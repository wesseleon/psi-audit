import asyncio
import inspect
import os
from dataclasses import dataclass
from functools import cache
from typing import Any, get_type_hints

import mistralai.workflows as workflows
from mistralai.workflows import workflow as _wf
from mistralai.workflows.core.config.config import (
    RESERVED_QUERY_NAMES,
    RESERVED_UPDATE_NAMES,
    PayloadEncryptionConfig,
    WorkerConfig,
)
from mistralai.workflows.core.config.config import config as sdk_config
from mistralai.workflows.protocol.v1.workflow import DeploymentName
from pydantic import BaseModel, SecretStr, TypeAdapter

with _wf.unsafe.imports_passed_through():
    import structlog
    from env.mistral import env
    from env.workflows import env as workflows_env
    from mistralai.client import Mistral
    from mistralai.client.errors import MistralError
    from mistralai.client.models import (
        ScheduleDefinition,
        ScheduleInterval,
        SchedulePolicy,
    )
    from mistralai.extra.workflows.encoding.config import WorkflowEncodingConfig
    from mistralai.extra.workflows.encoding.helpers import configure_workflow_encoding
    from mistralai.extra.workflows.encoding.payload_encoder import PayloadEncoder
    from mistralai.workflows import ScheduleOverlapPolicy
    from utils.mistral import mistral_client

_SCHEDULE_CATCHUP_SECONDS = 86400

# The worker registers its workflows a few seconds after its container starts; until it has, the
# platform 404s a schedule naming one. `converge_schedule` retries across that window.
_REGISTRATION_RETRY_ATTEMPTS = 9
_REGISTRATION_RETRY_SECONDS = 10.0
# Retries of a connection the SDK's HTTP clients could not open (refused or timed out): the first at
# once, then after 0.5 s, doubling, about a minute of waiting plus the connect timeout per attempt.
# Their first call is config discovery, which ends the worker when it cannot connect.
_CONNECT_RETRIES = 8
# The note a disabled setting pauses its schedule with: the only pause `converge_schedule` undoes.
_DISABLED_NOTE = "disabled by init"

_logger = structlog.get_logger("workflows.encryption")
_schedules_logger = structlog.get_logger("workflows.schedules")

# Assigning a config attribute skips the SDK's validation, so the name is validated here instead.
_DEPLOYMENT_NAME: TypeAdapter[str | None] = TypeAdapter(DeploymentName)

# An SDK internal, not a published contract: it can move on a version bump.
_ENTRYPOINT_MARKER = "__workflows_workflow_entrypoint"

__all__ = [
    "ScheduleTimingError",
    "WorkflowCapabilities",
    "WorkflowRun",
    "converge_schedule",
    "get_client",
    "infer_workflow_models",
    "pause_schedule_if_present",
    "payload_encryption",
    "resume_schedule_if_paused",
    "run_worker",
    "start_workflow",
    "upsert_schedule",
    "workflow_capabilities",
]


@cache
def payload_encryption() -> PayloadEncryptionConfig | None:
    encryption = WorkerConfig().temporal_payload_encryption
    if encryption is not None:
        PayloadEncoder(WorkflowEncodingConfig(payload_encryption=encryption))
        _logger.info(
            "workflow payload encryption enabled",
            mode=encryption.mode.value,
            key_rotation=encryption.secondary_key is not None,
        )
        return encryption
    _logger.info("workflow payload encryption is off")
    legacy_mode = os.environ.get("WORKFLOWS_ENCRYPTION_MODE", "").strip()
    if legacy_mode not in ("", "off"):
        _logger.warning(
            "WORKFLOWS_ENCRYPTION_MODE is no longer read, so payloads are not encrypted; set "
            "TEMPORAL_PAYLOAD_ENCRYPTION__MODE and TEMPORAL_PAYLOAD_ENCRYPTION__MAIN_KEY to encrypt them",
            workflows_encryption_mode=legacy_mode,
        )
    return None


_clients: dict[asyncio.AbstractEventLoop, asyncio.Task[Mistral]] = {}


async def _build_client() -> Mistral:
    client = mistral_client(server_url=workflows_env.workflows_base_url)
    encryption = payload_encryption()
    if encryption is not None:
        await configure_workflow_encoding(WorkflowEncodingConfig(payload_encryption=encryption), client=client)
    return client


async def get_client() -> Mistral:
    """The Mistral client, with payload encoding installed before its first use.

    The SDK keeps the encoding config on the client instance, so the configured client itself is
    what is cached and reused: a fresh ``Mistral`` would carry no config and send cleartext. It is
    built as one task per event loop, so concurrent first calls share one namespace lookup. A
    failed build is evicted, not cached, so a transient blip does not brick the process; it is
    re-raised because proceeding puts cleartext on the wire under ``full``.
    """
    loop = asyncio.get_running_loop()
    if (task := _clients.get(loop)) is None:
        for other in list(_clients):
            if other.is_closed():
                _clients.pop(other, None)
        task = _clients[loop] = loop.create_task(_build_client())
        task.add_done_callback(_forget_failed)
    return await asyncio.shield(task)


def _forget_failed(task: asyncio.Task[Mistral]) -> None:
    if (task.cancelled() or task.exception() is not None) and _clients.get(task.get_loop()) is task:
        del _clients[task.get_loop()]


def _call(**kwargs: Any) -> dict[str, Any]:
    """The kwargs for one platform call: drop what the caller left unset.

    Unset optionals are dropped because the SDK distinguishes an omitted parameter from an
    explicit ``None``, which serializes as a null the server may reject.
    """
    return {name: value for name, value in kwargs.items() if value is not None}


class ScheduleTimingError(ValueError):
    """Raised when a schedule declares neither or both of an interval and a cron expression."""


def _validated_timing(interval_seconds: int | None, cron_expressions: list[str] | None) -> None:
    if bool(interval_seconds) == bool(cron_expressions):
        raise ScheduleTimingError(
            "Provide exactly one of interval_seconds or cron_expressions. Prefer cron for a fixed "
            "time of day: an interval is anchored to the epoch, so it drifts against wall-clock time."
        )


@dataclass(frozen=True)
class WorkflowRun:
    """What starting a workflow gives back: always an id, a result only if we waited."""

    execution_id: str
    result: Any = None


@dataclass(frozen=True)
class WorkflowCapabilities:
    """The command handlers a workflow class declares, as names a caller may legitimately invoke."""

    signals: tuple[str, ...]
    queries: tuple[str, ...]
    updates: tuple[str, ...]


def workflow_capabilities(workflow_class: type) -> WorkflowCapabilities:
    """Which of signal, query and update a workflow actually answers.

    Lets a caller-facing surface publish only the commands a workflow can serve. Framework-reserved
    handlers are excluded: the reserved sets come from the SDK, so a new internal handler drops out
    of the contract on the next bump. The SDK publishes no reserved-signal set, so every declared
    signal belongs to the author. Raises the SDK's ``ValueError`` for a class never ``@workflow.define``d.
    """
    definition = workflows.get_workflow_definition(workflow_class)
    return WorkflowCapabilities(
        signals=tuple(handler.name for handler in definition.signals),
        queries=tuple(handler.name for handler in definition.queries if handler.name not in RESERVED_QUERY_NAMES),
        updates=tuple(handler.name for handler in definition.updates if handler.name not in RESERVED_UPDATE_NAMES),
    )


def infer_workflow_models(workflow_class: type) -> tuple[type[BaseModel], Any | None]:
    """The request and response types a workflow's entrypoint declares.

    Here rather than in a caller because it needs ``_ENTRYPOINT_MARKER``: ``WorkflowSpec``
    publishes JSON schema, not the model classes. Raises ``ValueError`` like
    :func:`workflow_capabilities`.
    """
    entrypoint = next(
        (
            method
            for _, method in inspect.getmembers(workflow_class, predicate=inspect.isfunction)
            if getattr(method, _ENTRYPOINT_MARKER, False)
        ),
        None,
    )
    if entrypoint is None:
        raise ValueError(f"{workflow_class.__name__} has no workflow entrypoint")

    original = inspect.unwrap(entrypoint)
    hints = get_type_hints(original)
    parameters = [parameter for name, parameter in inspect.signature(original).parameters.items() if name != "self"]
    if len(parameters) != 1:
        raise ValueError(f"{workflow_class.__name__} must take exactly one entrypoint argument, not {len(parameters)}")

    request_type = hints.get(parameters[0].name)
    if not inspect.isclass(request_type) or not issubclass(request_type, BaseModel):
        raise ValueError(f"{workflow_class.__name__} entrypoint input must be a Pydantic model")

    return request_type, hints.get("return")


async def start_workflow(
    workflow_class: type,
    workflow_input: BaseModel,
    *,
    wait_for_result: bool,
    timeout_seconds: float | None = None,
    execution_id: str | None = None,
) -> WorkflowRun:
    """Start a workflow through the remote execution API.

    ``exclude_unset`` matters: the input models give optional fields defaults the SDK also
    defaults, and sending them explicitly is not the same request.

    ``execution_id`` lets a caller name the execution before it exists, which makes "authorise
    first, then start" possible. Omitted, ``_call`` drops it and the platform generates one.
    """
    response = await (await get_client()).workflows.execute_workflow_async(
        **_call(
            workflow_identifier=workflows.get_workflow_definition(workflow_class).name,
            execution_id=execution_id,
            input=workflow_input.model_dump(mode="json", exclude_unset=True),
            wait_for_result=wait_for_result,
            timeout_seconds=timeout_seconds,
            deployment_name=workflows_env.deployment_name,
        )
    )
    return WorkflowRun(execution_id=response.execution_id, result=getattr(response, "result", None))


def _configure_worker() -> None:
    """Point the SDK's module-level config at the app's own settings.

    The SDK reads its settings from the environment when ``mistralai.workflows`` is imported,
    which can be before ``env`` has loaded the repo-root ``.env``. Its own names differ too
    (``SERVER_URL``), so without this the worker would poll the public host whatever
    ``WORKFLOWS_BASE_URL`` says, and run under whatever key and queue the SDK happened to see.
    """
    worker = sdk_config.worker
    worker.temporal_payload_encryption = payload_encryption()
    # The agent client (and the telemetry endpoint derived from it) defaults to ``server_url`` at
    # construction, so it only follows the move when nobody set it on purpose.
    if worker.agent.mistral_client_server_url in (None, worker.server_url):
        worker.agent.mistral_client_server_url = workflows_env.workflows_base_url
    worker.server_url = workflows_env.workflows_base_url
    if env.mistral_api_key:
        sdk_config.common.mistral_api_key = SecretStr(env.mistral_api_key)
    if workflows_env.deployment_name:
        worker.deployment_name = _DEPLOYMENT_NAME.validate_python(workflows_env.deployment_name)
    # Unset, a connection refused during a network blip fails at once, and under `mistral apps dev`
    # nothing restarts the worker before the next file edit. `MISTRAL_WORKFLOWS_HTTP_RETRIES` wins.
    if "retries" not in sdk_config.http.model_fields_set:
        sdk_config.http.retries = _CONNECT_RETRIES


async def run_worker(workflow_classes: list[type]) -> Any:
    """Run the worker, configured from our own settings.

    The config is assigned immediately before delegating. ``workflows.run_worker`` deep-copies it
    on entry and restores that copy on exit. Config discovery is on, so it authenticates from
    ``common.mistral_api_key``: its own ``api_key`` argument applies only with discovery off.
    """
    _configure_worker()
    return await workflows.run_worker(workflow_classes)


async def pause_schedule_if_present(*, schedule_id: str, note: str) -> bool:
    """Pause an existing schedule; return False when there is nothing to pause.

    Pausing rather than deleting keeps history and allows resume. A missing schedule surfaces as a
    generic SDKError carrying a 404, not a typed not-found, so the status check is deliberate.
    """
    schedules = (await get_client()).workflows.schedules
    try:
        await schedules.pause_schedule_async(**_call(schedule_id=schedule_id, note=note))
    except MistralError as error:
        if getattr(error, "status_code", None) != 404:
            raise
        return False
    return True


async def resume_schedule_if_paused(*, schedule_id: str, note: str, paused_by: str | None = None) -> bool:
    """Resume a paused schedule; return False when it is absent, already active, or (with
    ``paused_by``) paused with another note.

    The counterpart of ``pause_schedule_if_present``, and what an enabled setting needs after
    ``upsert_schedule``, which cannot resume. It reads the schedule first so an active one is left
    alone rather than relying on how the platform answers a resume it does not need. A schedule
    deleted between the read and the resume is absent too, so the resume's 404 is False as well.
    ``paused_by`` limits it to the pause that note made, so one made by ``pause_on_failure`` or by
    hand stays.
    """
    schedules = (await get_client()).workflows.schedules
    try:
        schedule = await schedules.get_schedule_async(**_call(schedule_id=schedule_id))
        if not schedule.paused or (paused_by is not None and schedule.note != paused_by):
            return False
        await schedules.resume_schedule_async(**_call(schedule_id=schedule_id, note=note))
    except MistralError as error:
        if getattr(error, "status_code", None) != 404:
            raise
        return False
    return True


async def upsert_schedule(
    *,
    schedule_id: str,
    workflow_identifier: str,
    input: dict[str, Any],
    interval_seconds: int | None = None,
    cron_expressions: list[str] | None = None,
    pause_on_failure: bool = False,
    deployment_name: str | None = None,
) -> Any:
    """Idempotently ensure a recurring workflow schedule exists and matches this config.

    Timing is exactly one of ``interval_seconds`` or ``cron_expressions`` (5-field UTC cron); an
    interval is anchored to the epoch and drifts. Re-running on every deploy converges, because the
    platform resolves an existing ``schedule_id`` to an update. This cannot resume: no create or
    update model carries ``paused``, so a paused schedule stays paused until a caller resumes it
    (``resume_schedule_if_paused``).
    """
    _validated_timing(interval_seconds, cron_expressions)
    schedules = (await get_client()).workflows.schedules
    return await schedules.schedule_workflow_async(
        **_call(
            schedule=ScheduleDefinition(
                schedule_id=schedule_id,
                input=input,
                intervals=[ScheduleInterval(every=f"PT{interval_seconds}S")] if interval_seconds else [],
                cron_expressions=cron_expressions or [],
                policy=SchedulePolicy(
                    overlap=ScheduleOverlapPolicy.SKIP,
                    catchup_window_seconds=_SCHEDULE_CATCHUP_SECONDS,
                    pause_on_failure=pause_on_failure,
                ),
            ),
            workflow_identifier=workflow_identifier,
            deployment_name=deployment_name,
        )
    )


async def converge_schedule(
    *,
    schedule_id: str,
    workflow_identifier: str,
    enabled: bool,
    input: dict[str, Any],
    interval_seconds: int | None = None,
    cron_expressions: list[str] | None = None,
    pause_on_failure: bool = False,
    deployment_name: str | None = None,
) -> None:
    """Converge one schedule to its settings: upserted when ``enabled``, paused otherwise.

    For init one-shots, which run once the worker container is up but possibly before it has
    registered the workflow: the platform's 404 for an unregistered workflow is retried a few times,
    some seconds apart, before it is raised. A schedule an earlier run paused because the setting was
    off is resumed once it is upserted, so turning the setting back on restarts it (``upsert_schedule``
    alone cannot resume); a schedule paused for any other reason (by ``pause_on_failure``, or by hand)
    stays paused.
    """
    if not enabled:
        was_present = await pause_schedule_if_present(schedule_id=schedule_id, note=_DISABLED_NOTE)
        _schedules_logger.info("schedule disabled", schedule_id=schedule_id, was_present=was_present)
        return

    for attempt in range(1, _REGISTRATION_RETRY_ATTEMPTS + 1):
        try:
            await upsert_schedule(
                schedule_id=schedule_id,
                workflow_identifier=workflow_identifier,
                input=input,
                interval_seconds=interval_seconds,
                cron_expressions=cron_expressions,
                pause_on_failure=pause_on_failure,
                deployment_name=deployment_name,
            )
            break
        except MistralError as error:
            if getattr(error, "status_code", None) != 404 or attempt == _REGISTRATION_RETRY_ATTEMPTS:
                raise
            _schedules_logger.info(
                "workflow not registered yet; waiting for the worker",
                schedule_id=schedule_id,
                workflow=workflow_identifier,
                attempt=attempt,
            )
            await asyncio.sleep(_REGISTRATION_RETRY_SECONDS)
    resumed = await resume_schedule_if_paused(schedule_id=schedule_id, note="enabled by init", paused_by=_DISABLED_NOTE)
    _schedules_logger.info(
        "schedule registered",
        schedule_id=schedule_id,
        resumed=resumed,
        workflow=workflow_identifier,
        cron=cron_expressions,
        interval_seconds=interval_seconds,
        pause_on_failure=pause_on_failure,
    )
