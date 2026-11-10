"""The file-watching loop the worker runs under in development (the app's ``worker.entrypoints.dev``)."""

import signal
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

from watchfiles import watch

__all__ = ["serve"]

# How often the worker is checked for an exit while no file changes.
_POLL_MS = 500
# How long the worker gets to shut down after SIGINT before it is killed.
_STOP_TIMEOUT_SECONDS = 5


def serve(command: Sequence[str], paths: Sequence[str | Path]) -> int:
    """Run ``command``, and restart it whenever a file under ``paths`` changes.

    If the run started at launch exits on its own, return its exit status, so a worker that cannot
    start ends the dev process and ``mistral apps dev`` reports it as exited. If a run started by an
    edit exits, keep waiting and start it again on the next edit, since a save can catch the code
    half-written. SIGINT or SIGTERM stops the run and returns 0. Call it from the main thread: it
    handles both signals while it runs, and puts back the handlers it found when it returns.
    """
    # Stop signals only set `stopping`. An exception raised from a handler inside `Popen` would lose a
    # worker that had already started.
    stopping = threading.Event()
    previous = {
        stop_signal: signal.signal(stop_signal, lambda _signum, _frame: stopping.set())
        for stop_signal in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        return _serve(command, paths, stopping)
    finally:
        for stop_signal, handler in previous.items():
            signal.signal(stop_signal, handler)


def _serve(command: Sequence[str], paths: Sequence[str | Path], stopping: threading.Event) -> int:
    worker = subprocess.Popen(command)
    edited = False
    try:
        for changes in watch(*paths, stop_event=stopping, rust_timeout=_POLL_MS, yield_on_timeout=True):
            if stopping.is_set():
                break
            if changes:
                _stop(worker)
                if stopping.is_set():
                    break
                worker = subprocess.Popen(command)
                edited = True
                continue
            # Polled on every tick, so a run an edit started is reaped as soon as it exits.
            exited = worker.poll() is not None
            # A stop signal sent to the whole process group (Ctrl+C, `mistral apps dev`) also ends the
            # worker; that exit is the stop, not a worker that died. Checked after `poll`: once the
            # signal has ended the worker it is pending here too, and `is_set` runs its handler first.
            if exited and not edited and not stopping.is_set():
                # A signal death is -N here; report it as a shell would, 128 + N.
                return worker.returncode if worker.returncode >= 0 else 128 - worker.returncode
    finally:
        _stop(worker)
    return 0


def _stop(worker: subprocess.Popen[bytes]) -> None:
    if worker.poll() is not None:
        return
    worker.send_signal(signal.SIGINT)
    try:
        worker.wait(_STOP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
