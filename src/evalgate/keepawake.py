"""Keep the computer from sleeping during long eval runs.

A laptop that sleeps mid-run freezes the process; on wake every connection is dead and the
conversations in flight fail. On Windows we ask the OS not to sleep while the run is active (the
screen may still turn off). On macOS we use `caffeinate`. Elsewhere this does nothing.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from collections.abc import Iterator

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextlib.contextmanager
def keep_awake() -> Iterator[bool]:
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ok = bool(kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
        try:
            yield ok
        finally:
            kernel32.SetThreadExecutionState(ES_CONTINUOUS)  # back to normal sleep rules
    elif sys.platform == "darwin":
        try:
            proc = subprocess.Popen(["caffeinate", "-i", "-w", str(__import__("os").getpid())])
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            proc.terminate()
    else:
        yield False
