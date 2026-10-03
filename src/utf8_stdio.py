"""Make stdout and stderr UTF-8 on every platform.

On Windows, Python writes redirected output (a pipe or a file) in the ANSI code page (cp1252),
which cannot encode − (U+2212) and writes ° and ± as single bytes that a UTF-8 reader rejects.
Every v2 entry script calls setup() before printing anything.
"""

from __future__ import annotations

import sys


def setup() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
