"""Predictive maintenance pipeline for NASA C-MAPSS FD001."""

import sys


def _use_utf8_console():
    """
    Make stdout/stderr able to carry the non-ASCII characters the
    pipeline prints (R^2 as "R²", "✓ New best model").

    On Windows the console defaults to cp1252, which cannot encode
    either one. Without this, a training run dies part-way through
    with UnicodeEncodeError after the expensive work is already done.
    """

    for stream in (sys.stdout, sys.stderr):

        reconfigure = getattr(
            stream,
            "reconfigure",
            None,
        )

        if reconfigure is None:
            # Not a regular text stream (e.g. pytest capture).
            continue

        try:
            reconfigure(
                encoding="utf-8",
                errors="replace",
            )
        except (ValueError, OSError):
            # Already detached or not reconfigurable; the
            # errors="replace" fallback below is not available, but
            # printing must never be the thing that fails a run.
            pass


_use_utf8_console()
