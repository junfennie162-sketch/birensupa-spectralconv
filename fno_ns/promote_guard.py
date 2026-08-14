"""Shared promote gate for public NS64 autochains.

Default: never call promote_public_ckpt automatically.
Requires both (1) beat declaration gate = live_best - 1e-4 and
(2) ALLOW_AUTO_PROMOTE=1. Otherwise only logs SIGNAL-pending.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


def evaluate_promote(
    src: Path,
    tag: str,
    live_best: float,
    log: Callable[[str], None],
    gate_delta: float = 1e-4,
) -> tuple[bool, float, float]:
    """Return (may_auto_promote, l2, gate). Never mutates checkpoints."""
    if not src.exists():
        log(f"skip promote {tag}: missing {src}")
        return False, 1e9, live_best - gate_delta

    import torch

    blob = torch.load(src, map_location="cpu", weights_only=False)
    l2 = float(blob.get("test_l2", 1e9))
    gate = float(live_best) - gate_delta

    if l2 >= live_best - 1e-9:
        log(f"no promote {tag}: {l2:.8f} >= live {live_best:.8f}")
        return False, l2, gate

    if l2 >= gate:
        log(
            f"no promote {tag}: {l2:.8f} improved vs live {live_best:.8f} "
            f"but NOT < gate {gate:.8f} (delta={gate_delta:g})"
        )
        return False, l2, gate

    if os.environ.get("ALLOW_AUTO_PROMOTE", "").strip() != "1":
        log(
            f"SIGNAL pending human confirm {tag}: {l2:.8f} < gate {gate:.8f}; "
            f"set ALLOW_AUTO_PROMOTE=1 to auto-run promote_public_ckpt"
        )
        return False, l2, gate

    log(f"auto-promote allowed {tag}: {l2:.8f} < gate {gate:.8f}")
    return True, l2, gate
