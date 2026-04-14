from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp_sgd.privacy import PrivacyArtifacts, attach_privacy, ensure_dp_compatible, get_epsilon

__all__ = [
    "PrivacyArtifacts",
    "attach_privacy",
    "ensure_dp_compatible",
    "get_epsilon",
]
