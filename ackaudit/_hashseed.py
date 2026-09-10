"""Re-exec with a fixed PYTHONHASHSEED.

Set iteration order varies with the hash seed, and that leaks into the
partitioner and the evaluator. Pinning it is a reproducibility measure for this
harness, not a fix for the upstream bug.
"""

import os
import sys

SEED = "0"


def pin() -> None:
    if os.environ.get("PYTHONHASHSEED") != SEED:
        os.environ["PYTHONHASHSEED"] = SEED
        os.execv(sys.executable, [sys.executable] + sys.argv)
