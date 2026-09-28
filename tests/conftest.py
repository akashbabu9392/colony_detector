import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests must not pick up whatever weights happen to sit in models/.
os.environ.setdefault("CD_ENGINES", "classical")
os.environ.setdefault("CD_WARMUP", "false")


@pytest.fixture(scope="session")
def counter():
    from colony_detector.config import Settings
    from colony_detector.pipeline import ColonyCounter

    c = ColonyCounter(Settings(engines="classical"))
    c.load()
    return c
