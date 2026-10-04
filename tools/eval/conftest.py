from __future__ import annotations

import shutil
import sys
from pathlib import Path
import uuid

import pytest


EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

TMP_ROOT = EVAL_DIR / ".pytest_tmp"


@pytest.fixture
def tmp_path() -> Path:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
