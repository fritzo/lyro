import os

import pytest

from lyro.interpreters import MemoizeSqlite, Standard, ThreadRandomKey, set_interpreter

REPO = os.path.dirname(os.path.dirname(__file__))
TEST_DB = os.path.join(REPO, "data", "test.db")


@pytest.fixture(autouse=True)
def reset_interpreters():
    set_interpreter(Standard() + MemoizeSqlite(TEST_DB) + ThreadRandomKey())
