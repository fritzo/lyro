import os

import pytest

from lyro.interpreters import MemoizeSqlite, ThreadRandomKey, set_interpreter

REPO = os.path.dirname(os.path.dirname(__file__))
TEST_DB = os.path.join(REPO, "data", "test.db")


@pytest.fixture(autouse=True)
def reset_interpreters():
    set_interpreter(MemoizeSqlite(TEST_DB) + ThreadRandomKey())
