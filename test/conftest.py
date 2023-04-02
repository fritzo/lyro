import pytest

from lyro.interpreters import MemoizeSqlite, Standard, ThreadRandomKey, set_interpreter


@pytest.fixture(autouse=True)
def reset_interpreters():
    set_interpreter(Standard() + MemoizeSqlite() + ThreadRandomKey())
