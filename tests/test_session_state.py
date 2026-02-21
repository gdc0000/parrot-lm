from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from parrotlm.ui import session_state


class _FakeLocalStorage:
    def __init__(self, initial=None):
        self.store = dict(initial or {})

    def getItem(self, key):
        return self.store.get(key)

    def setItem(self, key, value):
        self.store[key] = value

    def deleteItem(self, key):
        if key not in self.store:
            raise KeyError(key)
        del self.store[key]


class _FakeLocalStorageWithErase(_FakeLocalStorage):
    def eraseItem(self, key, default=None):
        self.store.pop(key, default)


class _FakeLocalStorageGetItemError(_FakeLocalStorage):
    def getItem(self, key):
        raise TypeError("storage not available")


class _FakeLocalStorageSetItemError(_FakeLocalStorage):
    def setItem(self, key, value):
        raise ValueError("write failed")


def test_initialize_session_state_sets_defaults_without_saved_logs():
    fake_st = SimpleNamespace(session_state={})
    storage = _FakeLocalStorage()

    with patch.object(session_state, "st", fake_st):
        session_state.initialize_session_state(storage)

    assert "last_generated_config" in fake_st.session_state
    assert fake_st.session_state["all_logs"].empty


def test_initialize_session_state_loads_saved_logs():
    fake_st = SimpleNamespace(session_state={})
    saved_logs = [{"content": "hello", "output_tokens": 3}]
    storage = _FakeLocalStorage({session_state.LOCAL_STORAGE_LOG_KEY: saved_logs})

    with patch.object(session_state, "st", fake_st):
        session_state.initialize_session_state(storage)

    all_logs_df = fake_st.session_state["all_logs"]
    assert len(all_logs_df) == 1
    assert all_logs_df.loc[0, "content"] == "hello"


def test_initialize_session_state_handles_malformed_saved_logs():
    fake_st = SimpleNamespace(session_state={})
    storage = _FakeLocalStorage({session_state.LOCAL_STORAGE_LOG_KEY: 1})

    with patch.object(session_state, "st", fake_st):
        session_state.initialize_session_state(storage)

    assert fake_st.session_state["all_logs"].empty


def test_clear_local_data_supports_erase_item():
    fake_st = SimpleNamespace(session_state={"all_logs": pd.DataFrame([{"x": 1}])})
    storage = _FakeLocalStorageWithErase({session_state.LOCAL_STORAGE_LOG_KEY: [{"x": 1}]})

    with patch.object(session_state, "st", fake_st):
        session_state.clear_local_data(storage)

    assert session_state.LOCAL_STORAGE_LOG_KEY not in storage.store
    assert fake_st.session_state["all_logs"].empty


def test_clear_local_data_supports_delete_item():
    fake_st = SimpleNamespace(session_state={"all_logs": pd.DataFrame([{"x": 1}])})
    storage = _FakeLocalStorage({session_state.LOCAL_STORAGE_LOG_KEY: [{"x": 1}]})

    with patch.object(session_state, "st", fake_st):
        session_state.clear_local_data(storage)

    assert session_state.LOCAL_STORAGE_LOG_KEY not in storage.store
    assert fake_st.session_state["all_logs"].empty


def test_append_and_persist_logs_appends_and_syncs():
    fake_st = SimpleNamespace(session_state={"all_logs": pd.DataFrame([{"content": "a"}])})
    storage = _FakeLocalStorage()
    new_logs_df = pd.DataFrame([{"content": "b"}])

    with patch.object(session_state, "st", fake_st):
        session_state.append_and_persist_logs(storage, new_logs_df)

    updated_df = fake_st.session_state["all_logs"]
    assert len(updated_df) == 2
    assert updated_df.iloc[-1]["content"] == "b"
    assert len(storage.store[session_state.LOCAL_STORAGE_LOG_KEY]) == 2


def test_append_and_persist_logs_sets_new_logs_when_current_is_empty():
    fake_st = SimpleNamespace(session_state={"all_logs": pd.DataFrame()})
    storage = _FakeLocalStorage()
    new_logs_df = pd.DataFrame([{"content": "new"}])

    with patch.object(session_state, "st", fake_st):
        session_state.append_and_persist_logs(storage, new_logs_df)

    assert len(fake_st.session_state["all_logs"]) == 1
    assert storage.store[session_state.LOCAL_STORAGE_LOG_KEY][0]["content"] == "new"


def test_initialize_session_state_handles_storage_read_errors():
    fake_st = SimpleNamespace(session_state={})
    storage = _FakeLocalStorageGetItemError()

    with patch.object(session_state, "st", fake_st):
        session_state.initialize_session_state(storage)

    assert fake_st.session_state["all_logs"].empty


def test_append_and_persist_logs_keeps_session_data_when_storage_write_fails():
    fake_st = SimpleNamespace(session_state={"all_logs": pd.DataFrame()})
    storage = _FakeLocalStorageSetItemError()
    new_logs_df = pd.DataFrame([{"content": "persist-me"}])

    with patch.object(session_state, "st", fake_st):
        session_state.append_and_persist_logs(storage, new_logs_df)

    assert len(fake_st.session_state["all_logs"]) == 1
    assert fake_st.session_state["all_logs"].iloc[0]["content"] == "persist-me"
