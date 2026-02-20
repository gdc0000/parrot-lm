import unittest
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


class TestSessionState(unittest.TestCase):
    def test_initialize_session_state_sets_defaults_without_saved_logs(self):
        fake_st = SimpleNamespace(session_state={})
        storage = _FakeLocalStorage()

        with patch.object(session_state, "st", fake_st):
            session_state.initialize_session_state(storage)

        self.assertIn("last_generated_config", fake_st.session_state)
        self.assertTrue(fake_st.session_state["all_logs"].empty)

    def test_initialize_session_state_loads_saved_logs(self):
        fake_st = SimpleNamespace(session_state={})
        saved_logs = [{"content": "hello", "output_tokens": 3}]
        storage = _FakeLocalStorage({session_state.LOCAL_STORAGE_LOG_KEY: saved_logs})

        with patch.object(session_state, "st", fake_st):
            session_state.initialize_session_state(storage)

        all_logs_df = fake_st.session_state["all_logs"]
        self.assertEqual(len(all_logs_df), 1)
        self.assertEqual(all_logs_df.loc[0, "content"], "hello")

    def test_clear_local_data_supports_erase_item(self):
        fake_st = SimpleNamespace(session_state={"all_logs": pd.DataFrame([{"x": 1}])})
        storage = _FakeLocalStorageWithErase({session_state.LOCAL_STORAGE_LOG_KEY: [{"x": 1}]})

        with patch.object(session_state, "st", fake_st):
            session_state.clear_local_data(storage)

        self.assertNotIn(session_state.LOCAL_STORAGE_LOG_KEY, storage.store)
        self.assertTrue(fake_st.session_state["all_logs"].empty)

    def test_append_and_persist_logs_appends_and_syncs(self):
        fake_st = SimpleNamespace(session_state={"all_logs": pd.DataFrame([{"content": "a"}])})
        storage = _FakeLocalStorage()
        new_logs_df = pd.DataFrame([{"content": "b"}])

        with patch.object(session_state, "st", fake_st):
            session_state.append_and_persist_logs(storage, new_logs_df)

        updated_df = fake_st.session_state["all_logs"]
        self.assertEqual(len(updated_df), 2)
        self.assertEqual(updated_df.iloc[-1]["content"], "b")
        self.assertEqual(len(storage.store[session_state.LOCAL_STORAGE_LOG_KEY]), 2)
