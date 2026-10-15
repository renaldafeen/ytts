"""Unit tests for ytts_client session history.

Run from client/: python -m unittest test_ytts_client.py
"""

import os
import unittest
from unittest.mock import patch

import readline

from ytts_client import _normalize_history, default_url, interpret_line, read_prompt


def _history_list():
    return [
        readline.get_history_item(i)
        for i in range(1, readline.get_current_history_length() + 1)
    ]


class NormalizeHistoryTests(unittest.TestCase):
    def setUp(self):
        readline.clear_history()

    def tearDown(self):
        readline.clear_history()

    def test_adds_stripped_line(self):
        _normalize_history("  hello  ", "hello")
        self.assertEqual(_history_list(), ["hello"])

    def test_skips_empty(self):
        _normalize_history("hello", "hello")
        _normalize_history("   ", "")
        self.assertEqual(_history_list(), ["hello"])

    def test_skips_consecutive_duplicate(self):
        _normalize_history("hello", "hello")
        _normalize_history("  hello  ", "hello")
        self.assertEqual(_history_list(), ["hello"])

    def test_appends_new_distinct_line(self):
        _normalize_history("hello", "hello")
        _normalize_history("world", "world")
        self.assertEqual(_history_list(), ["hello", "world"])

    def test_replaces_readline_auto_appended_raw_line(self):
        readline.add_history("  foo  ")
        _normalize_history("  foo  ", "foo")
        self.assertEqual(_history_list(), ["foo"])

    def test_edited_recall_is_a_new_entry(self):
        _normalize_history("hello", "hello")
        _normalize_history("hello there", "hello there")
        self.assertEqual(_history_list(), ["hello", "hello there"])


class ReadPromptTests(unittest.TestCase):
    def setUp(self):
        readline.clear_history()

    def tearDown(self):
        readline.clear_history()

    def test_strips_and_records(self):
        with patch("builtins.input", return_value="  hello  "):
            result = read_prompt("Prompt: ")
        self.assertEqual(result, "hello")
        self.assertEqual(_history_list(), ["hello"])

    def test_empty_input_not_recorded(self):
        with patch("builtins.input", return_value="  hello  "):
            read_prompt()
        with patch("builtins.input", return_value="   "):
            result = read_prompt()
        self.assertEqual(result, "")
        self.assertEqual(_history_list(), ["hello"])

    def test_consecutive_duplicate_not_recorded_twice(self):
        with patch("builtins.input", return_value="hello"):
            read_prompt()
        with patch("builtins.input", return_value="hello"):
            read_prompt()
        self.assertEqual(_history_list(), ["hello"])


class InterpretLineTests(unittest.TestCase):
    def test_quit_sends_nothing(self):
        self.assertEqual(interpret_line("quit"), ("quit", None))
        self.assertEqual(interpret_line("QUIT"), ("quit", None))

    def test_shutdown_server_is_a_directive(self):
        action, payload = interpret_line("shutdown-server")
        self.assertEqual(action, "shutdown")
        self.assertEqual(payload, {"directive": {"shutdown": True}})

    def test_normal_line_is_text(self):
        self.assertEqual(interpret_line("hello"), ("send", {"text": "hello"}))

    def test_set_keeps_voice_as_text_and_numbers_as_floats(self):
        _action, payload = interpret_line("set volume=0.5 voice=Alan seed=7")
        self.assertEqual(
            payload,
            {"directive": {"volume": 0.5, "voice": "Alan", "seed": 7.0}},
        )


class DefaultUrlTests(unittest.TestCase):
    def test_without_cbox_port(self):
        env = os.environ.copy()
        env.pop("CBOX_PORT", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(default_url(), "http://127.0.0.1:9126/yttss")

    def test_uses_cbox_port_when_set(self):
        with patch.dict(os.environ, {"CBOX_PORT": "9100"}):
            self.assertEqual(default_url(), "http://127.0.0.1:9100/yttss")


if __name__ == "__main__":
    unittest.main()
