"""Unit tests for ytts_client session history.

Run from client/: python -m unittest test_ytts_client.py
"""

import unittest
from unittest.mock import patch

import readline

from ytts_client import _normalize_history, read_prompt


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


if __name__ == "__main__":
    unittest.main()
