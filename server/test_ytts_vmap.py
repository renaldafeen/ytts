"""Unit tests for voice-name map loading and [aref:] target resolution.

Run from server/: python -m unittest test_ytts_vmap.py
"""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ytts_vmap import load_voice_map, resolve_aref_target


def _load_capturing(path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = load_voice_map(path)
    return result, buf.getvalue()


class LoadVoiceMapTests(unittest.TestCase):
    def test_omitted_path_returns_empty_and_logs_nothing(self):
        result, log = _load_capturing(None)
        self.assertEqual(result, {})
        self.assertEqual(log, "")

    def test_empty_path_logs_missing_filename(self):
        result, log = _load_capturing("")
        self.assertEqual(result, {})
        self.assertIn("filename is missing", log)

    def test_whitespace_path_logs_missing_filename(self):
        result, log = _load_capturing("   ")
        self.assertEqual(result, {})
        self.assertIn("filename is missing", log)

    def test_missing_file(self):
        result, log = _load_capturing("/tmp/does-not-exist-ytts-vmap.json")
        self.assertEqual(result, {})
        self.assertIn("file not found", log)

    def test_unreadable_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            handle.write('{"A": "a.wav"}')
            path = handle.name
        try:
            with patch("ytts_vmap.open", side_effect=OSError("permission denied")):
                result, log = _load_capturing(path)
            self.assertEqual(result, {})
            self.assertIn("cannot read", log)
        finally:
            os.unlink(path)

    def test_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            handle.write("{not json")
            path = handle.name
        try:
            result, log = _load_capturing(path)
            self.assertEqual(result, {})
            self.assertIn("cannot read", log)
        finally:
            os.unlink(path)

    def test_json_array_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            handle.write('["A", "B"]')
            path = handle.name
        try:
            result, log = _load_capturing(path)
            self.assertEqual(result, {})
            self.assertIn("top-level JSON must be an object", log)
        finally:
            os.unlink(path)

    def test_valid_map_logs_each_pair(self):
        payload = {
            "Bette": "~/temp/M2.wav",
            "Mark": "default",
            "Nobody": "/tmp/nonexistentfile.wav",
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = handle.name
        try:
            result, log = _load_capturing(path)
            self.assertEqual(result, payload)
            self.assertIn("Voice map: Bette -> ~/temp/M2.wav", log)
            self.assertIn("Voice map: Mark -> default", log)
            self.assertIn("Voice map: Nobody -> /tmp/nonexistentfile.wav", log)
        finally:
            os.unlink(path)

    def test_duplicate_keys_last_wins(self):
        raw = '{"A": "first.wav", "B": "b.wav", "A": "second.wav"}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            handle.write(raw)
            path = handle.name
        try:
            result, log = _load_capturing(path)
            self.assertEqual(result, {"A": "second.wav", "B": "b.wav"})
            self.assertIn("duplicate key: A", log)
            self.assertIn("Voice map: A -> second.wav", log)
            self.assertIn("Voice map: B -> b.wav", log)
        finally:
            os.unlink(path)

    def test_non_string_value_skipped(self):
        raw = '{"A": "a.wav", "B": 123, "C": "c.wav"}'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
            handle.write(raw)
            path = handle.name
        try:
            result, log = _load_capturing(path)
            self.assertEqual(result, {"A": "a.wav", "C": "c.wav"})
            self.assertIn("non-string value for key B", log)
            self.assertIn("Voice map: A -> a.wav", log)
            self.assertIn("Voice map: C -> c.wav", log)
            self.assertNotIn("Voice map: B ->", log)
        finally:
            os.unlink(path)

    def test_tilde_in_filename_is_expanded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vmap.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"Vera": "/voices/M1.wav"}, handle)
            with patch.dict(os.environ, {"HOME": tmp}):
                result, log = _load_capturing("~/vmap.json")
            self.assertEqual(result, {"Vera": "/voices/M1.wav"})
            self.assertIn("Voice map: Vera -> /voices/M1.wav", log)


class ResolveArefTargetTests(unittest.TestCase):
    def setUp(self):
        self.voice_map = {
            "Bette": "~/temp/M2.wav",
            "Abby": "~/tmp/M2.wav",
            "Mark": "default",
            "Vera": "./voices/M1.wav",
            "Nobody": "/tmp/nonexistentfile.wav",
        }

    def test_wav_extension_skips_map(self):
        self.assertEqual(
            resolve_aref_target("/x/voice.wav", self.voice_map),
            ("path", "/x/voice.wav"),
        )

    def test_wav_extension_case_insensitive(self):
        self.assertEqual(
            resolve_aref_target("/x/voice.WAV", self.voice_map),
            ("path", "/x/voice.WAV"),
        )

    def test_wav_path_not_looked_up_even_if_key_exists(self):
        self.assertEqual(
            resolve_aref_target("Nobody.wav", self.voice_map),
            ("path", "Nobody.wav"),
        )

    def test_mapped_path(self):
        self.assertEqual(
            resolve_aref_target("Vera", self.voice_map),
            ("path", "./voices/M1.wav"),
        )

    def test_mapped_tilde_path_passed_through(self):
        self.assertEqual(
            resolve_aref_target("Bette", self.voice_map),
            ("path", "~/temp/M2.wav"),
        )

    def test_mapped_default(self):
        self.assertEqual(
            resolve_aref_target("Mark", self.voice_map),
            ("default", None),
        )

    def test_mapped_default_with_surrounding_whitespace(self):
        self.assertEqual(
            resolve_aref_target("Padded", {"Padded": "  default  "}),
            ("default", None),
        )

    def test_missing_key_uses_default(self):
        self.assertEqual(
            resolve_aref_target("Unknown", self.voice_map),
            ("default", None),
        )

    def test_empty_map_uses_default(self):
        self.assertEqual(
            resolve_aref_target("Vera", {}),
            ("default", None),
        )

    def test_lookup_is_case_sensitive(self):
        self.assertEqual(
            resolve_aref_target("vera", self.voice_map),
            ("default", None),
        )

    def test_strips_target_whitespace(self):
        self.assertEqual(
            resolve_aref_target("  Vera  ", self.voice_map),
            ("path", "./voices/M1.wav"),
        )


if __name__ == "__main__":
    unittest.main()
