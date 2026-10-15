"""HTTP tests for the ytts server. Synthesis is mocked.

Run from server/ with the Chatterbox venv, which can import torch and chatterbox:

    python -m unittest test_ytts_http.py
"""

import io
import json
import time
import unittest
import wave
from io import BytesIO
from unittest.mock import patch

import ytts_server
from ytts_server import MAX_QUEUE_ITEMS, MAX_TEXT_CHARS, app


TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _wav_buffer():
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\x00\x00" * 8)
    buffer.seek(0)
    return buffer


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        ytts_server.auth_token = TOKEN
        ytts_server.model_ready = True
        ytts_server.is_server_running = True
        ytts_server.flush_requested = False
        ytts_server.server = None
        ytts_server.voice_map = {}
        ytts_server.audio_queue.clear()
        ytts_server.session.volume = 0.7
        ytts_server.session.exaggeration = 0.35
        ytts_server.session.pacing = 0.8
        ytts_server.session.seed = 13
        ytts_server.session.audio_prompt = None
        ytts_server.session.voice_name = None
        self.generate = patch(
            "ytts_server.generate_tts", return_value=_wav_buffer()
        ).start()
        self.addCleanup(patch.stopall)

    def tearDown(self):
        for item in ytts_server.audio_queue:
            item["buffer"].close()
        ytts_server.audio_queue.clear()
        ytts_server.flush_requested = False
        ytts_server.is_server_running = True
        ytts_server.server = None

    def test_health_does_not_require_a_token(self):
        ytts_server.model_ready = False
        loading = self.client.get("/health")
        self.assertEqual(loading.status_code, 503)
        self.assertEqual(loading.get_json(), {"status": "loading"})
        ytts_server.model_ready = True
        ready = self.client.get("/health")
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.get_json(), {"status": "ready"})

    def test_missing_and_wrong_token_are_rejected(self):
        missing = self.client.post("/yttss", json={"text": "hi"})
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.get_json(), {"error": "unauthorized"})
        wrong = self.client.post(
            "/yttss",
            json={"text": "hi"},
            headers={"Authorization": "Bearer no"},
        )
        self.assertEqual(wrong.status_code, 401)
        self.generate.assert_not_called()

    def test_success_reports_voice_name_and_not_a_path(self):
        ytts_server.session.voice_name = "Alan"
        ytts_server.session.audio_prompt = "/secret/path.wav"
        response = self.client.post("/yttss", json={"text": "hi"}, headers=AUTH)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "audio queued")
        self.assertEqual(body["parameters"]["voice"], "Alan")
        self.assertNotIn("audio_prompt", body["parameters"])
        self.assertNotIn("/secret/path.wav", json.dumps(body))

    def test_partial_success_reports_counts(self):
        def _generate(*_args, **_kwargs):
            if self.generate.call_count == 1:
                return _wav_buffer()
            return None

        self.generate.side_effect = _generate
        text = "word " * 80
        response = self.client.post("/yttss", json={"text": text}, headers=AUTH)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["status"], "partial")
        self.assertGreaterEqual(body["queued"], 1)
        self.assertGreaterEqual(body["failed"], 1)

    def test_generation_failure_is_a_fixed_message(self):
        self.generate.return_value = None
        response = self.client.post("/yttss", json={"text": "hi"}, headers=AUTH)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "synthesis failed"})

    def test_quit_text_does_not_stop_the_server(self):
        response = self.client.post("/yttss", json={"text": "quit"}, headers=AUTH)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(ytts_server.is_server_running)
        self.generate.assert_called()

    def test_shutdown_directive_schedules_server_shutdown(self):
        class FakeServer:
            def __init__(self):
                self.shutdowns = 0

            def shutdown(self):
                self.shutdowns += 1

        fake = FakeServer()
        ytts_server.server = fake
        response = self.client.post(
            "/yttss",
            json={"directive": {"shutdown": True}},
            headers=AUTH,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "shutting down")
        self.assertFalse(ytts_server.is_server_running)
        for _ in range(50):
            if fake.shutdowns:
                break
            time.sleep(0.01)
        self.assertEqual(fake.shutdowns, 1)

    def test_unknown_voice_token_keeps_the_current_prompt(self):
        ytts_server.session.audio_prompt = "/kept/voice.wav"
        ytts_server.session.voice_name = "Alan"
        response = self.client.post(
            "/yttss", json={"text": "[aref: NotAVoice] hi"}, headers=AUTH
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ytts_server.session.audio_prompt, "/kept/voice.wav")
        self.assertEqual(self.generate.call_args.args[1], "/kept/voice.wav")

    def test_wav_path_token_does_not_change_the_prompt(self):
        ytts_server.session.audio_prompt = "/kept/voice.wav"
        self.client.post(
            "/yttss", json={"text": "[aref: /tmp/x.wav] hi"}, headers=AUTH
        )
        self.assertEqual(ytts_server.session.audio_prompt, "/kept/voice.wav")
        self.assertEqual(self.generate.call_args.args[1], "/kept/voice.wav")

    def test_token_volume_does_not_stick(self):
        first = self.client.post(
            "/yttss", json={"text": "[vol: 0.4] hi"}, headers=AUTH
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(ytts_server.session.volume, 0.7)
        self.assertAlmostEqual(ytts_server.audio_queue[0]["volume"], 0.4)
        second = self.client.post("/yttss", json={"text": "hello"}, headers=AUTH)
        self.assertEqual(second.status_code, 200)
        self.assertAlmostEqual(ytts_server.audio_queue[1]["volume"], 0.7)

    def test_directive_volume_persists(self):
        updated = self.client.post(
            "/yttss", json={"directive": {"volume": 0.2}}, headers=AUTH
        )
        self.assertEqual(updated.status_code, 200)
        spoken = self.client.post("/yttss", json={"text": "hello"}, headers=AUTH)
        self.assertEqual(spoken.status_code, 200)
        self.assertAlmostEqual(ytts_server.session.volume, 0.2)
        self.assertAlmostEqual(ytts_server.audio_queue[-1]["volume"], 0.2)

    def test_boolean_volume_is_rejected(self):
        response = self.client.post(
            "/yttss", json={"directive": {"volume": True}}, headers=AUTH
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(ytts_server.session.volume, 0.7)

    def test_full_queue_returns_429_without_generating(self):
        for _ in range(MAX_QUEUE_ITEMS):
            ytts_server.audio_queue.append(
                {"buffer": io.BytesIO(b"x"), "volume": 1.0}
            )
        response = self.client.post("/yttss", json={"text": "hello"}, headers=AUTH)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.get_json()["status"], "queue full")
        self.generate.assert_not_called()

    def test_oversize_body_is_413(self):
        response = self.client.post(
            "/yttss", json={"text": "a" * 70000}, headers=AUTH
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"], "request too large")

    def test_text_over_character_cap_is_413(self):
        response = self.client.post(
            "/yttss", json={"text": "a" * (MAX_TEXT_CHARS + 1)}, headers=AUTH
        )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.get_json()["error"], "request too large")
        self.generate.assert_not_called()

    def test_flush_clears_the_queue(self):
        ytts_server.audio_queue.append({"buffer": io.BytesIO(b"x"), "volume": 1.0})
        response = self.client.post(
            "/yttss", json={"directive": {"flush": True}}, headers=AUTH
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "flushed")
        self.assertEqual(ytts_server.audio_queue, [])
        self.assertTrue(ytts_server.flush_requested)


if __name__ == "__main__":
    unittest.main()
