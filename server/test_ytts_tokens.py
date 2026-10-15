"""Unit tests for token parsing plus speak-event chunking.

Run from server/: python -m unittest test_ytts_tokens.py
"""

import unittest

from ytts_tokens import NUMPY_SEED_MAX, chunk_speak_text, parse_tokens


class ChunkSpeakTextTests(unittest.TestCase):
    def test_short_string_unchanged(self):
        self.assertEqual(chunk_speak_text("Hello there.", 20), ["Hello there."])

    def test_empty_and_whitespace(self):
        self.assertEqual(chunk_speak_text("", 20), [])
        self.assertEqual(chunk_speak_text("   \n\t  ", 20), [])
        self.assertEqual(chunk_speak_text(None, 20), [])

    def test_packs_short_sentences_until_max(self):
        text = "Hi. Hi. Hi. Hi. Hi. Hi."
        chunks = chunk_speak_text(text, 20)
        self.assertTrue(all(len(c) <= 20 for c in chunks))
        self.assertEqual(chunks[0], "Hi. Hi. Hi. Hi. Hi.")
        self.assertEqual(chunks[1], "Hi.")
        self.assertEqual(" ".join(chunks), "Hi. Hi. Hi. Hi. Hi. Hi.")

    def test_exclamation_and_question_are_sentence_ends(self):
        chunks = chunk_speak_text("Hello! Yes? OK.", 10)
        self.assertTrue(all(len(c) <= 10 for c in chunks))
        self.assertEqual(chunks, ["Hello!", "Yes? OK."])

    def test_long_sentence_splits_on_spaces(self):
        text = "This is a very long sentence without a period"
        chunks = chunk_speak_text(text, 20)
        self.assertTrue(all(len(c) <= 20 for c in chunks))
        self.assertTrue(len(chunks) > 1)
        self.assertEqual(" ".join(chunks), text)

    def test_oversize_word_is_hard_split(self):
        chunks = chunk_speak_text("abcdefghij", 4)
        self.assertEqual(chunks, ["abcd", "efgh", "ij"])
        self.assertTrue(all(len(c) <= 4 for c in chunks))

    def test_exactly_max_length_is_one_chunk(self):
        text = "a" * 20
        self.assertEqual(chunk_speak_text(text, 20), [text])

    def test_long_sentence_not_mixed_with_neighbors(self):
        # Short sentence, then a sentence too long for max, then a short one.
        long_sentence = "word " * 8 + "word."
        self.assertGreater(len(long_sentence), 20)
        text = "Hi. " + long_sentence + " Bye."
        chunks = chunk_speak_text(text, 20)
        self.assertTrue(all(len(c) <= 20 for c in chunks))
        self.assertEqual(chunks[0], "Hi.")
        self.assertEqual(chunks[-1], "Bye.")
        self.assertTrue(all("Hi." not in c or c == "Hi." for c in chunks))

    def test_production_cap_value(self):
        sentences = ["This is a complete sentence."] * 20
        text = " ".join(sentences)
        self.assertGreater(len(text), 300)
        chunks = chunk_speak_text(text, 300)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks), text)


class ParseThenChunkTests(unittest.TestCase):
    def test_aref_token_not_split_on_wav_dot(self):
        events = parse_tokens("Hello. [aref: /x/voice.wav] More text.")
        kinds = [event["kind"] for event in events]
        self.assertEqual(kinds, ["speak", "aref", "speak"])
        self.assertEqual(events[1]["path"], "/x/voice.wav")
        for event in events:
            if event["kind"] != "speak":
                continue
            for chunk in chunk_speak_text(event["text"], 20):
                self.assertNotIn("[", chunk)
                self.assertNotIn("]", chunk)
                self.assertNotIn(".wav", chunk)

    def test_blank_line_is_pause_not_a_chunk(self):
        events = parse_tokens("Hello\n\nWorld")
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["kind"], "speak")
        self.assertEqual(events[0]["text"], "Hello")
        self.assertEqual(events[1]["kind"], "pause")
        self.assertEqual(events[1]["seconds"], 1)
        self.assertEqual(events[2]["kind"], "speak")
        self.assertEqual(events[2]["text"], "World")
        self.assertEqual(chunk_speak_text(events[0]["text"], 20), ["Hello"])
        self.assertEqual(chunk_speak_text(events[2]["text"], 20), ["World"])


def _only(text):
    events = parse_tokens(text)
    if len(events) != 1:
        raise AssertionError(f"expected one event, got {events}")
    return events[0]


class PauseParseTests(unittest.TestCase):
    def test_fractional_pause_is_kept(self):
        event = _only("[pause: 0.5s]")
        self.assertEqual(event["kind"], "pause")
        self.assertEqual(event["seconds"], 0.5)
        self.assertFalse(event["adjusted"])

    def test_longer_fraction_is_not_truncated(self):
        event = _only("[pause: 2.9s]")
        self.assertEqual(event["seconds"], 2.9)
        self.assertFalse(event["adjusted"])

    def test_pause_above_cap_clamps(self):
        event = _only("[pause: 100s]")
        self.assertEqual(event["seconds"], 30)
        self.assertTrue(event["adjusted"])

    def test_zero_negative_and_garbage_are_ignored(self):
        for text in ("[pause: 0]", "[pause: -3s]", "[pause: banana]"):
            event = _only(text)
            self.assertEqual(event["kind"], "ignore", text)


class RpauseParseTests(unittest.TestCase):
    def test_reversed_range_swaps(self):
        event = _only("[rpause: 5-2]")
        self.assertEqual(event["kind"], "rpause")
        self.assertEqual(event["low"], 2)
        self.assertEqual(event["high"], 5)
        self.assertTrue(event["adjusted"])

    def test_high_end_above_cap_clamps(self):
        event = _only("[rpause: 1-100s]")
        self.assertEqual(event["low"], 1)
        self.assertEqual(event["high"], 30)
        self.assertTrue(event["adjusted"])


class ClosedRangeParseTests(unittest.TestCase):
    def test_volume_endpoints_accepted(self):
        for text, value in (("[vol: 0]", 0.0), ("[vol: 1]", 1.0), ("[vol: 0.5]", 0.5)):
            event = _only(text)
            self.assertEqual(event["kind"], "vol", text)
            self.assertEqual(event["value"], value)

    def test_volume_outside_and_non_numeric_ignored(self):
        for text in ("[vol: 1.5]", "[vol: true]", "[vol: -0.1]"):
            self.assertEqual(_only(text)["kind"], "ignore", text)

    def test_pace_endpoints_accepted_and_outside_ignored(self):
        self.assertEqual(_only("[pace: 0]")["kind"], "pace")
        self.assertEqual(_only("[pace: 1]")["value"], 1.0)
        self.assertEqual(_only("[pace: 1.1]")["kind"], "ignore")

    def test_exp_endpoints_accepted_and_outside_ignored(self):
        self.assertEqual(_only("[exp: 0.25]")["kind"], "exp")
        self.assertEqual(_only("[exp: 2.0]")["value"], 2.0)
        self.assertEqual(_only("[exp: 0.2]")["kind"], "ignore")
        self.assertEqual(_only("[exp: 2.1]")["kind"], "ignore")


class SeedParseTests(unittest.TestCase):
    def test_zero_adjusts_to_one(self):
        event = _only("[seed: 0]")
        self.assertEqual(event["kind"], "seed")
        self.assertEqual(event["value"], 1)
        self.assertTrue(event["adjusted"])

    def test_above_numpy_max_clamps(self):
        event = _only("[seed: 9999999999]")
        self.assertEqual(event["value"], NUMPY_SEED_MAX)
        self.assertTrue(event["adjusted"])


if __name__ == "__main__":
    unittest.main()
