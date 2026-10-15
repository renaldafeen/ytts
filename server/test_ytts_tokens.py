"""Unit tests for token parsing plus speak-event chunking.

Run from server/: python -m unittest test_ytts_tokens.py
"""

import unittest

from ytts_tokens import chunk_speak_text, parse_tokens


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


if __name__ == "__main__":
    unittest.main()
