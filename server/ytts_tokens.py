"""Parse ytts inline tokens from incoming TTS text.

Tokens are square-bracket spans that are never spoken. This module has no
TTS / Flask / torch imports so it can be exercised without the Chatterbox venv.
"""

import math
import re
import sys
from decimal import Decimal, InvalidOperation

# numpy.random.seed only accepts 0 .. 2**32-1; after forcing >0 the max is this.
NUMPY_SEED_MAX = 2 ** 32 - 1

TOKEN_RE = re.compile(r"\[[^\]]*\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
NAME_VALUE_RE = re.compile(r"^([A-Za-z]+)\s*:\s*(.*)$")
NUMBER_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$"
)
PAUSE_VALUE_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*s?$",
    re.IGNORECASE,
)
RPAUSE_VALUE_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*-\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*s?$",
    re.IGNORECASE,
)

KNOWN_TOKENS = ("pause", "rpause", "vol", "pace", "exp", "seed", "aref")

VOL_MIN, VOL_MAX = 0.0, 1.0
PACE_MIN, PACE_MAX = 0.0, 1.0
EXP_MIN, EXP_MAX = 0.25, 2.0
MAX_PAUSE_SECONDS = 30.0
# Shipped references are about 3–8 MB. Chatterbox uses the first 10 seconds
# after the file is opened, so the cap only has to stop a huge read.
MAX_REF_BYTES = 16 * 1024 * 1024
TTS_MAX_CHARS = 300


def normalize_int(incoming):
    """Truncate, force >0, and clamp to sys.maxsize.

    Returns (value, adjusted, incoming_str) or None if incoming is not numeric.
    """
    incoming_str = incoming if isinstance(incoming, str) else str(incoming)
    try:
        decimal_value = Decimal(incoming_str)
    except (InvalidOperation, ValueError):
        return None
    if not decimal_value.is_finite():
        return None

    truncated = int(decimal_value)
    value = truncated
    if value <= 0:
        value = 1
    if value > sys.maxsize:
        value = sys.maxsize
    adjusted = (value != truncated) or (Decimal(truncated) != decimal_value)
    return value, adjusted, incoming_str


def normalize_seed(incoming):
    """Integer normalize, then clamp into numpy's seed range."""
    result = normalize_int(incoming)
    if result is None:
        return None
    value, adjusted, incoming_str = result
    if value > NUMPY_SEED_MAX:
        value = NUMPY_SEED_MAX
        adjusted = True
    return value, adjusted, incoming_str


def _blank_pause():
    return {
        "kind": "pause",
        "seconds": 1,
        "adjusted": False,
        "incoming": "1",
        "raw": "<blank line>",
    }


def _ignore(raw, reason):
    return {"kind": "ignore", "reason": reason, "raw": raw}


def closed_float_ok(value, lo, hi):
    """True when value is a finite real number inside the closed interval.

    Booleans are rejected. bool is a subclass of int, so a JSON true would
    otherwise pass a numeric range check.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    value = float(value)
    return math.isfinite(value) and lo <= value <= hi


def _parse_pause_seconds(number_text):
    """Return (seconds, adjusted) or None when the duration must be ignored.

    Positive finite values are kept. Values above MAX_PAUSE_SECONDS clamp to
    that ceiling. Zero, negative, and non-finite values are rejected.
    """
    try:
        value = float(number_text)
    except ValueError:
        return None
    if not math.isfinite(value) or value <= 0:
        return None
    adjusted = False
    if value > MAX_PAUSE_SECONDS:
        value = MAX_PAUSE_SECONDS
        adjusted = True
    return value, adjusted


def _parse_closed_float(raw, rest, kind, lo, hi):
    match = NUMBER_RE.match(rest.strip())
    if not match:
        return _ignore(raw, "malformed")
    try:
        value = float(match.group(1))
    except ValueError:
        return _ignore(raw, "malformed")
    if not math.isfinite(value) or value < lo or value > hi:
        return _ignore(raw, f"out of range [{lo}, {hi}]")
    return {"kind": kind, "value": value, "raw": raw}


def parse_one_token(raw):
    """Turn a single `[...]` span into an event dict."""
    inner = raw[1:-1].strip()
    if not inner:
        return _ignore(raw, "malformed")

    name_value = NAME_VALUE_RE.match(inner)
    if not name_value:
        bare = inner.lower()
        if bare in KNOWN_TOKENS:
            return _ignore(raw, "malformed")
        return _ignore(raw, "unrecognized")

    name = name_value.group(1).lower()
    rest = name_value.group(2)

    if name == "pause":
        match = PAUSE_VALUE_RE.match(rest.strip())
        if not match:
            return _ignore(raw, "malformed")
        parsed = _parse_pause_seconds(match.group(1))
        if parsed is None:
            return _ignore(raw, "malformed")
        seconds, adjusted = parsed
        return {
            "kind": "pause",
            "seconds": seconds,
            "adjusted": adjusted,
            "incoming": match.group(1),
            "raw": raw,
        }

    if name == "rpause":
        match = RPAUSE_VALUE_RE.match(rest.strip())
        if not match:
            return _ignore(raw, "malformed")
        low_parsed = _parse_pause_seconds(match.group(1))
        high_parsed = _parse_pause_seconds(match.group(2))
        if low_parsed is None or high_parsed is None:
            return _ignore(raw, "malformed")
        low, low_adj = low_parsed
        high, high_adj = high_parsed
        swapped = False
        if low > high:
            low, high = high, low
            swapped = True
        return {
            "kind": "rpause",
            "low": low,
            "high": high,
            "adjusted": low_adj or high_adj or swapped,
            "incoming": f"{match.group(1)}-{match.group(2)}",
            "raw": raw,
        }

    if name == "vol":
        return _parse_closed_float(raw, rest, "vol", VOL_MIN, VOL_MAX)
    if name == "pace":
        return _parse_closed_float(raw, rest, "pace", PACE_MIN, PACE_MAX)
    if name == "exp":
        return _parse_closed_float(raw, rest, "exp", EXP_MIN, EXP_MAX)

    if name == "seed":
        match = NUMBER_RE.match(rest.strip())
        if not match:
            return _ignore(raw, "malformed")
        normalized = normalize_seed(match.group(1))
        if normalized is None:
            return _ignore(raw, "malformed")
        value, adjusted, incoming = normalized
        return {
            "kind": "seed",
            "value": value,
            "adjusted": adjusted,
            "incoming": incoming,
            "raw": raw,
        }

    if name == "aref":
        path = rest.strip()
        if not path:
            return _ignore(raw, "malformed")
        return {"kind": "aref", "path": path, "raw": raw}

    return _ignore(raw, "unrecognized")


def _emit_text_with_blank_lines(text):
    if not text:
        return []
    if "\n" not in text and "\r" not in text:
        return [{"kind": "speak", "text": text}]

    events = []
    for line in text.splitlines():
        if line.strip() == "":
            events.append(_blank_pause())
        else:
            events.append({"kind": "speak", "text": line})
    return events


def parse_tokens(text):
    """Parse raw request text into an ordered list of speak/token events.

    Empty or whitespace-only payloads become a one-second pause. Tokens are
    stripped from spoken text. Blank lines in non-token gaps are 1s pauses.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)

    if text.strip() == "":
        return [_blank_pause()]

    events = []
    pos = 0
    for match in TOKEN_RE.finditer(text):
        if match.start() > pos:
            events.extend(_emit_text_with_blank_lines(text[pos:match.start()]))
        events.append(parse_one_token(match.group(0)))
        pos = match.end()
    if pos < len(text):
        events.extend(_emit_text_with_blank_lines(text[pos:]))
    return events


def _hard_split(text, max_length):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def _split_on_spaces(text, max_length):
    """Pack words into pieces of at most max_length; hard-split oversize words."""
    chunks = []
    current = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    for word in text.split():
        if len(word) > max_length:
            flush()
            chunks.extend(_hard_split(word, max_length))
            continue
        added = len(word) if not current else 1 + len(word)
        if current_len + added <= max_length:
            current.append(word)
            current_len += added
        else:
            flush()
            current.append(word)
            current_len = len(word)
    flush()
    return chunks


def chunk_speak_text(text, max_length):
    """Split spoken text into chunks of at most max_length characters.

    Prefers sentence boundaries (. ! ?). If a sentence is still too long,
    splits on spaces. A single oversize word is hard-split at max_length
    so generate_tts never receives oversize input. Tokens are not handled
    here; call parse_tokens first and chunk only speak events.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.split())
    if not text:
        return []
    if max_length < 1:
        return [text]
    if len(text) <= max_length:
        return [text]

    result = []
    current = ""

    def flush():
        nonlocal current
        if current:
            result.append(current)
            current = ""

    def pack_sentence(sentence):
        nonlocal current
        if not current:
            current = sentence
            return
        joined = current + " " + sentence
        if len(joined) <= max_length:
            current = joined
        else:
            flush()
            current = sentence

    for sentence in SENTENCE_SPLIT_RE.split(text):
        if not sentence:
            continue
        if len(sentence) <= max_length:
            pack_sentence(sentence)
        else:
            flush()
            result.extend(_split_on_spaces(sentence, max_length))
    flush()
    return result
