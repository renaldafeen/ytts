"""Voice-name map for [aref:] tokens.

JSON file of name -> WAV path (or the sentinel \"default\"). This module has
no TTS / Flask / torch imports so it can be exercised without Chatterbox.
"""

import json
import os


def load_voice_map(path):
    """Read a voice-name JSON map. Never raises; logs errors and returns {}."""
    if path is None:
        return {}
    if not isinstance(path, str) or path.strip() == "":
        print("Voice map error: --vmap specified but filename is missing")
        return {}

    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        print(f"Voice map error: file not found: {path}")
        return {}

    try:
        with open(expanded, "r", encoding="utf-8") as handle:
            raw = handle.read()
    except OSError as exc:
        print(f"Voice map error: cannot read {path}: {exc}")
        return {}

    duplicates = []

    def _pairs_hook(pairs):
        seen = {}
        for key, value in pairs:
            if key in seen:
                duplicates.append(key)
            seen[key] = value
        return seen

    try:
        data = json.loads(raw, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as exc:
        print(f"Voice map error: cannot read {path}: {exc}")
        return {}

    if not isinstance(data, dict):
        print(f"Voice map error: cannot read {path}: top-level JSON must be an object")
        return {}

    for key in duplicates:
        print(f"Voice map error: duplicate key: {key}")

    result = {}
    for key, value in data.items():
        if not isinstance(value, str):
            print(f"Voice map error: non-string value for key {key}")
            continue
        result[key] = value
        print(f"Voice map: {key} -> {value}")
    return result


def resolve_aref_target(target, voice_map):
    """Resolve an [aref:] target to a WAV path or the default voice.

    Returns (\"path\", path_string) or (\"default\", None).
    """
    if target is None:
        target = ""
    if not isinstance(target, str):
        target = str(target)
    target = target.strip()

    if target.lower().endswith(".wav"):
        return ("path", target)

    if not voice_map:
        return ("default", None)

    if target in voice_map:
        value = voice_map[target]
        if isinstance(value, str):
            value = value.strip()
        if value == "default":
            return ("default", None)
        return ("path", value)

    return ("default", None)
