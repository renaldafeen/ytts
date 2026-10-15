"""Voice-name map for [aref:] tokens.

JSON file of name -> WAV path (or the sentinel \"default\"). This module has
no TTS / Flask / torch imports so it can be exercised without Chatterbox.
"""

import json
import os
import wave

from ytts_tokens import MAX_REF_BYTES


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
    """Resolve an [aref:] target to a map entry.

    Returns (\"path\", path_string), (\"default\", None), or (\"missing\", None).
    Only map keys match. A value that ends in .wav is not treated as a path.
    \"default\" is returned only when the mapped value is that sentinel.
    """
    if target is None:
        target = ""
    if not isinstance(target, str):
        target = str(target)
    target = target.strip()

    if not voice_map or target not in voice_map:
        return ("missing", None)

    value = voice_map[target]
    if isinstance(value, str):
        value = value.strip()
    if value == "default":
        return ("default", None)
    return ("path", value)


def _repo_root_for_voices(voices_root):
    return os.path.dirname(os.path.realpath(voices_root))


def resolved_wav_path(value, voices_root, max_bytes=MAX_REF_BYTES, confine=True):
    """Return a real WAV path, or None when the value cannot be used.

    Relative paths resolve against the parent of voices_root (the repo root).
    When confine is true, the real path must sit inside voices_root. Symlinks
    are followed by realpath, so a link that escapes the directory is rejected.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    expanded = os.path.expanduser(value.strip())
    if expanded == "" or expanded == "default":
        return None
    if not os.path.isabs(expanded):
        expanded = os.path.join(_repo_root_for_voices(voices_root), expanded)
    real = os.path.realpath(expanded)

    if confine:
        root = os.path.realpath(voices_root)
        prefix = root + os.sep
        if real != root and not real.startswith(prefix):
            print(f"Voice map rejected: outside voices: {value}")
            return None

    if not real.lower().endswith(".wav"):
        print(f"Voice map rejected: not a wav: {value}")
        return None
    if not os.path.isfile(real):
        print(f"Voice map rejected: file not found: {value}")
        return None
    try:
        size = os.path.getsize(real)
    except OSError as exc:
        print(f"Voice map rejected: cannot stat {value}: {exc}")
        return None
    if size > max_bytes:
        print(f"Voice map rejected: too large: {value}")
        return None
    try:
        with wave.open(real, "rb"):
            pass
    except (wave.Error, OSError, EOFError) as exc:
        print(f"Voice map rejected: cannot open wav {value}: {exc}")
        return None
    return real


def confine_voice_map(voice_map, voices_root, max_bytes=MAX_REF_BYTES):
    """Keep default sentinels and WAV paths that stay inside voices_root.

    Rejected entries are logged and omitted. load_voice_map is unchanged;
    call this after it.
    """
    if not voice_map:
        return {}
    result = {}
    for key, value in voice_map.items():
        if isinstance(value, str) and value.strip() == "default":
            result[key] = "default"
            print(f"Voice map confined: {key} -> default")
            continue
        real = resolved_wav_path(value, voices_root, max_bytes=max_bytes, confine=True)
        if real is None:
            continue
        result[key] = real
        print(f"Voice map confined: {key} -> {real}")
    return result
