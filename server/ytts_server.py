import argparse
import hashlib
import hmac
import logging
import os
import random
import sys
import threading
import warnings
from dataclasses import dataclass
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
import wave

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pygame
import torch
from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.serving import make_server

from chatterbox.tts import ChatterboxTTS
from ytts_tokens import (
    EXP_MAX,
    EXP_MIN,
    MAX_PAUSE_SECONDS,
    MAX_REF_BYTES,
    PACE_MAX,
    PACE_MIN,
    TTS_MAX_CHARS,
    VOL_MAX,
    VOL_MIN,
    chunk_speak_text,
    closed_float_ok,
    normalize_seed,
    parse_tokens,
)
from ytts_vmap import confine_voice_map, load_voice_map, resolve_aref_target, resolved_wav_path

warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.backends.cuda"
)
warnings.filterwarnings(
    "ignore", message=".*LlamaSdpaAttention.*scaled_dot_product_attention.*"
)
warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*sdp_kernel.*")

REPO_ROOT = Path(__file__).resolve().parent.parent
VOICES_ROOT = REPO_ROOT / "voices"

MAX_CONTENT_LENGTH = 64 * 1024
MAX_TEXT_CHARS = 4000
MAX_QUEUE_ITEMS = 32
MAX_QUEUE_BYTES = 32 * 1024 * 1024

TTS_TEMPERATURE = 0.8
TTS_MIN_P = 0.05
TTS_TOP_P = 1.0
TTS_REPETITION_PENALTY = 1.2

logger = logging.getLogger("ytts")
logger.setLevel(logging.INFO)
logger.addHandler(logging.NullHandler())

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

model = None
builtin_conds = None
prepared_path = None
sample_rate = 24000
model_ready = False
is_server_running = True
server = None
auth_token = None
voice_map = {}
flush_requested = False

# Separate from torch seeding so [rpause:] is not reset by each speech chunk.
pause_rng = random.Random()
generate_lock = threading.Lock()
params_lock = threading.Lock()
audio_lock = threading.Condition()
audio_queue = []


@dataclass
class SessionParams:
    volume: float = 0.7
    exaggeration: float = 0.35
    pacing: float = 0.8
    seed: int = 13
    audio_prompt: str | None = None
    voice_name: str | None = None


@dataclass
class ProcessResult:
    queued: int = 0
    failed: int = 0
    queue_full: bool = False


session = SessionParams()


def configure_logging(log_path):
    path = Path(log_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = RotatingFileHandler(
        path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def set_seed(seed: int):
    """Seed torch and numpy. Does not touch Python's random module."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def initialize_model(device=None):
    """Load Chatterbox once and remember the built-in voice conditionals."""
    global model, builtin_conds, prepared_path, sample_rate, model_ready
    if model is not None:
        return model
    if device is None:
        device = _select_device()
    logger.info("Using device: %s", device)
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
    except Exception:
        logger.exception("Failed to initialize model")
        sys.exit(1)
    builtin_conds = model.conds
    prepared_path = None
    sample_rate = model.sr
    model_ready = True
    logger.info("Model initialized on %s at %s Hz", device, sample_rate)
    return model


def _float_to_int16(audio_tensor):
    audio_np = np.asarray(audio_tensor.detach().cpu().numpy(), dtype=np.float32).flatten()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    return (audio_np * 32767.0).astype(np.int16)


def _wav_buffer(audio_np):
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer


def generate_tts(text, audio_prompt_path=None, exaggeration=0.35, pacing=0.8, seed=13):
    """Generate TTS audio and return a BytesIO WAV, or None on failure.

    The reference wav is encoded only when audio_prompt_path changes.
    Chatterbox reloads and replaces model.conds whenever a path is passed.
    """
    global model, prepared_path
    if len(text) > TTS_MAX_CHARS:
        logger.warning(
            "TTS text length %s exceeds %s characters", len(text), TTS_MAX_CHARS
        )
    if model is None:
        initialize_model()

    with generate_lock:
        set_seed(seed)
        use_path = audio_prompt_path
        if audio_prompt_path is None:
            if prepared_path is not None and builtin_conds is not None:
                model.conds = builtin_conds
            prepared_path = None
            use_path = None
        elif prepared_path == audio_prompt_path:
            use_path = None
        try:
            audio_tensor = model.generate(
                text=text,
                audio_prompt_path=use_path,
                temperature=TTS_TEMPERATURE,
                min_p=TTS_MIN_P,
                top_p=TTS_TOP_P,
                repetition_penalty=TTS_REPETITION_PENALTY,
                exaggeration=exaggeration,
                cfg_weight=pacing,
            )
        except Exception:
            logger.exception("TTS generation failed")
            return None
        if use_path is not None:
            prepared_path = audio_prompt_path

    return _wav_buffer(_float_to_int16(audio_tensor))


def make_silence_buffer(seconds):
    """Return a WAV BytesIO of at most MAX_PAUSE_SECONDS of int16 silence."""
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds < 0:
        seconds = 0.0
    if seconds > MAX_PAUSE_SECONDS:
        seconds = MAX_PAUSE_SECONDS
    frames = int(sample_rate * seconds)
    return _wav_buffer(np.zeros(frames, dtype=np.int16))


def _queue_has_item_room():
    with audio_lock:
        return len(audio_queue) < MAX_QUEUE_ITEMS


def enqueue_audio(buffer, volume):
    """Queue a WAV buffer. Return False when the queue cap is already reached."""
    with audio_lock:
        if len(audio_queue) >= MAX_QUEUE_ITEMS:
            buffer.close()
            return False
        total = 0
        for item in audio_queue:
            total += item["buffer"].getbuffer().nbytes
        if total + buffer.getbuffer().nbytes > MAX_QUEUE_BYTES:
            buffer.close()
            return False
        audio_queue.append({"buffer": buffer, "volume": float(volume)})
        audio_lock.notify()
        return True


def request_flush():
    """Drop queued audio and ask the playback thread to stop the current clip."""
    global flush_requested
    with audio_lock:
        for item in audio_queue:
            item["buffer"].close()
        audio_queue.clear()
        flush_requested = True
        audio_lock.notify_all()


def play_audio():
    """Play queued clips. The request thread never calls pygame."""
    global flush_requested
    clock = pygame.time.Clock()
    while is_server_running:
        with audio_lock:
            if flush_requested:
                for item in audio_queue:
                    item["buffer"].close()
                audio_queue.clear()
                flush_requested = False
                item = None
            elif not audio_queue:
                audio_lock.wait()
                continue
            else:
                item = audio_queue.pop(0)
        if item is None:
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            continue

        audio_buffer = item["buffer"]
        volume = item["volume"]
        if flush_requested:
            audio_buffer.close()
            with audio_lock:
                for queued in audio_queue:
                    queued["buffer"].close()
                audio_queue.clear()
                flush_requested = False
            continue
        if not 0.0 <= volume <= 1.0:
            logger.warning("Volume %s out of range [0.0, 1.0]. Clamping.", volume)
            volume = max(0.0, min(1.0, volume))
        try:
            pygame.mixer.music.set_volume(volume)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if flush_requested or not is_server_running:
                    pygame.mixer.music.stop()
                    break
                clock.tick(10)
        finally:
            audio_buffer.close()
        if flush_requested:
            with audio_lock:
                for queued in audio_queue:
                    queued["buffer"].close()
                audio_queue.clear()
                flush_requested = False


def _adjusted_suffix(event):
    if event.get("adjusted"):
        return f" (adjusted from {event.get('incoming')})"
    return ""


def _copy_session():
    return SessionParams(
        volume=session.volume,
        exaggeration=session.exaggeration,
        pacing=session.pacing,
        seed=session.seed,
        audio_prompt=session.audio_prompt,
        voice_name=session.voice_name,
    )


def _commit_session(params):
    global session
    with params_lock:
        session = params


def apply_aref(name, params):
    """Apply a voice-map name onto params. Unknown names leave params unchanged."""
    action, resolved = resolve_aref_target(name, voice_map)
    if action == "missing":
        logger.info("Token ignored: aref unknown voice %s", name)
        return False
    if action == "default":
        params.audio_prompt = None
        params.voice_name = None
        logger.info("Token action: aref default voice")
        return True
    real = resolved_wav_path(resolved, str(VOICES_ROOT), confine=True)
    if real is None:
        logger.info("Token ignored: aref rejected %s", name)
        return False
    params.audio_prompt = real
    params.voice_name = name
    logger.info("Token action: aref %s", name)
    return True


def process_tokenized_text(text, params, commit=False):
    """Parse tokens, generate speech and silence, and apply params.

    HTTP calls pass commit=False so token changes die with the request.
    The REPL and startup text pass commit=True.
    """
    result = ProcessResult()
    try:
        events = parse_tokens(text)
    except Exception:
        logger.exception("Token error: failed to parse tokens")
        result.failed += 1
        return result

    for event in events:
        if result.queue_full:
            break
        try:
            kind = event.get("kind")
            if kind == "speak":
                speak_text = event.get("text", "").strip()
                if not speak_text:
                    continue
                for chunk in chunk_speak_text(speak_text, TTS_MAX_CHARS):
                    if not _queue_has_item_room():
                        result.queue_full = True
                        break
                    audio_buffer = generate_tts(
                        chunk,
                        params.audio_prompt,
                        params.exaggeration,
                        params.pacing,
                        params.seed,
                    )
                    if not audio_buffer:
                        result.failed += 1
                        continue
                    if not enqueue_audio(audio_buffer, params.volume):
                        result.queue_full = True
                        break
                    result.queued += 1
                    logger.info("Audio generated successfully")
            elif kind == "pause":
                seconds = event["seconds"]
                logger.info("Token action: pause %ss%s", seconds, _adjusted_suffix(event))
                if not _queue_has_item_room():
                    result.queue_full = True
                    break
                if not enqueue_audio(make_silence_buffer(seconds), params.volume):
                    result.queue_full = True
                    break
                result.queued += 1
            elif kind == "rpause":
                low = event["low"]
                high = event["high"]
                seconds = pause_rng.uniform(low, high)
                logger.info(
                    "Token action: rpause %ss (range %s-%s)%s",
                    seconds,
                    low,
                    high,
                    _adjusted_suffix(event),
                )
                if not _queue_has_item_room():
                    result.queue_full = True
                    break
                if not enqueue_audio(make_silence_buffer(seconds), params.volume):
                    result.queue_full = True
                    break
                result.queued += 1
            elif kind == "vol":
                params.volume = event["value"]
                logger.info("Token action: vol %s", event["value"])
            elif kind == "pace":
                params.pacing = event["value"]
                logger.info("Token action: pace %s", event["value"])
            elif kind == "exp":
                params.exaggeration = event["value"]
                logger.info("Token action: exp %s", event["value"])
            elif kind == "seed":
                params.seed = event["value"]
                logger.info(
                    "Token action: seed %s%s",
                    event["value"],
                    _adjusted_suffix(event),
                )
            elif kind == "aref":
                apply_aref(event["path"], params)
            elif kind == "ignore":
                logger.info(
                    "Token ignored: %s (%s)", event.get("raw"), event.get("reason")
                )
            else:
                logger.info("Token error: unknown event kind %r", kind)
        except Exception:
            logger.exception("Token error")
            result.failed += 1
            continue
    if commit:
        _commit_session(params)
    return result


def _current_parameters():
    with params_lock:
        return {
            "volume": session.volume,
            "exaggeration": session.exaggeration,
            "pacing": session.pacing,
            "seed": session.seed,
            "voice": session.voice_name,
        }


def _authorized():
    header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not auth_token or not header.startswith(prefix):
        return False
    provided = header[len(prefix):]
    if provided == "":
        return False
    expected_digest = hashlib.sha256(auth_token.encode("utf-8")).digest()
    provided_digest = hashlib.sha256(provided.encode("utf-8")).digest()
    return hmac.compare_digest(expected_digest, provided_digest)


def _speech_response(result):
    parameters = _current_parameters()
    if result.queue_full:
        return jsonify({
            "status": "queue full",
            "queued": result.queued,
            "parameters": parameters,
        }), 429
    if result.queued == 0 and result.failed > 0:
        return jsonify({"error": "synthesis failed"}), 500
    if result.queued > 0 and result.failed > 0:
        return jsonify({
            "status": "partial",
            "queued": result.queued,
            "failed": result.failed,
            "parameters": parameters,
        }), 200
    if result.queued > 0:
        return jsonify({"status": "audio queued", "parameters": parameters}), 200
    return jsonify({"status": "no audio", "parameters": parameters}), 200


def _voice_update(name):
    """Return (audio_prompt, voice_name) or None when the name is unusable."""
    cleaned = name.strip()
    if cleaned == "default":
        return None, None
    action, resolved = resolve_aref_target(cleaned, voice_map)
    if action == "missing":
        return None
    if action == "default":
        return None, None
    real = resolved_wav_path(resolved, str(VOICES_ROOT), confine=True)
    if real is None:
        return None
    return real, cleaned


def _validate_directive(directive):
    """Validate every field before any session write. Return (pending, error)."""
    if not isinstance(directive, dict):
        return None, (jsonify({"error": "No valid parameters provided in directive"}), 400)

    pending = {}
    recognized = False

    if "volume" in directive:
        recognized = True
        if not closed_float_ok(directive["volume"], VOL_MIN, VOL_MAX):
            return None, (jsonify({"error": "Volume must be between 0.0 and 1.0"}), 400)
        pending["volume"] = float(directive["volume"])
    if "exaggeration" in directive:
        recognized = True
        if not closed_float_ok(directive["exaggeration"], EXP_MIN, EXP_MAX):
            return None, (jsonify({"error": "Exaggeration must be between 0.25 and 2.0"}), 400)
        pending["exaggeration"] = float(directive["exaggeration"])
    if "pacing" in directive:
        recognized = True
        if not closed_float_ok(directive["pacing"], PACE_MIN, PACE_MAX):
            return None, (jsonify({"error": "Pacing must be between 0.0 and 1.0"}), 400)
        pending["pacing"] = float(directive["pacing"])
    if "seed" in directive:
        recognized = True
        normalized = normalize_seed(directive["seed"])
        if normalized is None:
            return None, (jsonify({"error": "Seed must be a positive number"}), 400)
        pending["seed"] = normalized[0]
    if "voice" in directive:
        recognized = True
        voice = directive["voice"]
        if not isinstance(voice, str) or not voice.strip():
            return None, (jsonify({"error": "Unknown voice"}), 400)
        updated = _voice_update(voice)
        if updated is None:
            return None, (jsonify({"error": "Unknown voice"}), 400)
        pending["audio_prompt"], pending["voice_name"] = updated

    flush = False
    if "flush" in directive:
        recognized = True
        if not isinstance(directive["flush"], bool):
            return None, (jsonify({"error": "flush must be a boolean"}), 400)
        flush = directive["flush"]
    shutdown = False
    if "shutdown" in directive:
        recognized = True
        if not isinstance(directive["shutdown"], bool):
            return None, (jsonify({"error": "shutdown must be a boolean"}), 400)
        shutdown = directive["shutdown"]

    if not pending and not flush and not shutdown:
        return None, (jsonify({"error": "No valid parameters provided in directive"}), 400)
    if not recognized:
        return None, (jsonify({"error": "No valid parameters provided in directive"}), 400)
    return {"pending": pending, "flush": flush, "shutdown": shutdown}, None


def request_shutdown():
    global is_server_running
    is_server_running = False
    request_flush()
    current = server
    if current is not None:
        threading.Thread(target=current.shutdown, daemon=True).start()


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_exc):
    return jsonify({"error": "request too large"}), 413


@app.route("/health", methods=["GET"])
def health():
    if model_ready:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "loading"}), 503


@app.route("/yttss", methods=["POST"])
def tts_endpoint():
    """Handle HTTP POST requests for TTS or parameter updates."""
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Missing JSON payload"}), 400

        if "directive" in data:
            validated, error = _validate_directive(data["directive"])
            if error is not None:
                return error
            with params_lock:
                for key, value in validated["pending"].items():
                    setattr(session, key, value)
            if validated["flush"] or validated["shutdown"]:
                request_flush()
            if validated["shutdown"]:
                request_shutdown()
                return jsonify({"status": "shutting down"}), 200
            if validated["flush"] and not validated["pending"]:
                status = "flushed"
            elif validated["flush"]:
                status = "flushed"
            else:
                status = "Parameters updated"
            logger.info("Updated parameters: %s", status)
            return jsonify({"status": status, "parameters": _current_parameters()}), 200

        if "text" not in data:
            return jsonify({"error": "Missing text in JSON payload"}), 400
        text = data["text"]
        if not isinstance(text, str):
            return jsonify({"error": "Text must be a string"}), 400
        if len(text) > MAX_TEXT_CHARS:
            return jsonify({"error": "request too large"}), 413

        logger.info(
            "Received text: %s chars remote=%s", len(text), request.remote_addr
        )
        with params_lock:
            params = _copy_session()
        result = process_tokenized_text(text, params, commit=False)
        return _speech_response(result)
    except RequestEntityTooLarge:
        raise
    except Exception:
        logger.exception("request failed")
        return jsonify({"error": "request failed"}), 500


def init_mixer():
    try:
        pygame.mixer.init(frequency=sample_rate)
    except Exception:
        logger.exception("Failed to initialize pygame mixer")
        sys.exit(1)


def start_audio_thread():
    audio_thread = threading.Thread(target=play_audio, name="ytts-playback")
    audio_thread.daemon = True
    audio_thread.start()
    return audio_thread


def run_server(host, port, ssl_context=None):
    """Listen until shutdown. The model is already loaded."""
    global server
    scheme = "https" if ssl_context else "http"
    logger.info("Starting HTTP server on %s://%s:%s/yttss", scheme, host, port)
    server = make_server(host, port, app, ssl_context=ssl_context)
    server.serve_forever()


def _resolve_user_path(path):
    expanded = os.path.expanduser(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(str(REPO_ROOT), expanded)
    return expanded


def _require_closed(name, value, lo, hi):
    if not closed_float_ok(value, lo, hi):
        logger.error("%s must be between %s and %s", name, lo, hi)
        sys.exit(1)


def _log_startup_result(result):
    if result.queued > 0:
        logger.info("Initial audio queued (%s)", result.queued)
    elif result.failed:
        logger.error("Initial audio failed")
    else:
        logger.info("Initial text produced no audio")


def main():
    parser = argparse.ArgumentParser(
        description="HTTP-based TTS server with custom pacing and volume"
    )
    parser.add_argument("--text", default="", help="Optional startup text to synthesize")
    parser.add_argument(
        "--aref",
        default=None,
        help="Startup reference WAV. Relative paths resolve from the repo root.",
    )
    parser.add_argument("--exaggeration", type=float, default=0.35)
    parser.add_argument("--pacing", type=float, default=0.8)
    parser.add_argument("--volume", type=float, default=0.7)
    parser.add_argument("--server", action="store_true", help="Listen for HTTP requests")
    parser.add_argument("--port", type=int, default=9126)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--vmap", default=None, help="Voice-name JSON map")
    parser.add_argument("--tls-cert", default=None)
    parser.add_argument("--tls-key", default=None)
    parser.add_argument(
        "--log-file",
        default=str(REPO_ROOT / "ytts_server.log"),
        help="Rotating log file",
    )
    args = parser.parse_args()
    configure_logging(args.log_file)

    token = os.environ.get("YTTS_TOKEN", "")
    if token == "":
        logger.error("YTTS_TOKEN is required")
        sys.exit(1)
    global auth_token, voice_map, session
    auth_token = token

    if bool(args.tls_cert) != bool(args.tls_key):
        logger.error("--tls-cert and --tls-key must be set together")
        sys.exit(1)
    ssl_context = None
    if args.tls_cert and args.tls_key:
        cert = _resolve_user_path(args.tls_cert)
        key = _resolve_user_path(args.tls_key)
        if not os.path.isfile(cert) or not os.path.isfile(key):
            logger.error("TLS certificate or key file is missing")
            sys.exit(1)
        ssl_context = (cert, key)

    _require_closed("volume", args.volume, VOL_MIN, VOL_MAX)
    _require_closed("exaggeration", args.exaggeration, EXP_MIN, EXP_MAX)
    _require_closed("pacing", args.pacing, PACE_MIN, PACE_MAX)
    normalized_seed = normalize_seed(args.seed)
    if normalized_seed is None:
        logger.error("seed is not usable")
        sys.exit(1)
    seed_value, seed_adjusted, _incoming = normalized_seed
    if seed_adjusted:
        logger.info("seed adjusted to %s", seed_value)

    voice_map = {}
    if args.vmap:
        loaded = load_voice_map(_resolve_user_path(args.vmap))
        voice_map = confine_voice_map(loaded, str(VOICES_ROOT), MAX_REF_BYTES)
        for key, value in voice_map.items():
            logger.info("Voice map: %s -> %s", key, value)

    audio_prompt = None
    if args.aref:
        audio_prompt = resolved_wav_path(
            args.aref, str(VOICES_ROOT), max_bytes=MAX_REF_BYTES, confine=False
        )
        if audio_prompt is None:
            logger.error("startup reference is not a usable wav")
            sys.exit(1)
        logger.info("Startup reference: %s", audio_prompt)

    with params_lock:
        session = SessionParams(
            volume=float(args.volume),
            exaggeration=float(args.exaggeration),
            pacing=float(args.pacing),
            seed=seed_value,
            audio_prompt=audio_prompt,
            voice_name=None,
        )

    initialize_model()
    init_mixer()
    start_audio_thread()

    if args.text:
        with params_lock:
            params = _copy_session()
        _log_startup_result(process_tokenized_text(args.text, params, commit=True))

    if args.server:
        run_server(args.host, args.port, ssl_context)
        return

    while True:
        try:
            user_input = input("Say: ")
        except EOFError:
            logger.info("Exiting")
            break
        if user_input.strip().lower() == "quit":
            logger.info("Exiting")
            break
        with params_lock:
            params = _copy_session()
        result = process_tokenized_text(user_input, params, commit=True)
        if result.failed and result.queued == 0:
            logger.error("Synthesis failed")


if __name__ == "__main__":
    main()
