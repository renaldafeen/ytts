import os
import random
import sys
import threading
import warnings
from io import BytesIO
import wave
import numpy as np
import torch
import pygame
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.serving import make_server
import argparse
from chatterbox.tts import ChatterboxTTS
from ytts_tokens import chunk_speak_text, parse_tokens
from ytts_vmap import load_voice_map, resolve_aref_target

# Suppress warnings for cleaner console
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=FutureWarning,
                        module="torch.backends.cuda")
warnings.filterwarnings(
    "ignore", message=".*LlamaSdpaAttention.*scaled_dot_product_attention.*")
warnings.filterwarnings(
    "ignore", message=".*past_key_values.*tuple of tuples.*")

# Suppress pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Flask app
app = Flask(__name__)
# Allow all origins for testing
CORS(app, resources={r"/yttss": {"origins": "*"}})

# Global model and sample rate
model = None
sample_rate = 24000
is_server_running = True  # Global flag for server shutdown
server = None  # Global server instance for shutdown
cli_args = None  # Global args for dynamic updates
voice_map = {}  # Voice name -> audio reference path (from --vmap)

# Thread-safe queue for audio items: {"buffer": BytesIO, "volume": float}
audio_queue = []
audio_lock = threading.Condition()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def initialize_model(device="cuda"):
    """Initialize the global model if not already loaded."""
    global model
    if model is None:
        try:
            model = ChatterboxTTS.from_pretrained(device=device)
            model.seed = 42
            print(f"Model initialized on {device}")
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            sys.exit(1)
    return model


def _current_seed():
    if cli_args is not None and getattr(cli_args, "seed", None) is not None:
        return cli_args.seed
    return 13


TTS_MAX_CHARS = 300  # Chatterbox limit; the only copy of this number


def generate_tts(text, audio_prompt_path=None, exaggeration=1.0, pacing=0.5):
    """Generate TTS audio and return a BytesIO buffer."""
    global model
    if len(text) > TTS_MAX_CHARS:
        print(
            f"Warning: TTS text length {len(text)} exceeds "
            f"{TTS_MAX_CHARS} characters"
        )
    if model is None:
        device = "cuda" if torch.cuda.is_available(
        ) else "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {device}")
        initialize_model(device)

    set_seed(_current_seed())
    try:
        audio_tensor = model.generate(
            text=text,
            audio_prompt_path=audio_prompt_path,
            temperature=0.8,
            min_p=0.05,
            top_p=1.0,
            repetition_penalty=1.2,
            exaggeration=exaggeration,
            cfg_weight=pacing,
            progress_bar=False
        )
    except TypeError as e:
        if "progress_bar" in str(e):
            audio_tensor = model.generate(
                text=text,
                audio_prompt_path=audio_prompt_path,
                temperature=0.8,
                min_p=0.05,
                top_p=1.0,
                repetition_penalty=1.2,
                exaggeration=exaggeration,
                cfg_weight=pacing
            )
        else:
            raise e
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None

    audio_np = audio_tensor.cpu().numpy().flatten()
    if audio_np.dtype == np.float32:
        max_amplitude = np.max(np.abs(audio_np))
        if max_amplitude > 0:
            audio_np = audio_np / max_amplitude
        audio_np = (audio_np * 32767).astype(np.int16)

    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer


def make_silence_buffer(seconds):
    """Return a WAV BytesIO of `seconds` of int16 silence."""
    frames = int(sample_rate * max(0, seconds))
    audio_np = np.zeros(frames, dtype=np.int16)
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_np.tobytes())
    buffer.seek(0)
    return buffer


def enqueue_audio(buffer, volume=None):
    """Queue a WAV buffer, snapshotting playback volume at enqueue time."""
    if volume is None:
        volume = cli_args.volume if cli_args is not None else 1.0
    with audio_lock:
        audio_queue.append({"buffer": buffer, "volume": volume})
        audio_lock.notify()


def play_audio():
    """Play audio from the queue with the volume snapshotted on each item."""
    global is_server_running
    while is_server_running:
        with audio_lock:
            if not audio_queue:
                audio_lock.wait()
                continue
            item = audio_queue.pop(0)

        audio_buffer = item["buffer"]
        volume = item["volume"]
        if not 0.0 <= volume <= 1.0:
            print(
                f"Warning: Volume {volume} out of range [0.0, 1.0]. Clamping.")
            volume = max(0.0, min(1.0, volume))

        pygame.mixer.music.set_volume(volume)
        pygame.mixer.music.load(audio_buffer)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        audio_buffer.close()


def _adjusted_suffix(event):
    if event.get("adjusted"):
        return f" (adjusted from {event.get('incoming')})"
    return ""


def apply_aref(path):
    """Validate and apply an [aref:] path or voice name. Returns True if applied."""
    action, resolved = resolve_aref_target(path, voice_map)
    if action == "default":
        cli_args.audio_prompt = None
        print("Token action: aref default voice")
        return True
    path = resolved
    expanded = os.path.expanduser(path)
    if not os.path.isfile(expanded):
        print(f"Token error: aref file not found: {path}")
        return False
    if not expanded.lower().endswith(".wav"):
        print(f"Token error: aref is not a .WAV file: {path}")
        return False
    try:
        with wave.open(expanded, "rb"):
            pass
    except Exception as e:
        print(f"Token error: aref failed to open as WAV {path}: {e}")
        return False
    cli_args.audio_prompt = expanded
    print(f"Token action: aref {expanded}")
    return True


def process_tokenized_text(text):
    """Parse tokens in `text`, generate/enqueue speech and silence, apply params.

    Never raises for token-level errors; logs via print() and continues.
    Returns True if at least one audio (speech or silence) buffer was enqueued.
    """
    enqueued = False
    try:
        events = parse_tokens(text)
    except Exception as e:
        print(f"Token error: failed to parse tokens: {e}")
        return False

    for event in events:
        try:
            kind = event.get("kind")
            if kind == "speak":
                speak_text = event.get("text", "").strip()
                if not speak_text:
                    continue
                for chunk in chunk_speak_text(speak_text, TTS_MAX_CHARS):
                    audio_buffer = generate_tts(
                        chunk,
                        cli_args.audio_prompt,
                        cli_args.exaggeration,
                        cli_args.pacing,
                    )
                    if audio_buffer:
                        enqueue_audio(audio_buffer)
                        enqueued = True
                        print("Audio generated successfully")
                    else:
                        print("TTS generation failed")
            elif kind == "pause":
                seconds = event["seconds"]
                print(
                    f"Token action: pause {seconds}s{_adjusted_suffix(event)}")
                enqueue_audio(make_silence_buffer(seconds))
                enqueued = True
            elif kind == "rpause":
                low = event["low"]
                high = event["high"]
                seconds = random.randint(low, high)
                print(
                    f"Token action: rpause {seconds}s (range {low}-{high})"
                    f"{_adjusted_suffix(event)}"
                )
                enqueue_audio(make_silence_buffer(seconds))
                enqueued = True
            elif kind == "vol":
                cli_args.volume = event["value"]
                print(f"Token action: vol {event['value']}")
            elif kind == "pace":
                cli_args.pacing = event["value"]
                print(f"Token action: pace {event['value']}")
            elif kind == "exp":
                cli_args.exaggeration = event["value"]
                print(f"Token action: exp {event['value']}")
            elif kind == "seed":
                cli_args.seed = event["value"]
                print(
                    f"Token action: seed {event['value']}"
                    f"{_adjusted_suffix(event)}"
                )
            elif kind == "aref":
                apply_aref(event["path"])
            elif kind == "ignore":
                print(
                    f"Token ignored: {event.get('raw')} ({event.get('reason')})")
            else:
                print(f"Token error: unknown event kind {kind!r}")
        except Exception as e:
            print(f"Token error: {e}")
            continue
    return enqueued


def _current_parameters():
    return {
        "volume": cli_args.volume,
        "exaggeration": cli_args.exaggeration,
        "pacing": cli_args.pacing,
        "seed": cli_args.seed,
        "audio_prompt": cli_args.audio_prompt,
    }


@app.route('/yttss', methods=['POST'])
def tts_endpoint():
    """Handle HTTP POST requests for TTS or parameter updates."""
    global is_server_running, server, cli_args
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON payload'}), 400

        # Handle directive to update parameters
        if 'directive' in data:
            directive = data['directive']
            updated = []
            if 'volume' in directive:
                if isinstance(directive['volume'], (int, float)) and 0.0 <= directive['volume'] <= 1.0:
                    cli_args.volume = float(directive['volume'])
                    updated.append(f"volume={cli_args.volume}")
                else:
                    return jsonify({'error': 'Volume must be between 0.0 and 1.0'}), 400
            if 'exaggeration' in directive:
                if isinstance(directive['exaggeration'], (int, float)) and 0.25 <= directive['exaggeration'] <= 2.0:
                    cli_args.exaggeration = float(directive['exaggeration'])
                    updated.append(f"exaggeration={cli_args.exaggeration}")
                else:
                    return jsonify({'error': 'Exaggeration must be between 0.25 and 2.0'}), 400
            if 'pacing' in directive:
                if isinstance(directive['pacing'], (int, float)) and 0.0 <= directive['pacing'] <= 1.0:
                    cli_args.pacing = float(directive['pacing'])
                    updated.append(f"pacing={cli_args.pacing}")
                else:
                    return jsonify({'error': 'Pacing must be between 0.0 and 1.0'}), 400
            if updated:
                print(f"Updated parameters: {', '.join(updated)}")
                return jsonify({'status': 'Parameters updated', 'parameters': updated}), 200
            return jsonify({'error': 'No valid parameters provided in directive'}), 400

        # Handle quit command
        if 'text' in data and isinstance(data['text'], str) and data['text'].strip().lower() == "quit":
            print("Received quit command")
            is_server_running = False
            with audio_lock:
                audio_lock.notify_all()
            if server:
                threading.Thread(target=server.shutdown).start()
            return jsonify({'status': 'Quit received, server shutting down'}), 200

        # Handle TTS text (including empty / blank lines → 1s pause)
        if 'text' not in data:
            return jsonify({'error': 'Missing text in JSON payload'}), 400

        text = data['text']
        if not isinstance(text, str):
            return jsonify({'error': 'Text must be a string'}), 400

        """print(f"Received text: {text}")"""
        print(f"Received text: {len(text)} chars")
        process_tokenized_text(text)
        return jsonify({'status': 'Audio generated', 'parameters': _current_parameters()}), 200
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500


def init_mixer():
    try:
        pygame.mixer.init(frequency=sample_rate)
    except Exception as e:
        print(f"Failed to initialize pygame mixer: {e}")
        sys.exit(1)


def start_audio_thread():
    audio_thread = threading.Thread(target=play_audio)
    audio_thread.daemon = True
    audio_thread.start()
    return audio_thread


def run_server(port, text, audio_prompt_path, exaggeration, pacing, volume):
    """Start the Flask server with shutdown capability."""
    global server

    # Process initial audio (tokens in the startup text are honored)
    if text:
        process_tokenized_text(text)
        print("Initial audio generated successfully")

    print(f"Starting HTTP server on http://127.0.0.1:{port}/yttss...")
    server = make_server('127.0.0.1', port, app)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(
        description="HTTP-based TTS server with custom pacing and volume")
    parser.add_argument("text", type=str, help="Initial text to synthesize")
    parser.add_argument("audio_prompt", type=str,
                        help="Path to reference audio file")
    parser.add_argument("exaggeration", type=float, default=1.0,
                        help="Expressiveness (e.g., 0.25-2.0)")
    parser.add_argument("pacing", type=float, default=0.5,
                        help="Pacing control (0.0-1.0)")
    parser.add_argument("volume", type=float, default=1.0,
                        help="Playback volume (0.0-1.0)")
    parser.add_argument("--server", action="store_true",
                        help="Run as HTTP server on port 9126")
    parser.add_argument("--port", type=int, default=9126,
                        help="Port for server mode")
    parser.add_argument("--seed", type=int, default=13,
                        help="RNG seed for TTS generation")
    parser.add_argument("--vmap", nargs="?", const="", default=None,
                        help="JSON file mapping voice names to audio reference files")
    global cli_args, voice_map
    cli_args = parser.parse_args()
    voice_map = load_voice_map(cli_args.vmap)

    init_mixer()
    start_audio_thread()

    if cli_args.server:
        run_server(cli_args.port, cli_args.text, cli_args.audio_prompt,
                   cli_args.exaggeration, cli_args.pacing, cli_args.volume)
    else:
        process_tokenized_text(cli_args.text)
        print("Initial audio generated successfully")

        while True:
            user_input = input("Say: ")
            if user_input.strip().lower() == "quit":
                print("Exiting")
                break
            process_tokenized_text(user_input)


if __name__ == "__main__":
    main()
