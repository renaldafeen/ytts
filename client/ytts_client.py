import os
import json
import requests

try:
    import readline
except ImportError:
    readline = None

HISTORY_LIMIT = 1000
REQUEST_TIMEOUT = 300
DEFAULT_PORT = "9126"


def _history_last():
    if not readline:
        return None
    length = readline.get_current_history_length()
    if length <= 0:
        return None
    return readline.get_history_item(length)


def _drop_last_history_item():
    """Remove the most recent readline history entry.

    readline.remove_history_item is 0-based; get_history_item is 1-based.
    """
    if not readline:
        return
    length = readline.get_current_history_length()
    if length <= 0:
        return
    try:
        readline.remove_history_item(length - 1)
    except ValueError:
        pass


def _normalize_history(raw, stripped):
    """Keep session history stripped, skipping empties and consecutive dups.

    input() auto-appends the raw line when GNU readline is active. Drop that
    entry only when it is actually present, then re-add the stripped form.
    Never reads or writes a history file.
    """
    if not readline:
        return
    if _history_last() == raw:
        _drop_last_history_item()
    if stripped and stripped != _history_last():
        readline.add_history(stripped)


def read_prompt(prompt="Prompt: "):
    """Read a line, keeping session-only stripped history for Up/Down."""
    raw = input(prompt)
    stripped = raw.strip()
    _normalize_history(raw, stripped)
    return stripped


def default_url():
    """Local server URL. CBOX_PORT is optional and is read when called."""
    port = os.environ.get("CBOX_PORT", DEFAULT_PORT)
    return f"http://127.0.0.1:{port}/yttss"


def _directive_from_set(text):
    parts = text[4:].split()
    directive = {}
    for part in parts:
        key, value = part.split("=", 1)
        if key == "voice":
            directive[key] = value
        else:
            directive[key] = float(value)
    return directive


def interpret_line(user_input):
    """Return (action, payload). quit sends nothing. shutdown stops the server."""
    lowered = user_input.lower()
    if lowered == "quit":
        return "quit", None
    if lowered == "shutdown-server":
        return "shutdown", {"directive": {"shutdown": True}}
    if lowered.startswith("set "):
        return "send", {"directive": _directive_from_set(user_input)}
    return "send", {"text": user_input}


def _auth_headers():
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("YTTS_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_payload(url, payload):
    response = requests.post(
        url, json=payload, headers=_auth_headers(), timeout=REQUEST_TIMEOUT
    )
    if response.status_code == 401:
        print("unauthorized")
        if not os.environ.get("YTTS_TOKEN"):
            print("YTTS_TOKEN is required")
        return response
    print(f"Server response: {response.json()}")
    return response


def loop_yttss(url=None):
    """Persistent client to send text prompts or directives to TTS server."""
    if url is None:
        url = default_url()
    print(
        "TTS Client: type a prompt, "
        "'set volume=X exaggeration=Y pacing=Z seed=S voice=Name', "
        "'shutdown-server' to stop the server, or 'quit' to exit this client"
    )
    if readline:
        readline.set_history_length(HISTORY_LIMIT)
    else:
        print("Warning: readline unavailable; arrow-key history disabled")

    while True:
        try:
            user_input = read_prompt("Prompt: ")
            action, payload = interpret_line(user_input)
            if action == "quit":
                break
            post_payload(url, payload)
            if action == "shutdown":
                break
        except requests.RequestException as e:
            print(f"Error connecting to server: {e}")
        except json.JSONDecodeError:
            print("Error: Invalid response from server")
        except ValueError:
            print(
                "Error: Invalid directive format. "
                "Use 'set volume=X exaggeration=Y pacing=Z seed=S voice=Name'"
            )
        except KeyboardInterrupt:
            print("\nExiting")
            break


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="ytts test client")
    parser.add_argument(
        "--url",
        default=None,
        help="Server URL. Defaults to http://127.0.0.1:$CBOX_PORT/yttss",
    )
    args = parser.parse_args(argv)
    loop_yttss(args.url if args.url else default_url())


if __name__ == "__main__":
    main()
