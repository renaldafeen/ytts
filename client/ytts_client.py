import os
import json
import requests

try:
    import readline
except ImportError:
    readline = None

HISTORY_LIMIT = 1000


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


def loop_yttss(url="http://localhost:" + os.environ['CBOX_PORT'] + "/yttss"):
    """Persistent client to send text prompts or directives to TTS server."""
    print("TTS Client: Type your prompt, or 'set volume=X exaggeration=Y pacing=Z' (or 'quit' to exit)")
    if readline:
        readline.set_history_length(HISTORY_LIMIT)
    else:
        print("Warning: readline unavailable; arrow-key history disabled")

    while True:
        try:
            user_input = read_prompt("Prompt: ")

            if user_input.lower() == "quit":
                payload = {"text": "quit"}
                response = requests.post(url, json=payload, headers={
                                         "Content-Type": "application/json"})
                print(f"Server response: {response.json()}")
                break

            # Check for directive (e.g., "set volume=0.8 exaggeration=0.6 pacing=0.9")
            if user_input.lower().startswith("set "):
                parts = user_input[4:].split()
                directive = {}
                for part in parts:
                    key, value = part.split("=")
                    directive[key] = float(value)
                payload = {"directive": directive}
                response = requests.post(url, json=payload, headers={
                                         "Content-Type": "application/json"})
                print(f"Server response: {response.json()}")
                continue

            # Normal text prompt
            payload = {"text": user_input}
            response = requests.post(url, json=payload, headers={
                                     "Content-Type": "application/json"})
            print(f"Server response: {response.json()}")

        except requests.RequestException as e:
            print(f"Error connecting to server: {e}")
        except json.JSONDecodeError:
            print("Error: Invalid response from server")
        except ValueError:
            print(
                "Error: Invalid directive format. Use 'set volume=X exaggeration=Y pacing=Z'")
        except KeyboardInterrupt:
            print("\nExiting")
            break


if __name__ == "__main__":
    loop_yttss()
