### ytts

`ytts` is a TTS system that comprises a TTS Server module and test Client, using the ChatterboxTTS API.

Client: client/ytts_client.py - Server test CLI, with Up/Down arrow keys to recall
    input lines from the current session (history is not persisted). Provides a code reference
    for calling the server.

Server: server/ytts_server.py - exposes a localhost endpoint to listen for incoming TTS and parameter change requests.
    Incoming POST text is token-parsed first (server/ytts_tokens.py). Speak events are then
    chunked to TTS_MAX_CHARS (300), the Chatterbox limit, preferring sentence boundaries
    and never splitting inside [ ... ] tokens. One POST may enqueue many speech and silence
    buffers. Clients do not need to know the 300-character limit.

Start scripts:
    ytts_server.sh - activates the python virtual environment, starts ytts_server.py with specified parameters
    ytts_client.sh - starts up the ytts_client.py CLI

Run-time functionality requires installation of Resemble AI's ChatterboxTTS python virtual environment.
- see https://github.com/resemble-ai/chatterbox. This code is currently based on v0.1.2.

### Steps:

- clone the repo
- cd ytts
- add any desired reference files to ./voices and add them to vmap.json for easy reference
- export CBOX_VENV to identify the location of your ChatterboxTTS venv
- export CBOX_PORT to specify the port you'd like the server to listen on
- run ./scripts/ytts_server.sh
- if desired, run ./scripts/ytts_client.sh to test

In-line tags are supported for ChatterboxTTS parameters like exaggeration, pacing and reference audio (voice clone). See the file token_test_messages.txt for examples.

### Supported tokens:

[pause: Ns] - pauses the audio playback for a duration of *N* seconds. The 's' denotes Seconds, and is optional, i.e., it may or may not appear in the token. If 's' is absent, assume the duration is in Seconds. 

NOTE: a blank or empty line in the incoming text is treated as the equivalent of [pause: 1s].

[rpause: n-Ns] - pauses the audio playback for a random duration between *n* and *N* seconds. The 's' denotes Seconds, and is optional, i.e., it may or may not appear in the token. If 's' is absent, assume the duration is in Seconds.

[vol: V] - sets the *volume* parameter for audio generation to *V*, where *V* can be a fractional value between 0.0 and 1.0, non-inclusive. If *V* is outside that range of values, an error is logged and the token is ignored.

[pace: P] - sets the *pacing* parameter for audio generation to *P*, where *P* can be a fractional value between 0.0 and 1.0, non-inclusive. If *P* is outside that range of values, an error is logged and the token is ignored.

[exp: E] - sets the *exaggeration* parameter for audio generation to *E*, where *E* can be a fractional value between 0.25 and 2.0, non-inclusive. If *E* is outside that range of values, an error is logged and the token is ignored.

[seed: S] - sets the *seed* parameter for audio generation to *S*, where *S* can be any positive integer value that fits in the system's integer value type.

[aref: A] - sets the reference audio value to use the WAV file found at the path specified by *A*. If the file is not found, if it is not a .WAV file, or if an error occurs when attempting to use it as a reference, an error is logged and the token is ignored.

[aref: Name] - sets the reference audio value to use the WAV file mapped to *Name* in vmap.json. If the name or file is not found, if it is not a .WAV file, or if an error occurs when attempting to use it as a reference, an error is logged and the token is ignored.
