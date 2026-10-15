### ytts

`ytts` is a TTS system with a server and a test client. Speech is synthesized with ChatterboxTTS 0.1.2 and played on the machine that runs the server. A remote client can send text and parameter changes. It does not receive audio.

This build is English-only.

- Client: `client/ytts_client.py` is a test CLI. Up and Down recall lines from the current session. History is not saved. `quit` exits the client. `shutdown-server` stops the server process.
- Server: `server/ytts_server.py` listens for TTS and parameter requests. Incoming text is parsed by `server/ytts_tokens.py`. Spoken pieces are split to 300 characters, the Chatterbox limit, on sentence boundaries, and never inside `[...]` tokens. One POST can queue many speech and silence buffers. Clients do not need to know the 300-character limit.

Local playback buffers are mono, 16-bit PCM WAV at the sample rate reported by the loaded model (`model.sr`, 24000 Hz on Chatterbox 0.1.2).

Start scripts, both of which `cd` to the repository root:

- `scripts/ytts_server.sh` activates the Chatterbox virtual environment and starts the server.
- `scripts/ytts_client.sh` starts the test client.

Run-time synthesis requires Resemble AI's ChatterboxTTS virtual environment. See https://github.com/resemble-ai/chatterbox. This code is based on v0.1.2. `requirements.txt` records the package versions in that environment. The start script does not install them.

### Setup

- Clone the repo and `cd ytts`.
- Add reference WAV files under `voices/` and name them in `voices/vmap.json`.
- `export CBOX_VENV` to the Chatterbox virtual environment, for example `"$HOME/AI/CBox/chatterbox_venv"`.
- `export CBOX_PORT` to the listen port, for example `9100`.
- `export YTTS_TOKEN` to a shared secret. The server will not start without it, including on localhost. The client sends it as `Authorization: Bearer <token>`.
- Optional: `export YTTS_HOST` to the address the server binds. The default is `127.0.0.1`, which other machines cannot reach.
- Optional, on a remote client: `export YTTS_URL` to `http://<server-lan-ip>:<port>/yttss`.
- Run `./scripts/ytts_server.sh`.
- Run `./scripts/ytts_client.sh` to test.
- Application logs rotate in `ytts_server.log`. The log records text length, not the text.

The server command is flag-based. The script passes `--server`, `--host`, `--port`, `--vmap voices/vmap.json`, `--aref voices/ytts_ref_1.wav`, a startup sentence, and the startup exaggeration, pacing, and volume.

Fixed sampling values, not set by clients: temperature `0.8`, `min_p` `0.05`, `top_p` `1.0`, `repetition_penalty` `1.2`.

`volume` is pygame playback gain, from `0.0` through `1.0` inclusive. `pacing` is Chatterbox `cfg_weight`, from `0.0` through `1.0` inclusive. It is not a speech-rate percentage. `exaggeration` is Chatterbox's emotion control, from `0.25` through `2.0` inclusive.

Optional TLS: start the server yourself with `--tls-cert` and `--tls-key` (both required together), or use an SSH tunnel, when spoken text should not cross the LAN in cleartext.

`GET /health` does not require the token. It returns `{"status":"ready"}` after the model has loaded.

### Requests

`POST /yttss` with `Content-Type: application/json` and the bearer token.

A body with `text` is spoken with the current session parameters. Inline tokens in that text apply only to that request. The next request uses the session again.

A body with `directive` changes the session. `text` in the same body is ignored. Directive fields:

- `volume`, `exaggeration`, `pacing` — closed ranges above. A boolean is rejected.
- `seed` — positive integer, clamped to numpy's seed range.
- `voice` — a name from `vmap.json`, or `default` for the built-in Chatterbox voice.
- `flush` — `true` drops the playback queue and stops the current clip.
- `shutdown` — `true` stops the server. Requires the same token.

The JSON result includes `voice` (a map name, or `null` for the built-in voice). It does not include a filesystem path.

`quit` as spoken text is ordinary speech. It does not stop the server.

### Supported tokens

Tokens are case-sensitive for voice names. Numeric ranges below are inclusive. A value outside the range is logged and ignored.

`[pause: Ns]` pauses playback for N seconds. The `s` is optional. N may be fractional. `[pause: 0.5s]` is half a second. Values above 30 seconds are clamped to 30. Zero and negative values are ignored.

A blank line in the incoming text is a one-second pause.

`[rpause: n-Ns]` pauses for a random duration between n and N seconds, with the same cap and the same optional `s`. A reversed range is swapped.

`[vol: V]` sets playback volume for later audio in this request to V, from `0.0` through `1.0`.

`[pace: P]` sets pacing (`cfg_weight`) for later audio in this request to P, from `0.0` through `1.0`.

`[exp: E]` sets exaggeration for later audio in this request to E, from `0.25` through `2.0`.

`[seed: S]` sets the sampling seed for later audio in this request. Non-positive values become 1. Values above numpy's limit are clamped.

`[aref: Name]` selects the WAV mapped to Name in `vmap.json` for later audio in this request. A path is not accepted. An unknown name is logged and ignored, and the current voice stays in place. Use a directive `voice` when the change should stick for later requests.

See `token_test_messages.txt` for spoken examples.

### Remote access and the host firewall

Another machine can send requests only when two settings are in place together. `YTTS_HOST` is the address the process binds. The host firewall must allow TCP to that address and port. Opening the firewall while `YTTS_HOST` stays `127.0.0.1` does nothing for other machines, because the process is not listening on the LAN address. Set `YTTS_HOST` to this host's LAN address. Binding `0.0.0.0` is possible and is not the default.

`127.0.0.1` does not include IPv6 `::1`. If clients connect over IPv6, set `YTTS_HOST` to that address and allow the same port in the IPv6 firewall.

Allow the port only from the client addresses that should speak. Do not add a rule that allows the whole internet. The bearer token is still required. The firewall is the network limit.

Confirm what is listening:

```bash
ss -ltnp | grep "$CBOX_PORT"
```

The examples below use placeholders `SERVER_LAN_IP`, `CLIENT_LAN_IP`, and `CBOX_PORT`. Each allows TCP `CBOX_PORT` from `CLIENT_LAN_IP` to `SERVER_LAN_IP` and survives a reboot.

firewalld (RHEL, Fedora, and derivatives):

```bash
sudo firewall-cmd --permanent --add-rich-rule="rule family=ipv4 source address=CLIENT_LAN_IP destination address=SERVER_LAN_IP port port=CBOX_PORT protocol=tcp accept"
sudo firewall-cmd --reload
sudo firewall-cmd --list-rich-rules
```

ufw (Ubuntu and derivatives):

```bash
sudo ufw allow from CLIENT_LAN_IP to SERVER_LAN_IP port CBOX_PORT proto tcp
sudo ufw status verbose
```

nftables, on a host that manages its own ruleset. Put this in the existing inet filter chain. A second table that is never hooked will not take effect:

```nft
tcp dport CBOX_PORT ip saddr CLIENT_LAN_IP ip daddr SERVER_LAN_IP accept
```

From the client machine, check reachability without sending speech:

```bash
curl -sS "http://SERVER_LAN_IP:CBOX_PORT/health"
```

A ready server returns `{"status":"ready"}`. `/health` does not require the token. A refused connection is the firewall or the bind address. A JSON 401 on `POST /yttss` means the path is open and the token is the remaining step.

On the client:

```bash
export YTTS_TOKEN="the-same-secret"
export YTTS_URL="http://SERVER_LAN_IP:CBOX_PORT/yttss"
./scripts/ytts_client.sh
```

A router, a cloud security group, or another firewall in front of this host needs a rule of the same shape: TCP to the port, from the client address, to this host. This section is the host firewall. It does not configure the router.

Any host allowed by that rule can attempt requests. The bearer token rejects them. Remove the client address from the rule to revoke that network access.
