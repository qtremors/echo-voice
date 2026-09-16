# Echo Voice

A small, self-hosted voice echo server. It accepts mono PCM16 audio over a WebSocket,
detects complete utterances with Silero VAD, transcribes with faster-whisper, and streams
the same words back through Piper. There is no LLM.

The default CPU path is intentionally lean:

- Silero runs from faster-whisper's bundled ONNX model, with no PyTorch dependency.
- Whisper uses `tiny.en`, INT8, beam size 1, and one shared model.
- Serving uses only the local model cache, with no request-time network checks.
- Piper uses the small `en_US-lessac-low` voice and streams sentence chunks.
- CPU inference runs off the async event loop and is concurrency-limited.
- Models load on first use, so health checks and idle startup remain fast.

## Run locally

Install [uv](https://docs.astral.sh/uv/), then:

```sh
uv sync
uv run python scripts/download_models.py
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check `http://localhost:8000/health`. The voice socket is
`ws://localhost:8000/voice`.

Open `http://localhost:8000/` in a browser and select **Start microphone** to have a
live voice conversation. Microphone access works on localhost or HTTPS. The client pauses
capture while the echoed audio plays, preventing a feedback loop.

Copy `.env.example` to `.env` to tune the server. The application loads that file
automatically, or you can provide another one explicitly:

```sh
uv run --env-file .env uvicorn app.main:app
```

## Raw WebSocket protocol

Send binary frames containing 16 kHz, mono, signed 16-bit little-endian PCM. The server
returns binary frames in the same format. Frames can be any size up to
`MAX_MESSAGE_BYTES`; VAD internally consumes fixed 32 ms windows.

The server may send short JSON text events for observability:

- `speech_start`
- `transcript`
- `audio_start`
- `audio_end`
- `error`

Clients that only need audio can ignore text frames. Run the file client with a mono
PCM16 WAV input:

```sh
uv run python scripts/test_client.py sample.wav --output returned.wav
```

Set `VOICE_TOKEN` before exposing the service. A client then connects to
`/voice?token=...`. Put TLS and rate limiting at the reverse proxy.

## Telephony WebSocket

Configure the telephony stream with this secure WebSocket URL:

```text
wss://YOUR-DOMAIN/voice/telephony
```

If `VOICE_TOKEN` is set, append `?token=YOUR_TOKEN`. This endpoint implements JSON
`connected`, `start`, `media`, `mark`, and `stop` events. Inbound base64 G.711 A-law
at 8 kHz is converted to the pipeline's 16 kHz PCM. Replies are converted back to A-law
and sent in 20 ms media packets. The adapter also sends `mark` and `clear` events so queued
audio can be interrupted when the caller starts speaking again.

## Deploy on Vercel

This repository is configured for Vercel's native FastAPI runtime. No container is needed.
Import the Git repository in Vercel or run:

```sh
vercel --prod
```

The Vercel build hook downloads the Whisper and Piper models before the function bundle is
created. The repository enables Fluid Compute and opts into Large Functions during the build.
Its runtime excludes the media decoder and Xet client that the PCM-only pipeline does not use,
and stores the Whisper model in a flat directory to avoid duplicate cache files.

Vercel Hobby WebSocket functions have a 300-second maximum duration. On Vercel, this app
defaults to a 285-second connection limit so calls close cleanly before the platform limit.
That makes the free plan suitable for tests and calls under roughly 4 minutes 45 seconds,
but not for longer production calls. The public endpoints will be:

```text
https://YOUR-PROJECT.vercel.app/health
wss://YOUR-PROJECT.vercel.app/voice/telephony
```

The browser client remains available at the project root. Models are lazy-loaded on the
first voice request, so the first call after a cold start will take longer than warm calls.
