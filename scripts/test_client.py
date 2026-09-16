"""Send a PCM16 WAV file to the voice socket and save the echoed result."""

import argparse
import asyncio
import json
import wave
from pathlib import Path

from websockets.asyncio.client import connect


async def run(input_path: Path, output_path: Path, uri: str) -> None:
    with wave.open(str(input_path), "rb") as source:
        if source.getnchannels() != 1 or source.getsampwidth() != 2:
            raise ValueError("Input must be mono 16-bit PCM WAV")
        if source.getframerate() != 16_000:
            raise ValueError("Input sample rate must be 16000 Hz")
        pcm = source.readframes(source.getnframes())

    received = bytearray()
    audio_started = False
    async with connect(uri, max_size=1_048_576) as websocket:
        for offset in range(0, len(pcm), 1_024):
            await websocket.send(pcm[offset : offset + 1_024])
            await asyncio.sleep(0.032)
        await websocket.send(bytes(16_000 * 2))

        async for message in websocket:
            if isinstance(message, bytes):
                received.extend(message)
                continue
            event = json.loads(message)
            if event.get("event") == "transcript":
                print(f"Transcript: {event['text']}")
            elif event.get("event") == "audio_start":
                audio_started = True
            elif event.get("event") == "audio_end" and audio_started:
                break
            elif event.get("event") == "error":
                raise RuntimeError(event.get("detail", "server error"))

    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(received)
    print(f"Wrote {len(received)} bytes to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("returned.wav"))
    parser.add_argument("--uri", default="ws://localhost:8000/voice")
    args = parser.parse_args()
    asyncio.run(run(args.input, args.output, args.uri))


if __name__ == "__main__":
    main()
