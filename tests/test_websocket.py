import asyncio
import base64
import json
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from app.audio.vad import VADEvent, VADFrame
from app.config import Settings
from app.interfaces import AudioChunk
from app.main import app
from app.processors import EchoProcessor


class FakeVAD:
    def __init__(self) -> None:
        self.calls = 0

    def feed(self, chunk: bytes) -> list[VADFrame]:
        self.calls += 1
        event = VADEvent.SPEECH_START if self.calls == 1 else VADEvent.SPEECH_END
        return [VADFrame(pcm=chunk, event=event)]

    def reset(self) -> None:
        pass


class FakeSTT:
    async def transcribe(self, pcm: bytes) -> str:
        assert len(pcm) == 2_048
        return "hello"


class FakeTTS:
    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        assert text == "hello"
        yield AudioChunk(pcm=b"echo", sample_rate=16_000)


class FakeServices:
    def __init__(self) -> None:
        self.settings = Settings(vad_min_speech_ms=32)
        self.stt = FakeSTT()
        self.processor = EchoProcessor()
        self.tts = FakeTTS()
        self.inference_gate = asyncio.Semaphore(1)

    def new_vad(self) -> FakeVAD:
        return FakeVAD()


def test_websocket_echo_pipeline() -> None:
    original = app.state.services
    app.state.services = FakeServices()
    try:
        with TestClient(app) as client, client.websocket_connect("/voice") as websocket:
            websocket.send_bytes(bytes(1_024))
            assert json.loads(websocket.receive_text()) == {"event": "speech_start"}
            websocket.send_bytes(bytes(1_024))
            assert json.loads(websocket.receive_text()) == {
                "event": "transcript",
                "text": "hello",
            }
            assert json.loads(websocket.receive_text()) == {
                "event": "audio_start",
                "sample_rate": 16_000,
            }
            assert websocket.receive_bytes() == b"echo"
            assert json.loads(websocket.receive_text()) == {"event": "audio_end"}
    finally:
        app.state.services = original


def test_telephony_websocket_echo_pipeline() -> None:
    original = app.state.services
    app.state.services = FakeServices()
    payload = base64.b64encode(bytes([0xD5]) * 256).decode("ascii")
    try:
        with TestClient(app) as client, client.websocket_connect(
            "/voice/telephony"
        ) as websocket:
            websocket.send_json({"event": "connected"})
            websocket.send_json(
                {
                    "event": "start",
                    "stream_sid": "stream-1",
                    "start": {
                        "stream_sid": "stream-1",
                        "call_sid": "call-1",
                        "media_format": {
                            "encoding": "audio/alaw",
                            "sample_rate": "8000",
                        },
                    },
                }
            )
            websocket.send_json(
                {"event": "media", "media": {"track": "inbound", "payload": payload}}
            )
            websocket.send_json(
                {"event": "media", "media": {"track": "inbound", "payload": payload}}
            )

            media = websocket.receive_json()
            assert media["event"] == "media"
            assert base64.b64decode(media["media"]["payload"])
            assert websocket.receive_json() == {
                "event": "mark",
                "mark": {"name": "reply-1"},
            }
    finally:
        app.state.services = original
