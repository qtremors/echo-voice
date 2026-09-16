import base64
import json

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from app.audio.codecs import pcm16_to_alaw
from app.providers.telephony import TelephonyProvider


class FakeWebSocket:
    def __init__(self, messages: list[dict[str, object]] | None = None) -> None:
        self.messages = iter(messages or [])
        self.sent: list[dict[str, object]] = []

    async def receive(self) -> dict[str, object]:
        return next(self.messages)

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


def start_message() -> dict[str, object]:
    return {
        "type": "websocket.receive",
        "text": json.dumps(
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
        ),
    }


@pytest.mark.asyncio
async def test_receives_alaw_as_16k_pcm() -> None:
    pcm_8k = np.arange(160, dtype="<i2").tobytes()
    payload = base64.b64encode(pcm16_to_alaw(pcm_8k)).decode("ascii")
    websocket = FakeWebSocket(
        [
            {"type": "websocket.receive", "text": '{"event":"connected"}'},
            start_message(),
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {"event": "media", "media": {"track": "inbound", "payload": payload}}
                ),
            },
        ]
    )
    provider = TelephonyProvider()

    result = await provider.receive(websocket)  # type: ignore[arg-type]

    assert provider.stream_sid == "stream-1"
    assert provider.call_sid == "call-1"
    assert len(result) == 640


@pytest.mark.asyncio
async def test_sends_small_alaw_packets_mark_and_clear() -> None:
    websocket = FakeWebSocket()
    provider = TelephonyProvider()
    provider._handle_start(json.loads(str(start_message()["text"])))

    pcm_16k = np.arange(640, dtype="<i2").tobytes()
    await provider.send_audio(websocket, pcm_16k)  # type: ignore[arg-type]
    await provider.send_event(websocket, "audio_end")  # type: ignore[arg-type]
    await provider.send_event(websocket, "speech_start")  # type: ignore[arg-type]

    media = [message for message in websocket.sent if message["event"] == "media"]
    assert len(media) == 2
    assert all(len(base64.b64decode(message["media"]["payload"])) <= 160 for message in media)
    assert websocket.sent[-2] == {"event": "mark", "mark": {"name": "reply-1"}}
    assert websocket.sent[-1] == {"event": "clear", "stream_sid": "stream-1"}


@pytest.mark.asyncio
async def test_stop_ends_session() -> None:
    websocket = FakeWebSocket(
        [{"type": "websocket.receive", "text": '{"event":"stop","stop":{}}'}]
    )
    with pytest.raises(WebSocketDisconnect):
        await TelephonyProvider().receive(websocket)  # type: ignore[arg-type]
