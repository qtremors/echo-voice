import base64
import json
import logging
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.audio.codecs import alaw_to_pcm16, pcm16_to_alaw, resample_pcm16

logger = logging.getLogger(__name__)


class TelephonyProvider:
    """JSON telephony transport using 8 kHz G.711 A-law audio."""

    source_rate = 8_000
    target_rate = 16_000
    packet_bytes = 160  # 20 ms at 8 kHz

    def __init__(self, max_message_bytes: int = 65_536) -> None:
        self.max_message_bytes = max_message_bytes
        self.stream_sid = ""
        self.call_sid = ""
        self._mark_counter = 0
        self._pending_mark = ""

    async def receive(self, websocket: WebSocket) -> bytes:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            text = message.get("text")
            if text is None:
                raise ValueError("Telephony messages must be JSON text")
            try:
                event = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError("Telephony message is not valid JSON") from error
            if not isinstance(event, dict):
                raise ValueError("Telephony message must be a JSON object")

            event_name = event.get("event")
            if event_name == "connected":
                continue
            if event_name == "start":
                self._handle_start(event)
                continue
            if event_name == "media":
                return self._decode_media(event)
            if event_name == "mark":
                name = event.get("mark", {}).get("name")
                if name == self._pending_mark:
                    self._pending_mark = ""
                continue
            if event_name == "stop":
                raise WebSocketDisconnect(1000)
            logger.debug("Ignoring telephony event %r", event_name)

    async def send_audio(self, websocket: WebSocket, pcm: bytes) -> None:
        if not self.stream_sid:
            raise ValueError("Telephony start event must arrive before outbound audio")
        pcm_8k = resample_pcm16(pcm, self.target_rate, self.source_rate)
        encoded = pcm16_to_alaw(pcm_8k)
        for offset in range(0, len(encoded), self.packet_bytes):
            payload = base64.b64encode(encoded[offset : offset + self.packet_bytes]).decode("ascii")
            await self._send(websocket, {"event": "media", "media": {"payload": payload}})

    async def send_event(self, websocket: WebSocket, event: str, **data: Any) -> None:
        if event == "speech_start" and self._pending_mark and self.stream_sid:
            await self._send(
                websocket,
                {"event": "clear", "stream_sid": self.stream_sid},
            )
            self._pending_mark = ""
        elif event == "audio_end" and self.stream_sid:
            self._mark_counter += 1
            self._pending_mark = f"reply-{self._mark_counter}"
            await self._send(
                websocket,
                {"event": "mark", "mark": {"name": self._pending_mark}},
            )

    def _handle_start(self, event: dict[str, Any]) -> None:
        start = event.get("start")
        if not isinstance(start, dict):
            raise ValueError("Telephony start event is missing start data")
        media_format = start.get("media_format", {})
        if not isinstance(media_format, dict):
            raise ValueError("Telephony start event has an invalid media format")
        encoding = str(media_format.get("encoding", "")).lower()
        try:
            sample_rate = int(media_format.get("sample_rate", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("Telephony sample rate is invalid") from error
        if encoding != "audio/alaw" or sample_rate != self.source_rate:
            raise ValueError("Telephony transport requires audio/alaw at 8000 Hz")
        self.stream_sid = str(start.get("stream_sid") or event.get("stream_sid") or "")
        self.call_sid = str(start.get("call_sid") or "")
        if not self.stream_sid:
            raise ValueError("Telephony start event is missing stream_sid")
        logger.info(
            "Telephony stream started stream_sid=%s call_sid=%s",
            self.stream_sid,
            self.call_sid,
        )

    def _decode_media(self, event: dict[str, Any]) -> bytes:
        if not self.stream_sid:
            raise ValueError("Telephony media arrived before start")
        media = event.get("media")
        if not isinstance(media, dict):
            raise ValueError("Telephony media event is missing media data")
        payload = media.get("payload")
        if not isinstance(payload, str):
            raise ValueError("Telephony media payload must be base64 text")
        try:
            encoded = base64.b64decode(payload, validate=True)
        except ValueError as error:
            raise ValueError("Telephony media payload is not valid base64") from error
        if len(encoded) * 4 > self.max_message_bytes:
            raise ValueError("Telephony media frame is too large")
        pcm_8k = alaw_to_pcm16(encoded)
        return resample_pcm16(pcm_8k, self.source_rate, self.target_rate)

    @staticmethod
    async def _send(websocket: WebSocket, message: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(message, separators=(",", ":")))
