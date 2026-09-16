import json
from typing import Any

from fastapi import WebSocket


class RawPCMProvider:
    """Binary PCM transport with optional JSON lifecycle events."""

    async def receive(self, websocket: WebSocket) -> bytes:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            from starlette.websockets import WebSocketDisconnect

            raise WebSocketDisconnect(message.get("code", 1000))
        data = message.get("bytes")
        if data is None:
            raise ValueError("Only binary PCM frames are accepted")
        return data

    async def send_audio(self, websocket: WebSocket, pcm: bytes) -> None:
        await websocket.send_bytes(pcm)

    async def send_event(self, websocket: WebSocket, event: str, **data: Any) -> None:
        await websocket.send_text(json.dumps({"event": event, **data}, separators=(",", ":")))
