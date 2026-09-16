from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import WebSocket


class AudioProvider(Protocol):
    async def receive(self, websocket: WebSocket) -> bytes: ...

    async def send_audio(self, websocket: WebSocket, pcm: bytes) -> None: ...

    async def send_event(self, websocket: WebSocket, event: str, **data: Any) -> None: ...


class SpeechToText(Protocol):
    async def transcribe(self, pcm: bytes) -> str: ...


class Processor(Protocol):
    async def process(self, text: str) -> str: ...


@dataclass(frozen=True, slots=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> AsyncIterator[AudioChunk]: ...
