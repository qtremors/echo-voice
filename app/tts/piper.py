import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from app.audio.codecs import resample_pcm16
from app.interfaces import AudioChunk


class PiperTTS:
    def __init__(self, model_path: Path, target_sample_rate: int, use_cuda: bool = False) -> None:
        self._model_path = model_path
        self._target_sample_rate = target_sample_rate
        self._use_cuda = use_cuda
        self._voice: Any = None
        self._load_lock = threading.Lock()

    def _get_voice(self) -> Any:
        if self._voice is None:
            with self._load_lock:
                if self._voice is None:
                    if not self._model_path.is_file():
                        raise FileNotFoundError(
                            f"Piper model not found: {self._model_path}. "
                            "Run scripts/download_models.py first."
                        )
                    from piper import PiperVoice

                    self._voice = PiperVoice.load(
                        self._model_path,
                        use_cuda=self._use_cuda,
                    )
        return self._voice

    def _start(self, text: str) -> Iterator[Any]:
        return iter(self._get_voice().synthesize(text))

    def _next(self, iterator: Iterator[Any]) -> tuple[bool, AudioChunk | None]:
        try:
            chunk = next(iterator)
        except StopIteration:
            return False, None
        pcm = resample_pcm16(
            chunk.audio_int16_bytes,
            source_rate=chunk.sample_rate,
            target_rate=self._target_sample_rate,
        )
        return True, AudioChunk(pcm=pcm, sample_rate=self._target_sample_rate)

    async def synthesize(self, text: str) -> AsyncIterator[AudioChunk]:
        iterator = await asyncio.to_thread(self._start, text)
        while True:
            has_chunk, chunk = await asyncio.to_thread(self._next, iterator)
            if not has_chunk:
                return
            assert chunk is not None
            yield chunk
