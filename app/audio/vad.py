from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import numpy as np

from app.audio.codecs import pcm16_to_float32
from app.compat import enable_array_only_whisper

FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64


class VADEvent(Enum):
    NONE = auto()
    SPEECH_START = auto()
    SPEECH_END = auto()


@dataclass(frozen=True, slots=True)
class VADFrame:
    pcm: bytes
    event: VADEvent


class ProbabilityModel(Protocol):
    def predict(self, audio: np.ndarray) -> float: ...


class StreamingSileroModel:
    """Stateful wrapper around faster-whisper's bundled Silero v6 ONNX session."""

    def __init__(self) -> None:
        enable_array_only_whisper()
        from faster_whisper.vad import get_vad_model

        self._session = get_vad_model().session
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def predict(self, audio: np.ndarray) -> float:
        model_input = np.concatenate((self._context, audio.reshape(1, -1)), axis=1)
        output, self._h, self._c = self._session.run(
            None,
            {"input": model_input, "h": self._h, "c": self._c},
        )
        self._context = audio[-CONTEXT_SAMPLES:].reshape(1, -1)
        return float(output.item())


class VoiceActivityDetector:
    """Turns arbitrary PCM chunks into fixed frames with speech boundary events."""

    def __init__(
        self,
        threshold: float,
        silence_ms: int,
        sample_rate: int = 16_000,
        model: ProbabilityModel | None = None,
    ) -> None:
        if sample_rate != 16_000:
            raise ValueError("Silero VAD supports this streaming wrapper at 16 kHz only")
        self._model = model or StreamingSileroModel()
        self._threshold = threshold
        self._negative_threshold = max(threshold - 0.15, 0.01)
        frame_ms = FRAME_SAMPLES * 1_000 / sample_rate
        self._end_frames = max(1, round(silence_ms / frame_ms))
        self._pending = bytearray()
        self._speaking = False
        self._silent_frames = 0

    def feed(self, chunk: bytes) -> list[VADFrame]:
        self._pending.extend(chunk)
        frame_bytes = FRAME_SAMPLES * 2
        frames: list[VADFrame] = []
        while len(self._pending) >= frame_bytes:
            pcm = bytes(self._pending[:frame_bytes])
            del self._pending[:frame_bytes]
            probability = self._model.predict(pcm16_to_float32(pcm))
            event = self._classify(probability)
            frames.append(VADFrame(pcm=pcm, event=event))
        return frames

    def _classify(self, probability: float) -> VADEvent:
        if not self._speaking:
            if probability >= self._threshold:
                self._speaking = True
                self._silent_frames = 0
                return VADEvent.SPEECH_START
            return VADEvent.NONE

        if probability >= self._negative_threshold:
            self._silent_frames = 0
            return VADEvent.NONE

        self._silent_frames += 1
        if self._silent_frames >= self._end_frames:
            self._speaking = False
            self._silent_frames = 0
            return VADEvent.SPEECH_END
        return VADEvent.NONE

    def reset(self) -> None:
        self._pending.clear()
        self._speaking = False
        self._silent_frames = 0
        reset = getattr(self._model, "reset", None)
        if reset is not None:
            reset()
