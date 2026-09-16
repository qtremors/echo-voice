import numpy as np

from app.audio.vad import FRAME_SAMPLES, VADEvent, VoiceActivityDetector


class FakeModel:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)

    def predict(self, audio: np.ndarray) -> float:
        assert len(audio) == FRAME_SAMPLES
        return next(self.probabilities)


def test_vad_detects_start_and_end_across_arbitrary_chunks() -> None:
    model = FakeModel([0.1, 0.8, 0.1, 0.1])
    vad = VoiceActivityDetector(threshold=0.5, silence_ms=64, model=model)
    pcm = bytes(FRAME_SAMPLES * 2 * 4)
    first = vad.feed(pcm[:700])
    second = vad.feed(pcm[700:])
    assert first == []
    assert [frame.event for frame in second] == [
        VADEvent.NONE,
        VADEvent.SPEECH_START,
        VADEvent.NONE,
        VADEvent.SPEECH_END,
    ]
