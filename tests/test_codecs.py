import numpy as np
import pytest

from app.audio.codecs import alaw_to_pcm16, pcm16_to_alaw, pcm16_to_float32, resample_pcm16


def test_pcm16_conversion() -> None:
    result = pcm16_to_float32(np.array([-32768, 0, 32767], dtype="<i2").tobytes())
    assert result.tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])


def test_resample_changes_sample_count() -> None:
    pcm = np.arange(2205, dtype="<i2").tobytes()
    result = resample_pcm16(pcm, source_rate=22_050, target_rate=16_000)
    assert len(result) == 1600 * 2


def test_alaw_known_values_and_round_trip() -> None:
    assert alaw_to_pcm16(bytes([0x55, 0xD5])) == np.array([-8, 8], dtype="<i2").tobytes()
    source = np.array([-32_768, -5_504, -8, 0, 8, 5_504, 32_767], dtype="<i2")
    decoded = np.frombuffer(alaw_to_pcm16(pcm16_to_alaw(source.tobytes())), dtype="<i2")
    assert decoded.tolist() == [-32_256, -5_504, -8, 8, 8, 5_504, 32_256]
