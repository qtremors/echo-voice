import numpy as np


def _decode_alaw_sample(value: int) -> int:
    value ^= 0x55
    magnitude = ((value & 0x0F) << 4) + 8
    segment = (value & 0x70) >> 4
    if segment:
        magnitude = (magnitude + 0x100) << (segment - 1)
    return magnitude if value & 0x80 else -magnitude


def _encode_alaw_sample(sample: int) -> int:
    sign = 0x80 if sample >= 0 else 0
    magnitude = sample if sample >= 0 else ~sample
    compressed = magnitude >> 4
    if compressed > 15:
        segment = compressed.bit_length() - 4
        compressed = (compressed >> (segment - 1)) - 16 + (segment << 4)
    return (sign | compressed) ^ 0x55


_ALAW_DECODE_TABLE = np.array(
    [_decode_alaw_sample(value) for value in range(256)], dtype="<i2"
)
_ALAW_ENCODE_TABLE = np.array(
    [_encode_alaw_sample(value) for value in range(-32_768, 32_768)], dtype=np.uint8
)


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    if len(pcm) % 2:
        raise ValueError("PCM16 data must contain complete 2-byte samples")
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32_768.0


def resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """Resample mono PCM16 with low-overhead linear interpolation."""
    if source_rate == target_rate or not pcm:
        return pcm
    samples = np.frombuffer(pcm, dtype="<i2")
    output_size = max(1, round(len(samples) * target_rate / source_rate))
    source_positions = np.arange(len(samples), dtype=np.float64)
    target_positions = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    output = np.interp(target_positions, source_positions, samples).astype("<i2")
    return output.tobytes()


def alaw_to_pcm16(alaw: bytes) -> bytes:
    """Decode G.711 A-law into little-endian signed PCM16."""
    if not alaw:
        return b""
    encoded = np.frombuffer(alaw, dtype=np.uint8)
    return _ALAW_DECODE_TABLE[encoded].astype("<i2", copy=False).tobytes()


def pcm16_to_alaw(pcm: bytes) -> bytes:
    """Encode little-endian signed PCM16 as G.711 A-law."""
    if len(pcm) % 2:
        raise ValueError("PCM16 data must contain complete 2-byte samples")
    if not pcm:
        return b""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.int32)
    return _ALAW_ENCODE_TABLE[samples + 32_768].tobytes()
