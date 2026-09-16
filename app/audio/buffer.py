class AudioBuffer:
    """Bounded PCM buffer for one utterance."""

    __slots__ = ("_data", "_max_bytes")

    def __init__(self, max_bytes: int) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._data = bytearray()
        self._max_bytes = max_bytes

    def append(self, chunk: bytes) -> bool:
        remaining = self._max_bytes - len(self._data)
        self._data.extend(chunk[:remaining])
        return len(chunk) > remaining

    def clear(self) -> None:
        self._data.clear()

    def bytes(self) -> bytes:
        return bytes(self._data)

    def __len__(self) -> int:
        return len(self._data)
