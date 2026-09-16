import importlib.util
import sys
from types import ModuleType


def enable_array_only_whisper() -> None:
    """Allow faster-whisper's array path without bundling its FFmpeg decoder."""
    if "av" in sys.modules:
        return
    try:
        if importlib.util.find_spec("av") is not None:
            return
    except (ImportError, ValueError):
        pass

    module = ModuleType("av")

    def missing_decoder(name: str) -> object:
        raise RuntimeError(
            f"PyAV attribute {name!r} is unavailable; pass decoded PCM arrays to Whisper"
        )

    module.__getattr__ = missing_decoder  # type: ignore[attr-defined]
    sys.modules["av"] = module
