import sys
from types import ModuleType

from app.compat import enable_array_only_whisper


def test_preloaded_module_without_spec_is_accepted(monkeypatch) -> None:
    placeholder = ModuleType("av")
    assert placeholder.__spec__ is None
    monkeypatch.setitem(sys.modules, "av", placeholder)

    enable_array_only_whisper()

    assert sys.modules["av"] is placeholder
