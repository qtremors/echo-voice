import asyncio
import threading
from pathlib import Path
from typing import Any

from app.audio.codecs import pcm16_to_float32
from app.compat import enable_array_only_whisper


def prepare_model_source(
    model_name: str, download_root: Path, local_files_only: bool
) -> str:
    model_path = Path(model_name)
    if model_path.is_dir():
        return str(model_path)

    if local_files_only:
        raise FileNotFoundError(
            f"Whisper model not found at {model_path}. Run scripts/download_models.py first."
        )

    download_root.mkdir(parents=True, exist_ok=True)
    return model_name


class WhisperSTT:
    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: str | None,
        cpu_threads: int,
        download_root: Path,
        local_files_only: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._cpu_threads = cpu_threads
        self._download_root = download_root
        self._local_files_only = local_files_only
        self._model: Any = None
        self._load_lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    enable_array_only_whisper()
                    from faster_whisper import WhisperModel

                    model_source = prepare_model_source(
                        self._model_name,
                        self._download_root,
                        self._local_files_only,
                    )
                    self._model = WhisperModel(
                        model_source,
                        device=self._device,
                        compute_type=self._compute_type,
                        cpu_threads=self._cpu_threads,
                        num_workers=1,
                        download_root=str(self._download_root),
                        local_files_only=self._local_files_only,
                    )
        return self._model

    def _transcribe(self, pcm: bytes) -> str:
        segments, _ = self._get_model().transcribe(
            pcm16_to_float32(pcm),
            language=self._language,
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    async def transcribe(self, pcm: bytes) -> str:
        return await asyncio.to_thread(self._transcribe, pcm)
