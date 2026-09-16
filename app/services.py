import asyncio

from app.audio.vad import VoiceActivityDetector
from app.config import Settings
from app.processors import EchoProcessor
from app.stt.whisper import WhisperSTT
from app.tts.piper import PiperTTS


class Services:
    """Process-wide model services and per-connection VAD factory."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stt = WhisperSTT(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=settings.whisper_language,
            cpu_threads=settings.whisper_cpu_threads,
            download_root=settings.whisper_download_root,
            local_files_only=settings.whisper_local_files_only,
        )
        self.processor = EchoProcessor()
        self.tts = PiperTTS(
            model_path=settings.piper_model,
            target_sample_rate=settings.audio_sample_rate,
            use_cuda=settings.piper_use_cuda,
        )
        self.inference_gate = asyncio.Semaphore(settings.max_inference_jobs)

    def new_vad(self) -> VoiceActivityDetector:
        return VoiceActivityDetector(
            threshold=self.settings.vad_threshold,
            silence_ms=self.settings.vad_silence_ms,
            sample_rate=self.settings.audio_sample_rate,
        )
