import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = "INFO"
    voice_token: str = ""

    audio_sample_rate: int = 16_000
    max_message_bytes: int = Field(default=65_536, ge=1_024)
    max_utterance_seconds: float = Field(default=30.0, gt=0)
    max_connection_seconds: float = Field(
        default_factory=lambda: 285.0 if os.getenv("VERCEL") else 3_600.0,
        gt=0,
    )
    client_timeout_seconds: float = Field(default=60.0, gt=0)

    vad_threshold: float = Field(default=0.5, gt=0, lt=1)
    vad_silence_ms: int = Field(default=450, ge=32)
    vad_min_speech_ms: int = Field(default=160, ge=32)
    vad_pre_roll_ms: int = Field(default=160, ge=0)

    whisper_model: str = "models/whisper-tiny-en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = "en"
    whisper_cpu_threads: int = Field(default=0, ge=0)
    whisper_download_root: Path = Path("models/whisper")
    whisper_local_files_only: bool = True

    piper_model: Path = Path("models/en_US-lessac-low.onnx")
    piper_use_cuda: bool = False
    max_inference_jobs: int = Field(default=1, ge=1)

    @field_validator("audio_sample_rate")
    @classmethod
    def validate_sample_rate(cls, value: int) -> int:
        if value != 16_000:
            raise ValueError("Silero streaming VAD currently requires AUDIO_SAMPLE_RATE=16000")
        return value

    @field_validator("whisper_language", mode="before")
    @classmethod
    def empty_language_is_auto(cls, value: object) -> object:
        return None if value == "" else value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
