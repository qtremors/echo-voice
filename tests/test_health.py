import httpx

from app.config import Settings
from app.main import app


async def test_health() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_live_client_assets() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        page = await client.get("/")
        processor = await client.get("/audio-processor.js")
    assert page.status_code == 200
    assert "Start microphone" in page.text
    assert processor.status_code == 200
    assert "registerProcessor" in processor.text


def test_empty_environment_values_use_defaults(monkeypatch) -> None:
    empty_variables = [
        "PORT",
        "AUDIO_SAMPLE_RATE",
        "MAX_MESSAGE_BYTES",
        "MAX_UTTERANCE_SECONDS",
        "MAX_CONNECTION_SECONDS",
        "CLIENT_TIMEOUT_SECONDS",
        "VAD_THRESHOLD",
        "VAD_SILENCE_MS",
        "VAD_MIN_SPEECH_MS",
        "VAD_PRE_ROLL_MS",
        "WHISPER_MODEL",
        "WHISPER_CPU_THREADS",
        "WHISPER_LOCAL_FILES_ONLY",
        "PIPER_MODEL",
        "PIPER_USE_CUDA",
        "MAX_INFERENCE_JOBS",
    ]
    for name in empty_variables:
        monkeypatch.setenv(name, "")
    monkeypatch.delenv("VERCEL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.port == 8_000
    assert settings.audio_sample_rate == 16_000
    assert settings.whisper_model == "models/whisper-tiny-en"
    assert settings.whisper_local_files_only is True
    assert settings.piper_use_cuda is False
