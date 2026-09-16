import logging
import secrets
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from app.config import get_settings
from app.interfaces import AudioProvider
from app.pipeline.session import VoiceSession
from app.providers.raw import RawPCMProvider
from app.providers.telephony import TelephonyProvider
from app.services import Services

settings = get_settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Echo Voice", version="0.1.0")
app.state.services = Services(settings)
frontend_dir = Path(__file__).parent / "frontend"


@app.get("/", include_in_schema=False)
async def live_client() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/audio-processor.js", include_in_schema=False)
async def audio_processor() -> FileResponse:
    return FileResponse(frontend_dir / "audio-processor.js", media_type="text/javascript")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/voice")
async def voice(websocket: WebSocket) -> None:
    await _run_voice_session(websocket, RawPCMProvider())


@app.websocket("/voice/telephony")
async def telephony_voice(websocket: WebSocket) -> None:
    await _run_voice_session(
        websocket,
        TelephonyProvider(max_message_bytes=settings.max_message_bytes),
    )


async def _run_voice_session(websocket: WebSocket, provider: AudioProvider) -> None:
    supplied_token = websocket.query_params.get("token", "")
    if settings.voice_token and not secrets.compare_digest(supplied_token, settings.voice_token):
        await websocket.close(code=1008, reason="invalid token")
        return

    await websocket.accept()
    services: Services = websocket.app.state.services
    session = VoiceSession(
        websocket=websocket,
        settings=services.settings,
        provider=provider,
        vad=services.new_vad(),
        stt=services.stt,
        processor=services.processor,
        tts=services.tts,
        inference_gate=services.inference_gate,
    )
    await session.run()
