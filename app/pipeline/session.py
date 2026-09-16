import asyncio
import logging
import time
from collections import deque
from contextlib import suppress
from uuid import uuid4

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from app.audio.buffer import AudioBuffer
from app.audio.vad import FRAME_SAMPLES, VADEvent, VoiceActivityDetector
from app.config import Settings
from app.interfaces import AudioProvider, Processor, SpeechToText, TextToSpeech

logger = logging.getLogger(__name__)


class VoiceSession:
    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        provider: AudioProvider,
        vad: VoiceActivityDetector,
        stt: SpeechToText,
        processor: Processor,
        tts: TextToSpeech,
        inference_gate: asyncio.Semaphore,
    ) -> None:
        self.id = uuid4().hex[:8]
        self.websocket = websocket
        self.settings = settings
        self.provider = provider
        self.vad = vad
        self.stt = stt
        self.processor = processor
        self.tts = tts
        self.inference_gate = inference_gate
        max_bytes = int(settings.audio_sample_rate * 2 * settings.max_utterance_seconds)
        self.buffer = AudioBuffer(max_bytes=max_bytes)
        frame_ms = FRAME_SAMPLES * 1_000 / settings.audio_sample_rate
        pre_roll_frames = max(1, round(settings.vad_pre_roll_ms / frame_ms))
        self.pre_roll: deque[bytes] = deque(maxlen=pre_roll_frames)
        self.speaking = False
        self.speech_started_at = 0.0

    async def run(self) -> None:
        started_at = time.monotonic()
        logger.info("[session:%s] connected", self.id)
        try:
            while time.monotonic() - started_at < self.settings.max_connection_seconds:
                try:
                    chunk = await asyncio.wait_for(
                        self.provider.receive(self.websocket),
                        timeout=self.settings.client_timeout_seconds,
                    )
                except TimeoutError:
                    await self.provider.send_event(self.websocket, "error", detail="client timeout")
                    await self.websocket.close(code=1001)
                    return
                if not chunk:
                    continue
                if len(chunk) > self.settings.max_message_bytes:
                    await self.provider.send_event(
                        self.websocket, "error", detail="frame too large"
                    )
                    await self.websocket.close(code=1009)
                    return
                for frame in self.vad.feed(chunk):
                    await self._handle_frame(frame.pcm, frame.event)
            await self.websocket.close(code=1000, reason="maximum connection duration reached")
        except WebSocketDisconnect:
            pass
        except ValueError as error:
            logger.warning("[session:%s] invalid_input error=%s", self.id, error)
            with suppress(RuntimeError, WebSocketDisconnect):
                await self.provider.send_event(self.websocket, "error", detail=str(error))
                await self.websocket.close(code=1003)
        except Exception:
            logger.exception("[session:%s] pipeline_error", self.id)
            with suppress(RuntimeError, WebSocketDisconnect):
                await self.provider.send_event(self.websocket, "error", detail="pipeline failure")
                await self.websocket.close(code=1011)
        finally:
            logger.info("[session:%s] disconnected", self.id)

    async def _handle_frame(self, pcm: bytes, event: VADEvent) -> None:
        if not self.speaking:
            self.pre_roll.append(pcm)
            if event is not VADEvent.SPEECH_START:
                return
            self.speaking = True
            self.speech_started_at = time.monotonic()
            for frame in self.pre_roll:
                self.buffer.append(frame)
            self.pre_roll.clear()
            logger.info("[session:%s] speech_start", self.id)
            await self.provider.send_event(self.websocket, "speech_start")
            return

        overflowed = self.buffer.append(pcm)
        if overflowed:
            logger.warning("[session:%s] utterance_limit", self.id)
            await self._finish_utterance()
            self.vad.reset()
        elif event is VADEvent.SPEECH_END:
            await self._finish_utterance()

    async def _finish_utterance(self) -> None:
        self.speaking = False
        duration = len(self.buffer) / (self.settings.audio_sample_rate * 2)
        audio = self.buffer.bytes()
        self.buffer.clear()
        if duration * 1_000 < self.settings.vad_min_speech_ms:
            logger.debug("[session:%s] speech_discarded duration=%.3f", self.id, duration)
            return

        logger.info("[session:%s] speech_end duration=%.3f", self.id, duration)
        async with self.inference_gate:
            started = time.perf_counter()
            text = await self.stt.transcribe(audio)
            logger.info(
                "[session:%s] stt_done duration=%.3f transcript=%r",
                self.id,
                time.perf_counter() - started,
                text,
            )
            if not text:
                return
            await self.provider.send_event(self.websocket, "transcript", text=text)
            response = await self.processor.process(text)
            if not response:
                return

            await self.provider.send_event(
                self.websocket,
                "audio_start",
                sample_rate=self.settings.audio_sample_rate,
            )
            started = time.perf_counter()
            bytes_sent = 0
            async for chunk in self.tts.synthesize(response):
                bytes_sent += len(chunk.pcm)
                await self.provider.send_audio(self.websocket, chunk.pcm)
            await self.provider.send_event(self.websocket, "audio_end")
            logger.info(
                "[session:%s] tts_done duration=%.3f bytes=%d",
                self.id,
                time.perf_counter() - started,
                bytes_sent,
            )
