# Changelog

## 0.1.0

- Added a self-hosted PCM16 WebSocket voice echo pipeline with Silero VAD, faster-whisper
  transcription, and streaming Piper speech synthesis.
- Kept the CPU runtime small with the bundled Silero ONNX model, INT8 greedy transcription,
  lazy shared models, bounded buffers, and limited inference concurrency.
- Added environment configuration, optional token authentication, a model downloader, a WAV
  test client, and focused tests.
- Added a live browser microphone client with in-browser 16 kHz resampling, streamed playback,
  transcript display, and feedback-loop prevention.
- Added a telephony WebSocket adapter with base64 G.711 A-law conversion, 20 ms outbound
  media packets, playback marks, and interruption clearing.
- Added native FastAPI deployment configuration for Vercel, including build-time model
  downloads and a connection limit below the Hobby function ceiling.
- Reduced the deployment bundle by omitting the unused media decoder and Xet client, storing
  the Whisper model without a duplicate cache layout, and enabling Fluid Compute.
