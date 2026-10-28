# Thursday AI Project
Thursday AI Assistant & Automator. Helps around your home by keeping you connected and up to date with the world!

## Purpose
The purpose of this project is to create an AI Assistant/Agent that is able to interact not only with the user but also interact with the user's applications. Applications such as

- Google Gmail
- Apple Music/YouTube
- Google Calendar/Apple Calendar
- Google Home/Apple HomeKit

The creation of a semi-agentic assistant can benefit users greatly. Additionally, edge computing will be involved as this AI will be connected to multiple Raspberry Pis around the home in order to provide users with assistance in everyday life as well as any agentic features.

## Tools and Dependencies Used
- LLM Model: Gemma 4 26B-A4B (Google) via native Ollama
- Language: Python
- API: FastAPI
- Storage: SQLite

## Milestones:
- **v0.1** - LLama Implementation and Text Chat + Tool Router
- **v0.2** - Real Smart Home Controls
- **v0.3** - Safety & Security Layer
- **v0.4** - Voice Loop (Wake Word Implementation)
- **v0.5** - Raspberry Pi Node Publish Events & Spatial Awareness

## Status
- Early Design / Scaffolding Stage

## Running on Apple Silicon (M-series Mac)
Docker Desktop on macOS can't pass the Apple GPU (Metal) into containers, so
the GPU-heavy services run **natively** and the rest stays in Docker:

| Service | Where | Accelerator |
|---|---|---|
| Ollama (LLM) | native macOS app | Metal / MLX |
| TTS (`voice_out.py`) | native, `scripts/run_tts_native.sh` | MPS |
| STT, core API + web UI, SearXNG | Docker (arm64) | CPU |

Containers reach the native services via `host.docker.internal`.

```bash
brew install sox ffmpeg          # audio tools qwen-tts expects
ollama pull gemma4:26b           # ~16 GB at Q4
./scripts/run_tts_native.sh      # TTS on :9000 (first run builds .venv-tts)
docker compose up -d --build     # core (UI at :8000), stt, searxng
```

Set `OLLAMA_MODEL=gemma4:26b` in `.env`. A CPU-only TTS container is still
available with `docker compose --profile tts-docker up`.

### THURSDAY's voice

TTS clones the voice in `voices/thursday.wav` with Qwen3-TTS 1.7B Base. To design
a new one from a text description (1.7B VoiceDesign), render candidates, listen,
and pick one, then restart the TTS service:

```bash
.venv-tts/bin/python scripts/design_voice.py --description "..."   # writes voices/candidates/
afplay voices/candidates/candidate_1.wav
.venv-tts/bin/python scripts/design_voice.py --pick 1
```

## venv-tts
- qwen-tts
- soundfile