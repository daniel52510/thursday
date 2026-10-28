# THURSDAY

A self-hosted, voice-first personal AI assistant. The target experience is a **local version of ChatGPT / Claude voice mode**: you talk, it answers almost instantly, and the conversation flows naturally.

## North star: it has to feel fast
Judge every change by how it affects the feel of a conversation, not just whether it works.
- **Latency budget:** aim for under ~1s from the end of user speech to the first audio. Treat time-to-first-audio as the main metric.
- **Stream everything:** LLM tokens → sentence chunks → TTS → playback. Never wait for a full reply when a partial one can start playing.
- **No dead air:** audio cues (`assets/sounds/thursday-ping.mp3` when listening starts, `thursday-thinking.mp3` looped until the first TTS audio plays) cover any wait that can't be avoided.
- **Interruptible:** barge-in stops playback and cancels in-flight LLM/TTS work.
- **Smooth playback:** no gaps, clicks, or overlap between chunks or cues.
- If a change adds latency, say so and give the cost in milliseconds where possible.

## Architecture
- **LLM:** Ollama runs **natively on macOS** (Mac Studio M4 Max, 64GB) with a Gemma model. Containers reach it at `http://host.docker.internal:11434`. Never containerize Ollama on the Mac, because Docker there can't use the GPU.
- **TTS:** Qwen TTS runs **natively** with MPS on port 9000 via `scripts/run_tts_native.sh`, for the same GPU reason.
- **Containers (Docker Compose):** `thursday-core` (FastAPI orchestrator, tool-calling, and the web UI in `static/index.html` at `:8000`) and faster-whisper STT.
- **Other hardware:** an Ubuntu server (RTX 4070 12GB) is available for offloading work. A Raspberry Pi Zero 2W is the audio satellite. OpenWakeWord is planned.

## How to work with me
**Work in small slices.** I'm watching my usage.
- One focused change per task. If a request is large, propose a breakdown into slices and wait for me to pick one.
- Read only the files the task needs. Don't scan the whole repo, re-read files you just edited, or dump large outputs. Use `grep` and targeted reads.
- Stop when the slice is done. Don't start the next one or refactor unrelated code without asking.

**Explain, don't just fix.** I want to understand the system.
- Before a non-trivial change, give a short plan: which files, what changes, and why.
- For bugs, find and explain the root cause before fixing it.
- Before running any command that deletes, overwrites, installs, or touches Docker volumes or images, say what it does and wait for my OK.

**Report concisely.** End each task with:
- **Changed:** files plus a one-line purpose each
- **Why:** 1–2 sentences
- **Latency impact:** better / worse / neutral (with numbers if measured)
- **Verify:** the command or step that proves it works
- **Next:** one suggested slice

Keep it short: no restating my request and no long recaps.

## Guardrails
- Never commit, print, or bake `.env` or other secrets into images. Keep `.dockerignore` and `.gitignore` covering them. This repo will be public.
- Services bound to `0.0.0.0` have no auth, so flag anything that would expose them beyond the LAN.
- Prefer measuring to guessing. Log per-stage timings (STT, time to first token, time to first audio).
