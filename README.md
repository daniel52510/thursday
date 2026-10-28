# THURSDAY 🤖
**Tool-Handling, User-Respecting, Self-Hosted Digital Assistant (Yours)**

THURSDAY is a local-first AI assistant designed to feel like a real “home operator”: conversational, up-to-date (via web search tooling), and eventually voice-enabled across Raspberry Pi “satellite” nodes. The goal is simple: **make an assistant that’s genuinely useful day-to-day, while keeping control and privacy in your hands.**

---

## Why THURSDAY exists
Most assistants are either:
- smart but **not private** (cloud-first),
- private but **not useful** (offline-only + no tools),
- or useful but **not reliable** (tooling is brittle, no guardrails, no observability).

THURSDAY is built to be:
- **Local-first** (run on your own hardware)
- **Tool-capable** (weather, time, web search, and more)
- **Extensible** (drop-in tools + standardized ToolResult format)
- **Deployable** (Docker/Compose, server-ready)
- **Security-oriented** (treat web content as untrusted; minimize exposure)

---

## Current Capabilities (Progress so far)
### Core Agent Loop
- Structured tool-calling with **Pydantic-validated** agent responses
- Tool execution + ToolResult envelope (success/error/meta) for reliability
- Second-pass summarization supported (with a push toward **sanitizing tool output before summarizing** for speed)

### Tools
- **Web Search** via **SearXNG** (container-friendly, local endpoint)
- **Weather** via **Open-Meteo** (geocode + forecast) with improved disambiguation logic
- **Time / Timezone** support (IANA timezone handling)
- More tools planned: tasks, reminders, “open_url”, calendar/email integrations, etc.

### UI / API
- **FastAPI** server for chat/tool routing
- **Streamlit** UI (optional front-end; helpful for rapid iteration)

### Infrastructure
- Docker-ready (THURSDAY + dependencies can be run as services)
- Designed to move from dev machine → **Ubuntu server** (eventually leveraging RTX 4070 via GPU inference through Ollama)

---

## Architecture (High Level)
THURSDAY is intentionally modular:

- **THURSDAY (Brain)**: Python app (tool router + memory)
- **SearXNG (Eyes)**: web_search backend
- **Ollama (Voice/Brain Runtime)**: local LLM inference endpoint
- **SQLite (Memory)**: local “brain” database + persistence

Typical container topology
- `thursday` → calls `searxng` for web_search
- `thursday` → calls `ollama` for LLM inference

---

## Tech Stack
- **Language:** Python
- **API:** FastAPI
- **UI:** Streamlit (optional)
- **LLM Runtime:** Ollama
- **Models:** Qwen (local-first experimentation; configurable)
- **Web Search:** SearXNG (self-hosted)
- **Storage:** SQLite (“brain”)
- **Deployment:** Docker + Docker Compose
- **Planned Observability:** Prometheus + Grafana

---
