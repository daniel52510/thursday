"""THURSDAY's health check (system_status): is every part up, how much room is left, how fast the last turns were.

One read-only snapshot, gathered in about a second: Ollama (loaded models, context, size), the Mac's memory
(scripts/host_mem.sh, mounted at /hostmem), the voice and speech-to-text servers, connected satellites, the
self-improvement runner, the brain disk, and recent turn latency. "How are you doing?" in engineer terms.
"""
from __future__ import annotations

import json
import logging
from collections import deque
import os
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Dict

import httpx

from accounts import OWNER
from agent_loop import URL as OLLAMA_URL
from calendar_tool import TURN
from tools import ToolResult
OLLAMA_BASE = OLLAMA_URL.split("/api/")[0]
HOSTMEM = Path(os.getenv("HOSTMEM_FILE", "/hostmem/host-mem.json"))
RUNNER_OUT = Path(os.getenv("REPO_DIR", "/repo")) / "runner-out"
BRAIN = Path(os.getenv("THURSDAY_BRAIN_DIR", "brain"))


# The last warnings and errors THURSDAY logged, newest last, for the health page: the "why" behind a red dot.
RECENT_WARNINGS: deque = deque(maxlen=20)


class _Remember(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            RECENT_WARNINGS.append({"at": record.created, "level": record.levelname,
                                    "where": record.name.removeprefix("thursday."), "message": record.getMessage()[:300]})
        except Exception:
            pass


_remember = _Remember(level=logging.WARNING)
for _name in ("thursday", "thursday.voice"):  # thursday.voice doesn't propagate (its own handler)
    logging.getLogger(_name).addHandler(_remember)


def _get(url: str) -> tuple[bool, Any]:
    try:
        r = httpx.get(url, timeout=3)
        return r.status_code < 400, (r.json() if "json" in r.headers.get("content-type", "") else r.status_code)
    except (httpx.HTTPError, ValueError) as exc:
        return False, type(exc).__name__


def _age_s(path: Path) -> float | None:
    try:
        return round(time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _p50(rows: list[dict], key: str) -> int | None:
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(statistics.median(vals)) if vals else None


def system_status(args: Dict[str, Any]) -> ToolResult:
    if (TURN.get() or {}).get("user") != OWNER:
        return ToolResult(ok=False, tool_name="system_status", error="diagnostics_are_daniels_only")
    data = snapshot()
    data.pop("services", None), data.pop("recent_warnings", None), data.pop("mac", None)  # the page's; out loud he needs the summary
    return ToolResult(ok=True, tool_name="system_status", data={**data,
        "note": "Your own health check. Say what's wrong first (problems), then the one or two numbers that matter, "
                "in 2-3 sentences. If nothing's wrong, say so and give the reply speed. No lists out loud."})


def snapshot() -> dict:
    """Everything system_status says, plus a per-service list for the health page (/api/health)."""
    import approvals
    import presence
    import telephony
    import voice_ws  # here: voice_ws imports the tools
    problems: list[str] = []

    ok, ps = _get(f"{OLLAMA_BASE}/api/ps")
    models = [{"name": m.get("name"), "gb": round(m.get("size", 0) / 1e9, 1), "context": m.get("context_length")}
              for m in (ps.get("models", []) if ok and isinstance(ps, dict) else [])]
    if not ok:
        problems.append(f"Ollama isn't answering ({ps})")

    mac = None
    try:
        mem = json.loads(HOSTMEM.read_text())
        if "cpu_pct" in mem:  # scripts/host_mem.sh since 2026-11-06: CPU and GPU too
            total = mem.get("total_gb") or 0
            mac = {"cpu_pct": mem["cpu_pct"], "load1": mem.get("load1"), "cores": mem.get("cores"),
                   "gpu_pct": mem.get("gpu_pct"), "gpu_mem_gb": mem.get("gpu_mem_gb"),
                   "ram_used_gb": round(total - mem.get("available_gb", 0), 1) if total else None, "ram_total_gb": total,
                   "swap_used_gb": round(mem.get("swap_used_mb", 0) / 1024, 1),
                   "swap_total_gb": round(mem.get("swap_total_mb", 0) / 1024, 1), "age_s": round(time.time() - mem["at"])}
        mem_age = round(time.time() - mem.get("at", 0))
        memory = {"available_gb": mem.get("available_gb"), "pressure": {1: "normal", 2: "warning", 4: "critical"}.get(
            mem.get("pressure"), mem.get("pressure")), "reading_age_s": mem_age}
        if mem.get("pressure", 1) != 1:
            problems.append(f"memory pressure is {memory['pressure']}")
        if mem_age > 120:
            problems.append("the memory reading is stale (com.thursday.hostmem not running?)")
    except (OSError, ValueError):
        memory = None

    tts_ok, tts = _get(f"{voice_ws.TTS_BASE_URL}/health")
    stt_ok, _ = _get(f"{voice_ws.STT_BASE_URL}/health")
    if not tts_ok:
        problems.append("the voice server (TTS) is down")
    if not stt_ok:
        problems.append("speech-to-text is down")

    sats = [{"state": s.state} for s in voice_ws.SATELLITES]
    if not sats:
        problems.append("no satellite is connected (it can't hear the wake word)")
    queue = len(list((RUNNER_OUT / "queue").glob("*.json"))) if (RUNNER_OUT / "queue").exists() else 0
    busy = _age_s(RUNNER_OUT / "runner-busy")
    disk = shutil.disk_usage(BRAIN if BRAIN.exists() else "/")

    recent = [r for r in voice_ws.RECENT_TURNS if time.time() - r["at"] < 6 * 3600]
    turns = {"count_6h": len(recent), "p50_to_first_audio_ms": _p50(recent, "speech_end_to_playback_ms"),
             "p50_first_token_ms": _p50(recent, "llm_ttft_ms"), "p50_stt_ms": _p50(recent, "stt_ms")}
    if (turns["p50_to_first_audio_ms"] or 0) > 2500:
        problems.append(f"replies are slow: {turns['p50_to_first_audio_ms']} ms to first audio (median)")

    ntfy_ok = bool(approvals.NTFY_URL) and _get(f"{approvals.NTFY_URL}/v1/health")[0]
    calls_on = telephony.configured()
    sat_detail = ", ".join(s["state"] for s in sats) if sats else "none connected"
    services = [
        {"name": "Language model", "ok": ok, "detail": ", ".join(f"{m['name']} ({m['gb']} GB)" for m in models)
         or ("nothing loaded" if ok else f"Ollama isn't answering ({ps})")},
        {"name": "Voice (TTS)", "ok": tts_ok, "detail": f"{tts.get('engine', '')} {tts.get('voice', '')}".strip()
         if isinstance(tts, dict) else "down"},
        {"name": "Speech to text", "ok": stt_ok, "detail": "up" if stt_ok else "down"},
        {"name": "Satellite", "ok": bool(sats), "detail": sat_detail},
        {"name": "Phone calls", "ok": calls_on, "detail": ("two-way" if telephony.TWO_WAY else "one-way")
         if calls_on else "off (TWILIO_PUBLIC_URL not set)"},
        {"name": "Push (ntfy)", "ok": ntfy_ok, "detail": "up" if ntfy_ok else "down or not set up"},
        {"name": "Presence", "ok": presence.phone() in ("home", "away"),
         "detail": {"home": "phone is home", "away": "phone is out", "offline": "phone offline",
                    "unknown": "no reading (helper not running?)"}[presence.phone()]},
        {"name": "Mac memory", "ok": bool(memory) and memory["pressure"] == "normal",
         "detail": f"{memory['available_gb']} GB free, pressure {memory['pressure']}" if memory else "no reading"},
        {"name": "Brain disk", "ok": disk.free > 5e9, "detail": f"{round(disk.free / 1e9, 1)} GB free"},
    ]
    if not ntfy_ok and approvals.NTFY_URL:
        problems.append("push notifications (ntfy) are down")
    return {
        "mac": mac,
        "services": services,
        "recent_warnings": list(RECENT_WARNINGS),
        "problems": problems,
        "ollama": {"up": ok, "loaded": models},
        "mac_memory": memory,
        "voice_server": {"up": tts_ok, "engine": tts.get("engine") if isinstance(tts, dict) else None,
                         "voice": tts.get("voice") if isinstance(tts, dict) else None},
        "speech_to_text": {"up": stt_ok},
        "satellites": {"connected": len(sats), "states": [s["state"] for s in sats]},
        "runner": {"queued_jobs": queue, "busy": busy is not None and busy < 120},
        "brain_disk_free_gb": round(disk.free / 1e9, 1),
        "recent_turns": turns,
    }

