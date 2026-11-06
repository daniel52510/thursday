# Security notes

THURSDAY is a personal assistant that runs on your own hardware. It is built for **one household on a trusted
network**, not for the open internet. This page says exactly what is reachable from where, what it assumes,
and the guardrails around the things that can cost money or leak something private.

Found a problem? Open a GitHub issue without details and ask for a private channel, or email the maintainer.

## What is exposed, and to whom

| Surface | Bound to | Reachable from | Protected by |
|---|---|---|---|
| Web UI + API (`thursday-core`, :8000) | `0.0.0.0` | your LAN, and your Tailscale tailnet | login (scrypt passwords, HttpOnly session cookie, same-origin check on writes) or a per-device token |
| Voice socket `/ws/voice` | same | same | login or device token |
| `/twilio/*` only | Tailscale **Funnel** on :8443 | **the internet** | Twilio's request signature (HMAC-SHA1) on every webhook and the media-stream handshake, plus one-use random call tokens |
| Apple Health ingest `/api/health/ingest` | same as the API | LAN / tailnet only, never Funnel | the owner's device token |
| Ollama :11434, TTS :9000/:9001, STT :9300 | `127.0.0.1` (native on the Mac) | the Mac and its containers | not exposed |
| SearXNG, ntfy | Docker network / LAN | LAN | ntfy topic is a long random secret |

**Nothing but `/twilio/*` is on the public internet.** Verified by probing the Funnel address: `/`, `/api/*`,
`/static/*` and path tricks (`/twilio/../`, `%2e%2e`) return 404; unsigned or forged webhook calls return 403.

Services bound to `0.0.0.0` have **no protection beyond the login**. Don't port-forward :8000 on your router, and
don't put it behind a public reverse proxy. For access away from home, use Tailscale (the tailnet is private to
your devices).

## Secrets

- All keys live in `.env` (gitignored, excluded from Docker builds by `.dockerignore`) and are never printed or logged.
- Credentials saved from the UI (Apple Calendar, music) are encrypted in the vault with a key outside the repo
  (`~/.config/thursday/master.key`, mounted read-only).
- Passwords are scrypt-hashed; session and device tokens are stored only as SHA-256, so a copy of the database
  can't log anyone in. Tokens can be revoked one by one (Settings → Devices).
- CI scans the whole git history for secrets (gitleaks) on every push.

## Private data stays on the Mac

The language model, speech-to-text and text-to-speech all run locally. Conversations, memories, contacts and
Apple Health data are in `brain/` (gitignored). Web search goes through your own SearXNG; only search queries
leave the house. Health data is the owner's only, never available to guests, missions or phone calls.

## Phone call guardrails (Twilio)

Calls cost money and talk to strangers, so they're the most fenced-in feature:

- **Who:** only the owner's own number and contacts he added himself (Settings → Calls). No tool can add a
  contact, so nothing said to THURSDAY, on a call or in a web page, can add a number. US/Canada (`+1`) only;
  900/976 premium numbers are refused (toll fraud).
- **Consent:** every call is read back and dials only after the owner's plain "yes" on the next turn; on the
  satellite, voice ID must recognize him.
- **Cost caps:** 10 calls a day (persisted, survives restarts), a hard 3-minute limit set on every call at
  Twilio, plus a soft goodbye at 2:30, a 12-turn cap and a 20-second silence hang-up.
- **On the call:** the model gets only the call's brief (the message and goal). No tools, no memory, no calendar.
  What the other person says is treated as data, never instructions; it never invents a number, address or
  payment detail, and shares the owner's name or number only when he approved it for that call.
- **Recommended in the Twilio console:** turn auto-recharge off and set Geo Permissions to US/Canada.

## Satellite (Raspberry Pi)

Connects out to the Mac with a device token over the LAN; listens for the wake word locally. Voice ID gates
sensitive tools (calendar, approvals, calls, health) to the owner's voice.

## Self-improvement runner

Code changes proposed by THURSDAY run in a sandbox container: no network beyond Ollama, no secrets, no Docker
socket, not root, the repo mounted read-only. Nothing merges without the owner's approval.
