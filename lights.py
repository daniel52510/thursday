"""LIFX bulbs on the home Wi-Fi, over LIFX's LAN protocol (UDP port 56700): no cloud, no account,
no package. LIFX_LIGHTS in .env lists their IPs, comma-separated; each bulb's name is its own label
(set in the LIFX app), so "turn off the desk lamp" finds it. One bulb answers in ~90 ms.

    light_set     power, brightness (or a change in it), color (names, a hex value, a white in kelvin)
                  or a scene ("movie mode": dim and warm)
    light_status  what each bulb is doing now

Every change fades over FADE_MS. Changes are acknowledged by the bulb, and he reports the state it
acknowledged (read straight back, the bulb is still mid-fade and reports where it's coming from).
"""
from __future__ import annotations

import os
import random
import re
import socket
import struct
from typing import Any, Dict

from tools import ToolResult

PORT = 56700
FADE_MS = 400
TRIES = 3
TIMEOUT_S = 0.4  # per try: UDP, so a lost packet is retried
_GET, _STATE, _SET_COLOR, _SET_POWER, _ACK = 101, 107, 102, 117, 45

# name -> (hue in degrees, saturation 0-1) or a white in kelvin
COLORS: dict[str, tuple[float, float] | int] = {
    "red": (0, 1), "orange": (30, 1), "amber": (40, 1), "yellow": (55, 1), "lime": (90, 1), "green": (120, 1),
    "teal": (170, 1), "cyan": (180, 1), "sky blue": (200, 0.7), "blue": (240, 1), "indigo": (260, 1),
    "purple": (280, 1), "violet": (290, 0.8), "magenta": (300, 1), "pink": (330, 0.6), "hot pink": (330, 1),
    "candle": 2000, "warm white": 2700, "soft white": 2700, "warm": 2700, "white": 3500, "neutral white": 4000,
    "cool white": 5000, "cool": 5000, "daylight": 5500, "bright white": 6500,
}


# scene -> (brightness %, color): what "movie mode" or "reading light" means
SCENES = {"movie": (10, "candle"), "relax": (40, "warm white"), "reading": (80, "neutral white"),
          "focus": (100, "cool white"), "night": (5, "candle"), "bright": (100, "white")}


def _ips() -> list[str]:
    return [ip.strip() for ip in os.getenv("LIFX_LIGHTS", "").split(",") if ip.strip()]


def _packet(kind: int, payload: bytes, seq: int, ack: bool) -> bytes:
    # Frame (size, protocol 1024 + addressable + tagged, source), address (any target, flags, sequence),
    # protocol header (message type). Little-endian, 36 bytes, then the payload.
    flags = 0x02 if ack else 0x01  # ack_required for changes, res_required for reads
    return struct.pack("<HHI8s6sBBQHH", 36 + len(payload), 1024 | 1 << 12 | 1 << 13, 0x7448,
                       bytes(8), bytes(6), flags, seq, 0, kind, 0) + payload


def _ask(ip: str, kind: int, payload: bytes = b"", want: int = _STATE) -> bytes | None:
    """Send one message, return the reply of type `want` (None if the bulb never answered)."""
    seq = random.randrange(256)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(TIMEOUT_S)
        for _ in range(TRIES):
            s.sendto(_packet(kind, payload, seq, ack=want == _ACK), (ip, PORT))
            try:
                while True:
                    data, _addr = s.recvfrom(1024)
                    if len(data) >= 36 and data[23] == seq and struct.unpack_from("<H", data, 32)[0] == want:
                        return data
            except socket.timeout:
                continue
    return None


def _state(ip: str) -> dict | None:
    data = _ask(ip, _GET)
    if not data or len(data) < 80:
        return None
    hue, sat, bri, kelvin, _r, power = struct.unpack_from("<HHHHhH", data, 36)
    return {"ip": ip, "name": data[48:80].split(b"\0")[0].decode(errors="replace").strip() or ip,
            "power": "on" if power else "off", "brightness": round(bri * 100 / 65535),
            "hue": round(hue * 360 / 65536), "saturation": round(sat * 100 / 65535), "kelvin": kelvin}


def _describe(st: dict) -> dict:
    """The state the way he'd say it: a color name if one is close, else the white's kelvin."""
    if st["saturation"] < 15:
        color = min((n for n, v in COLORS.items() if isinstance(v, int)),
                    key=lambda n: abs(COLORS[n] - st["kelvin"]))  # type: ignore[operator]
        color = f"{color} ({st['kelvin']}K)"
    else:
        color = min((n for n, v in COLORS.items() if not isinstance(v, int)),
                    key=lambda n: min(abs(COLORS[n][0] - st["hue"]), 360 - abs(COLORS[n][0] - st["hue"])))  # type: ignore[index]
    return {"name": st["name"], "power": st["power"], "brightness": st["brightness"], "color": color}


def _color(raw: Any, st: dict) -> tuple[int, int, int] | str:
    """(hue 0-65535, saturation 0-65535, kelvin) for a color name, a "#rrggbb" or "2700K"; or an error."""
    text = " ".join(str(raw).lower().replace("-", " ").split())
    if m := re.fullmatch(r"(\d{4,5})\s*k?(?:elvin)?", text):
        return 0, 0, max(1500, min(9000, int(m[1])))
    if m := re.fullmatch(r"#?([0-9a-f]{6})", text):
        r, g, b = (int(m[1][i:i + 2], 16) / 255 for i in (0, 2, 4))
        hi, lo = max(r, g, b), min(r, g, b)
        if hi == lo:
            return 0, 0, 3500
        h = (60 * ((g - b) / (hi - lo) % 6) if hi == r else 60 * ((b - r) / (hi - lo) + 2) if hi == g
             else 60 * ((r - g) / (hi - lo) + 4))
        return round(h / 360 * 65535), round((hi - lo) / hi * 65535), st["kelvin"]
    value = COLORS.get(text) or COLORS.get(text.removesuffix(" light").strip())
    if value is None:
        return f"unknown color '{raw}'"
    if isinstance(value, int):
        return 0, 0, value
    return round(value[0] / 360 * 65535), round(value[1] * 65535), st["kelvin"]


def _pick(raw: Any) -> tuple[list[dict], ToolResult | None]:
    """The bulbs a request means: by name (loosely), or all of them."""
    ips = _ips()
    if not ips:
        return [], ToolResult(ok=False, tool_name="light_set", error="no_lights_configured")
    states = [st for ip in ips if (st := _state(ip))]
    if not states:
        return [], ToolResult(ok=False, tool_name="light_set", error="lights_unreachable")
    want = re.sub(r"\b(the|my|light|lights|lamp|bulb)\b", "", str(raw or "").lower()).strip()
    if want and want not in ("all", "every", "everything"):
        named = [st for st in states if want in st["name"].lower() or st["name"].lower() in want]
        if named:
            return named, None
        if len(states) > 1:
            return [], ToolResult(ok=False, tool_name="light_set", error="unknown_light",
                                  data={"lights": [st["name"] for st in states]})
    return states, None


def light_set(args: Dict[str, Any]) -> ToolResult:
    picked, err = _pick(args.get("light"))
    if err:
        return err
    power, color = str(args.get("power") or "").lower(), args.get("color")
    bri, change = args.get("brightness"), args.get("brightness_change")
    if args.get("scene") is not None:
        scene = SCENES.get(re.sub(r"\s*(mode|light|lighting|scene)$", "", str(args["scene"]).lower().strip()))
        if scene is None:
            return ToolResult(ok=False, tool_name="light_set", error=f"unknown scene '{args['scene']}'",
                              data={"scenes": sorted(SCENES)})
        bri = scene[0] if bri is None else bri
        color = scene[1] if color is None else color
    done = []
    for st in picked:
        # A bulb keeps its brightness while off: "make it blue" turns it back on at that level.
        base = st["brightness"] if st["power"] == "on" else 0
        level = int(bri) if bri is not None else base + int(change) if change is not None else st["brightness"]
        level = max(0, min(100, level))
        asked = bri is not None or change is not None or color is not None
        off = power == "off" or (asked and level == 0)
        on = not off and (power == "on" or asked)
        if on and level == 0:  # "on" for a bulb last left at 0%
            level = 100
        target = dict(st)
        if on and (asked or st["brightness"] == 0):
            hsk = _color(color, st) if color is not None else (round(st["hue"] / 360 * 65536) % 65536,
                                                               round(st["saturation"] / 100 * 65535), st["kelvin"])
            if isinstance(hsk, str):
                return ToolResult(ok=False, tool_name="light_set", error=hsk, data={"colors": sorted(COLORS)})
            hue, sat, kelvin = hsk
            target.update(hue=round(hue * 360 / 65536), saturation=round(sat * 100 / 65535), kelvin=kelvin,
                          brightness=level)
            payload = struct.pack("<BHHHHI", 0, hue, sat, round(level / 100 * 65535), kelvin, FADE_MS)
            if not _ask(st["ip"], _SET_COLOR, payload, want=_ACK):
                return ToolResult(ok=False, tool_name="light_set", error="lights_unreachable")
        if on or off:
            payload = struct.pack("<HI", 0 if off else 65535, FADE_MS)
            if not _ask(st["ip"], _SET_POWER, payload, want=_ACK):
                return ToolResult(ok=False, tool_name="light_set", error="lights_unreachable")
        if on or off:
            target["power"] = "off" if off else "on"
        done.append(_describe(target))
    return ToolResult(ok=True, tool_name="light_set", data={"lights": done})


def light_status(args: Dict[str, Any]) -> ToolResult:
    picked, err = _pick(args.get("light"))
    if err:
        return ToolResult(ok=False, tool_name="light_status", error=err.error, data=err.data)
    return ToolResult(ok=True, tool_name="light_status", data={"lights": [_describe(st) for st in picked]})


_LIGHT = {"type": "string", "description": "Which light, by name; leave out for all of them"}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "light_set", "description": "Change the lights: on/off, brightness, color. Fades in 0.4 s.",
        "parameters": {"type": "object", "properties": {
            "light": _LIGHT,
            "power": {"type": "string", "enum": ["on", "off"]},
            "brightness": {"type": "integer", "description": "Percent, 1-100 (0 turns it off)"},
            "brightness_change": {"type": "integer", "description": "Percent up (+) or down (-), for 'brighter' "
                                  "(+25), 'dim it' (-25), 'a little' (+-10)"},
            "color": {"type": "string", "description": "A color (red, blue, purple, pink...), a white (candle, warm "
                      "white, white, cool white, daylight), '#ff8800' or kelvin like '2700K'"},
            "scene": {"type": "string", "enum": sorted(SCENES), "description": "A preset brightness and color: "
                      "movie (dim, warm), relax, reading, focus (bright, cool), night (very dim), bright"}}}}},
    {"type": "function", "function": {
        "name": "light_status", "description": "What the lights are doing now: on or off, brightness, color.",
        "parameters": {"type": "object", "properties": {"light": _LIGHT}}}},
]
NAMES = tuple(t["function"]["name"] for t in TOOL_SCHEMAS)
