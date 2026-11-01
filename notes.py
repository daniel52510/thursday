"""Lists and notes he keeps for you ("add eggs to the grocery list", "remember the shelf is 34
inches", "what's on my packing list?"). Each user's own, by the user in the agent's TURN,
which the model can't set. Missions can add to them too ("start a packing list for Japan").

List names are normalized so "groceries", "my shopping list" and "grocery" are one list.
Removing matches loosely ("milk" takes "2% milk"), because that's how people say it.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from agent_loop import initalize_db
from calendar_tool import TURN
from tools import ToolResult

MAX_ITEMS = 100   # per list
NOTE_CHARS = 1500  # of each note, back to the model
_SAME = {"groceries": "grocery", "shopping": "grocery", "food": "grocery", "to do": "todo", "todos": "todo",
         "tasks": "todo", "chores": "todo"}


def _user(tool: str) -> tuple[str, ToolResult | None]:
    user = (TURN.get() or {}).get("user")
    if not user:
        return "", ToolResult(ok=False, tool_name=tool, error="unknown_user")
    return user, None


def _list_name(raw: Any) -> str:
    name = re.sub(r"[^a-z0-9 ]", "", str(raw or "").lower()).strip()
    name = re.sub(r"^(my|the|our|a)\s+", "", name)
    name = re.sub(r"\s*\blists?$", "", name).strip()
    return _SAME.get(name, name)


def _items(raw: Any) -> list[str]:
    items = raw if isinstance(raw, list) else [raw]
    return [" ".join(str(i).split())[:200] for i in items if str(i or "").strip()]


def _names(user: str) -> list[dict]:
    counts: dict[str, int] = {}
    for row in initalize_db().list_items(user):
        counts[row["list_name"]] = counts.get(row["list_name"], 0) + 1
    return [{"list": k, "items": v} for k, v in counts.items()]


def list_add(args: Dict[str, Any]) -> ToolResult:
    user, err = _user("list_add")
    if err:
        return err
    name, items = _list_name(args.get("list")), _items(args.get("items"))
    if not name or not items:
        return ToolResult(ok=False, tool_name="list_add", error="needs a list name and at least one item")
    db = initalize_db()
    have = db.list_items(user, name)
    seen = {r["text"].lower() for r in have}
    added = [i for i in dict.fromkeys(items) if i.lower() not in seen][:max(0, MAX_ITEMS - len(have))]
    for item in added:
        db.add_list_item(user, name, item)
    return ToolResult(ok=True, tool_name="list_add", data={
        "list": name, "added": added, "already_there": [i for i in items if i.lower() in seen],
        "total": len(have) + len(added)})


def list_show(args: Dict[str, Any]) -> ToolResult:
    user, err = _user("list_show")
    if err:
        return err
    name = _list_name(args.get("list"))
    if not name:
        return ToolResult(ok=True, tool_name="list_show", data={"lists": _names(user)})
    items = [r["text"] for r in initalize_db().list_items(user, name)]
    data = {"list": name, "items": items}
    if not items:
        data["other_lists"] = [n["list"] for n in _names(user)]
    return ToolResult(ok=True, tool_name="list_show", data=data)


def list_remove(args: Dict[str, Any]) -> ToolResult:
    user, err = _user("list_remove")
    if err:
        return err
    name = _list_name(args.get("list"))
    db = initalize_db()
    rows = db.list_items(user, name) if name else []
    if not rows:
        return ToolResult(ok=False, tool_name="list_remove", error="no_such_list",
                          data={"lists": [n["list"] for n in _names(user)]})
    if args.get("clear"):
        db.delete_list_items(user, [r["id"] for r in rows])
        return ToolResult(ok=True, tool_name="list_remove", data={"list": name, "cleared": len(rows)})
    removed, missing = [], []
    for want in _items(args.get("items")):
        w = want.lower()
        hit = next((r for r in rows if w in r["text"].lower() or r["text"].lower() in w), None)
        if hit:
            rows.remove(hit)
            removed.append(hit)
        else:
            missing.append(want)
    db.delete_list_items(user, [r["id"] for r in removed])
    return ToolResult(ok=True, tool_name="list_remove", data={
        "list": name, "removed": [r["text"] for r in removed], "not_found": missing, "left": len(rows)})


def note_save(args: Dict[str, Any]) -> ToolResult:
    user, err = _user("note_save")
    if err:
        return err
    title = " ".join(str(args.get("title") or "").lower().split())[:80]
    text = str(args.get("text") or "").strip()[:4000]
    if not title or not text:
        return ToolResult(ok=False, tool_name="note_save", error="needs a title and text")
    note = initalize_db().save_note(user, title, text)
    return ToolResult(ok=True, tool_name="note_save", data={"title": title, "added_to_existing": note["body"] != text})


def note_find(args: Dict[str, Any]) -> ToolResult:
    user, err = _user("note_find")
    if err:
        return err
    notes = initalize_db().list_notes(user)
    terms = [t for t in re.findall(r"[a-z0-9]+", str(args.get("query") or "").lower()) if len(t) > 2]
    if terms:
        scored = [(sum(t in f"{n['title']} {n['body']}".lower() for t in terms), n) for n in notes]
        notes = [n for score, n in sorted(scored, key=lambda x: -x[0]) if score]
    found = [{"title": n["title"], "text": n["body"][:NOTE_CHARS], "updated": n["updated_at"][:10]} for n in notes[:5]]
    return ToolResult(ok=True, tool_name="note_find", data={"notes": found, "total_notes": len(notes)})


def note_delete(args: Dict[str, Any]) -> ToolResult:
    user, err = _user("note_delete")
    if err:
        return err
    title = " ".join(str(args.get("title") or "").lower().split())
    if not initalize_db().delete_note(user, title):
        return ToolResult(ok=False, tool_name="note_delete", error="no_such_note: use note_find for the exact title")
    return ToolResult(ok=True, tool_name="note_delete", data={"deleted": title})


_LIST = {"type": "string", "description": "Short list name: 'grocery', 'packing', 'todo', 'gift ideas'"}
_ITEMS = {"type": "array", "items": {"type": "string"}, "description": "One entry per item, e.g. ['eggs', 'oat milk']"}
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_add", "description": "Add items to one of his lists (created if new).",
        "parameters": {"type": "object", "properties": {"list": _LIST, "items": _ITEMS}, "required": ["list", "items"]}}},
    {"type": "function", "function": {
        "name": "list_show", "description": "What's on a list; without a list name, which lists he has.",
        "parameters": {"type": "object", "properties": {"list": _LIST}}}},
    {"type": "function", "function": {
        "name": "list_remove", "description": "Take items off a list: whenever he says he got, bought, did or finished "
        "something, or to cross off, remove or clear. The list only changes when you call this.",
        "parameters": {"type": "object", "properties": {"list": _LIST, "items": _ITEMS,
                                                         "clear": {"type": "boolean", "description": "Empty the list"}},
                       "required": ["list"]}}},
    {"type": "function", "function": {
        "name": "note_save", "description": "Write something down for him to look up later (measurements, codes he "
        "chose to share, ideas, what he decided). Same title adds to that note.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string", "description": "A few words he'd use to ask for it: 'shelf', 'wifi at the cabin'"},
            "text": {"type": "string"}}, "required": ["title", "text"]}}},
    {"type": "function", "function": {
        "name": "note_find", "description": "Look up his notes by what they're about; without a query, the latest.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "note_delete", "description": "Delete one note, by its exact title from note_find.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}},
]
NAMES = tuple(t["function"]["name"] for t in TOOL_SCHEMAS)
