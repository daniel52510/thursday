"""Mission research tools (missions.py): reading long pages in parts, following links, remembering."""
import missions

PAGE = ("<html><head><title>Vacuums</title></head><body><article><h1>Best robot vacuums</h1>"
        + "".join(f"<p>Paragraph {i} about suction and dog hair and battery life, tested at home for weeks.</p>" for i in range(300))
        + '<p>See <a href="/roborock-q5">Roborock Q5 review</a> and <a href="https://example.org/shop">the shop</a>,'
        ' <a href="#top">top</a>, <a href="javascript:void(0)">x</a>.</p></article></body></html>')


class FakeResp:
    is_redirect, ok, status_code, encoding = False, True, 200, "utf-8"
    headers = {"content-type": "text/html"}

    def iter_content(self, n):
        yield PAGE.encode()

    def close(self):
        pass


def test_long_page_reads_on_and_links_come_back(monkeypatch):
    monkeypatch.setattr(missions, "_public", lambda url: True)
    monkeypatch.setattr(missions.requests, "get", lambda *a, **k: FakeResp())
    focused = missions.read_page({"url": "https://example.com/vacuums", "focus": "roborock"})
    assert focused.data["links"][0] == {"text": "Roborock Q5 review", "url": "https://example.com/roborock-q5"}  # focus first
    assert all(link["url"].startswith("http") for link in focused.data["links"])  # no #top, no javascript:
    first = missions.read_page({"url": "https://example.com/vacuums"})
    assert first.ok and first.data["truncated"] and first.data["next_start"] == missions.PAGE_CHARS
    more = missions.read_page({"url": "https://example.com/vacuums", "start": first.data["next_start"]})
    assert more.ok and more.data["text"] and more.data["text"] != first.data["text"] and "links" not in more.data


class FakeDB:
    def __init__(self, done):
        self.done = done

    def list_missions(self, user, status=None, limit=20):
        return self.done if status == "done" else []


def _m(i, goal, summary="found things", **k):
    return {"id": i, "goal": goal, "summary": summary, "announced_at": "x", "watch_id": None, "finished_at": "2026-11-01", **k}


def test_related_missions_need_two_shared_words():
    db = FakeDB([_m("a", "Best robot vacuum under $400 for a dog"), _m("b", "Best pizza near me"),
                 _m("c", "Robot lawn mower prices"), _m("d", "Watch: vacuum robot deals", watch_id="w")])
    got = missions._related(db, {"id": "new", "user_id": "owner", "goal": "Compare robot vacuum models for dog hair"})
    assert [m["id"] for m in got] == ["a"]
