"""THURSDAY on his iPhone (phone.py, /api/phone): the phone_ tools only record what the phone should do, only on a
phone turn; the Shortcut gets flat keys; and the phone tools are offered only to phone turns."""
import pytest

import agent_stream
import phone


@pytest.fixture(autouse=True)
def prefs(monkeypatch, tmp_path):
    monkeypatch.setattr(phone, "PREFS_FILE", str(tmp_path / "phone.json"))


def _turn():
    box = {}
    return box, phone.ACTION.set(box)


def test_call_goes_to_his_phone():
    box, t = _turn()
    try:
        assert phone.phone_call({"contact": "my Mom"}).ok and box == {"do": "call", "contact": "Mom"}
    finally:
        phone.ACTION.reset(t)


def test_a_text_sends_only_after_his_spoken_yes():
    from calendar_tool import TURN
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "text Sam"})
    box, t = _turn()
    try:
        r = phone.phone_message({"contact": "Sam", "text": "Running   10 late"})
        assert r.ok and "NOT SENT" in r.data["status"] and box == {}       # nothing for the phone yet
        out = phone.respond("Text Sam: 'Running 10 late.' Send it?")
        assert "do" not in out and "url" not in out and out["listen"] == "yes"                    # it listens for his answer
    finally:
        phone.ACTION.reset(t)
    assert phone.confirm_message("owner", "what's the weather") is None     # not an answer: the model takes over
    assert phone.confirm_message("owner", "yes") is None                    # and the pending text was dropped
    phone._pending["owner"] = {"contact": "Sam", "body": "Running 10 late", "at": __import__("time").monotonic()}
    assert phone.confirm_message("owner", "nope") == {"speak": "Okay, not sending it."}
    phone._pending["owner"] = {"contact": "Sam", "body": "Running 10 late", "at": __import__("time").monotonic()}
    sent = phone.confirm_message("owner", "yes, send it")
    assert (sent["do"], sent["contact"], sent["body"]) == ("message", "Sam", "Running 10 late")
    assert phone.confirm_message("owner", "yes") is None                    # one yes, one text


def test_music_opens_in_his_app_and_directions_and_timer():
    box, t = _turn()
    try:
        phone.phone_music({"query": "Tame Impala"})
        assert box["url"] == "https://music.youtube.com/search?q=Tame%20Impala"
        phone.set_prefs("apple")
        phone.phone_music({"query": "workout playlist"})
        assert box["url"].startswith("https://music.apple.com/us/search?term=workout")
        phone.phone_directions({"destination": "Whole Foods"})
        assert box["url"] == "https://maps.apple.com/?daddr=Whole%20Foods&dirflg=d"
        phone.phone_timer({"minutes": 10})
        assert box == {"do": "timer", "minutes": 10}
        assert not phone.phone_timer({"minutes": 0}).ok
    finally:
        phone.ACTION.reset(t)


def test_not_a_phone_turn_does_nothing():
    r = phone.phone_call({"contact": "Mom"})
    assert not r.ok and r.error.startswith("only_from_his_iphone")


def test_listen_again_when_he_was_asked_something():
    box, t = _turn()
    try:
        assert phone.respond("Which Sam, question?")["listen"] == "yes"
        assert phone.respond("Done.") == {"speak": "Done."}            # nothing empty for the Shortcut to trip on
    finally:
        phone.ACTION.reset(t)


@pytest.mark.parametrize("said,hit", [("call Mom", True), ("text Sam I'm late", True), ("play some Tame Impala", True),
                                      ("tell Jordana I'm on my way", True), ("tell me a joke", False),
                                      ("what's the weather", False), ("set a timer for 5 minutes", True)])
def test_phone_requests_are_recognised(said, hit):
    assert bool(agent_stream._PHONE_ASK.search(said)) is hit


def test_phone_tools_are_only_for_phone_turns():
    names = {t["function"]["name"] for t in agent_stream.TOOLS}
    assert not names & {"phone_call", "phone_message", "phone_music"}


@pytest.mark.parametrize("kwargs", [
    {"json": {"text": "what time is it"}}, {"json": {"Text": "what time is it"}}, {"json": {" TEXT ": "what time is it"}}, {"json": {"text =": "what time is it"}},
    {"data": {"text": "what time is it"}}, {"data": {"Text": "what time is it"}}])
def test_shortcut_body_any_shape(kwargs, monkeypatch):
    """JSON or Form, "text" in any case: a Shortcut's field boxes make all of these."""
    from fastapi.testclient import TestClient
    import server
    monkeypatch.setattr(server, "stream_prompt", _fake_stream)
    server.app.dependency_overrides[server.current_user] = lambda: {"id": "owner"}
    try:
        r = TestClient(server.app).post("/api/phone", **kwargs)
        assert r.status_code == 200 and r.json()["speak"] == "It's noon."
        assert TestClient(server.app).post("/api/phone", json={"words": "hi"}).status_code == 422
    finally:
        server.app.dependency_overrides.clear()


async def _fake_stream(text, mode, conversation_id=None, context=None, user=None):
    assert text == "what time is it" and context["channel"] == "phone"
    yield {"type": "conversation", "id": "c1"}
    yield {"type": "done", "reply": "It's noon."}


def test_every_link_is_https_for_the_shortcuts_check():
    """The Shortcut opens url only when it "contains http" (its "has any value" passed with no url, 2026-11-09)."""
    box, t = _turn()
    try:
        for app in phone.MUSIC_APPS:
            phone.set_prefs(app)
            phone.phone_music({"query": "Tame Impala"})
            assert box["url"].startswith("https://")
        phone.phone_directions({"destination": "home"})
        assert box["url"].startswith("https://")
    finally:
        phone.ACTION.reset(t)


def test_apple_music_opens_the_page_itself_from_the_artists_catalog(monkeypatch):
    import httpx

    def fake_get(url, params=None, **kw):
        if url.endswith("/search"):     # the artist
            data = {"results": [{"artistId": 1, "artistName": "Tame Impala", "artistLinkUrl": "https://music.apple.com/a/1?uo=4"}]}
        else:                           # that artist's songs
            data = {"results": [{"trackName": "The Less I Know the Better (Club Edit)", "trackViewUrl": "https://music.apple.com/x?i=1&uo=4"},
                                {"trackName": "The Less I Know the Better", "trackViewUrl": "https://music.apple.com/y?i=2&uo=4"},
                                {"trackName": "Borderline", "trackViewUrl": "https://music.apple.com/z?i=3&uo=4"}]}
        return type("R", (), {"json": lambda self: data})()
    monkeypatch.setattr(httpx, "get", fake_get)
    assert phone._apple_page("The Less I Know The Better", "song", "Tame Impala") == "https://music.apple.com/y?i=2"
    assert phone._apple_page("Tame Impala", "artist") == "https://music.apple.com/a/1"
    assert phone._apple_page("The Less I Know The Better", "song") is None   # no artist: the search page, not a guess
    phone.set_prefs("apple")
    box, t = _turn()
    try:
        assert phone.phone_music({"query": "The Less I Know The Better", "artist": "Tame Impala", "kind": "song"}).data["straight_to_it"]
        assert box["url"] == "https://music.apple.com/y?i=2"
        phone.phone_music({"query": "workout", "kind": "playlist"})            # playlists: the search page
        assert "search?term=workout" in box["url"]
    finally:
        phone.ACTION.reset(t)


def test_kind_and_artist_are_read_from_his_words(monkeypatch):
    seen = []
    monkeypatch.setattr(phone, "_apple_page", lambda q, kind, artist="", exact=False: seen.append((q, kind, artist, exact)) or None)
    phone.set_prefs("apple")
    box, t = _turn()
    try:
        phone.phone_music({"query": "The Less I Know The Better by Tame Impala"})
        phone.phone_music({"query": "the album Currents by Tame Impala"})
        from calendar_tool import TURN
        TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "play some Tame Impala"})
        phone.phone_music({"query": "Tame Impala"})
        TURN.set({"user": "owner", "conversation": "c", "turn": 2, "user_said": "play Blinding Lights"})
        phone.phone_music({"query": "Blinding Lights"})                 # a bare title: no artist guess
        assert "search?term=Blinding" in box["url"]
    finally:
        phone.ACTION.reset(t)
    assert seen == [("The Less I Know The Better", "song", "Tame Impala", False),
                    ("Currents", "album", "Tame Impala", False), ("Tame Impala", "artist", "", True)]


def test_the_artist_comes_from_his_words_when_the_model_drops_it(monkeypatch):
    from calendar_tool import TURN
    seen = []
    monkeypatch.setattr(phone, "_apple_page", lambda q, kind, artist="", exact=False: seen.append((q, kind, artist)) or None)
    phone.set_prefs("apple")
    TURN.set({"user": "owner", "conversation": "c", "turn": 1, "user_said": "Play The Less I Know the Better by Tame Impala."})
    box, t = _turn()
    try:
        phone.phone_music({"query": "The Less I Know The Better"})
    finally:
        phone.ACTION.reset(t)
    assert seen == [("The Less I Know The Better", "song", "Tame Impala")]
