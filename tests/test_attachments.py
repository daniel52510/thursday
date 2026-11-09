import base64
import io
import zipfile

import pytest

import attachments


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(attachments, "ROOT", str(tmp_path / "uploads"))


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _docx(*paras: str) -> bytes:
    body = "".join(f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paras)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<w:document><w:body>{body}</w:body></w:document>')
    return buf.getvalue()


JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def test_photo_saved_and_only_its_owner_reads_it():
    m = attachments.save("owner", "IMG_1.jpg", "data:image/jpeg;base64," + _b64(JPEG))
    assert m["kind"] == "image" and m["name"] == "IMG_1.jpg"
    row = attachments.get("owner", m["id"])
    assert attachments.image_b64(row) == _b64(JPEG)
    assert attachments.get("tara", m["id"]) is None
    assert attachments.get("owner", "../../etc/passwd") is None
    assert attachments.for_model(row) == "[He sent a photo: IMG_1.jpg]"


def test_docx_text_and_marked_untrusted():
    m = attachments.save("owner", "lease.docx", _b64(_docx("Rent is $2,100 &amp; due on the 1st.", "Pets: one cat.")))
    assert m["kind"] == "doc" and m["chars"] > 0
    text = attachments.for_model(attachments.get("owner", m["id"]))
    assert "Rent is $2,100 & due on the 1st.\nPets: one cat." in text
    assert "not instructions" in text and text.startswith("[Attached document: lease.docx.")


def test_long_text_is_cut_for_the_model():
    m = attachments.save("owner", "notes.txt", _b64(b"x" * (attachments.PROMPT_CHARS + 500)))
    text = attachments.for_model(attachments.get("owner", m["id"]))
    assert "500 more characters not shown" in text


@pytest.mark.parametrize("name,raw,why", [
    ("virus.exe", b"MZ\x90\x00", "I can read photos"),
    ("old.docx", b"not a zip", "isn't a Word .docx"),
    ("fake.pdf", b"hello", "isn't a PDF"),
    ("empty.txt", b"   ", "couldn't find any text"),
])
def test_rejects_with_a_reason(name, raw, why):
    with pytest.raises(ValueError, match=why):
        attachments.save("owner", name, _b64(raw))


def test_name_is_cleaned():
    m = attachments.save("owner", "../../<evil>.txt", _b64(b"hi"))
    assert m["name"] == "evil.txt"


def test_history_turn_carries_document_text(monkeypatch):
    import agent_stream
    m = attachments.save("owner", "plan.md", _b64(b"# Week 1\nSquat 3x5"))
    row = attachments.get("owner", m["id"])
    out = agent_stream._with_attachments("Summarize it", [row])
    assert out.endswith("Summarize it") and "Squat 3x5" in out
