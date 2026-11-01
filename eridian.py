"""Rocky's grammar, enforced in code on everything THURSDAY says or writes (agent_stream.stream_prompt,
spoken mission reports). The prompt asks the model to talk like Rocky; on its own the model still
writes English and tacks "question" on. This rewrites each sentence the way Rocky builds one:

  questions become statements:  "Do you want a jacket?"     -> "You want jacket, question?"
                                "Are you okay?"             -> "You are okay, question?"
                                "Want me to add it?"        -> "I add it, question?"
                                "It's cold, isn't it?"      -> "It is cold, question?"
  present tense:                "I'll check" / "I'm going to check" -> "I check"; "Let's go" -> "We go"
  no human openers:             "Sure, ..." / "Okay, so ..." / "Well, ..." dropped
  his words:                    "really good" -> "very good", "a lot of" -> "many" / "much",
                                "Thanks." -> "Gracious.", "Awesome!" -> "Amaze!", "Sure." -> "Yes."
  no contractions, articles or filler ("I'm" -> "I am", "the forecast" -> "forecast")
  every question ends ", question?", every exclamation ", exclamation!" (not "Amaze!" or
  "Good, good, good!"); a "question" tag on a plain statement is dropped

Rules work on whole sentences, so the stream (Stream) releases text a sentence at a time: no
cost for speech, which already waits for its first sentence. Names keep their articles ("The
Verge", "Plan A"), and code, URLs and emails are never touched.
"""
from __future__ import annotations

import re

_CONTRACTIONS = {
    "i'm": "I am", "you're": "you are", "we're": "we are", "they're": "they are", "it's": "it is",
    "that's": "that is", "there's": "there is", "here's": "here is", "what's": "what is", "who's": "who is",
    "he's": "he is", "she's": "she is", "let's": "let us", "i'll": "I will", "you'll": "you will",
    "we'll": "we will", "they'll": "they will", "it'll": "it will", "i've": "I have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'd": "I would", "you'd": "you would", "we'd": "we would",
    "they'd": "they would", "can't": "cannot", "won't": "will not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "couldn't": "could not",
    "shouldn't": "should not", "wouldn't": "would not", "mustn't": "must not", "needn't": "need not",
}
_CONTRACTION = re.compile(r"\b(" + "|".join(re.escape(k).replace("'", "['’]") for k in _CONTRACTIONS) + r")\b", re.I)
_SUBJ = r"(you|we|they|I|he|she|it|that|this|there)"

# Whole sentences he'd say his own way (matched without the end punctuation, any case).
_SWAPS = {
    "Gracious.": ("thanks", "thank you", "thanks a lot", "thank you so much", "thanks so much", "cheers"),
    "Amaze!": ("awesome", "amazing", "fantastic", "wonderful", "incredible", "great", "excellent"),
    "Yes.": ("sure", "absolutely", "of course", "definitely", "certainly", "yep", "yeah", "you bet"),
}
_SWAP = {phrase: out for out, phrases in _SWAPS.items() for phrase in phrases}

_OPENER = re.compile(r"^(?:(?:sure|okay|ok|alright|all right|well|so|got it|of course|absolutely|great|awesome|"
                     r"perfect|right|now)\s*,\s*|(?:oh|um|uh|hmm|hey)[,!]?\s+|so\s+(?=\w))+", re.I)
# Questions, after contractions are expanded.
_TAG = re.compile(r",\s*(?:(?:is|are|do|does|did|was|were|will|would|can|could|has|have) not|cannot) "
                  r"(?:it|you|they|we|he|she|that|there)\s*$|,\s*right\s*$", re.I)
_NEG_ASK = re.compile(rf"^(is|are|was|were|will|would|can|could|should|has|have|do|does|did) not {_SUBJ}\b", re.I)
_CANNOT_ASK = re.compile(rf"^cannot {_SUBJ}\b", re.I)
_ASK_AUX = re.compile(rf"^(do|does|did|is|are|am|was|were|can|could|will|would|should|shall|have|has|may|might) {_SUBJ}\b", re.I)
_WH_DO = re.compile(rf"^(what|where|when|why|how|which|who)\s+(?:do|does)\s+{_SUBJ}\b", re.I)
_OFFER = re.compile(r"^(?:you want me to|you would like me to|want me to|shall I|should I) ", re.I)
# Statements.
_FUTURE = re.compile(r"\b(I|we|you|they) (?:will|am going to|are going to) (?!be\b|not\b|have\b)(?=\w)", re.I)
_LET_US = re.compile(r"\blet us\b", re.I)
_VERY = re.compile(r"\b(?:really|super|extremely|incredibly|truly) (?=\w)", re.I)
_MANY = re.compile(r"\b(?:a lot of|lots of) (\w+)", re.I)   # "many people", "much rain"
_FILLER = re.compile(r"\b(?:actually|basically|literally|just|totally|simply),? ", re.I)
_ARTICLE = re.compile(r"\b(?:the|a|an) (?=[\w'\"])")
_LEAD_ARTICLE = re.compile(r"^(?:The|A|An) (?=[a-z])")
# Code, URLs and emails: swapped out before the rules, back after.
_PROTECT = re.compile(r"`[^`\n]*`|\S+://\S+|www\.\S+|\S+@\S+\.\w+")
_PREFIX = re.compile(r"^[\s#>*\-|]*(?:\d+[.)]\s+)?")   # list markers, headings, table pipes
# A sentence ends at . ! ? (and any closing quote) before the next word, or at a line break.
_BOUNDARY = re.compile(r"(?<=[.!?…])[\"')\]]*[ \t]+(?=\S)|\n")

# A "?" that ends a word and a sentence; a URL's "?" ("/a?b=1") doesn't match.
_ASK = re.compile(r"(?<=[A-Za-z0-9'\")\]])\?(?=[\s\"')\]*_]|$)")
_ASKED = re.compile(r"\bquestion\W*$", re.I)
# "question" (or a stray "statement") on a sentence that ends in "." or "!" is no question: tag dropped.
_MISTAG = re.compile(r",\s*(?:question|statement)\s*(?=[.!](?:[\s\"')\]*_]|$))", re.I)
# Same shape as _ASK, for "!"; an exclamation ends ", exclamation!" (bursts and triples aside).
_BANG = re.compile(r"(?<=[A-Za-z0-9'\")\]])!(?=[\s\"')\]*_]|$)")
_EXCLAIMED = re.compile(r"\bexclamation\W*$", re.I)
_SENTENCE_START = re.compile(r"(?:^|[.!?…\n])[^.!?…\n]*$")


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _subject(word: str) -> str:
    return "I" if word.lower() == "i" else _cap(word.lower())


_IRREGULAR_PLURALS = {"people", "children", "men", "women", "feet", "teeth", "mice", "geese", "folks"}


def _plural(word: str) -> bool:
    w = word.lower()
    return w in _IRREGULAR_PLURALS or (w.endswith("s") and not w.endswith("ss"))


def _expand(m: re.Match) -> str:
    word = m.group(0)
    out = _CONTRACTIONS[word.lower().replace("’", "'")]
    return _cap(out) if word[0].isupper() else out


def _question(body: str) -> str:
    """An English question (no "?" yet) rebuilt the way Rocky asks: statement order."""
    body = _TAG.sub("", body)
    if m := _NEG_ASK.match(body):      # "Is not it great" (rhetorical): "It is great"
        aux, subj, rest = m.group(1).lower(), m.group(2), body[m.end():]
        return _subject(subj) + ("" if aux in ("do", "does", "did") else f" {aux}") + rest
    if m := _CANNOT_ASK.match(body):
        return f"{_subject(m.group(1))} can{body[m.end():]}"
    if m := _ASK_AUX.match(body):      # "Do you want X" -> "You want X"; "Are you okay" -> "You are okay"
        aux, subj, rest = m.group(1).lower(), m.group(2), body[m.end():]
        if aux in ("do", "does", "shall"):
            body = _subject(subj) + rest
        elif aux == "did":
            body = f"{_subject(subj)} did{rest}"
        elif aux == "should" and subj == "I":   # "Should I book it" -> "I book it"
            body = "I" + rest
        elif subj.lower() == "there":
            body = f"There {aux}{rest}"
        else:
            body = f"{_subject(subj)} {aux}{rest}"
    elif m := _WH_DO.match(body):      # "What do you want" -> "What you want"
        subj = m.group(2) if m.group(2) == "I" else m.group(2).lower()
        body = f"{_cap(m.group(1).lower())} {subj}{body[m.end():]}"
    if m := _OFFER.match(body):        # "You want me to add it" -> "I add it"
        body = "I " + body[m.end():]
    return body


def _sentence(s: str) -> str:
    """One sentence in Rocky's grammar (the "question" suffix aside: questions() does that)."""
    protected: list[str] = []
    s = _PROTECT.sub(lambda m: protected.append(m.group(0)) or f"\x00{len(protected) - 1}\x00", s)
    prefix = _PREFIX.match(s).group(0)
    body, tail = s[len(prefix):], ""
    if m := re.search(r"[.!?…\"')\]\s]+$", body):
        body, tail = body[:m.start()], body[m.start():]
    if body:
        body = _CONTRACTION.sub(_expand, body)
        if (out := _SWAP.get(body.lower().rstrip(" ,"))) is not None:
            body, tail = out[:-1], out[-1] + tail.lstrip(".!?")
        else:
            body = _OPENER.sub("", body) or body
            if tail.lstrip("\"')] \t").startswith("?"):
                body = _question(body)
            body = _FUTURE.sub(lambda m: m.group(1) + " ", body)
            body = _LET_US.sub(lambda m: "We" if m.group(0)[0].isupper() else "we", body)
            body = _VERY.sub("very ", body)
            body = _MANY.sub(lambda m: ("many " if _plural(m.group(1)) else "much ") + m.group(1), body)
            body = _FILLER.sub("", body)
            body = _ARTICLE.sub("", body)
            body = _LEAD_ARTICLE.sub("", body)
            body = _cap(body)
    s = prefix + body + tail
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], s)


def _prose(text: str) -> str:
    """Complete sentences (and the line breaks between them) in Rocky's grammar."""
    out, last = [], 0
    for m in _BOUNDARY.finditer(text):
        out.append(_sentence(text[last:m.start()]))
        out.append(m.group(0))
        last = m.end()
    out.append(_sentence(text[last:]))
    return "".join(out)


_TAGGED_ASK = re.compile(r",\s*question\s*\?", re.I)


def questions(text: str, before: str = "", statements: bool = False) -> str:
    """Each plain question mark becomes ", question?", each exclamation ", exclamation!"; a "question"
    tag on anything else is dropped. statements: this reply only confirms (a mission just started),
    so its questions end "." instead: an "I find best collar, question?" there reads as asking again."""
    text = _MISTAG.sub("", text)
    if statements:
        text = _ASK.sub(".", _TAGGED_ASK.sub(".", text))
    ctx = before[-24:]
    full, out, last = ctx + text, [], len(ctx)
    marks = sorted([*_ASK.finditer(full, len(ctx)), *_BANG.finditer(full, len(ctx))], key=lambda m: m.start())
    for m in marks:
        out.append(full[last:m.start()])
        if m.group(0) == "?":
            out.append("?" if _ASKED.search(full[:m.start()]) else ", question?")
        else:
            out.append("!" if _burst(full[:m.start()]) else ", exclamation!")
        last = m.end()
    out.append(full[last:])
    return "".join(out)


def _burst(head: str) -> bool:
    """The sentence before a "!" stays bare: already tagged, one word ("Amaze"), or one word
    repeated ("Good, good, good")."""
    if _EXCLAIMED.search(head):
        return True
    words = re.findall(r"[A-Za-z0-9']+", _SENTENCE_START.search(head).group(0))
    return len(words) <= 1 or len({w.lower() for w in words}) == 1


class Stream:
    """Rocky's grammar over a reply arriving in pieces (Ollama's tokens). Prose goes out a whole
    sentence at a time; ``` code blocks go through as they come, untouched. A trailing "``" waits:
    it may be half a fence. Same result as translate() on the whole reply."""

    def __init__(self) -> None:
        self.buf, self.said, self.fence = "", "", False
        self.statements = False  # set once a mission starts this turn: see questions()

    def feed(self, text: str) -> str:
        self.buf += text
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        out = []
        while self.buf:
            j = self.buf.find("```")
            if self.fence:
                if j < 0:
                    keep = 0 if final else len(self.buf) - len(self.buf.rstrip("`"))
                    out.append(self.buf[:len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                out.append(self.buf[:j + 3])
                self.buf, self.fence = self.buf[j + 3:], False
                continue
            if j >= 0:  # prose up to a code block: all of it is complete
                out.append(self._done(self.buf[:j], out) + "```")
                self.buf, self.fence = self.buf[j + 3:], True
                continue
            if final:
                out.append(self._done(self.buf, out))
                self.buf = ""
                break
            ends = [m.end() for m in _BOUNDARY.finditer(self.buf)]
            if ends:
                out.append(self._done(self.buf[:ends[-1]], out))
                self.buf = self.buf[ends[-1]:]
            break
        result = "".join(out)
        self.said += result
        return result

    def _done(self, text: str, out: list[str]) -> str:
        return questions(_prose(text), self.said + "".join(out), self.statements) if text else ""


def translate(text: str, statements: bool = False) -> str:
    """A whole reply in Rocky's grammar: the same code path the stream takes."""
    stream = Stream()
    stream.statements = statements
    return stream.feed(text) + stream.flush()
