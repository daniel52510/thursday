"""The runner benchmark (scripts/bench_runner.py): fixed tasks, easiest first, each with a check of
the behavior it asks for. A check runs in thursday-core (which has the dependencies) against the
job's checkout, and must FAIL on the unchanged code, so passing means the job did the work.

Add a task when the runner gets a kind of job it should be able to do; never edit one in place,
or old scores stop being comparable (bump its id instead).
"""

TASKS = [
    {
        "id": "t1-swap",
        "level": "one line, one file",
        "task": "In eridian.py, when the whole sentence is 'cheers mate', THURSDAY should answer 'Gracious.' "
                "the way it does for 'thanks'. Add it to the existing list.",
        "check": """
import eridian
assert eridian.translate("Cheers mate.") == "Gracious.", eridian.translate("Cheers mate.")
assert eridian.translate("Thanks.") == "Gracious."
assert eridian.translate("Awesome!") == "Amaze!"
""",
    },
    {
        "id": "t2-regex",
        "level": "a regex case, one file",
        "task": "In agent_stream.py, 'hit pause' and 'hit pause on the music' should pause the music in code, the same "
                "way 'pause the music' does (_music_intent). Don't change what other phrases do.",
        "check": """
import agent_stream as a
assert a._music_intent("hit pause on the music") == ("music_control", {"action": "pause"}), a._music_intent("hit pause on the music")
assert a._music_intent("hit pause") == ("music_control", {"action": "pause"})
assert a._music_intent("pause the music") == ("music_control", {"action": "pause"})
assert a._music_intent("turn off the lights") is None
assert a._music_intent("play a game with me") is None
""",
    },
    {
        "id": "t3-action",
        "level": "new tool action, one file",
        "task": "In music.py, music_control should accept a 'stop' action that pauses the music (the same app command as "
                "'pause'), and its tool schema should list 'stop' as an allowed action.",
        "check": """
import music
sent = []
music._cmd = lambda command, data=None: sent.append(command)
r = music.music_control({"action": "stop"})
assert r.ok and sent == ["pause"], (r, sent)
enum = next(t for t in music.TOOL_SCHEMAS if t["function"]["name"] == "music_control")["function"]["parameters"]["properties"]["action"]["enum"]
assert "stop" in enum, enum
assert music.music_control({"action": "pause"}).ok and sent[-1] == "pause"
""",
    },
    {
        "id": "t4-two-files",
        "level": "a feature across two files",
        "task": "Add shuffle to the music. In music.py, music_control gets a 'shuffle' action that sends the app's "
                "'shuffle' command (and the schema lists it). In agent_stream.py, 'shuffle the music' and 'shuffle' run "
                "it in code from _music_intent, like skip does.",
        "check": """
import music, agent_stream as a
sent = []
music._cmd = lambda command, data=None: sent.append(command)
assert music.music_control({"action": "shuffle"}).ok and sent == ["shuffle"], sent
assert a._music_intent("shuffle the music") == ("music_control", {"action": "shuffle"}), a._music_intent("shuffle the music")
assert a._music_intent("shuffle") == ("music_control", {"action": "shuffle"})
assert a._music_intent("skip") == ("music_control", {"action": "next"})
""",
    },
    {
        "id": "t5-bug",
        "level": "a bug fix with a reason",
        "task": "Bug in music.py: music_recommend with like='some song' that YouTube Music can't find returns the error "
                "'no_seed: ask what he likes', which makes THURSDAY ask what he likes although he just named a song. When "
                "a named song isn't found, return an error starting with 'not_found' instead. Keep 'no_seed' for when "
                "nothing is playing and no song was named.",
        "check": """
import music
class Empty:
    def search(self, *a, **k): return []
music._ytmusic = lambda: Empty()
r = music.music_recommend({"like": "zzqx unknown song"})
assert not r.ok and r.error.startswith("not_found"), r.error
music.state = lambda fresh=False: {"video": {}}
r = music.music_recommend({})
assert not r.ok and r.error.startswith("no_seed"), r.error
""",
    },
]
