"""Runs openWakeWord's train.py unchanged, with two macOS compatibility shims:
- torch-audiomentations 0.11 hands torchaudio pathlib paths, and torchaudio 2.2's sox
  backend (the one it picks on macOS) only takes str. Paths become str here.
- Its DataLoader workers get lambdas, which macOS's default "spawn" start method
  can't pickle. Linux forks, so train.py assumes fork; use it here too.

    wake-train/venv/bin/python scripts/wake_train/run_train.py <train.py> [train.py args...]
"""
import multiprocessing
import runpy
import sys

import torchaudio

multiprocessing.set_start_method("fork")


def _str_paths(fn):
    def wrapped(uri, *args, **kwargs):
        return fn(str(uri) if hasattr(uri, "__fspath__") else uri, *args, **kwargs)
    return wrapped


torchaudio.info = _str_paths(torchaudio.info)
torchaudio.load = _str_paths(torchaudio.load)

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
