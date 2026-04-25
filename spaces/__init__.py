from .runner import HeadlessRunner, Snapshot
from .scripted_player import scripted_episode
from .random_player import random_episode
from .plots import battery_curve, reputation_curve, promise_keep_curve, ledger_table
from .transcript import render_transcript
from .recorder import save_episode, load_episode, load_prerecorded

def gradio_auto_wrap(fn):
    # Gradio on HF Spaces imports `spaces.gradio_auto_wrap` for GPU queuing.
    # Our package shadows the HF `spaces` module, so we provide a CPU no-op.
    return fn


__all__ = [
    "HeadlessRunner",
    "Snapshot",
    "scripted_episode",
    "random_episode",
    "battery_curve",
    "reputation_curve",
    "promise_keep_curve",
    "ledger_table",
    "render_transcript",
    "save_episode",
    "load_episode",
    "load_prerecorded",
    "gradio_auto_wrap",
]
