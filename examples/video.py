"""Questions about a video given as frames: every image in a folder, in name order.

    qev serve --model OmniJev/OneJev/4B
    python examples/video.py frames/ --fps 2
"""
import argparse
from pathlib import Path

from qev import Choice, Client, Noul
from qev.media import data_uri

ap = argparse.ArgumentParser()
ap.add_argument("frames", help="folder of frames (jpg or png)")
ap.add_argument("--fps", type=float, default=2.0)
args = ap.parse_args()
frames = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
r = Client("http://localhost:8000").system_one(
    state={"goal": "Make a cup of pour-over coffee", "recording": "<video:1>"},
    media=[{"type": "video", "frames": [{"data": data_uri(p)} for p in frames], "fps": args.fps}],
    questions={
        "finished": Noul("The coffee is ready at the end of the recording"),
        "step": Choice("Which step is shown at the end?", {"grind": "grinding the beans", "heat": "heating water",
                                                           "pour": "pouring water over the grounds", "serve": "serving"}),
    },
)
print(f"{len(frames)} frames")
print(f"finished   {r.answers['finished'].noul:.3f}")
print(f"step       {r.answers['step'].choice}  {r.answers['step'].probabilities}")
