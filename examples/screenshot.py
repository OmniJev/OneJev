"""Three questions about one screenshot, answered in one request.

    qev serve --model OmniJev/OneJev/4B
    python examples/screenshot.py screenshot.png
"""
import sys

from qev import Choice, Client, Noul, Score
from qev.media import data_uri

path = sys.argv[1] if len(sys.argv) > 1 else "screenshot.png"
r = Client("http://localhost:8000").system_one(
    state={"task": "Pay the open invoice from ACME", "screen": "<image:1>"},
    media=[{"type": "image", "data": data_uri(path)}],
    questions={
        "done": Noul("The invoice has been paid"),
        "next": Choice("What should the agent do next?", {"click": "click an element", "type": "type text",
                                                           "scroll": "scroll the page", "stop": "stop, the task is finished"}),
        "progress": Score("How far along is the task?", ["not started", "halfway", "almost done", "done"]),
    },
)
print(f"paid       {r.answers['done'].noul:.3f}")
print(f"next       {r.answers['next'].choice}  {r.answers['next'].probabilities}")
print(f"progress   {r.answers['progress'].score:.2f} of 3")
