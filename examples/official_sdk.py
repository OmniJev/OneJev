"""The official TypeSafe SDK, unchanged, against a local OneJev server (text requests).

    pip install typesafe-sdk
    qev serve --model OmniJev/OneJev/4B
    TYPESAFE_API_KEY=local python examples/official_sdk.py
"""
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient(base_url="http://localhost:8000")
r = client.system_one(
    state={"ticket": "I was charged twice for my subscription this month. Please refund one of them."},
    questions={
        "billing": Noul(instructions="Is this about billing?"),
        "team": Choice(instructions="Which team should handle it?",
                       criteria={"billing": "payments and refunds", "tech": "technical problems", "sales": "new purchases"}),
        "urgency": Score(instructions="How urgent is it?", criteria=["can wait", "soon", "today", "right now"]),
    },
)
print(f"billing    {r.answers['billing'].noul:.3f}")
print(f"team       {r.answers['team'].choice}  {r.answers['team'].probabilities}")
print(f"urgency    {r.answers['urgency'].score:.2f} of 3")
