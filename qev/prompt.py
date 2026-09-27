"""Rendering: request -> shared prefix text + one suffix per question + answer slots.

Everything the model sees is produced here, so this file is the single place to audit for
leakage of anything the model should not see (question ids, gold labels, internal bookkeeping).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion

PROMPT_VERSION = "qev-labels-v2"

SYSTEM_PROMPT = (
    "Apply the question to the state. Choose exactly one of the listed options. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NOT_ONE_TOKEN = {"BQ", "BZ", "CJ", "CQ", "CZ", "DQ", "DZ", "EJ", "EY", "FJ", "FQ", "FV", "FZ", "GJ", "GK", "GQ", "GZ",
                  "HJ", "IY", "JF", "JG", "JH", "JL", "JN", "JQ"}
SLOT_LABELS: tuple[str, ...] = tuple(
    list(LETTERS) + [a + b for a in LETTERS for b in LETTERS if a + b not in _NOT_ONE_TOKEN][: 256 - len(LETTERS)])
MAX_SLOTS = len(SLOT_LABELS)


@dataclass(frozen=True)
class PromptStyle:
    """Wording and layout knobs. The default instance is the frozen production template; other
    instances exist only for ablations."""

    name: str = PROMPT_VERSION
    system: str = SYSTEM_PROMPT
    layout: str = "tagged"
    show_names: bool = True
    question_header: str = "Question:"
    score_header: str = "Rate the state:"
    answer_line: str = "Answer with one letter: {letters}."
    answer_line_many: str = "Answer with one label: {letters}."


DEFAULT_STYLE = PromptStyle()

NOUL_DEFAULT_INSTRUCTIONS = "Is the statement true, or is the answer to the question yes?"
NOUL_DEFAULT_TRUE = "the statement is true / the answer is yes"
NOUL_DEFAULT_FALSE = "the statement is false / the answer is no"


def render_entry(entry: Any) -> str:
    """Render an instructions / criteria entry (string, object, array or null) as text."""
    if entry is None:
        return ""
    if isinstance(entry, str):
        return entry.strip()
    return json.dumps(entry, ensure_ascii=False, indent=2)


def render_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


@dataclass
class RenderedQuestion:
    """One branch: the suffix text and the labels its slots map to."""

    qid: str
    kind: str
    labels: list[str]
    descriptions: list[Any]
    suffix: str
    n_slots: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_slots = len(self.labels)


def _options_block(names: Sequence[str], descriptions: Sequence[Any], show_names: bool = True) -> str:
    lines = []
    for letter, name, desc in zip(SLOT_LABELS, names, descriptions):
        text = render_entry(desc)
        if name is None or (not show_names and text):
            lines.append(f"{letter}. {text}")
        elif text:
            lines.append(f"{letter}. {name}: {text}")
        else:
            lines.append(f"{letter}. {name}")
    return "\n".join(lines)


def render_question(qid: str, q: Question, order: Sequence[int] | None = None, style: PromptStyle = DEFAULT_STYLE) -> RenderedQuestion:
    """Render one question into a suffix. `order` optionally permutes the slot order (used for
    permutation debiasing and training augmentation); labels/descriptions follow the permutation."""
    if isinstance(q, NoulQuestion):
        kind = "noul"
        instructions = render_entry(q.instructions) or NOUL_DEFAULT_INSTRUCTIONS
        crit = q.criteria
        t = render_entry(crit.true) if crit and crit.true is not None else NOUL_DEFAULT_TRUE
        f = render_entry(crit.false) if crit and crit.false is not None else NOUL_DEFAULT_FALSE
        labels, names, descs = ["yes", "no"], ["yes", "no"], [t, f]
    elif isinstance(q, ChoiceQuestion):
        kind = "choice"
        instructions = render_entry(q.instructions) or "Which option applies to the state?"
        labels = list(q.criteria.keys())
        names = list(labels)
        descs = [q.criteria[k] for k in labels]
    elif isinstance(q, ScoreQuestion):
        kind = "score"
        instructions = render_entry(q.instructions) or "Which level describes the state?"
        labels = [str(i) for i in range(len(q.criteria))]
        names = [f"level {i}" for i in range(len(q.criteria))]
        descs = list(q.criteria)
    else:
        raise TypeError(f"unknown question type {type(q)}")

    if len(labels) > MAX_SLOTS:
        raise ValueError(f"{len(labels)} options exceed the {MAX_SLOTS} slots of prompt {PROMPT_VERSION}")

    if order is not None:
        labels = [labels[i] for i in order]
        names = [names[i] for i in order]
        descs = [descs[i] for i in order]

    letters = list(SLOT_LABELS[: len(labels)])
    if style.layout == "json":
        payload = {"criterion": instructions,
                   "options": [{"letter": l, "description": (f"{n}: {render_entry(d)}" if (style.show_names and n is not None and render_entry(d)) else (render_entry(d) or n))}
                               for l, n, d in zip(letters, names, descs)]}
        suffix = json.dumps(payload, ensure_ascii=False)[1:]
    else:
        header = style.question_header if kind != "score" else style.score_header
        suffix = (
            f"{header} {instructions}\n\n"
            f"Options:\n{_options_block(names, descs, style.show_names)}\n\n"
            + (style.answer_line if len(letters) <= len(LETTERS) else style.answer_line_many).format(letters=", ".join(letters))
        )
    return RenderedQuestion(qid=qid, kind=kind, labels=labels, descriptions=descs, suffix=suffix)


def state_block(state: Any, style: PromptStyle = DEFAULT_STYLE) -> str:
    if style.layout == "json":
        return "{" + json.dumps({"evidence": state}, ensure_ascii=False)[1:-1] + ", "
    return f"<state>\n{render_state(state)}\n</state>\n\n"


def build_messages(state: Any, suffix: str, style: PromptStyle = DEFAULT_STYLE) -> list[dict[str, str]]:
    """Chat messages for one branch. The user turn is `state_block + suffix`, so every branch of a
    request shares the same prefix up to the end of the state block."""
    return [
        {"role": "system", "content": style.system},
        {"role": "user", "content": state_block(state, style) + suffix},
    ]
