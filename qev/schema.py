"""Request and response types for the TypeSafe-compatible System One API.

Shapes follow TypeSafe's public OpenAPI description (notes/typesafe_openapi.json) and the
behaviour of the official Python SDK (msgspec structs). Extra keys are tolerated everywhere;
Score legend and probability keys are integer-parseable strings.
"""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

Entry = Union[str, dict[str, Any], list[Any], None]

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class NoulCriteria(_Lenient):
    true: Entry = None
    false: Entry = None


class NoulQuestion(_Lenient):
    type: Literal["noul"]
    instructions: Entry = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_Lenient):
    type: Literal["choice"]
    instructions: Entry = None
    criteria: dict[str, Entry]

    @model_validator(mode="after")
    def _check(self) -> "ChoiceQuestion":
        if not self.criteria:
            raise ValueError("choice question needs at least one option in criteria")
        if len(self.criteria) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice question has {len(self.criteria)} options, maximum is {MAX_CHOICE_OPTIONS}")
        return self


class ScoreQuestion(_Lenient):
    type: Literal["score"]
    instructions: Entry = None
    criteria: list[Entry]

    @model_validator(mode="after")
    def _check(self) -> "ScoreQuestion":
        if not self.criteria:
            raise ValueError("score question needs at least one level in criteria")
        if len(self.criteria) > MAX_SCORE_LEVELS:
            raise ValueError(f"score question has {len(self.criteria)} levels, maximum is {MAX_SCORE_LEVELS}")
        return self


Question = Union[NoulQuestion, ChoiceQuestion, ScoreQuestion]


class SystemOneRequest(_Lenient):
    state: Any = None
    model: str = "jev-latest"
    questions: dict[str, Question] = Field(discriminator=None)

    @model_validator(mode="after")
    def _check(self) -> "SystemOneRequest":
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, Entry]
    probabilities: dict[str, float]
    confidence: float


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelMetadata(BaseModel):
    name: str
    description: str
    release_date: str


class ModelMetadataList(BaseModel):
    models: list[ModelMetadata]
