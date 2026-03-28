"""
src/models.py — ReportAgent
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ReportInput(BaseModel):
    question:      str = Field(..., min_length=3,  max_length=2_000)
    data_analysis: str = Field(..., min_length=10, max_length=20_000)
    patterns:      str = Field(..., min_length=5,  max_length=10_000)
    forecast:      str = Field(..., min_length=5,  max_length=10_000)
    insights:      str = Field(..., min_length=5,  max_length=10_000)

    @field_validator("question")
    @classmethod
    def no_injection(cls, v: str) -> str:
        import re
        m = re.search(
            r"ignore\s+(previous|above|all|prior)"
            r"|forget\s+(your\s+)?instructions"
            r"|you\s+are\s+now|act\s+as|jailbreak|\bDAN\b"
            r"|drop\s+table|delete\s+from|<script|javascript:|eval\s*\(",
            v, re.IGNORECASE,
        )
        if m:
            raise ValueError(f"Question contains a disallowed pattern: '{m.group()}'.")
        return v.strip()

    @field_validator("data_analysis", "patterns", "forecast", "insights")
    @classmethod
    def not_placeholder(cls, v: str) -> str:
        if v.strip().lower() in {"none", "n/a", "null", "empty", "-", "tbd", ""}:
            raise ValueError(f"Input is a placeholder — provide actual agent output.")
        return v

    @model_validator(mode="after")
    def all_inputs_substantial(self) -> "ReportInput":
        sources = [self.data_analysis, self.patterns, self.forecast, self.insights]
        if sum(1 for s in sources if len(s.strip()) >= 20) < 3:
            raise ValueError("At least 3 of the 4 content inputs must have ≥ 20 chars of substance.")
        return self


class ContentPlan(BaseModel):
    """Node 2 output — plan before writing."""
    question_type:       str
    direct_answer_point: str
    top_findings:        list[str] = Field(..., min_length=1, max_length=5)
    top_recommendations: list[str] = Field(..., min_length=1, max_length=3)
    key_number:          Optional[str] = None
    word_limit:          int  = 300
    tone:                str  = "professional"
    include_bullets:     bool = True
    include_forecast:    bool = True
    include_risk:        bool = True


class ReportOutput(BaseModel):
    """Node 3 output — the actual report sections."""
    direct_answer:    str
    key_findings:     list[str] = Field(..., min_length=1, max_length=5)
    recommendations:  list[str] = Field(..., min_length=1, max_length=3)
    summary_sentence: str
    word_count:       int = 0

    @field_validator("direct_answer")
    @classmethod
    def answer_min_length(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError("direct_answer too short — must answer the question (min 20 chars).")
        return v.strip()

    @field_validator("summary_sentence")
    @classmethod
    def summary_min_length(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("summary_sentence too short (min 10 chars).")
        return v.strip()

    @field_validator("key_findings")
    @classmethod
    def findings_nonempty(cls, v: list[str]) -> list[str]:
        cleaned = [f.strip() for f in v if f.strip()]
        if not cleaned:
            raise ValueError("key_findings must contain at least one non-empty item.")
        return cleaned

    @field_validator("recommendations")
    @classmethod
    def recs_actionable(cls, v: list[str]) -> list[str]:
        cleaned = [r.strip() for r in v if r.strip()]
        if not cleaned:
            raise ValueError("recommendations must contain at least one item.")
        for rec in cleaned:
            if len(rec) < 10:
                raise ValueError(f"Recommendation '{rec}' too short (min 10 chars).")
        return cleaned

    @model_validator(mode="after")
    def check_word_count(self) -> "ReportOutput":
        full = " ".join([
            self.direct_answer,
            " ".join(self.key_findings),
            " ".join(self.recommendations),
            self.summary_sentence,
        ])
        self.word_count = len(full.split())
        if self.word_count > 350:
            raise ValueError(
                f"Report is {self.word_count} words — exceeds 300-word limit. Condense."
            )
        return self


class ReportAgentState(BaseModel):
    question:      str = ""
    data_analysis: str = ""
    patterns:      str = ""
    forecast:      str = ""
    insights:      str = ""

    input_valid:  bool      = False
    input_errors: list[str] = Field(default_factory=list)

    plan:            Optional[ContentPlan]  = None
    report:          Optional[ReportOutput] = None
    report_valid:    bool      = False
    report_errors:   list[str] = Field(default_factory=list)
    retry_count:     int       = 0
    max_retries:     int       = 2

    formatted_text:  str = ""

    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class ReportResult(BaseModel):
    success: bool

    # Drop-in for original str return value
    text:             str = ""

    # Structured breakdown
    direct_answer:    str       = ""
    key_findings:     list[str] = Field(default_factory=list)
    recommendations:  list[str] = Field(default_factory=list)
    summary_sentence: str       = ""
    word_count:       int       = 0
    question_type:    str       = ""
    tone:             str       = ""

    model_used:   str        = ""
    duration_ms:  float      = 0.0
    retry_count:  int        = 0
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)
    timestamp:    str        = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )