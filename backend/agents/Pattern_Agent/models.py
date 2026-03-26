"""
src/models.py
━━━━━━━━━━━━━
All Pydantic models for PatternAgent.

PatternInput        → validates data + question
ParsedFinancials    → Node 2: LLM extracts rows of numbers from raw text
PatternOutput       → Node 3: LLM detects patterns, anomalies, trends
PatternAgentState   → flows through all graph nodes
PatternResult       → final typed output returned to caller
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TrendDirection(str, Enum):
    GROWING   = "growing"
    DECLINING = "declining"
    STABLE    = "stable"
    VOLATILE  = "volatile"
    MIXED     = "mixed"


class Severity(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatternInput(BaseModel):
    """
    Validated inputs for PatternAgent.run().
    Mirrors the original function signature:
        pattern_agent(data, question)
    """
    data:     str = Field(..., min_length=10, max_length=20_000)
    question: str = Field(..., min_length=3,  max_length=2_000)

    @field_validator("question")
    @classmethod
    def no_injection(cls, v: str) -> str:
        import re
        banned = re.compile(
            r"ignore\s+(previous|above|all|prior)"
            r"|forget\s+(your\s+)?instructions"
            r"|you\s+are\s+now|act\s+as|jailbreak|\bDAN\b"
            r"|drop\s+table|delete\s+from|union\s+select"
            r"|<script|javascript:|eval\s*\(",
            re.IGNORECASE,
        )
        m = banned.search(v)
        if m:
            raise ValueError(
                f"Question contains a disallowed pattern: '{m.group()}'. "
                "Please ask a genuine financial pattern question."
            )
        return v.strip()

    @field_validator("data")
    @classmethod
    def data_has_numbers(cls, v: str) -> str:
        import re
        if len(re.findall(r"\d+(?:[.,]\d+)?", v)) < 3:
            raise ValueError(
                "data must contain at least 3 numeric figures. "
                "Provide real financial data with actual numbers."
            )
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM RESPONSE — NODE 2: PARSED FINANCIALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FinancialPeriod(BaseModel):
    """One time-period row extracted from raw data."""
    period:        str            # e.g. "Q1 2023"
    revenue:       Optional[float] = None
    expenses:      Optional[float] = None
    net_profit:    Optional[float] = None
    profit_margin: Optional[float] = None   # decimal e.g. 0.20
    growth_rate:   Optional[float] = None   # QoQ or YoY, decimal


class ParsedFinancials(BaseModel):
    """
    Node 2 — structured extraction of time-series rows
    so Node 3 gets clean numbers, not a raw text blob.
    """
    periods:          list[FinancialPeriod] = Field(..., min_length=1)
    currency:         str                   = "USD"
    data_period:      str                   = ""    # "Q1 2023 to Q1 2024"
    num_periods:      int                   = 0
    has_quarterly:    bool                  = True
    has_annual:       bool                  = False
    parsing_notes:    str                   = ""

    @field_validator("periods")
    @classmethod
    def periods_have_at_least_revenue_or_expenses(
        cls, v: list[FinancialPeriod]
    ) -> list[FinancialPeriod]:
        for p in v:
            if p.revenue is None and p.expenses is None and p.net_profit is None:
                raise ValueError(
                    f"Period '{p.period}' has no numeric values — "
                    "every period must have at least one of: revenue, expenses, net_profit."
                )
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM RESPONSE — NODE 3: PATTERN OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DetectedPattern(BaseModel):
    """One detected pattern."""
    name:        str             # short label e.g. "Accelerating Revenue Growth"
    description: str             # 1-2 sentences explaining the pattern
    evidence:    str             # specific numbers that prove it
    severity:    Severity        = Severity.LOW
    periods_affected: list[str]  = Field(default_factory=list)


class Anomaly(BaseModel):
    """One detected anomaly."""
    description:  str
    period:       str            # which period(s)
    magnitude:    str            # e.g. "+32% vs prior quarter average"
    severity:     Severity       = Severity.MEDIUM
    possible_cause: str          = ""


class YoYComparison(BaseModel):
    """Year-over-year comparison for one metric."""
    metric:       str            # "revenue", "expenses", "net_profit"
    prior_value:  Optional[float] = None
    current_value:Optional[float] = None
    change_pct:   Optional[float] = None   # decimal e.g. 0.25 = 25%
    direction:    TrendDirection  = TrendDirection.STABLE


class PatternOutput(BaseModel):
    """
    Full structured output from Node 3 (detect_patterns).
    Validated by Node 4.
    """
    patterns:          list[DetectedPattern]  = Field(..., min_length=1)
    anomalies:         list[Anomaly]          = Field(default_factory=list)
    trend_direction:   TrendDirection
    trend_explanation: str
    yoy_comparisons:   list[YoYComparison]    = Field(default_factory=list)
    key_finding:       str                    # most important finding in one sentence
    revenue_growth_rate:  Optional[float] = None   # decimal, latest period
    expense_growth_rate:  Optional[float] = None
    margin_trend:         str             = ""     # "expanding", "contracting", "stable"
    seasonal_patterns:    list[str]       = Field(default_factory=list)

    @field_validator("key_finding")
    @classmethod
    def finding_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError(
                "key_finding must be at least 10 characters — "
                "provide a substantive one-sentence summary."
            )
        return v.strip()

    @field_validator("revenue_growth_rate", "expense_growth_rate")
    @classmethod
    def growth_normalise(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if abs(v) > 2.0:     # looks like a percentage, normalise to ratio
            return round(v / 100, 6)
        return v

    @model_validator(mode="after")
    def patterns_have_evidence(self) -> "PatternOutput":
        for p in self.patterns:
            if not p.evidence.strip():
                raise ValueError(
                    f"Pattern '{p.name}' has no evidence. "
                    "Every pattern must cite specific numbers."
                )
        return self


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatternAgentState(BaseModel):
    """Mutable state that flows through every graph node."""

    # ── Inputs ──────────────────────────────────────────
    data:     str = ""
    question: str = ""

    # ── Node 1 ──────────────────────────────────────────
    input_valid:  bool      = False
    input_errors: list[str] = Field(default_factory=list)

    # ── Node 2 ──────────────────────────────────────────
    parsed:   Optional[ParsedFinancials] = None

    # ── Node 3 / 4 ──────────────────────────────────────
    patterns:         Optional[PatternOutput] = None
    patterns_valid:   bool      = False
    patterns_errors:  list[str] = Field(default_factory=list)
    retry_count:      int       = 0
    max_retries:      int       = 2

    # ── Node 5 ──────────────────────────────────────────
    narrative:  str = ""

    # ── Graph control ────────────────────────────────────
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PatternResult(BaseModel):
    """
    What PatternAgent.run() returns to the caller.
    Structured replacement for the original str return value.
    """
    success: bool

    # ── Core findings ────────────────────────────────────
    key_finding:       str           = ""
    trend_direction:   Optional[str] = None
    trend_explanation: str           = ""

    # ── Patterns ─────────────────────────────────────────
    patterns: list[dict] = Field(default_factory=list)
    # Each: {name, description, evidence, severity, periods_affected}

    # ── Anomalies ────────────────────────────────────────
    anomalies: list[dict] = Field(default_factory=list)
    # Each: {description, period, magnitude, severity, possible_cause}
    has_anomalies: bool = False

    # ── Growth rates ─────────────────────────────────────
    revenue_growth_rate:    Optional[float] = None
    revenue_growth_pct:     Optional[str]   = None
    expense_growth_rate:    Optional[float] = None
    margin_trend:           str             = ""
    seasonal_patterns:      list[str]       = Field(default_factory=list)

    # ── YoY comparisons ──────────────────────────────────
    yoy_comparisons: list[dict] = Field(default_factory=list)

    # ── Data context ─────────────────────────────────────
    data_period:   str = ""
    num_periods:   int = 0

    # ── Narrative ────────────────────────────────────────
    narrative:     str = ""

    # ── Audit ────────────────────────────────────────────
    model_used:   str        = ""
    duration_ms:  float      = 0.0
    retry_count:  int        = 0
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)
    timestamp:    str        = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )