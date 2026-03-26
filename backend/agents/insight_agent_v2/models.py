from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HealthTrend(str, Enum):
    IMPROVING  = "improving"
    STABLE     = "stable"
    DECLINING  = "declining"
    VOLATILE   = "volatile"


class UrgencyLevel(str, Enum):
    IMMEDIATE = "immediate"   # act this quarter
    SHORT_TERM = "short_term" # act within 6 months
    LONG_TERM  = "long_term"  # act within 1–2 years
    MONITOR    = "monitor"    # watch only


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InsightInput(BaseModel):
    """
    Validated inputs for InsightAgent.run().
    Mirrors the original function signature:
        insight_agent(data, patterns, forecast, question)
    """
    data:     str = Field(..., min_length=10, max_length=20_000)
    patterns: str = Field(..., min_length=5,  max_length=10_000)
    forecast: str = Field(..., min_length=5,  max_length=10_000)
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
                "Please ask a genuine financial insight question."
            )
        return v.strip()

    @field_validator("data")
    @classmethod
    def data_has_numbers(cls, v: str) -> str:
        import re
        if len(re.findall(r"\d+(?:[.,]\d+)?", v)) < 3:
            raise ValueError(
                "data must contain at least 3 numeric figures. "
                "Provide actual financial data with numbers."
            )
        return v

    @field_validator("forecast")
    @classmethod
    def forecast_not_placeholder(cls, v: str) -> str:
        placeholders = {"none", "n/a", "no forecast", "null", "empty", "-", "tbd"}
        if v.strip().lower() in placeholders:
            raise ValueError(
                f"forecast is a placeholder ('{v}'). "
                "Provide actual forecast output from the Forecast Agent."
            )
        return v

    @field_validator("patterns")
    @classmethod
    def patterns_not_placeholder(cls, v: str) -> str:
        placeholders = {"none", "n/a", "no patterns", "null", "empty", "-"}
        if v.strip().lower() in placeholders:
            raise ValueError(
                f"patterns is a placeholder ('{v}'). "
                "Provide actual pattern analysis."
            )
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM RESPONSE — NODE 2: SYNTHESIS CONTEXT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SynthesisContext(BaseModel):
    """
    Node 2 — Gemini extracts the key numbers and signals from all
    three input sources before Node 3 synthesizes insights.
    Gives Node 3 a clean, structured view instead of raw text blobs.
    """
    # From data
    latest_revenue:        Optional[float] = None
    latest_net_profit:     Optional[float] = None
    latest_profit_margin:  Optional[float] = None   # decimal e.g. 0.23
    revenue_trend:         HealthTrend     = HealthTrend.STABLE
    data_period:           str             = ""
    currency:              str             = "USD"

    # From patterns
    top_pattern:           str             = ""   # single most important pattern
    pattern_count:         int             = 0
    has_anomaly:           bool            = False
    anomaly_description:   str             = ""

    # From forecast
    forecast_growth_rate:  Optional[float] = None   # decimal
    forecast_confidence:   str             = ""     # High / Medium / Low
    forecast_period:       str             = ""     # "Q2 2025"
    forecast_revenue:      Optional[float] = None

    # Overall signal
    overall_health_signal: HealthTrend     = HealthTrend.STABLE
    synthesis_notes:       str             = ""

    @field_validator("latest_profit_margin")
    @classmethod
    def margin_normalise(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v > 1.5:
            return round(v / 100, 6)
        return v

    @field_validator("forecast_growth_rate")
    @classmethod
    def growth_normalise(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if abs(v) > 2.0:
            return round(v / 100, 6)
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM RESPONSE — NODE 3: INSIGHT OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class StrategicInsight(BaseModel):
    """One of the top strategic insights."""
    title:       str
    explanation: str
    evidence:    str             # specific data point that supports this
    urgency:     UrgencyLevel   = UrgencyLevel.MONITOR


class RecommendedAction(BaseModel):
    """One concrete recommended action."""
    action:          str
    rationale:       str
    expected_impact: str
    urgency:         UrgencyLevel = UrgencyLevel.SHORT_TERM


class InsightOutput(BaseModel):
    """
    Full structured output from Node 3 (generate_insights).
    Validated by Node 4.
    """
    direct_answer:      str                          # answer to the user's exact question
    insights:           list[StrategicInsight]       = Field(..., min_length=1, max_length=5)
    actions:            list[RecommendedAction]      = Field(..., min_length=1, max_length=3)
    key_risk:           str                          # one key risk to watch
    health_score:       int                          # 1–10
    health_explanation: str
    health_trend:       HealthTrend                  = HealthTrend.STABLE
    executive_summary:  str                          # 2-3 sentence CFO-level summary

    @field_validator("health_score")
    @classmethod
    def score_in_range(cls, v: int) -> int:
        if not (1 <= v <= 10):
            raise ValueError(
                f"health_score={v} is outside the 1–10 range."
            )
        return v

    @field_validator("key_risk")
    @classmethod
    def risk_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("key_risk cannot be empty.")
        return v.strip()

    @field_validator("direct_answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError(
                "direct_answer is too short — must be a substantive response."
            )
        return v.strip()

    @model_validator(mode="after")
    def insights_have_evidence(self) -> "InsightOutput":
        for i, insight in enumerate(self.insights):
            if not insight.evidence.strip():
                raise ValueError(
                    f"Insight {i+1} ('{insight.title}') has no evidence. "
                    "Every insight must cite a specific data point."
                )
        return self


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InsightAgentState(BaseModel):
    """Mutable state that flows through every graph node."""

    # ── Inputs ──────────────────────────────────────────
    data:     str = ""
    patterns: str = ""
    forecast: str = ""
    question: str = ""

    # ── Node 1 ──────────────────────────────────────────
    input_valid:  bool      = False
    input_errors: list[str] = Field(default_factory=list)

    # ── Node 2 ──────────────────────────────────────────
    context:  Optional[SynthesisContext] = None

    # ── Node 3 / 4 ──────────────────────────────────────
    insights:          Optional[InsightOutput] = None
    insights_valid:    bool      = False
    insights_errors:   list[str] = Field(default_factory=list)
    retry_count:       int       = 0
    max_retries:       int       = 2

    # ── Node 5 ──────────────────────────────────────────
    narrative: str = ""   # CFO-ready paragraph

    # ── Graph control ────────────────────────────────────
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class InsightResult(BaseModel):
    """
    What InsightAgent.run() returns to the caller.
    Structured replacement for the original str return value.
    """
    success: bool

    # ── Direct answer ────────────────────────────────────
    direct_answer:      str = ""

    # ── Insights (list of dicts for easy iteration) ──────
    insights: list[dict] = Field(default_factory=list)
    # Each dict: {title, explanation, evidence, urgency}

    # ── Actions ─────────────────────────────────────────
    actions: list[dict] = Field(default_factory=list)
    # Each dict: {action, rationale, expected_impact, urgency}

    # ── Risk & Health ────────────────────────────────────
    key_risk:           str           = ""
    health_score:       Optional[int] = None    # 1–10
    health_explanation: str           = ""
    health_trend:       Optional[str] = None

    # ── Summary ─────────────────────────────────────────
    executive_summary:  str = ""
    narrative:          str = ""   # full CFO-ready paragraph

    # ── Context (from parsed inputs) ─────────────────────
    latest_revenue:       Optional[float] = None
    forecast_growth_rate: Optional[float] = None
    forecast_growth_pct:  Optional[str]   = None
    data_period:          str             = ""

    # ── Audit ────────────────────────────────────────────
    model_used:   str        = ""
    duration_ms:  float      = 0.0
    retry_count:  int        = 0
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)
    timestamp:    str        = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
