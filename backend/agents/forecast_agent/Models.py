from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ConfidenceLevel(str, Enum):
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


class TrendDirection(str, Enum):
    GROWING   = "growing"
    DECLINING = "declining"
    STABLE    = "stable"
    VOLATILE  = "volatile"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastInput(BaseModel):
    """
    Validated inputs for ForecastAgent.run().
    Mirrors the original function signature:
        forecast_agent(data, patterns, question)
    """
    data:     str = Field(..., min_length=10, max_length=20_000,
                          description="Historical financial data string")
    patterns: str = Field(..., min_length=5,  max_length=10_000,
                          description="Patterns detected by Pattern Agent")
    question: str = Field(..., min_length=3,  max_length=2_000,
                          description="User question / forecast context")

    @field_validator("question")
    @classmethod
    def no_injection(cls, v: str) -> str:
        import re
        banned = re.compile(
            r"ignore\s+(previous|above|all|prior)"
            r"|forget\s+(your\s+)?instructions"
            r"|you\s+are\s+now|act\s+as|jailbreak|\bDAN\b"
            r"|drop\s+table|delete\s+from|union\s+select|--\s"
            r"|<script|javascript:|eval\s*\(",
            re.IGNORECASE,
        )
        m = banned.search(v)
        if m:
            raise ValueError(
                f"Question contains a disallowed pattern: '{m.group()}'. "
                "Please ask a genuine forecasting question."
            )
        return v.strip()

    @field_validator("data")
    @classmethod
    def data_has_numbers(cls, v: str) -> str:
        """Financial data must contain at least some numeric content."""
        import re
        numbers_found = re.findall(r"\d+(?:[.,]\d+)?", v)
        if len(numbers_found) < 3:
            raise ValueError(
                "data appears to contain no numeric financial figures. "
                "Please provide historical data with actual numbers."
            )
        return v

    @field_validator("patterns")
    @classmethod
    def patterns_not_placeholder(cls, v: str) -> str:
        lowered = v.lower().strip()
        placeholders = {"none", "n/a", "no patterns", "null", "empty", "-", ""}
        if lowered in placeholders:
            raise ValueError(
                "patterns cannot be a placeholder value. "
                "Provide actual pattern analysis from the Pattern Agent."
            )
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM RESPONSE — NODE 2: PARSED FINANCIALS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FinancialSnapshot(BaseModel):
    """
    Structured extraction of key numbers from the raw data string.
    Node 2 asks Gemini to parse these — gives Node 3 clean inputs.
    """
    latest_revenue:       Optional[float] = None   # most recent period revenue
    latest_expenses:      Optional[float] = None
    latest_net_profit:    Optional[float] = None
    latest_profit_margin: Optional[float] = None   # as decimal e.g. 0.35
    avg_growth_rate:      Optional[float] = None   # as decimal e.g. 0.12
    trend_direction:      TrendDirection  = TrendDirection.STABLE
    data_period:          str             = ""     # e.g. "2021 Q1 to 2024 Q4"
    currency:             str             = "USD"
    num_periods:          int             = 0      # how many data points found
    parsing_notes:        str             = ""

    @field_validator("latest_profit_margin")
    @classmethod
    def margin_is_ratio(cls, v: Optional[float]) -> Optional[float]:
        """Accept both 0.35 (ratio) and 35.0 (percent) — normalise to ratio."""
        if v is None:
            return v
        if v > 1.5:               # looks like a percentage, convert
            return round(v / 100, 6)
        if not (-1.0 <= v <= 1.5):
            raise ValueError(f"profit_margin={v} is outside plausible range −100%…150%.")
        return v

    @field_validator("avg_growth_rate")
    @classmethod
    def growth_is_ratio(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if abs(v) > 2.0:          # looks like a percentage
            return round(v / 100, 6)
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM RESPONSE — NODE 3: FORECAST
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class QuarterForecast(BaseModel):
    period:    str           # e.g. "Q2 2025"
    revenue:   Optional[float] = None
    expenses:  Optional[float] = None
    net_profit:Optional[float] = None
    reasoning: str = ""


class AnnualForecast(BaseModel):
    year:           int
    revenue:        Optional[float] = None
    expenses:       Optional[float] = None
    net_profit:     Optional[float] = None
    profit_margin:  Optional[float] = None
    reasoning:      str = ""

    @field_validator("year")
    @classmethod
    def year_is_future(cls, v: int) -> int:
        current = datetime.utcnow().year
        if not (current <= v <= current + 5):
            raise ValueError(
                f"Forecast year {v} must be within the next 5 years "
                f"({current}–{current + 5})."
            )
        return v


class ForecastOutput(BaseModel):
    """
    Structured forecast produced by Node 3 (generate_forecast).
    Validated by Node 4 before being included in the final result.
    """
    short_term:      QuarterForecast
    annual:          AnnualForecast
    growth_rate:     float            # e.g. 0.12 = 12%
    growth_reasoning:str
    risks:           list[str]        = Field(..., min_length=1, max_length=10)
    confidence:      ConfidenceLevel
    confidence_explanation: str
    assumptions:     list[str]        = Field(default_factory=list)

    @field_validator("growth_rate")
    @classmethod
    def growth_rate_plausible(cls, v: float) -> float:
        """Normalise percent → ratio if needed, then range-check."""
        if abs(v) > 2.0:
            v = v / 100
        if not (-0.9 <= v <= 5.0):
            raise ValueError(
                f"growth_rate={v:.2%} is implausible (allowed: −90% to +500%). "
                "Check if the model produced a hallucinated number."
            )
        return round(v, 6)

    @field_validator("risks")
    @classmethod
    def risks_not_empty_strings(cls, v: list[str]) -> list[str]:
        cleaned = [r.strip() for r in v if r.strip()]
        if not cleaned:
            raise ValueError("risks list must contain at least one non-empty risk.")
        return cleaned

    @model_validator(mode="after")
    def annual_revenue_gt_short_term(self) -> "ForecastOutput":
        """Annual revenue should be ≥ quarterly if both are present."""
        qr = self.short_term.revenue
        ar = self.annual.revenue
        if qr and ar and ar < qr * 0.5:
            raise ValueError(
                f"Annual revenue forecast ({ar:,.0f}) is less than half of "
                f"the quarterly forecast ({qr:,.0f}) — likely a magnitude error."
            )
        return self


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastAgentState(BaseModel):
    """Mutable state object that flows through every graph node."""

    # ── Inputs ────────────────────────────────────────────
    data:     str = ""
    patterns: str = ""
    question: str = ""

    # ── Node 1 output ─────────────────────────────────────
    input_valid:  bool       = False
    input_errors: list[str]  = Field(default_factory=list)

    # ── Node 2 output ─────────────────────────────────────
    snapshot:     Optional[FinancialSnapshot] = None

    # ── Node 3 / 4 output ─────────────────────────────────
    forecast:         Optional[ForecastOutput] = None
    forecast_valid:   bool      = False
    forecast_errors:  list[str] = Field(default_factory=list)
    retry_count:      int       = 0
    max_retries:      int       = 2

    # ── Node 5 output ─────────────────────────────────────
    narrative:    str = ""   # human-readable explanation paragraph

    # ── Graph control ─────────────────────────────────────
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ForecastResult(BaseModel):
    """
    What ForecastAgent.run() returns to the caller.
    Drop-in structured replacement for the original str return value.
    """
    success: bool

    # ── Core forecast fields ───────────────────────────────
    short_term_period:  Optional[str]   = None   # "Q2 2025"
    short_term_revenue: Optional[float] = None
    short_term_expenses:Optional[float] = None
    short_term_profit:  Optional[float] = None
    short_term_reasoning: str           = ""

    annual_year:        Optional[int]   = None
    annual_revenue:     Optional[float] = None
    annual_expenses:    Optional[float] = None
    annual_profit:      Optional[float] = None
    annual_margin:      Optional[float] = None
    annual_reasoning:   str             = ""

    growth_rate:        Optional[float] = None   # as decimal
    growth_rate_pct:    Optional[str]   = None   # "12.5%" — pre-formatted
    growth_reasoning:   str             = ""

    risks:              list[str]       = Field(default_factory=list)
    confidence:         Optional[str]   = None
    confidence_explanation: str         = ""
    assumptions:        list[str]       = Field(default_factory=list)

    # ── Snapshot (parsed inputs) ───────────────────────────
    data_period:        str             = ""
    latest_revenue:     Optional[float] = None
    trend_direction:    Optional[str]   = None

    # ── Narrative ─────────────────────────────────────────
    narrative:          str             = ""

    # ── Audit / observability ─────────────────────────────
    model_used:   str        = ""
    duration_ms:  float      = 0.0
    retry_count:  int        = 0
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)
    timestamp:    str        = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )