from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENUMS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Intent(str, Enum):
    TREND        = "trend"
    COMPARISON   = "comparison"
    SNAPSHOT     = "snapshot"
    FORECAST     = "forecast"
    ANOMALY      = "anomaly"
    RANKING      = "ranking"
    SUMMARY      = "summary"
    UNKNOWN      = "unknown"


class Granularity(str, Enum):
    ANNUAL    = "annual"
    QUARTERLY = "quarterly"
    MONTHLY   = "monthly"


class Quarter(str, Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VALIDATION SETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_TABLES = {"revenue", "expenses", "kpis", "ai_insights"}

VALID_COLUMNS: dict[str, set[str]] = {
    "revenue":     {"year", "quarter", "amount", "region", "product_line"},
    "expenses":    {"year", "quarter", "amount", "category", "department"},
    "kpis":        {"year", "revenue", "expenses", "net_profit", "profit_margin", "growth_rate"},
    "ai_insights": {"year", "quarter", "insight_text", "category", "confidence_score"},
}

VALID_METRICS = {"revenue", "expenses", "profit", "net_profit", "profit_margin", "growth_rate", "kpi"}
VALID_AGG_FUNCS = {"SUM", "AVG", "MIN", "MAX"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentInput(BaseModel):
    """Validated input for DataAgent.run()."""

    question:   str = Field(..., min_length=3, max_length=2000)
    db_summary: str = Field(..., min_length=10, max_length=8000)

    @field_validator("question")
    @classmethod
    def no_injection(cls, v: str) -> str:
        """Block prompt injection and SQL/script attempts."""
        banned_patterns = [
            "ignore previous", "ignore above", "forget your instructions",
            "you are now", "act as", "jailbreak", "DAN",
            "drop table", "delete from", "insert into", "--", "/*",
            "<script", "javascript:", "eval(",
        ]
        lowered = v.lower()
        for pat in banned_patterns:
            if pat.lower() in lowered:
                raise ValueError(
                    f"Question contains a disallowed pattern: '{pat}'. "
                    "Please ask a genuine financial question."
                )
        return v.strip()

    @field_validator("question")
    @classmethod
    def not_empty_after_strip(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace only.")
        return v


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM RESPONSE SCHEMAS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMIntentResponse(BaseModel):
    """Output from Node 2 (classify_intent)."""

    intent:       Intent
    confidence:   float = Field(..., ge=0.0, le=1.0)
    time_period:  Optional[str] = None
    year_start:   Optional[int] = None
    year_end:     Optional[int] = None
    quarter:      Optional[Quarter] = None
    granularity:  Granularity = Granularity.QUARTERLY
    key_metrics:  list[str] = Field(default_factory=list)
    reasoning:    str = ""

    @field_validator("year_start", "year_end")
    @classmethod
    def year_sane(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (2000 <= v <= 2035):
            raise ValueError(f"Year {v} is outside the valid range 2000–2035.")
        return v

    @model_validator(mode="after")
    def year_order(self) -> "LLMIntentResponse":
        if self.year_start and self.year_end and self.year_start > self.year_end:
            raise ValueError(
                f"year_start ({self.year_start}) must be ≤ year_end ({self.year_end})."
            )
        return self


class LLMDataPlanResponse(BaseModel):
    """Output from Node 3 (build_data_plan)."""

    tables_needed:    list[str] = Field(..., min_length=1)
    columns_needed:   dict[str, list[str]]   # {table: [col1, col2]}
    filters:          dict[str, str]          # {column: value}
    aggregations:     list[str]
    order_by:         Optional[str] = None
    limit:            Optional[int] = Field(None, ge=1, le=10000)
    explanation:      str = ""

    @field_validator("tables_needed")
    @classmethod
    def tables_exist(cls, v: list[str]) -> list[str]:
        unknown = set(v) - VALID_TABLES
        if unknown:
            raise ValueError(
                f"Unknown table(s): {unknown}. Valid tables: {VALID_TABLES}"
            )
        return v

    @model_validator(mode="after")
    def columns_exist(self) -> "LLMDataPlanResponse":
        # Check columns exist
        for table, cols in self.columns_needed.items():
            if table not in VALID_TABLES:
                continue
            valid = VALID_COLUMNS.get(table, set())
            unknown = set(cols) - valid - {"*"}
            if unknown:
                raise ValueError(f"Unknown column(s) in '{table}': {unknown}. Valid: {valid}")
        # Check filters columns
        for col in self.filters.keys():
            found = any(col in VALID_COLUMNS.get(t, {}) for t in self.tables_needed)
            if not found:
                raise ValueError(f"Filter column '{col}' not present in any of the tables {self.tables_needed}")
        # Check aggregations
        for agg in self.aggregations:
            try:
                func, rest = agg.split("(", 1)
                col = rest.rstrip(")").split(".")[-1]
                table = rest.rstrip(")").split(".")[0]
                if func.upper() not in VALID_AGG_FUNCS:
                    raise ValueError(f"Invalid aggregation function: {func}")
                if col not in VALID_COLUMNS.get(table, {}):
                    raise ValueError(f"Aggregation column '{col}' not in table '{table}'")
            except Exception as e:
                raise ValueError(f"Invalid aggregation format: {agg}") from e
        return self


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataAgentState(BaseModel):
    """Mutable state flowing through all nodes."""

    # Inputs
    question: str = ""
    db_summary: str = ""

    # Node 1
    clean_question: str = ""
    input_valid: bool = False
    input_errors: list[str] = Field(default_factory=list)

    # Node 2
    intent: Optional[LLMIntentResponse] = None

    # Node 3
    data_plan: Optional[LLMDataPlanResponse] = None

    # Node 4
    plan_valid: bool = False
    plan_errors: list[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2

    # Graph control
    node_history: list[dict] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DataAgentResult(BaseModel):
    """Output of DataAgent.run() for production."""

    success: bool

    # Core fields
    intent: Optional[str] = None
    time_period: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    quarter: Optional[str] = None
    granularity: Optional[str] = None
    key_metrics: list[str] = Field(default_factory=list)
    tables_needed: list[str] = Field(default_factory=list)
    columns_needed: dict[str, list[str]] = Field(default_factory=dict)
    filters: dict[str, str] = Field(default_factory=dict)
    aggregations: list[str] = Field(default_factory=list)
    data_explanation: str = ""

    # Audit / observability
    model_used: str = ""
    duration_ms: float = 0.0
    retry_count: int = 0
    node_history: list[dict] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
