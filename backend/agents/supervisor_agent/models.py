"""
src/models.py — SupervisorAgent
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Models for the orchestrator that runs all 5 sub-agents.

SupervisorInput     → validates question + financial_data dict
AgentRunRecord      → audit record for each sub-agent call
SupervisorState     → flows through all graph nodes
SupervisorResult    → final typed output (replaces original dict)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SupervisorInput(BaseModel):
    """
    Validates the two inputs:
        supervisor(question, financial_data)
    """
    question:       str  = Field(..., min_length=3, max_length=2_000)
    financial_data: dict = Field(...)

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

    @field_validator("financial_data")
    @classmethod
    def data_has_content(cls, v: dict) -> dict:
        if not v:
            raise ValueError("financial_data is empty — provide KPIs and/or revenue data.")
        has_kpis    = bool(v.get("kpis"))
        has_revenue = bool(v.get("revenue"))
        if not has_kpis and not has_revenue:
            raise ValueError(
                "financial_data must contain at least one of: 'kpis' or 'revenue'. "
                f"Found keys: {list(v.keys())}"
            )
        return v

    @model_validator(mode="after")
    def question_is_financial(self) -> "SupervisorInput":
        import re
        FINANCIAL_KW = {
            "revenue", "profit", "expense", "growth", "margin", "forecast",
            "trend", "pattern", "insight", "recommend", "strategy", "risk",
            "performance", "financial", "quarter", "annual", "year", "q1",
            "q2", "q3", "q4", "data", "analysis", "report", "summary",
            "result", "finding", "cost", "income", "loss", "budget", "kpi",
            "2023", "2024", "2025", "salesforce",
        }
        if not any(kw in self.question.lower() for kw in FINANCIAL_KW):
            raise ValueError(
                "Question does not appear to be about financial data. "
                "Use terms like revenue, profit, forecast, trend, etc."
            )
        return self


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT RUN RECORD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentRunRecord(BaseModel):
    """Audit record for each sub-agent call."""
    agent_name:   str
    status:       str          # "success" | "failed" | "skipped"
    duration_ms:  float        = 0.0
    output_len:   int          = 0    # character count of output
    error:        Optional[str]= None
    retry_count:  int          = 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUPERVISOR STATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SupervisorState(BaseModel):
    """Mutable state flowing through all 9 graph nodes."""

    # ── Raw inputs ──────────────────────────────────────
    question:       str  = ""
    financial_data: dict = Field(default_factory=dict)

    # ── Node 1: validate ────────────────────────────────
    input_valid:  bool      = False
    input_errors: list[str] = Field(default_factory=list)

    # ── Node 2: format data ─────────────────────────────
    db_summary:   str = ""   # formatted text for sub-agents

    # ── Node 3–7: sub-agent outputs ─────────────────────
    data_analysis: str = ""
    patterns:      str = ""
    forecast:      str = ""
    insights:      str = ""
    final_report:  str = ""

    # ── Sub-agent audit records ──────────────────────────
    agent_records: list[AgentRunRecord] = Field(default_factory=list)

    # ── Node 8: validate outputs ─────────────────────────
    outputs_valid:  bool      = False
    outputs_errors: list[str] = Field(default_factory=list)

    # ── Graph control ────────────────────────────────────
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL OUTPUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SupervisorResult(BaseModel):
    """
    Structured replacement for the original dict return value.

    Original returned:
        {"question": ..., "answer": ..., "details": {...}}

    This adds: typed fields, audit trail, per-agent timing, error reporting.
    """
    success:  bool

    # ── Original dict fields (drop-in compatible) ────────
    question: str      = ""
    answer:   str      = ""    # = details.final_report
    details:  dict     = Field(default_factory=dict)
    # details keys: data_analysis, patterns, forecast, insights

    # ── Per-agent audit ──────────────────────────────────
    agent_records: list[dict] = Field(default_factory=list)
    # Each: {agent_name, status, duration_ms, output_len, error}

    # ── Pipeline metadata ────────────────────────────────
    db_summary_len:    int   = 0
    agents_succeeded:  int   = 0
    agents_failed:     int   = 0
    total_output_chars:int   = 0

    # ── Audit ────────────────────────────────────────────
    model_used:   str        = ""
    duration_ms:  float      = 0.0
    node_history: list[dict] = Field(default_factory=list)
    errors:       list[dict] = Field(default_factory=list)
    timestamp:    str        = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )