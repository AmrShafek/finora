from __future__ import annotations

import re
from typing import Optional

from .models import InsightAgentState, InsightOutput


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1 — INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_INJECTION_RE = re.compile(
    r"ignore\s+(previous|above|all|prior)"
    r"|forget\s+(your\s+)?instructions"
    r"|you\s+are\s+now|act\s+as|jailbreak|\bDAN\b"
    r"|drop\s+table|delete\s+from|union\s+select"
    r"|<script|javascript:|eval\s*\(",
    re.IGNORECASE,
)

_FINANCIAL_KW = {
    "revenue", "profit", "margin", "expense", "growth", "insight", "recommend",
    "strategy", "action", "risk", "health", "performance", "financial", "kpi",
    "quarter", "annual", "year", "q1", "q2", "q3", "q4", "trend", "forecast",
    "2024", "2025", "2026", "cost", "income", "loss", "budget", "invest",
    "decision", "improve", "increase", "decrease", "optimize",
}

_PLACEHOLDERS = {"none", "n/a", "no patterns", "no forecast", "null", "empty", "-", "tbd", ""}


def validate_inputs(state: InsightAgentState) -> InsightAgentState:
    """
    Node 1 — validate all four inputs.

    Checks per field:
      data     : present, ≥ 3 numbers, ≤ 20 000 chars
      patterns : present, not placeholder, ≤ 10 000 chars
      forecast : present, not placeholder, ≤ 10 000 chars
      question : present, no injection, financial keyword, ≤ 2 000 chars
    """
    errors: list[str] = []

    # ── data ─────────────────────────────────────────────
    data = (state.data or "").strip()
    if not data:
        errors.append("data is empty — provide financial data summary.")
    elif len(data) > 20_000:
        errors.append(f"data too long ({len(data):,} chars, max 20 000).")
    elif len(re.findall(r"\d+(?:[.,]\d+)?", data)) < 3:
        errors.append(
            "data has fewer than 3 numeric values — "
            "provide actual financial figures."
        )

    # ── patterns ─────────────────────────────────────────
    patterns = (state.patterns or "").strip()
    if not patterns:
        errors.append("patterns is empty — provide Pattern Agent output.")
    elif patterns.lower() in _PLACEHOLDERS:
        errors.append(f"patterns is a placeholder ('{patterns}').")
    elif len(patterns) > 10_000:
        errors.append(f"patterns too long ({len(patterns):,} chars, max 10 000).")

    # ── forecast ─────────────────────────────────────────
    forecast = (state.forecast or "").strip()
    if not forecast:
        errors.append("forecast is empty — provide Forecast Agent output.")
    elif forecast.lower() in _PLACEHOLDERS:
        errors.append(f"forecast is a placeholder ('{forecast}').")
    elif len(forecast) > 10_000:
        errors.append(f"forecast too long ({len(forecast):,} chars, max 10 000).")

    # ── question ─────────────────────────────────────────
    question = (state.question or "").strip()
    if not question or len(question) < 3:
        errors.append("question is too short (min 3 chars).")
    elif len(question) > 2_000:
        errors.append(f"question too long ({len(question):,} chars, max 2 000).")
    else:
        m = _INJECTION_RE.search(question)
        if m:
            errors.append(
                f"question contains disallowed pattern: '{m.group()}'."
            )
        elif not any(kw in question.lower() for kw in _FINANCIAL_KW):
            errors.append(
                "question does not appear to be about financial insights. "
                "Use terms like revenue, profit, risk, strategy, recommend, etc."
            )

    state.input_valid  = len(errors) == 0
    state.input_errors = errors
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 4 — INSIGHT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MIN_INSIGHT_EXPLANATION_LEN = 20   # chars
_MIN_ACTION_RATIONALE_LEN    = 15


def validate_insights(state: InsightAgentState) -> InsightAgentState:
    """
    Node 4 — verify the InsightOutput from Node 3 meets quality standards.

    Checks:
      1. Output object exists
      2. direct_answer is substantive (≥ 20 chars)
      3. At least 1 insight, each with title + explanation + evidence
      4. At least 1 action with rationale
      5. health_score is 1–10
      6. key_risk is non-empty and specific (≥ 10 chars)
      7. executive_summary is present
      8. No insight explanation shorter than 20 chars (low-quality detection)
    """
    errors: list[str] = []
    out: Optional[InsightOutput] = state.insights

    if out is None:
        state.insights_valid  = False
        state.insights_errors = ["No insights produced — LLM call may have failed."]
        state.retry_count    += 1
        return state

    # ── 1. Direct answer substantive ─────────────────────
    if len(out.direct_answer.strip()) < 20:
        errors.append(
            f"direct_answer is too short ({len(out.direct_answer)} chars). "
            "Must provide a substantive answer to the user's question."
        )

    # ── 2. Insights quality ───────────────────────────────
    if not out.insights:
        errors.append("insights list is empty.")
    for i, ins in enumerate(out.insights):
        if not ins.title.strip():
            errors.append(f"Insight {i+1} has no title.")
        if len(ins.explanation.strip()) < _MIN_INSIGHT_EXPLANATION_LEN:
            errors.append(
                f"Insight {i+1} ('{ins.title}') explanation is too short "
                f"({len(ins.explanation)} chars, min {_MIN_INSIGHT_EXPLANATION_LEN})."
            )
        if not ins.evidence.strip():
            errors.append(
                f"Insight {i+1} ('{ins.title}') has no evidence — "
                "must cite a specific data point."
            )

    # ── 3. Actions quality ────────────────────────────────
    if not out.actions:
        errors.append("actions list is empty.")
    for i, act in enumerate(out.actions):
        if not act.action.strip():
            errors.append(f"Action {i+1} has no action text.")
        if len(act.rationale.strip()) < _MIN_ACTION_RATIONALE_LEN:
            errors.append(
                f"Action {i+1} rationale is too short "
                f"({len(act.rationale)} chars)."
            )

    # ── 4. Health score ───────────────────────────────────
    if not (1 <= out.health_score <= 10):
        errors.append(
            f"health_score={out.health_score} is outside 1–10 range."
        )

    # ── 5. Key risk specific ──────────────────────────────
    if len(out.key_risk.strip()) < 10:
        errors.append(
            f"key_risk is too short ({len(out.key_risk)} chars). "
            "Must describe a specific risk, not a generic placeholder."
        )

    # ── 6. Executive summary present ─────────────────────
    if len(out.executive_summary.strip()) < 30:
        errors.append(
            "executive_summary is missing or too short (min 30 chars)."
        )

    state.insights_valid  = len(errors) == 0
    state.insights_errors = errors
    if errors:
        state.retry_count += 1

    return state
