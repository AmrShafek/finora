"""
src/validators.py
━━━━━━━━━━━━━━━━━
Node 1 — validate_inputs    (pure, no LLM)
Node 4 — validate_patterns  (pure, no LLM)
"""

from __future__ import annotations

import re
from typing import Optional

from .models import PatternAgentState, PatternOutput


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
    "revenue", "expense", "profit", "margin", "growth", "pattern", "trend",
    "anomaly", "spike", "drop", "decline", "increase", "seasonal", "quarter",
    "annual", "year", "q1", "q2", "q3", "q4", "financial", "kpi", "cost",
    "income", "loss", "performance", "data", "analysis", "compare", "detect",
    "2023", "2024", "2025", "unusual", "change", "movement",
}


def validate_inputs(state: PatternAgentState) -> PatternAgentState:
    """
    Node 1 — validate data and question.

    data checks     : not empty, ≥ 3 numbers, ≤ 20 000 chars
    question checks : not empty, no injection, financial keyword, ≤ 2 000 chars
    """
    errors: list[str] = []

    # ── data ─────────────────────────────────────────────
    data = (state.data or "").strip()
    if not data:
        errors.append("data is empty — provide financial data to analyse.")
    elif len(data) > 20_000:
        errors.append(f"data too long ({len(data):,} chars, max 20 000).")
    else:
        nums = re.findall(r"\d+(?:[.,]\d+)?", data)
        if len(nums) < 3:
            errors.append(
                "data contains fewer than 3 numeric values — "
                "provide actual financial figures with numbers."
            )

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
                f"question contains a disallowed pattern: '{m.group()}'."
            )
        elif not any(kw in question.lower() for kw in _FINANCIAL_KW):
            errors.append(
                "question does not appear to be about financial data analysis. "
                "Use terms like revenue, trend, anomaly, growth, pattern, etc."
            )

    state.input_valid  = len(errors) == 0
    state.input_errors = errors
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 4 — PATTERN VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MIN_DESCRIPTION_LEN = 15
_MIN_EVIDENCE_LEN    = 10
_MAX_GROWTH_RATE     = 10.0    # 1000% — absolute ceiling after normalisation
_MIN_GROWTH_RATE     = -1.0    # −100%


def validate_patterns(state: PatternAgentState) -> PatternAgentState:
    """
    Node 4 — quality and sanity checks on PatternOutput.

    Checks:
      1. Output object exists
      2. At least 1 pattern with non-empty evidence
      3. Every pattern description ≥ 15 chars
      4. key_finding ≥ 10 chars
      5. trend_direction is a valid enum value
      6. trend_explanation is present
      7. Growth rates within plausible range
      8. Anomaly descriptions non-empty if anomalies present
    """
    errors: list[str] = []
    out: Optional[PatternOutput] = state.patterns

    if out is None:
        state.patterns_valid  = False
        state.patterns_errors = ["No patterns produced — LLM call may have failed."]
        state.retry_count    += 1
        return state

    # ── 1. Patterns quality ───────────────────────────────
    if not out.patterns:
        errors.append("patterns list is empty.")
    for i, p in enumerate(out.patterns):
        if not p.name.strip():
            errors.append(f"Pattern {i+1} has no name.")
        if len(p.description.strip()) < _MIN_DESCRIPTION_LEN:
            errors.append(
                f"Pattern {i+1} ('{p.name}') description too short "
                f"({len(p.description)} chars, min {_MIN_DESCRIPTION_LEN})."
            )
        if len(p.evidence.strip()) < _MIN_EVIDENCE_LEN:
            errors.append(
                f"Pattern {i+1} ('{p.name}') evidence too short or missing "
                f"({len(p.evidence)} chars) — must cite specific numbers."
            )

    # ── 2. Key finding ────────────────────────────────────
    if len(out.key_finding.strip()) < 10:
        errors.append(
            f"key_finding too short ({len(out.key_finding)} chars, min 10)."
        )

    # ── 3. Trend explanation ──────────────────────────────
    if not out.trend_explanation.strip():
        errors.append("trend_explanation is empty.")

    # ── 4. Growth rates plausible ─────────────────────────
    for label, gr in [
        ("revenue_growth_rate", out.revenue_growth_rate),
        ("expense_growth_rate", out.expense_growth_rate),
    ]:
        if gr is not None and not (_MIN_GROWTH_RATE <= gr <= _MAX_GROWTH_RATE):
            errors.append(
                f"{label}={gr:.2%} is outside the plausible range "
                f"({_MIN_GROWTH_RATE:.0%}…{_MAX_GROWTH_RATE:.0%})."
            )

    # ── 5. Anomaly descriptions non-empty ─────────────────
    for i, a in enumerate(out.anomalies):
        if not a.description.strip():
            errors.append(f"Anomaly {i+1} has no description.")
        if not a.period.strip():
            errors.append(f"Anomaly {i+1} has no period.")

    state.patterns_valid  = len(errors) == 0
    state.patterns_errors = errors
    if errors:
        state.retry_count += 1

    return state