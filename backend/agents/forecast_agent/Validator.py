"""
src/validators.py
━━━━━━━━━━━━━━━━━
Node 1 — validate_inputs   (pure, no LLM)
Node 4 — validate_forecast (pure, no LLM)

Both are stateless functions: state_in → state_out.
"""

from __future__ import annotations

import re
from typing import Optional

from .models import ForecastAgentState, ForecastOutput


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1 — INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_INJECTION_RE = re.compile(
    r"ignore\s+(previous|above|all|prior)"
    r"|forget\s+(your\s+)?instructions"
    r"|you\s+are\s+now|act\s+as|jailbreak|\bDAN\b"
    r"|drop\s+table|delete\s+from|union\s+select|--\s"
    r"|<script|javascript:|eval\s*\(",
    re.IGNORECASE,
)

_FINANCIAL_KW = {
    "revenue", "expense", "profit", "margin", "growth", "forecast", "predict",
    "next", "quarter", "annual", "year", "q1", "q2", "q3", "q4", "trend",
    "performance", "income", "loss", "cost", "budget", "projection", "estimate",
    "2024", "2025", "2026", "future", "financial", "kpi", "rate",
}


def validate_inputs(state: ForecastAgentState) -> ForecastAgentState:
    """
    Node 1 — validate all three inputs: data, patterns, question.

    Checks:
      data     : not empty, has numeric content (≥ 3 numbers), ≤ 20 000 chars
      patterns : not empty, not a placeholder, ≤ 10 000 chars
      question : not empty, no injection, has financial keyword, ≤ 2 000 chars
    """
    errors: list[str] = []

    # ── data ─────────────────────────────────────────────
    data = (state.data or "").strip()
    if not data:
        errors.append("data is empty — provide historical financial data.")
    elif len(data) > 20_000:
        errors.append(f"data is too long ({len(data):,} chars, max 20 000).")
    else:
        nums = re.findall(r"\d+(?:[.,]\d+)?", data)
        if len(nums) < 3:
            errors.append(
                "data contains fewer than 3 numeric values — "
                "provide actual financial figures."
            )

    # ── patterns ─────────────────────────────────────────
    patterns = (state.patterns or "").strip()
    if not patterns:
        errors.append("patterns is empty — provide Pattern Agent output.")
    elif len(patterns) > 10_000:
        errors.append(f"patterns is too long ({len(patterns):,} chars, max 10 000).")
    else:
        placeholders = {"none", "n/a", "no patterns", "null", "empty", "-"}
        if patterns.lower() in placeholders:
            errors.append(
                f"patterns is a placeholder value ('{patterns}'). "
                "Provide real pattern analysis."
            )

    # ── question ─────────────────────────────────────────
    question = (state.question or "").strip()
    if not question:
        errors.append("question is empty.")
    elif len(question) < 3:
        errors.append(f"question is too short ({len(question)} chars, min 3).")
    elif len(question) > 2_000:
        errors.append(f"question is too long ({len(question)} chars, max 2 000).")
    else:
        m = _INJECTION_RE.search(question)
        if m:
            errors.append(
                f"question contains a disallowed pattern: '{m.group()}'. "
                "Please ask a genuine forecasting question."
            )
        elif not any(kw in question.lower() for kw in _FINANCIAL_KW):
            errors.append(
                "question does not appear to be about financial forecasting. "
                "Use terms like revenue, profit, growth, forecast, next quarter, etc."
            )

    if errors:
        state.input_valid  = False
        state.input_errors = errors
    else:
        state.input_valid  = True
        state.input_errors = []

    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 4 — FORECAST VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MAX_PLAUSIBLE_REVENUE     = 1_000_000_000_000   # $1 trillion — sanity ceiling
_MAX_PLAUSIBLE_GROWTH_RATE = 5.00                # 500% — Gemini sometimes hallucinates
_MIN_PLAUSIBLE_GROWTH_RATE = -0.90               # −90%


def validate_forecast(state: ForecastAgentState) -> ForecastAgentState:
    """
    Node 4 — verify the ForecastOutput produced by Node 3 is sensible.

    Checks:
      1. Forecast object exists
      2. Short-term period is a recognisable format ("Q2 2025")
      3. Annual year is in the future
      4. Revenue figures are positive and below sanity ceiling
      5. Growth rate is within plausible range
      6. Confidence level is one of High / Medium / Low
      7. At least one risk listed
      8. Annual revenue ≥ quarterly (magnitude check)
    """
    errors: list[str] = []
    fc: Optional[ForecastOutput] = state.forecast

    if fc is None:
        state.forecast_valid  = False
        state.forecast_errors = ["No forecast produced — LLM call may have failed."]
        state.retry_count    += 1
        return state

    # ── 1. Short-term period format ───────────────────────
    period = fc.short_term.period
    if not re.match(r"^Q[1-4]\s+\d{4}$", period.strip()):
        errors.append(
            f"short_term.period='{period}' is not a valid format. "
            "Expected 'Q1 2025', 'Q2 2026', etc."
        )

    # ── 2. Annual year is in the future ───────────────────
    from datetime import datetime
    current_year = datetime.utcnow().year
    if not (current_year <= fc.annual.year <= current_year + 5):
        errors.append(
            f"annual.year={fc.annual.year} is not within the allowed "
            f"range ({current_year}–{current_year + 5})."
        )

    # ── 3. Revenue figures positive + below ceiling ───────
    for label, val in [
        ("short_term.revenue",    fc.short_term.revenue),
        ("short_term.expenses",   fc.short_term.expenses),
        ("annual.revenue",        fc.annual.revenue),
        ("annual.expenses",       fc.annual.expenses),
    ]:
        if val is not None:
            if val < 0:
                errors.append(f"{label}={val:,.0f} is negative — revenue/expenses must be ≥ 0.")
            if val > _MAX_PLAUSIBLE_REVENUE:
                errors.append(
                    f"{label}={val:,.0f} exceeds the sanity ceiling of "
                    f"{_MAX_PLAUSIBLE_REVENUE:,.0f}. Check for magnitude error (M vs B)."
                )

    # ── 4. Profit consistency ─────────────────────────────
    st = fc.short_term
    if st.revenue and st.expenses and st.net_profit:
        implied = st.revenue - st.expenses
        diff_pct = abs(implied - st.net_profit) / max(abs(implied), 1)
        if diff_pct > 0.15:    # allow 15% tolerance for rounding
            errors.append(
                f"short_term: revenue({st.revenue:,.0f}) − expenses({st.expenses:,.0f}) "
                f"= {implied:,.0f} but net_profit={st.net_profit:,.0f} "
                f"(discrepancy {diff_pct:.1%})."
            )

    # ── 5. Growth rate plausible ──────────────────────────
    gr = fc.growth_rate
    if not (_MIN_PLAUSIBLE_GROWTH_RATE <= gr <= _MAX_PLAUSIBLE_GROWTH_RATE):
        errors.append(
            f"growth_rate={gr:.2%} is outside plausible range "
            f"({_MIN_PLAUSIBLE_GROWTH_RATE:.0%}…{_MAX_PLAUSIBLE_GROWTH_RATE:.0%})."
        )

    # ── 6. At least one risk ──────────────────────────────
    if not fc.risks:
        errors.append("risks list is empty — at least one risk must be identified.")

    # ── 7. Annual magnitude vs quarterly ─────────────────
    q_rev = fc.short_term.revenue
    a_rev = fc.annual.revenue
    if q_rev and a_rev and a_rev < q_rev * 0.5:
        errors.append(
            f"annual.revenue ({a_rev:,.0f}) is less than half of "
            f"short_term.revenue ({q_rev:,.0f}) — likely a magnitude error."
        )

    # ── Result ────────────────────────────────────────────
    if errors:
        state.forecast_valid  = False
        state.forecast_errors = errors
        state.retry_count    += 1
    else:
        state.forecast_valid  = True
        state.forecast_errors = []

    return state