"""
src/validators.py — ReportAgent
Node 1: validate_inputs   (pure, no LLM)
Node 4: validate_report   (pure, no LLM)
"""
from __future__ import annotations
import re
from typing import Optional
from .models import ReportAgentState, ReportOutput

_INJECTION_RE = re.compile(
    r"ignore\s+(previous|above|all|prior)"
    r"|forget\s+(your\s+)?instructions"
    r"|you\s+are\s+now|act\s+as|jailbreak|\bDAN\b"
    r"|drop\s+table|delete\s+from|<script|javascript:|eval\s*\(",
    re.IGNORECASE,
)
_FINANCIAL_KW = {
    "revenue","profit","expense","growth","margin","forecast","trend","pattern",
    "insight","recommend","strategy","risk","performance","financial","quarter",
    "annual","year","q1","q2","q3","q4","data","analysis","report","summary",
    "result","finding","2024","2025","cost","income","loss","budget","kpi",
}
_PLACEHOLDERS = {"none","n/a","null","empty","-","tbd",""}


def validate_inputs(state: ReportAgentState) -> ReportAgentState:
    errors: list[str] = []

    q = (state.question or "").strip()
    if not q or len(q) < 3:
        errors.append("question too short (min 3 chars).")
    elif len(q) > 2_000:
        errors.append(f"question too long ({len(q):,} chars, max 2 000).")
    else:
        m = _INJECTION_RE.search(q)
        if m:
            errors.append(f"question contains disallowed pattern: '{m.group()}'.")
        elif not any(kw in q.lower() for kw in _FINANCIAL_KW):
            errors.append(
                "question does not appear to be about financial data. "
                "Use terms like revenue, profit, report, forecast, etc."
            )

    for label, val, min_len, max_len in [
        ("data_analysis", state.data_analysis, 10, 20_000),
        ("patterns",      state.patterns,       5, 10_000),
        ("forecast",      state.forecast,       5, 10_000),
        ("insights",      state.insights,       5, 10_000),
    ]:
        v = (val or "").strip()
        if not v:
            errors.append(f"{label} is empty.")
        elif v.lower() in _PLACEHOLDERS:
            errors.append(f"{label} is a placeholder ('{v}').")
        elif len(v) < min_len:
            errors.append(f"{label} too short ({len(v)} chars, min {min_len}).")
        elif len(v) > max_len:
            errors.append(f"{label} too long ({len(v):,} chars, max {max_len:,}).")

    sources = [state.data_analysis, state.patterns, state.forecast, state.insights]
    if sum(1 for s in sources if len((s or "").strip()) >= 20) < 3 and not errors:
        errors.append("At least 3 of the 4 content inputs must have ≥ 20 chars of substance.")

    state.input_valid  = len(errors) == 0
    state.input_errors = errors
    return state


def validate_report(state: ReportAgentState) -> ReportAgentState:
    errors: list[str] = []
    out: Optional[ReportOutput] = state.report

    if out is None:
        state.report_valid  = False
        state.report_errors = ["No report produced — LLM call failed."]
        state.retry_count  += 1
        return state

    if len(out.direct_answer.strip()) < 20:
        errors.append(f"direct_answer too short ({len(out.direct_answer)} chars, min 20).")

    if not out.key_findings:
        errors.append("key_findings is empty.")
    for i, f in enumerate(out.key_findings):
        if len(f.strip()) < 5:
            errors.append(f"Finding {i+1} too short (min 5 chars).")

    if not out.recommendations:
        errors.append("recommendations is empty.")
    for i, r in enumerate(out.recommendations):
        if len(r.strip()) < 10:
            errors.append(f"Recommendation {i+1} too short (min 10 chars).")

    if len(out.summary_sentence.strip()) < 10:
        errors.append("summary_sentence too short (min 10 chars).")

    if out.word_count > 350:
        errors.append(f"Report is {out.word_count} words — exceeds 300 limit.")
    elif out.word_count < 50:
        errors.append(f"Report is only {out.word_count} words — too brief (min ~50).")

    state.report_valid  = len(errors) == 0
    state.report_errors = errors
    if errors:
        state.retry_count += 1
    return state
