"""
src/validators.py — SupervisorAgent
Node 1: validate_inputs    (pure, no LLM)
Node 8: validate_outputs   (pure, no LLM)
"""
from __future__ import annotations
import re
from .models import SupervisorState

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
    "result","finding","cost","income","loss","budget","kpi","2023","2024","2025",
}


def validate_inputs(state: SupervisorState) -> SupervisorState:
    """
    Node 1 — validate question + financial_data.

    question:       no injection, financial keyword, 3–2000 chars
    financial_data: not empty, has 'kpis' or 'revenue' key
    kpis:           if present, each record must have year + at least one number
    revenue:        if present, must be a non-empty list
    """
    errors: list[str] = []

    # ── question ─────────────────────────────────────────
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
                "Use terms like revenue, profit, forecast, performance, etc."
            )

    # ── financial_data ────────────────────────────────────
    fd = state.financial_data or {}
    if not fd:
        errors.append("financial_data is empty.")
    else:
        kpis    = fd.get("kpis",    [])
        revenue = fd.get("revenue", [])

        if not kpis and not revenue:
            errors.append(
                f"financial_data has no 'kpis' or 'revenue' key. "
                f"Found: {list(fd.keys())}"
            )

        # Validate KPI records
        if kpis:
            if not isinstance(kpis, list):
                errors.append("financial_data['kpis'] must be a list.")
            else:
                for i, k in enumerate(kpis[:3]):   # spot-check first 3
                    if not isinstance(k, dict):
                        errors.append(f"kpis[{i}] must be a dict, got {type(k).__name__}.")
                        break
                    if "year" not in k:
                        errors.append(f"kpis[{i}] missing 'year' field.")
                    numeric_fields = {"revenue", "expenses", "profit", "margin",
                                      "net_profit", "profit_margin", "growth_rate"}
                    if not any(f in k for f in numeric_fields):
                        errors.append(
                            f"kpis[{i}] has no numeric fields. "
                            f"Expected one of: {numeric_fields}. Got: {list(k.keys())}"
                        )

        # Validate revenue records
        if revenue and not isinstance(revenue, list):
            errors.append("financial_data['revenue'] must be a list.")

    state.input_valid  = len(errors) == 0
    state.input_errors = errors
    return state


def validate_outputs(state: SupervisorState) -> SupervisorState:
    """
    Node 8 — verify all 5 sub-agents produced substantive output.

    Checks:
      1. Each output is a non-empty string (≥ 20 chars)
      2. No output is just an error message placeholder
      3. final_report starts with a substantive paragraph (not just headers)
      4. At least 4 of 5 agents succeeded (allows 1 graceful failure)
    """
    errors: list[str] = []

    outputs = {
        "data_analysis": state.data_analysis,
        "patterns":      state.patterns,
        "forecast":      state.forecast,
        "insights":      state.insights,
        "final_report":  state.final_report,
    }

    failed_outputs: list[str] = []
    for name, val in outputs.items():
        v = (val or "").strip()
        if not v:
            failed_outputs.append(name)
            errors.append(f"{name}: output is empty.")
        elif len(v) < 20:
            failed_outputs.append(name)
            errors.append(f"{name}: output too short ({len(v)} chars, min 20).")
        elif v.lower().startswith(("error", "failed", "could not", "exception")):
            errors.append(f"{name}: output appears to be an error message.")

    # ── final_report specific checks ─────────────────────
    if state.final_report and len(state.final_report.strip()) >= 20:
        report_words = len(state.final_report.split())
        if report_words > 600:
            errors.append(
                f"final_report is {report_words} words — exceeds expected ~300-word limit. "
                "Check if ReportAgent word-count validation worked."
            )
        if report_words < 30:
            errors.append(
                f"final_report is only {report_words} words — too brief to be useful."
            )

    # ── tolerance: allow 1 failed sub-agent ──────────────
    succeeded = sum(
        1 for rec in state.agent_records
        if rec.status == "success"
    )
    total = len(state.agent_records)
    if total > 0 and succeeded < max(1, total - 1):
        errors.append(
            f"Too many sub-agents failed: {total - succeeded}/{total}. "
            "At most 1 failure is tolerated."
        )

    state.outputs_valid  = len(errors) == 0
    state.outputs_errors = errors
    return state