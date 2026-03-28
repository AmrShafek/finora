from __future__ import annotations

import re
from typing import Optional

from .models import (
    DataAgentState,
    LLMDataPlanResponse,
    VALID_TABLES,
    VALID_COLUMNS,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 1 — INPUT VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Patterns that should never appear in a legitimate financial question
_INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all|prior)",
    r"forget\s+(your\s+)?instructions",
    r"you\s+are\s+now",
    r"act\s+as",
    r"new\s+persona",
    r"jailbreak",
    r"\bDAN\b",
    # SQL injection
    r"drop\s+table",
    r"delete\s+from",
    r"insert\s+into",
    r"union\s+select",
    r"--\s",
    r"/\*.*\*/",
    # Script injection
    r"<script",
    r"javascript:",
    r"eval\s*\(",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.DOTALL,
)

# Financial keywords — at least one must appear for the question to be relevant
_FINANCIAL_KEYWORDS = {
    "revenue", "expense", "profit", "margin", "kpi", "cost", "income",
    "loss", "growth", "quarter", "annual", "year", "q1", "q2", "q3", "q4",
    "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026",
    "salesforce", "finora", "financial", "budget", "forecast", "trend",
    "performance", "metric", "data", "insight", "compare", "best", "worst",
    "highest", "lowest", "average", "total", "sum",
}


def validate_input(state: DataAgentState) -> DataAgentState:
    """
    Node 1: Sanitize and validate the raw question.

    Checks:
      1. Not empty after stripping whitespace
      2. Not too short (< 5 chars) or too long (> 2000 chars)
      3. No injection patterns
      4. Contains at least one financial keyword
      5. db_summary is present and not empty

    On failure: sets state.input_valid=False and appends to state.input_errors.
    On success: sets state.clean_question and state.input_valid=True.
    """
    errors: list[str] = []
    question = (state.question or "").strip()

    # ── 1. Length checks ───────────────────────────────
    if not question:
        errors.append("Question is empty.")
    elif len(question) < 5:
        errors.append(
            f"Question is too short ({len(question)} chars). "
            "Please ask a complete question."
        )
    elif len(question) > 2000:
        errors.append(
            f"Question is too long ({len(question)} chars, max 2000). "
            "Please shorten your question."
        )

    # ── 2. Injection guard ─────────────────────────────
    if question and _INJECTION_RE.search(question):
        match = _INJECTION_RE.search(question)
        errors.append(
            f"Question contains a disallowed pattern ('{match.group()}'). "
            "Please ask a genuine financial question."
        )

    # ── 3. Financial relevance ─────────────────────────
    if question and not errors:
        lowered = question.lower()
        has_keyword = any(kw in lowered for kw in _FINANCIAL_KEYWORDS)
        if not has_keyword:
            errors.append(
                "Question does not appear to be about financial data. "
                "Please ask about revenue, expenses, profit, KPIs, or a specific time period."
            )

    # ── 4. DB summary present ──────────────────────────
    if not (state.db_summary or "").strip():
        errors.append("db_summary is empty — cannot identify available data.")

    # ── Result ─────────────────────────────────────────
    if errors:
        state.input_valid   = False
        state.input_errors  = errors
        state.clean_question = ""
    else:
        state.input_valid    = True
        state.input_errors   = []
        # Light sanitization: collapse whitespace, trim
        state.clean_question = re.sub(r"\s+", " ", question).strip()

    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODE 4 — PLAN VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_plan(state: DataAgentState) -> DataAgentState:
    """
    Node 4: Verify the DataPlan produced by the LLM is executable.

    Checks:
      1. At least one table is requested
      2. Every table exists in VALID_TABLES
      3. Every column exists in the table's VALID_COLUMNS
      4. Year range is sane (2011–2026, matches Salesforce data range)
      5. Quarter format is valid if provided
      6. Aggregation strings are non-empty if present

    On failure: sets state.plan_valid=False + errors.
                If retry_count < max_retries → signal retry to graph.
    On success: sets state.plan_valid=True.
    """
    errors: list[str] = []
    plan: Optional[LLMDataPlanResponse] = state.data_plan
    intent = state.intent

    if plan is None:
        state.plan_valid   = False
        state.plan_errors  = ["No data plan was produced by the LLM."]
        return state

    # ── 1. Tables ──────────────────────────────────────
    if not plan.tables_needed:
        errors.append("No tables specified in data plan.")
    else:
        unknown_tables = set(plan.tables_needed) - VALID_TABLES
        if unknown_tables:
            errors.append(
                f"Unknown table(s) in plan: {sorted(unknown_tables)}. "
                f"Valid: {sorted(VALID_TABLES)}"
            )

    # ── 2. Columns ─────────────────────────────────────
    for table, cols in plan.columns_needed.items():
        if table not in VALID_TABLES:
            continue
        valid_cols = VALID_COLUMNS[table] | {"*"}
        unknown_cols = set(cols) - valid_cols
        if unknown_cols:
            errors.append(
                f"Unknown column(s) in '{table}': {sorted(unknown_cols)}. "
                f"Valid: {sorted(VALID_COLUMNS[table])}"
            )

    # ── 3. Year range ──────────────────────────────────
    DATA_MIN_YEAR = 2011
    DATA_MAX_YEAR = 2026

    if intent:
        for label, yr in [("year_start", intent.year_start), ("year_end", intent.year_end)]:
            if yr is not None and not (DATA_MIN_YEAR <= yr <= DATA_MAX_YEAR):
                errors.append(
                    f"{label}={yr} is outside the available data range "
                    f"({DATA_MIN_YEAR}–{DATA_MAX_YEAR})."
                )
        if intent.year_start and intent.year_end:
            if intent.year_start > intent.year_end:
                errors.append(
                    f"year_start ({intent.year_start}) is after "
                    f"year_end ({intent.year_end})."
                )

    # ── 4. Quarter format ──────────────────────────────
    if intent and intent.quarter:
        if intent.quarter not in {"Q1", "Q2", "Q3", "Q4"}:
            errors.append(
                f"Invalid quarter '{intent.quarter}'. Must be Q1, Q2, Q3, or Q4."
            )

    # ── 5. Aggregations non-empty ──────────────────────
    for agg in plan.aggregations:
        if not agg.strip():
            errors.append("Empty string found in aggregations list.")

    # ── Result ─────────────────────────────────────────
    if errors:
        state.plan_valid  = False
        state.plan_errors = errors
        state.retry_count += 1
    else:
        state.plan_valid  = True
        state.plan_errors = []

    return state
