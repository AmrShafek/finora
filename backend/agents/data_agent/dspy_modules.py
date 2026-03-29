"""
agents/data_agent/dspy_modules.py
──────────────────────────────────
DSPy Signatures and Modules for every LLM-calling node in the DataAgent graph.

NODE MAP:
  Node 2 — classify_intent  → ClassifyIntentModule
  Node 3 — build_data_plan  → BuildDataPlanModule

WHY SEPARATE MODULES?
  Each module:
    1. Documents the typed input/output contract via a dspy.Signature.
    2. Encapsulates the exact prompt templates from data_planning_agent.py,
       so the LLM receives the same detailed JSON-schema instructions as before.
    3. Is independently optimizable: DSPy optimizers (BootstrapFewShot,
       MIPROv2) can improve each module using labelled examples without
       touching the agent orchestration code.
    4. Calls FinancialJSONModule internally — the single DSPy adapter that
       replaces GeminiAdapter.generate_json().

REPLACING GeminiAdapter:
  BEFORE: raw = await self._gemini.generate_json(system_prompt, user_prompt)
  AFTER:  raw = await self._classify_intent.acall(question, db_summary)
          raw = await self._build_plan.acall(question, intent, ...)
"""

from __future__ import annotations

from typing import Optional

import dspy

from ..deep_agent.dspy_module import FinancialJSONModule


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy SIGNATURES  (typed contracts per node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClassifyIntentSignature(dspy.Signature):
    """
    Classify the intent and time scope of a Salesforce financial question.
    Returns a JSON object describing the intent type, year range, relevant metrics,
    and granularity.  Data range: 2011–2026.
    """

    question:   str = dspy.InputField(desc="Natural language financial question to classify")
    db_summary: str = dspy.InputField(desc="Summary of available database tables and columns")

    intent:      str       = dspy.OutputField(
        desc="One of: trend|comparison|snapshot|forecast|anomaly|ranking|summary|unknown"
    )
    confidence:  float     = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    time_period: str       = dspy.OutputField(
        desc="Human-readable period, e.g. '2023' or 'Q1-Q2 2024' or '2020-2024'"
    )
    year_start:  int       = dspy.OutputField(desc="Start year integer in range 2011–2026")
    year_end:    int       = dspy.OutputField(desc="End year integer in range 2011–2026")
    quarter:     str       = dspy.OutputField(desc="Q1, Q2, Q3, or Q4 — or null if not quarterly")
    granularity: str       = dspy.OutputField(desc="One of: annual|quarterly|monthly")
    key_metrics: list[str] = dspy.OutputField(
        desc="Subset of: revenue, expenses, net_profit, profit_margin, growth_rate, amount"
    )
    reasoning:   str       = dspy.OutputField(desc="One sentence explaining the intent choice")


class BuildDataPlanSignature(dspy.Signature):
    """
    Build a precise SQL data-retrieval plan from a classified financial question.
    Returns the tables, columns, filters, and aggregations needed to answer it.
    """

    question:    str       = dspy.InputField(desc="Original financial question")
    intent:      str       = dspy.InputField(desc="Detected intent type from classification step")
    time_period: str       = dspy.InputField(desc="Human-readable time period string")
    key_metrics: list[str] = dspy.InputField(desc="Metrics to retrieve")
    year_start:  int       = dspy.InputField(desc="Start year of the data range")
    year_end:    int       = dspy.InputField(desc="End year of the data range")

    tables_needed:  list[str]     = dspy.OutputField(
        desc="Table names from: revenue, expenses, kpis, ai_insights"
    )
    columns_needed: dict          = dspy.OutputField(
        desc="Mapping {table: [columns]} — only real columns that exist in each table"
    )
    filters:        dict          = dspy.OutputField(
        desc="SQL WHERE fragments {column: condition}, e.g. {'year': 'BETWEEN 2020 AND 2024'}"
    )
    aggregations:   list[str]     = dspy.OutputField(
        desc="SQL aggregation expressions e.g. SUM(revenue.amount). Empty list if raw rows suffice."
    )
    order_by:       Optional[str] = dspy.OutputField(desc="ORDER BY clause or null")
    limit:          Optional[int] = dspy.OutputField(desc="Row limit (1–10000) or null")
    explanation:    str           = dspy.OutputField(
        desc="One sentence describing what data is fetched and why"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy MODULES  (one per LLM node)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ClassifyIntentModule(dspy.Module):
    """
    Node 2 DSPy Module: classify the intent of a financial question.

    Wraps FinancialJSONModule with the DataAgent's _INTENT_PROMPT template.
    The LLM receives the same detailed JSON-schema instructions as before the
    refactor — only the SDK path (GeminiAdapter → DSPy LM) has changed.

    Signature ClassifyIntentSignature documents the typed interface and enables
    DSPy optimization without modifying the agent orchestration code.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        intent_prompt_template: str,
    ):
        """
        Args:
            json_module:             Shared FinancialJSONModule for this agent.
            system_prompt:           The agent-level _SYSTEM prompt.
            intent_prompt_template:  The _INTENT_PROMPT template string (with {question},
                                     {db_summary} placeholders).
        """
        super().__init__()
        self._json_module = json_module
        self._system      = system_prompt
        self._template    = intent_prompt_template

    async def acall(self, question: str, db_summary: str) -> dict:
        """
        Call the LM and return the raw JSON dict for intent classification.

        Args:
            question:   Financial question string (already validated/cleaned).
            db_summary: Database structure summary.

        Returns:
            Parsed dict suitable for LLMIntentResponse.model_validate().
        """
        user_prompt = self._template.format(
            question=question,
            db_summary=db_summary,
        )
        return await self._json_module.acall(self._system, user_prompt)


class BuildDataPlanModule(dspy.Module):
    """
    Node 3 DSPy Module: build a SQL data retrieval plan from a classified intent.

    Wraps FinancialJSONModule with the DataAgent's _DATA_PLAN_PROMPT template.
    Supports retry augmentation: when called after a failed validation attempt,
    the caller passes the list of plan_errors which are appended as a
    _RETRY_ADDENDUM so the LLM can self-correct.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        plan_prompt_template: str,
        retry_addendum_template: str,
    ):
        """
        Args:
            json_module:              Shared FinancialJSONModule for this agent.
            system_prompt:            The agent-level _SYSTEM prompt.
            plan_prompt_template:     The _DATA_PLAN_PROMPT template string.
            retry_addendum_template:  The _RETRY_ADDENDUM template (with {errors}).
        """
        super().__init__()
        self._json_module       = json_module
        self._system            = system_prompt
        self._template          = plan_prompt_template
        self._retry_addendum    = retry_addendum_template

    async def acall(
        self,
        question:     str,
        intent:       str,
        time_period:  str,
        key_metrics:  list,
        year_start:   int,
        year_end:     int,
        retry_errors: Optional[list[str]] = None,
    ) -> dict:
        """
        Call the LM and return the raw JSON dict for the data plan.

        Args:
            question:     Original financial question.
            intent:       Detected intent type string.
            time_period:  Human-readable time period.
            key_metrics:  List of metrics to retrieve.
            year_start:   Start year integer.
            year_end:     End year integer.
            retry_errors: If retrying after a failed validation, pass the list
                          of validation error strings so the LM can self-correct.

        Returns:
            Parsed dict suitable for LLMDataPlanResponse.model_validate().
        """
        user_prompt = self._template.format(
            question=question,
            intent=intent,
            time_period=time_period,
            key_metrics=key_metrics,
            year_start=year_start,
            year_end=year_end,
        )
        if retry_errors:
            user_prompt += self._retry_addendum.format(
                errors="\n".join(f"  - {e}" for e in retry_errors)
            )
        return await self._json_module.acall(self._system, user_prompt)
