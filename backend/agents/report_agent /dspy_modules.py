"""
agents/report_agent/dspy_modules.py
──────────────────────────────────────
DSPy Signatures and Modules for every LLM-calling node in the ReportAgent graph.

NODE MAP:
  Node 2 — plan_content  → PlanContentModule
  Node 3 — write_report  → WriteReportModule

REPLACING GeminiAdapter:
  BEFORE: raw = await self._gemini.generate_json(system_prompt, user_prompt)
  AFTER:  raw = await self._plan_mod.acall(question=..., ...)
          raw = await self._write_mod.acall(plan=..., data_analysis=..., ...)
"""

from __future__ import annotations

from typing import Optional

import dspy

from ..deep_agent.dspy_module import FinancialJSONModule


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy SIGNATURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlanContentSignature(dspy.Signature):
    """
    Create a content plan before writing the report.
    Identifies question type, key facts, and structure to ensure
    the written report stays under 300 words and directly answers the question.
    """
    question: str = dspy.InputField(desc="User's financial question")
    da_len:   int = dspy.InputField(desc="Length of data_analysis string in characters")
    pat_len:  int = dspy.InputField(desc="Length of patterns string in characters")
    fc_len:   int = dspy.InputField(desc="Length of forecast string in characters")
    ins_len:  int = dspy.InputField(desc="Length of insights string in characters")

    question_type:        str             = dspy.OutputField(desc="trend|forecast|comparison|risk|performance|general")
    direct_answer_point:  str             = dspy.OutputField(desc="Single most important fact answering the question (≤20 words)")
    top_findings:         list[str]       = dspy.OutputField(desc="2-3 key findings each ≤15 words")
    top_recommendations:  list[str]       = dspy.OutputField(desc="1-2 recommended actions each ≤15 words")
    key_number:           Optional[str]   = dspy.OutputField(desc="Most important number e.g. '$1.5M revenue' or null")
    word_limit:           int             = dspy.OutputField(desc="Word limit for the report (always 300)")
    tone:                 str             = dspy.OutputField(desc="professional")
    include_bullets:      bool            = dspy.OutputField(desc="Whether to include bullet points")
    include_forecast:     bool            = dspy.OutputField(desc="Whether to include forecast section")
    include_risk:         bool            = dspy.OutputField(desc="Whether to include risk section")


class WriteReportSignature(dspy.Signature):
    """
    Write a concise financial report under 300 words following the content plan.
    Must directly answer the user's question with specific numbers. Plain English only.
    """
    plan:          str = dspy.InputField(desc="JSON-serialised ContentPlan")
    data_analysis: str = dspy.InputField(desc="Data analysis text (truncated)")
    patterns:      str = dspy.InputField(desc="Patterns text (truncated)")
    forecast:      str = dspy.InputField(desc="Forecast text (truncated)")
    insights:      str = dspy.InputField(desc="Insights text (truncated)")
    question:      str = dspy.InputField(desc="User's financial question")
    key_number:    str = dspy.InputField(desc="Key number to include in direct answer")

    direct_answer:    str       = dspy.OutputField(desc="2-3 sentences directly answering the question (min 20 chars)")
    key_findings:     list[str] = dspy.OutputField(desc="2-3 findings each with a specific number (min 5 chars each)")
    recommendations:  list[str] = dspy.OutputField(desc="1-2 verb-led action recommendations (min 10 chars each)")
    summary_sentence: str       = dspy.OutputField(desc="One closing sentence summarising financial position (≤20 words)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DSPy MODULES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PlanContentModule(dspy.Module):
    """
    Node 2 DSPy Module: create content plan before writing.
    Wraps FinancialJSONModule with _PLAN_PROMPT from ReportAgent.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        plan_prompt_template: str,
    ):
        super().__init__()
        self._json_module = json_module
        self._system      = system_prompt
        self._template    = plan_prompt_template

    async def acall(
        self,
        question: str,
        da_len:   int,
        pat_len:  int,
        fc_len:   int,
        ins_len:  int,
    ) -> dict:
        """Returns dict for ContentPlan.model_validate()."""
        user_prompt = self._template.format(
            question=question,
            da_len=da_len, pat_len=pat_len,
            fc_len=fc_len, ins_len=ins_len,
        )
        return await self._json_module.acall(self._system, user_prompt)


class WriteReportModule(dspy.Module):
    """
    Node 3 DSPy Module: write the final concise report.
    Wraps FinancialJSONModule with _WRITE_PROMPT template.
    Appends _RETRY addendum when retry_errors are provided.
    """

    def __init__(
        self,
        json_module: FinancialJSONModule,
        system_prompt: str,
        write_prompt_template: str,
        retry_template: str,
    ):
        super().__init__()
        self._json_module = json_module
        self._system      = system_prompt
        self._template    = write_prompt_template
        self._retry       = retry_template

    async def acall(
        self,
        plan:          str,
        data_analysis: str,
        patterns:      str,
        forecast:      str,
        insights:      str,
        question:      str,
        key_number:    str,
        retry_errors:  Optional[list[str]] = None,
    ) -> dict:
        """Returns dict for ReportOutput.model_validate()."""
        user_prompt = self._template.format(
            plan=plan,
            data_analysis=data_analysis,
            patterns=patterns,
            forecast=forecast,
            insights=insights,
            question=question,
            key_number=key_number,
        )
        if retry_errors:
            user_prompt += self._retry.format(
                errors="\n".join(f"  - {e}" for e in retry_errors)
            )
        return await self._json_module.acall(self._system, user_prompt)
