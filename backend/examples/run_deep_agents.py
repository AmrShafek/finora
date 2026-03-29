"""
examples/run_deep_agents.py
────────────────────────────
End-to-end demonstration of the refactored DeepAgent + DSPy + LangGraph pipeline.

Runs all five agents in sequence:
  DataAgent → PatternAgent → ForecastAgent → InsightAgent → ReportAgent

Each agent reads GEMINI_API_KEY from the environment (or .env file).

Usage:
    cd backend
    python -m examples.run_deep_agents

    # Or with a custom question:
    FINORA_QUESTION="What is the revenue trend?" python -m examples.run_deep_agents
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from pathlib import Path

# ── Allow running from the backend root ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional

from loguru import logger

from agents.data_agent.data_planning_agent import DataAgent
from agents.Pattern_Agent.pattern_Agent import PatternAgent
from agents.forecast_agent.Forcasting_agent import ForecastAgent
from agents.insight_agent.Insight_Agent import InsightAgent
from agents.report_agent.report_agent import ReportAgent


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SAMPLE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SAMPLE_DATA = """
Salesforce Financial Data (USD, quarterly)

Quarter | Revenue    | Expenses   | Net Profit | Profit Margin
--------|------------|------------|------------|---------------
2022 Q1 | 1,230,000  | 980,000    | 250,000    | 20.3%
2022 Q2 | 1,290,000  | 1,010,000  | 280,000    | 21.7%
2022 Q3 | 1,340,000  | 1,050,000  | 290,000    | 21.6%
2022 Q4 | 1,410,000  | 1,100,000  | 310,000    | 22.0%
2023 Q1 | 1,480,000  | 1,140,000  | 340,000    | 23.0%
2023 Q2 | 1,550,000  | 1,185,000  | 365,000    | 23.5%
2023 Q3 | 1,620,000  | 1,225,000  | 395,000    | 24.4%
2023 Q4 | 1,710,000  | 1,270,000  | 440,000    | 25.7%
2024 Q1 | 1,780,000  | 1,310,000  | 470,000    | 26.4%
2024 Q2 | 1,860,000  | 1,360,000  | 500,000    | 26.9%

Notes:
- Revenue growth: ~51% from 2022 Q1 to 2024 Q2
- Expenses growth: ~38% over same period (slower than revenue)
- Net profit growth: ~100% — margin expansion from 20% to 27%
- No negative quarters. Strong and consistent upward trajectory.
"""

QUESTION = os.getenv(
    "FINORA_QUESTION",
    "What is the current financial health of the company and what should we focus on next quarter?"
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _section(title: str) -> None:
    bar = "━" * 60
    logger.info(f"\n{bar}\n  {title}\n{bar}")


def _result_summary(agent_name: str, result, extra: str = "") -> None:
    status = "✅ SUCCESS" if result.success else "❌ FAILED"
    logger.info(
        f"  {status} | {agent_name} | {result.duration_ms}ms"
        + (f" | {extra}" if extra else "")
    )
    if not result.success:
        for err in result.errors:
            logger.error(f"    Error [{err.get('node','?')}]: {err.get('error','?')}")


def _print_report(result) -> None:
    if result.success and result.text:
        print("\n" + "═" * 60)
        print("  FINAL REPORT")
        print("═" * 60)
        for line in result.text.splitlines():
            print(f"  {line}")
        print("═" * 60 + "\n")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def run_pipeline() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.error(
            "GEMINI_API_KEY not found.\n"
            "Set it in your environment or a .env file:\n"
            "  export GEMINI_API_KEY=your_key_here"
        )
        sys.exit(1)

    logger.info(f"Question: {QUESTION}")
    logger.info(f"Model:    gemini-2.5-flash\n")

    # ── DataAgent ─────────────────────────────────────────────────────────────
    _section("AGENT 1 — DataAgent")
    data_agent = DataAgent(api_key=api_key)
    data_result = await data_agent.run(question=QUESTION, data=SAMPLE_DATA)
    _result_summary("DataAgent", data_result,
                    f"intent={data_result.intent or '?'} tables={data_result.tables or []}")

    data_analysis_str = (
        data_result.plan_json or SAMPLE_DATA
        if data_result.success else SAMPLE_DATA
    )

    # ── PatternAgent ──────────────────────────────────────────────────────────
    _section("AGENT 2 — PatternAgent")
    pattern_agent  = PatternAgent(api_key=api_key)
    pattern_result = await pattern_agent.run(data=SAMPLE_DATA)
    _result_summary("PatternAgent", pattern_result,
                    f"patterns={pattern_result.pattern_count or 0} anomalies={pattern_result.anomaly_count or 0}")

    patterns_str = pattern_result.narrative or str(pattern_result.patterns or "")

    # ── ForecastAgent ─────────────────────────────────────────────────────────
    _section("AGENT 3 — ForecastAgent")
    forecast_agent  = ForecastAgent(api_key=api_key)
    forecast_result = await forecast_agent.run(
        data=SAMPLE_DATA,
        patterns=patterns_str,
        question=QUESTION,
    )
    _result_summary("ForecastAgent", forecast_result,
                    f"growth={forecast_result.growth_rate_pct or '?'} confidence={forecast_result.confidence or '?'}")

    forecast_str = forecast_result.narrative or ""

    # ── InsightAgent ──────────────────────────────────────────────────────────
    _section("AGENT 4 — InsightAgent")
    insight_agent  = InsightAgent(api_key=api_key)
    insight_result = await insight_agent.run(
        data=SAMPLE_DATA,
        patterns=patterns_str,
        forecast=forecast_str,
        question=QUESTION,
    )
    _result_summary("InsightAgent", insight_result,
                    f"health={insight_result.health_score or '?'}/10 insights={len(insight_result.insights or [])}")

    if insight_result.success and insight_result.insights:
        for i, ins in enumerate(insight_result.insights[:3], 1):
            logger.info(f"    [{i}] {ins['title']}: {textwrap.shorten(ins['evidence'], 80)}")

    insights_str = insight_result.narrative or insight_result.executive_summary or ""

    # ── ReportAgent ───────────────────────────────────────────────────────────
    _section("AGENT 5 — ReportAgent")
    report_agent  = ReportAgent(api_key=api_key)
    report_result = await report_agent.run(
        question=QUESTION,
        data_analysis=data_analysis_str,
        patterns=patterns_str,
        forecast=forecast_str,
        insights=insights_str,
    )
    _result_summary("ReportAgent", report_result,
                    f"words={report_result.word_count or 0}")

    # ── Final output ──────────────────────────────────────────────────────────
    _print_report(report_result)

    # ── Pipeline summary ──────────────────────────────────────────────────────
    _section("PIPELINE SUMMARY")
    agents = [
        ("DataAgent",     data_result),
        ("PatternAgent",  pattern_result),
        ("ForecastAgent", forecast_result),
        ("InsightAgent",  insight_result),
        ("ReportAgent",   report_result),
    ]
    total_ms    = sum(r.duration_ms for _, r in agents)
    all_success = all(r.success for _, r in agents)

    for name, r in agents:
        icon = "✅" if r.success else "❌"
        retries = getattr(r, "retry_count", 0)
        logger.info(f"  {icon} {name:<15} {r.duration_ms:>7.1f}ms  retries={retries}")

    logger.info(f"\n  Total time : {total_ms:.1f}ms  ({total_ms/1000:.2f}s)")
    logger.info(f"  Pipeline   : {'ALL PASSED ✅' if all_success else 'PARTIAL FAILURES ❌'}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    asyncio.run(run_pipeline())
