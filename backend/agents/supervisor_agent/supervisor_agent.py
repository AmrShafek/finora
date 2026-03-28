from __future__ import annotations

import os
import time
from typing import Callable, Optional

from loguru import logger

from .models import AgentRunRecord, SupervisorInput, SupervisorResult, SupervisorState
from .validators import validate_inputs, validate_outputs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA FORMATTER  (Node 2 — pure Python)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def format_financial_data(financial_data: dict) -> str:
    """
    Convert the financial_data dict into a readable text summary
    that all sub-agents can consume.

    Handles both the original dict structure and variations.
    """
    lines: list[str] = []

    # ── KPIs ─────────────────────────────────────────────
    kpis = financial_data.get("kpis", [])
    if kpis:
        lines.append("Annual KPIs (most recent 6 years):")
        for k in kpis[-6:]:
            year    = k.get("year", "?")
            rev     = k.get("revenue",     k.get("total_revenue", 0)) or 0
            exp     = k.get("expenses",    k.get("total_expenses", 0)) or 0
            profit  = k.get("profit",      k.get("net_profit", 0)) or 0
            margin  = k.get("margin",      k.get("profit_margin", 0)) or 0

            # Handle both raw numbers and millions/billions
            def fmt(v: float) -> str:
                if v == 0:
                    return "$0"
                if abs(v) >= 1e9:
                    return f"${v/1e9:.2f}B"
                if abs(v) >= 1e6:
                    return f"${v/1e6:.1f}M"
                return f"${v:,.0f}"

            margin_str = f"{margin:.1f}%" if margin > 1 else f"{margin*100:.1f}%"
            lines.append(
                f"  {year}: Revenue={fmt(rev)}, Expenses={fmt(exp)}, "
                f"Net Profit={fmt(profit)}, Margin={margin_str}"
            )

    # ── Quarterly Revenue ─────────────────────────────────
    revenue = financial_data.get("revenue", [])
    if revenue:
        lines.append("\nQuarterly Revenue (most recent 8 periods):")
        for r in revenue[-8:]:
            period = r.get("month", r.get("quarter", r.get("period", "?")))
            amount = r.get("amount", r.get("revenue", 0)) or 0
            def fmt_r(v: float) -> str:
                if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
                if abs(v) >= 1e6: return f"${v/1e6:.1f}M"
                return f"${v:,.0f}"
            lines.append(f"  {period}: {fmt_r(amount)}")

    # ── Any other keys ────────────────────────────────────
    known = {"kpis", "revenue"}
    extras = {k: v for k, v in financial_data.items() if k not in known and isinstance(v, list)}
    for key, values in extras.items():
        if values:
            lines.append(f"\n{key.replace('_', ' ').title()} ({len(values)} records):")
            for item in values[:3]:
                if isinstance(item, dict):
                    lines.append("  " + ", ".join(f"{k}={v}" for k, v in list(item.items())[:4]))

    return "\n".join(lines) if lines else "No financial data provided."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AGENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SupervisorAgent:
    """
    Production DeepAgent orchestrating the 5 financial sub-agents.

    Drop-in replacement for:
        def supervisor(question: str, financial_data: dict) -> dict

    Usage:
        agent  = SupervisorAgent()
        result = await agent.run(question, financial_data)

        # Drop-in compatibility:
        print(result.answer)           # same as original["answer"]
        print(result.details)          # same as original["details"]

        # New structured fields:
        print(result.agents_succeeded) # 5 if all worked
        print(result.agent_records)    # per-agent timing + status
        print(result.duration_ms)      # total pipeline time
    """

    def __init__(
        self,
        model:   str           = "gemini-2.5-flash",
        api_key: Optional[str] = None,
    ):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it in .env or pass api_key= to SupervisorAgent()."
            )

        self.model      = model
        self.api_key    = resolved_key
        self.agent_name = "SupervisorAgent"

        # Lazy-import sub-agents to avoid circular imports
        # They are instantiated in _build_sub_agents()
        self._sub_agents: Optional[dict] = None

    def _build_sub_agents(self) -> dict:
        """
        Build all 5 sub-agents lazily.
        Each uses the same Gemini model + API key.
        """
        if self._sub_agents is not None:
            return self._sub_agents

        # Import the actual DeepAgent classes
        # Adjust import paths to match your project structure
        try:
            from agents.data_agent    import DataAgent
            from agents.pattern_agent import PatternAgent
            from agents.forecast_agent import ForecastAgent
            from agents.insight_agent import InsightAgent
            from agents.report_agent  import ReportAgent
        except ImportError:
            # Fallback: use local stub agents if the real ones aren't installed
            logger.warning(
                "Sub-agent imports failed — using stub wrappers. "
                "Make sure agents/ directory is in your Python path."
            )
            DataAgent    = _StubAgent("DataAgent")
            PatternAgent = _StubAgent("PatternAgent")
            ForecastAgent = _StubAgent("ForecastAgent")
            InsightAgent = _StubAgent("InsightAgent")
            ReportAgent  = _StubAgent("ReportAgent")

        kwargs = {"model": self.model, "api_key": self.api_key}

        self._sub_agents = {
            "data":     DataAgent(**kwargs),
            "pattern":  PatternAgent(**kwargs),
            "forecast": ForecastAgent(**kwargs),
            "insight":  InsightAgent(**kwargs),
            "report":   ReportAgent(**kwargs),
        }
        return self._sub_agents

    # ──────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────

    async def run(
        self,
        question:       str,
        financial_data: dict,
    ) -> SupervisorResult:
        """
        Exact same signature as the original supervisor().
        Returns SupervisorResult — typed, with full audit trail.
        """
        t_start = time.perf_counter()

        # Pre-validate with Pydantic
        try:
            validated = SupervisorInput(
                question=question,
                financial_data=financial_data,
            )
        except Exception as e:
            return SupervisorResult(
                success=False,
                question=question,
                errors=[{"node": "pre_validation", "error": str(e)}],
                model_used=self.model,
                duration_ms=round((time.perf_counter() - t_start) * 1000, 1),
            )

        state = SupervisorState(
            question=validated.question,
            financial_data=validated.financial_data,
        )

        logger.info(f"🤖 [{self.agent_name}] Starting pipeline | model={self.model}")
        logger.info(f"   Q: {question[:80]}{'...' if len(question) > 80 else ''}")

        state       = await self._execute_graph(state)
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        result      = self._build_result(state, duration_ms)

        if result.success:
            logger.success(
                f"✅ [{self.agent_name}] {duration_ms:.0f}ms | "
                f"agents={result.agents_succeeded}/5 | "
                f"output={result.total_output_chars:,} chars"
            )
        else:
            logger.error(
                f"❌ [{self.agent_name}] {duration_ms:.0f}ms | errors={result.errors}"
            )

        return result

    # ──────────────────────────────────────────────────────────
    # GRAPH ENGINE
    # ──────────────────────────────────────────────────────────

    async def _execute_graph(self, state: SupervisorState) -> SupervisorState:
        graph: list[tuple[str, Callable, bool]] = [
            ("validate_inputs",    self._node_validate_inputs,    True),
            ("format_data",        self._node_format_data,        True),
            ("run_data_agent",     self._node_run_data_agent,     False),
            ("run_pattern_agent",  self._node_run_pattern_agent,  False),
            ("run_forecast_agent", self._node_run_forecast_agent, False),
            ("run_insight_agent",  self._node_run_insight_agent,  False),
            ("run_report_agent",   self._node_run_report_agent,   False),
            ("validate_outputs",   self._node_validate_outputs,   False),
            ("format_result",      self._node_format_result,      False),
        ]

        for node_name, node_fn, is_critical in graph:
            t = time.perf_counter()
            logger.debug(f"  ▶ Node [{node_name}]")
            try:
                state = await node_fn(state)
                ms    = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({"node": node_name, "status": "completed", "ms": ms})
                logger.debug(f"  ✅ [{node_name}] {ms}ms")
            except Exception as exc:
                ms  = round((time.perf_counter() - t) * 1000, 1)
                msg = f"{type(exc).__name__}: {exc}"
                state.node_history.append({"node": node_name, "status": "failed", "ms": ms, "error": msg})
                state.errors.append({"node": node_name, "error": msg})
                logger.error(f"  ❌ [{node_name}] FAILED: {msg}")
                if is_critical:
                    logger.error("  🛑 Critical node — aborting graph")
                    break

        return state

    # ──────────────────────────────────────────────────────────
    # NODE 1 — VALIDATE INPUTS
    # ──────────────────────────────────────────────────────────

    async def _node_validate_inputs(self, state: SupervisorState) -> SupervisorState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError(
                "Input validation failed:\n"
                + "\n".join(f"  • {e}" for e in state.input_errors)
            )
        kpis_n = len(state.financial_data.get("kpis", []))
        rev_n  = len(state.financial_data.get("revenue", []))
        logger.debug(f"    inputs valid | kpis={kpis_n} rev={rev_n} | q='{state.question[:50]}'")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 2 — FORMAT DATA  (pure Python, no LLM)
    # ──────────────────────────────────────────────────────────

    async def _node_format_data(self, state: SupervisorState) -> SupervisorState:
        """Convert the financial_data dict to a readable text summary."""
        state.db_summary = format_financial_data(state.financial_data)
        logger.debug(f"    db_summary: {len(state.db_summary)} chars")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 3–7 — SUB-AGENT RUNNERS
    # Each runs independently — one failure doesn't block the others.
    # ──────────────────────────────────────────────────────────

    async def _run_sub_agent(
        self,
        agent_name: str,
        coro,
    ) -> tuple[str, AgentRunRecord]:
        """
        Runs one sub-agent coroutine with isolation + timing.
        Returns (output_str, AgentRunRecord).
        On failure: returns ("", record with error) — does NOT raise.
        """
        t = time.perf_counter()
        try:
            result     = await coro
            ms         = round((time.perf_counter() - t) * 1000, 1)
            # Sub-agents return typed Result objects with .text or .narrative
            output_str = _extract_text(result, agent_name)
            record = AgentRunRecord(
                agent_name=agent_name,
                status="success",
                duration_ms=ms,
                output_len=len(output_str),
                retry_count=getattr(result, "retry_count", 0),
            )
            logger.info(f"  ✅ {agent_name} | {ms:.0f}ms | {len(output_str):,} chars")
            return output_str, record

        except Exception as exc:
            ms  = round((time.perf_counter() - t) * 1000, 1)
            msg = f"{type(exc).__name__}: {exc}"
            logger.error(f"  ❌ {agent_name} FAILED: {msg}")
            record = AgentRunRecord(
                agent_name=agent_name,
                status="failed",
                duration_ms=ms,
                error=msg,
            )
            return "", record

    async def _node_run_data_agent(self, state: SupervisorState) -> SupervisorState:
        logger.info("🔍 DataAgent: Analyzing question...")
        agents = self._build_sub_agents()
        output, record = await self._run_sub_agent(
            "DataAgent",
            agents["data"].run(state.question, state.db_summary),
        )
        state.data_analysis = output
        state.agent_records.append(record)
        return state

    async def _node_run_pattern_agent(self, state: SupervisorState) -> SupervisorState:
        logger.info("🧠 PatternAgent: Detecting patterns...")
        agents = self._build_sub_agents()
        output, record = await self._run_sub_agent(
            "PatternAgent",
            agents["pattern"].run(state.db_summary, state.question),
        )
        state.patterns = output
        state.agent_records.append(record)
        return state

    async def _node_run_forecast_agent(self, state: SupervisorState) -> SupervisorState:
        logger.info("🔮 ForecastAgent: Generating forecast...")
        agents = self._build_sub_agents()
        patterns_input = state.patterns or state.db_summary   # fallback if patterns failed
        output, record = await self._run_sub_agent(
            "ForecastAgent",
            agents["forecast"].run(state.db_summary, patterns_input, state.question),
        )
        state.forecast = output
        state.agent_records.append(record)
        return state

    async def _node_run_insight_agent(self, state: SupervisorState) -> SupervisorState:
        logger.info("💡 InsightAgent: Generating insights...")
        agents = self._build_sub_agents()
        # Fallbacks in case earlier agents failed
        patterns_input = state.patterns or "No pattern analysis available."
        forecast_input = state.forecast or "No forecast available."
        output, record = await self._run_sub_agent(
            "InsightAgent",
            agents["insight"].run(state.db_summary, patterns_input, forecast_input, state.question),
        )
        state.insights = output
        state.agent_records.append(record)
        return state

    async def _node_run_report_agent(self, state: SupervisorState) -> SupervisorState:
        logger.info("📣 ReportAgent: Writing final report...")
        agents = self._build_sub_agents()
        data_input     = state.data_analysis or "No data analysis available."
        patterns_input = state.patterns      or "No patterns available."
        forecast_input = state.forecast      or "No forecast available."
        insights_input = state.insights      or "No insights available."
        output, record = await self._run_sub_agent(
            "ReportAgent",
            agents["report"].run(
                state.question,
                data_input,
                patterns_input,
                forecast_input,
                insights_input,
            ),
        )
        state.final_report = output
        state.agent_records.append(record)
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 8 — VALIDATE OUTPUTS
    # ──────────────────────────────────────────────────────────

    async def _node_validate_outputs(self, state: SupervisorState) -> SupervisorState:
        state = validate_outputs(state)
        if not state.outputs_valid:
            logger.warning(f"    output validation issues: {state.outputs_errors}")
            # Non-critical — we still return whatever we have
        else:
            logger.debug("    all outputs valid ✓")
        return state

    # ──────────────────────────────────────────────────────────
    # NODE 9 — FORMAT RESULT
    # ──────────────────────────────────────────────────────────

    async def _node_format_result(self, state: SupervisorState) -> SupervisorState:
        """No LLM — _build_result() reads state directly."""
        logger.info("✅ SupervisorAgent: Pipeline complete!")
        return state

    # ──────────────────────────────────────────────────────────
    # RESULT BUILDER
    # ──────────────────────────────────────────────────────────

    def _build_result(self, state: SupervisorState, duration_ms: float) -> SupervisorResult:
        succeeded = sum(1 for r in state.agent_records if r.status == "success")
        failed    = sum(1 for r in state.agent_records if r.status == "failed")
        total_chars = sum(
            len(s) for s in [
                state.data_analysis, state.patterns,
                state.forecast, state.insights, state.final_report,
            ]
        )

        success = (
            state.input_valid
            and bool(state.final_report)
            and len(state.errors) == 0
            and succeeded >= 4   # allow 1 failure
        )

        return SupervisorResult(
            success=success,
            question=state.question,
            answer=state.final_report,
            details={
                "data_analysis": state.data_analysis,
                "patterns":      state.patterns,
                "forecast":      state.forecast,
                "insights":      state.insights,
            },
            agent_records=[
                {
                    "agent_name":  r.agent_name,
                    "status":      r.status,
                    "duration_ms": r.duration_ms,
                    "output_len":  r.output_len,
                    "error":       r.error,
                    "retry_count": r.retry_count,
                }
                for r in state.agent_records
            ],
            db_summary_len=len(state.db_summary),
            agents_succeeded=succeeded,
            agents_failed=failed,
            total_output_chars=total_chars,
            model_used=self.model,
            duration_ms=duration_ms,
            node_history=state.node_history,
            errors=state.errors,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_text(result: object, agent_name: str) -> str:
    """
    Extract the string output from a DeepAgent result object.
    Handles: .text, .narrative, .answer, and raw str.
    """
    if isinstance(result, str):
        return result

    # Try common result field names
    for attr in ("text", "narrative", "answer", "direct_answer"):
        val = getattr(result, attr, None)
        if val and isinstance(val, str) and len(val.strip()) > 5:
            return val.strip()

    # Fallback: convert to string
    logger.warning(f"    {agent_name}: could not extract text from {type(result).__name__}, using str()")
    return str(result)


class _StubAgent:
    """
    Stub sub-agent used when the real agent classes are not importable.
    Returns a minimal placeholder string.
    """
    def __init__(self, name: str):
        self.name = name

    def __call__(self, **kwargs):
        return self

    async def run(self, *args, **kwargs) -> str:
        return f"[{self.name} stub output — install the real agent class]"
    