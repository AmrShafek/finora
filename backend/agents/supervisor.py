"""
Supervisor Agent - Orchestrates all 5 sub-agents
New architecture with validators and OpenRouter
"""

import json
import re
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from agents.data_agent import data_agent, DataAgent
from agents.pattern_agent import pattern_agent, PatternAgent
from agents.forecast_agent import forecast_agent, ForecastAgent
from agents.insight_agent import insight_agent, InsightAgent
from agents.report_agent import report_agent, ReportAgent


def format_financial_data(financial_data: dict) -> str:
    """Convert financial_data dict to readable text summary"""
    lines = []
    
    kpis = financial_data.get("kpis", [])
    if kpis:
        lines.append("Annual KPIs (most recent 6 years):")
        for k in kpis[-6:]:
            year = k.get("year", "?")
            rev = k.get("revenue", k.get("total_revenue", 0)) or 0
            exp = k.get("expenses", k.get("total_expenses", 0)) or 0
            profit = k.get("profit", k.get("net_profit", 0)) or 0
            margin = k.get("margin", k.get("profit_margin", 0)) or 0
            
            def fmt(v):
                if v == 0:
                    return "$0"
                if abs(v) >= 1e9:
                    return f"${v/1e9:.2f}B"
                if abs(v) >= 1e6:
                    return f"${v/1e6:.1f}M"
                return f"${v:,.0f}"
            
            margin_str = f"{margin:.1f}%" if margin > 1 else f"{margin*100:.1f}%"
            lines.append(f"  {year}: Revenue={fmt(rev)}, Expenses={fmt(exp)}, Net Profit={fmt(profit)}, Margin={margin_str}")
    
    revenue = financial_data.get("revenue", [])
    if revenue:
        lines.append("\nQuarterly Revenue (most recent 8 periods):")
        for r in revenue[-8:]:
            period = r.get("month", r.get("quarter", r.get("period", "?")))
            amount = r.get("amount", r.get("revenue", 0)) or 0
            
            def fmt_r(v):
                if abs(v) >= 1e9:
                    return f"${v/1e9:.2f}B"
                if abs(v) >= 1e6:
                    return f"${v/1e6:.1f}M"
                return f"${v:,.0f}"
            lines.append(f"  {period}: {fmt_r(amount)}")
    
    known = {"kpis", "revenue"}
    extras = {k: v for k, v in financial_data.items() if k not in known and isinstance(v, list)}
    for key, values in extras.items():
        if values:
            lines.append(f"\n{key.replace('_', ' ').title()} ({len(values)} records):")
            for item in values[:3]:
                if isinstance(item, dict):
                    lines.append("  " + ", ".join(f"{k}={v}" for k, v in list(item.items())[:4]))
    
    return "\n".join(lines) if lines else "No financial data provided."


@dataclass
class SupervisorState:
    question: str = ""
    financial_data: dict = field(default_factory=dict)
    db_summary: str = ""
    
    input_valid: bool = False
    input_errors: list = field(default_factory=list)
    
    data_analysis: str = ""
    patterns: str = ""
    forecast: str = ""
    insights: str = ""
    final_report: str = ""
    
    outputs_valid: bool = False
    outputs_errors: list = field(default_factory=list)
    
    agent_records: list = field(default_factory=list)
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def validate_inputs(state: SupervisorState) -> SupervisorState:
    """Validate supervisor inputs"""
    errors = []
    
    q = (state.question or "").strip()
    if not q or len(q) < 3:
        errors.append("question too short.")
    elif len(q) > 2000:
        errors.append("question too long.")
    
    injection_patterns = [
        r"ignore\s+(previous|above|all|prior)",
        r"forget\s+(your\s+)?instructions",
        r"you\s+are\s+now",
        r"act\s+as",
        r"jailbreak",
        r"\bDAN\b",
        r"drop\s+table",
        r"delete\s+from",
        r"insert\s+into",
        r"union\s+select",
        r"--\s",
        r"/\*.*\*/",
        r"<script",
        r"javascript:",
        r"eval\s*\(",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, q, re.IGNORECASE):
            errors.append("Question contains disallowed pattern.")
            break
    
    financial_keywords = {
        "revenue", "profit", "expense", "growth", "margin", "forecast",
        "trend", "pattern", "insight", "performance", "financial", "kpi",
    }
    if q and not any(kw in q.lower() for kw in financial_keywords):
        errors.append("Question does not appear to be about financial data.")
    
    fd = state.financial_data or {}
    if not fd:
        errors.append("financial_data is empty.")
    else:
        kpis = fd.get("kpis", [])
        revenue = fd.get("revenue", [])
        if not kpis and not revenue:
            errors.append("financial_data has no 'kpis' or 'revenue' key.")
    
    if errors:
        state.input_valid = False
        state.input_errors = errors
    else:
        state.input_valid = True
        state.input_errors = []
    
    return state


def validate_outputs(state: SupervisorState) -> SupervisorState:
    """Validate all sub-agent outputs"""
    errors = []
    
    outputs = {
        "data_analysis": state.data_analysis,
        "patterns": state.patterns,
        "forecast": state.forecast,
        "insights": state.insights,
        "final_report": state.final_report,
    }
    
    for name, val in outputs.items():
        v = (val or "").strip()
        if not v:
            errors.append(f"{name}: output is empty.")
        elif len(v) < 20:
            errors.append(f"{name}: output too short.")
        elif v.lower().startswith(("error", "failed", "could not")):
            errors.append(f"{name}: appears to be an error message.")
    
    if state.final_report:
        report_words = len(state.final_report.split())
        if report_words > 600:
            errors.append("final_report exceeds expected word limit.")
        if report_words < 30:
            errors.append("final_report is too brief.")
    
    succeeded = sum(1 for rec in state.agent_records if rec.get("status") == "success")
    total = len(state.agent_records)
    if total > 0 and succeeded < max(1, total - 1):
        errors.append(f"Too many sub-agents failed: {total - succeeded}/{total}.")
    
    if errors:
        state.outputs_valid = False
        state.outputs_errors = errors
    else:
        state.outputs_valid = True
        state.outputs_errors = []
    
    return state


class SupervisorAgent:
    """Supervisor Agent - orchestrates all 5 sub-agents"""
    
    def __init__(self):
        self.agent_name = "SupervisorAgent"
        self.data_agent = DataAgent()
        self.pattern_agent = PatternAgent()
        self.forecast_agent = ForecastAgent()
        self.insight_agent = InsightAgent()
        self.report_agent = ReportAgent()
    
    async def run(self, question: str, financial_data: dict) -> dict:
        """Execute the full supervisor pipeline"""
        t_start = time.perf_counter()
        
        state = SupervisorState(
            question=question,
            financial_data=financial_data,
        )
        
        nodes = [
            ("validate_inputs", self._node_validate_inputs),
            ("format_data", self._node_format_data),
            ("run_data_agent", self._node_run_data_agent),
            ("run_pattern_agent", self._node_run_pattern_agent),
            ("run_forecast_agent", self._node_run_forecast_agent),
            ("run_insight_agent", self._node_run_insight_agent),
            ("run_report_agent", self._node_run_report_agent),
            ("validate_outputs", self._node_validate_outputs),
            ("format_result", self._node_format_result),
        ]
        
        for node_name, node_fn in nodes:
            t = time.perf_counter()
            try:
                state = await node_fn(state)
                ms = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({"node": node_name, "status": "completed", "ms": ms})
                
                # Stop on critical failures
                if node_name == "validate_inputs" and not state.input_valid:
                    break
            except Exception as e:
                ms = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({"node": node_name, "status": "failed", "ms": ms, "error": str(e)})
                state.errors.append({"node": node_name, "error": str(e)})
                # Stop on validation failures
                if "validation failed" in str(e).lower():
                    break
        
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        
        return self._build_result(state, duration_ms)
    
    async def _node_validate_inputs(self, state: SupervisorState) -> SupervisorState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError("Input validation failed: " + " | ".join(state.input_errors))
        return state
    
    async def _node_format_data(self, state: SupervisorState) -> SupervisorState:
        state.db_summary = format_financial_data(state.financial_data)
        return state
    
    async def _node_run_data_agent(self, state: SupervisorState) -> SupervisorState:
        print("[Data Agent] Analyzing question...")
        try:
            result = await self.data_agent.run(state.question, state.db_summary)
            state.data_analysis = json.dumps({
                "intent": result.intent,
                "time_period": result.time_period,
                "tables_needed": result.tables_needed,
                "filters": result.filters,
                "explanation": result.data_explanation,
            })
            state.agent_records.append({"agent": "data", "status": "success"})
        except Exception as e:
            state.data_analysis = f"Error: {str(e)}"
            state.agent_records.append({"agent": "data", "status": "failed", "error": str(e)})
        return state
    
    async def _node_run_pattern_agent(self, state: SupervisorState) -> SupervisorState:
        print("[Pattern Agent] Detecting patterns...")
        try:
            result = await self.pattern_agent.run(state.db_summary, state.question)
            state.patterns = json.dumps({
                "key_patterns": result.key_patterns,
                "anomalies": result.anomalies,
                "trend_direction": result.trend_direction,
                "most_important": result.most_important,
            })
            state.agent_records.append({"agent": "pattern", "status": "success"})
        except Exception as e:
            state.patterns = f"Error: {str(e)}"
            state.agent_records.append({"agent": "pattern", "status": "failed", "error": str(e)})
        return state
    
    async def _node_run_forecast_agent(self, state: SupervisorState) -> SupervisorState:
        print("[Forecast Agent] Generating forecast...")
        patterns_input = state.patterns or state.db_summary
        try:
            result = await self.forecast_agent.run(state.db_summary, patterns_input, state.question)
            state.forecast = json.dumps({
                "short_term": {
                    "period": result.short_term_period,
                    "revenue": result.short_term_revenue,
                    "profit": result.short_term_profit,
                },
                "annual": {
                    "year": result.annual_year,
                    "revenue": result.annual_revenue,
                    "profit": result.annual_profit,
                },
                "growth_rate": result.growth_rate,
                "confidence": result.confidence,
                "risks": result.risks,
                "narrative": result.narrative,
            })
            state.agent_records.append({"agent": "forecast", "status": "success"})
        except Exception as e:
            state.forecast = f"Error: {str(e)}"
            state.agent_records.append({"agent": "forecast", "status": "failed", "error": str(e)})
        return state
    
    async def _node_run_insight_agent(self, state: SupervisorState) -> SupervisorState:
        print("[Insight Agent] Generating insights...")
        patterns_input = state.patterns or "No patterns available."
        forecast_input = state.forecast or "No forecast available."
        try:
            result = await self.insight_agent.run(state.db_summary, patterns_input, forecast_input, state.question)
            state.insights = json.dumps({
                "direct_answer": result.direct_answer,
                "health_score": result.health_score,
                "health_trend": result.health_trend,
                "insights": result.insights,
                "actions": result.actions,
                "key_risk": result.key_risk,
                "executive_summary": result.executive_summary,
            })
            state.agent_records.append({"agent": "insight", "status": "success"})
        except Exception as e:
            state.insights = f"Error: {str(e)}"
            state.agent_records.append({"agent": "insight", "status": "failed", "error": str(e)})
        return state
    
    async def _node_run_report_agent(self, state: SupervisorState) -> SupervisorState:
        print("[Report Agent] Writing final report...")
        data_input = state.data_analysis or "No data analysis available."
        patterns_input = state.patterns or "No patterns available."
        forecast_input = state.forecast or "No forecast available."
        insights_input = state.insights or "No insights available."
        
        try:
            result = await self.report_agent.run(
                state.question,
                data_input,
                patterns_input,
                forecast_input,
                insights_input
            )
            state.final_report = result.text or result.direct_answer
            state.agent_records.append({"agent": "report", "status": "success"})
        except Exception as e:
            state.final_report = f"Error: {str(e)}"
            state.agent_records.append({"agent": "report", "status": "failed", "error": str(e)})
        return state
    
    async def _node_validate_outputs(self, state: SupervisorState) -> SupervisorState:
        state = validate_outputs(state)
        return state
    
    async def _node_format_result(self, state: SupervisorState) -> SupervisorState:
        print("[Supervisor] Analysis complete!")
        return state
    
    def _build_result(self, state: SupervisorState, duration_ms: float) -> dict:
        succeeded = sum(1 for r in state.agent_records if r.get("status") == "success")
        
        # If validation failed, return error message
        if not state.input_valid:
            error_msg = "I couldn't process your question: " + " | ".join(state.input_errors)
            return {
                "question": state.question,
                "answer": error_msg,
                "details": {},
                "agents_succeeded": 0,
                "duration_ms": duration_ms,
                "node_history": state.node_history,
                "errors": state.errors,
            }
        
        # Check if user asked for a report
        uses_report = "report" in state.question.lower()
        
        # Build answer based on whether report was requested
        if uses_report:
            answer = state.final_report or "I couldn't generate a report. Please try again."
        else:
            # Summarized response - extract key parts
            answer = self._create_summary(state)
        
        return {
            "question": state.question,
            "answer": answer,
            "details": {
                "data_analysis": state.data_analysis,
                "patterns": state.patterns,
                "forecast": state.forecast,
                "insights": state.insights,
            },
            "agents_succeeded": succeeded,
            "duration_ms": duration_ms,
            "node_history": state.node_history,
            "errors": state.errors,
        }
    
    def _create_summary(self, state: SupervisorState) -> str:
        """Create a concise summary when report is not requested"""
        parts = []
        
        # Parse and format data analysis
        if state.data_analysis:
            da = state.data_analysis.strip()
            if da.startswith("{"):
                try:
                    da_obj = json.loads(da)
                    if da_obj.get("direct_answer"):
                        parts.append(da_obj["direct_answer"][:300])
                    elif da_obj.get("answer"):
                        parts.append(da_obj["answer"][:300])
                except:
                    da_lines = da.split('\n')[:3]
                    for line in da_lines:
                        if line.strip():
                            parts.append(line.strip()[:200])
            else:
                # Plain text response
                parts.append(da[:400])
        
        # Parse and format patterns
        if state.patterns:
            pat_str = state.patterns.strip()
            if pat_str.startswith("{"):
                try:
                    pat = json.loads(pat_str)
                    if pat.get("most_important"):
                        parts.append(f"Answer: {pat['most_important'][:200]}")
                    patterns = pat.get("key_patterns", [])
                    if patterns:
                        for p in patterns[:2]:
                            parts.append(f"Detail: {p[:150]}")
                    anomalies = pat.get("anomalies", [])
                    if anomalies:
                        for a in anomalies[:1]:
                            parts.append(f"Anomaly: {a[:150]}")
                except:
                    parts.append(f"Patterns: {pat_str[:300]}")
            else:
                # Plain text response
                parts.append(f"Analysis: {pat_str[:400]}")
        
        # Parse and format forecast
        if state.forecast:
            fc_str = state.forecast.strip()
            if fc_str.startswith("{"):
                try:
                    fc = json.loads(fc_str)
                    st = fc.get("short_term", {})
                    if st.get("period"):
                        parts.append(f"Forecast: {st.get('period')} - Revenue ${st.get('revenue', 0):,.0f}, Profit ${st.get('profit', 0):,.0f}")
                except:
                    parts.append(f"Forecast: {fc_str[:300]}")
            else:
                parts.append(f"Forecast: {fc_str[:400]}")
        
        # Parse and format insights
        if state.insights:
            ins_str = state.insights.strip()
            if ins_str.startswith("{"):
                try:
                    ins = json.loads(ins_str)
                    key_insights = ins.get("key_insights", [])
                    if key_insights:
                        for i in key_insights[:2]:
                            parts.append(f"Insight: {i[:150]}")
                except:
                    parts.append(f"Insights: {ins_str[:300]}")
            else:
                parts.append(f"Insights: {ins_str[:400]}")
        
        return "\n\n".join(parts)[:600] if parts else "Analysis complete. Ask for a detailed report for more information."


def supervisor(question: str, financial_data: dict) -> dict:
    """Synchronous wrapper for SupervisorAgent"""
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        # We're already in an async context
        async def run():
            agent = SupervisorAgent()
            result = await agent.run(question, financial_data)
            return result
        # Create a new task - this won't work for sync wrapper
        # So we need to handle this differently
        raise RuntimeError("Cannot run sync supervisor from async context")
    else:
        async def run():
            agent = SupervisorAgent()
            result = await agent.run(question, financial_data)
            return result
        
        return asyncio.run(run())
