"""
Forecast Agent - Financial forecasting specialist
New 6-node architecture with validators and OpenRouter
"""

import json
import re
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "mistralai/mistral-small-24b-instruct-2501"

_MAX_PLAUSIBLE_REVENUE = 1_000_000_000_000
_MAX_PLAUSIBLE_GROWTH_RATE = 5.00
_MIN_PLAUSIBLE_GROWTH_RATE = -0.90


@dataclass
class ForecastAgentState:
    data: str = ""
    patterns: str = ""
    question: str = ""
    max_retries: int = 2
    
    input_valid: bool = False
    input_errors: list = field(default_factory=list)
    
    snapshot: Optional[Dict] = None
    forecast: Optional[Dict] = None
    forecast_valid: bool = False
    forecast_errors: list = field(default_factory=list)
    retry_count: int = 0
    
    narrative: str = ""
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class ForecastAgentResult:
    success: bool = False
    short_term_period: str = ""
    short_term_revenue: float = 0
    short_term_expenses: float = 0
    short_term_profit: float = 0
    short_term_reasoning: str = ""
    annual_year: int = 0
    annual_revenue: float = 0
    annual_expenses: float = 0
    annual_profit: float = 0
    annual_margin: float = 0
    annual_reasoning: str = ""
    growth_rate: float = 0
    growth_rate_pct: str = ""
    growth_reasoning: str = ""
    risks: list = field(default_factory=list)
    confidence: str = ""
    confidence_explanation: str = ""
    narrative: str = ""
    model_used: str = ""
    duration_ms: float = 0
    retry_count: int = 0
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


_SYSTEM = """You are an expert financial forecasting specialist for Finora AI.
You analyse Salesforce financial data and produce precise, evidence-based forecasts.
Base every number on the actual data provided — never hallucinate figures.
Return ONLY valid JSON — no markdown fences, no explanation outside the JSON object.
"""

_PARSE_PROMPT = """
Parse the key financial numbers from this historical data.

Raw data:
{data}

Return EXACTLY this JSON:
{{
  "latest_revenue": 1234567.89,
  "latest_expenses": 987654.32,
  "latest_net_profit": 246913.57,
  "latest_profit_margin": 0.20,
  "avg_growth_rate": 0.12,
  "trend_direction": "growing|declining|stable|volatile",
  "data_period": "2021 Q1 to 2024 Q4",
  "currency": "USD",
  "num_periods": 16,
  "parsing_notes": "any caveats about data quality or gaps"
}}

Rules:
- All monetary values in the SAME currency unit
- profit_margin as decimal: 20% → 0.20
- avg_growth_rate as decimal: 12% → 0.12
"""

_FORECAST_PROMPT = """
Generate a structured financial forecast.

Parsed financial snapshot:
{snapshot}

Detected patterns:
{patterns}

User question: {question}

Return EXACTLY this JSON:
{{
  "short_term": {{
    "period": "Q2 2025",
    "revenue": 1300000.00,
    "expenses": 1040000.00,
    "net_profit": 260000.00,
    "reasoning": "one sentence grounded in the data"
  }},
  "annual": {{
    "year": 2025,
    "revenue": 5200000.00,
    "expenses": 4160000.00,
    "net_profit": 1040000.00,
    "profit_margin": 0.20,
    "reasoning": "one sentence grounded in the data"
  }},
  "growth_rate": 0.12,
  "growth_reasoning": "explains why this growth rate is expected",
  "risks": [
    "Risk 1 — specific and grounded",
    "Risk 2 — specific and grounded"
  ],
  "confidence": "High|Medium|Low",
  "confidence_explanation": "why this confidence level"
}}

Critical rules:
- Use EXACT numbers from the snapshot as the baseline
- growth_rate as decimal: 12% → 0.12
- profit_margin as decimal: 20% → 0.20
- annual.revenue must be ≈ 4× short_term.revenue
- Return ONLY valid JSON
"""

_NARRATIVE_PROMPT = """
Write a concise, professional financial forecast narrative.

Forecast data:
{forecast_json}

User asked: {question}

Write 3–4 sentences that:
1. State the short-term and annual forecasts with actual numbers
2. Explain the growth rate estimate and its key drivers
3. Name the top 2 risks
4. Give the confidence level and why

Return EXACTLY this JSON:
{{
  "narrative": "your 3-4 sentence paragraph"
}}
"""

_RETRY_ADDENDUM = """

IMPORTANT - Previous forecast had these validation errors:
{errors}

Fix ALL of these errors in your new response.
"""


def _llm_json(prompt: str, system: str = _SYSTEM) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000
    )
    raw = response.choices[0].message.content.strip()
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    
    return json.loads(raw)


def validate_inputs(state: ForecastAgentState) -> ForecastAgentState:
    """Node 1: Validate inputs"""
    errors = []
    
    data = (state.data or "").strip()
    if not data:
        errors.append("data is empty.")
    elif len(data) > 20000:
        errors.append("data is too long (max 20000 chars).")
    
    nums = re.findall(r"\d+(?:[.,]\d+)?", data)
    if len(nums) < 3:
        errors.append("data must contain at least 3 numeric values.")
    
    patterns = (state.patterns or "").strip()
    if not patterns:
        errors.append("patterns is empty.")
    
    question = (state.question or "").strip()
    if not question:
        errors.append("question is empty.")
    
    if errors:
        state.input_valid = False
        state.input_errors = errors
    else:
        state.input_valid = True
        state.input_errors = []
    
    return state


def validate_forecast(state: ForecastAgentState) -> ForecastAgentState:
    """Node 4: Validate forecast output"""
    errors = []
    fc = state.forecast
    
    if fc is None:
        state.forecast_valid = False
        state.forecast_errors = ["No forecast produced."]
        state.retry_count += 1
        return state
    
    st = fc.get("short_term", {})
    ann = fc.get("annual", {})
    
    if st.get("period"):
        if not re.match(r"^Q[1-4]\s+\d{4}$", st["period"].strip()):
            errors.append(f"Invalid short_term period format: {st.get('period')}")
    
    current_year = datetime.utcnow().year
    if ann.get("year"):
        if not (current_year <= ann["year"] <= current_year + 5):
            errors.append(f"Annual year {ann['year']} outside valid range.")
    
    for label, val in [
        ("short_term.revenue", st.get("revenue")),
        ("short_term.expenses", st.get("expenses")),
        ("annual.revenue", ann.get("revenue")),
        ("annual.expenses", ann.get("expenses")),
    ]:
        if val is not None and val < 0:
            errors.append(f"{label} is negative.")
        if val is not None and val > _MAX_PLAUSIBLE_REVENUE:
            errors.append(f"{label} exceeds sanity ceiling.")
    
    gr = fc.get("growth_rate")
    if gr is not None:
        if not (_MIN_PLAUSIBLE_GROWTH_RATE <= gr <= _MAX_PLAUSIBLE_GROWTH_RATE):
            errors.append(f"Growth rate {gr} outside plausible range.")
    
    if not fc.get("risks"):
        errors.append("No risks listed.")
    
    confidence = fc.get("confidence", "").lower()
    if confidence not in {"high", "medium", "low"}:
        errors.append(f"Invalid confidence level: {fc.get('confidence')}")
    
    q_rev = st.get("revenue")
    a_rev = ann.get("revenue")
    if q_rev and a_rev and a_rev < q_rev * 0.5:
        errors.append("Annual revenue less than half of quarterly - likely magnitude error.")
    
    if errors:
        state.forecast_valid = False
        state.forecast_errors = errors
        state.retry_count += 1
    else:
        state.forecast_valid = True
        state.forecast_errors = []
    
    return state


class ForecastAgent:
    """Forecast Agent - Financial forecasting specialist"""
    
    def __init__(self, max_retries: int = 2, temperature: float = 0.1):
        self.max_retries = max_retries
        self.temperature = temperature
        self.agent_name = "ForecastAgent"
    
    async def run(self, data: str, patterns: str, question: str) -> ForecastAgentResult:
        """Execute the 6-node forecast agent graph"""
        t_start = time.perf_counter()
        
        state = ForecastAgentState(
            data=data,
            patterns=patterns,
            question=question,
            max_retries=self.max_retries,
        )
        
        nodes = [
            ("validate_inputs", self._node_validate_inputs),
            ("parse_financials", self._node_parse_financials),
            ("generate_forecast", self._node_generate_forecast),
            ("validate_forecast", self._node_validate_forecast),
            ("enrich_narrative", self._node_enrich_narrative),
            ("format_output", self._node_format_output),
        ]
        
        for node_name, node_fn in nodes:
            t = time.perf_counter()
            try:
                state = await node_fn(state)
                ms = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({"node": node_name, "status": "completed", "ms": ms})
            except Exception as e:
                ms = round((time.perf_counter() - t) * 1000, 1)
                state.node_history.append({"node": node_name, "status": "failed", "ms": ms, "error": str(e)})
                state.errors.append({"node": node_name, "error": str(e)})
                break
        
        duration_ms = round((time.perf_counter() - t_start) * 1000, 1)
        
        return self._build_result(state, duration_ms)
    
    async def _node_validate_inputs(self, state: ForecastAgentState) -> ForecastAgentState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError("Input validation failed: " + " | ".join(state.input_errors))
        return state
    
    async def _node_parse_financials(self, state: ForecastAgentState) -> ForecastAgentState:
        prompt = _PARSE_PROMPT.format(data=state.data)
        state.snapshot = _llm_json(prompt)
        return state
    
    async def _node_generate_forecast(self, state: ForecastAgentState) -> ForecastAgentState:
        snapshot_str = json.dumps(state.snapshot) if state.snapshot else "{}"
        
        prompt = _FORECAST_PROMPT.format(
            snapshot=snapshot_str,
            patterns=state.patterns,
            question=state.question,
        )
        
        if state.retry_count > 0 and state.forecast_errors:
            prompt += _RETRY_ADDENDUM.format(errors="\n".join(f"  - {e}" for e in state.forecast_errors))
        
        state.forecast = _llm_json(prompt)
        return state
    
    async def _node_validate_forecast(self, state: ForecastAgentState) -> ForecastAgentState:
        state = validate_forecast(state)
        
        if not state.forecast_valid and state.retry_count <= state.max_retries:
            state = await self._node_generate_forecast(state)
            state = validate_forecast(state)
        
        return state
    
    async def _node_enrich_narrative(self, state: ForecastAgentState) -> ForecastAgentState:
        if not state.forecast:
            state.narrative = "Forecast could not be generated."
            return state
        
        prompt = _NARRATIVE_PROMPT.format(
            forecast_json=json.dumps(state.forecast),
            question=state.question,
        )
        
        try:
            result = _llm_json(prompt)
            state.narrative = result.get("narrative", "")
        except:
            state.narrative = "Narrative generation failed."
        
        return state
    
    async def _node_format_output(self, state: ForecastAgentState) -> ForecastAgentState:
        return state
    
    def _build_result(self, state: ForecastAgentState, duration_ms: float) -> ForecastAgentResult:
        success = state.input_valid and state.forecast_valid and len(state.errors) == 0
        
        result = ForecastAgentResult(
            success=success,
            model_used=MODEL,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
            narrative=state.narrative,
        )
        
        if state.forecast:
            fc = state.forecast
            st = fc.get("short_term", {})
            ann = fc.get("annual", {})
            
            result.short_term_period = st.get("period", "")
            result.short_term_revenue = st.get("revenue", 0)
            result.short_term_expenses = st.get("expenses", 0)
            result.short_term_profit = st.get("net_profit", 0)
            result.short_term_reasoning = st.get("reasoning", "")
            
            result.annual_year = ann.get("year", 0)
            result.annual_revenue = ann.get("revenue", 0)
            result.annual_expenses = ann.get("expenses", 0)
            result.annual_profit = ann.get("net_profit", 0)
            result.annual_margin = ann.get("profit_margin", 0)
            result.annual_reasoning = ann.get("reasoning", "")
            
            result.growth_rate = fc.get("growth_rate", 0)
            result.growth_rate_pct = f"{fc.get('growth_rate', 0) * 100:.1f}%"
            result.growth_reasoning = fc.get("growth_reasoning", "")
            result.risks = fc.get("risks", [])
            result.confidence = fc.get("confidence", "")
            result.confidence_explanation = fc.get("confidence_explanation", "")
        
        return result


def forecast_agent(data: str, patterns: str, question: str) -> str:
    """Synchronous wrapper for ForecastAgent"""
    import asyncio
    
    async def run():
        agent = ForecastAgent()
        result = await agent.run(data, patterns, question)
        return result
    
    result = asyncio.run(run())
    
    return json.dumps({
        "success": result.success,
        "short_term": {
            "period": result.short_term_period,
            "revenue": result.short_term_revenue,
            "expenses": result.short_term_expenses,
            "profit": result.short_term_profit,
            "reasoning": result.short_term_reasoning,
        },
        "annual": {
            "year": result.annual_year,
            "revenue": result.annual_revenue,
            "expenses": result.annual_expenses,
            "profit": result.annual_profit,
            "margin": result.annual_margin,
            "reasoning": result.annual_reasoning,
        },
        "growth_rate": result.growth_rate,
        "growth_rate_pct": result.growth_rate_pct,
        "growth_reasoning": result.growth_reasoning,
        "risks": result.risks,
        "confidence": result.confidence,
        "confidence_explanation": result.confidence_explanation,
        "narrative": result.narrative,
    })
