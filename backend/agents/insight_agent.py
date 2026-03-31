"""
Insight Agent - Strategic insight generation specialist
New 6-node architecture with validators and OpenRouter
"""

import json
import re
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "mistralai/mistral-small-24b-instruct-2501"


@dataclass
class InsightAgentState:
    data: str = ""
    patterns: str = ""
    forecast: str = ""
    question: str = ""
    max_retries: int = 2
    
    input_valid: bool = False
    input_errors: list = field(default_factory=list)
    
    context: Optional[Dict] = None
    insights: Optional[Dict] = None
    insights_valid: bool = False
    insights_errors: list = field(default_factory=list)
    retry_count: int = 0
    
    narrative: str = ""
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class InsightAgentResult:
    success: bool = False
    direct_answer: str = ""
    insights: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    key_risk: str = ""
    health_score: int = 0
    health_explanation: str = ""
    health_trend: str = ""
    executive_summary: str = ""
    narrative: str = ""
    model_used: str = ""
    duration_ms: float = 0
    retry_count: int = 0
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


_SYSTEM = """You are a senior financial advisor and strategic insight specialist for Finora AI.
You synthesize financial data, patterns, and forecasts into clear, actionable recommendations for CFOs.
Every insight must be grounded in specific evidence from the provided data.
Return ONLY valid JSON — no markdown fences, no explanation outside the JSON object.
"""

_CONTEXT_PROMPT = """
Extract and synthesize the key signals from these three financial sources.

Financial data:
{data}

Detected patterns:
{patterns}

Forecast summary:
{forecast}

Return EXACTLY this JSON:
{{
  "latest_revenue": 1500000.0,
  "latest_net_profit": 350000.0,
  "latest_profit_margin": 0.233,
  "revenue_trend": "growing|stable|declining|volatile",
  "data_period": "Q1 2023 to Q1 2024",
  "currency": "USD",
  "top_pattern": "single most important pattern in one sentence",
  "forecast_growth_rate": 0.08,
  "forecast_confidence": "High|Medium|Low",
  "overall_health_signal": "growing|stable|declining|volatile"
}}
"""

_INSIGHTS_PROMPT = """
Synthesize all findings into strategic insights for a CFO.

Context summary:
{context}

Original financial data:
{data}

Patterns:
{patterns}

Forecast:
{forecast}

User question: "{question}"

Return EXACTLY this JSON:
{{
  "direct_answer": "direct, specific answer to the user's question in 2-3 sentences using actual numbers",
  "insights": [
    {{
      "title": "Insight title in 5-8 words",
      "explanation": "2-3 sentence explanation of the insight",
      "evidence": "specific data point with numbers",
      "urgency": "immediate|short_term|long_term|monitor"
    }}
  ],
  "actions": [
    {{
      "action": "Specific action verb + what to do",
      "rationale": "Why this action, grounded in the data",
      "expected_impact": "Quantified or qualified expected outcome",
      "urgency": "immediate|short_term|long_term|monitor"
    }}
  ],
  "key_risk": "One specific risk in 1-2 sentences with data reference",
  "health_score": 7,
  "health_explanation": "2 sentence explanation of why this score",
  "health_trend": "improving|stable|declining|volatile",
  "executive_summary": "2-3 sentence CFO-level summary"
}}

Rules:
- direct_answer must reference actual numbers from the data
- Every insight must have specific evidence
- health_score: 1=critical, 5=average, 8=strong, 10=exceptional
- Return ONLY valid JSON
"""

_NARRATIVE_PROMPT = """
Write a polished, CFO-ready financial insight narrative.

Insights data:
{insights_json}

User asked: "{question}"

Write 4-5 sentences that:
1. Opens with the health score and overall position
2. States the top strategic insight with specific numbers
3. Names the top recommended action and its expected impact
4. Closes with the key risk to watch

Return EXACTLY this JSON:
{{
  "narrative": "your 4-5 sentence paragraph"
}}
"""

_RETRY_ADDENDUM = """

IMPORTANT - Previous response had these quality issues:
{errors}

Fix ALL of these in your new response.
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
    msg_content = response.choices[0].message.content
    if msg_content is None or msg_content.strip() == "":
        raise ValueError("Empty response from model")
    raw = msg_content.strip()
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    
    return json.loads(raw)


def validate_inputs(state: InsightAgentState) -> InsightAgentState:
    """Node 1: Validate inputs"""
    errors = []
    
    data = (state.data or "").strip()
    if not data:
        errors.append("data is empty.")
    elif len(data) > 20000:
        errors.append("data is too long.")
    
    nums = re.findall(r"\d+(?:[.,]\d+)?", data)
    if len(nums) < 3:
        errors.append("data must contain at least 3 numeric values.")
    
    patterns = (state.patterns or "").strip()
    if not patterns:
        errors.append("patterns is empty.")
    
    forecast = (state.forecast or "").strip()
    if not forecast:
        errors.append("forecast is empty.")
    
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


def validate_insights(state: InsightAgentState) -> InsightAgentState:
    """Node 4: Validate insights output"""
    errors = []
    out = state.insights
    
    if out is None:
        state.insights_valid = False
        state.insights_errors = ["No insights produced."]
        state.retry_count += 1
        return state
    
    if len(out.get("direct_answer", "").strip()) < 20:
        errors.append("direct_answer is too short.")
    
    if not out.get("insights"):
        errors.append("insights list is empty.")
    
    if not out.get("actions"):
        errors.append("actions list is empty.")
    
    hs = out.get("health_score")
    if hs is not None and not (1 <= hs <= 10):
        errors.append("health_score must be 1-10.")
    
    if len(out.get("key_risk", "").strip()) < 10:
        errors.append("key_risk is too short.")
    
    if len(out.get("executive_summary", "").strip()) < 30:
        errors.append("executive_summary is too short.")
    
    if errors:
        state.insights_valid = False
        state.insights_errors = errors
        state.retry_count += 1
    else:
        state.insights_valid = True
        state.insights_errors = []
    
    return state


class InsightAgent:
    """Insight Agent - Strategic insight generation specialist"""
    
    def __init__(self, max_retries: int = 2, temperature: float = 0.1):
        self.max_retries = max_retries
        self.temperature = temperature
        self.agent_name = "InsightAgent"
    
    async def run(self, data: str, patterns: str, forecast: str, question: str) -> InsightAgentResult:
        """Execute the 6-node insight agent graph"""
        t_start = time.perf_counter()
        
        state = InsightAgentState(
            data=data,
            patterns=patterns,
            forecast=forecast,
            question=question,
            max_retries=self.max_retries,
        )
        
        nodes = [
            ("validate_inputs", self._node_validate_inputs),
            ("build_context", self._node_build_context),
            ("generate_insights", self._node_generate_insights),
            ("validate_insights", self._node_validate_insights),
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
    
    async def _node_validate_inputs(self, state: InsightAgentState) -> InsightAgentState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError("Input validation failed: " + " | ".join(state.input_errors))
        return state
    
    async def _node_build_context(self, state: InsightAgentState) -> InsightAgentState:
        prompt = _CONTEXT_PROMPT.format(
            data=state.data,
            patterns=state.patterns,
            forecast=state.forecast,
        )
        state.context = _llm_json(prompt)
        return state
    
    async def _node_generate_insights(self, state: InsightAgentState) -> InsightAgentState:
        context_str = json.dumps(state.context) if state.context else "{}"
        
        prompt = _INSIGHTS_PROMPT.format(
            context=context_str,
            data=state.data,
            patterns=state.patterns,
            forecast=state.forecast,
            question=state.question,
        )
        
        if state.retry_count > 0 and state.insights_errors:
            prompt += _RETRY_ADDENDUM.format(errors="\n".join(f"  - {e}" for e in state.insights_errors))
        
        state.insights = _llm_json(prompt)
        return state
    
    async def _node_validate_insights(self, state: InsightAgentState) -> InsightAgentState:
        state = validate_insights(state)
        
        if not state.insights_valid and state.retry_count <= self.max_retries:
            state = await self._node_generate_insights(state)
            state = validate_insights(state)
        
        return state
    
    async def _node_enrich_narrative(self, state: InsightAgentState) -> InsightAgentState:
        if not state.insights:
            state.narrative = "Insights could not be generated."
            return state
        
        prompt = _NARRATIVE_PROMPT.format(
            insights_json=json.dumps(state.insights),
            question=state.question,
        )
        
        try:
            result = _llm_json(prompt)
            state.narrative = result.get("narrative", "")
        except:
            state.narrative = "Narrative generation failed."
        
        return state
    
    async def _node_format_output(self, state: InsightAgentState) -> InsightAgentState:
        return state
    
    def _build_result(self, state: InsightAgentState, duration_ms: float) -> InsightAgentResult:
        success = state.input_valid and state.insights_valid and len(state.errors) == 0
        
        result = InsightAgentResult(
            success=success,
            model_used=MODEL,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
            narrative=state.narrative,
        )
        
        if state.insights:
            ins = state.insights
            result.direct_answer = ins.get("direct_answer", "")
            result.insights = ins.get("insights", [])
            result.actions = ins.get("actions", [])
            result.key_risk = ins.get("key_risk", "")
            result.health_score = ins.get("health_score", 0)
            result.health_explanation = ins.get("health_explanation", "")
            result.health_trend = ins.get("health_trend", "")
            result.executive_summary = ins.get("executive_summary", "")
        
        return result


def insight_agent(data: str, patterns: str, forecast: str, question: str) -> str:
    """Synchronous wrapper for InsightAgent"""
    import asyncio
    
    async def run():
        agent = InsightAgent()
        result = await agent.run(data, patterns, forecast, question)
        return result
    
    result = asyncio.run(run())
    
    return json.dumps({
        "success": result.success,
        "direct_answer": result.direct_answer,
        "insights": result.insights,
        "actions": result.actions,
        "key_risk": result.key_risk,
        "health_score": result.health_score,
        "health_explanation": result.health_explanation,
        "health_trend": result.health_trend,
        "executive_summary": result.executive_summary,
        "narrative": result.narrative,
    })
