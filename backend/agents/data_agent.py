"""
Data Agent - Financial data planning specialist
New 5-node architecture with LangGraph and OpenRouter
"""

import json
import re
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "mistralai/mistral-small-24b-instruct-2501"


class IntentType(str, Enum):
    TREND = "trend"
    COMPARISON = "comparison"
    SNAPSHOT = "snapshot"
    FORECAST = "forecast"
    ANOMALY = "anomaly"
    RANKING = "ranking"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


class GranularityType(str, Enum):
    ANNUAL = "annual"
    QUARTERLY = "quarterly"
    MONTHLY = "monthly"


VALID_TABLES = {"revenue", "expenses", "kpis", "ai_insights"}
VALID_COLUMNS = {
    "revenue": {"year", "quarter", "amount", "region", "product_line"},
    "expenses": {"year", "quarter", "amount", "category", "department"},
    "kpis": {"year", "revenue", "expenses", "net_profit", "profit_margin", "growth_rate"},
    "ai_insights": {"year", "quarter", "insight_text", "category", "confidence_score"},
}


@dataclass
class DataAgentState:
    question: str = ""
    db_summary: str = ""
    max_retries: int = 2
    
    input_valid: bool = False
    input_errors: list = field(default_factory=list)
    clean_question: str = ""
    
    intent: Optional[Dict] = None
    data_plan: Optional[Dict] = None
    plan_valid: bool = False
    plan_errors: list = field(default_factory=list)
    retry_count: int = 0
    
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class DataAgentResult:
    success: bool = False
    intent: Optional[str] = None
    time_period: Optional[str] = None
    year_start: Optional[int] = None
    year_end: Optional[int] = None
    quarter: Optional[str] = None
    granularity: Optional[str] = None
    key_metrics: list = field(default_factory=list)
    tables_needed: list = field(default_factory=list)
    columns_needed: Dict = field(default_factory=dict)
    filters: Dict = field(default_factory=dict)
    aggregations: list = field(default_factory=list)
    data_explanation: str = ""
    model_used: str = ""
    duration_ms: float = 0
    retry_count: int = 0
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


_SYSTEM = """You are a financial data planning specialist for Finora AI.
You analyze financial questions and produce precise data retrieval plans.
You only answer about Salesforce financial data (2011–2026).
Return ONLY valid JSON — no markdown, no explanation outside JSON.
"""

_INTENT_PROMPT = """
Analyze this financial question and return a JSON intent classification.

Question: "{question}"

Available database:
{db_summary}

Return EXACTLY this JSON:
{{
  "intent": "trend|comparison|snapshot|forecast|anomaly|ranking|summary|unknown",
  "confidence": 0.95,
  "time_period": "natural language description e.g. '2023' or 'Q1-Q2 2024' or '2020-2024'",
  "year_start": 2020,
  "year_end": 2024,
  "quarter": "Q1 or null",
  "granularity": "annual|quarterly|monthly",
  "key_metrics": ["revenue", "net_profit"],
  "reasoning": "one sentence why you chose this intent"
}}

Rules:
- year_start and year_end must be integers between 2011 and 2026
- If no specific year is mentioned, use the full range: year_start=2011, year_end=2026
- key_metrics must only reference: revenue, expenses, net_profit, profit_margin, growth_rate, amount
- intent "unknown" only if the question is completely unrelated to finance
"""

_DATA_PLAN_PROMPT = """
Build a precise data retrieval plan for this financial question.

Question: "{question}"
Detected intent: {intent}
Time period: {time_period}
Key metrics: {key_metrics}
Year range: {year_start} to {year_end}

Available tables and columns:
- revenue:     year, quarter, amount, region, product_line
- expenses:    year, quarter, amount, category, department
- kpis:        year, revenue, expenses, net_profit, profit_margin, growth_rate
- ai_insights: year, quarter, insight_text, category, confidence_score

Return EXACTLY this JSON:
{{
  "tables_needed": ["kpis", "revenue"],
  "columns_needed": {{
    "kpis": ["year", "net_profit", "profit_margin"],
    "revenue": ["year", "quarter", "amount"]
  }},
  "filters": {{
    "year": "BETWEEN 2020 AND 2024",
    "quarter": "Q1"
  }},
  "aggregations": ["SUM(revenue.amount)", "AVG(kpis.profit_margin)"],
  "order_by": "year ASC",
  "limit": null,
  "explanation": "one sentence describing what data is being fetched and why"
}}

Rules:
- Only include tables actually needed to answer the question
- columns_needed must contain ONLY columns that exist in the table
- filters values are SQL fragments (e.g. "BETWEEN 2020 AND 2024", "= 'Q1'")
- aggregations can be empty list [] if raw rows are sufficient
- Return ONLY valid JSON
"""

_RETRY_ADDENDUM = """

IMPORTANT — Your previous attempt had these validation errors:
{errors}

Fix these errors in your response. The corrected JSON must pass all validation checks.
"""


def _llm_json(prompt: str, system: str = _SYSTEM) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    raw = response.choices[0].message.content.strip()
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    
    return json.loads(raw)


def validate_input_state(state: "DataAgentState") -> "DataAgentState":
    """Node 1: Validate and sanitize input"""
    errors = []
    question = (state.question or "").strip()
    
    if not question:
        errors.append("Question is empty.")
    elif len(question) < 5:
        errors.append(f"Question is too short ({len(question)} chars).")
    elif len(question) > 2000:
        errors.append(f"Question is too long ({len(question)} chars, max 2000).")
    
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
    ]
    
    if question:
        for pattern in injection_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                errors.append("Question contains disallowed pattern.")
                break
    
    financial_keywords = {
        "revenue", "expense", "profit", "margin", "kpi", "cost", "income",
        "loss", "growth", "quarter", "annual", "year", "q1", "q2", "q3", "q4",
        "salesforce", "finora", "financial", "budget", "forecast", "trend",
    }
    
    if question and not errors:
        if not any(kw in question.lower() for kw in financial_keywords):
            errors.append("Question does not appear to be about financial data.")
    
    if not (state.db_summary or "").strip():
        errors.append("db_summary is empty.")
    
    if errors:
        state.input_valid = False
        state.input_errors = errors
    else:
        state.input_valid = True
        state.input_errors = []
        state.clean_question = re.sub(r"\s+", " ", question).strip()
    
    return state


def validate_plan_state(state: "DataAgentState") -> "DataAgentState":
    """Node 4: Validate the data plan"""
    errors = []
    plan = state.data_plan
    
    if plan is None:
        state.plan_valid = False
        state.plan_errors = ["No data plan was produced."]
        return state
    
    if not plan.get("tables_needed"):
        errors.append("No tables specified in data plan.")
    else:
        unknown_tables = set(plan["tables_needed"]) - VALID_TABLES
        if unknown_tables:
            errors.append(f"Unknown table(s): {sorted(unknown_tables)}")
    
    for table, cols in plan.get("columns_needed", {}).items():
        if table in VALID_COLUMNS:
            valid_cols = VALID_COLUMNS[table] | {"*"}
            unknown_cols = set(cols) - valid_cols
            if unknown_cols:
                errors.append(f"Unknown column(s) in '{table}': {sorted(unknown_cols)}")
    
    intent = state.intent
    if intent:
        for label, yr in [("year_start", intent.get("year_start")), ("year_end", intent.get("year_end"))]:
            if yr is not None and not (2011 <= yr <= 2026):
                errors.append(f"{label}={yr} is outside data range (2011-2026).")
        
        if intent.get("year_start") and intent.get("year_end"):
            if intent["year_start"] > intent["year_end"]:
                errors.append("year_start is after year_end.")
    
    if intent and intent.get("quarter"):
        if intent["quarter"] not in {"Q1", "Q2", "Q3", "Q4"}:
            errors.append(f"Invalid quarter: {intent['quarter']}")
    
    if errors:
        state.plan_valid = False
        state.plan_errors = errors
        state.retry_count += 1
    else:
        state.plan_valid = True
        state.plan_errors = []
    
    return state


class DataAgent:
    """
    Data Agent - Financial data planning specialist
    Uses OpenRouter (mistral) instead of Gemini
    """
    
    def __init__(self, max_retries: int = 2, temperature: float = 0.0):
        self.max_retries = max_retries
        self.temperature = temperature
        self.agent_name = "DataAgent"
    
    async def run(self, question: str, db_summary: str) -> DataAgentResult:
        """Execute the 5-node data agent graph"""
        t_start = time.perf_counter()
        
        state = DataAgentState(
            question=question,
            db_summary=db_summary,
            max_retries=self.max_retries,
        )
        
        nodes = [
            ("validate_input", self._node_validate_input),
            ("classify_intent", self._node_classify_intent),
            ("build_data_plan", self._node_build_data_plan),
            ("validate_plan", self._node_validate_plan),
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
    
    async def _node_validate_input(self, state: DataAgentState) -> DataAgentState:
        state = validate_input_state(state)
        if not state.input_valid:
            raise ValueError("Input validation failed: " + " | ".join(state.input_errors))
        return state
    
    async def _node_classify_intent(self, state: DataAgentState) -> DataAgentState:
        prompt = _INTENT_PROMPT.format(
            question=state.clean_question,
            db_summary=state.db_summary,
        )
        state.intent = _llm_json(prompt)
        return state
    
    async def _node_build_data_plan(self, state: DataAgentState) -> DataAgentState:
        intent = state.intent or {}
        prompt = _DATA_PLAN_PROMPT.format(
            question=state.clean_question,
            intent=intent.get("intent", "unknown"),
            time_period=intent.get("time_period", "unspecified"),
            key_metrics=intent.get("key_metrics", []),
            year_start=intent.get("year_start", 2011),
            year_end=intent.get("year_end", 2026),
        )
        
        if state.retry_count > 0 and state.plan_errors:
            prompt += _RETRY_ADDENDUM.format(errors="\n".join(f"  - {e}" for e in state.plan_errors))
        
        state.data_plan = _llm_json(prompt)
        return state
    
    async def _node_validate_plan(self, state: DataAgentState) -> DataAgentState:
        state = validate_plan_state(state)
        
        if not state.plan_valid and state.retry_count <= state.max_retries:
            state = await self._node_build_data_plan(state)
            state = validate_plan_state(state)
        
        return state
    
    async def _node_format_output(self, state: DataAgentState) -> DataAgentState:
        return state
    
    def _build_result(self, state: DataAgentState, duration_ms: float) -> DataAgentResult:
        success = state.input_valid and state.plan_valid and len(state.errors) == 0
        
        result = DataAgentResult(
            success=success,
            model_used=MODEL,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
        )
        
        if state.intent:
            result.intent = state.intent.get("intent")
            result.time_period = state.intent.get("time_period")
            result.year_start = state.intent.get("year_start")
            result.year_end = state.intent.get("year_end")
            result.quarter = state.intent.get("quarter")
            result.granularity = state.intent.get("granularity")
            result.key_metrics = state.intent.get("key_metrics", [])
        
        if state.data_plan:
            result.tables_needed = state.data_plan.get("tables_needed", [])
            result.columns_needed = state.data_plan.get("columns_needed", {})
            result.filters = state.data_plan.get("filters", {})
            result.aggregations = state.data_plan.get("aggregations", [])
            result.data_explanation = state.data_plan.get("explanation", "")
        
        return result


def data_agent(question: str, db_summary: str) -> str:
    """
    Synchronous wrapper for DataAgent
    Returns JSON string with the data plan
    """
    import asyncio
    
    async def run():
        agent = DataAgent()
        result = await agent.run(question, db_summary)
        return result
    
    result = asyncio.run(run())
    
    return json.dumps({
        "success": result.success,
        "intent": result.intent,
        "time_period": result.time_period,
        "year_start": result.year_start,
        "year_end": result.year_end,
        "tables_needed": result.tables_needed,
        "columns_needed": result.columns_needed,
        "filters": result.filters,
        "aggregations": result.aggregations,
        "explanation": result.data_explanation,
    })
