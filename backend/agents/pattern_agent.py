"""
Pattern Agent - Financial pattern detection specialist
New architecture with validators and OpenRouter
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
class PatternAgentState:
    data: str = ""
    question: str = ""
    max_retries: int = 2
    
    input_valid: bool = False
    input_errors: list = field(default_factory=list)
    
    patterns: Optional[Dict] = None
    patterns_valid: bool = False
    patterns_errors: list = field(default_factory=list)
    retry_count: int = 0
    
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class PatternAgentResult:
    success: bool = False
    key_patterns: list = field(default_factory=list)
    anomalies: list = field(default_factory=list)
    trend_direction: str = ""
    most_important: str = ""
    model_used: str = ""
    duration_ms: float = 0
    retry_count: int = 0
    node_history: list = field(default_factory=list)
    errors: list = field(default_factory=list)


_SYSTEM = """You are a financial Pattern Detection Agent for Finora AI.
You analyze financial data and detect patterns, anomalies, and trends.
IMPORTANT: Always answer the user's specific question with relevant data points.
Return ONLY valid JSON — no markdown, no explanation outside JSON.
"""

_PATTERN_PROMPT = """
Analyze financial data to answer this specific question: "{question}"

Financial data to analyze:
{data}

Focus on:
1. What does the data show about the question?
2. Use specific numbers and percentages from the data
3. Identify any anomalies or unusual changes

Return EXACTLY this JSON:
{{
  "key_patterns": [
    "Answer point 1 with specific numbers addressing the question",
    "Answer point 2 with specific numbers",
    "Answer point 3 (optional)"
  ],
  "anomalies": [
    "Anomaly 1 - specific unusual movement related to the question",
    "Anomaly 2 (optional)"
  ],
  "trend_direction": "growing|declining|stable|volatile",
  "most_important": "One sentence answering the user's question directly",
  "seasonal_patterns": "Description of any seasonal patterns found",
  "year_over_year": "Year-over-year comparison summary"
}}

Rules:
- Answer the USER'S QUESTION directly with specific data
- Include actual figures from the data
- Return ONLY valid JSON
"""

_RETRY_ADDENDUM = """

IMPORTANT - Your previous response had these validation errors:
{errors}

Fix these errors in your response.
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


def validate_inputs(state: PatternAgentState) -> PatternAgentState:
    """Node 1: Validate inputs"""
    errors = []
    
    data = (state.data or "").strip()
    if not data:
        errors.append("data is empty.")
    elif len(data) < 10:
        errors.append("data is too short.")
    elif len(data) > 50000:
        errors.append("data is too long (max 50000 chars).")
    
    nums = re.findall(r"\d+(?:[.,]\d+)?", data)
    if len(nums) < 3:
        errors.append("data must contain at least 3 numeric values.")
    
    question = (state.question or "").strip()
    if not question:
        errors.append("question is empty.")
    
    injection_patterns = [
        r"ignore\s+(previous|above|all|prior)",
        r"forget\s+(your\s+)?instructions",
        r"jailbreak",
    ]
    
    if question:
        for pattern in injection_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                errors.append("Question contains disallowed pattern.")
                break
    
    if errors:
        state.input_valid = False
        state.input_errors = errors
    else:
        state.input_valid = True
        state.input_errors = []
    
    return state


def validate_patterns(state: PatternAgentState) -> PatternAgentState:
    """Node 4: Validate pattern output"""
    errors = []
    patterns = state.patterns
    
    if patterns is None:
        state.patterns_valid = False
        state.patterns_errors = ["No patterns produced."]
        return state
    
    if not patterns.get("key_patterns"):
        errors.append("No key patterns found.")
    
    if not patterns.get("trend_direction"):
        errors.append("No trend direction specified.")
    
    if not patterns.get("most_important"):
        errors.append("No most important finding specified.")
    
    valid_directions = {"growing", "declining", "stable", "volatile"}
    if patterns.get("trend_direction", "").lower() not in valid_directions:
        errors.append(f"Invalid trend direction: {patterns.get('trend_direction')}")
    
    if errors:
        state.patterns_valid = False
        state.patterns_errors = errors
        state.retry_count += 1
    else:
        state.patterns_valid = True
        state.patterns_errors = []
    
    return state


class PatternAgent:
    """Pattern Agent - Financial pattern detection specialist"""
    
    def __init__(self, max_retries: int = 2, temperature: float = 0.0):
        self.max_retries = max_retries
        self.temperature = temperature
        self.agent_name = "PatternAgent"
    
    async def run(self, data: str, question: str) -> PatternAgentResult:
        """Execute the pattern agent graph"""
        t_start = time.perf_counter()
        
        state = PatternAgentState(
            data=data,
            question=question,
            max_retries=self.max_retries,
        )
        
        nodes = [
            ("validate_inputs", self._node_validate_inputs),
            ("detect_patterns", self._node_detect_patterns),
            ("validate_patterns", self._node_validate_patterns),
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
    
    async def _node_validate_inputs(self, state: PatternAgentState) -> PatternAgentState:
        state = validate_inputs(state)
        if not state.input_valid:
            raise ValueError("Input validation failed: " + " | ".join(state.input_errors))
        return state
    
    async def _node_detect_patterns(self, state: PatternAgentState) -> PatternAgentState:
        prompt = _PATTERN_PROMPT.format(
            data=state.data,
            question=state.question,
        )
        
        if state.retry_count > 0 and state.patterns_errors:
            prompt += _RETRY_ADDENDUM.format(errors="\n".join(f"  - {e}" for e in state.patterns_errors))
        
        state.patterns = _llm_json(prompt)
        return state
    
    async def _node_validate_patterns(self, state: PatternAgentState) -> PatternAgentState:
        state = validate_patterns(state)
        
        if not state.patterns_valid and state.retry_count <= state.max_retries:
            state = await self._node_detect_patterns(state)
            state = validate_patterns(state)
        
        return state
    
    async def _node_format_output(self, state: PatternAgentState) -> PatternAgentState:
        return state
    
    def _build_result(self, state: PatternAgentState, duration_ms: float) -> PatternAgentResult:
        success = state.input_valid and state.patterns_valid and len(state.errors) == 0
        
        result = PatternAgentResult(
            success=success,
            model_used=MODEL,
            duration_ms=duration_ms,
            retry_count=state.retry_count,
            node_history=state.node_history,
            errors=state.errors,
        )
        
        if state.patterns:
            result.key_patterns = state.patterns.get("key_patterns", [])
            result.anomalies = state.patterns.get("anomalies", [])
            result.trend_direction = state.patterns.get("trend_direction", "")
            result.most_important = state.patterns.get("most_important", "")
        
        return result


def pattern_agent(data: str, question: str) -> str:
    """Synchronous wrapper for PatternAgent"""
    import asyncio
    
    async def run():
        agent = PatternAgent()
        result = await agent.run(data, question)
        return result
    
    result = asyncio.run(run())
    
    return json.dumps({
        "success": result.success,
        "key_patterns": result.key_patterns,
        "anomalies": result.anomalies,
        "trend_direction": result.trend_direction,
        "most_important": result.most_important,
    })
