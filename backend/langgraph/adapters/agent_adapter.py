"""Adapters to call existing Finora agents from LangGraph."""

import json
from typing import Any, Dict, Union

# Import existing agent entry points
from agents.data_agent import data_agent
from agents.forecast_agent import forecast_agent
from agents.insight_agent import insight_agent
from agents.pattern_agent import pattern_agent
from agents.report_agent import report_agent
from agents.supervisor import supervisor
from ocr_agent import extract_document_data


class AgentAdapter:
    def __init__(self):
        pass

    def call(self, agent_type: str, inputs: Dict[str, Any]) -> str:
        if agent_type == "data":
            return data_agent(inputs.get("question", "") or "", inputs.get("db_summary", "") or "")  # type: ignore
        if agent_type == "forecast":
            return forecast_agent(inputs.get("data", "") or "", inputs.get("patterns", "") or "", inputs.get("question", "") or "")  # type: ignore
        if agent_type == "insight":
            return insight_agent(inputs.get("data", "") or "", inputs.get("patterns", "") or "", inputs.get("forecast", "") or "", inputs.get("question", "") or "")  # type: ignore
        if agent_type == "pattern":
            return pattern_agent(inputs.get("data", "") or "", inputs.get("question", "") or "")  # type: ignore
        if agent_type == "report":
            return report_agent(inputs.get("question", "") or "", inputs.get("data_analysis", "") or "", inputs.get("patterns", "") or "", inputs.get("forecast", "") or "", inputs.get("insights", "") or "")  # type: ignore
        if agent_type == "supervisor":
            return str(supervisor(inputs.get("question", "") or "", inputs.get("financial_data", {}) or {}))  # type: ignore
        if agent_type == "ocr":
            image_bytes = inputs.get("image_bytes")
            mime_type = inputs.get("mime_type", "image/jpeg")
            if not image_bytes:
                raise ValueError("OCR agent requires image_bytes in inputs")
            result = extract_document_data(image_bytes, mime_type)
            return json.dumps(result)
        raise ValueError(f"Unknown agent_type: {agent_type}")


def run_agent(agent_type: str, inputs: Dict[str, Any]) -> str:
    ad = AgentAdapter()
    return ad.call(agent_type, inputs)
