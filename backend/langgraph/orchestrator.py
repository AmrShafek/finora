"""
LangGraph Orchestrator - Uses the new multi-agent architecture
"""

import json
import asyncio
from typing import Any, Dict
from .store import new_run_id, persist_run
from agents.supervisor import SupervisorAgent


class LangGraph:
    def __init__(self):
        self.run_id = new_run_id()
        self.supervisor = SupervisorAgent()

    def run(self, question: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full supervisor pipeline with the new architecture"""
        try:
            # Run async supervisor in sync context
            result = asyncio.run(self.supervisor.run(question, financial_data))
            
            payload = {
                "run_id": self.run_id,
                "question": question,
                "result": result,
            }
            persist_run(self.run_id, payload)
            
            return {
                "run_id": self.run_id,
                "status": "completed",
                "question": result.get("question"),
                "answer": result.get("answer"),
                "details": result.get("details"),
                "agents_succeeded": result.get("agents_succeeded"),
                "duration_ms": result.get("duration_ms"),
            }
        except Exception as e:
            import traceback
            return {
                "run_id": self.run_id,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

    def run_ocr(self, image_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
        """Run OCR agent"""
        from .adapters.agent_adapter import run_agent
        
        node_inputs = {"image_bytes": image_bytes, "mime_type": mime_type}
        node_output = run_agent("ocr", node_inputs)
        
        try:
            parsed = json.loads(node_output)
        except json.JSONDecodeError:
            return {"run_id": self.run_id, "status": "failed", "node": "ocr", "issues": ["Failed to parse OCR output as JSON"]}

        payload = {
            "run_id": self.run_id,
            "node": "ocr",
            "output": parsed,
        }
        persist_run(self.run_id, payload)

        return {"run_id": self.run_id, "status": "completed", "result": parsed}
