import json
import os
from datetime import datetime

STORAGE_DIR = os.path.join("backend", "langgraph", "store")
os.makedirs(STORAGE_DIR, exist_ok=True)


def persist_run(run_id: str, payload: dict) -> None:
    path = os.path.join(STORAGE_DIR, f"run_{run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_run(run_id: str) -> dict:
    path = os.path.join(STORAGE_DIR, f"run_{run_id}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def new_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
