import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_OCR_KEY"))
 

SYSTEM_PROMPT = """
You are a financial document OCR agent.
Analyze the provided image (invoice or receipt) and extract the financial data.

You MUST respond with ONLY a valid JSON object — no explanation, no markdown, no backticks.
The JSON must have exactly these fields:

{
  "date": "YYYY-MM-DD",
  "amount": 0.00,
  "category": "string",
  "description": "string"
}

Rules:
- date: ISO format (YYYY-MM-DD). If not found, use today's date.
- amount: float number only. No currency symbols.
- category: one of these — Salaries, Software, Marketing, Operations, Travel, Utilities, Other
- description: short description of what the document is about (max 100 chars)

Respond with ONLY the JSON. Nothing else.
"""

def extract_document_data(image_bytes: bytes, mime_type: str) -> dict:
    image_part = {
        "mime_type": mime_type,
        "data": image_bytes
    }
    response = model.generate_content([SYSTEM_PROMPT, image_part])
    raw = response.text.strip()
    # Strip markdown fences if Gemini adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
