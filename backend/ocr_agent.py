
from openai import OpenAI  


import os
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY"),
)


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

import base64

def extract_document_data(image_bytes: bytes, mime_type: str) -> dict:

    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    response = client.chat.completions.create(
        model="meta-llama/llama-3.2-11b-vision-instruct", 
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                    }
                ]
            }
        ]
    )
    
    msg_content = response.choices[0].message.content
    if msg_content is None or msg_content.strip() == "":
        raise ValueError("Cannot read image (model does not support image input)")
    
    raw = msg_content.strip()
    
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
        
    return json.loads(raw)
