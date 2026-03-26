import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(" gemini-2.5-flash")

def pattern_agent(data: str, question: str) -> str:
    """
    Pattern Agent — detects anomalies, trends, and unusual
    patterns in the financial data.
    """
    prompt = f"""
You are a financial Pattern Detection Agent for Finora AI.
Your job is to analyze financial data and detect:
- Unusual spikes or drops in revenue or expenses
- Growth trends (accelerating or decelerating)
- Seasonal patterns
- Anomalies that need attention
- Year-over-year comparisons

Financial data to analyze:
{data}

User question context: {question}

Provide:
1. Key patterns you detected
2. Any anomalies or unusual movements
3. Trend direction (growing, declining, stable)
4. Most important finding in one sentence

Be specific with numbers and percentages.
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text