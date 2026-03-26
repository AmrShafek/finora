import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(" gemini-2.5-flash")

def forecast_agent(data: str, patterns: str, question: str) -> str:
    """
    Forecast Agent — predicts future financial performance
    based on historical data and detected patterns.
    """
    prompt = f"""
You are a financial Forecasting Agent for Finora AI.
Your job is to predict future financial performance based on historical data and patterns.

Historical financial data:
{data}

Patterns detected by Pattern Agent:
{patterns}

User question context: {question}

Provide:
1. Short-term forecast (next quarter)
2. Annual forecast (next year)
3. Growth rate estimate with reasoning
4. Key risks that could affect the forecast
5. Confidence level (High / Medium / Low) with explanation

Use specific numbers in your forecast. Base everything on the actual data provided.
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text