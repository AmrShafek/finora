import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(" gemini-2.5-flash")

def insight_agent(data: str, patterns: str, forecast: str, question: str) -> str:
    """
    Insight Agent — synthesizes all findings into strategic
    recommendations and actionable insights.
    """
    prompt = f"""
You are a financial Insight Agent for Finora AI.
Your job is to synthesize all findings from other agents and produce strategic insights and recommendations for business decision makers.

Financial data summary:
{data}

Patterns detected:
{patterns}

Forecast:
{forecast}

User question: {question}

Provide:
1. Direct answer to the user's question
2. Top 3 strategic insights
3. Top 2 recommended actions
4. One key risk to watch
5. Overall financial health score (1-10) with brief explanation

Write as if you are a senior financial advisor speaking to a CFO.
Be confident, specific, and actionable.
"""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text