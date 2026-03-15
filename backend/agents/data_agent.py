import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def data_agent(question: str, db_summary: str) -> str:
    """
    Data Agent — understands the user question and
    identifies what data is needed from the database.
    """
    prompt = f"""
You are a financial Data Agent for Finora AI.
Your job is to analyze a user question and identify exactly what financial data is needed to answer it.

The database contains real Salesforce financial data with these tables:
- revenue: quarterly revenue from 2011 to 2026
- expenses: quarterly operating expenses from 2011 to 2026  
- kpis: annual KPIs including revenue, expenses, net_profit, profit_margin
- ai_insights: AI-generated financial insights

Current database summary:
{db_summary}

User question: {question}

Respond with:
1. What specific data is needed
2. What time period is relevant
3. What metrics should be focused on

Be concise and specific.
"""
    response = model.generate_content(prompt)
    return response.text