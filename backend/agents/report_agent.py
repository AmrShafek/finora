import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

def report_agent(question: str, data_analysis: str, patterns: str, forecast: str, insights: str) -> str:
    """
    Report Agent — transforms all agent outputs into a clean,
    human-readable final response for the user.
    """
    prompt = f"""
You are a financial Report Agent for Finora AI.
Your job is to take all the analysis from other agents and write a clean, clear, professional response for the user.

User asked: {question}

Data Analysis:
{data_analysis}

Patterns Found:
{patterns}

Forecast:
{forecast}

Strategic Insights:
{insights}

Write a final response that:
1. Directly answers the user's question in the first paragraph
2. Highlights the most important findings
3. Gives clear actionable recommendations
4. Is written in plain English (no jargon)
5. Uses bullet points where helpful
6. Ends with a one-sentence summary

Keep the total response under 300 words. Be clear, confident, and helpful.
"""
    response = model.generate_content(prompt)
    return response.text
    