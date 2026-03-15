from agents.data_agent import data_agent
from agents.pattern_agent import pattern_agent
from agents.forecast_agent import forecast_agent
from agents.insight_agent import insight_agent
from agents.report_agent import report_agent

def supervisor(question: str, financial_data: dict) -> dict:
    """
    Supervisor Agent — orchestrates all 5 agents in sequence.
    Each agent builds on the previous one's output.
    """

    # Format the financial data into a readable summary
    kpis = financial_data.get("kpis", [])
    revenue = financial_data.get("revenue", [])

    # Build a text summary of the data
    kpi_summary = "\n".join([
        f"{k['year']}: Revenue=${k['revenue']/1e9:.1f}B, "
        f"Expenses=${k['expenses']/1e9:.1f}B, "
        f"Net Profit=${k['profit']/1e9:.1f}B, "
        f"Margin={k['margin']}%"
        for k in kpis[-6:]  # last 6 years
    ])

    revenue_summary = "\n".join([
        f"{r['month']}: ${r['amount']/1e9:.2f}B"
        for r in revenue[-8:]  # last 8 quarters
    ])

    db_summary = f"""
Recent Annual KPIs (last 6 years):
{kpi_summary}

Recent Quarterly Revenue (last 8 quarters):
{revenue_summary}
"""

    print(f"🤖 Supervisor: Starting analysis for: '{question}'")

    # Step 1: Data Agent — identify what data is needed
    print("🔍 Data Agent: Analyzing question...")
    data_analysis = data_agent(question, db_summary)

    # Step 2: Pattern Agent — detect patterns
    print("🧠 Pattern Agent: Detecting patterns...")
    patterns = pattern_agent(db_summary, question)

    # Step 3: Forecast Agent — predict future
    print("🔮 Forecast Agent: Generating forecast...")
    forecast = forecast_agent(db_summary, patterns, question)

    # Step 4: Insight Agent — strategic insights
    print("💡 Insight Agent: Generating insights...")
    insights = insight_agent(db_summary, patterns, forecast, question)

    # Step 5: Report Agent — final clean response
    print("📣 Report Agent: Writing final report...")
    final_report = report_agent(question, data_analysis, patterns, forecast, insights)

    print("✅ Supervisor: Analysis complete!")

    return {
        "question": question,
        "answer": final_report,
        "details": {
            "data_analysis": data_analysis,
            "patterns": patterns,
            "forecast": forecast,
            "insights": insights,
        }
    }