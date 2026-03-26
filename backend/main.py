from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_db
from pydantic import BaseModel
from agents.supervisor import supervisor

app = FastAPI(title="Finora API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Finora API is running!"}

@app.get("/revenue")
def get_revenue(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT month, amount, source
        FROM revenue
        WHERE company_id = 1
        ORDER BY month ASC
    """)).fetchall()
    return [{"month": str(r[0]), "amount": float(r[1]), "source": r[2]} for r in result]

@app.get("/expenses")
def get_expenses(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT month, amount, category
        FROM expenses
        WHERE company_id = 1
        ORDER BY month ASC
    """)).fetchall()
    return [{"month": str(r[0]), "amount": float(r[1]), "category": r[2]} for r in result]

@app.get("/kpis")
def get_kpis(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT month, revenue, expenses, net_profit, profit_margin
        FROM kpis
        WHERE company_id = 1
        ORDER BY month ASC
    """)).fetchall()
    return [{"month": str(r[0]), "revenue": float(r[1]), "expenses": float(r[2]), "net_profit": float(r[3]), "profit_margin": float(r[4])} for r in result]

@app.get("/insights")
def get_insights(db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT agent, insight_type, title, body, severity, created_at
        FROM ai_insights
        WHERE company_id = 1
        ORDER BY created_at DESC
    """)).fetchall()
    return [{"agent": r[0], "type": r[1], "title": r[2], "body": r[3], "severity": r[4], "date": str(r[5])} for r in result]

# ── AI CHAT endpoint — triggers all 5 agents ──
class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: AskRequest, db: Session = Depends(get_db)):
    # Fetch latest financial data from database
    kpis_raw = db.execute(text("""
        SELECT month, revenue, expenses, net_profit, profit_margin
        FROM kpis WHERE company_id = 1 ORDER BY month ASC
    """)).fetchall()

    revenue_raw = db.execute(text("""
        SELECT month, amount FROM revenue
        WHERE company_id = 1 ORDER BY month ASC
    """)).fetchall()

    # Format for agents
    kpis_data = [
        {
            "year": str(r[0])[:4],
            "revenue": float(r[1]),
            "expenses": float(r[2]),
            "profit": float(r[3]),
            "margin": float(r[4])
        }
        for r in kpis_raw
    ]

    revenue_data = [
        {"month": str(r[0]), "amount": float(r[1])}
        for r in revenue_raw
    ]

    financial_data = {
        "kpis": kpis_data,
        "revenue": revenue_data,
    }

    # Run all 5 agents through the supervisor
    result = supervisor(request.question, financial_data)
    return result
