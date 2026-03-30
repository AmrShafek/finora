
"use client";
import FileUpload from "./components/FileUpload";
import { useEffect, useState, useRef } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from "recharts";

const API = "http://127.0.0.1:8001";

function fmt(n: number) {
  if (n >= 1e9) return "$" + (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return "$" + (n / 1e6).toFixed(1) + "M";
  return "$" + n.toLocaleString();
}

interface Message {
  role: "user" | "assistant";
  content: string;
  loading?: boolean;
}

interface Invoice {
  id: number;
  month: string;
  amount: number;
  category: string;
  description: string;
  created_at: string;
}

export default function Dashboard() {
  const [revenue, setRevenue]   = useState<any[]>([]);
  const [kpis, setKpis]         = useState<any[]>([]);
  const [insights, setInsights] = useState<any[]>([]);
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [loading, setLoading]   = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>("");

  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Hello! I'm Finora AI. Ask me anything about Salesforce's financial performance — revenue trends, profit margins, forecasts, or strategic insights.",
    },
  ]);
  const [question, setQuestion]   = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

useEffect(() => {
  const fetchData = () => {
    Promise.all([
      fetch(`${API}/revenue`).then(r => r.json()),
      fetch(`${API}/kpis`).then(r => r.json()),
      fetch(`${API}/insights`).then(r => r.json()),
      fetch(`${API}/recent-invoices`).then(r => r.json()),
    ]).then(([rev, kpi, ins, inv]) => {
      setRevenue(rev.slice(-20).map((r: any) => ({
        month: r.month.slice(0, 7),
        amount: r.amount / 1e9,
      })));
      setKpis(kpi.map((k: any) => ({
        year: k.month.slice(0, 4),
        revenue: k.revenue / 1e9,
        expenses: k.expenses / 1e9,
        profit: k.net_profit / 1e9,
        margin: k.profit_margin,
      })));
      setInsights(ins);
      setInvoices(inv);
      setLoading(false);
      setLastUpdated(new Date().toLocaleTimeString());
    });
  };

  // Fetch immediately on load
  fetchData();

  // Then refresh every 1 hour
  const interval = setInterval(fetchData, 3600000);

  // Cleanup on unmount
  return () => clearInterval(interval);
}, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = async () => {
    if (!question.trim() || chatLoading) return;

    const userMessage = question.trim();
    setQuestion("");

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);

    // Add loading message
    setMessages(prev => [...prev, { role: "assistant", content: "", loading: true }]);
    setChatLoading(true);

    try {
      const response = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage }),
      });
      const data = await response.json();

      // Replace loading with real answer
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: data.answer },
      ]);
    } catch {
      setMessages(prev => [
        ...prev.slice(0, -1),
        { role: "assistant", content: "Sorry, I couldn't connect to the AI agents. Please make sure the backend is running." },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const latestKpi = kpis[kpis.length - 1];
  const prevKpi   = kpis[kpis.length - 2];

  const cards = latestKpi ? [
    { label: "Annual Revenue",  value: fmt(latestKpi.revenue * 1e9),  delta: prevKpi ? `+${((latestKpi.revenue - prevKpi.revenue) / prevKpi.revenue * 100).toFixed(1)}%` : "", color: "#00e4a0" },
    { label: "Annual Expenses", value: fmt(latestKpi.expenses * 1e9), delta: "", color: "#4dd0e1" },
    { label: "Net Profit",      value: fmt(latestKpi.profit * 1e9),   delta: "", color: "#7c4dff" },
    { label: "Profit Margin",   value: latestKpi.margin + "%",        delta: prevKpi ? `prev ${prevKpi.margin}%` : "", color: "#ffb300" },
  ] : [];

  const severityColor: Record<string, string> = {
    info:    "#00e4a0",
    warning: "#ffb300",
    critical:"#ff5555",
  };

  const suggestedQuestions = [
    "What was Salesforce best year?",
    "Why did profit drop in 2023?",
    "What is the revenue forecast for 2027?",
    "How did expenses change over time?",
  ];

  if (loading) return (
    <div style={{ background: "#050d1a", height: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <div style={{ color: "#00e4a0", fontSize: 22, fontFamily: "monospace", letterSpacing: 4 }}>
        FINORA AI — Loading...
      </div>
    </div>
  );

  return (
    <div style={{
      background: "linear-gradient(135deg, #050d1a 0%, #08142a 60%, #050d1a 100%)",
      minHeight: "100vh",
      fontFamily: "'DM Sans', sans-serif",
      color: "#f0f8ff",
      padding: "0 0 60px 0",
    }}>

      {/* ── HEADER ── */}
      <div style={{
        borderBottom: "1px solid rgba(0,228,160,0.15)",
        padding: "20px 48px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: "rgba(5,13,26,0.8)",
        backdropFilter: "blur(12px)",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 28, fontWeight: 800, letterSpacing: -1, color: "#00e4a0" }}>F</span>
          <span style={{ fontSize: 28, fontWeight: 800, letterSpacing: -1 }}>inora</span>
          <span style={{
            fontSize: 11, letterSpacing: 3, color: "rgba(0,228,160,0.7)",
            background: "rgba(0,228,160,0.08)", border: "1px solid rgba(0,228,160,0.2)",
            padding: "3px 10px", borderRadius: 100, textTransform: "uppercase", marginLeft: 8,
          }}>AI</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#00e4a0", boxShadow: "0 0 8px #00e4a0" }} />
          <span style={{ fontSize: 13, color: "rgba(0,228,160,0.7)", letterSpacing: 1 }}>Live • Salesforce CRM</span>
          {lastUpdated && (
  <span style={{ fontSize: 11, color: "rgba(180,210,200,0.4)", marginLeft: 8 }}>
    Updated {lastUpdated}
  </span>
)}
        </div>
      </div>

      <div style={{ padding: "40px 48px", maxWidth: 1400, margin: "0 auto" }}>

        {/* ── PAGE TITLE ── */}
        <div style={{ marginBottom: 36 }}>
          <div style={{ fontSize: 13, letterSpacing: 5, textTransform: "uppercase", color: "rgba(0,228,160,0.7)", marginBottom: 8 }}>
            Financial Intelligence Dashboard
          </div>
          <div style={{ fontSize: 36, fontWeight: 800, letterSpacing: -1 }}>
            Salesforce Performance Overview
          </div>
          <div style={{ fontSize: 15, color: "rgba(180,210,200,0.5)", marginTop: 6 }}>
            FY2012 — FY2026 • Real data from Macrotrends
          </div>
        </div>

        {/* ── KPI CARDS ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 20, marginBottom: 36 }}>
          {cards.map((c, i) => (
            <div key={i} style={{
              background: "rgba(10,20,35,0.8)",
              border: `1px solid ${c.color}30`,
              borderRadius: 20,
              padding: "24px 28px",
              position: "relative",
              overflow: "hidden",
              boxShadow: `0 0 30px ${c.color}10`,
            }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 3, background: `linear-gradient(90deg, transparent, ${c.color}, transparent)`, opacity: 0.6 }} />
              <div style={{ fontSize: 12, letterSpacing: 3, textTransform: "uppercase", color: "rgba(180,210,200,0.5)", marginBottom: 12 }}>{c.label}</div>
              <div style={{ fontSize: 30, fontWeight: 800, color: c.color, letterSpacing: -1 }}>{c.value}</div>
              {c.delta && <div style={{ fontSize: 12, color: "rgba(180,210,200,0.5)", marginTop: 6 }}>{c.delta} vs prev year</div>}
            </div>
          ))}
        </div>

        {/* ── CHARTS ROW ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1.6fr 1fr", gap: 24, marginBottom: 24 }}>
          <div style={{ background: "rgba(10,20,35,0.8)", border: "1px solid rgba(0,228,160,0.1)", borderRadius: 20, padding: "28px 24px" }}>
            <div style={{ fontSize: 13, letterSpacing: 3, textTransform: "uppercase", color: "rgba(0,228,160,0.6)", marginBottom: 4 }}>Quarterly Revenue</div>
            <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Last 20 Quarters (Billions)</div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={revenue}>
                <defs>
                  <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor="#00e4a0" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00e4a0" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="month" tick={{ fill: "rgba(180,210,200,0.4)", fontSize: 10 }} tickLine={false} interval={3} />
                <YAxis tick={{ fill: "rgba(180,210,200,0.4)", fontSize: 10 }} tickLine={false} tickFormatter={v => `$${v}B`} />
                <Tooltip contentStyle={{ background: "#0a1628", border: "1px solid rgba(0,228,160,0.3)", borderRadius: 10, color: "#f0f8ff" }} formatter={(v: any) => [`$${Number(v).toFixed(2)}B`, "Revenue"]} />
                <Area type="monotone" dataKey="amount" stroke="#00e4a0" strokeWidth={2} fill="url(#revGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div style={{ background: "rgba(10,20,35,0.8)", border: "1px solid rgba(124,77,255,0.1)", borderRadius: 20, padding: "28px 24px" }}>
            <div style={{ fontSize: 13, letterSpacing: 3, textTransform: "uppercase", color: "rgba(124,77,255,0.6)", marginBottom: 4 }}>Profit Margin</div>
            <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Annual % by Year</div>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={kpis}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="year" tick={{ fill: "rgba(180,210,200,0.4)", fontSize: 10 }} tickLine={false} />
                <YAxis tick={{ fill: "rgba(180,210,200,0.4)", fontSize: 10 }} tickLine={false} tickFormatter={v => `${v}%`} />
                <Tooltip contentStyle={{ background: "#0a1628", border: "1px solid rgba(124,77,255,0.3)", borderRadius: 10, color: "#f0f8ff" }} formatter={(v: any) => [`${v}%`, "Margin"]} />
                <Bar dataKey="margin" fill="#7c4dff" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* ── REVENUE vs EXPENSES ── */}
        <div style={{ background: "rgba(10,20,35,0.8)", border: "1px solid rgba(77,208,225,0.1)", borderRadius: 20, padding: "28px 24px", marginBottom: 24 }}>
          <div style={{ fontSize: 13, letterSpacing: 3, textTransform: "uppercase", color: "rgba(77,208,225,0.6)", marginBottom: 4 }}>Revenue vs Expenses</div>
          <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Annual Comparison (Billions)</div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={kpis}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="year" tick={{ fill: "rgba(180,210,200,0.4)", fontSize: 11 }} tickLine={false} />
              <YAxis tick={{ fill: "rgba(180,210,200,0.4)", fontSize: 11 }} tickLine={false} tickFormatter={v => `$${v}B`} />
              <Tooltip contentStyle={{ background: "#0a1628", border: "1px solid rgba(77,208,225,0.3)", borderRadius: 10, color: "#f0f8ff" }} formatter={(v: any) => [`$${Number(v).toFixed(1)}B`]} />
              <Line type="monotone" dataKey="revenue"  stroke="#00e4a0" strokeWidth={2.5} dot={false} name="Revenue" />
              <Line type="monotone" dataKey="expenses" stroke="#ff6b6b" strokeWidth={2.5} dot={false} name="Expenses" strokeDasharray="5 3" />
              <Line type="monotone" dataKey="profit"   stroke="#ffb300" strokeWidth={2}   dot={false} name="Net Profit" />
            </LineChart>
          </ResponsiveContainer>
          <div style={{ display: "flex", gap: 24, marginTop: 16, paddingLeft: 8 }}>
            {[["Revenue", "#00e4a0"], ["Expenses", "#ff6b6b"], ["Net Profit", "#ffb300"]].map(([l, c]) => (
              <div key={l} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 24, height: 2, background: c }} />
                <span style={{ fontSize: 12, color: "rgba(180,210,200,0.5)" }}>{l}</span>
              </div>
            ))}
          </div>
        </div>

        {/* ── MAIN BOTTOM ROW: AI CHAT + INSIGHTS ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr", gap: 24 }}>

          {/* ── AI CHAT ── */}
          <div style={{
            background: "rgba(10,20,35,0.8)",
            border: "1px solid rgba(0,228,160,0.15)",
            borderRadius: 20,
            display: "flex",
            flexDirection: "column",
            height: 520,
            overflow: "hidden",
          }}>
            {/* Chat header */}
            <div style={{
              padding: "20px 24px",
              borderBottom: "1px solid rgba(0,228,160,0.1)",
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}>
              <div style={{
                width: 36, height: 36, borderRadius: "50%",
                background: "rgba(0,228,160,0.15)",
                border: "1px solid rgba(0,228,160,0.3)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 18,
              }}>🤖</div>
              <div>
                <div style={{ fontSize: 15, fontWeight: 700 }}>Finora AI Chat</div>
                <div style={{ fontSize: 11, color: "rgba(0,228,160,0.6)", letterSpacing: 1 }}>
                  {chatLoading ? "Agents thinking..." : "5 agents ready"}
                </div>
              </div>
              {chatLoading && (
                <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
                  {[0,1,2].map(i => (
                    <div key={i} style={{
                      width: 6, height: 6, borderRadius: "50%", background: "#00e4a0",
                      animation: `pulse 1s ease-in-out ${i * 0.2}s infinite`,
                    }} />
                  ))}
                </div>
              )}
            </div>

            {/* Messages */}
            <div style={{
              flex: 1,
              overflowY: "auto",
              padding: "20px 24px",
              display: "flex",
              flexDirection: "column",
              gap: 16,
            }}>
              {messages.map((msg, i) => (
                <div key={i} style={{
                  display: "flex",
                  justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                }}>
                  <div style={{
                    maxWidth: "85%",
                    padding: "12px 16px",
                    borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                    background: msg.role === "user"
                      ? "linear-gradient(135deg, #00e4a0, #00b884)"
                      : "rgba(255,255,255,0.05)",
                    border: msg.role === "user" ? "none" : "1px solid rgba(255,255,255,0.08)",
                    color: msg.role === "user" ? "#050d1a" : "#f0f8ff",
                    fontSize: 13,
                    lineHeight: 1.6,
                    fontWeight: msg.role === "user" ? 600 : 400,
                  }}>
                    {msg.loading ? (
                      <div style={{ display: "flex", gap: 6, alignItems: "center", padding: "4px 0" }}>
                        <span style={{ fontSize: 12, color: "rgba(180,210,200,0.6)" }}>Agents analyzing</span>
                        {[0,1,2].map(i => (
                          <div key={i} style={{
                            width: 5, height: 5, borderRadius: "50%",
                            background: "#00e4a0", opacity: 0.7,
                          }} />
                        ))}
                      </div>
                    ) : (
                      <div style={{ whiteSpace: "pre-wrap" }}>{msg.content}</div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>

            {/* Suggested questions */}
            {messages.length <= 1 && (
              <div style={{ padding: "0 24px 12px", display: "flex", flexWrap: "wrap", gap: 8 }}>
                {suggestedQuestions.map((q, i) => (
                  <button key={i} onClick={() => setQuestion(q)} style={{
                    background: "rgba(0,228,160,0.08)",
                    border: "1px solid rgba(0,228,160,0.2)",
                    borderRadius: 100,
                    padding: "6px 14px",
                    fontSize: 11,
                    color: "rgba(0,228,160,0.8)",
                    cursor: "pointer",
                    letterSpacing: 0.3,
                  }}>
                    {q}
                  </button>
                ))}
              </div>
            )}

            {/* Input */}
            <div style={{
              padding: "16px 24px",
              borderTop: "1px solid rgba(0,228,160,0.1)",
              display: "flex",
              gap: 12,
            }}>
              <input
                value={question}
                onChange={e => setQuestion(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about revenue, profits, forecasts..."
                disabled={chatLoading}
                style={{
                  flex: 1,
                  background: "rgba(255,255,255,0.05)",
                  border: "1px solid rgba(0,228,160,0.2)",
                  borderRadius: 12,
                  padding: "12px 16px",
                  color: "#f0f8ff",
                  fontSize: 13,
                  outline: "none",
                  fontFamily: "'DM Sans', sans-serif",
                }}
              />
              <button
                onClick={sendMessage}
                disabled={chatLoading || !question.trim()}
                style={{
                  background: chatLoading || !question.trim()
                    ? "rgba(0,228,160,0.2)"
                    : "linear-gradient(135deg, #00e4a0, #00b884)",
                  border: "none",
                  borderRadius: 12,
                  padding: "12px 20px",
                  color: chatLoading || !question.trim() ? "rgba(0,228,160,0.5)" : "#050d1a",
                  fontWeight: 700,
                  fontSize: 13,
                  cursor: chatLoading || !question.trim() ? "not-allowed" : "pointer",
                  fontFamily: "'DM Sans', sans-serif",
                }}
              >
                {chatLoading ? "..." : "Send →"}
              </button>
            </div>
          </div>

          {/* ── AI INSIGHTS ── */}
          <div style={{
            background: "rgba(10,20,35,0.8)",
            border: "1px solid rgba(0,228,160,0.1)",
            borderRadius: 20,
            padding: "28px 24px",
            height: 520,
            overflowY: "auto",
          }}>
            <div style={{ fontSize: 13, letterSpacing: 3, textTransform: "uppercase", color: "rgba(0,228,160,0.6)", marginBottom: 4 }}>AI Agents</div>
            <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 24 }}>Latest Insights</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {insights.map((ins, i) => (
                <div key={i} style={{
                  background: "rgba(5,13,26,0.6)",
                  border: `1px solid ${severityColor[ins.severity]}30`,
                  borderLeft: `3px solid ${severityColor[ins.severity]}`,
                  borderRadius: 14,
                  padding: "16px 18px",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                    <div style={{ width: 7, height: 7, borderRadius: "50%", background: severityColor[ins.severity], boxShadow: `0 0 6px ${severityColor[ins.severity]}`, flexShrink: 0 }} />
                    <span style={{ fontSize: 10, color: severityColor[ins.severity], letterSpacing: 2, textTransform: "uppercase" }}>{ins.agent}</span>
                    <span style={{ fontSize: 10, color: "rgba(180,210,200,0.3)", background: "rgba(255,255,255,0.04)", padding: "2px 8px", borderRadius: 100 }}>{ins.type}</span>
                  </div>
                  <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 5 }}>{ins.title}</div>
                  <div style={{ fontSize: 12, color: "rgba(180,210,200,0.55)", lineHeight: 1.6 }}>{ins.body}</div>
                </div>
              ))}
            </div>
          </div>

        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(0,228,160,0.2); border-radius: 2px; }
        input::placeholder { color: rgba(180,210,200,0.3); }
      `}</style>
              {/* ── DOCUMENT UPLOAD ── */}
        <div style={{ marginTop: 24 }}>
          <FileUpload onUploadSuccess={() => fetch(`${API}/recent-invoices`).then(r => r.json()).then(setInvoices)} />
        </div>

        {/* ── RECENT INVOICES PANEL ── */}
        {invoices.length > 0 && (
          <div style={{
            marginTop: 24,
            background: "rgba(10,20,35,0.8)",
            border: "1px solid rgba(0,228,160,0.15)",
            borderRadius: 20,
            padding: "28px 28px",
          }}>
            <div style={{ fontSize: 13, letterSpacing: 4, textTransform: "uppercase", color: "rgba(0,228,160,0.7)", marginBottom: 4 }}>
              Recent Invoices
            </div>
            <div style={{ fontSize: 20, fontWeight: 700, marginBottom: 20 }}>Latest Uploaded Documents</div>

            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {invoices.map((inv) => (
                <div key={inv.id} style={{
                  background: "rgba(5,13,26,0.6)",
                  border: "1px solid rgba(0,228,160,0.1)",
                  borderRadius: 14,
                  padding: "16px 20px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 16,
                  flexWrap: "wrap",
                }}>
                  {/* Invoice details */}
                  <div style={{ display: "flex", gap: 32, flexWrap: "wrap", flex: 1 }}>
                    <div>
                      <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(180,210,200,0.4)", textTransform: "uppercase", marginBottom: 3 }}>ID</div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#f0f8ff" }}>#{inv.id}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(180,210,200,0.4)", textTransform: "uppercase", marginBottom: 3 }}>Date</div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: "#f0f8ff" }}>{inv.created_at.slice(0, 10)}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(180,210,200,0.4)", textTransform: "uppercase", marginBottom: 3 }}>Amount</div>
                      <div style={{ fontSize: 13, fontWeight: 700, color: "#00e4a0" }}>${inv.amount.toLocaleString()}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(180,210,200,0.4)", textTransform: "uppercase", marginBottom: 3 }}>Category</div>
                      <div style={{
                        fontSize: 11, fontWeight: 600,
                        background: "rgba(0,228,160,0.1)",
                        border: "1px solid rgba(0,228,160,0.25)",
                        borderRadius: 100,
                        padding: "2px 10px",
                        color: "#00e4a0",
                        display: "inline-block",
                      }}>{inv.category}</div>
                    </div>
                    <div style={{ flex: 1, minWidth: 160 }}>
                      <div style={{ fontSize: 10, letterSpacing: 2, color: "rgba(180,210,200,0.4)", textTransform: "uppercase", marginBottom: 3 }}>Description</div>
                      <div style={{ fontSize: 13, color: "rgba(180,210,200,0.7)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis", maxWidth: 300 }}>{inv.description}</div>
                    </div>
                  </div>

                  {/* Ask AI button */}
                  <button
                    onClick={() => setQuestion(`Tell me about invoice #${inv.id}: ${inv.description} costing $${inv.amount} in ${inv.category} category on ${inv.created_at.slice(0,10)}`)}
                    style={{
                      background: "rgba(0,228,160,0.1)",
                      border: "1px solid rgba(0,228,160,0.3)",
                      borderRadius: 10,
                      padding: "8px 16px",
                      fontSize: 12,
                      color: "#00e4a0",
                      cursor: "pointer",
                      fontFamily: "'DM Sans', sans-serif",
                      fontWeight: 600,
                      whiteSpace: "nowrap",
                      transition: "all 0.2s",
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = "rgba(0,228,160,0.2)")}
                    onMouseLeave={e => (e.currentTarget.style.background = "rgba(0,228,160,0.1)")}
                  >
                    🤖 Ask AI about this
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    
  );
}