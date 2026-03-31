# Finora - Financial Intelligence Dashboard

Finora is an AI-powered financial intelligence dashboard designed to analyze and present company performance, specifically tailored for revenue, expenses, profit margins, and forecast data. The application uses a robust combination of a Python-based **FastAPI** backend with a multi-agent AI system and a modern **Next.js** React frontend.

It also includes advanced OCR (Optical Character Recognition) capabilities to ingest financial invoices and receipts automatically.

---

## System Architecture

### 1. The Backend (Python + FastAPI)
The backend acts as a central hub. It serves basic REST endpoints to fetch data directly from a SQLite database and provides advanced AI endpoints powered by **OpenRouter API** (using Mistral models). 
The AI system is built with a custom **LangGraph** orchestrator that delegates tasks logically to a team of specialized AI sub-agents.

### 2. The Frontend (Next.js + React)
A cutting-edge, dark-themed UI built to visualize complex financial metrics. It includes interactive `Recharts` graphs, dynamic Key Performance Indicators (KPIs), an embedded conversational AI chat, and a drag-and-drop OCR document scanner for automated expense ingestion.

---

## Architecture Flow

```
User Question
     ↓
[FastAPI /ask endpoint]
     ↓
[LangGraph Orchestrator]
     ↓
[Supervisor Agent] ← Orchestrates 5 agents
     ↓
┌─────────────────────────────────────────────┐
│  1. Data Agent    (5 nodes + validator)     │
│  2. Pattern Agent (4 nodes + validator)      │
│  3. Forecast Agent (6 nodes + validator)    │
│  4. Insight Agent (6 nodes + validator)     │
│  5. Report Agent  (5 nodes + validator)     │
└─────────────────────────────────────────────┘
     ↓
[Validation at each step - prevents hallucinations]
     ↓
Final Answer to User
```

---

## File Directory & Responsibilities

### `backend/` Directory
The brains of the operation. Handles API routing, database communication, and AI orchestration.

* **`main.py`**
  * **Job:** The primary entry point for the FastAPI server. It configures CORS, mounts the route handlers (`upload_route.py`), and defines the standard API endpoints (`/revenue`, `/expenses`, `/kpis`, `/insights`, `/recent-invoices`). It also defines the `/ask` endpoint, which delegates natural language questions to the LangGraph AI orchestrator, and `/ocr`, another endpoint for image processing.

* **`database.py`**
  * **Job:** Sets up the SQLite database connection using SQLAlchemy. Contains the `get_db` dependency used across API routes to fetch real-time financial data.

* **`ocr_agent.py`**
  * **Job:** Contains the logic to process images (receipts and invoices). Uses the `meta-llama/llama-3.2-11b-vision-instruct` model via OpenRouter to read an image and strictly extract standard JSON containing: `date`, `amount`, `category`, and `description`.

* **`upload_route.py`**
  * **Job:** Defines the `/upload-doc` REST endpoint. It validates uploaded image files, passes them to `ocr_agent.py` to extract financial data, normalizes the dates to match quarterly financial formats, and inserts the data as a new expense record in the SQLite database.

---

### `backend/langgraph/` Directory
The `langgraph` folder is responsible for the structured execution of AI agents, providing a robust wrapper around the individual agent logic.

* **`orchestrator.py`**
  * **Job:** Implements the core LangGraph execution flow. Creates SupervisorAgent instance, runs the async pipeline in a synchronous context, persists run history for provenance, and returns structured response. Acts as the bridge between FastAPI routes and the multi-agent AI pipeline.

* **`graph.py`**
  * **Job:** Defines graph data structures (Node, Graph classes) for potential future LangGraph features.

* **`store.py`**
  * **Job:** Handles state persistence for AI agent run history. Functions: `new_run_id()`, `persist_run()`, `load_run()`.

---

### `backend/langgraph/adapters/`
* **`agent_adapter.py`**
  * **Job:** Provides a standardized interface for calling any of the Finora agents. It maps agent types (e.g., "data", "forecast", "pattern", "insight", "report", "ocr") to their respective Python function entry points, ensuring input/output consistency and allowing the LangGraph system to interact with agent code.

---

### `backend/langgraph/validators/`
This subsystem ensures that AI-generated outputs meet the strict financial formatting requirements before being saved or shown to the user.

* **`base_validator.py`**
  * **Job:** Defines the abstract `Validator` class that all other validators inherit from.

* **`ocr_validator.py`**
  * **Job:** Strictly validates OCR output. Ensures the result is valid JSON, contains mandatory fields (`date`, `amount`, `category`, `description`), and verifies that the `amount` is a number and the `category` follows the approved financial labels.

* **`data_validator.py`**, **`forecast_validator.py`**, **`insight_validator.py`**, **`pattern_validator.py`**, **`report_validator.py`**
  * **Job:** Specialized validators for each sub-agent's output, checking for length, presence of key financial indicators, and error-free formatting.

---

### `backend/agents/` Directory
A complex multi-agent system where 5 sub-agents execute sequentially under a supervisor. Each agent has built-in validation nodes to prevent hallucinations.

* **`supervisor.py`**
  * **Job:** The orchestrator. Defines the `SupervisorAgent` which controls the execution flow of the sub-agents:
    1. `validate_inputs` - Validates question and financial data
    2. `format_data` - Converts financial_data dict to readable text
    3. `run_data_agent` - Executes DataAgent
    4. `run_pattern_agent` - Executes PatternAgent
    5. `run_forecast_agent` - Executes ForecastAgent
    6. `run_insight_agent` - Executes InsightAgent
    7. `run_report_agent` - Executes ReportAgent
    8. `validate_outputs` - Verifies all agent outputs
    9. `format_result` - Builds final response
  * It checks outputs and catches failures to ensure a high-quality answer.
  * Includes input validation (injection detection, financial keywords check).

* **`data_agent.py`** (DataAgent)
  * **Job:** Step 1 - Financial data planning specialist
  * **Nodes (5):**
    1. `validate_input` - Checks question validity (injection detection, financial keywords)
    2. `classify_intent` - Classifies user intent (trend, forecast, comparison, snapshot, anomaly, ranking, summary)
    3. `build_data_plan` - Creates data retrieval plan (tables, columns, filters)
    4. `validate_plan` - Verifies plan validity (valid table/column names, date ranges 2011-2026)
    5. `format_output` - Formats final JSON response
  * **Output:** Structured data plan with `intent`, `tables_needed`, `columns_needed`, `filters`, `explanation`

* **`pattern_agent.py`** (PatternAgent)
  * **Job:** Step 2 - Pattern detection in financial data
  * **Nodes (4):**
    1. `validate_inputs` - Checks data/question validity
    2. `detect_patterns` - Analyzes data for trends, anomalies, seasonal patterns
    3. `validate_patterns` - Verifies pattern output quality
    4. `format_output` - Formats response
  * **Output:** `key_patterns[]`, `anomalies[]`, `trend_direction`, `most_important`

* **`forecast_agent.py`** (ForecastAgent)
  * **Job:** Step 3 - Financial forecasting
  * **Nodes (6):**
    1. `validate_inputs` - Checks inputs
    2. `parse_financials` - Extracts key numbers from data
    3. `generate_forecast` - Generates predictions
    4. `validate_forecast` - Sanity checks (magnitude limits, growth rate -90% to +500%)
    5. `enrich_narrative` - Writes professional forecast text
    6. `format_output` - Formats response
  * **Output:** `short_term` forecast, `annual` forecast, `growth_rate`, `confidence`, `risks[]`

* **`insight_agent.py`** (InsightAgent)
  * **Job:** Step 4 - Strategic insights for CFOs
  * **Nodes (6):**
    1. `validate_inputs` - Checks all inputs
    2. `build_context` - Synthesizes data, patterns, forecast
    3. `generate_insights` - Creates strategic recommendations
    4. `validate_insights` - Verifies insight quality (health score 1-10, evidence required)
    5. `enrich_narrative` - Writes executive narrative
    6. `format_output` - Formats response
  * **Output:** `direct_answer`, `insights[]`, `actions[]`, `health_score`, `key_risk`, `executive_summary`

* **`report_agent.py`** (ReportAgent)
  * **Job:** Step 5 - Final report generation
  * **Nodes (5):**
    1. `validate_inputs` - Checks all source inputs
    2. `plan_content` - Plans report structure
    3. `write_report` - Generates final report (under 300 words)
    4. `validate_report` - Word count & quality checks
    5. `format_output` - Formats with bullet points
  * **Output:** `direct_answer`, `key_findings[]`, `recommendations[]`, `summary_sentence`

---

## Data Flow

### Chat Flow (The AI Team)
1. **User asks a question** via the frontend chat input.
2. **`main.py`** receives the request and gathers a snapshot of the database (KPIs, revenue, latest expenses).
3. **`orchestrator.py`** (LangGraph) is invoked.
4. **`supervisor.py`** guides the team:
   - **`data_agent`** - Analyzes intent and creates data plan
   - **`pattern_agent`** - Finds trends and anomalies
   - **`forecast_agent`** - Predicts future performance
   - **`insight_agent`** - Scores business health and suggests actions
   - **`report_agent`** - Writes the final human-friendly response
5. Each agent has **built-in validators** that check inputs and outputs to prevent hallucinations.
6. The final report is returned to the user.

### Document Flow (The OCR Engine)
1. **User uploads an image** in the `FileUpload.tsx` component.
2. **`upload_route.py`** validates the image format.
3. **`ocr_agent.py`** uses high-end Llama-vision models to read the text.
4. The output is **validated** by `ocr_validator.py`.
5. The normalized data is **inserted into the database**, and the dashboard charts auto-refresh to show the new expense immediately.

---

## Validation & Hallucination Prevention

### Input Validation (Supervisor)
- Question length: 3-2000 characters
- **Injection pattern detection**: DROP, DELETE, INSERT, UNION, SCRIPT, JAILBREAK, etc.
- **Financial keywords check**: revenue, profit, expense, growth, margin, forecast, etc.
- Financial data: must have kpis or revenue keys

### Per-Agent Validation (Built into each agent)
Each agent validates:
- **Input**: Not empty, contains numeric data, no injection patterns
- **Output**: Required fields present, reasonable values, proper format

### Hallucination Prevention Mechanisms
- Structured JSON prompts with specific field requirements
- Validation of numeric ranges (growth rates -90% to +500%)
- Confidence scores from LLM
- Word count limits on reports (max 300 words)
- Retry mechanism (up to 2 retries) on validation failure

---

### `frontend/` Directory
A React 19 / Next.js 16 application using Tailwind CSS functionality for responsive design and charting packages for visualizations.

#### `frontend/app/`
* **`page.tsx`**
  * **Job:** The central Dashboard component. 
  * It fetches the financial data (`/revenue`, `/kpis`, etc.) on load and auto-refreshes every 1 hour. 
  * Renders KPI summary cards (Annual Revenue, Expenses, Margin).
  * Renders complex interactive charts (Quarterly Revenue AreaChart, Profit Margin BarChart, and a combined LineChart) using the `recharts` library.
  * Contains the Chat UI embedded within the page, enabling users to message the Backend AI `SupervisorAgent`.
  * Renders the "Recent Invoices" panel visually, tying them sequentially to uploaded data.

* **`layout.tsx`** & **`globals.css`**
  * **Job:** Serves as the HTML shell definition and manages global CSS variables (including typography and CSS resets required to maintain the dark/glassmorphism design).

#### `frontend/app/components/`
* **`FileUpload.tsx`**
  * **Job:** Displays the drag-and-drop OCR Document Scanner component. It manages drag events, handles image file previews, and uploads files to the backend `/upload-doc` route via standard form data. It tracks upload status to provide visual feedback to the user.

---

## Technology Stack

- **Backend**: FastAPI, Python 3.10+
- **LLM**: OpenRouter (mistralai/mistral-small-24b-instruct-2501)
- **Database**: SQLite with SQLAlchemy
- **Orchestration**: Custom LangGraph pattern implementation
- **Frontend**: Next.js 16, React 19, Tailwind CSS 4, Recharts

---

## Running the Project

### Backend
```bash
cd backend
python -m uvicorn main:app --reload
# Runs on http://localhost:8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

### API Endpoints
- `GET http://localhost:8000/revenue` - Revenue data
- `GET http://localhost:8000/expenses` - Expenses data
- `GET http://localhost:8000/kpis` - KPI data
- `GET http://localhost:8000/insights` - AI insights
- `POST http://localhost:8000/ask` - Chatbot endpoint
- `POST http://localhost:8000/ocr` - Document OCR
- `POST http://localhost:8000/upload-doc` - File upload for OCR