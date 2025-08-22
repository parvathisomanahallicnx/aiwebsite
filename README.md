# CNX LangGraph Agent — README

A FastAPI-powered assistant that routes user requests to:
- Product search (via MCP server)
- Order creation + status (via MCP server)
- Info search (RAG over Pinecone with HuggingFace embeddings)

This document explains local setup, environment, running, and deployment to Render.

---

## Project layout

- `langgraph_agent_workflow.py` — FastAPI app and LangGraph workflow (`/agent-assistant/`, `/health`)
- `Train_Your_Own_Data.py` — Streamlit tool for local embedding (UI)
- `docs/` — Place your DOCX content (e.g., policies, offers)
- `test_rag_debug.py` — Simple local tester for RAG behavior
- `requirements.txt` — Python dependencies
- `render.yaml` — Render.com deployment config

---

## Prerequisites

- Python 3.11 or 3.12
- Pinecone account and index (1024-dim, e.g., `cxnaidemo`)
- Google Gemini API key

---

## Environment variables

Set in a `.env` file locally and in Render dashboard for deployment.

Required:
- `GOOGLE_API_KEY` or `GEMINI_API_KEY` — Gemini key (only one is required by code)
- `PINECONE_API_KEY` — Pinecone API key
- `PINECONE_INDEX` — Pinecone index name (must match index dims: 1024)

Optional:
- `DOC_DIR_PATH` — Path to documents folder (default `docs`)
- `PRODUCT_SEARCH_MCP_URL` — MCP tool endpoint for product search (default set in code)
- `ORDER_MCP_URL` — MCP tool endpoint for orders (default set in code)

---

## Local development

We recommend using the included virtual environment at `aiwebsite/`.

PowerShell:
```powershell
# Activate venv
& .\aiwebsite\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r requirements.txt

# Run API locally
uvicorn langgraph_agent_workflow:app --host 0.0.0.0 --port 8002 --reload

# Test endpoint
curl -X POST http://localhost:8002/agent-assistant/ ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"source\":\"user\",\"content\":\"Any current offers or discounts?\"}]}"

## Embedding your documents

Two options:

1) Streamlit UI (local):
```powershell
& .\aiwebsite\Scripts\Activate.ps1
streamlit run Train_Your_Own_Data.py
```
Upload DOCX files under `docs/`. It will split and upload embeddings to Pinecone.

Note: Your Pinecone index must be created with dimension 1024 to match `intfloat/e5-large`.

---

## Deployment to Render

We provide `render.yaml` for zero-config deployment.

Steps:
1. Push this repo to GitHub.
2. In Render dashboard: New → Web Service → Select the repo.
3. Ensure the following environment variables are set on Render:
   - `GOOGLE_API_KEY` or `GEMINI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX`
4. Deploy. Render will install deps from `requirements.txt` and start the API: `uvicorn langgraph_agent_workflow:app ...`.

Health check: `GET /health`

Main endpoint: `POST /agent-assistant/`

Example body:
```json
{
  "messages": [
    { "source": "user", "content": "Any current offers or discounts?" }
  ]
}

