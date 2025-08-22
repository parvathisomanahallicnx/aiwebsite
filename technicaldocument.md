# CNX LangGraph Agent — Technical Document

This document details the architecture, components, data flow, and operational concerns of the CNX LangGraph Agent.

---

## 1. Architecture overview

- **API Layer**: `FastAPI` app defined in `langgraph_agent_workflow.py` exposes:
  - `POST /agent-assistant/` — main chat endpoint
  - `GET /health` — healthcheck
- **Workflow Orchestration**: `LangGraph` manages routing via a state machine:
  - `analyze_intent` → routes to nodes based on user intent
  - `product_search` → MCP product search
  - `order_creation` → MCP order creation
  - `order_status` → MCP order status
  - `info_search` → RAG (Pinecone + Gemini)
- **RAG Stack**:
  - Embeddings: `HuggingFaceEmbeddings("intfloat/e5-large")` (1024-d)
  - Vector store: Pinecone (existing index, name from `PINECONE_INDEX`)
  - LLM: `ChatGoogleGenerativeAI` (Gemini) for QA and formatting

---

## 2. Code components

- `langgraph_agent_workflow.py`
  - `AgentState`: workflow state dictionary
  - `call_mcp_server(url, tool_name, arguments)`: generic JSON-RPC call to MCP servers
  - `call_gemini_llm(prompt)`: low-level Gemini invocation
  - `analyze_user_intent(state)`: intent classifier (Gemini + fallbacks)
  - `product_search_node(state)`: parses filters → calls product MCP → LLM post-filtering
  - `order_creation_node(state)`: extracts fields → calls order MCP → formats result
  - `order_status_node(state)`: extracts ID → calls order MCP → formats result
  - `info_search_node(state)`: RAG over Pinecone with:
    - Embeddings: `HuggingFaceEmbeddings("intfloat/e5-large")`
    - Retriever: `k=8`
    - QA chain: `RetrievalQA.from_chain_type(… retriever=retriever)`
    - Two-pass answering:
      1) Retrieve with `user_q`
      2) Brand-formatting pass (offers-aware or general structured)
  - `create_agent_workflow()` and `process_user_message()`
  - FastAPI app with `MessageRequest` and `AgentResponse` models

- `Train_Your_Own_Data.py`
  - Streamlit UI version of document embedding for local/manual use

---

## 3. Data flow

1. Client sends messages to `POST /agent-assistant/`
2. `process_user_message()` compiles workflow and runs it
3. `analyze_intent` picks a path
4. For `info_search`:
   - Pinecone client initialized with `PINECONE_API_KEY`
   - Embedding model: `intfloat/e5-large` (1024-d)
   - Retrieve with `user_q` using `k=8`
   - `RetrievalQA` returns `result["result"]`
   - Second LLM pass formats to CNXStore guidelines (offers-aware when relevant)
5. Response returned to client with formatted text in `final_response`

---

## 4. Environment and configuration

- `.env` (local) and Render environment (production):
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY` — Gemini access
  - `PINECONE_API_KEY` — Pinecone access
  - `PINECONE_INDEX` — index name (must be 1024-d)
  - `DOC_DIR_PATH` — docs folder (default `docs`)
  - `PRODUCT_SEARCH_MCP_URL` — product MCP
  - `ORDER_MCP_URL` — order MCP

- Pinecone index dimension must be 1024 to match `intfloat/e5-large`.

---

## 5. Security considerations

- Never commit real API keys; use environment variables.
- Consider removing any default API key values in code.
- Validate and sanitize external responses from MCP servers.
- Add rate limiting and auth in FastAPI if exposed publicly.

---

## 6. Deployment (Render)

- File: `render.yaml`
  - `buildCommand`: install requirements
  - `startCommand`: `uvicorn langgraph_agent_workflow:app --host 0.0.0.0 --port $PORT --proxy-headers`
  - `healthCheckPath`: `/health`
  - Environment variables as listed above

- Logs: Use Render dashboard to view app logs and embedding output `[embed] …`.

---

## 7. Testing and debugging

- Local quick test: `test_rag_debug.py`
- Manual query test:
```python
from langgraph_agent_workflow import process_user_message
process_user_message("Any current offers or discounts?")
```
- Observe logs for:
  - `[DEBUG] Pinecone client initialized`
  - `[DEBUG] Vector store connected`
  - `[DEBUG] QA chain initialized`
  - `[DEBUG] RAG successful - Answer length: …`

---

## 8. Extensibility

- Add new intents by extending `analyze_user_intent()` mappings and creating new nodes.
- Swap embedding models by ensuring index dimension alignment and changing `HuggingFaceEmbeddings` model.
- Add reranking or hybrid search by plugging a reranker after initial retrieval.

---

## 9. Known limitations

- Answers strictly grounded in indexed docs for info search; missing topics will get minimal answers unless new content is added.
- The structured, brand-styled formatter depends on the content fidelity in the docs.

---

## 10. Monitoring

- Surface key events via logs.
- Optionally add metrics (request counts, latency, retrieval hit rate) using a middleware or an APM agent.
