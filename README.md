# Agentic AI Workbench

### 🚀 Agentic AI examples with LangChain + LangGraph + Groq:
- ✅ A simple agent
- 📚 An agentic RAG assistant
- 🧠 Multi-agent orchestration

This repository contains three Python agent workflows built with `LangChain/LangGraph and Groq`:

- `simple_agent.py`: single-agent web search assistant
- `retrieval_agent.py`: retrieval-augmented Markdown assistant (PDF + LanceDB + web fallback)
- `multi_agent.py`: multi-agent investment workflow (web + finance + supervisor)

---

## 📁 1) Repository Structure

```text
agentic-ai-workbench/
├── simple_agent.py
├── retrieval_agent.py
├── multi_agent.py
├── requirements.txt
├── markdown-guide-sample.pdf
└── lancedb/
```

---

## 🏗️ 2) High-Level Architecture

## A. `simple_agent.py` ✅
**Goal:** Demonstrates a minimal LangChain agent with Groq + DuckDuckGo.

Flow:
1. Load `.env`
2. Validate `GROQ_API_KEY`
3. Create `ChatGroq`
4. Add `DuckDuckGoSearchResults` tool
5. Build agent with `create_agent`
6. Ask a sample question and print result

---

## B. `retrieval_agent.py` 📚
**Goal:** Markdown Q&A with local knowledge base + web fallback.

Flow:
1. Load `.env` and validate `GROQ_API_KEY`
2. Load `markdown-guide-sample.pdf` using `PyPDFLoader`
3. Chunk text with `RecursiveCharacterTextSplitter`
4. Embed chunks with `HuggingFaceEmbeddings`
5. Store vectors in LanceDB (`lancedb/markdown_guide.lance`)
6. Define custom tool: `markdown_kb_search`
7. Add DuckDuckGo web tool
8. Run agent on sample Markdown questions

---

## C. `multi_agent.py` 🧠
**Goal:** Multi-agent stock analysis with orchestration in LangGraph.

Flow:
1. Define tools:
   - `search_web`
   - `get_stock_price`
   - `get_stock_fundamentals`
   - `get_analyst_recommendations`
   - `get_company_info`
2. Create specialized agents:
   - Web Agent (qualitative context)
   - Finance Agent (quantitative metrics)
3. Build LangGraph with state:
   - `messages`
   - `web_research`
   - `financial_data`
   - `final_response`
4. Graph order:
   - `web_agent -> finance_agent -> supervisor -> END`
5. Print synthesized final recommendation

---

## 📦 3) Dependencies

From current code usage:

- `python-dotenv`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-groq`
- `langgraph`
- `pypdf`
- `langchain-text-splitters`
- `lancedb`
- `sentence-transformers`
- `langchain-huggingface`
- `ddgs`
- `yfinance`

---

## 🔐 4) Environment Variables

Create a `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🛠️ 5) Setup Instructions

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ 6) How to Run

### A. Run simple agent

```bash
python simple_agent.py
```

### B. Run retrieval agent

```bash
python retrieval_agent.py
```

Notes:
- Requires `markdown-guide-sample.pdf` in project root.
- First run builds the LanceDB index; later runs reuse it.
- **Source PDF:** https://www.markdownguide.org/assets/markdown-guide-sample.pdf

### C. Run multi-agent investment workflow

```bash
python multi_agent.py
```

---

## 🔎 7) Script-by-Script Details

## `simple_agent.py`
- Uses `DuckDuckGoSearchResults(num_results=5)`.
- Input format:
  - `agent.invoke({"messages": [("user", question)]})`

## `retrieval_agent.py`
- Uses local document retrieval before web fallback.
- Custom tool returns labeled snippets `[KB-1]`, `[KB-2]`, etc.
- Agent construction includes system prompt:
  - `agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)`

## `multi_agent.py`
- Uses `create_agent(..., system_prompt=...)` for both sub-agents.
- Uses LangGraph typed state and node chaining.
- Supervisor combines web + finance outputs into final recommendation.

---

## 🧪 8) Sample Output (Placeholder)

> Replace this section with real output after running each script.

### `simple_agent.py` output

```text
[PLACEHOLDER]
USER: <sample query>
ASSISTANT:
<sample response from simple agent>
```

### `retrieval_agent.py` output

```text
[PLACEHOLDER]
USER: <sample markdown question>
ASSISTANT:
<sample response using KB + optional web fallback>
```

### `multi_agent.py` output

```text
[PLACEHOLDER]
QUERY: <sample investment query>
FINAL RECOMMENDATION:
<sample synthesized recommendation>
```

---

## 🧯 9) Troubleshooting

### Error: Missing `GROQ_API_KEY`
- Ensure `.env` exists and has a valid key.
- Ensure scripts run from project root.

Quick check:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('GROQ_API_KEY')))"
```

### Error: `No module named yfinance`

```bash
pip install yfinance
```

### DuckDuckGo tool returns sparse/empty results
- Retry with clearer keywords.
- Increase `num_results` in the relevant script.

### PDF not found in retrieval flow
- Place `markdown-guide-sample.pdf` in root:
  - `agentic-ai-workbench/markdown-guide-sample.pdf`

### Download the sample PDF if missing:
- **Source PDF:** https://www.markdownguide.org/assets/markdown-guide-sample.pdf

---

## ✅ 10) Quick Start Checklist

- [ ] Create and activate virtual environment
- [ ] Install dependencies
- [ ] Add `GROQ_API_KEY` to `.env`
- [ ] Ensure `markdown-guide-sample.pdf` exists
- [ ] Run `simple_agent.py`
- [ ] Run `retrieval_agent.py`
- [ ] Run `multi_agent.py`

---

## 📝 11) Usage Notes

- External API/tool outputs may change over time.
- Financial output is informational only, not investment advice.
- Always verify with up-to-date sources before decisions.
