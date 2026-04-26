# Agentic AI Workspace Documentation

This repository contains three Python agent workflows built with LangChain/LangGraph and Groq:

- `simple_agent.py`: single-agent web search assistant
- `retrieval_agent.py`: retrieval-augmented Thai recipe assistant (PDF + LanceDB + web fallback)
- `multi_agent.py`: multi-agent investment workflow (web + finance + supervisor)

---

## 1) Repository Structure

```text
agentic-ai/
├── simple_agent.py
├── retrieval_agent.py
├── multi_agent.py
├── requirements.txt
├── ThaiRecipes.pdf
└── lancedb/
```

---

## 2) High-Level Architecture

## A. `simple_agent.py`
**Goal:** Demonstrates a minimal LangChain agent with Groq + DuckDuckGo.

Flow:
1. Load `.env`
2. Validate `GROQ_API_KEY`
3. Create `ChatGroq`
4. Add `DuckDuckGoSearchResults` tool
5. Build agent with `create_agent`
6. Ask a sample question and print result

---

## B. `retrieval_agent.py`
**Goal:** Thai recipe Q&A with local knowledge base + web fallback.

Flow:
1. Load `.env` and validate `GROQ_API_KEY`
2. Load `ThaiRecipes.pdf` using `PyPDFLoader`
3. Chunk text with `RecursiveCharacterTextSplitter`
4. Embed chunks with `HuggingFaceEmbeddings`
5. Store vectors in LanceDB (`lancedb/recipes`)
6. Define custom tool: `thai_recipe_kb_search`
7. Add DuckDuckGo web tool
8. Run agent on sample Thai cuisine questions

---

## C. `multi_agent.py`
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

## 3) Dependencies (Current vs Recommended)

From current code usage:

### Required by scripts
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
- `yfinance`  (used in `multi_agent.py`)

### Likely unused right now
- `requests` (not imported in current scripts)

Recommendation: add `yfinance>=0.2.0` to `requirements.txt` if missing.

---

## 4) Environment Variables

Create a `.env` file in project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

If you later switch any agent to OpenAI models, also add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 5) Setup Instructions

Use your existing virtual environment or create one:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `yfinance` is not listed yet:

```bash
pip install yfinance
```

---

## 6) How to Run

### A. Run simple agent

```bash
python simple_agent.py
```

### B. Run retrieval agent

```bash
python retrieval_agent.py
```

Notes:
- Requires `ThaiRecipes.pdf` in project root.
- Builds/updates LanceDB index at startup.

### C. Run multi-agent investment workflow

```bash
python multi_agent.py
```

---

## 7) Script-by-Script Details

## `simple_agent.py`
- Best for quick demos.
- Uses `DuckDuckGoSearchResults(num_results=5)`.
- Input format:
  - `agent.invoke({"messages": [("user", question)]})`

## `retrieval_agent.py`
- Uses local document retrieval before web fallback.
- Custom tool returns labeled snippets `[KB-1]`, `[KB-2]`, etc.
- Current agent construction in code:
  - `agent = create_agent(llm, tools)`
- `system_prompt` variable exists but is not passed to `create_agent` in current file.

## `multi_agent.py`
- Uses `create_agent(..., system_prompt=...)` for both sub-agents.
- Uses LangGraph typed state and node chaining.
- Supervisor combines web + finance outputs into final recommendation.

---

## 8) Known Observations / Improvements

1. `requirements.txt`
   - Add `yfinance>=0.2.0`
   - Remove `requests` if you want a strict minimal dependency list

2. `retrieval_agent.py`
   - `system_prompt` is defined but not applied to `create_agent`
   - Suggested change:
     ```python
     agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
     ```

3. `multi_agent.py`
   - Uses `create_agent` API (good; avoids deprecated `create_react_agent`)
   - If future LangChain versions change response shape, add safe extraction logic for `result["messages"]`

4. Data freshness
   - Web search output can vary by time and source quality
   - `yfinance` fields may be missing for some tickers

---

## 8.5) Sample Outputs

### A. `simple_agent.py` Output

```
==========================================================================================
USER: Who won the India vs New Zealand finals in CT 2025?

A:

Based on the search results, India won the ICC Men's T20 Cricket World Cup 2024 final against South Africa (not New Zealand). India defeated South Africa by 6 runs in the final held in Barbados.

However, if you're asking about a specific India vs New Zealand final in CT 2025, please note that this information may not be available yet as of the current date. You could try rephrasing the question with more specific details about the tournament or match date for accurate results.
```

---

### B. `retrieval_agent.py` Output

```
==========================================================================================
USER: How do I make Thai Green Curry?

A:

Based on Thai Recipes knowledge base:

**Thai Green Curry (Gaeng Keow Wan)** 

Ingredients [KB-1]:
- 2 tbsp green curry paste
- 1 can (14 oz) coconut milk
- 300g chicken, sliced
- 100g green beans
- 3-4 Thai basil leaves
- 2 tbsp fish sauce
- 1 tbsp palm sugar

Instructions [KB-1]:
1. Heat 1/4 cup coconut milk in a large pan over medium-high heat
2. Add curry paste and stir-fry for 2 minutes until fragrant
3. Add chicken and cook until no longer pink (5-7 minutes)
4. Pour remaining coconut milk and bring to simmer
5. Add green beans and simmer for 5 minutes
6. Season with fish sauce and palm sugar
7. Stir in Thai basil leaves just before serving
8. Serve over jasmine rice

Total cooking time: ~20 minutes
Serves: 2-3 people
```

---

### C. `multi_agent.py` Output

```
==========================================================================================
USER: Analyze Tesla stock (TSLA) for investment recommendation

==========================================================================================
Running Web Research...
[Web Agent Processing...]

==========================================================================================
Running Financial Analysis...
[Finance Agent Processing...]

==========================================================================================
SUPERVISOR SYNTHESIS:

**Stock:** Tesla (TSLA)

**Web Research Findings [WEB-1]:**
- Latest news indicates strong EV market recovery
- Analyst sentiment: Mixed to Positive
- Key events: Q4 earnings beat expectations

**Financial Metrics [FIN-1]:**
- Current Price: $245.50
- P/E Ratio: 68.5
- Market Cap: $780B
- 52-week Range: $138 - $278
- Dividend Yield: 0%

**Technical Indicators:**
- Moving Average (50d): $242.30
- RSI (14): 58 (Neutral)
- Volume Trend: Above Average

**FINAL RECOMMENDATION:**

**HOLD / CAUTIOUS BUY** (for long-term investors)

Rationale:
- Strong fundamentals and market position
- Valuation is elevated (P/E > 60)
- Recent positive earnings momentum
- Recommend dollar-cost averaging for new positions
- Monitor Q1 delivery numbers closely

*Disclaimer: This analysis is for informational purposes only, not investment advice.*
```

---

## 9) Troubleshooting

### Error: Missing `GROQ_API_KEY`
- Ensure `.env` exists and has a valid key
- Ensure scripts run from project root
- Verify quickly:

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(bool(os.getenv('GROQ_API_KEY')))"
```

### Error: `No module named yfinance`

```bash
pip install yfinance
```

### DuckDuckGo tool returns sparse/empty results
- Retry with clearer keywords
- Increase `num_results` in relevant script

### PDF not found in retrieval flow
- Place `ThaiRecipes.pdf` in root:
  - `agentic-ai/ThaiRecipes.pdf`

---

## 10) Suggested Next Enhancements

1. Add CLI arguments for runtime query input
2. Add structured logging instead of `print`
3. Add retries/backoff for network tool calls
4. Add caching for repeated retrieval/web queries
5. Add tests for:
   - tool output contracts
   - graph state transitions
   - prompt regression snapshots

---

## 11) Quick Start Checklist

- [ ] Create and activate virtual environment
- [ ] Install dependencies
- [ ] Add `GROQ_API_KEY` to `.env`
- [ ] Ensure `ThaiRecipes.pdf` exists
- [ ] Run `simple_agent.py`
- [ ] Run `retrieval_agent.py`
- [ ] Run `multi_agent.py`

---

## 12) Usage Notes

- External API/tool outputs may change over time.
- Financial output is informational only, not investment advice.

