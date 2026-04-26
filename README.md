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
python simple_agent.py

==========================================================================================
USER: Who won the India vs New Zealand finals in CT 2025?

ASSISTANT:

India won the Champions Trophy (CT) 2025 final against New Zealand by 4 wickets. The match, held in Dubai on March 9, 2025, saw India chase down a target of 252, securing their third Champions Trophy title. Key highlights include Rohit Sharma's 76-run innings and Axar Patel's crucial contributions. 

For details:  
- [ESPNcricinfo Match Report](https://www.espncricinfo.com/series/icc-champions-trophy-2024-25-1459031/india-vs-new-zealand-final-1466428/live-cricket-score)  
- [BBC Sport Summary](https://www.bbc.com/sport/cricket/live/c1lv2p7nnmzt)  
- [Zee News Highlights](https://zeenews.india.com/cricket/live-updates/ind-vs-nz-final-live-cricket-score-and-updates-icc-champions-trophy-2025-india-vs-new-zealand-odi-match-full-scorecard-dubai-international-stadium-rohit-sharma-vs-mitchell-santner-virat-kohli-2869348.html?fromapp=yes)
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
python multi_agent.py

======================================================================
🤖 Multi-Agent Investment Analyst (LangChain + LangGraph + Groq)
======================================================================

📌 Query: Analyze companies like Tesla, NVDA, Apple and suggest which to buy for long term


🌐 [Web Agent] Searching the web for latest insights...


📈 [Finance Agent] Fetching financial data...


🧠 [Supervisor] Synthesizing findings and generating final recommendation...


======================================================================
📊 FINAL RECOMMENDATION
======================================================================
Okay, I need to create a comprehensive investment recommendation based on the provided research on Tesla (TSLA), NVIDIA (NVDA), and Apple (AAPL). Let me start by understanding the data given from both the Web Research Findings and the Financial Data & Analysis.

First, the Web Research highlights each company's strengths and risks. Tesla has strong financial reserves and global expansion plans but faces sales volatility and product reliability issues. NVIDIA is leading in AI with high market cap but has regulatory risks and a high valuation. Apple has a strong ecosystem and stable revenue but faces market saturation and regulatory challenges. The recommendation from the Web Research suggests NVIDIA for growth, Apple for stability, and Tesla as a high-risk option.

Next, the Financial Data provides metrics like P/E ratios, revenue, EPS, etc. NVIDIA has the highest market cap at $5.06T with a strong P/E of 42.59, while Apple has a P/E of 34.35. Tesla's P/E is very high at 345.23. Analyst ratings also show NVIDIA having the strongest consensus with many "Strong Buy" ratings, followed by Apple and Tesla more cautiously.

My task is to create a summary table comparing key metrics, list pros and cons for each, and provide a clear recommendation. Let me start by structuring the summary table. The key metrics should include Market Cap, P/E Ratio, Revenue (Q3 2024), Profit Margin, 52-Week Range, and Analyst Ratings. I'll need to extract these from the Financial Data section.

For the pros and cons, I'll need to combine the strengths and risks from the Web Research with the financial metrics. For example, Tesla's pros include financial reserves and global expansion, but cons are high P/E and sales volatility. NVIDIA's pros are AI dominance and high profit margins, but cons are regulatory issues and high valuation. Apple's pros are ecosystem and stable revenue, cons are market saturation and competition.

The recommendation section needs to align with the previous summaries, suggesting allocations based on risk tolerance. The example action plan from the Financial Data suggests 50% NVDA, 30% AAPL, and avoiding TSLA unless high risk is acceptable. I should explain the rationale, such as NVIDIA's AI growth potential, Apple's stability, and Tesla's high risk.

I should also ensure that sources are cited, but since this is a synthetic analysis, maybe just reference the provided data. Use markdown tables for the summary and pros/cons. Need to be clear and concise, making sure the justification for each recommendation is backed by the data presented.

Let me check if I have all the necessary data points. For each company, the table needs Market Cap, P/E, Revenue, Profit Margin, 52-Week Range, and Analyst Ratings. The pros and cons should integrate both qualitative and quantitative aspects. The recommendation should be actionable, advising percentages and explaining why each stock is chosen or avoided.

I think that's all. Now I'll structure the summary table, then the pros and cons, and then the recommendation with justification. Make sure to use clear markdown formatting and check for any errors in the data.

# Investment Analysis Summary: Tesla (TSLA), NVIDIA (NVDA), Apple (AAPL)  

---

## **1. Summary Table of Key Metrics**  

| **Metric**               | **Tesla (TSLA)**       | **NVIDIA (NVDA)**      | **Apple (AAPL)**       |  
|--------------------------|------------------------|------------------------|------------------------|  
| **Market Cap**           | $1.41T                 | $5.06T                 | $3.98T                 |  
| **P/E Ratio**            | 345.23 (very high)     | 42.59 (growth-oriented)| 34.35 (balanced)       |  
| **Q3 2024 Revenue**      | $97.9B                 | $215.9B                | $435.6B                |  
| **Profit Margin**        | 3.9%                   | 55.6% (industry-leading)| 27.0%                 |  
| **52-Week Price Range**  | $270.78 - $498.83      | $104.08 - $212.19      | $193.25 - $288.62      |  
| **Analyst Ratings**      | Mixed (5 SB, 18 B, 18 H)| Strong Bullish (9 SB, 47 B, 2 H)| Positive (7 SB, 24 B, 14 H) |  

---

## **2. Pros and Cons for Long-Term Investment**  

### **Tesla (TSLA)**  
**Pros**:  
- **Financial Reserves**: $37B in cash supports R&D and expansion.  
- **Global Expansion**: $2–3B India factory investment could unlock growth.  
- **EV Leadership**: Dominant brand in electric vehicles.  

**Cons**:  
- **High Valuation**: P/E of 345.23 reflects overvaluation risks.  
- **Sales Volatility**: 20% forecasted sales drop in 2024.  
- **Quality Concerns**: Frequent Cybertruck recalls and depreciation issues.  

---

### **NVIDIA (NVDA)**  
**Pros**:  
- **AI/ML Leadership**: Powers 90% of AI infrastructure with $100B+ investments.  
- **High Profitability**: 55.6% profit margin, industry-leading.  
- **Diversified Growth**: Robotics, self-driving, and biotech investments.  

**Cons**:  
- **Valuation Risks**: High P/E of 42.59; sensitive to AI growth slowdowns.  
- **Regulatory Scrutiny**: Export restrictions (e.g., China) may limit expansion.  
- **Competitive Pressure**: AMD/Intel gaining traction in AI chips.  

---

### **Apple (AAPL)**  
**Pros**:  
- **Ecosystem Lock-In**: Strong hardware-software integration and brand loyalty.  
- **Stable Cash Flows**: Services revenue provides consistent returns.  
- **Leadership Continuity**: Strategic stability post-Tim Cook transition.  

**Cons**:  
- **Market Saturation**: iPhone growth plateauing in mature markets.  
- **Innovation Risks**: Pressure from competitors like Samsung and Google.  
- **Regulatory Challenges**: Antitrust lawsuits threaten App Store revenue.  

---

## **3. Investment Recommendation**  

### **Recommended Allocation**  
| **Stock**          | **Allocation** | **Rationale**                                                                 |  
|--------------------|----------------|-------------------------------------------------------------------------------|  
| **NVIDIA (NVDA)**  | **50%**        | **Strong Buy**: AI and semiconductors represent secular growth. High profit margins and dominant market position justify long-term upside. Monitor regulatory risks. |  
| **Apple (AAPL)**   | **30%**        | **Buy**: Safe-haven stock with stable returns. Strong ecosystem and services revenue offset innovation pressures. Lower volatility compared to NVDA. |  
| **Tesla (TSLA)**   | **Avoid or ~20%** | **Hold**: High-risk/high-reward profile. Only consider if India expansion and autonomy bets succeed. Depreciation and recall risks outweigh current valuations. |  

---

### **Final Justification**  
1. **NVIDIA** is the top pick for long-term growth due to its leadership in AI infrastructure, which is central to the next decade of tech innovation. Its robust profit margins and diversified portfolio (gaming, robotics) reduce sector-specific risks.  
2. **Apple** provides stability and predictable cash flows, making it ideal for risk-averse investors. Its ecosystem lock-in ensures customer retention, even if growth slows.  
3. **Tesla** should be approached cautiously. While its EV dominance and global expansion plans offer upside, current risks (valuation, quality control) outweigh potential rewards for most long-term investors.  

**Diversification Note**: Balancing 50% NVDA and 30% AAPL creates a mix of high-growth and defensive assets. A small Tesla allocation (~20%) could be considered for speculative gains, but only with strict risk management.  

**Sources**: Financial data and analyst ratings from provided research. Valuation metrics and sector analysis synthesized from Web Research Findings.
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

