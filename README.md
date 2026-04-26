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

## 🖥️ Sample Console Output

Below are example logs from all the client scripts. These are sample outputs and may vary depending on model response, server data, and runtime environment.

### `simple_agent.py` output

```text
python simple_agent.py

[transformers] Disabling PyTorch because PyTorch >= 2.4 is required but found 2.2.2
[transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.

==========================================================================================
USER: Who won the India vs New Zealand finals in CT 2025?

ASSISTANT:

India won the Champions Trophy 2025 final against New Zealand by four wickets. They chased a target of 252, with Rohit Sharma playing a crucial 76. This victory marked India's third Champions Trophy title. 

**Source:**  
- [BBC Sport: India beat New Zealand to win 3rd Champions Trophy](https://www.bbc.com/sport/cricket/live/c1lv2p7nnmzt)  
- [Cricbuzz: IND vs NZ Final Highlights](https://www.cricbuzz.com/live-cricket-scores/112469/ind-vs-nz-final-icc-champions-trophy-2025)
```

### `retrieval_agent.py` output
~~~text
python retrieval_agent.py                                      

python-dotenv could not parse statement starting at line 20
python-dotenv could not parse statement starting at line 22
Loading PDF from: /Users/naveenkumar/Code/agentic-ai-workbench/markdown-guide-sample.pdf
Ignoring wrong pointing object 49 0 (offset 0)
Loaded 10 pages from PDF.
Split into 12 chunks.
Vector DB ready.

==========================================================================================
USER: How do I create headings, bold text, and bullet lists in Markdown?

ASSISTANT:

### Headings
To create headings in Markdown, you can use the `#` symbol followed by the heading text. For example:
# Heading 1
## Heading 2
### Heading 3

### Bold Text
To create bold text in Markdown, you can surround the text with double asterisks `**`. For example:
**This text will be bold**

### Bullet Lists
To create bullet lists in Markdown, you can use the `*` symbol followed by the list item text. For example:
* Item 1
* Item 2
* Item 3

Note: You can also use `-` or `+` symbols to create bullet lists.

These are the basic ways to create headings, bold text, and bullet lists in Markdown. Let me know if you have any further questions!

==========================================================================================
USER: What is the difference between inline code and fenced code blocks in Markdown?

ASSISTANT:

The main difference between inline code and fenced code blocks in Markdown is how they are formatted and used.

Inline code is used to format a small piece of code within a line of text. It is created by surrounding the code with backticks (`). For example: `This is an example of inline code`.

Fenced code blocks, on the other hand, are used to format a larger block of code. They are created by surrounding the code with three backticks (```) on either side. For example:
```
This is an example of a fenced code block
```
Fenced code blocks can also be used to specify the language of the code, which can help with syntax highlighting. For example:
```python
This is an example of a fenced code block with syntax highlighting for Python
```
In summary, inline code is used for small pieces of code within a line of text, while fenced code blocks are used for larger blocks of code and can include syntax highlighting.
~~~

### `multi_agent.py` output

```text
python multi_agent.py

python-dotenv could not parse statement starting at line 20
python-dotenv could not parse statement starting at line 22
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
Okay, let's tackle this. The user wants me to synthesize the provided research on Tesla, NVIDIA, and Apple. First, I need to create a summary table comparing their key metrics. The financial data from both the web research and the detailed analysis should be combined here. I'll need to make sure the table includes stock price, market cap, P/E ratio, profit margin, revenue, and analyst ratings.

Next, the pros and cons for each company. For Tesla, strengths are growth in software and AI, but weaknesses include low profit margins and production issues. NVIDIA's strengths are AI dominance and high profit margins, but risks are geopolitical and competition. Apple's pros are stability and strong cash flow, but cons are market saturation. I'll need to list these clearly, maybe using bullet points for each.

For the recommendation, the user wants a clear recommendation on which stocks to buy. From the data, NVIDIA has the highest upside and strong fundamentals, Apple is the safest, and Tesla is high-risk. The portfolio suggestion from the web research mentions a mix of 50% Apple, 30% NVIDIA, and 20% Tesla. I should align with that, explaining the rationale for each stock based on their risk profiles and growth potential.

I need to make sure the sources are cited. The information comes from the provided web research and financial data analysis. Since the user mentioned using markdown for tables, I'll structure the summary table accordingly. Also, check that the pros and cons sections are concise and directly reference the data points given.

Wait, the financial data section has a P/E ratio for Tesla as 345.23 which is very high. That should be highlighted as a con. NVIDIA's P/E is 42.59, which is reasonable compared to its growth. Apple's 34.35 is also good. Profit margins are another key metric—Tesla's is 3.95%, which is low, while NVIDIA's is 55.6%, which is strong. These should be in the table and in the pros and cons.

Analyst ratings show NVIDIA has the most "Strong Buy" and "Buy" ratings, which supports the recommendation. Tesla has a lot of "Hold" and "Sell" ratings, indicating caution. Apple's ratings are more balanced but still positive.

I need to make sure the final recommendation explains the mix and the reasoning. For example, NVIDIA for growth, Apple for stability, Tesla as a speculative bet. Also, note the risks involved with each. Maybe suggest a diversified portfolio as per the web research's portfolio strategy.

Check for any missing metrics or inconsistencies. The market cap in the financial data section for TSLA is $1.41T, but in the web research, it was $807B. Wait, there's a discrepancy here. The web research says Tesla's market cap is $807B, but the financial data table shows $1.41T. Need to clarify which is correct. The financial data section might have more recent data from 2025, so I should use that. But in the web research, the TSLA market cap was listed as $807B. Maybe it's a typo or time difference. Since the financial data is more detailed and includes dates (2025), I'll go with the $1.41T.

Also, in the web research, Apple's market cap is $3.8T, and in the financial data table, it's $3.98T. Close enough, so use the more recent $3.98T.

Putting it all together, the summary table should include the latest numbers from the financial data. Pros and cons should reflect the strengths and risks from both sections. The recommendation should balance growth and safety, as per the user's task.

### **Summary Table: Tesla (TSLA), NVIDIA (NVDA), Apple (AAPL)**  
| **Metric**               | **Tesla (TSLA)**          | **NVIDIA (NVDA)**          | **Apple (AAPL)**          |
|--------------------------|---------------------------|----------------------------|---------------------------|
| **Stock Price**          | $376.30                   | $208.27                    | $271.06                   |
| **Market Cap**           | $1.41T                    | $5.06T                     | $3.98T                    |
| **P/E Ratio**            | 345.23 (High)             | 42.59                      | 34.35                     |
| **Profit Margin**        | 3.95%                     | 55.6%                      | 27.0%                     |
| **Annual Revenue**       | $97.9B                    | $215.9B                    | $435.6B                   |
| **Analyst Ratings**      | 5 Strong Buy / 18 Buy     | 9 Strong Buy / 47 Buy      | 7 Strong Buy / 24 Buy     |
| **Key Strengths**        | EV leadership, AI/energy  | AI/semiconductor dominance | Brand loyalty, ecosystem  |
| **Key Risks**            | Low margins, production   | Geopolitical, competition  | Market saturation         |

---

### **Pros and Cons for Long-Term Investment**  

#### **Tesla (TSLA)**  
**Pros**:  
- **Growth Catalysts**: EV market expansion, software (FSD, energy solutions), and AI innovation.  
- **Mission Appeal**: Strong ESG alignment for sustainability-focused investors.  
- **Valuation Potential**: Low profit margin suggests room for improvement if production scales.  

**Cons**:  
- **High P/E Ratio (345.23)**: May overvalue future growth assumptions.  
- **Operational Risks**: Supply chain bottlenecks, margin pressures, and competition from legacy automakers.  
- **Mixed Analyst Sentiment**: 4 "Sell" ratings and 18 "Hold" indicate execution concerns.  

---

#### **NVIDIA (NVDA)**  
**Pros**:  
- **AI/ML Dominance**: Leader in data center GPUs and AI infrastructure (CUDA ecosystem).  
- **High Profitability**: 55.6% profit margin and $215.9B revenue reflect strong pricing power.  
- **Bullish Analysts**: 56/60 "Buy" or "Strong Buy" ratings (9+47) signal confidence in AI growth.  

**Cons**:  
- **Geopolitical Exposure**: China’s AI chip self-reliance efforts could disrupt demand.  
- **Emerging Competition**: Custom chips from Google/Meta may erode margins in inference markets.  
- **High Valuation**: $5.06T market cap leaves less room for error.  

---

#### **Apple (AAPL)**  
**Pros**:  
- **Financial Fortress**: $3.98T market cap, $41.7B cash reserves, and 27% profit margin.  
- **Ecosystem Stickiness**: Premium iPhone demand, Apple Watch/Services revenue, and brand loyalty.  
- **Stable Growth**: Consistent "Buy" ratings and diversified product portfolio.  

**Cons**:  
- **Mature Market Saturation**: iPhone growth slowing in developed markets.  
- **Innovation Risks**: Lagging in AI advancements compared to peers like NVDA.  
- **Low Marginal Growth**: Revenue ($435.6B) may plateau without new product breakthroughs.  

---

### **Investment Recommendation**  

**1. Top Long-Term Buy: NVIDIA (NVDA)**  
   - **Justification**: NVIDIA is the **AI semiconductor growth engine**. Its dominance in data center GPUs (driven by generative AI demand), high profit margins, and strong analyst sentiment (9 "Strong Buy" ratings) justify its valuation. Analysts project a **36% upside by 2025**, making it a core holding for investors seeking exposure to AI-driven secular trends.  

**2. Safe Long-Term Buy: Apple (AAPL)**  
   - **Justification**: Apple offers **blue-chip stability** with $435.6B in revenue and $41.7B cash reserves. Its diversified ecosystem (iPhone, Services, Wearables) and consistent dividend yield make it a **defensive anchor** for portfolios. While growth may lag behind disruptors, its brand loyalty and financial resilience are unmatched.  

**3. High-Risk/High-Reward: Tesla (TSLA)**  
   - **Justification**: Tesla remains a **speculative play** for investors comfortable with volatility. Its EV leadership and AI software potential (e.g., FSD, energy solutions) could unlock long-term value. However, its **high P/E ratio (345.23)** and operational risks (production delays, margin pressures) make it a secondary allocation.  

---

### **Portfolio Strategy**  
- **Recommended Mix**: **50% AAPL (stability), 30% NVDA (growth), 20% TSLA (speculative upside)**.  
  - **Rationale**: Balances NVIDIA’s AI-driven growth, Apple’s defensive qualities, and Tesla’s disruptive potential. Adjust weights based on risk tolerance.  
- **Alternatives**: Consider reducing TSLA exposure if valuations rise further or geopolitical risks intensify.  

---

**Sources**:  
- Web Research Findings (2025): Tesla, NVIDIA, Apple financials and analyst ratings.  
- Financial Data (2025): Market caps, P/E ratios, and profit margins retrieved from stock market analysis platforms (e.g.,
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
