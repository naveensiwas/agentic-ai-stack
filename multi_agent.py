"""
Multi-agent investment workflow using LangChain + LangGraph + Groq.

What it does:
- Loads environment variables from `.env`.
- Creates one shared `ChatGroq` model for all agents.
- Defines tools for web search and stock data retrieval.
- Builds two specialized agents (web research + finance analysis).
- Orchestrates both agents with LangGraph and synthesizes a final recommendation.
- Runs one sample investment query and prints the final response.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from typing import TypedDict, Annotated, List
import operator
import yfinance as yf
import json

# Load environment variables from .env (for example, GROQ_API_KEY).
load_dotenv()

# Create one shared LLM instance used by all agents.
llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Tools ─────────────────────────────────────────────────────────────────────
# Tools are defined as functions decorated with @tool.
# - They can be invoked by agents during their reasoning process.
# - Here we define tools for web search and stock data retrieval.


# Shared web search utility used by the web tool.
web_search = DuckDuckGoSearchRun()


@tool
def search_web(query: str) -> str:
    """Search the web for recent company news and qualitative context."""
    return web_search.run(query)


@tool
def get_stock_price(ticker: str) -> str:
    """Return latest close price for a ticker (best-effort via yfinance)."""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1d")
    if hist.empty:
        return f"Could not fetch price for {ticker}"
    price = hist["Close"].iloc[-1]
    return f"{ticker} current price: ${price:.2f}"


@tool
def get_stock_fundamentals(ticker: str) -> str:
    """Return core valuation and performance fundamentals as JSON text."""
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamentals = {
        "ticker": ticker,
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "eps": info.get("trailingEps"),
        "revenue": info.get("totalRevenue"),
        "profit_margin": info.get("profitMargins"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }
    return json.dumps(fundamentals, indent=2)


@tool
def get_analyst_recommendations(ticker: str) -> str:
    """Return the most recent analyst recommendation rows for a ticker."""
    stock = yf.Ticker(ticker)
    recs = stock.recommendations
    if recs is None or recs.empty:
        return f"No analyst recommendations found for {ticker}"
    latest = recs.tail(5).to_string()
    return f"Latest analyst recommendations for {ticker}:\n{latest}"


@tool
def get_company_info(ticker: str) -> str:
    """Return high-level company profile fields for a ticker as JSON text."""
    stock = yf.Ticker(ticker)
    info = stock.info
    summary = {
        "name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "employees": info.get("fullTimeEmployees"),
        "summary": info.get("longBusinessSummary", "")[:400] + "...",
    }
    return json.dumps(summary, indent=2)


# ── Agents ────────────────────────────────────────────────────────────────────
# Agents are created with `create_agent`, which takes an LLM, a list of tools, and a system prompt.
# - Each agent can invoke its tools during reasoning, and the system prompt guides its behavior and response formatting.
# - Here we create two agents: one focused on qualitative web research, and another focused on quantitative financial data retrieval.
# - The supervisor node later will synthesize outputs from both agents to generate a final investment recommendation.

# Create specialist agents for qualitative and quantitative analysis.
web_agent = create_agent(
    model=llm,
    tools=[search_web],
    system_prompt=(
        "You are a Web Research Agent. Search the web for the latest news, "
        "sentiment, and qualitative insights about the companies provided. "
    )
)

finance_agent = create_agent(
    model=llm,
    tools=[get_stock_price, get_stock_fundamentals, get_analyst_recommendations, get_company_info],
    system_prompt=(
        "You are a Finance Agent. Retrieve stock prices, fundamentals, analyst "
        "recommendations, and company information for the requested tickers. "
        "Present all data in well-formatted markdown tables."
    )
)


# ── LangGraph State & Orchestrator ────────────────────────────────────────────
# We define a shared state structure `AgentTeamState` that will be passed between graph nodes.
# - Each node (agent) will read from this state, perform its task, and write its output back to the state for the next node to consume.
# - The graph orchestrator will manage the flow of execution between the web agent, finance agent, and supervisor node,
#   ensuring that outputs are combined effectively to produce a final recommendation.
# - This structure allows us to maintain a clear separation of concerns while enabling collaboration between specialized agents in a cohesive workflow.

# Define shared state passed between graph nodes.
class AgentTeamState(TypedDict):
    """Shared graph state passed node-to-node during orchestration."""
    # Merge message history across nodes.
    messages: Annotated[List, operator.add]

    # Output from the web research node.
    web_research: str

    # Output from the finance analysis node.
    financial_data: str

    # Final synthesized recommendation from the supervisor node.
    final_response: str


def web_agent_node(state: AgentTeamState) -> AgentTeamState:
    """Run the web agent and persist qualitative findings into graph state."""
    print("\n🌐 [Web Agent] Searching the web for latest insights...\n")
    user_query = state["messages"][-1].content
    result = web_agent.invoke({"messages": [HumanMessage(content=user_query)]})
    web_output = result["messages"][-1].content
    return {"web_research": web_output, "messages": [AIMessage(content=f"**Web Agent:**\n{web_output}")]}


def finance_agent_node(state: AgentTeamState) -> AgentTeamState:
    """Run the finance agent and persist quantitative findings into graph state."""
    print("\n📈 [Finance Agent] Fetching financial data...\n")
    user_query = state["messages"][0].content
    result = finance_agent.invoke({"messages": [HumanMessage(content=user_query)]})
    finance_output = result["messages"][-1].content
    return {"financial_data": finance_output, "messages": [AIMessage(content=f"**Finance Agent:**\n{finance_output}")]}


def supervisor_node(state: AgentTeamState) -> AgentTeamState:
    """Combine prior node outputs into one investment recommendation."""
    print("\n🧠 [Supervisor] Synthesizing findings and generating final recommendation...\n")

    synthesis_prompt = f"""
You are a senior investment analyst and team supervisor.

You have received research from two specialized agents. Synthesize their findings
and provide a comprehensive investment recommendation.

## Web Research Findings:
{state.get("web_research", "N/A")}

## Financial Data & Analysis:
{state.get("financial_data", "N/A")}

## Task:
Based on the above, analyze Tesla (TSLA), NVIDIA (NVDA), and Apple (AAPL) and provide:
1. A **summary table** comparing the three companies across key metrics.
2. **Pros and cons** for long-term investment in each company.
3. A clear **recommendation** on which stock(s) to buy for the long term with justification.

Always include sources and use markdown tables where applicable.
"""
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    return {"final_response": response.content, "messages": [AIMessage(content=response.content)]}


# ── Build the Graph ───────────────────────────────────────────────────────────
# We construct a LangGraph `StateGraph` to orchestrate the flow between our agents.
# - Each node corresponds to one of our agents or the supervisor.
# - Edges define the execution order: web agent → finance agent → supervisor → END.
# - The graph will manage the state transitions and ensure that outputs from each node are available to subsequent nodes,
#   enabling a cohesive multi-agent workflow that culminates in a final investment recommendation.
# - This structure allows us to maintain modularity while facilitating collaboration between specialized agents in a clear and organized manner.

builder = StateGraph(AgentTeamState)

builder.add_node("web_agent", web_agent_node)
builder.add_node("finance_agent", finance_agent_node)
builder.add_node("supervisor", supervisor_node)

# Run web -> finance -> supervisor, then finish.
builder.set_entry_point("web_agent")
builder.add_edge("web_agent", "finance_agent")
builder.add_edge("finance_agent", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()

# ── Run ───────────────────────────────────────────────────────────────────────
# Finally, we run a sample query through the graph when the script is executed directly.
# - The initial state contains the user's query as the first message.
# - The graph orchestrator will manage the execution flow, passing state between nodes and ultimately producing
#   a final recommendation that is printed to the console.
# - This allows us to see the multi-agent collaboration in action, with each agent contributing its expertise to the final output.
# - In a real application, you could replace the hardcoded query with user input for an interactive experience.

# Run a sample analysis when the script is executed directly.
if __name__ == "__main__":
    # Sample query; replace with user input for interactive use.
    query = "Analyze companies like Tesla, NVDA, Apple and suggest which to buy for long term"

    print("=" * 70)
    print("🤖 Multi-Agent Investment Analyst (LangChain + LangGraph + Groq)")
    print("=" * 70)
    print(f"\n📌 Query: {query}\n")

    # Seed initial graph state and execute the workflow.
    initial_state: AgentTeamState = {
        "messages": [HumanMessage(content=query)],
        "web_research": "",
        "financial_data": "",
        "final_response": "",
    }

    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 70)
    print("📊 FINAL RECOMMENDATION")
    print("=" * 70)
    print(final_state["final_response"])
