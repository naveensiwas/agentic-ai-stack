"""
Script overview
---------------
This module runs a 3-step multi-agent investment workflow:
1) Web agent gathers qualitative market/news context via DuckDuckGo.
2) Finance agent gathers structured stock metrics via yfinance tools.
3) Supervisor synthesizes both outputs into a final recommendation.

Tech stack: LangChain agents, LangGraph orchestration, Groq Chat model.
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

# Initialize environment variables from .env (e.g., GROQ_API_KEY).
load_dotenv()

# Shared LLM for all agents in this script.
llm = ChatGroq(
    model="qwen-2.5-32b",
    api_key=os.getenv("GROQ_API_KEY"),
)

# ── Tools ─────────────────────────────────────────────────────────────────────

# Web search utility wrapped as a tool callable by the web agent.
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

# Web agent focuses on qualitative signal collection from public web sources.
web_agent = create_agent(
    model=llm,
    tools=[search_web],
    system_prompt=(
        "You are a Web Research Agent. Search the web for the latest news, "
        "sentiment, and qualitative insights about the companies provided. "
    )
)

# Finance agent focuses on quantitative retrieval and tabular presentation.
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

class AgentTeamState(TypedDict):
    """Shared graph state passed node-to-node during orchestration."""
    # Message history merged across nodes; operator.add appends lists.
    messages: Annotated[List, operator.add]
    # Output from web research node.
    web_research: str
    # Output from finance analysis node.
    financial_data: str
    # Final synthesized recommendation from supervisor node.
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

builder = StateGraph(AgentTeamState)

builder.add_node("web_agent", web_agent_node)
builder.add_node("finance_agent", finance_agent_node)
builder.add_node("supervisor", supervisor_node)

# Both agents run first (sequentially), then supervisor synthesizes
builder.set_entry_point("web_agent")
builder.add_edge("web_agent", "finance_agent")
builder.add_edge("finance_agent", "supervisor")
builder.add_edge("supervisor", END)

graph = builder.compile()


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo query; replace with user input for interactive use.
    query = "Analyze companies like Tesla, NVDA, Apple and suggest which to buy for long term"

    print("=" * 70)
    print("🤖 Multi-Agent Investment Analyst (LangChain + LangGraph + Groq)")
    print("=" * 70)
    print(f"\n📌 Query: {query}\n")

    # Seed initial graph state and execute synchronously.
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
