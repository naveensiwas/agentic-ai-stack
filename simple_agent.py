"""
Simple LangChain + Groq agent example.

What it does:
- Loads environment variables from `.env`.
- Validates that `GROQ_API_KEY` is set.
- Creates a `ChatGroq` model and a DuckDuckGo search tool.
- Builds an agent with a concise system prompt.
- Runs one sample question and prints the final assistant response.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_agent


# This example shows how to build a simple ReAct agent with ChatGroq and a web search tool.
def build_agent():
    """Create and return a configured LangChain agent."""
    load_dotenv()

    # Fail fast when credentials are missing.
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Missing GROQ_API_KEY. Please set it in .env or environment variables.")

    # Configure the Groq chat model.
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )

    # Add DuckDuckGo search as the external tool.
    ddg = DuckDuckGoSearchResults(num_results=5)
    tools = [ddg]

    # Keep instructions brief and response-focused.
    system_prompt = (
        "You are an assistant. Reply based on the user question.\n"
        "Use web search when needed, and cite what you found.\n"
        "Format the final answer in Markdown."
    )

    # Create the agent with model, tools, and prompt.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


# Helper function to run a question through the agent and print the final response.
def ask_question(agent, question: str):
    """Invoke the agent with one user question and print the final message."""
    print("\n" + "=" * 90)
    print(f"USER: {question}\n")

    result = agent.invoke({"messages": [("user", question)]})
    final_msg = result["messages"][-1]

    print("ASSISTANT:\n")
    print(final_msg.content)


# Main function to build the agent and run a sample question.
def main():
    """Build the agent and run a sample query."""
    agent = build_agent()
    ask_question(agent, "Who won the India vs New Zealand finals in CT 2025?")


# Run the main function when this script is executed.
if __name__ == "__main__":
    main()
