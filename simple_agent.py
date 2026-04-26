"""
Script Overview:
- Loads environment variables and validates `GROQ_API_KEY`.
- Creates a Groq chat model and a DuckDuckGo web-search tool.
- Builds a LangChain agent with a system prompt and tool access.
- Sends a user question to the agent and prints the final response.

Usage:
- Run this script directly to execute a sample query.
- Update the sample question in `main()` for quick testing.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import create_agent


# This example shows how to build a simple ReAct agent with ChatGroq and a web search tool.
def build_agent():
    load_dotenv()

    # Ensure the API key is available at runtime, otherwise the agent won't work.
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Missing GROQ_API_KEY. Please set it in .env or environment variables.")

    # LLM configuration - adjust model name and parameters as needed.
    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    )

    # Tool configuration - DuckDuckGo search for web queries. Adjust num_results as needed.
    ddg = DuckDuckGoSearchResults(num_results=5)
    tools = [ddg]

    # Prompt configuration - system prompt sets the agent's behavior, and we include a placeholder for the conversation history.
    system_prompt = (
        "You are an assistant. Reply based on the user question.\n"
        "Use web search when needed, and cite what you found.\n"
        "Format the final answer in Markdown."
    )

    # Create the agent with the specified LLM, tools, and system prompt.
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    return agent


# Helper function to run a question through the agent and print the final response.
def ask_question(agent, question: str):
    print("\n" + "=" * 90)
    print(f"USER: {question}\n")

    result = agent.invoke({"messages": [("user", question)]})
    final_msg = result["messages"][-1]

    print("ASSISTANT:\n")
    print(final_msg.content)


# Main function to build the agent and run a sample question.
def main():
    agent = build_agent()
    ask_question(agent, "Who won the India vs New Zealand finals in CT 2025?")


# Run the main function when this script is executed.
if __name__ == "__main__":
    main()
