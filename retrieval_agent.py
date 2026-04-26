"""
Overview:
- Loads Thai recipe content from `ThaiRecipes.pdf`.
- Splits PDF pages into chunks, embeds them, and stores vectors in LanceDB.
- Exposes two tools to the agent:
  1) `thai_recipe_kb_search` for local PDF-backed retrieval.
  2) DuckDuckGo search for web fallback.
- Runs two sample user questions through a Groq-hosted chat model.

Notes:
- Requires `GROQ_API_KEY` in environment or `.env`.
- Re-ingests vector data on startup via `LanceDB.from_documents(...)`.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_agent

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load local environment variables (e.g., GROQ_API_KEY).
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "Missing GROQ_API_KEY. Set it in your environment or .env file."
    )


# Configuration
# This section defines paths, model names, and the function to build/load the vector database.
# Source PDF: https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf
DATA_DIR = Path(".")
PDF_PATH = DATA_DIR / "ThaiRecipes.pdf"   # Keep ThaiRecipes.pdf in the project root.
LANCEDB_URI = str(DATA_DIR / "lancedb")
TABLE_NAME = "recipes"


# Lightweight embedding model with good speed/quality tradeoff.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Build or load the vector database from the PDF.
# This runs at startup and is used by the `thai_recipe_kb_search` tool.
def build_or_load_vectordb() -> LanceDB:
    """Load PDF, chunk text, embed chunks, and persist to LanceDB."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at: {PDF_PATH.resolve()}\n"
            "Please place 'ThaiRecipes.pdf' in the project root folder."
        )

    print(f"Loading PDF from: {PDF_PATH.resolve()}")
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")

    # Chunk settings balance retrieval recall and context size.
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Ingest chunks into the configured LanceDB table.
    vector_db = LanceDB.from_documents(
        documents=chunks,
        embedding=embeddings,
        uri=LANCEDB_URI,
        table_name=TABLE_NAME,
    )
    print("Vector DB ready.")
    return vector_db


# Initialize the vector database and retriever at startup.
vector_db = build_or_load_vectordb()
retriever = vector_db.as_retriever(search_kwargs={"k": 5})


# The `thai_recipe_kb_search` tool queries the local PDF vector database for relevant chunks.
@tool("thai_recipe_kb_search")
def thai_recipe_kb_search(query: str) -> str:
    """
    Search the local Thai recipe PDF for relevant chunks.
    Prefer this tool first for Thai cuisine and recipe questions.
    """
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant knowledge-base results found."

    # Return concise, labeled snippets for model context.
    out = []
    for i, d in enumerate(docs, 1):
        text = d.page_content.strip().replace("\n", " ")
        out.append(f"[KB-{i}] {text[:900]}")
    return "\n".join(out)


# Web fallback when local KB evidence is insufficient.
ddg = DuckDuckGoSearchResults(num_results=5)


# Configure the Groq LLM with desired settings.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=60,
    max_retries=2
)


# The system prompt guides the agent's behavior, emphasizing tool usage and answer formatting.
system_prompt = """You are a Thai cuisine expert.
Rules:
1) Prefer the knowledge base tool `thai_recipe_kb_search` for Thai recipes and Thai cuisine questions.
2) If the knowledge base does not contain enough information, use DuckDuckGo search to fill gaps.
3) Clearly separate what comes from the knowledge base vs the web when you answer.
4) Keep answers concise, practical, and structured. Use Markdown.
"""


# Create the agent with the Groq LLM and both tools.
tools = [thai_recipe_kb_search, ddg]
agent = create_agent(llm, tools)


# Helper function to run a question through the agent and print the final response.
def ask_question(question: str):
    """Run one question through the agent and print the final assistant reply."""
    print("\n" + "=" * 90)
    print(f"USER: {question}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_msg = result["messages"][-1]

    print("ASSISTANT:\n")
    print(final_msg.content)

def main() -> None:
    """Quick smoke run with two sample Thai cuisine queries."""
    ask_question("How do I make chicken and galangal in coconut milk soup?")
    ask_question("What is the history of Thai curry?")


if __name__ == "__main__":
    main()
