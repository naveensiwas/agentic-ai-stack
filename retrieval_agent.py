"""
Retrieval-augmented Markdown assistant using LangChain + Groq + LanceDB.

What it does:
- Loads environment variables from `.env` and validates `GROQ_API_KEY`.
- Loads an existing LanceDB table when available; otherwise ingests `markdown-guide-sample.pdf`.
- Exposes two tools: local KB retrieval and web fallback search.
- Runs sample Markdown questions and prints the final response.
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

# Load environment variables (for example, GROQ_API_KEY).
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "Missing GROQ_API_KEY. Set it in your environment or .env file."
    )

# ── Configuration ────────────────────────────────────────────────────────────────────
# Adjust these paths and settings as needed for your environment and PDF source.
# The sample PDF used here is from the Markdown Guide, which provides a variety of formatting examples.
# Source PDF: https://www.markdownguide.org/assets/markdown-guide-sample.pdf

DATA_DIR = Path(".")
PDF_PATH = DATA_DIR / "markdown-guide-sample.pdf"  # Keep markdown-guide-sample.pdf in project root.
LANCEDB_URI = str(DATA_DIR / "lancedb")
TABLE_NAME = "markdown_guide"

# Embedding model with a good speed/quality tradeoff.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ── Build or Load Vector DB ───────────────────────────────────────────────────────────
# Build or load the vector database from the Markdown Guide PDF.
# This runs at startup and powers local KB retrieval.

def build_or_load_vectordb() -> LanceDB:
    """Load an existing LanceDB table when available; otherwise ingest the PDF."""
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at: {PDF_PATH.resolve()}\n"
            "Please place 'markdown-guide-sample.pdf' in the project root folder."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

    # Reuse existing table if present to avoid rebuilding vectors every run.
    db_root = Path(LANCEDB_URI)
    table_dir = db_root / f"{TABLE_NAME}.lance"
    if table_dir.exists():
        print(f"Loading existing vector DB table: {table_dir}")
        return LanceDB(
            embedding=embeddings,
            uri=LANCEDB_URI,
            table_name=TABLE_NAME,
        )

    print(f"Loading PDF from: {PDF_PATH.resolve()}")
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF.")

    # Chunk settings balance retrieval quality and context size.
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Ingest chunks into the configured LanceDB table.
    vector_db = LanceDB.from_documents(
        documents=chunks,
        embedding=embeddings,
        uri=LANCEDB_URI,
        table_name=TABLE_NAME,
    )
    print("Vector DB ready.")
    return vector_db


# Initialize vector DB and retriever at startup.
vector_db = build_or_load_vectordb()
retriever = vector_db.as_retriever(search_kwargs={"k": 5})


# ── Tool ────────────────────────────────────────────────────────────────────
# Local KB tool for Markdown Guide retrieval.

@tool("markdown_kb_search")
def markdown_kb_search(query: str) -> str:
    """Search the local Markdown Guide knowledge base and return top snippets."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant knowledge-base results found."

    # Return concise, labeled snippets for model context.
    out = []
    for i, d in enumerate(docs, 1):
        text = d.page_content.strip().replace("\n", " ")
        out.append(f"[KB-{i}] {text[:900]}")
    return "\n".join(out)


# Web fallback when local KB coverage is insufficient.
ddg = DuckDuckGoSearchResults(num_results=5)

# Configure the Groq chat model.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=60,
    max_retries=2
)

# Keep tool usage and answer format explicit.
system_prompt = """You are a Markdown documentation assistant.
Rules:
1) Prefer the knowledge base tool `markdown_kb_search` for questions about Markdown syntax, formatting, and examples from the Markdown Guide PDF.
2) If the knowledge base does not contain enough information, use DuckDuckGo search to fill gaps.
3) Clearly separate what comes from the knowledge base vs the web when you answer.
4) Keep answers concise, practical, and structured. Use Markdown.
"""

# Build the agent with local KB + web fallback tools.
tools = [markdown_kb_search, ddg]
agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)


# Helper function to run a question through the agent and print the final response.
def ask_question(question: str):
    """Invoke the agent with one question and print the final response."""
    print("\n" + "=" * 90)
    print(f"USER: {question}\n")

    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    final_msg = result["messages"][-1]

    print("ASSISTANT:\n")
    print(final_msg.content)


def main() -> None:
    """Run a quick smoke test with two Markdown Guide questions."""
    ask_question("How do I create headings, bold text, and bullet lists in Markdown?")
    ask_question("What is the difference between inline code and fenced code blocks in Markdown?")


if __name__ == "__main__":
    main()
