"""
config.py — Loads all settings from .env
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# ── Embeddings & Vector Store ─────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

# ── LangSmith Observability ───────────────────────
# These are read automatically by LangChain if set in .env:
# LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT


def validate_config():
    """Raise early if required keys are missing."""
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Copy .env.example → .env and fill in your keys."
        )
    print("✅ Config loaded successfully.")
    print(f"   LLM model      : {LLM_MODEL}")
    print(f"   Embedding model: {EMBEDDING_MODEL}")
    print(f"   ChromaDB path  : {CHROMA_DB_PATH}")
