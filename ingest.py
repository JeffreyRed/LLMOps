"""
ingest.py — Load Wikipedia dataset → chunk → embed → store in ChromaDB
Run once before using the agent: python src/ingest.py
"""

from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from rich.console import Console

from config import EMBEDDING_MODEL, CHROMA_DB_PATH, validate_config

console = Console()

# ── How many Wikipedia articles to ingest ─────────
NUM_ARTICLES = 200  # increase for a richer knowledge base


def load_wikipedia_docs(num: int = NUM_ARTICLES) -> list[str]:
    """Stream a slice of Wikipedia (English) from HuggingFace — free, no login needed."""
    console.print(f"[bold cyan]📥 Loading {num} Wikipedia articles...[/bold cyan]")
    dataset = load_dataset(
        "wikipedia",
        "20220301.en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    docs = []
    for i, item in enumerate(tqdm(dataset, total=num, desc="Fetching articles")):
        if i >= num:
            break
        docs.append(item["text"])
    console.print(f"[green]✅ Loaded {len(docs)} articles.[/green]")
    return docs


def chunk_documents(texts: list[str]) -> list[str]:
    """Split long articles into overlapping chunks for better retrieval."""
    console.print("[bold cyan]✂️  Chunking documents...[/bold cyan]")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for text in tqdm(texts, desc="Chunking"):
        chunks.extend(splitter.split_text(text))
    console.print(f"[green]✅ Created {len(chunks)} chunks.[/green]")
    return chunks


def build_vector_store(chunks: list[str]) -> Chroma:
    """Embed chunks with a local HuggingFace model and persist to ChromaDB."""
    console.print(
        f"[bold cyan]🧠 Loading embedding model: {EMBEDDING_MODEL}...[/bold cyan]"
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    console.print("[bold cyan]💾 Building ChromaDB vector store...[/bold cyan]")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )
    console.print(f"[green]✅ Vector store saved to: {CHROMA_DB_PATH}[/green]")
    return vectorstore


if __name__ == "__main__":
    validate_config()
    raw_docs = load_wikipedia_docs()
    chunks = chunk_documents(raw_docs)
    build_vector_store(chunks)
    console.print(
        "\n[bold green]🎉 Ingestion complete! You can now run the agent.[/bold green]"
    )
