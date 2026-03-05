"""
retriever.py — Loads the persisted ChromaDB and exposes it as a LangChain tool
the agent can decide to call.
"""

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool

from config import EMBEDDING_MODEL, CHROMA_DB_PATH


def load_retriever_tool():
    """
    Returns a LangChain Tool wrapping the ChromaDB retriever.
    The agent autonomously decides when to call this tool.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance — diverse results
        search_kwargs={"k": 5},  # Return top 5 chunks
    )

    # Wrap as a named tool the agent can invoke
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="search_wikipedia_knowledge_base",
        description=(
            "Search the Wikipedia knowledge base for factual information. "
            "Use this tool when the user asks a factual question about science, "
            "history, people, places, or events. "
            "Input should be a clear search query."
        ),
    )
    return retriever_tool
