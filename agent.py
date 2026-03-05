"""
agent.py — Agentic RAG using LangChain + Groq (LLaMA 3.3 70B)

The agent decides:
  1. Whether to retrieve documents or answer from its own knowledge
  2. What query to use for retrieval
  3. When it has enough context to answer

All decisions and traces are logged to LangSmith automatically.
"""
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from config import GROQ_API_KEY, LLM_MODEL, validate_config
from retriever import load_retriever_tool

console = Console()

SYSTEM_PROMPT = """You are a knowledgeable research assistant powered by a Wikipedia knowledge base.

When answering questions:
- Use the `search_wikipedia_knowledge_base` tool to look up factual information
- If the question is conversational or doesn't need facts (e.g. "hello", "thanks"), answer directly
- If retrieved documents are not relevant, say so honestly — don't fabricate information
- Always cite which topic/article the information came from when possible
- Be concise but thorough

You are part of an LLMOps demo pipeline. Your responses are being traced in LangSmith."""


def build_agent():
    """Construct the LangChain tool-calling agent with Groq LLM."""
    validate_config()

    # ── LLM via Groq ─────────────────────────────
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=0.1,       # Low temp for factual accuracy
        max_tokens=1024,
    )

    # ── Tools available to the agent ─────────────
    tools = [load_retriever_tool()]

    # ── Prompt template ───────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),   # where agent reasoning goes
    ])

    # ── Build agent ───────────────────────────────
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,           # prints agent reasoning steps
        max_iterations=5,       # prevent infinite loops
        handle_parsing_errors=True,
    )
    return agent_executor


def run_interactive_session():
    """Run a multi-turn chat session in the terminal."""
    console.print(Panel.fit(
        "[bold green]🤖 Agentic RAG Pipeline[/bold green]\n"
        "[dim]Powered by Groq (LLaMA 3.3 70B) + ChromaDB + LangSmith[/dim]\n"
        "Type [bold]exit[/bold] to quit.",
        border_style="green"
    ))

    agent = build_agent()
    chat_history = []

    while True:
        try:
            user_input = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        console.print("\n[bold yellow]Agent thinking...[/bold yellow]")

        try:
            result = agent.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })

            answer = result["output"]

            # Append to history for multi-turn context
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

            console.print(Panel(
                Markdown(answer),
                title="[bold green]Agent[/bold green]",
                border_style="green"
            ))

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    run_interactive_session()
