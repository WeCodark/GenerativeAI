"""
NewsNinja - LangGraph Agent
Basically, here we are defining the StateGraph with 3 nodes only: parser, searcher, and writer.
Working fine only.
"""

from datetime import datetime
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from src.tools import search_news


# ---------------------------------------------------------------------------
# State Definition itself
# Actually, we need to declare the state first.
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State that is flowing through all the LangGraph nodes, you know."""
    topic: str               # Raw user topic only
    search_query: str        # Actually cleaned / optimized search query
    search_results: str      # Raw search output we are getting from Tavily
    digest: str              # Final polished news digest here


# ---------------------------------------------------------------------------
# Node Functions
# So these are the main functions we are executing nodes-wise.
# ---------------------------------------------------------------------------

def parse_topic(state: AgentState) -> dict:
    """Node 1 — Simply parse & refine the user's topic into an optimized search query itself."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    prompt = (
        "You are a search query optimizer. Given the following topic, create a concise, "
        "effective search query to find the latest news about it. "
        "Return ONLY the optimized search query, nothing else.\n\n"
        f"Topic: {state['topic']}"
    )

    response = llm.invoke(prompt)
    search_query = response.content.strip()
    print(f"\nOptimized search query: {search_query}")

    return {"search_query": search_query}


def search_web(state: AgentState) -> dict:
    """Node 2 — Basically, use the Tavily tool to search for news articles only."""
    query = state.get("search_query", state["topic"])
    print(f"\nSearching the web for: {query}")

    results = search_news.invoke({"query": query})
    print(f"Found {results.count('Article')} articles")

    return {"search_results": results}


def write_digest(state: AgentState) -> dict:
    """Node 3 — Use Groq LLM to synthesize a polished news digest, isn't it."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    prompt = (
        "You are NewsNinja, an elite AI news analyst. Based on the following search "
        "results, create a comprehensive yet concise news digest.\n\n"
        "FORMAT YOUR RESPONSE AS:\n"
        "**NewsNinja Digest**\n"
        f"**Topic:** {state['topic']}\n"
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n\n"
        "---\n\n"
        "Then provide:\n"
        "1. A brief **Executive Summary** (2-3 sentences)\n"
        "2. **Key Headlines** with bullet points\n"
        "3. **Analysis & Insights** (what this means)\n"
        "4. **Sources** (list the URLs)\n\n"
        "---\n\n"
        f"SEARCH RESULTS:\n{state['search_results']}"
    )

    response = llm.invoke(prompt)
    print("\nDigest written!")

    return {"digest": response.content}


# ---------------------------------------------------------------------------
# Graph Construction
# Finally, we are building the graph from start to end like this.
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the 3-node LangGraph StateGraph, that's it."""
    graph = StateGraph(AgentState)

    # Actually adding nodes here
    graph.add_node("parser", parse_topic)
    graph.add_node("searcher", search_web)
    graph.add_node("writer", write_digest)

    # Define edges (like a linear pipeline only)
    graph.set_entry_point("parser")
    graph.add_edge("parser", "searcher")
    graph.add_edge("searcher", "writer")
    graph.add_edge("writer", END)

    return graph.compile()
