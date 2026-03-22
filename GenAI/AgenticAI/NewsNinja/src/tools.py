"""
NewsNinja - Tavily Search Tool
Basically, this defines the search tool we are using for the LangGraph agent to fetch news articles only.
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


@tool
def search_news(query: str) -> str:
    """Search the web for the latest news articles on a given topic, simply.

    Args:
        query: Basically the search query string for finding relevant news articles.

    Returns:
        A formatted string of search results with titles, URLs, and snippets like that.
    """
    search = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
    )

    results = search.invoke({"query": query})

    if not results:
        return "No results found for the given query."

    # Format results into a readable string and send back
    formatted = []
    for i, result in enumerate(results, 1):
        title = result.get("title", "No Title")
        url = result.get("url", "")
        content = result.get("content", "No content available.")
        formatted.append(
            f"**Article {i}:** {title}\n"
            f"   URL: {url}\n"
            f"   Summary: {content}\n"
        )

    return "\n".join(formatted)
