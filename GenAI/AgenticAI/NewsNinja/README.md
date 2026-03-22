# NewsNinja

**AI-powered news digest agent** built with **LangGraph**, **Groq**, and **Tavily**.

Give it any topic, it searches the web for the latest articles, and delivers a polished, structured news digest in seconds.

---

## Table of Contents

- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Code Walkthrough](#code-walkthrough)
  - [tools.py -- Search Tool](#toolspy----search-tool)
  - [agent.py -- LangGraph Agent](#agentpy----langgraph-agent)
  - [main.py -- Entry Point](#mainpy----entry-point)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [API Keys](#api-keys)
- [License](#license)

---

## Architecture

NewsNinja uses **LangGraph's StateGraph** to create a 3-node sequential pipeline. Each node performs one step of the digest generation process:

```
+---------------+     +---------------+     +---------------+
|  Parse Topic  | --> |  Search Web   | --> | Write Digest  |
|   (Node 1)    |     |   (Node 2)    |     |   (Node 3)    |
+---------------+     +---------------+     +---------------+
       |                      |                      |
  Groq LLM refines      Tavily API fetches      Groq LLM writes
  user's raw topic       top 5 articles         final digest
  into a search query    from the web           from search results
```

A shared **AgentState** dictionary flows through all three nodes. Each node reads from the state, does its work, and writes its output back into the state for the next node.

---

## How It Works

1. **User enters a topic** (e.g., "Cricket", "AI startups", "climate change").
2. **Node 1 (Parser)** -- The Groq LLM takes the raw topic and rewrites it into an optimized search query designed to surface the most relevant, recent news.
3. **Node 2 (Searcher)** -- The optimized query is sent to the Tavily Search API, which returns the top 5 articles with titles, URLs, and content snippets.
4. **Node 3 (Writer)** -- The Groq LLM reads all the search results and synthesizes them into a structured news digest with an executive summary, key headlines, analysis, and source links.
5. **Output** -- The digest is printed to the console and optionally saved as a timestamped `.txt` file in the `outputs/` folder.

---

## Project Structure

```
NewsNinja/
├── .env.example          # Template for required API keys
├── .env                  # Your actual API keys (git-ignored)
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── src/
│   ├── __init__.py       # Makes src a Python package
│   ├── tools.py          # Tavily search tool definition
│   ├── agent.py          # LangGraph state, nodes, and graph
│   └── main.py           # CLI entry point
└── outputs/              # Where saved digests are stored
```

---

## Code Walkthrough

### tools.py -- Search Tool

This file defines a single LangChain tool that wraps the Tavily Search API.

**Key components:**

- **`@tool` decorator** -- Registers `search_news` as a LangChain-compatible tool so it can be used by agents and invoked programmatically.
- **`TavilySearchResults`** -- Configured with:
  - `max_results=5` -- Returns the top 5 most relevant articles.
  - `search_depth="advanced"` -- Uses Tavily's deeper search mode for higher quality results.
  - `include_answer=True` -- Includes Tavily's AI-generated answer alongside raw results.
- **Result formatting** -- Each result is formatted into a readable block with article number, title, URL, and a content summary. If no results are found, a fallback message is returned.

```python
@tool
def search_news(query: str) -> str:
    search = TavilySearchResults(max_results=5, search_depth="advanced", include_answer=True)
    results = search.invoke({"query": query})
    # Formats each result as: Article N: Title, URL, Summary
```

---

### agent.py -- LangGraph Agent

This is the core of the application. It defines the state schema, three processing nodes, and the graph that connects them.

**1. AgentState (TypedDict)**

A typed dictionary that acts as the shared state flowing through the graph:

| Field | Type | Purpose |
|-------|------|---------|
| `topic` | `str` | The raw topic entered by the user |
| `search_query` | `str` | The LLM-optimized search query |
| `search_results` | `str` | Formatted results from Tavily |
| `digest` | `str` | The final polished news digest |

**2. Node Functions**

Each node is a plain Python function that receives the current state and returns a dictionary with updated fields:

- **`parse_topic(state)`** -- Uses Groq's `llama-3.3-70b-versatile` model (temperature=0 for deterministic output) to convert the user's raw topic into a concise, effective search query. For example, "Cricket" becomes `"cricket news" OR "cricket updates" site:bbc.com OR site:espn.com`.

- **`search_web(state)`** -- Takes the optimized `search_query` from state and passes it to the `search_news` tool. Falls back to the raw `topic` if `search_query` is missing. Returns the formatted articles as `search_results`.

- **`write_digest(state)`** -- Uses Groq's `llama-3.3-70b-versatile` model (temperature=0.7 for creative writing) with a detailed prompt that instructs the LLM to produce a structured digest containing:
  - An Executive Summary (2-3 sentences)
  - Key Headlines as bullet points
  - Analysis and Insights
  - Source URLs
  - The actual current date (via `datetime.now()`)

**3. build_graph()**

Constructs and compiles the LangGraph `StateGraph`:

```python
graph = StateGraph(AgentState)
graph.add_node("parser", parse_topic)      # Node 1
graph.add_node("searcher", search_web)     # Node 2
graph.add_node("writer", write_digest)     # Node 3
graph.set_entry_point("parser")
graph.add_edge("parser", "searcher")       # parser -> searcher
graph.add_edge("searcher", "writer")       # searcher -> writer
graph.add_edge("writer", END)              # writer -> done
return graph.compile()
```

The `compile()` call validates the graph structure and returns a runnable object that can be invoked with an initial state.

---

### main.py -- Entry Point

The CLI application that ties everything together.

**Key functions:**

- **`validate_env()`** -- Checks that both `GROQ_API_KEY` and `TAVILY_API_KEY` are present in the environment. If either is missing, it prints a helpful error message and exits. This prevents confusing API errors later in the pipeline.

- **`save_digest(topic, digest)`** -- Saves the digest to `outputs/` with a filename format of `YYYYMMDD_HHMMSS_TopicName.txt`. The topic is sanitized (special characters removed, spaces replaced with underscores, truncated to 50 chars) to create a safe filename.

- **`main()`** -- The main execution flow:
  1. Loads `.env` via `python-dotenv`
  2. Validates API keys
  3. Accepts topic from command-line args (`python -m src.main "AI news"`) or interactive prompt
  4. Calls `build_graph()` to construct the LangGraph pipeline
  5. Invokes the graph with `{"topic": topic}` -- this runs all 3 nodes in sequence
  6. Prints the final digest
  7. Offers to save the output to a file

```python
graph = build_graph()
result = graph.invoke({"topic": topic})   # Runs parser -> searcher -> writer
print(result["digest"])                   # Final output
```

---

## Tech Stack

| Package | Version | Purpose |
|---------|---------|---------|
| **langchain** | >= 0.3.0 | Core framework for LLM application development |
| **langgraph** | >= 0.2.0 | State machine framework for building multi-step AI agents |
| **langchain-groq** | >= 0.2.0 | LangChain integration for Groq's ultra-fast LLM inference |
| **tavily-python** | >= 0.5.0 | Search API client for fetching real-time web results |
| **python-dotenv** | >= 1.0.0 | Loads environment variables from `.env` files |

**Why these choices?**

- **Groq** -- Provides blazing-fast inference on open-source models (Llama 3.3 70B). Free tier is generous enough for development and light usage.
- **Tavily** -- Purpose-built search API for AI agents. Returns clean, structured results optimized for LLM consumption (unlike raw Google results).
- **LangGraph** -- Gives us a clear, visual pipeline architecture with typed state. Makes it easy to add new nodes (e.g., a fact-checker or translator) without restructuring the code.

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/NewsNinja.git
cd NewsNinja
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
cp .env.example .env
```
Open `.env` and add your keys:
```
GROQ_API_KEY=gsk_your_groq_key_here
TAVILY_API_KEY=tvly-your_tavily_key_here
```

---

## Usage

### Interactive Mode
```bash
python -m src.main
```
You will be prompted to enter a news topic.

### Command-Line Mode
```bash
python -m src.main "Artificial Intelligence startups 2026"
```
The topic is passed directly as arguments -- no prompt needed.

### Sample Output
```
============================================================
  Welcome to NewsNinja -- AI News Digest Agent
============================================================

Enter a news topic to research: Cricket

Topic: Cricket
------------------------------------------------------------

Optimized search query: "cricket news" OR "cricket updates"

Searching the web for: "cricket news" OR "cricket updates"
Found 5 articles

Digest written!

============================================================
**NewsNinja Digest**
**Topic:** Cricket
**Date:** March 17, 2026

---

**Executive Summary**
The cricketing world is buzzing with action as ...

**Key Headlines**
- India dominates in T20 World Cup qualifier ...
- England announces squad changes for upcoming series ...
...

**Sources**
- https://www.bbc.com/sport/cricket/...
- https://www.espn.com/cricket/...
============================================================

Save digest to file? (y/n): y
Saved to: outputs/20260317_143000_Cricket.txt

Thanks for using NewsNinja! Stay informed.
```

---

## API Keys

| Service | Get Your Key | Free Tier |
|---------|-------------|-----------|
| **Groq** | [console.groq.com](https://console.groq.com) | Generous free tier |
| **Tavily** | [tavily.com](https://tavily.com) | 1,000 free searches/month |

---

## License

MIT License -- feel free to use, modify, and distribute.
