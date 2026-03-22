"""
NewsNinja - Main Entry Point
Basically, this is a CLI application that just takes a user topic, runs the LangGraph agent,
prints the digest, and optionally saves it to the outputs/ directory only.
"""

import os
import sys
from datetime import datetime




def save_digest(topic: str, digest: str) -> str:
    """Simply save the digest to the outputs/ directory with a timestamp itself."""
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "" for c in topic)
    safe_topic = safe_topic.replace(" ", "_")[:50]
    filename = f"outputs/{timestamp}_{safe_topic}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(digest)

    return filename


def main():
    """So basically run the NewsNinja agent here."""
    print("=" * 60)
    print("  Welcome to NewsNinja -- AI News Digest Agent")
    print("=" * 60)

    # Actually get topic from user (or command-line argument) like that
    topic = " ".join(sys.argv[1:]) or input("\nEnter a news topic to research: ").strip()
    if not topic:
        sys.exit("No topic provided. Exiting.")

    print(f"\nTopic: {topic}")
    print("-" * 60)

    # Then build and invoke the LangGraph agent, isn't it
    from src.agent import build_graph

    graph = build_graph()
    result = graph.invoke({"topic": topic})

    # Finally display the digest and see
    print("\n" + "=" * 60)
    print(result["digest"])
    print("=" * 60)

    # Optionally we can save to file also
    if input("\nSave digest to file? (y/n): ").strip().lower() in ("y", "yes"):
        print(f"Saved to: {save_digest(topic, result['digest'])}")

    print("\nThanks for using NewsNinja! Stay informed.")


if __name__ == "__main__":
    main()
