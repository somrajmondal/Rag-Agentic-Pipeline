"""
Quick CLI, no server needed.

Usage:
    python cli.py ingest ./data/documents
    python cli.py query "What does the contract say about termination?"
    python cli.py agent "What does the contract say about termination?"
"""
from __future__ import annotations

import sys

from app.agents.graph import run_agentic_query
from app.agents.tools import call_ai
from app.rag.pipeline import RAGPipeline


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command, arg = sys.argv[1], sys.argv[2]
    pipeline = RAGPipeline()

    if command == "ingest":
        n = pipeline.ingest_directory(arg)
        print(f"Ingested {n} chunks from '{arg}'.")

    elif command == "query":
        context = pipeline.retrieve_as_context(arg)
        system = "Answer using ONLY the provided context. Cite sources like [source: filename]."
        answer = call_ai(system, f"Question: {arg}\n\nContext:\n{context}")
        print("\n--- Answer ---\n")
        print(answer)

    elif command == "agent":
        result = run_agentic_query(arg)
        print("\n--- Final Answer ---\n")
        print(result.get("final_answer"))
        print(f"\n(sources: {result.get('sources')}, revisions: {result.get('revision_count')})")

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
