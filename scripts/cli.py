#!/usr/bin/env python
"""RAG system command-line interface.

Usage:
    python scripts/cli.py build <document_path>  [--store-path DIR]
    python scripts/cli.py chat                    [--store-path DIR]
    python scripts/cli.py evaluate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.core.config import get_settings


def _setup_logging(settings):
    fmt = "%(levelname)s: %(message)s"
    if not settings.logging.json_format:
        logging.basicConfig(level=settings.logging.level, format=fmt)
    else:
        logging.basicConfig(level=settings.logging.level)
    logging.getLogger("pdfminer").setLevel(logging.ERROR)
    logging.getLogger("tabula").setLevel(logging.ERROR)


# ------------------------------------------------------------------------ cmd

def _build(args):
    """Build a vector store index from one or more documents."""
    settings = get_settings()
    _setup_logging(settings)

    from rag.chatbot import build_index

    doc_path = args.document_path
    store_path = args.store_path or settings.vector_store.path

    print(f"Building index from: {doc_path}")
    summary = build_index(doc_path, store_path=store_path, settings=settings)
    print("Index built successfully.")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def _chat(args):
    """Start an interactive chat session using the local index."""
    import uuid

    settings = get_settings()
    _setup_logging(settings)

    from rag.chatbot import get_chatbot

    store_path = args.store_path or settings.vector_store.path
    chatbot = get_chatbot(store_path=store_path)
    conversation_id = args.conversation_id or "cli-" + uuid.uuid4().hex
    print("RAG Chatbot (type 'quit' to exit)")
    print(f"Conversation: {conversation_id}")
    print("-" * 50)

    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break
            if not question:
                continue

            result = chatbot.chat(
                question,
                conversation_id=conversation_id,
                project_id=args.project_id,
            )
            if result['success']:
                print(f"\nBot: {result['answer']}\n")
                print(f"Sources: {result['num_chunks']} chunks used")
            else:
                print(f"\nError: {result['error']}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
    finally:
        chatbot.close()


def _evaluate(args):
    """Run the evaluation suite."""
    from scripts.evaluate import main as eval_main

    sys.exit(eval_main())


# --------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = sub.add_parser("build", help="Build vector store from documents")
    build_p.add_argument("document_path", help="PDF/DOCX/TXT file or directory")
    build_p.add_argument("--store-path", help="Override vector store snapshot path")

    # chat
    chat_p = sub.add_parser("chat", help="Interactive chat session")
    chat_p.add_argument("--store-path", help="Vector store snapshot path")
    chat_p.add_argument("--conversation-id", help="Reuse a conversation id (default: new)")
    chat_p.add_argument("--project-id", help="Scope retrieval to a project")

    # evaluate
    sub.add_parser("evaluate", help="Run evaluation suite")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {"build": _build, "chat": _chat, "evaluate": _evaluate}
    dispatch[args.command](args)


if __name__ == "__main__":
    main()