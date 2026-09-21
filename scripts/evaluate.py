"""
Run RAG Evaluation Framework

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

_parent = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_parent / "src"))
sys.path.insert(0, str(_parent))

from tools.evaluation import TEST_CASES, RAGEvaluator  # noqa: E402 - src-layout bootstrap above

from rag.chatbot import get_chatbot  # noqa: E402 - src-layout bootstrap above


def main() -> int:
    """Run complete RAG evaluation."""
    print("\n" + "=" * 70)
    print("LOADING RAG SYSTEM")
    print("=" * 70)

    # Load chatbot
    try:
        chatbot = get_chatbot()
        print("Chatbot loaded successfully")
    except Exception as e:
        print(f"Failed to load chatbot: {e}")
        print("\nMake sure to:")
        print("  1. Build index: python scripts/cli.py build <document_path>")
        print("  2. Set API key in .env file")
        return 1

    # Initialize evaluator
    evaluator = RAGEvaluator(chatbot)

    # Run evaluation
    results = evaluator.evaluate(cast(list[dict[str, Any]], TEST_CASES))

    # Save results
    evaluator.save_results(results)

    # Print final verdict
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    overall_score = results['overall']['overall_score']
    grade = results['overall']['grade']

    if overall_score >= 0.80:
        print(f"\nEXCELLENT! Your RAG system scored {overall_score:.3f} (Grade {grade})")
        print("   Ready for production deployment!")
    elif overall_score >= 0.70:
        print(f"\nGOOD! Your RAG system scored {overall_score:.3f} (Grade {grade})")
        print("   Minor improvements recommended before production.")
    else:
        print(f"\nNEEDS WORK! Your RAG system scored {overall_score:.3f} (Grade {grade})")
        print("   Significant improvements needed.")

    print("\nDetailed results saved to: evaluation_results.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())