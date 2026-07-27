"""
Run RAG Evaluation Framework

Usage:
    python run_evaluation.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from evaluation import RAGEvaluator, TEST_CASES
from main import get_chatbot


def main():
    """Run complete RAG evaluation"""
    
    print("\n" + "="*70)
    print("LOADING RAG SYSTEM")
    print("="*70)
    
    # Load chatbot
    try:
        chatbot = get_chatbot()
        print("✅ Chatbot loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load chatbot: {e}")
        print("\nMake sure to:")
        print("  1. Build index: python main.py build <document_path>")
        print("  2. Set API key in .env file")
        return 1
    
    # Initialize evaluator
    evaluator = RAGEvaluator(chatbot)
    
    # Run evaluation
    results = evaluator.evaluate(TEST_CASES)
    
    # Save results
    evaluator.save_results(results)
    
    # Print final verdict
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    overall_score = results['overall']['overall_score']
    grade = results['overall']['grade']
    
    if overall_score >= 0.80:
        print(f"\n🎉 EXCELLENT! Your RAG system scored {overall_score:.3f} (Grade {grade})")
        print("   Ready for production deployment!")
    elif overall_score >= 0.70:
        print(f"\n👍 GOOD! Your RAG system scored {overall_score:.3f} (Grade {grade})")
        print("   Minor improvements recommended before production.")
    else:
        print(f"\n⚠️ NEEDS WORK! Your RAG system scored {overall_score:.3f} (Grade {grade})")
        print("   Significant improvements needed.")
    
    print("\n📊 Detailed results saved to: evaluation_results.json")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
