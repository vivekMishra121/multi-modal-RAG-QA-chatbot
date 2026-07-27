"""Main RAG Evaluator - Orchestrates all metrics"""

import json
import logging
from typing import List, Dict, Any
from pathlib import Path

from .metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    MultiModalMetrics,
    LatencyMetrics
)

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Complete RAG evaluation framework
    
    Metrics:
    1. Retrieval (30%): Precision, Recall, MRR, NDCG
    2. Generation (30%): Semantic similarity, Faithfulness, Length
    3. Multi-Modal (25%): Coverage, Table accuracy, Chart detection
    4. Latency (15%): Response time
    """
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.retrieval_metrics = RetrievalMetrics()
        self.generation_metrics = GenerationMetrics()
        self.multimodal_metrics = MultiModalMetrics()
        self.latency_metrics = LatencyMetrics()
    
    def evaluate(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run complete evaluation
        
        Args:
            test_cases: List of test cases with:
                - query: str
                - relevant_pages: List[int] (optional)
                - reference_answer: str (optional)
        
        Returns:
            Complete evaluation results
        """
        print("\n" + "="*70)
        print("RAG EVALUATION - INDUSTRY STANDARD METRICS")
        print("="*70)
        print(f"Test Cases: {len(test_cases)}\n")
        
        results = {
            'retrieval': self._evaluate_retrieval(test_cases),
            'generation': self._evaluate_generation(test_cases),
            'multimodal': self._evaluate_multimodal(test_cases),
            'latency': self._evaluate_latency(test_cases)
        }
        
        # Calculate overall score
        results['overall'] = self._calculate_overall_score(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _evaluate_retrieval(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate retrieval quality"""
        print("📊 1. RETRIEVAL METRICS")
        print("-" * 70)
        
        precisions, recalls, mrrs, ndcgs = [], [], [], []
        
        for case in test_cases:
            query = case['query']
            relevant_pages = case.get('relevant_pages', [])
            
            if not relevant_pages:
                continue
            
            # Get retrieval results
            query_embedding = self.chatbot.embedder.embed_query(query)
            results = self.chatbot.rag.retriever.retrieve(
                query_embedding, query, top_k=5, strategy='auto', rerank=True
            )
            
            # Extract pages
            retrieved_pages = [
                c['metadata'].get('page') 
                for c, _ in results 
                if c['metadata'].get('page') is not None
            ]
            
            if retrieved_pages:
                precisions.append(self.retrieval_metrics.precision_at_k(retrieved_pages, relevant_pages))
                recalls.append(self.retrieval_metrics.recall_at_k(retrieved_pages, relevant_pages))
                mrrs.append(self.retrieval_metrics.mrr(retrieved_pages, relevant_pages))
                ndcgs.append(self.retrieval_metrics.ndcg_at_k(retrieved_pages, relevant_pages))
        
        metrics = {
            'precision@5': sum(precisions) / len(precisions) if precisions else 0,
            'recall@5': sum(recalls) / len(recalls) if recalls else 0,
            'mrr': sum(mrrs) / len(mrrs) if mrrs else 0,
            'ndcg@5': sum(ndcgs) / len(ndcgs) if ndcgs else 0
        }
        
        print(f"  Precision@5: {metrics['precision@5']:.3f} (Target: >0.70)")
        print(f"  Recall@5:    {metrics['recall@5']:.3f} (Target: >0.60)")
        print(f"  MRR:         {metrics['mrr']:.3f} (Target: >0.50)")
        print(f"  NDCG@5:      {metrics['ndcg@5']:.3f} (Target: >0.60)")
        
        return metrics
    
    def _evaluate_generation(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate answer generation"""
        print("\n📊 2. GENERATION METRICS")
        print("-" * 70)
        
        similarities, faithfulness_scores, length_scores, citation_scores = [], [], [], []
        
        for case in test_cases:
            query = case['query']
            reference = case.get('reference_answer', '')
            
            # Get answer
            result = self.chatbot.chat(query)
            answer = result['answer']
            sources = result['sources']
            
            # Semantic similarity
            if reference:
                sim = self.generation_metrics.semantic_similarity(answer, reference)
                similarities.append(sim)
            
            # Faithfulness
            if sources:
                context = " ".join([s['content_preview'] for s in sources])
                faith = self.generation_metrics.faithfulness(answer, context)
                faithfulness_scores.append(faith)
            
            # Length
            length_scores.append(self.generation_metrics.answer_length_score(answer))
            
            # Citation
            citation_scores.append(self.generation_metrics.citation_present(answer, sources))
        
        metrics = {
            'semantic_similarity': sum(similarities) / len(similarities) if similarities else 0,
            'faithfulness': sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0,
            'length_score': sum(length_scores) / len(length_scores) if length_scores else 0,
            'citation_score': sum(citation_scores) / len(citation_scores) if citation_scores else 0
        }
        
        print(f"  Semantic Similarity: {metrics['semantic_similarity']:.3f} (Target: >0.70)")
        print(f"  Faithfulness:        {metrics['faithfulness']:.3f} (Target: >0.85)")
        print(f"  Length Score:        {metrics['length_score']:.3f} (Target: >0.80)")
        print(f"  Citation Score:      {metrics['citation_score']:.3f} (Target: >0.80)")
        
        return metrics
    
    def _evaluate_multimodal(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate multi-modal capabilities"""
        print("\n📊 3. MULTI-MODAL METRICS")
        print("-" * 70)
        
        coverage_scores, table_scores = [], []
        
        for case in test_cases:
            query = case['query']
            
            # Get results
            query_embedding = self.chatbot.embedder.embed_query(query)
            results = self.chatbot.rag.retriever.retrieve(
                query_embedding, query, top_k=5, strategy='auto', rerank=True
            )
            
            # Modality coverage
            coverage = self.multimodal_metrics.modality_coverage(results)
            coverage_scores.append(coverage['score'])
            
            # Table retrieval
            table_acc = self.multimodal_metrics.table_retrieval_accuracy(results, query)
            table_scores.append(table_acc)
        
        # Chart detection (on all chunks)
        chart_quality = self.multimodal_metrics.chart_detection_quality(
            self.chatbot.rag.retriever.vector_store.chunks
        )
        
        # OCR quality
        ocr_score = self.multimodal_metrics.ocr_quality(
            self.chatbot.rag.retriever.vector_store.chunks
        )
        
        metrics = {
            'modality_coverage': sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0,
            'table_accuracy': sum(table_scores) / len(table_scores) if table_scores else 0,
            'chart_detection': chart_quality['detection_rate'],
            'ocr_quality': ocr_score
        }
        
        print(f"  Modality Coverage:   {metrics['modality_coverage']:.3f} (Target: >0.67)")
        print(f"  Table Accuracy:      {metrics['table_accuracy']:.3f} (Target: >0.75)")
        print(f"  Chart Detection:     {metrics['chart_detection']:.3f} (Target: >0.75)")
        print(f"  OCR Quality:         {metrics['ocr_quality']:.3f} (Target: >0.85)")
        
        return metrics
    
    def _evaluate_latency(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate system performance"""
        print("\n📊 4. LATENCY METRICS")
        print("-" * 70)
        
        latencies = []
        
        for case in test_cases[:5]:  # Test 5 queries
            query = case['query']
            
            _, latency = self.latency_metrics.measure_latency(
                self.chatbot.chat, query
            )
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        
        metrics = {
            'avg_latency': avg_latency,
            'latency_score': self.latency_metrics.latency_score(avg_latency)
        }
        
        print(f"  Avg Latency:  {metrics['avg_latency']:.3f}s (Target: <3.5s)")
        print(f"  Latency Score: {metrics['latency_score']:.3f}")
        
        return metrics
    
    def _calculate_overall_score(self, results: Dict) -> Dict[str, float]:
        """Calculate weighted overall score"""
        # Retrieval (30%)
        retrieval_score = (
            results['retrieval']['precision@5'] * 0.3 +
            results['retrieval']['recall@5'] * 0.3 +
            results['retrieval']['mrr'] * 0.2 +
            results['retrieval']['ndcg@5'] * 0.2
        )
        
        # Generation (30%)
        generation_score = (
            results['generation']['semantic_similarity'] * 0.4 +
            results['generation']['faithfulness'] * 0.4 +
            results['generation']['length_score'] * 0.1 +
            results['generation']['citation_score'] * 0.1
        )
        
        # Multi-modal (25%)
        multimodal_score = (
            results['multimodal']['modality_coverage'] * 0.3 +
            results['multimodal']['table_accuracy'] * 0.3 +
            results['multimodal']['chart_detection'] * 0.2 +
            results['multimodal']['ocr_quality'] * 0.2
        )
        
        # Latency (15%)
        latency_score = results['latency']['latency_score']
        
        # Weighted overall
        overall = (
            retrieval_score * 0.30 +
            generation_score * 0.30 +
            multimodal_score * 0.25 +
            latency_score * 0.15
        )
        
        return {
            'retrieval_score': retrieval_score,
            'generation_score': generation_score,
            'multimodal_score': multimodal_score,
            'latency_score': latency_score,
            'overall_score': overall,
            'grade': self._get_grade(overall)
        }
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        overall = results['overall']
        
        print(f"\n🎯 Component Scores:")
        print(f"  Retrieval:   {overall['retrieval_score']:.3f} (30% weight)")
        print(f"  Generation:  {overall['generation_score']:.3f} (30% weight)")
        print(f"  Multi-Modal: {overall['multimodal_score']:.3f} (25% weight)")
        print(f"  Latency:     {overall['latency_score']:.3f} (15% weight)")
        
        print(f"\n🏆 OVERALL SCORE: {overall['overall_score']:.3f} (Grade: {overall['grade']})")
        
        # Industry comparison
        print("\n" + "="*70)
        print("INDUSTRY BENCHMARKS")
        print("="*70)
        print("""
Grade Scale:
  A+ (0.90-1.00): Production-ready, top-tier
  A  (0.80-0.89): Production-ready, good
  B  (0.70-0.79): Acceptable, minor improvements needed
  C  (0.60-0.69): Needs improvements
  D  (<0.60):     Not production-ready

Top Companies (OpenAI, Anthropic, Google):
  Overall: 0.85-0.95
  Retrieval: 0.80-0.90
  Generation: 0.85-0.95
  Multi-Modal: 0.80-0.90
  Latency: <2s
        """)
        
        # Recommendations
        self._print_recommendations(overall)
    
    def _print_recommendations(self, overall: Dict):
        """Print improvement recommendations"""
        print("="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        score = overall['overall_score']
        
        if score >= 0.80:
            print("\n✅ Your RAG system is PRODUCTION-READY!")
        elif score >= 0.70:
            print("\n⚠️ Your RAG system is acceptable but needs minor improvements")
        else:
            print("\n❌ Your RAG system needs significant improvements")
        
        # Specific recommendations
        if overall['retrieval_score'] < 0.70:
            print("\n🔧 Improve Retrieval:")
            print("  - Use better embedding model (bge-large)")
            print("  - Optimize chunk size")
            print("  - Add query expansion")
        
        if overall['generation_score'] < 0.70:
            print("\n🔧 Improve Generation:")
            print("  - Use better LLM (GPT-4)")
            print("  - Improve prompts")
            print("  - Add few-shot examples")
        
        if overall['multimodal_score'] < 0.70:
            print("\n🔧 Improve Multi-Modal:")
            print("  - Enhance chart detection")
            print("  - Improve OCR quality")
            print("  - Better table formatting")
        
        if overall['latency_score'] < 0.70:
            print("\n🔧 Improve Speed:")
            print("  - Cache frequent queries")
            print("  - Use faster models")
            print("  - Optimize retrieval")
    
    def _get_grade(self, score: float) -> str:
        """Convert score to grade"""
        if score >= 0.90:
            return "A+"
        elif score >= 0.80:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.60:
            return "C"
        else:
            return "D"
    
    def save_results(self, results: Dict, filepath: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {filepath}")
