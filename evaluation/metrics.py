"""Core evaluation metrics - Industry standard"""

import time
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util


class RetrievalMetrics:
    """Retrieval quality metrics (Used by: OpenAI, Google, Anthropic)"""
    
    @staticmethod
    def precision_at_k(retrieved_pages: List[int], relevant_pages: List[int]) -> float:
        """
        Precision@K: % of retrieved chunks that are relevant
        Industry Standard: >0.70
        """
        if not retrieved_pages:
            return 0.0
        relevant_retrieved = len(set(retrieved_pages) & set(relevant_pages))
        return relevant_retrieved / len(retrieved_pages)
    
    @staticmethod
    def recall_at_k(retrieved_pages: List[int], relevant_pages: List[int]) -> float:
        """
        Recall@K: % of relevant chunks that were retrieved
        Industry Standard: >0.60
        """
        if not relevant_pages:
            return 0.0
        relevant_retrieved = len(set(retrieved_pages) & set(relevant_pages))
        return relevant_retrieved / len(relevant_pages)
    
    @staticmethod
    def mrr(retrieved_pages: List[int], relevant_pages: List[int]) -> float:
        """
        Mean Reciprocal Rank: Position of first relevant result
        Industry Standard: >0.50
        """
        for i, page in enumerate(retrieved_pages, 1):
            if page in relevant_pages:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved_pages: List[int], relevant_pages: List[int], k: int = 5) -> float:
        """
        Normalized Discounted Cumulative Gain
        Industry Standard: >0.60
        """
        dcg = 0.0
        for i, page in enumerate(retrieved_pages[:k], 1):
            if page in relevant_pages:
                dcg += 1.0 / (i + 1)  # Simplified NDCG
        
        # Ideal DCG
        idcg = sum(1.0 / (i + 1) for i in range(min(len(relevant_pages), k)))
        
        return dcg / idcg if idcg > 0 else 0.0


class GenerationMetrics:
    """Answer generation quality metrics (Used by: OpenAI, Microsoft, Meta)"""
    
    def __init__(self):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def semantic_similarity(self, answer: str, reference: str) -> float:
        """
        Semantic similarity between answer and reference
        Industry Standard: >0.70
        """
        emb1 = self.similarity_model.encode(answer, convert_to_tensor=True)
        emb2 = self.similarity_model.encode(reference, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).item())
    
    def faithfulness(self, answer: str, context: str) -> float:
        """
        Answer grounded in context (hallucination check)
        Industry Standard: >0.85
        """
        if not context:
            return 0.0
        
        emb_answer = self.similarity_model.encode(answer, convert_to_tensor=True)
        emb_context = self.similarity_model.encode(context, convert_to_tensor=True)
        similarity = float(util.cos_sim(emb_answer, emb_context).item())
        
        # High similarity = grounded in context
        return similarity
    
    @staticmethod
    def answer_length_score(answer: str, min_words: int = 20, max_words: int = 150) -> float:
        """
        Answer length appropriateness
        Industry Standard: 20-150 words
        """
        word_count = len(answer.split())
        
        if word_count < min_words:
            return word_count / min_words
        elif word_count > max_words:
            return max_words / word_count
        else:
            return 1.0
    
    @staticmethod
    def citation_present(answer: str, sources: List[Dict]) -> float:
        """
        Check if answer references sources
        Industry Standard: >0.80
        """
        if not sources:
            return 0.0
        
        # Check for page numbers or source references
        has_citation = any(
            str(src.get('page', '')) in answer or
            src.get('file_name', '') in answer
            for src in sources
        )
        
        return 1.0 if has_citation else 0.0


class MultiModalMetrics:
    """Multi-modal specific metrics (Used by: Google, Microsoft)"""
    
    @staticmethod
    def modality_coverage(results: List[Tuple[Dict, float]]) -> Dict[str, Any]:
        """
        Check if all modalities are used
        Industry Standard: 2-3 modalities in top-5
        """
        modality_counts = {'text': 0, 'table': 0, 'image': 0}
        
        for chunk, _ in results:
            chunk_type = chunk.get('chunk_type', 'text')
            modality_counts[chunk_type] += 1
        
        modalities_used = sum(1 for count in modality_counts.values() if count > 0)
        
        return {
            'modalities_used': modalities_used,
            'distribution': modality_counts,
            'score': modalities_used / 3.0
        }
    
    @staticmethod
    def table_retrieval_accuracy(results: List[Tuple[Dict, float]], query: str) -> float:
        """
        Check if tables retrieved for data queries
        Industry Standard: >0.75
        """
        # Data query indicators
        data_keywords = ['rate', 'number', 'amount', 'value', 'data', 'statistics', 'figure']
        is_data_query = any(kw in query.lower() for kw in data_keywords)
        
        if not is_data_query:
            return 1.0  # Not applicable
        
        # Check if table in top-3
        has_table = any(
            chunk.get('chunk_type') == 'table'
            for chunk, _ in results[:3]
        )
        
        return 1.0 if has_table else 0.0
    
    @staticmethod
    def chart_detection_quality(chunks: List[Dict]) -> Dict[str, float]:
        """
        Evaluate chart detection accuracy
        Industry Standard: >0.75 detection rate
        """
        image_chunks = [c for c in chunks if c.get('chunk_type') == 'image']
        
        if not image_chunks:
            return {'detection_rate': 0.0, 'classification_rate': 0.0}
        
        charts_detected = sum(
            1 for c in image_chunks
            if c.get('metadata', {}).get('is_chart')
        )
        
        charts_classified = sum(
            1 for c in image_chunks
            if c.get('metadata', {}).get('chart_type')
        )
        
        return {
            'detection_rate': charts_detected / len(image_chunks),
            'classification_rate': charts_classified / len(image_chunks)
        }
    
    @staticmethod
    def ocr_quality(chunks: List[Dict]) -> float:
        """
        OCR text quality
        Industry Standard: >0.85 success rate
        """
        image_chunks = [c for c in chunks if c.get('chunk_type') == 'image']
        
        if not image_chunks:
            return 0.0
        
        ocr_success = sum(
            1 for c in image_chunks
            if c.get('content') and len(c.get('content', '')) > 20
        )
        
        return ocr_success / len(image_chunks)


class LatencyMetrics:
    """Performance metrics (Used by: All companies)"""
    
    @staticmethod
    def measure_latency(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function execution time"""
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        return result, latency
    
    @staticmethod
    def latency_score(latency: float, threshold: float = 3.5) -> float:
        """
        Convert latency to score
        Industry Standard: <3.5s total
        """
        if latency <= threshold:
            return 1.0
        else:
            return threshold / latency
