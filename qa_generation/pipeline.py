"""End-to-end RAG pipeline: Query → Retrieve → Generate"""

import logging
from typing import Dict, Any, List, Optional
from .qa_chain import QAChain

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation"""
    
    def __init__(
        self,
        retriever,
        embedder,
        api_key: str,
        model_name: str = "gpt-3.5-turbo",
        max_context_tokens: int = 4096,
        retrieval_strategy: str = 'standard',
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline
        
        Args:
            retriever: Retriever instance
            embedder: Embedder instance
            api_key: OpenAI API key
            model_name: LLM model name
            max_context_tokens: Max context window
            retrieval_strategy: 'standard', 'mmr', or 'hybrid'
            top_k: Number of chunks to retrieve
        """
        self.retriever = retriever
        self.embedder = embedder
        self.retrieval_strategy = retrieval_strategy
        self.top_k = top_k
        self.max_context_tokens = max_context_tokens
        
        self.qa_chain = QAChain(api_key, model_name)
        
        logger.info(f"Initialized RAG pipeline (strategy={retrieval_strategy}, top_k={top_k})")
    
    def query(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        **retrieval_kwargs
    ) -> Dict[str, Any]:
        """
        Process a query end-to-end
        
        Args:
            question: User question
            filters: Optional metadata filters
            **retrieval_kwargs: Additional retrieval parameters
            
        Returns:
            Dict with answer, sources, and metadata
        """
        logger.info(f"Processing query: {question}")
        
        # Step 1: Embed query
        query_embedding = self.embedder.embed_query(question)
        
        # Step 2: Retrieve relevant chunks
        chunks_with_scores = self.retriever.retrieve(
            query_embedding=query_embedding,
            query_text=question,
            top_k=self.top_k,
            strategy=self.retrieval_strategy,
            filters=filters,
            **retrieval_kwargs
        )
        
        if not chunks_with_scores:
            return {
                'answer': "No relevant information found in the documents.",
                'sources': [],
                'success': False,
                'error': 'No chunks retrieved'
            }
        
        logger.info(f"Retrieved {len(chunks_with_scores)} chunks")
        
        # Step 3: Select top chunks and format context
        selected_chunks = [chunk for chunk, score in chunks_with_scores[:self.top_k]]
        context = self._format_context(selected_chunks)
        
        # Step 4: Generate answer
        result = self.qa_chain.generate_answer(question, context)
        
        # Step 5: Add source information
        sources = self._extract_sources(selected_chunks)
        
        return {
            'answer': result['answer'],
            'sources': sources,
            'success': result['success'],
            'error': result['error'],
            'num_chunks_used': len(selected_chunks),
            'retrieval_strategy': self.retrieval_strategy
        }
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format chunks into context string"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            source = metadata.get('file_name', 'Unknown')
            page = metadata.get('page', 'N/A')
            chunk_type = chunk.get('chunk_type', 'text')
            
            citation = f"[Source {i}: {source}, Page {page}, Type: {chunk_type}]"
            content = chunk['content']
            
            context_parts.append(f"{citation}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source citations from chunks"""
        sources = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.get('metadata', {})
            sources.append({
                'source_id': i,
                'file_name': metadata.get('file_name', 'Unknown'),
                'page': metadata.get('page', 'N/A'),
                'chunk_type': chunk.get('chunk_type', 'text'),
                'content_preview': chunk['content'][:200] + '...'
            })
        
        return sources
