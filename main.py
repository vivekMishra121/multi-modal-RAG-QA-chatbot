"""Main RAG System API for Frontend Integration"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from document_ingestion.enhanced_file_reader import DocumentIngestionPipeline
from chunking.pipeline import MultiModalChunkingPipeline
from chunking.openai_embedder import OpenAIEmbedder
from vector_store import VectorStore
from retrieval import Retriever
from qa_generation import RAGPipeline

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("tabula").setLevel(logging.ERROR)


class RAGChatbot:
    """RAG-powered chatbot for frontend integration"""
    
    def __init__(
        self,
        store_path: str = "./vector_store_data",
        api_key: Optional[str] = None,
        model_name: str = "gpt-5.1",
        max_context_tokens: int = 4096,
        top_k: int = 5
    ):
        """
        Initialize RAG chatbot
        
        Args:
            store_path: Path to vector store
            api_key: OpenAI API key (defaults to env var)
            model_name: LLM model name (default: gpt-5.1)
            max_context_tokens: Max tokens for context
            top_k: Number of chunks to retrieve
        """
        self.store_path = store_path
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("API key required (set OPENAI_API_KEY in .env)")
        
        # Load or create vector store
        if Path(store_path).exists():
            self.store = VectorStore.load(store_path)
            
            # Check if Azure or OpenAI
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                from chunking.azure_embedder import AzureOpenAIEmbedder
                self.embedder = AzureOpenAIEmbedder()
            else:
                self.embedder = OpenAIEmbedder()
        else:
            raise FileNotFoundError(f"Vector store not found at {store_path}. Run build_index() first.")
        
        # Initialize RAG pipeline with enhanced retrieval
        retriever = Retriever(self.store, use_reranker=True, use_query_expansion=True)
        self.rag = RAGPipeline(
            retriever=retriever,
            embedder=self.embedder,
            api_key=self.api_key,
            model_name=model_name,
            max_context_tokens=max_context_tokens,
            retrieval_strategy='auto',  # Auto strategy selection
            top_k=top_k
        )
    
    def chat(self, question: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get answer for a question
        
        Args:
            question: User question
            filters: Optional metadata filters
            
        Returns:
            {
                'answer': str,
                'sources': List[Dict],
                'success': bool,
                'error': Optional[str]
            }
        """
        result = self.rag.query(question, filters=filters)
        
        return {
            'answer': result['answer'],
            'sources': result['sources'],
            'success': result['success'],
            'error': result.get('error'),
            'num_chunks': result.get('num_chunks_used', 0)
        }
    
    @staticmethod
    def build_index(
        document_path: str,
        store_path: str = "./vector_store_data"
    ) -> Dict[str, Any]:
        """
        Build vector store from documents
        
        Args:
            document_path: Path to PDF/DOCX/TXT file or folder
            store_path: Where to save vector store
            
        Returns:
            Summary of indexing process
        """
        # Ingest documents
        ingestion = DocumentIngestionPipeline()
        documents = ingestion.ingest_documents(document_path)
        summary = ingestion.get_content_summary(documents)
        
        # Chunk and embed with OpenAI
        chunking = MultiModalChunkingPipeline(
            text_chunk_size=1000,
            text_chunk_overlap=200,
            use_openai=True
        )
        chunks_with_embeddings = chunking.process_documents(documents)
        
        # Index
        store = VectorStore(dimension=chunking.embedding_dimension, distance_metric='cosine')
        chunk_dicts = [c.to_dict() for c in chunks_with_embeddings]
        store.add_chunks(chunk_dicts)
        store.save(store_path)
        
        return {
            'documents_processed': summary['successful'],
            'total_chunks': len(chunks_with_embeddings),
            'tables_extracted': summary['total_tables'],
            'images_extracted': summary['total_images']
        }


# Global chatbot instance
_chatbot = None


def get_chatbot(**kwargs) -> RAGChatbot:
    """Get or create chatbot instance"""
    global _chatbot
    if _chatbot is None:
        _chatbot = RAGChatbot(**kwargs)
    return _chatbot


def ask(question: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Simple function to ask questions
    
    Args:
        question: User question
        filters: Optional metadata filters
        
    Returns:
        Answer dictionary
    """
    chatbot = get_chatbot()
    return chatbot.chat(question, filters)


# CLI Interface
if __name__ == "__main__":
    import sys
    
    # Check if building index
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        if len(sys.argv) < 3:
            print("Usage: python main.py build <document_path>")
            sys.exit(1)
        
        doc_path = sys.argv[2]
        print(f"📄 Building index from: {doc_path}")
        
        result = RAGChatbot.build_index(doc_path)
        print(f"✅ Index built successfully!")
        print(f"   Documents: {result['documents_processed']}")
        print(f"   Chunks: {result['total_chunks']}")
        print(f"   Tables: {result['tables_extracted']}")
        print(f"   Images: {result['images_extracted']}")
        sys.exit(0)
    
    # Interactive chat
    print("🤖 RAG Chatbot (type 'quit' to exit)")
    print("-" * 50)
    
    try:
        chatbot = get_chatbot()
        print("✅ Chatbot ready!\n")
        
        while True:
            question = input("You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            result = chatbot.chat(question)
            
            if result['success']:
                print(f"\nBot: {result['answer']}")
                print(f"\n📚 Sources: {result['num_chunks']} chunks used")
            else:
                print(f"\n❌ Error: {result['error']}")
            
            print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
