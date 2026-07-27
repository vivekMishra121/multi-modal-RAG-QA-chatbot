"""Complete End-to-End RAG System Demo"""

import logging
import os
from pathlib import Path
from dotenv import load_dotenv

from document_ingestion.enhanced_file_reader import DocumentIngestionPipeline
from chunking.pipeline import MultiModalChunkingPipeline
from vector_store import VectorStore
from retrieval import Retriever
from qa_generation import RAGPipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("tabula").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def main():
    """Complete RAG pipeline demonstration"""
    
    # Configuration
    PDF_PATH = Path(r"C:\Users\HP\Downloads\Qatar Test Document.pdf")
    STORE_PATH = "./vector_store_data"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    
    print("🤖 COMPLETE RAG SYSTEM - Multi-Modal Document QA")
    
    # Check if index exists
    if Path(STORE_PATH).exists():
        print("\n📂 Loading existing vector store...")
        store = VectorStore.load(STORE_PATH)
        
        # Load embedder for query embedding
        from chunking.embedder import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()
    else:
        print("\n📄 Building vector store from scratch...")
        
        # Step 1: Document Ingestion
        print("\n[1/4] Document Ingestion...")
        ingestion = DocumentIngestionPipeline()
        documents = ingestion.ingest_documents(PDF_PATH)
        summary = ingestion.get_content_summary(documents)
        print(f"  ✓ Processed {summary['successful']} documents")
        print(f"  ✓ Extracted {summary['total_tables']} tables, {summary['total_images']} images")
        
        # Step 2: Chunking & Embedding
        print("\n[2/4] Chunking & Embedding...")
        chunking = MultiModalChunkingPipeline()
        chunks_with_embeddings = chunking.process_documents(documents)
        print(f"  ✓ Created {len(chunks_with_embeddings)} chunks")
        
        embedder = chunking.embedder
        
        # Step 3: Vector Store Indexing
        print("\n[3/4] Vector Store Indexing...")
        store = VectorStore(dimension=chunking.embedding_dimension, distance_metric='cosine')
        chunk_dicts = [c.to_dict() for c in chunks_with_embeddings]
        store.add_chunks(chunk_dicts)
        store.save(STORE_PATH)
        print(f"  ✓ Indexed {len(store)} chunks")
    
    # Step 4: Initialize RAG Pipeline
    print("\n[4/4] Initializing RAG Pipeline...")
    retriever = Retriever(store)
    
    rag = RAGPipeline(
        retriever=retriever,
        embedder=embedder,
        api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        max_context_tokens=4096,
        retrieval_strategy='standard',  # Best score strategy
        top_k=5
    )
    print("  ✓ RAG pipeline ready")
    
    # Interactive QA
    print("\n" + "="*70)
    print("💬 Interactive QA System (type 'quit' to exit)")
    print("="*70)
    
    # Demo queries
    demo_queries = [
        "What is Qatar's economic growth rate?",
        "What are the main drivers of economic growth?",
        "What is the inflation rate?"
    ]
    
    print("\n📝 Demo Queries:")
    for i, q in enumerate(demo_queries, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-"*70)
    
    while True:
        query = input("\n❓ Your question (or number 1-3, or 'quit'): ").strip()
        
        if query.lower() == 'quit':
            print("\n👋 Goodbye!")
            break
        
        # Handle demo query selection
        if query in ['1', '2', '3']:
            query = demo_queries[int(query) - 1]
            print(f"   Selected: {query}")
        
        if not query:
            continue
        
        print(f"\n🔍 Processing query...")
        
        # Get answer
        result = rag.query(query)
        
        if result['success']:
            print(f"\n✅ Answer:")
            print(f"{result['answer']}")
            
            print(f"\n📚 Sources ({result['num_chunks_used']} chunks used):")
            for source in result['sources']:
                print(f"  [{source['source_id']}] {source['file_name']} - Page {source['page']} ({source['chunk_type']})")
                print(f"      Preview: {source['content_preview'][:100]}...")
        else:
            print(f"\n❌ Error: {result['error']}")
        
        print("\n" + "-"*70)


if __name__ == "__main__":
    main()
