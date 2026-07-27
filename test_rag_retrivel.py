"""Example: Proper RAG architecture with separated Indexing and Retrieval"""

import logging
from pathlib import Path
from document_ingestion.enhanced_file_reader import DocumentIngestionPipeline
from chunking.pipeline import MultiModalChunkingPipeline
from vector_store import VectorStore
from retrieval import Retriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    pdf_path = Path(r"C:\Users\HP\Downloads\Qatar Test Document.pdf")
    store_path = "./vector_store_data"
    
    print("\n" + "="*60)
    print("RAG PIPELINE: Indexing + Retrieval")
    print("="*60)
    
    # Step 1: Ingestion
    print("\n[1/4] Document Ingestion...")
    reader = DocumentIngestionPipeline()
    documents = reader.ingest_documents(pdf_path)
    
    # Step 2: Chunking & Embedding
    print("[2/4] Chunking & Embedding...")
    pipeline = MultiModalChunkingPipeline()
    chunks_with_embeddings = pipeline.process_documents(documents)
    
    # Step 3: INDEXING - Vector Store
    print("[3/4] Indexing (Vector Store)...")
    store = VectorStore(dimension=pipeline.embedding_dimension, distance_metric='cosine')
    chunk_dicts = [chunk.to_dict() for chunk in chunks_with_embeddings]
    store.add_chunks(chunk_dicts)
    store.save(store_path)
    
    stats = store.get_stats()
    print(f"  ✓ Indexed {stats['total_chunks']} chunks")
    print(f"  ✓ Chunk types: {stats['chunk_types']}")
    
    # Step 4: RETRIEVAL - Advanced Search
    print("[4/4] Retrieval (Search Strategies)...")
    retriever = Retriever(store)
    
    query = "What is Qatar's economic growth rate?"
    query_embedding = pipeline.embedder.embed_query(query)
    
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}")
    
    # Standard retrieval
    print("\n[Strategy 1] STANDARD (Vector Similarity)")
    results = retriever.retrieve(query_embedding, query, top_k=3, strategy='standard')
    for i, (chunk, score) in enumerate(results, 1):
        print(f"{i}. Score: {score:.4f} | {chunk['chunk_type']} | Page {chunk['metadata'].get('page')}")
        print(f"   {chunk['content'][:80]}...")
    
    # MMR retrieval
    print("\n[Strategy 2] MMR (Diverse Results)")
    mmr_results = retriever.retrieve(query_embedding, query, top_k=3, strategy='mmr', 
                                     fetch_k=10, lambda_mult=0.5)
    for i, (chunk, score) in enumerate(mmr_results, 1):
        print(f"{i}. Score: {score:.4f} | {chunk['chunk_type']} | Page {chunk['metadata'].get('page')}")
        print(f"   {chunk['content'][:80]}...")
    
    # Hybrid retrieval
    print("\n[Strategy 3] HYBRID (Vector + Keyword)")
    hybrid_results = retriever.retrieve(query_embedding, query, top_k=3, strategy='hybrid', alpha=0.7)
    for i, (chunk, score) in enumerate(hybrid_results, 1):
        print(f"{i}. Score: {score:.4f} | {chunk['chunk_type']} | Page {chunk['metadata'].get('page')}")
        print(f"   {chunk['content'][:80]}...")
    
    print(f"\n{'='*60}")
    print("✓ RAG Pipeline Complete!")
    print(f"{'='*60}")
    print("\nArchitecture:")
    print("  1. Ingestion → Extract content")
    print("  2. Chunking → Create embeddings")
    print("  3. Indexing → Store in vector DB (vector_store/)")
    print("  4. Retrieval → Advanced search (retrieval/)")


if __name__ == "__main__":
    main()
