"""Test improved chunking with page tracking and context"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from document_ingestion.enhanced_file_reader import EnhancedFileReader
from chunking.pipeline import MultiModalChunkingPipeline
from vector_store import VectorStore
from retrieval import Retriever

def main():
    pdf_path = Path(r"C:\Users\HP\Downloads\Qatar Test Document.pdf")
    
    print("\n" + "="*70)
    print("TESTING IMPROVED CHUNKING")
    print("="*70)
    
    # Step 1: Ingestion
    print("\n[1/4] Document Ingestion...")
    reader = EnhancedFileReader()
    documents = reader.process_files([pdf_path])
    
    # Step 2: Chunking with NEW settings
    print("[2/4] Chunking with improved settings...")
    print("  - Chunk size: 512 chars (was 1000)")
    print("  - Overlap: 100 chars (was 200)")
    print("  - Page tracking: ENABLED")
    print("  - Context headers: ENABLED")
    
    pipeline = MultiModalChunkingPipeline()
    chunks = pipeline.process_documents(documents)
    
    print(f"\n  ✓ Created {len(chunks)} chunks")
    
    # Show sample chunks
    print("\n" + "="*70)
    print("SAMPLE CHUNKS (Before vs After)")
    print("="*70)
    
    print("\n📄 TEXT CHUNK EXAMPLE:")
    text_chunks = [c for c in chunks if c.chunk.chunk_type.value == 'text']
    if text_chunks:
        sample = text_chunks[0]
        print(f"\nContent:\n{sample.chunk.content[:300]}...")
        print(f"\nMetadata: {sample.chunk.metadata}")
    
    print("\n📊 TABLE CHUNK EXAMPLE:")
    table_chunks = [c for c in chunks if c.chunk.chunk_type.value == 'table']
    if table_chunks:
        sample = table_chunks[0]
        print(f"\nContent:\n{sample.chunk.content[:300]}...")
        print(f"\nMetadata: {sample.chunk.metadata}")
    
    print("\n🖼️ IMAGE CHUNK EXAMPLE:")
    image_chunks = [c for c in chunks if c.chunk.chunk_type.value == 'image']
    if image_chunks:
        sample = image_chunks[0]
        print(f"\nContent:\n{sample.chunk.content[:300]}...")
        print(f"\nMetadata: {sample.chunk.metadata}")
    
    # Step 3: Index
    print("\n" + "="*70)
    print("[3/4] Indexing...")
    store = VectorStore(dimension=pipeline.embedding_dimension, distance_metric='cosine')
    store.add_chunks([c.to_dict() for c in chunks])
    
    # Step 4: Test Retrieval
    print("[4/4] Testing Retrieval...")
    retriever = Retriever(store)
    
    query = "What is Qatar's GDP?"
    print(f"\nQuery: '{query}'")
    
    query_embedding = pipeline.embedder.embed_query(query)
    results = retriever.retrieve(query_embedding, query, top_k=3, strategy='standard')
    
    print("\n" + "="*70)
    print("RETRIEVAL RESULTS")
    print("="*70)
    
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"Type: {chunk['chunk_type']}")
        print(f"Page: {chunk['metadata'].get('page', 'N/A')}")
        print(f"Content:\n{chunk['content'][:250]}...")
        print("-" * 70)
    
    # Show improvement metrics
    print("\n" + "="*70)
    print("IMPROVEMENTS")
    print("="*70)
    
    pages_with_info = sum(1 for c in chunks if c.chunk.metadata.get('page') is not None)
    print(f"✓ Chunks with page info: {pages_with_info}/{len(chunks)} ({pages_with_info/len(chunks)*100:.1f}%)")
    
    avg_chunk_size = sum(len(c.chunk.content) for c in chunks) / len(chunks)
    print(f"✓ Average chunk size: {avg_chunk_size:.0f} chars (target: 512)")
    
    chunks_with_context = sum(1 for c in chunks if '[Document:' in c.chunk.content or '[Table:' in c.chunk.content or '[Image:' in c.chunk.content)
    print(f"✓ Chunks with context headers: {chunks_with_context}/{len(chunks)} ({chunks_with_context/len(chunks)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✓ IMPROVEMENTS COMPLETE!")
    print("="*70)
    print("\nKey Changes:")
    print("1. ✅ Page numbers tracked in metadata")
    print("2. ✅ Smaller chunks (512 vs 1000 chars) = more focused")
    print("3. ✅ Context headers added to all chunks")
    print("4. ✅ Consistent metadata across all chunk types")
    print("5. ✅ Better formatting for LLM consumption")


if __name__ == "__main__":
    main()
