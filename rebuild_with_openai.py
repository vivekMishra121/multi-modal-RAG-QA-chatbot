"""
Rebuild vector store with OpenAI embeddings for better retrieval

Usage:
    python rebuild_with_openai.py <document_path>
"""

import sys
import shutil
from pathlib import Path
from main import RAGChatbot

def main():
    if len(sys.argv) < 2:
        print("Usage: python rebuild_with_openai.py <document_path>")
        sys.exit(1)
    
    doc_path = sys.argv[1]
    store_path = "./vector_store_data"
    
    print("\n" + "="*70)
    print("REBUILDING INDEX WITH OPENAI EMBEDDINGS")
    print("="*70)
    
    # Backup old index
    if Path(store_path).exists():
        backup_path = "./vector_store_data_backup"
        print(f"\n📦 Backing up old index to: {backup_path}")
        if Path(backup_path).exists():
            shutil.rmtree(backup_path)
        shutil.copytree(store_path, backup_path)
        shutil.rmtree(store_path)
    
    # Build new index
    print(f"\n🔨 Building new index with OpenAI embeddings...")
    print(f"   Document: {doc_path}")
    print(f"   Chunk size: 1000 chars (improved from 512)")
    print(f"   Embeddings: text-embedding-3-large (1536 dims)")
    
    result = RAGChatbot.build_index(doc_path, store_path)
    
    print(f"\n✅ Index rebuilt successfully!")
    print(f"   Documents: {result['documents_processed']}")
    print(f"   Chunks: {result['total_chunks']}")
    print(f"   Tables: {result['tables_extracted']}")
    print(f"   Images: {result['images_extracted']}")
    
    print("\n" + "="*70)
    print("IMPROVEMENTS APPLIED:")
    print("="*70)
    print("✅ OpenAI text-embedding-3-large (better semantic understanding)")
    print("✅ Larger chunk size: 1000 chars (better context)")
    print("✅ Query expansion enabled (better recall)")
    print("✅ Fusion retrieval (combines multiple strategies)")
    print("✅ Enhanced reranking (better precision)")
    
    print("\n🎯 Run evaluation to see improvements:")
    print("   python run_evaluation.py")
    print()

if __name__ == "__main__":
    main()
