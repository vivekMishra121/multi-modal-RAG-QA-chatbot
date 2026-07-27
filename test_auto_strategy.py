"""Test automatic strategy selection"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import VectorStore
from retrieval import Retriever
from chunking.embedder import SentenceTransformerEmbedder

def main():
    print("\n" + "="*70)
    print("AUTOMATIC STRATEGY SELECTION")
    print("="*70)
    
    # Load vector store
    print("\n[1/2] Loading vector store...")
    try:
        store = VectorStore.load('./vector_store_data')
        print(f"  ✓ Loaded {len(store)} chunks")
    except:
        print("  ❌ No vector store found. Run test_improved_chunking.py first!")
        return
    
    # Initialize retriever
    print("\n[2/2] Initializing retriever...")
    retriever = Retriever(store, use_reranker=True)
    embedder = SentenceTransformerEmbedder()
    
    # Test different query types
    test_queries = [
        # Specific factual queries → Standard
        ("What is Qatar's GDP in 2023?", "standard"),
        ("What is the inflation rate?", "standard"),
        
        # Broad exploratory queries → MMR
        ("What are the main economic sectors in Qatar?", "mmr"),
        ("Give me an overview of Qatar's economy", "mmr"),
        ("Explain the different types of economic policies", "mmr"),
        
        # Keyword/reference queries → Hybrid
        ("Find table 3.2", "hybrid"),
        ("Article 5 section 2", "hybrid"),
        ("GDP 2023 exact figure", "hybrid"),
        ("Page 42 economic data", "hybrid"),
    ]
    
    print("\n" + "="*70)
    print("TESTING AUTO STRATEGY SELECTION")
    print("="*70)
    
    for query, expected_strategy in test_queries:
        print(f"\n📝 Query: \"{query}\"")
        
        # Get query embedding
        query_embedding = embedder.embed_query(query)
        
        # Use AUTO strategy
        results = retriever.retrieve(
            query_embedding,
            query,
            top_k=3,
            strategy='auto',  # ← Automatic selection!
            rerank=True
        )
        
        # Show selected strategy (from debug log)
        selected = retriever._select_strategy(query)
        match = "✅" if selected == expected_strategy else "⚠️"
        print(f"   {match} Selected: {selected.upper()} (expected: {expected_strategy.upper()})")
        
        # Show top result
        if results:
            chunk, score = results[0]
            page = chunk['metadata'].get('page', 'N/A')
            print(f"   Top result: Score {score:.3f} | Page {page}")
            print(f"   Content: {chunk['content'][:100]}...")
    
    print("\n" + "="*70)
    print("STRATEGY SELECTION RULES")
    print("="*70)
    print("""
📊 HYBRID (Keyword-based):
   - Has numbers/codes: "GDP 2023", "Table 3.2"
   - References: "article", "section", "page", "table"
   - Precision words: "exactly", "specific", "precise"
   - Short queries (≤5 words)
   
🔍 MMR (Diverse results):
   - Broad terms: "overview", "summary", "main", "key"
   - Variety words: "different", "types", "various", "aspects"
   - Exploratory: "explain", "describe", "tell me about"
   - Long queries (>8 words)
   
⚡ STANDARD (Default):
   - Specific factual questions
   - "What is...", "How much...", "When did..."
   - Most common query type
   - Best for focused answers
    """)
    
    print("\n" + "="*70)
    print("USAGE IN YOUR CHATBOT")
    print("="*70)
    print("""
# Simple - just use 'auto'!
results = retriever.retrieve(
    query_embedding,
    user_query,
    top_k=5,
    strategy='auto',  # ← Handles everything automatically
    rerank=True
)

# The system will:
# 1. Analyze the query
# 2. Choose best strategy (standard/mmr/hybrid)
# 3. Retrieve candidates
# 4. Rerank for accuracy
# 5. Return best results

# You don't need to worry about strategy selection!
    """)


if __name__ == "__main__":
    main()
