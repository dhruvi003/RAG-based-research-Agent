from sentence_transformers import CrossEncoder
from langchain_classic.schema import Document
from typing import List, Tuple

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """
    Loads a cross-encoder reranker model locally.
    'ms-marco-MiniLM-L-6-v2' is fast, small, and good quality.
    For better accuracy: 'cross-encoder/ms-marco-electra-base'
    """
    print(f"🔄 Loading reranker model: {model_name}")
    model = CrossEncoder(model_name)
    print("✅ Reranker loaded")
    return model

def rerank(
    query: str,
    documents: List[Document],
    reranker: CrossEncoder,
    top_n: int = 4
) -> List[Document]:
    """
    Reranks retrieved documents using cross-encoder scores.
    Returns top_n most relevant documents.
    """
    # Pair query with each document content
    pairs = [(query, doc.page_content) for doc in documents]
    
    # Score each pair
    scores = reranker.predict(pairs)
    
    # Attach scores and sort
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Print scores for visibility
    print(f"\n📊 Reranking scores:")
    for score, doc in scored_docs:
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  {score:.4f} | {preview}...")
    
    # Return top_n docs without scores
    return [doc for _, doc in scored_docs[:top_n]]