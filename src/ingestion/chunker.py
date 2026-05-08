from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from src.ingestion.loader import load_documents



def fixed_size_chunker(documents: List[Document], chunk_size=500, chunk_overlap=50):
    """
    Splits text every N characters regardless of content,
    fast, simple but cuts mid sentence. 
    Good baseline to compare againsts.
    """

    splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separator=""
    )

    return splitter.split_documents(documents)


def recursive_chunker(documents: List[Document], chunk_size=500, chunk_overlap=50):
    """
    Tries to split on paragraphs → sentences → words → characters.
    Respects text structure. Best general-purpose strategy.
    This is what most production RAG uses.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )
    return splitter.split_documents(documents)


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2",
        model_kwargs = {'device':'cpu'},
        encode_kwargs = {'normalize_embeddings': True}
    )

def semantic_chunker(documents: List[Document], embeddings=None):
    """
    Groups sentences by semantic similarity using embeddings.
    Produces semantically coherent chunks — best for complex documents.
    Slower and costs embedding API calls.
    """
    if embeddings is None:
        embeddings = get_embeddings()

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    return splitter.split_documents(documents)


def compare_strategies(documents):
    print('--'*60)

    fixed = fixed_size_chunker(documents=documents)
    print(f'Fixed size chunker: {len(fixed)} chunks | avg: '
          f'{sum(len(c.page_content) for c in fixed) // len(fixed)} chars'

          )
    recursive = recursive_chunker(documents)
    print(f"Recursive  : {len(recursive)} chunks | avg: "
          f"{sum(len(c.page_content) for c in recursive) // len(recursive)} chars")

    print("Semantic   : loading HuggingFace model... (first run downloads ~90MB)")
    semantic = semantic_chunker(documents)
    print(f"Semantic   : {len(semantic)} chunks | avg: "
          f"{sum(len(c.page_content) for c in semantic) // len(semantic)} chars")

    print("=" * 60)
    return {"fixed": fixed, "recursive": recursive, "semantic": semantic}

# d = load_documents("data\W17-4419.pdf")
# res = compare_strategies(d)
# print(res)
def filter_noise_chunks(chunks: List[Document], min_length: int=200)-> List[Document]:
    """
    Removes chunks that are too short to be meaningful.
    Catches reference lists, headers, page numbers etc.
    """
    noise_patterns = [
        "association for computational linguistics",
        "proceedings of",
        "pages 148",
        "copyright",
        "all rights reserved"
    ]
    
    filtered = []
    for chunk in chunks:
        content = chunk.page_content.strip().lower()
        
        # Length filter
        if len(content) < min_length:
            continue
        
        # Pattern filter — skip if chunk starts with noise
        if any(content.startswith(pattern) for pattern in noise_patterns):
            continue
        
        filtered.append(chunk)
    
    print(f"🧹 Filtered {len(chunks) - len(filtered)} noise chunks | {len(filtered)} remaining")
    return filtered