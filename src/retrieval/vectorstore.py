from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.schema import Document
from typing import List 
import os 

def get_embeddings():
    """
    We will use same embedding used in chunker
    we'll keep it in one place 
    """
    return HuggingFaceEmbeddings(
        model="all-MiniLM-L6-v2",
        model_kwargs={'device':"cpu"},
        encode_kwargs={"normalize_embeddings":True}
    )


def build_vectorstore(chunks: List[Document], save_path: str = "vectorstore/faiss_index"):
    """
    Embeds chunks and store in FAISS index.
    Save to disk so we won't have to re-embed every run
    """
    print(f'Embedding {len(chunks)} chunks...')
    embedding = get_embeddings()
    
    vectorstore = FAISS.from_documents(chunks, embedding)

    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

    print(f'Vector embeddings are saved {save_path}')
    return vectorstore


def load_vectorstore(save_path: str = "vectorstore/faiss_index"):
    """
    Loads an existing FAISS from disk.
    """
    embedding = get_embeddings()
    vectorstore = FAISS.load_local(
        save_path,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )
    print(f'Got embeddings from {save_path}')
    return vectorstore


def get_retriever(vectorstore, top_k:int = 4):
    """
    Return retriever that fetches top-k most similart chunks

    """
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs = {'k': top_k}
    )
