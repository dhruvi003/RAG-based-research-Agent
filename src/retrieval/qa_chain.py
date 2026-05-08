from langchain_groq import ChatGroq
from langchain_classic.prompts import ChatPromptTemplate
from langchain_classic.schema.runnable import RunnablePassthrough
from langchain_classic.schema.output_parser import StrOutputParser
from langchain_classic.schema import Document
from typing import List 
import os
from dotenv import load_dotenv

def get_llm(model: str="Llama-3.3-70B-Versatile"):
    """
    Groq hosted Llama 3-fast and free
    """
    return ChatGroq(
        model = model,
        api_key= os.getenv('GROQ_API_KEY'),
        temperature=0
    )


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(retriever, llm=None):
    """
    Builds simple RAG chain: retriever -> format -> prompt -> llm
    """
    if llm is None:
        llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template(
       """
        You are a helpful assistant. Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm 
        | StrOutputParser()
    )

    return chain