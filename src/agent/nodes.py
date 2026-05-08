from langchain_groq import ChatGroq
from langchain_classic.schema import Document
from src.agent.state import AgentState
from src.retrieval.vectorstore import load_vectorstore, get_retriever
from src.retrieval.reranker import get_reranker, rerank
from typing import List
import os 
from dotenv import load_dotenv
from src.retrieval.qa_chain import get_llm
from src.agent.state import AgentState, set_docs, get_docs, clear_store
from src.agent.state import AgentState, set_docs, get_docs, clear_store, set_metrics, get_metrics
import time


load_dotenv()


def planner_node(state: AgentState) -> AgentState:
    start = time.time()
    """
    Takes user query. and returns
    - rewritten query for better retrieval
    - Breaks into sub-question for complex task
    """
    print(f'Planner planning....')
    
    llm = get_llm()
    prompt = f"""You are a query planning expert.
Given a user question, do two things:
1. Rewrite the question to be more specific and retrieval-friendly
2. Break it into 2-3 focused sub-questions if it's complex. If simple, just return the rewritten question as the only sub-question.

User question: {state["query"]}

Respond in this exact format:
REFINED: <rewritten question>
SUB1: <first sub-question>
SUB2: <second sub-question (optional)>
SUB3: <third sub-question (optional)>
"""
    response = llm.invoke(prompt).content 
    lines = response.strip().split('\n')

    refined_query = state['query']
    sub_questions = []

    for line in lines:
        if line.startswith("REFINED:"):
            refined_query = line.replace("REFINED:","").strip()
        elif line.startswith("SUB1:") or line.startswith("SUB2:") or line.startswith("SUB3:"):
            q = line.split(":", 1)[1].strip()
            if q:
                sub_questions.append(q)
    
    if not sub_questions:
        sub_questions = [refined_query]

    print(f'refined subquery: {refined_query}')
    print(f' sub-questions: {sub_questions}')

    metrics = get_metrics()
    metrics["planner_latency"] = round(time.time() - start, 2)
    set_metrics(metrics)

    return {
        "refined_query": refined_query,
        "sub_questions": sub_questions
    }
from src.agent.state import AgentState, set_docs, get_docs, clear_store

def retriever_node(state: AgentState) -> AgentState:
    start = time.time()
    print("\n🔍 [Retriever] Retrieving documents...")

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, top_k=10)
    reranker = get_reranker()

    queries = state.get("sub_questions") or [state.get("refined_query")] or [state["query"]]
    retry_count = state.get("retry_count") or 0

    if retry_count > 0 and state.get("critic_feedback"):
        print(f"  Retry #{retry_count} — refining with critic feedback")
        queries = [f"{q} {state['critic_feedback']}" for q in queries]

    print(f"  Searching with queries: {queries}")

    all_docs = []
    seen_contents = set()

    for q in queries:
        docs = retriever.invoke(q)
        print(f"  Query '{q[:50]}' → {len(docs)} docs retrieved")
        
        reranked = rerank(q, docs, reranker, top_n=4)
        for doc in reranked:
            if doc.page_content not in seen_contents:
                all_docs.append(doc)
                seen_contents.add(doc.page_content)

    print(f"  Returning retrieved_docs with {len(all_docs)} items")
    set_docs(all_docs)  # ✅ store outside LangGraph state

    metrics = get_metrics()
    retriever_runs = metrics.get("retriever_runs", [])
    retriever_runs.append({
        "latency": round(time.time() - start, 2),
        "chunks_retrieved": len(all_docs),
        "retry": retry_count
    })
    metrics["retriever_runs"] = retriever_runs
    set_metrics(metrics)

    return {
        "retry_count": retry_count  # return something so LangGraph is happy
    }


def critic_node(state: AgentState) -> AgentState:
    start = time.time()
    print("\n🔎 [Critic] Evaluating context quality...")

    retrieved_docs = get_docs()  # ✅ read from store
    print(f"  Critic sees {len(retrieved_docs)} documents")

    if len(retrieved_docs) == 0:
        print("  No documents retrieved — forcing retry")
        return {
            "context_sufficient": False,
            "critic_feedback": "No documents were retrieved. Try a broader search query."
        }

    llm = get_llm()
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = f"""You are a strict quality evaluator for a RAG system.

User question: {state["query"]}

Retrieved context:
{context}

Evaluate if the context contains enough information to answer the question.
Respond in this exact format:
SUFFICIENT: <yes or no>
REASON: <one sentence explanation>
FEEDBACK: <if no, what specific information is missing>
"""

    response = llm.invoke(prompt).content
    lines = response.strip().split("\n")

    sufficient = False
    feedback = ""

    for line in lines:
        if line.startswith("SUFFICIENT:"):
            sufficient = "yes" in line.lower()
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()

    print(f"  Context sufficient: {sufficient}")
    metrics = get_metrics()
    metrics["critic_latency"] = round(time.time() - start, 2)
    metrics["context_sufficient"] = sufficient
    set_metrics(metrics)
    if not sufficient:
        print(f"  Feedback: {feedback}")

    return {
        "context_sufficient": sufficient,
        "critic_feedback": feedback
    }


def synthesizer_node(state: AgentState) -> AgentState:
    start = time.time()
    print("\nSynthesizer node generating answer")
    llm = get_llm()

    retrieved_docs = get_docs()  # ✅ read from store
    context = "\n\n".join(doc.page_content for doc in retrieved_docs) if retrieved_docs else "No context available."

    prompt = f"""You are a helpful research assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have enough information to answer this."
Be concise and cite specific details from the context.

Context:
{context}

Question: {state["query"]}

Answer:
"""

    answer = llm.invoke(prompt).content
    print(f"\n💡 Final Answer:\n{answer}")
    metrics = get_metrics()
    metrics["synthesizer_latency"] = round(time.time() - start, 2)
    metrics["answer_length"] = len(answer.split())
    metrics["total_latency"] = round(
        metrics.get("planner_latency", 0) +
        sum(r["latency"] for r in metrics.get("retriever_runs", [])) +
        metrics.get("critic_latency", 0) +
        (time.time() - start), 2
    )
    set_metrics(metrics)
    return {"final_answer": answer}    