from src.ingestion.loader import load_documents
from src.ingestion.chunker import compare_strategies
from src.ingestion.loader import load_documents
from src.ingestion.chunker import compare_strategies, filter_noise_chunks
from src.retrieval.vectorstore import build_vectorstore, load_vectorstore, get_retriever
from src.retrieval.reranker import get_reranker, rerank
from src.retrieval.qa_chain import build_qa_chain, get_llm
from src.evaluation.ragas_eval import build_eval_dataset, run_ragas_evaluation
from src.retrieval.vectorstore import get_embeddings
from src.agent.graph import build_graph
import os
from src.agent.state import clear_store


if __name__ == "__main__":
    # documents = load_documents("data\W17-4419.pdf")
    # results = compare_strategies(documents)

    # for i, chunk in enumerate(results["recursive"][:3]):
    #     print(f"\n--- Chunk {i+1} ---")
    #     print(chunk.page_content)
    # chunks = results['recursive']
    # chunks = filter_noise_chunks(chunks)
    # index_path = "vectorstore/faiss_index"

    # if os.path.exists(index_path):
    #     vectorstore = load_vectorstore(index_path)
    # else:
    #     vectorstore = build_vectorstore(chunks, index_path)
    
    # retriever = get_retriever(vectorstore)
    # query = "What is the Named Entity Recognition?"
    # retrieved_docs = retriever.invoke(query)

    # print(f'Query {query}')
    # print(f'retrieved docs: {len(retrieved_docs)} chunks')
    # for i, doc in enumerate(retrieved_docs):
    #     print(f"--- Result {i+1} ---")
    #     print(doc.page_content[:300])
    #     print()


# reranker = get_reranker()
# reranked_docs = rerank(query, retrieved_docs, reranker, top_n=4)
# ── Step 1c: QA chain ────────────────────────────────────────────
    # llm = get_llm()
    # qa_chain = build_qa_chain(retriever, llm)

    # Quick test
    query = "What is Named Entity Recognition?"
    # print("\n🤖 Answer:", qa_chain.invoke(query))

# print(f"\n Top 4 after reranking:")
# for i, doc in enumerate(reranked_docs):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content[:300])

    questions = [
        "What is Named Entity Recognition?",
        "What dataset was used in the paper?",
        "What was the F1 score achieved?",
        "What makes social media NER challenging?",
    ]

    ground_truths = [
        "Named Entity Recognition aims at identifying different types of entities such as people names, companies, and locations within a given text.",
        "The WNUT-2017 dataset was used, from the 3rd Workshop on Noisy User-generated Text.",
        "The system achieved a 41.86% entity F1-score and a 40.24% surface F1-score.",
        "Social media NER is challenging due to improper grammatical structures, spelling inconsistencies, and numerous informal abbreviations.",
    ]

    # print("\n📊 Building evaluation dataset...")
    # eval_dataset = build_eval_dataset(questions, ground_truths, qa_chain, retriever)

    # embeddings = get_embeddings()
    # print("\n🧪 Running RAGAS evaluation...")
    # results_df = run_ragas_evaluation(eval_dataset, llm, embeddings)

    # print("\n📈 RAGAS Results:")
    # print(results_df.to_string())  # print all columns first so we can see what's there

    # print("\n📊 Average Scores:")
    # numeric_cols = results_df.select_dtypes(include='number').columns
    # print(results_df[numeric_cols].mean())


    graph = build_graph()

    initial_state = {
    "query": "What multi-task approach was proposed for NER and what results did it achieve?",
    "refined_query": None,
    "sub_questions": None,
    "context_sufficient": None,
    "critic_feedback": None,
    "retry_count": 0,
    "final_answer": None
}

    print("\n🚀 Running Agentic RAG...\n")
    clear_store()
    final_state = graph.invoke(initial_state)

    print("\n" + "="*60)
    print("FINAL ANSWER:")
    print("="*60)
    print(final_state["final_answer"])