from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from langchain_classic.schema import Document
from typing import List
import pandas as pd

def build_eval_dataset(
    questions: List[str],
    ground_truths: List[str],
    qa_chain,
    retriever
) -> Dataset:
    rows = []

    for question, ground_truth in zip(questions, ground_truths):
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]
        answer = qa_chain.invoke(question)

        rows.append({
            "question": question,        # ✅ must be "question"
            "answer": answer,            # ✅ must be "answer"
            "contexts": contexts,        # ✅ must be "contexts" (list of strings)
            "ground_truth": ground_truth # ✅ must be "ground_truth" (single string)
        })

        print(f"✅ Evaluated: {question[:60]}...")

    # Debug — print before converting
    print("\n🔍 Sample row keys:", rows[0].keys())
    print("🔍 Contexts type:", type(rows[0]["contexts"]))
    print("🔍 Ground truth type:", type(rows[0]["ground_truth"]))

    dataset = Dataset.from_list(rows)
    print("🔍 Dataset columns:", dataset.column_names)
    return dataset

def run_ragas_evaluation(dataset: Dataset, llm, embeddings) -> pd.DataFrame:
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ✅ 0.4.x API — assign llm/embeddings to each metric directly
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embeddings
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    df = results.to_pandas()
    return df