from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from src.agent.graph import build_graph
from src.agent.state import clear_store
from src.ingestion.loader import load_documents
from src.ingestion.chunker import recursive_chunker, filter_noise_chunks
from src.retrieval.vectorstore import build_vectorstore
import shutil
import os
import asyncio
import json
from src.evaluation.ragas_eval import build_eval_dataset, run_ragas_evaluation
from src.retrieval.vectorstore import load_vectorstore, get_retriever, get_embeddings
from src.retrieval.reranker import get_reranker, rerank
from src.retrieval.qa_chain import build_qa_chain, get_llm

class EvaluateRequest(BaseModel):
    history: list

app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = build_graph()

# ── Request schema ────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── Upload + index a document ─────────────────────────────────────────────────
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF or txt file, chunks it, and builds the vectorstore.
    """
    upload_path = f"data/{file.filename}"
    os.makedirs("data", exist_ok=True)

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process
    documents = load_documents(upload_path)
    chunks = recursive_chunker(documents)
    chunks = filter_noise_chunks(chunks, min_length=200)
    build_vectorstore(chunks, save_path="vectorstore/faiss_index")

    return {
        "message": f"Indexed {file.filename}",
        "chunks": len(chunks)
    }


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Streams agent events back to the client using Server-Sent Events.
    Each agent step is sent as an event so the frontend can show progress.
    """
    async def event_generator():
        clear_store()

        initial_state = {
            "query": request.query,
            "refined_query": None,
            "sub_questions": None,
            "context_sufficient": None,
            "critic_feedback": None,
            "retry_count": 0,
            "final_answer": None
        }

        # Stream each node's output as it happens
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event")
            name = event.get("name", "")

            # Node started
            if kind == "on_chain_start" and name in ["planner", "retriever", "critic", "synthesizer"]:
                data = json.dumps({"type": "node_start", "node": name})
                yield {"data": data}
                await asyncio.sleep(0)

            # Node finished
            elif kind == "on_chain_end" and name in ["planner", "retriever", "critic", "synthesizer"]:
                output = event.get("data", {}).get("output", {})
                data = json.dumps({
                    "type": "node_end",
                    "node": name,
                    "output": {k: str(v)[:200] for k, v in output.items() if v is not None}
                })
                yield {"data": data}
                await asyncio.sleep(0)

        # Send final answer at the end
        final_state = graph.invoke(initial_state)
        data = json.dumps({
            "type": "final_answer",
            "answer": final_state.get("final_answer", "No answer generated.")
        })
        yield {"data": data}

    return EventSourceResponse(event_generator())


# ── Non-streaming query (simpler fallback) ────────────────────────────────────
from src.agent.state import clear_store, get_metrics
@app.post("/query")
async def query(request: QueryRequest):
    clear_store()

    initial_state = {
        "query": request.query,
        "refined_query": None,
        "sub_questions": None,
        "context_sufficient": None,
        "critic_feedback": None,
        "retry_count": 0,
        "final_answer": None
    }

    final_state = graph.invoke(initial_state)
    metrics = get_metrics()
    # print("FINAL STATE:", final_state)
    return {
        "query": request.query,
        "answer": final_state.get("final_answer", "No answer generated."),
        "refined_query": final_state.get("refined_query"),
        "sub_questions": final_state.get("sub_questions"),
        "context_sufficient": final_state.get("context_sufficient"),
        "retry_count": final_state.get("retry_count", 0),
        "metrics": metrics  # ✅ return metrics to frontend
    }

@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """
    Runs RAGAS evaluation on query history.
    Called from the dashboard — runs offline, not per query.
    """
    history = request.history

    questions = [h["query"] for h in history]
    ground_truths = [h["answer"] for h in history]  # using generated answers as proxy

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, top_k=10)
    llm = get_llm()
    qa_chain = build_qa_chain(retriever, llm)

    dataset = build_eval_dataset(questions, ground_truths, qa_chain, retriever)
    embeddings = get_embeddings()
    results_df = run_ragas_evaluation(dataset, llm, embeddings)

    numeric_cols = results_df.select_dtypes(include='number').columns
    averages = results_df[numeric_cols].mean().to_dict()

    return {k: round(v, 3) for k, v in averages.items()}