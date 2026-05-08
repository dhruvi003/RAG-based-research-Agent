from typing import TypedDict, List, Optional
from langchain_classic.schema import Document

# ── Side-channel store (bypasses LangGraph state merging issues) ──────────────
_store = {}

def set_docs(docs: List[Document]):
    _store["retrieved_docs"] = docs

def get_docs() -> List[Document]:
    return _store.get("retrieved_docs", [])

def set_metrics(metrics: dict):
    _store["metrics"] = metrics

def get_metrics() -> dict:
    return _store.get("metrics", {})

def clear_store():
    _store.clear()

# ── State schema ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query: str
    refined_query: Optional[str]
    sub_questions: Optional[List[str]]
    context_sufficient: Optional[bool]
    critic_feedback: Optional[str]
    retry_count: Optional[int]
    final_answer: Optional[str]