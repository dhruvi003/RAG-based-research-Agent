import streamlit as st
import requests
import json
import os
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Agentic RAG Research Assistant")
st.caption("Multi-agent system: Planner → Retriever → Critic → Synthesizer")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file:
        if st.button("Index Document"):
            with st.spinner("Chunking and indexing..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Indexed {data['chunks']} chunks from {uploaded_file.name}")
                else:
                    st.error("Failed to index document")

    st.divider()
    st.header("⚙️ API Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        st.success("API is running ✅") if health.status_code == 200 else st.error("API is down")
    except:
        st.error("Cannot reach API")

# ── Session state for query history ──────────────────────────────────────────
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Ask", "📊 Evaluation Dashboard", "🏗️ Architecture"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Ask
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    query = st.text_input(
        "Ask a question about your document",
        placeholder="e.g. What multi-task approach was proposed for NER?"
    )

    if st.button("🔍 Ask", disabled=not query):
        with st.spinner("Running agentic pipeline..."):
            start = time.time()
            response = requests.post(f"{API_URL}/query", json={"query": query})
            wall_time = round(time.time() - start, 2)

        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics", {})

            # Save to history
            st.session_state.query_history.append({
                "query": query,
                "answer": data["answer"],
                "metrics": metrics,
                "retry_count": data.get("retry_count", 0),
                "context_sufficient": data.get("context_sufficient", False)
            })

            # ── Answer ────────────────────────────────────────────────────────
            st.subheader("💡 Answer")
            st.success(data["answer"])

            # ── Latency breakdown ─────────────────────────────────────────────
            st.subheader("⏱️ Latency Breakdown")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🧠 Planner", f"{metrics.get('planner_latency', 0)}s")
            col2.metric("🔍 Retriever", f"{metrics.get('retriever_runs', [{}])[0].get('latency', 0)}s")
            col3.metric("🔎 Critic", f"{metrics.get('critic_latency', 0)}s")
            col4.metric("✍️ Synthesizer", f"{metrics.get('synthesizer_latency', 0)}s")

            st.info(f"**Total latency:** {metrics.get('total_latency', wall_time)}s")

            # ── Agent trace ───────────────────────────────────────────────────
            with st.expander("🧠 Agent Trace", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**🗺️ Planner**")
                    st.markdown("*Refined query:*")
                    st.info(data.get("refined_query", "N/A"))
                    st.markdown("*Sub-questions:*")
                    for i, sq in enumerate(data.get("sub_questions", []), 1):
                        st.markdown(f"{i}. {sq}")

                with col2:
                    st.markdown("**🔎 Critic**")
                    if data.get("context_sufficient"):
                        st.success("Context approved ✅")
                    else:
                        st.warning("Context insufficient ⚠️")
                    st.metric("Retries", data.get("retry_count", 0))
                    chunks = metrics.get("retriever_runs", [{}])[0].get("chunks_retrieved", 0)
                    st.metric("Chunks retrieved", chunks)

        else:
            st.error("Something went wrong. Is the API running?")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Evaluation Dashboard
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Evaluation Dashboard")
    history = st.session_state.query_history

    if not history:
        st.info("Ask some questions in the 'Ask' tab to populate this dashboard.")
    else:
        # ── Aggregate performance metrics (existing) ──────────────────────────
        total_queries = len(history)
        avg_latency = round(sum(h["metrics"].get("total_latency", 0) for h in history) / total_queries, 2)
        success_rate = round(sum(1 for h in history if h["context_sufficient"]) / total_queries * 100, 1)
        avg_retries = round(sum(h["retry_count"] for h in history) / total_queries, 2)

        st.markdown("### ⚡ Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Queries", total_queries)
        col2.metric("Avg Latency", f"{avg_latency}s")
        col3.metric("Context Success Rate", f"{success_rate}%")
        col4.metric("Avg Retries", avg_retries)

        st.divider()

        # ── RAGAS quality metrics ─────────────────────────────────────────────
        st.markdown("### 🧪 Quality Metrics (RAGAS)")
        st.caption("Runs offline — evaluates answer quality, faithfulness, and retrieval precision.")

        if "ragas_results" not in st.session_state:
            st.session_state.ragas_results = None

        if st.button("▶️ Run RAGAS Evaluation", help="This may take 2-3 mins"):
            with st.spinner("Running RAGAS evaluation on query history..."):
                response = requests.post(
                    f"{API_URL}/evaluate",
                    json={"history": history}
                )
                if response.status_code == 200:
                    st.session_state.ragas_results = response.json()
                else:
                    st.error("Evaluation failed")

        if st.session_state.ragas_results:
            r = st.session_state.ragas_results
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Faithfulness", r.get("faithfulness", "N/A"))
            col2.metric("Answer Relevancy", r.get("answer_relevancy", "N/A"))
            col3.metric("Context Precision", r.get("context_precision", "N/A"))
            col4.metric("Context Recall", r.get("context_recall", "N/A"))

        st.divider()

        # ── Per query breakdown (existing) ────────────────────────────────────
        st.markdown("### Per Query Breakdown")
        for i, h in enumerate(reversed(history), 1):
            with st.expander(f"Query {total_queries - i + 1}: {h['query'][:60]}..."):
                st.markdown(f"**Answer:** {h['answer']}")
                m = h["metrics"]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Planner", f"{m.get('planner_latency', 0)}s")
                c2.metric("Retriever", f"{m.get('retriever_runs', [{}])[0].get('latency', 0)}s")
                c3.metric("Critic", f"{m.get('critic_latency', 0)}s")
                c4.metric("Synthesizer", f"{m.get('synthesizer_latency', 0)}s")
# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Architecture
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🏗️ System Architecture")

    st.markdown("""
    ### How it works

    This system is a **multi-agent RAG pipeline** built with LangGraph.
    Each agent has a single responsibility and they communicate via shared state.
    """)

    st.code("""
    User Query
        ↓
    [Planner]     — rewrites query, generates sub-questions
        ↓
    [Retriever]   — FAISS retrieval + cross-encoder reranking
        ↓
    [Critic]      — evaluates context quality
        ↓ (insufficient → retry retriever, max 2 retries)
    [Synthesizer] — generates grounded answer
        ↓
    Final Answer
    """)

    st.markdown("### Tech Stack")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Component | Tool |
        |---|---|
        | Agent orchestration | LangGraph |
        | LLM | Groq (Llama 3) |
        | Embeddings | HuggingFace (MiniLM) |
        | Vector store | FAISS |
        | Reranker | Cross-Encoder (MS-MARCO) |
        """)
    with col2:
        st.markdown("""
        | Component | Tool |
        |---|---|
        | Backend | FastAPI |
        | Frontend | Streamlit |
        | Evaluation | RAGAS |
        | Chunking | LangChain |
        | Containerization | Docker |
        """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with LangGraph · FAISS · HuggingFace · Groq · FastAPI · Streamlit")