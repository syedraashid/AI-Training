import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_core import INDEX_PATH, REFUSAL_TEXT, answer_question, build_index, load_index

load_dotenv()
st.set_page_config(page_title="Customer Support RAG Lab", page_icon="🎫", layout="wide")

st.title("🎫 Customer Support RAG Lab")
st.caption("Weeks 3–6: build, inspect, measure, analyse, and evaluate a RAG app.")

with st.sidebar:
    st.header("1. Configure")
    embedding_model = st.text_input("Local embedding model", "all-MiniLM-L6-v2")
    generation_model = st.text_input("Groq answer model", "openai/gpt-oss-20b")
    chunk_words = st.slider("Chunk size (words)", 150, 1_000, 400, 50)
    overlap_words = st.slider("Chunk overlap (words)", 0, 300, 80, 10)
    top_k = st.slider("Retrieved chunks (top-k)", 1, 8, 3)
    retrieval_mode = st.selectbox(
        "Retrieval mode", ["dense", "bm25", "hybrid"], index=2,
        help="dense = Week 3 baseline (embeddings only). bm25 = exact-keyword search. "
             "hybrid = Week 4 improvement: reciprocal rank fusion of both.",
    )
    st.divider()
    st.header("2. Upload tickets")
    uploads = st.file_uploader("PDF, DOCX, TXT, or MD", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
    if st.button("Build / rebuild index", type="primary", disabled=not uploads):
        try:
            count = build_index([(item.name, item.getvalue()) for item in uploads], chunk_words, overlap_words, embedding_model)
            st.success(f"Indexed {count} chunks. Now ask a question.")
        except Exception as error:
            st.error(str(error))

if not os.getenv("GROQ_API_KEY"):
    st.warning("Set GROQ_API_KEY in a .env file before asking questions. The embedding model downloads locally on first use.")

tab_chat, tab_inspect, tab_traces, tab_weekly = st.tabs(["Ask", "Inspect retrieval", "Traces", "Week checklist"])

with tab_chat:
    question = st.text_area("Ask a customer-support question", placeholder="What is the refund policy for a damaged product?")
    if st.button("Answer from documents", disabled=not question.strip()):
        try:
            answer, results = answer_question(question, top_k, generation_model, retrieval_mode)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Retrieved evidence")
            for result in results:
                st.markdown(f"**{result['source']} — {result['location']}** · similarity `{result['score']}`")
                st.caption(result["text"][:500] + ("..." if len(result["text"]) > 500 else ""))
        except Exception as error:
            st.error(str(error))

with tab_inspect:
    inspect_question = st.text_input("Question to inspect", key="inspect")
    if st.button("Show retrieved chunks", disabled=not inspect_question.strip()):
        try:
            from rag_core import retrieve
            results, _ = retrieve(inspect_question, top_k, retrieval_mode)
            for rank, result in enumerate(results, start=1):
                with st.expander(f"#{rank}: {result['source']} — {result['location']} (score {result['score']})", expanded=True):
                    st.write(result["text"])
        except Exception as error:
            st.error(str(error))

with tab_traces:
    traces_path = Path("data/traces.jsonl")
    if traces_path.exists():
        st.download_button("Download traces", traces_path.read_bytes(), "traces.jsonl", "application/jsonl")
        st.code("\n".join(traces_path.read_text(encoding="utf-8").splitlines()[-10:]), language="json")
    else:
        st.info("Traces will appear after you ask questions. Use them for Week 5 error analysis.")

with tab_weekly:
    st.markdown(f"""
### Week 3 — baseline
- Upload your ticket documents, test two chunk sizes, and record the results.
- Ask supported and unsupported questions. Unsupported questions must produce: `{REFUSAL_TEXT}`

### Week 4 — retrieval debugging
- Use **Inspect retrieval** to label each failure as retrieval or generation.
- Make one change only, then measure hit-rate@3 using `python run_evals.py`.

### Week 5 — error analysis
- Download traces after roughly 20 real questions.
- Write one honest note per failure before grouping them into named error types.

### Week 6 — automated evaluation
- Edit `data/eval_cases.json` with real questions, expected sources, refusal cases, and problem types.
- Run `python run_evals.py` before and after one improvement.
""")

if INDEX_PATH.exists():
    try:
        index = load_index()
        st.sidebar.success(f"Current index: {len(index['chunks'])} chunks ({index['chunk_words']} words, {index['overlap_words']} overlap)")
    except Exception:
        pass
