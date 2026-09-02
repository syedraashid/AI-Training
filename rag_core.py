"""Small, transparent RAG pipeline for the customer-support-ticket training project."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from docx import Document
from groq import Groq
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "index.json"
TRACES_PATH = DATA_DIR / "traces.jsonl"
REFUSAL_TEXT = "I don't know based on the provided customer-support documents."


def client() -> Groq:
    """Create the Groq client using the user's GROQ_API_KEY."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing. Add it to .env or your environment.")
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def extract_pages(filename: str, content: bytes) -> list[dict[str, Any]]:
    """Extract text while retaining a source name and page/section identifier."""
    suffix = Path(filename).suffix.lower()
    temp_path = DATA_DIR / f"_upload{suffix}"
    temp_path.write_bytes(content)
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(temp_path))
            return [
                {"source": filename, "location": f"page {i + 1}", "text": page.extract_text() or ""}
                for i, page in enumerate(reader.pages)
            ]
        if suffix == ".docx":
            document = Document(str(temp_path))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            return [{"source": filename, "location": "document", "text": text}]
        if suffix in {".txt", ".md"}:
            return [{"source": filename, "location": "document", "text": content.decode("utf-8", errors="replace")}]
        raise ValueError(f"Unsupported file type: {suffix}. Use PDF, DOCX, TXT, or Markdown.")
    finally:
        temp_path.unlink(missing_ok=True)


def chunk_text(text: str, chunk_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap_words)
    return [" ".join(words[start : start + chunk_words]) for start in range(0, len(words), step)]


@lru_cache(maxsize=2)
def _load_embedding_model(embedding_model: str) -> SentenceTransformer:
    return SentenceTransformer(embedding_model)


def embed_texts(texts: list[str], embedding_model: str) -> list[list[float]]:
    """Create free, local dense embeddings; first use downloads the selected model."""
    model = _load_embedding_model(embedding_model)
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def build_index(files: list[tuple[str, bytes]], chunk_words: int, overlap_words: int, embedding_model: str) -> int:
    DATA_DIR.mkdir(exist_ok=True)
    chunks: list[dict[str, Any]] = []
    for filename, content in files:
        for page in extract_pages(filename, content):
            for number, text in enumerate(chunk_text(page["text"], chunk_words, overlap_words), start=1):
                chunks.append({
                    "id": f"{filename}:{page['location']}:chunk-{number}",
                    "source": filename,
                    "location": page["location"],
                    "text": text,
                })
    if not chunks:
        raise ValueError("No readable text was found in the uploaded files.")
    vectors = embed_texts([chunk["text"] for chunk in chunks], embedding_model)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    INDEX_PATH.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "embedding_model": embedding_model,
        "chunks": chunks,
    }), encoding="utf-8")
    return len(chunks)


def load_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        raise RuntimeError("No index exists yet. Upload documents and click Build / rebuild index.")
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


RRF_K = 60
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/code tokenizer. Keeps hyphenated codes like ERR-4032 as one token,
    and also emits their parts (err, 4032) so a bare-number query like "4032" still matches."""
    tokens = TOKEN_PATTERN.findall(text.lower())
    for token in list(tokens):
        if "-" in token:
            tokens.extend(token.split("-"))
    return tokens


def _dense_scores(question: str, index: dict[str, Any]) -> dict[str, float]:
    query_vector = np.array(embed_texts([question], index["embedding_model"])[0], dtype=float)
    scores: dict[str, float] = {}
    for chunk in index["chunks"]:
        vector = np.array(chunk["embedding"], dtype=float)
        scores[chunk["id"]] = float(
            np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
        )
    return scores


def _bm25_scores(question: str, index: dict[str, Any]) -> dict[str, float]:
    corpus = [_tokenize(chunk["text"]) for chunk in index["chunks"]]
    bm25 = BM25Okapi(corpus)
    raw_scores = bm25.get_scores(_tokenize(question))
    return {chunk["id"]: float(score) for chunk, score in zip(index["chunks"], raw_scores)}


def _rrf_fuse(*rankings: dict[str, float]) -> dict[str, float]:
    """Reciprocal rank fusion: combine multiple id->score maps into one id->fused-score map."""
    fused: dict[str, float] = {}
    for scores in rankings:
        for rank, chunk_id in enumerate(sorted(scores, key=lambda key: scores[key], reverse=True), start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
    return fused


def retrieve(
    question: str, top_k: int = 3, mode: str = "dense"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve the top-k chunks. mode is 'dense' (embeddings only), 'bm25' (keyword only),
    or 'hybrid' (reciprocal rank fusion of both) — hybrid is the Week 4 improvement over
    the Week 3 dense-only baseline, added so exact terms like error codes aren't missed."""
    index = load_index()
    by_id = {chunk["id"]: chunk for chunk in index["chunks"]}

    if mode == "dense":
        scored = _dense_scores(question, index)
    elif mode == "bm25":
        scored = _bm25_scores(question, index)
    elif mode == "hybrid":
        scored = _rrf_fuse(_dense_scores(question, index), _bm25_scores(question, index))
    else:
        raise ValueError(f"Unknown retrieval mode: {mode}. Use 'dense', 'bm25', or 'hybrid'.")

    ranked_ids = sorted(scored, key=lambda key: scored[key], reverse=True)[:top_k]
    results = [{**by_id[chunk_id], "score": round(scored[chunk_id], 4)} for chunk_id in ranked_ids]
    return results, index


def answer_question(
    question: str, top_k: int = 3, generation_model: str = "openai/gpt-oss-20b", mode: str = "dense"
) -> tuple[str, list[dict[str, Any]]]:
    results, _ = retrieve(question, top_k, mode)
    context = "\n\n".join(
        f"[Source: {item['source']} | {item['location']} | chunk: {item['id']}]\n{item['text']}"
        for item in results
    )
    prompt = f"""You are a careful customer-support assistant.
Answer the user's question using ONLY the supplied support-document excerpts.
If an excerpt states a rule or fact that directly answers the question once applied to the specifics
given — including a single, direct inference, such as explaining why a stated rule produced an
outcome, or confirming/denying a request against a rule the excerpts state explicitly — answer using
that rule. Do not refuse just because the answer isn't quoted verbatim; refuse only when the excerpts
contain no rule or fact that addresses the question at all.
Every factual answer must cite its source in the format [filename | page/section].
If the documents do not contain a rule or fact that addresses the question, reply exactly: {REFUSAL_TEXT}
Do not use outside knowledge. Do not invent policies, dates, refunds, or steps.

User question: {question}

Support-document excerpts:
{context}
"""
    response = client().chat.completions.create(
        model=generation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = (response.choices[0].message.content or "").strip()
    log_trace(question, results, answer, generation_model, mode)
    return answer, results


def log_trace(
    question: str, retrieved: list[dict[str, Any]], answer: str, generation_model: str, mode: str = "dense"
) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "mode": mode,
        "retrieved": [{key: item[key] for key in ("id", "source", "location", "score")} for item in retrieved],
        "answer": answer,
        "generation_model": generation_model,
    }
    with TRACES_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")


_DASH_VARIANTS = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",  # hyphen/dash variants
    "’": "'", "‘": "'",  # curly single quotes, e.g. "don't"
})


def is_refusal(answer: str) -> bool:
    """Detect the exact refusal string. Some generation models render an ASCII hyphen or
    apostrophe as a typographic Unicode variant (seen in practice: U+2011 NON-BREAKING HYPHEN
    in "customer-support"), which silently broke a naive substring match — a real refusal was
    scored as a normal answer. Normalize common variants before comparing."""
    normalized = answer.lower().translate(_DASH_VARIANTS)
    return "i don't know based on the provided customer-support documents" in normalized


JUDGE_RUBRIC = """You are grading one customer-support ticket reply for quality. Score it 1-5:
5 = fully correct, grounded only in the excerpts, cites a source, and reads like a helpful agent.
4 = correct and grounded, but citation is missing/malformed or tone is slightly off.
3 = partially correct: right idea but missing an important detail the excerpts contained.
2 = mostly wrong, or states something the excerpts do not support (partial hallucination).
1 = confidently wrong, invents a policy/number, or refuses when the excerpts clearly answer it
    (or answers when it should have refused).
Judge ONLY against the excerpts given below — do not use outside knowledge of what a "normal"
policy should say.

Question: {question}

Support-document excerpts the assistant had access to:
{context}

Assistant's reply:
{answer}

Reply with strict JSON only, no other text: {{"score": <integer 1-5>, "reasoning": "<one sentence>"}}
"""


def judge_answer(
    question: str, answer: str, context: str, generation_model: str = "openai/gpt-oss-20b"
) -> dict[str, Any]:
    """LLM-as-judge for subjective reply quality (helpfulness/tone), scored 1-5.
    Must be validated against human grades on a sample before being trusted (see
    scripts/week6_judge_validation.py) — an unchecked judge is just a confident number."""
    prompt = JUDGE_RUBRIC.format(question=question, context=context, answer=answer)
    response = client().chat.completions.create(
        model=generation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
        return {"score": int(parsed["score"]), "reasoning": str(parsed["reasoning"])}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": None, "reasoning": f"unparseable judge output: {raw[:200]}"}
