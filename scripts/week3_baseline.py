"""Week 3: build the index at two chunk sizes and compare hit-rate@3 / refusal accuracy.

Usage:
    python scripts/week3_baseline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import rag_core

load_dotenv()

ROOT = Path(__file__).parent.parent
DOCS_DIR = ROOT / "data" / "docs"
QUESTIONS_PATH = ROOT / "data" / "week3_questions.json"
RESULTS_DIR = ROOT / "data" / "results"

CONFIGS = [
    {"name": "chunk_60_overlap_0", "chunk_words": 60, "overlap_words": 0},
    {"name": "chunk_700_overlap_120", "chunk_words": 700, "overlap_words": 120},
]

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_docs() -> list[tuple[str, bytes]]:
    files = sorted(DOCS_DIR.glob("*.md"))
    if not files:
        raise SystemExit(f"No documents found in {DOCS_DIR}")
    return [(path.name, path.read_bytes()) for path in files]


def run_config(config: dict, docs: list[tuple[str, bytes]], questions: list[dict]) -> dict:
    chunk_count = rag_core.build_index(docs, config["chunk_words"], config["overlap_words"], EMBEDDING_MODEL)
    rows = []
    for case in questions:
        answer, retrieved = rag_core.answer_question(case["question"], top_k=3, generation_model="openai/gpt-oss-20b")
        sources = [item["source"] for item in retrieved]
        hit_at_3 = (case["expected_source"] in sources) if case["expected_source"] else None
        refused = rag_core.is_refusal(answer)
        refusal_ok = refused == case["should_refuse"]
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "expected_source": case["expected_source"],
            "retrieved_sources": sources,
            "should_refuse": case["should_refuse"],
            "refused": refused,
            "hit_at_3": hit_at_3,
            "refusal_ok": refusal_ok,
            "answer": answer,
        })
        mark = "OK" if (hit_at_3 in (True, None) and refusal_ok) else "FAIL"
        print(f"  [{mark}] {case['id']}: hit@3={hit_at_3} refusal_ok={refusal_ok}")

    retrieval_rows = [r for r in rows if r["hit_at_3"] is not None]
    hit_rate = sum(r["hit_at_3"] for r in retrieval_rows) / len(retrieval_rows) if retrieval_rows else 0.0
    refusal_rate = sum(r["refusal_ok"] for r in rows) / len(rows)
    return {
        "config": config,
        "chunk_count": chunk_count,
        "hit_rate_at_3": hit_rate,
        "refusal_pass_rate": refusal_rate,
        "rows": rows,
    }


def main() -> int:
    docs = load_docs()
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for config in CONFIGS:
        print(f"\n=== Building index: {config['name']} ({config['chunk_words']}w / {config['overlap_words']}o) ===")
        result = run_config(config, docs, questions)
        out_path = RESULTS_DIR / f"week3_{config['name']}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"hit-rate@3: {result['hit_rate_at_3']:.1%}  refusal pass rate: {result['refusal_pass_rate']:.1%}  chunks: {result['chunk_count']}")
        summary.append({
            "config": config["name"],
            "chunk_words": config["chunk_words"],
            "overlap_words": config["overlap_words"],
            "chunk_count": result["chunk_count"],
            "hit_rate_at_3": result["hit_rate_at_3"],
            "refusal_pass_rate": result["refusal_pass_rate"],
        })

    (RESULTS_DIR / "week3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== Summary ===")
    for row in summary:
        print(f"{row['config']}: chunks={row['chunk_count']} hit@3={row['hit_rate_at_3']:.1%} refusal={row['refusal_pass_rate']:.1%}")

    # Rebuild with the better-performing config (by hit_rate, tie-break refusal) so the app's active index is the good one.
    best = max(summary, key=lambda r: (r["hit_rate_at_3"], r["refusal_pass_rate"]))
    best_config = next(c for c in CONFIGS if c["name"] == best["config"])
    print(f"\nRebuilding active index with best config: {best['config']}")
    rag_core.build_index(docs, best_config["chunk_words"], best_config["overlap_words"], EMBEDDING_MODEL)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
