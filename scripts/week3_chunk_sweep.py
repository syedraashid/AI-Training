"""Week 3 (supplement): hit-rate@3 saturates at 100% with only 6 source documents and
top_k=3, so it hides the real effect of chunk size. This sweep uses hit-rate@1 (is the
single best-ranked chunk from the right document?) across several chunk sizes, using
retrieval only (no LLM calls, fast) so more configs can be compared directly.

Usage:
    python scripts/week3_chunk_sweep.py
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
    {"name": "60w/0o", "chunk_words": 60, "overlap_words": 0},
    {"name": "150w/20o", "chunk_words": 150, "overlap_words": 20},
    {"name": "400w/80o", "chunk_words": 400, "overlap_words": 80},
    {"name": "700w/120o", "chunk_words": 700, "overlap_words": 120},
]
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_docs():
    files = sorted(DOCS_DIR.glob("*.md"))
    return [(path.name, path.read_bytes()) for path in files]


def main() -> int:
    docs = load_docs()
    questions = [q for q in json.loads(QUESTIONS_PATH.read_text(encoding="utf-8")) if q["expected_source"]]

    summary = []
    for config in CONFIGS:
        chunk_count = rag_core.build_index(docs, config["chunk_words"], config["overlap_words"], EMBEDDING_MODEL)
        hits_1 = hits_3 = 0
        misses = []
        for case in questions:
            results, _ = rag_core.retrieve(case["question"], top_k=3, mode="dense")
            sources = [r["source"] for r in results]
            if sources and sources[0] == case["expected_source"]:
                hits_1 += 1
            else:
                misses.append(case["id"])
            if case["expected_source"] in sources:
                hits_3 += 1
        row = {
            "config": config["name"],
            "chunk_count": chunk_count,
            "hit_rate_at_1": hits_1 / len(questions),
            "hit_rate_at_3": hits_3 / len(questions),
            "misses_at_1": misses,
        }
        summary.append(row)
        print(f"{config['name']:12s} chunks={chunk_count:3d}  hit@1={row['hit_rate_at_1']:.1%}  hit@3={row['hit_rate_at_3']:.1%}  misses@1={misses}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week3_chunk_sweep.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
