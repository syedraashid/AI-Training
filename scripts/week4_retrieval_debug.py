"""Week 4: label failures as retrieval-vs-generation, then compare dense (before) vs
hybrid RRF (after) retrieval with a before/after hit-rate number.

Usage:
    python scripts/week4_retrieval_debug.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import rag_core

load_dotenv()

ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = ROOT / "data" / "week4_questions.json"
RESULTS_DIR = ROOT / "data" / "results"


def evaluate(mode: str, questions: list[dict]) -> dict:
    rows = []
    for case in questions:
        results, _ = rag_core.retrieve(case["question"], top_k=3, mode=mode)
        sources = [item["source"] for item in results]
        hit_at_1 = sources[0] == case["expected_source"] if sources else False
        hit_at_3 = case["expected_source"] in sources
        rows.append({
            "id": case["id"],
            "question": case["question"],
            "expected_source": case["expected_source"],
            "retrieved_sources": sources,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
        })
    hit_rate_1 = sum(r["hit_at_1"] for r in rows) / len(rows)
    hit_rate_3 = sum(r["hit_at_3"] for r in rows) / len(rows)
    return {"mode": mode, "hit_rate_at_1": hit_rate_1, "hit_rate_at_3": hit_rate_3, "rows": rows}


def main() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    before = evaluate("dense", questions)
    after = evaluate("hybrid", questions)

    (RESULTS_DIR / "week4_before_dense.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    (RESULTS_DIR / "week4_after_hybrid.json").write_text(json.dumps(after, indent=2), encoding="utf-8")

    lines_before = [f"Retrieval mode: dense (Week 3 baseline)"]
    lines_after = [f"Retrieval mode: hybrid (Week 4 improvement: RRF fusion of dense + BM25)"]
    for label, report, lines in (("before", before, lines_before), ("after", after, lines_after)):
        lines.append(f"hit-rate@1: {report['hit_rate_at_1']:.1%}")
        lines.append(f"hit-rate@3: {report['hit_rate_at_3']:.1%}")
        lines.append("")
        for row in report["rows"]:
            mark = "OK " if row["hit_at_1"] else ("~  " if row["hit_at_3"] else "FAIL"[:4])
            lines.append(f"[{mark}] {row['id']}: '{row['question']}' -> expected={row['expected_source']} got_top3={row['retrieved_sources']}")
        print("\n".join(lines))
        print()

    (RESULTS_DIR / "before.txt").write_text("\n".join(lines_before), encoding="utf-8")
    (RESULTS_DIR / "after.txt").write_text("\n".join(lines_after), encoding="utf-8")

    print("=== Delta ===")
    print(f"hit-rate@1: {before['hit_rate_at_1']:.1%} -> {after['hit_rate_at_1']:.1%}")
    print(f"hit-rate@3: {before['hit_rate_at_3']:.1%} -> {after['hit_rate_at_3']:.1%}")

    unfixed = [
        r_after["id"] for r_before, r_after in zip(before["rows"], after["rows"])
        if not r_before["hit_at_1"] and not r_after["hit_at_1"]
    ]
    if unfixed:
        print(f"\nStill failing after the change (hybrid did not fix these): {unfixed}")
    else:
        print("\nAll cases that failed under dense-only retrieval are fixed by hybrid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
