"""Run Week 6 retrieval/refusal regression tests with one command.

Usage:
    python run_evals.py                  # dense retrieval (Week 3 baseline)
    python run_evals.py hybrid           # hybrid retrieval (Week 4 improvement)
    python run_evals.py hybrid --judge   # also score subjective quality with the LLM judge
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_core import answer_question, is_refusal, judge_answer

load_dotenv()
ROOT = Path(__file__).parent
CASES_PATH = ROOT / "data" / "eval_cases.json"


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mode = args[0] if args else "dense"
    use_judge = "--judge" in sys.argv

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if any(case["id"].startswith("replace-") for case in cases):
        print("Replace the example cases in data/eval_cases.json before evaluating.")
        return 2

    results = []
    for case in cases:
        answer, retrieved = answer_question(case["question"], top_k=3, mode=mode)
        sources = [item["source"] for item in retrieved]
        hit_at_3 = case["expected_source"] in sources if case["expected_source"] else None
        refusal_ok = is_refusal(answer) == case["should_refuse"]
        row = {**case, "mode": mode, "sources": sources, "answer": answer, "hit_at_3": hit_at_3, "refusal_ok": refusal_ok}
        if use_judge:
            context = "\n\n".join(f"[Source: {r['source']} | {r['location']}]\n{r['text']}" for r in retrieved)
            row["judge"] = judge_answer(case["question"], answer, context)
        results.append(row)
        judge_note = f" judge={row['judge']['score']}" if use_judge else ""
        print(f"{case['id']}: hit@3={hit_at_3}, refusal={refusal_ok}{judge_note}")

    retrieval_cases = [item for item in results if item["hit_at_3"] is not None]
    hit_rate = sum(item["hit_at_3"] for item in retrieval_cases) / len(retrieval_cases) if retrieval_cases else 0
    refusal_rate = sum(item["refusal_ok"] for item in results) / len(results)
    print(f"\nRetrieval mode: {mode}")
    print(f"Hit-rate@3: {hit_rate:.1%}")
    print(f"Refusal assertion pass rate: {refusal_rate:.1%}")
    if use_judge:
        judged = [item for item in results if item["judge"]["score"] is not None]
        avg_judge = sum(item["judge"]["score"] for item in judged) / len(judged) if judged else 0
        print(f"Average judge score: {avg_judge:.2f}/5 (n={len(judged)})")
    print("\nScores by problem type:")
    for problem_type in sorted({item["problem_type"] for item in results}):
        group = [item for item in results if item["problem_type"] == problem_type]
        group_retrieval = [item for item in group if item["hit_at_3"] is not None]
        group_hit = sum(item["hit_at_3"] for item in group_retrieval) / len(group_retrieval) if group_retrieval else None
        group_refusal = sum(item["refusal_ok"] for item in group) / len(group)
        hit_text = f"{group_hit:.1%}" if group_hit is not None else "n/a"
        line = f"- {problem_type}: hit@3={hit_text}, refusal={group_refusal:.1%}, n={len(group)}"
        if use_judge:
            group_judged = [item for item in group if item["judge"]["score"] is not None]
            if group_judged:
                line += f", judge={sum(item['judge']['score'] for item in group_judged) / len(group_judged):.2f}/5"
        print(line)
    (ROOT / "data" / f"eval_results_{mode}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
