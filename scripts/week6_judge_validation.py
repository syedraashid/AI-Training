"""Week 6: validate the LLM judge against human grades on a small sample before trusting it.

For each sample question we (a) generate a fresh answer through the real pipeline, (b) apply a
human grade with an explicit written reason (the "human" here is a careful manual read against
the source documents, done before looking at the judge's score), and (c) ask the LLM judge to
grade the same answer blind. We then report agreement.

Usage:
    python scripts/week6_judge_validation.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import rag_core

load_dotenv()

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "results"

# Human grade is written BEFORE running the judge, based on what the source documents actually say.
SAMPLE = [
    {"question": "How many days do I have to return an unopened item for a full refund?",
     "human_score": 5, "human_reason": "Correct (30 days), grounded, cited."},
    {"question": "What does error code ERR-4032 mean and how should I fix it?",
     "human_score": 5, "human_reason": "Correct cause and fix, grounded, cited."},
    {"question": "Can I get a refund as cash or store credit instead of back to my card?",
     "human_score": 5, "human_reason": "Docs explicitly say original-payment-method only, no cash/store credit — a clear no."},
    {"question": "What is the CEO's direct phone number?",
     "human_score": 5, "human_reason": "Correctly out of scope; refusal is the right answer."},
    {"question": "Do you ship to other countries?",
     "human_score": 5, "human_reason": "Correct no, grounded, cited."},
    {"question": "which is cheaper for a $50 order, standard or express shipping",
     "human_score": 4, "human_reason": "Requires combining two facts (free standard over $35, express $12.99); correct if it states standard is free and cheaper, but easy to get partially right."},
    {"question": "if i return an opened but broken item do i still pay the 15% restocking fee",
     "human_score": 4, "human_reason": "Docs say restocking fee applies only to non-defective opened returns, so a broken item should NOT pay it. Easy to conflate 'opened' with the fee and answer wrong."},
    {"question": "how much does the extended warranty cost",
     "human_score": 5, "human_reason": "No price is stated anywhere in the docs, so the only correct answer is a refusal — anything else is a hallucinated price."},
]


def main() -> int:
    rows = []
    for case in SAMPLE:
        answer, retrieved = rag_core.answer_question(case["question"], top_k=3, mode="hybrid")
        context = "\n\n".join(f"[Source: {r['source']} | {r['location']}]\n{r['text']}" for r in retrieved)
        judge = rag_core.judge_answer(case["question"], answer, context)
        agree = judge["score"] is not None and abs(judge["score"] - case["human_score"]) <= 1
        rows.append({
            "question": case["question"],
            "answer": answer,
            "human_score": case["human_score"],
            "human_reason": case["human_reason"],
            "judge_score": judge["score"],
            "judge_reasoning": judge["reasoning"],
            "agree_within_1": agree,
        })
        print(f"Q: {case['question']}")
        print(f"  human={case['human_score']} judge={judge['score']} agree={agree}")
        print(f"  judge said: {judge['reasoning']}")
        print()

    n = len(rows)
    agreement_rate = sum(r["agree_within_1"] for r in rows) / n
    exact_rate = sum(r["human_score"] == r["judge_score"] for r in rows) / n
    print(f"Agreement within 1 point: {agreement_rate:.1%} ({sum(r['agree_within_1'] for r in rows)}/{n})")
    print(f"Exact agreement: {exact_rate:.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week6_judge_validation.json").write_text(
        json.dumps({"agreement_within_1": agreement_rate, "exact_agreement": exact_rate, "rows": rows}, indent=2),
        encoding="utf-8",
    )

    if agreement_rate < 0.7:
        print("\nJudge does not agree closely enough with human grading — do not trust it yet.")
    else:
        print("\nJudge agrees closely enough with human grading to use for subjective scoring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
