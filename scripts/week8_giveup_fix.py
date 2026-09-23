"""Week 8: measure the fix for the "gives up quietly" failure found by
scripts/week8_trajectory_eval.py -- openai/gpt-oss-20b sometimes ends its turn with
finish_reason=stop, empty content, and no tool call, even right after successfully
retrieving everything the ticket needs. agent_core.run_agent now retries that specific
condition (give_up_retries, default 2) instead of accepting an empty reply as done.

This is a repeated-trials measurement, not a single race: the failure is stochastic at
temperature=0 (Groq's fast inference isn't perfectly deterministic), so a single run of
each ticket proves nothing either way. Runs N repeats per ticket with give_up_retries=0
(before/baseline) and the default (after/fixed) and reports the empty-answer rate.

Usage:
    python scripts/week8_giveup_fix.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import agent_core

load_dotenv()

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "results"

REPEATS = 6
TICKETS = [
    "A customer placed an order 3 hours ago and wants to cancel it. They also have a promo "
    "code that already expired and want it applied anyway. Are either of those possible?",
    "A customer's tracking hasn't updated in 12 days, and separately they're asking if the "
    "product warranty would cover it if the item turns out to be damaged. What should I tell "
    "them about both?",
    "A customer never received their password reset email, and separately wants to know if "
    "they can get a cash refund instead of a refund to their card for an unused item still "
    "within the return window.",
]


def trial(ticket: str, give_up_retries: int) -> dict:
    result = agent_core.run_agent(ticket, give_up_retries=give_up_retries)
    return {
        "empty": result["answer"] == "",
        "retries_used": result["steps"][-1].get("gave_up_retries_used", 0),
        "total_tokens": result["total_tokens"],
        "num_llm_calls": result["num_llm_calls"],
    }


def main() -> int:
    rows = []
    for ticket in TICKETS:
        print(f"--- {ticket[:70]}... ---")
        before = [trial(ticket, give_up_retries=0) for _ in range(REPEATS)]
        after = [trial(ticket, give_up_retries=agent_core.GIVE_UP_RETRIES) for _ in range(REPEATS)]
        before_empty_rate = sum(t["empty"] for t in before) / REPEATS
        after_empty_rate = sum(t["empty"] for t in after) / REPEATS
        print(f"  before (no retry): empty rate {before_empty_rate:.0%}")
        print(f"  after  (fixed):    empty rate {after_empty_rate:.0%}  "
              f"(avg retries used: {sum(t['retries_used'] for t in after) / REPEATS:.1f})")
        rows.append({"ticket": ticket, "before": before, "after": after,
                      "before_empty_rate": before_empty_rate, "after_empty_rate": after_empty_rate})

    overall_before = sum(r["before_empty_rate"] for r in rows) / len(rows)
    overall_after = sum(r["after_empty_rate"] for r in rows) / len(rows)
    print(f"\n=== Overall (n={REPEATS} reps x {len(TICKETS)} tickets = {REPEATS * len(TICKETS)} runs each) ===")
    print(f"before: {overall_before:.0%} empty   after: {overall_after:.0%} empty")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week8_giveup_fix.json").write_text(
        json.dumps({"overall_before_empty_rate": overall_before, "overall_after_empty_rate": overall_after,
                     "repeats_per_ticket": REPEATS, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
