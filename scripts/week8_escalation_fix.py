"""Week 8: before/after measurement for the escalation-precheck fix in agent_core.run_agent.

Week 7's agent only called check_escalation when the LLM itself judged it was needed (see
SYSTEM_PROMPT's old wording: "Only call check_escalation if..."). week8_trajectory_eval.py
found a ticket (t09) where that judgment failed outright -- the agent looped on search_kb
and never called check_escalation at all for a ticket literally designed to need it.

The fix (agent_core.run_agent, force_escalation_precheck=True by default): run the
deterministic check_escalation rule on the raw ticket text BEFORE the LLM loop starts,
unconditionally, and hand the LLM the result. This can't be skipped by a bad tool-choice
decision. It does NOT fix keyword coverage -- check_escalation is still a literal substring
match (data/agent_core.py ESCALATION_RULES), so a true escalation case phrased without any
trigger word still won't fire. data/week8_escalation_tickets.json deliberately includes a
couple of naturally-phrased cases (e03, e04) that describe a data-deletion / fraud situation
without using the literal trigger words, specifically to measure that residual gap honestly
rather than only testing cases engineered to pass.

before = force_escalation_precheck=False (Week 7 behavior: LLM decides whether to call it)
after  = force_escalation_precheck=True  (Week 8 fix: forced, deterministic, pre-LLM)

Usage:
    python scripts/week8_escalation_fix.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import agent_core

load_dotenv()

ROOT = Path(__file__).parent.parent
TICKETS_PATH = ROOT / "data" / "week8_escalation_tickets.json"
RESULTS_DIR = ROOT / "data" / "results"

ESCALATION_PHRASES = ("escalate", "escalation")


def escalation_correct(result: dict, expect_escalation: bool) -> bool:
    mentions = any(p in result["answer"].lower() for p in ESCALATION_PHRASES)
    return mentions == expect_escalation


def main() -> int:
    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in tickets:
        print(f"=== {case['id']} (expect_escalation={case['expect_escalation']}) ===")
        before = agent_core.run_agent(case["ticket"], force_escalation_precheck=False)
        after = agent_core.run_agent(case["ticket"], force_escalation_precheck=True)
        before_ok = escalation_correct(before, case["expect_escalation"])
        after_ok = escalation_correct(after, case["expect_escalation"])
        print(f"  before: correct={before_ok}  after: correct={after_ok}"
              f"{'  <-- fix changed outcome' if before_ok != after_ok else ''}")
        rows.append({
            "id": case["id"], "ticket": case["ticket"], "expect_escalation": case["expect_escalation"],
            "before": {"answer": before["answer"], "correct": before_ok, "steps": before["steps"]},
            "after": {"answer": after["answer"], "correct": after_ok, "steps": after["steps"]},
        })

    n = len(rows)
    before_rate = sum(r["before"]["correct"] for r in rows) / n
    after_rate = sum(r["after"]["correct"] for r in rows) / n
    print(f"\n=== Summary (n={n}) ===")
    print(f"before: {before_rate:.0%} correct   after: {after_rate:.0%} correct")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week8_escalation_fix.json").write_text(
        json.dumps({"before_rate": before_rate, "after_rate": after_rate, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
