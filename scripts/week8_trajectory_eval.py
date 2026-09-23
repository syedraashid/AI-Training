"""Week 8: trajectory evaluation for the ticket agent (agent_core.run_agent).

Outcome scoring (page coverage, escalation mention) only looks at the final answer -- it
can't tell a right answer reached by a sound path from one reached by a lucky wrong path
that will break under a slightly different ticket. This script scores both, independently,
on the same runs, and reports every case where they disagree (the outcome-vs-trajectory gap).

Trajectory correctness (process, not content) requires ALL of:
  1. stopped_safely       -- the loop ended via final_answer, not the step budget.
  2. no_looping           -- no two search_kb queries are near-duplicates of each other
                              (Jaccard token overlap > 0.6): the "going in circles" failure mode.
  3. search_count_sane    -- number of distinct search_kb calls is in a reasonable range for
                              the ticket's known topic count (data/week8_trajectory_tickets.json).
  4. escalation_tool_used -- check_escalation was called iff the ticket actually needed it
                              (matches ground truth, not just whether the final text mentions it).

Usage:
    python scripts/week8_trajectory_eval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import agent_core
import rag_core

load_dotenv()

ROOT = Path(__file__).parent.parent
TICKETS_PATH = ROOT / "data" / "week8_trajectory_tickets.json"
RESULTS_DIR = ROOT / "data" / "results"

ESCALATION_PHRASES = ("escalate", "escalation")
STOPWORDS = {"a", "an", "the", "is", "are", "for", "to", "of", "on", "in", "and", "or", "not"}


def _tokens(query: str) -> set[str]:
    return {w for w in query.lower().split() if w not in STOPWORDS}


def outcome_score(result: dict, case: dict) -> dict:
    pages_used = {s["location"] for s in result["sources"]}
    expected = set(case["expected_pages"])
    if expected:
        page_coverage = len(expected & pages_used) / len(expected)
    else:
        page_coverage = 1.0 if rag_core.is_refusal(result["answer"]) else 0.0
    mentions_escalation = any(p in result["answer"].lower() for p in ESCALATION_PHRASES)
    escalation_correct = mentions_escalation == case["expect_escalation"]
    outcome_correct = page_coverage == 1.0 and escalation_correct
    return {"page_coverage": page_coverage, "escalation_mentioned": mentions_escalation,
            "escalation_correct": escalation_correct, "outcome_correct": outcome_correct}


def trajectory_score(result: dict, case: dict) -> dict:
    steps = result["steps"]
    search_queries = [s["action_input"].get("query", "") for s in steps if s["action"] == "search_kb"]
    escalation_calls = [s for s in steps if s["action"] == "check_escalation"]

    no_looping = True
    dupes = []
    for i in range(len(search_queries)):
        for j in range(i + 1, len(search_queries)):
            a, b = _tokens(search_queries[i]), _tokens(search_queries[j])
            if not a or not b:
                continue
            jaccard = len(a & b) / len(a | b)
            if jaccard > 0.6:
                no_looping = False
                dupes.append((search_queries[i], search_queries[j], round(jaccard, 2)))

    n_topics = len(case["topics"])
    distinct_searches = len(set(search_queries))
    if n_topics == 0:
        search_count_sane = 1 <= distinct_searches <= 3
    else:
        search_count_sane = n_topics <= distinct_searches <= n_topics + 1

    escalation_tool_used = bool(escalation_calls) == case["expect_escalation"]

    trajectory_correct = (
        result["stopped_safely"] and no_looping and search_count_sane and escalation_tool_used
    )
    return {
        "stopped_safely": result["stopped_safely"],
        "no_looping": no_looping,
        "duplicate_queries": dupes,
        "search_count_sane": search_count_sane,
        "distinct_searches": distinct_searches,
        "expected_topics": n_topics,
        "escalation_tool_used": escalation_tool_used,
        "trajectory_correct": trajectory_correct,
    }


def main() -> int:
    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in tickets:
        print(f"=== {case['id']} ===")
        print(case["ticket"])
        result = agent_core.run_agent(case["ticket"])
        outcome = outcome_score(result, case)
        trajectory = trajectory_score(result, case)
        gap = outcome["outcome_correct"] != trajectory["trajectory_correct"]
        print(f"  outcome_correct={outcome['outcome_correct']} trajectory_correct={trajectory['trajectory_correct']}"
              f"{'  <-- GAP' if gap else ''}")
        if trajectory["duplicate_queries"]:
            print(f"  duplicate/near-duplicate queries: {trajectory['duplicate_queries']}")
        print()
        rows.append({
            "id": case["id"], "ticket": case["ticket"], "topics": case["topics"],
            "expect_escalation": case["expect_escalation"],
            "answer": result["answer"], "steps": result["steps"],
            "outcome": outcome, "trajectory": trajectory, "gap": gap,
        })

    n = len(rows)
    outcome_rate = sum(r["outcome"]["outcome_correct"] for r in rows) / n
    trajectory_rate = sum(r["trajectory"]["trajectory_correct"] for r in rows) / n
    gaps = [r for r in rows if r["gap"]]

    print("=== Summary (n=%d tickets) ===" % n)
    print(f"outcome-correct rate:    {outcome_rate:.1%}")
    print(f"trajectory-correct rate: {trajectory_rate:.1%}")
    print(f"gap cases (outcome and trajectory disagree): {[r['id'] for r in gaps]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week8_trajectory_eval.json").write_text(
        json.dumps({
            "summary": {"outcome_correct_rate": outcome_rate, "trajectory_correct_rate": trajectory_rate,
                        "gap_ids": [r["id"] for r in gaps]},
            "rows": rows,
        }, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
