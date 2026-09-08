"""Week 7: race the hand-built agent (agent_core.run_agent) against the plain fixed
workflow already shipped in Weeks 3-6 (rag_core.answer_question, hybrid retrieval,
top_k=3) on the same 10 tickets. Measures speed, cost (tokens), and reliability
(did every expected article page get used, and was escalation flagged correctly).

Usage:
    python scripts/week7_race.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import agent_core
import rag_core

load_dotenv()

ROOT = Path(__file__).parent.parent
TICKETS_PATH = ROOT / "data" / "week7_tickets.json"
RESULTS_DIR = ROOT / "data" / "results"

ESCALATION_PHRASES = ("escalate", "escalation")


def run_fixed(ticket: str) -> dict:
    started = time.perf_counter()
    answer, results, usage = rag_core.answer_question(ticket, top_k=3, mode="hybrid")
    elapsed = time.perf_counter() - started
    return {
        "answer": answer,
        "sources": [{"source": r["source"], "location": r["location"]} for r in results],
        "num_llm_calls": 1,
        "total_tokens": usage["total_tokens"],
        "elapsed_s": round(elapsed, 3),
    }


def score(result: dict, case: dict) -> dict:
    pages_used = {s["location"] for s in result["sources"]}
    expected = set(case["expected_pages"])
    if expected:
        page_coverage = len(expected & pages_used) / len(expected)
    else:
        page_coverage = 1.0 if rag_core.is_refusal(result["answer"]) else 0.0
    mentions_escalation = any(p in result["answer"].lower() for p in ESCALATION_PHRASES)
    escalation_correct = mentions_escalation == case["expect_escalation"]
    return {"page_coverage": page_coverage, "escalation_correct": escalation_correct}


def main() -> int:
    tickets = json.loads(TICKETS_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in tickets:
        print(f"=== {case['id']} ===")
        print(case["ticket"])

        fixed = run_fixed(case["ticket"])
        fixed_score = score(fixed, case)
        print(f"  fixed: calls={fixed['num_llm_calls']} tokens={fixed['total_tokens']} "
              f"time={fixed['elapsed_s']}s coverage={fixed_score['page_coverage']:.0%} "
              f"escalation_ok={fixed_score['escalation_correct']}")

        agent = agent_core.run_agent(case["ticket"])
        agent_score = score(agent, case)
        print(f"  agent: calls={agent['num_llm_calls']} tokens={agent['total_tokens']} "
              f"time={agent['elapsed_s']}s coverage={agent_score['page_coverage']:.0%} "
              f"escalation_ok={agent_score['escalation_correct']} stopped_safely={agent['stopped_safely']}")
        print()

        rows.append({
            "id": case["id"], "ticket": case["ticket"], "expected_pages": case["expected_pages"],
            "expect_escalation": case["expect_escalation"],
            "fixed": {**fixed, **fixed_score},
            "agent": {**agent, **agent_score},
        })

    def agg(key: str, sub: str) -> float:
        return sum(r[key][sub] for r in rows) / len(rows)

    summary = {
        "fixed": {
            "avg_llm_calls": agg("fixed", "num_llm_calls"),
            "avg_tokens": agg("fixed", "total_tokens"),
            "avg_time_s": agg("fixed", "elapsed_s"),
            "avg_page_coverage": agg("fixed", "page_coverage"),
            "escalation_pass_rate": sum(r["fixed"]["escalation_correct"] for r in rows) / len(rows),
        },
        "agent": {
            "avg_llm_calls": agg("agent", "num_llm_calls"),
            "avg_tokens": agg("agent", "total_tokens"),
            "avg_time_s": agg("agent", "elapsed_s"),
            "avg_page_coverage": agg("agent", "page_coverage"),
            "escalation_pass_rate": sum(r["agent"]["escalation_correct"] for r in rows) / len(rows),
        },
    }
    print("=== Summary (n=%d tickets) ===" % len(rows))
    for label, stats in summary.items():
        print(f"{label}: calls={stats['avg_llm_calls']:.1f} tokens={stats['avg_tokens']:.0f} "
              f"time={stats['avg_time_s']:.2f}s coverage={stats['avg_page_coverage']:.1%} "
              f"escalation={stats['escalation_pass_rate']:.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week7_race.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
