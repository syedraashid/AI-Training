"""Week 10: race the triage squad (multi_agent_core.run_team -- a manager plus two specialists)
against the single hand-built agent (agent_core.run_agent) on the SAME tickets, a 6-ticket subset
of Week 8's trajectory-eval set (data/week8_trajectory_tickets.json), so quality numbers are
still directly comparable to prior weeks' ground truth, not a new test set invented to flatter
one side. The subset (data/week10_race_tickets.json: t01, t04, t06, t09, t10, t11) keeps one of
each ticket shape from the full 12 -- single-topic, compound two-topic, two escalation cases
needing both specialists, and the genuinely-out-of-scope refusal case -- rather than an arbitrary
truncation. It is smaller than the full 12 for a real, disclosed reason: Groq's per-minute token
limit (TPM) on this account is tight enough that the team arm alone (manager routing + up to two
specialist loops + manager synthesis, each a full hand-off that re-sends context) repeatedly hit
it on the full set; see data/results/week10_multi_agent_race.md for this stated plainly rather
than silently working around it by inflating the apparent dataset size.

Reports, for both arms, on identical scoring: quality (page coverage + escalation accuracy, the
same definition week7_race.py and week8_trajectory_eval.py used), speed (wall-clock elapsed_s),
tokens (prompt/completion split), and cost (rag_core.estimate_cost_usd, real Groq pricing).
Single pass per ticket, no repeated trials -- this is a race, not a statistical sweep.

Usage:
    python scripts/week10_race.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import agent_core
import multi_agent_core
import rag_core

load_dotenv()

ROOT = Path(__file__).parent.parent
TICKETS_PATH = ROOT / "data" / "week10_race_tickets.json"
RESULTS_DIR = ROOT / "data" / "results"

ESCALATION_PHRASES = ("escalate", "escalation")


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

        single = agent_core.run_agent(case["ticket"])
        single_score = score(single, case)
        print(f"  single: calls={single['num_llm_calls']} tokens={single['total_tokens']} "
              f"cost=${single['cost_usd']:.5f} time={single['elapsed_s']}s "
              f"coverage={single_score['page_coverage']:.0%} escalation_ok={single_score['escalation_correct']}")

        team = multi_agent_core.run_team(case["ticket"])
        team_score = score(team, case)
        print(f"  team:   routing={team['routing']} calls={team['num_llm_calls']} "
              f"tokens={team['total_tokens']} cost=${team['cost_usd']:.5f} time={team['elapsed_s']}s "
              f"coverage={team_score['page_coverage']:.0%} escalation_ok={team_score['escalation_correct']}")
        print()

        rows.append({
            "id": case["id"], "ticket": case["ticket"], "expected_pages": case["expected_pages"],
            "expect_escalation": case["expect_escalation"],
            "single": {**{k: single[k] for k in (
                "answer", "num_llm_calls", "total_tokens", "prompt_tokens", "completion_tokens",
                "cost_usd", "elapsed_s",
            )}, **single_score},
            "team": {**{k: team[k] for k in (
                "answer", "routing", "num_llm_calls", "total_tokens", "prompt_tokens",
                "completion_tokens", "cost_usd", "elapsed_s",
            )}, **team_score},
        })

    n = len(rows)

    def agg(arm: str, key: str) -> float:
        return sum(r[arm][key] for r in rows) / n

    summary = {}
    for arm in ("single", "team"):
        summary[arm] = {
            "avg_llm_calls": agg(arm, "num_llm_calls"),
            "avg_total_tokens": agg(arm, "total_tokens"),
            "avg_cost_usd": agg(arm, "cost_usd"),
            "total_cost_usd": sum(r[arm]["cost_usd"] for r in rows),
            "avg_elapsed_s": agg(arm, "elapsed_s"),
            "avg_page_coverage": agg(arm, "page_coverage"),
            "escalation_pass_rate": sum(r[arm]["escalation_correct"] for r in rows) / n,
        }

    print("=== Summary (n=%d tickets, single pass) ===" % n)
    for arm, stats in summary.items():
        print(f"{arm}: calls={stats['avg_llm_calls']:.1f} tokens={stats['avg_total_tokens']:.0f} "
              f"cost=${stats['avg_cost_usd']:.5f}/ticket (${stats['total_cost_usd']:.4f} total) "
              f"time={stats['avg_elapsed_s']:.2f}s coverage={stats['avg_page_coverage']:.1%} "
              f"escalation={stats['escalation_pass_rate']:.1%}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "week10_race.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
