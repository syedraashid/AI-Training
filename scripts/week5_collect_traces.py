"""Week 5: ask ~20 realistic questions through the real pipeline and let answer_question's
own trace logging record them to data/traces.jsonl, exactly as the app would.

Usage:
    python scripts/week5_collect_traces.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

import rag_core

load_dotenv()

ROOT = Path(__file__).parent.parent
QUESTIONS_PATH = ROOT / "data" / "week5_questions.json"


def main() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    for i, question in enumerate(questions, start=1):
        answer, results = rag_core.answer_question(question, top_k=3, generation_model="openai/gpt-oss-20b", mode="hybrid")
        sources = [r["source"] for r in results]
        print(f"[{i}/{len(questions)}] Q: {question}")
        print(f"    sources: {sources}")
        print(f"    A: {answer[:200].encode('ascii', 'replace').decode()}")
        print()
    print(f"Done. Traces appended to {rag_core.TRACES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
