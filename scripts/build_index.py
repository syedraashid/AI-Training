"""Build the production index from the real assigned knowledge base
(data/docs/customer-support-ticket-knowledge-base.pdf), so the app doesn't depend on
someone re-uploading it through the Streamlit UI. Rebuilds data/index.json at 400/80,
the config settled on in Week 3 (see data/results/week3_notes.md).

Usage:
    python scripts/build_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import rag_core

ROOT = Path(__file__).parent.parent
PDF_PATH = ROOT / "data" / "docs" / "customer-support-ticket-knowledge-base.pdf"


def main() -> int:
    count = rag_core.build_index(
        [(PDF_PATH.name, PDF_PATH.read_bytes())],
        chunk_words=400,
        overlap_words=80,
        embedding_model="all-MiniLM-L6-v2",
    )
    print(f"Indexed {count} chunks from {PDF_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
