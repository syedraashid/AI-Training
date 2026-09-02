# Customer Support RAG Lab

A deliberately small RAG application for completing Weeks 3–6 of the course with customer-support tickets. It is designed to make the full pipeline visible: documents become chunks, chunks become embeddings, retrieval is inspectable, every answer is logged, and regression tests can run from one command.

## Setup

1. Install Python 3.11 or newer.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install packages:

   ```powershell
   pip install -r requirements.txt
   ```

5. Create a file named `.env` in this folder containing your Groq API key:

   ```text
   GROQ_API_KEY=your_key_here
   ```

   Do not upload or commit `.env`.

   The app creates embeddings locally with `all-MiniLM-L6-v2`; the first index build downloads that model. Groq is used only to generate answers.

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

## Status: Weeks 3–6 complete (customer support tickets track)

Everything below is done and reproducible from a fresh checkout — the documents, questions, scripts,
and result files are all committed. Full write-ups are in `data/results/`:

- `data/results/week3_notes.md` — chunk-size comparison and the Week 3 question table
- `data/results/week5_error_analysis.md` — honest per-failure notes, named/ranked error types
- `data/results/week6_before_after.md` — the one Week 6 prompt change, measured, including what it
  did **not** fix
- `data/results/*.json` — raw per-question results behind every number in the write-ups above

**Known limitation:** the LLM-judge validation (last Week 6 step) is implemented
(`scripts/week6_judge_validation.py`, `rag_core.judge_answer()`) but has not been run — it needs Groq
API calls and this project's free-tier daily token quota (200,000 tokens/day for
`openai/gpt-oss-20b`) was exhausted while measuring the Week 6 before/after. Run it once the quota
resets (see `data/results/week6_before_after.md` for details):
```powershell
python scripts/week6_judge_validation.py
```

## Week 3: Baseline RAG

1. `data/docs/` has 6 help-centre documents (returns, shipping, error codes, billing, warranty, promo).
2. Production index uses 400-word chunks / 80-word overlap (`python scripts/week3_baseline.py` rebuilds it).
3. `data/week3_questions.json` has 18 questions: 15 answerable + 3 the documents cannot answer.
4. Every supported answer cited its source; every unsupported question got the exact refusal — verified
   programmatically, see `data/results/week3_notes.md`.
5. Chunk size was compared across 4 configs (60/0, 150/20, 400/80, 700/120,
   `python scripts/week3_chunk_sweep.py`) using hit-rate@1, since hit-rate@3 saturates at 100% with
   only 6 source documents. 150w and 700w each dropped a *different* question — see the write-up for
   why. 400/80 was kept as the config that avoided both failure modes.

Full table (question, expected source, retrieved source, answer correct?, citation correct?, notes)
is in `data/results/week3_notes.md`.

## Week 4: Debug and improve retrieval

1. `rag_core.retrieve()` now supports `mode="dense"` (Week 3 baseline), `"bm25"`, and `"hybrid"`
   (reciprocal-rank fusion of both — the Week 4 improvement). Toggle it in the app sidebar, or pass it
   directly to `retrieve()` / `answer_question()`.
2. `data/week4_questions.json` + `python scripts/week4_retrieval_debug.py` targets the retrieval-vs-
   generation split explicitly, using exact error-code lookups (the brief's own ERR-4032 example) as
   the adversarial case for keyword vs. semantic search.
3. Real, reproducible before/after: hit-rate@1 went from **85.7% → 92.9%**. The clearest single case:
   the bare code `"3305"` fails under dense-only AND bm25-only (it matches two different documents for
   two different reasons) but is correctly resolved by hybrid fusion. One case (`w4-10`, a
   promo-vs-shipping vocabulary collision on "order value") was **not** fixed by hybrid and is
   documented as such — not every retrieval failure is a hybrid-search problem.
4. Along the way, found and fixed a real bug: `is_refusal()` used an exact ASCII substring match, so a
   model output using a Unicode non-breaking hyphen silently passed as "not a refusal," corrupting
   every downstream refusal metric. Fixed by normalizing dash/quote variants before matching.

See `data/results/before.txt` / `after.txt` and `data/results/week4_*.json` for full output.

## Week 5: Error analysis

1. 20 realistic questions (typos, compound questions, roleplay/override attempts, out-of-scope,
   vocabulary mismatches — not a cherry-picked easy set) were run through the real pipeline via
   `python scripts/week5_collect_traces.py`; traces are in `data/traces.jsonl`.
2. Every trace was read and, for each failure, a plain-language note was written before naming any
   category (`data/results/week5_error_analysis.md`).
3. Grouped into 4 named, ranked types: **unwarranted refusal** (highest priority — right document
   retrieved, model still refused), **refusal-detection blind spot** (the Unicode-hyphen bug above),
   **retrieval miss on vocabulary mismatch** ("Bitcoin" vs. the doc's "cryptocurrency"), and
   **citation formatting drift** (cosmetic).
4. Target for next fix and a written prediction are both in the write-up, ahead of the Week 6 change.

## Week 6: Automated evals

1. `data/eval_cases.json` — 26 permanent regression cases, each tagged with a `problem_type`
   (`retrieval`, `unsupported-question`, `exact-code-retrieval`, `unwarranted-refusal`,
   `vocabulary-mismatch`), including the real failures found in Week 5.
2. One command runs the whole suite: `python run_evals.py hybrid` (pass `dense` to compare against the
   Week 3 baseline retrieval mode, or add `--judge` to also score with the LLM judge).
3. The Week 5 prediction (permit one direct inference from a stated rule) was implemented as the one
   Week 6 prompt change and measured before/after per problem type — **honestly, it did not move the
   number** on the two specific failures it targeted, which is reported and root-caused rather than
   hidden. See `data/results/week6_before_after.md` for the full story, including why an untested
   follow-up prompt wasn't shipped.
4. LLM-as-judge (`rag_core.judge_answer`, rubric-based, 1–5) is implemented with a human-vs-judge
   validation harness (`scripts/week6_judge_validation.py`) — not yet run, see the quota note above.

## Files you will edit

- `app.py`: Streamlit screen and controls.
- `rag_core.py`: extraction, chunking, embedding, retrieval, answer generation, and trace logging.
- `data/eval_cases.json`: your test cases. This is your main Week 6 artifact.
- `run_evals.py`: the one-command automated retrieval/refusal checks.

## Current limits and good next improvements

This baseline uses dense semantic retrieval only. After recording a baseline, add one feature at a time: BM25, hybrid search, reranking, metadata filters, or a better chunking strategy. Keep the evidence and test results for each change.
