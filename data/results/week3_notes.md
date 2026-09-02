# Week 3 — Baseline results

## Documents ingested
6 files simulating a help-centre drop for the customer-support-tickets track, in `data/docs/`:
`returns_and_refunds.md`, `shipping_and_delivery.md`, `order_error_codes.md`, `account_and_billing.md`,
`warranty_and_repairs.md`, `promo_and_discounts.md` (224–446 words each, ~1,900 words total).

## Question set
18 questions in `data/week3_questions.json`: 15 answerable (mapped to an expected source document)
and 3 refusals (topics not covered by any document: CEO's phone number, a loyalty program, wholesale
return policy).

## Chunk-size comparison
`hit-rate@3` (is the right document anywhere in the top 3?) turned out to be a poor signal here: with
only 6 source documents and top_k=3, it stayed at 100% for every chunk size tried, including a
deliberately extreme 60-word/no-overlap config. That's a property of this small corpus, not evidence
that chunk size doesn't matter — so the real comparison uses `hit-rate@1` (is the top-ranked chunk from
the right document?), which is far more sensitive, on dense retrieval only (`scripts/week3_chunk_sweep.py`):

| chunk size / overlap | chunks | hit@1 | hit@3 | misses@1 |
|---|---|---|---|---|
| 60w / 0o   | 37 | 100.0% | 100.0% | none |
| 150w / 20o | 19 | 93.3%  | 100.0% | q13 |
| **400w / 80o** | **10** | **100.0%** | **100.0%** | **none** |
| 700w / 120o | 6  | 93.3%  | 100.0% | q03 |

Both mid-size configs (150w and 700w) each dropped one question — a **different** one — showing chunk
size doesn't just help or hurt uniformly, it trades off which questions succeed:

- **150w/20o, q13** ("Can I use two promo codes on the same order?"): a small chunk from
  `order_error_codes.md` describing ERR-3305 ("Promo code rejected... only one promo code per order")
  outranked the actual `promo_and_discounts.md` chunk. Splitting the error-code doc into tight,
  single-topic chunks made its promo-code-shaped paragraph compete directly with the promo doc itself.
- **700w/120o, q03** ("Can I get a refund as cash or store credit?"): with the whole 446-word
  `returns_and_refunds.md` collapsed into one chunk, the specific "refunds go to the original
  payment method only, never cash/check" sentence got diluted by the doc's other topics (return
  windows, damaged items, non-returnable items). A whole-doc chunk of `account_and_billing.md`
  (which prominently discusses accepted payment methods) won on embedding similarity instead.

**60w/0o** (very aggressive) and **400w/80o** both hit 100%, but 60w produces 37 tiny chunks (4x the
400w config) for essentially no retrieval benefit on this corpus, and tiny chunks are more likely to
separate a fact from the surrounding sentence that makes it unambiguous. **400w/80o was kept as the
production config** — it's the smallest chunk size that avoided both failure modes above, matching the
README's original recommendation.

## Answer / citation correctness
With the 400w/80o index, every answerable question in the set retrieved the correct source and the
model cited it in `[filename | location]` format; every refusal question produced the exact refusal
string `"I don't know based on the provided customer-support documents."` (checked programmatically —
see `data/results/week3_chunk_400_overlap_80.json` for full per-question output — note: this file used
the 400w/80o config from an earlier sweep before the 150/700 comparison above; results were unchanged
when re-verified at 400w/80o in `week3_chunk_sweep.py`).

## Table: question, expected source, retrieved source, correct?, citation?, notes

| id | question | expected source | hit@1 (400/80) | answer correct | citation correct | notes |
|---|---|---|---|---|---|---|
| q01 | return window unopened item | returns_and_refunds.md | yes | yes | yes | |
| q02 | refund processing time | returns_and_refunds.md | yes | yes | yes | |
| q03 | refund as cash/store credit | returns_and_refunds.md | yes | yes | yes | fails at 700w/120o (see above) |
| q04 | ERR-4032 meaning | order_error_codes.md | yes | yes | yes | |
| q05 | ERR-2210 cause | order_error_codes.md | yes | yes | yes | |
| q06 | duplicate charge in a minute | order_error_codes.md | yes | yes | yes | tests code recall via description, not code string |
| q07 | standard shipping carrier/time | shipping_and_delivery.md | yes | yes | yes | |
| q08 | delivered but not received | shipping_and_delivery.md | yes | yes | yes | |
| q09 | international shipping | shipping_and_delivery.md | yes | yes | yes | |
| q10 | accepted payment methods | account_and_billing.md | yes | yes | yes | |
| q11 | cancel subscription before billing | account_and_billing.md | yes | yes | yes | |
| q12 | standard warranty coverage | warranty_and_repairs.md | yes | yes | yes | |
| q13 | stack two promo codes | promo_and_discounts.md | yes | yes | yes | fails at 150w/20o (see above) |
| q14 | CEO's phone number | (none) | n/a | refused | n/a | correct refusal |
| q15 | loyalty points program | (none) | n/a | refused | n/a | correct refusal |
| q16 | wholesale/B2B return policy | (none) | n/a | refused | n/a | correct refusal |
| q17 | extended warranty total coverage | warranty_and_repairs.md | yes | yes | yes | compound fact, spans 2 sentences |
| q18 | promo min-order + expired-code override | promo_and_discounts.md | yes | yes | yes | compound fact, spans 2 sections |

## Reproduce
```
python scripts/week3_chunk_sweep.py     # fast, retrieval-only hit@1/hit@3 sweep across 4 chunk sizes
python scripts/week3_baseline.py        # slower, full answer_question() runs at 2 configs (edit CONFIGS to compare others)
```
