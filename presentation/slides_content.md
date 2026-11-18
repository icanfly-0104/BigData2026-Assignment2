# Asg2 slide content (English, 10 pages) — paste into Assignment2_report_template

Every number below comes from this repo's actual run (`work/results.csv`, logs of
`prepare.py`, `train_lgbm.py`, `train_gnn.py`, `ensemble.py`). If you re-run with newer data,
update the numbers before recording.

Figures are in `figures/`.

---

## Slide 1 — Title

**Assignment 2: Predicting GitHub User Activity**
Relational Deep Learning (temporal heterogeneous GNN) + point-in-time LightGBM

- Name / Student ID: `<your name>` / 315551143
- GitHub (public): `https://github.com/<your-account>/<repo-name>`
- Test week: 2026/11/22 00:00 – 11/29 00:00 (Taiwan time) = 11/21 16:00 – 11/28 16:00 UTC

---

## Slide 2 — Task, data and the key difficulty

**Task.** For 4,465 test users, predict 6 binary activities (open_pr, open_issue, issue_comment,
pr_review, pr_review_comment, push) in the test week. Score = (macro-F1 + micro-F1) / 2.

**Data I built.**
- GH Archive hour files 2025-12-01 00:00 → 2026-11-18 09:00 UTC (8,458 hours), streamed and
  filtered to cohort actors, then the **official `preprocess.py`** → 8 tables.
- 2,393,662 cohort events; Pushes 1.68M, PullRequests 311k, Issues 178k, IssueComments 108k,
  PRReviews 93k, PRReviewComments 67k, Repositories 111k, Users 46k rows.
- Data ends 3 d 6 h before the test week → **gap = 3 d 6 h** (no data in this window).

**Key difficulty: the data distribution is not stationary** (figure `a_regime_shift.png`).
- push: ~80k events/week (May) → ~500/week (November).
- Other activities collapse in Jul–Aug and jump in early Sep.
- Active cohort users per week: ~2,900 (Sep) → ~950 (Nov).
→ This is a change in GH Archive coverage, not in users, so the model has to cope with regime shift.

Figure: `figures/a_regime_shift.png`

---

## Slide 3 — Q1 (a) Schema graph

Figure: `figures/q1_schema_graph.png`

- 8 tables = 8 nodes; 20 foreign keys = 20 edges (FK → referenced PK).
- PK in red, FK in blue, row-time column in grey for each table.
- Hubs: **Users** (8 incoming FKs: author ×4 / reviewer / pusher / assignee / merged_by) and
  **Repositories** (6 incoming `repo_id`).
- Self-reference: PRReviewComments.in_reply_to_id → PRReviewComments.
- Issues.pull_request_id links a PR's conversation to its PullRequests row.

---

## Slide 4 — Q1 (b) Relational entity graph (users 22996444 & 28983449, 2026-09-15 → 09-17 UTC)

Figure: `figures/q1_relational_entity_graph.png` (28 nodes, 53 edges)

- The raw GitHub ids are anonymised by `code_int("user", id)` → 601395036021679 (user_id 778)
  and 283459088190688 (user_id 1025).
- Rows in the window: 5 PRReviews, 5 PRReviewComments, 5 PullRequests, 3 Issues,
  2 IssueComments, 2 Pushes, 4 Repositories (incl. referenced PR / issue / repo rows).
- The two users are connected through **Repo 2219**: user 28983449 reviewed (PRReview 87184)
  and commented on (PRReviewComment 63406) **PR 287357, authored by user 22996444**.
- This is exactly the multi-hop signal a GNN can use (user → review → PR → author).

---

## Slide 5 — Q2 How I modelled relationships across tables

**1. Relational Deep Learning graph (main model).** For every prediction time, I turn the
database into a heterogeneous relational entity graph:
- 8 node types (one per table: user, repo, pr, issue, issue-comment, review, review-comment, push).
- 18 of the 20 FKs as edge types (all except `assignee_id` and `in_reply_to_id`), each with a
  reverse edge (e.g. push→user, review→pr, review-comment→review,
  issue→pr via `pull_request_id`, pr→user via `merged_by_id`).
- 2 layers of HeteroConv(GraphSAGE), hidden 128, residual + LayerNorm.
  After 2 hops a user embedding sees its events, those events' PRs/issues/repos, and the
  other users on them (e.g. reviews received on my PR).

**2. Cross-table join features (shared with the GNN user nodes).**
- Feedback received: reviews / review comments / issue comments that other users wrote on
  *my* PRs and issues (14 d) — joins review → PR → author.
- Open PRs I authored, PRs I merged (maintainer signal), assigned open issues.
- Repo neighbourhood: activity of all cohort users in the repos I touched (28 d), new PRs in
  those repos (review opportunities), number of co-workers, org / own-repo share.

**Evidence** (validation, test users):
- Same user features, add message passing over the FK graph: LightGBM 37.4 → RDL-GNN 39.2.
- GNN on the raw graph only (no engineered features): 32.6, which beats copy-last-week (29.8).
- Hand-crafted relational features on top of temporal ones in LightGBM: +0.9 with
  probability thresholds, −0.3 with the final rate rule → the useful relational signal comes
  mainly from the **GNN**.

---

## Slide 6 — Q3 How I modelled temporal information

1. **Point-in-time correctness.** Features at time t use only rows with row time
   (`first_seen_at` / `created_at`) < cutoff = t − gap; `merged_at`, `closed_at`, … only if < cutoff.
   Each training sample reproduces the real **3 d 6 h** blind gap before the test week.
2. **Multi-scale temporal features.** Per activity: counts in 1/3/7/14/28/56/112 d and all time,
   recency, exponential decay (half-life 7 d), fraction of active weeks (4/8/16 w), trend
   (7 d vs previous 21 d), active days, hour-of-day and weekend habits.
3. **Relative time encoding in the GNN.** Every event node gets
   log(1+Δt), sin/cos of Δt with periods 1, 3.5, 7, 14, 28 days, and hour/day-of-week sin/cos
   (Δt = cutoff − row time). The snapshot keeps the last 28 days.
4. **Regime-shift handling.**
   - Global context features (cohort activity per type, active users) and user activity
     *relative* to the global level (`rel_*`).
   - **Recency-weighted training**, weight = 0.5^(age in weeks / 3).
   - **Rate-based decision rule** (slide 9).

Evidence: recency half-life (LightGBM, per-activity F1-opt thresholds on validation):
none 35.0 · 12 w 35.6 · 6 w 35.7 · **3 w 36.2**.
Time encoding in the GNN: 38.9 → 39.2.

---

## Slide 7 — Q4 Framework and full pipeline

Figure: `figures/q4_framework.png`

1. **Data**: stream GH Archive (2025-12-01 → 2026-11-18 09:00 UTC); keep cohort actors only
   (identical to Step 2 of preprocess.py) → official `preprocess.py` → 8 Parquet tables.
2. **Samples**: 45 weekly prediction times aligned with the test week (Sat 16:00 UTC),
   2026-01-03 … 2026-11-07; cutoff = t − 3 d 6 h; label = activity in [t, t+7 d);
   9,000 cohort users × 45 weeks; 141 point-in-time features.
3. **Split by time**: train = 42 weeks whose labels end before the first validation cutoff;
   validation = weeks of 10/31 and 11/07 (evaluated on the 4,465 test users).
4. **Branch A — LightGBM**: one binary model per activity, recency-weighted, early stopping on
   validation, then refit on all weeks.
5. **Branch B — RDL-GNN**: 45 snapshot graphs; multi-task BCE (6 outputs), AdamW lr 5e-4 with
   cosine decay, 15 epochs, recency-weighted loss, 3 seeds averaged, refit on all weeks.
6. **Ensemble**: 0.7·LightGBM + 0.3·GNN (weight chosen on validation).
7. **Decision**: per activity, predict the top-q% users as 1 (q tuned for the Score)
   → `output_<ID>.csv`, checked with the official `evaluate.py`.

---

## Slide 8 — Results and ablation (validation, 4,465 test users)

Protocol: decision rule tuned on one validation week and scored on the other, averaged.

| Model | Score | macro-F1 | micro-F1 |
|---|---|---|---|
| Copy last week (baseline) | 29.8 | 30.6 | 29.0 |
| LightGBM, temporal features only | 37.7 | 36.7 | 38.8 |
| LightGBM, temporal + relational | 37.4 | 36.3 | 38.6 |
| RDL-GNN, raw graph only (+time enc.) | 32.6 | 32.1 | 33.1 |
| RDL-GNN + user features, no time enc. | 38.9 | 38.5 | 39.3 |
| RDL-GNN + user features + time enc. | 39.2 | 38.9 | 39.5 |
| **Ensemble (final)** | **40.3** | **40.0** | **40.7** |

Per-activity F1 of the final model: open_pr 42.6 · open_issue 43.4 · issue_comment 38.4 ·
pr_review 43.3 · pr_review_comment 35.6 · push 36.6.

Official `evaluate.py` on the week of 11/07 (rule tuned on 10/31): Score **39.80%**.

Figure: `figures/b_model_comparison.png`

---

## Slide 9 — Additional experiments and analysis

**(a) Decision rule under regime shift: probability threshold vs. top-q% rate**

| Model | prob. threshold | top-q% rate |
|---|---|---|
| LightGBM (temporal + relational) | 32.2 | 37.4 |
| RDL-GNN + features + time enc. | 38.5 | 39.2 |
| Ensemble | 35.6 | 40.3 |

The validation cutoffs still contain the push-heavy history from before 10/25, but the test
cutoff does not. The probabilities therefore drop at test time (final ensemble, test users:
mean push probability 0.29 / 0.25 in the two validation weeks vs 0.06 at test). With
probability thresholds tuned on validation, only 0.6% of test users would be predicted to
push (2.9% for open_pr), while 5.8–6.0% of them pushed in each validation week. Ranking is stable across regimes, so we tune the predicted
positive rate instead. Final rates: open_pr 6.6%, open_issue 4.0%, issue_comment 3.8%,
pr_review 6.6%, pr_review_comment 1.6%, push 8.4%.

**(b) Recency weighting**: half-life 3 weeks is best (slide 6).

**(c) Feature importance** (`figures/c_feature_importance.png`): recency-decay features dominate
(decay_pr_review, recency_any, decay_pr_review_comment, decay_open_issue), followed by
regime-relative `rel_push_28d` and the global `global_active_users_7d`.

---

## Slide 10 — Conclusion and limitations

- RDL-GNN with point-in-time user features is the best single model (39.2);
  the ensemble with LightGBM reaches 40.3 vs 29.8 for copy-last-week.
- Temporal modelling matters most: point-in-time features, a realistic gap, recency
  weighting, and a regime-robust decision rule.
- Relational modelling helps mainly through GNN message passing over the FK graph (+1.8 over
  LightGBM with the same features).
- Limitations: only 2 validation weeks in the new regime, so weight / rate tuning is noisy.
  The ensemble weight was chosen on the same 2 weeks, so 40.3 is slightly optimistic.
  GH Archive coverage in the test week is unknown.
- Reproduce: `README.md` → `download_filter.py` → `preprocess.py` → `bash run_all.sh <ID>`.
