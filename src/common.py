"""Shared helpers: paths, time settings, loading the 8 tables, and the activity log."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os
DB = Path(os.environ.get("A2_DB", ROOT / "data" / "db"))
WORK = ROOT / "work"
WORK.mkdir(exist_ok=True)

TARGETS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]

# Test week 2026/11/22 00:00 - 11/29 00:00 Taiwan time (UTC+8)  ->  UTC
TEST_T = pd.Timestamp(os.environ.get("A2_TEST_T", "2026-11-21 16:00:00"))
WEEK = pd.Timedelta(days=7)

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
          "PRReviews", "PRReviewComments", "Pushes"]


def load_tables():
    return {t: pd.read_parquet(DB / f"{t}.parquet") for t in TABLES}


def data_end(tables):
    """Last observed row time in the data (= the newest information we have)."""
    return max(tables[t][c].max() for t, c in [("Pushes", "created_at"),
                                               ("IssueComments", "created_at"),
                                               ("PRReviews", "first_seen_at"),
                                               ("PRReviewComments", "created_at"),
                                               ("Issues", "first_seen_at"),
                                               ("PullRequests", "first_seen_at")])


def activity_log(tb):
    """One row per (user_id, activity, time, repo_id) for the 6 target activities.

    This is how labels are defined, and the same log feeds the temporal features.
    """
    pr, iss = tb["PullRequests"], tb["Issues"]
    parts = [
        pd.DataFrame({"user_id": pr.author_id, "act": "open_pr", "time": pr.created_at,
                      "seen": pr.first_seen_at, "repo_id": pr.repo_id}),
        pd.DataFrame({"user_id": iss.author_id[~iss.is_pull_request.fillna(False)],
                      "act": "open_issue",
                      "time": iss.created_at[~iss.is_pull_request.fillna(False)],
                      "seen": iss.first_seen_at[~iss.is_pull_request.fillna(False)],
                      "repo_id": iss.repo_id[~iss.is_pull_request.fillna(False)]}),
        pd.DataFrame({"user_id": tb["IssueComments"].author_id, "act": "issue_comment",
                      "time": tb["IssueComments"].created_at,
                      "repo_id": tb["IssueComments"].repo_id}),
        pd.DataFrame({"user_id": tb["PRReviews"].reviewer_id, "act": "pr_review",
                      "time": tb["PRReviews"].first_seen_at, "repo_id": tb["PRReviews"].repo_id}),
        pd.DataFrame({"user_id": tb["PRReviewComments"].author_id, "act": "pr_review_comment",
                      "time": tb["PRReviewComments"].created_at,
                      "repo_id": tb["PRReviewComments"].repo_id}),
        pd.DataFrame({"user_id": tb["Pushes"].user_id, "act": "push",
                      "time": tb["Pushes"].created_at, "repo_id": tb["Pushes"].repo_id}),
    ]
    log = pd.concat(parts, ignore_index=True).dropna(subset=["user_id", "time"])
    # row time: when the row becomes visible in the data (PR/issue rows can appear late)
    log["seen"] = log.seen.fillna(log.time)
    log["seen"] = log.seen.where(log.seen > log.time, log.time)
    log["user_id"] = log.user_id.astype("int64")
    log["act_id"] = log.act.map({a: i for i, a in enumerate(TARGETS)}).astype("int8")
    # only cohort users have complete activity
    cohort = tb["Users"].loc[tb["Users"].is_cohort.fillna(False), "user_id"]
    log = log[log.user_id.isin(cohort)]
    return log.sort_values("time", ignore_index=True)


def labels_for(log, users, t):
    """Binary (n_users, 6) matrix: did user do activity k in [t, t+7d)?"""
    w = log[(log.time >= t) & (log.time < t + WEEK)]
    y = np.zeros((len(users), 6), dtype=np.float32)
    idx = pd.Series(np.arange(len(users)), index=users)
    hit = w[w.user_id.isin(idx.index)]
    y[idx[hit.user_id].to_numpy(), hit.act_id.to_numpy()] = 1
    return y


def anchors(end_label_time, n_weeks, stride_days=7):
    """Prediction times aligned with the test week start (Sat 16:00 UTC), going back."""
    out, t = [], TEST_T
    while t + WEEK > end_label_time:
        t -= pd.Timedelta(days=stride_days)
    for _ in range(n_weeks * 7 // stride_days):
        out.append(t)
        t -= pd.Timedelta(days=stride_days)
    return sorted(out)


def f1_scores(y, p):
    """Per-activity F1, macro-F1, micro-F1 and Score (as in evaluate.py)."""
    tp = (y * p).sum(0); fp = ((1 - y) * p).sum(0); fn = (y * (1 - p)).sum(0)
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 1.0)
    micro = 2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
    macro = f1.mean()
    return {"per": dict(zip(TARGETS, f1.round(4))), "macro": macro, "micro": micro,
            "score": (macro + micro) / 2}
