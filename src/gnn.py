"""Relational Deep Learning model: a heterogeneous temporal GNN over the 8 tables.

For one prediction time we build a *snapshot* of the relational entity graph:
  * every table row becomes a node (users, repositories, PRs, issues, issue comments,
    reviews, review comments, pushes) whose row time is < cutoff and >= cutoff - LOOKBACK;
  * every foreign key becomes an edge (and a reverse edge);
  * every event node gets a relative time encoding of (cutoff - row time).
A HeteroConv(SAGE) network passes messages along the FK edges, and a multi-task head
predicts the 6 activities for each cohort user node.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
import torch_geometric.transforms as T

DAY = pd.Timedelta(days=1)
LOOKBACK = 28  # days of event rows kept in a snapshot


def time_enc(age_days, abs_time, use_time=True):
    """Relative-time encoding: log age + sinusoids at several periods + time of day/week."""
    n = len(age_days)
    if not use_time:
        return np.zeros((n, 1), np.float32)
    a = np.asarray(age_days, dtype=np.float64)
    feats = [np.log1p(a)[:, None] / 4]
    for period in (1, 3.5, 7, 14, 28):
        feats += [np.sin(2 * np.pi * a / period)[:, None], np.cos(2 * np.pi * a / period)[:, None]]
    ts = pd.DatetimeIndex(abs_time)
    h = ts.hour.to_numpy() / 24.0; d = ts.dayofweek.to_numpy() / 7.0
    feats += [np.sin(2 * np.pi * h)[:, None], np.cos(2 * np.pi * h)[:, None],
              np.sin(2 * np.pi * d)[:, None], np.cos(2 * np.pi * d)[:, None]]
    return np.hstack(feats).astype(np.float32)


def _b(s):
    return s.astype("float").fillna(0).to_numpy(np.float32)[:, None]


def build_snapshot(tb, user_feat, cutoff, use_time=True, lookback=LOOKBACK):
    """HeteroData for one cutoff. user_feat: DataFrame (index = cohort user_id), scaled."""
    lo = cutoff - pd.Timedelta(days=lookback)
    win = lambda df, c: df[(df[c] < cutoff) & (df[c] >= lo)]  # noqa: E731
    pushes = win(tb["Pushes"], "created_at")
    icom = win(tb["IssueComments"], "created_at")
    revs = win(tb["PRReviews"], "first_seen_at")
    rcom = win(tb["PRReviewComments"], "created_at")
    pr_all, iss_all = tb["PullRequests"], tb["Issues"]
    pr_ids = set(win(pr_all, "first_seen_at").pr_id) | set(revs.pr_id) | set(rcom.pr_id.dropna())
    iss_ids = set(win(iss_all, "first_seen_at").issue_id) | set(icom.issue_id.dropna())
    prs = pr_all[pr_all.pr_id.isin(pr_ids) & (pr_all.first_seen_at < cutoff)]
    iss = iss_all[iss_all.issue_id.isin(iss_ids) & (iss_all.first_seen_at < cutoff)]

    # ----- users: all cohort users first (fixed order) + other users referenced -----
    cohort = user_feat.index.to_numpy()
    others = pd.concat([prs.author_id, prs.merged_by_id.where(prs.merged_at < cutoff),
                        iss.author_id, icom.author_id, revs.reviewer_id, rcom.author_id,
                        pushes.user_id]).dropna().astype("int64").unique()
    others = np.setdiff1d(others, cohort)
    u_ids = np.concatenate([cohort, others])
    u_map = pd.Series(np.arange(len(u_ids)), index=u_ids)
    xu = np.zeros((len(u_ids), user_feat.shape[1] + 1), np.float32)
    xu[:len(cohort), :-1] = user_feat.to_numpy(np.float32)
    xu[:len(cohort), -1] = 1.0  # is_cohort flag

    # ----- repositories referenced in the snapshot -----
    r_ids = pd.concat([prs.repo_id, iss.repo_id, icom.repo_id, revs.repo_id, rcom.repo_id,
                       pushes.repo_id]).dropna().astype("int64").unique()
    r_map = pd.Series(np.arange(len(r_ids)), index=r_ids)
    repo_tab = tb["Repositories"].set_index("repo_id").loc[r_ids]
    cnt = lambda df: np.log1p(df.repo_id.value_counts().reindex(r_ids).fillna(0).to_numpy())  # noqa: E731
    xr = np.stack([repo_tab.org_id.notna().to_numpy(float), cnt(pushes), cnt(icom), cnt(revs),
                   cnt(rcom), cnt(prs), cnt(iss)], 1).astype(np.float32)

    data = HeteroData()
    data["user"].x = torch.from_numpy(xu)
    data["user"].n_cohort = len(cohort)
    data["repo"].x = torch.from_numpy(xr)

    def te(df, col):
        return time_enc(((cutoff - df[col]) / DAY).to_numpy(), df[col].to_numpy(), use_time)

    def node(name, df, tcol, extra):
        data[name].x = torch.from_numpy(np.hstack([te(df, tcol)] + extra).astype(np.float32))
        return pd.Series(np.arange(len(df)), index=df.iloc[:, 0].to_numpy())

    blen = lambda s: np.log1p(s.fillna("").str.len().to_numpy(np.float32))[:, None]  # noqa: E731
    p_map = node("pr", prs, "first_seen_at", [
        _b(prs.is_cross_repo), _b(prs.draft),
        (prs.merged_at < cutoff).to_numpy(np.float32)[:, None],
        (prs.closed_at < cutoff).to_numpy(np.float32)[:, None]])
    i_map = node("issue", iss, "first_seen_at", [
        _b(iss.is_pull_request), np.log1p(iss.num_labels.fillna(0).to_numpy(np.float32))[:, None],
        (iss.closed_at < cutoff).to_numpy(np.float32)[:, None], blen(iss.body)])
    node("icom", icom, "created_at", [_b(icom.is_pull_request), blen(icom.body)])
    st = revs.state.fillna("")
    node("review", revs, "first_seen_at", [
        np.stack([(st == s).to_numpy(np.float32) for s in ("approved", "commented",
                                                          "changes_requested")], 1), blen(revs.body)])
    node("rcom", rcom, "created_at", [(rcom.subject_type == "line").to_numpy(np.float32)[:, None],
                                      blen(rcom.body)])
    node("push", pushes, "created_at", [(pushes.ref_type == "tag").to_numpy(np.float32)[:, None],
                                        _b(pushes.is_new_ref)])
    rv_map = pd.Series(np.arange(len(revs)), index=revs.review_id.to_numpy())

    def edge(src_name, src_df, fk, dst_name, dst_map, rel, mask=None):
        s = pd.Series(np.arange(len(src_df)))
        tgt = src_df[fk].reset_index(drop=True)
        ok = tgt.notna() & tgt.isin(dst_map.index)
        if mask is not None:
            ok &= pd.Series(np.asarray(mask)).reset_index(drop=True)
        ei = np.stack([s[ok].to_numpy(), dst_map[tgt[ok].astype("int64")].to_numpy()])
        data[src_name, rel, dst_name].edge_index = torch.from_numpy(ei).long()

    edge("push", pushes, "user_id", "user", u_map, "pushed_by")
    edge("push", pushes, "repo_id", "repo", r_map, "in_repo")
    edge("icom", icom, "author_id", "user", u_map, "written_by")
    edge("icom", icom, "issue_id", "issue", i_map, "on_issue")
    edge("icom", icom, "repo_id", "repo", r_map, "in_repo")
    edge("issue", iss, "author_id", "user", u_map, "opened_by")
    edge("issue", iss, "repo_id", "repo", r_map, "in_repo")
    edge("issue", iss, "pull_request_id", "pr", p_map, "conversation_of")
    edge("pr", prs, "author_id", "user", u_map, "opened_by")
    edge("pr", prs, "merged_by_id", "user", u_map, "merged_by", mask=(prs.merged_at < cutoff))
    edge("pr", prs, "repo_id", "repo", r_map, "in_repo")
    edge("review", revs, "reviewer_id", "user", u_map, "reviewed_by")
    edge("review", revs, "pr_id", "pr", p_map, "reviews")
    edge("review", revs, "repo_id", "repo", r_map, "in_repo")
    edge("rcom", rcom, "author_id", "user", u_map, "written_by")
    edge("rcom", rcom, "pr_id", "pr", p_map, "on_pr")
    edge("rcom", rcom, "pull_request_review_id", "review", rv_map, "part_of")
    edge("rcom", rcom, "repo_id", "repo", r_map, "in_repo")
    return T.ToUndirected()(data)


class RelGNN(nn.Module):
    """Per-type input encoders -> L x HeteroConv(SAGE) -> user head with skip connection."""

    def __init__(self, metadata, in_dims, hidden=128, layers=2, dropout=0.2, n_out=6):
        super().__init__()
        self.enc = nn.ModuleDict({t: nn.Sequential(nn.Linear(in_dims[t], hidden), nn.ReLU(),
                                                   nn.Linear(hidden, hidden))
                                  for t in metadata[0]})
        self.convs = nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in metadata[1]}, aggr="sum")
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(hidden) for t in metadata[0]})
                                    for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, n_out))

    def forward(self, data):
        h = {t: self.enc[t](data[t].x) for t in data.node_types}
        h0 = h["user"]
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, data.edge_index_dict)
            h = {t: self.drop(Fn.relu(norm[t](out[t] + h[t]))) if t in out else h[t] for t in h}
        n = data["user"].n_cohort
        return self.head(torch.cat([h["user"][:n], h0[:n]], 1))
