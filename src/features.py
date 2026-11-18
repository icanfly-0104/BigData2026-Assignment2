"""Point-in-time user features (temporal + cross-table relational).

Every feature for a prediction time t is computed only from rows whose row time is
< cutoff = t - gap, where gap reproduces the real situation at test time (our data ends
some days before the test week starts). "Known only later" columns (merged_at, closed_at,
...) are used only when their value is < cutoff.
"""
import numpy as np
import pandas as pd

from common import TARGETS

DAY = pd.Timedelta(days=1)
WINDOWS = [1, 3, 7, 14, 28, 56, 112]


class FeatureBuilder:
    def __init__(self, tb, log):
        self.tb, self.log = tb, log
        users = tb["Users"]
        self.cohort = users.loc[users.is_cohort.fillna(False), "user_id"].to_numpy()
        self.uidx = pd.Series(np.arange(len(self.cohort)), index=self.cohort)
        self.login = users.set_index("user_id").login
        repos = tb["Repositories"]
        self.repo_is_org = repos.set_index("repo_id").org_id.notna()
        self.repo_owner = repos.set_index("repo_id").owner_login

    # ------------------------------------------------------------------ helpers
    def _per_user(self, s, fill=0.0):
        """Series indexed by user_id -> array aligned with self.cohort."""
        return s.reindex(self.cohort).fillna(fill).to_numpy(dtype=np.float32)

    def build(self, cutoff):
        tb, n = self.tb, len(self.cohort)
        F = {}
        lg = self.log[self.log.seen < cutoff]   # row-time filter (no leakage)
        age_days = (cutoff - lg.time) / DAY

        # ---------- (A) temporal: multi-scale counts per activity ----------
        for k, act in enumerate(TARGETS):
            la = lg[lg.act_id == k]
            ad = age_days[la.index]
            for w in WINDOWS:
                F[f"cnt_{act}_{w}d"] = self._per_user(la.user_id[ad < w].value_counts())
            F[f"cnt_{act}_all"] = self._per_user(la.user_id.value_counts())
            last = ad.groupby(la.user_id).min()
            F[f"recency_{act}"] = self._per_user(last, fill=999.0)
            # weekly regularity: fraction of past weeks (relative to cutoff) with this activity
            wk = (ad // 7).astype(int)
            for nw in (4, 8, 16):
                m = wk < nw
                F[f"weekfrac_{act}_{nw}w"] = self._per_user(
                    pd.Series(wk[m].values, index=la.user_id[m].values)
                    .groupby(level=0).nunique()) / nw
            # exponentially-decayed activity (half-life 7 days)
            F[f"decay_{act}"] = self._per_user(np.exp(-ad * np.log(2) / 7).groupby(la.user_id).sum())

        # any-activity summaries
        F["recency_any"] = self._per_user(age_days.groupby(lg.user_id).min(), fill=999.0)
        F["first_seen_days"] = self._per_user(age_days.groupby(lg.user_id).max(), fill=0.0)
        for w in (7, 28, 56):
            m = age_days < w
            days = (age_days[m] // 1).astype(int)
            F[f"active_days_{w}d"] = self._per_user(
                pd.Series(days.values, index=lg.user_id[m].values).groupby(level=0).nunique())
            F[f"repos_{w}d"] = self._per_user(lg[m].groupby("user_id").repo_id.nunique())
        # trend: last week vs previous 3 weeks average
        a7 = (age_days < 7); a28 = (age_days >= 7) & (age_days < 28)
        F["trend_7_vs_21"] = (self._per_user(lg.user_id[a7].value_counts()) + 1) / (
            self._per_user(lg.user_id[a28].value_counts()) / 3 + 1)
        # time-of-week habits (last 56 days)
        m = age_days < 56
        dow = lg.time[m].dt.dayofweek
        F["weekend_share"] = self._per_user((dow >= 5).groupby(lg.user_id[m]).mean())
        hour = lg.time[m].dt.hour
        F["hour_mean"] = self._per_user(hour.groupby(lg.user_id[m]).mean(), fill=-1)
        F["hour_std"] = self._per_user(hour.groupby(lg.user_id[m]).std(), fill=-1)

        # ---------- (B) relational: signals that come through other tables ----------
        pr = tb["PullRequests"][tb["PullRequests"].first_seen_at < cutoff]
        merged = pr.merged_at.where(pr.merged_at < cutoff)          # known-only-later rule
        closed = pr.closed_at.where(pr.closed_at < cutoff)
        open_pr = pr[closed.isna()]
        recent = lambda s, d: (cutoff - s) / DAY < d                # noqa: E731
        F["open_prs_authored"] = self._per_user(open_pr.author_id.value_counts())
        F["open_prs_authored_28d"] = self._per_user(
            open_pr.author_id[recent(open_pr.created_at, 28)].value_counts())
        F["prs_merged_by_28d"] = self._per_user(
            pr.merged_by_id[recent(merged, 28).fillna(False)].value_counts())
        F["own_prs_merged_28d"] = self._per_user(
            pr.author_id[recent(merged, 28).fillna(False)].value_counts())

        rv = tb["PRReviews"][tb["PRReviews"].first_seen_at < cutoff]
        rc = tb["PRReviewComments"][tb["PRReviewComments"].created_at < cutoff]
        ic = tb["IssueComments"][tb["IssueComments"].created_at < cutoff]
        iss = tb["Issues"][tb["Issues"].first_seen_at < cutoff]
        pr_author = pr.set_index("pr_id").author_id
        iss_author = iss.set_index("issue_id").author_id
        # feedback the user RECEIVED on own PRs / issues from others (join review -> PR -> author)
        for name, df, key, who, tcol, amap in [
            ("reviews_recv", rv, "pr_id", "reviewer_id", "first_seen_at", pr_author),
            ("revcomments_recv", rc, "pr_id", "author_id", "created_at", pr_author),
            ("comments_recv", ic, "issue_id", "author_id", "created_at", iss_author)]:
            d = df[recent(df[tcol], 14)]
            owner = d[key].map(amap)
            d = d[owner.notna() & (owner != d[who])]
            F[f"{name}_14d"] = self._per_user(owner[d.index].value_counts())
        # how often the user reviews OTHER people's PRs (maintainer role)
        F["review_others_ratio"] = self._per_user(
            (rv.pr_id.map(pr_author) != rv.reviewer_id).groupby(rv.reviewer_id).mean())
        # requested / assigned work
        F["assigned_open_issues"] = self._per_user(
            iss.assignee_id[iss.closed_at.where(iss.closed_at < cutoff).isna()].value_counts())

        # repo neighbourhood: activity of ALL cohort users in the repos this user used (28d)
        l28 = lg[age_days < 28]
        repo_act = l28.groupby(["repo_id", "act_id"]).size().unstack(fill_value=0)
        repo_users = l28.groupby("repo_id").user_id.nunique()
        ur = l28[["user_id", "repo_id"]].drop_duplicates()
        for k, act in enumerate(TARGETS):
            col = repo_act[k] if k in repo_act.columns else pd.Series(0, index=repo_act.index)
            F[f"nbr_repo_{act}_28d"] = self._per_user(
                ur.assign(v=ur.repo_id.map(col).fillna(0).values).groupby("user_id").v.sum())
        F["nbr_repo_users_28d"] = self._per_user(
            ur.assign(v=ur.repo_id.map(repo_users).values).groupby("user_id").v.sum())
        # open PRs by others in the user's repos opened in last 7 days (review opportunities)
        pr7 = pr[recent(pr.created_at, 7)]
        pr7_cnt = pr7.groupby("repo_id").size()
        F["repo_new_prs_7d"] = self._per_user(
            ur.assign(v=ur.repo_id.map(pr7_cnt).fillna(0).values).groupby("user_id").v.sum())
        # organisation / ownership structure
        l56 = lg[age_days < 56]
        F["org_repo_share"] = self._per_user(
            l56.repo_id.map(self.repo_is_org).astype(float).groupby(l56.user_id).mean(), fill=-1)
        own = l56.repo_id.map(self.repo_owner).values == l56.user_id.map(self.login).values
        F["own_repo_share"] = self._per_user(pd.Series(own, index=l56.user_id.values)
                                             .groupby(level=0).mean(), fill=-1)
        # co-activity: how many other cohort users share a repo with the user (28d)
        pairs = ur.merge(ur, on="repo_id")
        pairs = pairs[pairs.user_id_x != pairs.user_id_y]
        F["coworkers_28d"] = self._per_user(pairs.groupby("user_id_x").user_id_y.nunique())

        # ---------- (C) global context (same for every user at this time) ----------
        # GH Archive coverage changes strongly over time (e.g. PushEvent almost disappears in
        # late Oct 2026), so we also give the model the regime level and user activity
        # *relative* to that level, which transfers between regimes.
        act7 = age_days < 7
        F["global_active_users_7d"] = np.full(
            n, lg.user_id[act7].nunique() / len(self.cohort), np.float32)
        for k, act in enumerate(TARGETS):
            g7 = ((lg.act_id == k) & act7).sum() / len(self.cohort)
            g28 = ((lg.act_id == k) & (age_days < 28)).sum() / len(self.cohort)
            F[f"global_{act}_7d"] = np.full(n, g7, np.float32)
            F[f"global_{act}_28d"] = np.full(n, g28, np.float32)
            F[f"rel_{act}_7d"] = F[f"cnt_{act}_7d"] / (g7 + 1e-3)
            F[f"rel_{act}_28d"] = F[f"cnt_{act}_28d"] / (g28 + 1e-3)
            F[f"user_rate_{act}_7d"] = np.full(n, (F[f"cnt_{act}_7d"] > 0).mean(), np.float32)

        X = pd.DataFrame(F)
        X.index = self.cohort
        return X
