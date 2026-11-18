"""Analysis figures for the slides: regime shift, model comparison, feature importance."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import ROOT, TARGETS, WORK  # noqa: E402

FIG = ROOT / "figures"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF,
                     "figure.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False})


def regime():
    log = pd.read_parquet(WORK / "activity_log.parquet")
    log = log[(log.time >= "2025-12-01") & (log.time < "2026-11-16")]
    wk = log.groupby([log.time.dt.to_period("W").dt.start_time, "act"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, a in enumerate(TARGETS):
        ax.plot(wk.index, wk[a], color=SERIES[i], lw=2, label=a)
    ax.set_yscale("log"); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylabel("events per week (cohort, log scale)")
    ax.axvspan(pd.Timestamp("2026-10-26"), wk.index[-1] + pd.Timedelta(days=7), color="#eeeeee",
               zorder=0)
    ax.text(pd.Timestamp("2026-10-27"), ax.get_ylim()[1] * 0.6, "validation\nweeks", fontsize=9,
            color=INK2)
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
    ax.set_xlim(wk.index[0], pd.Timestamp("2026-11-28"))
    ax.set_title("GH Archive coverage changes over time: weekly target events of the 9,000 users",
                 loc="left", color=INK, pad=28)
    fig.savefig(FIG / "a_regime_shift.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def models():
    r = pd.read_csv(WORK / "results.csv")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(r))[::-1]
    best = r.score.idxmax()
    cols = ["#9e9d98" if i == 0 else ("#2a78d6" if i != best else "#0b5cad") for i in range(len(r))]
    ax.barh(y, r.score * 100, color=cols, height=0.6)
    for yi, v in zip(y, r.score * 100):
        ax.text(v + 0.4, yi, f"{v:.1f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(r.model, fontsize=10)
    ax.set_xlabel("Score = (macro-F1 + micro-F1) / 2  (%)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_xlim(0, max(r.score * 100) + 6)
    ax.set_title("Validation score on the test users (weeks of 10/31 & 11/07, decision rule tuned on the other week)",
                 loc="left", fontsize=11.5, color=INK)
    fig.savefig(FIG / "b_model_comparison.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def importance():
    imp = pd.read_csv(WORK / "importance_lgbm_full.csv", index_col=0)
    s = (imp / imp.sum()).mean(axis=1).sort_values(ascending=False).head(18)[::-1]

    def group(c):
        if c.startswith(("global_", "user_rate")):
            return 2
        if c.startswith(("cnt_", "recency_", "weekfrac_", "decay_", "active_days", "first_seen",
                         "trend", "weekend", "hour_", "repos_", "rel_")):
            return 0
        return 1
    names = ["temporal (user history)", "relational (cross-table)", "global regime"]
    gc = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    g = [group(c) for c in s.index]
    ax.barh(range(len(s)), s.values * 100, color=[gc[i] for i in g], height=0.65)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=9)
    ax.set_xlabel("share of total gain, averaged over the 6 activity models (%)")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for i, n in enumerate(names):
        ax.barh([0], [0], color=gc[i], label=n)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("Top-18 LightGBM features", loc="left", color=INK)
    fig.savefig(FIG / "c_feature_importance.png", dpi=200, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    regime(); models(); importance()
