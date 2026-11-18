"""Stage 4: baselines, ensemble, decision rule, results table and the submission file.

Decision rule. GH Archive coverage drifts strongly (e.g. PushEvent nearly disappears after
2026-10-25), so predicted probabilities are not calibrated across weeks: a fixed probability
threshold tuned on validation predicts almost no positives at test time. We therefore tune,
per activity, the *fraction of users predicted positive* (top-q by score) to maximise
Score = (macro-F1 + micro-F1) / 2. Rankings transfer between regimes much better than
probabilities (see the "prob" vs "rate" columns of work/results.csv).

"Honest" scores: tune on one validation week, evaluate on the other, average both ways.
"""
import argparse
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from common import ROOT, TARGETS, WORK, f1_scores
from train_lgbm import load_split

PROB_GRID = np.round(np.arange(0.04, 0.80, 0.01), 2)
RATE_GRID = np.round(np.arange(0.002, 0.20, 0.002), 3)


def apply_prob(p, th):
    return (p > th).astype(float)


def apply_rate(p, q):
    out = np.zeros_like(p)
    for k in range(p.shape[1]):
        out[:, k] = p[:, k] >= np.quantile(p[:, k], 1 - q[k])
    return out


def tune(y, p, mode="rate", iters=3):
    grid, apply = (RATE_GRID, apply_rate) if mode == "rate" else (PROB_GRID, apply_prob)
    v = np.full(6, 0.05 if mode == "rate" else 0.3)
    for _ in range(iters):
        for k in range(6):
            best = (-1, v[k])
            for g in grid:
                v[k] = g
                s = f1_scores(y, apply(p, v))["score"]
                if s > best[0]:
                    best = (s, g)
            v[k] = best[1]
    return v


def honest(y, p, n_users, mode="rate"):
    """Tune on one validation week, evaluate on the other, average."""
    apply = apply_rate if mode == "rate" else apply_prob
    w = [slice(0, n_users), slice(n_users, 2 * n_users)]
    out = []
    for a, b in [(0, 1), (1, 0)]:
        v = tune(y[w[a]], p[w[a]], mode)
        out.append(f1_scores(y[w[b]], apply(p[w[b]], v)))
    return {k: np.mean([o[k] for o in out]) for k in ("macro", "micro", "score")}, \
        {t: np.mean([o["per"][t] for o in out]) for t in TARGETS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_id", default="studentID")
    a = ap.parse_args()
    ds, tr_ts, val_ts = load_split()
    n = len(ds["cohort"])
    is_test_user = np.isin(ds["gh_of_cohort"], ds["test_ids"])
    m2 = np.concatenate([is_test_user, is_test_user])          # score on the test users only
    nt = is_test_user.sum()

    preds = {}
    for name in ["lgbm_temporal", "lgbm_full", "gnn_none", "gnn_full_notime", "gnn_full"]:
        f = WORK / f"pred_{name}.npz"
        if f.exists():
            preds[name] = dict(np.load(f))
    y = preds["lgbm_full"]["val_y"]

    # ensemble weight search (LightGBM full + GNN full) with the rate decision rule
    if "gnn_full" in preds:
        best = (-1, 0)
        for wgt in np.linspace(0, 1, 11):
            p = wgt * preds["lgbm_full"]["val_p"] + (1 - wgt) * preds["gnn_full"]["val_p"]
            sc = honest(y[m2], p[m2], nt)[0]["score"]
            print(f"  w_lgbm={wgt:.1f}  honest score {sc:.4f}")
            if sc > best[0]:
                best = (sc, wgt)
        w = best[1]
        preds["ensemble"] = {k: w * preds["lgbm_full"][k] + (1 - w) * preds["gnn_full"][k]
                             for k in ("val_p", "test_p")}
        print(f"ensemble weight on LightGBM = {w}")

    labels = {"lgbm_temporal": "LightGBM, temporal features only",
              "lgbm_full": "LightGBM, temporal + relational",
              "gnn_none": "RDL-GNN, raw graph only (+time enc.)",
              "gnn_full_notime": "RDL-GNN + user features, no time enc.",
              "gnn_full": "RDL-GNN + user features + time enc.",
              "ensemble": "Ensemble (LightGBM + RDL-GNN)"}
    # baseline: copy last week (activity in the 7 days before the cutoff)
    cl_val = np.vstack([np.stack([(ds["samples"][t][0][f"cnt_{k}_7d"] > 0).to_numpy(float)
                                  for k in TARGETS], 1) for t in val_ts])
    s = f1_scores(y[m2], cl_val[m2])
    rows = [{"model": "Copy last week (baseline)", "score_prob_thr": np.nan,
             "macro_f1": s["macro"], "micro_f1": s["micro"], "score": s["score"],
             **{f"F1_{k}": v for k, v in s["per"].items()}}]
    for name, pr in preds.items():
        sp = honest(y[m2], pr["val_p"][m2], nt, "prob")[0]["score"]
        sc, per = honest(y[m2], pr["val_p"][m2], nt, "rate")
        rows.append({"model": labels[name], "score_prob_thr": sp, "macro_f1": sc["macro"],
                     "micro_f1": sc["micro"], "score": sc["score"],
                     **{f"F1_{k}": v for k, v in per.items()}})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(4).to_string(index=False))
    res.to_csv(WORK / "results.csv", index=False)

    # final decision rule tuned on both validation weeks, then the submission
    final = "ensemble" if "ensemble" in preds else "lgbm_full"
    q = tune(y[m2], preds[final]["val_p"][m2], "rate")
    print("final model:", final, "positive rate per activity:", dict(zip(TARGETS, q)))
    row_of = pd.Series(np.arange(n), index=ds["gh_of_cohort"])
    rows_test = row_of[ds["test_ids"]].to_numpy()          # every test user is in the data
    pred_test = apply_rate(preds[final]["test_p"][rows_test], q).astype(int)
    out = pd.DataFrame(pred_test, columns=TARGETS)
    out.insert(0, "UserID", ds["test_ids"])
    path = ROOT / f"output_{a.student_id}.csv"
    out.to_csv(path, index=False)
    print("positive rate in submission:", out[TARGETS].mean().round(3).to_dict())
    pickle.dump({"rates": q, "final": final, "w_lgbm": w}, open(WORK / "final.pkl", "wb"))

    # official format check + scoring the last validation week with the official script
    ev = ROOT / "assignment2" / "scripts" / "evaluate.py"
    subprocess.run([sys.executable, str(ev), "--pred", str(path), "--input",
                    str(ROOT / "assignment2" / "submission" / "test_input.csv")])
    last = slice(n, 2 * n)
    q1 = tune(y[:n][is_test_user], preds[final]["val_p"][:n][is_test_user], "rate")
    vp = apply_rate(preds[final]["val_p"][last][is_test_user], q1).astype(int)
    gh = ds["gh_of_cohort"][is_test_user]
    for fn, arr in [("val_pred.csv", vp), ("val_truth.csv", y[last][is_test_user])]:
        df = pd.DataFrame(arr.astype(int), columns=TARGETS); df.insert(0, "UserID", gh)
        df.to_csv(WORK / fn, index=False)
    print(f"\nOfficial evaluate.py on the last validation week ({val_ts[-1]} UTC, "
          f"rates tuned on the previous week):")
    subprocess.run([sys.executable, str(ev), "--pred", str(WORK / "val_pred.csv"),
                    "--truth", str(WORK / "val_truth.csv")])


if __name__ == "__main__":
    main()
