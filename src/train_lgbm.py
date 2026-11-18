"""Stage 2: LightGBM on the point-in-time features (one binary model per activity).

Variants (for the ablation study):
  temporal  : only the user's own temporal features (group A)
  full      : temporal + cross-table relational + global features (A + B + C)

Validation = the last N_VAL labelled weeks (time split). The final model is retrained on
all labelled weeks with the number of trees found on validation, then predicts the test week.
"""
import argparse
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import WORK, f1_scores

N_VAL = int(__import__("os").environ.get("A2_NVAL", 2))
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=100,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, n_jobs=8)
TEMPORAL_PREFIX = ("cnt_", "recency_", "weekfrac_", "decay_", "active_days", "first_seen",
                   "trend", "weekend", "hour_", "repos_", "rel_", "global_", "user_rate")


def load_split():
    ds = pickle.load(open(WORK / "dataset.pkl", "rb"))
    ts = sorted(ds["samples"])
    val_ts = ts[-N_VAL:]
    gap = ds["gap"]
    tr_ts = [t for t in ts if t + pd.Timedelta(days=7) <= val_ts[0] - gap]
    return ds, tr_ts, val_ts


def stack(ds, ts, cols, halflife=None):
    """Stack weekly samples; optional exponential recency weights (half-life in weeks)."""
    X = pd.concat([ds["samples"][t][0][cols] for t in ts], ignore_index=True)
    y = np.vstack([ds["samples"][t][1] for t in ts])
    if halflife is None:
        return X, y
    age = np.concatenate([np.full(len(ds["samples"][t][1]), (ts[-1] - t).days / 7) for t in ts])
    return X, y, 0.5 ** (age / halflife)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full", choices=["temporal", "full"])
    ap.add_argument("--halflife", type=float, default=3.0, help="recency weight half-life (weeks); 0 = none")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    hl = a.halflife if a.halflife > 0 else 1e9
    ds, tr_ts, val_ts = load_split()
    cols = list(ds["X_test"].columns)
    if a.variant == "temporal":
        cols = [c for c in cols if c.startswith(TEMPORAL_PREFIX)]
    print(f"{a.variant}: {len(cols)} features, {len(tr_ts)} train weeks, val weeks {val_ts}")
    Xtr, ytr, wtr = stack(ds, tr_ts, cols, hl)
    Xva, yva = stack(ds, val_ts, cols)
    Xall, yall, wall = stack(ds, tr_ts + val_ts, cols, hl)

    val_p = np.zeros_like(yva); test_p = np.zeros((len(ds["X_test"]), 6), np.float32)
    importance = {}
    for k, name in enumerate(ds["targets"]):
        m = lgb.train(PARAMS, lgb.Dataset(Xtr, ytr[:, k], weight=wtr), 3000,
                      valid_sets=[lgb.Dataset(Xva, yva[:, k])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        val_p[:, k] = m.predict(Xva, num_iteration=m.best_iteration)
        n_iter = max(50, int(m.best_iteration * 1.1))
        final = lgb.train(PARAMS, lgb.Dataset(Xall, yall[:, k], weight=wall), n_iter)
        test_p[:, k] = final.predict(ds["X_test"][cols])
        importance[name] = pd.Series(final.feature_importance("gain"), index=cols)
        print(f"  {name:18s} best_iter {m.best_iteration}")
    from train_gnn import best_f1_threshold_score
    print(f"val score (F1-opt thr): {best_f1_threshold_score(yva, val_p):.4f}")
    np.savez(WORK / f"pred_lgbm_{a.variant}{a.tag}.npz", val_p=val_p, val_y=yva, test_p=test_p)
    pd.DataFrame(importance).to_csv(WORK / f"importance_lgbm_{a.variant}{a.tag}.csv")


if __name__ == "__main__":
    main()
