"""Stage 3: train the heterogeneous temporal GNN (RDL) on weekly snapshots.

Variants (ablation):
  --user_feats full  : user nodes start from the engineered point-in-time features
  --user_feats none  : user nodes start from a constant (pure RDL: everything learned from graph)
  --no_time          : drop the relative time encoding on event nodes
"""
import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn

from common import TEST_T, WORK, f1_scores, load_tables
from gnn import RelGNN, build_snapshot
from train_lgbm import load_split


def best_f1_threshold_score(y, p):
    """Score with per-activity thresholds that maximise F1 (quick model-selection metric)."""
    pred = np.zeros_like(p)
    for k in range(y.shape[1]):
        best, bt = -1, 0.5
        for th in np.linspace(0.05, 0.95, 37):
            tp = ((p[:, k] > th) & (y[:, k] == 1)).sum(); fp = ((p[:, k] > th) & (y[:, k] == 0)).sum()
            fn = ((p[:, k] <= th) & (y[:, k] == 1)).sum()
            f = 2 * tp / max(2 * tp + fp + fn, 1)
            if f > best:
                best, bt = f, th
        pred[:, k] = p[:, k] > bt
    return f1_scores(y, pred)["score"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_feats", default="full", choices=["full", "none"])
    ap.add_argument("--no_time", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--halflife", type=float, default=3.0, help="recency weight half-life (weeks)")
    a = ap.parse_args()
    tag = f"gnn_{a.user_feats}" + ("_notime" if a.no_time else "")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, tr_ts, val_ts = load_split()
    tb = load_tables()
    gap = ds["gap"]

    def scale_fit(X):
        Z = np.sign(X) * np.log1p(np.abs(X))
        return Z.mean(), Z.std() + 1e-3

    cols = list(ds["X_test"].columns)
    mu, sd = scale_fit(pd.concat([ds["samples"][t][0] for t in tr_ts]))

    def user_x(X):
        if a.user_feats == "none":
            return pd.DataFrame(np.zeros((len(X), 1), np.float32), index=X.index)
        Z = np.sign(X) * np.log1p(np.abs(X))
        return ((Z - mu) / sd).clip(-5, 5).astype(np.float32)

    t0 = time.time()
    snaps = {}
    for t in tr_ts + val_ts:
        X, y = ds["samples"][t]
        g = build_snapshot(tb, user_x(X[cols]), t - gap, use_time=not a.no_time)
        snaps[t] = (g, torch.from_numpy(y))
    test_g = build_snapshot(tb, user_x(ds["X_test"][cols]), TEST_T - gap,
                            use_time=not a.no_time)
    print(f"{tag}: built {len(snaps) + 1} snapshots in {time.time() - t0:.0f}s")
    g0 = snaps[tr_ts[0]][0]
    in_dims = {nt: g0[nt].x.shape[1] for nt in g0.node_types}

    def run(train_ts, epochs, seed, eval_ts=None):
        torch.manual_seed(seed); np.random.seed(seed)
        model = RelGNN(g0.metadata(), in_dims, a.hidden, a.layers).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1) * len(train_ts))
        hist = []
        for ep in range(epochs):
            model.train()
            for i in np.random.permutation(len(train_ts)):
                g, y = snaps[train_ts[i]]
                g = g.to(dev)
                w = 0.5 ** ((train_ts[-1] - train_ts[i]).days / 7 / a.halflife)
                loss = w * Fn.binary_cross_entropy_with_logits(model(g), y.to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if eval_ts:
                p, yy = predict(model, eval_ts)
                vl = Fn.binary_cross_entropy(torch.tensor(p), torch.tensor(yy)).item()
                hist.append((vl, best_f1_threshold_score(yy, p), p))
                print(f"  seed {seed} ep {ep:02d} val bce {vl:.4f} score {hist[-1][1]:.4f}")
        return model, hist

    @torch.no_grad()
    def predict(model, ts):
        model.eval()
        ps, ys = [], []
        for t in ts:
            g, y = snaps[t]
            ps.append(torch.sigmoid(model(g.to(dev))).cpu().numpy()); ys.append(y.numpy())
        return np.vstack(ps), np.vstack(ys)

    val_ps, test_ps, best_eps = [], [], []
    for seed in range(a.seeds):
        _, hist = run(tr_ts, a.epochs, seed, eval_ts=val_ts)
        be = int(np.argmax([h[1] for h in hist]))   # epoch with best validation score
        best_eps.append(be + 1)
        val_ps.append(hist[be][2])
        model, _ = run(tr_ts + val_ts, be + 1, seed)            # refit on all weeks
        model.eval()
        with torch.no_grad():
            test_ps.append(torch.sigmoid(model(test_g.to(dev))).cpu().numpy())
    val_p = np.mean(val_ps, 0)
    _, val_y = predict(model, val_ts)
    print(f"{tag}: best epochs {best_eps}, val score (F1-opt thr) {best_f1_threshold_score(val_y, val_p):.4f}")
    np.savez(WORK / f"pred_{tag}.npz", val_p=val_p, val_y=val_y, test_p=np.mean(test_ps, 0))


if __name__ == "__main__":
    main()
