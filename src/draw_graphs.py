"""Q1 figures (schema graph, relational entity graph) + Q4 framework figure.

Writes PNGs to figures/ and Graphviz .dot sources (render with `dot -Tpng x.dot -o x.png`).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from common import ROOT, load_tables  # noqa: E402

sys.path.insert(0, str(ROOT / "assignment2" / "scripts"))
from preprocess import TABLES, code_int  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SINGULAR = {"Users": "User", "Repositories": "Repo", "PullRequests": "PR", "Issues": "Issue",
            "IssueComments": "IssueComment", "PRReviews": "PRReview",
            "PRReviewComments": "PRReviewComment", "Pushes": "Push"}
COL = {"Users": "#4C78A8", "Repositories": "#F58518", "PullRequests": "#54A24B",
       "Issues": "#B279A2", "IssueComments": "#E45756", "PRReviews": "#72B7B2",
       "PRReviewComments": "#9D755D", "Pushes": "#EECA3B"}


# ------------------------------------------------------------------ schema graph
def schema_graph():
    import numpy as np
    pos = {"Users": (0.50, 0.92), "Repositories": (0.50, 0.06), "Pushes": (0.02, 0.50),
           "IssueComments": (0.24, 0.50), "Issues": (0.46, 0.50), "PullRequests": (0.70, 0.50),
           "PRReviews": (0.95, 0.76), "PRReviewComments": (0.95, 0.27)}
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.set_xlim(-0.08, 1.12); ax.set_ylim(-0.06, 1.02); ax.axis("off")
    boxes = {}
    for t, (x, y) in pos.items():
        spec = TABLES[t]
        lines = [f"PK  {spec['pk']}"] + [f"FK  {c}" for c in spec["fks"]] + [f"time {spec['time']}"]
        w, h = 0.17, 0.05 + 0.03 * len(lines)
        boxes[t] = (x, y, w, h)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008",
                                    fc="white", ec=COL[t], lw=2.5, zorder=3))
        ax.text(x, y + h / 2 - 0.022, t, ha="center", va="center", fontsize=12, weight="bold",
                color=COL[t], zorder=4)
        for i, ln in enumerate(lines):
            ax.text(x - w / 2 + 0.01, y + h / 2 - 0.052 - 0.03 * i, ln, ha="left", va="center",
                    fontsize=9, family="monospace", zorder=4,
                    weight="bold" if ln.startswith("PK") else "normal",
                    color="#B00020" if ln.startswith("PK") else ("#1a4d8f" if ln.startswith("FK") else "#555"))

    def clip(x, y, w, h, dx, dy):
        """point where the ray from the box centre in direction (dx,dy) leaves the box"""
        sx = (w / 2 + 0.008) / abs(dx) if dx else 1e9
        sy = (h / 2 + 0.008) / abs(dy) if dy else 1e9
        k = min(sx, sy)
        return x + dx * k, y + dy * k

    pairs = {}
    for t, spec in TABLES.items():
        for c, r in spec["fks"].items():
            pairs.setdefault(frozenset((t, r)), []).append((t, c, r))
    for group in pairs.values():
        n = len(group)
        for i, (t, c, r) in enumerate(group):
            if t == r:   # self reference
                x, y, w, h = boxes[t]
                ax.annotate("", xy=(x - w / 2 - 0.008, y - 0.06), xytext=(x - w / 2 - 0.008, y + 0.0),
                            arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.3,
                                            connectionstyle="arc3,rad=1.6"), zorder=2)
                ax.text(x - w / 2 - 0.07, y - 0.03, c, fontsize=8, ha="center", va="center",
                        bbox=dict(fc="#f4f4f4", ec="none", pad=0.8))
                continue
            (x1, y1, w1, h1), (x2, y2, w2, h2) = boxes[t], boxes[r]
            d = np.array([x2 - x1, y2 - y1]); L = np.linalg.norm(d); u = d / L
            nrm = np.array([-u[1], u[0]]) * (i - (n - 1) / 2) * 0.03
            p1 = clip(x1 + nrm[0], y1 + nrm[1], w1, h1, *u)
            p2 = clip(x2 + nrm[0], y2 + nrm[1], w2, h2, *(-u))
            def hits(rad):
                p1a, p2a = np.array(p1), np.array(p2)
                dd = p2a - p1a
                ctrl = (p1a + p2a) / 2 + rad * np.array([dd[1], -dd[0]])
                for tt in np.linspace(0.08, 0.92, 40):
                    q = (1 - tt) ** 2 * p1a + 2 * (1 - tt) * tt * ctrl + tt ** 2 * p2a
                    for b, (bx, by, bw, bh) in boxes.items():
                        if b not in (t, r) and abs(q[0] - bx) < bw / 2 + 0.01 and abs(q[1] - by) < bh / 2 + 0.01:
                            return True, ctrl
                return False, ctrl
            for rad in (0, 0.2, -0.2, 0.35, -0.35, 0.5, -0.5):
                bad, ctrl = hits(rad)
                if not bad:
                    break
            ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3,
                        mutation_scale=14, connectionstyle=f"arc3,rad={rad}"), zorder=2)
            m = 0.25 * np.array(p1) + 0.5 * ctrl + 0.25 * np.array(p2) if rad else                 np.array(p1) * (0.62 - 0.22 * i) + np.array(p2) * (0.38 + 0.22 * i)
            ax.text(m[0], m[1], c, fontsize=8, color="#1a4d8f", ha="center", va="center", zorder=5,
                    bbox=dict(fc="#f4f4f4", ec="none", pad=0.8))
    ax.text(0.5, -0.03, "Arrow = foreign key (FK column) → primary key (PK) of the referenced table;"
            " red = PK, blue = FK, grey = row-time column", ha="center", fontsize=10, color="#333")
    ax.set_title("Schema graph: 8 tables, every foreign key → referenced primary key", fontsize=14)
    fig.savefig(FIG / "q1_schema_graph.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    dot = ["digraph schema {", "  rankdir=LR; node [shape=plaintext];"]
    for t, spec in TABLES.items():
        rows = f'<tr><td bgcolor="{COL[t]}"><b>{t}</b></td></tr>'
        rows += f'<tr><td align="left">PK {spec["pk"]}</td></tr>'
        rows += "".join(f'<tr><td align="left">FK {c}</td></tr>' for c in spec["fks"])
        dot.append(f'  {t} [label=<<table border="1" cellborder="0">{rows}</table>>];')
    for t, spec in TABLES.items():
        for c, r in spec["fks"].items():
            dot.append(f'  {t} -> {r} [label="{c}"];')
    dot.append("}")
    (FIG / "q1_schema_graph.dot").write_text("\n".join(dot))


# ------------------------------------------------------------------ relational entity graph
def entity_graph(tb, raw_ids=(22996444, 28983449), lo="2026-09-15", hi="2026-09-17"):
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    users = tb["Users"]
    codes = {code_int("user", r): r for r in raw_ids}
    sel = users[users.github_user_id.isin(codes)]
    uids = dict(zip(sel.user_id, sel.github_user_id.map(codes)))
    G = nx.DiGraph()
    ntype = {}

    def add(n, t, label):
        if n not in G:
            G.add_node(n, label=label); ntype[n] = t

    for uid, raw in uids.items():
        add(("Users", uid), "Users", f"User\n{raw}")
    rep = lambda r: add(("Repositories", r), "Repositories", f"Repo {r}")  # noqa: E731

    def link(a, b, lab):
        G.add_edge(a, b, label=lab)

    ev = []
    for t, ucol, tcol in [("Pushes", "user_id", "created_at"),
                          ("IssueComments", "author_id", "created_at"),
                          ("PRReviews", "reviewer_id", "first_seen_at"),
                          ("PRReviewComments", "author_id", "created_at"),
                          ("PullRequests", "author_id", "first_seen_at"),
                          ("Issues", "author_id", "first_seen_at")]:
        df = tb[t]
        d = df[df[ucol].isin(uids) & (df[tcol] >= lo) & (df[tcol] < hi)]
        pk = TABLES[t]["pk"]
        for _, r in d.iterrows():
            n = (t, r[pk])
            add(n, t, f"{t[:-1] if t.endswith('s') else t}\n{r[pk]}")
            ev.append((t, r[pk], r[tcol], r[ucol]))
            for c, ref in TABLES[t]["fks"].items():
                v = r[c]
                if pd.isna(v):
                    continue
                v = int(v)
                if ref == "Users" and v not in uids:
                    add(("Users", v), "Users", f"User\n#{v}")
                elif ref == "Repositories":
                    rep(v)
                elif ref not in ("Users", "Repositories"):
                    pr = tb[ref][tb[ref][TABLES[ref]["pk"]] == v]
                    add((ref, v), ref, f"{ref[:-1]}\n{v}")
                    if len(pr) and "repo_id" in pr and pd.notna(pr.repo_id.iloc[0]):
                        rep(int(pr.repo_id.iloc[0]))
                        link((ref, v), ("Repositories", int(pr.repo_id.iloc[0])), "repo_id")
                link(n, (ref, v), c)
    pd.DataFrame(ev, columns=["table", "pk", "time", "user_id"]).sort_values("time").to_csv(
        FIG / "q1_entity_rows.csv", index=False)

    fig, ax = plt.subplots(figsize=(18, 12))
    p = nx.kamada_kawai_layout(G.to_undirected()) if len(G) > 2 else nx.spring_layout(G)
    for t in COL:
        ns = [n for n in G if ntype[n] == t]
        nx.draw_networkx_nodes(G, p, nodelist=ns, node_color=COL[t], ax=ax, label=t,
                               node_size=[1500 if (n[0] == "Users" and n[1] in uids) else 500
                                          for n in ns], edgecolors="black", linewidths=0.6)
    nx.draw_networkx_edges(G, p, ax=ax, arrows=True, arrowsize=10, edge_color="#888", width=0.8)
    nx.draw_networkx_labels(G, p, {n: G.nodes[n]["label"] for n in G}, font_size=6.5, ax=ax)
    if G.number_of_edges() < 120:
        nx.draw_networkx_edge_labels(G, p, nx.get_edge_attributes(G, "label"), font_size=5.5,
                                     ax=ax, alpha=0.8)
    ax.legend(scatterpoints=1, fontsize=9, loc="upper left", markerscale=0.6)
    ax.set_title(f"Relational entity graph: users {raw_ids[0]} & {raw_ids[1]}, "
                 f"{lo:%Y-%m-%d %H:%M} – {hi:%Y-%m-%d %H:%M} UTC "
                 f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)", fontsize=13)
    ax.axis("off")
    fig.savefig(FIG / "q1_relational_entity_graph.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    dot = ["digraph reg {", "  node [style=filled, fontsize=9];"]
    for n in G:
        dot.append(f'  "{n[0]}_{n[1]}" [label="{G.nodes[n]["label"]}", fillcolor="{COL[ntype[n]]}"];')
    for a, b, d in G.edges(data=True):
        dot.append(f'  "{a[0]}_{a[1]}" -> "{b[0]}_{b[1]}" [label="{d["label"]}", fontsize=7];')
    dot.append("}")
    (FIG / "q1_relational_entity_graph.dot").write_text("\n".join(dot))
    counts = pd.Series([ntype[n] for n in G]).value_counts()
    print("entity graph nodes by table:\n", counts.to_string(), "\nedges:", G.number_of_edges())


# ------------------------------------------------------------------ framework figure
def framework():
    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.2); ax.axis("off")

    def box(x, y, w, h, title, body, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06", fc=c, ec="#333",
                                    lw=1.2, alpha=0.95))
        ax.text(x + w / 2, y + h - 0.25, title, ha="center", va="top", fontsize=11, weight="bold")
        ax.text(x + w / 2, y + h - 0.65, body, ha="center", va="top", fontsize=8.6, linespacing=1.35)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.6, color="#333"))

    box(0.2, 5.4, 2.6, 2.4, "① Data", "GH Archive hour files\n2025-12-01 → data end\n"
        "stream + keep cohort\nactors (= preprocess\nStep 2)", "#dbe9f6")
    box(3.3, 5.4, 2.6, 2.4, "② 8 relational tables", "official preprocess.py\nUsers, Repositories,\n"
        "PullRequests, Issues,\nIssueComments, PRReviews,\nPRReviewComments, Pushes", "#dbe9f6")
    box(6.4, 5.4, 3.3, 2.4, "③ Point-in-time samples", "weekly prediction times t\n(Sat 16:00 UTC)\n"
        "cutoff = t − gap (data lag)\nonly rows with row time < cutoff\nlabel = 6 activities in [t, t+7d)",
        "#fde9d9")
    box(0.2, 1.6, 4.4, 3.2, "④a Feature branch (tabular)",
        "Temporal: multi-scale counts (1–112d),\nrecency, weekly regularity, decay,\n"
        "trend, time-of-day/week habits\nRelational (joins across tables):\nfeedback received on own PRs/issues,\n"
        "open PRs, merges, repo-neighbour\nactivity, co-workers, org/owner share\n→ LightGBM × 6 (recency-weighted)", "#e2f0d9")
    box(5.0, 1.6, 4.7, 3.2, "④b Graph branch (RDL-GNN)",
        "snapshot relational entity graph\n(28-day window before cutoff)\n8 node types, 18 FK edge types (+rev)\n"
        "event nodes: relative time encoding\nlog Δt + sin/cos(1,3.5,7,14,28d)\nuser nodes: point-in-time features\n"
        "2× HeteroConv(GraphSAGE) → 6-way head\n(multi-task BCE, recency-weighted, 3 seeds)", "#e2f0d9")
    box(10.3, 2.6, 2.6, 2.6, "⑤ Ensemble", "0.7 · LightGBM\n+ 0.3 · RDL-GNN\n(weight chosen on\n"
        "validation weeks)", "#fff2cc")
    box(13.3, 2.6, 2.5, 2.6, "⑥ Decision", "per activity: top-q%\nusers → 1 (q tuned\nfor (macro+micro F1)/2)\n"
        "→ output_{ID}.csv\n→ evaluate.py check", "#fff2cc")
    box(10.3, 5.6, 5.5, 2.2, "Validation protocol", "time split: last 2 labelled weeks = validation\n"
        "train weeks end before validation cutoff\nrule tuned on one week, scored on the other\n"
        "final models refit on all weeks", "#eeeeee")
    arr(2.8, 6.6, 3.3, 6.6); arr(5.9, 6.6, 6.4, 6.6)
    arr(7.4, 5.4, 2.4, 4.8); arr(8.0, 5.4, 7.35, 4.8)
    ax.plot([2.4, 2.4, 11.0], [1.55, 0.9, 0.9], color="#333", lw=1.6)
    arr(11.0, 0.9, 11.0, 2.55); arr(9.75, 3.4, 10.3, 3.4); arr(12.95, 3.9, 13.3, 3.9)
    ax.plot([4.65, 4.95], [3.2, 3.2], color="#333", lw=1.2, ls="--")
    ax.text(4.8, 3.35, "user\nfeatures", fontsize=7.5, ha="center", style="italic")
    ax.text(6.7, 0.65, "LightGBM probabilities", fontsize=9, style="italic", ha="center")
    fig.savefig(FIG / "q4_framework.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    schema_graph()
    framework()
    entity_graph(load_tables())
    print("figures written to", FIG)
