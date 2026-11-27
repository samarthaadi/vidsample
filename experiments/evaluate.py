"""Shortcut test and CPU baseline detector.

python experiments/evaluate.py EXP_DIR OUT_MD

Feature sets (each on raw clips and on --normalize'd clips):
  meta  container/stream metadata only (size, bitrate, resolution, fps, codec, audio ...)
        -> gradient-boosted trees. High AUC here means real/fake is separable without
           looking at a single pixel: a shortcut.
  clip  frozen CLIP ViT-B/32 embedding (4 frames, first 2 s) -> logistic regression.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from common import eval_sets, load_clips  # noqa: E402

DATASETS = ["aigvdbench", "genbuster200k", "genvidbench", "genvideo100k"]
NUMERIC = ["file_bytes", "duration", "fps", "width", "height", "aspect", "bitrate",
           "bits_per_pixel", "has_audio"]


def load_features(exp, df):
    feats = {}
    for var in ("raw", "norm"):
        meta = pd.read_csv(os.path.join(exp, "features", f"{var}_meta.csv")).sort_values("id")
        ok = meta["codec"].notna().to_numpy()
        cat = pd.get_dummies(meta[["codec", "pix_fmt", "container"]].fillna("?"), dtype=float)
        X = pd.concat([meta[NUMERIC].fillna(0), cat], axis=1).to_numpy(float)
        feats[f"meta_{var}"] = (X, ok, meta)
        C = np.load(os.path.join(exp, "features", f"{var}_clip.npy"))
        feats[f"clip_{var}"] = (C, ~np.isnan(C[:, 0]), None)
    return feats


def model(kind):
    if kind.startswith("meta"):
        return HistGradientBoostingClassifier(max_iter=200, random_state=0)
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000))


def fit_eval(kind, X, ok, y, train, tests):
    tr = train & ok
    if len(set(y[tr])) < 2:
        return None, {}
    clf = model(kind).fit(X[tr], y[tr])
    out = {}
    for name, m in tests.items():
        m = m & ok
        if m.sum() == 0 or len(set(y[m])) < 2:
            continue
        p = clf.predict_proba(X[m])[:, 1]
        yt = y[m]
        acc = 0.5 * ((p[yt == 1] >= 0.5).mean() + (p[yt == 0] < 0.5).mean())
        out[name] = {"auc": roc_auc_score(yt, p), "bacc": acc, "n": int(m.sum())}
    return clf, out


def single_feature_auc(meta, y, mask):
    best = []
    for f in ["file_bytes", "bitrate", "bits_per_pixel", "width", "height", "fps", "duration"]:
        v = meta[f].fillna(0).to_numpy()[mask]
        if len(set(v)) < 2:
            continue
        a = roc_auc_score(y[mask], v)
        best.append((max(a, 1 - a), f))
    return max(best) if best else (0.5, "-")


def main(exp, out_md):
    df = load_clips(exp)
    y = df["y"].to_numpy()
    sets = {k: {n: m.to_numpy() for n, m in v.items()} for k, v in eval_sets(df).items()}
    feats = load_features(exp, df)
    kinds = ["meta_raw", "meta_norm", "clip_raw", "clip_norm"]
    res = {"within": {}, "cross": {}, "single": {}, "per_gen": {}, "counts": {}}

    for ds in DATASETS:
        s = sets[ds]
        res["counts"][ds] = {n: {"real": int((m & (y == 0)).sum()), "fake": int((m & (y == 1)).sum())}
                             for n, m in s.items()}
        for k in kinds:
            X, ok, meta = feats[k]
            clf, r = fit_eval(k, X, ok, y, s["train"], {n: m for n, m in s.items() if n != "train"})
            res["within"][(ds, k)] = r
            if k == "clip_norm" and clf is not None:
                for n in ("test", "ood"):
                    if n not in s:
                        continue
                    m = s[n] & ok
                    p = clf.predict_proba(X[m])[:, 1] >= 0.5
                    g = df.loc[m, ["generator", "label"]].assign(pred=p)
                    g["correct"] = np.where(g["label"] == "fake", g["pred"], ~g["pred"])
                    res["per_gen"][(ds, n)] = g.groupby(["label", "generator"])["correct"].agg(["mean", "size"])
        for var in ("raw", "norm"):
            res["single"][(ds, var)] = single_feature_auc(feats[f"meta_{var}"][2], y, s["train"])

    for k in kinds:
        X, ok, _ = feats[k]
        trains = {ds: sets[ds]["train"] for ds in DATASETS}
        trains["all"] = np.logical_or.reduce(list(trains.values()))
        for tr_name, tr in trains.items():
            _, r = fit_eval(k, X, ok, y, tr, {ds: sets[ds]["test"] for ds in DATASETS})
            res["cross"][(k, tr_name)] = {ds: v["auc"] for ds, v in r.items()}
        for held in DATASETS:       # leave one dataset out
            tr = np.logical_or.reduce([sets[d]["train"] for d in DATASETS if d != held])
            _, r = fit_eval(k, X, ok, y, tr, {held: sets[held]["test"]})
            res["cross"][(k, f"all-but-{held}")] = {held: r[held]["auc"]} if held in r else {}

    md = render(res, df, feats)
    with open(out_md, "w") as fh:
        fh.write(md)
    with open(os.path.splitext(out_md)[0] + ".json", "w") as fh:
        json.dump({"within": {f"{a}|{b}": v for (a, b), v in res["within"].items()},
                   "cross": {f"{a}|{b}": v for (a, b), v in res["cross"].items()},
                   "single": {f"{a}|{b}": v for (a, b), v in res["single"].items()},
                   "per_gen": {f"{a}|{b}": {f"{lab}|{gen}": {"acc": float(r["mean"]), "n": int(r["size"])}
                                            for (lab, gen), r in g.iterrows()}
                               for (a, b), g in res["per_gen"].items()},
                   "counts": res["counts"]}, fh, indent=1)
    print(md)


def _f(x):
    return "–" if x is None else f"{x:.3f}"


def render(res, df, feats):
    L = ["# vidsample experiments: shortcut test and CPU baseline", ""]
    L += ["## Data", "", "| dataset | train real/fake | test real/fake | ood real/fake |", "|---|---|---|---|"]
    for ds in DATASETS:
        c = res["counts"][ds]
        cell = lambda n: f"{c[n]['real']}/{c[n]['fake']}" if n in c else "–"
        L.append(f"| {ds} | {cell('train')} | {cell('test')} | {cell('ood')} |")
    bad = {k: int((~v[1]).sum()) for k, v in feats.items()}
    L += ["", f"Clips without usable features (dropped): {bad}", ""]

    L += ["## 1. Within-dataset AUC (train → in-distribution test / unseen generators)", "",
          "| dataset | meta raw | meta norm | CLIP raw | CLIP norm |", "|---|---|---|---|---|"]
    for ds in DATASETS:
        row = []
        for k in ["meta_raw", "meta_norm", "clip_raw", "clip_norm"]:
            r = res["within"][(ds, k)]
            cell = _f(r.get("test", {}).get("auc"))
            if "ood" in r:
                cell += f" / {_f(r['ood']['auc'])}"
            row.append(cell)
        L.append(f"| {ds} | " + " | ".join(row) + " |")
    L += ["", "Cells are `test AUC / ood AUC`. Balanced accuracy at threshold 0.5:", "",
          "| dataset | meta raw | meta norm | CLIP raw | CLIP norm |", "|---|---|---|---|---|"]
    for ds in DATASETS:
        row = []
        for k in ["meta_raw", "meta_norm", "clip_raw", "clip_norm"]:
            r = res["within"][(ds, k)]
            cell = _f(r.get("test", {}).get("bacc"))
            if "ood" in r:
                cell += f" / {_f(r['ood']['bacc'])}"
            row.append(cell)
        L.append(f"| {ds} | " + " | ".join(row) + " |")

    L += ["", "## 2. Best single metadata feature (AUC on the training set)", "",
          "| dataset | raw | normalized |", "|---|---|---|"]
    for ds in DATASETS:
        a, b = res["single"][(ds, "raw")], res["single"][(ds, "norm")]
        L.append(f"| {ds} | {a[0]:.3f} ({a[1]}) | {b[0]:.3f} ({b[1]}) |")

    for k, title in [("meta_raw", "metadata, raw"), ("clip_raw", "CLIP, raw"), ("clip_norm", "CLIP, normalized")]:
        L += ["", f"## 3. Cross-dataset AUC ({title}): rows = trained on, columns = tested on", "",
              "| train \\ test | " + " | ".join(DATASETS) + " |", "|---" * (len(DATASETS) + 1) + "|"]
        for tr in DATASETS + ["all"]:
            r = res["cross"][(k, tr)]
            L.append(f"| {tr} | " + " | ".join(_f(r.get(ds)) for ds in DATASETS) + " |")
        loo = [res["cross"][(k, f"all-but-{d}")].get(d) for d in DATASETS]
        L.append("| all but column | " + " | ".join(_f(x) for x in loo) + " |")

    L += ["", "## 4. Per-generator accuracy (CLIP normalized, within-dataset, threshold 0.5)", ""]
    for (ds, n), g in res["per_gen"].items():
        L.append(f"**{ds} – {n}**: " + ", ".join(
            f"{gen} {row['mean']:.2f}" for (lab, gen), row in g.iterrows()) + "  ")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
