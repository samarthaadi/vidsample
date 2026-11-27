"""Shared helpers for the experiments: clip table and train/test/ood sets per dataset."""
import hashlib
import os

import pandas as pd

NORM_SPEC = dict(size=512, fps=16, max_seconds=2, crf=23)


def load_clips(exp_dir):
    """The raw manifest with an `id` column (row order is the feature order everywhere)."""
    df = pd.read_csv(os.path.join(exp_dir, "raw", "manifest.csv"))
    df = df.sort_values("path").reset_index(drop=True)
    df["id"] = range(len(df))
    df["y"] = (df["label"] == "fake").astype(int)
    df["norm_path"] = df["path"].map(lambda p: os.path.splitext(p)[0] + ".mp4")
    return df


def _holdout(path, frac=0.3):
    h = int(hashlib.md5(path.encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < frac


def eval_sets(df):
    """{dataset: {"train": mask, "test": mask, "ood": mask?}}

    test = same generators as train (in-distribution), ood = unseen generators/sources.
    aigvdbench    train split / open-source test split / closed-source test split (+ test reals)
    genbuster200k train / test / benchmark (8 closed-source generators, own reals)
    genvidbench   70% / 30% of pair1 / pair2 (different generators and real source)
    genvideo100k  70% / 30% (no official split)
    """
    out = {}
    hold = df["path"].map(_holdout)
    d = df["dataset"]

    m = d == "aigvdbench"
    open_src = df["generator"].str.contains(r"_(?:T2V|I2V|V2V)$")
    test = m & (df["split"] == "test")
    out["aigvdbench"] = {"train": m & (df["split"] == "train"),
                         "test": test & (open_src | (df["label"] == "real")),
                         "ood": test & (~open_src | (df["label"] == "real"))}

    m = d == "genbuster200k"
    out["genbuster200k"] = {"train": m & (df["split"] == "train"), "test": m & (df["split"] == "test"),
                            "ood": m & (df["split"] == "benchmark")}

    m = d == "genvidbench"
    p1 = m & (df["split"] == "pair1")
    out["genvidbench"] = {"train": p1 & ~hold, "test": p1 & hold, "ood": m & (df["split"] == "pair2")}

    m = d == "genvideo100k"
    out["genvideo100k"] = {"train": m & ~hold, "test": m & hold}
    return out
