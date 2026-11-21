"""GenBuster-200K (BusterX) - HF l8cv/GenBuster-200K.

A 27-volume 7z (~113 GB) stored with the Copy coder, so every clip is a plain byte range.
Member paths: GenBuster-200K/{train,test,benchmark}/{real,fake}/<generator>/<file>.mp4
"""
from .. import hosts
from ..archive import Archive, Meta, is_video

NAME = "genbuster200k"
REPO = "l8cv/GenBuster-200K"


def classify(name):
    if not is_video(name):
        return None
    parts = name.split("/")
    if len(parts) < 4:
        return None
    split, label = parts[1], parts[2]
    gen = "real" if label == "real" else (parts[3] if len(parts) > 4 else "unknown")
    return Meta(label, gen, split)


def archives():
    files = hosts.hf_files(REPO)
    vols = sorted(p for p in files if p.startswith("GenBuster-200K.7z."))
    return [Archive(NAME, "GenBuster-200K", "7z",
                    [hosts.hf_url(REPO, p) for p in vols], [files[p] for p in vols], classify)]
