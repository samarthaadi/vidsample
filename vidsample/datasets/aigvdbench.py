"""AIGVDBench (Ma et al., CVPR 2026) - HF AIGVDBench/AIGVDBench.

One zip per generator under AIGVDBench/{OpenSource/{T2V,I2V,V2V},ClosedSource,Real}/.
Train/val/test membership comes from Split/*.jsonl (keyed by Video_id, shared by every
generator). Closed-source generators are test-only.
"""
import json

from .. import hosts
from ..archive import Archive, Meta, is_video
from ..remote import HTTPRangeFile

NAME = "aigvdbench"
REPO = "AIGVDBench/AIGVDBench"


def _split_map(files):
    m = {}
    for split in ("train", "val", "test"):
        path = f"Split/{split}.jsonl"
        if path not in files:
            continue
        f = HTTPRangeFile(hosts.hf_url(REPO, path), files[path])
        for line in f.read().decode().splitlines():
            if line.strip():
                m[json.loads(line)["Video_id"]] = split
    return m


def archives():
    files = hosts.hf_files(REPO)
    splits = _split_map(files)
    out = []
    for path, size in sorted(files.items()):
        if not (path.startswith("AIGVDBench/") and path.endswith(".zip")):
            continue
        parts = path[:-4].split("/")           # AIGVDBench/OpenSource/T2V/LTX
        if parts[1] == "Real":
            label, gen, fixed = "real", "real", None
        elif parts[1] == "ClosedSource":
            label, gen, fixed = "fake", parts[2], "test"
        else:
            label, gen, fixed = "fake", f"{parts[3]}_{parts[2]}", None

        def classify(name, label=label, gen=gen, fixed=fixed):
            if not is_video(name):
                return None
            split = fixed or splits.get(name.rsplit("/", 1)[-1], "unspecified")
            return Meta(label, gen, split)

        out.append(Archive(NAME, f"{label}_{gen}", "zip", [hosts.hf_url(REPO, path)], [size], classify))
    return out
