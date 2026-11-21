"""GenVideo-100K (DeMamba's lightweight GenVideo release) - ModelScope cccnju/GenVideo-100K.

Every generator is a tar.gz (large ones split into <Name>_part_aa, _ab, ...), so sampling
streams from the start of each archive and stops once its quota is filled. Members come in
shuffled order, so a prefix is a fair sample. I2VGEN_XL clips are GIFs.
Member paths look like ./train_OpenSora/OpenSora_610.mp4 or ./Real/real_51563.mp4.
"""
import re
from collections import defaultdict

from .. import hosts
from ..archive import Archive, Meta, is_video

NAME = "genvideo100k"
REPO = "cccnju/GenVideo-100K"


def _classify(gen, label):
    def classify(name):
        if not is_video(name):
            return None
        top = name.split("/")[0]
        m = re.match(r"(train|val|test)_", top, re.I)
        return Meta(label, gen, m.group(1).lower() if m else "unspecified")
    return classify


def archives():
    files = hosts.ms_files(REPO)
    groups = defaultdict(list)
    for p in sorted(files):
        m = re.match(r"(.+?)(\.tar\.gz|\.tgz|\.tar|_part_[a-z]{2})$", p)
        if m:
            groups[m.group(1)].append(p)
    out = []
    for base, parts in sorted(groups.items()):
        label = "real" if base.lower() == "real" else "fake"
        gen = "real" if label == "real" else base
        out.append(Archive(NAME, base, "tar_stream", [hosts.ms_url(REPO, p) for p in parts],
                           [files[p] for p in parts], _classify(gen, label),
                           stream_groups=[(label, gen)]))
    return out
