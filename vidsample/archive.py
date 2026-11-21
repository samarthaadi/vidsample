"""An archive inside a dataset, plus a cached listing of its members."""
import json
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from . import hosts
from .formats import rar5_remote, sevenz_remote, stream_reader, zip_remote
from .formats.entry import Entry
from .remote import HTTPRangeFile, MultiPartFile

VIDEO_EXT = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".gif")

RANDOM_ACCESS = {"zip": zip_remote, "7z": sevenz_remote, "rar": rar5_remote}
STREAM = {"tar_stream": stream_reader.iter_tar, "rar_in_7z_stream": stream_reader.iter_rar_in_7z}


@dataclass(frozen=True)
class Meta:
    label: str          # "real" | "fake"
    generator: str
    split: str


@dataclass
class Archive:
    dataset: str
    key: str                        # short unique name inside the dataset
    kind: str                       # zip | 7z | rar | tar_stream | rar_in_7z_stream
    urls: List[str]
    sizes: List[int]
    classify: Callable[[str], Optional[Meta]]
    # stream archives only: what the archive contains, for quota planning
    stream_groups: List[tuple] = field(default_factory=list)   # [(label, generator)]
    stream_split: Optional[str] = None                          # if every member shares a split

    @property
    def random_access(self):
        return self.kind in RANDOM_ACCESS

    def open(self):
        parts = [HTTPRangeFile(u, s) for u, s in zip(self.urls, self.sizes)]
        return parts[0] if len(parts) == 1 else MultiPartFile(parts)

    def entries(self):
        """Member listing, cached under index/<dataset>/<key>.jsonl."""
        path = os.path.join(hosts.INDEX_DIR, self.dataset, self.key + ".jsonl")
        if os.path.exists(path):
            with open(path) as fh:
                return [Entry(**json.loads(line)) for line in fh]
        es = RANDOM_ACCESS[self.kind].list_entries(self.open())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            for e in es:
                fh.write(json.dumps(e.to_dict()) + "\n")
        os.replace(tmp, path)
        return es

    def read(self, f, entry):
        return RANDOM_ACCESS[self.kind].read_entry(f, entry)

    def stream(self, max_bytes=None):
        return STREAM[self.kind](self.urls, max_bytes=max_bytes)


def is_video(name):
    base = name.rsplit("/", 1)[-1]
    return not base.startswith(".") and "__MACOSX" not in name and base.lower().endswith(VIDEO_EXT)
