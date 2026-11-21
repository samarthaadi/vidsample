"""Plan balanced per-generator samples, fetch only those clips, write clips + manifest.csv."""
import csv
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import normalize as norm
from .remote import STATS

MIN_BYTES = 1024          # some source archives contain 0-byte placeholder clips
MANIFEST_FIELDS = ["path", "dataset", "label", "generator", "split", "archive", "member", "bytes",
                   "src_bytes", "norm"]


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _safe(name):
    return re.sub(r"[^\w.\-]+", "_", name.rsplit("/", 1)[-1]).strip("_") or "clip"


def candidates(archives, splits=None, max_bytes=None):
    """{(label, gen): [(archive, entry, meta)]} for random-access archives, with filters."""
    groups = defaultdict(list)
    for a in archives:
        if not a.random_access:
            continue
        for e in a.entries():
            m = a.classify(e.name)
            if m is None or (splits and m.split not in splits):
                continue
            if e.size < MIN_BYTES or (max_bytes and e.size > max_bytes):
                continue
            groups[(m.label, m.generator)].append((a, e, m))
    return groups


def _fill(total, avail):
    """Split `total` across groups as evenly as availability allows. avail None = unbounded."""
    quota = {g: 0 for g in avail}
    open_ = [g for g in avail if avail[g] is None or avail[g] > 0]
    left = total
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for g in open_:
            if left <= 0:
                break
            cap = share if avail[g] is None else min(share, avail[g] - quota[g])
            quota[g] += cap
            left -= cap
            if avail[g] is None or quota[g] < avail[g]:
                nxt.append(g)
        open_ = nxt
    return quota


def plan(archives, per_generator, real=None, splits=None, max_bytes=None, use_streams=True):
    groups = candidates(archives, splits, max_bytes)
    avail = {g: len(v) for g, v in groups.items()}
    if use_streams:
        for a in archives:
            if a.random_access or (splits and a.stream_split and a.stream_split not in splits):
                continue
            for g in a.stream_groups:
                avail.setdefault(g, None)
    fake = {g: (per_generator if n is None else min(per_generator, n))
            for g, n in avail.items() if g[0] == "fake"}
    real_avail = {g: n for g, n in avail.items() if g[0] == "real"}
    real_total = sum(fake.values()) if real is None else real
    quotas = {**fake, **_fill(real_total, real_avail)}
    return groups, avail, quotas


def decodes(path):
    """True if ffprobe can read at least one video packet."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() not in ("", "0", "N/A")


class Writer:
    def __init__(self, out, verify=False, spec=None):
        if verify and not shutil.which("ffprobe"):
            raise RuntimeError("--verify needs ffprobe (apt install ffmpeg)")
        if spec and not norm.available():
            raise RuntimeError("--normalize needs ffmpeg (apt install ffmpeg)")
        self.verify = verify
        self.spec = spec
        self.tag = str(spec) if spec else ""
        self.out = out
        self.path = os.path.join(out, "manifest.csv")
        self.rows = {}
        self.lock = threading.Lock()
        if os.path.exists(self.path):
            with open(self.path) as fh:
                for r in csv.DictReader(fh):
                    self.rows[r["path"]] = r

    def target(self, archive, meta, member):
        name = _safe(member)
        if self.spec:
            name = os.path.splitext(name)[0] + ".mp4"
        rel = os.path.join(archive.dataset, meta.label, meta.generator, name)
        return rel, os.path.join(self.out, rel)

    def done(self, rel, src_size):
        """Already written from the same source member with the same normalization."""
        full = os.path.join(self.out, rel)
        row = self.rows.get(rel)
        if not os.path.exists(full) or row is None:
            return False
        if row.get("norm", "") != self.tag:
            return False
        return int(row.get("src_bytes") or row["bytes"]) == src_size

    def write(self, rel, data, archive, meta, member):
        full = os.path.join(self.out, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        tmp = full + ".part"
        try:
            if self.spec:
                src = full + ".src"
                with open(src, "wb") as fh:
                    fh.write(data)
                try:
                    norm.normalize(src, tmp, self.spec)
                except ValueError as err:
                    raise ValueError(f"cannot normalize {member}: {err}") from None
                finally:
                    os.remove(src)
            else:
                with open(tmp, "wb") as fh:
                    fh.write(data)
            if self.verify and not decodes(tmp):
                raise ValueError(f"undecodable clip {member}")
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        os.replace(tmp, full)
        self.record(rel, archive, meta, member, len(data), os.path.getsize(full))

    def record(self, rel, archive, meta, member, src_size, size=None):
        with self.lock:
            self.rows[rel] = {"path": rel, "dataset": archive.dataset, "label": meta.label,
                              "generator": meta.generator, "split": meta.split,
                              "archive": archive.key, "member": member,
                              "bytes": src_size if size is None else size,
                              "src_bytes": src_size, "norm": self.tag}

    def save(self):
        with self.lock:
            os.makedirs(self.out, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, MANIFEST_FIELDS)
                w.writeheader()
                for k in sorted(self.rows):
                    w.writerow(self.rows[k])
            os.replace(tmp, self.path)


def run(archives, out, per_generator, real=None, splits=None, seed=0, workers=8,
        max_clip_mb=None, max_stream_gb=2.0, stream_keep_prob=1.0, use_streams=True, dry_run=False,
        verify=False, normalize=None):
    max_bytes = int(max_clip_mb * 2**20) if max_clip_mb else None
    groups, avail, quotas = plan(archives, per_generator, real, splits, max_bytes, use_streams)

    _log(f"{'label':5} {'generator':28} {'available':>10} {'quota':>6}")
    for g in sorted(quotas):
        n = avail[g]
        _log(f"{g[0]:5} {g[1]:28} {'stream' if n is None else n:>10} {quotas[g]:>6}")
    if dry_run:
        return

    writer = Writer(out, verify, normalize)
    rng = random.Random(seed)
    jobs = []
    for g in sorted(groups):
        pool = sorted(groups[g], key=lambda t: (t[0].key, t[1].name))
        jobs += rng.sample(pool, quotas.get(g, 0))

    handles, t0, b0 = {}, time.time(), STATS["bytes"]
    for a, _, _ in jobs:
        if a.key not in handles:
            handles[a.key] = a.open()

    def fetch(job):
        a, e, m = job
        rel, _ = writer.target(a, m, e.name)
        if writer.done(rel, e.size):
            return "skip"
        data = a.read(handles[a.key], e)
        if len(data) != e.size:
            raise IOError(f"size mismatch for {e.name}: {len(data)} != {e.size}")
        writer.write(rel, data, a, m, e.name)
        return "ok"

    status = Counter()
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fetch, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                status[f.result()] += 1
            except Exception as err:          # keep going; report at the end
                status["error"] += 1
                _log(f"  ! {futs[f][1].name}: {err}")
            if i % 50 == 0 or i == len(jobs):
                _log(f"  {i}/{len(jobs)} clips  {(STATS['bytes'] - b0) / 2**20:.0f} MiB fetched  "
                     f"{time.time() - t0:.0f}s")
                writer.save()

    if use_streams:
        for a in archives:
            if a.random_access:
                continue
            if splits and a.stream_split and a.stream_split not in splits:
                continue
            need = {g: quotas.get(g, 0) for g in a.stream_groups}
            if sum(need.values()) > 0:
                status.update(_run_stream(a, need, writer, splits, max_bytes,
                                          max_stream_gb, stream_keep_prob, rng))
    writer.save()
    _log(f"done: {dict(status)}  fetched {(STATS['bytes'] - b0) / 2**20:.0f} MiB in {time.time() - t0:.0f}s"
         f"  -> {writer.path}")


def _run_stream(a, need, writer, splits, max_bytes, max_stream_gb, keep_prob, rng):
    cap = int(max_stream_gb * 2**30) if max_stream_gb else None
    status = Counter()
    _log(f"  streaming {a.key} (cap {max_stream_gb} GB) for {need}")
    start, split_misses = STATS["bytes"], 0
    try:
        for name, size, read in a.stream(max_bytes=cap):
            m = a.classify(name)
            g = m and (m.label, m.generator)
            if m and splits and m.split not in splits:
                split_misses += 1
                if split_misses >= 50 and not status:
                    _log(f"  {a.key}: no members in split {sorted(splits)}, skipping")
                    return status
                continue
            if (m is None or need.get(g, 0) <= 0
                    or size < MIN_BYTES or (max_bytes and size > max_bytes)
                    or rng.random() > keep_prob):
                continue
            rel, _ = writer.target(a, m, name)
            if writer.done(rel, size):
                status["skip"] += 1
            else:
                try:
                    writer.write(rel, read(), a, m, name)
                except ValueError as err:     # failed --verify; keep streaming for a replacement
                    _log(f"  ! {err}")
                    status["bad"] += 1
                    continue
                status["ok"] += 1
            need[g] -= 1
            if all(v <= 0 for v in need.values()):
                break
    except (EOFError, tarfile.ReadError, IOError) as err:
        if cap is None or STATS["bytes"] - start < cap * 0.95:
            _log(f"  ! {a.key}: {err}")
            status["error"] += 1
    short = {g: v for g, v in need.items() if v > 0}
    if short:
        _log(f"  {a.key}: stream ended, still short {short} (raise --max-stream-gb)")
    return status
