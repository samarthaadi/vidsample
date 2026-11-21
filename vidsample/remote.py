"""Seekable, read-only file objects backed by HTTP Range requests.

Nothing is written to disk: bytes are fetched on demand and a small in-memory
block cache absorbs the many tiny reads archive parsers make.
"""
import io
import os
import threading
import time
from collections import OrderedDict

import requests

BLOCK = 1 << 20          # readahead granularity for small reads
CACHE_BLOCKS = 16        # per-file LRU of BLOCK-sized chunks
_RETRIES = 5

_stats_lock = threading.Lock()
STATS = {"bytes": 0, "requests": 0}

_local = threading.local()


def session():
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "vidsample/0.1"
        tok = os.environ.get("HF_TOKEN")
        if tok:
            s.headers["Authorization"] = f"Bearer {tok}"
        _local.session = s
    return s


def _count(n):
    with _stats_lock:
        STATS["bytes"] += n
        STATS["requests"] += 1


class HTTPRangeFile(io.RawIOBase):
    """A seekable file over one URL. `size` may be passed to skip a probe request."""

    def __init__(self, url, size=None):
        self.url = url
        self._resolved = None        # signed CDN URL after redirects (expires)
        self._size = size
        self._pos = 0
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    # -- HTTP -------------------------------------------------------------
    def _get(self, start, end):
        """Fetch bytes [start, end] inclusive."""
        last = None
        for attempt in range(_RETRIES):
            url = self._resolved or self.url
            try:
                r = session().get(url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=60, allow_redirects=True)
                if r.status_code in (401, 403, 404, 410) and self._resolved:
                    self._resolved = None    # signed URL expired; resolve again
                    continue
                if r.status_code == 200 and (start != 0 or end + 1 < len(r.content)):
                    raise IOError(f"server ignored Range header: {self.url}")
                r.raise_for_status()
                if r.url != url:
                    self._resolved = r.url
                if self._size is None:
                    cr = r.headers.get("Content-Range", "")
                    if "/" in cr and not cr.endswith("/*"):
                        self._size = int(cr.rsplit("/", 1)[1])
                data = r.content
                _count(len(data))
                return data
            except (requests.ConnectionError, requests.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                last = e
                time.sleep(2 ** attempt)
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 500, 502, 503, 504):
                    last = e
                    time.sleep(2 ** (attempt + 1))
                    continue
                raise
        raise IOError(f"giving up on {self.url} [{start}-{end}]: {last}")

    @property
    def size(self):
        if self._size is None:
            self._get(0, 0)
        return self._size

    def pread(self, offset, n):
        """Read n bytes at offset (thread-safe, does not move the file pointer)."""
        n = max(0, min(n, self.size - offset))
        if n == 0:
            return b""
        if n > BLOCK * 2:
            return self._get(offset, offset + n - 1)
        out = bytearray()
        while n:
            bi, bo = divmod(offset, BLOCK)
            blk = self._block(bi)
            chunk = blk[bo:bo + n]
            if not chunk:
                break
            out += chunk
            offset += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def fetch(self, offset, n):
        """Uncached exact read, for pulling whole members in one request."""
        n = max(0, min(n, self.size - offset))
        return self._get(offset, offset + n - 1) if n else b""

    def _block(self, bi):
        with self._lock:
            if bi in self._cache:
                self._cache.move_to_end(bi)
                return self._cache[bi]
        start = bi * BLOCK
        data = self._get(start, min(start + BLOCK, self.size) - 1)
        with self._lock:
            self._cache[bi] = data
            while len(self._cache) > CACHE_BLOCKS:
                self._cache.popitem(last=False)
        return data

    # -- io.RawIOBase -----------------------------------------------------
    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        if whence == 0:
            self._pos = off
        elif whence == 1:
            self._pos += off
        else:
            self._pos = self.size + off
        return self._pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self._pos
        data = self.pread(self._pos, n)
        self._pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class MultiPartFile(HTTPRangeFile):
    """Concatenation of several remote files (split archives: .7z.001, part_aa ...)."""

    def __init__(self, parts):
        self.parts = parts
        self._offsets = []
        total = 0
        for p in parts:
            self._offsets.append(total)
            total += p.size
        self._size = total
        self._pos = 0
        self.url = parts[0].url

    @property
    def size(self):
        return self._size

    def pread(self, offset, n):
        n = max(0, min(n, self._size - offset))
        out = bytearray()
        i = 0
        while i + 1 < len(self.parts) and self._offsets[i + 1] <= offset:
            i += 1
        while n and i < len(self.parts):
            local = offset - self._offsets[i]
            chunk = self.parts[i].pread(local, n)
            if not chunk:
                i += 1
                continue
            out += chunk
            offset += len(chunk)
            n -= len(chunk)
            if offset >= self._offsets[i] + self.parts[i].size:
                i += 1
        return bytes(out)


    def fetch(self, offset, n):
        out = bytearray()
        for i, p in enumerate(self.parts):
            lo, hi = self._offsets[i], self._offsets[i] + p.size
            if offset + n <= lo or offset >= hi:
                continue
            a = max(offset, lo)
            out += p.fetch(a - lo, min(offset + n, hi) - a)
        return bytes(out)


class SequentialHTTPStream(io.RawIOBase):
    """Forward-only streaming reader over one or more URLs (for tar.gz / solid 7z).

    Uses a single long-lived GET per part rather than many Range requests.
    `max_bytes` stops the stream early to cap bandwidth.
    """

    def __init__(self, urls, max_bytes=None, chunk=1 << 20):
        self.urls = list(urls)
        self.max_bytes = max_bytes
        self.chunk = chunk
        self.read_total = 0
        self._it = None
        self._buf = b""
        self._part = -1

    def readable(self):
        return True

    def _next_part(self):
        self._part += 1
        if self._part >= len(self.urls):
            self._it = None
            return False
        r = session().get(self.urls[self._part], stream=True, timeout=120)
        r.raise_for_status()
        self._it = r.iter_content(self.chunk)
        return True

    def readinto(self, b):
        if self.max_bytes is not None and self.read_total >= self.max_bytes:
            return 0
        while not self._buf:
            if self._it is None and not self._next_part():
                return 0
            try:
                self._buf = next(self._it)
                with _stats_lock:
                    STATS["bytes"] += len(self._buf)
            except StopIteration:
                if not self._next_part():
                    return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        self.read_total += n
        return n
