"""Forward-only readers for archives that cannot be randomly accessed.

Each iterator yields (name, size, read) where read() returns the member bytes. Members
whose read() is not called are skipped (their bytes still cross the network). Stop
iterating as soon as you have enough; `max_bytes` caps the compressed bytes pulled.
"""
import io
import lzma
import tarfile

from ..remote import SequentialHTTPStream
from . import rar5_remote


def iter_tar(urls, max_bytes=None):
    """tar, tar.gz or split tar.gz (part_aa, part_ab ... concatenated)."""
    stream = SequentialHTTPStream(urls, max_bytes=max_bytes)
    try:
        with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|*") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                yield m.name.lstrip("./"), m.size, (lambda m=m: tf.extractfile(m).read())
    except (tarfile.ReadError, EOFError):
        if stream.max_bytes is None or stream.read_total < stream.max_bytes:
            raise                       # truncation we did not ask for


class _LZMA2Reader(io.RawIOBase):
    def __init__(self, raw, dict_size):
        self.raw = raw
        self.d = lzma.LZMADecompressor(lzma.FORMAT_RAW,
                                       filters=[{"id": lzma.FILTER_LZMA2, "dict_size": dict_size}])
        self.buf = b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            if self.d.eof:
                return 0
            if self.d.needs_input:
                chunk = self.raw.read(1 << 20)
                if not chunk:
                    return 0
            else:
                chunk = b""
            self.buf = self.d.decompress(chunk, max_length=1 << 22)
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.buf = self.buf[n:]
        return n


def iter_rar_in_7z(urls, max_bytes=None, dict_prop=0x18):
    """A 7z whose single LZMA2 folder holds one non-solid RAR5 (GenVidBench HD-VG).

    The LZMA2 stream starts right after the 32-byte signature header (pack pos 0).
    """
    dict_size = (2 | (dict_prop & 1)) << (dict_prop // 2 + 11)
    raw = io.BufferedReader(SequentialHTTPStream(urls, max_bytes=max_bytes), 1 << 20)
    raw.read(32)
    fh = io.BufferedReader(_LZMA2Reader(raw, dict_size), 1 << 20)
    try:
        for h, read in rar5_remote.iter_stream(fh):
            yield h["name"], h["unpacked"], read
    except (lzma.LZMAError, IndexError, ValueError, EOFError):
        if max_bytes is None:
            raise                       # cut mid-header by the byte cap
