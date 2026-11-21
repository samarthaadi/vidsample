"""Remote RAR5 access for non-solid archives (GenVidBench).

Listing uses the archive's quick-open record (a cached copy of every file header near the end of
the archive), falling back to hopping from header to header. A single member is extracted by
range-reading its header and packed data, wrapping them in a minimal one-file RAR5 archive and
decompressing it with 7-Zip.
"""
import os
import shutil
import subprocess
import tempfile
import zlib

from .entry import Entry

SIG = b"Rar!\x1a\x07\x01\x00"
T_MAIN, T_FILE, T_SERVICE, T_END = 1, 2, 3, 5


def _vint(b, i):
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return r, i


def _enc_vint(n):
    out = bytearray()
    while True:
        c = n & 0x7F
        n >>= 7
        out.append(c | (0x80 if n else 0))
        if not n:
            return bytes(out)


def parse_header(b, p=0):
    """Parse a block header at b[p:]. Returns dict with total header length and fields."""
    hsize, q = _vint(b, p + 4)
    body_start = q
    end = body_start + hsize
    typ, q = _vint(b, q)
    flags, q = _vint(b, q)
    extra = data = 0
    if flags & 1:
        extra, q = _vint(b, q)
    if flags & 2:
        data, q = _vint(b, q)
    h = {"type": typ, "flags": flags, "hdr_len": end - p, "data_size": data, "end": end}
    if typ == T_MAIN:
        h["arch_flags"], q = _vint(b, q)
        h["extra"] = b[q:end]
    elif typ in (T_FILE, T_SERVICE):
        ff, q = _vint(b, q)
        h["unpacked"], q = _vint(b, q)
        _attr, q = _vint(b, q)
        if ff & 2:
            q += 4
        if ff & 4:
            q += 4
        comp, q = _vint(b, q)
        _host, q = _vint(b, q)
        nlen, q = _vint(b, q)
        h["name"] = b[q:q + nlen].decode("utf-8", "replace")
        h["is_dir"] = bool(ff & 1)
        h["solid"] = bool(comp & 0x40)
        h["method"] = (comp >> 7) & 7
        h["split"] = bool(flags & 0x18)
    return h


def _main_header(f):
    b = f.pread(0, 256)
    if b[:8] != SIG:
        raise ValueError("not a RAR5 archive")
    h = parse_header(b, 8)
    if h["arch_flags"] & 0x05:
        raise NotImplementedError("solid or multi-volume RAR5 is not randomly accessible")
    return h, b


def _quick_open_offset(main):
    ex, q = main["extra"], 0
    while q < len(ex):
        size, r = _vint(ex, q)
        typ, s = _vint(ex, r)
        if typ == 1:                      # locator record
            fl, s = _vint(ex, s)
            if fl & 1:
                off, s = _vint(ex, s)
                return off or None
        q = r + size
    return None


def _entry(h, hdr_off):
    return Entry(name=h["name"], size=h["unpacked"], offset=hdr_off,
                 packed=h["hdr_len"] + h["data_size"], method=h["method"])


def list_entries(f):
    main, _ = _main_header(f)
    first = 8 + main["hdr_len"]
    qo = _quick_open_offset(main)
    if qo:
        try:
            return _list_quick_open(f, 8 + qo)
        except Exception:
            pass
    return _list_hopping(f, first)


def _list_quick_open(f, qo_start):
    hb = f.pread(qo_start, 512)
    h = parse_header(hb)
    if h["type"] != T_SERVICE or h.get("name") != "QO" or h["method"] != 0:
        raise ValueError("no usable quick-open record")
    data = f.pread(qo_start + h["hdr_len"], h["data_size"])
    out, p = [], 0
    while p < len(data):
        size, q = _vint(data, p + 4)
        rec_end = q + size
        _fl, q = _vint(data, q)
        back, q = _vint(data, q)
        dsize, q = _vint(data, q)
        cached = data[q:q + dsize]
        p = rec_end
        ch = parse_header(cached)
        if ch["type"] == T_FILE and not ch["is_dir"]:
            out.append(_entry(ch, qo_start - back))
    if not out:
        raise ValueError("empty quick-open record")
    return out


def _list_hopping(f, pos):
    out = []
    while pos < f.size:
        b = f.pread(pos, 4096)
        h = parse_header(b)
        if h["type"] == T_END:
            break
        if h["type"] == T_FILE and not h["is_dir"]:
            out.append(_entry(h, pos))
        pos += h["hdr_len"] + h["data_size"]
    return out


def _block(typ, body):
    """Header block with CRC: crc32(size vint + type + flags + body)."""
    payload = _enc_vint(typ) + b"\x00" + body
    sized = _enc_vint(len(payload)) + payload
    return zlib.crc32(sized).to_bytes(4, "little") + sized


_MINI_MAIN = _block(T_MAIN, b"\x00")
_MINI_END = _block(T_END, b"\x00")


def sevenzip_bin():
    for name in ("7zz", "7z", "7za"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError("RAR5 extraction needs 7-Zip: apt install 7zip (or p7zip-full p7zip-rar)")


def decompress_member(header_and_data):
    """Decompress one RAR5 file header + packed data, given as raw bytes."""
    h = parse_header(header_and_data)
    if h["method"] == 0:
        return header_and_data[h["hdr_len"]:h["hdr_len"] + h["data_size"]]
    if h["solid"]:
        raise NotImplementedError("solid RAR member")
    with tempfile.TemporaryDirectory() as td:
        arc = os.path.join(td, "m.rar")
        with open(arc, "wb") as fh:
            fh.write(SIG + _MINI_MAIN + header_and_data + _MINI_END)
        r = subprocess.run([sevenzip_bin(), "e", "-so", "-y", arc], capture_output=True)
        if r.returncode != 0 or len(r.stdout) != h["unpacked"]:
            raise IOError(f"7z failed on {h['name']}: {r.stderr.decode()[-300:]}")
        return r.stdout


def read_entry(f, e):
    return decompress_member(f.fetch(e.offset, e.packed))


def iter_stream(fh):
    """Walk a forward-only RAR5 stream. Yields (header, read_member) pairs, where
    read_member() returns the decompressed bytes. If it is not called, the data is skipped."""
    if _read_exact(fh, 8) != SIG:
        raise ValueError("not a RAR5 stream")
    while True:
        pre = _read_exact(fh, 7)
        if len(pre) < 7:
            return
        hsize, q = _vint(pre, 4)                    # header size vint is 1..3 bytes
        raw = pre + _read_exact(fh, q + hsize - len(pre))
        h = parse_header(raw)
        if h["type"] == T_END:
            return
        state = {"done": False}

        def read_member(raw=raw, h=h, state=state):
            state["done"] = True
            data = _read_exact(fh, h["data_size"])
            if len(data) < h["data_size"]:
                raise EOFError(f"stream ended inside {h['name']}")
            return decompress_member(raw + data)

        if h["type"] == T_FILE and not h["is_dir"]:
            yield h, read_member
        if not state["done"]:
            _skip(fh, h["data_size"])


def _read_exact(fh, n):
    out = bytearray()
    while len(out) < n:
        chunk = fh.read(n - len(out))
        if not chunk:
            break
        out += chunk
    return bytes(out)


def _skip(fh, n):
    while n > 0:
        chunk = fh.read(min(n, 1 << 20))
        if not chunk:
            return
        n -= len(chunk)
