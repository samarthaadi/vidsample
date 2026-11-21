"""Random access into a remote zip: read the central directory, then range-read single members."""
import struct
import zlib

from .entry import Entry

_EOCD = b"PK\x05\x06"
_EOCD64_LOC = b"PK\x06\x07"
_CDH = b"PK\x01\x02"


def list_entries(f):
    tail_len = min(f.size, 65536 + 22)
    tail = f.pread(f.size - tail_len, tail_len)
    i = tail.rfind(_EOCD)
    if i < 0:
        raise ValueError("no end-of-central-directory record")
    (_, _, _, _, n_total, cd_size, cd_off, _) = struct.unpack("<4sHHHHIIH", tail[i:i + 22])
    if 0xFFFFFFFF in (cd_size, cd_off) or n_total == 0xFFFF:
        loc = tail[i - 20:i]
        if loc[:4] != _EOCD64_LOC:
            raise ValueError("zip64 locator missing")
        eocd64_off = struct.unpack("<Q", loc[8:16])[0]
        rec = f.pread(eocd64_off, 56)
        n_total, cd_size, cd_off = struct.unpack("<QQQ", rec[32:56])

    cd = f.pread(cd_off, cd_size)
    out, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == _CDH:
        (method, csize, usize, nlen, xlen, clen, lho) = (
            struct.unpack("<H", cd[p + 10:p + 12])[0],
            *struct.unpack("<II", cd[p + 20:p + 28]),
            *struct.unpack("<HHH", cd[p + 28:p + 34]),
            struct.unpack("<I", cd[p + 42:p + 46])[0],
        )
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, lho):
            usize, csize, lho = _zip64_extra(extra, usize, csize, lho)
        p += 46 + nlen + xlen + clen
        if name.endswith("/"):
            continue
        out.append(Entry(name=name, size=usize, offset=lho, packed=csize, method=method))
    return out


def _zip64_extra(extra, usize, csize, lho):
    q = 0
    while q + 4 <= len(extra):
        hid, hlen = struct.unpack("<HH", extra[q:q + 4])
        body = extra[q + 4:q + 4 + hlen]
        if hid == 1:
            vals, k = [], 0
            for v in (usize, csize, lho):
                if v == 0xFFFFFFFF:
                    v = struct.unpack("<Q", body[k:k + 8])[0]
                    k += 8
                vals.append(v)
            return vals
        q += 4 + hlen
    return usize, csize, lho


def read_entry(f, e):
    # one request for local header + data; the local name/extra length is a guess, topped up below
    buf = f.fetch(e.offset, 30 + 1024 + e.packed)
    nlen, xlen = struct.unpack("<HH", buf[26:30])
    start = 30 + nlen + xlen
    raw = buf[start:start + e.packed]
    if len(raw) < e.packed:
        raw += f.fetch(e.offset + start + len(raw), e.packed - len(raw))
    if e.method == 0:
        return raw
    if e.method == 8:
        return zlib.decompressobj(-15).decompress(raw)
    raise NotImplementedError(f"zip method {e.method} for {e.name}")
