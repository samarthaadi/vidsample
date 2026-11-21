"""Remote 7z access. Header parsing is delegated to py7zr (reading only the header bytes).

Members of folders that use the Copy coder are stored verbatim, so they are fetched with a
single range read. Members of compressed folders get method=-1 and can only be streamed
(see stream_reader.iter_7z).
"""
import py7zr
import py7zr.archiveinfo as _ai

from .entry import Entry

_ai.MAX_NUMSTREAMS = 1 << 28   # GenBuster-200K has ~206k pack streams (one per file)

_SIG_HEADER = 32
_COPY = b"\x00"


def open_archive(f):
    return py7zr.SevenZipFile(f)


def list_entries(f):
    z = open_archive(f)
    ms = z.header.main_streams
    folders = ms.unpackinfo.folders
    packsizes = ms.packinfo.packsizes
    per_folder = ms.substreamsinfo.num_unpackstreams_folders
    unpacksizes = ms.substreamsinfo.unpacksizes

    # absolute start of each folder's first pack stream
    pack_off, pos = [], _SIG_HEADER + ms.packinfo.packpos
    for s in packsizes:
        pack_off.append(pos)
        pos += s
    folder_pack, pi = [], 0
    for fo in folders:
        folder_pack.append(pi)
        pi += len(fo.packed_indices) if getattr(fo, "packed_indices", None) else 1

    # walk substreams in order: (folder index, byte offset within folder)
    streams = []
    k = 0
    for fi, n in enumerate(per_folder):
        inner = 0
        for _ in range(n):
            streams.append((fi, inner, unpacksizes[k]))
            inner += unpacksizes[k]
            k += 1

    out, si = [], 0
    for info in z.header.files_info.files:
        if info.get("emptystream"):
            continue
        fi, inner, size = streams[si]
        si += 1
        name = info["filename"].replace("\\", "/")
        coders = folders[fi].coders
        if len(coders) == 1 and coders[0]["method"] == _COPY:
            out.append(Entry(name=name, size=size, offset=pack_off[folder_pack[fi]] + inner,
                             packed=size, method=0))
        else:
            out.append(Entry(name=name, size=size, offset=fi, packed=0, method=-1))
    z.close()
    return out


def read_entry(f, e):
    if e.method != 0:
        raise NotImplementedError(f"{e.name} is in a compressed 7z folder; stream it instead")
    return f.fetch(e.offset, e.size)
