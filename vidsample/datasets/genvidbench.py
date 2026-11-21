"""GenVidBench (Ni et al., AAAI 2026) - HF jian-0/GenVidBench.

Pair1: ms, pika, t2vz, vc2 (fake) + vript (real)        non-solid RAR5, random access
Pair2: cogvideo, mora, musev, svd (fake)                non-solid RAR5, random access
       hd_vg_130m.7z.00x (real)                         one LZMA2 stream around a RAR5: stream only
6.7m:  OpenAI_Sora.zip, keling.zip                      zip, random access
       OpenSora_13800.tar.gz                            stream only
The paper's cross-source protocol trains on one pair and tests on the other, so the split
column is "pair1" / "pair2" / "extra" rather than train/test.
"""
from .. import hosts
from ..archive import Archive, Meta, is_video

NAME = "genvidbench"
REPO = "jian-0/GenVidBench"
ROOT = "GenVidBench/"
REAL = {"vript", "hd_vg_130m"}


def _folder_classifier(split, rename=None):
    def classify(name):
        if not is_video(name):
            return None
        folder = name.split("/")[0]
        gen = (rename or {}).get(folder, folder)
        return Meta("real" if folder in REAL else "fake", gen, split)
    return classify


def archives():
    files = hosts.hf_files(REPO)
    url = lambda p: hosts.hf_url(REPO, ROOT + p)
    size = lambda p: files[ROOT + p]
    out = []
    for pair, gens in (("Pair1", ["ms", "pika", "t2vz", "vc2", "vript"]),
                       ("Pair2", ["cogvideo", "mora", "musev", "svd"])):
        for g in gens:
            p = f"{pair}/{g}.rar"
            out.append(Archive(NAME, f"{pair}_{g}", "rar", [url(p)], [size(p)],
                               _folder_classifier(pair.lower())))

    vols = sorted(p[len(ROOT):] for p in files if p.startswith(ROOT + "Pair2/hd_vg_130m.7z."))
    out.append(Archive(NAME, "Pair2_hd_vg_130m", "rar_in_7z_stream", [url(p) for p in vols],
                       [size(p) for p in vols], _folder_classifier("pair2", {"hd_vg_130m": "hdvg"}),
                       stream_groups=[("real", "hdvg")], stream_split="pair2"))

    for p, gen in (("6.7m/OpenAI_Sora.zip", "sora"), ("6.7m/keling.zip", "kling")):
        out.append(Archive(NAME, f"extra_{gen}", "zip", [url(p)], [size(p)],
                           lambda n, gen=gen: Meta("fake", gen, "extra") if is_video(n) else None))
    p = "6.7m/OpenSora_13800.tar.gz"
    out.append(Archive(NAME, "extra_opensora", "tar_stream", [url(p)], [size(p)],
                       lambda n: Meta("fake", "opensora", "extra") if is_video(n) else None,
                       stream_groups=[("fake", "opensora")], stream_split="extra"))
    return out
