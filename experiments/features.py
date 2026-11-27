"""Extract per-clip features for the raw and normalized clips.

python experiments/features.py EXP_DIR [raw|norm ...]
  EXP_DIR/features/<variant>_meta.csv   container/stream metadata (ffprobe)
  EXP_DIR/features/<variant>_clip.npy   mean CLIP ViT-B/32 image embedding of 4 frames taken
                                        uniformly from the first 2 s (NaN row if undecodable)
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pandas as pd
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

sys.path.insert(0, os.path.dirname(__file__))
from common import load_clips  # noqa: E402

MODEL = "openai/clip-vit-base-patch32"
N_FRAMES = 4
WINDOW_S = 2.0


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                       capture_output=True, text=True)
    if r.returncode:
        return {}
    j = json.loads(r.stdout)
    v = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), None)
    if v is None:
        return {}
    fmt = j.get("format", {})
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    dur = float(fmt.get("duration") or v.get("duration") or 0)
    size = os.path.getsize(path)
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    return {
        "file_bytes": size, "duration": dur, "fps": fps, "width": w, "height": h,
        "aspect": w / h if h else 0.0, "bitrate": size * 8 / dur if dur else 0.0,
        "bits_per_pixel": size * 8 / (dur * fps * w * h) if dur and fps and w and h else 0.0,
        "has_audio": int(any(s.get("codec_type") == "audio" for s in j["streams"])),
        "codec": v.get("codec_name", "?"), "pix_fmt": v.get("pix_fmt", "?"),
        "container": fmt.get("format_name", "?").split(",")[0],
    }


def frames(path):
    """N_FRAMES RGB PIL images spread over the first WINDOW_S seconds."""
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        dur = c.duration / 1e6 if c.duration else WINDOW_S
        span = min(dur, WINDOW_S)
        targets = [(k + 0.5) / N_FRAMES * span for k in range(N_FRAMES)]
        got, seen = [], []
        for i, fr in enumerate(c.decode(s)):
            t = fr.time if fr.time is not None else i / float(s.average_rate or 8)
            seen.append(fr)
            while got.__len__() < N_FRAMES and t >= targets[len(got)]:
                got.append(fr.to_image().convert("RGB"))
            if len(got) == N_FRAMES or t > span + 1 or len(seen) > 400:
                break
        if len(got) < N_FRAMES and seen:          # short clip: spread over what we decoded
            idx = np.linspace(0, len(seen) - 1, N_FRAMES).round().astype(int)
            got = [seen[i].to_image().convert("RGB") for i in idx]
        return got


_worker = {}


def _init_worker():
    torch.set_num_threads(1)          # one thread per process: no oversubscription
    _worker["proc"] = CLIPImageProcessor.from_pretrained(MODEL)
    _worker["model"] = CLIPVisionModelWithProjection.from_pretrained(MODEL).eval()


def _embed(job):
    i, path = job
    try:
        imgs = frames(path)
        if not imgs:
            return i, None
        x = _worker["proc"](images=imgs, return_tensors="pt")["pixel_values"]
        with torch.inference_mode():
            emb = _worker["model"](pixel_values=x).image_embeds
        return i, torch.nn.functional.normalize(emb, dim=-1).mean(0).numpy()
    except Exception:
        return i, None


def clip_features(paths, out_path, procs=4):
    feats = None
    with mp.get_context("spawn").Pool(procs, initializer=_init_worker) as pool:
        for n, (i, e) in enumerate(pool.imap_unordered(_embed, list(enumerate(paths)), chunksize=8), 1):
            if e is not None:
                if feats is None:
                    feats = np.full((len(paths), e.shape[0]), np.nan, dtype=np.float32)
                feats[i] = e
            if n % 500 == 0:
                print(f"  clip features {n}/{len(paths)}", flush=True)
    np.save(out_path, feats)
    print(f"  {np.isnan(feats[:, 0]).sum()} clips undecodable")


def main(exp, variants):
    df = load_clips(exp)
    os.makedirs(os.path.join(exp, "features"), exist_ok=True)
    for var in variants:
        col = "path" if var == "raw" else "norm_path"
        paths = [os.path.join(exp, var, p) for p in df[col]]
        meta_csv = os.path.join(exp, "features", f"{var}_meta.csv")
        if not os.path.exists(meta_csv):
            with ThreadPoolExecutor(8) as ex:
                meta = list(ex.map(lambda p: probe(p) if os.path.exists(p) else {}, paths))
            pd.DataFrame(meta).assign(id=df["id"]).to_csv(meta_csv, index=False)
            print(f"{var}: metadata for {len(paths)} clips")
        npy = os.path.join(exp, "features", f"{var}_clip.npy")
        if not os.path.exists(npy):
            print(f"{var}: CLIP features")
            clip_features(paths, npy)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:] or ["raw", "norm"])
