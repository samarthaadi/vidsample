"""Build the PDF report: what vidsample is, and the experiment results.

python experiments/make_report.py EXP_DIR OUT_PDF

Every number is computed from files: EXP_DIR/RESULTS.json, the feature tables, the raw
manifest, the sampling/run logs, and the archive listings cached under ./index.
"""
import glob
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from common import NORM_SPEC, load_clips  # noqa: E402
from vidsample.datasets import DATASETS  # noqa: E402

DS = ["aigvdbench", "genbuster200k", "genvidbench", "genvideo100k"]
NAME = {"aigvdbench": "AIGVDBench", "genbuster200k": "GenBuster-200K",
        "genvidbench": "GenVidBench", "genvideo100k": "GenVideo-100K"}
REPO_URL = "https://github.com/samarthaadi/vidsample"

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
pdfmetrics.registerFont(TTFont("DV", f"{FONT_DIR}/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", f"{FONT_DIR}/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-M", f"{FONT_DIR}/DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.2, leading=13,
                      alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.8, leading=10.5, textColor=colors.HexColor("#444444"))
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=15, leading=19, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=9.8, leading=13, spaceBefore=6, spaceAfter=3)
TITLE = ParagraphStyle("title", parent=BODY, fontName="DV-B", fontSize=20, leading=25, spaceAfter=4)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=11, bulletIndent=2, spaceAfter=3)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.8, leading=10, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
CODE = ParagraphStyle("code", parent=BODY, fontName="DV-M", fontSize=7.8, leading=10.5,
                      backColor=colors.HexColor("#f3f3f3"), borderPadding=4, spaceBefore=3, spaceAfter=7)
ACCENT = colors.HexColor("#1f4e79")


def P(text, style=BODY):
    return Paragraph(text, style)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, header_rows=1, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(c, CELLB if r < header_rows else CELL)
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=header_rows)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
          ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dde6f0")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
          ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5)]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f6f8fa")))
    t.setStyle(TableStyle(st))
    return t


def f3(x):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def hanley_mcneil(a, n1, n0):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return 1.96 * math.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0))


# ---------------------------------------------------------------- data gathering
def gather(exp):
    R = json.load(open(os.path.join(exp, "RESULTS.json")))
    df = load_clips(exp)
    meta = {v: pd.read_csv(os.path.join(exp, "features", f"{v}_meta.csv")).sort_values("id").reset_index(drop=True)
            for v in ("raw", "norm")}
    raw = pd.concat([df[["dataset", "split", "label", "generator", "member"]], meta["raw"]], axis=1)

    # sampling log: per-call clips / MiB / seconds
    samp = [tuple(map(float, m)) for m in re.findall(r"'ok': (\d+)\}\s+fetched (\d+) MiB in (\d+)s",
                                                      open(os.path.join(exp, "sample.log")).read())]
    run_log = open(os.path.join(exp, "run.log")).read()
    run_real = re.search(r"real\s+(\d+)m([\d.]+)s", run_log)
    # first normalization pass: file mtimes of the normalized non-GenVideo clips
    mt = [os.path.getmtime(p) for p in glob.glob(os.path.join(exp, "norm", "*", "*", "*", "*.mp4"))
          if "/genvideo100k/" not in p]

    # dataset listings from the cached archive indexes (no network)
    listing = {}
    for ds in DS:
        mod = DATASETS[ds]
        kinds, n_real, n_fake, stream_gb, archives = {}, 0, 0, 0.0, mod.archives()
        for a in archives:
            kinds[a.kind] = kinds.get(a.kind, 0) + 1
            if a.random_access:
                for e in a.entries():
                    m = a.classify(e.name)
                    if m:
                        n_real += m.label == "real"
                        n_fake += m.label == "fake"
            else:
                stream_gb += sum(a.sizes) / 2**30
        listing[ds] = dict(kinds=kinds, real=n_real, fake=n_fake, stream_gb=stream_gb,
                           total_gb=sum(sum(a.sizes) for a in archives) / 2**30)
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    return dict(R=R, df=df, raw=raw, meta=meta, samp=samp, run_real=run_real, norm_mtimes=mt,
                listing=listing, commit=commit)


# ---------------------------------------------------------------- report
def build(exp, out_pdf):
    G = gather(exp)
    R, df, raw = G["R"], G["df"], G["raw"]
    W = R["within"]
    C = R["counts"]
    cross = R["cross"]
    single = R["single"]

    def within(ds, k, s="test", m="auc"):
        return W.get(f"{ds}|{k}", {}).get(s, {}).get(m)

    def ci(ds, k, s="test"):
        a = within(ds, k, s)
        if a is None:
            return None
        c = C[ds][s]
        return hanley_mcneil(a, c["fake"], c["real"])

    n_total = len(df)
    n_real, n_fake = int((df.label == "real").sum()), int((df.label == "fake").sum())
    loo = lambda k: [cross.get(f"{k}|all-but-{d}", {}).get(d) for d in DS]
    meta_loo, clipn_loo = loo("meta_raw"), loo("clip_norm")
    aig = raw[raw.dataset == "aigvdbench"]
    aig_fake_1920 = int(((aig.label == "fake") & (aig.width == 1920)).sum())
    aig_real_not_1920 = int(((aig.label == "real") & (aig.width != 1920)).sum())
    n_aig_fake = int((aig.label == "fake").sum())
    gb = raw[raw.dataset == "genbuster200k"].groupby(["split", "label"])["file_bytes"].median() / 1e3
    gvb_ood = R["per_gen"]["genvidbench|ood"]
    samp_clips = sum(s[0] for s in G["samp"])
    samp_mib = sum(s[1] for s in G["samp"])
    samp_s = sum(s[2] for s in G["samp"])
    run_min = (int(G["run_real"].group(1)) + float(G["run_real"].group(2)) / 60) if G["run_real"] else None
    norm_min = (max(G["norm_mtimes"]) - min(G["norm_mtimes"])) / 60 if G["norm_mtimes"] else None
    meta_norm_3 = [within(d, "meta_norm") for d in ("aigvdbench", "genvidbench", "genvideo100k")]
    clipn_within = [within(d, "clip_norm") for d in DS]
    pg_n = [v["n"] for g in R["per_gen"].values() for k, v in g.items() if k.startswith("fake|")]
    cross_into_aig = [cross[f"clip_norm|{d}"].get("aigvdbench") for d in DS if d != "aigvdbench"]

    story = []
    story += [P("vidsample", TITLE),
              P("Remote sampling of AI-generated-video detection datasets — what was built, "
                "and results of a shortcut test and a CPU baseline detector", H3),
              P(f"Repository: <a href='{REPO_URL}' color='#1f4e79'>{REPO_URL}</a> (commit {G['commit']}) · "
                f"generated {datetime.now(timezone.utc):%Y-%m-%d}", SMALL), Spacer(1, 6)]

    # ---------------- summary
    story.append(P("Summary", H2))
    story += bullets([
        "<b>Built:</b> a Python tool, <font name='DV-M'>vidsample</font>, that samples clips from four public "
        "AI-generated-video detection datasets (GenVidBench, GenBuster-200K, AIGVDBench, GenVideo-100K) "
        "<b>without downloading their archives</b>. It reads archive directories over HTTP range requests and "
        "fetches only the chosen clips, writing them with a <font name='DV-M'>manifest.csv</font>. An optional "
        "<font name='DV-M'>--normalize</font> flag re-encodes every clip to one format. A fifth requested source, "
        "VAE-rebuilt real clips, was dropped because no public release was found.",
        f"<b>Experiment data:</b> {n_total:,} sampled clips ({n_real:,} real, {n_fake:,} fake), "
        f"each also re-encoded with <font name='DV-M'>--normalize</font>.",
        f"<b>Shortcut:</b> on raw clips, a classifier given only file metadata (size, bitrate, resolution, fps, "
        f"duration, codec, audio) reaches test AUC {f3(within('aigvdbench','meta_raw'))} on AIGVDBench, "
        f"{f3(within('genvidbench','meta_raw'))} on GenVidBench and {f3(within('genvideo100k','meta_raw'))} on "
        f"GenVideo-100K. Trained on three datasets, it scores {f3(min(meta_loo))}–{f3(max(meta_loo))} on the "
        f"fourth. Normalizing lowers these three to {f3(min(meta_norm_3))}–{f3(max(meta_norm_3))}.",
        f"<b>GenBuster-200K</b> is format-matched, but its real clips are smaller than its fakes in the train "
        f"split (median {gb[('train','real')]:.0f} vs {gb[('train','fake')]:.0f} KB) and larger in test "
        f"({gb[('test','real')]:.0f} vs {gb[('test','fake')]:.0f} KB). So a metadata model trained on its train "
        f"split scores AUC {f3(within('genbuster200k','meta_raw'))} on test, worse than chance.",
        f"<b>Baseline detector</b> (frozen CLIP ViT-B/32 + logistic regression, normalized clips): within-dataset "
        f"test AUC {f3(min(clipn_within))}–{f3(max(clipn_within))} "
        f"(lowest: {NAME[DS[clipn_within.index(min(clipn_within))]]}). But leave-one-dataset-out AUC is only "
        f"{f3(min(clipn_loo))}–{f3(max(clipn_loo))}.",
    ])

    # ---------------- what was built
    story.append(P("1. What was built", H2))
    story.append(P("1.1 The tool", H3))
    story.append(P(
        "The datasets are distributed as archives of tens to hundreds of GB. <font name='DV-M'>vidsample</font> "
        "opens each remote archive as a seekable file backed by HTTP <i>Range</i> requests. It parses the "
        "archive's directory and caches the member list locally. It then plans a balanced sample (N clips per "
        "fake generator, the same total of real clips by default) and fetches only those byte ranges, in "
        "parallel. Archives that cannot be randomly accessed (gzip-compressed tar, and a solid LZMA2 7z) are "
        "streamed from the start. Streaming stops when the quota is filled or a byte cap is reached."))
    story.append(table([
        ["Component", "File", "What it does"],
        ["Remote file", "vidsample/remote.py", "Seekable HTTP-range file with block cache and retries; multi-volume "
         "concatenation; forward-only stream with a byte cap"],
        ["zip", "formats/zip_remote.py", "Reads the (zip64) central directory; extracts stored/deflated members"],
        ["7z", "formats/sevenz_remote.py", "Parses the header with py7zr; Copy-coder members become plain byte "
         "ranges"],
        ["RAR5", "formats/rar5_remote.py", "Lists members from the quick-open record (or header hopping); extracts "
         "one member by wrapping it in a minimal RAR5 and running 7-Zip"],
        ["Streams", "formats/stream_reader.py", "tar / tar.gz / split tar.gz; LZMA2 7z holding a RAR5 (GenVidBench "
         "HD-VG)"],
        ["Datasets", "vidsample/datasets/*.py", "Per-dataset archive lists and label / generator / split rules"],
        ["Sampler + CLI", "sampler.py, cli.py", "Balanced quotas, parallel fetch, resumable output, "
         "manifest.csv, --verify (ffprobe), --dry-run"],
        ["Normalization", "normalize.py", "--normalize: H.264 yuv420p mp4, S×S scale+centre-crop, fixed fps, "
         "max duration, no audio, metadata stripped"],
    ], [26, 42, 104]))

    story.append(P("1.2 Datasets supported", H3))
    L = G["listing"]
    kind_txt = lambda k: ", ".join(f"{n}× {t}" for t, n in sorted(k.items()))
    story.append(table([
        ["Dataset", "Host", "Archives", "Clips listed (random access)", "Stream-only", "Total size"],
        *[[NAME[d], {"genvidbench": "HF jian-0/GenVidBench", "genbuster200k": "HF l8cv/GenBuster-200K",
                     "aigvdbench": "HF AIGVDBench/AIGVDBench", "genvideo100k": "ModelScope cccnju/GenVideo-100K"}[d],
           kind_txt(L[d]["kinds"]),
           f"{L[d]['real']:,} real / {L[d]['fake']:,} fake" if L[d]["real"] + L[d]["fake"] else "none",
           f"{L[d]['stream_gb']:.1f} GB" if L[d]["stream_gb"] else "–", f"{L[d]['total_gb']:.0f} GB"]
          for d in DS],
    ], [26, 37, 33, 34, 20, 18]))
    story.append(P(
        "Archive kinds: zip and rar (non-solid RAR5) and 7z (Copy coder) are random access; tar_stream and "
        "rar_in_7z_stream are read from the start. Counts are clips that pass the label/generator rules (video "
        "files only).", SMALL))

    story.append(P("1.3 Data issues found while building it", H3))
    gv_gif = int(((raw.dataset == "genvideo100k") & raw.member.str.lower().str.endswith(".gif")).sum())
    story += bullets([
        "GenVideo-100K contains 0-byte mp4 members (seen in DynamicCrafter, SEINE and ZeroScope). The sampler "
        "now always skips members under 1 KB, and <font name='DV-M'>--verify</font> drops undecodable clips.",
        f"GenVideo-100K's I2VGEN_XL clips are GIFs ({gv_gif} of the sampled clips here); "
        "<font name='DV-M'>--normalize</font> turns them into mp4.",
        "GenVidBench's HD-VG real clips are one solid LZMA2 7z wrapping a single RAR5, so they can only be "
        "streamed. Individual clips can exceed 100 MB, so the experiments capped them at 40 MB "
        "(<font name='DV-M'>--max-clip-mb 40</font>).",
        "Real and fake clips differ in format in three of the four datasets: resolution, length, audio and "
        "bitrate. Section 3 measures how much a detector can learn from that alone.",
    ])

    # ---------------- experiment design
    story.append(P("2. Experiment design", H2))
    story.append(P(
        f"Clips were sampled with <font name='DV-M'>experiments/sample.sh</font> (seed 0): {int(samp_clips):,} "
        f"clips, {samp_mib/1024:.1f} GiB transferred in {samp_s/60:.1f} min of fetch time. Each clip was then "
        f"re-encoded locally with the normalization used in the experiments: {NORM_SPEC['size']}×"
        f"{NORM_SPEC['size']}, {NORM_SPEC['fps']} fps, at most {NORM_SPEC['max_seconds']} s, CRF "
        f"{NORM_SPEC['crf']}, no audio, metadata stripped."))
    story.append(table([
        ["Dataset", "train", "test (seen generators)", "ood (unseen generators / sources)"],
        ["AIGVDBench", "official train split, 20 open-source generators", "official test split, same generators",
         "closed-source generators (Sora, Kling, Gen3, …) + test reals"],
        ["GenBuster-200K", "train", "test", "benchmark (8 closed-source generators, own reals)"],
        ["GenVidBench", "70% of Pair1", "30% of Pair1", "Pair2 (other generators; HD-VG reals)"],
        ["GenVideo-100K", "70%", "30%", "–"],
    ], [28, 46, 46, 52]))
    story.append(Spacer(1, 4))
    story.append(table([
        ["Dataset", "train real / fake", "test real / fake", "ood real / fake"],
        *[[NAME[d]] + [f"{C[d][s]['real']} / {C[d][s]['fake']}" if s in C[d] else "–"
                       for s in ("train", "test", "ood")] for d in DS],
    ], [40, 44, 44, 44]))
    story.append(Spacer(1, 4))
    story += bullets([
        "<b>meta</b>: ffprobe features only (file size, duration, fps, width, height, aspect, bitrate, bits per "
        "pixel, has-audio, codec, pixel format, container) → gradient-boosted trees "
        "(<font name='DV-M'>HistGradientBoostingClassifier</font>).",
        "<b>CLIP</b>: frozen CLIP ViT-B/32 image embeddings of 4 frames spread over the first 2 s, L2-normalized "
        "and averaged → standardized logistic regression (C = 0.5).",
        "Each is run on raw clips and on normalized clips. The metric is ROC AUC. The ± values are approximate "
        "95% intervals (Hanley–McNeil). "
        + (f"On 4 CPUs, the first normalization pass (3,136 clips) spanned {norm_min:.0f} min. The final run "
           f"(normalizing the remaining 800 clips, feature extraction and evaluation) took {run_min:.0f} min."
           if norm_min and run_min else ""),
    ])

    # ---------------- results: shortcut
    story.append(PageBreak())
    story.append(P("3. Results: metadata shortcut", H2))
    story.append(P("Test AUC of the metadata-only classifier (ood AUC in brackets where there is an ood set):"))
    story.append(table([
        ["Dataset", "meta, raw clips", "meta, normalized", "best single feature, raw", "best single feature, normalized"],
        *[[NAME[d],
           f3(within(d, "meta_raw")) + (f" ({f3(within(d, 'meta_raw', 'ood'))})" if within(d, "meta_raw", "ood") is not None else ""),
           f3(within(d, "meta_norm")) + (f" ({f3(within(d, 'meta_norm', 'ood'))})" if within(d, "meta_norm", "ood") is not None else ""),
           f"{single[f'{d}|raw'][0]:.3f} ({single[f'{d}|raw'][1]})",
           f"{single[f'{d}|norm'][0]:.3f} ({single[f'{d}|norm'][1]})"] for d in DS],
    ], [30, 30, 30, 41, 41]))
    story.append(P("Single-feature AUC is measured on the training set, as max(AUC, 1 − AUC) of the raw feature value.", SMALL))

    story.append(P("Why: the formats differ (medians of the sampled raw clips)", H3))
    rows = [["Dataset", "label", "median size", "median duration", "median fps", "most common widths", "with audio"]]
    for d in DS:
        for lab in ("real", "fake"):
            s = raw[(raw.dataset == d) & (raw.label == lab)]
            widths = ", ".join(f"{int(w)} ({n})" for w, n in s.width.value_counts().head(3).items())
            rows.append([NAME[d] if lab == "real" else "", lab, f"{s.file_bytes.median()/1e3:,.0f} KB",
                         f"{s.duration.median():.1f} s", f"{s.fps.median():.1f}", widths,
                         f"{100*s.has_audio.mean():.0f}%"])
    story.append(table(rows, [28, 12, 22, 24, 18, 50, 18], zebra=False))
    story += bullets([
        f"AIGVDBench: {int((aig.label=='real').sum()) - aig_real_not_1920} of the {int((aig.label=='real').sum())} "
        f"real clips are 1920 px wide, against {aig_fake_1920} of the {n_aig_fake} fakes. So width alone separates them.",
        f"Trained on any three datasets, the metadata model scores {', '.join(f'{NAME[d]} {f3(x)}' for d, x in zip(DS, meta_loo))} "
        "on the held-out one. The shortcut transfers because reals are high-resolution, long and often have audio "
        "almost everywhere.",
    ])

    story.append(P("GenBuster-200K: format-matched, but real-clip bitrate flips between splits", H3))
    gbr = raw[raw.dataset == "genbuster200k"]
    fmt = gbr.groupby(["codec", "width", "height", "fps"]).size()
    rows = [["split", "real median size", "fake median size", "real smaller?"]]
    for sp in ("train", "test", "benchmark"):
        r_, f_ = gb[(sp, "real")], gb[(sp, "fake")]
        rows.append([sp, f"{r_:.0f} KB", f"{f_:.0f} KB", "yes" if r_ < f_ else "no"])
    story.append(KeepTogether([
        P(f"All {len(gbr)} sampled GenBuster clips share one format: "
          + "; ".join(f"{c} {w}×{h} {fp:g} fps ({n})" for (c, w, h, fp), n in fmt.items())
          + f". Median duration {gbr.duration.median():.2f} s."),
        table(rows, [30, 40, 40, 30]),
        P(f"The metadata model learns 'smaller = real' on train, then scores AUC {f3(within('genbuster200k','meta_raw'))} "
          f"on test and {f3(within('genbuster200k','meta_raw','ood'))} on the benchmark. That is well below 0.5, "
          "so the signal is still there but inverted. Any detector that picks up bitrate from GenBuster's train "
          "split will be systematically wrong on its test split."),
    ]))

    # ---------------- results: CLIP
    story.append(P("4. Results: CLIP baseline detector", H2))
    rows = [["Dataset", "CLIP raw: test", "CLIP raw: ood", "CLIP norm: test", "CLIP norm: ood"]]
    for d in DS:
        cells = []
        for k in ("clip_raw", "clip_norm"):
            for s in ("test", "ood"):
                a = within(d, k, s)
                cells.append("–" if a is None else f"{a:.3f} ± {ci(d, k, s):.3f}")
        rows.append([NAME[d]] + cells)
    story.append(table(rows, [30, 35, 35, 35, 35]))
    story.append(Spacer(1, 3))
    story.append(P(
        f"Normalizing lowers AIGVDBench from {f3(within('aigvdbench','clip_raw'))} to "
        f"{f3(within('aigvdbench','clip_norm'))} (intervals barely overlap). So part of the raw score came from "
        f"format differences. The other datasets change by at most "
        f"{max(abs(within(d,'clip_raw')-within(d,'clip_norm')) for d in DS if d!='aigvdbench'):.3f}. Within a "
        f"dataset the baseline holds on unseen generators (AIGVDBench closed-source "
        f"{f3(within('aigvdbench','clip_norm','ood'))}, GenBuster benchmark {f3(within('genbuster200k','clip_norm','ood'))}). "
        f"But it drops when the real source changes: GenVidBench Pair1 → Pair2 scores "
        f"{f3(within('genvidbench','clip_norm','ood'))}, catching MuseV fakes "
        f"{100*gvb_ood['fake|musev']['acc']:.0f}% and SVD fakes {100*gvb_ood['fake|svd']['acc']:.0f}% of the time "
        f"(n = {gvb_ood['fake|musev']['n']} each)."))

    for k, title in (("clip_norm", "CLIP, normalized clips"), ("clip_raw", "CLIP, raw clips"),
                     ("meta_raw", "metadata, raw clips")):
        rows = [["trained on ↓ / tested on →"] + [NAME[d] for d in DS]]
        for tr in DS + ["all"]:
            rows.append([NAME.get(tr, "all four")] + [f3(cross[f"{k}|{tr}"].get(d)) for d in DS])
        rows.append(["the other three"] + [f3(x) for x in loo(k)])
        story.append(KeepTogether([P(f"Cross-dataset AUC: {title}", H3), table(rows, [40, 33, 33, 33, 33])]))
    story.append(P(
        f"Cross-dataset generalization is the weak point. AIGVDBench is the hardest target from every other "
        f"dataset ({f3(min(cross_into_aig))}–{f3(max(cross_into_aig))}, normalized). Within-dataset scores near 1.0 "
        "(GenBuster, GenVideo-100K) suggest their reals and fakes also differ in content or style, not only in "
        "generation artifacts."))

    story.append(P("Per-generator accuracy (CLIP, normalized, threshold 0.5)", H3))
    rows = [["Dataset / set", "fakes (accuracy, n)", "reals"]]
    for key, g in R["per_gen"].items():
        d, s = key.split("|")
        fakes = ", ".join(f"{k.split('|')[1]} {v['acc']:.2f}" for k, v in g.items() if k.startswith("fake|"))
        ns = sorted({v["n"] for k, v in g.items() if k.startswith("fake|")})
        reals = ", ".join(f"{k.split('|')[1]} {v['acc']:.2f} (n={v['n']})" for k, v in g.items() if k.startswith("real|"))
        rows.append([f"{NAME[d]} – {s}", fakes + f"  [n = {'/'.join(map(str, ns))} per generator]", reals])
    story.append(table(rows, [32, 110, 30]))
    half = lambda n: 1.96 * math.sqrt(0.25 / n)
    story.append(P(f"With {min(pg_n)}–{max(pg_n)} clips per generator these accuracies carry up to roughly "
                   f"±{half(max(pg_n)):.2f}–{half(min(pg_n)):.2f}, so they are not suitable for ranking generators.", SMALL))

    cis = [ci(d, k, s) for d in DS for k in ("clip_raw", "clip_norm") for s in ("test", "ood")
           if within(d, k, s) is not None]

    # ---------------- caveats + reproduce
    story.append(P("5. Caveats", H2))
    story += bullets([
        f"Sample sizes are small. Test-set AUC intervals range from ±{min(cis):.3f} to ±{max(cis):.3f} (see the "
        f"± values above), and per-generator numbers ({min(pg_n)}–{max(pg_n)} clips each) are noisy.",
        "The baseline is deliberately weak: a frozen image model, a linear probe, 4 frames and no temporal "
        "modelling. A fine-tuned or video model will score higher. The metadata findings do not depend on the "
        "detector.",
        "GenVidBench Pair2 reals are the first HD-VG clips under 40 MB in the archive stream, not a uniform sample. "
        "GenVideo-100K has no official split, so a hash-based 70/30 split is used.",
        "Run on one CPU-only machine. No hyperparameter tuning was done.",
    ])
    story.append(P("6. Reproduce", H2))
    story.append(P("pip install -r requirements.txt<br/>"
                   "pip install torch --index-url https://download.pytorch.org/whl/cpu<br/>"
                   "pip install scikit-learn transformers pandas av reportlab<br/>"
                   "experiments/run.sh /path/to/exp_dir<br/>"
                   "python experiments/make_report.py /path/to/exp_dir report.pdf", CODE))
    story.append(P("System tools: 7-Zip with RAR support (7zip + 7zip-rar) and ffmpeg. Results, the clip list and "
                   "a Markdown write-up are in experiments/results/ and experiments/README.md.", SMALL))

    def footer(c, doc):
        c.saveState()
        c.setFont("DV", 7)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(18 * mm, 10 * mm, f"vidsample · {REPO_URL}")
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"page {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(out_pdf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="vidsample: build and results",
                            author="vidsample")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
