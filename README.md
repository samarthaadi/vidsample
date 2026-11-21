# vidsample

Samples clips from remote AI-generated-video detection datasets **without downloading the
archives**. The tool reads archive directories over HTTP Range requests, then fetches only the
selected clips. Output is the clip files plus a `manifest.csv` for training.

| `--dataset` | Source | How clips are fetched |
|---|---|---|
| `genvidbench` | HF `jian-0/GenVidBench` (AAAI 2026) | RAR5 + zip: random access. HD-VG real (7z/LZMA2) and OpenSora_13800 (tar.gz): streamed |
| `genbuster200k` | HF `l8cv/GenBuster-200K` | split 7z, Copy coder: random access (pure byte ranges) |
| `aigvdbench` | HF `AIGVDBench/AIGVDBench` (CVPR 2026) | zip per generator: random access; split from `Split/*.jsonl` |
| `genvideo100k` | ModelScope `cccnju/GenVideo-100K` (DeMamba) | tar.gz: streamed, stops when the quota is filled |

"Streamed" means reading from the start of the archive and stopping early. Members arrive in
shuffled order, so the prefix is a fair sample. `--max-stream-gb` caps the bandwidth.

VAE-rebuilt real clips are not included because there is no public release.

## Setup

```bash
pip install -r requirements.txt
```

```bash
sudo apt install 7zip 7zip-rar ffmpeg
```

You need `7zip-rar` (or `p7zip-full p7zip-rar`) only for GenVidBench's RAR files. You need
`ffmpeg` only for `--verify` and `--normalize`. Set `HF_TOKEN` if you get rate-limited by Hugging Face.

## Usage

```bash
python -m vidsample list --dataset all
```

```bash
python -m vidsample sample --dataset all --per-generator 200 --out data --verify
```

```bash
python -m vidsample sample --dataset aigvdbench --split train --per-generator 500 --out data
```

```bash
python -m vidsample sample --dataset genvidbench --split pair1 --per-generator 300 --no-streams --out data
```

- `--per-generator N`: clips per fake generator.
- `--real N`: total real clips. The default matches the total fake count, spread evenly over
  the real sources.
- `--split`: AIGVDBench `train/val/test`; GenBuster `train/test/benchmark`; GenVidBench
  `pair1/pair2/extra` (the paper trains on one pair and tests on the other); GenVideo-100K
  `train/unspecified`.
- `--max-clip-mb`: skips huge clips. Some HD-VG clips are more than 150 MB.
- `--verify`: runs ffprobe on every clip and drops undecodable ones. Some source archives
  contain broken or 0-byte clips. Clips under 1 KB are always skipped.
- `--dry-run`: prints the per-generator plan without fetching anything.
- `--seed`: fixes the sample. Re-running skips clips already on disk, so runs are resumable.

Output: `data/<dataset>/<real|fake>/<generator>/<clip>` plus `data/manifest.csv` with the
columns `path, dataset, label, generator, split, archive, member, bytes, src_bytes, norm`.

## Format normalization (`--normalize`)

In these datasets, real and fake clips differ in their format. For example, AIGVDBench real
clips have a median size of ~2.9 MB at 1080p, while most open-source fakes are 50–540 KB.
HD-VG real clips carry audio, and the I2VGEN_XL fakes are GIFs. A detector trained on raw clips
can learn those differences instead of generation artifacts.

```bash
python -m vidsample sample --dataset all --per-generator 200 --normalize --verify --out data_norm
```

`--normalize` re-encodes every clip with ffmpeg to the same format:

- H.264, yuv420p, mp4
- `--norm-size` × `--norm-size` (default 512), scaled then center-cropped, so there are no padding bars
- `--norm-fps` (default 16), at most `--norm-max-seconds` long (default 10), `--norm-crf` 23
- no audio, subtitle or data streams, and the source metadata is stripped

In the manifest, `src_bytes` is the original size, `bytes` is the size of the written file, and
`norm` holds the spec. Use a separate `--out` for each spec: a different spec in the same
directory re-encodes and replaces the clips.

After normalization, bitrates match closely, but clip length still differs. GenVidBench and
GenVideo-100K real clips run 8–10 s, while their fakes are ~2 s. Either sample a fixed number of
frames in your data loader, or pass `--norm-max-seconds 2`.

Archive listings are cached in `./index/` (override with `VIDSAMPLE_INDEX`). Only the first
`list` or `sample` of each archive reads its directory.

## Layout

- `vidsample/remote.py`: seekable HTTP Range file, multi-volume concatenation, sequential stream
- `vidsample/formats/`: `zip_remote`, `sevenz_remote` (py7zr header + direct ranges),
  `rar5_remote` (quick-open listing, per-member extraction via 7-Zip), `stream_reader`
- `vidsample/datasets/`: per-dataset archive lists and label/split rules
- `vidsample/sampler.py`: balanced planning, parallel fetch, manifest
- `vidsample/normalize.py`: ffmpeg re-encode used by `--normalize`
