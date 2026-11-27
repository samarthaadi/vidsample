# Experiments: shortcut test and CPU baseline

Two questions, answered on 3,936 sampled clips (about half real, half fake) from the four datasets:

1. **Shortcut test.** Can container metadata alone tell real from fake, before and after
   `--normalize`? The metadata is file size, bitrate, resolution, fps, duration, codec and audio.
   A high AUC here means a detector could score well without looking at a single pixel.
2. **CPU baseline.** How well does a frozen CLIP ViT-B/32 work, with mean embeddings of 4 frames
   from the first 2 s fed to logistic regression? It is tested within each dataset and across
   datasets, on raw and on normalized clips.

Full tables: [`results/RESULTS.md`](results/RESULTS.md). The clips used are listed in
[`results/manifest.csv`](results/manifest.csv).

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install scikit-learn transformers pandas av
```

```bash
experiments/run.sh /path/to/exp_dir
```

The four steps are `sample.sh`, `normalize_local.py`, `features.py` and `evaluate.py`. The run
needs ~6.5 GB of disk and took 26 min on 4 CPUs, plus ~15 min of sampling and ~25 min of
normalization. Normalization uses 512×512, 16 fps, 2 s, CRF 23.

## Sets

| dataset | train | test (seen generators) | ood (unseen generators / sources) |
|---|---|---|---|
| AIGVDBench | official train split, 20 open-source gens | official test split, same gens | closed-source gens (Sora, Kling, Gen3, …) |
| GenBuster-200K | train | test | benchmark (8 closed-source gens) |
| GenVidBench | 70% of Pair1 | 30% of Pair1 | Pair2 (other gens, HD-VG reals) |
| GenVideo-100K | 70% | 30% | – |

## Findings

1. **On raw clips, metadata alone is a near-perfect detector in 3 of 4 datasets.** The test AUC
   is 1.000 for AIGVDBench, GenVidBench and GenVideo-100K.
   - A single feature is enough. In AIGVDBench every real clip is 1920 px wide, and only 1 of the
     548 fakes is (train-set AUC 1.000). In GenVidBench, width alone gives AUC 0.992. In GenVideo-100K, duration alone
     gives 0.973: reals run ~10 s, fakes ~2 s.
   - GenVidBench and GenVideo reals also carry audio, and their fakes don't.
   - The shortcut also carries across datasets. A metadata model trained on the other three
     datasets scores AUC 0.82–1.00 on the held-out one, because real clips are high-resolution
     and long almost everywhere.
2. **GenBuster-200K is format-matched, but its real clips change bitrate between splits.**
   - Every clip is HEVC, 1024², 24 fps, 5.04 s.
   - In train, real clips are *smaller* than fakes (median 356 vs 542 KB). In test and benchmark
     they are *larger* (~800–850 vs ~500 KB).
   - So the metadata model's rule flips, and it scores AUC 0.21 (test) and 0.14 (benchmark),
     far below chance. Any detector that picks up bitrate on GenBuster train will be
     systematically wrong on its test sets.
3. **`--normalize` removes most of the metadata signal, but not all of it.**
   - In AIGVDBench, GenVidBench and GenVideo-100K, metadata AUC falls from 1.000 to 0.64–0.75.
     GenBuster stays at the inverted ~0.21.
   - What remains is clip length (clips shorter than 2 s) and bits per pixel, which at a fixed
     CRF tracks content complexity.
4. **Normalization costs the CLIP baseline some AIGVDBench accuracy, so part of that score was
   format.**
   - AIGVDBench drops from 0.880 to 0.780 (±0.05).
   - The other datasets barely move: GenBuster 0.977 → 0.963, GenVidBench 0.997 → 0.983,
     GenVideo 0.999 → 0.997.
5. **The baseline holds up on unseen generators within the same dataset, but not on a new real
   source or a new dataset.**
   - On unseen generators it holds: AIGVDBench closed-source 0.82, GenBuster benchmark 0.985.
   - It drops when the real source changes: GenVidBench Pair1 → Pair2 scores 0.71 (±0.07), and
     MuseV and SVD fakes are caught only 20% of the time.
   - Across datasets it is weak. Trained on the other three datasets (normalized clips), it
     scores AIGVDBench 0.61, GenBuster 0.75, GenVidBench 0.86, GenVideo 0.84. AIGVDBench is the
     hardest target from any source (0.49–0.60).
   - The GenBuster and GenVideo in-dataset scores are near 1.0, which suggests their reals and
     fakes also differ in content or style, not only in generation artifacts.

## Caveats

- **Small samples.** Test AUCs have roughly ±0.02–0.07 at 95%. The per-generator accuracies use
  8–25 clips each (AIGVDBench: 8, about ±0.35), so don't rank generators by them.
- **One weak baseline.** It is a frozen image model, linear probe, 4 frames and no temporal
  modelling. Fine-tuned or video models will score higher. The shortcut findings don't depend
  on the model.
- **GenVidBench Pair2 reals are a narrow sample.** They are the first HD-VG clips under 40 MB in
  the stream, not a uniform sample.
