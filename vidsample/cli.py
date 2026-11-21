"""vidsample: sample clips from remote AI-video-detection datasets without downloading them.

  python -m vidsample.cli list   --dataset aigvdbench
  python -m vidsample.cli sample --dataset genbuster200k --per-generator 200 --split train --out data
"""
import argparse
from collections import Counter

from . import sampler
from .normalize import Spec
from .datasets import DATASETS


def _datasets(name):
    return list(DATASETS.values()) if name == "all" else [DATASETS[name]]


def cmd_list(args):
    for mod in _datasets(args.dataset):
        print(f"\n== {mod.NAME}")
        counts, streams = Counter(), []
        for a in mod.archives():
            if not a.random_access:
                streams.append(a)
                continue
            for e in a.entries():
                m = a.classify(e.name)
                if m:
                    counts[(m.split, m.label, m.generator)] += 1
        for (split, label, gen), n in sorted(counts.items()):
            print(f"  {split:12} {label:5} {gen:28} {n:>8}")
        for a in streams:
            gb = sum(a.sizes) / 2**30
            print(f"  {'(stream)':12} {a.stream_groups[0][0]:5} {a.stream_groups[0][1]:28} "
                  f"{'?':>8}   {gb:.1f} GB, {a.kind}")


def cmd_sample(args):
    splits = set(args.split) if args.split else None
    spec = (Spec(args.norm_size, args.norm_fps, args.norm_max_seconds, args.norm_crf)
            if args.normalize else None)
    for mod in _datasets(args.dataset):
        print(f"\n== {mod.NAME}", flush=True)
        sampler.run(mod.archives(), args.out, args.per_generator, real=args.real, splits=splits,
                    seed=args.seed, workers=args.workers, max_clip_mb=args.max_clip_mb,
                    max_stream_gb=args.max_stream_gb, stream_keep_prob=args.stream_keep_prob,
                    use_streams=not args.no_streams, dry_run=args.dry_run, verify=args.verify,
                    normalize=spec)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vidsample", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    choices = sorted(DATASETS) + ["all"]

    pl = sub.add_parser("list", help="count clips per split/label/generator (no video bytes)")
    pl.add_argument("--dataset", choices=choices, required=True)
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("sample", help="fetch a balanced sample of clips")
    ps.add_argument("--dataset", choices=choices, required=True)
    ps.add_argument("--out", default="data")
    ps.add_argument("--per-generator", type=int, default=100, help="clips per fake generator")
    ps.add_argument("--real", type=int, default=None,
                    help="total real clips (default: same as total fake, spread over real sources)")
    ps.add_argument("--split", nargs="*", help="keep only these splits, e.g. train / pair1 / test")
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--workers", type=int, default=8)
    ps.add_argument("--max-clip-mb", type=float, default=None, help="skip clips larger than this")
    ps.add_argument("--max-stream-gb", type=float, default=2.0,
                    help="bandwidth cap per stream-only archive")
    ps.add_argument("--stream-keep-prob", type=float, default=1.0,
                    help="<1 spreads stream samples further into the archive (costs bandwidth)")
    ps.add_argument("--no-streams", action="store_true", help="skip stream-only archives")
    ps.add_argument("--verify", action="store_true",
                    help="ffprobe each clip and drop undecodable ones")
    ps.add_argument("--normalize", action="store_true",
                    help="re-encode every clip to one format (H.264, square, fixed fps, no audio)")
    ps.add_argument("--norm-size", type=int, default=512, help="output is SIZE x SIZE (centre crop)")
    ps.add_argument("--norm-fps", type=float, default=16)
    ps.add_argument("--norm-max-seconds", type=float, default=10)
    ps.add_argument("--norm-crf", type=int, default=23)
    ps.add_argument("--dry-run", action="store_true", help="print the plan, fetch nothing")
    ps.set_defaults(func=cmd_sample)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
