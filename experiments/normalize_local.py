"""Re-encode the sampled raw clips with vidsample.normalize (no re-download).

python experiments/normalize_local.py EXP_DIR   ->  EXP_DIR/norm/<same layout>.mp4
"""
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import NORM_SPEC, load_clips  # noqa: E402
from vidsample.normalize import Spec, normalize  # noqa: E402

SPEC = Spec(**NORM_SPEC)


def work(args):
    src, dst = args
    if os.path.exists(dst):
        return None
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        normalize(src, dst + ".part", SPEC)
        os.replace(dst + ".part", dst)
        return None
    except ValueError as e:
        return f"{src}: {e}"


def main(exp):
    df = load_clips(exp)
    jobs = [(os.path.join(exp, "raw", p), os.path.join(exp, "norm", n))
            for p, n in zip(df["path"], df["norm_path"])]
    errors = []
    with Pool(4) as pool:
        for i, err in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            if err:
                errors.append(err)
            if i % 250 == 0:
                print(f"{i}/{len(jobs)} normalized, {len(errors)} failed", flush=True)
    print(f"done: {len(jobs)} clips, {len(errors)} failed ({SPEC})")
    for e in errors:
        print("  !", e[:300])


if __name__ == "__main__":
    main(sys.argv[1])
