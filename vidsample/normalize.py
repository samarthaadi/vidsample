"""Re-encode clips to one fixed format so a detector cannot learn source-format shortcuts.

Real and fake clips in these datasets differ in bitrate, resolution, codec, container, audio
and metadata (e.g. AIGVDBench real ~2.9 MB 1080p vs ~50-540 KB fakes). Normalizing removes
those cues: H.264 yuv420p, square SxS (scale then centre-crop, no padding bars), fixed fps,
capped duration, no audio/subtitles/data streams, no metadata.
"""
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Spec:
    size: int = 512
    fps: float = 16
    max_seconds: float = 10
    crf: int = 23

    def __str__(self):
        return f"h264-{self.size}sq-{self.fps:g}fps-{self.max_seconds:g}s-crf{self.crf}"


def available():
    return shutil.which("ffmpeg") is not None


def normalize(src, dst, spec):
    s = spec.size
    vf = (f"fps={spec.fps:g},scale={s}:{s}:force_original_aspect_ratio=increase,"
          f"crop={s}:{s},setsar=1")
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", src, "-t", f"{spec.max_seconds:g}",
           "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
           "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", str(spec.crf),
           "-pix_fmt", "yuv420p", "-threads", "2", "-fflags", "+bitexact",
           "-flags:v", "+bitexact", "-movflags", "+faststart", "-f", "mp4", dst]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"ffmpeg failed: {r.stderr.strip()[-300:]}")
