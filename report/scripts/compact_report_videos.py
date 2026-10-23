"""Encode compact report videos, preserving originals and recording both hashes."""

import hashlib
import json
from pathlib import Path
import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "report/videos"
BACKUP = VIDEOS / ".originals"
RECORD = ROOT / "report/video-encoding.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(path):
    with imageio.get_reader(path) as reader:
        info = reader.get_meta_data()
        reader.get_data(round(info["duration"] * info["fps"]) - 1)
    return info


def main():
    BACKUP.mkdir(exist_ok=True)
    record = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    for path in sorted(VIDEOS.glob("*.mp4")):
        if not path.name.startswith(("walk01_", "flatfeet_", "policy_transfer01_")):
            continue
        if path.name in record:
            assert digest(path) == record[path.name]["encoded_sha256"], path
            continue
        original = BACKUP / path.name
        temporary = VIDEOS / (".compact-" + path.name)
        assert path.resolve().is_relative_to(ROOT)
        assert original.resolve().is_relative_to(ROOT)
        if original.exists():
            raise RuntimeError(
                f"Original already exists without an encoding record: {original}"
            )
        before = metadata(path)
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(path),
                "-c:v",
                "libx264",
                "-crf",
                "27",
                "-preset",
                "medium",
                "-threads",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
        )
        after = metadata(temporary)
        assert before["size"] == after["size"]
        assert abs(before["fps"] - after["fps"]) < 0.01
        assert abs(before["duration"] - after["duration"]) < 0.06
        entry = dict(
            original_sha256=digest(path),
            encoded_sha256=digest(temporary),
            original_bytes=path.stat().st_size,
            encoded_bytes=temporary.stat().st_size,
            fps=after["fps"],
            seconds=after["duration"],
            size=list(after["size"]),
            encoder="libx264",
            crf=27,
            preset="medium",
            note="Report encoding only; simulation, speed and frame order unchanged. "
            "Original retained locally in ignored report/videos/.originals/.",
        )
        path.replace(original)
        temporary.replace(path)
        record[path.name] = entry
        RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(
            f"{path.name}: {entry['original_bytes'] / 1e6:.1f} -> {entry['encoded_bytes'] / 1e6:.1f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
