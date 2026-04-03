#!/usr/bin/env python3
import argparse
import pathlib
import subprocess
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-list", type=str, required=True,
                        help="Text file with one YouTube video ID per line")
    parser.add_argument("--out-dir", type=str, default="data_raw",
                        help="Directory to store downloaded videos")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.video_list, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    for vid in tqdm(video_ids, desc="Downloading videos"):
        out_path = out_dir / f"{vid}.mp4"
        if out_path.exists():
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        cmd = [
            "yt-dlp",
            "-f", "mp4",
            "-o", str(out_path),
            url
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[WARN] Failed to download {vid}")

if __name__ == "__main__":
    main()
