#!/usr/bin/env python3
import argparse
import pathlib

import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-csv",
        type=str,
        default="processed/labels_sponsor_segments.csv"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=500,
        help="Number of unique videos to sample"
    )
    parser.add_argument(
        "--out-list",
        type=str,
        default="video_ids_sample.txt"
    )
    args = parser.parse_args()

    labels = pd.read_csv(args.labels_csv)
    vids = labels["video_id"].dropna().unique()
    if len(vids) > args.max_videos:
        vids = vids[:args.max_videos]

    out_path = pathlib.Path(args.out_list)
    with open(out_path, "w") as f:
        for v in vids:
            f.write(str(v).strip() + "\n")

    print(f"[INFO] Wrote {len(vids)} video IDs to {out_path}")

if __name__ == "__main__":
    main()
