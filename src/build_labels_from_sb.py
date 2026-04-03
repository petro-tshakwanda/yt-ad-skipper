#!/usr/bin/env python3
import argparse
import pathlib

import pandas as pd

AD_LIKE_CATEGORIES = [
    "sponsor",
    "selfpromo",
    "interaction",
    "affiliate",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sb-csv",
        type=str,
        required=True,
        help="/home/petro/sponsorblock_dataset/sb-mirror_*.csv from sb-mirror"
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="labels_sponsor_segments.csv",
        help="Output labels CSV path"
    )
    args = parser.parse_args()

    sb_path = pathlib.Path(args.sb_csv)
    out_path = pathlib.Path(args.out_csv)

    print(f"[INFO] Reading SponsorBlock CSV: {sb_path}")
    df = pd.read_csv(sb_path)

    cols_lower = {c.lower(): c for c in df.columns}
    vid_col = cols_lower.get("videoid")
    s_col = cols_lower.get("starttime")
    e_col = cols_lower.get("endtime")
    cat_col = cols_lower.get("category")

    if not all([vid_col, s_col, e_col, cat_col]):
        raise SystemExit(f"[ERROR] Could not find expected columns in {sb_path}. Found: {df.columns}")

    df = df[[vid_col, s_col, e_col, cat_col]]
    df.columns = ["video_id", "start", "end", "category"]

    before = len(df)
    df = df[df["category"].isin(AD_LIKE_CATEGORIES)]
    print(f"[INFO] Filtered categories {AD_LIKE_CATEGORIES}: {before} -> {len(df)} rows")

    # Optional: drop duplicates
    df = df.drop_duplicates(subset=["video_id", "start", "end", "category"])

    out_path = pathlib.Path("processed") / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote labels to {out_path}")

if __name__ == "__main__":
    main()
