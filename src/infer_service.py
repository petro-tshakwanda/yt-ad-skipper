#!/usr/bin/env python3
import pathlib
import subprocess
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd

MODEL_PATH = pathlib.Path("models/tf_model/final.keras")
FEATURE_DIR = pathlib.Path("data_features")
RAW_DIR = pathlib.Path("data_raw")
LABELS_CSV = pathlib.Path("processed/labels_sponsor_segments.csv")

app = FastAPI()
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Load labels once
LABELS_BY_VID = {}
if LABELS_CSV.exists():
    df = pd.read_csv(LABELS_CSV)
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        LABELS_BY_VID.setdefault(vid, []).append((float(row["start"]), float(row["end"]), row["category"]))

def ensure_video_downloaded(video_id: str):
    out_path = RAW_DIR / f"{video_id}.mp4"
    if not out_path.exists():
        url = f"https://www.youtube.com/watch?v={video_id}"
        cmd = ["yt-dlp", "-f", "mp4", "-o", str(out_path), url]
        subprocess.run(cmd, check=True)
    return out_path

def ensure_features(video_id: str):
    feat_path = FEATURE_DIR / f"{video_id}_segments.npz"
    if not feat_path.exists():
        print(f"[INFO] Re-extracting features for {video_id}")
        # Create temp list
        with open("tmp_single_video.txt", "w") as f:
            f.write(video_id + "\n")
        cmd = [
            "python", "src/extract_features.py",
            "--labels-csv", str(LABELS_CSV),
            "--video-list", "tmp_single_video.txt",
            "--video-dir", str(RAW_DIR),
            "--out-dir", str(FEATURE_DIR)
        ]
        subprocess.run(cmd, check=True)
    return feat_path

def compute_ad_intervals(video_id: str, threshold=0.7, min_duration=2.0):
    feat_path = ensure_features(video_id)
    d = np.load(feat_path)
    Z, times = d["z"], d["times"]
    Z_batch = np.expand_dims(Z, 0)
    preds = MODEL.predict(Z_batch, verbose=0)[0, :, 0]
    is_ad = preds > threshold

    intervals = []
    start = None
    for i, flag in enumerate(is_ad):
        if flag and start is None:
            start = times[i, 0]
        elif not flag and start is not None:
            end = times[i-1, 1]
            if end - start >= min_duration:
                intervals.append({"start": float(start), "end": float(end)})
            start = None
    if start is not None:
        end = times[-1, 1]
        if end - start >= min_duration:
            intervals.append({"start": float(start), "end": float(end)})
    return intervals

@app.get("/segments")
def get_segments(video_id: str):
    try:
        ensure_video_downloaded(video_id)
        intervals = compute_ad_intervals(video_id)
        return JSONResponse(content={"video_id": video_id, "segments": intervals})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5005)
