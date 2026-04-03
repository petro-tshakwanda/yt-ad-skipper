#!/usr/bin/env python3
import argparse
import pathlib
import json

import numpy as np
import pandas as pd
import cv2
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
import whisper

VIDEO_EMBED_MODEL = None
WHISPER_MODEL = None
AUDIO_SR = 16000

def init_models():
    global VIDEO_EMBED_MODEL, WHISPER_MODEL
    if VIDEO_EMBED_MODEL is None:
        VIDEO_EMBED_MODEL = hub.KerasLayer(
            "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4",
            trainable=False
        )
    if WHISPER_MODEL is None:
        WHISPER_MODEL = whisper.load_model("tiny")  # tiny/base/small as resources allow

def load_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    by_vid = {}
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        by_vid.setdefault(vid, []).append((float(row["start"]), float(row["end"]), row["category"]))
    return by_vid

def video_duration(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps <= 0:
        return 0.0
    return frames / fps

def extract_frame_embedding_segment(video_path, start, end, frame_rate=1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return np.zeros((1280,), dtype=np.float32)  # fallback size

    start_frame = int(start * fps)
    end_frame = int(end * fps)
    step = int(max(int(fps / frame_rate), 1))

    frames = []
    for i in range(start_frame, end_frame, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))
        frames.append(frame_resized)
    cap.release()

    if not frames:
        return np.zeros((1280,), dtype=np.float32)

    frames = np.array(frames) / 255.0
    emb = VIDEO_EMBED_MODEL(frames).numpy()
    return emb.mean(axis=0)  # [D]

def extract_audio_embedding_segment(video_path, start, end):
    y, sr = librosa.load(str(video_path), sr=AUDIO_SR, offset=start, duration=end-start)
    if y.size == 0:
        return np.zeros((128,), dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    log_mel = librosa.power_to_db(mel)
    # simple pooling
    mean = log_mel.mean(axis=1)
    return mean.astype(np.float32)  # [64]

def extract_text_embedding_segments(video_path, segments, total_dur):
    """
    segments: whisper segments [{start,end,text},...]
    Returns: dict { (start,end) -> text_embedding }
    For simplicity, we just use bag-of-words in this script; you can swap in sentence-transformers.
    """
    # Simple numeric encoding: (length, num sponsor words)
    sponsor_words = ["sponsored", "this video is sponsored", "our sponsor", "use code", "link in the description"]
    # We'll compute features on the fly in main loop; here we just keep segments.
    return segments, sponsor_words

def compute_text_features(segments, sponsor_words, start, end):
    texts = [s["text"] for s in segments if not (s["end"] < start or s["start"] > end)]
    full_text = " ".join(texts).lower()
    length = len(full_text)
    sponsor_hits = sum(1 for w in sponsor_words if w in full_text)
    return np.array([length, sponsor_hits], dtype=np.float32)

def overlap_fraction(seg_start, seg_end, label_ranges):
    """
    fraction of [seg_start, seg_end] that overlaps any label range
    """
    seg_len = seg_end - seg_start
    if seg_len <= 0:
        return 0.0
    total_overlap = 0.0
    for s, e, _ in label_ranges:
        if e <= seg_start or s >= seg_end:
            continue
        inter_start = max(seg_start, s)
        inter_end = min(seg_end, e)
        total_overlap += max(0.0, inter_end - inter_start)
    return total_overlap / seg_len

def process_video(video_path, label_ranges, out_dir, window=3.0, stride=1.0, overlap_threshold=0.5):
    vid = video_path.stem
    duration = video_duration(video_path)
    if duration <= 0:
        print(f"[WARN] Could not get duration for {vid}")
        return

    init_models()

    # Transcribe once per video
    print(f"[INFO] Transcribing {vid}")
    result = WHISPER_MODEL.transcribe(str(video_path), verbose=False)
    whisper_segments = result.get("segments", [])
    text_segments, sponsor_words = extract_text_embedding_segments(video_path, whisper_segments, duration)

    features = []
    labels = []
    times = []

    t = 0.0
    while t + window <= duration:
        seg_start = t
        seg_end = t + window

        v_emb = extract_frame_embedding_segment(video_path, seg_start, seg_end)   # [Dv]
        a_emb = extract_audio_embedding_segment(video_path, seg_start, seg_end)   # [Da]
        l_emb = compute_text_features(text_segments, sponsor_words, seg_start, seg_end)  # [2]
        pos = np.array([seg_start / duration], dtype=np.float32)

        z = np.concatenate([v_emb, a_emb, l_emb, pos], axis=0)
        frac = overlap_fraction(seg_start, seg_end, label_ranges)
        y = 1.0 if frac >= overlap_threshold else 0.0

        features.append(z)
        labels.append(y)
        times.append((seg_start, seg_end))

        t += stride

    features = np.stack(features, axis=0)
    labels = np.array(labels, dtype=np.float32)
    times = np.array(times, dtype=np.float32)

    out_path = out_dir / f"{vid}_segments.npz"
    np.savez(out_path, z=features, y=labels, times=times)
    print(f"[INFO] Saved {out_path} ({features.shape[0]} segments)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", type=str, default="processed/labels_sponsor_segments.csv")
    parser.add_argument("--video-list", type=str, default="video_ids_sample.txt")
    parser.add_argument("--video-dir", type=str, default="data_raw")
    parser.add_argument("--out-dir", type=str, default="data_features")
    args = parser.parse_args()

    labels_by_vid = load_labels(args.labels_csv)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.video_list, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    for vid in tqdm(video_ids, desc="Processing videos"):
        video_path = pathlib.Path(args.video_dir) / f"{vid}.mp4"
        if not video_path.exists():
            print(f"[WARN] Video not found: {video_path}")
            continue
        label_ranges = labels_by_vid.get(vid, [])
        if not label_ranges:
            # we still can include negative-only examples, but for first pass you can skip them
            print(f"[INFO] No ad labels for {vid}, skipping for now")
            continue
        process_video(video_path, label_ranges, out_dir)

if __name__ == "__main__":
    main()
