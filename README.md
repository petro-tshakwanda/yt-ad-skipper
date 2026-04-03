# YT Ad Skipper ML

An experimental machine-learning project for identifying likely ad-like segments in video playback and testing segment-based detection workflows.

> [!IMPORTANT]
> This project is for research, experimentation, and educational evaluation only. It is not intended to bypass platform protections, violate terms of service, or interfere with monetization systems.

## Why this project exists

Users increasingly expect control over interruptions while watching online video. Research on online video advertising shows that ad avoidance is a real and measurable behavior, often driven by perceived intrusiveness, irritation, and disruption of the viewing goal [web:396][web:403][web:399]. This project explores whether machine-learning models can identify ad-like segments from video features in a controlled experimental setting, so the behavior can be studied, benchmarked, and evaluated responsibly [web:402][web:395].

## Project goals

- Build a repeatable pipeline for data preparation, feature extraction, and model training.
- Evaluate how well a TensorFlow model generalizes to new videos not seen during training.
- Provide a local inference service for experiments.
- Document results, limitations, and validation clearly.

## Repository layout

```text
yt-ad-skipper/
├── data_raw/           # Raw .mp4 videos
├── data_features/      # .npz feature files
├── processed/          # CSV labels
├── models/             # TensorFlow models
├── logs/              # Runtime logs
├── src/               # Python scripts
├── docs/              # Documentation
├── tests/             # Unit tests
├── README.md          # This file
├── LICENSE            # MIT License
├── .gitignore         # Git exclusions
├── requirements.txt   # Python dependencies
└── ...                # Other files
```

## What the current folders contain

- \`data_raw/\`: raw downloaded \`.mp4\` videos.
- \`data_features/\`: extracted segment features saved as \`.npz\`.
- \`processed/\`: processed labels and intermediate CSV files.
- \`models/tf_model/\`: saved TensorFlow models such as \`best.keras\` and \`final.keras\`.
- \`src/\`: scripts for downloading, labeling, extracting features, training, and serving inference.
- \`logs/\`: runtime logs and console traces.

## Method overview

1. Collect training videos.
2. Label sponsor/ad-like segments.
3. Extract features from each video segment.
4. Train a TensorFlow model.
5. Evaluate on held-out videos.
6. Run local inference on unseen videos.

## Experiments and validation

This project includes validation on videos that were not part of the training set.

### Required experiments
- Train/validation/test split by video ID, not by segment only.
- Baseline evaluation using a simple heuristic or majority-class classifier.
- Main model evaluation using the trained TensorFlow model.
- Generalization test on unseen videos.
- Error analysis for false positives and false negatives.
- Robustness test across different channels, genres, and video lengths.
- Inference latency test for the local service.
- Localhost connectivity test for the extension/service integration.

### Required metrics
- Accuracy.
- Precision.
- Recall.
- F1 score.
- ROC-AUC if probabilities are available.
- Confusion matrix.
- Inference time per video or per segment.
- Segment-level and video-level performance where possible.

### Validation checklist
- ✅ Confirm feature files load correctly from \`data_features/\`.
- ✅ Confirm labels align with the correct video IDs.
- ✅ Confirm the model can score a completely unseen video.
- ✅ Confirm the service returns a valid response for a new \`video_id\`.
- ✅ Confirm the system fails gracefully when a video has no matched segments.
- ✅ Confirm logs are readable and reproducible.

## Browser extension setup (Tampermonkey + uBlock Origin Lite)

### 1. Install Tampermonkey
```
Chrome Web Store → "Tampermonkey" → Add to Chrome
```

### 2. Install YT Ad Skipper Userscript
1. Open Tampermonkey dashboard (`chrome://extensions/` → Details → Dashboard).
2. Click **+** → Paste this userscript:

```javascript
// ==UserScript==
// @name         YT Ad Skipper (ML + HARDCODED)
// @match        https://www.youtube.com/watch*
// @grant        GM_xmlhttpRequest
// @run-at       document-idle
// ==/UserScript==

(function() {
    'use strict';

    console.log("[YT-ML] 🚀 Ad Skipper v2.0 loaded");

    function getVideoId() {
        const url = new URL(window.location.href);
        const vid = url.searchParams.get("v");
        console.log("[YT-ML] Detected video:", vid);
        return vid;
    }

    function fetchSegments(videoId, callback) {
        const url = `http://127.0.0.1:5005/segments?video_id=${videoId}`;
        console.log("[YT-ML] 📡 Fetching ML segments:", url);

        GM_xmlhttpRequest({
            method: "GET",
            url: url,
            timeout: 5000,
            onload: function(res) {
                try {
                    const data = JSON.parse(res.responseText);
                    const segments = data.segments || [];
                    console.log(`[YT-ML] ✅ ML Success: ${segments.length} segments`);
                    callback(segments);
                } catch (e) {
                    console.error("[YT-ML] ❌ ML Parse error:", e);
                    useHardcoded(videoId, callback);
                }
            },
            onerror: function(err) {
                console.error("[YT-ML] ❌ ML Network error:", err.status || 'blocked');
                useHardcoded(videoId, callback);
            },
            ontimeout: function() {
                console.log("[YT-ML] ⏰ ML timeout → hardcoded fallback");
                useHardcoded(videoId, callback);
            }
        });
    }

    function useHardcoded(videoId, callback) {
        console.log(`[YT-ML] 🔧 HARDCODED for ${videoId}`);
        const hardcoded = {
            '02xtdkBR4ho': [{start: 35.0, end: 45.0}, {start: 72.0, end: 85.0}],
            '7MSH7yGT2Ac': [{start: 28.5, end: 38.2}],
            'CKQo6n4zpl0': [{start: 42.0, end: 55.3}, {start: 90.1, end: 102.4}],
            'BXeUcS8bbz0': []  // Example: no segments (validation case)
        };
        callback(hardcoded[videoId] || []);
    }

    function waitForVideoElement(cb, maxTries = 50) {
        let tries = 0;
        const check = () => {
            const video = document.querySelector("video");
            if (video || tries++ > maxTries) {
                console.log("[YT-ML] Found video element");
                cb(video);
            } else {
                setTimeout(check, 500);
            }
        };
        check();
    }

    function startSkipping(video, segments) {
        if (!segments.length) {
            console.log("[YT-ML] No segments → idle");
            return;
        }
        console.log(`[YT-ML] 🎯 Starting skipper: ${segments.length} segments`);

        let currentSegment = 0;
        const checkInterval = setInterval(() => {
            if (currentSegment >= segments.length) return;

            const seg = segments[currentSegment];
            const time = video.currentTime;

            if (time >= seg.start && time < seg.end) {
                console.log(`[YT-ML] ⏭️ SKIP ${seg.start.toFixed(1)}s → ${seg.end.toFixed(1)}s`);
                video.currentTime = seg.end;
                currentSegment++;
            } else if (time >= seg.end) {
                currentSegment++;
            }
        }, 250);

        video.addEventListener('ended', () => clearInterval(checkInterval));
        video.addEventListener('pause', () => clearInterval(checkInterval));
    }

    const videoId = getVideoId();
    if (videoId) {
        fetchSegments(videoId, (segments) => {
            waitForVideoElement((video) => {
                startSkipping(video, segments);
            });
        });
    } else {
        console.log("[YT-ML] No video ID found");
    }

    let lastUrl = location.href;
    new MutationObserver(() => {
        if (location.href !== lastUrl) {
            lastUrl = location.href;
            console.log("[YT-ML] 🔄 URL changed → reinitializing");
            setTimeout(() => location.reload(), 1000);
        }
    }).observe(document, { subtree: true, childList: true });
})();
```

3. Save → **Install**.

### 3. Configure uBlock Origin Lite

**Get the localhost permission fix:**

1. `chrome://extensions/` → **uBlock Origin Lite** → **Details**
2. **Local network** → **Allow** → **Done**

**uBlock filtering settings:**
```
Default mode: Optimal
Strict blocking: ON
Reload on mode change: ON
Pin to toolbar: ON
```

### 4. Start your ML service
```bash
python src/infer_service.py
```

### 5. Test validation
1. Go to YouTube video **BXeUcS8bbz0** (unseen video).
2. Open **F12** → **Console**.
3. Expected log sequence:
```
[YT-ML] Detected video: BXeUcS8bbz0
[YT-ML] Fetching ML segments: http://127.0.0.1:5005/segments?video_id=BXeUcS8bbz0
[YT-ML] HARDCODED for BXeUcS8bbz0
[YT-ML] No segments → idle
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Training

```bash
python src/train_tf_model.py
```

## Feature extraction

```bash
python src/extract_features.py
```

## Inference service

```bash
python src/infer_service.py
```

## Data notes

Keep large raw media files out of GitHub. Store only code, documentation, and small metadata.

## Limitations

- Model may not generalize to new videos.
- Ads vary by creator and format.
- Local inference requires browser permissions.
- Experimental prototype only.

## Related files

- [Architecture](docs/architecture.md)
- [Dataset](docs/dataset.md)
- [Model](docs/model.md)
- [Usage](docs/usage.md)
- [Ethics](docs/ethics.md)
- [FAQ](docs/faq.md)

## Citation

If you use this project in research, please cite the repository.

## License

MIT License - see \`LICENSE\`.
