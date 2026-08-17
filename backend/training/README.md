# AirCanvas training loop

The project now uses **active learning**, not blind self-retraining.

## 1. Collect mistakes

Run the Python API locally and send corrections to:

`POST /learning/feedback`

Every sample stores the 21 landmarks, predicted gesture, optional corrected gesture, confidence, and whether it is a hard example.

## 2. Inspect the dataset

`GET /learning/stats`

Hard examples are the highest-value samples for the next training run.

## 3. Train the first learned model

From the repository root, after installing `backend/requirements-training.txt`:

```bash
python backend/training/train_temporal.py
```

The first model is a small GRU over 24 frames of 21 hand landmarks. This learns temporal gesture behavior without replacing the proven hand landmark detector.

## 4. Model progression

**Stage A:** MediaPipe landmarks + Python temporal tracker.

**Stage B:** GRU learns gesture/state transitions and fingertip motion patterns.

**Stage C:** RTMPose hand becomes the Python detector candidate and is fine-tuned on AirCanvas camera data.

**Stage D:** YOLO hand-keypoint model is benchmarked as a second detector.

A candidate only replaces the current model after it beats a frozen validation set on fingertip error, missed-frame rate, gesture F1, and real-time latency.

## Important

The system should never modify its production weights just because it saw a new frame. New data is collected, reviewed/validated, trained into a candidate, and promoted only when metrics improve. That is the safe version of a system that “builds itself” over time.
