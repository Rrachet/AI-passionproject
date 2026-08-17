# AirCanvas Vision Model Strategy

## Model candidates

### 1. MediaPipe Hand Landmarker — production baseline
- 21 hand landmarks.
- Designed for real-time hand tracking on mobile/web.
- Keep this as the browser fallback and teacher/label generator.

### 2. RTMPose hand — Python primary candidate
- MMPose provides a dedicated 21-keypoint hand RTMPose model.
- Better fit for a Python training pipeline than replacing the whole system with a generic object detector.
- Train/fine-tune on AirCanvas camera samples after the active-learning dataset is large enough.

### 3. Ultralytics YOLO pose — experimental challenger
- Supports custom pose datasets and 21-keypoint hand layouts.
- Useful as a second model for benchmarking and difficult camera conditions.
- Do not make it the production model until it beats the RTMPose/MediaPipe baseline on our own validation set.

## What we train first

Do **not** immediately retrain the hand detector. The current system already produces 21 landmarks. The first trainable intelligence should learn from those landmarks over time:

1. fingertip trajectory prediction;
2. gesture recognition;
3. tracking-loss recovery;
4. drawing-state prediction;
5. shape intent classification.

This is a temporal sequence-learning problem, so a small GRU/TCN is a better first experiment than training a large vision model from scratch.

## Self-learning loop

```text
Live landmarks
    -> confidence / uncertainty
    -> hard-example sampler
    -> user correction
    -> JSONL dataset
    -> train candidate
    -> validation set
    -> compare against current model
    -> promote only if metrics improve
```

The system must never silently replace the production model. Every new model is evaluated against a frozen validation set before promotion.

## What counts as learning

- Low-confidence frames are candidates for review.
- User corrections become labeled examples.
- Repeated failure patterns are oversampled during training.
- The validation set is kept separate so the model cannot simply memorize its training data.
- Promotion is metric-driven, not based on training loss alone.
