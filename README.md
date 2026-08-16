# AirCanvas

> Draw in the air. Turn motion into pixels.

AirCanvas is a real-time computer-vision passion project that uses a webcam to track a user's index finger and render its movement as a digital drawing on an HTML canvas.

## What it does

- Opens the user's webcam directly in the browser.
- Detects one hand using MediaPipe Hand Landmarker.
- Uses the index fingertip landmark as the drawing cursor.
- Draws only while the index finger is extended and the other fingers are folded.
- Smooths fingertip coordinates to reduce webcam jitter.
- Supports brush sizes, ink colors, canvas clearing, and PNG export.
- Keeps the computer-vision loop local to the browser; no camera frames are uploaded by the app.

## Stack

- React + Vite
- JavaScript
- MediaPipe Tasks Vision
- HTML Canvas 2D
- Browser MediaDevices API

MediaPipe's current Hand Landmarker API supports video/live-stream workflows and exposes hand landmarks for real-time tracking.

## Run locally

```bash
npm install
npm run dev
```

Then open the local Vite URL, allow camera access, and click **OPEN CAMERA**.

## Roadmap

### Phase 1 — Air drawing
- [x] Webcam input
- [x] Index fingertip tracking
- [x] Gesture-gated drawing
- [x] Coordinate smoothing
- [x] Brush controls
- [x] PNG export

### Phase 2 — Gesture controls
- [ ] Pinch to change brush size
- [ ] Two-finger gesture for cursor mode
- [ ] Fist to erase
- [ ] Swipe to undo/redo

### Phase 3 — AI drawing agent
- [ ] Recognize rough shapes
- [ ] Convert rough sketches into clean geometry
- [ ] Recognize handwritten symbols/text
- [ ] Ask an AI model to interpret the canvas
- [ ] Generate structured diagrams from freehand drawings

### Phase 4 — Portfolio-grade system
- [ ] Drawing history and replay
- [ ] Performance telemetry
- [ ] Web Worker / off-main-thread experimentation
- [ ] Mobile camera support
- [ ] Public deployment
- [ ] Technical architecture documentation

## Project philosophy

This is intentionally being built as an engineering experiment rather than a tutorial clone. The goal is to explore real-time human-computer interaction, computer vision, gesture interfaces, rendering, and AI-assisted interpretation in one product.
