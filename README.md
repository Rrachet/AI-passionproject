# AirCanvas

> Put the idea in the air.

AirCanvas is a real-time computer-vision passion project exploring a simple question: **can a hand become a useful interface for explaining ideas?**

The long-term target is a Chrome extension for calls and meetings. The user should be able to draw, point, highlight or clean up a rough sketch directly over a conversation.

## Architecture

```text
Camera (phone / laptop)
        |
        v
React + Canvas + MediaPipe
        |
        | 21 hand landmarks / frame
        v
Python FastAPI + WebSocket
        |
        +--> temporal filtering
        +--> gesture memory
        +--> short-gap prediction
        +--> movement analysis
        +--> shape alignment
        +--> future learned models
        |
        v
AR interaction layer
```

The browser owns camera permissions and fast visual rendering. Python is the temporal intelligence layer. We intentionally send compact hand landmarks rather than uploading raw camera video frame-by-frame.

> More Python is not automatically better. The useful goal is to put the parts that need temporal reasoning, evaluation and training in Python, then measure whether they actually improve tracking.

## Interaction modes

- **Draw** — continuous freehand AR ink.
- **Shape** — sketch a rough line, circle or rectangle; make a fist to align it.
- **Pointer** — fingertip becomes a clean presentation pointer.
- **Laser** — temporary glowing trail for emphasis.

Core gestures:

- ☝️ index finger — interact / draw
- ✊ fist — pause; in Shape mode, commit alignment
- 🖐️ open palm — hold to clear

## Python vision engine

The Python backend now contains a temporal tracker using an adaptive One Euro filter, velocity estimation, confidence decay and short prediction bridges for missed frames. Gesture classification uses joint geometry plus temporal majority voting rather than trusting one noisy frame.

It also exposes `/shape` for Python-side shape alignment and `/ws/vision` for real-time gesture/tip results.

### Run locally

Frontend:

```bash
npm install
npm run dev
```

Python vision engine:

```bash
cd backend
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

For the local Python bridge, set:

```text
VITE_VISION_WS_URL=ws://localhost:8000/ws/vision
```

The public GitHub Pages build keeps a local MediaPipe fallback so the site remains usable without a Python server. Production Python integration will be enabled once the persistent WebSocket service is deployed.

## Stack

### Frontend
- React + Vite
- JavaScript
- MediaPipe Tasks Vision
- HTML Canvas 2D
- Browser MediaDevices API

### Vision / AI
- Python
- FastAPI
- WebSockets
- NumPy
- OpenCV
- MediaPipe
- Pydantic

## Roadmap

### Phase 1 — Air drawing
- [x] Webcam input
- [x] Index fingertip tracking
- [x] Gesture-gated drawing
- [x] Adaptive browser smoothing
- [x] Brush controls
- [x] PNG export

### Phase 2 — Python temporal engine
- [x] FastAPI service
- [x] WebSocket vision channel
- [x] Server-side gesture classification
- [x] Adaptive temporal filtering
- [x] Confidence tracking
- [x] Short tracking-drop prediction
- [x] Python shape alignment
- [ ] Production WebSocket deployment

### Phase 3 — Intelligent drawing agent
- [ ] Pinch / swipe vocabulary
- [ ] Stroke segmentation
- [ ] Learned shape classifier
- [ ] Handwriting recognition
- [ ] Canvas understanding
- [ ] Natural-language commands
- [ ] AI-assisted drawing cleanup

### Phase 4 — Meeting extension
- [ ] Chrome Manifest V3 extension
- [ ] Transparent AR overlay
- [ ] Meet / browser tab integration
- [ ] Persistent annotations
- [ ] Shared collaboration layer

## Project philosophy

Build → test → observe → measure → improve.

This is intentionally an engineering experiment rather than a tutorial clone. Every tracking failure becomes a data point for the next iteration.

**Made by Amar.**
