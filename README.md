# AirCanvas

> Draw in the air. Let AI understand the motion.

AirCanvas is a real-time computer-vision and AI passion project that turns hand movement into digital strokes and is being evolved into an intelligent drawing agent.

## Architecture

```text
Camera (phone / laptop)
        |
        v
React + Canvas + MediaPipe landmarks
        |
        | compact 21-point hand landmarks
        v
Python FastAPI + WebSocket
        |
        +--> gesture classification
        +--> trajectory analysis
        +--> future computer-vision models
        +--> future AI agent
        |
        v
Canvas + intelligent drawing interpretation
```

The browser owns camera permissions and real-time rendering. Python owns the intelligence layer. We intentionally avoid uploading raw camera video on every frame; the browser can send compact landmark data to the Python service instead.

## Current capabilities

- Browser webcam access
- MediaPipe hand landmark tracking
- Index fingertip drawing
- Gesture-gated drawing
- Coordinate smoothing
- Brush sizes and ink colors
- Canvas clearing and PNG export
- Python FastAPI vision engine
- WebSocket protocol for real-time gesture events
- Mobile-friendly browser architecture

## Stack

### Frontend
- React + Vite
- JavaScript
- MediaPipe Tasks Vision
- HTML Canvas 2D
- Browser MediaDevices API

### AI / Vision backend
- Python
- FastAPI
- WebSockets
- NumPy
- OpenCV
- MediaPipe
- Pydantic

## Run locally

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

Then open the Vite URL, allow camera access, and click **OPEN CAMERA**.

## Roadmap

### Phase 1 — Air drawing
- [x] Webcam input
- [x] Index fingertip tracking
- [x] Gesture-gated drawing
- [x] Coordinate smoothing
- [x] Brush controls
- [x] PNG export

### Phase 2 — Python vision engine
- [x] FastAPI service
- [x] WebSocket vision channel
- [x] Server-side gesture classification
- [x] Landmark-based movement analysis
- [ ] Connect production frontend to deployed Python service

### Phase 3 — Intelligent drawing agent
- [ ] Pinch / fist / swipe gesture vocabulary
- [ ] Stroke segmentation
- [ ] Shape recognition
- [ ] Handwriting recognition
- [ ] Canvas understanding
- [ ] Natural-language commands
- [ ] AI-assisted drawing cleanup

### Phase 4 — Portfolio-grade AI system
- [ ] Convert sketches into structured diagrams
- [ ] Drawing history and replay
- [ ] Performance telemetry
- [ ] Web Worker experimentation
- [ ] Mobile production deployment
- [ ] Model evaluation and benchmarks
- [ ] Technical architecture and research notes

## Project philosophy

This is intentionally being built as an engineering experiment rather than a tutorial clone. The goal is to explore real-time human-computer interaction, computer vision, Python AI systems, gesture interfaces, rendering, and intelligent interpretation in one product.
