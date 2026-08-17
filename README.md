# AirCanvas

**A camera-first interface for drawing, pointing and explaining ideas with your hand.**

AirCanvas is a personal computer-vision project by **Amarnath Mishra**. It turns a live camera feed into an interactive canvas: move your index finger and the system follows it in real time.

The project is being built as an engineering experiment, not as a clone of an existing drawing app. The goal is to understand what happens when hand tracking, temporal reasoning and a lightweight AR layer become one interface.

[Live product](https://ai-passionproject.vercel.app) · [Builder](https://www.linkedin.com/in/amarnath-mishra)

---

## Why AirCanvas exists

A mouse and keyboard are excellent for structured input. They are not always the best tools for explaining a shape, pointing at something during a call, or sketching an idea while speaking.

AirCanvas explores a different interaction model:

**See the hand. Understand the movement. Turn the movement into an interface.**

The long-term direction is a browser extension that can place this interaction layer over meetings, calls, presentations and browser content.

---

## What it can do today

| Mode | Purpose |
| --- | --- |
| **Draw** | Follow the index fingertip and create continuous freehand strokes. |
| **Shape** | Turn rough geometry into cleaner lines and polygons. |
| **Pointer** | Use the fingertip as a presentation pointer. |
| **Laser** | Create a temporary visual trail for emphasis. |

The interface is designed around a camera-first workflow. Camera permission stays in the browser, while the canvas renders the interaction locally.

---

## System design

```text
                  Camera
                    |
                    v
          MediaPipe Hand Landmarks
                    |
          +---------+---------+
          |                   |
          v                   v
     Browser Canvas       Python Engine
     low-latency UI       temporal reasoning
          |                   |
          +---------+---------+
                    |
                    v
             AirCanvas Layer
          draw / shape / pointer / laser
```

The browser handles camera access, hand-landmark detection and immediate visual feedback. The Python service is the experimental intelligence layer for temporal filtering, gesture reasoning, movement analysis and shape processing.

Only compact hand-landmark data is intended to cross the browser-to-Python boundary; the architecture does not require continuously uploading raw camera video to the Python service.

---

## Vision engine

The Python side is where the project is deliberately evolving.

Current work includes:

- temporal fingertip tracking
- adaptive smoothing
- velocity estimation
- confidence handling
- short tracking-gap prediction
- temporal gesture voting
- polygon and shape alignment
- WebSocket communication between the browser and Python
- an experimental training path for future learned models

The important engineering principle is simple:

> More Python is not automatically better. Python earns its place when it improves tracking, prediction, evaluation or learning.

The project is therefore being developed through a repeated loop:

**Build → test → observe → measure → improve.**

---

## Repository structure

```text
AI-passionproject/
├── backend/
│   ├── app/
│   │   ├── learning.py
│   │   ├── main.py
│   │   ├── polygon.py
│   │   └── tracker.py
│   └── training/
│       ├── MODEL_STRATEGY.md
│       └── train_temporal.py
├── docs/
│   └── architecture.md
├── public/
│   └── aircanvas-mark.svg
├── src/
│   ├── main.jsx
│   ├── styles.css
│   └── visionClient.js
├── index.html
├── package.json
└── vite.config.js
```

---

## Run the frontend

```bash
npm install
npm run dev
```

Open the local Vite URL in a browser with camera access enabled.

Camera access requires a secure context such as `localhost` or HTTPS.

---

## Run the Python vision service

Create a virtual environment inside `backend`:

```bash
cd backend
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start FastAPI:

```bash
uvicorn app.main:app --reload --port 8000
```

For the local browser-to-Python bridge:

```text
VITE_VISION_WS_URL=ws://localhost:8000/ws/vision
```

The public frontend can fall back to browser-side MediaPipe when the Python WebSocket service is not configured.

---

## Development roadmap

### Interaction

- [x] Camera input
- [x] Index-fingertip tracking
- [x] Gesture-gated drawing
- [x] Browser-side smoothing
- [x] Draw mode
- [x] Shape mode
- [x] Pointer mode
- [x] Laser mode
- [x] Shape alignment

### Python vision layer

- [x] FastAPI service
- [x] WebSocket vision channel
- [x] Server-side gesture classification
- [x] Temporal filtering
- [x] Confidence tracking
- [x] Tracking-gap prediction
- [x] Python shape alignment
- [ ] Production WebSocket deployment
- [ ] Measured tracking benchmark
- [ ] Training dataset and evaluation pipeline

### Intelligent interaction

- [ ] Pinch and swipe vocabulary
- [ ] Stroke segmentation
- [ ] Learned shape classifier
- [ ] Handwriting recognition
- [ ] Canvas understanding
- [ ] Natural-language drawing commands
- [ ] AI-assisted cleanup

### Browser extension

- [ ] Chrome Manifest V3 extension
- [ ] Transparent AR overlay
- [ ] Browser-tab integration
- [ ] Meeting integration
- [ ] Persistent annotations
- [ ] Shared collaboration

---

## Design philosophy

AirCanvas should feel like a real product, not a generated demo.

The interface is intentionally restrained: system typography, a small visual language, direct interaction and the camera as the central product surface.

The same principle applies to the codebase. Features are added because they improve the interaction, not because they make the project look larger.

---

## Built by

**Amarnath Mishra**

Computer science engineer building AirCanvas as an independent passion project around computer vision, interaction design and real-time systems.

[LinkedIn](https://www.linkedin.com/in/amarnath-mishra)

---

## License

MIT License. See `LICENSE` for details.
