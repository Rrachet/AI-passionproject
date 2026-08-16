# AirCanvas Architecture

```text
Phone / Laptop Camera
        |
        v
React + Canvas + MediaPipe landmark extraction
        |
        | 21 normalized hand landmarks
        v
FastAPI + WebSocket (Python)
        |
        +--> gesture classification
        +--> trajectory analysis
        +--> future CV models
        +--> future AI agent
        |
        v
Drawing / AI interpretation
```

## Design principle

JavaScript is the interaction layer. Python is the intelligence layer. We intentionally keep the camera stream local to the browser and send compact landmark data to Python rather than uploading raw video frames on every tick.
