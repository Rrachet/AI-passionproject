# AirCanvas Python Vision Engine

FastAPI service for gesture classification and real-time vision events.

## Local development

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Health check: `http://localhost:8000/health`

WebSocket: `ws://localhost:8000/ws/vision`

The browser remains responsible for camera permissions and interactive rendering. The Python service owns gesture classification and the future AI reasoning layer.
