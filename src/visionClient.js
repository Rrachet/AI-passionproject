const DEFAULT_WS_URL = 'ws://localhost:8000/ws/vision';

let socket = null;
let latestResult = null;
let status = 'disconnected';

export function connectVisionEngine(onResult, onStatus) {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
  const url = import.meta.env.VITE_VISION_WS_URL || DEFAULT_WS_URL;
  try {
    socket = new WebSocket(url);
    status = 'connecting';
    onStatus?.(status);
    socket.onopen = () => { status = 'connected'; onStatus?.(status); };
    socket.onmessage = (event) => {
      try {
        latestResult = JSON.parse(event.data);
        onResult?.(latestResult);
      } catch { latestResult = null; }
    };
    socket.onerror = () => { status = 'error'; onStatus?.(status); };
    socket.onclose = () => { status = 'disconnected'; socket = null; onStatus?.(status); };
  } catch {
    socket = null;
    status = 'error';
    onStatus?.(status);
  }
}

export function sendHandLandmarks(landmarks, timestamp, confidence = 1) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return false;
  socket.send(JSON.stringify({
    landmarks: landmarks.map(({ x, y, z }) => ({ x, y, z })),
    timestamp,
    confidence,
  }));
  return true;
}

export function getLatestVisionResult() { return latestResult; }
export function getVisionStatus() { return status; }
export function disconnectVisionEngine() {
  socket?.close();
  socket = null;
  latestResult = null;
  status = 'disconnected';
}
