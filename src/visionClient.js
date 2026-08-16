const DEFAULT_WS_URL = 'ws://localhost:8000/ws/vision';

let socket = null;
let latestGesture = 'UNKNOWN';

export function connectVisionEngine() {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;

  const url = import.meta.env.VITE_VISION_WS_URL || DEFAULT_WS_URL;
  try {
    socket = new WebSocket(url);
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        latestGesture = payload.gesture || 'UNKNOWN';
      } catch {
        latestGesture = 'UNKNOWN';
      }
    };
    socket.onclose = () => {
      socket = null;
    };
  } catch {
    socket = null;
  }
}

export function sendHandLandmarks(landmarks) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  socket.send(JSON.stringify({
    landmarks: landmarks.map(({ x, y, z }) => ({ x, y, z })),
  }));
}

export function getLatestGesture() {
  return latestGesture;
}

export function disconnectVisionEngine() {
  socket?.close();
  socket = null;
  latestGesture = 'UNKNOWN';
}
