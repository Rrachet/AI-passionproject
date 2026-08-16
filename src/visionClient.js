let socket = null;
let latestResult = null;
let status = 'standalone';

export function connectVisionEngine(onResult, onStatus) {
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;

  // Python is optional. Never try localhost from a deployed HTTPS site.
  const url = import.meta.env.VITE_VISION_WS_URL;
  if (!url) {
    status = 'standalone';
    onStatus?.(status);
    return;
  }

  try {
    socket = new WebSocket(url);
    status = 'connecting';
    onStatus?.(status);
    socket.onopen = () => { status = 'connected'; onStatus?.(status); };
    socket.onmessage = (event) => {
      try {
        latestResult = JSON.parse(event.data);
        onResult?.(latestResult);
      } catch {
        latestResult = null;
      }
    };
    socket.onerror = () => { status = 'standalone'; onStatus?.(status); };
    socket.onclose = () => { status = 'standalone'; socket = null; onStatus?.(status); };
  } catch {
    socket = null;
    status = 'standalone';
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
  status = 'standalone';
}
