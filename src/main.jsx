import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Camera, Circle, Hand, Play, RotateCcw, Sparkles, Trash2, ShieldCheck, AlertTriangle } from 'lucide-react';
import './styles.css';

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const frameRef = useRef(null);
  const previousPointRef = useRef(null);
  const lastTimestampRef = useRef(-1);
  const smoothingRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [tracking, setTracking] = useState(false);
  const [loading, setLoading] = useState(false);
  const [visionLoading, setVisionLoading] = useState(false);
  const [error, setError] = useState('');
  const [permissionState, setPermissionState] = useState('prompt');
  const [brushSize, setBrushSize] = useState(6);
  const [brushColor, setBrushColor] = useState('#ffffff');
  const [strokes, setStrokes] = useState(0);

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    previousPointRef.current = null;
    smoothingRef.current = null;
    setStrokes(0);
  }, []);

  const isIndexOnly = (landmarks) => {
    if (!landmarks || landmarks.length < 21) return false;
    const indexExtended = landmarks[8].y < landmarks[6].y;
    const middleFolded = landmarks[12].y > landmarks[10].y;
    const ringFolded = landmarks[16].y > landmarks[14].y;
    const pinkyFolded = landmarks[20].y > landmarks[18].y;
    return indexExtended && middleFolded && ringFolded && pinkyFolded;
  };

  const drawFromHand = useCallback((result) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const hand = result?.landmarks?.[0];
    if (!hand || !isIndexOnly(hand)) {
      previousPointRef.current = null;
      smoothingRef.current = null;
      setTracking(false);
      return;
    }

    const tip = hand[8];
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const rawX = (1 - tip.x) * width;
    const rawY = tip.y * height;
    const previous = smoothingRef.current || { x: rawX, y: rawY };
    const smooth = {
      x: previous.x * 0.72 + rawX * 0.28,
      y: previous.y * 0.72 + rawY * 0.28,
    };
    smoothingRef.current = smooth;

    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;

    if (previousPointRef.current) {
      ctx.beginPath();
      ctx.moveTo(previousPointRef.current.x, previousPointRef.current.y);
      ctx.lineTo(smooth.x, smooth.y);
      ctx.stroke();
      setStrokes((value) => value + 1);
    }

    previousPointRef.current = smooth;
    setTracking(true);
  }, [brushColor, brushSize]);

  const detectLoop = useCallback(() => {
    const video = videoRef.current;
    const landmarker = handLandmarkerRef.current;
    if (!video || !cameraOn) return;

    if (!landmarker || video.readyState < 2) {
      frameRef.current = requestAnimationFrame(detectLoop);
      return;
    }

    const timestamp = performance.now();
    if (timestamp <= lastTimestampRef.current) {
      frameRef.current = requestAnimationFrame(detectLoop);
      return;
    }
    lastTimestampRef.current = timestamp;

    try {
      const result = landmarker.detectForVideo(video, timestamp);
      drawFromHand(result);
    } catch (err) {
      console.error('Hand detection error:', err);
    }

    frameRef.current = requestAnimationFrame(detectLoop);
  }, [cameraOn, drawFromHand]);

  const initializeHandLandmarker = async () => {
    if (handLandmarkerRef.current) return handLandmarkerRef.current;
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    handLandmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });
    return handLandmarkerRef.current;
  };

  const startCamera = async () => {
    if (loading) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not available here. Open this site in a modern browser over HTTPS.');
      return;
    }

    if (!window.isSecureContext) {
      setError('Camera access requires HTTPS. Open the deployed site instead of an insecure HTTP page.');
      return;
    }

    try {
      setError('');
      setLoading(true);

      // Camera permission must be requested directly from the user's button click.
      // Do not initialize MediaPipe before this request: the camera should appear
      // immediately after the browser permission is granted.
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: { ideal: 'user' },
        },
        audio: false,
      });

      setPermissionState('granted');
      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) throw new Error('Camera preview element is unavailable.');

      video.srcObject = stream;
      await video.play();

      // Camera is now considered live. The AI model loads independently.
      setCameraOn(true);
      setLoading(false);
      requestAnimationFrame(setupCanvas);

      const [track] = stream.getVideoTracks();
      if (track) {
        track.addEventListener('ended', () => {
          setCameraOn(false);
          setTracking(false);
          setError('The camera was stopped by the browser or device. Press Open Camera to reconnect.');
        }, { once: true });
      }

      // Do not block camera display while MediaPipe downloads/initializes.
      setVisionLoading(true);
      try {
        await initializeHandLandmarker();
      } catch (visionError) {
        console.error('MediaPipe initialization error:', visionError);
        setError('Camera is working, but hand tracking could not load. Refresh the page and try again.');
      } finally {
        setVisionLoading(false);
      }
    } catch (err) {
      console.error('Camera startup error:', err);
      setCameraOn(false);

      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setPermissionState('denied');
        setError('Camera permission is blocked. Allow camera access for this site, then press Open Camera again.');
      } else if (err?.name === 'NotFoundError' || err?.name === 'DevicesNotFoundError') {
        setError('No camera was found on this device. Connect a camera and try again.');
      } else if (err?.name === 'NotReadableError' || err?.name === 'TrackStartError') {
        setError('The camera is already being used by another app or browser tab. Close it and try again.');
      } else if (err?.name === 'OverconstrainedError') {
        setError('The selected camera settings are unavailable. Try again or choose another camera in your browser.');
      } else {
        setError('We could not start the camera. Check browser camera permissions and try again.');
      }

      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    } finally {
      setLoading(false);
    }
  };

  const stopCamera = useCallback(() => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    frameRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    previousPointRef.current = null;
    smoothingRef.current = null;
    setTracking(false);
    setCameraOn(false);
    setVisionLoading(false);
  }, []);

  const downloadDrawing = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `aircanvas-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  useEffect(() => {
    if (cameraOn) {
      frameRef.current = requestAnimationFrame(detectLoop);
    }
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
    };
  }, [cameraOn, detectLoop]);

  useEffect(() => {
    const onResize = () => setupCanvas();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [setupCanvas]);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    handLandmarkerRef.current?.close();
  }, []);

  const permissionMessage = permissionState === 'denied'
    ? 'Camera permission is blocked. Allow access in your browser site settings, then press Open Camera.'
    : 'Click Open Camera. Your browser will ask for permission before the camera starts.';

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Sparkles size={16} /></div>
          <div>
            <strong>AirCanvas</strong>
            <span>AI PASSION PROJECT / 001</span>
          </div>
        </div>
        <div className={`system-status ${cameraOn ? 'live' : ''}`}>
          <span className="status-dot" />
          {cameraOn ? 'VISION SYSTEM LIVE' : 'SYSTEM STANDBY'}
        </div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">REAL-TIME COMPUTER VISION</p>
          <h1>Draw with<br /><em>your hand.</em></h1>
          <p className="lede">Point one finger at the camera. AirCanvas tracks your fingertip and turns every movement into a live digital stroke.</p>

          <div className="controls">
            {!cameraOn ? (
              <button className="primary" onClick={startCamera} disabled={loading}>
                {loading ? <Circle className="spin" size={17} /> : <Camera size={17} />}
                {loading ? 'OPENING CAMERA...' : permissionState === 'denied' ? 'TRY CAMERA AGAIN' : 'OPEN CAMERA'}
              </button>
            ) : (
              <button className="primary stop" onClick={stopCamera}><Camera size={17} /> STOP CAMERA</button>
            )}
            <button className="secondary" onClick={clearCanvas}><Trash2 size={17} /> CLEAR</button>
            <button className="secondary" onClick={downloadDrawing}>EXPORT PNG</button>
          </div>

          {!cameraOn && !error && (
            <div className="permission-note">
              <ShieldCheck size={16} />
              <span>{permissionMessage}</span>
            </div>
          )}
          {error && (
            <div className="error-box">
              <AlertTriangle size={15} />
              <p>{error}</p>
            </div>
          )}
        </div>

        <div className="workspace">
          <div className="workspace-head">
            <div className="workspace-label"><Hand size={15} /> LIVE DRAWING SURFACE</div>
            <div className="metrics"><span>{tracking ? 'TRACKING INDEX' : visionLoading ? 'LOADING VISION' : cameraOn ? 'CAMERA READY' : 'WAITING FOR CAMERA'}</span><b>{strokes} pts</b></div>
          </div>

          <div className="stage">
            <video ref={videoRef} className="camera" playsInline muted autoPlay />
            <canvas ref={canvasRef} className="drawing-layer" />

            {!cameraOn && (
              <div className="stage-empty">
                <div className="empty-icon"><Camera size={34} strokeWidth={1.3} /></div>
                <strong>Open your camera to start</strong>
                <span>Press Open Camera below and allow camera access when your browser asks.</span>
                <button className="stage-cta" onClick={startCamera} disabled={loading}>
                  {loading ? 'Opening camera...' : 'Open Camera'}
                </button>
              </div>
            )}

            {cameraOn && visionLoading && (
              <div className="vision-loading">
                <Circle className="spin" size={15} />
                <span>Camera is live · loading hand tracking...</span>
              </div>
            )}

            <div className={`tracking-pill ${tracking ? 'active' : ''}`}>
              <span /> {tracking ? 'INDEX DETECTED' : cameraOn ? 'SHOW ONE FINGER' : 'CAMERA OFF'}
            </div>
          </div>
        </div>
      </section>

      <section className="tool-panel">
        <div className="tool-group">
          <span className="tool-title">BRUSH</span>
          <div className="brush-row">
            {[3, 6, 10, 16].map((size) => (
              <button key={size} className={`brush ${brushSize === size ? 'selected' : ''}`} onClick={() => setBrushSize(size)}>
                <span style={{ width: size, height: size }} />
              </button>
            ))}
          </div>
        </div>
        <div className="tool-group">
          <span className="tool-title">INK</span>
          <div className="color-row">
            {['#ffffff', '#8bffb0', '#75a7ff', '#ff7e9f', '#ffd166'].map((color) => (
              <button key={color} aria-label={`Select ${color}`} className={`color ${brushColor === color ? 'selected' : ''}`} style={{ background: color }} onClick={() => setBrushColor(color)} />
            ))}
          </div>
        </div>
        <div className="gesture-note">
          <RotateCcw size={16} />
          <span><b>Gesture:</b> one finger extended = draw · fold it = pause</span>
        </div>
        <div className="tech-stack">MEDIA PIPE · CANVAS 2D · REACT · JAVASCRIPT</div>
      </section>

      <footer className="footer">Made by - Amar</footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
