import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Camera, Circle, Hand, Play, RotateCcw, Sparkles, Trash2, ShieldCheck } from 'lucide-react';
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
  const mountedRef = useRef(true);

  const [cameraOn, setCameraOn] = useState(false);
  const [tracking, setTracking] = useState(false);
  const [visionReady, setVisionReady] = useState(false);
  const [loading, setLoading] = useState(false);
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
    const existing = canvas.toDataURL();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (strokes > 0 && existing !== 'data:,') {
      const image = new Image();
      image.onload = () => ctx.drawImage(image, 0, 0, rect.width, rect.height);
      image.src = existing;
    }
  }, [strokes]);

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
    return landmarks[8].y < landmarks[6].y && landmarks[12].y > landmarks[10].y && landmarks[16].y > landmarks[14].y && landmarks[20].y > landmarks[18].y;
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
    const smooth = { x: previous.x * 0.72 + rawX * 0.28, y: previous.y * 0.72 + rawY * 0.28 };
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
    if (!cameraOn) return;
    const video = videoRef.current;
    const landmarker = handLandmarkerRef.current;

    // Keep the loop alive while the AI model is downloading/initializing.
    if (video && landmarker && video.readyState >= 2) {
      const timestamp = performance.now();
      if (timestamp > lastTimestampRef.current) {
        lastTimestampRef.current = timestamp;
        try {
          const result = landmarker.detectForVideo(video, timestamp);
          drawFromHand(result);
        } catch (err) {
          console.error('Hand detection error:', err);
        }
      }
    }
    frameRef.current = requestAnimationFrame(detectLoop);
  }, [cameraOn, drawFromHand]);

  const initializeHandLandmarker = async () => {
    if (handLandmarkerRef.current) return handLandmarkerRef.current;
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    const landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    handLandmarkerRef.current = landmarker;
    if (mountedRef.current) setVisionReady(true);
    return landmarker;
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not supported by this browser. Use Chrome or Safari.');
      return;
    }
    if (!window.isSecureContext) {
      setError('Camera access requires HTTPS. Open the deployed AirCanvas link.');
      return;
    }

    try {
      setError('');
      setLoading(true);
      setVisionReady(false);
      setPermissionState('prompt');

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: 'user' } },
        audio: false,
      });

      setPermissionState('granted');
      streamRef.current = stream;
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      // Camera is intentionally made visible immediately after permission.
      setCameraOn(true);
      requestAnimationFrame(setupCanvas);
      setLoading(false);

      // Vision initializes independently; the drawing loop stays alive meanwhile.
      initializeHandLandmarker().catch((err) => {
        console.error('Vision initialization error:', err);
        if (mountedRef.current) {
          setVisionReady(false);
          setError('Camera is working, but hand tracking could not load. Check your internet connection and refresh the page.');
        }
      });
    } catch (err) {
      console.error('Camera startup error:', err);
      setCameraOn(false);
      setLoading(false);
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setPermissionState('denied');
        setError('Camera permission was denied. Allow camera access in your browser settings, then press Open Camera again.');
      } else if (err?.name === 'NotFoundError') setError('No camera was found on this device.');
      else if (err?.name === 'NotReadableError') setError('The camera is already being used by another app or browser tab.');
      else setError('We could not start the camera. Check your browser permissions and try again.');
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  const stopCamera = () => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    previousPointRef.current = null;
    smoothingRef.current = null;
    setTracking(false);
    setCameraOn(false);
    setVisionReady(false);
  };

  const downloadDrawing = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `aircanvas-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  useEffect(() => {
    if (cameraOn) frameRef.current = requestAnimationFrame(detectLoop);
    return () => { if (frameRef.current) cancelAnimationFrame(frameRef.current); };
  }, [cameraOn, detectLoop]);

  useEffect(() => {
    const onResize = () => setupCanvas();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [setupCanvas]);

  useEffect(() => () => {
    mountedRef.current = false;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    handLandmarkerRef.current?.close();
  }, []);

  const permissionMessage = permissionState === 'denied'
    ? 'Camera permission is blocked. Enable it in your browser settings, then try again.'
    : 'Press Open Camera. Your browser will ask for permission.';

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><Sparkles size={16} /></div><div><strong>AirCanvas</strong><span>AI PASSION PROJECT / 001</span></div></div>
        <div className={`system-status ${cameraOn ? 'live' : ''}`}><span className="status-dot" />{cameraOn ? 'VISION SYSTEM LIVE' : 'SYSTEM STANDBY'}</div>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">REAL-TIME COMPUTER VISION</p>
          <h1>Draw with<br /><em>your hand.</em></h1>
          <p className="lede">Point one finger at the camera. AirCanvas tracks your fingertip and turns every movement into a live digital stroke.</p>
          <div className="controls">
            {!cameraOn ? <button className="primary" onClick={startCamera} disabled={loading}>{loading ? <Circle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{loading ? 'OPENING CAMERA' : permissionState === 'denied' ? 'TRY CAMERA AGAIN' : 'OPEN CAMERA'}</button> : <button className="primary stop" onClick={stopCamera}><Camera size={17} /> STOP CAMERA</button>}
            <button className="secondary" onClick={clearCanvas}><Trash2 size={17} /> CLEAR</button>
            <button className="secondary" onClick={downloadDrawing}>EXPORT PNG</button>
          </div>
          {!cameraOn && !error && <div className="permission-note"><ShieldCheck size={16} /><span>{permissionMessage}</span></div>}
          {error && <p className="error">{error}</p>}
        </div>

        <div className="workspace">
          <div className="workspace-head"><div className="workspace-label"><Hand size={15} /> LIVE DRAWING SURFACE</div><div className="metrics"><span>{tracking ? 'TRACKING INDEX' : cameraOn && !visionReady ? 'LOADING VISION' : 'WAITING FOR HAND'}</span><b>{strokes} pts</b></div></div>
          <div className="stage">
            <video ref={videoRef} className="camera" playsInline muted />
            <canvas ref={canvasRef} className="drawing-layer" />
            {!cameraOn && <div className="stage-empty"><div className="empty-icon"><Hand size={34} strokeWidth={1.3} /></div><strong>Open your camera to start</strong><span>Press Open Camera and allow access.</span></div>}
            {cameraOn && !visionReady && !error && <div className="vision-loading"><Circle className="spin" size={18} /><span>Loading hand tracking…</span></div>}
            <div className={`tracking-pill ${tracking ? 'active' : ''}`}><span /> {tracking ? 'INDEX DETECTED' : visionReady ? 'SHOW ONE FINGER' : 'VISION LOADING'}</div>
          </div>
        </div>
      </section>

      <section className="tool-panel">
        <div className="tool-group"><span className="tool-title">BRUSH</span><div className="brush-row">{[3, 6, 10, 16].map((size) => <button key={size} className={`brush ${brushSize === size ? 'selected' : ''}`} onClick={() => setBrushSize(size)}><span style={{ width: size, height: size }} /></button>)}</div></div>
        <div className="tool-group"><span className="tool-title">INK</span><div className="color-row">{['#ffffff', '#8bffb0', '#75a7ff', '#ff7e9f', '#ffd166'].map((color) => <button key={color} aria-label={`Select ${color}`} className={`color ${brushColor === color ? 'selected' : ''}`} style={{ background: color }} onClick={() => setBrushColor(color)} />)}</div></div>
        <div className="gesture-note"><RotateCcw size={16} /><span><b>Gesture:</b> one finger extended = draw · fold it = pause</span></div>
        <div className="tech-stack">MEDIA PIPE · CANVAS 2D · REACT · JAVASCRIPT</div>
      </section>

      <footer className="footer">Made by - Amar</footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
