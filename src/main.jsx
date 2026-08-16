import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Camera, Circle, Hand, Play, Sparkles, Trash2, ShieldCheck } from 'lucide-react';
import './styles.css';

const MP_VERSION = '0.10.35';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const TARGET_FPS = 45;

// Geometry-based finger detection is more tolerant of hand rotation than checking Y only.
function fingerIsExtended(hand, mcp, pip, tip) {
  const a = { x: hand[mcp].x - hand[pip].x, y: hand[mcp].y - hand[pip].y };
  const b = { x: hand[tip].x - hand[pip].x, y: hand[tip].y - hand[pip].y };
  const dot = a.x * b.x + a.y * b.y;
  const mag = Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y) || 1;
  const angle = Math.acos(Math.max(-1, Math.min(1, dot / mag))) * 180 / Math.PI;
  const tipFromMcp = Math.hypot(hand[tip].x - hand[mcp].x, hand[tip].y - hand[mcp].y);
  const pipFromMcp = Math.hypot(hand[pip].x - hand[mcp].x, hand[pip].y - hand[mcp].y);
  return angle > 145 && tipFromMcp > pipFromMcp * 1.12;
}

function classifyGesture(hand) {
  if (!hand || hand.length < 21) return 'none';
  const index = fingerIsExtended(hand, 5, 6, 8);
  const middle = fingerIsExtended(hand, 9, 10, 12);
  const ring = fingerIsExtended(hand, 13, 14, 16);
  const pinky = fingerIsExtended(hand, 17, 18, 20);
  if (index && !middle && !ring && !pinky) return 'draw';
  if (!index && !middle && !ring && !pinky) return 'pause';
  if (index && middle && ring && pinky) return 'clear';
  return 'none';
}

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const frameRef = useRef(null);
  const previousPointRef = useRef(null);
  const smoothPointRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastDetectionRef = useRef(0);
  const clearHoldRef = useRef(0);
  const lastClearRef = useRef(0);
  const mountedRef = useRef(true);

  const [cameraOn, setCameraOn] = useState(false);
  const [visionReady, setVisionReady] = useState(false);
  const [tracking, setTracking] = useState(false);
  const [loadingCamera, setLoadingCamera] = useState(false);
  const [loadingVision, setLoadingVision] = useState(false);
  const [error, setError] = useState('');
  const [permissionState, setPermissionState] = useState('prompt');
  const [brushSize, setBrushSize] = useState(6);
  const [brushColor, setBrushColor] = useState('#ffffff');
  const [strokes, setStrokes] = useState(0);
  const [gesture, setGesture] = useState('none');
  const [pointer, setPointer] = useState(null);

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const rect = video.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr));
    canvas.height = Math.max(1, Math.round(rect.height * dpr));
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
    canvas.getContext('2d').clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    previousPointRef.current = null;
    smoothPointRef.current = null;
    setStrokes(0);
  }, []);

  const handleGesture = useCallback((nextGesture) => {
    const now = performance.now();
    setGesture(nextGesture);
    if (nextGesture !== 'draw') {
      previousPointRef.current = null;
      smoothPointRef.current = null;
      setTracking(false);
    }
    if (nextGesture === 'clear') {
      if (!clearHoldRef.current) clearHoldRef.current = now;
      if (now - clearHoldRef.current >= 850 && now - lastClearRef.current >= 1400) {
        lastClearRef.current = now;
        clearHoldRef.current = 0;
        clearCanvas();
      }
    } else {
      clearHoldRef.current = 0;
    }
  }, [clearCanvas]);

  const drawResult = useCallback((result) => {
    const canvas = canvasRef.current;
    const hand = result?.landmarks?.[0];
    if (!canvas || !hand) {
      handleGesture('none');
      setPointer(null);
      return;
    }

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const tip = hand[8];
    const rawPoint = { x: (1 - tip.x) * width, y: tip.y * height };
    const previousSmooth = smoothPointRef.current || rawPoint;
    const distance = Math.hypot(rawPoint.x - previousSmooth.x, rawPoint.y - previousSmooth.y);
    // Adaptive smoothing: high alpha for fast movement, lower alpha for tiny jitter.
    const alpha = distance > 55 ? 0.86 : distance > 18 ? 0.72 : 0.55;
    const point = {
      x: previousSmooth.x + (rawPoint.x - previousSmooth.x) * alpha,
      y: previousSmooth.y + (rawPoint.y - previousSmooth.y) * alpha,
    };
    smoothPointRef.current = point;
    setPointer(point);

    const nextGesture = classifyGesture(hand);
    handleGesture(nextGesture);
    if (nextGesture !== 'draw') return;

    const ctx = canvas.getContext('2d');
    ctx.strokeStyle = brushColor;
    ctx.lineWidth = brushSize;
    if (previousPointRef.current) {
      ctx.beginPath();
      ctx.moveTo(previousPointRef.current.x, previousPointRef.current.y);
      ctx.lineTo(point.x, point.y);
      ctx.stroke();
      setStrokes((n) => n + 1);
    }
    previousPointRef.current = point;
    setTracking(true);
  }, [brushColor, brushSize, handleGesture]);

  const processFrames = useCallback(() => {
    if (!cameraOn) return;
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    const now = performance.now();
    const minInterval = 1000 / TARGET_FPS;
    if (video && landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTimeRef.current && now - lastDetectionRef.current >= minInterval) {
      lastVideoTimeRef.current = video.currentTime;
      lastDetectionRef.current = now;
      try {
        drawResult(landmarker.detectForVideo(video, now));
      } catch (err) {
        console.error('MediaPipe frame error:', err);
      }
    }
    frameRef.current = requestAnimationFrame(processFrames);
  }, [cameraOn, drawResult]);

  const createLandmarker = async (delegate) => {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    return HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.28,
      minHandPresenceConfidence: 0.28,
      minTrackingConfidence: 0.28,
    });
  };

  const initializeVision = async () => {
    if (landmarkerRef.current) return landmarkerRef.current;
    setLoadingVision(true);
    setError('');
    try {
      try {
        landmarkerRef.current = await createLandmarker('GPU');
      } catch (gpuError) {
        console.warn('GPU unavailable; falling back to CPU.', gpuError);
        landmarkerRef.current = await createLandmarker('CPU');
      }
      if (mountedRef.current) {
        setVisionReady(true);
        setLoadingVision(false);
      }
      return landmarkerRef.current;
    } catch (err) {
      console.error('MediaPipe initialization failed:', err);
      landmarkerRef.current = null;
      if (mountedRef.current) {
        setLoadingVision(false);
        setVisionReady(false);
        setError(`Hand tracking failed to load: ${err?.message || 'MediaPipe could not initialize.'}`);
      }
      return null;
    }
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return setError('Camera access is not supported by this browser. Use a current version of Chrome or Safari.');
    if (!window.isSecureContext) return setError('Camera access requires HTTPS. Open the deployed AirCanvas website.');
    try {
      setError('');
      setLoadingCamera(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'user' }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      setPermissionState('granted');
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      setCameraOn(true);
      setLoadingCamera(false);
      requestAnimationFrame(setupCanvas);
      initializeVision();
    } catch (err) {
      console.error('Camera startup failed:', err);
      setLoadingCamera(false);
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') {
        setPermissionState('denied');
        setError('Camera permission was denied. Allow camera access in browser settings and press Open Camera again.');
      } else if (err?.name === 'NotFoundError') setError('No camera was found on this device.');
      else if (err?.name === 'NotReadableError') setError('The camera is already being used by another application or browser tab.');
      else setError(`Camera could not start: ${err?.message || 'unknown error'}`);
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
    smoothPointRef.current = null;
    lastVideoTimeRef.current = -1;
    lastDetectionRef.current = 0;
    clearHoldRef.current = 0;
    setTracking(false);
    setGesture('none');
    setPointer(null);
    setCameraOn(false);
    setVisionReady(false);
    setLoadingVision(false);
  };

  const exportDrawing = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `aircanvas-${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
  };

  useEffect(() => {
    if (cameraOn) frameRef.current = requestAnimationFrame(processFrames);
    return () => frameRef.current && cancelAnimationFrame(frameRef.current);
  }, [cameraOn, processFrames]);
  useEffect(() => { window.addEventListener('resize', setupCanvas); return () => window.removeEventListener('resize', setupCanvas); }, [setupCanvas]);
  useEffect(() => () => {
    mountedRef.current = false;
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    landmarkerRef.current?.close();
  }, []);

  const gestureLabel = gesture === 'draw' ? 'DRAWING' : gesture === 'pause' ? 'PAUSED — FIST' : gesture === 'clear' ? 'CLEAR — HOLD PALM' : visionReady ? 'SHOW A GESTURE' : loadingVision ? 'VISION LOADING' : 'VISION OFFLINE';

  return (
    <main className="app-shell">
      <header className="topbar"><div className="brand"><div className="brand-mark"><Sparkles size={16} /></div><div><strong>AirCanvas</strong><span>AI PASSION PROJECT / 001</span></div></div><div className={`system-status ${cameraOn ? 'live' : ''}`}><span className="status-dot" />{cameraOn ? 'VISION SYSTEM LIVE' : 'SYSTEM STANDBY'}</div></header>
      <section className="hero">
        <div className="hero-copy"><p className="eyebrow">REAL-TIME COMPUTER VISION</p><h1>Draw with<br /><em>your hand.</em></h1><p className="lede">Write naturally in the air. AirCanvas follows your index fingertip and turns your movement into a clean digital stroke.</p>
          <div className="controls">{!cameraOn ? <button className="primary" onClick={startCamera} disabled={loadingCamera}>{loadingCamera ? <Circle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{loadingCamera ? 'OPENING CAMERA' : permissionState === 'denied' ? 'TRY CAMERA AGAIN' : 'OPEN CAMERA'}</button> : <button className="primary stop" onClick={stopCamera}><Camera size={17} /> STOP CAMERA</button>}<button className="secondary" onClick={clearCanvas}><Trash2 size={17} /> CLEAR</button><button className="secondary" onClick={exportDrawing}>EXPORT PNG</button></div>
          {!cameraOn && !error && <div className="permission-note"><ShieldCheck size={16} /><span>Press Open Camera. Your browser will ask for permission.</span></div>}{error && <p className="error-box">{error}</p>}
        </div>
        <div className="workspace"><div className="workspace-head"><div className="workspace-label"><Hand size={15} /> LIVE DRAWING SURFACE</div><div className="metrics"><span>{tracking ? 'FINGERTIP LOCKED' : gestureLabel}</span><b>{strokes} pts</b></div></div>
          <div className="stage"><video ref={videoRef} className="camera" playsInline muted autoPlay /><canvas ref={canvasRef} className="drawing-layer" />{pointer && cameraOn && <div className={`finger-cursor ${tracking ? 'drawing' : ''}`} style={{ left: pointer.x, top: pointer.y }}><span /></div>}{!cameraOn && <div className="stage-empty"><div className="empty-icon"><Hand size={34} strokeWidth={1.3} /></div><strong>Open your camera to start</strong><span>Press Open Camera and allow access.</span></div>}{cameraOn && loadingVision && !error && <div className="vision-loading"><Circle className="spin" size={18} /><span>Learning your hand…</span></div>}<div className={`tracking-pill ${tracking ? 'active' : ''}`}><span />{tracking ? 'FINGERTIP LOCKED · DRAWING' : gestureLabel}</div></div>
        </div>
      </section>
      <section className="gesture-guide"><div className="guide-heading"><span>LEARN THE GESTURES</span><b>Start simple. Improve every session.</b></div><div className="gesture-grid"><div className={`gesture-card ${gesture === 'draw' ? 'active' : ''}`}><div className="gesture-visual">☝</div><div><strong>DRAW</strong><span>Raise one index finger and move slowly to start writing.</span></div></div><div className={`gesture-card ${gesture === 'pause' ? 'active' : ''}`}><div className="gesture-visual">✊</div><div><strong>PAUSE</strong><span>Make a fist. Your stroke stops without clearing the canvas.</span></div></div><div className={`gesture-card ${gesture === 'clear' ? 'active' : ''}`}><div className="gesture-visual">🖐</div><div><strong>CLEAR</strong><span>Hold an open palm for about one second to clear.</span></div></div></div></section>
      <section className="tool-panel"><div className="tool-group"><span className="tool-title">BRUSH</span><div className="brush-row">{[3, 6, 10, 16].map((size) => <button key={size} className={`brush ${brushSize === size ? 'selected' : ''}`} onClick={() => setBrushSize(size)}><span style={{ width: size, height: size }} /></button>)}</div></div><div className="tool-group"><span className="tool-title">INK</span><div className="color-row">{['#ffffff', '#8bffb0', '#75a7ff', '#ff7e9f', '#ffd166'].map((color) => <button key={color} aria-label={`Select ${color}`} className={`color ${brushColor === color ? 'selected' : ''}`} style={{ background: color }} onClick={() => setBrushColor(color)} />)}</div></div><div className="tech-stack">MEDIAPIPE · CANVAS 2D · REACT · JAVASCRIPT</div></section>
      <footer className="footer">Made by - Amar</footer>
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
