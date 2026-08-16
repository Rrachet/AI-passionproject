import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Camera, Circle, Hand, Play, Sparkles, Trash2, ShieldCheck } from 'lucide-react';
import './styles.css';

const MP_VERSION = '0.10.35';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const TARGET_FPS = 30;
const LOST_HAND_GRACE_MS = 140;

function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function angleAt(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y }, cb = { x: c.x - b.x, y: c.y - b.y };
  const denom = (Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y)) || 1;
  return Math.acos(Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / denom))) * 180 / Math.PI;
}
function fingerIsExtended(hand, mcp, pip, tip) {
  const angle = angleAt(hand[mcp], hand[pip], hand[tip]);
  return angle > 135 && distance(hand[mcp], hand[tip]) > distance(hand[mcp], hand[pip]) * 1.05;
}
function classifyGesture(hand) {
  if (!hand || hand.length < 21) return 'none';
  const index = fingerIsExtended(hand, 5, 6, 8), middle = fingerIsExtended(hand, 9, 10, 12), ring = fingerIsExtended(hand, 13, 14, 16), pinky = fingerIsExtended(hand, 17, 18, 20);
  if (index && !middle && !ring && !pinky) return 'draw';
  if (!index && !middle && !ring && !pinky) return 'pause';
  if (index && middle && ring && pinky) return 'clear';
  return 'none';
}

function App() {
  const videoRef = useRef(null), canvasRef = useRef(null), landmarkerRef = useRef(null), streamRef = useRef(null), frameRef = useRef(null);
  const previousPointRef = useRef(null), smoothPointRef = useRef(null), lastVideoTimeRef = useRef(-1), lastDetectionRef = useRef(0), lastHandSeenRef = useRef(0);
  const clearHoldRef = useRef(0), lastClearRef = useRef(0), activeGestureRef = useRef('none'), mountedRef = useRef(true);
  const [cameraOn, setCameraOn] = useState(false), [visionReady, setVisionReady] = useState(false), [tracking, setTracking] = useState(false);
  const [loadingCamera, setLoadingCamera] = useState(false), [loadingVision, setLoadingVision] = useState(false), [error, setError] = useState('');
  const [permissionState, setPermissionState] = useState('prompt'), [brushSize, setBrushSize] = useState(6), [brushColor, setBrushColor] = useState('#ffffff');
  const [strokes, setStrokes] = useState(0), [gesture, setGesture] = useState('none'), [pointer, setPointer] = useState(null), [velocity, setVelocity] = useState(0);

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current, video = videoRef.current; if (!canvas || !video) return;
    const rect = video.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr));
    canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    canvas.getContext('2d').clearRect(0, 0, canvas.clientWidth, canvas.clientHeight); previousPointRef.current = null; smoothPointRef.current = null; setStrokes(0);
  }, []);

  const renderStroke = useCallback((from, to) => {
    const canvas = canvasRef.current; if (!canvas) return;
    const ctx = canvas.getContext('2d'), d = distance(from, to), steps = Math.max(1, Math.ceil(d / 6)), speed = Math.min(1, d / 45);
    ctx.strokeStyle = brushColor; ctx.lineWidth = Math.max(2, brushSize * (1.05 - speed * 0.32)); ctx.shadowColor = brushColor; ctx.shadowBlur = 7 + speed * 13;
    ctx.beginPath(); ctx.moveTo(from.x, from.y);
    for (let i = 1; i <= steps; i += 1) { const t = i / steps; ctx.lineTo(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t); }
    ctx.stroke(); ctx.shadowBlur = 0;
  }, [brushColor, brushSize]);

  const drawResult = useCallback((result) => {
    const canvas = canvasRef.current, hand = result?.landmarks?.[0], now = performance.now();
    if (!canvas || !hand) {
      if (now - lastHandSeenRef.current > LOST_HAND_GRACE_MS) { previousPointRef.current = null; smoothPointRef.current = null; setTracking(false); setPointer(null); activeGestureRef.current = 'none'; setGesture('none'); }
      return;
    }
    lastHandSeenRef.current = now;
    const width = canvas.clientWidth, height = canvas.clientHeight, tip = hand[8];
    const raw = { x: (1 - tip.x) * width, y: tip.y * height }, prev = smoothPointRef.current || raw, moved = distance(raw, prev);
    const alpha = moved > 70 ? 0.94 : moved > 30 ? 0.84 : moved > 10 ? 0.68 : 0.48;
    const point = { x: prev.x + (raw.x - prev.x) * alpha, y: prev.y + (raw.y - prev.y) * alpha };
    smoothPointRef.current = point; setPointer(point); setVelocity(Math.min(1, moved / 45));
    const detected = classifyGesture(hand);

    if (detected === 'draw') {
      activeGestureRef.current = 'draw'; setGesture('draw');
      if (previousPointRef.current) { renderStroke(previousPointRef.current, point); setStrokes((n) => n + 1); }
      previousPointRef.current = point; setTracking(true); return;
    }
    if (detected === 'pause') {
      activeGestureRef.current = 'pause'; setGesture('pause'); previousPointRef.current = null; smoothPointRef.current = null; setTracking(false); clearHoldRef.current = 0; return;
    }
    if (detected === 'clear') {
      activeGestureRef.current = 'clear'; setGesture('clear'); previousPointRef.current = null; smoothPointRef.current = null; setTracking(false);
      if (!clearHoldRef.current) clearHoldRef.current = now;
      if (now - clearHoldRef.current >= 800 && now - lastClearRef.current >= 1400) { lastClearRef.current = now; clearHoldRef.current = 0; clearCanvas(); }
      return;
    }
    if (activeGestureRef.current === 'draw' && now - lastHandSeenRef.current < 100) return;
    activeGestureRef.current = 'none'; setGesture('none'); previousPointRef.current = null; smoothPointRef.current = null; setTracking(false); clearHoldRef.current = 0;
  }, [clearCanvas, renderStroke]);

  const processFrames = useCallback(() => {
    if (!cameraOn) return;
    const video = videoRef.current, landmarker = landmarkerRef.current, now = performance.now();
    if (video && landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTimeRef.current && now - lastDetectionRef.current >= 1000 / TARGET_FPS) {
      lastVideoTimeRef.current = video.currentTime; lastDetectionRef.current = now;
      try { drawResult(landmarker.detectForVideo(video, now)); } catch (err) { console.error('MediaPipe frame error:', err); }
    }
    frameRef.current = requestAnimationFrame(processFrames);
  }, [cameraOn, drawResult]);

  const createLandmarker = async (delegate) => {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    return HandLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath: MODEL_URL, delegate }, runningMode: 'VIDEO', numHands: 1, minHandDetectionConfidence: 0.28, minHandPresenceConfidence: 0.28, minTrackingConfidence: 0.28 });
  };

  const initializeVision = async () => {
    if (landmarkerRef.current) return landmarkerRef.current;
    setLoadingVision(true); setError('');
    try {
      try { landmarkerRef.current = await createLandmarker('GPU'); } catch (gpuError) { console.warn('GPU unavailable; falling back to CPU.', gpuError); landmarkerRef.current = await createLandmarker('CPU'); }
      if (mountedRef.current) { setVisionReady(true); setLoadingVision(false); }
      return landmarkerRef.current;
    } catch (err) { console.error('MediaPipe initialization failed:', err); landmarkerRef.current = null; if (mountedRef.current) { setLoadingVision(false); setVisionReady(false); setError(`Hand tracking failed to load: ${err?.message || 'MediaPipe could not initialize.'}`); } return null; }
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return setError('Camera access is not supported by this browser. Use a current version of Chrome or Safari.');
    if (!window.isSecureContext) return setError('Camera access requires HTTPS. Open the deployed AirCanvas website.');
    try {
      setError(''); setLoadingCamera(true);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'user' }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }, audio: false });
      streamRef.current = stream; setPermissionState('granted'); const video = videoRef.current; video.srcObject = stream; await video.play();
      setCameraOn(true); setLoadingCamera(false); requestAnimationFrame(setupCanvas); initializeVision();
    } catch (err) {
      console.error('Camera startup failed:', err); setLoadingCamera(false);
      if (err?.name === 'NotAllowedError' || err?.name === 'PermissionDeniedError') { setPermissionState('denied'); setError('Camera permission was denied. Allow camera access in browser settings and press Open Camera again.'); }
      else if (err?.name === 'NotFoundError') setError('No camera was found on this device.');
      else if (err?.name === 'NotReadableError') setError('The camera is already being used by another application or browser tab.');
      else setError(`Camera could not start: ${err?.message || 'unknown error'}`);
      streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null;
    }
  };

  const stopCamera = () => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current); streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null; previousPointRef.current = null; smoothPointRef.current = null; lastVideoTimeRef.current = -1; lastDetectionRef.current = 0; clearHoldRef.current = 0; activeGestureRef.current = 'none';
    setTracking(false); setGesture('none'); setPointer(null); setCameraOn(false); setVisionReady(false); setLoadingVision(false);
  };

  const exportDrawing = () => { const canvas = canvasRef.current; if (!canvas) return; const link = document.createElement('a'); link.download = `aircanvas-${Date.now()}.png`; link.href = canvas.toDataURL('image/png'); link.click(); };
  useEffect(() => { if (cameraOn) frameRef.current = requestAnimationFrame(processFrames); return () => frameRef.current && cancelAnimationFrame(frameRef.current); }, [cameraOn, processFrames]);
  useEffect(() => { window.addEventListener('resize', setupCanvas); return () => window.removeEventListener('resize', setupCanvas); }, [setupCanvas]);
  useEffect(() => () => { mountedRef.current = false; if (frameRef.current) cancelAnimationFrame(frameRef.current); streamRef.current?.getTracks().forEach((track) => track.stop()); landmarkerRef.current?.close(); }, []);

  const gestureLabel = gesture === 'draw' ? 'DRAWING' : gesture === 'pause' ? 'PAUSED — FIST' : gesture === 'clear' ? 'CLEAR — HOLD PALM' : visionReady ? 'SHOW A GESTURE' : loadingVision ? 'VISION LOADING' : 'VISION OFFLINE';
  return <main className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark"><Sparkles size={16} /></div><div><strong>AirCanvas</strong><span>AI PASSION PROJECT / 001</span></div></div><div className={`system-status ${cameraOn ? 'live' : ''}`}><span className="status-dot" />{cameraOn ? 'VISION SYSTEM LIVE' : 'SYSTEM STANDBY'}</div></header>
    <section className="hero"><div className="hero-copy"><p className="eyebrow">REAL-TIME COMPUTER VISION + AR INK</p><h1>Draw with<br /><em>your hand.</em></h1><p className="lede">Your fingertip becomes a digital brush. AirCanvas stabilizes brief tracking drops, bridges motion gaps and adds velocity-reactive AR ink in real time.</p><div className="controls">{!cameraOn ? <button className="primary" onClick={startCamera} disabled={loadingCamera}>{loadingCamera ? <Circle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{loadingCamera ? 'OPENING CAMERA' : permissionState === 'denied' ? 'TRY CAMERA AGAIN' : 'OPEN CAMERA'}</button> : <button className="primary stop" onClick={stopCamera}><Camera size={17} /> STOP CAMERA</button>}<button className="secondary" onClick={clearCanvas}><Trash2 size={17} /> CLEAR</button><button className="secondary" onClick={exportDrawing}>EXPORT PNG</button></div>{!cameraOn && !error && <div className="permission-note"><ShieldCheck size={16} /><span>Press Open Camera. Your browser will ask for permission.</span></div>}{error && <p className="error-box">{error}</p>}</div>
      <div className="workspace"><div className="workspace-head"><div className="workspace-label"><Hand size={15} /> LIVE AR DRAWING SURFACE</div><div className="metrics"><span>{tracking ? 'FINGERTIP LOCKED' : gestureLabel}</span><b>{strokes} pts</b></div></div><div className="stage"><video ref={videoRef} className="camera" playsInline muted autoPlay /><canvas ref={canvasRef} className="drawing-layer" />{pointer && cameraOn && <div className={`finger-cursor ${tracking ? 'drawing' : ''}`} style={{ left: pointer.x, top: pointer.y }}><span /></div>}{!cameraOn && <div className="stage-empty"><div className="empty-icon"><Hand size={34} strokeWidth={1.3} /></div><strong>Open your camera to start</strong><span>Press Open Camera and allow access.</span></div>}{cameraOn && loadingVision && !error && <div className="vision-loading"><Circle className="spin" size={18} /><span>Learning your hand…</span></div>}<div className={`tracking-pill ${tracking ? 'active' : ''}`}><span />{tracking ? `FINGERTIP LOCKED · ${Math.round(velocity * 100)}% SPEED` : gestureLabel}</div></div></div></section>
    <section className="gesture-guide"><div className="guide-heading"><span>AR GESTURE LAB</span><b>Draw · Pause · Clear</b></div><div className="gesture-grid"><div className={`gesture-card ${gesture === 'draw' ? 'active' : ''}`}><div className="gesture-visual">☝</div><div><strong>DRAW</strong><span>One index finger. The ink follows your fingertip.</span></div></div><div className={`gesture-card ${gesture === 'pause' ? 'active' : ''}`}><div className="gesture-visual">✊</div><div><strong>PAUSE</strong><span>Make a fist. Your ink freezes without losing the canvas.</span></div></div><div className={`gesture-card ${gesture === 'clear' ? 'active' : ''}`}><div className="gesture-visual">🖐</div><div><strong>CLEAR</strong><span>Hold an open palm for 0.8 seconds to wipe the canvas.</span></div></div></div></section>
    <section className="tool-panel"><div className="tool-group"><span className="tool-title">BRUSH</span><div className="brush-row">{[3, 6, 10, 16].map((size) => <button key={size} className={`brush ${brushSize === size ? 'selected' : ''}`} onClick={() => setBrushSize(size)}><span style={{ width: size, height: size }} /></button>)}</div></div><div className="tool-group"><span className="tool-title">INK</span><div className="color-row">{['#ffffff', '#8bffb0', '#75a7ff', '#ff7e9f', '#ffd166'].map((color) => <button key={color} aria-label={`Select ${color}`} className={`color ${brushColor === color ? 'selected' : ''}`} style={{ background: color }} onClick={() => setBrushColor(color)} />)}</div></div><div className="gesture-note"><span><b>AR FX:</b> speed changes glow + brush weight · motion gaps are bridged</span></div><div className="tech-stack">MEDIA PIPE · CANVAS 2D · REACT · AR INK</div></section>
    <footer className="footer">Made by - Amar</footer>
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
