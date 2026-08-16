import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Camera, Circle, Hand, Play, Sparkles, Trash2, ShieldCheck, MousePointer2, WandSparkles, Scan, Zap, Undo2 } from 'lucide-react';
import './styles.css';

const MP_VERSION = '0.10.35';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const TARGET_FPS = 30;
const LOST_HAND_GRACE_MS = 180;

const TOOLS = {
  draw: { label: 'Draw', icon: '✦', hint: 'Freehand AR ink' },
  shape: { label: 'AI Shape', icon: '◇', hint: 'Auto-align circles, boxes & arrows' },
  pointer: { label: 'AR Pointer', icon: '⌖', hint: 'Point at anything on screen' },
  laser: { label: 'AI Laser', icon: '⚡', hint: 'Live laser pointer trail' },
};

function distance(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function angleAt(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y }, cb = { x: c.x - b.x, y: c.y - b.y };
  const denom = Math.hypot(ab.x, ab.y) * Math.hypot(cb.x, cb.y) || 1;
  return Math.acos(Math.max(-1, Math.min(1, (ab.x * cb.x + ab.y * cb.y) / denom))) * 180 / Math.PI;
}
function fingerIsExtended(hand, mcp, pip, tip) {
  return angleAt(hand[mcp], hand[pip], hand[tip]) > 132 && distance(hand[mcp], hand[tip]) > distance(hand[mcp], hand[pip]) * 1.04;
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

function resample(points, count = 40) {
  if (points.length < 2) return points;
  const out = [points[0]];
  const total = points.reduce((s, p, i) => i ? s + distance(points[i - 1], p) : 0, 0);
  const step = total / (count - 1);
  let target = step, walked = 0, i = 1, prev = points[0];
  while (i < points.length && out.length < count) {
    const d = distance(prev, points[i]);
    if (walked + d >= target && d > 0) {
      const t = (target - walked) / d;
      const p = { x: prev.x + (points[i].x - prev.x) * t, y: prev.y + (points[i].y - prev.y) * t };
      out.push(p); prev = p; walked = 0; target = step;
    } else { walked += d; prev = points[i]; i += 1; }
  }
  while (out.length < count) out.push(points[points.length - 1]);
  return out;
}

function recognizeShape(points) {
  if (points.length < 12) return null;
  const p = resample(points, 36);
  const start = p[0], end = p[p.length - 1];
  const closed = distance(start, end) < Math.max(28, points.reduce((s, x) => s + distance(x, start), 0) / points.length * 0.28);
  const xs = p.map(x => x.x), ys = p.map(x => x.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const w = Math.max(1, maxX - minX), h = Math.max(1, maxY - minY);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  if (closed) {
    const radial = p.reduce((s, x) => s + Math.abs(Math.hypot(x.x - cx, x.y - cy) - Math.hypot(w / 2, h / 2) * 0.72), 0) / p.length;
    if (radial < Math.max(18, Math.min(w, h) * 0.18)) return { type: 'circle', cx, cy, rx: w / 2, ry: h / 2 };
    return { type: 'rect', x: minX, y: minY, w, h, radius: Math.min(18, Math.min(w, h) * 0.08) };
  }
  if (w > h * 2.8 || h > w * 2.8) return { type: 'line', from: start, to: end };
  const dx = end.x - start.x, dy = end.y - start.y;
  const tip = end, len = Math.max(16, Math.hypot(dx, dy)), ux = dx / len, uy = dy / len;
  const px = -uy, py = ux, wing = Math.min(28, len * 0.16);
  return { type: 'arrow', from: start, to: end, a: { x: tip.x - ux * wing + px * wing * 0.7, y: tip.y - uy * wing + py * wing * 0.7 }, b: { x: tip.x - ux * wing - px * wing * 0.7, y: tip.y - uy * wing - py * wing * 0.7 } };
}

function App() {
  const videoRef = useRef(null), canvasRef = useRef(null), landmarkerRef = useRef(null), streamRef = useRef(null), frameRef = useRef(null);
  const previousPointRef = useRef(null), smoothPointRef = useRef(null), pointsRef = useRef([]), lastVideoTimeRef = useRef(-1), lastDetectionRef = useRef(0), lastHandSeenRef = useRef(0);
  const clearHoldRef = useRef(0), lastClearRef = useRef(0), activeGestureRef = useRef('none'), laserPointsRef = useRef([]), mountedRef = useRef(true);
  const [cameraOn, setCameraOn] = useState(false), [visionReady, setVisionReady] = useState(false), [tracking, setTracking] = useState(false);
  const [loadingCamera, setLoadingCamera] = useState(false), [loadingVision, setLoadingVision] = useState(false), [error, setError] = useState('');
  const [permissionState, setPermissionState] = useState('prompt'), [brushSize, setBrushSize] = useState(6), [brushColor, setBrushColor] = useState('#9dffbd');
  const [strokes, setStrokes] = useState(0), [gesture, setGesture] = useState('none'), [pointer, setPointer] = useState(null), [velocity, setVelocity] = useState(0);
  const [tool, setTool] = useState('draw'), [showTools, setShowTools] = useState(false), [shapeName, setShapeName] = useState('');

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current, video = videoRef.current; if (!canvas || !video) return;
    const rect = video.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(rect.width * dpr)); canvas.height = Math.max(1, Math.round(rect.height * dpr));
    canvas.style.width = `${rect.width}px`; canvas.style.height = `${rect.height}px`;
    const ctx = canvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current; if (!canvas) return;
    canvas.getContext('2d').clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    previousPointRef.current = null; smoothPointRef.current = null; pointsRef.current = []; laserPointsRef.current = [];
    setStrokes(0); setShapeName('');
  }, []);

  const drawSegment = useCallback((from, to, mode = tool) => {
    const canvas = canvasRef.current; if (!canvas || mode === 'pointer') return;
    const ctx = canvas.getContext('2d'), d = distance(from, to), steps = Math.max(1, Math.ceil(d / 5));
    if (mode === 'laser') {
      laserPointsRef.current.push({ ...to, t: performance.now() });
      if (laserPointsRef.current.length > 45) laserPointsRef.current.shift();
      ctx.save(); ctx.lineCap = 'round';
      for (let i = 1; i < laserPointsRef.current.length; i += 1) {
        const a = laserPointsRef.current[i - 1], b = laserPointsRef.current[i], age = performance.now() - b.t, alpha = Math.max(0, 1 - age / 700);
        ctx.globalAlpha = alpha; ctx.strokeStyle = '#ff3b6b'; ctx.shadowColor = '#ff3b6b'; ctx.shadowBlur = 18; ctx.lineWidth = Math.max(2, brushSize * alpha);
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
      ctx.restore(); return;
    }
    ctx.save(); ctx.strokeStyle = brushColor; ctx.lineWidth = Math.max(2, brushSize * (1.08 - Math.min(1, d / 45) * 0.2)); ctx.shadowColor = brushColor; ctx.shadowBlur = 6 + Math.min(14, d / 4);
    ctx.beginPath(); ctx.moveTo(from.x, from.y);
    for (let i = 1; i <= steps; i += 1) { const t = i / steps; ctx.lineTo(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t); }
    ctx.stroke(); ctx.restore();
  }, [brushColor, brushSize, tool]);

  const commitShape = useCallback(() => {
    if (tool !== 'shape' || pointsRef.current.length < 12) return;
    const shape = recognizeShape(pointsRef.current); if (!shape) return;
    const canvas = canvasRef.current, ctx = canvas?.getContext('2d'); if (!ctx) return;
    ctx.save(); ctx.strokeStyle = brushColor; ctx.lineWidth = brushSize; ctx.shadowColor = brushColor; ctx.shadowBlur = 9;
    ctx.beginPath();
    if (shape.type === 'circle') ctx.ellipse(shape.cx, shape.cy, shape.rx, shape.ry, 0, 0, Math.PI * 2);
    if (shape.type === 'rect') ctx.roundRect(shape.x, shape.y, shape.w, shape.h, shape.radius);
    if (shape.type === 'line') { ctx.moveTo(shape.from.x, shape.from.y); ctx.lineTo(shape.to.x, shape.to.y); }
    if (shape.type === 'arrow') { ctx.moveTo(shape.from.x, shape.from.y); ctx.lineTo(shape.to.x, shape.to.y); ctx.moveTo(shape.to.x, shape.to.y); ctx.lineTo(shape.a.x, shape.a.y); ctx.moveTo(shape.to.x, shape.to.y); ctx.lineTo(shape.b.x, shape.b.y); }
    ctx.stroke(); ctx.restore(); setShapeName(shape.type === 'rect' ? 'RECTANGLE ALIGNED' : `${shape.type.toUpperCase()} ALIGNED`);
  }, [brushColor, brushSize, tool]);

  const drawResult = useCallback((result) => {
    const canvas = canvasRef.current, hand = result?.landmarks?.[0], now = performance.now();
    if (!canvas || !hand) {
      if (now - lastHandSeenRef.current > LOST_HAND_GRACE_MS) { previousPointRef.current = null; smoothPointRef.current = null; setTracking(false); setPointer(null); activeGestureRef.current = 'none'; setGesture('none'); }
      return;
    }
    lastHandSeenRef.current = now;
    const width = canvas.clientWidth, height = canvas.clientHeight, tip = hand[8];
    const raw = { x: (1 - tip.x) * width, y: tip.y * height }, prev = smoothPointRef.current || raw, moved = distance(raw, prev);
    const alpha = moved > 70 ? 0.95 : moved > 30 ? 0.86 : moved > 10 ? 0.7 : 0.5;
    const point = { x: prev.x + (raw.x - prev.x) * alpha, y: prev.y + (raw.y - prev.y) * alpha };
    smoothPointRef.current = point; setPointer(point); setVelocity(Math.min(1, moved / 45));
    const detected = classifyGesture(hand);
    if (detected === 'draw') {
      activeGestureRef.current = 'draw'; setGesture('draw');
      if (previousPointRef.current) drawSegment(previousPointRef.current, point);
      previousPointRef.current = point;
      if (tool === 'shape') pointsRef.current.push(point);
      setStrokes((n) => n + 1); setTracking(true); return;
    }
    if (detected === 'pause') {
      activeGestureRef.current = 'pause'; setGesture('pause');
      if (tool === 'shape') commitShape();
      previousPointRef.current = null; smoothPointRef.current = null; pointsRef.current = []; setTracking(false); clearHoldRef.current = 0; return;
    }
    if (detected === 'clear') {
      activeGestureRef.current = 'clear'; setGesture('clear'); previousPointRef.current = null; smoothPointRef.current = null; pointsRef.current = []; setTracking(false);
      if (!clearHoldRef.current) clearHoldRef.current = now;
      if (now - clearHoldRef.current >= 800 && now - lastClearRef.current >= 1400) { lastClearRef.current = now; clearHoldRef.current = 0; clearCanvas(); }
      return;
    }
    if (activeGestureRef.current === 'draw' && now - lastHandSeenRef.current < LOST_HAND_GRACE_MS) return;
    activeGestureRef.current = 'none'; setGesture('none'); previousPointRef.current = null; smoothPointRef.current = null; setTracking(false); clearHoldRef.current = 0;
  }, [clearCanvas, commitShape, drawSegment, tool]);

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
      if (mountedRef.current) { setVisionReady(true); setLoadingVision(false); } return landmarkerRef.current;
    } catch (err) { console.error('MediaPipe initialization failed:', err); landmarkerRef.current = null; if (mountedRef.current) { setLoadingVision(false); setVisionReady(false); setError(`Hand tracking failed to load: ${err?.message || 'MediaPipe could not initialize.'}`); } return null; }
  };
  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) return setError('Camera access is not supported by this browser.');
    if (!window.isSecureContext) return setError('Camera access requires HTTPS. Open the deployed AirCanvas website.');
    try {
      setError(''); setLoadingCamera(true);
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'user' }, width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30, max: 30 } }, audio: false });
      streamRef.current = stream; setPermissionState('granted'); const video = videoRef.current; video.srcObject = stream; await video.play(); setCameraOn(true); setLoadingCamera(false); requestAnimationFrame(setupCanvas); initializeVision();
    } catch (err) {
      setLoadingCamera(false); if (err?.name === 'NotAllowedError') { setPermissionState('denied'); setError('Camera permission was denied. Allow camera access and try again.'); } else if (err?.name === 'NotFoundError') setError('No camera was found.'); else if (err?.name === 'NotReadableError') setError('The camera is already being used.'); else setError(`Camera could not start: ${err?.message || 'unknown error'}`);
      streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null;
    }
  };
  const stopCamera = () => {
    if (frameRef.current) cancelAnimationFrame(frameRef.current); streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null; if (videoRef.current) videoRef.current.srcObject = null;
    previousPointRef.current = null; smoothPointRef.current = null; pointsRef.current = []; laserPointsRef.current = []; lastVideoTimeRef.current = -1; activeGestureRef.current = 'none';
    setTracking(false); setGesture('none'); setPointer(null); setCameraOn(false); setVisionReady(false); setLoadingVision(false);
  };
  const exportDrawing = () => { const canvas = canvasRef.current; if (!canvas) return; const link = document.createElement('a'); link.download = `aircanvas-${Date.now()}.png`; link.href = canvas.toDataURL('image/png'); link.click(); };
  const selectTool = (name) => { setTool(name); setShowTools(false); previousPointRef.current = null; smoothPointRef.current = null; pointsRef.current = []; laserPointsRef.current = []; setShapeName(''); };

  useEffect(() => { if (cameraOn) frameRef.current = requestAnimationFrame(processFrames); return () => frameRef.current && cancelAnimationFrame(frameRef.current); }, [cameraOn, processFrames]);
  useEffect(() => { window.addEventListener('resize', setupCanvas); return () => window.removeEventListener('resize', setupCanvas); }, [setupCanvas]);
  useEffect(() => () => { mountedRef.current = false; if (frameRef.current) cancelAnimationFrame(frameRef.current); streamRef.current?.getTracks().forEach((track) => track.stop()); landmarkerRef.current?.close(); }, []);

  useEffect(() => {
    let raf;
    const tick = () => { const canvas = canvasRef.current; if (canvas && tool === 'laser' && laserPointsRef.current.length) { const ctx = canvas.getContext('2d'); ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight); for (let i = 1; i < laserPointsRef.current.length; i += 1) { const a = laserPointsRef.current[i - 1], b = laserPointsRef.current[i], alpha = Math.max(0, 1 - (performance.now() - b.t) / 700); ctx.globalAlpha = alpha; ctx.strokeStyle = '#ff3b6b'; ctx.shadowColor = '#ff3b6b'; ctx.shadowBlur = 18; ctx.lineWidth = Math.max(2, brushSize * alpha); ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); } ctx.globalAlpha = 1; } raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick); return () => cancelAnimationFrame(raf);
  }, [tool, brushSize]);

  const gestureLabel = gesture === 'draw' ? 'ACTIVE' : gesture === 'pause' ? 'PAUSED — FIST' : gesture === 'clear' ? 'CLEAR — HOLD PALM' : visionReady ? TOOLS[tool].label.toUpperCase() : loadingVision ? 'VISION LOADING' : 'VISION OFFLINE';
  return <main className="app-shell">
    <header className="topbar"><div className="brand"><div className="brand-mark"><Sparkles size={16} /></div><div><strong>AirCanvas</strong><span>AI PASSION PROJECT / 001</span></div></div><div className={`system-status ${cameraOn ? 'live' : ''}`}><span className="status-dot" />{cameraOn ? 'VISION SYSTEM LIVE' : 'SYSTEM STANDBY'}</div></header>
    <section className="hero"><div className="hero-copy"><p className="eyebrow">REAL-TIME COMPUTER VISION + AR</p><h1>Draw what<br /><em>you imagine.</em></h1><p className="lede">Choose how your hand should behave: freehand ink, AI-aligned shapes, an AR pointer, or a laser that cuts through the screen.</p>
      <div className="controls">{!cameraOn ? <button className="primary" onClick={startCamera} disabled={loadingCamera}>{loadingCamera ? <Circle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{loadingCamera ? 'OPENING CAMERA' : permissionState === 'denied' ? 'TRY CAMERA AGAIN' : 'OPEN CAMERA'}</button> : <button className="primary stop" onClick={stopCamera}><Camera size={17} /> STOP CAMERA</button>}<button className="secondary" onClick={clearCanvas}><Trash2 size={17} /> CLEAR</button><button className="secondary" onClick={exportDrawing}>EXPORT PNG</button></div>
      {!cameraOn && !error && <div className="permission-note"><ShieldCheck size={16} /><span>Press Open Camera. Your browser will ask for permission.</span></div>}{error && <p className="error-box">{error}</p>}</div>
      <div className="workspace"><div className="workspace-head"><div className="workspace-label"><Hand size={15} /> LIVE AR SURFACE</div><div className="metrics"><span>{tracking ? 'FINGERTIP LOCKED' : gestureLabel}</span><b>{strokes} pts</b></div></div>
        <div className="stage"><video ref={videoRef} className="camera" playsInline muted autoPlay /><canvas ref={canvasRef} className="drawing-layer" />{pointer && cameraOn && <div className={`finger-cursor ${tracking ? 'drawing' : ''}`} style={{ left: pointer.x, top: pointer.y }}><span /></div>}
          {!cameraOn && <div className="stage-empty"><div className="empty-icon"><Hand size={34} strokeWidth={1.3} /></div><strong>Open your camera to start</strong><span>Choose a mode after the camera is live.</span></div>}
          {cameraOn && loadingVision && !error && <div className="vision-loading"><Circle className="spin" size={18} /><span>Learning your hand…</span></div>}
          {shapeName && <div className="shape-toast"><WandSparkles size={14} /> {shapeName}</div>}
          <div className={`tracking-pill ${tracking ? 'active' : ''}`}><span />{tracking ? `${TOOLS[tool].label.toUpperCase()} · ${Math.round(velocity * 100)}% SPEED` : gestureLabel}</div>
          {cameraOn && <div className="tool-dock-wrap" onMouseEnter={() => setShowTools(true)} onMouseLeave={() => setShowTools(false)}><button className="tool-dock" onClick={() => setShowTools((v) => !v)}><Sparkles size={16} /><span>{TOOLS[tool].label}</span><small>HOVER / TAP TO CHOOSE</small></button>{showTools && <div className="tool-menu">{Object.entries(TOOLS).map(([key, value]) => <button key={key} className={tool === key ? 'selected' : ''} onClick={() => selectTool(key)}><span className="tool-icon">{value.icon}</span><span><strong>{value.label}</strong><small>{value.hint}</small></span></button>)}</div>}</div>}
        </div>
      </div></section>

    <section className="mode-section"><div className="section-head"><span>CHOOSE YOUR SUPERPOWER</span><b>One camera. Four ways to communicate.</b></div><div className="mode-grid">{Object.entries(TOOLS).map(([key, value]) => <button key={key} className={`mode-card ${tool === key ? 'selected' : ''}`} onClick={() => selectTool(key)}><div className="mode-icon">{value.icon}</div><strong>{value.label}</strong><span>{value.hint}</span><em>{tool === key ? 'ACTIVE' : 'SELECT'}</em></button>)}</div></section>
    <section className="gesture-guide"><div className="section-head"><span>GESTURE LANGUAGE</span><b>☝ Draw · ✊ Pause · 🖐 Clear</b></div><div className="gesture-grid"><div className={`gesture-card ${gesture === 'draw' ? 'active' : ''}`}><div className="gesture-visual">☝</div><div><strong>DRAW / ACT</strong><span>Raise one index finger and move.</span></div></div><div className={`gesture-card ${gesture === 'pause' ? 'active' : ''}`}><div className="gesture-visual">✊</div><div><strong>PAUSE / COMMIT</strong><span>Fist pauses. In AI Shape mode it locks the shape.</span></div></div><div className={`gesture-card ${gesture === 'clear' ? 'active' : ''}`}><div className="gesture-visual">🖐</div><div><strong>CLEAR</strong><span>Hold an open palm for about one second.</span></div></div></div></section>
    <footer className="footer">Made by - Amar</footer>
  </main>;
}

createRoot(document.getElementById('root')).render(<App />);
