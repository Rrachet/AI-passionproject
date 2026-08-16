import React, { useCallback, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { Camera, Circle, Hand, Play, RotateCcw, Sparkles, Trash2, ShieldCheck } from 'lucide-react';
import './styles.css';

// MediaPipe's documented web setup uses the Tasks Vision WASM bundle plus the official hand model.
// Keep both pinned to the same known-good 0.10.x runtime and use CPU for broad mobile compatibility.
const MP_VERSION = '0.10.35';
const WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}/wasm`;
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const streamRef = useRef(null);
  const frameRef = useRef(null);
  const previousPointRef = useRef(null);
  const smoothPointRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
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

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
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
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    previousPointRef.current = null;
    smoothPointRef.current = null;
    setStrokes(0);
  }, []);

  const indexOnly = (hand) => {
    if (!hand || hand.length < 21) return false;
    // Only draw when the index is extended and the other three fingers are folded.
    return hand[8].y < hand[6].y && hand[12].y > hand[10].y && hand[16].y > hand[14].y && hand[20].y > hand[18].y;
  };

  const drawResult = useCallback((result) => {
    const canvas = canvasRef.current;
    const hand = result?.landmarks?.[0];
    if (!canvas || !hand || !indexOnly(hand)) {
      previousPointRef.current = null;
      smoothPointRef.current = null;
      setTracking(false);
      return;
    }

    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const target = { x: (1 - hand[8].x) * width, y: hand[8].y * height };
    const previousSmooth = smoothPointRef.current || target;
    const point = {
      x: previousSmooth.x * 0.72 + target.x * 0.28,
      y: previousSmooth.y * 0.72 + target.y * 0.28,
    };
    smoothPointRef.current = point;

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
  }, [brushColor, brushSize]);

  const processFrames = useCallback(() => {
    const video = videoRef.current;
    const landmarker = landmarkerRef.current;
    if (!cameraOn) return;

    if (video && landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      try {
        const result = landmarker.detectForVideo(video, performance.now());
        drawResult(result);
      } catch (err) {
        console.error('MediaPipe frame error:', err);
      }
    }
    frameRef.current = requestAnimationFrame(processFrames);
  }, [cameraOn, drawResult]);

  const initializeVision = async () => {
    if (landmarkerRef.current) return landmarkerRef.current;
    setLoadingVision(true);
    setError('');

    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      // CPU is intentional: it is the most portable delegate across desktop and mobile browsers.
      const landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      landmarkerRef.current = landmarker;
      if (mountedRef.current) {
        setVisionReady(true);
        setLoadingVision(false);
      }
      return landmarker;
    } catch (err) {
      console.error('MediaPipe initialization failed:', err);
      if (mountedRef.current) {
        setLoadingVision(false);
        setVisionReady(false);
        setError(`Hand tracking failed to load: ${err?.message || 'MediaPipe could not initialize.'}`);
      }
      return null;
    }
  };

  const startCamera = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not supported by this browser. Use a current version of Chrome or Safari.');
      return;
    }
    if (!window.isSecureContext) {
      setError('Camera access requires HTTPS. Open the deployed AirCanvas website.');
      return;
    }

    try {
      setError('');
      setLoadingCamera(true);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'user' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });

      streamRef.current = stream;
      setPermissionState('granted');
      const video = videoRef.current;
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        video.play().catch((err) => console.error('Video play failed:', err));
        requestAnimationFrame(setupCanvas);
      };
      await video.play();
      setCameraOn(true);
      setLoadingCamera(false);

      // Start vision separately from camera startup. The camera stays visible while the model loads.
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
    setTracking(false);
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

  useEffect(() => {
    window.addEventListener('resize', setupCanvas);
    return () => window.removeEventListener('resize', setupCanvas);
  }, [setupCanvas]);

  useEffect(() => () => {
    mountedRef.current = false;
    if (frameRef.current) cancelAnimationFrame(frameRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    landmarkerRef.current?.close();
  }, []);

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
            {!cameraOn ? <button className="primary" onClick={startCamera} disabled={loadingCamera}>{loadingCamera ? <Circle className="spin" size={17} /> : <Play size={17} fill="currentColor" />}{loadingCamera ? 'OPENING CAMERA' : permissionState === 'denied' ? 'TRY CAMERA AGAIN' : 'OPEN CAMERA'}</button> : <button className="primary stop" onClick={stopCamera}><Camera size={17} /> STOP CAMERA</button>}
            <button className="secondary" onClick={clearCanvas}><Trash2 size={17} /> CLEAR</button>
            <button className="secondary" onClick={exportDrawing}>EXPORT PNG</button>
          </div>
          {!cameraOn && !error && <div className="permission-note"><ShieldCheck size={16} /><span>Press Open Camera. Your browser will ask for permission.</span></div>}
          {error && <p className="error-box">{error}</p>}
        </div>

        <div className="workspace">
          <div className="workspace-head"><div className="workspace-label"><Hand size={15} /> LIVE DRAWING SURFACE</div><div className="metrics"><span>{tracking ? 'TRACKING INDEX' : loadingVision ? 'LOADING VISION' : visionReady ? 'SHOW ONE FINGER' : cameraOn ? 'VISION OFFLINE' : 'STANDBY'}</span><b>{strokes} pts</b></div></div>
          <div className="stage">
            <video ref={videoRef} className="camera" playsInline muted autoPlay />
            <canvas ref={canvasRef} className="drawing-layer" />
            {!cameraOn && <div className="stage-empty"><div className="empty-icon"><Hand size={34} strokeWidth={1.3} /></div><strong>Open your camera to start</strong><span>Press Open Camera and allow access.</span></div>}
            {cameraOn && loadingVision && !error && <div className="vision-loading"><Circle className="spin" size={18} /><span>Loading hand tracking…</span></div>}
            <div className={`tracking-pill ${tracking ? 'active' : ''}`}><span />{tracking ? 'INDEX DETECTED' : visionReady ? 'SHOW ONE FINGER' : loadingVision ? 'VISION LOADING' : 'VISION OFFLINE'}</div>
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
