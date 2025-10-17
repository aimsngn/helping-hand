// ======================= app.js =======================
// A / I / L / V / Y classifier (no sliders)
// - Camera + MediaPipe Hands (21 landmarks/frame)
// - Simple geometry thresholds (normalized by palm width)
// - Smoothing + confirmation gate
// - Web Serial: sends ONE ASCII letter to Arduino when correct
// - Console Coach: tells you which threshold to tweak and how
// =====================================================


// ------------------------------------------------------
// CONFIGURATION (tune these numbers as needed)
// ------------------------------------------------------

// Smoothing window: majority vote over last N frames
const SMOOTH_N = 5;

// Confirmation gate: require this many consecutive frames = target
const CONFIRM_FRAMES = 4;

/*
 All distances are normalized by palm width (MCP index ↔ MCP pinky),
 so thresholds are hand-size and distance independent.
*/
const TH = {
  // General: thumb considered "out" if horizontal spread exceeds this
  thumbSpread: 0.035,

  // A: fist with thumb outside along index
  A_thumbAwayFromIndexMCP: 0.55,

  // I: pinky up only (others down, thumb folded)
  // (kept for future numeric gate; current rule is shape-based)
  I_pinkyUpGap: 0.05,

  // L: index up + thumb out; ~right angle; others bent; index straight
  L_requiredAngleMinDeg: 55,
  L_requiredAngleMaxDeg: 115,
  L_minThumbSpread: 0.040,   // normalized thumb tip↔base
  L_indexStraightDeg: 165,   // PIP angle for index
  L_otherBentTol: 0.62,      // TIP↔PIP normalized gap (< tol ⇒ bent)

  // V: index & middle up; separate tips & angle
  V_sepMin: 0.24,            // tip gap (index↔middle)
  V_angleMinDeg: 16,         // ray angle (index vs middle)
  V_hyst: 0.02,              // hysteresis to reduce flicker

  // Y (shaka): thumb + pinky out
  Y_thumbSpread: 0.035       // normalized thumb tip↔base
};


// ------------------------------------------------------
// DOM REFERENCES
// ------------------------------------------------------
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const $pred = document.getElementById('prediction');
const $status = document.getElementById('status');
const $conf = document.getElementById('conf');
const $target = document.getElementById('target');
const $connectBtn = document.getElementById('connectBtn');
const $serialStatus = document.getElementById('serialStatus');
const $lastSent = document.getElementById('lastSent');
const $debug = document.getElementById('debug');
const $dbgToggle = document.getElementById('dbgToggle');
const $errbar = document.getElementById('errbar');

// Tiny overlay toggle
$dbgToggle.addEventListener('click', () => {
  const show = !$debug.style.display || $debug.style.display === 'none';
  $debug.style.display = show ? 'block' : 'none';
});

// Track desired letter (dropdown)
let targetLetter = $target.value;
$target.addEventListener('change', () => {
  targetLetter = $target.value;
  streak = 0;
  sentForThisSuccess = false;
  $conf.textContent = `0 / ${CONFIRM_FRAMES}`;
});


// ------------------------------------------------------
// GEOMETRY HELPERS (2D, normalized by palm width)
// ------------------------------------------------------

// Euclidean distance (image space)
function dist(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }

// Palm width = MCP index (5) ↔ MCP pinky (17)
function palmWidth(lm){ return Math.max(dist(lm[5], lm[17]), 1e-6); }

// Normalized distance (scale invariant)
function ndist(lm,i,j,pw){ return dist(lm[i], lm[j]) / pw; }

// Cosine similarity between two vectors
function cosSim(ax,ay,bx,by){
  const dot = ax*bx + ay*by, ma = Math.hypot(ax,ay)||1e-6, mb = Math.hypot(bx,by)||1e-6;
  return dot/(ma*mb);
}

// Angle in degrees between two rays
function rayAngleDeg(ax,ay,bx,by){
  const c = Math.max(-1, Math.min(1, cosSim(ax,ay,bx,by)));
  return Math.acos(c) * 180 / Math.PI;
}

// Normalized thumb spread (tip(4) ↔ base(2))
function thumbSpreadNorm(lm,pw){ return ndist(lm,4,2,pw); }

// Quick finger-up heuristic: tip above PIP (lower y) ⇒ up
// Thumb uses horizontal spread vs its base
function fingersUp(lm){
  const upIndex  = lm[8].y  < lm[6].y  ? 1 : 0;
  const upMiddle = lm[12].y < lm[10].y ? 1 : 0;
  const upRing   = lm[16].y < lm[14].y ? 1 : 0;
  const upPinky  = lm[20].y < lm[18].y ? 1 : 0;
  const thumbOut = Math.abs(lm[4].x - lm[2].x) > TH.thumbSpread ? 1 : 0;
  return [thumbOut, upIndex, upMiddle, upRing, upPinky]; // [T,I,M,R,P]
}

// Index PIP angle (higher ≈ straighter)
function angleAtPIP(lm, tip, dip, pip){
  const ax=lm[tip].x-lm[dip].x, ay=lm[tip].y-lm[dip].y;
  const bx=lm[pip].x-lm[dip].x, by=lm[pip].y-lm[dip].y;
  const dot=ax*bx+ay*by, na=Math.hypot(ax,ay)||1e-6, nb=Math.hypot(bx,by)||1e-6;
  const c = Math.max(-1, Math.min(1, dot/(na*nb)));
  return Math.acos(c)*180/Math.PI;
}
function fingerIsStraight_Index(lm){ return angleAtPIP(lm,8,7,6) >= TH.L_indexStraightDeg; }
function fingerIsBent(lm, tip, pip, pw){ return ndist(lm,tip,pip,pw) < TH.L_otherBentTol; }


// ------------------------------------------------------
// CLASSIFIER (returns 'A'|'I'|'L'|'V'|'Y'|'?')
// ------------------------------------------------------
let vLatch='V'; // hysteresis for V near boundary
function classifyLetters(lm){
  const pw = palmWidth(lm);
  const f  = fingersUp(lm);

  // Core measurements
  const dThumb_IndexMCP = ndist(lm,4,5,pw);  // A helper
  const dIdxMidTips     = ndist(lm,8,12,pw); // V separation
  const dThumb_PinkyTip = ndist(lm,4,20,pw); // Y span
  const ix=lm[8].x-lm[5].x,  iy=lm[8].y-lm[5].y;
  const tx=lm[4].x-lm[2].x,  ty=lm[4].y-lm[2].y;
  const mx=lm[12].x-lm[9].x, my=lm[12].y-lm[9].y;
  const i_t_deg = rayAngleDeg(ix,iy,tx,ty);  // L angle
  const i_m_deg = rayAngleDeg(ix,iy,mx,my);  // V angle

  // Macro shapes
  const allFourDown = (f[1]===0 && f[2]===0 && f[3]===0 && f[4]===0);
  const pinkyUpOnly = (f[4]===1 && f[1]===0 && f[2]===0 && f[3]===0);
  const twoUp_IM    = (f[1]===1 && f[2]===1 && f[3]===0 && f[4]===0);
  const L_shape     = (f[1]===1 && f[0]===1 && f[2]===0 && f[3]===0 && f[4]===0);
  const Y_shape     = (f[0]===1 && f[4]===1 && f[1]===0 && f[2]===0 && f[3]===0);

  // --- A: fist + thumb outside along index ---
  if(allFourDown && f[0]===1 && dThumb_IndexMCP >= TH.A_thumbAwayFromIndexMCP) return 'A';

  // --- I: pinky up only (thumb folded) ---
  if(pinkyUpOnly && f[0]===0) return 'I';

  // --- L: index up + thumb out; right-angle-ish; others bent; index straight ---
  if(L_shape){
    const thumbNorm = thumbSpreadNorm(lm,pw);
    const indexStraight = fingerIsStraight_Index(lm);
    const othersBent = fingerIsBent(lm,12,10,pw) && fingerIsBent(lm,16,14,pw) && fingerIsBent(lm,20,18,pw);
    const angleOK = (i_t_deg >= TH.L_requiredAngleMinDeg && i_t_deg <= TH.L_requiredAngleMaxDeg);
    if(angleOK && thumbNorm > TH.L_minThumbSpread && indexStraight && othersBent) return 'L';
  }

  // --- V: index + middle up; separated tips & angle ---
  if(twoUp_IM){
    const sepOK = dIdxMidTips >= TH.V_sepMin;
    const angOK = i_m_deg >= TH.V_angleMinDeg;
    if(sepOK && angOK){ vLatch='V'; return 'V'; }
    if(vLatch==='V'){
      const keepV = (dIdxMidTips >= (TH.V_sepMin - TH.V_hyst)) || (i_m_deg >= TH.V_angleMinDeg - 3);
      if(keepV) return 'V';
    }
  }

  // --- Y: thumb + pinky out; others down; span check ---
  if(Y_shape){
    const thumbOK = thumbSpreadNorm(lm,pw) > TH.Y_thumbSpread;
    const spanOK  = dThumb_PinkyTip > 0.35; // tweak if needed
    if(thumbOK && spanOK) return 'Y';
  }

  return '?';
}


// ------------------------------------------------------
// SMOOTHING & CONFIRMATION
// ------------------------------------------------------
const hist = [];   // rolling predictions
let streak = 0;    // consecutive frames hit target
let sentForThisSuccess = false; // send only once per success

function majority(arr){
  const c = Object.create(null);
  for (const v of arr) c[v] = (c[v]||0)+1;
  let best='?', cnt=0; for (const k in c){ if (c[k]>cnt){ best=k; cnt=c[k]; } }
  return best;
}


// ------------------------------------------------------
// WEB SERIAL (Browser ↔ Arduino)
// ------------------------------------------------------
let port=null, writer=null, serialConnected=false;

function updateSerialUI(){
  if(serialConnected){
    $connectBtn.textContent='Disconnect Robot';
    $serialStatus.textContent='Connected';
  }else{
    $connectBtn.textContent='Connect Robot';
    $serialStatus.textContent=('serial' in navigator) ? 'Not connected' : 'Unsupported';
  }
}

async function connectSerial(){
  try{
    port = await navigator.serial.requestPort();
    await port.open({ baudRate: 115200 });
    writer = port.writable.getWriter();
    serialConnected = true; updateSerialUI();
  }catch(e){
    console.error(e);
    serialConnected = false; updateSerialUI();
    $errbar.textContent = 'Serial connection failed. Check cable/permissions.';
    $errbar.style.display = 'block';
  }
}

async function disconnectSerial(){
  try{
    if(writer){ writer.releaseLock(); writer=null; }
    if(port){ await port.close(); port=null; }
  }finally{
    serialConnected=false; updateSerialUI();
  }
}

// Send exactly ONE ASCII letter ('A','I','L','V','Y')
async function sendToRobot(letter){
  if(!serialConnected || !writer) return;
  try{
    await writer.write(new TextEncoder().encode(letter));
    $lastSent.textContent = letter;
  }catch(e){
    console.error('Serial write failed', e);
    await disconnectSerial();
  }
}

if('serial' in navigator){
  $connectBtn.disabled = false;
  $connectBtn.addEventListener('click', () => serialConnected ? disconnectSerial() : connectSerial());
}else{
  $connectBtn.disabled = true;
}
updateSerialUI();


// ------------------------------------------------------
// MEDIAPIPE HANDS + CAMERA
// ------------------------------------------------------
const hands = new Hands({ locateFile: (f)=>`https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6
});
hands.onResults(onResults);

const cam = new Camera(video, {
  onFrame: async () => { await hands.send({ image: video }); },
  width: 960, height: 540
});

// Camera smoke test: surfaces permission issues early
async function smokeTestCamera(){
  try{
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 540 } });
    video.srcObject = stream; // raw stream appears immediately
  }catch(err){
    console.error('getUserMedia failed:', err);
    $errbar.textContent = 'Camera permission denied or no camera found. Use HTTPS/localhost and allow access.';
    $errbar.style.display = 'block';
    throw err;
  }
}


// ------------------------------------------------------
// CONSOLE COACH (prints actionable tuning tips)
// ------------------------------------------------------
const LOG_EVERY_MS = 800; let _lastLog = 0;
function ok(v){ return v ? '✅' : '✖'; }
function deg(x){ return x.toFixed(1)+'°'; }

function printDebugGuidance(lm, shown, target){
  const now = performance.now(); if(now - _lastLog < LOG_EVERY_MS) return; _lastLog = now;
  const pw = palmWidth(lm);

  // Core measures
  const dThumb_IndexMCP = ndist(lm,4,5,pw);
  const dIdxMidTips     = ndist(lm,8,12,pw);
  const dThumb_PinkyTip = ndist(lm,4,20,pw);
  const thumbNorm       = thumbSpreadNorm(lm,pw);

  const ix=lm[8].x-lm[5].x, iy=lm[8].y-lm[5].y;
  const tx=lm[4].x-lm[2].x, ty=lm[4].y-lm[2].y;
  const mx=lm[12].x-lm[9].x, my=lm[12].y-lm[9].y;
  const i_t_deg = rayAngleDeg(ix,iy,tx,ty);
  const i_m_deg = rayAngleDeg(ix,iy,mx,my);

  const f = fingersUp(lm);
  const allFourDown = (f[1]===0 && f[2]===0 && f[3]===0 && f[4]===0);
  const pinkyUpOnly = (f[4]===1 && f[1]===0 && f[2]===0 && f[3]===0);
  const twoUp_IM    = (f[1]===1 && f[2]===1 && f[3]===0 && f[4]===0);
  const L_shape     = (f[1]===1 && f[0]===1 && f[2]===0 && f[3]===0 && f[4]===0);
  const Y_shape     = (f[0]===1 && f[4]===1 && f[1]===0 && f[2]===0 && f[3]===0);

  console.groupCollapsed(
    `%c[DEBUG] pred:${shown}  target:${target}`,
    'color:#444;background:#eef;padding:2px 6px;border-radius:6px'
  );

  console.log('thumb→indexMCP=%s | idx↔mid tips=%s | i↔t=%s | i↔m=%s | thumbNorm=%s | thumb↔pinky=%s',
    dThumb_IndexMCP.toFixed(3), dIdxMidTips.toFixed(3), deg(i_t_deg), deg(i_m_deg),
    thumbNorm.toFixed(3), dThumb_PinkyTip.toFixed(3)
  );
  console.log('fingersUp [T,I,M,R,P] =', f);

  switch(target){
    case 'A': {
      const pass = (allFourDown && f[0]===1 && dThumb_IndexMCP >= TH.A_thumbAwayFromIndexMCP);
      console.log('A rules:', ok(pass));
      console.log('  allFourDown:', ok(allFourDown), ' thumbOut:', ok(f[0]===1));
      console.log('  thumb→indexMCP %s  vs  TH.A_thumbAwayFromIndexMCP %s',
        dThumb_IndexMCP.toFixed(3), TH.A_thumbAwayFromIndexMCP.toFixed(3));
      console.log('  Tune: If A rarely triggers → LOWER A_thumbAwayFromIndexMCP by ~0.02; if false A → RAISE it.');
      break;
    }
    case 'I': {
      const pass = (pinkyUpOnly && f[0]===0);
      console.log('I rules:', ok(pass));
      console.log('  pinkyUpOnly (P=1, I/M/R=0):', ok(pinkyUpOnly), ' thumb folded:', ok(f[0]===0));
      console.log('  Tune: If false I → enforce a stronger pinky-vs-ring height delta or ensure thumbOut=0 (tighten TH.thumbSpread).');
      break;
    }
    case 'L': {
      const idxPIP = angleAtPIP(lm,8,7,6);
      const indexStraight = idxPIP >= TH.L_indexStraightDeg;
      const othersBent = (ndist(lm,12,10,pw)<TH.L_otherBentTol) && (ndist(lm,16,14,pw)<TH.L_otherBentTol) && (ndist(lm,20,18,pw)<TH.L_otherBentTol);
      const angleOK = (i_t_deg >= TH.L_requiredAngleMinDeg && i_t_deg <= TH.L_requiredAngleMaxDeg);
      const thumbOK = thumbNorm > TH.L_minThumbSpread;
      const pass = (L_shape && angleOK && thumbOK && indexStraight && othersBent);

      console.log('L rules:', ok(pass));
      console.log('  L_shape:', ok(L_shape));
      console.log('  i↔t angle %s in [%s,%s]: %s', deg(i_t_deg), TH.L_requiredAngleMinDeg, TH.L_requiredAngleMaxDeg, ok(angleOK));
      console.log('  index PIP %s ≥ %s: %s', deg(idxPIP), TH.L_indexStraightDeg, ok(indexStraight));
      console.log('  thumbNorm %s > %s: %s', thumbNorm.toFixed(3), TH.L_minThumbSpread.toFixed(3), ok(thumbOK));
      console.log('  othersBent (M,R,P):', ok(othersBent));
      console.log('  Tune: Misses → widen angle band / LOWER L_minThumbSpread / LOWER L_indexStraightDeg. False L → narrow band / RAISE spreads.');
      break;
    }
    case 'V': {
      const sepOK = dIdxMidTips >= TH.V_sepMin;
      const angOK = i_m_deg >= TH.V_angleMinDeg;
      const pass = (twoUp_IM && sepOK && angOK);
      console.log('V rules:', ok(pass));
      console.log('  twoUp_IM (I & M only):', ok(twoUp_IM));
      console.log('  tipsSep %s ≥ %s: %s', dIdxMidTips.toFixed(3), TH.V_sepMin.toFixed(3), ok(sepOK));
      console.log('  i↔m angle %s ≥ %s: %s', deg(i_m_deg), TH.V_angleMinDeg, ok(angOK));
      console.log('  Tune: Misses → LOWER V_sepMin or LOWER V_angleMinDeg. False V → RAISE them. Hysteresis V_hyst smooths borders.');
      break;
    }
    case 'Y': {
      const thumbOK = thumbNorm > TH.Y_thumbSpread;
      const spanOK  = dThumb_PinkyTip > 0.35;
      const pass = (Y_shape && thumbOK && spanOK);
      console.log('Y rules:', ok(pass));
      console.log('  Y_shape:', ok(Y_shape));
      console.log('  thumbNorm %s > %s: %s', thumbNorm.toFixed(3), TH.Y_thumbSpread.toFixed(3), ok(thumbOK));
      console.log('  thumb↔pinky span %s > 0.35: %s', dThumb_PinkyTip.toFixed(3), ok(spanOK));
      console.log('  Tune: Misses → LOWER Y_thumbSpread or LOWER span (0.35→0.33). False Y → RAISE them.');
      break;
    }
  }
  console.groupEnd();
}


// ------------------------------------------------------
// FRAME HANDLER (core loop)
// ------------------------------------------------------
function onResults(res){
  // Canvas = video size (so drawings align)
  if (canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  // Draw frame background
  ctx.save();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(res.image, 0, 0, canvas.width, canvas.height);

  // Landmarks → classify
  let lm = null;
  if (res.multiHandLandmarks && res.multiHandLandmarks.length) {
    lm = res.multiHandLandmarks[0];

    // Visual overlay
    drawConnectors(ctx, lm, HAND_CONNECTIONS, { color:'#22c55e', lineWidth:3 });
    drawLandmarks(ctx, lm, { color:'#111', lineWidth:1, radius:2 });

    const pred = classifyLetters(lm);
    hist.push(pred); if (hist.length > SMOOTH_N) hist.shift();
  } else {
    hist.push('?');  if (hist.length > SMOOTH_N) hist.shift();
  }

  // Majority vote → UI
  const shown = majority(hist);
  $pred.textContent = shown;

  // Streak vs target
  if (shown === targetLetter) {
    streak++; $conf.textContent = `${Math.min(streak, CONFIRM_FRAMES)} / ${CONFIRM_FRAMES}`;
  } else {
    streak = 0; sentForThisSuccess = false; $conf.textContent = `0 / ${CONFIRM_FRAMES}`;
  }

  // Send once when confirmed
  if (streak >= CONFIRM_FRAMES) {
    $status.textContent = '✓ Correct'; $status.className = 'ok';
    if (!sentForThisSuccess) { sendToRobot(targetLetter); sentForThisSuccess = true; }
  } else {
    $status.textContent = `✗ Show ${targetLetter}`; $status.className = 'bad';
  }

  // Small on-screen debug (for quick glance)
  if ($debug.style.display==='block' && lm) {
    const pw = palmWidth(lm);
    const dThumb_IndexMCP = ndist(lm,4,5,pw);
    const dIdxMidTips = ndist(lm,8,12,pw);
    const i_m_deg = rayAngleDeg(
      lm[8].x-lm[5].x, lm[8].y-lm[5].y,
      lm[12].x-lm[9].x, lm[12].y-lm[9].y
    );

    $debug.textContent =
`pred:${shown}  target:${targetLetter}
A/I thumb→indexMCP: ${dThumb_IndexMCP.toFixed(3)}
V   tipsSep: ${dIdxMidTips.toFixed(3)}  i↔m: ${i_m_deg.toFixed(1)}°`;
  }

  // Console coach: prints actionable tuning guidance
  if (lm) printDebugGuidance(lm, shown, targetLetter);

  ctx.restore();
}


// ------------------------------------------------------
// INIT: request camera, then start MediaPipe pipeline
// ------------------------------------------------------
$conf.textContent = `0 / ${CONFIRM_FRAMES}`;
smokeTestCamera().then(()=> cam.start());
