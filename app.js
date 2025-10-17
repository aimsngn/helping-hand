// =============================================================
//  CONFIGURATION
// =============================================================
const SMOOTH_N = 5;       // majority window size to reduce jitter
const CONFIRM_FRAMES = 4; // consecutive frames required to confirm

// Thresholds (normalized by palm width unless noted)
const TH = {
  // Letter A & B helpers
  A_thumbNearIndexMCP: 0.60,
  B_thumbAcrossPalm:   0.65,
  B_minStraightLift:   0.015, // (unused; placeholder)
  B_strictLiftBonus:   0.035, // (unused; placeholder)

  // U vs V discriminators + hysteresis
  UV_normU: 0.45,
  UV_normV: 0.55,
  UV_cosU:  0.95,
  UV_cosV:  0.90,
  UV_baseU: 0.22,
  UV_baseV: 0.20,
  UV_hyst:  0.04
};

// =============================================================
//  DOM REFERENCES
// =============================================================
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const $pred = document.getElementById('prediction');
const $status = document.getElementById('status');
const $conf = document.getElementById('conf');
const $target = document.getElementById('target');
const $connectBtn = document.getElementById('connectBtn');
const $serialStatus = document.getElementById('serialStatus');
const $debug = document.getElementById('debug');
const $dbgToggle = document.getElementById('dbgToggle');
const $errbar = document.getElementById('errbar');
const $lastSent = document.getElementById('lastSent');

// Toggle debug panel visibility
$dbgToggle.addEventListener('click', ()=>{
  $debug.style.display = ($debug.style.display==='none'||!$debug.style.display)?'block':'none';
});

// Keep track of the currently selected target letter (A/B/U/V/W)
let targetLetter = $target.value;
$target.addEventListener('change', () => {
  targetLetter = $target.value;
  streak = 0;
  sentForThisSuccess = false;
  $conf.textContent = `0 / ${CONFIRM_FRAMES}`;
  // Optionally relax robot: sendBits([0,0,0,0,0]);
});

// =============================================================
//  GEOMETRY / UTILS
// =============================================================
function dist(a,b){ const dx=a.x-b.x, dy=a.y-b.y; return Math.hypot(dx,dy); }
function palmWidth(lm){ return Math.max(dist(lm[5], lm[17]), 1e-6); } // MCP(Idx) ↔ MCP(Pinky)
function ndist(lm,i,j,pw){ return dist(lm[i], lm[j]) / pw; }           // normalized distance
function cosSim(ax,ay,bx,by){ const dot=ax*bx+ay*by, ma=Math.hypot(ax,ay)||1e-6, mb=Math.hypot(bx,by)||1e-6; return dot/(ma*mb); }

// Quick heuristic: which fingers are up (1) vs down (0)
function fingersUp(lm){
  const upIndex  = lm[8].y  < lm[6].y  ? 1 : 0;
  const upMiddle = lm[12].y < lm[10].y ? 1 : 0;
  const upRing   = lm[16].y < lm[14].y ? 1 : 0;
  const upPinky  = lm[20].y < lm[18].y ? 1 : 0;
  const thumbSpread = Math.abs(lm[4].x - lm[2].x);
  const thumbOut = thumbSpread > 0.03 ? 1 : 0;
  return [thumbOut, upIndex, upMiddle, upRing, upPinky]; // [T,I,M,R,P]
}

// =============================================================
//  CLASSIFIER (A, B, U, V, W)
// =============================================================
let uvLatch = '?';

function classifyLetters(lm){
  const pw = palmWidth(lm);
  const f = fingersUp(lm);

  // Distances
  const d_4_5  = ndist(lm, 4, 5, pw);   // A: thumb tip ↔ index MCP
  const d_4_9  = ndist(lm, 4, 9, pw);   // B: thumb tip ↔ middle MCP
  const d_8_12 = ndist(lm, 8, 12, pw);  // U/V: index ↔ middle tips
  const d_5_9  = ndist(lm, 5, 9, pw);   // U/V: base MCP spread

  // Finger lengths for normalization (MCP→TIP)
  const idxLen = ndist(lm, 5, 8, pw);
  const midLen = ndist(lm, 9, 12, pw);
  const avgLen = Math.max((idxLen + midLen) * 0.5, 1e-6);
  const normSep = d_8_12 / avgLen;

  // Bent-ness (A helper)
  const d_8_6 = ndist(lm, 8, 6, pw);
  const d_12_10 = ndist(lm, 12, 10, pw);
  const d_16_14 = ndist(lm, 16, 14, pw);
  const d_20_18 = ndist(lm, 20, 18, pw);
  const bent = [d_8_6,d_12_10,d_16_14,d_20_18].map(v=>v<0.60?1:0);
  const bentCount = bent.reduce((a,b)=>a+b,0);

  // U/V angle (MCP→TIP vectors)
  const ix = lm[8].x - lm[5].x,  iy = lm[8].y - lm[5].y;
  const mx = lm[12].x - lm[9].x, my = lm[12].y - lm[9].y;
  const cos = cosSim(ix, iy, mx, my);

  // --- A ---
  const noFingersUp = (f[1]===0 && f[2]===0 && f[3]===0 && f[4]===0);
  if ((noFingersUp || bentCount>=3) && d_4_5 < TH.A_thumbNearIndexMCP) return 'A';

  // --- B ---
  if (f[1] && f[2] && f[3] && f[4] && d_4_9 < TH.B_thumbAcrossPalm) return 'B';

  // Patterns for U/V/W
  const twoUp   = (f[1] && f[2] && !f[3] && !f[4]);
  const threeUp = (f[1] && f[2] && f[3] && !f[4]);

  // --- U/V with hysteresis ---
  if (twoUp){
    const isUhard = (normSep <= TH.UV_normU && cos >= TH.UV_cosU && d_5_9 <= TH.UV_baseU);
    const isVhard = (normSep >= TH.UV_normV && (cos <= TH.UV_cosV || d_5_9 >= TH.UV_baseV));
    if (isUhard){ uvLatch='U'; return 'U'; }
    if (isVhard){ uvLatch='V'; return 'V'; }

    if (uvLatch === 'U'){
      const needFlip = (normSep >= (TH.UV_normV - TH.UV_hyst)) && (cos <= (TH.UV_cosV + 0.02));
      if (!needFlip) return 'U';
      uvLatch = 'V'; return 'V';
    }
    if (uvLatch === 'V'){
      const needFlip = (normSep <= (TH.UV_normU + TH.UV_hyst)) && (cos >= (TH.UV_cosU - 0.02));
      if (!needFlip) return 'V';
      uvLatch = 'U'; return 'U';
    }

    uvLatch = (normSep < (TH.UV_normU + TH.UV_normV)/2 && cos >= (TH.UV_cosU + TH.UV_cosV)/2) ? 'U' : 'V';
    return uvLatch;
  }

  // --- W ---
  if (threeUp) return 'W';

  uvLatch = '?';
  return '?';
}

// =============================================================
//  SMOOTHING & GATE
// =============================================================
const hist = [];
let streak = 0;
let sentForThisSuccess = false;
function majority(arr){
  const c = Object.create(null);
  for (const v of arr) c[v] = (c[v]||0)+1;
  let best='?', cnt=0; for (const k in c){ if (c[k]>cnt){ best=k; cnt=c[k]; } }
  return best;
}

// =============================================================
//  WEB SERIAL (Optional Robot Control)
// =============================================================
let port = null, writer = null, serialConnected = false;

function updateSerialUI(){
  if(serialConnected){
    $connectBtn.textContent = 'Disconnect Robot';
    $serialStatus.textContent = 'Connected';
    $serialStatus.className = 'muted';
  } else {
    $connectBtn.textContent = 'Connect Robot (Web Serial)';
    $serialStatus.textContent = navigator.serial ? 'Not connected' : 'Unsupported in this browser';
    $serialStatus.className = 'muted';
  }
}

async function connectSerial(){
  try{
    port = await navigator.serial.requestPort();
    await port.open({ baudRate: 9600 }); // match Arduino sketch
    writer = port.writable.getWriter();
    serialConnected = true; updateSerialUI();
  }catch(err){ console.error(err); serialConnected = false; updateSerialUI(); }
}
async function disconnectSerial(){
  try{
    if(writer){ writer.releaseLock(); writer = null; }
    if(port){ await port.close(); port = null; }
  }catch(e){} finally{ serialConnected = false; updateSerialUI(); }
}

// Letter → 5-bit (T,I,M,R,P). 1=open/straight, 0=closed/curled.
function letterToBits(L){
  const map = {
    'A': [0,0,0,0,0],
    'B': [0,1,1,1,1],
    'U': [0,1,1,0,0],
    'V': [0,1,1,0,0],
    'W': [0,1,1,1,0],
    'R': [0,0,0,0,0],
  };
  return map[L] || [0,0,0,0,0];
}

async function sendBits(bits){
  if(!serialConnected || !writer) return;
  const payload = `$${bits.join('')}`; // e.g. $01111
  try{
    await writer.write(new TextEncoder().encode(payload));
    $lastSent.textContent = payload;
  }catch(err){ console.error('Serial write failed', err); await disconnectSerial(); }
}

async function sendToRobot(letter){
  const bits = letterToBits(letter);
  await sendBits(bits);
}

if('serial' in navigator){
  $connectBtn.disabled = false;
  $connectBtn.addEventListener('click', () => {
    if(serialConnected) disconnectSerial(); else connectSerial();
  });
} else {
  $connectBtn.disabled = true;
}
updateSerialUI();

// =============================================================
//  MEDIAPIPE HANDS SETUP
// =============================================================
const hands = new Hands({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
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

async function startCamera(){
  try{ await cam.start(); }
  catch(err){
    console.error(err);
    $errbar.textContent = 'Camera unavailable. Use HTTPS or http://localhost and allow camera access.';
    $errbar.style.display = 'block';
    $status.textContent = 'Camera error';
    $status.className = 'bad';
  }
}

// =============================================================
//  FRAME RESULT HANDLER
// =============================================================
function onResults(res){
  if (canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  ctx.save();
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(res.image, 0, 0, canvas.width, canvas.height);

  let lm = null;
  if (res.multiHandLandmarks && res.multiHandLandmarks.length) {
    lm = res.multiHandLandmarks[0];
    drawConnectors(ctx, lm, HAND_CONNECTIONS, {color:'#22c55e', lineWidth:3});
    drawLandmarks(ctx, lm, {color:'#111', lineWidth:1, radius:2});

    const pred = classifyLetters(lm);
    hist.push(pred);
    if (hist.length > SMOOTH_N) hist.shift();
  } else {
    hist.push('?'); if (hist.length > SMOOTH_N) hist.shift();
  }

  const shown = majority(hist);
  $pred.textContent = shown;

  if (shown === targetLetter) {
    streak++; $conf.textContent = `${Math.min(streak, CONFIRM_FRAMES)} / ${CONFIRM_FRAMES}`;
  } else {
    streak = 0; sentForThisSuccess = false; $conf.textContent = `0 / ${CONFIRM_FRAMES}`;
  }

  if (streak >= CONFIRM_FRAMES) {
    $status.textContent = '✓ Correct'; $status.className = 'ok';
    if (!sentForThisSuccess){ sendToRobot(targetLetter); sentForThisSuccess = true; }
  } else {
    $status.textContent = `✗ Show ${targetLetter}`; $status.className = 'bad';
  }

  if ($debug.style.display==='block' && lm) {
    const pw = palmWidth(lm);
    const d_4_5 = ndist(lm, 4, 5, pw);
    const d_4_9 = ndist(lm, 4, 9, pw);
    const d_8_12 = ndist(lm, 8, 12, pw);
    const d_5_9  = ndist(lm, 5, 9, pw);
    const idxLen = ndist(lm, 5, 8, pw);
    const midLen = ndist(lm, 9, 12, pw);
    const avgLen = Math.max((idxLen + midLen) * 0.5, 1e-6);
    const normSep = d_8_12 / avgLen;
    const ix = lm[8].x - lm[5].x,  iy = lm[8].y - lm[5].y;
    const mx = lm[12].x - lm[9].x, my = lm[12].y - lm[9].y;
    const cos = cosSim(ix, iy, mx, my);
    $debug.textContent =
`pred:${shown} target:${targetLetter}
A d(4,5)=${d_4_5.toFixed(3)}  B d(4,9)=${d_4_9.toFixed(3)}
U/V: tipSep=${d_8_12.toFixed(3)} base=${d_5_9.toFixed(3)} normSep=${normSep.toFixed(3)} cos=${cos.toFixed(3)} latch=${uvLatch}`;
  }

  ctx.restore();
}

// =============================================================
//  INIT
// =============================================================
$conf.textContent = `0 / ${CONFIRM_FRAMES}`;
startCamera();
