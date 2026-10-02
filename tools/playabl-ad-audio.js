// Synthesizes the soundtrack for playabl-ad.html: an original track (tempo set by the ad's cuts) plus
// sound effects timed to the ad's timeline. Writes a 44.1kHz stereo WAV.
//   node tools/playabl-ad-audio.js playabl-ad-events.json out.wav
// The cue file comes from `node tools/render-playabl-ad.js events …`, so every effect
// lands on the exact frame it belongs to and the beat grid follows the ad's cuts.
const fs = require('fs');

const CUES = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const SR = 44100, DUR = CUES.duration / 1000, N = Math.round(SR * DUR);
const L = new Float32Array(N), R = new Float32Array(N);       // dry mix
const RL = new Float32Array(N), RR = new Float32Array(N);     // reverb send
const DL = new Float32Array(N), DR = new Float32Array(N);     // echo send
const duck = new Float32Array(N).fill(1);                     // sidechain gain for pads/bass

let seed = 1234;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const noise = () => rnd() * 2 - 1;
const mtof = m => 440 * Math.pow(2, (m - 69) / 12);
const TAU = Math.PI * 2;

function put(i, v, pan = 0, rev = 0, echo = 0){
  if (i < 0 || i >= N) return;
  const gl = Math.cos((pan + 1) * Math.PI / 4), gr = Math.sin((pan + 1) * Math.PI / 4);
  L[i] += v * gl; R[i] += v * gr;
  if (rev){ RL[i] += v * gl * rev; RR[i] += v * gr * rev; }
  if (echo){ DL[i] += v * gl * echo; DR[i] += v * gr * echo; }
}
// TPT state-variable filter
function svf(){
  let ic1 = 0, ic2 = 0;
  return (x, fc, q = .707) => {
    const g = Math.tan(Math.PI * Math.min(fc, SR * .45) / SR), k = 1 / q;
    const a1 = 1 / (1 + g * (g + k)), a2 = g * a1, a3 = g * a2;
    const v3 = x - ic2, v1 = a1 * ic1 + a2 * v3, v2 = ic2 + a2 * ic1 + a3 * v3;
    ic1 = 2 * v1 - ic1; ic2 = 2 * v2 - ic2;
    return {lp: v2, bp: v1, hp: x - k * v1 - v2};
  };
}

// ---------------- musical grid ----------------
const GROOVE = CUES.groove / 1000, DROP = CUES.drop / 1000;
const BAR = (DROP - GROOVE) / (CUES.bars || 7);   // whole bars between the groove start and the end-card drop
const BEAT = BAR / 4, T0 = GROOVE - 3 * BAR;
const barT = k => T0 + k * BAR;
const TR = CUES.transpose || 0;        // per-ad key change, in semitones
const CH = Object.fromEntries(Object.entries({Am:[57,60,64], F:[57,60,65], C:[55,60,64], G:[55,59,62]}).map(([k, v]) => [k, v.map(m => m + TR)]));
const ROOT = Object.fromEntries(Object.entries({Am:45, F:41, C:36, G:43}).map(([k, v]) => [k, v + TR]));
// 3 intro bars, NB groove bars (the last one is the build), then the drop and a tail bar
const NB = CUES.bars || 7, KB = 3 + NB - 1, KD = KB + 1, KT = KD + 1;
const PROG = Array.from({length: KT + 1}, (_, k) => k < KB - 1 ? ['Am','F','C','G'][k % 4] : k === KB - 1 ? 'F' : k === KB ? 'G' : 'C');

// ---------------- instruments ----------------
function kick(t, vol = .9){
  const s = Math.round(t * SR), len = SR * .4;
  let ph = 0;
  for (let i = 0; i < len; i++){
    const x = i / SR, f = 45 + 110 * Math.exp(-x * 28);
    ph += TAU * f / SR;
    const v = Math.sin(ph) * Math.exp(-x * 7) + (i < 90 ? noise() * .3 * (1 - i / 90) : 0);
    put(s + i, v * vol);
  }
  for (let i = 0; i < SR * .3; i++) if (s + i >= 0 && s + i < N) duck[s + i] = Math.min(duck[s + i], 1 - .55 * Math.exp(-i / SR / .09));
}
function clap(t, vol = .45){
  const s = Math.round(t * SR), f = svf();
  for (let i = 0; i < SR * .25; i++){
    const x = i / SR, burst = x < .03 ? (Math.floor(x / .01) % 1 === 0 ? Math.exp(-(x % .01) * 300) : 0) : Math.exp(-(x - .03) * 22);
    put(s + i, f(noise(), 1600, 1.2).bp * burst * vol * 2, .1, .35);
  }
}
function snare(t, vol = .4){
  const s = Math.round(t * SR), f = svf();
  for (let i = 0; i < SR * .18; i++){
    const x = i / SR;
    const v = f(noise(), 2200, .8).bp * Math.exp(-x * 25) * 1.6 + Math.sin(TAU * 185 * x) * Math.exp(-x * 30) * .5;
    put(s + i, v * vol, -.05, .25);
  }
}
function hat(t, vol = .12, open = false, pan = .25){
  const s = Math.round(t * SR), f = svf(), d = open ? 14 : 55;
  for (let i = 0; i < SR * (open ? .25 : .06); i++) put(s + i, f(noise(), 8000).hp * Math.exp(-i / SR * d) * vol, pan, .1);
}
function bass(t, dur, midi, vol = .32){
  const s = Math.round(t * SR), len = Math.round(dur * SR), f = svf(), fr = mtof(midi);
  let ph = 0;
  for (let i = 0; i < len; i++){
    const x = i / SR; ph = (ph + fr / SR) % 1;
    const raw = (2 * ph - 1) * .6 + Math.sin(TAU * ph) * .7;
    const env = Math.min(1, x / .005) * Math.min(1, (len - i) / (SR * .02)) * (.65 + .35 * Math.exp(-x * 6));
    const v = f(raw, 300 + 900 * Math.exp(-x * 9), 1.1).lp * env * vol;
    put(s + i, v * duck[Math.min(N - 1, Math.max(0, s + i))]);
  }
}
function pad(t, dur, notes, vol, cutoff){
  const s = Math.round(t * SR), len = Math.round(dur * SR);
  notes.forEach((m, ni) => {
    [-.09, 0, .1].forEach((det, vi) => {
      const fr = mtof(m + det), f = svf(), pan = (vi - 1) * .55;
      let ph = rnd();
      for (let i = 0; i < len; i++){
        const x = i / SR; ph = (ph + fr / SR) % 1;
        const env = Math.min(1, x / .35) * Math.min(1, (len - i) / (SR * .3));
        const c = typeof cutoff === 'function' ? cutoff(t + x) : cutoff;
        const v = f(2 * ph - 1, c, .9).lp * env * vol / 3;
        put(s + i, v * duck[Math.min(N - 1, s + i)], pan, .45);
      }
    });
    void ni;
  });
}
function pluck(t, midi, vol = .13, pan = 0, bright = 1){
  const s = Math.round(t * SR), fr = mtof(midi), f = svf();
  let ph = 0;
  for (let i = 0; i < SR * .35; i++){
    const x = i / SR; ph = (ph + fr / SR) % 1;
    const raw = (ph < .5 ? 1 : -1) * .5 + (2 * Math.abs(2 * ph - 1) - 1) * .6;
    const v = f(raw, 600 + 5000 * bright * Math.exp(-x * 18), 1).lp * Math.exp(-x * 9) * vol;
    put(s + i, v, pan, .3, .45);
  }
}

// ---------------- sound effects ----------------
function whoosh(t, dur = .65, vol = .22, dir = 1){
  const s = Math.round((t - dur * .6) * SR), len = Math.round(dur * SR), f = svf();
  for (let i = 0; i < len; i++){
    const p = i / len, env = Math.pow(Math.sin(Math.PI * Math.min(1, p * 1.25)), 2) * (p < .8 ? 1 : (1 - p) / .2);
    const fc = 250 * Math.pow(18, dir > 0 ? p : 1 - p);
    put(s + i, f(noise(), fc, 2.2).bp * env * vol * 2.2, (p * 2 - 1) * .6 * dir, .3);
  }
}
function click(t, vol = .06){
  const s = Math.round(t * SR), f = svf(), tone = 1800 + rnd() * 900;
  for (let i = 0; i < SR * .02; i++){
    const x = i / SR;
    put(s + i, (f(noise(), 5000).hp * .7 + Math.sin(TAU * tone * x) * .5) * Math.exp(-x * 320) * vol * (.7 + rnd() * .5), (rnd() - .5) * .3);
  }
}
function pop(t, vol = .3, f0 = 950, f1 = 320, pan = 0){
  const s = Math.round(t * SR); let ph = 0;
  for (let i = 0; i < SR * .12; i++){
    const x = i / SR, fr = f1 + (f0 - f1) * Math.exp(-x * 45);
    ph += TAU * fr / SR;
    put(s + i, Math.sin(ph) * Math.exp(-x * 38) * vol, pan, .15);
  }
}
function bell(t, midi, vol = .14, pan = 0, decay = 2.2){
  const s = Math.round(t * SR), fr = mtof(midi);
  const parts = [[1, 1], [2.01, .45], [3.0, .2], [4.2, .12]];
  for (let i = 0; i < SR * 1.4; i++){
    const x = i / SR; let v = 0;
    for (const [m, a] of parts) v += Math.sin(TAU * fr * m * x) * a * Math.exp(-x * decay * m);
    put(s + i, v * vol * Math.min(1, x / .002), pan, .5, .25);
  }
}
function boing(t, vol = .2){
  const s = Math.round(t * SR); let ph = 0;
  for (let i = 0; i < SR * .45; i++){
    const x = i / SR, fr = 260 + 380 * (x / .45) + Math.sin(x * 70) * 60 * Math.exp(-x * 5);
    ph += TAU * fr / SR;
    put(s + i, Math.sin(ph) * Math.exp(-x * 6) * Math.min(1, x / .01) * vol, .35, .25);
  }
}
function coin(t, vol = .16){
  // two-step coin blip + metallic shimmer
  [[83, 0], [88, .075]].forEach(([m, dt]) => {
    const s = Math.round((t + dt) * SR), fr = mtof(m); let ph = 0;
    const len = SR * (dt ? .45 : .075);
    for (let i = 0; i < len; i++){
      const x = i / SR; ph = (ph + fr / SR) % 1;
      put(s + i, (ph < .5 ? 1 : -1) * .5 * vol * (dt ? Math.exp(-x * 6) : 1), .15, .35, .2);
    }
  });
  const s = Math.round((t + .05) * SR), f = svf();
  for (let i = 0; i < SR * .6; i++) put(s + i, f(noise(), 9000, 4).bp * Math.exp(-i / SR * 6) * vol * 1.6, -.2, .4);
}
function riser(t0, t1, vol = .28){
  const s = Math.round(t0 * SR), len = Math.round((t1 - t0) * SR), f = svf(); let ph = 0;
  for (let i = 0; i < len; i++){
    const p = i / len, fr = 110 * Math.pow(8, p);
    ph = (ph + fr / SR) % 1;
    const v = f(noise(), 300 * Math.pow(30, p), 3).bp * 1.5 + (2 * ph - 1) * .25;
    put(s + i, v * Math.pow(p, 1.8) * vol, Math.sin(p * 18) * .4 * p, .35);
  }
}
function impact(t, vol = .8){
  const s = Math.round(t * SR); let ph = 0; const f = svf();
  for (let i = 0; i < SR * 2.2; i++){
    const x = i / SR, fr = 32 + 60 * Math.exp(-x * 6);
    ph += TAU * fr / SR;
    put(s + i, Math.sin(ph) * Math.exp(-x * 2.2) * vol);
    put(s + i, f(noise(), 5500).lp * Math.exp(-x * 2.6) * vol * .35, 0, .6);
  }
  for (let i = 0; i < SR * .5; i++) if (s + i < N) duck[s + i] = Math.min(duck[s + i], 1 - .7 * Math.exp(-i / SR / .25));
}

// ---------------- arrangement ----------------
const BUILD = barT(KB);
for (let k = 0; k < PROG.length; k++){
  const t = barT(k); if (t >= DUR) break;
  const ch = PROG[k], notes = CH[ch];
  const beats = [0, 1, 2, 3].map(b => t + b * BEAT);
  const full = k >= 3 && k < KB, drop = k >= KD;

  // drums
  if (full || drop){ beats.forEach(b => kick(b, drop ? .85 : .75)); }
  else if (k === 2){ kick(beats[0], .7); kick(beats[2], .7); }
  if (k === KB){ kick(beats[0], .8); kick(beats[1], .8); }
  if (full || drop){ clap(beats[1]); clap(beats[3]); }
  if (k >= 1 && k !== KB) for (let e = 0; e < 8; e++){
    const tt = t + e * BEAT / 2;
    hat(tt, (e % 2 ? .13 : .07) * (k === 1 ? .6 : 1), (full || drop) && e % 4 === 2, e % 2 ? .3 : -.2);
  }
  if (k === KB){ // snare roll building into the drop
    let tt = t + BEAT * 2, step = BEAT / 2, n = 0;
    while (tt < DROP - .02){ snare(tt, .12 + .3 * ((tt - t) / BAR)); tt += step; if (++n % 2 === 0 && step > BEAT / 8) step /= 2; }
  }

  // bass: pumping 8ths
  if (k >= 2 && k !== KB && k < KT){
    for (let e = 0; e < 8; e++){
      const oct = (e === 3 || e === 7) ? 12 : 0;
      bass(t + e * BEAT / 2, BEAT / 2 * .9, ROOT[ch] + oct, k === 2 ? .22 : .3);
    }
  }
  if (k === KB) bass(t, BAR * .5, ROOT[ch], .28);

  // pads: filter opens through the intro, wide open on the drop
  const cut = tt => tt < GROOVE ? 500 + 1800 * Math.min(1, tt / GROOVE) : tt < BUILD ? 2300 : tt < DROP ? 2300 + 3500 * ((tt - BUILD) / (DROP - BUILD)) : 4200;
  pad(t, k >= KT ? DUR - t : BAR + .05, notes.map(m => m - 12).concat(notes), k >= KD ? .16 : k < 2 ? .2 : .12, cut);

  // arp: 16ths over chord tones
  if (k !== KB && k < KT){
    const pat = [0, 1, 2, 3, 2, 1, 4, 2, 0, 1, 2, 3, 2, 4, 3, 1];
    const tones = notes.concat(notes.map(m => m + 12));
    for (let e = 0; e < 16; e++){
      const bright = Math.min(1, .25 + t / 8);
      pluck(t + e * BEAT / 4, tones[pat[e]] + 12, (k === 0 ? .07 : .1) * (e % 4 === 0 ? 1.15 : 1), e % 2 ? .35 : -.35, bright);
    }
  }
}
// final chord swell + sparkle on the end card
[72, 76, 79, 84].forEach((m, i) => bell(DROP + .1 + i * .09, m, .09, (i - 1.5) * .3, 1.6));
riser(BUILD, DROP - .03, .26);

// ---------------- SFX: one sound per cue exported by the ad (AD.events) ----------------
for (const e of CUES.events){
  const t = e.t / 1000;
  switch (e.type){
    case 'key':    click(t, .055); break;
    case 'back':   click(t, .045); break;
    case 'tap':    pop(t, .14, 520, 260); break;
    case 'send':   pop(t, .3, 950, 320); break;
    case 'cut':    whoosh(t + .05, .55, .2, 1); break;
    case 'swipe':  whoosh(t + .2, .5, .2, -1); break;
    case 'boot':   [79, 84, 88].forEach((m, i) => bell(t + i * .08, m, .08, .2, 3)); break;
    case 'reply':  pop(t, .07, 1500, 1000, .3); break;
    case 'rivals': boing(t + .1); boing(t + .25, .13); break;
    case 'eat':    pop(t, .05, 1700, 1100, (rnd() - .5) * .5); break;
    case 'like':   pop(t, .22, 700, 1100, .2); bell(t + .02, 84, .05, .2, 4); break;
    case 'tick':   pop(t, .04, 2600, 1900, (rnd() - .5) * .6); break;
    case 'coin':   coin(t, e.i === 0 ? .15 : .08); break;
    case 'stat':   kick(t, .55); pop(t, .2, 600 + e.i * 150, 300 + e.i * 80); bell(t, 76 + e.i * 3, .06, 0, 4); break;
    case 'drop':   impact(t, .75); break;
    case 'hop':    pop(t, .045, 420, 760, (rnd() - .5) * .6); break;
    case 'flip':   pop(t, .06, 500, 1200, (rnd() - .5) * .6); whoosh(t + .2, .25, .06, 1); break;
    case 'upgrade': whoosh(t + .1, .45, .14, 1); [76, 79, 83, 88].forEach((m, i) => bell(t + i * .06, m, .07, (i - 1.5) * .3, 3)); break;
  }
}

// ---------------- effects buses ----------------
// ping-pong dotted-8th echo
{
  const d = Math.round(BEAT * .75 * SR), fb = .38;
  const bl = new Float32Array(N), br = new Float32Array(N);
  for (let i = 0; i < N; i++){
    const il = i >= d ? br[i - d] : 0, ir = i >= d ? bl[i - d] : 0;
    bl[i] = DL[i] + il * fb; br[i] = DR[i] + ir * fb;
    RL[i] += il * .2; RR[i] += ir * .2;
    L[i] += il * .35; R[i] += ir * .35;
  }
}
// Schroeder reverb
function reverb(inp, offset){
  const out = new Float32Array(N);
  for (const [dl, fb] of [[1557, .86], [1617, .85], [1491, .87], [1422, .86], [1277, .85], [1356, .86]]){
    const D = dl + offset, buf = new Float32Array(D); let p = 0, lp = 0;
    for (let i = 0; i < N; i++){
      const y = buf[p]; lp = y * .7 + lp * .3;
      buf[p] = inp[i] + lp * fb; p = (p + 1) % D; out[i] += y / 6;
    }
  }
  for (const D of [225 + offset, 556 + offset, 441]){
    const buf = new Float32Array(D); let p = 0;
    for (let i = 0; i < N; i++){ const b = buf[p], y = -out[i] + b; buf[p] = out[i] + b * .5; p = (p + 1) % D; out[i] = y; }
  }
  return out;
}
const wl = reverb(RL, 0), wr = reverb(RR, 23);
for (let i = 0; i < N; i++){ L[i] += wl[i] * .55; R[i] += wr[i] * .55; }

// ---------------- master: gentle glue, fade, soft clip, write WAV ----------------
let peak = 0;
for (let i = 0; i < N; i++){
  const t = i / SR, fade = Math.min(1, t / .05) * (t > DUR - 1.4 ? Math.max(0, (DUR - t) / 1.4) : 1);
  L[i] = Math.tanh(L[i] * fade * .9); R[i] = Math.tanh(R[i] * fade * .9);
  peak = Math.max(peak, Math.abs(L[i]), Math.abs(R[i]));
}
const g = .89 / peak, out = Buffer.alloc(44 + N * 4);
out.write('RIFF', 0); out.writeUInt32LE(36 + N * 4, 4); out.write('WAVEfmt ', 8);
out.writeUInt32LE(16, 16); out.writeUInt16LE(1, 20); out.writeUInt16LE(2, 22); out.writeUInt32LE(SR, 24);
out.writeUInt32LE(SR * 4, 28); out.writeUInt16LE(4, 32); out.writeUInt16LE(16, 34); out.write('data', 36); out.writeUInt32LE(N * 4, 40);
for (let i = 0; i < N; i++){
  out.writeInt16LE(Math.round(L[i] * g * 32767), 44 + i * 4);
  out.writeInt16LE(Math.round(R[i] * g * 32767), 46 + i * 4);
}
fs.writeFileSync(process.argv[3], out);
console.log('wrote', process.argv[3], 'peak', peak.toFixed(3));
