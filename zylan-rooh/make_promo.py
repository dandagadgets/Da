#!/usr/bin/env python3
"""
10-second promo for ZYLAN "Rooh", made from the three product posters.

Four shots joined by flare-lit crossfades, each with a slow camera move:
  1. the dark "Introducing ROOH" poster, panning from the logo down to the bottle
  2. the ocean poster, pulling back from the bottle to reveal the scent notes
  3. a close-up of the bottle and lime on the wet rocks
  4. end card on the "ZYLAN | ROOH - Luxury in every drop" tagline
Gold lettering catches sweeping light, gold dust drifts on the dark shots and
the sea glitters on the ocean shots. The posters' own pixels are used, so all
text stays exactly as designed. A short synthesized score (pad, harp, chimes,
whooshes on the cuts) is added.

Usage:
    python3 make_promo.py                 # zylan_rooh_promo.mp4 (1080x1080)
    python3 make_promo.py --size 1440     # larger master
    python3 make_promo.py --stills 1,4,7,9.5
Requires numpy, scipy, Pillow, opencv-python-headless and ffmpeg on PATH.
The music helpers come from ../walima-invitation/make_video.py.
"""
import argparse
import math
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from PIL import Image
from scipy.signal import butter, fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'walima-invitation'))
from make_video import SR, bell, make_star, midi_hz, pad_voice, pluck, reverb_ir, whoosh  # noqa: E402

FPS = 30
DURATION = 10.0
XFADE = 0.4

# shot: poster, start, end, (cx, cy, window) at the start and at the end of the
# shot in poster pixels (window = side of the square that fills the frame),
# dust colour, whether the sea glitters
SHOTS = [
    dict(img=1, t0=0.0, t1=3.0, a=(572, 572, 1145), b=(572, 742, 1100),
         dust=(255, 210, 130), sea=False),
    dict(img=2, t0=2.6, t1=5.6, a=(585, 700, 900), b=(627, 627, 1254),
         dust=(220, 240, 255), sea=True),
    dict(img=3, t0=5.2, t1=7.9, a=(470, 860, 780), b=(480, 880, 690),
         dust=(220, 240, 255), sea=True),
    dict(img=1, t0=7.5, t1=10.0, a=(572, 899, 950), b=(572, 824, 1100),
         dust=(255, 210, 130), sea=False),
]
SWEEPS = [(0.5, 1.6, 1), (3.3, 1.5, 2), (8.3, 1.4, 4)]   # start, duration, shot number


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_in_out(x):
    x = clamp01(x)
    return 0.5 - 0.5 * np.cos(np.pi * x)


class Poster:
    def __init__(self, path, rng):
        self.img8 = np.asarray(Image.open(path).convert('RGB'))
        self.img = self.img8.astype(np.float32)
        hsv = cv2.cvtColor(self.img8, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        gold = ((h >= 10) & (h <= 32) & (s > 70) & (v > 140)).astype(np.float32)
        self.gold = cv2.GaussianBlur(gold * clamp01((v - 140) / 90), (0, 0), 0.8)
        H, W = h.shape
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        self.proj = xx * math.cos(math.radians(30)) + yy * math.sin(math.radians(30))
        hi = (v > 238) & (s < 60)                       # specular glints on water and glass
        ys, xs = np.nonzero(hi)
        pick = rng.choice(len(xs), size=min(3000, len(xs)), replace=False) if len(xs) else []
        self.spark = np.stack([xs[pick], ys[pick]], 1).astype(np.float32) if len(xs) else np.zeros((0, 2))


def camera(shot, t):
    u = ease_in_out((t - shot['t0']) / (shot['t1'] - shot['t0']))
    a, b = np.float32(shot['a']), np.float32(shot['b'])
    return a + (b - a) * u


def render_shot(P, shot, t, size, scene):
    cx, cy, win = camera(shot, t)
    s = size / win
    src = P.img
    for t0, dur, k in SWEEPS:                              # light running over the gold
        u = (t - t0) / dur
        if SHOTS[k - 1] is shot and 0 < u < 1:
            lo, hi = float(P.proj.min()), float(P.proj.max())
            pos = lo - 100 + (hi - lo + 200) * float(ease_in_out(u))
            band = np.exp(-0.5 * ((P.proj - pos) / 45.0) ** 2) * math.sin(math.pi * u) ** 0.5
            amt = (P.gold * band * 0.9)[..., None]
            src = src + amt * (np.float32([255, 246, 222]) - src)
    M = np.float32([[s, 0, size / 2 - s * cx], [0, s, size / 2 - s * cy]])
    out = cv2.warpAffine(src, M, (size, size), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    if shot['sea']:                                       # glittering highlights
        for ev in scene['sparkles'][id(shot)]:
            u = (t - ev[0]) / ev[1]
            if 0 < u < 1:
                x, y = M[0, 0] * ev[2] + M[0, 2], M[1, 1] * ev[3] + M[1, 2]
                env = math.sin(math.pi * u) ** 1.5
                star(out, scene['star'], x, y, ev[4] * s * (0.4 + 0.6 * env), env * ev[5], ev[6])
    return out


def star(out, sprite, x, y, size, gain, angle, color=(255, 250, 235)):
    S = sprite.shape[0]
    ps = int(size * 2) | 1
    if ps < 5:
        return
    R = cv2.getRotationMatrix2D((S / 2, S / 2), angle, ps / S)
    R[:, 2] += ps / 2 - S / 2
    add_light(out, cv2.warpAffine(sprite, R, (ps, ps)), x, y, np.float32(color) * gain)


def add_light(out, sp, x, y, color):
    h, w = sp.shape
    x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    H, W = out.shape[:2]
    a0, b0 = max(0, x0), max(0, y0)
    a1, b1 = min(W, x0 + w), min(H, y0 + h)
    if a1 <= a0 or b1 <= b0:
        return
    reg = out[b0:b1, a0:a1]
    light = sp[b0 - y0:b1 - y0, a0 - x0:a1 - x0, None] * color
    reg += light * (1 - np.clip(reg, 0, 255) / 255)


def render_frame(fi, posters, scene, size):
    t = fi / FPS
    out = None
    weight_sum = 0.0
    layers = []
    for shot in SHOTS:
        if shot['t0'] - 1e-6 <= t <= shot['t1'] + 1e-6:
            w_in = 1.0 if shot['t0'] == 0 else float(smoothstep((t - shot['t0']) / XFADE))
            w_out = 1.0 if shot['t1'] >= DURATION else float(smoothstep((shot['t1'] - t) / XFADE))
            layers.append((shot, w_in * w_out))
    for shot, w in layers:
        img = render_shot(posters[shot['img']], shot, t, size, scene)
        out = img * w if out is None else out + img * w
        weight_sum += w
    out /= max(weight_sum, 1e-6)
    # flare at each cut
    for shot in SHOTS[1:]:
        u = (t - shot['t0']) / XFADE
        if 0 < u < 1:
            g = math.sin(math.pi * u) ** 2
            x = size * (0.2 + 0.6 * u)
            add_light(out, scene['flare'], x, size * 0.38, np.float32([255, 236, 200]) * 0.6 * g)
    # drifting dust in the colour of the current shot
    dom = max(layers, key=lambda l: l[1])[0]
    for p in scene['dust']:
        y = (p['y0'] - p['v'] * t) % (size + 40) - 20
        x = (p['x0'] + p['a'] * math.sin(p['w'] * t + p['ph'])) % size
        tw = (0.5 + 0.5 * math.sin(p['wt'] * t + p['ph'] * 3)) ** 2
        add_light(out, p['sprite'], x, y, np.float32(dom['dust']) * p['g'] * (0.25 + 0.75 * tw))
    out *= scene['vignette']
    fade = float(smoothstep(t / 0.35)) * (1 - float(smoothstep((t - (DURATION - 0.45)) / 0.45)) * 0.85)
    return np.clip(out * fade + 0.5, 0, 255).astype(np.uint8)


def make_scene(posters, size, rng):
    sc = dict(star=make_star())
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.sqrt(((xx - size / 2) / (size / 2)) ** 2 + ((yy - size / 2) / (size / 2)) ** 2)
    sc['vignette'] = (1 - 0.28 * smoothstep((r - 0.6) / 0.8))[..., None]
    fr = np.exp(-r ** 2 / 0.08) + 0.35 * np.exp(-r ** 2 / 0.6)
    sc['flare'] = cv2.resize(fr.astype(np.float32), (size, size))
    unit = size / 1080
    dust = []
    for _ in range(70):
        sig = rng.choice([1.0, 1.5, 2.2, 3.0]) * unit
        n = int(sig * 6) | 1
        ax = np.arange(n, dtype=np.float32) - n // 2
        g = np.exp(-0.5 * (ax / sig) ** 2)
        dust.append(dict(x0=rng.uniform(0, size), y0=rng.uniform(0, size), v=rng.uniform(12, 40) * unit,
                         a=rng.uniform(8, 30) * unit, w=rng.uniform(0.3, 0.9), wt=rng.uniform(1, 4),
                         ph=rng.uniform(0, 6.3), sprite=np.outer(g, g) * 2.0, g=rng.uniform(0.4, 0.9)))
    sc['dust'] = dust
    sparkles = {}
    for shot in SHOTS:
        if not shot['sea']:
            continue
        P = posters[shot['img']]
        ev = []
        t = shot['t0']
        while t < shot['t1']:
            t += rng.exponential(1 / 14.0)
            cx, cy, win = camera(shot, t)
            pts = P.spark
            vis = np.nonzero((np.abs(pts[:, 0] - cx) < win / 2 - 10) & (np.abs(pts[:, 1] - cy) < win / 2 - 10))[0]
            if len(vis):
                p = pts[rng.choice(vis)]
                ev.append((t, rng.uniform(0.35, 0.8), p[0], p[1], rng.uniform(6, 16),
                           rng.uniform(0.5, 1.0), rng.uniform(0, 90)))
        sparkles[id(shot)] = ev
    sc['sparkles'] = sparkles
    return sc


def make_audio(path):
    n = int((DURATION + 0.5) * SR)
    dry = np.zeros((n, 2))
    rng = np.random.default_rng(9)

    def put(sig, t, gain, pan=0.0):
        i = int(t * SR)
        if i >= n:
            return
        sig = sig[:n - i]
        dry[i:i + len(sig), 0] += sig * gain * math.cos((pan + 1) * math.pi / 4)
        dry[i:i + len(sig), 1] += sig * gain * math.sin((pan + 1) * math.pi / 4)

    # D minor -> Bb -> F -> C, one chord per shot: cool, airy, a little mysterious
    chords = [(0.0, [38, 50, 57, 62, 65, 69]), (2.6, [34, 46, 53, 58, 62, 65]),
              (5.2, [41, 53, 60, 65, 69, 72]), (7.5, [36, 48, 55, 60, 64, 67])]
    for i, (t0, ch) in enumerate(chords):
        nxt = chords[i + 1][0] if i + 1 < len(chords) else DURATION + 0.5
        for note in (ch[0], ch[2], ch[4]):
            put(pad_voice(midi_hz(note), nxt - t0 + 1.2), t0, 0.09, pan=rng.uniform(-0.4, 0.4))
        for k, idx in enumerate([1, 3, 4, 5, 4, 3, 5, 4, 3]):
            tk = t0 + 0.12 + k * 0.29
            if tk < nxt:
                put(pluck(midi_hz(ch[idx] + 12), 2.0, seed=40 * i + k), tk, 0.10, pan=-0.3 + 0.07 * k)
    # sub pulse for weight
    t = np.arange(int(1.2 * SR)) / SR
    boom = np.sin(2 * np.pi * 46 * t) * np.exp(-t / 0.35) * (1 - np.exp(-t / 0.01))
    for tm in (0.0, 2.6, 5.2, 7.5):
        put(boom, tm, 0.22)
    for i, shot in enumerate(SHOTS[1:]):
        put(whoosh(0.9, seed=20 + i), shot['t0'] - 0.45, 0.10, pan=-0.5 + 0.5 * i)
    for tm in (0.5, 3.3, 8.3):                            # chimes on the gold sweeps
        for k, note in enumerate([86, 89, 93, 98]):
            put(bell(midi_hz(note), 1.6), tm + k * 0.06, 0.04, pan=-0.5 + 0.33 * k)
    put(bell(midi_hz(74), 3.0), 7.55, 0.10)
    put(bell(midi_hz(81), 3.0), 7.85, 0.07)
    wet = np.stack([fftconvolve(dry[:, c], reverb_ir(seconds=2.4, seed=7 + c)[:, c])[:n] for c in range(2)], 1)
    mix = dry * 0.7 + wet * 0.6
    b, a = butter(2, 30 / (SR / 2), 'highpass')
    mix = lfilter(b, a, mix, axis=0)
    tt = np.arange(n) / SR
    mix *= smoothstep(tt / 0.15)[:, None] * (1 - smoothstep((tt - (DURATION - 0.8)) / 0.8))[:, None]
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.1) * 0.95
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((mix * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--size', type=int, default=1080)
    ap.add_argument('--out', default=os.path.join(HERE, 'zylan_rooh_promo.mp4'))
    ap.add_argument('--stills', default=None)
    ap.add_argument('--no-audio', action='store_true')
    args = ap.parse_args()
    rng = np.random.default_rng(3)
    posters = {i: Poster(os.path.join(HERE, f'poster{i}.jpg'), rng) for i in (1, 2, 3)}
    scene = make_scene(posters, args.size, rng)
    if args.stills:
        for ts in args.stills.split(','):
            fr = render_frame(int(round(float(ts) * FPS)), posters, scene, args.size)
            Image.fromarray(fr).save(f'promo_still_{float(ts):05.2f}.png')
        return
    audio = None
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{args.size}x{args.size}', '-r', str(FPS), '-i', '-']
    if not args.no_audio:
        audio = os.path.splitext(args.out)[0] + '_audio.wav'
        make_audio(audio)
        cmd += ['-i', audio, '-c:a', 'aac', '-b:a', '192k', '-shortest']
    cmd += ['-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-x264-params', 'aq-mode=3',
            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart', args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(int(round(DURATION * FPS))):
        enc.stdin.write(render_frame(fi, posters, scene, args.size).tobytes())
    enc.stdin.close()
    enc.wait()
    if audio:
        os.remove(audio)
    print('done:', args.out)


if __name__ == '__main__':
    main()
