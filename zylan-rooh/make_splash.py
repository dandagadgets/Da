#!/usr/bin/env python3
"""
ZYLAN "Rooh" - "Summer Splash": a bright, beat-synced 10-second vertical ad
(1080x1920, 120 BPM). Deliberately the opposite of make_film.py's calm,
dark mood: vivid azure, sunburst, bold type slams and quick cuts.

  0-2 s   "FRESH" / "CLEAN" / "AQUATIC" slam in on the beat
  2-4 s   water burst; the bottle spins in over rotating sun rays, "ROOH" behind it
  4-6 s   four diagonal-wipe cuts on details from the photos, one scent note each
  6-8 s   hero bottle punching in on every beat, light sweep, droplets
  8-10 s  white flash, end card: ZYLAN / ROOH / Luxury in every drop

Reuses the bottle cut-out and text helpers from make_film.py.
Usage:  python3 make_splash.py      [--stills 1,3,5,7,9]
"""
import argparse
import math
import os
import subprocess
import wave

import cv2
import numpy as np
from PIL import Image
from scipy.signal import butter, fftconvolve, lfilter

from make_film import (H, W, HERE, add_light, blit, build_bottle, clamp01, ease, ease_out, flat_rgba,
                       font, gauss_dot, gold_rgba, smoothstep, text_mask)
from make_video import SR, bell, pluck, reverb_ir, whoosh, midi_hz, make_star

FPS = 30
DURATION = 10.0
BEAT = 0.5
AZURE_TOP, AZURE_MID, AZURE_LOW = np.float32([0, 70, 160]), np.float32([10, 160, 225]), np.float32([0, 55, 125])
WHITE = np.float32([255, 255, 255])
SHADOW = np.float32([0, 40, 95])

SLAMS = [(0.0, 'FRESH'), (0.5, 'CLEAN'), (1.0, 'AQUATIC')]
PANELS = [  # start, poster, crop box, caption
    (4.0, 'poster2.jpg', (0, 470, 430, 750), 'FRESH OCEAN BREEZE'),
    (4.5, 'poster2.jpg', (120, 560, 770, 982), 'CLEAN AQUATIC ACCORD'),
    (5.0, 'poster3.jpg', (150, 920, 520, 1160), 'SLIGHTLY SALTY'),
    (5.5, 'poster3.jpg', (860, 800, 1250, 1053), 'SOFT MUSKY DRY DOWN'),
]
BURST, HERO, END = 2.0, 6.0, 8.0


def bold_rgba(text, size, tracking=0.02, weight='Black', shadow=True):
    m = text_mask(text, font('Montserrat[wght].ttf', size, weight), tracking=tracking, pad=30)[0]
    rgba = flat_rgba(m, WHITE)
    if shadow:                                   # deep blue drop shadow for punch
        sh = np.roll(np.roll(cv2.GaussianBlur(m, (0, 0), 4), 10, 0), 8, 1) * 0.65
        rgba[..., :3] = rgba[..., :3] + SHADOW * sh[..., None] * (1 - m[..., None])
        rgba[..., 3] = np.maximum(rgba[..., 3], sh)
    return rgba


def build(rng):
    A = dict(bottle=build_bottle(), star=make_star())
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = yy / H
    k = smoothstep(v / 0.5)[..., None]
    bg = AZURE_TOP * (1 - k) + AZURE_MID * k
    k2 = smoothstep((v - 0.6) / 0.4)[..., None]
    A['bg'] = bg * (1 - k2) + AZURE_LOW * k2
    A['ang'] = np.arctan2(yy - 900, xx - W / 2)
    A['rad'] = np.sqrt((xx - W / 2) ** 2 + (yy - 900) ** 2)
    r = np.sqrt(((xx - W / 2) / (W * 0.7)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
    A['vignette'] = (1 - 0.35 * smoothstep((r - 0.6) / 0.6))[..., None]
    A['slams'] = [(t, bold_rgba(w, 165, 0.03)) for t, w in SLAMS]
    A['rooh_big'] = bold_rgba('ROOH', 265, 0.04, 'Black', shadow=False)
    A['rooh_big'][..., :3] *= 0.92
    A['panels'] = []
    for t0, poster, box, cap in PANELS:
        img = np.asarray(Image.open(os.path.join(HERE, poster)).convert('RGB')).astype(np.float32)
        x0, y0, x1, y1 = box
        sc = max((W + 120) / (x1 - x0), 1040 / (y1 - y0))      # cover the whole band
        crop = cv2.resize(img[y0:y1, x0:x1], (int((x1 - x0) * sc), int((y1 - y0) * sc)),
                          interpolation=cv2.INTER_CUBIC)
        A['panels'].append((t0, crop, bold_rgba(cap, 64, 0.06, 'ExtraBold')))
    A['logo'] = gold_rgba(text_mask('ZYLAN', font('Cinzel[wght].ttf', 150, 'Bold'), tracking=0.2)[0], glow=0.4)
    A['logo_small'] = gold_rgba(text_mask('ZYLAN', font('Cinzel[wght].ttf', 92, 'Bold'), tracking=0.2)[0], glow=0.3)
    A['end_rooh'] = bold_rgba('ROOH', 190, 0.12, 'Black')
    A['end_sub'] = bold_rgba('LUXURY IN EVERY DROP', 40, 0.3, 'SemiBold', shadow=False)
    A['end_fr'] = bold_rgba('LUXURY FRAGRANCES', 30, 0.4, 'Medium', shadow=False)
    # water burst droplets around the bottle at BURST
    A['drops'] = [dict(a=rng.uniform(0, 2 * np.pi), v=rng.uniform(500, 1500), r=rng.uniform(2.5, 9),
                       life=rng.uniform(0.5, 1.1)) for _ in range(140)]
    A['bokeh'] = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), r=rng.uniform(20, 70), g=rng.uniform(0.05, 0.16),
                       vx=rng.uniform(-20, 20), vy=rng.uniform(-30, -6), ph=rng.uniform(0, 6.3)) for _ in range(18)]
    A['dots'] = {}
    return A


def dot(A, r):
    k = int(round(r * 2))
    if k not in A['dots']:
        A['dots'][k] = gauss_dot(max(0.8, k / 4))
    return A['dots'][k]


def sunburst(A, t, strength):
    rays = 0.5 + 0.5 * np.cos(14 * A['ang'] + t * 0.9)
    rays = rays ** 6 * np.exp(-A['rad'] / 900)
    glow = np.exp(-(A['rad'] / 520) ** 2)
    return (rays * 0.55 + glow * 0.8)[..., None] * np.float32([200, 240, 255]) * strength


def draw_bottle(out, A, t, cx, base, hgt, spin=0.0, sweep=0.0):
    """Bottle with a fake Y-axis spin (width follows cos) and horizontal motion blur."""
    B = A['bottle']
    s = hgt / B['h']
    c = math.cos(spin)
    if abs(c) < 0.04:
        return
    src = B['rgba'] if c > 0 else B['rgba'][:, ::-1]
    hb, wb = src.shape[:2]
    sx = s * abs(c)
    patch = cv2.resize(src, (max(2, int(wb * sx)), int(hb * s)), interpolation=cv2.INTER_AREA)
    patch = patch.copy()
    patch[..., :3] *= 0.75 + 0.25 * abs(c)
    rgba_edge = cv2.resize(B['edge'], (patch.shape[1], patch.shape[0]))
    patch[..., :3] += rgba_edge[..., None] * np.float32([150, 230, 255]) * 0.8
    if sweep > 0:
        ph, pw = patch.shape[:2]
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        p = xx * 0.8 + yy * 0.6
        pos = -150 + (pw * 0.8 + ph * 0.6 + 300) * sweep
        band = np.exp(-0.5 * ((p - pos) / 30) ** 2) * math.sin(math.pi * sweep)
        patch[..., :3] += (band * patch[..., 3])[..., None] * WHITE * 0.75
    speed = abs(math.sin(spin))
    blur = int(speed * 18 * (abs(c) < 0.97))
    if blur > 2:
        patch = cv2.blur(patch, (blur, 1))
    cy = base - (B['base_y'] - hb / 2) * s
    blit(out, patch, cx, cy, 1.0)


def spin_angle(t):
    """Two fast turns that slow to a stop facing front at t = 3.2."""
    u = clamp01((t - BURST) / 1.2)
    return float(4 * math.pi * (1 - (1 - u) ** 3))


def render_frame(fi, A):
    t = fi / FPS
    beat_phase = (t % BEAT) / BEAT
    pulse = math.exp(-beat_phase * 6)                      # 1 right on the beat, decays
    out = A['bg'].copy()
    # sun rays behind everything (strong once the bottle arrives)
    out += sunburst(A, t, 0.35 + 0.35 * float(smoothstep((t - BURST) / 0.4)) + 0.15 * pulse)
    for b in A['bokeh']:
        x = (b['x'] + b['vx'] * t) % W
        y = (b['y'] + b['vy'] * t) % H
        add_light(out, dot(A, b['r'] / 1.6), x, y, WHITE * b['g'] * (0.7 + 0.3 * math.sin(t * 2 + b['ph'])))
    punch = 1.0

    if t < BURST:                                          # --- type slams
        for i, (t0, spr) in enumerate(A['slams']):
            t1 = A['slams'][i + 1][0] if i + 1 < len(A['slams']) else BURST - 0.15
            if t0 <= t < t1 + 0.12:
                u = (t - t0) / 0.14
                sc = 1.0 + 0.6 * (1 - ease_out(u)) if u < 1 else 1.0 + 0.04 * (t - t0)
                al = float(clamp01(u * 1.5)) * (1 - float(clamp01((t - t1) / 0.12)))
                blit(out, spr, W / 2, 900, al, sc)
        if t > 1.55:                                       # bottle flash-preview before the burst
            k = float(smoothstep((t - 1.55) / 0.45))
            out += sunburst(A, t * 3, 0.6 * k)
    elif t < 4.0:                                          # --- spin-in
        u = (t - BURST)
        blit(out, A['rooh_big'], W / 2, 640, float(smoothstep((u - 0.9) / 0.4)) * 0.9,
             1.15 - 0.15 * ease_out((u - 0.9) / 0.6))
        hgt = 520 + 380 * ease_out(u / 1.2)
        bob = 10 * math.sin(t * 3)
        draw_bottle(out, A, t, W / 2, 1520 + bob, hgt, spin_angle(t), sweep=float(clamp01((t - 3.3) / 0.6)))
        if u < 1.1:                                        # water burst
            for d in A['drops']:
                if u < d['life']:
                    x = W / 2 + math.cos(d['a']) * d['v'] * u
                    y = 1150 + math.sin(d['a']) * d['v'] * u * 0.8 + 900 * u * u
                    add_light(out, dot(A, d['r']), x, y, WHITE * (1 - u / d['life']) * 1.4)
    elif t < HERO:                                         # --- detail cuts
        idx = min(3, int((t - 4.0) / 0.5))
        for j in range(max(0, idx - 1), idx + 1):
            t0, crop, cap = A['panels'][j]
            if t < t0:
                continue
            u = (t - t0) / 0.5
            ch, cw = crop.shape[:2]
            band_h = 980
            zoom = 1.0 + 0.08 * u
            view = cv2.resize(crop, (int(cw * zoom), int(ch * zoom)))
            vh, vw = view.shape[:2]
            y0 = 940 - band_h // 2
            sub = np.zeros((band_h, W, 3), np.float32)
            src = view[max(0, (vh - band_h) // 2):, max(0, (vw - W) // 2):][:band_h, :W]
            sub[:src.shape[0], :src.shape[1]] = src
            # diagonal wipe in from the right
            wipe = ease_out(u / 0.35)
            xx = np.arange(W, dtype=np.float32)[None, :]
            yy = np.arange(band_h, dtype=np.float32)[:, None]
            edge = (W + 400) * (1 - wipe) - 200 - 0.35 * (yy - band_h / 2)
            m = clamp01((xx - edge) / 3.0)[..., None]
            reg = out[y0:y0 + band_h]
            reg[:] = reg * (1 - m) + sub * m
            line = np.exp(-0.5 * ((xx - edge) / 3.0) ** 2)[..., None]
            reg += line * WHITE * 0.9 * (wipe < 1)
            if j == idx:                               # only the current caption
                cu = ease_out((u - 0.12) / 0.3)
                blit(out, cap, W / 2, y0 + band_h + 120, cu, 1.0, dx=(1 - cu) * -80)
        blit(out, A['logo_small'], W / 2, 250, 1.0)
    elif t < END:                                          # --- hero, punching on the beat
        punch = 1.0 + 0.035 * pulse
        blit(out, A['rooh_big'], W / 2, 640, 0.9, 1.0 + 0.02 * pulse)
        draw_bottle(out, A, t, W / 2, 1560, 920 * punch, 0.0, sweep=float(clamp01((t - 6.9) / 0.7)))
        blit(out, A['logo_small'], W / 2, 230, 1.0)
        for ts, x, y in ((6.5, 640, 720), (7.0, 410, 1100), (7.5, 600, 1350)):
            u = (t - ts) / 0.45
            if 0 < u < 1:
                st = A['star']
                n = int(70 * math.sin(math.pi * u) + 4) | 1
                R = cv2.getRotationMatrix2D((st.shape[0] / 2,) * 2, 45 * u, n / st.shape[0])
                R[:, 2] += n / 2 - st.shape[0] / 2
                add_light(out, cv2.warpAffine(st, R, (n, n)), x, y, WHITE * math.sin(math.pi * u))
    else:                                                  # --- end card
        u = t - END
        draw_bottle(out, A, t, W / 2, 1450, 820 + 20 * ease(u / 2), 0.0, sweep=float(clamp01((u - 0.9) / 0.7)))
        lu = ease_out(u / 0.5)
        blit(out, A['logo'], W / 2, 210, lu, 1.2 - 0.2 * lu)
        blit(out, A['end_fr'], W / 2, 315, ease((u - 0.25) / 0.4))
        ru = ease_out((u - 0.15) / 0.45)
        blit(out, A['end_rooh'], W / 2, 500, ru, 1.25 - 0.25 * ru)
        blit(out, A['end_sub'], W / 2, 1640, ease((u - 0.5) / 0.4))

    # white flashes on the big moments
    for tf, dur in ((BURST, 0.3), (END, 0.35), (HERO, 0.2)):
        if tf <= t < tf + dur:
            out = out + (WHITE - out) * (1 - (t - tf) / dur) * 0.85
    if punch != 1.0:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, punch)
        out = cv2.warpAffine(out, M, (W, H), borderMode=cv2.BORDER_REFLECT)
    small = cv2.resize(out, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    out += cv2.resize(cv2.GaussianBlur(np.maximum(small - 200, 0), (0, 0), 5), (W, H)) * 0.45
    out *= A['vignette']
    out *= float(smoothstep(t / 0.15))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def make_audio(path):
    n = int((DURATION + 0.8) * SR)
    dry = np.zeros((n, 2))
    rng = np.random.default_rng(21)

    def put(sig, t, gain, pan=0.0):
        i = int(t * SR)
        if i >= n:
            return
        sig = sig[:n - i]
        dry[i:i + len(sig), 0] += sig * gain * math.cos((pan + 1) * math.pi / 4)
        dry[i:i + len(sig), 1] += sig * gain * math.sin((pan + 1) * math.pi / 4)

    def kick():
        tt = np.arange(int(0.35 * SR)) / SR
        f = 45 + 110 * np.exp(-tt / 0.03)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / 0.14)

    def noise_hit(dur, lo, hi, decay, seed):
        x = np.random.default_rng(seed).normal(0, 1, int(dur * SR))
        x = lfilter(*butter(2, [lo / (SR / 2), hi / (SR / 2)], 'band'), x)
        return x * np.exp(-np.arange(len(x)) / SR / decay) / (np.abs(x).max() + 1e-9) * 3

    def saw_stab(freqs, dur):
        tt = np.arange(int(dur * SR)) / SR
        y = sum(((tt * g) % 1.0) * 2 - 1 for f in freqs for g in (f * 0.997, f * 1.003))
        y = lfilter(*butter(2, 2600 / (SR / 2)), y)
        return y * np.exp(-tt / 0.18) * (1 - np.exp(-tt / 0.004)) / (len(freqs) * 2)

    chords = [[62, 66, 69], [57, 61, 64], [59, 62, 66], [55, 59, 62]]   # D A Bm G
    roots = [38, 33, 35, 31]
    k_drum = kick()
    for b in range(int(DURATION / BEAT)):
        tb = b * BEAT
        bar = (b // 4) % 4
        put(k_drum, tb, 0.55)
        if b % 2 == 1:
            put(noise_hit(0.25, 900, 6000, 0.06, 100 + b), tb, 0.25, pan=0.1)       # clap
        for off in (0.0, 0.25):
            put(noise_hit(0.06, 7000, 15000, 0.015, 300 + b * 2 + int(off * 4)), tb + off, 0.10, pan=0.4)
        put(pluck(midi_hz(roots[bar] + 12), 0.45, seed=b), tb + 0.25, 0.30)            # off-beat bass
        if b % 2 == 0:
            put(saw_stab([midi_hz(n_) for n_ in chords[bar]], 0.4), tb + 0.25, 0.22, pan=-0.2)
        put(pluck(midi_hz(chords[bar][b % 3] + 12), 1.0, seed=50 + b), tb + 0.125, 0.08, pan=0.3)
    # riser + impacts on the burst and the end card, whooshes on the cuts
    for t_hit in (BURST, END):
        m = int(1.0 * SR)
        tt = np.arange(m) / SR
        r = lfilter(*butter(2, [500 / (SR / 2), 9000 / (SR / 2)], 'band'), rng.normal(0, 1, m)) * (tt / tt[-1]) ** 3
        put(r / np.abs(r).max(), t_hit - 1.0, 0.18)
        put(noise_hit(1.2, 200, 9000, 0.25, int(t_hit * 7)), t_hit, 0.22)
        tt = np.arange(int(1.2 * SR)) / SR
        put(np.sin(2 * np.pi * np.cumsum(40 + 80 * np.exp(-tt / 0.05)) / SR) * np.exp(-tt / 0.4), t_hit, 0.5)
    for p in PANELS:
        put(whoosh(0.35, seed=int(p[0] * 10)), p[0] - 0.12, 0.10, pan=0.5)
    for i, note in enumerate([86, 90, 93, 98]):
        put(bell(midi_hz(note), 2.2), END + 0.05 + i * 0.06, 0.06, pan=-0.4 + 0.27 * i)
    wet = np.stack([fftconvolve(dry[:, c], reverb_ir(seconds=1.4, seed=31 + c)[:, c])[:n] for c in range(2)], 1)
    mix = dry * 0.85 + wet * 0.3
    mix = lfilter(*butter(2, 30 / (SR / 2), 'highpass'), mix, axis=0)
    tt = np.arange(n) / SR
    mix *= (1 - smoothstep((tt - (DURATION - 0.5)) / 0.5))[:, None]
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.6) * 0.95
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((mix * 32767).astype(np.int16).tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'zylan_rooh_splash.mp4'))
    ap.add_argument('--stills', default=None)
    args = ap.parse_args()
    A = build(np.random.default_rng(8))
    if args.stills:
        for ts in args.stills.split(','):
            Image.fromarray(render_frame(int(round(float(ts) * FPS)), A)).save(f'splash_still_{float(ts):05.2f}.png')
        return
    audio = os.path.splitext(args.out)[0] + '_audio.wav'
    make_audio(audio)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
           '-r', str(FPS), '-i', '-', '-i', audio, '-c:a', 'aac', '-b:a', '192k', '-shortest',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-x264-params', 'aq-mode=3',
           '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart', args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(int(round(DURATION * FPS))):
        enc.stdin.write(render_frame(fi, A).tobytes())
    enc.stdin.close()
    enc.wait()
    os.remove(audio)
    print('done:', args.out)


if __name__ == '__main__':
    main()
