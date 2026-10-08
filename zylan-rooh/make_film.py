#!/usr/bin/env python3
"""
ZYLAN "Rooh" - 10-second vertical product film (1080x1920).

A different take from make_promo.py: instead of panning over the posters,
the bottle is cut out of poster1.jpg and staged in a new animated scene.

  0.0-2.0  dark water, light rays and caustics; "ZYLAN" forms letter by letter
  2.0-2.8  the logo rises; the bottle drops in and lands with a splash
  2.8-4.2  hero shot: rim light, living liquid, a light sweep over the glass
  4.2-7.6  the bottle glides aside and the four scent notes draw themselves in
  7.6-10   back to centre: "ROOH", "More Than Just a Fragrance", the sign-off

Fonts (SIL Open Font License, in fonts/): Cinzel, Great Vibes, Montserrat.

Usage:
    python3 make_film.py                     # zylan_rooh_film.mp4
    python3 make_film.py --stills 1,3,5.5,9
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
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'walima-invitation'))
from make_video import SR, bell, make_star, midi_hz, pad_voice, pluck, reverb_ir, whoosh  # noqa: E402

W, H = 1080, 1920
FPS = 30
DURATION = 10.0
FLOOR = 1600                         # y of the mirror-water surface the bottle stands on
LAND = 2.75                          # the bottle touches down
FONT = os.path.join(HERE, 'fonts')

GOLD_LIGHT = np.float32([255, 226, 150])
GOLD_DARK = np.float32([165, 112, 40])
CHAMPAGNE = np.float32([238, 228, 206])

NOTES = [('FRESH OCEAN BREEZE', 'Clean & refreshing like the sea air', 'waves'),
         ('CLEAN AQUATIC ACCORD', 'Crisp and cool', 'drop'),
         ('SLIGHTLY SALTY', 'Adds natural depth', 'crystal'),
         ('SOFT MUSKY DRY DOWN', 'Smooth and long lasting', 'leaf')]
NOTES_T0, NOTES_STEP, NOTES_OUT = 4.75, 0.55, 7.25


# ---------------------------------------------------------------------------
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease(x):                         # ease in-out
    return float(smoothstep(x))


def ease_out(x):
    x = float(clamp01(x))
    return 1 - (1 - x) ** 3


def font(name, size, var=None):
    f = ImageFont.truetype(os.path.join(FONT, name), size)
    if var:
        f.set_variation_by_name(var)
    return f


def text_mask(text, f, tracking=0.0, pad=12):
    """Alpha mask of `text` with extra letter spacing (fraction of size)."""
    widths = [f.getlength(c) for c in text]
    extra = tracking * f.size
    total = sum(widths) + extra * (len(text) - 1)
    asc, desc = f.getmetrics()
    im = Image.new('L', (int(total + 2 * pad), asc + desc + 2 * pad), 0)
    d = ImageDraw.Draw(im)
    x = pad
    boxes = []
    for c, w in zip(text, widths):
        d.text((x, pad), c, font=f, fill=255)
        boxes.append((x, x + w))
        x += w + extra
    return np.asarray(im).astype(np.float32) / 255.0, boxes


def gold_rgba(mask, emboss=4.0, glow=0.0):
    """Polished gold lettering as premultiplied RGBA."""
    h, w = mask.shape
    ht = cv2.GaussianBlur(mask, (0, 0), 1.2)
    gx = cv2.Sobel(ht, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ht, cv2.CV_32F, 0, 1, ksize=3)
    n = np.dstack([-gx * emboss, -gy * emboss, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    L = np.float32([-0.4, -0.7, 0.6]) / np.float32(np.linalg.norm([-0.4, -0.7, 0.6]))
    diff = clamp01(n @ L)
    spec = clamp01(n @ (L + np.float32([0, 0, 1])) / np.float32(np.linalg.norm(L + np.float32([0, 0, 1])))) ** 20
    v = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    base = GOLD_LIGHT * (1 - v) + GOLD_DARK * v
    col = base * (0.55 + 0.6 * diff)[..., None] + 255 * 0.5 * spec[..., None]
    rgba = np.dstack([np.clip(col, 0, 255) * mask[..., None], mask])
    if glow:
        g = cv2.GaussianBlur(mask, (0, 0), 6) * glow
        rgba[..., :3] += (np.float32([255, 190, 90]) * g[..., None]) * (1 - mask[..., None])
        rgba[..., 3] = np.maximum(rgba[..., 3], g * 0.6)
    return rgba.astype(np.float32)


def flat_rgba(mask, color, glow=0.0, glow_color=(120, 210, 255)):
    rgba = np.dstack([np.float32(color) * mask[..., None], mask]).astype(np.float32)
    if glow:
        g = cv2.GaussianBlur(mask, (0, 0), 5) * glow
        rgba[..., :3] += np.float32(glow_color) * g[..., None] * (1 - mask[..., None])
        rgba[..., 3] = np.maximum(rgba[..., 3], g * 0.5)
    return rgba


def blit(out, rgba, cx, cy, alpha=1.0, scale=1.0, dx=0.0, dy=0.0):
    """Composite a premultiplied RGBA sprite centred at (cx, cy)."""
    if alpha <= 0.003:
        return
    h, w = rgba.shape[:2]
    if abs(scale - 1) > 1e-3:
        M = np.float32([[scale, 0, cx + dx - scale * w / 2], [0, scale, cy + dy - scale * h / 2]])
        x0 = int(max(0, math.floor(M[0, 2])))
        y0 = int(max(0, math.floor(M[1, 2])))
        x1 = int(min(W, math.ceil(M[0, 2] + scale * w)))
        y1 = int(min(H, math.ceil(M[1, 2] + scale * h)))
        if x1 <= x0 or y1 <= y0:
            return
        M[0, 2] -= x0
        M[1, 2] -= y0
        patch = cv2.warpAffine(rgba, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    else:
        x0, y0 = int(round(cx + dx - w / 2)), int(round(cy + dy - h / 2))
        a0, b0 = max(0, x0), max(0, y0)
        a1, b1 = min(W, x0 + w), min(H, y0 + h)
        if a1 <= a0 or b1 <= b0:
            return
        patch = rgba[b0 - y0:b1 - y0, a0 - x0:a1 - x0]
        x0, y0, x1, y1 = a0, b0, a1, b1
    reg = out[y0:y1, x0:x1]
    reg *= 1 - patch[..., 3:4] * alpha
    reg += patch[..., :3] * alpha


def add_light(out, sp, x, y, color):
    h, w = sp.shape
    x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    a0, b0 = max(0, x0), max(0, y0)
    a1, b1 = min(W, x0 + w), min(H, y0 + h)
    if a1 <= a0 or b1 <= b0:
        return
    reg = out[b0:b1, a0:a1]
    reg += sp[b0 - y0:b1 - y0, a0 - x0:a1 - x0, None] * np.float32(color) * (1 - np.clip(reg, 0, 255) / 255)


def gauss_dot(sigma):
    n = int(sigma * 6) | 1
    ax = np.arange(n, dtype=np.float32) - n // 2
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    return np.outer(g, g)


def caustics(w, h, t, scale=6.0):
    """Animated water caustics (classic iterated-sine pattern), values ~0..1."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px = xx / w * scale * (w / h) * 1.6 - 250
    py = yy / h * scale - 250
    ix, iy = px.copy(), py.copy()
    c = np.ones_like(px)
    inten = 0.005
    for n in range(5):
        tt = t * 0.55 * (1 - 3.5 / (n + 1))
        ix, iy = (px + np.cos(tt - ix) + np.sin(tt + iy), py + np.sin(tt - iy) + np.cos(tt + ix))
        c += 1.0 / np.sqrt((px / (np.sin(ix + tt) / inten)) ** 2 + (py / (np.cos(iy + tt) / inten)) ** 2)
    c /= 5
    c = 1.17 - c ** 1.4
    return clamp01(np.abs(c) ** 8 * 0.6)


# ---------------------------------------------------------------------------
# assets
# ---------------------------------------------------------------------------
def build_bottle():
    """Cut the bottle out of poster1 (rounded glass body + cap)."""
    img = np.asarray(Image.open(os.path.join(HERE, 'poster1.jpg')).convert('RGB')).astype(np.float32)
    X0, Y0, X1, Y1 = 362, 278, 784, 1056
    crop = img[Y0:Y1, X0:X1]
    S = 4
    m = Image.new('L', ((X1 - X0) * S, (Y1 - Y0) * S), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([(373 - X0) * S, (535 - Y0) * S, (772 - X0) * S, (1046 - Y0) * S], 22 * S, fill=255)
    d.rounded_rectangle([(488 - X0) * S, (288 - Y0) * S, (673 - X0) * S, (545 - Y0) * S], 6 * S, fill=255)
    mask = np.asarray(m.resize((X1 - X0, Y1 - Y0), Image.LANCZOS)).astype(np.float32) / 255
    hsv = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    liquid = ((hsv[..., 0] > 82) & (hsv[..., 0] < 112) & (hsv[..., 1] > 90) & (hsv[..., 2] > 50)).astype(np.float32)
    liquid = cv2.GaussianBlur(liquid * mask, (0, 0), 2)
    edge = clamp01(mask - cv2.erode(mask, np.ones((7, 7), np.uint8)))
    crop = crop * np.float32([0.96, 1.0, 1.06])            # nudge towards the cool scene
    rgba = np.dstack([crop * mask[..., None], mask]).astype(np.float32)
    base_y = 1046 - Y0                                     # glass bottom inside the crop
    glow = cv2.GaussianBlur(np.pad(mask, 160), (0, 0), 42)     # padded so the halo is not clipped
    return dict(rgba=rgba, liquid=liquid, edge=cv2.GaussianBlur(edge, (0, 0), 1.5),
                glow=glow / glow.max(), base_y=base_y, h=1046 - 288, top_y=288 - Y0)


def build_text():
    T = {}
    f = font('Cinzel[wght].ttf', 150, 'Regular')
    m, boxes = text_mask('ZYLAN', f, tracking=0.22)
    T['logo'] = gold_rgba(m, glow=0.5)
    T['logo_boxes'] = boxes
    mm = np.zeros_like(m)
    T['logo_letters'] = []
    for x0, x1 in boxes:
        lm = mm.copy()
        a, b = int(x0) - 4, int(x1) + 4
        lm[:, a:b] = m[:, a:b]
        T['logo_letters'].append((gold_rgba(lm, glow=0.5), (a + b) / 2 - m.shape[1] / 2))
    sub, _ = text_mask('LUXURY  FRAGRANCES', font('Montserrat[wght].ttf', 30, 'Medium'), tracking=0.38)
    T['logo_sub'] = flat_rgba(sub, CHAMPAGNE)
    T['rooh'] = gold_rgba(text_mask('ROOH', font('Cinzel[wght].ttf', 225, 'Bold'), tracking=0.08)[0], emboss=5, glow=0.7)
    T['script'] = flat_rgba(text_mask('More Than Just a Fragrance', font('GreatVibes-Regular.ttf', 82))[0],
                            CHAMPAGNE, glow=0.6, glow_color=(255, 210, 140))
    T['signoff'] = flat_rgba(text_mask('LUXURY IN EVERY DROP', font('Montserrat[wght].ttf', 30, 'Medium'),
                                       tracking=0.4)[0], CHAMPAGNE)
    T['heading'] = gold_rgba(text_mask('SCENT NOTES', font('Montserrat[wght].ttf', 32, 'SemiBold'),
                                       tracking=0.45)[0], emboss=2)
    T['notes'] = []
    for title, sub_, icon in NOTES:
        tm = text_mask(title, font('Montserrat[wght].ttf', 27, 'SemiBold'), tracking=0.06)[0]
        sm = text_mask(sub_, font('Montserrat[wght].ttf', 22, 'Regular'))[0]
        T['notes'].append((flat_rgba(tm, (245, 245, 240)), flat_rgba(sm, (165, 205, 222)), icon))
    return T


def icon_rgba(kind, progress, size=104):
    """Gold line icon in a circle; `progress` 0..1 draws the circle, then the symbol."""
    S = 3
    n = size * S
    m = np.zeros((n, n), np.uint8)
    c = n // 2
    r = int(n * 0.44)
    sweep = int(360 * clamp01(progress / 0.7))
    if sweep > 0:
        cv2.ellipse(m, (c, c), (r, r), -90, 0, sweep, 255, 2 * S, cv2.LINE_AA)
    inner = float(clamp01((progress - 0.55) / 0.45))
    if inner > 0:
        sym = np.zeros_like(m)
        k = n / 100.0
        if kind == 'waves':
            for j in range(3):
                xs = np.linspace(30, 70, 40)
                ys = 40 + j * 10 + 3.5 * np.sin((xs - 30) / 40 * 2 * np.pi)
                cv2.polylines(sym, [np.int32(np.stack([xs, ys], 1) * k)], False, 255, 2 * S, cv2.LINE_AA)
        elif kind == 'drop':
            th = np.linspace(0, 2 * np.pi, 80)
            xs = 50 + 13 * np.sin(th) * (1 - np.cos(th)) / 2 * 1.6
            ys = 33 + 32 * (1 - np.cos(th)) / 2
            cv2.polylines(sym, [np.int32(np.stack([xs, ys], 1) * k)], True, 255, 2 * S, cv2.LINE_AA)
        elif kind == 'crystal':
            pts = np.float32([[50, 28], [68, 44], [50, 72], [32, 44]])
            cv2.polylines(sym, [np.int32(pts * k)], True, 255, 2 * S, cv2.LINE_AA)
            for a, b in (((32, 44), (68, 44)), ((50, 28), (44, 44)), ((50, 28), (56, 44)),
                         ((44, 44), (50, 72)), ((56, 44), (50, 72))):
                cv2.line(sym, tuple(int(v * k) for v in a), tuple(int(v * k) for v in b), 255, S, cv2.LINE_AA)
        else:  # leaf
            th = np.linspace(0, np.pi, 40)
            up = np.stack([32 + 36 * th / np.pi, 50 - 15 * np.sin(th) - 6 * th / np.pi], 1)
            dn = np.stack([32 + 36 * th / np.pi, 50 + 15 * np.sin(th) - 6 * th / np.pi], 1)[::-1]
            cv2.polylines(sym, [np.int32(np.vstack([up, dn]) * k)], True, 255, 2 * S, cv2.LINE_AA)
            cv2.line(sym, (int(28 * k), int(54 * k)), (int(66 * k), int(45 * k)), 255, S, cv2.LINE_AA)
        m = np.maximum(m, (sym * inner).astype(np.uint8))
    a = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return gold_rgba(a, emboss=1.5)


def build_scene(rng):
    sc = {}
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = yy / H
    top, mid, low = np.float32([6, 26, 44]), np.float32([4, 40, 62]), np.float32([2, 10, 20])
    k = clamp01((v - 0.0) / 0.55)[..., None]
    bg = top * (1 - k) + mid * k
    k2 = smoothstep((v - 0.55) / 0.3)[..., None]
    bg = bg * (1 - k2) + low * k2
    halo = np.exp(-(((xx - W / 2) / 520) ** 2 + ((yy - 1180) / 620) ** 2))[..., None]
    sc['bg'] = bg + halo * np.float32([10, 70, 95])
    sc['halo'] = halo
    r = np.sqrt(((xx - W / 2) / (W * 0.62)) ** 2 + ((yy - H * 0.48) / (H * 0.6)) ** 2)
    sc['vignette'] = (1 - 0.45 * smoothstep((r - 0.55) / 0.7))[..., None]
    rays = np.zeros((H, W), np.float32)
    for _ in range(9):                                   # slanted beams from the surface
        x0 = rng.uniform(-200, W + 200)
        wdt = rng.uniform(25, 90)
        d = (xx - x0) - (yy * rng.uniform(0.15, 0.35))
        rays += np.exp(-0.5 * (d / wdt) ** 2) * rng.uniform(0.3, 1.0)
    sc['rays'] = rays * np.exp(-yy / 1100) * 0.5
    sc['floor_fade'] = smoothstep((yy - FLOOR) / 320)
    sc['star'] = make_star()
    sc['bubbles'] = [dict(x=rng.uniform(0, W), y0=rng.uniform(0, H + 200), v=rng.uniform(50, 170),
                          r=rng.uniform(3, 13), a=rng.uniform(10, 30), w=rng.uniform(1, 3),
                          ph=rng.uniform(0, 6.3), al=rng.uniform(0.3, 0.75)) for _ in range(46)]
    sc['motes'] = [dict(x=rng.uniform(0, W), y=rng.uniform(0, H), vx=rng.uniform(-8, 8), vy=rng.uniform(-14, 4),
                        s=gauss_dot(rng.uniform(1.0, 2.6)), g=rng.uniform(0.3, 0.8), ph=rng.uniform(0, 6.3))
                   for _ in range(70)]
    # splash droplets thrown up when the bottle lands (ballistic, stateless)
    sc['splash'] = [dict(x=W / 2 + rng.choice([-1, 1]) * rng.uniform(150, 290), vx=rng.uniform(-260, 260),
                         vy=-rng.uniform(380, 980), r=rng.uniform(2, 6), life=rng.uniform(0.6, 1.1))
                    for _ in range(80)]
    sc['bubble_sprite'] = {}
    return sc


def bubble_sprite(sc, r):
    key = int(round(r * 2))
    if key not in sc['bubble_sprite']:
        rr = key / 2
        n = int(rr * 2 + 6) | 1
        ax = np.arange(n, dtype=np.float32) - n // 2
        X, Y = np.meshgrid(ax, ax)
        d = np.sqrt(X ** 2 + Y ** 2)
        ring = np.exp(-0.5 * ((d - rr) / max(0.8, rr * 0.15)) ** 2)
        hl = np.exp(-(((X + rr * 0.35) / (rr * 0.3 + 0.5)) ** 2 + ((Y + rr * 0.35) / (rr * 0.3 + 0.5)) ** 2))
        sc['bubble_sprite'][key] = (ring * 0.8 + hl).astype(np.float32)
    return sc['bubble_sprite'][key]


# ---------------------------------------------------------------------------
# timeline
# ---------------------------------------------------------------------------
def bottle_pose(t):
    """(centre x, bottle height in px) - the base always sits on FLOOR."""
    hero = (540.0, 900.0)
    side = (300.0, 720.0)
    if t < LAND:
        return hero
    if t < 4.2:
        return (540.0, 900.0 + 25 * ease((t - LAND) / 1.45))
    if t < 4.8:
        u = ease((t - 4.2) / 0.6)
        return (540 + (side[0] - 540) * u, 925 + (side[1] - 925) * u)
    if t < 7.3:
        return (side[0], side[1] + 10 * ease((t - 4.8) / 2.5))
    if t < 7.9:
        u = ease((t - 7.3) / 0.6)
        return (side[0] + (540 - side[0]) * u, 750 + (880 - 750) * u)
    return (540.0, 880 + 30 * ease((t - 7.9) / 2.1))


def drop_offset(t):
    """Vertical offset of the bottle while it falls in (0 once landed)."""
    if t >= LAND:
        return 0.0
    u = clamp01((t - 2.0) / (LAND - 2.0))
    return float(-2100 * (1 - u ** 2)) if t >= 2.0 else -5000.0


def shake(t):
    if LAND <= t < LAND + 0.35:
        k = 1 - (t - LAND) / 0.35
        return 7 * k * math.sin(t * 90), 9 * k * math.cos(t * 77)
    return 0.0, 0.0


# ---------------------------------------------------------------------------
def draw_bottle(out, B, t, cx, hgt, yoff, sweep):
    s = hgt / B['h']
    src = B['rgba']
    hb, wb = src.shape[:2]
    # living liquid: caustic light inside the blue
    caus = cv2.resize(caustics(wb // 4, hb // 4, t * 1.3 + 3.0, 4.0), (wb, hb))
    rgba = src.copy()
    rgba[..., :3] += (B['liquid'] * caus * 1.6)[..., None] * np.float32([60, 200, 255])
    rgba[..., :3] += (B['edge'] * (0.55 + 0.25 * math.sin(t * 2.1)))[..., None] * np.float32([120, 220, 255])
    if sweep > 0:                                        # a band of light across the glass
        yy, xx = np.mgrid[0:hb, 0:wb].astype(np.float32)
        p = xx * 0.8 + yy * 0.6
        pos = -200 + (wb * 0.8 + hb * 0.6 + 400) * sweep
        band = np.exp(-0.5 * ((p - pos) / 38) ** 2) * math.sin(math.pi * sweep)
        rgba[..., :3] += (band * rgba[..., 3])[..., None] * np.float32([230, 245, 255]) * 0.7
    base = FLOOR + yoff
    cy = base - (B['base_y'] - hb / 2) * s
    # glow behind the bottle
    blit_add(out, B['glow'][..., None] * np.float32([45, 150, 200]), cx, cy, s * 1.05)
    blit(out, rgba, cx, cy, 1.0, s)
    return s, cy


def blit_add(out, rgb, cx, cy, scale):
    h, w = rgb.shape[:2]
    M = np.float32([[scale, 0, cx - scale * w / 2], [0, scale, cy - scale * h / 2]])
    x0, y0 = int(max(0, M[0, 2])), int(max(0, M[1, 2]))
    x1, y1 = int(min(W, M[0, 2] + scale * w)), int(min(H, M[1, 2] + scale * h))
    if x1 <= x0 or y1 <= y0:
        return
    M[0, 2] -= x0
    M[1, 2] -= y0
    out[y0:y1, x0:x1] += cv2.warpAffine(rgb, M, (x1 - x0, y1 - y0))


def draw_reflection(out, B, t, cx, hgt, yoff, ripple):
    if yoff < -400:
        return
    s = hgt / B['h']
    src = B['rgba'][:B['base_y'] + 2][::-1]               # flipped glass, starting at the base
    hb, wb = src.shape[:2]
    rh = int(min(hb * s, H - FLOOR))
    if rh <= 4:
        return
    patch = cv2.resize(src, (int(wb * s), int(hb * s)))[:rh]
    ph, pw = patch.shape[:2]
    yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    amp = 3 + 14 * ripple
    mx = xx + amp * np.sin(yy / 9.0 + t * 6) * (0.3 + yy / ph)
    patch = cv2.remap(patch, mx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    fade = (0.42 * (1 - yy / ph) ** 1.6)[..., None]
    x0 = int(round(cx - pw / 2))
    y0 = int(FLOOR + 2 + max(0.0, yoff))
    a0, a1 = max(0, x0), min(W, x0 + pw)
    b1 = min(H, y0 + ph)
    if a1 <= a0 or b1 <= y0:
        return
    pt = patch[:b1 - y0, a0 - x0:a1 - x0]
    f = fade[:b1 - y0]
    reg = out[y0:b1, a0:a1]
    reg *= 1 - pt[..., 3:4] * f
    reg += pt[..., :3] * f


def render_frame(fi, A):
    t = fi / FPS
    sc, B, T = A['scene'], A['bottle'], A['text']
    out = sc['bg'] * (0.93 + 0.07 * math.sin(t * 1.7))
    # light rays from above, swaying
    shift = int(40 * math.sin(t * 0.5))
    out += np.roll(sc['rays'], shift, axis=1)[..., None] * np.float32([60, 150, 190]) * (0.8 + 0.2 * math.sin(t * 1.3))
    # caustics on the water floor and faintly in the deep
    cz = cv2.resize(caustics(W // 6, H // 6, t), (W, H))
    out += cz[..., None] * np.float32([40, 120, 160]) * (0.25 + 0.9 * sc['floor_fade'])[..., None]
    # mirror surface line
    line = np.exp(-0.5 * ((np.arange(H, dtype=np.float32) - FLOOR) / 2.5) ** 2)[:, None, None]
    out += line * np.float32([70, 150, 190]) * 0.6
    # bubbles
    for b in sc['bubbles']:
        y = (b['y0'] - b['v'] * t) % (H + 200) - 100
        x = b['x'] + b['a'] * math.sin(b['w'] * t + b['ph'])
        add_light(out, bubble_sprite(sc, b['r']), x, y, np.float32([150, 220, 255]) * b['al'])

    cx, hgt = bottle_pose(t)
    yoff = drop_offset(t)
    landed = t >= LAND
    ripple = math.exp(-(t - LAND) / 0.8) if landed else 0.0
    # ripple rings on the surface when the bottle lands
    if landed and t < LAND + 1.6:
        u = (t - LAND) / 1.6
        layer = np.zeros((H, W), np.uint8)
        for j in range(3):
            uj = u - j * 0.12
            if 0 < uj < 1:
                rx = int(220 + 520 * uj)
                cv2.ellipse(layer, (int(cx), FLOOR + 6), (rx, int(rx * 0.12)), 0, 0, 360,
                            int(255 * (1 - uj) ** 1.5), 3, cv2.LINE_AA)
        out += cv2.GaussianBlur(layer.astype(np.float32) / 255, (0, 0), 1.5)[..., None] * np.float32([140, 220, 255])
    draw_reflection(out, B, t, cx, hgt, yoff, ripple)

    # finale title behind the bottle
    if t >= 7.6:
        u = ease_out((t - 7.6) / 0.8)
        blit(out, T['rooh'], W / 2, 430, u, 1.12 - 0.12 * u)

    if t >= 2.0:
        sweep = 0.0
        for s0, d in ((3.15, 1.0), (8.6, 1.0)):
            if s0 <= t < s0 + d:
                sweep = (t - s0) / d
        draw_bottle(out, B, t, cx, hgt, yoff, sweep)
        if landed and t < LAND + 0.25:                      # impact flash
            k = 1 - (t - LAND) / 0.25
            add_light(out, A['flare'], cx, FLOOR, np.float32([180, 230, 255]) * k)
    # splash droplets
    if landed and t < LAND + 1.2:
        tt = t - LAND
        for p in sc['splash']:
            if tt < p['life']:
                x = p['x'] + p['vx'] * tt + (cx - W / 2)
                y = FLOOR + p['vy'] * tt + 0.5 * 1800 * tt * tt
                if y < FLOOR + 4:
                    add_light(out, gauss_dot(p['r'] * 0.6), x, y, np.float32([200, 240, 255]) * (1 - tt / p['life']))

    # logo: letters form in the middle, then the logo rises to the top
    if t < 2.0:
        cyL = 860
        for i, (spr, _) in enumerate(T['logo_letters']):
            u = ease_out((t - 0.35 - i * 0.16) / 0.7)
            if u > 0:
                blit(out, spr, W / 2, cyL, u, 1.0, dx=(1 - u) * (i - 2) * 40, dy=(1 - u) * 26)
        su = ease((t - 1.15) / 0.6)
        blit(out, T['logo_sub'], W / 2, cyL + 110, su)
    else:
        u = ease((t - 2.0) / 0.55)
        cyL = 860 + (150 - 860) * u
        sl = 1 + (0.62 - 1) * u
        blit(out, T['logo'], W / 2, cyL, 1.0, sl)
        blit(out, T['logo_sub'], W / 2, cyL + 110 * sl, 1.0, sl)

    # scent notes
    if 4.45 <= t < NOTES_OUT + 0.5:
        out_k = 1 - ease((t - NOTES_OUT) / 0.45)
        hu = ease((t - 4.45) / 0.4) * out_k
        blit(out, T['heading'], 790, 560, hu, dx=(1 - hu) * 30)
        for i, (title, sub, kind) in enumerate(T['notes']):
            t0 = NOTES_T0 + i * NOTES_STEP
            if t < t0:
                continue
            yrow = 690 + i * 185
            p = clamp01((t - t0) / 0.55)
            blit(out, icon_rgba(kind, float(p)), 588, yrow, out_k)
            tu = ease_out((t - t0 - 0.2) / 0.45) * out_k
            blit(out, title, 650 + title.shape[1] / 2, yrow - 16, tu, dx=(1 - tu) * 50 + (1 - out_k) * 40)
            su = ease_out((t - t0 - 0.32) / 0.45) * out_k
            blit(out, sub, 650 + sub.shape[1] / 2, yrow + 22, su, dx=(1 - su) * 50 + (1 - out_k) * 40)

    # finale lines
    if t >= 8.3:
        sw = T['script']
        u = clamp01((t - 8.3) / 0.9)
        cut = int(sw.shape[1] * float(u))
        if cut > 2:
            part = sw.copy()
            part[:, cut:] = 0
            feather = min(30, cut)
            part[:, cut - feather:cut] *= np.linspace(1, 0, feather, dtype=np.float32)[None, :, None]
            blit(out, part, W / 2, 610, 1.0)
    if t >= 9.0:
        blit(out, T['signoff'], W / 2, 1780, ease((t - 9.0) / 0.5))

    # drifting motes
    for m in sc['motes']:
        x = (m['x'] + m['vx'] * t) % W
        y = (m['y'] + m['vy'] * t) % H
        tw = (0.5 + 0.5 * math.sin(t * 2.3 + m['ph'] * 5)) ** 2
        add_light(out, m['s'], x, y, np.float32([160, 225, 255]) * m['g'] * (0.3 + 0.7 * tw))
    # sparkles on the cap and the title at the end
    for ts, x, y, size in ((3.4, cx + 40, FLOOR - hgt * 0.97, 26), (9.3, W / 2 + 230, 360, 34),
                           (9.55, cx - 60, FLOOR - hgt * 0.97, 22)):
        u = (t - ts) / 0.6
        if 0 < u < 1:
            st = A['scene']['star']
            n = int(size * 2 * math.sin(math.pi * u) + 4) | 1
            R = cv2.getRotationMatrix2D((st.shape[0] / 2, st.shape[0] / 2), 40 * u, n / st.shape[0])
            R[:, 2] += n / 2 - st.shape[0] / 2
            add_light(out, cv2.warpAffine(st, R, (n, n)), x, y, np.float32([255, 245, 220]) * math.sin(math.pi * u))

    # bloom, vignette, camera shake, fade in
    small = cv2.resize(out, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bright = cv2.GaussianBlur(np.maximum(small - 150, 0), (0, 0), 6)
    out += cv2.resize(bright, (W, H)) * 0.55
    out *= sc['vignette']
    sx, sy = shake(t)
    if sx or sy:
        out = cv2.warpAffine(out, np.float32([[1, 0, sx], [0, 1, sy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    out *= float(smoothstep(t / 0.4))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# sound: low drone, a falling drop, riser into the landing, a soft pulse and chimes
# ---------------------------------------------------------------------------
def make_audio(path):
    n = int((DURATION + 0.6) * SR)
    dry = np.zeros((n, 2))

    def put(sig, t, gain, pan=0.0):
        i = int(t * SR)
        if i >= n:
            return
        sig = sig[:n - i]
        dry[i:i + len(sig), 0] += sig * gain * math.cos((pan + 1) * math.pi / 4)
        dry[i:i + len(sig), 1] += sig * gain * math.sin((pan + 1) * math.pi / 4)

    def tone(f0, f1, dur, decay):
        tt = np.arange(int(dur * SR)) / SR
        f = f0 * (f1 / f0) ** (tt / dur)
        ph = 2 * np.pi * np.cumsum(f) / SR
        return np.sin(ph) * np.exp(-tt / decay) * (1 - np.exp(-tt / 0.002))

    rng = np.random.default_rng(4)
    for note in (38, 45, 50, 53):                                  # D minor drone
        put(pad_voice(midi_hz(note), DURATION + 0.6), 0.0, 0.08, pan=rng.uniform(-0.4, 0.4))
    put(tone(1900, 950, 0.5, 0.12), 1.0, 0.18, pan=0.1)            # water drop
    put(tone(1500, 800, 0.4, 0.09), 1.18, 0.07, pan=-0.2)
    # riser into the landing
    m = int(0.75 * SR)
    tt = np.arange(m) / SR
    noise = rng.normal(0, 1, m)
    b, a = butter(2, [400 / (SR / 2), 6000 / (SR / 2)], 'band')
    riser = lfilter(b, a, noise) * (tt / tt[-1]) ** 2.5 + 0.4 * tone(200, 900, 0.75, 10.0)[:m] * (tt / tt[-1]) ** 2
    put(riser / np.abs(riser).max(), LAND - 0.75, 0.16)
    # impact: sub boom + splash
    put(tone(110, 40, 1.4, 0.35), LAND, 0.55)
    sp = lfilter(*butter(2, [700 / (SR / 2), 9000 / (SR / 2)], 'band'), rng.normal(0, 1, int(0.9 * SR)))
    sp *= np.exp(-np.arange(len(sp)) / SR / 0.18)
    put(sp / np.abs(sp).max(), LAND, 0.22)
    # soft pulse and harp figure through the middle
    beat = 0.5
    t0 = LAND + 0.5
    chords = [[50, 57, 62, 65, 69], [46, 53, 58, 62, 65], [41, 48, 53, 57, 60], [45, 52, 57, 61, 64]]
    k = 0
    while t0 + k * beat < 9.4:
        tk = t0 + k * beat
        put(tone(95, 45, 0.35, 0.12), tk, 0.22 if k % 2 == 0 else 0.12)
        ch = chords[int((tk - t0) // 2) % 4]
        put(pluck(midi_hz(ch[(k * 2) % 5] + 12), 1.6, seed=k), tk + 0.25, 0.09, pan=-0.4 + 0.1 * (k % 8))
        k += 1
    for i in range(4):                                             # a chime as each note appears
        put(bell(midi_hz(81 + [0, 3, 5, 7][i]), 1.4), NOTES_T0 + i * NOTES_STEP + 0.05, 0.07, pan=0.4)
    for tw in (4.2, 7.3):
        put(whoosh(0.7, seed=int(tw * 10)), tw - 0.1, 0.10)
    put(tone(80, 38, 1.6, 0.45), 7.6, 0.45)                        # title hit
    for kk, note in enumerate([86, 90, 93, 98]):
        put(bell(midi_hz(note), 2.0), 7.65 + kk * 0.07, 0.05, pan=-0.4 + 0.27 * kk)
    put(bell(midi_hz(74), 3.5), 9.0, 0.10)
    wet = np.stack([fftconvolve(dry[:, c], reverb_ir(seconds=2.2, seed=11 + c)[:, c])[:n] for c in range(2)], 1)
    mix = dry * 0.75 + wet * 0.5
    mix = lfilter(*butter(2, 28 / (SR / 2), 'highpass'), mix, axis=0)
    tt = np.arange(n) / SR
    mix *= (1 - smoothstep((tt - (DURATION - 0.6)) / 0.6))[:, None]
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.2) * 0.95
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((mix * 32767).astype(np.int16).tobytes())


def build_all():
    rng = np.random.default_rng(12)
    fl = gauss_dot(60.0)
    return dict(scene=build_scene(rng), bottle=build_bottle(), text=build_text(), flare=fl)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'zylan_rooh_film.mp4'))
    ap.add_argument('--stills', default=None)
    ap.add_argument('--no-audio', action='store_true')
    args = ap.parse_args()
    A = build_all()
    if args.stills:
        for ts in args.stills.split(','):
            Image.fromarray(render_frame(int(round(float(ts) * FPS)), A)).save(f'film_still_{float(ts):05.2f}.png')
        return
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-']
    audio = None
    if not args.no_audio:
        audio = os.path.splitext(args.out)[0] + '_audio.wav'
        make_audio(audio)
        cmd += ['-i', audio, '-c:a', 'aac', '-b:a', '192k', '-shortest']
    cmd += ['-c:v', 'libx264', '-preset', 'slow', '-crf', '17', '-x264-params', 'aq-mode=3',
            '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-movflags', '+faststart', args.out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(int(round(DURATION * FPS))):
        enc.stdin.write(render_frame(fi, A).tobytes())
    enc.stdin.close()
    enc.wait()
    if audio:
        os.remove(audio)
    print('done:', args.out)


if __name__ == '__main__':
    main()
