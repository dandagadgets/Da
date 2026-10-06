#!/usr/bin/env python3
"""
Animated Walima invitation.

Turns the static invitation card (card.jpg) into a short animated video:

  * the card starts blank and every line of text / gold ornament is revealed
    in reading order using the card's own pixels, so the lettering stays
    exactly as designed;
  * a slow cinematic camera pushes in on the text and pulls back to the full
    card at the end;
  * flickering candle light, gold shimmer sweeps and star glints on the gilding;
  * falling rose petals and drifting gold dust;
  * a soft synthesized harp/pad soundtrack with chimes on the key reveals.

Usage:
    python3 make_video.py                         # full video (2:3, 1080x1620)
    python3 make_video.py --format story          # 9:16 (1080x1920) for Status/Reels
    python3 make_video.py --stills 4,12,30        # preview PNG frames only

Requires numpy, scipy, Pillow, opencv-python-headless and ffmpeg on PATH.
"""
import argparse
import math
import multiprocessing as mp
import os
import subprocess
import time
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, fftconvolve, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
LUMA = np.array([0.299, 0.587, 0.114], np.float32)

FPS = 30
DURATION = 32.0
FORMATS = {'card': (1080, 1620), 'story': (1080, 1920)}

# ---------------------------------------------------------------------------
# Card layout, in pixel coordinates of card.jpg (843 x 1264).
# name, bbox (x0, y0, x1, y1), kind, reveal style, start (s), duration (s)
# ---------------------------------------------------------------------------
ELEMENTS = [
    ('bismillah', (345,  92, 492, 166), 'gold', 'wipe_rtl',    0.9, 1.8),
    ('divider',   (338, 170, 508, 206), 'gold', 'center_out',  2.5, 0.9),
    ('line1',     (222, 207, 632, 244), 'text', 'fade_up',     3.2, 1.1),
    ('line2',     (210, 244, 642, 280), 'text', 'fade_up',     3.8, 1.1),
    ('line3',     (210, 280, 642, 316), 'text', 'fade_up',     4.4, 1.1),
    ('name1',     (265, 326, 592, 376), 'text', 'wipe',        5.7, 1.5),
    ('son_of',    (258, 376, 596, 403), 'text', 'fade_up',     7.0, 1.0),
    ('amp',       (398, 403, 458, 446), 'text', 'pop',         7.9, 0.8),
    ('name2',     (228, 446, 624, 498), 'text', 'wipe',        8.6, 1.6),
    ('dau_of',    (318, 498, 536, 526), 'text', 'fade_up',     10.0, 1.0),
    ('walima',    (305, 526, 540, 602), 'text', 'write',       11.0, 2.0),
    ('celebr',    (326, 601, 528, 630), 'text', 'center_fade', 12.7, 1.0),
    ('date',      (222, 630, 638, 672), 'text', 'fade_up',     13.6, 1.1),
    ('time',      (234, 702, 622, 741), 'text', 'fade_up',     15.0, 1.1),
    ('venue1',    (328, 742, 530, 783), 'text', 'fade_up',     15.6, 1.1),
    ('venue2',    (274, 783, 582, 823), 'text', 'fade_up',     16.2, 1.1),
    ('love1',     (354, 858, 496, 896), 'text', 'fade_up',     17.6, 1.1),
    ('love2',     (328, 896, 524, 927), 'text', 'fade_up',     18.2, 1.1),
    ('love3',     (242, 927, 610, 967), 'text', 'fade_up',     18.8, 1.2),
    ('flourish',  (306, 970, 542, 1029), 'gold', 'center_out', 20.0, 1.2),
    ('rsvp',      (366, 1029, 482, 1069), 'text', 'pop_soft',  21.0, 0.9),
    ('phone1',    (248, 1069, 600, 1099), 'text', 'wipe',      21.6, 1.0),
    ('phone2',    (276, 1099, 572, 1125), 'text', 'wipe',      22.1, 1.0),
]
# elements whose letters flash gold as they are revealed, led by a moving light
GILDED = {'bismillah', 'divider', 'name1', 'name2', 'walima', 'flourish'}

# candle flames in the two lanterns: x, y, strength
FLAMES = [(822, 186, 1.0), (36, 728, 0.85)]

# camera: t, zoom (1 = whole card), vertical position (0 = top, 1 = bottom)
CAM_KEYS = [
    (0.0, 1.00, 0.30),
    (1.2, 1.05, 0.08),
    (3.6, 1.25, 0.03),
    (5.5, 1.30, 0.08),
    (9.0, 1.30, 0.26),
    (13.0, 1.30, 0.52),
    (17.5, 1.30, 0.84),
    (21.0, 1.30, 0.99),
    (23.0, 1.30, 1.00),
    (26.2, 1.00, 1.00),
    (32.0, 1.035, 0.50),
]

# gold light sweeps: start, duration, use the fully revealed gold mask?
SWEEPS = [(0.15, 1.7, False), (26.3, 1.9, True), (29.6, 1.9, True)]

# rose petal layers: size px, fall speed px/s (at 1080 px wide), blur, opacity,
# spawn rate per second while the text is revealed, and during the finale shower
PETAL_LAYERS = [
    ((26, 36), (45, 65), 0.8, 0.85, 0.14, 0.60),
    ((40, 58), (80, 115), 0.0, 1.00, 0.22, 1.30),
    ((100, 130), (200, 250), 4.0, 0.92, 0.035, 0.12),
]

# a final glint of light across the gilded lettering while the card is held
FINAL_GLINTS = {'bismillah': 28.0, 'name1': 28.25, 'name2': 28.45, 'walima': 28.7}
GLINT_SPEED = 520.0   # card pixels per second

GILD_COLOR = np.array([215, 160, 70], np.float32)
SHINE_COLOR = np.array([255, 248, 226], np.float32)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(x):
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def ease_out_cubic(x):
    x = clamp01(x)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out_sine(x):
    x = clamp01(x)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def ease_out_back(x, s=1.9):
    x = clamp01(x)
    return 1.0 + (s + 1.0) * (x - 1.0) ** 3 + s * (x - 1.0) ** 2


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def gauss_kernel2d(size, sigma):
    ax = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    g = np.exp(-0.5 * (ax / sigma) ** 2)
    return np.outer(g, g).astype(np.float32)


# ---------------------------------------------------------------------------
# Asset preparation: masks, blank card, per-element layers, gold masks
# ---------------------------------------------------------------------------
def build_assets(card_path):
    im8 = np.asarray(Image.open(card_path).convert('RGB'))
    im = im8.astype(np.float32)
    H, W = im.shape[:2]
    lum = im @ LUMA

    # 1. text / ornament pixels: darker than the locally "closed" parchment
    bg_l = cv2.GaussianBlur(cv2.morphologyEx(lum, cv2.MORPH_CLOSE, disk(15)), (0, 0), 6)
    d = bg_l - lum
    labels = np.zeros((H, W), np.int32)
    for i, (_, (x0, y0, x1, y1), kind, *_r) in enumerate(ELEMENTS):
        sd, sl = d[y0:y1, x0:x1], lum[y0:y1, x0:x1]
        m = ((sd > 22) & (sl < 175)) if kind == 'text' else (sd > 12)
        sub = labels[y0:y1, x0:x1]
        sub[m & (sub == 0)] = i + 1
    core = labels > 0
    fill_region = cv2.dilate(core.astype(np.uint8), disk(6)) > 0
    excl = cv2.dilate(core.astype(np.uint8), disk(10)) > 0
    dist_out = cv2.distanceTransform((~fill_region).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = smoothstep(1.0 - dist_out / 5.0)
    ext = alpha > 0

    # every affected pixel belongs to the element of its nearest text pixel
    _, idx = cv2.distanceTransformWithLabels((~core).astype(np.uint8), cv2.DIST_L2, 5,
                                             labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(idx.max() + 1, np.int32)
    lut[idx[core]] = labels[core]
    owner = lut[idx]
    owner[~ext] = 0

    # 2. plain parchment pixels (flood fill from the middle of the card)
    blur8 = cv2.GaussianBlur(im8, (0, 0), 1.0)
    ffmask = np.zeros((H + 2, W + 2), np.uint8)
    for sx, sy in [(420, 690), (420, 840), (420, 320), (420, 200), (260, 1000)]:
        if ffmask[sy + 1, sx + 1] == 0:
            cv2.floodFill(blur8.copy(), ffmask, (sx, sy), 0, (3, 3, 3), (3, 3, 3),
                          4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
    plain = ffmask[1:-1, 1:-1] > 0
    plain[1015:, :245] = False          # keep the little paper tag out of it
    w = (plain & ~excl).astype(np.float32)

    # 3. blank card: smooth polynomial model of the parchment light falloff,
    #    plus a local correction diffused in from the surrounding paper
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn, yn = (xx - 425) / 300.0, (yy - 620) / 560.0
    deg = 6
    terms = [xn ** i * yn ** j for i in range(deg + 1) for j in range(deg + 1 - i)]
    fitm = w > 0
    fitm[:, :180] = False
    fitm[:, 672:] = False
    fitm[:55] = False
    fitm[1140:] = False
    A = np.stack([t[fitm] for t in terms], 1).astype(np.float64)
    P = np.zeros_like(im)
    for ch in range(3):
        coef, *_ = np.linalg.lstsq(A, im[..., ch][fitm].astype(np.float64), rcond=None)
        P[..., ch] = sum(c * t for c, t in zip(coef, terms))
    r = (im - P) * w[..., None]
    corr = np.zeros_like(im)
    for s in (6, 14, 30):
        num = cv2.GaussianBlur(r, (0, 0), s)
        den = cv2.GaussianBlur(w, (0, 0), s)
        corr += num / (den[..., None] + 0.08)
    fill = P + corr / 3.0
    rng = np.random.default_rng(7)
    grain = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.6)
    fill += (grain / grain.std() * 1.1)[..., None]
    clean = np.clip(im * (1 - alpha[..., None]) + fill * alpha[..., None], 0, 255)
    clean = clean.astype(np.float32)
    diff = im - clean

    # 4. per-element difference layers (adding D back onto the blank card
    #    reproduces the original exactly)
    elements = []
    for i, (name, box, kind, style, t0, dur) in enumerate(ELEMENTS):
        sel = owner == i + 1
        ys, xs = np.nonzero(sel)
        bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        pad = 30
        X0, Y0 = max(0, bx0 - pad), max(0, by0 - pad)
        X1, Y1 = min(W, bx1 + pad), min(H, by1 + pad)
        D = (diff * sel[..., None])[Y0:Y1, X0:X1].copy()
        dark = -(D @ LUMA)
        Aa = clamp01(dark / (140.0 if kind == 'text' else 45.0)).astype(np.float32)
        # "pen" height per column: where the strokes are in that column
        colw = Aa.sum(0)
        rows = np.arange(Y0, Y1, dtype=np.float32)
        with np.errstate(invalid='ignore', divide='ignore'):
            pen = (Aa * rows[:, None]).sum(0) / colw
        good = colw > 0.5
        cols = np.arange(X1 - X0)
        pen = np.interp(cols, cols[good], pen[good]) if good.any() else np.full(len(cols), (Y0 + Y1) / 2)
        pen = np.convolve(np.pad(pen, 6, mode='edge'), np.ones(13) / 13, mode='valid')
        el = dict(name=name, kind=kind, style=style, t0=t0, dur=dur, box=(X0, Y0, X1, Y1),
                  tight=(bx0, by0, bx1, by1), D=D, A=Aa, pen=pen.astype(np.float32),
                  xs=np.arange(X0, X1, dtype=np.float32), gild=name in GILDED,
                  soft={'write': 12.0, 'wipe_rtl': 22.0}.get(style, 34.0))
        if style in ('wipe', 'wipe_rtl', 'write', 'center_out', 'center_fade'):
            ts = np.linspace(t0, t0 + dur, 600)
            fr = np.array([front_pos(el, t) for t in ts])
            if style == 'wipe_rtl':
                fr = -fr
            coord = el_coord(el)
            el['t_rev'] = np.interp(coord, fr, ts).astype(np.float32)
        el['t_done'] = t0 + dur + (2.6 if el['gild'] else 0.0)
        elements.append(el)

    # 5. gold: frame, filigree, lanterns (for shimmer sweeps and glints)
    hsv = cv2.cvtColor(im8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gold = ((hh >= 8) & (hh <= 32) & (ss > 90) & (vv > 130)).astype(np.float32)
    gold *= clamp01((vv - 130) / 100.0) * clamp01((ss - 90) / 70.0 + 0.4)
    text_ids = [i + 1 for i, e in enumerate(ELEMENTS) if e[2] == 'text']
    text_px = cv2.dilate(np.isin(owner, text_ids).astype(np.uint8), disk(2)) > 0
    gold[text_px] = 0                     # keep the lettering legible during sweeps
    gold = cv2.GaussianBlur(gold, (0, 0), 0.8)
    elem_px = cv2.dilate((owner > 0).astype(np.uint8), disk(3)) > 0
    gold_clean = gold * (~elem_px)
    bright = (gold > 0.35) & (vv > 185) & ~elem_px
    gy, gx = np.nonzero(bright)
    pick = rng.choice(len(gx), size=min(2500, len(gx)), replace=False)
    glint_pts = np.stack([gx[pick], gy[pick]], 1).astype(np.float32)

    # shimmer coordinate along a diagonal
    ang = math.radians(32)
    proj = (xx * math.cos(ang) + yy * math.sin(ang)).astype(np.float32)

    # candle glow sprites (source space)
    flames = []
    for fx, fy, strength in FLAMES:
        R = 150
        x0, y0 = max(0, fx - R), max(0, fy - R)
        x1, y1 = min(W, fx + R), min(H, fy + R)
        gx_, gy_ = np.meshgrid(np.arange(x0, x1) - fx, np.arange(y0, y1) - fy)
        r2 = (gx_ ** 2 + (gy_ * 0.8) ** 2).astype(np.float32)
        flames.append(dict(box=(x0, y0, x1, y1), strength=strength,
                           inner=np.exp(-r2 / (2 * 16.0 ** 2)).astype(np.float32),
                           outer=np.exp(-r2 / (2 * 55.0 ** 2)).astype(np.float32)))

    return dict(W=W, H=H, clean=clean, elements=elements, gold=gold,
                gold_clean=gold_clean, glint_pts=glint_pts, proj=proj,
                proj_range=(float(proj.min()), float(proj.max())), flames=flames)


def front_pos(el, t):
    """Position of the reveal front of a wipe-like element at time t."""
    bx0, _, bx1, _ = el['tight']
    soft = el['soft']
    u = (t - el['t0']) / el['dur']
    if el['style'] == 'write':
        e = 0.5 * ease_in_out_sine(u) + 0.5 * clamp01(u)      # near-constant pen speed
    else:
        e = ease_in_out_sine(u)
    if el['style'] == 'wipe_rtl':
        return bx1 + soft - (bx1 - bx0 + 2 * soft) * e
    if el['style'] in ('center_out', 'center_fade'):
        return ((bx1 - bx0) / 2 + soft) * e
    return bx0 - soft + (bx1 - bx0 + 2 * soft) * e


def el_coord(el):
    """Per-column coordinate that the front moves along (monotonic in time)."""
    xs = el['xs']
    bx0, _, bx1, _ = el['tight']
    if el['style'] == 'wipe_rtl':
        return -xs                       # front moves to smaller x
    if el['style'] in ('center_out', 'center_fade'):
        return np.abs(xs - (bx0 + bx1) / 2)
    return xs


def shower_amount(t):
    return float(smoothstep((t - 21.5) / 1.5) * (1.0 - smoothstep((t - 25.5) / 2.5)))


def gild_curve(tau):
    """Gold flash of a column as the front passes it (tau = time since)."""
    s = smoothstep((tau + 0.05) / 0.05)
    tp = np.maximum(tau, 0)
    return s * (0.65 * np.exp(-tp / 0.15) + 0.35 * np.exp(-tp / 0.8))


# ---------------------------------------------------------------------------
# Per-frame rendering
# ---------------------------------------------------------------------------
ASSETS = None
SCENE = None


def shift_patch(p, dx, dy, scale=1.0, center=None):
    h, w = p.shape[:2]
    cx, cy = center if center is not None else (w / 2, h / 2)
    M = np.float32([[scale, 0, cx - scale * cx + dx], [0, scale, cy - scale * cy + dy]])
    return cv2.warpAffine(p, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def add_element(src, el, t):
    u = (t - el['t0']) / el['dur']
    if u <= 0:
        return
    X0, Y0, X1, Y1 = el['box']
    tgt = src[Y0:Y1, X0:X1]
    D, A = el['D'], el['A']
    if t >= el['t_done']:
        tgt += D
        tg = FINAL_GLINTS.get(el['name'])
        if tg is not None:
            bx0, _, bx1, _ = el['tight']
            span = (bx1 - bx0) / GLINT_SPEED
            if tg - 0.3 < t < tg + span + 0.3:
                xs = el['xs']
                pos = (bx1 - xs) if el['style'] == 'wipe_rtl' else (xs - bx0)
                tau = t - (tg + pos / GLINT_SPEED)
                g = 0.7 * np.exp(-0.5 * (tau / 0.08) ** 2)
                tgt += (A * g[None, :])[..., None] * GILD_COLOR
        return
    style, xs, soft = el['style'], el['xs'], el['soft']
    bx0, by0, bx1, by1 = el['tight']

    if style == 'fade_up':
        e = float(ease_out_cubic(u))
        p = D
        sig = 2.6 * (1 - e)
        if sig > 0.3:
            p = cv2.GaussianBlur(p, (0, 0), sig)
        p = shift_patch(p, 0, 14 * (1 - e))
        tgt += p * e
    elif style in ('wipe', 'wipe_rtl', 'write', 'center_out', 'center_fade'):
        f = front_pos(el, t)
        if style == 'wipe_rtl':
            m = smoothstep((xs - f) / soft)
        elif style in ('center_out', 'center_fade'):
            m = smoothstep((f - np.abs(xs - (bx0 + bx1) / 2)) / soft)
        else:
            m = smoothstep((f - xs) / soft)
        op = 1.0
        if style == 'center_fade':
            op = float(ease_out_cubic(u * 1.3))
        tgt += D * (m * op)[None, :, None]
        if el['gild']:
            g = gild_curve(t - el['t_rev'])
            tgt += (A * g[None, :])[..., None] * GILD_COLOR
    elif style in ('pop', 'pop_soft'):
        if style == 'pop':
            sc = 1.0 + 0.9 * (1.0 - float(ease_out_back(u)))
            op = float(ease_out_cubic(u * 2.2))
            sig = 3.0 * (1 - float(ease_out_cubic(u)))
        else:
            sc = 1.0 + 0.22 * (1.0 - float(ease_out_cubic(u)))
            op = float(ease_out_cubic(u * 1.4))
            sig = 2.0 * (1 - float(ease_out_cubic(u)))
        c = ((bx0 + bx1) / 2 - X0, (by0 + by1) / 2 - Y0)
        p = shift_patch(D, 0, 0, sc, c)
        if sig > 0.3:
            p = cv2.GaussianBlur(p, (0, 0), sig)
        tgt += p * op


def add_candles(src, t):
    for k, fl in enumerate(ASSETS['flames']):
        n1 = SCENE['flicker'][k](t)
        n2 = SCENE['flicker'][k](t * 0.53 + 3.1)
        x0, y0, x1, y1 = fl['box']
        reg = src[y0:y1, x0:x1]
        reg *= (1.0 + 0.16 * n1 * fl['inner'])[..., None]
        amp = fl['strength'] * (26.0 + 12.0 * n2)
        reg += (fl['outer'] * amp)[..., None] * np.array([1.0, 0.62, 0.25], np.float32)


def add_shimmer(src, t):
    for t0, dur, full in SWEEPS:
        u = (t - t0) / dur
        if not 0 < u < 1:
            continue
        lo, hi = ASSETS['proj_range']
        pos = lo - 160 + (hi - lo + 320) * float(ease_in_out_sine(u))
        band = np.exp(-0.5 * ((ASSETS['proj'] - pos) / 55.0) ** 2)
        mask = ASSETS['gold'] if full else ASSETS['gold_clean']
        amt = np.sqrt(mask) * band * (0.92 * math.sin(math.pi * u) ** 0.5)
        src += amt[..., None] * (SHINE_COLOR - src)


def camera(t):
    W, H = ASSETS['W'], ASSETS['H']
    OW, OH = SCENE['out']
    z = math.exp(float(SCENE['cam_z'](t)))
    py = float(np.clip(SCENE['cam_y'](t), 0, 1))
    s = SCENE['s0'] * z
    hw, hh = OW / (2 * s), OH / (2 * s)
    cx = min(max(W / 2, hw), W - hw)
    cy = hh + py * max(0.0, H - 2 * hh)
    M = np.float32([[s, 0, OW / 2 - s * cx], [0, s, OH / 2 - s * cy]])
    return M, s


def screen_add(out, x, y, sprite, color, gain):
    """Screen-blend an additive light sprite centred at (x, y)."""
    h, w = sprite.shape[:2]
    x0, y0 = int(round(x - w / 2)), int(round(y - h / 2))
    OH, OW = out.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(OW, x0 + w), min(OH, y0 + h)
    if x1c <= x0c or y1c <= y0c:
        return
    sp = sprite[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    reg = out[y0c:y1c, x0c:x1c]
    light = sp[..., None] * (color * gain)
    reg += light * (1.0 - np.clip(reg, 0, 255) / 255.0)


def draw_star(out, x, y, size, gain, angle):
    st = SCENE['star']
    S = st.shape[0]
    sc = size / (S * 0.5)
    ps = int(S * sc) | 1
    if ps < 5:
        return
    M = cv2.getRotationMatrix2D((S / 2, S / 2), angle, sc)
    M[0, 2] += ps / 2 - S / 2
    M[1, 2] += ps / 2 - S / 2
    patch = cv2.warpAffine(st, M, (ps, ps), flags=cv2.INTER_LINEAR)
    screen_add(out, x, y, patch, np.array([255, 244, 214], np.float32), gain)


def draw_sparkles(out, t, M, s):
    def to_screen(x, y):
        return M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]

    for ev in SCENE['glints']:
        u = (t - ev[0]) / ev[1]
        if 0 < u < 1:
            x, y = to_screen(ev[2], ev[3])
            env = math.sin(math.pi * u) ** 1.5
            draw_star(out, x, y, ev[4] * s * (0.45 + 0.55 * env), ev[5] * env, ev[6] + 50 * u)
    # the moving "pen" light on reveals
    for el in ASSETS['elements']:
        if el['name'] not in GILDED:
            continue
        u = (t - el['t0']) / el['dur']
        if not 0 < u < 1:
            continue
        f = front_pos(el, t)
        env = math.sin(math.pi * min(1, u * 1.15)) ** 0.6
        X0 = el['box'][0]
        if el['style'] in ('center_out', 'center_fade'):
            cxm = (el['tight'][0] + el['tight'][2]) / 2
            xs_ = [cxm - f, cxm + f]
        else:
            xs_ = [f]
        for xf in xs_:
            ci = int(np.clip(round(xf - X0), 0, len(el['pen']) - 1))
            x, y = to_screen(xf, el['pen'][ci])
            draw_star(out, x, y, 26 * s, 1.0 * env, 20 + 180 * u)


def draw_particles(out, t):
    OW, OH = SCENE['out']
    intro = float(smoothstep((t - 0.3) / 2.0))
    # soft bokeh orbs
    for b in SCENE['bokeh']:
        y = (b['y0'] + b['vy'] * t) % (OH + 2 * b['r']) - b['r']
        x = b['x0'] + b['ax'] * math.sin(b['wx'] * t + b['ph'])
        pulse = 0.6 + 0.4 * math.sin(b['wp'] * t + b['ph'] * 2)
        screen_add(out, x, y, b['sprite'], b['color'], b['gain'] * pulse * intro)
    # gold dust
    for p in SCENE['dust']:
        y = (p['y0'] + p['vy'] * t) % (OH + 40) - 20
        x = (p['x0'] + p['ax'] * math.sin(p['wx'] * t + p['ph'])) % OW
        tw = (0.5 + 0.5 * math.sin(p['wt'] * t + p['ph'] * 3)) ** 2
        screen_add(out, x, y, p['sprite'], p['color'], p['gain'] * (0.25 + 0.75 * tw) * intro)
    # rose petals
    for p in SCENE['petals']:
        age = t - p['t0']
        if age < 0:
            continue
        m = p['size'] * 0.7
        y = -m + p['vy'] * age
        if y > OH + m:
            continue
        x = p['x0'] + p['ax'] * math.sin(p['wx'] * t + p['ph']) + p['vx'] * age
        ang = p['rot0'] + p['wr'] * t
        tumble = math.cos(p['wt'] * t + p['ph'])
        squash = max(0.32, abs(tumble))
        shade = 0.74 + 0.26 * abs(tumble)
        blit_sprite(out, p['sprite'], x, y, p['size'], ang, squash, p['alpha'], shade, p['blur'])


def blit_sprite(out, spr, x, y, size, angle, squash, alpha, shade, blur):
    """Alpha-composite a premultiplied RGBA sprite (rotated, squashed)."""
    if alpha <= 0.01:
        return
    sh, sw = spr.shape[:2]
    sc = size / max(sh, sw)
    ps = int(size * 1.25) + 6
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    # sprite -> patch: centre, squash x, scale, rotate, move to patch centre
    A = np.array([[ca, -sa], [sa, ca]]) @ np.diag([sc * squash, sc])
    off = np.array([ps / 2, ps / 2]) - A @ np.array([sw / 2, sh / 2])
    M = np.hstack([A, off[:, None]]).astype(np.float32)
    patch = cv2.warpAffine(spr, M, (ps, ps), flags=cv2.INTER_LINEAR)
    if blur > 0.3:
        patch = cv2.GaussianBlur(patch, (0, 0), blur)
    OH, OW = out.shape[:2]
    x0, y0 = int(round(x - ps / 2)), int(round(y - ps / 2))
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    x1c, y1c = min(OW, x0 + ps), min(OH, y0 + ps)
    if x1c <= x0c or y1c <= y0c:
        return
    pt = patch[sy0:sy0 + (y1c - y0c), sx0:sx0 + (x1c - x0c)]
    aa = pt[..., 3:4] * alpha
    reg = out[y0c:y1c, x0c:x1c]
    reg *= 1.0 - aa
    reg += pt[..., :3] * (alpha * shade)


def render_frame(fi):
    t = fi / FPS
    src = ASSETS['clean'].copy()
    for el in ASSETS['elements']:
        add_element(src, el, t)
    add_candles(src, t)
    add_shimmer(src, t)
    M, s = camera(t)
    OW, OH = SCENE['out']
    out = cv2.warpAffine(src, M, (OW, OH), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REFLECT)
    draw_sparkles(out, t, M, s)
    draw_particles(out, t)
    out *= SCENE['vignette']
    fade = 0.70 + 0.30 * float(ease_out_cubic(t / 1.0))
    fade *= 1.0 - float(smoothstep((t - (DURATION - 1.0)) / 1.0))
    out *= fade
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Scene setup: sprites, particles, sparkle schedule, camera path
# ---------------------------------------------------------------------------
def make_star(S=97):
    ax = np.arange(S, dtype=np.float32) - S // 2
    X, Y = np.meshgrid(ax, ax)
    R = np.sqrt(X ** 2 + Y ** 2)
    star = np.exp(-R ** 2 / (2 * (S * 0.03) ** 2)) * 1.2
    star += np.exp(-R ** 2 / (2 * (S * 0.11) ** 2)) * 0.30
    for ang, length, width, g in [(0, 0.24, 0.010, 1.0), (90, 0.24, 0.010, 1.0),
                                  (45, 0.11, 0.008, 0.45), (135, 0.11, 0.008, 0.45)]:
        a = math.radians(ang)
        u = X * math.cos(a) + Y * math.sin(a)
        v = -X * math.sin(a) + Y * math.cos(a)
        star += g * np.exp(-v ** 2 / (2 * (S * width) ** 2)) * np.exp(-np.abs(u) / (S * length))
    return np.clip(star, 0, 1.6).astype(np.float32)


def make_dot(sigma):
    n = int(sigma * 6) | 1
    return gauss_kernel2d(n, sigma)


def make_orb(r):
    n = int(r * 2.6) | 1
    ax = np.arange(n, dtype=np.float32) - n // 2
    X, Y = np.meshgrid(ax, ax)
    R = np.sqrt(X ** 2 + Y ** 2)
    orb = smoothstep((r - R) / (r * 0.18)) * (0.75 + 0.25 * smoothstep((R - r * 0.55) / (r * 0.45)))
    return cv2.GaussianBlur(orb.astype(np.float32), (0, 0), r * 0.06 + 0.5)


PETAL_PALETTES = [  # (base, edge) colours: deep red, crimson, rose pink
    ((105, 4, 24), (196, 26, 52)),
    ((140, 10, 38), (224, 52, 84)),
    ((176, 70, 96), (246, 162, 178)),
]


def make_rose_petal(h=180, seed=0, palette=0):
    """A shaded, slightly cupped rose petal as a premultiplied RGBA sprite."""
    rng = np.random.default_rng(seed)
    S = 4
    w = int(h * 0.94)
    wav = rng.uniform(0, 2 * np.pi)
    pts = []
    for th in np.linspace(0, 2 * np.pi, 480, endpoint=False):
        y = -math.cos(th)                                   # -1 = outer edge, +1 = base
        x = math.sin(th) * 0.95 * (1 - 0.50 * ((y + 1) / 2) ** 1.7)
        x *= 1 + 0.035 * math.sin(6 * th + wav)              # soft ruffle
        if y < 0:
            y += 0.12 * math.exp(-(x / 0.22) ** 2) * -y      # gentle notch at the top
        pts.append(((x * 0.5 + 0.5) * w * S, (y * 0.48 + 0.5) * h * S))
    mask = Image.new('L', (w * S, h * S), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=255)
    a = np.asarray(mask.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn, xn = yy / h, (xx - w / 2) / (w / 2)
    base, edge_c = (np.array(c, np.float32) for c in PETAL_PALETTES[palette])
    k = clamp01((yn - 0.08) / 0.85)[..., None] ** 1.4
    col = edge_c * (1 - k) + base * k                        # light outer edge -> deep base
    inner = cv2.GaussianBlur(a, (0, 0), h * 0.06)
    rim = clamp01(1 - inner / 0.75) * a * (yn < 0.6)
    col += (rim * 38)[..., None]                             # light catching the rim
    col *= (0.80 + 0.20 * (xn + 1) / 2)[..., None]           # cupped: one side in shade
    col *= (1 - 0.18 * np.exp(-(xn / 0.07) ** 2) * (yn > 0.35))[..., None]   # centre crease
    hl = np.exp(-(((xn + 0.35) / 0.35) ** 2 + ((yn - 0.33) / 0.22) ** 2))
    col += hl[..., None] * np.array([46, 30, 34], np.float32)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 1.2)
    col *= (1 + 0.05 * tex / tex.std())[..., None]           # velvety texture
    col = np.clip(col, 0, 255)
    return np.dstack([col * a[..., None], a]).astype(np.float32)


def make_scene(fmt):
    OW, OH = FORMATS[fmt]
    W, H = ASSETS['W'], ASSETS['H']
    rng = np.random.default_rng(2026)
    sc = dict(out=(OW, OH), s0=max(OW / W, OH / H))

    ts = np.array([k[0] for k in CAM_KEYS])
    sc['cam_z'] = PchipInterpolator(ts, np.log([k[1] for k in CAM_KEYS]), extrapolate=True)
    sc['cam_y'] = PchipInterpolator(ts, [k[2] for k in CAM_KEYS], extrapolate=True)

    # candle flicker: smooth noise from a few incommensurate sines
    def make_flicker(seed):
        r = np.random.default_rng(seed)
        f = r.uniform(1.3, 11.0, 7)
        ph = r.uniform(0, 2 * np.pi, 7)
        amp = 1 / np.sqrt(f)
        amp /= amp.sum()
        return lambda t: float(np.sum(amp * np.sin(2 * np.pi * f * t + ph)) * 1.6)
    sc['flicker'] = [make_flicker(11 + k) for k in range(len(FLAMES))]

    yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
    rr = np.sqrt(((xx - OW / 2) / (OW / 2)) ** 2 + ((yy - OH / 2) / (OH / 2)) ** 2)
    sc['vignette'] = (1.0 - 0.30 * smoothstep((rr - 0.55) / 0.75))[..., None].astype(np.float32)

    sc['star'] = make_star()

    # glints on the gold work: a Poisson schedule, placed where the camera looks
    glints = []
    pts = ASSETS['glint_pts']
    t = 0.4
    while t < DURATION - 0.8:
        rate = 3.0 if t < 23 else 6.0
        t += rng.exponential(1.0 / rate)
        M, s = _camera_with(sc, t)
        x = M[0, 0] * pts[:, 0] + M[0, 2]
        y = M[1, 1] * pts[:, 1] + M[1, 2]
        vis = np.nonzero((x > 20) & (x < OW - 20) & (y > 20) & (y < OH - 20))[0]
        if len(vis) == 0:
            continue
        p = pts[rng.choice(vis)]
        glints.append((t, rng.uniform(0.6, 1.1), p[0], p[1], rng.uniform(9, 17),
                       rng.uniform(0.6, 1.0), rng.uniform(0, 90)))
    # extra glints on the revealed gold ornaments + the ampersand burst
    for el in ASSETS['elements']:
        if el['kind'] == 'gold' or el['name'] in ('amp',):
            X0, Y0 = el['box'][:2]
            ys, xs = np.nonzero(el['A'] > 0.5)
            if len(xs) == 0:
                continue
            n = 8 if el['name'] == 'amp' else 5
            for k in range(n):
                i = rng.integers(len(xs))
                t0 = el['t0'] + el['dur'] * (0.55 if el['name'] == 'amp' else 1.0) + rng.uniform(0, 0.6)
                glints.append((t0, rng.uniform(0.5, 0.9), X0 + xs[i], Y0 + ys[i],
                               rng.uniform(8, 14), rng.uniform(0.7, 1.0), rng.uniform(0, 90)))
    # trailing twinkles behind the pen light
    for el in ASSETS['elements']:
        if el['name'] not in GILDED:
            continue
        X0 = el['box'][0]
        for tk in np.arange(el['t0'] + 0.05, el['t0'] + el['dur'] * 0.95, 0.06):
            f = front_pos(el, tk)
            if el['style'] in ('center_out', 'center_fade'):
                cxm = (el['tight'][0] + el['tight'][2]) / 2
                f = cxm + f * rng.choice([-1, 1])
            ci = int(np.clip(round(f - X0), 0, len(el['pen']) - 1))
            glints.append((tk, rng.uniform(0.35, 0.6), f + rng.normal(0, 4),
                           el['pen'][ci] + rng.normal(0, 7), rng.uniform(4, 8),
                           rng.uniform(0.5, 0.9), rng.uniform(0, 90)))
    sc['glints'] = glints

    # gold dust motes
    unit = OW / 1080.0
    dots = {s_: make_dot(s_ * unit) for s_ in (1.0, 1.5, 2.2, 3.2)}
    dust = []
    for _ in range(int(80 * OH / 1620)):
        s_ = rng.choice(list(dots.keys()), p=[0.35, 0.3, 0.22, 0.13])
        dust.append(dict(x0=rng.uniform(0, OW), y0=rng.uniform(0, OH + 40),
                         vy=-rng.uniform(10, 32) * unit, ax=rng.uniform(8, 30) * unit,
                         wx=rng.uniform(0.2, 0.7), wt=rng.uniform(1.0, 3.5),
                         ph=rng.uniform(0, 2 * np.pi), sprite=dots[s_] * 2.2,
                         color=np.array([255, 212, 140], np.float32),
                         gain=rng.uniform(0.55, 1.0)))
    sc['dust'] = dust

    # out-of-focus orbs, kept towards the sides of the frame
    bokeh = []
    for k in range(12):
        r = rng.uniform(16, 42) * unit
        side = rng.random() < 0.5
        x0 = rng.uniform(0, OW * 0.2) if side else rng.uniform(OW * 0.8, OW)
        bokeh.append(dict(x0=x0, y0=rng.uniform(0, OH), vy=-rng.uniform(4, 12) * unit, r=r,
                          ax=rng.uniform(5, 20) * unit, wx=rng.uniform(0.1, 0.3),
                          wp=rng.uniform(0.4, 1.2), ph=rng.uniform(0, 2 * np.pi),
                          sprite=make_orb(r), gain=rng.uniform(0.10, 0.20),
                          color=np.array([255, 196, 120], np.float32)))
    sc['bokeh'] = bokeh

    # falling rose petals at three depths, spawned above the frame on a rate
    # schedule: sparse while the text is being revealed, a shower when the
    # whole card comes into view (kept mostly to the sides so the details stay
    # readable)
    sprites = [make_rose_petal(180, seed=k, palette=k % 3) for k in range(6)]
    petals = []
    for size, speed, blur, alpha, r_reveal, r_finale in PETAL_LAYERS:
        t = -5.0                     # a few petals are already drifting in at the start
        while True:
            rate = r_reveal + (r_finale - r_reveal) * shower_amount(t)
            t += rng.exponential(1.0 / rate)
            if t > DURATION:
                break
            sz = rng.uniform(*size) * unit
            if t > 21.5 and rng.random() < 0.75:
                x0 = rng.uniform(0, OW * 0.22) if rng.random() < 0.5 else rng.uniform(OW * 0.78, OW)
            else:
                x0 = rng.uniform(0, OW)
            boost = 1.0 + 0.6 * shower_amount(t)
            petals.append(dict(t0=t, sprite=sprites[rng.choice([0, 0, 1, 1, 3, 4, 2, 5])],
                               size=sz, x0=x0, vy=rng.uniform(*speed) * unit * boost,
                               vx=rng.uniform(-6, 10) * unit,
                               ax=rng.uniform(20, 55) * unit, wx=rng.uniform(0.5, 1.2),
                               ph=rng.uniform(0, 2 * np.pi), rot0=rng.uniform(0, 360),
                               wr=rng.uniform(-50, 50), wt=rng.uniform(0.8, 2.0),
                               blur=blur * unit, alpha=alpha))
    sc['petals'] = petals
    return sc


def _camera_with(sc, t):
    global SCENE
    old = SCENE
    SCENE = sc
    try:
        return camera(t)
    finally:
        SCENE = old


# ---------------------------------------------------------------------------
# Soundtrack: harp arpeggios over a warm pad, chimes on the key reveals
# ---------------------------------------------------------------------------
SR = 44100


def midi_hz(n):
    return 440.0 * 2 ** ((n - 69) / 12.0)


def pluck(freq, dur, seed=0):
    """Karplus-Strong plucked string with a soft, harp-like excitation."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    P = SR / freq                       # loop delay = N + 0.5 (averager) + allpass delay
    N = max(2, int(math.floor(P - 0.6)))
    dly = P - N - 0.5
    c = (1 - dly) / (1 + dly)           # first-order allpass for the fractional part
    noise = lfilter([0.25], [1, -0.75], rng.uniform(-1, 1, N))      # mellow noise
    shape = np.sin(np.pi * np.arange(N) / N)                        # smooth finger pluck
    exc = 0.6 * noise / (np.abs(noise).max() + 1e-9) + 0.4 * shape
    x = np.zeros(n)
    x[:N] = exc - exc.mean()                                        # no DC thump
    g = 0.9985
    a = np.zeros(N + 3)
    a[0], a[1] = 1.0, c
    a[N] -= 0.5 * g * c
    a[N + 1] -= 0.5 * g * (1 + c)
    a[N + 2] -= 0.5 * g
    y = lfilter([1.0, c], a, x)
    y *= np.exp(-np.arange(n) / (SR * dur * 0.55))
    return y / (np.abs(y).max() + 1e-9)


def bell(freq, dur):
    """Soft glassy chime (inharmonic partials)."""
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for ratio, amp, dec in [(1, 1.0, 1.6), (2.76, 0.45, 0.9), (5.4, 0.25, 0.5), (8.93, 0.12, 0.3)]:
        if freq * ratio > SR * 0.42:
            continue
        y += amp * np.sin(2 * np.pi * freq * ratio * t) * np.exp(-t / dec)
    y *= 1 - np.exp(-t / 0.004)
    return y / np.abs(y).max()


def pad_voice(freq, dur):
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for k, det in enumerate((-0.12, 0.0, 0.11)):
        f = freq * 2 ** (det / 12)
        y += np.sin(2 * np.pi * f * t + k) + 0.25 * np.sin(4 * np.pi * f * t + k * 2)
    env = smoothstep(t / 1.2) * smoothstep((dur - t) / 1.4)
    return y * env / 3.0


def reverb_ir(seconds=2.8, seed=3):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, (n, 2)) * np.exp(-t / 0.7)[:, None]
    # darken the tail progressively
    lp = lfilter([0.18], [1, -0.82], ir, axis=0)
    mix = np.clip(t / seconds * 2.5, 0, 1)[:, None]
    ir = ir * (1 - mix) + lp * mix * 2.2
    ir[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))[:, None]
    return ir / np.sqrt((ir ** 2).sum(0))


def make_audio(path, duration=DURATION):
    n = int((duration + 0.5) * SR)
    dry = np.zeros((n, 2))
    rng = np.random.default_rng(5)

    def put(sig, t, gain, pan=0.0):
        i = int(t * SR)
        if i >= n:
            return
        sig = sig[: n - i]
        lg, rg = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
        dry[i:i + len(sig), 0] += sig * gain * lg
        dry[i:i + len(sig), 1] += sig * gain * rg

    # D major: D - Bm - G - A - D - Bm - G/A - D, one bar (8 harp notes) every 4 s
    D_, Bm, G_, A_ = ([50, 57, 62, 66, 69, 74], [47, 54, 59, 62, 66, 71],
                      [43, 50, 55, 59, 62, 67], [45, 52, 57, 61, 64, 69])
    bars = [[D_] * 8, [Bm] * 8, [G_] * 8, [A_] * 8, [D_] * 8, [Bm] * 8, [G_] * 4 + [A_] * 4, [D_] * 8]
    pattern = [0, 2, 3, 4, 5, 4, 3, 2]
    beat = 0.5
    for b, bar in enumerate(bars):
        t0 = 0.6 + b * 4.0
        last = b == len(bars) - 1
        for k in range(8):
            if last and k > 4:
                break
            note = bar[k][pattern[k]]
            put(pluck(midi_hz(note), 2.8, seed=b * 8 + k), t0 + k * beat + rng.normal(0, 0.008),
                0.16 * (1.25 if k in (0, 4) else 1.0), pan=-0.35 + 0.1 * k)
        for half in (0, 1):
            ch = bar[half * 4]
            if half and ch is bar[0]:
                continue
            length = 4.6 if ch is bar[-1] and ch is bar[0] else 2.6
            for note in (ch[0] - 12, ch[1], ch[3]):
                put(pad_voice(midi_hz(note), length + (2.0 if last else 0.0)), t0 + half * 2.0 - 0.3,
                    0.05, pan=rng.uniform(-0.3, 0.3))
    # a gentle music-box melody on top
    melody = [(4.6, 78), (5.6, 74), (6.1, 76), (6.6, 78), (7.6, 71),
              (8.6, 74), (9.6, 71), (10.1, 74), (10.6, 79), (11.6, 78),
              (12.6, 76), (13.6, 73), (14.1, 76), (14.6, 81), (15.6, 79),
              (16.6, 78), (17.6, 81), (18.1, 78), (18.6, 76), (19.6, 74),
              (20.6, 78), (21.6, 74), (22.1, 76), (22.6, 78), (23.6, 83),
              (24.6, 79), (25.1, 78), (25.6, 76), (26.6, 76), (27.1, 73), (27.6, 76),
              (28.6, 74)]
    for tm, note in melody:
        put(bell(midi_hz(note), 3.0 if tm >= 28 else 2.4), tm, 0.10, pan=0.25)
    # shimmering chime runs on the key reveals
    for tm in [0.9, 5.7, 7.95, 8.6, 11.0, 20.0, 21.0, 26.3, 29.6]:
        for k, note in enumerate([86, 90, 93, 98, 102]):
            put(bell(midi_hz(note), 1.6), tm + k * 0.055, 0.035 * (1 - k * 0.12), pan=-0.6 + 0.3 * k)

    wet = np.stack([fftconvolve(dry[:, c], reverb_ir(seed=3 + c)[:, c])[:n] for c in range(2)], 1)
    mix = dry * 0.75 + wet * 0.55
    b, a = butter(2, 45 / (SR / 2), 'highpass')
    mix = lfilter(b, a, mix, axis=0)
    t = np.arange(n) / SR
    mix *= smoothstep(t / 0.6)[:, None]
    mix *= (1 - smoothstep((t - (duration - 2.2)) / 2.2))[:, None]
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.1) * 0.95
    pcm = (mix * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
def _init_worker(card, fmt):
    # Workers are spawned (not forked: OpenCV/BLAS thread pools do not survive
    # a fork), so each one rebuilds the deterministic assets itself.
    global ASSETS, SCENE
    cv2.setNumThreads(1)
    ASSETS = build_assets(card)
    SCENE = make_scene(fmt)


def main():
    global ASSETS, SCENE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--card', default=os.path.join(HERE, 'card.jpg'))
    ap.add_argument('--format', choices=sorted(FORMATS), default='card')
    ap.add_argument('--out', default=None)
    ap.add_argument('--stills', default=None, help='comma separated times (s): write PNGs only')
    ap.add_argument('--no-audio', action='store_true')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    t_start = time.time()
    ASSETS = build_assets(args.card)
    SCENE = make_scene(args.format)
    print(f'assets ready in {time.time() - t_start:.1f}s', flush=True)

    if args.stills:
        for ts in args.stills.split(','):
            fi = int(round(float(ts) * FPS))
            fr = render_frame(fi)
            p = os.path.join(os.getcwd(), f'still_{args.format}_{float(ts):05.2f}.png')
            Image.fromarray(fr).save(p)
            print('wrote', p)
        return

    name = 'walima_invitation' + ('_story' if args.format == 'story' else '')
    out = args.out or os.path.join(HERE, name + ('_no_music' if args.no_audio else '') + '.mp4')
    OW, OH = SCENE['out']
    nframes = int(round(DURATION * FPS))
    audio = None
    if not args.no_audio:
        audio = os.path.splitext(out)[0] + '_audio.wav'
        make_audio(audio)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{OW}x{OH}', '-r', str(FPS), '-i', '-']
    if audio:
        cmd += ['-i', audio, '-c:a', 'aac', '-b:a', '160k', '-shortest']
    cmd += ['-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-profile:v', 'high', '-movflags', '+faststart', out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.jobs, initializer=_init_worker, initargs=(args.card, args.format)) as pool:
        for k, fr in enumerate(pool.imap(render_frame, range(nframes), chunksize=4)):
            enc.stdin.write(fr.tobytes())
            if k % 60 == 0:
                print(f'frame {k}/{nframes}  {time.time() - t_start:.0f}s', flush=True)
    enc.stdin.close()
    enc.wait()
    if audio and os.path.exists(audio):
        os.remove(audio)
    print(f'done: {out}  ({time.time() - t_start:.0f}s)')


if __name__ == '__main__':
    main()
