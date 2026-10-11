# Builds arcade-assets/ from the tryarcade.com screenshots used by arcade-promo.html.
#   python3 tools/arcade-prep-assets.py <screenshots dir> <models dir>
# Screenshots: 1.webp (home banner), 2.webp (trending / new games), 5.png (creator home),
# 6.png (Your Mold editor), 7.webp (Spooky Season banner). Outputs that already exist are kept.
# Models: EDSR_x4.pb (github.com/Saafke/EDSR_Tensorflow) for 4x upscaling and
# isnet-general-use.onnx (github.com/danielgatis/rembg releases) for cut-outs.
# Needs opencv-contrib-python-headless, onnxruntime and potracer.
import sys, os, numpy as np, cv2, onnxruntime as ort, potrace
from PIL import Image, ImageFilter

SRC, MODELS = sys.argv[1], sys.argv[2]
OUT = 'arcade-assets'
os.makedirs(OUT, exist_ok=True)
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(os.path.join(MODELS, 'EDSR_x4.pb')); sr.setModel('edsr', 4)
isnet = ort.InferenceSession(os.path.join(MODELS, 'isnet-general-use.onnx'), providers=['CPUExecutionProvider'])

def load(name): return Image.open(os.path.join(SRC, name)).convert('RGB')
def up4(im):
    a = sr.upsample(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    return Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
def mask(im):
    x = np.asarray(im.resize((1024, 1024), Image.LANCZOS)).astype(np.float32)
    x = x / x.max() - np.array([.485, .456, .406], np.float32)
    p = isnet.run(None, {isnet.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    p = (p - p.min()) / (p.max() - p.min())
    return Image.fromarray((p * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS)
def done(name):
    return any(os.path.exists(f'{OUT}/{name}.{e}') for e in ('png', 'jpg'))
def cutout(crop, name, clean=None):   # 4x upscale, then matte
    if done(name): return
    im = up4(crop); m = mask(im)
    if clean: m = clean(m)
    rgba = im.copy(); rgba.putalpha(m)
    rgba = rgba.crop(rgba.getbbox())
    rgba.save(f'{OUT}/{name}.png', optimize=True)
    print('cutout', name, rgba.size)
def save_jpg(crop, name, q=90):   # 4x upscale
    if done(name): return
    im = up4(crop)
    im.save(f'{OUT}/{name}.jpg', quality=q, optimize=True, progressive=True)
    print('image', name, im.size)

# --- game thumbnails (2.webp): two rows of four cards ---
games = load('2.webp')
COLS = [(2, 282), (304, 586), (609, 890), (912, 1192)]
ROWS = [(38, 196), (338, 496)]
NAMES = ['wrongland', 'burger-minute', 'the-heist', 'wanted', 'last-man', 'last-light', 'sunny-sprouts', 'bunker-signal']
for i, name in enumerate(NAMES):
    (x0, x1), (y0, y1) = COLS[i % 4], ROWS[i // 4]
    save_jpg(games.crop((x0 + 1, y0 + 1, x1 - 1, y1 - 1)), 'game-' + name)

# --- home banner (1.webp): purple backdrop, character groups and the clay podium ---
banner = load('1.webp')
save_jpg(banner.crop((137, 73, 1327, 408)), 'banner', 88)
def keep_largest(n):   # drop stray specks the matte picks up from the backdrop
    def f(m):
        a = (np.asarray(m) > 24).astype(np.uint8)
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(a, 8)
        big = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:n] + 1
        keep = np.isin(lab, big)
        return Image.fromarray(np.where(keep, np.asarray(m), 0).astype(np.uint8))
    return f
cutout(banner.crop((222, 112, 470, 402)), 'crew-left', keep_largest(6))
right = banner.crop((935, 135, 1318, 398))
right.paste(right.getpixel((8, 40)), (0, 0, 70, 32))   # paint out the tail of the banner's tagline
cutout(right, 'crew-right', keep_largest(8))
cutout(banner.crop((470, 200, 950, 380)), 'podium-3d', keep_largest(40))

# --- creator home (5.png): Morph over the galaxy, and the waving Discord buddy ---
home = load('5.png')
save_jpg(home.crop((22, 176, 680, 366)), 'galaxy', 88)
cutout(home.crop((198, 280, 306, 368)), 'morph', keep_largest(1))   # head only: the card's headline covers the rest
cutout(home.crop((874, 364, 1028, 521)), 'buddy', keep_largest(2))

# --- Your Mold editor (6.png): the preview figure and three texture swatches ---
mold = load('6.png')
cutout(mold.crop((212, 240, 348, 468)), 'mold-preview', keep_largest(2))
for name, x in (('clay', 712), ('bone', 891), ('amethyst', 1071)):
    cutout(mold.crop((x, 298, x + 90, 452)), 'mold-' + name, keep_largest(1))

# --- Spooky Season banner (7.webp): slime title, the graveyard scene, scared clay figure, peeking bunny ---
spooky = load('7.webp')
def key_title(crop, name):
    # the matte model drops slime lettering, so key it by colour instead: slime green and bone white,
    # plus the dark ink (outlines, shadows, prize text) touching them; the night sky behind is blue
    if done(name): return
    a = np.asarray(crop).astype(int); R, G, B = a[..., 0], a[..., 1], a[..., 2]
    core = (((G > R + 35) & (G > B + 35) & (G > 110)) | ((R > 175) & (G > 170) & (B > 140) & (np.abs(R - G) < 40))).astype(np.uint8)
    ink = (((.3 * R + .59 * G + .11 * B) < 75) & (B - np.maximum(R, G) < 22)).astype(np.uint8)
    m = core | (ink & cv2.dilate(core, np.ones((9, 9), np.uint8)))
    cnt, lab = cv2.connectedComponents((1 - m).astype(np.uint8), 4)
    for i in range(1, cnt):   # enclosed ink-coloured holes are the prize text, not the sky
        reg = lab == i
        if not (reg[0].any() or reg[-1].any() or reg[:, 0].any() or reg[:, -1].any()) and ink[reg].mean() > .5: m[reg] = 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    m = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 60]).astype(np.float32)
    alpha = Image.fromarray((cv2.GaussianBlur(m, (3, 3), 0) * 255).astype(np.uint8))
    rgba = up4(crop); rgba.putalpha(alpha.resize(rgba.size, Image.BICUBIC))
    rgba.crop(rgba.getbbox()).save(f'{OUT}/{name}.png', optimize=True)
    print('keyed', name, rgba.size)
key_title(spooky.crop((25, 14, 620, 462)), 'spooky-title')
save_jpg(spooky.crop((610, 0, 1159, 467)), 'spooky-scene', 88)
cutout(spooky.crop((728, 78, 985, 467)), 'spooky-scared', keep_largest(1))
cutout(spooky.crop((982, 305, 1072, 455)), 'spooky-bunny', keep_largest(1))

# --- logo: trace the white mark into a clean vector ---
if not os.path.exists(f'{OUT}/logo.svg'):
  logo = up4(banner.crop((24, 4, 92, 56))).convert('L').filter(ImageFilter.GaussianBlur(1.2))
  bits = np.asarray(logo) > 128   # potracer traces the False pixels
  path = potrace.Bitmap(~bits).trace(turdsize=40, alphamax=1.1, opttolerance=.3)
  H, W = bits.shape
  d = []
  for c in path:
      s = c.start_point; d.append(f'M{s.x:.1f} {s.y:.1f}')
      for seg in c.segments:
          if seg.is_corner: d.append(f'L{seg.c.x:.1f} {seg.c.y:.1f}L{seg.end_point.x:.1f} {seg.end_point.y:.1f}')
          else: d.append(f'C{seg.c1.x:.1f} {seg.c1.y:.1f} {seg.c2.x:.1f} {seg.c2.y:.1f} {seg.end_point.x:.1f} {seg.end_point.y:.1f}')
      d.append('Z')
  with open(f'{OUT}/logo.svg', 'w') as f:
      f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><path fill="#fff" fill-rule="evenodd" d="{"".join(d)}"/></svg>\n')
  print('logo', W, H)
