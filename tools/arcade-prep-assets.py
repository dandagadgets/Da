# Builds arcade-assets/ from the tryarcade.com screenshots used by arcade-promo.html.
#   python3 tools/arcade-prep-assets.py <screenshots dir> <models dir>
# Screenshots: 1.webp (home banner), 2.webp (trending / new games), 4.png (ladder), 5.png (creator home).
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
def cutout(im, name, clean=None):
    m = mask(im)
    if clean: m = clean(m)
    rgba = im.copy(); rgba.putalpha(m)
    rgba = rgba.crop(rgba.getbbox())
    rgba.save(f'{OUT}/{name}.png', optimize=True)
    print('cutout', name, rgba.size)
def save_jpg(im, name, q=90):
    im.save(f'{OUT}/{name}.jpg', quality=q, optimize=True, progressive=True)
    print('image', name, im.size)

# --- game thumbnails (2.webp): two rows of four cards ---
games = load('2.webp')
COLS = [(2, 282), (304, 586), (609, 890), (912, 1192)]
ROWS = [(38, 196), (338, 496)]
NAMES = ['wrongland', 'burger-minute', 'the-heist', 'wanted', 'last-man', 'last-light', 'sunny-sprouts', 'bunker-signal']
for i, name in enumerate(NAMES):
    (x0, x1), (y0, y1) = COLS[i % 4], ROWS[i // 4]
    save_jpg(up4(games.crop((x0 + 1, y0 + 1, x1 - 1, y1 - 1))), 'game-' + name)

# --- home banner (1.webp): purple backdrop, character groups and the clay podium ---
banner = load('1.webp')
save_jpg(up4(banner.crop((137, 73, 1327, 408))), 'banner', 88)
def keep_largest(n):   # drop stray specks the matte picks up from the backdrop
    def f(m):
        a = (np.asarray(m) > 24).astype(np.uint8)
        cnt, lab, stats, _ = cv2.connectedComponentsWithStats(a, 8)
        big = np.argsort(-stats[1:, cv2.CC_STAT_AREA])[:n] + 1
        keep = np.isin(lab, big)
        return Image.fromarray(np.where(keep, np.asarray(m), 0).astype(np.uint8))
    return f
cutout(up4(banner.crop((222, 112, 470, 402))), 'crew-left', keep_largest(6))
right = banner.crop((935, 135, 1318, 398))
right.paste(right.getpixel((8, 40)), (0, 0, 70, 32))   # paint out the tail of the banner's tagline
cutout(up4(right), 'crew-right', keep_largest(8))
cutout(up4(banner.crop((470, 200, 950, 380))), 'podium-3d', keep_largest(40))

# --- creator home (5.png): Morph over the galaxy, and the waving Discord buddy ---
home = load('5.png')
galaxy = up4(home.crop((22, 176, 680, 366)))
save_jpg(galaxy, 'galaxy', 88)
cutout(up4(home.crop((198, 280, 306, 368))), 'morph', keep_largest(1))   # head only: the card's headline covers the rest
cutout(up4(home.crop((874, 364, 1028, 521))), 'buddy', keep_largest(2))

# --- logo: trace the white mark into a clean vector ---
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
