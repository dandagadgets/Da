#!/usr/bin/env python3
"""
ZYLAN "Rooh" - a real 3D product film, rendered with Blender (Cycles).

The bottle is modelled (bevelled glass shell, thick glass base, blue liquid
with volume absorption, glossy black cap with a gold ring, the paper-and-gold
label rebuilt crisp from the poster). It stands on a wet stone plinth in dark
reflective water, lit by cyan rim lights, with water droplets floating around
it. One continuous camera move:

  0.0-2.6  macro on the cap, gliding down the glass
  2.6-5.0  orbiting round the side to a low angle over the water
  5.0-7.0  swinging to the front while a light sweeps across the glass
  6.6-8.2  3D gold "ROOH" rises out of the water, "ZYLAN" descends
  8.2-10   hero hold, tagline, light sweep

Usage:
    python3 make_3d.py --test 1.0                 # one preview frame (half size)
    python3 make_3d.py --frames 0-299             # render frames to build/frames
    python3 make_3d.py --assemble                 # music + tagline -> zylan_rooh_3d.mp4
Requires: pip install bpy==5.2.2 (Blender as a Python module), numpy, scipy,
Pillow, ffmpeg. Music helpers come from ../walima-invitation/make_video.py.
"""
import argparse
import math
import os
import subprocess
import sys
import time
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONTS = os.path.join(HERE, 'fonts')
BUILD = os.path.join(HERE, 'build')
FRAMES = os.path.join(BUILD, 'frames')
FPS = 24
DURATION = 10.0
NFRAMES = int(DURATION * FPS)
RES = (1080, 1920)

# bottle dimensions (Blender units; 1.0 = bottle width)
BW, BD, BH = 1.0, 0.62, 1.28          # body width, depth, height
WALL, BASE = 0.045, 0.20              # glass wall and thick base
CAP_R, CAP_H = 0.23, 0.64
LABEL_W, LABEL_H, LABEL_Z = 0.61, 0.617, 0.70
WATER_Z = -0.10

# camera keys: t, orbit angle (deg, 0 = front), distance, height, target z, lens mm, f-stop
CAM = [
    (0.0, -72, 2.05, 2.15, 1.62, 70, 2.4),
    (2.4, -48, 2.15, 1.45, 1.20, 65, 2.6),
    (5.0, 34, 2.75, 0.42, 0.78, 58, 3.2),
    (7.0, 0, 5.20, 1.45, 1.60, 50, 5.6),
    (10.0, 0, 4.75, 1.45, 1.62, 50, 5.6),
]
SWEEPS = [(3.0, 1.5), (8.4, 1.3)]     # light strip passing in front: start, duration
ROOH_RISE = (6.6, 1.2)
ZYLAN_DROP = (7.5, 0.9)


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 3


# ---------------------------------------------------------------------------
# label texture (rebuilt from the poster layout, crisp at any distance)
# ---------------------------------------------------------------------------
def fit_text(draw_w, text, font_path, cap_px, width_px):
    """Font size giving the wanted cap height, and tracking giving the width."""
    f = ImageFont.truetype(font_path, 100)
    cap = f.getbbox('H')[3] - f.getbbox('H')[1]
    f = ImageFont.truetype(font_path, max(8, int(round(100 * cap_px / cap))))
    natural = sum(f.getlength(c) for c in text)
    track = (width_px - natural) / max(1, len(text) - 1)
    return f, track


def draw_tracked(d, text, f, track, cx, cy, fill):
    widths = [f.getlength(c) for c in text]
    total = sum(widths) + track * (len(text) - 1)
    x = cx - total / 2
    top = f.getbbox('H')[1]
    cap = f.getbbox('H')[3] - top
    for c, w in zip(text, widths):
        d.text((x, cy - cap / 2 - top), c, font=f, fill=fill)
        x += w + track


def make_label(path_color, path_metal, W=1200, H=1215):
    rng = np.random.default_rng(5)

    def noise(scale, sx=1.0, sy=1.0):
        n = rng.normal(0, 1, (H // scale + 2, W // scale + 2)).astype(np.float32)
        img = Image.fromarray(((n - n.min()) / (np.ptp(n) + 1e-6) * 255).astype(np.uint8))
        img = img.resize((int(W * sx), int(H * sy)), Image.BICUBIC).resize((W, H), Image.BICUBIC)
        a = np.asarray(img).astype(np.float32) / 255
        return (a - a.mean()) / (a.std() + 1e-6)

    foil = 0.62 + 0.10 * noise(40) + 0.07 * noise(12, 1.0, 0.15) + 0.03 * noise(3)
    foil = np.clip(foil, 0, 1)[..., None]
    gold = np.float32([150, 104, 38]) * (1 - foil) + np.float32([246, 210, 132]) * foil
    img = gold.copy()
    x0, x1, y0, y1 = int(0.075 * W), int(0.925 * W), int(0.079 * H), int(0.921 * H)
    paper = np.float32([226, 223, 214]) * (1 + 0.012 * noise(2))[..., None]
    img[y0:y1, x0:x1] = paper[y0:y1, x0:x1]
    metal = np.ones((H, W), np.float32)
    metal[y0:y1, x0:x1] = 0
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    mm = Image.fromarray((metal * 255).astype(np.uint8))
    d, dm = ImageDraw.Draw(im), ImageDraw.Draw(mm)
    ink = (22, 20, 18)
    fz, tz = fit_text(W, 'ZYLAN', os.path.join(FONTS, 'Cinzel[wght].ttf'), 0.105 * H, 0.70 * W)
    draw_tracked(d, 'ZYLAN', fz, tz, W / 2, 0.206 * H, (150, 104, 34))
    draw_tracked(dm, 'ZYLAN', fz, tz, W / 2, 0.206 * H, 255)
    yl = 0.339 * H
    for a, b in ((0.255, 0.475), (0.525, 0.742)):
        d.line([(a * W, yl), (b * W, yl)], fill=ink, width=5)
    s = 0.016 * W
    d.polygon([(W / 2, yl - s), (W / 2 + s, yl), (W / 2, yl + s), (W / 2 - s, yl)], fill=ink)
    f, tr = fit_text(W, 'WHERE MYSTERY BECOMES SCENT', os.path.join(FONTS, 'Montserrat-ExtraBold.ttf'),
                     0.024 * H, 0.70 * W)
    draw_tracked(d, 'WHERE MYSTERY BECOMES SCENT', f, tr, W / 2, 0.386 * H, ink)
    f, tr = fit_text(W, 'ROOH', os.path.join(FONTS, 'Montserrat-ExtraBold.ttf'), 0.075 * H, 0.33 * W)
    draw_tracked(d, 'ROOH', f, tr, W / 2, 0.572 * H, ink)
    yo = 0.705 * H                                     # small ornament under ROOH
    d.line([(0.35 * W, yo), (0.465 * W, yo)], fill=ink, width=4)
    d.line([(0.535 * W, yo), (0.65 * W, yo)], fill=ink, width=4)
    s = 0.013 * W
    d.polygon([(W / 2, yo - s), (W / 2 + s, yo), (W / 2, yo + s), (W / 2 - s, yo)], fill=ink)
    for sx in (-1, 1):
        d.arc([W / 2 + sx * 0.035 * W - 14, yo - 14, W / 2 + sx * 0.035 * W + 14, yo + 14],
              0 if sx > 0 else 180, 180 if sx > 0 else 360, fill=ink, width=4)
    f, tr = fit_text(W, 'EXTRAIT DE PARFUM', os.path.join(FONTS, 'Montserrat-ExtraBold.ttf'), 0.03 * H, 0.50 * W)
    draw_tracked(d, 'EXTRAIT DE PARFUM', f, tr, W / 2, 0.847 * H, ink)
    im = im.filter(ImageFilter.GaussianBlur(0.6))
    im.save(path_color)
    mm.filter(ImageFilter.GaussianBlur(0.8)).save(path_metal)


def make_backdrop(path, W=1600, H=1000):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W - 0.5, yy / H
    glow = np.exp(-((u / 0.16) ** 2 + ((v - 0.66) / 0.26) ** 2))
    shafts = np.zeros_like(u)
    rng = np.random.default_rng(3)
    for _ in range(10):
        c, w = rng.uniform(-0.45, 0.45), rng.uniform(0.006, 0.03)
        shafts += np.exp(-0.5 * ((u - c - 0.2 * (v - 0.5)) / w) ** 2) * rng.uniform(0.15, 0.5)
    img = (np.float32([0.004, 0.02, 0.04]) + glow[..., None] * np.float32([0.05, 0.42, 0.62])
           + (shafts * (1 - v))[..., None] * np.float32([0.02, 0.12, 0.18]))
    # stored as a plain 16-bit-ish PNG in linear-ish space (emission strength scales it)
    Image.fromarray(np.clip(img / img.max() * 255, 0, 255).astype(np.uint8)).save(path)


# ---------------------------------------------------------------------------
# Blender scene
# ---------------------------------------------------------------------------
def build_scene(width, height, samples):
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    cy = scn.cycles
    cy.device = 'CPU'
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.03
    cy.use_denoising = True
    cy.max_bounces = 8
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 4
    cy.transmission_bounces = 8
    cy.transparent_max_bounces = 8
    cy.volume_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 1.0
    scn.render.resolution_x, scn.render.resolution_y = width, height
    scn.render.resolution_percentage = 100
    scn.render.fps = FPS
    scn.render.use_persistent_data = True
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGB'
    scn.view_settings.view_transform = 'AgX'
    for look in ('AgX - Medium High Contrast', 'Medium High Contrast', 'AgX - Punchy', 'Punchy'):
        try:
            scn.view_settings.look = look
            break
        except TypeError:
            continue

    os.makedirs(BUILD, exist_ok=True)
    lc, lm, bd = (os.path.join(BUILD, n) for n in ('label_color.png', 'label_metal.png', 'backdrop.png'))
    make_label(lc, lm)
    make_backdrop(bd)

    def mat(name):
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        return m, m.node_tree.nodes, m.node_tree.links, m.node_tree.nodes['Principled BSDF']

    def setp(bsdf, **kw):
        names = {'base': 'Base Color', 'metal': 'Metallic', 'rough': 'Roughness', 'ior': 'IOR',
                 'trans': 'Transmission Weight', 'coat': 'Coat Weight', 'coat_rough': 'Coat Roughness',
                 'spec': 'Specular IOR Level'}
        for k, v in kw.items():
            bsdf.inputs[names[k]].default_value = (*v, 1.0) if k == 'base' else v

    world = bpy.data.worlds.new('World')
    scn.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.002, 0.005, 0.01, 1)

    def smooth_obj(o):
        for p in o.data.polygons:
            p.use_smooth = True

    def bevel(o, width, segs):
        b = o.modifiers.new('bevel', 'BEVEL')
        b.width, b.segments, b.limit_method = width, segs, 'NONE'
        b.harden_normals = True
        smooth_obj(o)

    # glass shell
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, BH / 2))
    glass = bpy.context.object
    glass.scale = (BW, BD, BH)
    bpy.ops.object.transform_apply(scale=True)
    bevel(glass, 0.085, 8)
    sol = glass.modifiers.new('wall', 'SOLIDIFY')
    sol.thickness, sol.offset, sol.use_even_offset = WALL, -1, True
    mg, _, _, b = mat('glass')
    setp(b, base=(0.96, 0.985, 1.0), rough=0.0, ior=1.5, trans=1.0)
    glass.data.materials.append(mg)
    # thick glass base inside the shell
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, WALL + (BASE - WALL) / 2))
    base = bpy.context.object
    base.scale = (BW - 2 * WALL - 0.002, BD - 2 * WALL - 0.002, BASE - WALL)
    bpy.ops.object.transform_apply(scale=True)
    bevel(base, 0.04, 4)
    base.data.materials.append(mg)
    # liquid
    lh = BH - WALL - BASE - 0.004
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, BASE + 0.002 + lh / 2))
    liq = bpy.context.object
    liq.scale = (BW - 2 * WALL - 0.004, BD - 2 * WALL - 0.004, lh)
    bpy.ops.object.transform_apply(scale=True)
    bevel(liq, 0.045, 6)
    ml, _, _, b = mat('liquid')
    setp(b, base=(0.06, 0.66, 0.88), rough=0.0, ior=1.33, trans=1.0)
    b.inputs['Emission Color'].default_value = (0.05, 0.62, 0.85, 1)      # faint inner glow
    b.inputs['Emission Strength'].default_value = 0.12
    liq.data.materials.append(ml)

    # cap + gold ring
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=CAP_R, depth=CAP_H,
                                        location=(0, 0, BH + CAP_H / 2 - 0.005))
    cap = bpy.context.object
    bevel(cap, 0.02, 5)
    mc, _, _, b = mat('cap')
    setp(b, base=(0.006, 0.006, 0.008), rough=0.1, coat=1.0, coat_rough=0.02)
    cap.data.materials.append(mc)
    bpy.ops.mesh.primitive_cylinder_add(vertices=128, radius=CAP_R + 0.006, depth=0.035,
                                        location=(0, 0, BH + 0.012))
    ring = bpy.context.object
    bevel(ring, 0.008, 3)
    mgold, _, _, b = mat('gold')
    setp(b, base=(1.0, 0.68, 0.26), metal=1.0, rough=0.18)
    ring.data.materials.append(mgold)

    # label
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, -BD / 2 - 0.0015, LABEL_Z), rotation=(math.pi / 2, 0, 0))
    label = bpy.context.object
    label.scale = (LABEL_W, LABEL_H, 1)
    mlb, nodes, links, b = mat('label')
    tc = nodes.new('ShaderNodeTexImage')
    tc.image = bpy.data.images.load(lc)
    tm = nodes.new('ShaderNodeTexImage')
    tm.image = bpy.data.images.load(lm)
    tm.image.colorspace_settings.name = 'Non-Color'
    links.new(tc.outputs['Color'], b.inputs['Base Color'])
    mr = nodes.new('ShaderNodeMapRange')
    mr.inputs['To Min'].default_value, mr.inputs['To Max'].default_value = 0.8, 0.3
    b.inputs['Specular IOR Level'].default_value = 0.0      # matte paper; the gold is metallic
    mm_ = nodes.new('ShaderNodeMath')
    mm_.operation, mm_.inputs[1].default_value = 'MULTIPLY', 0.8
    links.new(tm.outputs['Color'], mm_.inputs[0])
    links.new(mm_.outputs[0], b.inputs['Metallic'])
    links.new(tm.outputs['Color'], mr.inputs['Value'])
    links.new(mr.outputs['Result'], b.inputs['Roughness'])
    label.data.materials.append(mlb)

    # wet stone plinth
    bpy.ops.mesh.primitive_cylinder_add(vertices=160, radius=1.15, depth=0.6, location=(0, 0, -0.3))
    plinth = bpy.context.object
    bevel(plinth, 0.04, 4)
    ms, nodes, links, b = mat('stone')
    setp(b, base=(0.010, 0.011, 0.013), spec=0.2)
    nz = nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value, nz.inputs['Detail'].default_value = 7.0, 8.0
    rr = nodes.new('ShaderNodeMapRange')
    rr.inputs['To Min'].default_value, rr.inputs['To Max'].default_value = 0.22, 0.6
    links.new(nz.outputs['Fac'], rr.inputs['Value'])
    links.new(rr.outputs['Result'], b.inputs['Roughness'])
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = 0.35
    links.new(nz.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], b.inputs['Normal'])
    plinth.data.materials.append(ms)

    # dark reflective water
    bpy.ops.mesh.primitive_plane_add(size=80, location=(0, 0, WATER_Z))
    water = bpy.context.object
    mw, nodes, links, b = mat('water')
    setp(b, base=(0.002, 0.008, 0.014), rough=0.03, ior=1.33, spec=0.6)
    tco = nodes.new('ShaderNodeTexCoord')
    wn = nodes.new('ShaderNodeTexNoise')
    wn.noise_dimensions = '4D'
    wn.inputs['Scale'].default_value, wn.inputs['Detail'].default_value = 2.2, 3.0
    links.new(tco.outputs['Object'], wn.inputs['Vector'])
    wb = nodes.new('ShaderNodeBump')
    wb.inputs['Strength'].default_value = 0.22
    wb.inputs['Distance'].default_value = 0.03
    links.new(wn.outputs['Fac'], wb.inputs['Height'])
    links.new(wb.outputs['Normal'], b.inputs['Normal'])
    water.data.materials.append(mw)

    # glowing backdrop far behind
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 9, 3.0), rotation=(math.pi / 2, 0, 0))
    back = bpy.context.object
    back.scale = (26, 16, 1)
    mb = bpy.data.materials.new('backdrop')
    mb.use_nodes = True
    nt = mb.node_tree
    nt.nodes.remove(nt.nodes['Principled BSDF'])
    em = nt.nodes.new('ShaderNodeEmission')
    ti = nt.nodes.new('ShaderNodeTexImage')
    ti.image = bpy.data.images.load(bd)
    nt.links.new(ti.outputs['Color'], em.inputs['Color'])
    em.inputs['Strength'].default_value = 0.5
    nt.links.new(em.outputs[0], nt.nodes['Material Output'].inputs['Surface'])
    back.data.materials.append(mb)

    # lights
    aim = bpy.data.objects.new('aim', None)
    aim.location = (0, 0, 0.9)
    scn.collection.objects.link(aim)

    def area(name, loc, size, size_y, energy, color, target=aim):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.shape = 'RECTANGLE'
        ld.size, ld.size_y = size, size_y
        ld.energy, ld.color = energy, color
        o = bpy.data.objects.new(name, ld)
        o.location = loc
        o.visible_camera = False                       # lights never show up as panels
        scn.collection.objects.link(o)
        c = o.constraints.new('TRACK_TO')
        c.target, c.track_axis, c.up_axis = target, 'TRACK_NEGATIVE_Z', 'UP_Y'
        return o

    area('key', (-2.6, -3.0, 3.6), 2.2, 2.2, 420, (1.0, 0.95, 0.9))
    area('fill', (0.8, -4.5, 1.6), 1.2, 1.6, 70, (1.0, 0.97, 0.92))
    area('back', (0.0, 2.6, 0.9), 1.6, 1.6, 220, (0.25, 0.85, 1.0))     # backlight through the liquid
    area('rimL', (-2.4, 2.2, 1.4), 0.35, 3.2, 650, (0.35, 0.8, 1.0))
    area('rimR', (2.4, 2.2, 1.4), 0.35, 3.2, 650, (0.35, 0.8, 1.0))
    top_aim = bpy.data.objects.new('top_aim', None)
    top_aim.location = (0, 0, 1.9)
    scn.collection.objects.link(top_aim)
    area('top', (0.4, -0.6, 3.6), 0.8, 0.8, 90, (1.0, 1.0, 1.0), top_aim)
    sweep = area('sweep', (-9, -2.2, 1.2), 0.22, 4.0, 550, (1.0, 1.0, 1.0))
    sweep.constraints.clear()
    sweep.rotation_euler = (math.pi / 2, 0, 0)
    sweep.visible_diffuse = False                      # only a glint on glass and gold

    # floating droplets
    rng = np.random.default_rng(17)
    mdrop, _, _, b = mat('droplet')
    setp(b, base=(0.9, 0.97, 1.0), rough=0.0, ior=1.33, trans=1.0)
    drops = []
    for i in range(28):
        r = float(rng.uniform(0.012, 0.05))
        a = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(0.85, 3.0)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=r)
        o = bpy.context.object
        smooth_obj(o)
        o.data.materials.append(mdrop)
        drops.append(dict(obj=o, x=rad * math.cos(a), y=rad * math.sin(a), z=rng.uniform(0.05, 3.1),
                          v=rng.uniform(0.03, 0.09), ph=rng.uniform(0, 6.3), sw=rng.uniform(0.02, 0.06)))

    # 3D gold type
    def text3d(body, font_file, size, extrude, bevel_d):
        bpy.ops.object.text_add(location=(0, 0, 0), rotation=(math.pi / 2, 0, 0))
        o = bpy.context.object
        o.data.body = body
        o.data.font = bpy.data.fonts.load(os.path.join(FONTS, font_file))
        o.data.size, o.data.extrude = size, extrude
        o.data.bevel_depth, o.data.bevel_resolution = bevel_d, 3
        o.data.align_x, o.data.align_y = 'CENTER', 'CENTER'
        o.data.space_character = 1.12
        o.data.materials.append(mgold)
        return o

    rooh = text3d('ROOH', 'Cinzel-Bold.ttf', 0.62, 0.05, 0.008)
    zylan = text3d('ZYLAN', 'Cinzel-Bold.ttf', 0.24, 0.025, 0.004)
    zylan.data.space_character = 1.5

    cam_d = bpy.data.cameras.new('cam')
    cam = bpy.data.objects.new('cam', cam_d)
    scn.collection.objects.link(cam)
    scn.camera = cam
    cam_d.sensor_fit = 'AUTO'
    cam_d.dof.use_dof = True
    target = bpy.data.objects.new('target', None)
    scn.collection.objects.link(target)
    c = cam.constraints.new('TRACK_TO')
    c.target, c.track_axis, c.up_axis = target, 'TRACK_NEGATIVE_Z', 'UP_Y'

    return dict(bpy=bpy, scn=scn, cam=cam, target=target, sweep=sweep, drops=drops, rooh=rooh,
                zylan=zylan, wnoise=wn)


def cam_params(t):
    from scipy.interpolate import PchipInterpolator
    ts = [k[0] for k in CAM]
    vals = [PchipInterpolator(ts, [k[i] for k in CAM])(t) for i in range(1, 7)]
    return [float(v) for v in vals]


def set_time(S, t):
    ang, dist, hgt, tz, lens, fstop = cam_params(t)
    a = math.radians(ang)
    S['cam'].location = (dist * math.sin(a), -dist * math.cos(a), hgt)
    S['target'].location = (0, 0, tz)
    cd = S['cam'].data
    cd.lens = lens
    cd.dof.aperture_fstop = fstop
    cd.dof.focus_distance = max(0.3, math.sqrt(dist ** 2 + (hgt - tz) ** 2) - 0.3)
    S['wnoise'].inputs['W'].default_value = t * 0.35
    x = -9.0
    for s0, d in SWEEPS:
        if s0 <= t < s0 + d:
            x = -3.2 + 6.4 * smooth((t - s0) / d)
    S['sweep'].location = (x, -2.2, 1.2)
    for dr in S['drops']:
        z = dr['z'] + dr['v'] * t
        dr['obj'].location = (dr['x'] + dr['sw'] * math.sin(t * 0.8 + dr['ph']),
                              dr['y'] + dr['sw'] * math.cos(t * 0.7 + dr['ph']), (z % 3.3) - 0.05)
    u = ease_out((t - ROOH_RISE[0]) / ROOH_RISE[1])
    S['rooh'].location = (0, 0.95, -1.2 + (2.62 + 1.2) * u)
    v = ease_out((t - ZYLAN_DROP[0]) / ZYLAN_DROP[1])
    S['zylan'].location = (0, 1.05, 4.6 - (4.6 - 3.22) * v)


def render(frames, width, height, samples, outdir, name='f_{:04d}.png'):
    S = build_scene(width, height, samples)
    bpy = S['bpy']
    os.makedirs(outdir, exist_ok=True)
    for f in frames:
        p = os.path.join(outdir, name.format(f))
        if os.path.exists(p) and name.startswith('f_'):
            continue
        t0 = time.time()
        set_time(S, f / FPS)
        S['scn'].render.filepath = p
        bpy.ops.render.render(write_still=True)
        print(f'frame {f} {time.time() - t0:.1f}s', flush=True)


# ---------------------------------------------------------------------------
# music + tagline overlay
# ---------------------------------------------------------------------------
def make_audio(path):
    from scipy.signal import butter, fftconvolve, lfilter
    sys.path.insert(0, os.path.join(HERE, '..', 'walima-invitation'))
    from make_video import SR, bell, midi_hz, pad_voice, pluck, reverb_ir, whoosh

    n = int((DURATION + 0.6) * SR)
    dry = np.zeros((n, 2))
    rng = np.random.default_rng(6)

    def put(sig, t, gain, pan=0.0):
        i = int(t * SR)
        sig = sig[:n - i]
        dry[i:i + len(sig), 0] += sig * gain * math.cos((pan + 1) * math.pi / 4)
        dry[i:i + len(sig), 1] += sig * gain * math.sin((pan + 1) * math.pi / 4)

    def sweep_tone(f0, f1, dur, decay):
        tt = np.arange(int(dur * SR)) / SR
        f = f0 * (f1 / f0) ** (tt / dur)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / decay) * (1 - np.exp(-tt / 0.003))

    for note, g in ((38, 0.09), (45, 0.07), (50, 0.06), (57, 0.04)):         # D drone
        put(pad_voice(midi_hz(note), 7.2), 0.0, g, pan=rng.uniform(-0.4, 0.4))
    for note, g in ((41, 0.08), (48, 0.06), (53, 0.05), (60, 0.04), (65, 0.03)):  # resolves at the reveal
        put(pad_voice(midi_hz(note), 4.2), 6.6, g, pan=rng.uniform(-0.4, 0.4))
    put(sweep_tone(90, 38, 2.2, 0.6), 0.0, 0.5)
    for k in range(13):                                                      # slow glassy arpeggio
        tk = 0.6 + k * 0.48
        put(pluck(midi_hz([62, 69, 74, 77, 81][k % 5]), 2.2, seed=k), tk, 0.08, pan=-0.4 + 0.065 * k)
    for s0, d in SWEEPS:
        put(whoosh(d, seed=int(s0 * 10)), s0, 0.07, pan=0.3)
        for i, note in enumerate([93, 98, 101, 105]):
            put(bell(midi_hz(note), 1.4), s0 + d * 0.45 + i * 0.05, 0.03, pan=-0.3 + 0.2 * i)
    m = int(1.2 * SR)                                                        # riser into the reveal
    tt = np.arange(m) / SR
    r = lfilter(*butter(2, [400 / (SR / 2), 7000 / (SR / 2)], 'band'), rng.normal(0, 1, m)) * (tt / tt[-1]) ** 3
    put(r / np.abs(r).max(), ROOH_RISE[0] - 1.1, 0.14)
    put(sweep_tone(120, 36, 2.0, 0.55), 7.0, 0.6)                            # hit as the type lands
    sp = lfilter(*butter(2, [500 / (SR / 2), 8000 / (SR / 2)], 'band'), rng.normal(0, 1, int(1.0 * SR)))
    put(sp * np.exp(-np.arange(len(sp)) / SR / 0.2) / np.abs(sp).max(), ROOH_RISE[0] + 0.15, 0.12)
    for i, note in enumerate([86, 90, 93, 98, 102]):
        put(bell(midi_hz(note), 2.4), 7.05 + i * 0.07, 0.05, pan=-0.5 + 0.25 * i)
    for tm, note in ((7.6, 74), (8.1, 78), (8.6, 81), (9.2, 86)):
        put(bell(midi_hz(note), 3.0), tm, 0.09, pan=0.2)
    wet = np.stack([fftconvolve(dry[:, c], reverb_ir(seconds=2.6, seed=41 + c)[:, c])[:n] for c in range(2)], 1)
    mix = dry * 0.7 + wet * 0.6
    mix = lfilter(*butter(2, 28 / (SR / 2), 'highpass'), mix, axis=0)
    tt = np.arange(n) / SR
    mix *= (np.clip(tt / 0.05, 0, 1) * (1 - np.clip((tt - (DURATION - 0.7)) / 0.7, 0, 1)))[:, None]
    mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.2) * 0.95
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((mix * 32767).astype(np.int16).tobytes())


def overlay(fi, img):
    """2D tagline on the water, and fade in/out."""
    t = fi / FPS
    a = np.asarray(img).astype(np.float32)
    k = min(1.0, max(0.0, (t - 8.4) / 0.6))
    if k > 0:
        f = ImageFont.truetype(os.path.join(FONTS, 'Montserrat-SemiBold.ttf'), 36)
        txt = 'LUXURY  IN  EVERY  DROP'
        layer = Image.new('L', (RES[0], 120), 0)
        draw_tracked(ImageDraw.Draw(layer), txt, f, 9, RES[0] / 2, 60, 255)
        m = np.asarray(layer).astype(np.float32)[..., None] / 255 * k
        y0 = 1745
        a[y0:y0 + 120] = a[y0:y0 + 120] * (1 - m) + np.float32([236, 226, 205]) * m
    a *= min(1.0, t / 0.4) * (1 - min(1.0, max(0.0, (t - (DURATION - 0.5)) / 0.5)))
    return np.clip(a, 0, 255).astype(np.uint8)


def assemble(out):
    audio = os.path.join(BUILD, 'audio.wav')
    make_audio(audio)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{RES[0]}x{RES[1]}', '-r', str(FPS), '-i', '-', '-i', audio,
           '-c:a', 'aac', '-b:a', '192k', '-shortest', '-c:v', 'libx264', '-preset', 'slow',
           '-crf', '16', '-x264-params', 'aq-mode=3', '-pix_fmt', 'yuv420p', '-profile:v', 'high',
           '-movflags', '+faststart', out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(NFRAMES):
        img = Image.open(os.path.join(FRAMES, f'f_{fi:04d}.png')).convert('RGB')
        if img.size != RES:
            img = img.resize(RES, Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=1.4, percent=55, threshold=2))
        enc.stdin.write(overlay(fi, img).tobytes())
    enc.stdin.close()
    enc.wait()
    print('done:', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--test', type=str, help='comma separated times: preview frames at half size')
    ap.add_argument('--frames', type=str, help='range a-b to render at full size')
    ap.add_argument('--samples', type=int, default=48)
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--assemble', action='store_true')
    ap.add_argument('--out', default=os.path.join(HERE, 'zylan_rooh_3d.mp4'))
    args = ap.parse_args()
    if args.test:
        fr = [int(round(float(t) * FPS)) for t in args.test.split(',')]
        render(fr, RES[0] // 2, RES[1] // 2, args.samples, os.path.join(BUILD, 'test'), 't_{:04d}.png')
    elif args.frames:
        a, b = (int(v) for v in args.frames.split('-'))
        w, h = int(RES[0] * args.scale) // 2 * 2, int(RES[1] * args.scale) // 2 * 2
        render(range(a, b + 1), w, h, args.samples, FRAMES)
    if args.assemble:
        assemble(args.out)


if __name__ == '__main__':
    main()
