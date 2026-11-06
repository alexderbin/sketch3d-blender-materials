"""Surface materials: the shared "sk_surface" material on library models. One node tree for all models — Eevee builds it once (measured: each tree
that is new in structure costs 0.2–0.4 s, identical ones with their own images and numbers are almost free).

What is inside:
  zones — paint / metal / organic by the team-colour mask, colour and gloss (components/materials/tex_maps.py, cache library/textures/masks/);
  shape masks — edges and cavities from the model's normal map (a curvature filter), "up", the underside of the model, AO shading (Eevee — live);
  state layers: edge wear and cavity dirt from the shape masks, and Lagarde's wetness model (Water drop 3b):
    wear (paint worn to metal), rust (Rust009), dirt, dust, burn, blood, snow, wet;
  substance values — Physically Based (iron, blood, coal, snow). Mask spread, dirt colour and grain — the object's 3D noise (does not
  repeat, no projection); dust and dirt — as a soft film with large-scale unevenness; the rust image — at two scales
  under large noise (macro variation). Repetition on a flat pipe — 0.05 against 0.43 for images.
States — the properties "sk_s1" = (wet, dirt, dust, rust) and "sk_s2" = (wear, blood, burn, snow), 0–1: on a scene — for everything
(Attribute node, View Layer), on an object — an addition for a figure (Object). They can be keyed — the state changes over time.

In Blender:  import bl_mat; bl_mat.apply(objs, up=False); bl_mat.set_states(scene_or_object, {"wet": 0.6, "blood": 0.3})
Demo and check:  blender -b -P core/blender/run.py -- components/materials/bl_mat.py -- demo <model folder> <output> [closeup=1] [hdri=<.hdr file>] [palette=off]
                  … -- time <model folder> <output> — states keyed over time (drying, dirt, blood), 5 frames
Workbench (previs) does not use the material: the active node is the colour image, as before."""
import core.paths         # the code is here, the core — core/<part> through kernel
import hashlib
import json
import math
import os
import re
import sys

import bpy
import numpy as np

ROOT = core.paths.ROOT
from components.materials import tex_maps as T

GROUP, VERSION = "sk_surface", 26
STATES = ("wet", "dirt", "dust", "rust", "wear", "blood", "burn", "snow")
MASKS = os.path.join(ROOT, "library/textures/masks")
_STAT = {}                                                  # last apply: time, materials, sets, new masks
UP = os.path.join(ROOT, "library/textures/closeup")
LAYERS = os.path.join(ROOT, "library/textures/layers/index.json")
IRON, DUST, CHAR, SNOW = (0.530, 0.513, 0.494), (0.34, 0.31, 0.27), (0.02, 0.02, 0.02), (0.85, 0.86, 0.88)
RUST, MUD = (0.20, 0.08, 0.03), (0.13, 0.095, 0.065)          # far variant: the average colours of Rust009 and dirt (no images)
BLOOD, BLOOD_DRY = (0.11, 0.003, 0.004), (0.05, 0.012, 0.007)   # thick blood (Physically Based gives 0.644 — a thin layer seen through)


# ── states ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def parse_states(s):
    """"wet:0.6,dirt:0.3" → {name: share}."""
    out = {}
    for kv in filter(None, (s or "").replace(";", ",").split(",")):
        k, _, v = kv.partition(":")
        k = k.strip()
        if k not in STATES:
            raise ValueError(f"state '{k}' does not exist; available: {', '.join(STATES)}")
        out[k] = float(v or 1)
    return out


GROUND = ("wet", "puddles", "dirt", "dust", "snow", "burn")     # for the ground: sk_g1 = (wet, puddles, dirt, dust), sk_g2 = (snow, burn)


def set_states(owner, st, world=False, ground=False):
    """owner — a scene or an object; st — {name: 0–1}, the rest — 0. For a scene: world=False — for fighters and vehicles (faction models),
    world=True — for buildings, objects, map pieces (materials with a gloss-strength map); ground=True — for the ground (terrain, map ground,
    rocks, walls, trees — the apply_weather weather layer); for an object — an addition to itself only."""
    if ground:
        v = [float(st.get(k, 0.0)) for k in GROUND] + [0.0, 0.0]
        owner["sk_g1"], owner["sk_g2"] = v[:4], v[4:]
        return
    v = [float(st.get(k, 0.0)) for k in STATES]
    p = "sk_w" if world else "sk_s"
    owner[p + "1"], owner[p + "2"] = v[:4], v[4:]


def key_states(owner, frame, st, world=False, ground=False):
    """A state at a frame — as a key (behaviour over time: drying, accumulating dirt, blood); smooth between keys."""
    set_states(owner, st, world, ground)
    p = "sk_g" if ground else ("sk_w" if world else "sk_s")
    for k in (p + "1", p + "2"):
        owner.keyframe_insert(f'["{k}"]', frame=frame)


# ── node tree ──────────────────────────────────────────────────────────────────────────────────────────────────────────
class _B:
    """Short node notation: numbers are input values, sockets are links."""

    def __init__(self, nt):
        self.nt = nt

    def n(self, typ, **kw):
        node = self.nt.nodes.new(typ)
        for k, v in kw.items():
            setattr(node, k, v)
        return node

    def put(self, sock, v):
        if isinstance(v, (int, float)):
            sock.default_value = v
        elif isinstance(v, tuple):
            sock.default_value = v if len(v) == len(sock.default_value) else (*v, 1.0)
        else:
            self.nt.links.new(v, sock)

    def math(self, op, a, b=None, clamp=False):
        node = self.n("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.put(node.inputs[0], a)
        if b is not None:
            self.put(node.inputs[1], b)
        return node.outputs[0]

    def add(self, *xs, clamp=False):
        s = xs[0]
        for x in xs[1:]:
            s = self.math("ADD", s, x)
        return self.math("ADD", s, 0.0, clamp=True) if clamp else s

    def mul(self, *xs):
        s = xs[0]
        for x in xs[1:]:
            s = self.math("MULTIPLY", s, x)
        return s

    def ss(self, x, lo, hi):
        """Smooth step 0→1 between lo and hi (lo, hi — numbers or sockets)."""
        node = self.n("ShaderNodeMapRange", interpolation_type="SMOOTHSTEP", clamp=True)
        self.put(node.inputs["Value"], x)
        self.put(node.inputs["From Min"], lo)
        self.put(node.inputs["From Max"], hi)
        return node.outputs["Result"]

    def _mix(self, kind, f, a, b):
        node = self.n("ShaderNodeMix", data_type=kind, clamp_factor=True)
        sfx = {"FLOAT": "Float", "RGBA": "Color", "VECTOR": "Vector"}[kind]
        ins = {s.identifier: s for s in node.inputs}
        self.put(ins["Factor_Float"], f)
        self.put(ins["A_" + sfx], a)
        self.put(ins["B_" + sfx], b)
        return next(s for s in node.outputs if s.identifier == "Result_" + sfx)

    def mixf(self, f, a, b):
        return self._mix("FLOAT", f, a, b)

    def mixc(self, f, a, b):
        return self._mix("RGBA", f, a, b)

    def mixv(self, f, a, b):
        return self._mix("VECTOR", f, a, b)


def _layer_images():
    if not os.path.exists(LAYERS):
        raise SystemExit("no layer textures: .venv/bin/python -m components.materials.mat_layers fetch")
    L = json.load(open(LAYERS))["layers"]
    out = {}
    for role, maps in L.items():
        for m, rel in maps.items():
            im = bpy.data.images.load(os.path.join(ROOT, rel), check_existing=True)
            im.colorspace_settings.name = "sRGB" if m == "Color" and role != "grunge" else "Non-Color"   # spread — data
            out[(role, m)] = im
    return out


def _code(used, lite):
    return ("lite_" if lite else "") + "".join(str(i) for i, k in enumerate(STATES) if k in used)


def ensure_group(used=STATES, lite=False):
    """A shared tree for a set of layers: only the states given in the scene go into the shader (the GPU does not skip branches —
    every layer costs every pixel). lite — the far variant: the same zones and
    states on simple masks (edges, cavities, "up", bottom, one coarse noise), without noise blotches, AO, relief and images.
    Name — sk_surface_<layer numbers>[_lite]; Eevee builds an identical tree once."""
    used = [k for k in STATES if k in set(used)]
    name = f"{GROUP}_{_code(used, lite)}"
    ng = bpy.data.node_groups.get(name)
    if ng is not None and ng.get("version") == VERSION:
        return ng
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    ng["version"], ng["sk_used"], ng["sk_lite"] = VERSION, ",".join(used), int(lite)
    I = ng.interface
    for nm, typ, dv in (("Base Color", "NodeSocketColor", (0.5, 0.5, 0.5, 1)), ("Alpha", "NodeSocketFloat", 1.0),
                        ("Normal", "NodeSocketVector", None), ("Gloss", "NodeSocketFloat", 0.22),
                        ("Masks", "NodeSocketColor", (0, 0, 0, 1)), ("Organic", "NodeSocketFloat", 0.5),
                        ("Emission", "NodeSocketColor", (0, 0, 0, 1)), ("World", "NodeSocketFloat", 0.0),
                        ("Spec", "NodeSocketFloat", 0.5)):
        sk = I.new_socket(nm, in_out="INPUT", socket_type=typ)
        if dv is not None:
            sk.default_value = dv
        else:
            sk.hide_value = True
    I.new_socket("BSDF", in_out="OUTPUT", socket_type="NodeSocketShader")
    b = _B(ng)
    gi, go = b.n("NodeGroupInput"), b.n("NodeGroupOutput")
    G = gi.outputs
    U = set(used)
    memo = {}

    def once(key, fn):
        if key not in memo:
            memo[key] = fn()
        return memo[key]

    # layer coordinates: the object's own axes + an offset by the object's random number (fighters are not identical)
    tc, oi, geo = b.n("ShaderNodeTexCoord"), b.n("ShaderNodeObjectInfo"), b.n("ShaderNodeNewGeometry")

    def co_():
        off = b.n("ShaderNodeCombineXYZ")
        for i, k in enumerate((13.7, 7.3, 3.1)):
            b.put(off.inputs[i], b.mul(oi.outputs["Random"], k))
        v = b.n("ShaderNodeVectorMath", operation="ADD")
        b.put(v.inputs[0], tc.outputs["Object"]); b.put(v.inputs[1], off.outputs[0])
        return v.outputs[0]
    co = lambda: once("co", co_)

    def noise(scale, detail=2.0, rough=0.5, dist=0.0):
        t = b.n("ShaderNodeTexNoise"); b.put(t.inputs["Vector"], co())
        t.inputs["Scale"].default_value, t.inputs["Detail"].default_value = scale, detail
        t.inputs["Roughness"].default_value, t.inputs["Distortion"].default_value = rough, dist
        return t.outputs["Fac"]

    def tex(role, m, scale, rot=(0, 0, 0), loc=(0, 0, 0)):
        mp = b.n("ShaderNodeMapping")
        b.put(mp.inputs["Vector"], co())
        mp.inputs["Scale"].default_value = (scale, scale, scale)
        mp.inputs["Rotation"].default_value, mp.inputs["Location"].default_value = rot, loc
        t = b.n("ShaderNodeTexImage", image=_layer_images()[(role, m)], projection="BOX", projection_blend=0.45)
        b.put(t.inputs["Vector"], mp.outputs[0])
        return t

    # macro variation: the image at two scales with a rotation, the share by large 3D noise;
    # otherwise on a large flat surface (pipe, wall) the image tile shows as a grid
    def tex2(role, m, scale):
        mac2 = once("mac2", lambda: b.ss(noise(0.45, 2.0), 0.4, 0.6))
        t1 = tex(role, m, scale).outputs["Color"]
        t2 = tex(role, m, scale * 0.37, rot=(0.6, 1.1, 0.4), loc=(3.1, 7.7, 1.9)).outputs["Color"]
        return b.mixc(mac2, t1, t2)

    def bw(sock):
        n = b.n("ShaderNodeRGBToBW"); b.put(n.inputs[0], sock); return n.outputs[0]

    def mulc(col, k):
        """colour × grey k (socket or number)"""
        m_ = b.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY", clamp_factor=True)
        ins = {s_.identifier: s_ for s_ in m_.inputs}
        b.put(ins["Factor_Float"], 1.0); b.put(ins["A_Color"], col)
        g = b.n("ShaderNodeCombineColor")
        for i in range(3):
            b.put(g.inputs[i], k)
        b.put(ins["B_Color"], g.outputs[0])
        return next(s_ for s_ in m_.outputs if s_.identifier == "Result_Color")

    # spread — the object's 3D noise, not an image (SurfaceImperfections with a 33 cm step gave a grid on a flat pipe): patches with a
    # ragged edge = medium noise + fine noise − Voronoi cells; large-scale unevenness of coverage (metres) — macro
    def grunge_():
        vg = b.n("ShaderNodeTexVoronoi", feature="SMOOTH_F1"); b.put(vg.inputs["Vector"], co())
        vg.inputs["Scale"].default_value, vg.inputs["Smoothness"].default_value = 5.0, 0.4
        st = b.add(noise(2.2, 4.0, 0.55, 0.6), b.mul(b.math("SUBTRACT", noise(9.0, 4.0, 0.65, 0.2), 0.5), 0.25),
                   b.mul(vg.outputs["Distance"], -0.35))
        return b.ss(st, 0.1, 0.5)                          # mean ≈ that of the former spread image (coverage matching)
    grunge = lambda: once("grunge", grunge_)
    macro = lambda: once("macro", lambda: b.ss(noise(0.35, 2.0), 0.3, 0.7))
    soft = lambda: once("soft", lambda: b.ss(noise(1.2, 3.0, 0.5, 0.3), 0.3, 0.7))     # large soft patches
    nz_s = lambda: once("nz_s", lambda: noise(3.0, 4.0))
    nz_l = lambda: once("nz_l", lambda: noise(0.8, 2.0))
    coarse = lambda: once("coarse", lambda: noise(0.8, 1.0))                            # far variant: one coarse noise

    # shape masks
    sep = b.n("ShaderNodeSeparateColor"); b.put(sep.inputs[0], G["Masks"])
    edge, cav, metal = sep.outputs["Red"], sep.outputs["Green"], sep.outputs["Blue"]
    za = G["Organic"]                                       # 0.5 + 0.5·(organic − army paint), components/materials/tex_maps.py
    org = b.math("SUBTRACT", b.mul(za, 2.0), 1.0, clamp=True)
    team = b.math("SUBTRACT", 1.0, b.mul(za, 2.0), clamp=True)
    paint = b.math("SUBTRACT", b.math("SUBTRACT", 1.0, metal), org, clamp=True)   # everything solid and non-metallic
    wearable = b.mixf(G["World"], team, paint)              # army paint wears down to the metal; on buildings — any paint

    def up_():
        nz = b.n("ShaderNodeSeparateXYZ"); b.put(nz.inputs[0], geo.outputs["Normal"])
        return b.ss(nz.outputs["Z"], 0.35, 0.85)
    up = lambda: once("up", up_)

    def occl_():
        if lite:
            return 0.0                                      # without AO: at a distance the cavities come from the shape masks
        ao = b.n("ShaderNodeAmbientOcclusion"); ao.inputs["Distance"].default_value = 0.35
        return b.math("SUBTRACT", 1.0, ao.outputs["AO"], clamp=True)
    occl = lambda: once("occl", occl_)

    def bottom_():
        rel = b.n("ShaderNodeVectorMath", operation="SUBTRACT")
        b.put(rel.inputs[0], geo.outputs["Position"]); b.put(rel.inputs[1], oi.outputs["Location"])
        rz = b.n("ShaderNodeSeparateXYZ"); b.put(rz.inputs[0], rel.outputs[0])
        return b.math("SUBTRACT", 1.0, b.ss(rz.outputs["Z"], 0.05, 0.7))
    bottom = lambda: once("bottom", bottom_)

    # states: scene (View Layer; for buildings — sk_w*) + object (sk_s*); nodes — only for the given layers
    def prop(name, wname):
        a = b.n("ShaderNodeAttribute", attribute_type="VIEW_LAYER", attribute_name=name)
        aw = b.n("ShaderNodeAttribute", attribute_type="VIEW_LAYER", attribute_name=wname)
        o = b.n("ShaderNodeAttribute", attribute_type="OBJECT", attribute_name=name)
        sv = b.mixv(G["World"], a.outputs["Vector"], aw.outputs["Vector"])
        v = b.n("ShaderNodeVectorMath", operation="ADD")
        b.put(v.inputs[0], sv); b.put(v.inputs[1], o.outputs["Vector"])
        sx = b.n("ShaderNodeSeparateXYZ"); b.put(sx.inputs[0], v.outputs[0])
        w = b.math("ADD", b.mixf(G["World"], a.outputs["Alpha"], aw.outputs["Alpha"]), o.outputs["Alpha"], clamp=True)
        return [sx.outputs["X"], sx.outputs["Y"], sx.outputs["Z"], w]

    def st(k):
        i = STATES.index(k)
        comps = once("s1" if i < 4 else "s2", lambda: prop("sk_s1", "sk_w1") if i < 4 else prop("sk_s2", "sk_w2"))
        return once("v_" + k, lambda: b.math("ADD", comps[i % 4], 0.0, clamp=True))
    gate = lambda s_: b.ss(s_, 0.0, 0.06)                    # at 0 there is no layer at all
    on = lambda x, s_, width: b.mul(b.ss(x, b.math("SUBTRACT", 1.0, s_), b.math("SUBTRACT", 1.0 + width, s_)), gate(s_))

    # base: the zones and gloss of the game
    world, spec, gloss = G["World"], G["Spec"], G["Gloss"]
    r_unit = b.math("MAXIMUM", 0.12, b.math("MINIMUM", 0.85, b.math("SUBTRACT", 0.8, b.mul(gloss, 1.4))))
    r_world = b.math("MAXIMUM", 0.2, spec)        # buildings, objects: as the import showed them (G — roughness); otherwise
                                                   # dark brown crates under a bright sky shine and turn grey (chaos_crop.jpg)
    R = b.mixf(world, r_unit, r_world)
    R = b.add(R, b.mul(metal, -0.12), b.mul(org, 0.1), clamp=True)
    M, C = metal, G["Base Color"]
    flat, bump_h, bump_s = [], [], []
    m_wear = m_rust = m_dirt = m_dust = m_burn = m_snow = 0.0

    if "wear" in U:       # 1 paint worn down to the metal on the edges
        w_in = edge if lite else b.mul(edge, b.add(0.45, b.mul(grunge(), 0.9)))
        m_wear = b.mul(on(w_in, st("wear"), 0.12), wearable)
        C, M, R = b.mixc(m_wear, C, IRON), b.mixf(m_wear, M, 1.0), b.mixf(m_wear, R, 0.3)
    if "rust" in U:    # 2 cavities, edges, top × noise; metal and paint chips rust; leather and cloth do not
        if lite:
            r_in, rc_col, rr = b.mul(b.add(b.mul(cav, 0.6), b.mul(edge, 0.45), b.mul(up(), 0.35), 0.15),
                                     b.add(0.6, b.mul(coarse(), 0.8))), RUST, 0.88
        else:
            rc_col = tex2("rust", "Color", 2.0)
            r_in = b.mul(b.add(b.mul(cav, 0.6), b.mul(edge, 0.45), b.mul(up(), 0.35), b.mul(occl(), 0.35), 0.15),
                         b.add(0.5, nz_s()), b.add(0.85, b.mul(macro(), 0.3)))
            rr = b.add(0.7, b.mul(bw(rc_col), 0.6))          # roughness — from the brightness of the same image (−6 samples)
            bump_h.append(b.mul(bw(rc_col), 1.0)); bump_s.append(0.5)
        rustable = b.mul(b.add(b.math("MAXIMUM", M, m_wear), b.mul(wearable, 0.35), clamp=True), b.math("SUBTRACT", 1.0, org))
        m_rust = b.mul(on(r_in, st("rust"), 0.2), rustable)
        C, M, R = b.mixc(m_rust, C, rc_col), b.mixf(m_rust, M, 0.0), b.mixf(m_rust, R, rr)
        if not lite:
            bump_s[-1] = b.mul(m_rust, 0.5); bump_h[-1] = b.mul(bump_h[-1], m_rust)
    if "dirt" in U:       # 3 cavities, shading, the underside of the model; colour — 3D noise with grain (no projection)
        if lite:
            d_in = b.mul(b.add(b.mul(cav, 0.6), b.mul(bottom(), 0.6), 0.37), b.add(0.7, b.mul(coarse(), 0.6)))
            dcol = MUD
        else:
            fine = noise(14.0, 4.0, 0.6, 0.0)
            grain = b.ss(noise(45.0, 2.0, 0.5, 0.0), 0.35, 0.65)            # soil grain
            g_amp = b.add(0.15, b.mul(soft(), 0.45))                          # grain strength — in patches, not an even speckle
            dc = b.mixc(b.ss(noise(1.3, 3.0, 0.5, 0.4), 0.35, 0.65), (0.16, 0.105, 0.06), (0.11, 0.09, 0.07))
            dcol = mulc(dc, b.mul(b.add(0.8, b.mul(fine, 0.4)), b.add(1.0, b.mul(b.math("SUBTRACT", grain, 0.5), g_amp))))
            d_in = b.mul(b.add(b.mul(cav, 0.6), b.mul(occl(), 0.7), b.mul(bottom(), 0.6), 0.37),
                         b.add(0.5, b.mul(grunge(), 0.6), b.mul(soft(), 0.3)), b.add(0.8, b.mul(macro(), 0.4)))
        m_dirt = b.mul(on(d_in, st("dirt"), 0.35), 0.9)
        C, M, R = b.mixc(m_dirt, C, dcol), b.mixf(m_dirt, M, 0.0), b.mixf(m_dirt, R, 0.92)
        if not lite:
            bump_h.append(b.mul(fine, m_dirt)); bump_s.append(b.mul(m_dirt, 0.25))
    if "dust" in U:        # 4 a soft film from above (thicker in corners, large-scale unevenness), not small patches
        if lite:
            u_in, dcl = b.add(b.mul(up(), b.add(0.7, b.mul(coarse(), 0.4))), b.mul(cav, 0.2)), DUST
        else:
            u_in = b.add(b.mul(up(), b.add(0.45, b.mul(soft(), 0.45), b.mul(grunge(), 0.2)), b.add(0.8, b.mul(macro(), 0.4))),
                         b.mul(cav, 0.2), b.mul(occl(), 0.15))
            dcl = mulc(DUST, b.add(0.8, b.mul(grunge(), 0.4)))
        m_dust = b.mul(on(u_in, st("dust"), 0.55), 0.8)
        C, M, R = b.mixc(m_dust, C, dcl), b.mixf(m_dust, M, 0.0), b.mixf(m_dust, R, 0.95)
        flat.append(b.mul(m_dust, 0.3))
    if "burn" in U:        # 5 soot in patches
        nzb = coarse() if lite else nz_l()
        m_burn = b.mul(on(b.mul(b.add(0.3, b.mul(cav, 0.5), b.mul(occl(), 0.4)), b.add(0.2, b.mul(b.ss(nzb, 0.4, 0.7), 1.1))),
                          st("burn"), 0.15), 0.95)
        C, M, R = b.mixc(m_burn, C, CHAR), b.mixf(m_burn, M, 0.0), b.mixf(m_burn, R, 0.85)
    if "blood" in U:       # 6 splatter (Voronoi cells, edges by noise) in clusters
        bl = st("blood")
        if lite:
            m_blood = b.mul(b.ss(coarse(), b.math("SUBTRACT", 0.7, b.mul(bl, 0.25)), b.math("SUBTRACT", 0.74, b.mul(bl, 0.25))), gate(bl))
            bl_c, rb = BLOOD, 0.3
        else:
            vor = b.n("ShaderNodeTexVoronoi", feature="F1"); b.put(vor.inputs["Vector"], co()); vor.inputs["Scale"].default_value = 4.0
            vd = b.add(vor.outputs["Distance"], b.mul(b.math("SUBTRACT", nz_s(), 0.5), 0.45))
            spl = b.math("SUBTRACT", 1.0, b.ss(vd, 0.02, b.add(0.04, b.mul(bl, 0.5))))
            clus = b.ss(nz_l(), b.math("SUBTRACT", 0.68, b.mul(bl, 0.22)), b.math("SUBTRACT", 0.72, b.mul(bl, 0.22)))
            m_blood = b.mul(b.math("MAXIMUM", b.mul(spl, clus), b.mul(clus, b.ss(nz_s(), 0.55, 0.6))), gate(bl))
            bl_c = b.mixc(b.ss(grunge(), 0.3, 0.8), BLOOD, BLOOD_DRY)          # fresh — dark red, dried — brown
            rb = b.mixf(b.ss(grunge(), 0.3, 0.8), 0.15, 0.5)
        C, M, R = b.mixc(m_blood, C, bl_c), b.mixf(m_blood, M, 0.0), b.mixf(m_blood, R, rb)
    if "snow" in U:        # 7 from above, by noise
        if lite:
            m_snow, snow_c = on(b.mul(up(), b.add(0.7, b.mul(coarse(), 0.5))), st("snow"), 0.25), SNOW
        else:
            m_snow = on(b.mul(up(), b.add(0.55, b.mul(nz_s(), 0.5), b.mul(grunge(), 0.3))), st("snow"), 0.25)
            snow_c = b.mixc(b.mul(b.math("SUBTRACT", 1.0, grunge()), 0.35), SNOW, (0.70, 0.73, 0.78))
            bump_h.append(b.mul(nz_s(), m_snow)); bump_s.append(b.mul(m_snow, 0.35))
        C, M, R = b.mixc(m_snow, C, snow_c), b.mixf(m_snow, M, 0.0), b.mixf(m_snow, R, 0.55)
        flat.append(b.mul(m_snow, 0.6))
    if "wet" in U:      # 8 Lagarde: factor = lerp(1; 0.2; porosity·(1−metal)); colour ×= lerp(1; factor; wetness);
        wet = st("wet")  # roughness ×= lerp(1; factor; 0.5·wetness); a water film — gloss and a smoothed normal
        por = b.mul(b.add(b.mul(org, 0.9), b.mul(paint, 0.25), m_dirt, m_dust, m_burn, b.mul(m_rust, 0.7), clamp=True),
                    b.math("SUBTRACT", 1.0, M))
        one_m = b.mul(por, 0.8)
        C = mulc(C, b.math("SUBTRACT", 1.0, b.mul(wet, one_m)))
        R = b.mul(R, b.math("SUBTRACT", 1.0, b.mul(b.mul(wet, 0.5), one_m)))
        R = b.mixf(b.mul(wet, 0.55, b.math("SUBTRACT", 1.0, m_snow)), R, 0.06)
        flat.append(b.mul(wet, 0.35))

    # normal: relief of rust, dirt, snow; smoothing by dust, snow, water
    N = G["Normal"]
    if bump_h:
        bump = b.n("ShaderNodeBump"); b.put(bump.inputs["Strength"], b.add(*bump_s) if len(bump_s) > 1 else bump_s[0])
        b.put(bump.inputs["Height"], b.add(*bump_h) if len(bump_h) > 1 else bump_h[0]); b.put(bump.inputs["Normal"], N)
        N = bump.outputs["Normal"]
    if flat:
        nn = b.n("ShaderNodeVectorMath", operation="NORMALIZE")
        b.put(nn.inputs[0], b.mixv(b.add(*flat, clamp=True) if len(flat) > 1 else flat[0], N, geo.outputs["Normal"]))
        N = nn.outputs[0]
    p = b.n("ShaderNodeBsdfPrincipled")
    b.put(p.inputs["Base Color"], C); b.put(p.inputs["Metallic"], M)
    b.put(p.inputs["Roughness"], b.math("MAXIMUM", 0.04, R)); p.inputs["Specular IOR Level"].default_value = 0.5
    b.put(p.inputs["Normal"], N); b.put(p.inputs["Alpha"], G["Alpha"])
    b.put(p.inputs["Emission Color"], G["Emission"]); p.inputs["Emission Strength"].default_value = 1.0
    b.put(go.inputs["BSDF"], p.outputs[0])
    return ng


# ── model materials ────────────────────────────────────────────────────────────────────────────────────────────────────
def _base_name(im_name):
    s = im_name.lower()
    s = re.sub(r"(_tc)+$", "", s)
    s = re.sub(r"\.\d{3}$", "", s)
    s = re.sub(r"\.(jpg|jpeg|png|tga|dds)$", "", s)
    return re.sub(r"_(dif|diff|diffuse)$", "", s)


def _px(im, alpha=True):
    im.alpha_mode = "STRAIGHT"
    w, h = im.size
    if w == 0:
        return None
    buf = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(buf)
    a = buf.reshape(h, w, 4)
    return a if alpha else a[..., :3]


def _store():
    try:
        from core.assets.models import bl_model as BM
    except ImportError:                                 # without the game-model modules (as in the public release) there is no store
        return {}
    return BM._store_index()


def _masks_image(base, nrm_im, dif_im, mr_im, world, hint=""):
    """Model masks (edges, cavities, metal, organic) — from the cache library/textures/masks, or computed and saved."""
    nrm = _px(nrm_im, alpha=False)
    if nrm is None:
        return None
    key = hashlib.md5(nrm[::max(1, nrm.shape[0] // 16), ::max(1, nrm.shape[1] // 16)].tobytes()).hexdigest()[:8]
    os.makedirs(MASKS, exist_ok=True)
    path = os.path.join(MASKS, f"{base}_{nrm.shape[1]}_{key}_v{T.VERSION}.png")
    if not os.path.exists(path):
        _STAT["masks_new"] = _STAT.get("masks_new", 0) + 1
        dif = _px(dif_im, alpha=False)
        gl = _px(mr_im, alpha=False)[..., 1] if mr_im is not None else None
        tem = None
        st = _store().get(base)
        if st and st[1]:
            ti = bpy.data.images.load(st[1], check_existing=True)
            tem = _px(ti)
            hint = hint + " " + st[0]
        m = T.masks(nrm[::-1], dif[::-1], None if tem is None else tem[::-1], None if gl is None else gl[::-1],
                    base + " " + hint, world)[::-1]                      # Blender stores rows bottom to top
        im = bpy.data.images.new(f"sk_msk_{base}", m.shape[1], m.shape[0], alpha=True)
        im.pixels.foreach_set((m.astype(np.float32) / 255).ravel())
        im.filepath_raw = path
        im.file_format = "PNG"
        im.save()
        bpy.data.images.remove(im)
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = "Non-Color"
    im.alpha_mode = "CHANNEL_PACKED"
    return im


def _up_set(base):
    """Upscaled set (library/textures/closeup): {dif, tem, nrm, gls, msk} or None."""
    p = os.path.join(UP, "index.json")
    if not os.path.exists(p):
        return None
    idx = json.load(open(p))
    e = idx.get(base)
    return {k: os.path.join(ROOT, v) for k, v in e.items()} if e else None


def _palette(m, base_im, base):
    """Army colours of the material: the team_palette property (bl_model.team_colour) or an estimate from the "_tc" image and the mask."""
    if m.get("team_palette"):
        return [list(c) for c in m["team_palette"]]
    st = _store().get(base)
    if not st or not st[1] or not base_im.name.lower().endswith("_tc"):
        return None
    tc = _px(base_im, alpha=False)
    dif = _px(bpy.data.images.load(st[0], check_existing=True), alpha=False)
    tem = _px(bpy.data.images.load(st[1], check_existing=True))
    if tem.shape[:2] != tc.shape[:2]:
        tem = T.resize(tem, tc.shape[:2])
    if dif.shape[:2] != tc.shape[:2]:
        dif = T.resize(dif, tc.shape[:2])
    pal = []
    for i in range(3):                                      # the emblem channel (A) is often all 1 — it is not parsed, colour 0.5
        sel = (tem[..., i] > 0.9) & (np.delete(tem[..., :3], i, -1).max(-1) < 0.1) & (dif.max(-1) > 0.08)
        if sel.sum() < 20:
            pal.append([0.5, 0.5, 0.5])
            continue
        pal.append([float(x) for x in np.clip(np.median(tc[sel] / np.maximum(dif[sel], 1e-3), 0) / 2, 0, 1)])
    return pal + [[0.5, 0.5, 0.5]]


def _load(path, colour):
    im = bpy.data.images.load(path, check_existing=True)
    im.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    return im


def surface_material(m, ng, up=False, tag="sk"):
    """Model material (glTF import: node labels BASE COLOR, NORMAL MAP, METALLIC ROUGHNESS, EMISSIVE, SPECULAR COLOR)
    → a copy on the shared tree. Not a library model (no colour image) → None."""
    lab = {n.label: n for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image}
    base_n = lab.get("BASE COLOR")
    if base_n is None:
        try:
            from core.assets.models import bl_model as BM
            base_n = BM._dif_node(m)
        except ImportError:                             # without the game-model modules — glTF labels only
            base_n = None
        if base_n is not None and not re.search(r"_dif(\.\d{3})?(\.(jpg|png))?(_tc)*$", base_n.image.name.lower()):
            base_n = None                                   # the frame's ground (ground_tex) and other own objects — not a library model
    if base_n is None:
        return None
    base = _base_name(base_n.image.name)
    nrm_n, mr_n, emi_n = lab.get("NORMAL MAP"), lab.get("METALLIC ROUGHNESS"), lab.get("EMISSIVE")
    world = "SPECULAR COLOR" in lab
    p0 = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    nm0 = next((n for n in m.node_tree.nodes if n.type == "NORMAL_MAP"), None)
    alpha_linked = p0 is not None and p0.inputs["Alpha"].is_linked

    base_im, nrm_im, mr_im = base_n.image, nrm_n.image if nrm_n else None, mr_n.image if mr_n else None
    msk_im = None
    U = _up_set(base) if up else None
    if U:                                                   # close-up: the upscaled set, team colour again
        dif = _load(U["dif"], True)
        pal = _palette(m, base_im, base) if base_im.name.lower().endswith("_tc") else None
        if pal and U.get("tem"):
            d, t = _px(dif, alpha=False), _px(_load(U["tem"], False))
            out = T.team_tint(d, t, pal)
            im = bpy.data.images.new(f"{base}_up_tc", out.shape[1], out.shape[0])
            im.pixels.foreach_set(np.concatenate([out, np.ones(out.shape[:2] + (1,), np.float32)], -1).ravel())
            im.pack()
            base_im = im
        else:
            base_im = dif
        nrm_im = _load(U["nrm"], False)
        if U.get("gls"):
            mr_im = _load(U["gls"], False)
        if U.get("msk"):
            msk_im = _load(U["msk"], False)
            msk_im.alpha_mode = "CHANNEL_PACKED"
    if msk_im is None and nrm_im is not None:
        msk_im = _masks_image(base, nrm_im, base_n.image, mr_im, world)

    m2 = bpy.data.materials.new(f"{m.name}·{tag}")
    m2["sk_mat"] = VERSION
    m2["sk_base"] = base
    for attr in ("surface_render_method", "blend_method", "use_backface_culling", "alpha_threshold"):
        if hasattr(m, attr):
            try:
                setattr(m2, attr, getattr(m, attr))
            except (AttributeError, TypeError):
                pass
    nt = m2.node_tree
    nt.nodes.clear()
    b = _B(nt)
    out = b.n("ShaderNodeOutputMaterial")
    g = b.n("ShaderNodeGroup", node_tree=ng)
    bt = b.n("ShaderNodeTexImage", image=base_im, label="BASE COLOR")
    b.put(g.inputs["Base Color"], bt.outputs["Color"])
    if alpha_linked:
        b.put(g.inputs["Alpha"], bt.outputs["Alpha"])
    nm = b.n("ShaderNodeNormalMap")
    if nm0 is not None:
        nm.uv_map = nm0.uv_map
        nm.inputs["Strength"].default_value = nm0.inputs["Strength"].default_value
    if nrm_im is not None:
        nrm_im.colorspace_settings.name = "Non-Color"
        nt_ = b.n("ShaderNodeTexImage", image=nrm_im, label="NORMAL MAP")
        b.put(nm.inputs["Color"], nt_.outputs["Color"])
    else:
        nm.inputs["Strength"].default_value = 0.0
    b.put(g.inputs["Normal"], nm.outputs["Normal"])
    if mr_im is not None:
        mr_im.colorspace_settings.name = "Non-Color"
        mt = b.n("ShaderNodeTexImage", image=mr_im, label="GLOSS")
        sp = b.n("ShaderNodeSeparateColor")
        b.put(sp.inputs[0], mt.outputs["Color"])
        b.put(g.inputs["Gloss"], sp.outputs["Green"])
        b.put(g.inputs["Spec"], sp.outputs["Green"])
    if msk_im is not None:
        mk = b.n("ShaderNodeTexImage", image=msk_im, label="MASKS")
        b.put(g.inputs["Masks"], mk.outputs["Color"])
        b.put(g.inputs["Organic"], mk.outputs["Alpha"])
    if emi_n is not None and emi_n.image.size[0] > 8:
        et = b.n("ShaderNodeTexImage", image=emi_n.image, label="EMISSIVE")
        b.put(g.inputs["Emission"], et.outputs["Color"])
    g.inputs["World"].default_value = 1.0 if world else 0.0
    b.put(out.inputs["Surface"], g.outputs["BSDF"])
    nt.nodes.active = bt                                    # Workbench draws the active image — the colour, as before
    return m2


def used_states(scene, objs=None):
    """Layers set anywhere at all: properties of the scene (fighters sk_s*, world sk_w*), of objects (sk_s*) and their time keys."""
    from bpy_extras import anim_utils
    mx = {k: 0.0 for k in STATES}
    owners = [scene] + [o for o in (objs if objs is not None else scene.objects) if "sk_s1" in o or "sk_s2" in o
                        or (o.animation_data and o.animation_data.action)]
    for ow in owners:
        for pfx in ("sk_s", "sk_w"):
            for j in (1, 2):
                v = ow.get(f"{pfx}{j}")
                if v is not None:
                    for i, x in enumerate(v):
                        mx[STATES[(j - 1) * 4 + i]] = max(mx[STATES[(j - 1) * 4 + i]], float(x))
        ad = getattr(ow, "animation_data", None)
        if ad and ad.action:
            cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
            for fc in (cb.fcurves if cb else []):
                for pfx in ("sk_s", "sk_w"):
                    for j in (1, 2):
                        if fc.data_path == f'["{pfx}{j}"]':
                            k = STATES[(j - 1) * 4 + fc.array_index]
                            mx[k] = max([mx[k]] + [kp.co[1] for kp in fc.keyframe_points])
    return {k for k, v in mx.items() if v > 1e-3}


def cams_from_json(cams):
    """cameras.json → [(position, tan of half the horizontal field)] for choosing the far variant."""
    from mathutils import Matrix
    out = []
    for c in cams:
        M = Matrix(c["matrix_world"])
        out.append((M.translation.copy(), c.get("sensor_width_mm", 36.0) / (2 * c.get("lens_mm", 50.0))))
    return out


def cams_from_scene(scene, step=48):
    """Scene cameras per frame (shots with markers switch the camera): positions every step frames."""
    out, f0 = [], scene.frame_current
    for f in range(scene.frame_start, scene.frame_end + 1, step):
        scene.frame_set(f)
        c = scene.camera
        if c is not None:
            out.append((c.matrix_world.translation.copy(), c.data.sensor_width / (2 * c.data.lens)))
    scene.frame_set(f0)
    return out


LOD_FRAC = 0.12          # an object under 12 % of the frame width on all cameras — the far (light) variant


def _is_hero(ob):
    while ob is not None:
        if ob.get("sketch_kind") == "figure":
            return True
        ob = ob.parent
    return False


def _near(ob, cams):
    """Full variant: a hero, or an object taking ≥ LOD_FRAC of the frame width on at least one camera. Crowd prototypes
    (invisible, shown by Geometry Nodes) are light: a crowd is a mass, close-up heroes are separate objects."""
    from mathutils import Vector
    if _is_hero(ob):
        return True
    if not ob.visible_get():
        return False
    pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    size, ctr = (hi - lo).length, (hi + lo) / 2
    for pos, th in cams:
        d = (ctr - pos).length
        if d < size or size / (2 * max(d, 1e-3) * th) >= LOD_FRAC:
            return True
    return False


def apply(objs=None, up=False, scene=None, lod=None, cams=None, used=None):
    """The shared material on the meshes of library models (other materials are left unchanged).
    used — the layers in the shader (by default those set in the scene and on objects, used_states: set the states BEFORE apply);
    lod: None — full for all; "auto" — light for far ones (under LOD_FRAC of the frame on all cameras cams); "lite" — light for all.
    Returns {(material, variant): new material}."""
    import time
    scene = scene or bpy.context.scene
    if "sk_s1" not in scene:
        set_states(scene, {})
    if "sk_w1" not in scene:
        set_states(scene, {}, world=True)
    t0, _STAT["masks_new"] = time.time(), 0
    used = used_states(scene, objs) if used is None else set(used)
    groups = {False: ensure_group(used), True: ensure_group(used, lite=True) if lod in ("auto", "lite") else None}
    done, nfar, nnear, near_sets = {}, 0, 0, set()
    obs = [ob for ob in (objs if objs is not None else scene.objects) if ob.type == "MESH" and not ob.hide_render]
    # the variant applies to a whole mesh: a per-object material (slot.link = OBJECT) breaks batched drawing of identical meshes
    # (measured on the Gates scene: auto with per-object materials — 21.2 s against 20.0 for full). A mesh is light only if all its
    # objects are far; one near object — full for all.
    far_of = {ob.name: (lod == "lite" or (lod == "auto" and not _near(ob, cams or []))) for ob in obs}
    mesh_far = {}
    for ob in obs:
        mesh_far[ob.data.name] = mesh_far.get(ob.data.name, True) and far_of[ob.name]
    for ob in obs:
        nfar, nnear = nfar + far_of[ob.name], nnear + (not far_of[ob.name])
        for slot in ob.material_slots:
            m = slot.material
            if m is None:
                continue
            far = far_of[ob.name] if slot.link == "OBJECT" else mesh_far[ob.data.name]
            if m.get("sk_mat"):
                m = bpy.data.materials.get(m.get("sk_src", "")) or m
                if m.get("sk_mat"):
                    continue
            key = (m.name, far)
            if key not in done:
                done[key] = surface_material(m, groups[far], up and not far, tag="sk·lite" if far else "sk")
                if done[key] is not None:
                    done[key]["sk_src"] = m.name
            if done[key] is not None:
                if not far:
                    near_sets.add(done[key]["sk_base"])
                slot.material = done[key]
    mats = [v for v in done.values() if v is not None]
    _STAT.update(materials=len(mats), seen=len({k[0] for k in done}), apply_s=round(time.time() - t0, 2),
                 sets=sorted({v["sk_base"] for v in mats}), sets_near=sorted(near_sets), used=[k for k in STATES if k in used],
                 objects_full=nnear, objects_lite=nfar)
    print(f"materials: layers [{', '.join(_STAT['used']) or 'no'}], materials {len(mats)} (of {_STAT['seen']}), sets "
          f"{len(_STAT['sets'])}, new masks {_STAT['masks_new']}, objects full {nnear} / light {nfar}"
          f"{' (close-up)' if up else ''}, {_STAT['apply_s']} s", flush=True)
    return done


# ── weather layer: ground and everything not from the library ──────────────────────────────────────────────────────────
WEATHER_LAYERS = ("burn", "dirt", "dust", "snow", "wet", "puddles", "rain")
MUD_WET = (0.075, 0.055, 0.04)                                 # dirt on the ground is darker than dirt on armour (wet soil)
SLUSH_K, SLUSH_DIRTY = (0.60, 0.65, 0.72), (0.20, 0.19, 0.18)   # snow in rain — slush: darker and bluish; near roads — dirty
WATER_IMG, WATER_RES, WATER_MAX = "sk_water", 0.5, 1200          # water map: 0.5 m step, grid up to 1200² (larger — a coarser step)
RAIN_TILT, RAIN_AZ = 10.0, 30.0                                  # rain: tilt from the vertical and its direction (degrees from east)


def rain_dir():
    """Direction toward the sky from which the rain comes (unit vector)."""
    t, a = math.radians(RAIN_TILT), math.radians(RAIN_AZ)
    return (math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t))


def set_rain(owner, v):
    """Rain is falling now: 0–1 (puddle ripples, splashes, streaks); wetness and puddles are ground states and remain after the rain."""
    owner["sk_rain"] = float(v)


def key_rain(owner, frame, v):
    set_rain(owner, v)
    owner.keyframe_insert('["sk_rain"]', frame=frame)


def _fcurve(owner, path, index=0):
    from bpy_extras import anim_utils
    ad = getattr(owner, "animation_data", None)
    if not (ad and ad.action):
        return None
    cb = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    return cb.fcurves.find(path, index=index) if cb else None


def rain_used(scene):
    fc = _fcurve(scene, '["sk_rain"]')
    return float(scene.get("sk_rain", 0.0)) > 1e-3 or bool(fc and any(kp.co[1] > 1e-3 for kp in fc.keyframe_points))


def time_prop(scene):
    """Scene time, s — the sk_time property with linear keys (puddle ripples advance with the frames; Blender 5.2 shaders
    have no time node — only Geometry Nodes do)."""
    fps = scene.render.fps / scene.render.fps_base
    for f, v in ((0, 0.0), (100, 100.0 / fps)):
        scene["sk_time"] = v
        scene.keyframe_insert('["sk_time"]', frame=f)
    fc = _fcurve(scene, '["sk_time"]')
    if fc:
        fc.extrapolation = "LINEAR"
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"


def water_image(arr=None):
    """Water map image (RGBA, linear): R — running water (rills, streams), G — 1 − the rain layer at which the cell
    floods (/30 mm; 0 — never floods, same as outside the map), B — shelter from the rain, A — trampled ground.
    Without a map — 1×1 zeros (the shader always takes an image; outside the map — zeros too)."""
    im = bpy.data.images.get(WATER_IMG)
    a = np.zeros((1, 1, 4), np.float32) if arr is None else arr.astype(np.float32)
    h, w = a.shape[:2]
    if im is None:
        im = bpy.data.images.new(WATER_IMG, w, h, alpha=True, float_buffer=True)
    elif tuple(im.size) != (w, h):
        im.scale(w, h)                                             # the same data block — node references stay valid
    im.colorspace_settings.name = "Non-Color"
    im.alpha_mode = "CHANNEL_PACKED"
    im.pixels.foreach_set(a.ravel())
    im.update()
    return im


_WATER = {}                                                       # last water map: arr, x0, y0, res (for the sk_wat attribute)
IMG_BUDGET = 9                                                    # more images in the material than this — the water map goes as an attribute (Metal: 16 textures
                                                                  # per shader, some are used by Eevee itself; scan ground — 12–13)


def water_attr(ob):
    """Water map into the vertex attribute sk_wat (FLOAT_COLOR): bilinear over the world XY; zeros outside the map."""
    a = _WATER.get("arr")
    me = ob.data
    n = len(me.vertices)
    if a is None or n == 0:
        return 0
    co = np.empty(n * 3, np.float32); me.vertices.foreach_get("co", co)
    M = np.array(ob.matrix_world); w = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    ny, nx = a.shape[:2]
    fx, fy = (w[:, 0] - _WATER["x0"]) / _WATER["res"], (w[:, 1] - _WATER["y0"]) / _WATER["res"]
    inside = (fx >= 0) & (fy >= 0) & (fx <= nx - 1) & (fy <= ny - 1)
    fx, fy = np.clip(fx, 0, nx - 1.001), np.clip(fy, 0, ny - 1.001)
    i, j = fx.astype(int), fy.astype(int); tx, ty = (fx - i)[:, None], (fy - j)[:, None]
    v = (a[j, i] * (1 - tx) * (1 - ty) + a[j, i + 1] * tx * (1 - ty) + a[j + 1, i] * (1 - tx) * ty + a[j + 1, i + 1] * tx * ty)
    v[~inside] = 0.0
    at = me.attributes.get("sk_wat") or me.attributes.new("sk_wat", "FLOAT_COLOR", "POINT")
    at.data.foreach_set("color", v.astype(np.float32).ravel())
    return int(inside.sum())


def _n_images(nt, seen=None):
    """Number of distinct images in the node tree (with groups)."""
    seen = set() if seen is None else seen
    for n in nt.nodes:
        if n.type in ("TEX_IMAGE", "TEX_ENVIRONMENT") and getattr(n, "image", None) is not None:
            seen.add(n.image.name)
        elif n.type == "GROUP" and n.node_tree is not None:
            _n_images(n.node_tree, seen)
    return len(seen)


def _shelter_objects(scene):
    """What shelters the ground from rain: meshes that are neither ground nor water (shelters, walls, arches, trees, towers), without fighters and crowds; those whose top
    is below 2 m above the ground (crates, boulders — they cover the ground under themselves) are filtered out by water_map."""
    out = []
    for ob in scene.objects:
        if ob.type != "MESH" or _weather_target(ob) is not False or max(ob.dimensions) < 0.4 \
                or "crowd" in ob.name.lower() or ob.get("kind") in ("crowd", "figure"):
            continue
        p_, fig = ob, False
        while p_ is not None and not fig:
            fig, p_ = p_.get("sketch_kind") == "figure", p_.parent
        if not fig:
            out.append(ob)
    return out


def water_map(scene, folder=None, cams=None):
    """Scene water map (components/materials/mat_water.py: hollows — Priority-Flood, runoff — MFD on terrain with micro-relief; shelter —
    rays to the sky against buildings, rocks, trees) → the image sk_water and the scene property sk_wmap = (x0, y0, 1/width,
    1/height) in world metres: the weather shader reads it by XY. Heights — the site's relief.npz (the folder is the terrain's relief property
    or the scene folder), otherwise the vertices of the scene ground: for a large ground (a battle scene — 8 × 7 km in one grid) — the area around the frame's cameras
    (cams — cams_from_json/cams_from_scene, ±250 m, up to 2.5 km). Runoff and hollows are cached in out/materials/water/. → a summary."""
    import time
    from mathutils import Vector
    from components.materials import mat_water as MW
    from core.surface import terrain_analysis as TA
    t0 = time.time()
    grounds = [ob for ob in scene.objects if _weather_target(ob) is True and not ob.name.lower().startswith("far_")]
    src = next((os.path.join(f, "relief.npz") for f in [ob.get("relief") for ob in grounds] + [folder]
                if f and os.path.exists(os.path.join(f, "relief.npz"))), None)
    use = None
    if src:
        R_ = np.load(src)
        H, (x0, y0, res) = R_["near"].astype(np.float64), (float(v) for v in R_["near_o"])
        if "road" in R_.files:
            use = np.maximum(R_["road"], R_["pad"]).astype(np.float64) / 255
    elif grounds:
        pts = []
        for ob in grounds:
            n = len(ob.data.vertices)
            co = np.empty(n * 3, np.float32); ob.data.vertices.foreach_get("co", co)
            M = np.array(ob.matrix_world); pts.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        w = np.concatenate(pts)
        if cams and max(np.ptp(w[:, 0]), np.ptp(w[:, 1])) > 1500:   # large ground — the area around the cameras
            cp_ = np.array([c[0][:2] for c in cams], float); c0 = cp_.mean(0)
            lo, hi = np.maximum(cp_.min(0) - 250.0, c0 - 1250.0), np.minimum(cp_.max(0) + 250.0, c0 + 1250.0)
            w = w[(w[:, 0] >= lo[0]) & (w[:, 0] <= hi[0]) & (w[:, 1] >= lo[1]) & (w[:, 1] <= hi[1])]
        if len(w) < 100:
            water_image()
            scene["sk_wmap"] = [0.0, 0.0, 0.0, 0.0]
            return {"map": "no ground near the cameras"}
        ext = max(np.ptp(w[:, 0]), np.ptp(w[:, 1]))
        res = max(WATER_RES, ext / WATER_MAX)
        step = math.sqrt(max(np.ptp(w[:, 0]) * np.ptp(w[:, 1]), 1.0) / len(w))   # average vertex spacing, m
        H, x0, y0 = MW.raster(w, res, sigma=0.6 * step / res)
    else:
        water_image()
        scene["sk_wmap"] = [0.0, 0.0, 0.0, 0.0]
        return {"map": "no ground"}
    key = hashlib.sha1(H.astype(np.float32).tobytes() + (b"" if use is None else use.astype(np.float32).tobytes())
                       + f"{MW.VERSION}|{res}".encode()).hexdigest()[:12]
    cp = os.path.join(ROOT, "out", "materials", "water", key + ".npz")
    if os.path.exists(cp):
        d_ = np.load(cp); run, need = d_["run"], d_["need"]
    else:
        o = MW.compute(H, res, use)
        run, need = o["run"].astype(np.float32), o["need"].astype(np.float32)
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        np.savez_compressed(cp, run=run, need=need)
    t1 = time.time()
    ny, nx = H.shape
    # shelter — a depth map from above along the rain (Lagarde, Water drop 2a: "like a shadow map"), in numpy: triangles are seeded
    # with points at the map step, a point at height h above the ground is shifted along the wind by h·tan(tilt); if anything higher than
    # 0.3 m is above a cell, it is dry. Without rays: a battle scene would need 4 million rays, 9 s.
    shel = np.zeros((ny, nx), np.float32)
    rd = rain_dir(); sx, sy = -rd[0] / rd[2], -rd[1] / rd[2]               # offset of the rain shadow per metre of height
    dg = bpy.context.evaluated_depsgraph_get()
    rng = np.random.default_rng(5)
    npts = nobj = 0
    for ob in _shelter_objects(scene):
        bb = np.array([ob.matrix_world @ Vector(c) for c in ob.bound_box])
        pad = (bb[:, 2].max() - bb[:, 2].min()) * math.tan(math.radians(RAIN_TILT)) + res
        if bb[:, 0].max() + pad < x0 or bb[:, 0].min() - pad > x0 + nx * res or bb[:, 1].max() + pad < y0 or bb[:, 1].min() - pad > y0 + ny * res:
            continue
        ic = min(max(int((bb[:, 0].mean() - x0) / res), 0), nx - 1); jc = min(max(int((bb[:, 1].mean() - y0) / res), 0), ny - 1)
        if bb[:, 2].max() - H[jc, ic] < 2.0:                                # top below 2 m: it covers the ground under itself
            continue
        eo = ob.evaluated_get(dg)
        me = eo.to_mesh()
        me.calc_loop_triangles()
        nv, nt = len(me.vertices), len(me.loop_triangles)
        if nt:
            co = np.empty(nv * 3, np.float32); me.vertices.foreach_get("co", co)
            M = np.array(ob.matrix_world); co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            tri = np.empty(nt * 3, np.int32); me.loop_triangles.foreach_get("vertices", tri); T3 = co[tri.reshape(-1, 3)]
            area = 0.5 * np.abs(np.cross(T3[:, 1, :2] - T3[:, 0, :2], T3[:, 2, :2] - T3[:, 0, :2]))   # area in plan view
            cnt = np.minimum(np.ceil(area / (res * res * 0.25)).astype(int) + 1, 4000)
            ti = np.repeat(np.arange(nt), cnt)
            u, v = rng.random(len(ti)), rng.random(len(ti)); fl_ = u + v > 1; u[fl_], v[fl_] = 1 - u[fl_], 1 - v[fl_]
            P = T3[ti, 0] + (T3[ti, 1] - T3[ti, 0]) * u[:, None] + (T3[ti, 2] - T3[ti, 0]) * v[:, None]
            i_ = np.clip(((P[:, 0] - x0) / res).round().astype(int), 0, nx - 1); j_ = np.clip(((P[:, 1] - y0) / res).round().astype(int), 0, ny - 1)
            h = P[:, 2] - H[j_, i_]
            P = P[h > 0.3]; h = h[h > 0.3]
            i_ = ((P[:, 0] + sx * h - x0) / res).round().astype(int); j_ = ((P[:, 1] + sy * h - y0) / res).round().astype(int)
            ok = (i_ >= 0) & (i_ < nx) & (j_ >= 0) & (j_ < ny)
            shel[j_[ok], i_[ok]] = 1.0
            npts += int(ok.sum()); nobj += 1
        eo.to_mesh_clear()
    nrays = npts
    from components.materials import tex_maps as T_
    shel = np.clip(T_.blur(shel, 0.8) * 1.3, 0, 1)
    arr = np.stack([np.clip(run, 0, 1), 1.0 - np.clip(np.nan_to_num(need / 0.03, posinf=1.0), 0, 1), shel,
                    np.zeros_like(shel) if use is None else use.astype(np.float32)], -1)
    water_image(arr)
    scene["sk_wmap"] = [float(x0 - res / 2), float(y0 - res / 2), float(1.0 / (nx * res)), float(1.0 / (ny * res))]
    _WATER.update(arr=arr, x0=float(x0), y0=float(y0), res=float(res))
    st = {"map": os.path.relpath(src, ROOT) if src else "ground vertices", "grid": [nx, ny], "step": res,
          "puddles_share": round(float((need < 0.009).mean()), 3), "occlusion": [nobj, nrays],
          "sec": [round(t1 - t0, 2), round(time.time() - t1, 2)], "cache": os.path.basename(cp)}
    _STAT["water"] = st
    return st


WM_OUT = (("burn", "F"), ("burn_core", "F"), ("mud", "F"), ("dust", "F"), ("snow", "F"), ("wet", "F"), ("rill", "F"), ("lev", "F"),
          ("over_b", "F"), ("hole", "F"), ("pond_k", "F"), ("w0", "F"), ("drench", "F"), ("rx", "F"), ("ry", "F"), ("fine", "F"),
          ("bump", "V"))                                       # weather masks → overlay on each BSDF


def _wtree(name, inputs, outputs):
    """An empty shader group with inputs and outputs [(name, socket type, value | None)]."""
    ng = bpy.data.node_groups.get(name)
    if ng is not None and ng.get("version") == VERSION:
        return ng, False
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(name, "ShaderNodeTree")
    ng["version"] = VERSION
    for io, items in (("INPUT", inputs), ("OUTPUT", outputs)):
        for nm, typ, dv in items:
            sk = ng.interface.new_socket(nm, in_out=io, socket_type=typ)
            if io == "INPUT":
                if dv is None:
                    sk.hide_value = True
                else:
                    sk.default_value = dv
    return ng, True


class _WH:
    """Nodes of the weather layer: noises and masks in world space (shared by both groups)."""

    def __init__(self, ng):
        self.b = b = _B(ng)
        self.gi, self.go = b.n("NodeGroupInput"), b.n("NodeGroupOutput")
        self.geo = b.n("ShaderNodeNewGeometry")
        self.memo = {}

    def once(self, key, fn):
        if key not in self.memo:
            self.memo[key] = fn()
        return self.memo[key]

    def noise(self, scale, detail=2.0, rough=0.5, dist=0.0):
        t = self.b.n("ShaderNodeTexNoise"); self.b.put(t.inputs["Vector"], self.geo.outputs["Position"])
        t.inputs["Scale"].default_value, t.inputs["Detail"].default_value = scale, detail
        t.inputs["Roughness"].default_value, t.inputs["Distortion"].default_value = rough, dist
        return t.outputs["Fac"]

    def sep(self, v):
        s_ = self.b.n("ShaderNodeSeparateXYZ"); self.b.put(s_.inputs[0], v)
        return s_.outputs

    def mulc(self, col, k):
        b = self.b
        m_ = b.n("ShaderNodeMix", data_type="RGBA", blend_type="MULTIPLY", clamp_factor=True)
        ins = {s_.identifier: s_ for s_ in m_.inputs}
        b.put(ins["Factor_Float"], 1.0); b.put(ins["A_Color"], col)
        if isinstance(k, tuple):
            b.put(ins["B_Color"], k)
        else:
            g = b.n("ShaderNodeCombineColor")
            for i in range(3):
                b.put(g.inputs[i], k)
            b.put(ins["B_Color"], g.outputs[0])
        return next(s_ for s_ in m_.outputs if s_.identifier == "Result_Color")

    def upz(self):
        return self.once("upz", lambda: self.sep(self.geo.outputs["Normal"])["Z"])

    def pos(self):
        return self.once("pos", lambda: self.sep(self.geo.outputs["Position"]))

    def vl(self, nm):
        return self.once("vl_" + nm, lambda: self.b.n("ShaderNodeAttribute", attribute_type="VIEW_LAYER", attribute_name=nm))

    def one_m(self, x):
        return self.b.math("SUBTRACT", 1.0, x)


def ensure_weather_groups(used, attr=False):
    """Weather layer over a finished material (terrain, map ground, rocks, walls, trees) — two groups: masks (one per
    material: water map, noises, puddle levels, ripples — expensive) and overlay (per each Principled of the material: colour, roughness,
    layer normal — cheap; site ground from scans — one BSDF per layer, computing masks for each would be four to six times more expensive).
    Ground (Ground = 1) reads the scene's sk_g*, everything else reads sk_w* (like buildings). Water follows Lagarde's canon (Water drop 2b, 3b) on
    the site's water map (water_map): wet — porosity, light colours darken less; runoff streams — water as a film; puddles — in
    hollows up to the flooding level (trampled ground holds water, snow drinks it) and small pits, a wet rim; under shelter — dry; snow in the
    rain — slush; dirt — where there is water and traffic, with no noise blotches; when it rains — ripples as rings (time — sk_time). Layers — only the given ones.
    attr — the water map from the vertex attribute sk_wat (a material with many images). → (mask group, overlay group)."""
    used = [k for k in WEATHER_LAYERS if k in set(used)]
    U = set(used)
    code = "".join(str(i) for i, k in enumerate(WEATHER_LAYERS) if k in used)
    water = bool({"wet", "puddles", "dirt"} & U)
    rain_fx = "rain" in U and bool({"wet", "puddles"} & U)
    outs = [(nm, "NodeSocketVector" if t == "V" else "NodeSocketFloat", None) for nm, t in WM_OUT]
    # ── masks ──
    ngm, fresh = _wtree("sk_wm_" + code + ("a" if attr else ""), [("Ground", "NodeSocketFloat", 1.0)], outs)
    if fresh:
        h = _WH(ngm); b = h.b; G = h.gi.outputs; O = {}
        soft = lambda: h.once("soft", lambda: b.ss(h.noise(0.35, 3.0, 0.5, 0.3), 0.3, 0.7))       # patches metres across
        big = lambda: h.once("big", lambda: b.ss(h.noise(0.06, 2.0), 0.35, 0.65))                  # unevenness tens of metres across
        fine = lambda: h.once("fine", lambda: h.noise(6.0, 3.0))
        gate = lambda s_: b.ss(s_, 0.0, 0.06)
        on = lambda x, s_, width: b.mul(b.ss(x, b.math("SUBTRACT", 1.0, s_), b.math("SUBTRACT", 1.0 + width, s_)), gate(s_))

        def st(k):
            """Layer share: for ground — sk_g*, for everything else — sk_w* (buildings)."""
            gi_ = {"wet": (1, 0), "puddles": (1, 1), "dirt": (1, 2), "dust": (1, 3), "snow": (2, 0), "burn": (2, 1)}[k]
            wi_ = {"wet": (1, 0), "dirt": (1, 1), "dust": (1, 2), "snow": (2, 3), "burn": (2, 2), "puddles": None}[k]
            ga = h.vl(f"sk_g{gi_[0]}")
            gs = h.once(f"gs{gi_[0]}", lambda: h.sep(ga.outputs["Vector"]))
            gv = ga.outputs["Alpha"] if gi_[1] == 3 else gs["XYZ"[gi_[1]]]
            wv = 0.0
            if wi_ is not None:
                wa = h.vl(f"sk_w{wi_[0]}")
                ws = h.once(f"ws{wi_[0]}", lambda: h.sep(wa.outputs["Vector"]))
                wv = wa.outputs["Alpha"] if wi_[1] == 3 else ws["XYZ"[wi_[1]]]
            return h.once("v_" + k, lambda: b.mixf(G["Ground"], wv, gv))

        def wm_():
            """Water map over the world XY: running water, 1 − rain layer before flooding (/30 mm), shelter, trampled ground (0–1; outside the map —
            zeros). attr — from the vertex attribute sk_wat (a Metal shader allows 16 textures)."""
            if attr:
                a_ = b.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="sk_wat")
                c_ = b.n("ShaderNodeSeparateColor"); b.put(c_.inputs[0], a_.outputs["Color"])
                return c_.outputs["Red"], c_.outputs["Green"], c_.outputs["Blue"], a_.outputs["Alpha"]
            m_ = h.vl("sk_wmap"); o_ = h.sep(m_.outputs["Vector"]); p_ = h.pos()
            uv = b.n("ShaderNodeCombineXYZ")
            b.put(uv.inputs[0], b.mul(b.math("SUBTRACT", p_["X"], o_["X"]), o_["Z"]))
            b.put(uv.inputs[1], b.mul(b.math("SUBTRACT", p_["Y"], o_["Y"]), m_.outputs["Alpha"]))
            tex = b.n("ShaderNodeTexImage", image=bpy.data.images.get(WATER_IMG) or water_image(), extension="CLIP",
                      interpolation="Cubic")                         # cubic: the shore on a 0.5 m grid — no stair steps
            b.put(tex.inputs["Vector"], uv.outputs[0])
            c_ = b.n("ShaderNodeSeparateColor"); b.put(c_.inputs[0], tex.outputs["Color"])
            return c_.outputs["Red"], c_.outputs["Green"], c_.outputs["Blue"], tex.outputs["Alpha"]

        m_snow = 0.0
        if water:
            fl, pdn, shl, use = wm_()
            ex = h.one_m(b.mul(shl, G["Ground"]))                          # open to the sky (under a building or rock — dry)
        if "burn" in U:    # rare scorched crater patches: Voronoi cells ~25 m, a patch only in some of the cells, round,
            bl_ = st("burn")   # soft edge, darker toward the middle (texture bombing; explosion sites are handled as events)
            vo = b.n("ShaderNodeTexVoronoi", feature="F1"); b.put(vo.inputs["Vector"], h.geo.outputs["Position"])
            vo.inputs["Scale"].default_value = 0.04
            rnd = b.n("ShaderNodeSeparateColor"); b.put(rnd.inputs[0], vo.outputs["Color"])
            pick = b.math("LESS_THAN", rnd.outputs["Red"], b.mul(bl_, 0.6))
            rad = b.add(0.12, b.mul(bl_, 0.18), b.mul(b.math("SUBTRACT", soft(), 0.5), 0.08))
            O["burn"] = b.mul(h.one_m(b.ss(vo.outputs["Distance"], b.mul(rad, 0.55), rad)), pick, gate(bl_), 0.95)
            O["burn_core"] = h.one_m(b.ss(vo.outputs["Distance"], 0.0, b.mul(rad, 0.6)))
        if "dirt" in U:   # dirt — where there is water and foot traffic: road and pad, hollows, streams on gentle ground; no noise threshold
            flat_ = b.ss(h.upz(), 0.85, 0.97)
            g_in = b.math("MAXIMUM", b.mul(use, 0.9), b.math("MAXIMUM", b.mul(fl, flat_, 0.75), b.mul(b.ss(pdn, 0.7, 0.98), 0.6)))
            g_in = b.mixf(G["Ground"], b.mul(b.add(0.6, b.mul(soft(), 0.6)), 0.3), g_in)   # everything else — a weak film
            O["mud"] = b.math("MINIMUM", 1.0, b.mul(g_in, st("dirt"), 2.2, b.add(0.75, b.mul(soft(), 0.25))))
        if "dust" in U:    # a film from above, with no threshold (on large flat terrain a threshold cut it into patches)
            O["dust"] = b.mul(st("dust"), b.ss(h.upz(), 0.3, 0.8), b.add(0.65, b.mul(big(), 0.35)), b.add(0.85, b.mul(soft(), 0.15)), 0.9)
        if "snow" in U:    # by slope: flat ground — full coverage, steeper than ~55° — none
            m_snow = O["snow"] = on(b.mul(b.ss(h.upz(), 0.55, 0.85), b.add(0.7, b.mul(soft(), 0.5))), st("snow"), 0.25)
        if "puddles" in U:    # Lagarde 2b: flooding level — hollows (by rain volume: L²·30 mm stands, trampled ground holds everything,
            L = st("puddles")  # soil — ¼; snow — in the overlay) and pits from the height map (more on the road and pad)
            O["lev"] = b.mul(L, L, b.mixf(use, 0.25, 1.0))
            shore = b.math("SUBTRACT", h.noise(3.0, 3.0, 0.55), 0.5)           # the shore is uneven (~30 cm), not following the map grid
            O["over_b"] = b.add(pdn, b.mul(shore, 0.03), -1.0)
            hl = b.mul(L, 0.55, b.add(0.3, b.mul(use, 0.7)), b.add(0.7, b.mul(big(), 0.6)))
            O["hole"] = b.mul(h.one_m(b.ss(h.noise(1.3, 4.0, 0.6), b.math("SUBTRACT", hl, 0.03), hl)), b.ss(h.upz(), 0.965, 0.99))
            O["pond_k"] = b.mul(ex, G["Ground"], gate(L), h.one_m(m_snow))
        if "wet" in U:  # Lagarde: walls get less wet; runoff streams — water as a film
            W0 = O["w0"] = st("wet")
            O["wet"] = b.mul(W0, ex, h.one_m(m_snow), b.add(0.7, b.mul(b.math("MAXIMUM", h.upz(), 0.0), 0.3)))
            O["rill"] = b.mul(fl, W0, ex, G["Ground"])
            if "rain" in U:   # it rains — "drenched" (Lagarde: drenched): a thin film on flat ground
                O["drench"] = b.mul(h.vl("sk_rain").outputs["Fac"], b.ss(h.upz(), 0.9, 0.99), ex, G["Ground"])
        if rain_fx:        # ripples — Lagarde's formula (Water drop 2b) as is: x = 1 − d/r within the drop's circle, DropFrac =
            t_, rn = h.vl("sk_time").outputs["Fac"], h.vl("sk_rain").outputs["Fac"]   # fract(offset + t), ring
            nx_, ny_ = [], []   # sin(clamp((DropFrac − 1 + x)·9, 0, 3)·π); circles — 2D Voronoi cells (centre — Position, m);
            for cell, rate, dy in ((0.36, 1.0, 0.0), (0.24, 0.85, 37.1)):   # no images and no Bump
                py = b.add(h.pos()["Y"], dy)
                pv = b.n("ShaderNodeCombineXYZ"); b.put(pv.inputs[0], h.pos()["X"]); b.put(pv.inputs[1], py)
                vo = b.n("ShaderNodeTexVoronoi", feature="F1", voronoi_dimensions="2D"); b.put(vo.inputs["Vector"], pv.outputs[0])
                vo.inputs["Scale"].default_value = 1.0 / cell
                fp = h.sep(vo.outputs["Position"])
                dx_, dy_ = b.math("SUBTRACT", h.pos()["X"], fp["X"]), b.math("SUBTRACT", py, fp["Y"])
                d = b.math("MAXIMUM", b.math("SQRT", b.add(b.mul(dx_, dx_), b.mul(dy_, dy_))), 1e-4)
                rc = b.n("ShaderNodeSeparateColor"); b.put(rc.inputs[0], vo.outputs["Color"])
                x = b.math("SUBTRACT", 1.0, b.math("DIVIDE", d, b.mul(cell, b.add(0.28, b.mul(rc.outputs["Green"], 0.17)))), clamp=True)
                drop = b.math("FRACT", b.add(rc.outputs["Red"], b.mul(t_, rate)))
                dfac = b.math("SUBTRACT", b.add(0.2, b.mul(rn, 0.8)), drop, clamp=True)
                ring = b.math("SINE", b.mul(b.math("MINIMUM", b.math("MAXIMUM", b.mul(b.add(drop, -1.0, x), 9.0), 0.0), 3.0), math.pi))
                fin = b.math("DIVIDE", b.mul(dfac, x, ring), d)
                nx_.append(b.mul(dx_, fin)); ny_.append(b.mul(dy_, fin))
            O["rx"], O["ry"] = b.add(*nx_), b.add(*ny_)
        if {"dirt", "dust", "snow"} & U:
            O["fine"] = fine()
        if water and ({"dirt", "wet"} & U):   # wet dirt and soil — fine unevenness (normal shift; strength — in the overlay)
            bp = b.n("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.35; b.put(bp.inputs["Distance"], 0.01)
            b.put(bp.inputs["Height"], fine()); b.put(bp.inputs["Normal"], h.geo.outputs["Normal"])
            dv = b.n("ShaderNodeVectorMath", operation="SUBTRACT"); b.put(dv.inputs[0], bp.outputs["Normal"]); b.put(dv.inputs[1], h.geo.outputs["Normal"])
            O["bump"] = dv.outputs[0]
        for nm, v in O.items():
            b.put(h.go.inputs[nm], v)
    # ── overlay on the BSDF ──
    nga, fresh = _wtree("sk_wa_" + code, [("Color", "NodeSocketColor", (0.5, 0.5, 0.5, 1)), ("Roughness", "NodeSocketFloat", 0.5),
                                          ("Normal", "NodeSocketVector", None), ("Ground", "NodeSocketFloat", 1.0)] + outs,
                        [("Color", "NodeSocketColor", None), ("Roughness", "NodeSocketFloat", None), ("Normal", "NodeSocketVector", None)])
    if fresh:
        h = _WH(nga); b = h.b; G = h.gi.outputs
        C, R, N0 = G["Color"], G["Roughness"], G["Normal"]
        def snowb_():
            """Layer snow: light and almost colourless (measured on a game map: snow V 0.54, S 0.03; yard S 0.2)."""
            hsv = b.n("ShaderNodeSeparateColor", mode="HSV"); b.put(hsv.inputs[0], G["Color"])
            return b.mul(b.ss(hsv.outputs["Blue"], 0.42, 0.52), h.one_m(b.ss(hsv.outputs["Green"], 0.06, 0.14)), G["Ground"])
        snowb = lambda: h.once("snowb", snowb_)
        flat = []
        mud = dust = 0.0
        if "burn" in U:
            C, R = b.mixc(G["burn"], C, b.mixc(G["burn_core"], h.mulc(C, 0.45), CHAR)), b.mixf(G["burn"], R, 0.9)
        if "dirt" in U:   # on snow — dirty slush
            mud = G["mud"]
            mc = b.mixc(snowb(), b.mixc(0.35, h.mulc(C, 0.55), MUD_WET), SLUSH_DIRTY)
            C, R = b.mixc(mud, C, h.mulc(mc, b.add(0.8, b.mul(G["fine"], 0.4)))), b.mixf(mud, R, 0.85)
        if "dust" in U:
            dust = G["dust"]
            C, R = b.mixc(dust, C, h.mulc(DUST, b.add(0.85, b.mul(G["fine"], 0.3)))), b.mixf(dust, R, 0.95)
            flat.append(b.mul(dust, 0.3))
        if "snow" in U:
            C, R = b.mixc(G["snow"], C, b.mixc(b.mul(G["fine"], 0.4), SNOW, (0.74, 0.77, 0.82))), b.mixf(G["snow"], R, 0.5)
            flat.append(b.mul(G["snow"], 0.8))
        acc = ring = over = wet = None
        if "puddles" in U:    # snow absorbs water: level ×0.12, pits fade out
            over = b.add(b.mul(G["lev"], b.mixf(snowb(), 1.0, 0.12)), G["over_b"])
            acc = b.mul(b.math("MAXIMUM", b.ss(over, 0.0, 0.01), b.mul(G["hole"], h.one_m(b.mul(snowb(), 0.8)))), G["pond_k"])
            ring = b.mul(b.ss(over, -0.02, 0.0), G["pond_k"])                # wet rim around a puddle (saturated soil)
        if "wet" in U:  # Lagarde: porosity 0.7, light colours darken less than dark ones
            wet = b.math("MAXIMUM", G["wet"], G["rill"])
            if ring is not None:
                wet = b.math("MAXIMUM", wet, b.mul(ring, G["w0"]))
            por = b.add(0.7, b.mul(mud, 0.3), b.mul(dust, 0.3), clamp=True)
            lum = b.n("ShaderNodeRGBToBW"); b.put(lum.inputs[0], C)
            dk = b.mul(por, 0.8, h.one_m(b.mul(lum.outputs[0], 0.6)))
            Cw = h.mulc(C, h.one_m(b.mul(wet, dk)))
            Rw = b.mixf(b.mul(wet, 0.45), b.mul(R, h.one_m(b.mul(wet, 0.5, dk))), 0.08)
            sl = b.mul(snowb(), wet)                                      # snow in rain — slush: darker, bluish, glossier
            C, R = b.mixc(b.mul(sl, 0.85), Cw, h.mulc(C, SLUSH_K)), b.mixf(sl, Rw, 0.22)
            C, R = h.mulc(C, h.one_m(b.mul(G["rill"], 0.25))), b.mixf(G["rill"], R, 0.06)   # stream: water as a film
            if "rain" in U:   # "drenched": soil stays rough (Lagarde: wet-soil gloss ×2.5 — not a mirror)
                dr = b.mul(G["drench"], h.one_m(b.mul(snowb(), 0.5)))
                C, R = h.mulc(C, h.one_m(b.mul(dr, 0.1))), b.mixf(b.mul(dr, 0.35), R, 0.2)
            flat.append(b.mul(wet, 0.3)); flat.append(b.mul(G["rill"], 0.6))
        if acc is not None:    # water in a puddle: the bottom is visible, darker with depth; smooth, flat normal
            C = b.mixc(acc, C, h.mulc(C, b.mixf(b.ss(over, 0.0, 0.1), 0.72, 0.42)))
            R = b.mixf(acc, R, 0.025)
            flat.append(acc)
        N = N0
        if flat:
            nn = b.n("ShaderNodeVectorMath", operation="NORMALIZE")
            b.put(nn.inputs[0], b.mixv(b.add(*flat, clamp=True) if len(flat) > 1 else flat[0], N0, h.geo.outputs["Normal"]))
            N = nn.outputs[0]
        if water and ({"dirt", "wet"} & U):
            mb = b.mul(b.add(b.mul(mud, 0.6), b.mul(wet if wet is not None else 0.0, 0.25)), G["Ground"],
                       h.one_m(acc) if acc is not None else 1.0)
            sc_ = b.n("ShaderNodeVectorMath", operation="SCALE"); b.put(sc_.inputs[0], G["bump"]); b.put(sc_.inputs["Scale"], mb)
            ad = b.n("ShaderNodeVectorMath", operation="ADD"); b.put(ad.inputs[0], N); b.put(ad.inputs[1], sc_.outputs[0])
            nn = b.n("ShaderNodeVectorMath", operation="NORMALIZE"); b.put(nn.inputs[0], ad.outputs[0]); N = nn.outputs[0]
        if rain_fx and (acc is not None or "wet" in U):     # ripples: the water normal tilts in rings
            wmask = b.math("MAXIMUM", acc if acc is not None else 0.0, b.mul(G["rill"], 0.35) if "wet" in U else 0.0)
            k_ = b.mul(wmask, 0.6)
            d_ = b.n("ShaderNodeCombineXYZ"); b.put(d_.inputs[0], b.mul(G["rx"], k_)); b.put(d_.inputs[1], b.mul(G["ry"], k_))
            ad = b.n("ShaderNodeVectorMath", operation="ADD"); b.put(ad.inputs[0], N); b.put(ad.inputs[1], d_.outputs[0])
            nn = b.n("ShaderNodeVectorMath", operation="NORMALIZE"); b.put(nn.inputs[0], ad.outputs[0]); N = nn.outputs[0]
        b.put(h.go.inputs["Color"], C); b.put(h.go.inputs["Roughness"], b.math("MAXIMUM", 0.02, R)); b.put(h.go.inputs["Normal"], N)
    return ngm, nga


GROUND_NAMES = ("terrain", "ground", "far_terrain")
SKIP_NAMES = ("far_hz", "probe", "sky", "cloud", "smoke", "fire", "far_city", "sk_rain")


def _weather_target(ob):
    if ob.type != "MESH" or ob.hide_render or ob.get("sketch_fx") or ob.get("kind") == "water":
        return None
    if ob.name.lower().startswith(SKIP_NAMES):
        return None
    return bool(ob.name.lower().startswith(GROUND_NAMES) or ob.get("kind") == "ground")


def apply_weather(scene=None, used=None, folder=None, reflect=True, cams=None):
    """Weather layer for everything that is not from the library: ground (terrain, map ground, far terrain), rocks, walls, trees, own
    buildings. Inserted before the Principled BSDF of a finished material (colour, roughness, normal); materials without Principled,
    transparent and emissive ones are left unchanged. Ground reads the scene's sk_g*, everything else reads sk_w*; water follows the water map (water_map,
    folder — the scene folder for relief.npz). Returns the number of materials."""
    import time
    scene = scene or bpy.context.scene
    for k in ("sk_g1", "sk_w1"):
        if k not in scene:
            set_states(scene, {}, world=(k == "sk_w1"), ground=(k == "sk_g1"))
    t0 = time.time()
    if used is None:
        used = set()
        for pfx, names in (("sk_g", GROUND), ("sk_w", STATES)):
            vals = list(scene.get(pfx + "1", [])) + list(scene.get(pfx + "2", []))
            used |= {nm for nm, v in zip(names, vals) if v > 1e-3}
        if rain_used(scene):
            used.add("rain")
    used = {k for k in used if k in WEATHER_LAYERS}
    if not used:
        return 0
    if {"wet", "puddles", "dirt"} & used:
        water_map(scene, folder, cams)
    if reflect and {"wet", "puddles"} & used:   # wet surfaces and puddles reflect buildings and fighters, not just the sky: Eevee ray tracing
        e_ = scene.eevee                    # (without it, only the world: wet ground is flat grey, puddles have no reflections)
        e_.use_raytracing, e_.ray_tracing_method = True, "SCREEN"
        e_.ray_tracing_options.resolution_scale, e_.ray_tracing_options.trace_max_roughness = "2", 0.5
    if "rain" in used:
        time_prop(scene)
    ngs = {}
    mats = {}
    for ob in scene.objects:
        g = _weather_target(ob)
        if g is None:
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is None or m.get("sk_mat") or m.get("sk_weather"):
                continue
            if any(n.type == "TEX_IMAGE" and n.label == "BASE COLOR" for n in m.node_tree.nodes):
                continue                                   # a library model — its weather is in sk_surface (apply)
            mats[m.name] = mats.get(m.name, False) or g
    done = nbsdf = 0
    attr_mats = set()
    for name, g in mats.items():
        m = bpy.data.materials[name]
        nt = m.node_tree
        ps = [n for n in nt.nodes if n.type == "BSDF_PRINCIPLED" and not (n.inputs["Transmission Weight"].default_value > 0.5
              or n.inputs["Alpha"].is_linked or (n.inputs["Emission Strength"].default_value > 0 and n.inputs["Emission Color"].is_linked))]
        attr = bool({"wet", "puddles", "dirt"} & used) and _n_images(nt) > IMG_BUDGET
        if attr:
            attr_mats.add(name)
        if attr not in ngs:
            ngs[attr] = ensure_weather_groups(used, attr)
        ngm, ng = ngs[attr]
        if not ps:
            continue
        mk = nt.nodes.new("ShaderNodeGroup"); mk.node_tree = ngm; mk.inputs["Ground"].default_value = 1.0 if g else 0.0
        for p in ps:                       # for each Principled (site ground from scans — a BSDF per layer): the masks are shared
            gn = nt.nodes.new("ShaderNodeGroup"); gn.node_tree = ng; gn.inputs["Ground"].default_value = 1.0 if g else 0.0
            for nm, _t in WM_OUT:
                nt.links.new(mk.outputs[nm], gn.inputs[nm])
            for sock, inp in (("Base Color", "Color"), ("Roughness", "Roughness"), ("Normal", "Normal")):
                src = p.inputs[sock]
                if src.is_linked:
                    lk = src.links[0]; nt.links.new(lk.from_socket, gn.inputs[inp]); nt.links.remove(lk)
                elif inp == "Normal":
                    gm = nt.nodes.new("ShaderNodeNewGeometry"); nt.links.new(gm.outputs["Normal"], gn.inputs["Normal"])
                else:
                    gn.inputs[inp].default_value = src.default_value
                nt.links.new(gn.outputs[inp], p.inputs[sock])
        if ps:
            m["sk_weather"] = ng.name
            done += 1; nbsdf += len(ps)
    nattr = 0
    if attr_mats:                          # water map as an attribute — for objects with such materials
        for ob in scene.objects:
            if ob.type == "MESH" and any(sl.material is not None and sl.material.name in attr_mats for sl in ob.material_slots):
                nattr += water_attr(ob)
    _STAT.update(weather_materials=done, weather_bsdf=nbsdf, weather_used=[k for k in WEATHER_LAYERS if k in used],
                 weather_s=round(time.time() - t0, 2))
    wst = _STAT.get("water") if {"wet", "puddles", "dirt"} & used else None
    print(f"weather: layers [{', '.join(_STAT['weather_used'])}], ground and world materials {done} (BSDF {nbsdf}"
          + (f"; water map as an attribute — {len(attr_mats)}, vertices {nattr}" if attr_mats else "") + f"), {_STAT['weather_s']} s"
          + (f"; water map {wst['grid'][0]}×{wst['grid'][1]} at {wst['step']} m ({wst['map']}), hollows {wst['puddles_share']}, "
             f"shelter {wst['occlusion'][0]} objects / {wst['occlusion'][1]} points, {wst['sec'][0]} + {wst['sec'][1]} s" if wst else ""),
          flush=True)
    return done


def apply_plan(plan, scene=None, cams=None, sets_out=None):
    """Plan components/materials/mat_plan.py (scenes/<name>/materials.yaml + command-line keys) → states of the scene and the world, additions for groups of
    objects (a name pattern or the sketch_kind tag on the object or its ancestor), time keys, the shared material; far objects get a
    light variant (simplification: auto, cameras cams). sets_out — JSON with the list of texture sets of the near objects (for an upscaling tool).
    Crowd instances (Geometry Nodes) take the states of their prototype; an addition to the crowd does not change them.
    Rain (strength > 0 or in the time keys) — ripples in puddles and streaks with splashes in the frame (components/materials/bl_rain.py)."""
    import fnmatch
    scene = scene or bpy.context.scene
    base, world, ground = plan.get("state", {}), plan.get("world", {}), plan.get("ground", {})
    set_states(scene, base)
    set_states(scene, world, world=True)
    set_states(scene, ground, ground=True)
    set_rain(scene, plan.get("rain", 0.0))
    for t in plan.get("timeline", []):                         # behaviour over time: smooth between the keys
        key_states(scene, t["frame"], {**base, **t.get("state", {})})
        if t.get("world") or world:
            key_states(scene, t["frame"], {**world, **t.get("world", {})}, world=True)
        if t.get("ground") or ground:
            key_states(scene, t["frame"], {**ground, **t.get("ground", {})}, ground=True)
        if "rain" in t:
            key_rain(scene, t["frame"], t["rain"])
    wused = {k for k, v in ground.items() if v > 0} | {k for k, v in world.items() if v > 0}
    for t in plan.get("timeline", []):
        wused |= {k for k, v in {**t.get("ground", {}), **t.get("world", {})}.items() if v > 0}
    if rain_used(scene):
        wused.add("rain")
    apply_weather(scene, used=wused, folder=plan.get("folder"), reflect=plan.get("reflections", True), cams=cams)
    if "rain" in wused:
        from components.materials import bl_rain
        bl_rain.add(scene)
    ng = 0
    for g in plan.get("groups", []):
        for ob in scene.objects:
            hit = g.get("name") and fnmatch.fnmatch(ob.name, g["name"])
            p_ = ob
            while not hit and g.get("tag") and p_ is not None:
                hit, p_ = p_.get("sketch_kind") == g["tag"], p_.parent
            if hit and ob.type == "MESH":
                set_states(ob, g.get("state", {}))
                ng += 1
    done = apply(None, up=plan.get("closeup", False), scene=scene, lod=plan.get("simplify", "auto"), cams=cams)
    _STAT["group_objects"] = ng
    if sets_out:
        os.makedirs(os.path.dirname(sets_out), exist_ok=True)
        json.dump({"near": _STAT.get("sets_near", []), "all": _STAT.get("sets", [])}, open(sets_out, "w"), ensure_ascii=False, indent=1)
    return done


# ── demo and check ──────────────────────────────────────────────────────────────────────────────────────────────────────
DEMO = [("clean", {}), ("wear", {"wear": 0.7}), ("rust", {"rust": 0.6, "wear": 0.5}),
        ("dirt", {"dirt": 0.6}), ("dust", {"dust": 0.6}), ("wet", {"wet": 0.9, "dirt": 0.3}),
        ("blood", {"blood": 0.7}), ("burn", {"burn": 0.6}), ("snow", {"snow": 0.7}),
        ("battle", {"wear": 0.5, "dirt": 0.5, "blood": 0.4, "burn": 0.3, "dust": 0.2})]


def demo(model_dir, out, opt):
    import math
    import time
    from core.assets.models import bl_model as BM
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    objs = BM.load_light(model_dir)
    pal = [[0.10, 0.24, 0.72], [0.10, 0.24, 0.72], [0.92, 0.70, 0.22], [0.5, 0.5, 0.5]]   # a blue team palette; palette=off — no colour
    if opt.get("palette") != "off":
        BM.team_colour(objs, pal)
    meshes = [o for o in objs if o.type == "MESH" and o.visible_get()]
    lo, hi = BM.bbox(meshes)
    H = hi[2] - lo[2]
    w = bpy.data.worlds.new("w"); sc.world = w
    nt = w.node_tree; nt.nodes.clear()
    hdri = opt.get("hdri")
    bg = nt.nodes.new("ShaderNodeBackground"); wo = nt.nodes.new("ShaderNodeOutputWorld")
    if hdri:
        env = nt.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(hdri)
        nt.links.new(env.outputs[0], bg.inputs[0]); bg.inputs["Strength"].default_value = 0.8
    else:
        bg.inputs["Color"].default_value = (0.5, 0.55, 0.6, 1)
    nt.links.new(bg.outputs[0], wo.inputs[0])
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 3.0; sun.rotation_euler = (math.radians(50), 0, math.radians(-35))
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    cam.data.lens = 85
    c = (lo + hi) / 2
    tgt = np.array([c[0], c[1], lo[2] + float(opt.get("height", 0.62)) * H])
    pos = tgt + np.array([0.55 * H, -1.6 * H * float(opt.get("distance", 1.0)), 0.12 * H])
    cam.location = pos
    d = tgt - pos
    cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(d[1], d[0]) - math.pi / 2)
    sc.render.engine = "BLENDER_EEVEE"
    sc.eevee.taa_render_samples = int(opt.get("samples", 32))
    sc.render.resolution_x = sc.render.resolution_y = int(opt.get("px", 640))
    sc.view_settings.view_transform = "AgX"
    os.makedirs(out, exist_ok=True)
    sc.render.filepath = os.path.join(out, "original.png")         # as it is now: glTF import + army colour, without the shared material
    bpy.ops.render.render(write_still=True)
    t = time.time()
    done = apply(objs, up=opt.get("closeup") == "1", used=STATES, lod=opt.get("lod") or None)   # sheet — all layers at once
    t_apply = time.time() - t
    mats = [v for v in done.values() if v is not None]
    groups = {n.node_tree.name for m in mats for n in m.node_tree.nodes if n.type == "GROUP"}
    res = {"apply_s": round(t_apply, 2), "materials": len(mats), "groups": sorted(groups), "frames": []}
    for name, st in DEMO:
        if opt.get("only") and name != opt["only"]:
            continue
        set_states(sc, st)
        sc.render.filepath = os.path.join(out, f"{name}.png")
        t = time.time(); bpy.ops.render.render(write_still=True)
        res["frames"].append([name, round(time.time() - t, 2)])
    json.dump(res, open(os.path.join(out, "result.json"), "w"), ensure_ascii=False, indent=1)
    print("__RESULT__", json.dumps(res, ensure_ascii=False), flush=True)


def demo_time(model_dir, out, opt):
    """Behaviour over time: state keys on frames 1 and 24 (wetness dries, dirt accumulates, blood dries into patches)."""
    import time
    opt = dict(opt, px=opt.get("px", "480"))
    demo(model_dir, os.path.join(out, "_base"), dict(opt, only="clean"))
    sc = bpy.context.scene
    key_states(sc, 1, {"wet": 1.0, "dirt": 0.1})
    key_states(sc, 24, {"wet": 0.0, "dirt": 0.6, "blood": 0.5})
    res = {"frames": []}
    for f in (1, 6, 12, 18, 24):
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, f"frame {f:02d}.png")
        t = time.time(); bpy.ops.render.render(write_still=True)
        res["frames"].append([f"frame {f:02d}", round(time.time() - t, 2)])
    json.dump(res, open(os.path.join(out, "result.json"), "w"), ensure_ascii=False, indent=1)
    print("__RESULT__", json.dumps(res, ensure_ascii=False), flush=True)


if __name__ == "__main__" and "--" in sys.argv:
    a = sys.argv[sys.argv.index("--") + 1:]
    if a and a[0] in ("demo", "time"):
        (demo if a[0] == "demo" else demo_time)(os.path.abspath(a[1]), os.path.abspath(a[2]),
                                                 dict(kv.split("=", 1) for kv in a[3:]))


FRAME_KEYS = ("materials", "state", "state_world", "state_ground", "rain", "closeup", "simplify", "reflections", "weather")


def frame_materials(sc, scene, cams, opt, out):
    """Core hook frame_materials (core/hooks.py; the scene frame from the .blend — core/outputs/bl_color): the scene's materials plan
    (<scene>/materials.yaml and command-line keys; weather=rain|snow|… — a preset) is written by the project's Python (mat_plan resolve), the shared material
    of models and the weather layer of ground and world — apply_plan (far cameras get the light variant). → {"weather", "sky"} of the plan; not needed — None."""
    import subprocess
    if opt.get("materials") == "0" or not (opt.get("light") or any(k in opt for k in FRAME_KEYS[:-1])): return None
    mp = os.path.join(out, "materials_plan.json")
    r = subprocess.run([os.path.join(ROOT, ".venv/bin/python"), "-m", "components.materials.mat_plan", "resolve", scene,
                        *[f"{k}={opt[k]}" for k in FRAME_KEYS if k in opt], f"out={mp}"], capture_output=True, text=True, cwd=ROOT)
    if r.returncode: raise SystemExit("mat_plan.py: " + r.stderr[-600:])
    plan = json.load(open(mp))
    if plan["needed"]:
        apply_plan(plan, sc, cams=cams_from_json(cams), sets_out=os.path.join(ROOT, "out/materials/sets", os.path.basename(scene) + ".json"))
    return {"weather": plan.get("weather"), "sky": plan.get("sky")}
