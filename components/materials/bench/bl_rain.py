# Rain check: a 60×60 m plot — a slope down to a flat pad, a pit, a mound; the left half is
# snow, the right is soil; a 4×4 m shelter on posts (dry under it). Rain weather: puddles in hollows and pits on the flat, runoff streams,
# slush on the snow, ripples, streaks and splashes. A low camera (as in the game) — the water reflects the sky.
# blender -b -P core/blender/run.py -- components/materials/bench/bl_rain.py -- <output.png> [rain=0.6] [puddles=0.55] [wet=0.9] [dirt=0.3] [frame=1] [px=960]
#   [view=low|top|puddle] [streaks=0] [reflections=0] [samples=16] [repeat=1 — time of the second frame, without building shaders]
#   [layers=N — ground from N BSDFs mixed by noise, as in bl_terrain_surf] [images=N — images in the ground material] [budget=N]
import json, math, os, sys, time
import bpy
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))   # this script lives in components/materials/bench/
from components.materials import bl_mat
a = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(a[0]); opt = dict(kv.split("=", 1) for kv in a[1:])
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.mesh.primitive_grid_add(x_subdivisions=240, y_subdivisions=240, size=60)
ob = bpy.context.object; ob.name = "terrain"; me = ob.data
co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
x, y = co[:, 0], co[:, 1]
z = np.where(y > 6, (y - 6) * 0.35, 0.0)                                  # slope to the north ~19°
z += 1.2 * np.exp(-((x + 14) ** 2 + (y + 12) ** 2) / (2 * 4.0 ** 2))       # mound
z -= 0.35 * np.exp(-((x - 8) ** 2 + (y + 6) ** 2) / (2 * 1.6 ** 2))        # pit
z += 0.02 * np.sin(x * 0.9) * np.cos(y * 0.7)                              # slight unevenness
co[:, 2] = z; me.vertices.foreach_set("co", co.ravel()); me.update(); bpy.ops.object.shade_smooth()
m = bpy.data.materials.new("ground_test"); nt = m.node_tree; p = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
gc = nt.nodes.new("ShaderNodeNewGeometry"); sx = nt.nodes.new("ShaderNodeSeparateXYZ"); nt.links.new(gc.outputs["Position"], sx.inputs[0])
mr = nt.nodes.new("ShaderNodeMapRange"); mr.inputs["From Min"].default_value, mr.inputs["From Max"].default_value = -1.0, 1.0
nt.links.new(sx.outputs["X"], mr.inputs["Value"])
mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"
ins = {s.identifier: s for s in mx.inputs}
ins["A_Color"].default_value = (0.54, 0.545, 0.56, 1); ins["B_Color"].default_value = (0.24, 0.17, 0.11, 1)   # snow | soil
nt.links.new(mr.outputs["Result"], ins["Factor_Float"])
nt.links.new(next(s for s in mx.outputs if s.identifier == "Result_Color"), p.inputs["Base Color"])
p.inputs["Roughness"].default_value = 0.85; me.materials.append(m)
col_in = p.inputs["Base Color"].links[0].from_socket                    # images in the material (as with scans: Metal — 16 textures)
for k in range(int(opt.get("images", 0))):
    im = bpy.data.images.new(f"img{k}", 4, 4); im.pixels.foreach_set(np.full(64, 0.5 + 0.03 * k, np.float32))
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = im; tx.interpolation = "Closest"
    mx_ = nt.nodes.new("ShaderNodeMix"); mx_.data_type = "RGBA"; ins_ = {x.identifier: x for x in mx_.inputs}
    ins_["Factor_Float"].default_value = 0.02; nt.links.new(col_in, ins_["A_Color"]); nt.links.new(tx.outputs["Color"], ins_["B_Color"])
    col_in = next(x for x in mx_.outputs if x.identifier == "Result_Color")
nt.links.new(col_in, p.inputs["Base Color"])
for k in range(1, int(opt.get("layers", 1))):                           # layered ground (as in bl_terrain_surf): one BSDF per layer
    p2 = nt.nodes.new("ShaderNodeBsdfPrincipled"); p2.inputs["Base Color"].default_value = (0.3 + 0.1 * k, 0.25, 0.2, 1)
    p2.inputs["Roughness"].default_value = 0.8
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.2 + 0.1 * k
    out_ = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"); src = out_.inputs["Surface"].links[0].from_socket
    mxs = nt.nodes.new("ShaderNodeMixShader"); nt.links.new(nz.outputs["Fac"], mxs.inputs[0])
    nt.links.new(src, mxs.inputs[1]); nt.links.new(p2.outputs[0], mxs.inputs[2]); nt.links.new(mxs.outputs[0], out_.inputs["Surface"])
for px_, py_ in ((-6, -2), (-2, -2), (-6, 2), (-2, 2)):                    # shelter on posts
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=2.6, location=(px_, py_, 1.3))
bpy.ops.mesh.primitive_cube_add(size=1, location=(-4, 0, 2.7)); bpy.context.object.scale = (5.0, 5.0, 0.2)
wm = bpy.data.materials.new("wood"); wm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.2, 0.12, 0.07, 1)
for o in sc.objects:
    if o.name != "terrain": o.data.materials.append(wm)
bl_mat.set_states(sc, {"wet": float(opt.get("wet", 0.9)), "puddles": float(opt.get("puddles", 0.55)),
                       "dirt": float(opt.get("dirt", 0.3))}, ground=True)
bl_mat.set_states(sc, {"wet": float(opt.get("wet", 0.9)) * 0.9}, world=True)
bl_mat.set_rain(sc, float(opt.get("rain", 0.6)))
w = bpy.data.worlds.new("w"); sc.world = w; wn = w.node_tree; bg = wn.nodes["Background"]   # overcast sky: lighter toward the zenith
tc = wn.nodes.new("ShaderNodeTexCoord"); sxyz = wn.nodes.new("ShaderNodeSeparateXYZ"); wn.links.new(tc.outputs["Generated"], sxyz.inputs[0])
cr = wn.nodes.new("ShaderNodeValToRGB"); wn.links.new(sxyz.outputs["Z"], cr.inputs["Fac"])
cr.color_ramp.elements[0].position, cr.color_ramp.elements[0].color = 0.45, (0.25, 0.26, 0.28, 1)
cr.color_ramp.elements[1].position, cr.color_ramp.elements[1].color = 0.75, (0.75, 0.78, 0.82, 1)
wn.links.new(cr.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 1.0
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 0.6; sun.rotation_euler = (math.radians(50), 0, math.radians(30))   # overcast: weak sun
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
if opt.get("view", "low") == "top":
    cam.data.type = "ORTHO"; cam.data.ortho_scale = 44; cam.location = (0, 0, 40); cam.rotation_euler = (0, 0, 0)
elif opt.get("view") == "puddle":                                           # a puddle in the pit up close: ripples, splashes
    cam.data.lens = 35; cam.location = (8, -10.5, 1.6)
    d = np.array([8.0, -6.0, -0.3]) - np.array(cam.location)
    cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(d[1], d[0]) - math.pi / 2)
else:
    cam.data.lens = 30; cam.location = (6, -24, 2.2)
    d = np.array([-2.0, 0.0, 0.3]) - np.array(cam.location)
    cam.rotation_euler = (math.atan2(math.hypot(d[0], d[1]), -d[2]), 0, math.atan2(d[1], d[0]) - math.pi / 2)
if "budget" in opt:                                                      # test: the water map as an image at any number of images
    bl_mat.IMG_BUDGET = int(opt["budget"])
t = time.time(); n = bl_mat.apply_weather(sc); dt = time.time() - t
if opt.get("reflections") == "0":
    sc.eevee.use_raytracing = False
from components.materials import bl_rain
t = time.time(); rob = bl_rain.add(sc) if float(opt.get("rain", 0.6)) > 0 and opt.get("streaks", "1") == "1" else None; dt2 = time.time() - t
sc.frame_set(int(opt.get("frame", 1)))
sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = int(opt.get("samples", 16))
px = int(opt.get("px", 960)); sc.render.resolution_x, sc.render.resolution_y = px, px * 9 // 16 if opt.get("view", "low") != "top" else px
sc.view_settings.view_transform = "AgX"; sc.render.filepath = out
t = time.time(); bpy.ops.render.render(write_still=True); dr = time.time() - t
if opt.get("repeat") == "1":                                              # second frame — without building shaders (pixel cost)
    t = time.time(); bpy.ops.render.render(write_still=True); dr = time.time() - t
dg = bpy.context.evaluated_depsgraph_get()
nr = len(rob.evaluated_get(dg).data.vertices) if rob else 0
im = bpy.data.images.get(bl_mat.WATER_IMG) or bl_mat.water_image(); W_, H_ = im.size
px_ = np.empty(W_ * H_ * 4, np.float32); im.pixels.foreach_get(px_); px_ = px_.reshape(H_, W_, 4)
x0_, y0_, iw, ih = sc.get("sk_wmap", [0.0, 0.0, 0.0, 0.0])
at = lambda x, y: px_[min(max(int((y - y0_) * ih * H_), 0), H_ - 1), min(max(int((x - x0_) * iw * W_), 0), W_ - 1)]
under = float(np.mean([at(x, y)[2] for x in np.linspace(-5.5, -2.5, 7) for y in np.linspace(-1.5, 1.5, 7)]))
around = float(np.mean([at(x, y)[2] for x in np.linspace(4, 12, 9) for y in np.linspace(-14, -10, 5)]))
ri = bpy.data.images.load(out); rp = np.empty(ri.size[0] * ri.size[1] * 4, np.float32); ri.pixels.foreach_get(rp); rp = rp.reshape(-1, 4)
R_, G_, B_ = rp[:, 0], rp[:, 1], rp[:, 2]                                     # purple (dark after lighting) — the shader failed to build
mag = float(((R_ - G_ > 0.2) & (B_ - G_ > 0.2) & (np.abs(R_ - B_) < 0.1)).mean())
print("__RESULT__", json.dumps({"n_materials": n, "bsdf": bl_mat._STAT.get("weather_bsdf", 0), "violet": round(mag, 4),
                                "weather_s": round(dt, 2), "rain_s": round(dt2, 2), "render_s": round(dr, 2),
                                "n_vertices_rain": nr, "shelter_under_awning": round(under, 2), "shelter_around": round(around, 2),
                                "pit_mm": round(float(1.0 - at(8, -6)[1]) * 30, 2), "water": bl_mat._STAT.get("water")}, ensure_ascii=False))
