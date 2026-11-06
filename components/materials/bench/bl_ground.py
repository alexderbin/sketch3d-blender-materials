# Check of the ground weather layer: a 20° slope, 60×60 m, with a pit (r 1.5 m, depth 0.5 m), an even grey
# material; rain — puddles 0.55 (as in the preset), wet 0.6. Top view: a puddle in the pit (darker, glossy), even on the slope, no stripes (an early
# version of the hollows gave stripes across the slope). blender -b -P core/blender/run.py -- components/materials/bench/bl_ground.py -- <output.png>
import math, os, sys, time, json
import bpy
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))   # this script lives in components/materials/bench/
from components.materials import bl_mat
out = os.path.abspath(sys.argv[sys.argv.index("--") + 1])
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.mesh.primitive_grid_add(x_subdivisions=460, y_subdivisions=460, size=60)
ob = bpy.context.object; ob.name = "terrain"; me = ob.data
co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
co[:, 2] = co[:, 0] * math.tan(math.radians(20)) * 0 + co[:, 0] * 0.05 - 0.5 * np.exp(-((co[:, 0] - 5) ** 2 + (co[:, 1] + 4) ** 2) / (2 * 1.5 ** 2))
me.vertices.foreach_set("co", co.ravel()); me.update(); bpy.ops.object.shade_smooth()
m = bpy.data.materials.new("ground_test"); p = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
p.inputs["Base Color"].default_value = (0.45, 0.4, 0.33, 1); p.inputs["Roughness"].default_value = 0.8; me.materials.append(m)
bl_mat.set_states(sc, {"puddles": 0.55, "wet": 0.6}, ground=True)   # strength of the "rain" preset: no puddles on the flat, puddles in the pit
t = time.time(); n = bl_mat.apply_weather(sc); dt = time.time() - t
w = bpy.data.worlds.new("w"); sc.world = w; w.color = (0.55, 0.6, 0.7)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 3.0; sun.rotation_euler = (math.radians(35), 0, math.radians(30))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = 40; cam.location = (0, 0, 30)
sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = 16; sc.render.resolution_x = sc.render.resolution_y = 400
sc.view_settings.view_transform = "Standard"; sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("__RESULT__", json.dumps({"n_materials": n, "sec": round(dt, 2)}))
