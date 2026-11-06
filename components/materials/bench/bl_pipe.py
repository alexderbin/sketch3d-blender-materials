# Check for repetition of state masks on a flat surface (on a pipe, dirt and dust used to lay down as an identical grid).
# A pipe 6 m × r 0.35 m lies along X, the texture is an even grey (like an asset with a uniform texture), the state applies to the whole scene.
# blender -b -P core/blender/run.py -- components/materials/bench/bl_pipe.py -- <folder with bl_mat.py> <output.png> [state=dirt:0.6,dust:0.5] [px=1280]
import math, os, sys
import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
mod_dir, out = os.path.abspath(a[0]), os.path.abspath(a[1])
opt = dict(kv.split("=", 1) for kv in a[2:])
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))   # this script lives in components/materials/bench/
sys.path.insert(0, mod_dir)
from components.materials import bl_mat

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.35, depth=6.0, rotation=(0, math.pi / 2, 0))
ob = bpy.context.object
bpy.ops.object.shade_smooth()
bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.uv.smart_project(); bpy.ops.object.mode_set(mode="OBJECT")


def flat(name, rgb):
    im = bpy.data.images.new(name, 64, 64)
    im.pixels.foreach_set(np.tile(np.array([*rgb, 1.0], np.float32), 64 * 64))
    im.pack()
    return im


m = bpy.data.materials.new("pipe")
nt = m.node_tree
p = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
b = nt.nodes.new("ShaderNodeTexImage"); b.image = flat("pipe_dif", (0.45, 0.45, 0.46)); b.label = "BASE COLOR"
nt.links.new(b.outputs["Color"], p.inputs["Base Color"])
n = nt.nodes.new("ShaderNodeTexImage"); n.image = flat("pipe_nrm", (0.5, 0.5, 1.0)); n.label = "NORMAL MAP"
nm = nt.nodes.new("ShaderNodeNormalMap"); nt.links.new(n.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], p.inputs["Normal"])
ob.data.materials.append(m)

bl_mat.set_states(sc, bl_mat.parse_states(opt.get("state", "dirt:0.6,dust:0.5")))   # states — before apply:
bl_mat.apply([ob])                                                                          # the given layers go into the shader
w = bpy.data.worlds.new("w"); sc.world = w; w.color = (0.6, 0.62, 0.66)
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
sun.data.energy = 3.0; sun.rotation_euler = (math.radians(40), 0, math.radians(20))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = "ORTHO"; cam.data.ortho_scale = 6.2
cam.location = (0, -6, 2.2); cam.rotation_euler = (math.radians(70), 0, 0)
sc.render.engine = "BLENDER_EEVEE"; sc.eevee.taa_render_samples = 16
px = int(opt.get("px", 1280)); sc.render.resolution_x, sc.render.resolution_y = px, px // 4
sc.view_settings.view_transform = "AgX"
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("__RESULT__", out)
