"""Rain in the frame: streaks of falling drops and splashes on the ground — Geometry Nodes on the object
"sk_rain". The canon is Lagarde, Water drop 2a: drops in layers from the camera, the count follows the rain strength; a drop is a motion-blurred
streak elongated along the fall, with no light of its own (lighter than a dark background, almost invisible against a bright sky); splashes — on the ground near the
camera, only where open to the sky (a ray from above; shelter — the water map sk_water, bl_mat.water_map). The field of view is that of the active
camera (the Active Camera and Camera Info nodes): the drops follow the camera on every frame; the pattern is new on every frame (in 1/12 s
a drop falls ~0.75 m — the eye does not see any link between frames). Strength — the scene property sk_rain (0–1, keyable: bl_mat.set_rain,
key_rain) — through a driver on the modifier input. Drops cast no shadows and give no reflections.
  In Blender: import bl_rain; bl_rain.add(scene)         (bl_mat.apply_plan adds it itself if the plan has rain)"""
import math
import os
import sys

import bpy

from components.materials import bl_mat as BM

NAME, VERSION = "sk_rain", 2
# drop layers: from, to (m from the camera), count at strength 1, opacity — near ones sparse and large, far ones as a veil
LAYERS = ((0.7, 4.0, 700, 0.5), (4.0, 14.0, 4000, 0.38), (14.0, 45.0, 12000, 0.26), (45.0, 120.0, 14000, 0.15))
STREAK = 0.6                                  # streak length, m (exposure blur; at 1/50 s — 0.18 m, longer for a drawn frame)
SPLASH = (1.2, 14.0, 1500)                    # splashes: from, to (m), count at strength 1 (beyond 14 m the splashes cannot be made out)
ASPECT = 0.62                                 # vertical to horizontal field ratio with a margin (16:9 — 0.56)


def _mat_streak():
    m = bpy.data.materials.get("sk_rain_streak") or bpy.data.materials.new("sk_rain_streak")
    nt = m.node_tree; nt.nodes.clear(); b = BM._B(nt)
    uv = b.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="sk_uv")
    a = b.n("ShaderNodeAttribute", attribute_type="GEOMETRY", attribute_name="sk_a")
    s = b.n("ShaderNodeSeparateXYZ"); b.put(s.inputs[0], uv.outputs["Vector"])
    across = b.math("POWER", b.math("SUBTRACT", 1.0, b.math("ABSOLUTE", b.math("SUBTRACT", b.mul(s.outputs["X"], 2.0), 1.0))), 1.5)
    along = b.math("SINE", b.mul(s.outputs["Y"], math.pi))
    alpha = b.mul(a.outputs["Fac"], across, along)
    dif = b.n("ShaderNodeBsdfDiffuse"); dif.inputs["Color"].default_value = (0.9, 0.92, 0.95, 1)
    tr = b.n("ShaderNodeBsdfTranslucent"); tr.inputs["Color"].default_value = (0.7, 0.72, 0.75, 1)
    add = b.n("ShaderNodeAddShader"); b.put(add.inputs[0], dif.outputs[0]); b.put(add.inputs[1], tr.outputs[0])
    tp = b.n("ShaderNodeBsdfTransparent")
    mx = b.n("ShaderNodeMixShader"); b.put(mx.inputs["Fac"], alpha); b.put(mx.inputs[1], tp.outputs[0]); b.put(mx.inputs[2], add.outputs[0])
    out = b.n("ShaderNodeOutputMaterial"); b.put(out.inputs["Surface"], mx.outputs[0])
    m.surface_render_method = "DITHERED"; m.use_transparent_shadow = True
    return m


def _mat_splash():
    m = bpy.data.materials.get("sk_rain_splash") or bpy.data.materials.new("sk_rain_splash")
    nt = m.node_tree; nt.nodes.clear(); b = BM._B(nt)
    dif = b.n("ShaderNodeBsdfDiffuse"); dif.inputs["Color"].default_value = (0.92, 0.94, 0.97, 1)
    tr = b.n("ShaderNodeBsdfTranslucent"); tr.inputs["Color"].default_value = (0.8, 0.82, 0.85, 1)
    add = b.n("ShaderNodeAddShader"); b.put(add.inputs[0], dif.outputs[0]); b.put(add.inputs[1], tr.outputs[0])
    tp = b.n("ShaderNodeBsdfTransparent")
    mx = b.n("ShaderNodeMixShader"); b.put(mx.inputs["Fac"], 0.35); b.put(mx.inputs[1], tp.outputs[0]); b.put(mx.inputs[2], add.outputs[0])
    out = b.n("ShaderNodeOutputMaterial"); b.put(out.inputs["Surface"], mx.outputs[0])
    m.surface_render_method = "DITHERED"
    return m


def _group(scene, res_x, ground_coll):
    ng = bpy.data.node_groups.get(NAME + "_gn")
    if ng is not None:
        bpy.data.node_groups.remove(ng)
    ng = bpy.data.node_groups.new(NAME + "_gn", "GeometryNodeTree")
    ng["version"] = VERSION
    I = ng.interface
    I.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    I.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    s = I.new_socket("Intensity", in_out="INPUT", socket_type="NodeSocketFloat"); s.default_value, s.min_value, s.max_value = 0.6, 0.0, 1.0
    s = I.new_socket("Seed", in_out="INPUT", socket_type="NodeSocketInt"); s.default_value = 0
    b = BM._B(ng)
    gi, go = b.n("NodeGroupInput"), b.n("NodeGroupOutput")
    cam = b.n("GeometryNodeInputActiveCamera")
    oi = b.n("GeometryNodeObjectInfo", transform_space="RELATIVE"); b.put(oi.inputs["Object"], cam.outputs[0])
    ci = b.n("GeometryNodeCameraInfo"); b.put(ci.inputs["Camera"], cam.outputs[0])
    frame = b.math("FLOOR", b.n("GeometryNodeInputSceneTime").outputs["Frame"])   # the pattern is per whole frame: motion blur
    # motion blur evaluates fractions of a frame — a fractional frame would give a new pattern
    idx = b.n("GeometryNodeInputIndex").outputs[0]
    sens = b.n("ShaderNodeSeparateXYZ"); b.put(sens.inputs[0], ci.outputs["Sensor"])
    tanx = b.mul(b.math("DIVIDE", sens.outputs["X"], b.mul(ci.outputs["Focal Length"], 2.0)), 1.12)
    pix = b.math("DIVIDE", sens.outputs["X"], b.mul(ci.outputs["Focal Length"], float(res_x)))   # pixel angle, rad
    rd = BM.rain_dir()
    fall = (-rd[0], -rd[1], -rd[2])
    inten = gi.outputs["Intensity"]

    def vec(x, y, z):
        c = b.n("ShaderNodeCombineXYZ")
        for i, v in enumerate((x, y, z)):
            b.put(c.inputs[i], v)
        return c.outputs[0]

    def rand3(seed_k, lo=(-1.0, -1.0, 0.0), hi=(1.0, 1.0, 1.0)):
        rv = b.n("FunctionNodeRandomValue", data_type="FLOAT_VECTOR")
        rv.inputs["Min"].default_value, rv.inputs["Max"].default_value = lo, hi
        b.put(rv.inputs["ID"], idx)
        b.put(rv.inputs["Seed"], b.add(b.mul(frame, 7919.0), seed_k, gi.outputs["Seed"]))
        s_ = b.n("ShaderNodeSeparateXYZ"); b.put(s_.inputs[0], rv.outputs[0])
        return s_.outputs

    def in_view(r, z):
        """A point in the field of view: r.X, r.Y — −1…1 across the frame, z — m from the camera → world."""
        tp = b.n("FunctionNodeTransformPoint")
        b.put(tp.inputs["Vector"], vec(b.mul(r["X"], tanx, z), b.mul(r["Y"], tanx, z, ASPECT), b.mul(z, -1.0)))
        b.put(tp.inputs["Transform"], oi.outputs["Transform"])
        return tp.outputs[0]

    # streak: a 1×1 quad in XY, the fall runs along Y, normal Z faces the camera; uv goes into the sk_uv attribute
    grid = b.n("GeometryNodeMeshGrid"); grid.inputs["Vertices X"].default_value = grid.inputs["Vertices Y"].default_value = 2
    gs = b.n("GeometryNodeStoreNamedAttribute", data_type="FLOAT_VECTOR", domain="CORNER")
    b.put(gs.inputs["Geometry"], grid.outputs["Mesh"]); gs.inputs["Name"].default_value = "sk_uv"; b.put(gs.inputs["Value"], grid.outputs["UV Map"])
    parts = []
    for k, (z0, z1, n, alpha) in enumerate(LAYERS):
        pts = b.n("GeometryNodePoints"); b.put(pts.inputs["Count"], b.mul(inten, float(n)))
        r = rand3(101.0 * (k + 1))
        z = b.math("POWER", b.add(z0 ** 3, b.mul(r["Z"], z1 ** 3 - z0 ** 3)), 1.0 / 3.0)   # uniform over the layer's volume
        p = in_view(r, z)
        sp = b.n("GeometryNodeSetPosition"); b.put(sp.inputs["Geometry"], pts.outputs[0]); b.put(sp.inputs["Position"], p)
        ra = b.n("FunctionNodeRandomValue"); ra.inputs["Min"].default_value, ra.inputs["Max"].default_value = 0.55, 1.0
        b.put(ra.inputs["ID"], idx); b.put(ra.inputs["Seed"], b.add(b.mul(frame, 31.0), 7.0 + k))
        st_ = b.n("GeometryNodeStoreNamedAttribute", data_type="FLOAT", domain="POINT")
        b.put(st_.inputs["Geometry"], sp.outputs[0]); st_.inputs["Name"].default_value = "sk_a"
        b.put(st_.inputs["Value"], b.mul(ra.outputs[0], alpha))
        to_cam = b.n("ShaderNodeVectorMath", operation="SUBTRACT"); b.put(to_cam.inputs[0], oi.outputs["Location"]); b.put(to_cam.inputs[1], p)
        rot = b.n("FunctionNodeAxesToRotation", primary_axis="Y", secondary_axis="Z")
        rot.inputs["Primary Axis"].default_value = fall; b.put(rot.inputs["Secondary Axis"], to_cam.outputs[0])
        wid = b.math("MAXIMUM", 0.0012, b.mul(pix, z, 1.1))
        ln = b.mul(STREAK, b.add(1.0, b.mul(z, 0.012)))
        ins = b.n("GeometryNodeInstanceOnPoints"); b.put(ins.inputs["Points"], st_.outputs[0]); b.put(ins.inputs["Instance"], gs.outputs[0])
        b.put(ins.inputs["Rotation"], rot.outputs[0]); b.put(ins.inputs["Scale"], vec(wid, ln, 1.0))
        parts.append(ins.outputs[0])
    js = b.n("GeometryNodeJoinGeometry")
    for p_ in reversed(parts):
        b.put(js.inputs[0], p_)
    rs = b.n("GeometryNodeRealizeInstances"); b.put(rs.inputs[0], js.outputs[0])
    sm = b.n("GeometryNodeSetMaterial"); b.put(sm.inputs["Geometry"], rs.outputs[0]); sm.inputs["Material"].default_value = _mat_streak()
    out = [sm.outputs[0]]
    if ground_coll is not None:                 # splashes: points near the camera → a ray down to the ground → ground open to the sky
        z0, z1, n = SPLASH
        pts = b.n("GeometryNodePoints"); b.put(pts.inputs["Count"], b.mul(inten, float(n)))
        r = rand3(977.0)
        z = b.math("SQRT", b.add(z0 ** 2, b.mul(r["Z"], z1 ** 2 - z0 ** 2)))               # uniform over the ground area
        p = in_view(r, z)
        gc = b.n("GeometryNodeCollectionInfo", transform_space="RELATIVE"); gc.inputs["Collection"].default_value = ground_coll
        gr = b.n("GeometryNodeRealizeInstances"); b.put(gr.inputs[0], gc.outputs[0])
        rc = b.n("GeometryNodeRaycast"); b.put(rc.inputs["Target Geometry"], gr.outputs[0])
        up = b.n("ShaderNodeVectorMath", operation="ADD"); b.put(up.inputs[0], p); up.inputs[1].default_value = (0.0, 0.0, 120.0)
        b.put(rc.inputs["Source Position"], up.outputs[0]); rc.inputs["Ray Direction"].default_value = (0.0, 0.0, -1.0)
        rc.inputs["Ray Length"].default_value = 600.0
        hp = b.n("ShaderNodeSeparateXYZ"); b.put(hp.inputs[0], rc.outputs["Hit Position"])
        hn = b.n("ShaderNodeSeparateXYZ"); b.put(hn.inputs[0], rc.outputs["Hit Normal"])
        wm = list(scene.get("sk_wmap", [0.0, 0.0, 0.0, 0.0]))
        shel = 0.0
        if wm[2] > 0 and bpy.data.images.get(BM.WATER_IMG) is not None:
            tx = b.n("GeometryNodeImageTexture", interpolation="Linear", extension="CLIP")
            tx.inputs["Image"].default_value = bpy.data.images[BM.WATER_IMG]
            b.put(tx.inputs["Vector"], vec(b.mul(b.math("SUBTRACT", hp.outputs["X"], wm[0]), wm[2]),
                                            b.mul(b.math("SUBTRACT", hp.outputs["Y"], wm[1]), wm[3]), 0.0))
            sc_ = b.n("FunctionNodeSeparateColor"); b.put(sc_.inputs[0], tx.outputs["Color"])
            shel = sc_.outputs["Blue"]
        bad = b.math("MAXIMUM", b.math("SUBTRACT", 1.0, rc.outputs["Is Hit"]),
                     b.math("MAXIMUM", b.math("LESS_THAN", hn.outputs["Z"], 0.55), b.math("GREATER_THAN", shel, 0.5)))
        dl = b.n("GeometryNodeDeleteGeometry", domain="POINT"); b.put(dl.inputs["Geometry"], pts.outputs[0])
        b.put(dl.inputs["Selection"], b.math("GREATER_THAN", bad, 0.5))
        sp = b.n("GeometryNodeSetPosition"); b.put(sp.inputs["Geometry"], dl.outputs[0]); b.put(sp.inputs["Position"], rc.outputs["Hit Position"])
        cone = b.n("GeometryNodeMeshCone", fill_type="NONE")
        cone.inputs["Vertices"].default_value = 7
        cone.inputs["Radius Top"].default_value, cone.inputs["Radius Bottom"].default_value, cone.inputs["Depth"].default_value = 0.5, 0.12, 0.45
        tg = b.n("GeometryNodeTransform"); b.put(tg.inputs["Geometry"], cone.outputs["Mesh"]); tg.inputs["Translation"].default_value = (0.0, 0.0, 0.22)
        rs_ = rand3(4111.0, (0.018, 0.0, 0.0), (0.045, 6.283, 1.0))
        er = b.n("FunctionNodeEulerToRotation"); b.put(er.inputs[0], vec(0.0, 0.0, rs_["Y"]))
        ins = b.n("GeometryNodeInstanceOnPoints"); b.put(ins.inputs["Points"], sp.outputs[0]); b.put(ins.inputs["Instance"], tg.outputs[0])
        b.put(ins.inputs["Rotation"], er.outputs[0]); b.put(ins.inputs["Scale"], vec(rs_["X"], rs_["X"], b.mul(rs_["X"], 0.8)))
        rz = b.n("GeometryNodeRealizeInstances"); b.put(rz.inputs[0], ins.outputs[0])
        sm2 = b.n("GeometryNodeSetMaterial"); b.put(sm2.inputs["Geometry"], rz.outputs[0]); sm2.inputs["Material"].default_value = _mat_splash()
        out.append(sm2.outputs[0])
    jo = b.n("GeometryNodeJoinGeometry")
    for o_ in reversed(out):
        b.put(jo.inputs[0], o_)
    b.put(go.inputs[0], jo.outputs[0])
    return ng


def add(scene=None, res_x=None):
    """The scene's rain object (one per scene; a repeated call rebuilds it). Strength — a driver from the scene property sk_rain."""
    scene = scene or bpy.context.scene
    old = bpy.data.objects.get(NAME)
    if old is not None:
        bpy.data.objects.remove(old)
    grounds = [o for o in scene.objects if BM._weather_target(o) is True and not o.name.lower().startswith("far_")
               and len(o.data.vertices) < 3_000_000]                 # ground for the splash rays: without the far terrain
    coll = bpy.data.collections.get(NAME + "_ground") or bpy.data.collections.new(NAME + "_ground")
    for o in list(coll.objects):
        coll.objects.unlink(o)
    for o in grounds:
        coll.objects.link(o)
    ng = _group(scene, res_x or scene.render.resolution_x, coll if grounds else None)
    me = bpy.data.meshes.get(NAME) or bpy.data.meshes.new(NAME)
    ob = bpy.data.objects.new(NAME, me)
    scene.collection.objects.link(ob)
    ob["sketch_fx"] = "rain"
    ob.visible_shadow = False
    ob.visible_diffuse = ob.visible_glossy = ob.visible_transmission = ob.visible_volume_scatter = False
    md = ob.modifiers.new("rain", "NODES"); md.node_group = ng
    sid = next(i.identifier for i in ng.interface.items_tree if getattr(i, "name", "") == "Intensity" and getattr(i, "in_out", "") == "INPUT")
    getattr(md.properties.inputs, sid).value = float(scene.get("sk_rain", 0.6))   # Blender 5.2: inputs are properties.inputs.<id>.value
    fc = ob.driver_add(f'modifiers["{md.name}"].properties.inputs.{sid}.value')
    fc.driver.type = "AVERAGE"
    v = fc.driver.variables.new(); v.type = "SINGLE_PROP"
    v.targets[0].id_type = "SCENE"; v.targets[0].id = scene; v.targets[0].data_path = '["sk_rain"]'
    return ob
