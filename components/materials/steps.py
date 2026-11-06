"""Entry of the "Materials and weather" component for the step-by-step build (core/blender/build.py): the "weather" step.
plan (project Python): the run scene's scenes/<name>/materials.yaml (no file — the default plan) + the step knob `weather` — a preset
of mat_plan.WEATHER (rain, storm, after_rain, snow, fog, dust_storm, after_battle, mud, clear) and explicit knobs on top
(state, state_world, state_ground, rain, closeup, simplify, reflections — as in mat_plan.py resolve) → mat_plan.resolve → plan.json.
A sky weather (far.WEATHER) in place of a materials preset is a key for the sky only: the materials follow the plan from materials.yaml, the sky is as given.
build (Blender): bl_mat.apply_plan on the open scene — states, the weather layer of all non-library materials (ground, rocks, walls,
trees, buildings: apply_weather), water by terrain (the water map), wet and puddles with ray tracing, rain (bl_rain), the shared material of library
models (apply, a light variant for far ones by the scene's cameras).
ORDER: the step goes after all the steps that make geometry and materials (terrain, building, plants, masonry, textures): the weather layer
is inserted before the Principled of a finished material, so new objects after it would stay dry. Light and sky go after weather: the node's sky
takes the weather from gives (sky_weather = mat_plan.WEATHER[…]["sky"], as in frames.py --weather); the old path did the same.
gives: weather, sky_weather (for the "light" step); log statistics — states (layers set in the scene), objects (with a material), rain …"""
import json, os

OPTIONS = ("state", "state_world", "state_ground", "rain", "closeup", "simplify", "reflections", "materials")   # knobs are the options of mat_plan.resolve


def plan_weather(M, keys, ctx):
    from components.materials import mat_plan
    w = keys.get("weather")
    kv = {k: str(v) for k, v in keys.items() if k in OPTIONS}
    if w in mat_plan.WEATHER: kv["weather"] = w                     # a materials preset (rain, storm …) — together with its sky
    plan = mat_plan.resolve(ctx["scene"], kv)
    if w and w not in mat_plan.WEATHER:                               # a sky weather (cumulus, overcast, mist …): the materials follow the scene description
        plan["sky"] = str(w)
    plan["needed"] = bool(plan["needed"] or kv)
    plan["sky_weather"] = plan.get("sky") or ""                     # the sky weather for light even without a build (draft stage: plan only)
    return plan


def _cams(sc):
    """Scene cameras (position, tan of half the field) — for choosing the far variant of the material; the light node's far cameras do not count."""
    out = []
    for o in sc.objects:
        if o.type == "CAMERA" and not o.name.startswith("far_"):
            out.append((o.matrix_world.translation.copy(), (o.data.sensor_width or 36.0) / (2 * o.data.lens)))
    return out


def build_weather(ctx):
    import bpy
    from components.materials import bl_mat
    sc, plan = ctx["sc"], dict(ctx["plan"])
    plan["folder"] = ctx["scene"]                                     # the site's relief.npz (the water map) comes from the run's scene, not from another process's plan
    ret = {"weather": plan.get("weather") or "", "sky_weather": plan.get("sky") or "", "needed": bool(plan.get("needed"))}
    if not plan.get("needed"):
        return ret | {"states": "", "objects": 0}
    bl_mat.apply_plan(plan, sc, cams=_cams(sc), sets_out=os.path.join(ctx["out"], "sets.json"))
    st = bl_mat._STAT
    n = sum(1 for o in sc.objects if o.type == "MESH" and any(s.material is not None and (s.material.get("sk_mat") or s.material.get("sk_weather"))
                                                                for s in o.material_slots))
    w = [k for k in (st.get("used") or [])] + [k for k in (st.get("weather_used") or []) if k not in (st.get("used") or [])]
    return ret | {"states": ",".join(w), "objects": n, "materials": int(st.get("materials", 0)), "ground_world_materials": int(st.get("weather_materials", 0)),
                  "rain": float(plan.get("rain") or 0.0), "rain_streaks": sum(1 for o in sc.objects if o.name.lower().startswith("sk_rain")),
                  "sec": float(st.get("apply_s", 0)) + float(st.get("weather_s", 0))}
