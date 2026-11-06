"""Scene materials by description: scenes/<name>/materials.yaml → a JSON plan for Blender (components/materials/bl_mat.py apply_plan; Blender
has no PyYAML — the plan is written by the project's Python, as with far.py). The names of the description and of the command-line keys are English.
  .venv/bin/python -m components.materials.mat_plan resolve <scene folder> out=<plan.json> [weather=rain] [state=…] [state_world=…]
      [state_ground=…] [rain=0–1] [closeup=1] [simplify=…] [reflections=off]
Command-line keys override the description. The description (all keys are optional):
  state: {dirt: 0.4, wear: 0.3}            # for fighters and vehicles (faction models), 0–1
  weather: rain                            # the whole frame at once: clear, rain, storm, after_rain, snow, fog, dust_storm, after_battle,
                                           # mud — states for fighters, the world, the ground and the far sky weather (the explicit keys below go on top)
  world: {dust: 0.3}                       # for buildings, objects, map pieces
  ground: {puddles: 0.5}                   # for terrain and map ground, rocks, walls: wet, puddles (in hollows), dirt, dust, snow, burn
  rain: 0.6                                # raining now, 0–1: puddle ripples, streaks and splashes in the frame (wet and puddles are ground states)
  closeup: true                            # upscaled texture sets where available (library/textures/closeup)
  simplify: auto                           # auto — a light variant for far objects (default); off — full for all; lite — light for all
  reflections: auto                        # auto — with wet and puddles, Eevee screen-space ray tracing (buildings in puddles); off — without it
  groups:                                  # additions for objects: name (fnmatch pattern) or tag sketch_kind
    - {name: "chaos_*", state: {blood: 0.5}}
    - {tag: figure, state: {wet: 0.4}}
  timeline:                                # scene keys over time (frame → fighters' state; world: — for buildings)
    - {frame: 1, state: {wet: 1.0}}
    - {frame: 120, state: {wet: 0.2, dirt: 0.6}, world: {dust: 0.4}, ground: {puddles: 0.2}, rain: 0}
States: wet, dirt, dust, rust, wear, blood, burn (soot), snow; the ground also has puddles."""
import json
import os
import sys

import yaml

STATES = ("wet", "dirt", "dust", "rust", "wear", "blood", "burn", "snow")
GROUND = ("wet", "puddles", "dirt", "dust", "snow", "burn")          # for the ground (terrain, map ground) and other non-library objects
# Weather over the whole frame: states for fighters, the world, the ground and the sky weather for the far terrain (core/light/far.py WEATHER), if the sky is not set
# by the key weather=<clear|overcast|…>. The values are initial ones, checked against a sheet.
WEATHER = {
    "clear": dict(fighters={}, world={}, ground={}, sky="clear"),
    "rain": dict(fighters={"wet": 0.85, "dirt": 0.25}, world={"wet": 0.8, "dirt": 0.15},
                  ground={"wet": 0.9, "puddles": 0.55, "dirt": 0.3}, sky="rain", rain=0.6),
    "storm": dict(fighters={"wet": 0.95, "dirt": 0.3}, world={"wet": 0.9, "dirt": 0.2},
                  ground={"wet": 1.0, "puddles": 0.75, "dirt": 0.35}, sky="storm", rain=1.0),
    "after_rain": dict(fighters={"wet": 0.35, "dirt": 0.3}, world={"wet": 0.3}, ground={"wet": 0.55, "puddles": 0.4, "dirt": 0.3},
                        sky="overcast"),
    "snow": dict(fighters={"snow": 0.45, "wet": 0.15}, world={"snow": 0.6}, ground={"snow": 0.85}, sky="overcast"),
    "fog": dict(fighters={"wet": 0.25}, world={"wet": 0.2}, ground={"wet": 0.3}, sky="mist"),
    "dust_storm": dict(fighters={"dust": 0.7, "dirt": 0.15}, world={"dust": 0.75}, ground={"dust": 0.6}, sky="haze"),
    "after_battle": dict(fighters={"dirt": 0.45, "wear": 0.35, "dust": 0.3, "burn": 0.25, "blood": 0.2},
                      world={"dust": 0.4, "burn": 0.3, "dirt": 0.2}, ground={"burn": 0.4, "dirt": 0.2, "dust": 0.25}, sky="war_smoke"),
    "mud": dict(fighters={"dirt": 0.7, "wet": 0.45}, world={"dirt": 0.4, "wet": 0.4},
                      ground={"dirt": 0.75, "wet": 0.6, "puddles": 0.45}, sky="overcast", rain=0.15),
}
LOD = {"auto": "auto", "off": None, "lite": "lite"}                    # simplification: auto — a light variant for far objects, off — none, lite — for all


def states(d, where, names=STATES):
    out = {}
    for k, v in (d or {}).items():
        if k not in names:
            raise SystemExit(f"{where}: state '{k}' does not exist; available: {', '.join(names)}")
        v = float(v)
        if not 0 <= v <= 1:
            raise SystemExit(f"{where}: {k} = {v} — must be 0–1")
        out[k] = v
    return out


def rain(v, where):
    v = float(v or 0.0)
    if not 0 <= v <= 1:
        raise SystemExit(f"{where} = {v} — must be 0–1")
    return v


def parse_line(s):
    return {kv.split(":")[0].strip(): float(kv.split(":")[1] or 1)
            for kv in filter(None, (s or "").replace(";", ",").split(","))}


def weather_presets():
    """Core hook weather (core/hooks.py; the "frames" output — core/outputs/frames.py): materials weather presets {name: {sky, …}}."""
    return WEATHER


def resolve(scene_dir, kv):
    p = os.path.join(scene_dir, "materials.yaml")
    d = (yaml.safe_load(open(p)) or {}) if os.path.exists(p) else {}
    d = d.get("materials", d)
    wname = kv.get("weather") if kv.get("weather") in WEATHER else d.get("weather")
    if wname is not None and wname not in WEATHER:
        raise SystemExit(f"weather '{wname}' does not exist; available: {', '.join(WEATHER)}")
    W = WEATHER.get(wname, dict(fighters={}, world={}, ground={}, sky=None))
    plan = {
        "weather": wname, "sky": W["sky"],
        "state": {**W["fighters"], **states(d.get("state"), "state")},
        "world": {**W["world"], **states(d.get("world"), "world")},
        "ground": {**W["ground"], **states(d.get("ground"), "ground", GROUND)},
        "rain": rain(d.get("rain", W.get("rain", 0.0)), "rain"),
        "closeup": bool(d.get("closeup", False)),
        "simplify": LOD.get(str(d.get("simplify", "auto")).lower(), "auto"),
        "reflections": str(d.get("reflections", "auto")).lower() not in ("off", "0", "false"),
        "groups": [],
        "timeline": [],
        "source": p if os.path.exists(p) else None,
        "folder": os.path.abspath(scene_dir),
    }
    for i, g in enumerate(d.get("groups") or []):
        if not (g.get("name") or g.get("tag")):
            raise SystemExit(f"groups[{i}]: need the key name (pattern) or tag (sketch_kind)")
        plan["groups"].append({"name": g.get("name"), "tag": g.get("tag"),
                               "state": states(g.get("state"), f"groups[{i}]")})
    for i, t in enumerate(d.get("timeline") or []):
        plan["timeline"].append({"frame": int(t["frame"]), "state": states(t.get("state"), f"time[{i}]"),
                              "world": states(t.get("world"), f"time[{i}]"), "ground": states(t.get("ground"), f"time[{i}]", GROUND),
                              **({"rain": rain(t["rain"], f"time[{i}].rain")} if "rain" in t else {})})
    if "state" in kv:
        plan["state"] = states(parse_line(kv["state"]), "key state=")
    if "state_world" in kv:
        plan["world"] = states(parse_line(kv["state_world"]), "key state_world=")
    if "state_ground" in kv:
        plan["ground"] = states(parse_line(kv["state_ground"]), "key state_ground=", GROUND)
    if "rain" in kv:
        plan["rain"] = rain(kv["rain"], "key rain=")
    if "closeup" in kv:
        plan["closeup"] = kv["closeup"] == "1"
    if "simplify" in kv:
        plan["simplify"] = LOD.get(kv["simplify"].lower(), "auto")
    if "reflections" in kv:
        plan["reflections"] = kv["reflections"].lower() not in ("off", "0", "false")
    plan["needed"] = bool(plan["source"] or kv.get("materials") == "1" or kv.get("weather") in WEATHER
                         or any(k in kv for k in ("state", "state_world", "state_ground", "rain", "closeup", "simplify")))
    return plan


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2 or a[0] != "resolve":
        raise SystemExit(__doc__)
    kv = dict(x.split("=", 1) for x in a[2:])
    plan = resolve(a[1], kv)
    out = kv.get("out")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(plan, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(plan, ensure_ascii=False))
