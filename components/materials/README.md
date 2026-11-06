# Materials — procedural weather and weathering for Blender: rain, puddles, wet ground, snow, dirt and rust

Materials is a procedural weather and weathering system for Blender, written in Python: it makes a finished scene rainy,
wet, snowy, dusty, muddy or burnt and ages models with rust, dirt and wear. A weather layer is inserted in front of the
Principled BSDF of every material that is already there — ground, rocks, walls, trees, buildings — and imported glTF
models get one shared surface material with eight states; all of it is driven by a few scene-wide numbers that can be
keyed over time. Where water stands and runs is computed from the relief, not painted with noise: puddles fill
depressions by Priority-Flood (Barnes et al. 2014) and rivulets follow multiple-flow-direction accumulation (Freeman
1991). The `weather` step runs after every geometry and material step and before `light`, and gives the sky weather to
the light node.

**Use it to:**

- Make a Blender scene rainy — wet ground, puddles in the hollows, rain streaks and splashes:
  `weather: {weather: rain, rain: 0.8}`
- Switch the whole frame to snow, mud, fog, a dust storm, a downpour or a battlefield aftermath:
  `weather: {weather: after_battle}`, or `snow`, `mud`, `fog`, `dust_storm`, `storm`
- Age and dirty a scene or single models with wet, dirt, dust, rust, wear, blood, burn and snow states from 0 to 1:
  `weather: {state: "rust:0.5,wear:0.3", state_world: "dust:0.4", state_ground: "burn:0.4"}`
- Find where puddles and rivulets form on a site's heightmap (`relief.npz`) instead of painting noise:
  `.venv/bin/python -m components.materials.mat_water <relief.npz> <out.png>`
- Animate wetness, dirt and rain over time, such as a street drying after a shower: `timeline` in the scene's
  `materials.yaml`
- Give the sky the same weather automatically — the step hands `sky_weather` to `light`:
  `weather: {weather: storm}` followed by `light: {time: dusk}`

## Features

- **9 weather presets for the whole frame**: clear, rain, storm, after_rain, snow, fog, dust_storm, after_battle, mud.
  A preset sets the states of models, the world and the ground, the rain strength and the sky weather of the light node
  (rain → rain, storm → storm, fog → mist, after_battle → war_smoke); explicit keys override it.
- **Weather layer** (ground and world): wet, puddles, dirt, dust, snow, burn — dirt where people walk and water runs,
  dust as a film, snow by slope, burn in sparse craters, snow under rain turns to slush.
- **Wet surfaces and puddles** by Lagarde's wet-surface model: darker albedo, lower roughness, a water film on flat
  ground; puddles where depressions fill with rain volume (roads and pads hold water, soil absorbs it), a wet rim,
  ripples while it rains; wet ground switches on Eevee screen-space ray tracing, so buildings mirror in puddles.
- **Water map from the relief** (`mat_water.py`): depth to spill (Priority-Flood), flow by MFD (Freeman 1991, Quinn et
  al. 1991) over 1–8 m micro-relief (ruts on roads and pads), shelter from a top-down depth map along the rain (dry under
  roofs and rocks); from the site's `relief.npz`, else from the ground mesh.
- **Rain in the frame** (`bl_rain.py`): Geometry Nodes streaks in layers in front of the active camera — count by rain
  strength, motion-blurred, no own light (Lagarde's rain streaks) — and splashes on ground open to the sky;
  a new pattern every frame; the strength is a keyable scene property.
- **Shared surface material** for glTF-imported models (`sk_surface`, labels BASE COLOR, NORMAL MAP, METALLIC
  ROUGHNESS…): one node tree for every model, so Eevee compiles it once; edges and cavities from the normal map, up-facing
  and bottom masks, AO; eight states — wear down to metal, rust, dirt, dust, burn, blood, snow, wet — with physically
  based values; 3D object noise instead of tiling; a lighter variant for far objects.
- **Scene description** `materials.yaml`: `state`, `weather`, `world`, `ground`, `rain`, `groups` (by object name
  pattern or `sketch_kind` tag), `timeline` (frame → states, smooth between keys), `closeup`, `simplify`, `reflections`.
- **Only the layers in use** enter the shader; shader groups and the water map are cached.

## How it works

Plan (Python, `mat_plan.resolve`): `materials.yaml` of the scene (optional) + step keys → `plan.json`. Build (Blender,
`bl_mat.apply_plan`): scene and object states, keys over time, the weather layer on every material that is not a model
(`apply_weather`; ground objects read the ground states, others the world states; materials without a Principled BSDF,
transparent or emissive ones stay as they are), the water map as an image in world XY (a vertex attribute when a
material already holds more than 9 images), rain, and the shared material on models (`apply`). Hooks:
`frame_materials` — materials and weather in lit frames of a `.blend` scene; `weather` — presets for `--weather` of the
frames output. Textures of building surfaces belong to the core surface engine (`core/surface/`).

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `weather` | — (+ `site` relief for the water map) → `materials`; gives `weather`, `sky_weather` | `weather` (clear), `state`, `state_world`, `state_ground` (as `"wet:0.6,dirt:0.2"`), `rain` (0–1), `closeup`, `simplify` (auto \| off \| lite), `reflections` (auto \| off), `materials` |

`weather` also takes a sky weather of the light node (overcast, mist…): then only the sky changes. Plan only in the
`sketch` profile. All keys with descriptions: [component.yaml](/components/materials/component.yaml)

## Usage

```yaml
steps:   # as core/manager/pipelines/castle_weather.yaml
  - relief: {preset: fortress_hill, seed: 7}
  - building: {type: castle, culture: german, scale: medium, seed: 7}
  - camera: {view: gate, azimuth: 200, elevation: 10, distance: 170, orbit: 8, res: 640x360}
  - masonry: {detail: cam}
  - textures
  - weather: {weather: rain}            # or {weather: after_battle, state_ground: "burn:0.6,dust:0.3"}
  - light: {time: night}
```

```yaml
# <scene folder>/materials.yaml — drying after a shower
weather: after_rain
timeline:
  - {frame: 1, ground: {wet: 1.0, puddles: 0.6}, rain: 0.6}
  - {frame: 120, ground: {wet: 0.3, puddles: 0.2}, rain: 0}
```

```sh
.venv/bin/python -m components.materials.mat_plan resolve <scene folder> out=<plan.json> [weather=rain] [rain=0.5]
.venv/bin/python -m components.materials.mat_water <relief.npz> <out.png>      # blue — puddles, cyan — rivulets
blender -b -P core/blender/run.py -- components/materials/bench/bl_rain.py -- <out.png> [rain=0.6] [view=low|top|puddle]
blender -b -P core/blender/run.py -- components/materials/bench/bl_ground.py -- <out.png>
blender -b -P core/blender/run.py -- components/materials/bench/bl_pipe.py -- components/materials <out.png> [state=dirt:0.6]
.venv/bin/python -m components.materials.bench.repeat <pipe image>
```

The stands: `bench/bl_rain.py` — a 60 × 60 m slope with a pit, snow and soil halves and a shelter; `bench/bl_ground.py`
— a puddle in a pit and no stripes on the slope; `bench/bl_pipe.py` + `bench/repeat.py` — tiling of state masks on a pipe.

## Driving it from an AI agent

One `weather` line after all geometry and textures and before `light`:

```yaml
  - weather: {weather: after_battle, state_ground: "burn:0.6,dust:0.3"}
  - light: {time: dusk}                      # takes the sky weather (war_smoke) from the weather step
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry`. All keys, defaults and
  values: `components/materials/component.yaml`; a scene's `materials.yaml` takes `state`, `weather`, `world`, `ground`,
  `rain`, `groups`, `timeline`, `closeup`, `simplify`, `reflections`.
- A run prints `__RESULT__{"pipeline", "out", "failures", …}` as its last line; the run folder holds `journal.md` and,
  with the `frames` output, a contact sheet `contact.jpg`. The step passes `weather` and `sky_weather` on to `light`.
- Request to keys: "a muddy road after a downpour" → `weather: mud`; "heavy rain at night" → `weather: storm` and
  `light: {time: night}`; "a half-wet yard" → `state_ground: "wet:0.5"`; "a burnt, dusty battlefield" → `after_battle`.

## Files

| File | Role |
|---|---|
| `steps.py`, `mat_plan.py`, `api.py` | step entry; description and presets → plan; public entry |
| `bl_mat.py` | Blender: shared material, states, weather layer, water map, keys over time |
| `mat_water.py`, `bl_rain.py` | water on the ground; rain streaks and splashes |
| `mat_layers.py`, `mat_sheet.py` | layer texture downloader; contact sheet of states |

## Assets

Rust (Rust009), ambientCG, CC0: `.venv/bin/python -m components.materials.mat_layers fetch` → `library/textures/layers/`.
Everything else is procedural.

## Limitations

- Paint, metal and organic zones from team-colour masks (`tex_maps.py`) and the close-up texture sets
  (`library/textures/closeup/`) expect model textures with such masks; none are included.
- Rain with puddles nearly doubles the frame time (2.8 → 5.4 s on the castle sample); the water map costs 1.8–6 s per
  site once, then comes from the cache; `rain: 0` drops the streaks for cheaper trials.

## FAQ

**How do I add rain, wet ground and puddles to a Blender scene with Python?** Put `weather: {weather: rain}` (or
`storm`) after the geometry steps. Wet ground gets a darker albedo, lower roughness and a water film, puddles stand in
the hollows, and Eevee screen-space ray tracing is switched on so buildings mirror in them; `rain: 0` drops the
streaks for cheaper trials.

**Are puddles placed by noise or by the terrain?** By the terrain: depth to the spill level (Priority-Flood) and flow
accumulation over 1–8 m micro-relief, from the site's `relief.npz`, otherwise from the ground mesh. To see the map,
run `.venv/bin/python -m components.materials.mat_water <relief.npz> <out.png>`.

**How do I make snow, mud, fog or a dust storm?** Pick the preset: `weather: snow`, `mud`, `fog`, `dust_storm`,
`after_rain` or `after_battle`. Explicit keys (`state_ground`, `state_world`, `state`, `rain`) go on top of a preset.

**Can I use it without the rest of sketch3d?** Yes, it also ships as its own repository, `sketch3d-blender-materials`, with the
core it needs. The plan is plain Python and `apply_plan` in `components/materials/api.py` applies it to an open Blender
scene; `.venv/bin/python -m components.materials.mat_plan resolve <scene folder> weather=rain` prints the plan as JSON.
The only download is the rust layer: `.venv/bin/python -m components.materials.mat_layers fetch`.

**Why is there no rain or wear in my sketch render?** In the `sketch` profile the `weather` step builds only the plan:
`light` still gets the sky weather, but no states or rain are applied. Use `--profile preview`.
