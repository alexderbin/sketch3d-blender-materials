# curvature from a normal map: the X derivative along the horizontal + the Y derivative along the vertical, blur, normalisation.
# Convolutions are whole-array numpy shifts, the blur is three box passes (≈ Gaussian), normalisation is by the 1–99 percentiles
# (a single outlier does not flatten the whole map), the output is two masks: edges (convex) and cavities (concave).
"""Shape masks and material zones from a model's textures.
Pure numpy: works both in Blender (components/materials/bl_mat.py) and in the project's Python.

masks(nrm, dif, tem, gloss, name, world) → RGBA uint8: R — edges, G — cavities, B — metal, A = 0.5 + 0.5·(organic − team
colour): organic (skin, bone, fur, tanned leather) and team colour (channels R, G, A of the team-colour mask) do not overlap — they share one signed
channel; 0.5 — everything else (cloth, stone, wood). The trim (channel B of the mask) — metal.
The normals are like Blender's (OpenGL, Y up): verified with an armour map."""
import numpy as np

VERSION = 3          # when it changes, the mask cache (library/textures/masks) is recomputed


def _box(a, r):
    """Mean in a (2r+1)² window — via a cumulative sum; edges — by repeating the edge pixels."""
    if r < 1:
        return a
    p = np.pad(a, ((r + 1, r), (r + 1, r)), mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def blur(a, sigma):
    """≈ Gaussian: three box passes (central limit theorem); sigma in pixels."""
    r = int(round(np.sqrt(12 * sigma * sigma / 3 + 1) / 2)) if sigma > 0 else 0
    for _ in range(3):
        a = _box(a, r)
    return a


def curvature(nrm, sigma=1.0):
    """nrm: H×W×3 float 0–1 (tangent space, OpenGL). → H×W, 0.5 — flat, above — convex (edge), below — concave."""
    x = nrm[..., 0] * 2 - 1
    y = nrm[..., 1] * 2 - 1
    c = (np.roll(x, -1, 1) - np.roll(x, 1, 1)) + (np.roll(y, 1, 0) - np.roll(y, -1, 0))
    c = blur(c, sigma)
    lo, hi = np.percentile(c, [1, 99])
    m = max(abs(lo), abs(hi), 1e-6)
    return np.clip(0.5 + 0.5 * c / m, 0, 1)


def _hsv(rgb):
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    s = np.where(mx > 1e-6, d / np.maximum(mx, 1e-6), 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.where(d < 1e-6, 0,
                 np.where(mx == r, ((g - b) / np.maximum(d, 1e-6)) % 6,
                          np.where(mx == g, (b - r) / np.maximum(d, 1e-6) + 2, (r - g) / np.maximum(d, 1e-6) + 4))) * 60
    return h, s, mx


def _band(x, a, b, soft):
    """1 inside [a, b], soft edges of width soft."""
    return np.clip(np.minimum((x - a) / soft + 1, (b - x) / soft + 1), 0, 1)


SKIN_HINT = ("head", "face", "skin", "flesh", "body_", "arm", "hand", "torso", "chest", "bare")
GREEN_SKIN = ("orc", "goblin")
BONE_HINT = ("bone", "skull", "wraith", "ivory", "horn", "tusk", "talon", "claw", "carapace", "chitin")


def zones(dif, tem=None, gloss=None, name="", world=False):
    """Zone weights per texel: (metal, organic, army paint). dif H×W×3 (sRGB 0–1); tem H×W×4 army mask or None; gloss H×W —
    the G channel of the glTF packing of a light model: for fighters it is the game's **gloss** (light — glossy: stones, bare metal of scratches;
    0.2 — "no map"), for buildings — a grey highlight-strength map. name — the name of the texture set
    and its file path (…/orc/…): organic hints. Metal with care: army trim + grey and shiny
    (by colour alone, grey cloth and white paint get confused with metal)."""
    h, s, v = _hsv(dif)
    nm = name.lower()
    team = np.zeros(dif.shape[:2], np.float32)
    trim = np.zeros_like(team)
    if tem is not None:
        team = np.clip(tem[..., 0] + tem[..., 1] + tem[..., 3], 0, 1)
        trim = tem[..., 2]
    free = np.clip(1 - team - trim, 0, 1)
    grey = np.clip((0.22 - s) / 0.1, 0, 1) * _band(v, 0.12, 0.92, 0.08)
    if gloss is None or world:     # buildings and objects: the game has a highlight model there (highlight strength, not metal); metal guessed from
        shiny = 0.0                # a grey colour and a bright highlight turned brown crates into grey mirrors (chaos_crop.jpg)
    else:
        shiny = np.clip((gloss - 0.22) / 0.15, 0, 1)
    metal = trim + free * grey * shiny
    brown = _band(h, 12, 48, 8) * _band(s, 0.22, 0.75, 0.08) * _band(v, 0.08, 0.62, 0.08)
    skin = _band(h, 0, 32, 8) * _band(s, 0.18, 0.6, 0.08) * _band(v, 0.35, 0.95, 0.08)
    ivory = _band(h, 30, 62, 8) * _band(s, 0.08, 0.38, 0.06) * _band(v, 0.55, 1.0, 0.08)
    organic = brown * 0.8
    if any(k in nm for k in SKIN_HINT):
        organic = np.maximum(organic, skin)
    if any(k in nm for k in GREEN_SKIN):
        organic = np.maximum(organic, _band(h, 65, 150, 12) * _band(s, 0.2, 0.9, 0.08) * _band(v, 0.12, 0.9, 0.08))
    if any(k in nm for k in BONE_HINT):
        organic = np.maximum(organic, ivory)
    organic = organic * free
    metal = np.clip(metal * (1 - organic), 0, 1)
    return metal.astype(np.float32), organic.astype(np.float32), team.astype(np.float32)


def masks(nrm, dif, tem=None, gloss=None, name="", world=False):
    """RGBA uint8 of the same size as nrm: R — edges, G — cavities (small + medium), B — metal, A — organic/paint (signed)."""
    H, W = nrm.shape[:2]
    fine = curvature(nrm, 0.8 * max(1, W / 512))
    mid = curvature(nrm, 3.0 * max(1, W / 512))
    edge = np.clip((fine - 0.55) / 0.3, 0, 1)
    cav = np.clip(np.maximum((0.45 - fine) / 0.3, (0.47 - mid) / 0.25), 0, 1)
    if dif.shape[:2] != (H, W):
        dif = resize(dif, (H, W))
    if tem is not None and tem.shape[:2] != (H, W):
        tem = resize(tem, (H, W))
    if gloss is not None and gloss.shape[:2] != (H, W):
        gloss = resize(gloss[..., None], (H, W))[..., 0]
    metal, organic, team = zones(dif, tem, gloss, name, world)
    out = np.stack([edge, cav, metal, np.clip(0.5 + 0.5 * (organic - team), 0, 1)], -1)
    return (out * 255 + 0.5).astype(np.uint8)


def resize(a, hw):
    """Bilinear (without PIL/cv2 — works in Blender)."""
    H, W = hw
    h, w = a.shape[:2]
    if (h, w) == (H, W):
        return a
    ys = (np.arange(H) + 0.5) * h / H - 0.5
    xs = (np.arange(W) + 0.5) * w / W - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1); y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1); x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None, None]; fx = np.clip(xs - x0, 0, 1)[None, :, None]
    a3 = a if a.ndim == 3 else a[..., None]
    r = (a3[y0][:, x0] * (1 - fy) * (1 - fx) + a3[y0][:, x1] * (1 - fy) * fx
         + a3[y1][:, x0] * fy * (1 - fx) + a3[y1][:, x1] * fy * fx)
    return r if a.ndim == 3 else r[..., 0]


def team_tint(dif, tem, pal):
    """Team colour: diffuse × Π mix(1, 2·color_i, channel_i)."""
    tint = np.ones(dif.shape[:2] + (3,), np.float32)
    for i in range(4):
        m = tem[..., i:i + 1]
        tint *= 1 - m + m * 2 * np.asarray(pal[i][:3], np.float32)
    return np.clip(dif[..., :3] * tint, 0, 1)
