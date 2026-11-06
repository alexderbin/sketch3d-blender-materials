"""Water on the ground: where water stands and where it runs — by Lagarde's canon (Water drop 2b, 3b:
puddles — in hollows up to the flooding level, small pits — from the height map, streams — from the slope) computed from terrain data, not from noise.
  hollows — Priority-Flood filling (terrain_analysis.fill, Barnes et al. 2014): depth to the spill level = where a puddle will stand;
  runoff — MFD flow accumulation (Freeman 1991, Quinn et al. 1991: a share to each lower neighbour by slope^p) — branching streams, not
  straight D8 lines; terrain — plus fine micro-relief 1–8 m (trampled ground: road, pad — rougher: ruts, pits).
Pure numpy — works in Blender too (bl_mat.water_map). Map: run — visible running water 0–1 (rills and streams), flow —
log10 of the catchment, m²; pond — hollow depth, m.
  .venv/bin/python -m components.materials.mat_water <relief.npz> <output.png> — a map for the eye (blue — puddles, light blue — streams)."""
import math
import os
import sys

import numpy as np

from core.surface import terrain_analysis as TA

VERSION = 4


def value_noise(shape, cell, seed):
    """Value noise: a random lattice with a step of cell cells → smoothly (smoothstep) onto the grid."""
    rng = np.random.default_rng(seed)
    ny, nx = shape
    G = rng.random((int(math.ceil(ny / cell)) + 2, int(math.ceil(nx / cell)) + 2))
    fy, fx = np.arange(ny) / cell, np.arange(nx) / cell
    iy, ix = fy.astype(int), fx.astype(int)
    ty, tx = (fy - iy)[:, None], (fx - ix)[None, :]
    ty, tx = ty * ty * (3 - 2 * ty), tx * tx * (3 - 2 * tx)
    return (G[iy][:, ix] * (1 - ty) * (1 - tx) + G[iy][:, ix + 1] * (1 - ty) * tx
            + G[iy + 1][:, ix] * ty * (1 - tx) + G[iy + 1][:, ix + 1] * ty * tx)


def micro(shape, res, seed=1):
    """Micro-relief: octaves 8, 4, 2, 1 m, amplitude ±1."""
    m, w = 0.0, 0.0
    for k, (sz, wt) in enumerate(((8.0, 1.0), (4.0, 0.6), (2.0, 0.35), (1.0, 0.2))):
        m = m + wt * (value_noise(shape, max(sz / res, 1.0), seed + k) - 0.5)
        w += wt
    return m / w * 2


def mfd(F, res, p=1.5):
    """Catchment, m², over many directions: top to bottom over the filled terrain F, the share to a lower neighbour ∝ slope^p."""
    ny, nx = F.shape
    P = np.pad(F, 1, constant_values=np.inf)
    W = np.zeros((8, ny, nx))
    for k, (dy, dx) in enumerate(TA.N8):
        drop = (F - P[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]) / (res * math.hypot(dy, dx))
        W[k] = np.where(drop > 0, drop, 0.0) ** p
    S = W.sum(0)
    S[S == 0] = 1.0
    Wl = [(W[k] / S).ravel().tolist() for k in range(8)]
    A = [res * res] * (ny * nx)
    for i in np.argsort(F.ravel())[::-1].tolist():
        a, (r, c) = A[i], divmod(i, nx)
        for k, (dy, dx) in enumerate(TA.N8):
            wk = Wl[k][i]
            if wk > 0:
                rr, cc = r + dy, c + dx
                if 0 <= rr < ny and 0 <= cc < nx:
                    A[rr * nx + cc] += a * wk
    return np.array(A).reshape(ny, nx)


def ss(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def flood_need(Hm, F, A, res):
    """How much rain (m of layer) it takes to flood a cell: a hollow (F > Hm) is filled by the rain volume from its own catchment
    (the largest MFD catchment in it — the runoff from above ends up here too), from the bottom up: water up to the height of cell k has volume Σ(z_k − z_i)·res²
    over the cells below. A small pit floods from a light rain, a large shallow bowl — only at the bottom. Outside hollows — inf."""
    ny, nx = Hm.shape
    m = ((F - Hm) > 1e-4).ravel()
    lab = np.full(ny * nx, -1, np.int64)
    h, a = Hm.ravel(), A.ravel()
    need = np.full(ny * nx, np.inf)
    cur = 0
    for s0 in np.flatnonzero(m).tolist():
        if lab[s0] >= 0:
            continue
        stack, cells = [s0], []
        lab[s0] = cur
        while stack:
            i = stack.pop(); cells.append(i); r, c = divmod(i, nx)
            for dy, dx in TA.N8:
                rr, cc = r + dy, c + dx
                if 0 <= rr < ny and 0 <= cc < nx:
                    j = rr * nx + cc
                    if m[j] and lab[j] < 0:
                        lab[j] = cur; stack.append(j)
        idx = np.array(cells)
        o = idx[np.argsort(h[idx])]
        z = h[o]
        vol = (np.arange(len(z)) * z - np.concatenate([[0.0], np.cumsum(z)[:-1]])) * res * res
        need[o] = vol / max(float(a[idx].max()), len(idx) * res * res)
        cur += 1
    return need.reshape(ny, nx)


def compute(H, res, use=None, seed=1, amp=0.04, amp_use=0.12):
    """H — heights, m (grid res m); use — trampled ground 0–1 (road, pad) or None. → {flow, pond, run, Hm}:
    need — the rain layer (m) at which a cell floods (flood_need); run — visible running water 0–1: rills — where the runoff converges more than around it (log A above its own 2 m average),
    only on a slope of 2–7° (on flat ground water stands, and a network of lines between puddles is not streams); streams — a large catchment
    (10³–10³·⁸ m²) on any slope; none in standing water."""
    m = micro(H.shape, res, seed)
    a = amp if use is None else amp + (amp_use - amp) * np.clip(use, 0, 1)
    Hm = H + a * m
    F = TA.fill(Hm, eps=1e-5)
    A = mfd(F, res)
    lA = np.log10(A)
    pond = F - Hm
    rill = ss(lA - TA.blur(lA, 2.0 / res), 0.15, 0.6) * ss(TA.slope(H, res), 2.0, 7.0) * ss(lA, 0.8, 1.5)
    run = np.maximum(rill, ss(lA, 3.0, 3.8)) * (1 - ss(pond, 0.001, 0.005))
    return {"flow": lA, "pond": pond, "need": flood_need(Hm, F, A, res), "run": run, "Hm": Hm}


def raster(w, cell, pad=2, sigma=1.0):
    """Vertex heights (N×3, world) → a grid of cell m: the average per cell, empty cells — by normalised convolution (σ in cells — from the vertex
    spacing; where still empty — twice as wide, until filled). → (H, x0, y0) — x0, y0 is the centre of cell [0, 0]."""
    x0, y0 = w[:, 0].min() - pad * cell, w[:, 1].min() - pad * cell
    ix = np.round((w[:, 0] - x0) / cell).astype(int)
    iy = np.round((w[:, 1] - y0) / cell).astype(int)
    S = np.zeros((iy.max() + pad + 1, ix.max() + pad + 1))
    C = np.zeros_like(S)
    np.add.at(S, (iy, ix), w[:, 2])
    np.add.at(C, (iy, ix), 1.0)
    H = np.where(C > 0, S / np.maximum(C, 1e-9), np.nan)
    sg = max(sigma, 1.0)
    while np.isnan(H).any() and sg < 64:
        bc = TA.blur(C, sg)
        fill = TA.blur(S, sg) / np.maximum(bc, 1e-9)
        H = np.where(np.isnan(H) & (bc > 1e-3), fill, H)
        sg *= 2
    return np.nan_to_num(H, nan=float(np.nanmean(H))), x0, y0


if __name__ == "__main__":
    from PIL import Image
    R = np.load(sys.argv[1])
    H, (x0, y0, res) = R["near"].astype(np.float64), R["near_o"]
    use = np.maximum(R["road"], R["pad"]).astype(np.float64) / 255 if "road" in R.files else None
    o = compute(H, float(res), use)
    gy, gx = np.gradient(H, res)
    sh = np.clip(0.6 + 0.4 * (-gx * 0.5 - gy * 0.5) / np.sqrt(1 + gx * gx + gy * gy), 0, 1)
    img = np.stack([sh, sh, sh], -1) * 0.8
    fl = o["run"][..., None]
    img = img * (1 - fl * 0.8) + np.array([0.2, 0.6, 1.0]) * fl * 0.8
    pd = (o["need"] < 0.02)[..., None] * 1.0                  # flooded at a 20 mm rain layer
    img = img * (1 - pd) + np.array([0.0, 0.1, 0.6]) * pd
    Image.fromarray((np.clip(img[::-1], 0, 1) * 255).astype(np.uint8)).save(sys.argv[2])
