"""Measure of pattern repetition along the pipe (components/materials/bench/bl_pipe.py): autocorrelation of the brightness of a strip along the axis; after its first
dip below zero — the highest secondary peak relative to zero shift. A tile gives a strong secondary peak at its own period; non-repeating patches
(even large ones) do not: they give only a wide first hump, which the measure skips.
.venv/bin/python -m components.materials.bench.repeat <pipe image> → number"""
import sys
import numpy as np
from PIL import Image


def repeat_peak(path):
    a = np.asarray(Image.open(path).convert("L"), np.float32)
    H, W = a.shape
    band = a[int(H * 0.35):int(H * 0.65), int(W * 0.06):int(W * 0.94)]     # the middle strip of the pipe, without the ends
    k = np.exp(-0.5 * (np.arange(-60, 61) / 20.0) ** 2); k /= k.sum()     # remove the large scale (σ 20 px): the tile shows in the fine detail
    low = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, band)
    prof = band - low
    best = 0.0
    n = prof.shape[1]
    f = np.fft.rfft(prof, n=2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :n].mean(0)
    ac /= max(ac[0], 1e-6)
    neg = np.nonzero(ac[1:n // 2] < 0)[0]
    if len(neg) == 0:
        return 0.0
    lo = int(neg[0]) + 1
    return float(max(0.0, ac[lo:n // 2].max()))


if __name__ == "__main__":
    print(round(repeat_peak(sys.argv[1]), 3))
