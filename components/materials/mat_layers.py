"""Textures of the state layers for the shared material (components/materials/bl_mat.py): downloaded from ambientCG per library/textures/layers.yaml.
.venv/bin/python -m components.materials.mat_layers fetch [--res 2K]  → library/textures/layers/<id>/<id>_<res>-JPG_<map>.jpg, index.json
The ambientCG licence is CC0."""
import io, json, sys, urllib.request, zipfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "library/textures"


def fetch(res=None):
    cfg = yaml.safe_load((LIB / "layers.yaml").read_text())
    res = res or cfg.get("res", "1K")
    out = LIB / "layers"
    idx = {}
    for role, L in cfg["layers"].items():
        aid = L["id"]
        d = out / aid
        d.mkdir(parents=True, exist_ok=True)
        want = {m: d / f"{aid}_{res}-JPG_{m}.jpg" for m in L["maps"]}
        if not all(p.exists() for p in want.values()):
            url = f"https://ambientcg.com/get?file={aid}_{res}-JPG.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "sketch3d/1.0"})
            data = urllib.request.urlopen(req, timeout=120).read()
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for m, p in want.items():
                    name = next((n for n in z.namelist() if n.endswith(f"_{m}.jpg")), None)
                    if name is None:
                        raise SystemExit(f"{aid}: no map {m} in the archive ({z.namelist()})")
                    p.write_bytes(z.read(name))
            print(f"{role}: {aid} {res} — {len(data) / 1e6:.1f} MB archive", flush=True)
        idx[role] = {m: str(p.relative_to(ROOT)) for m, p in want.items()}
    (out / "index.json").write_text(json.dumps({"res": res, "layers": idx}, ensure_ascii=False, indent=1))
    print("done →", (out / "index.json").relative_to(ROOT))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] != "fetch":
        raise SystemExit(__doc__)
    fetch(a[a.index("--res") + 1] if "--res" in a else None)
