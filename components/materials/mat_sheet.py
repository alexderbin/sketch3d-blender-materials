"""Sheet of the material states (components/materials/bl_mat.py demo): frames <folder>/<state>.png → one sheet with captions.
.venv/bin/python -m components.materials.mat_sheet <demo folder> [<output.jpg>] [--cols 5] [--crop x0,y0,x1,y1]"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

a = sys.argv[1:]
d = a[0]
out = a[1] if len(a) > 1 and not a[1].startswith("--") else os.path.join(d, "sheet.jpg")
cols = int(a[a.index("--cols") + 1]) if "--cols" in a else 5
crop = tuple(int(x) for x in a[a.index("--crop") + 1].split(",")) if "--crop" in a else None
names = [n for n, _ in json.load(open(os.path.join(d, "result.json")))["frames"]]
ims = []
for n in names:
    im = Image.open(os.path.join(d, n + ".png")).convert("RGB")
    ims.append(im.crop(crop) if crop else im)
W, H = ims[0].size
F = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", max(14, W // 22))
rows = (len(ims) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, (H + F.size + 10) * rows), (18, 18, 18))
dr = ImageDraw.Draw(sheet)
for i, (n, im) in enumerate(zip(names, ims)):
    x, y = (i % cols) * W, (i // cols) * (H + F.size + 10)
    sheet.paste(im, (x, y + F.size + 10))
    dr.text((x + 8, y + 4), n, fill=(255, 230, 0), font=F)
sheet.save(out, quality=88)
print(out)
