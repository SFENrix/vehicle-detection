from pathlib import Path
import random
from collections import Counter
import cv2
import numpy as np

ROOT = Path("dataset")   # ganti path hasil download
lbls = list((ROOT/"train"/"labels").glob("*.txt")) + list((ROOT/"valid"/"labels").glob("*.txt"))
cnt = Counter()
bbox_areas = []
img_count = 0

for f in lbls:
    img_path = (f.parent.parent/"images"/(f.stem + ".jpg"))
    if not img_path.exists():
        img_path = (f.parent.parent/"images"/(f.stem + ".png"))
    if img_path.exists():
        img = cv2.imread(str(img_path))
        h,w = img.shape[:2]
    else:
        continue
    img_count += 1
    for ln in open(f,'r'):
        parts = ln.strip().split()
        if not parts: continue
        cls = int(parts[0])
        cnt[cls] += 1
        cx,cy,nw,nh = map(float, parts[1:5])
        bw = nw * w
        bh = nh * h
        bbox_areas.append((bw*bh)/(w*h))  # relative area

print("Total labelled images:", img_count)
print("Class counts (id:count):", cnt)
areas = np.array(bbox_areas)
print("Median bbox rel area:", np.median(areas))
print("Fraction small (<0.005 rel area):", (areas < 0.005).mean())
