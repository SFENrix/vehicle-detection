# preview_samples.py
import cv2, random
from pathlib import Path

ROOT = Path("dataset")
imgs = list((ROOT/"train"/"images").glob("*.*"))
for _ in range(8):
    p = random.choice(imgs)
    img = cv2.imread(str(p))
    lbl = (p.parent.parent/"labels"/(p.stem + ".txt"))
    if lbl.exists():
        for ln in open(lbl):
            cls, cx, cy, w, h = ln.split()
            cls=int(cls); cx=float(cx); cy=float(cy); w=float(w); h=float(h)
            H,W = img.shape[:2]
            x1 = int((cx - w/2)*W); y1 = int((cy-h/2)*H)
            x2 = int((cx + w/2)*W); y2 = int((cy+h/2)*H)
            cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img, str(cls),(x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imshow("sample", cv2.resize(img, (960,540)))
    cv2.waitKey(0)
cv2.destroyAllWindows()
