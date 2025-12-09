import numpy as np
import json, cv2
img = cv2.imread("sample_frame.jpg")   # ganti nama file
zones = json.load(open("zonestest.json"))
vis = img.copy()
for name, poly in zones.items():
    pts = cv2.convexHull(np.array(poly, dtype=int))
    cv2.polylines(vis, [np.array(poly, int)], True, (0,255,255), 2)
    cx = int(sum(p[0] for p in poly)/len(poly)); cy = int(sum(p[1] for p in poly)/len(poly))
    cv2.putText(vis, f"{name}", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
cv2.imshow("zones", cv2.resize(vis, (1280,720)))
cv2.waitKey(0)
cv2.destroyAllWindows()
