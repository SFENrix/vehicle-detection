#!/usr/bin/env python3
"""
[run code]
python zone_adaptive.py --mode run --source sample-video.mkv --zones zones.json

zone_adaptive.py
Zone-based vehicle detector + adaptive green timing prototype.

Usage:
  # interactive mode to draw zones from a sample image (one-time)
  python zone_adaptive.py --mode draw --image sample_frame.jpg --zones zones.json

  # run on video or camera
  python zone_adaptive.py --mode run --source video.mp4 --zones zones.json

Notes:
- Requires: ultralytics, opencv-python, numpy
  pip install ultralytics opencv-python numpy
- Default counts car & motorbike only. Adjust CLASS_IDS_TO_COUNT to match your data.yaml.
"""
import argparse
import json
import time
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# ----------------- CONFIG -----------------
MODEL_PATH = "vehicle_detector.pt"   # change if needed
CLASS_IDS_TO_COUNT = {1, 2, 3, 4}  # car=1, motorbike=2 (update if your data.yaml differs)

WINDOW_SEC = 10.0     # sliding window (seconds) to count unique vehicles
MIN_GREEN = 10.0      # min green time (seconds)
MAX_GREEN = 60.0      # max green time (seconds)
ALLOCATABLE = 60.0    # total green seconds to split among zones (adjust)
EMA_ALPHA = 0.4       # smoothing factor for green time
IOU_TRACK_THRESH = 0.3
MIN_TRACK_AGE = 2     # frames
# ------------------------------------------

def draw_text(img, text, pos, color=(0,255,0), scale=0.6, thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

# ---------- simple IoU tracker ----------
class SimpleTracker:
    def __init__(self, iou_thresh=IOU_TRACK_THRESH, forget_frames=30):
        self.next_id = 1
        self.tracks = {}  # id -> dict {bbox, last_seen, age, counted_zones:set}
        self.iou_thresh = iou_thresh
        self.forget_frames = forget_frames

    @staticmethod
    def iou(a, b):
        # a,b = (x1,y1,x2,y2)
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        inter = interW * interH
        areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        denom = areaA + areaB - inter + 1e-9
        return inter / denom if denom > 0 else 0.0

    def update(self, detections, frame_idx):
        """
        detections: list of (x1,y1,x2,y2, cls_id, conf)
        returns: list of (track_id or None) aligned with detections
        """
        assigned = [None]*len(detections)
        # attempt match existing tracks
        used_det = set()
        for tid, tr in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, det in enumerate(detections):
                if j in used_det: continue
                i = self.iou(tr['bbox'], det[:4])
                if i > best_iou:
                    best_iou = i; best_j = j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                # update track
                det = detections[best_j]
                self.tracks[tid]['bbox'] = det[:4]
                self.tracks[tid]['last_seen'] = frame_idx
                self.tracks[tid]['age'] += 1
                self.tracks[tid]['cls'] = det[4]
                self.tracks[tid]['conf'] = det[5]
                assigned[best_j] = tid
                used_det.add(best_j)
            else:
                # forget if stale
                if frame_idx - tr['last_seen'] > self.forget_frames:
                    del self.tracks[tid]
        # add remaining detections as new tracks
        for j, det in enumerate(detections):
            if j in used_det: continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'bbox': det[:4],
                'last_seen': frame_idx,
                'age': 1,
                'counted_zones': set(),
                'cls': det[4],
                'conf': det[5]
            }
            assigned[j] = tid
        return assigned

# ---------- zone utilities ----------
def point_in_poly(pt, poly):
    # pt=(x,y), poly list of (x,y)
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (int(pt[0]), int(pt[1])), False) >= 0

def load_zones(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, 'r') as f:
        data = json.load(f)
    return data  # dict: zone_name -> list of [ [x,y], ... ]

def save_zones(path, zones):
    with open(path, 'w') as f:
        json.dump(zones, f, indent=2)

def interactive_draw_zones(image_path, out_json):
    img = cv2.imread(str(image_path))
    if img is None:
        print("Cannot load image:", image_path); return
    clone = img.copy()
    zones = {}
    current_poly = []
    zone_idx = 1
    drawing = False
    print("Interactive draw mode:")
    print(" - Left click to add point")
    print(" - Right click to remove last point")
    print(" - Press 'n' to finalize current zone and start next")
    print(" - Press 's' to save zones to", out_json)
    print(" - Press 'q' to quit without saving")

    def mouse_cb(event, x, y, flags, param):
        nonlocal current_poly, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_poly:
                current_poly.pop()

    cv2.namedWindow("draw")
    cv2.setMouseCallback("draw", mouse_cb)

    while True:
        vis = clone.copy()
        # draw existing zones
        for name, poly in zones.items():
            pts = np.array(poly, np.int32)
            cv2.polylines(vis, [pts], True, (0,255,255), 2)
            # label
            cx = int(np.mean(pts[:,0])); cy = int(np.mean(pts[:,1]))
            cv2.putText(vis, name, (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        # draw current poly
        if current_poly:
            pts = np.array(current_poly, np.int32)
            cv2.polylines(vis, [pts], False, (0,128,255), 2)
            for p in current_poly:
                cv2.circle(vis, tuple(p), 4, (0,128,255), -1)

        cv2.imshow("draw", vis)
        k = cv2.waitKey(10) & 0xFF
        if k == ord('n'):
            if current_poly:
                name = f"zone_{zone_idx}"
                zones[name] = current_poly.copy()
                current_poly = []
                zone_idx += 1
                print("Saved zone:", name)
            else:
                print("No points in current polygon.")
        elif k == ord('s'):
            if current_poly:
                name = f"zone_{zone_idx}"
                zones[name] = current_poly.copy()
                current_poly = []
            save_zones(out_json, zones)
            print("Zones saved to", out_json)
            break
        elif k == ord('q'):
            print("Exit without saving.")
            break

    cv2.destroyAllWindows()
    return zones

# ---------- adaptive logic ----------
def compute_green_times(counts, prev_green_times, min_green=MIN_GREEN, max_green=MAX_GREEN, allocatable=ALLOCATABLE, ema_alpha=EMA_ALPHA):
    total = sum(counts.values()) + 1e-6
    raw = {z: (counts[z]/total)*allocatable for z in counts}
    green = {}
    for z in counts:
        g = max(min_green, min(max_green, raw[z]))
        # EMA smoothing
        prev = prev_green_times.get(z, g)
        sm = ema_alpha * g + (1-ema_alpha) * prev
        green[z] = sm
    return green

# ---------- main run ----------
def run_video(source, zones_file, model_path):
    zones = load_zones(zones_file)
    if zones is None:
        print("Zones file not found. Run draw mode first to create zones.json")
        return
    # load model
    model = YOLO(model_path)
    # video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open source:", source); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    frame_time = 1.0 / max(1.0, fps)
    tracker = SimpleTracker()
    history = []  # list of (timestamp, {zone:set(track_ids)})
    prev_green = {z: MIN_GREEN for z in zones.keys()}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream")
            break
        frame_idx += 1
        tnow = time.time()
        # predict (single frame)
        res = model.predict(source=[frame], imgsz=960, conf=0.25, verbose=False)
        preds = res[0]
        dets = []
        if hasattr(preds, 'boxes') and len(preds.boxes):
            for b in preds.boxes:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                cls_id = int(b.cls[0].item()) if hasattr(b, 'cls') else int(b.cls)
                conf = float(b.conf[0].item()) if hasattr(b, 'conf') else float(b.conf)
                if cls_id in CLASS_IDS_TO_COUNT:
                    dets.append((x1,y1,x2,y2, cls_id, conf))
        # update tracker
        det2track = tracker.update(dets, frame_idx)
        # check zones
        zone_seen = {zn: set() for zn in zones.keys()}
        for j, det in enumerate(dets):
            tid = det2track[j]
            bbox = det[:4]
            cx = (bbox[0]+bbox[2]) / 2.0
            cy = (bbox[1]+bbox[3]) / 2.0
            for zn, poly in zones.items():
                if point_in_poly((cx,cy), poly):
                    zone_seen[zn].add(tid)
                    # mark track as counted for zone (optional)
                    tracker.tracks[tid].setdefault('counted_zones', set()).add(zn)
        # append history and purge old
        history.append((tnow, zone_seen))
        cutoff = tnow - WINDOW_SEC
        history = [h for h in history if h[0] >= cutoff]
        # aggregate unique counts per zone
        counts = {zn: 0 for zn in zones.keys()}
        for _, zmap in history:
            for zn, s in zmap.items():
                counts[zn] = len(set().union(*(zmap.get(zn, set()) for _, zmap in history)))
        # compute green times
        green_times = compute_green_times(counts, prev_green)
        prev_green = green_times
        # visualization
        vis = frame.copy()
        # draw zones
        for zn, poly in zones.items():
            pts = np.array(poly, np.int32)
            cv2.polylines(vis, [pts], True, (0,255,255), 2)
            # fill with slight transparency
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0,255,255))
            cv2.addWeighted(overlay, 0.05, vis, 0.95, 0, vis)
            # label counts & green time
            cx = int(np.mean(pts[:,0])); cy = int(np.mean(pts[:,1]))
            draw_text(vis, f"{zn}: cnt={counts[zn]}", (cx-60, cy-10), (0,255,0))
            draw_text(vis, f"green={int(round(green_times[zn]))}s", (cx-60, cy+12), (0,200,255))
        # draw detections & track ids
        for tid, tr in tracker.tracks.items():
            x1,y1,x2,y2 = map(int, tr['bbox'])
            color = (0,200,0) if tr['cls'] in CLASS_IDS_TO_COUNT else (200,0,0)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
            draw_text(vis, f"ID:{tid}", (x1, y1-6), color)
        # HUD
        hud_y = 20
        draw_text(vis, f"Counts: " + ", ".join([f"{k}={v}" for k,v in counts.items()]), (10, hud_y))
        draw_text(vis, f"GreenTimes: " + ", ".join([f"{k}={int(round(v))}s" for k,v in green_times.items()]), (10, hud_y+22))
        draw_text(vis, f"Window:{WINDOW_SEC}s FPS:{1.0/max(0.001, time.time()-tnow):.1f}", (10, hud_y+44))
        cv2.imshow("zone_adaptive", cv2.resize(vis, (1280,720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["draw","run"], default="run", help="draw zones or run detection")
    parser.add_argument("--image", type=str, help="image for draw mode")
    parser.add_argument("--zones", type=str, default="zones.json", help="zones json file")
    parser.add_argument("--source", type=str, default=0, help="video file or camera index or rtsp url")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="path to model weights")
    args = parser.parse_args()

    if args.mode == "draw":
        if not args.image:
            print("Provide --image sample_frame.jpg to draw zones")
        else:
            interactive_draw_zones(args.image, args.zones)
    else:
        run_video(args.source, args.zones, args.model)
