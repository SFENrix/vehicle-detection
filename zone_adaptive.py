"""
Updated version of zone_adaptive.py based on lecturer's suggestions

Added:
1. Observation zone are now implemented as multi-section zones (near/mid/far) for distance-aware detection
2. Occupancy-based measurement (area coverage)
3. Confidence-weighted counting
4. Density estimation for overlapping vehicles
5. Velocity-aware tracking

Usage:  
  # Still works with old simple zones
  python zone_adaptive.py --mode run --source video.mp4/camera index --zones zones.json
  >>>> run python camera_test.py to find camera index

"""
#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from collections import deque
from ultralytics import YOLO
import cv2
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

MODEL_PATH = "vehicle_detector.pt"
CLASS_IDS_TO_COUNT = {1, 2, 3, 4}

WINDOW_SEC = 10.0
MIN_GREEN = 15.0
MAX_GREEN = 75.0
ALLOCATABLE = 90.0
EMA_ALPHA = 0.4
IOU_TRACK_THRESH = 0.5

MIN_CONF_DISPLAY = 0.3
SHOW_ALL_DETECTIONS = True

SATURATION_FLOW = 1800
HEADWAY_TIME = 2.0
STARTUP_LOST_TIME = 2.0

OCCUPANCY_WEIGHT = 0.6
COUNT_WEIGHT = 0.4
NEAR_ZONE_SATURATION_THRESHOLD = 0.75

STOPPED_THRESHOLD = 2.0
SLOW_THRESHOLD = 5.0
MOVING_THRESHOLD = 10.0

USE_VELOCITY_TRACKING = True

def draw_text(img, text, pos, color=(0,255,0), scale=0.8, thick=3):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), (int(pt[0]), int(pt[1])), False) >= 0

def calculate_polygon_area(polygon):
    try:
        poly = Polygon(polygon)
        return poly.area
    except:
        return cv2.contourArea(np.array(polygon, dtype=np.float32))

def calculate_bbox_polygon_intersection(bbox, polygon):
    try:
        x1, y1, x2, y2 = bbox
        bbox_poly = box(x1, y1, x2, y2)
        zone_poly = Polygon(polygon)
        intersection = bbox_poly.intersection(zone_poly)
        return intersection.area
    except:
        return 0.0

def calculate_occupancy(polygon, bboxes):
    if not bboxes:
        return 0.0
    
    polygon_area = calculate_polygon_area(polygon)
    if polygon_area == 0:
        return 0.0
    
    try:
        bbox_shapes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            bbox_poly = box(x1, y1, x2, y2)
            zone_poly = Polygon(polygon)
            
            if bbox_poly.intersects(zone_poly):
                intersection = bbox_poly.intersection(zone_poly)
                bbox_shapes.append(intersection)
        
        if not bbox_shapes:
            return 0.0
        
        covered_area = unary_union(bbox_shapes).area
        occupancy_ratio = min(1.0, covered_area / polygon_area)
        
        return occupancy_ratio
    except:
        total_bbox_area = 0
        for bbox in bboxes:
            intersection_area = calculate_bbox_polygon_intersection(bbox, polygon)
            total_bbox_area += intersection_area
        
        return min(1.0, total_bbox_area / polygon_area)

def estimate_hidden_vehicles(bboxes, iou_threshold=0.3):
    if len(bboxes) < 2:
        return 0.0
    
    hidden_count = 0.0
    processed = set()
    
    for i, bbox_a in enumerate(bboxes):
        if i in processed:
            continue
        
        x1a, y1a, x2a, y2a = bbox_a[:4]
        area_a = max(0, x2a - x1a) * max(0, y2a - y1a)
        
        for j, bbox_b in enumerate(bboxes[i+1:], start=i+1):
            if j in processed:
                continue
            
            x1b, y1b, x2b, y2b = bbox_b[:4]
            
            x1i = max(x1a, x1b)
            y1i = max(y1a, y1b)
            x2i = min(x2a, x2b)
            y2i = min(y2a, y2b)
            
            inter_area = max(0, x2i - x1i) * max(0, y2i - y1i)
            
            if inter_area > 0:
                area_b = max(0, x2b - x1b) * max(0, y2b - y1b)
                union_area = area_a + area_b - inter_area
                iou = inter_area / (union_area + 1e-9)
                
                if iou_threshold < iou < 0.7:
                    hidden_count += 0.3
                elif iou >= 0.7:
                    hidden_count += 0.1
    
    return hidden_count

def load_zones(path):
    p = Path(path)
    if not p.exists():
        return None, None
    with open(p, 'r') as f:
        data = json.load(f)
    
    if data:
        first_zone = next(iter(data.values()))
        if isinstance(first_zone, dict) and 'sections' in first_zone:
            return data, 'multisection'
        else:
            converted = {}
            for zone_name, polygon in data.items():
                converted[zone_name] = {
                    'sections': [
                        {
                            'name': 'full',
                            'polygon': polygon,
                            'distance_weight': 1.0,
                            'max_contribution': MAX_GREEN,
                            'expected_capacity': 10
                        }
                    ]
                }
            return converted, 'simple'
    
    return None, None

def save_zones(path, zones):
    with open(path, 'w') as f:
        json.dump(zones, f, indent=2)

def interactive_draw_multisection_zones(image_path, out_json):
    img = cv2.imread(str(image_path))
    if img is None:
        print("Cannot load image:", image_path)
        return
    
    clone = img.copy()
    zones = {}
    current_poly = []
    zone_idx = 1
    section_idx = 0
    section_names = ['near', 'mid', 'far']
    current_zone_sections = []
    
    print("=== Multi-Section Zone Drawing Mode ===")
    print("Draw 3 sections per zone (near → mid → far)")
    print("")
    print("Controls:")
    print("  Left click    - Add point to current section")
    print("  Right click   - Remove last point")
    print("  'n'           - Finalize current section")
    print("  'z'           - Complete zone (finish all 3 sections)")
    print("  's'           - Save all zones and exit")
    print("  'q'           - Quit without saving")
    print("")
    print("Start drawing NEAR section for Zone 1...")

    def mouse_cb(event, x, y, flags, param):
        nonlocal current_poly
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if current_poly:
                current_poly.pop()

    cv2.namedWindow("draw_multisection", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("draw_multisection", mouse_cb)

    while True:
        vis = clone.copy()
        
        for zname, zdata in zones.items():
            for i, section in enumerate(zdata['sections']):
                pts = np.array(section['polygon'], np.int32)
                color = [(0,255,0), (0,200,200), (0,100,255)][i % 3]
                cv2.polylines(vis, [pts], True, color, 2)
                
                cx = int(np.mean(pts[:,0]))
                cy = int(np.mean(pts[:,1]))
                cv2.putText(vis, f"{zname}-{section['name']}", 
                           (cx-40, cy), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, color, 2)
        
        for i, section_data in enumerate(current_zone_sections):
            pts = np.array(section_data['polygon'], np.int32)
            color = [(0,255,0), (0,200,200), (0,100,255)][i]
            cv2.polylines(vis, [pts], True, color, 3)
        
        if current_poly:
            pts = np.array(current_poly, np.int32)
            color = [(255,255,0), (255,200,0), (255,150,0)][section_idx % 3]
            cv2.polylines(vis, [pts], False, color, 2)
            for p in current_poly:
                cv2.circle(vis, tuple(p), 5, color, -1)
        
        status_text = f"Zone {zone_idx} - Section: {section_names[section_idx].upper()}"
        cv2.putText(vis, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(vis, f"Points: {len(current_poly)}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("draw_multisection", vis)
        k = cv2.waitKey(10) & 0xFF
        
        if k == ord('n'):
            if len(current_poly) >= 3:
                section_data = {
                    'name': section_names[section_idx],
                    'polygon': current_poly.copy(),
                    'distance_weight': [1.5, 1.0, 0.7][section_idx],
                    'max_contribution': [30, 50, 70][section_idx],
                    'expected_capacity': [8, 12, 15][section_idx]
                }
                current_zone_sections.append(section_data)
                current_poly = []
                section_idx += 1
                
                if section_idx >= 3:
                    zone_name = f"zone_{zone_idx}"
                    zones[zone_name] = {'sections': current_zone_sections.copy()}
                    print(f"✓ {zone_name} completed (3 sections)")
                    
                    current_zone_sections = []
                    section_idx = 0
                    zone_idx += 1
                else:
                    print(f"→ Draw {section_names[section_idx].upper()} section")
            else:
                print("Need at least 3 points for a section")
        
        elif k == ord('z'):
            if current_poly and len(current_poly) >= 3:
                section_data = {
                    'name': section_names[section_idx],
                    'polygon': current_poly.copy(),
                    'distance_weight': [1.5, 1.0, 0.7][section_idx],
                    'max_contribution': [30, 50, 70][section_idx],
                    'expected_capacity': [8, 12, 15][section_idx]
                }
                current_zone_sections.append(section_data)
            
            if current_zone_sections:
                zone_name = f"zone_{zone_idx}"
                zones[zone_name] = {'sections': current_zone_sections.copy()}
                print(f"✓ {zone_name} completed ({len(current_zone_sections)} sections)")
                
                current_zone_sections = []
                current_poly = []
                section_idx = 0
                zone_idx += 1
        
        elif k == ord('s'):
            if current_poly and len(current_poly) >= 3:
                section_data = {
                    'name': section_names[section_idx],
                    'polygon': current_poly.copy(),
                    'distance_weight': [1.5, 1.0, 0.7][section_idx],
                    'max_contribution': [30, 50, 70][section_idx],
                    'expected_capacity': [8, 12, 15][section_idx]
                }
                current_zone_sections.append(section_data)
            
            if current_zone_sections:
                zone_name = f"zone_{zone_idx}"
                zones[zone_name] = {'sections': current_zone_sections.copy()}
            
            if zones:
                save_zones(out_json, zones)
                print(f"\n✓ Saved {len(zones)} zones to {out_json}")
            break
        
        elif k == ord('q'):
            print("Cancelled")
            break

    cv2.destroyAllWindows()
    return zones

class VelocityAwareTracker:
    def __init__(self, iou_thresh=IOU_TRACK_THRESH, forget_frames=15, history_length=5):
        self.next_id = 1
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.forget_frames = forget_frames
        self.history_length = history_length

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        inter = interW * interH
        areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        denom = areaA + areaB - inter + 1e-9
        return inter / denom if denom > 0 else 0.0
    
    @staticmethod
    def calculate_center(bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
    
    @staticmethod
    def calculate_distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
    
    def calculate_velocity(self, position_history):
        if len(position_history) < 2:
            return 0.0, 0.0
        
        velocities = []
        for i in range(1, len(position_history)):
            prev_pos = position_history[i-1]
            curr_pos = position_history[i]
            
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            
            velocity = np.sqrt(dx**2 + dy**2)
            velocities.append(velocity)
        
        avg_velocity = np.mean(velocities) if velocities else 0.0
        
        if len(position_history) >= 2:
            oldest = position_history[0]
            newest = position_history[-1]
            dx_total = newest[0] - oldest[0]
            dy_total = newest[1] - oldest[1]
            direction = np.arctan2(dy_total, dx_total)
        else:
            direction = 0.0
        
        return avg_velocity, direction
    
    def classify_movement(self, velocity):
        if velocity < STOPPED_THRESHOLD:
            return 'stopped'
        elif velocity < SLOW_THRESHOLD:
            return 'slow'
        elif velocity < MOVING_THRESHOLD:
            return 'moving'
        else:
            return 'fast'

    def update(self, detections, frame_idx):
        assigned = [None] * len(detections)
        used_det = set()
        
        for tid, tr in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, det in enumerate(detections):
                if j in used_det:
                    continue
                i = self.iou(tr['bbox'], det[:4])
                if i > best_iou:
                    best_iou = i
                    best_j = j
            
            if best_j >= 0 and best_iou >= self.iou_thresh:
                det = detections[best_j]
                new_bbox = det[:4]
                
                new_center = self.calculate_center(new_bbox)
                
                self.tracks[tid]['position_history'].append(new_center)
                if len(self.tracks[tid]['position_history']) > self.history_length:
                    self.tracks[tid]['position_history'].popleft()
                
                velocity, direction = self.calculate_velocity(
                    list(self.tracks[tid]['position_history'])
                )
                
                self.tracks[tid]['bbox'] = new_bbox
                self.tracks[tid]['last_seen'] = frame_idx
                self.tracks[tid]['age'] += 1
                self.tracks[tid]['cls'] = det[4]
                self.tracks[tid]['conf'] = det[5]
                self.tracks[tid]['velocity'] = velocity
                self.tracks[tid]['direction'] = direction
                self.tracks[tid]['movement_state'] = self.classify_movement(velocity)
                
                assigned[best_j] = tid
                used_det.add(best_j)
            else:
                if frame_idx - tr['last_seen'] > self.forget_frames:
                    del self.tracks[tid]
        
        for j, det in enumerate(detections):
            if j in used_det:
                continue
            
            tid = self.next_id
            self.next_id += 1
            
            bbox = det[:4]
            center = self.calculate_center(bbox)
            
            self.tracks[tid] = {
                'bbox': bbox,
                'last_seen': frame_idx,
                'age': 1,
                'counted_zones': set(),
                'cls': det[4],
                'conf': det[5],
                'position_history': deque([center], maxlen=self.history_length),
                'velocity': 0.0,
                'direction': 0.0,
                'movement_state': 'unknown'
            }
            assigned[j] = tid
        
        return assigned

class SimpleTracker:
    def __init__(self, iou_thresh=IOU_TRACK_THRESH, forget_frames=15):
        self.next_id = 1
        self.tracks = {}
        self.iou_thresh = iou_thresh
        self.forget_frames = forget_frames

    @staticmethod
    def iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[2], b[2]); yB = min(a[3], b[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        inter = interW * interH
        areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        denom = areaA + areaB - inter + 1e-9
        return inter / denom if denom > 0 else 0.0

    def update(self, detections, frame_idx):
        assigned = [None]*len(detections)
        used_det = set()
        
        for tid, tr in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, det in enumerate(detections):
                if j in used_det: continue
                i = self.iou(tr['bbox'], det[:4])
                if i > best_iou:
                    best_iou = i; best_j = j
            
            if best_j >= 0 and best_iou >= self.iou_thresh:
                det = detections[best_j]
                self.tracks[tid]['bbox'] = det[:4]
                self.tracks[tid]['last_seen'] = frame_idx
                self.tracks[tid]['age'] += 1
                self.tracks[tid]['cls'] = det[4]
                self.tracks[tid]['conf'] = det[5]
                assigned[best_j] = tid
                used_det.add(best_j)
            else:
                if frame_idx - tr['last_seen'] > self.forget_frames:
                    del self.tracks[tid]
        
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

def strategy_multisection(zone_metrics, prev_green, min_green, max_green, allocatable, ema_alpha):
    green = {}
    
    for zone_name, metrics in zone_metrics.items():
        total_weighted_count = metrics.get('weighted_count', 0)
        total_occupancy = metrics.get('occupancy', 0)
        near_saturation = metrics.get('near_saturation', 0)
        
        stopped_count = metrics.get('stopped_count', 0)
        slow_count = metrics.get('slow_count', 0)
        moving_count = metrics.get('moving_count', 0)
        total_count = metrics.get('total_count', 0)
        
        if USE_VELOCITY_TRACKING and total_count > 0:
            effective_count = (
                stopped_count * 1.0 +
                slow_count * 0.6 +
                moving_count * 0.2
            )
            count_score = effective_count
        else:
            count_score = total_weighted_count
        
        occupancy_score = total_occupancy * 100
        
        combined_score = (COUNT_WEIGHT * count_score + 
                         OCCUPANCY_WEIGHT * occupancy_score)
        
        if combined_score < 0.1:
            raw_time = min_green
        else:
            scaled_score = np.log(combined_score + 1) * 10
            raw_time = min_green + (scaled_score / 100) * (max_green - min_green)
            raw_time = max(min_green, min(max_green, raw_time))
        
        if near_saturation > NEAR_ZONE_SATURATION_THRESHOLD:
            if USE_VELOCITY_TRACKING and stopped_count > 0:
                saturation_cap = min_green + (1 - near_saturation) * 20
                raw_time = min(raw_time, saturation_cap)
            elif not USE_VELOCITY_TRACKING:
                saturation_cap = min_green + (1 - near_saturation) * 20
                raw_time = min(raw_time, saturation_cap)
        
        if USE_VELOCITY_TRACKING and total_count > 0:
            moving_ratio = (moving_count + slow_count) / total_count
            if moving_ratio > 0.8:
                raw_time *= 0.7
        
        prev = prev_green.get(zone_name, raw_time)
        smoothed = ema_alpha * raw_time + (1 - ema_alpha) * prev
        
        green[zone_name] = smoothed
    
    return green

def strategy_hybrid_enhanced(zone_metrics, prev_green, min_green, max_green, allocatable, ema_alpha):
    green = {}
    
    for zone_name, metrics in zone_metrics.items():
        weighted_count = metrics.get('weighted_count', 0)
        occupancy = metrics.get('occupancy', 0)
        near_saturation = metrics.get('near_saturation', 0)
        
        stopped_count = metrics.get('stopped_count', 0)
        slow_count = metrics.get('slow_count', 0)
        moving_count = metrics.get('moving_count', 0)
        total_count = metrics.get('total_count', 0)
        
        if USE_VELOCITY_TRACKING and total_count > 0:
            effective_count = (
                stopped_count * 1.0 +
                slow_count * 0.6 +
                moving_count * 0.2
            )
            count_to_use = effective_count
        else:
            count_to_use = weighted_count
        
        if count_to_use < 0.5:
            base_time = min_green
        elif count_to_use < 8:
            base_time = STARTUP_LOST_TIME + (count_to_use * HEADWAY_TIME)
        else:
            base_time = STARTUP_LOST_TIME + (8 * HEADWAY_TIME)
            extra = np.sqrt(count_to_use - 8) * 3
            base_time += extra
        
        occupancy_factor = 1.0 + (occupancy * 0.3)
        adjusted_time = base_time * occupancy_factor
        
        if near_saturation > NEAR_ZONE_SATURATION_THRESHOLD:
            if USE_VELOCITY_TRACKING and stopped_count > 0:
                cap = 25 + (1 - near_saturation) * 15
                adjusted_time = min(adjusted_time, cap)
            elif not USE_VELOCITY_TRACKING:
                cap = 25 + (1 - near_saturation) * 15
                adjusted_time = min(adjusted_time, cap)
        
        if USE_VELOCITY_TRACKING and total_count > 0:
            moving_ratio = (moving_count + slow_count) / total_count
            if moving_ratio > 0.8:
                adjusted_time *= 0.7
        
        adjusted_time = max(min_green, min(max_green, adjusted_time))
        
        prev = prev_green.get(zone_name, adjusted_time)
        smoothed = ema_alpha * adjusted_time + (1 - ema_alpha) * prev
        
        green[zone_name] = smoothed
    
    return green
STRATEGIES = {
    'multisection': strategy_multisection, #heterogeneous traffic
    'hybrid_enhanced': strategy_hybrid_enhanced, #car dominant traffic
}

def run_video(source, zones_file, model_path, strategy='multisection', show_raw_detections=SHOW_ALL_DETECTIONS):
    zones_data, zone_format = load_zones(zones_file)
    if zones_data is None:
        print("Zones file not found. Run draw mode first.")
        return
    
    print(f"Loaded zones in '{zone_format}' format")
    print(f"Using strategy: {strategy}")
    if USE_VELOCITY_TRACKING:
        print("Velocity tracking: ENABLED")
    else:
        print("Velocity tracking: DISABLED (using count-only mode)")
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open source:", source)
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    
    if USE_VELOCITY_TRACKING:
        tracker = VelocityAwareTracker()
        print("Taking in to account vehicle velocity for tracking and duration adjustment")
    else:
        tracker = SimpleTracker()
        print("Using basic tracking")
    
    history = []
    prev_green = {z: MIN_GREEN for z in zones_data.keys()}

    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of stream")
            break
        
        frame = cv2.resize(frame, (1280, 720))
        frame_idx += 1
        tnow = time.time()
        
        res = model.predict(
            source=[frame],
            imgsz=640,
            conf=0.35,
            iou=0.6,
            classes=list(CLASS_IDS_TO_COUNT),
            verbose=False
        )
        
        preds = res[0]
        dets = []
        if hasattr(preds, 'boxes') and len(preds.boxes):
            for b in preds.boxes:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                cls_id = int(b.cls[0].item()) if hasattr(b, 'cls') else int(b.cls)
                conf = float(b.conf[0].item()) if hasattr(b, 'conf') else float(b.conf)
                if cls_id in CLASS_IDS_TO_COUNT:
                    dets.append((x1,y1,x2,y2, cls_id, conf))
        
        det2track = tracker.update(dets, frame_idx)
        
        zone_metrics = {}
        
        for zone_name, zone_data in zones_data.items():
            sections = zone_data['sections']
            
            zone_weighted_count = 0.0
            zone_occupancy = 0.0
            zone_vehicle_count = 0
            near_section_occupancy = 0.0
            section_scores = []
            
            stopped_vehicles = set()
            slow_vehicles = set()
            moving_vehicles = set()
            fast_vehicles = set()
            
            tracked_ids_in_zone = set()
            
            for section_idx, section in enumerate(sections):
                section_name = section['name']
                polygon = section['polygon']
                distance_weight = section['distance_weight']
                expected_capacity = section.get('expected_capacity', 10)
                
                section_bboxes = []
                section_tracked_ids = set()
                section_conf_sum = 0.0
                
                section_stopped = 0
                section_slow = 0
                section_moving = 0
                section_fast = 0
                
                for j, det in enumerate(dets):
                    bbox = det[:4]
                    conf = det[5]
                    tid = det2track[j]
                    
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = (bbox[1] + bbox[3]) / 2.0
                    
                    if point_in_poly((cx, cy), polygon):
                        section_bboxes.append(bbox)
                        section_tracked_ids.add(tid)
                        tracked_ids_in_zone.add(tid)
                        section_conf_sum += conf
                        
                        tracker.tracks[tid].setdefault('counted_zones', set()).add(zone_name)
                        
                        if USE_VELOCITY_TRACKING:
                            track = tracker.tracks.get(tid)
                            if track:
                                movement_state = track.get('movement_state', 'unknown')
                                
                                if movement_state == 'stopped':
                                    section_stopped += 1
                                    stopped_vehicles.add(tid)
                                elif movement_state == 'slow':
                                    section_slow += 1
                                    slow_vehicles.add(tid)
                                elif movement_state == 'moving':
                                    section_moving += 1
                                    moving_vehicles.add(tid)
                                elif movement_state == 'fast':
                                    section_fast += 1
                                    fast_vehicles.add(tid)
                
                section_vehicle_count = len(section_tracked_ids)
                section_occupancy = calculate_occupancy(polygon, section_bboxes)
                section_weighted_count = section_conf_sum
                
                hidden_estimate = estimate_hidden_vehicles(section_bboxes)
                adjusted_count = section_vehicle_count + hidden_estimate
                
                weighted_contribution = (
                    adjusted_count * distance_weight * 0.5 +
                    section_occupancy * distance_weight * 50
                )
                
                zone_weighted_count += section_conf_sum * distance_weight
                zone_occupancy += section_occupancy * distance_weight / len(sections)
                zone_vehicle_count += section_vehicle_count
                
                if section_name == 'near' or section_idx == 0:
                    near_section_occupancy = section_occupancy
                
                section_scores.append({
                    'name': section_name,
                    'count': section_vehicle_count,
                    'occupancy': section_occupancy,
                    'weighted_count': section_weighted_count,
                    'contribution': weighted_contribution,
                    'stopped': section_stopped,
                    'slow': section_slow,
                    'moving': section_moving,
                    'fast': section_fast
                })
            
            zone_metrics[zone_name] = {
                'vehicle_count': zone_vehicle_count,
                'weighted_count': zone_weighted_count,
                'occupancy': zone_occupancy,
                'near_saturation': near_section_occupancy,
                'section_scores': section_scores,
                'tracked_ids': tracked_ids_in_zone,
                'stopped_count': len(stopped_vehicles),
                'slow_count': len(slow_vehicles),
                'moving_count': len(moving_vehicles),
                'fast_count': len(fast_vehicles),
                'total_count': len(tracked_ids_in_zone)
            }
        
        history.append((tnow, zone_metrics))
        cutoff = tnow - WINDOW_SEC
        history = [h for h in history if h[0] >= cutoff]
        
        aggregated_metrics = {}
        for zone_name in zones_data.keys():
            all_ids = set()
            weighted_counts = []
            occupancies = []
            near_sats = []
            stopped_sets = []
            slow_sets = []
            moving_sets = []
            
            for _, zmetrics in history:
                if zone_name in zmetrics:
                    all_ids |= zmetrics[zone_name]['tracked_ids']
                    weighted_counts.append(zmetrics[zone_name]['weighted_count'])
                    occupancies.append(zmetrics[zone_name]['occupancy'])
                    near_sats.append(zmetrics[zone_name]['near_saturation'])
                    
                    if USE_VELOCITY_TRACKING:
                        stopped_sets.append(zmetrics[zone_name]['stopped_count'])
                        slow_sets.append(zmetrics[zone_name]['slow_count'])
                        moving_sets.append(zmetrics[zone_name]['moving_count'])
            
            aggregated_metrics[zone_name] = {
                'unique_count': len(all_ids),
                'weighted_count': np.mean(weighted_counts) if weighted_counts else 0,
                'occupancy': np.max(occupancies) if occupancies else 0,
                'near_saturation': np.max(near_sats) if near_sats else 0,
                'section_scores': zone_metrics[zone_name]['section_scores'],
                'stopped_count': int(np.mean(stopped_sets)) if stopped_sets else 0,
                'slow_count': int(np.mean(slow_sets)) if slow_sets else 0,
                'moving_count': int(np.mean(moving_sets)) if moving_sets else 0,
                'total_count': len(all_ids)
            }
        
        green_times = STRATEGIES[strategy](
            aggregated_metrics, prev_green, 
            MIN_GREEN, MAX_GREEN, ALLOCATABLE, EMA_ALPHA
        )
        prev_green = green_times
        
        vis = frame.copy()
        
        for zone_name, zone_data in zones_data.items():
            sections = zone_data['sections']
            
            for section_idx, section in enumerate(sections):
                polygon = section['polygon']
                section_name = section['name']
                pts = np.array(polygon, np.int32)
                
                colors = {
                    'near': (0, 255, 0),
                    'mid': (0, 200, 200),
                    'far': (0, 100, 255),
                    'full': (0, 255, 255)
                }
                color = colors.get(section_name, (0, 255, 255))
                
                cv2.polylines(vis, [pts], True, color, 2)
                
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, 0.05, vis, 0.95, 0, vis)
                
                if section_name in ['near', 'mid', 'far', 'full']:
                    section_metric = next(
                        (s for s in zone_metrics[zone_name]['section_scores'] 
                         if s['name'] == section_name), 
                        None
                    )
                    
                    if section_metric:
                        cx = int(np.mean(pts[:,0]))
                        cy = int(np.mean(pts[:,1]))
                        
                        label = f"{section_name[:3].upper()}: {section_metric['count']}v"
                        draw_text(vis, label, (cx+30, cy-10), color, scale=0.5, thick=1)
                        
                        occ_pct = int(section_metric['occupancy'] * 100)
                        draw_text(vis, f"Occ:{occ_pct}%", (cx+30, cy+8), color, scale=0.4, thick=1)
            
            zone_center_x = int(np.mean([np.mean(np.array(s['polygon'])[:,0]) 
                                        for s in sections]))
            zone_center_y = int(np.mean([np.mean(np.array(s['polygon'])[:,1]) 
                                        for s in sections]))
            
            total_count = aggregated_metrics[zone_name]['unique_count']
            green_sec = int(round(green_times[zone_name]))
            
            draw_text(vis, f"{zone_name}", 
                     (zone_center_x-40, zone_center_y-35), 
                     (255,255,0), scale=0.7, thick=2)
            draw_text(vis, f"Total: {total_count}v", 
                     (zone_center_x-40, zone_center_y-15), 
                     (0,255,0), scale=0.6, thick=1)
            draw_text(vis, f"Green: {green_sec}s", 
                     (zone_center_x-40, zone_center_y+5), 
                     (0,200,255), scale=0.6, thick=1)
        
        if show_raw_detections:
            for det in dets:
                x1, y1, x2, y2, cls_id, conf = det
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                draw_text(vis, f"{conf:.2f}", (x1, y1-3), (0, 255, 255), scale=0.4, thick=1)
        
        active_tracks = 0
        for tid, tr in tracker.tracks.items():
            frames_since_seen = frame_idx - tr['last_seen']
            
            if USE_VELOCITY_TRACKING:
                movement_state = tr.get('movement_state', 'unknown')
                velocity = tr.get('velocity', 0)
                
                colors_vel = {
                    'stopped': (0, 0, 255),
                    'slow': (0, 165, 255),
                    'moving': (0, 255, 255),
                    'fast': (0, 255, 0),
                    'unknown': (128, 128, 128)
                }
                color = colors_vel.get(movement_state, (255, 255, 255))
                thickness = 2
                
                if frames_since_seen <= 5 and tr['age'] >= 2:
                    active_tracks += 1
            else:
                if frames_since_seen > 5:
                    color = (0, 0, 200)
                    thickness = 1
                elif tr['age'] < 2:
                    color = (255, 200, 0)
                    thickness = 2
                else:
                    color = (0, 220, 0)
                    thickness = 2
                    active_tracks += 1
            
            x1, y1, x2, y2 = map(int, tr['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            if USE_VELOCITY_TRACKING:
                label = f"ID:{tid} {movement_state[:3].upper()} {velocity:.1f}"
            else:
                label = f"ID:{tid}"
                if tr['age'] < 2:
                    label += " (new)"
            
            draw_text(vis, label, (x1, y1-6), color, scale=0.4, thick=1)
            
            if 'conf' in tr:
                draw_text(vis, f"{tr['conf']:.2f}", (x2-35, y1-6), color, scale=0.4, thick=1)
            
            if USE_VELOCITY_TRACKING:
                velocity = tr.get('velocity', 0)
                if velocity > 2.0 and len(tr.get('position_history', [])) >= 2:
                    positions = list(tr['position_history'])
                    if len(positions) >= 2:
                        prev_pos = positions[-2]
                        curr_pos = positions[-1]
                        
                        cv2.arrowedLine(vis, 
                                       (int(prev_pos[0]), int(prev_pos[1])),
                                       (int(curr_pos[0]), int(curr_pos[1])),
                                       color, 2, tipLength=0.3)
        
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (430, 190), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.3, 0, vis)
        
        hud_y = 30
        vel_status = "+VEL" if USE_VELOCITY_TRACKING else ""
        draw_text(vis, f"Strategy: {strategy.upper()}{vel_status}", (10, hud_y), (255,255,0), scale=0.9, thick=3)
        draw_text(vis, f"Format: {zone_format.upper()}", (10, hud_y+32), (255,200,0), scale=0.7, thick=3)
        draw_text(vis, f"Tracks: {len(tracker.tracks)} | Active: {active_tracks} | Dets: {len(dets)}", 
                 (10, hud_y+60), (255,255,255), scale=0.7, thick=1)
        
        zone_summary = ', '.join([
            f"{k}:{aggregated_metrics[k]['unique_count']}v({int(aggregated_metrics[k]['occupancy']*100)}%)"
            for k in aggregated_metrics.keys()
        ])
        draw_text(vis, f"Zones: {zone_summary}", (10, hud_y+88), (200,255,200), scale=0.6, thick=1)
        
        green_summary = ', '.join([f"{k}={int(round(v))}s" for k,v in green_times.items()])
        draw_text(vis, f"Green: {green_summary}", (10, hud_y+116), (100,200,255), scale=0.7, thick=1)
        
        fps_val = 1.0/max(0.001, time.time()-tnow)
        draw_text(vis, f"Window:{WINDOW_SEC}s | FPS:{fps_val:.1f}", 
                 (10, hud_y+144), (200,200,200), scale=0.6, thick=1)
        
        legend_y = frame.shape[0] - 120
        cv2.rectangle(vis, (5, legend_y-5), (150, legend_y+130), (0,0,0), -1)
        cv2.rectangle(vis, (5, legend_y-5), (150, legend_y+130), (255,255,255), 1)
        draw_text(vis, "LEGEND:", (15, legend_y+10), (255,255,255), scale=0.5, thick=2)
        
        cv2.line(vis, (15, legend_y+24), (40, legend_y+24), (0,255,0), 2)
        draw_text(vis, "Near section", (45, legend_y+26), (255,255,255), scale=0.4, thick=1)
        cv2.line(vis, (15, legend_y+36), (40, legend_y+36), (0,200,200), 2)
        draw_text(vis, "Mid section", (45, legend_y+40), (255,255,255), scale=0.4, thick=1)
        cv2.line(vis, (15, legend_y+48), (40, legend_y+48), (0,100,255), 2)
        draw_text(vis, "Far section", (45, legend_y+52), (255,255,255), scale=0.4, thick=1)

        if USE_VELOCITY_TRACKING:
            cv2.rectangle(vis, (15, legend_y+58), (40, legend_y+68), (0,0,255), 2)
            draw_text(vis, "Stopped", (45, legend_y+67), (255,255,255), scale=0.4, thick=1)
            cv2.rectangle(vis, (15, legend_y+75), (40, legend_y+85), (0,165,255), 2)
            draw_text(vis, "Slow", (45, legend_y+83), (255,255,255), scale=0.4, thick=1)
            cv2.rectangle(vis, (15, legend_y+90), (40, legend_y+100), (0,255,255), 2)
            draw_text(vis, "Moving", (45, legend_y+99), (255,255,255), scale=0.4, thick=1)
            cv2.rectangle(vis, (15, legend_y+105), (40, legend_y+115), (0,255,0), 2)
            draw_text(vis, "Fast", (45, legend_y+114), (255,255,255), scale=0.4, thick=1)
        else:
            cv2.rectangle(vis, (15, legend_y+52), (40, legend_y+62), (0,220,0), 2)
            draw_text(vis, "Active track", (45, legend_y+60), (255,255,255), scale=0.4, thick=1)
            cv2.rectangle(vis, (15, legend_y+67), (40, legend_y+77), (255,200,0), 2)
            draw_text(vis, "New track", (45, legend_y+75), (255,255,255), scale=0.4, thick=1)
        
        cv2.imshow("zone_adaptive_velocity", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_raw_detections = not show_raw_detections
            print(f"Raw detections: {show_raw_detections}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    def smart_source(value):
        try:
            return int(value)
        except ValueError:
            return value
    
    parser = argparse.ArgumentParser(
        description="Adaptive traffic light timing with velocity tracking"
    )
    parser.add_argument("--mode", choices=["draw","run"], default="run")
    parser.add_argument("--image", type=str, help="Image for draw mode")
    parser.add_argument("--zones", type=str, default="zones.json")
    parser.add_argument("--source", type=smart_source, default=0)
    parser.add_argument("--model", type=str, default=MODEL_PATH)
    parser.add_argument("--strategy", choices=list(STRATEGIES.keys()), default='multisection')
    parser.add_argument("--show-raw", action='store_true', help="Show raw detections")
    args = parser.parse_args()

    if args.mode == "draw":
        if not args.image:
            print("Provide --image for draw mode")
        else:
            interactive_draw_multisection_zones(args.image, args.zones)
    else:
        run_video(args.source, args.zones, args.model, args.strategy, args.show_raw)