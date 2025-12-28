""""
python train_model.py

"""

from ultralytics import YOLO
from pathlib import Path
import shutil

# CONFIGURATION
DATA = "dataset/data.yaml"
MODEL = "yolov8m.pt"
PROJECT = "runs/vehicle"
NAME = "vehicle_detector"
EPOCHS = 300
IMGSZ = 640
BATCH = 4

# Train 
model = YOLO(MODEL)
model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    project=PROJECT,
    name=NAME,
    exist_ok=True,
    augment=True,
    workers=0  
)


best = Path(PROJECT) / NAME / "weights" / "best.pt"
if best.exists():
    shutil.copy2(best, Path.cwd() / "vehicle_detector.pt")
    print("Saved best model -> vehicle_detector.pt")
else:
    print("best.pt not found")
