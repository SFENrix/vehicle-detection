# check_model.py
from ultralytics import YOLO

def run_validation():
    model = YOLO("vehicle_detector.pt")   # ubah jika nama beda
    # jalankan val; ubah imgsz/other args kalau perlu
    results = model.val(data="dataset/data.yaml", imgsz=960)
    # print ringkasan dari object results (Ultralytics juga sudah print tabel otomatis)
    try:
        print("\n=== SUMMARY METRICS ===")
        print(f"mAP50: {results.box.map50:.3f}")
        print(f"mAP50-95: {results.box.map:.3f}")
        print(f"Precision (overall): {results.box.p:.3f}")
        print(f"Recall (overall): {results.box.r:.3f}")
    except Exception:
        # fallback: hasil mungkin berbeda struktur tergantung versi ultralytics
        print("Validation finished. Check printed table above for full metrics.")

if __name__ == "__main__":
    run_validation()
