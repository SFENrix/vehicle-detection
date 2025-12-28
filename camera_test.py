# test_cameras.py
import cv2

print("Testing camera indices...")
for i in range(10):  # Test indices 0-9
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ Camera {i}: Working - Resolution {frame.shape[1]}x{frame.shape[0]}")
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(1000)  # Show for 1 second
        cap.release()
    else:
        print(f"❌ Camera {i}: Not available")
    
cv2.destroyAllWindows()
print("\nOBS Virtual Camera is usually index 1 or 2")