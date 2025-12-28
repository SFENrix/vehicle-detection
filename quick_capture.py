import cv2

TARGET_W, TARGET_H = 1280, 720  # must match runtime resize

cap = cv2.VideoCapture(2)  # OBS virtual cam index
ret, frame = cap.read()

if ret:
    frame_resized = cv2.resize(frame, (TARGET_W, TARGET_H))
    cv2.imwrite("obs_sample_frame_1280x720.jpg", frame_resized)

    print("✅ Frame captured & resized: obs_sample_frame.jpg")
    cv2.imshow("Captured Frame (1280x720)", frame_resized)
    cv2.waitKey(0)
else:
    print("❌ Failed to capture")

cap.release()
cv2.destroyAllWindows()
