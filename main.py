from ultralytics import YOLO
import cv2
from datetime import datetime
import os

# Load local pretrained model
model = YOLO("yolo12s_RDD2022_best.pt")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # If damage detected
    if len(results[0].boxes) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/damage_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print("Road damage saved:", filename)

    # Show live detection
    cv2.imshow("Smart Road Monitoring", results[0].plot())

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()