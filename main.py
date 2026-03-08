from ultralytics import YOLO
import cv2
from datetime import datetime
import os
import geocoder
import csv

# Load model
model = YOLO("yolo12s_RDD2022_best.pt")

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Create CSV file
if not os.path.exists("detections.csv"):
    with open("detections.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Latitude", "Longitude", "Image"])

# Function to get location
def get_location():
    g = geocoder.ip('me')
    return g.latlng

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    # Temporary test (force saving)
    if True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/damage_{timestamp}.jpg"

        cv2.imwrite(filename, frame)

        location = get_location()

        if location:
            lat, lon = location
            print(f"Road damage saved: {filename}")
            print(f"Location: {lat}, {lon}")

            with open("detections.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, lat, lon, filename])

    # Show detection window
    cv2.imshow("Smart Road Monitoring", results[0].plot())

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
