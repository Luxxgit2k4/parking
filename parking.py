import cv2
from cv2.gapi import video
import torch
import numpy as np
from typing import Any

# Load the YOLOv5 model
model: Any = torch.hub.load('/home/lakshmanan/projects/parkingspace/yolov5', 'custom',
                            path='/home/lakshmanan/projects/parkingspace/model/best.pt',
                            source='local')

# Check if model loaded correctly
if hasattr(model, 'names'):
    print("Model loaded successfully, class names available.")
else:
    print("Error: Model does not have 'names' attribute. Check loading process.")
    exit()

def process_detections(frame, results, confidence_threshold=0.3):
    total_spaces = 0
    filled_spaces = 0
    data = []

    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()

        if confidence < confidence_threshold:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)

        if cls == 0:
            total_spaces += 1
            data.append(0)
            color = (0, 0, 255)  # Red for empty spaces
        elif cls == 1:
            total_spaces += 1
            filled_spaces += 1
            data.append(1)
            color = (0, 255, 0)  # Green for occupied spaces
        else:
            continue

        label = f"{model.names[cls]} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    not_filled_spaces = total_spaces - filled_spaces
    output = {
        "Total spaces": total_spaces,
        "Filled": filled_spaces,
        "Not Filled": not_filled_spaces,
        "Data": data
    }

    return frame, output
url = "http://192.168.1.4:4747/video"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Error: Unable to access the camera or video feed.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame from feed.")
        break

    # Convert BGR to RGB before passing to model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ensure model is callable
    try:
        results = model(rgb_frame)
    except TypeError as e:
        print(f"Error: Model inference failed. {e}")
        break

    frame_with_boxes, parking_data = process_detections(frame, results)
    print(parking_data)

    cv2.imshow("Parking Space Detection", frame_with_boxes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

