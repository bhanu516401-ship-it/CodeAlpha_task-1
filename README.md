Name:G.Bhanu prakash reddy
ID:CA/SE1/18257
Domain:Artificial intelligence
Duration:10th OCT 2025 to 10th NOV 2025
# CodeAlpha_task-1
import cv2
import numpy as np
from sort import Sort  # For SORT tracking

# Load YOLO model (weights, config, and class labels)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Load class labels (from coco.names)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the SORT tracker
tracker = Sort()

# Set up webcam (or video file)
cap = cv2.VideoCapture(0)  # '0' for webcam, replace with file path for video

def detect_objects(frame):
    """Function to detect objects using YOLO"""
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Threshold for detection
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w // 2
                y = center_y - h // 2

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    boxes, confidences, class_ids = detect_objects(frame)

    # Format boxes for tracker (x1, y1, x2, y2, confidence)
    detection_boxes = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        detection_boxes.append([x, y, x + w, y + h, confidences[i]])

    # Update tracker
    tracked_objects = tracker.update(np.array(detection_boxes))

    # Draw bounding boxes and labels
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        label = str(classes[class_ids[int(track_id)]] if int(track_id) < len(classes) else "Unknown")
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {int(track_id)} {label}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Object Detection and Tracking", frame)

    # Exit if 'q' is pressed
    if cv2.waitKe
<img width="731" height="491" alt="Screenshot 2025-11-01 141006" src="https://github.com/user-attachments/assets/6e3797dc-ae84-4684-85bb-4039e6a8dc6d" />
