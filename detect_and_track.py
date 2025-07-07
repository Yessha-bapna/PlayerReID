import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLO model (no special torch config needed after downgrade)
model = YOLO("yolov8n.pt")  # Or yolov8s.pt for slightly bigger


# Initialize Deep SORT
tracker = DeepSort(max_age=30)

# Input/output video
video_path = '15sec_input_720p.mp4'
cap = cv2.VideoCapture(video_path)

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output writer
out = cv2.VideoWriter('output/tracked_output.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO
    results = model(frame)[0]
    detections = []

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = float(result.conf[0])
        cls_id = int(result.cls[0])

        # Only track persons (assumes class 0 = person)
        if cls_id == 0 and conf > 0.3:
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Track with Deep SORT
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
