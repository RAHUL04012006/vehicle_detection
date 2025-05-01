import cv2
import torch
import numpy as np
from collections import defaultdict

def detect_vehicles(video_path):
    # Load the smaller YOLOv5 model for faster processing
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Set confidence threshold
    model.conf = 0.5  # Confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions (reduce size for faster processing)
    new_width = 640  # Standard YOLO input size
    new_height = int(height * (new_width / width))
    
    # Vehicle classes we want to count
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    
    # Dictionary to store vehicle counts per class
    class_counts = defaultdict(int)
    
    # Frame skipping counter
    frame_skip = 0
    skip_frames = 1  # Process every other frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Skip frames to improve performance
        frame_skip += 1
        if frame_skip % (skip_frames + 1) != 0:
            continue
            
        # Resize frame for faster processing
        frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert frame to RGB (YOLOv5 expects RGB)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = model(img)
        
        # Get detection results
        detections = results.xyxy[0]
        
        # Reset counts for this frame
        for cls in vehicle_classes:
            class_counts[cls] = 0
        
        # Process detections
        for *box, conf, cls in detections:
            label = model.names[int(cls)]
            if label in vehicle_classes:
                class_counts[label] += 1
                
                # Draw bounding box
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate total vehicles
        total_vehicles = sum(class_counts.values())
        
        # Add counts to the frame
        y_offset = 30
        cv2.putText(frame, f'Total Vehicles: {total_vehicles}', (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display counts for each vehicle type
        for cls, count in class_counts.items():
            y_offset += 30
            cv2.putText(frame, f'{cls}: {count}', (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow('Vehicle Detection', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to your video file
    video_path = "vehicle.mp4"
    detect_vehicles(video_path)
