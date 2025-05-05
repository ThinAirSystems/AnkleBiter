import cv2
import numpy as np
import time

class DroneDetector:
    def __init__(self, use_pretrained_model=True):
        self.use_pretrained_model = use_pretrained_model
        
        # Initialize variables for motion detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, 
            varThreshold=16, 
            detectShadows=False
        )
        
        # For object tracking
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking_initialized = False
        self.track_box = None
        
        # For deep learning model detection
        if use_pretrained_model:
            # Load YOLOv4 or another model trained for drone detection
            # Note: You'll need to download weights file and cfg file
            self.net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
            self.model = cv2.dnn_DetectionModel(self.net)
            self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
            
            # Load class names
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Drone is typically class index 0 in a custom model, but in COCO might be identified as:
            # "airplane" (5) or similar - you'll need to adjust based on your model
            self.drone_class_ids = [0]  # Adjust based on your model's classes
    
    def detect_with_motion(self, frame):
        """Simple motion-based detection"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and aspect ratio
        drone_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 400:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                # Drones typically have aspect ratios close to 1 (square-ish)
                aspect_ratio = float(w) / h
                if 0.5 < aspect_ratio < 2.0:
                    drone_contours.append((x, y, w, h))
        
        return drone_contours
    
    def detect_with_model(self, frame):
        """Deep learning based detection"""
        classes, scores, boxes = self.model.detect(frame, confThreshold=0.6, nmsThreshold=0.4)
        
        detections = []
        for class_id, score, box in zip(classes, scores, boxes):
            if class_id in self.drone_class_ids:
                x, y, w, h = box
                detections.append((x, y, w, h))
        
        return detections
    
    def process_frame(self, frame):
        """Main processing function for each frame"""
        result_frame = frame.copy()
        detections = []
        
        # Detect using either motion detection or deep learning model
        if self.use_pretrained_model:
            detections = self.detect_with_model(frame)
        else:
            detections = self.detect_with_motion(frame)
        
        # Draw bounding boxes
        for x, y, w, h in detections:
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result_frame, "Drone", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # If we want to start tracking a detected drone
            if not self.tracking_initialized and len(detections) > 0:
                self.track_box = (x, y, w, h)
                self.tracker.init(frame, (x, y, w, h))
                self.tracking_initialized = True
        
        # If no detections but we're tracking, update the tracker
        if len(detections) == 0 and self.tracking_initialized:
            success, box = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(result_frame, "Tracked Drone", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                self.tracking_initialized = False
        
        return result_frame, len(detections) > 0

def main():
    # Initialize camera capture
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide a video file path
    
    # If you're using a video file for testing:
    # cap = cv2.VideoCapture("drone_video.mp4")
    
    # Initialize detector
    # Set use_pretrained_model=True to use deep learning (more accurate but requires model files)
    # Set use_pretrained_model=False to use simple motion detection (works without model files)
    detector = DroneDetector(use_pretrained_model=False)
    
    # Set up FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        result_frame, detected = detector.process_frame(frame)
        
        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Display FPS and detection status
        status = "Drone Detected" if detected else "No Drone"
        cv2.putText(result_frame, f"FPS: {fps} | Status: {status}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow("Drone Detection", result_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()