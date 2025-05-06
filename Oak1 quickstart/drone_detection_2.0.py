import cv2
import depthai as dai
import numpy as np
import time
import argparse
from collections import deque

# Add command line arguments for modes
parser = argparse.ArgumentParser(description='OAK-1 Drone Detection')
parser.add_argument('--production', action='store_true', help='Enable production mode with no video display')
parser.add_argument('--model', type=str, default="mobilenet-ssd", help='Model to use for detection')
parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold (0.0-1.0)')
args = parser.parse_args()

# Debug mode is the default (video output)
DEBUG_MODE = not args.production
print(f"Running in {'PRODUCTION' if args.production else 'DEBUG'} mode")

# --- Configuration ---
NN_MODEL = args.model
NN_INPUT_SIZE = 300
CONFIDENCE_THRESHOLD = args.confidence  # Increased from 0.5 to reduce false positives
MIN_DETECTION_FRAMES = 3  # Number of consecutive frames needed for valid detection
DETECTION_HISTORY_SIZE = 10  # Store last N detections for tracking

# Size filtering - ignore unrealistic detections (adjust these based on your use case)
MIN_DRONE_SIZE = 20  # Minimum width or height in pixels
MAX_DRONE_SIZE = 250  # Maximum width or height in pixels

# Non-maximum suppression parameters
NMS_THRESHOLD = 0.4  # For filtering overlapping detections

# Mobilenet SSD label mapping
label_map = ["background", "drone", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Specify which labels are drones (in this case, likely "drone" and possibly "bird")
DRONE_LABELS = ["drone", "bird"]  # Add other relevant labels as needed

# Detection history queue for temporal filtering
detection_history = deque(maxlen=DETECTION_HISTORY_SIZE)

# Helper to convert normalized NN coordinates to pixel coordinates
def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# Apply non-maximum suppression to filter overlapping detections
def apply_nms(detections):
    if not detections:
        return []
    
    boxes = []
    scores = []
    classes = []
    
    for detection in detections:
        boxes.append([detection.xmin, detection.ymin, detection.xmax, detection.ymax])
        scores.append(detection.confidence)
        classes.append(detection.label)
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # NMS is applied per class
    filtered_indices = []
    unique_classes = np.unique(classes)
    
    for cls in unique_classes:
        cls_indices = np.where(classes == cls)[0]
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]
        
        # Apply NMS
        keep_indices = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(), 
            cls_scores.tolist(), 
            CONFIDENCE_THRESHOLD, 
            NMS_THRESHOLD
        )
        
        if len(keep_indices) > 0:
            # OpenCV 4.5.4+ returns a flat array
            if isinstance(keep_indices, tuple):
                keep_indices = keep_indices[0]
            
            filtered_indices.extend(cls_indices[keep_indices])
    
    # Return filtered detections
    return [detections[i] for i in filtered_indices]

# Helper function to draw detections (only used in debug mode)
def display_frame(name, frame, detections, valid_drones):
    if frame is None:
        print("Received None frame, cannot display.")
        return
    
    # Only process in debug mode
    if not DEBUG_MODE:
        return
    
    # Draw all detections in yellow (candidates)
    for detection in detections:
        try:
            # Denormalize bounding box
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            
            # Default color is yellow (candidate)
            color = (0, 255, 255)
            
            # Check if this is a confirmed drone (green)
            for drone in valid_drones:
                if np.array_equal(drone["bbox"], bbox):
                    color = (0, 255, 0)  # Green for confirmed drones
                    break
                    
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Get label text
            label_index = detection.label
            if 0 <= label_index < len(label_map):
                label = label_map[label_index]
            else:
                label = f"Label {label_index}"  # Fallback
                
            cv2.putText(frame, label, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            
            # Get confidence text
            confidence = f"{int(detection.confidence * 100)}%"
            cv2.putText(frame, confidence, (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        except Exception as e:
            print(f"Error drawing detection: {e}, Detection data: {detection}")

    # Show the frame only in debug mode
    cv2.imshow(name, frame)

# Update detection history and determine if the detection is stable
def update_detection_history(drone_info):
    global detection_history
    
    # Add current detection to history
    detection_history.append(drone_info)
    
    # If we don't have enough history yet, return False
    if len(detection_history) < MIN_DETECTION_FRAMES:
        return False
    
    # Check last N frames for consistent detection
    recent_frames = list(detection_history)[-MIN_DETECTION_FRAMES:]
    
    # Count detections with similar position and size
    matching_count = 0
    reference = recent_frames[-1]
    
    for det in recent_frames:
        # Check if centers are close (within 20% of width/height)
        dist_threshold = max(reference["width"], reference["height"]) * 0.2
        
        center_dist = np.sqrt(
            (det["center_x"] - reference["center_x"])**2 + 
            (det["center_y"] - reference["center_y"])**2
        )
        
        # Check if size is similar (within 30%)
        size_ratio_w = det["width"] / reference["width"] if reference["width"] > 0 else 0
        size_ratio_h = det["height"] / reference["height"] if reference["height"] > 0 else 0
        
        size_match = (0.7 < size_ratio_w < 1.3) and (0.7 < size_ratio_h < 1.3)
        
        if center_dist < dist_threshold and size_match:
            matching_count += 1
    
    # Return True if we have enough matching frames
    return matching_count >= MIN_DETECTION_FRAMES

# Process and filter drone detections
def process_drone_detections(frame, detections):
    # Apply NMS to filter redundant detections
    filtered_detections = apply_nms(detections)
    
    drone_candidates = []
    valid_drones = []
    
    for detection in filtered_detections:
        label_index = detection.label
        if 0 <= label_index < len(label_map):
            label = label_map[label_index]
            
            # Check if detected object is a potential drone
            if label in DRONE_LABELS and detection.confidence >= CONFIDENCE_THRESHOLD:
                # Get bounding box in pixel coordinates
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                # Calculate center position and size
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Apply size filtering
                if width < MIN_DRONE_SIZE or height < MIN_DRONE_SIZE:
                    continue  # Too small, probably noise
                
                if width > MAX_DRONE_SIZE or height > MAX_DRONE_SIZE:
                    continue  # Too large, probably a false positive
                
                # Store drone data
                drone_info = {
                    "label": label,
                    "confidence": detection.confidence,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height,
                    "bbox": bbox,
                    "aspect_ratio": width / height if height > 0 else 0
                }
                
                # Add to drone candidates
                drone_candidates.append(drone_info)
                
                # Check for temporal consistency
                if update_detection_history(drone_info):
                    valid_drones.append(drone_info)
                    print(f"CONFIRMED DRONE: {label} ({detection.confidence:.2f}) at ({center_x}, {center_y}), size: {width}x{height}")
    
    return drone_candidates, valid_drones

# Motion detection background subtractor (optional)
if DEBUG_MODE:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

# Motion validation
def validate_motion(frame, drone_info, prev_frame):
    if prev_frame is None or frame is None:
        return True  # Can't verify motion without previous frame
    
    # Extract region of interest around detection
    bbox = drone_info["bbox"]
    x1, y1, x2, y2 = bbox
    
    # Add some padding
    pad = 20
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad)
    y2 = min(frame.shape[0], y2 + pad)
    
    # Extract ROI from current and previous frame
    roi_curr = frame[y1:y2, x1:x2]
    roi_prev = prev_frame[y1:y2, x1:x2]
    
    # Make sure ROIs are valid
    if roi_curr.size == 0 or roi_prev.size == 0 or roi_curr.shape != roi_prev.shape:
        return True
    
    # Calculate absolute difference
    diff = cv2.absdiff(roi_curr, roi_prev)
    
    # Convert to grayscale
    if len(diff.shape) == 3:
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    else:
        diff_gray = diff
        
    # Apply threshold
    _, thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels (motion)
    motion_pixels = np.count_nonzero(thresh)
    
    # Calculate percentage of motion
    total_pixels = thresh.shape[0] * thresh.shape[1]
    motion_percentage = motion_pixels / total_pixels if total_pixels > 0 else 0
    
    # Return True if enough motion is detected
    return motion_percentage > 0.05  # 5% of pixels need to show movement

try:
    # --- Define the Pipeline ---
    pipeline = dai.Pipeline()

    # 1. Define sources
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE)  # Match NN input
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # 2. Define Neural Network node
    print(f"Loading Neural Network model: {NN_MODEL}")
    detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

    # Define the MODEL_PATH
    from pathlib import Path
    MODEL_PATH = str(Path("C:/Users/armer/OneDrive/ThinAir_Main/ThinAirSystems_Root/5 - Active projects/CUAS/Oak1 quickstart/models/mobilenet-ssd_openvino_2021.2_6shave.blob"))
    detection_nn.setBlobPath(MODEL_PATH)
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.input.setBlocking(False)

    # 3. Define Output nodes
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    # 4. Linking
    cam_rgb.preview.link(detection_nn.input)
    
    # Only link RGB output in debug mode to save bandwidth
    if DEBUG_MODE:
        detection_nn.passthrough.link(xout_rgb.input)  # Use passthrough for synchronized frame
    
    detection_nn.out.link(xout_nn.input)

    print("Pipeline defined. Attempting to connect to device...")

    # --- Connect and Run ---
    found, device_info = dai.Device.getFirstAvailableDevice()
    if not found:
        print("No OAK device found. Make sure it's connected.")
        exit()

    with dai.Device(pipeline, device_info) as device:
        print(f"Connected to OAK device: {device.getDeviceInfo().name}")
        print("Downloading/Compiling model blob if necessary (this may take a moment)...")

        # Get output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False) if DEBUG_MODE else None
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        prev_frame = None
        detections = []

        print("Starting object detection loop...")
        while True:
            # Process RGB frame if in debug mode
            if DEBUG_MODE and q_rgb is not None:
                in_rgb = q_rgb.tryGet()
                if in_rgb is not None:
                    prev_frame = frame.copy() if frame is not None else None
                    frame = in_rgb.getCvFrame()
                    
                    # Apply background subtraction for motion (debug visualization)
                    if frame is not None:
                        fg_mask = bg_subtractor.apply(frame)
                        cv2.imshow("Motion Detection", fg_mask)

            # Always process neural network results
            in_nn = q_nn.tryGet()
            if in_nn is not None:
                detections = in_nn.detections
                
                # Process drone detections
                if frame is not None or not DEBUG_MODE:
                    # In production mode, we don't need the actual frame content
                    # Just pass a dummy frame with correct dimensions for coordinate calculations
                    if not DEBUG_MODE and frame is None:
                        frame_dimensions = (NN_INPUT_SIZE, NN_INPUT_SIZE, 3)
                        dummy_frame = np.zeros(frame_dimensions, dtype=np.uint8)
                        drone_candidates, valid_drones = process_drone_detections(dummy_frame, detections)
                    else:
                        drone_candidates, valid_drones = process_drone_detections(frame, detections)
                        
                        # Additional motion validation if we have previous frame
                        if prev_frame is not None:
                            for drone in valid_drones:
                                if not validate_motion(frame, drone, prev_frame):
                                    print(f"Warning: Drone at ({drone['center_x']}, {drone['center_y']}) failed motion validation, may be static object")

            # If we have a valid frame and in debug mode, display it with current detections
            if DEBUG_MODE and frame is not None:
                display_frame("OAK-1 Object Detection", frame, detections, valid_drones if 'valid_drones' in locals() else [])
            
            # Small delay if no frame yet
            if frame is None and DEBUG_MODE:
                time.sleep(0.001)

            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1) if DEBUG_MODE else -1
            if key == ord('q'):
                break

except Exception as e:
    print(f"An error occurred during object detection: {e}")
    import traceback
    traceback.print_exc()
    print("Make sure the OAK device is connected, libraries are installed, and the model is accessible.")

finally:
    # Close OpenCV windows if in debug mode
    if DEBUG_MODE:
        cv2.destroyAllWindows()
    print("Detection stopped or failed.")