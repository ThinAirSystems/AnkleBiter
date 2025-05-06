import cv2
import depthai as dai
import numpy as np
import time
import argparse

# Add command line arguments for modes
parser = argparse.ArgumentParser(description='OAK-1 Drone Detection')
parser.add_argument('--production', action='store_true', help='Enable production mode with no video display')
parser.add_argument('--model', type=str, default="mobilenet-ssd", help='Model to use for detection')
args = parser.parse_args()

# Debug mode is the default (video output)
DEBUG_MODE = not args.production
print(f"Running in {'PRODUCTION' if args.production else 'DEBUG'} mode")

# --- Configuration ---
NN_MODEL = args.model
NN_INPUT_SIZE = 300
CONFIDENCE_THRESHOLD = 0.5

# Mobilenet SSD label mapping
label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Specify which labels are drones (in this case, likely "aeroplane")
DRONE_LABELS = ["aeroplane"]  # Add other relevant labels as needed

# Helper to convert normalized NN coordinates to pixel coordinates
def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# Helper function to draw detections (only used in debug mode)
def display_frame(name, frame, detections):
    if frame is None:
        print("Received None frame, cannot display.")
        return
    
    # Only process in debug mode
    if not DEBUG_MODE:
        return
    
    color = (0, 255, 0)  # Green
    for detection in detections:
        try:
            # Denormalize bounding box
            bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
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

# Process and report drone detections
def process_drone_detections(frame, detections):
    drone_data = []
    
    for detection in detections:
        label_index = detection.label
        if 0 <= label_index < len(label_map):
            label = label_map[label_index]
            
            # Check if detected object is a drone
            if label in DRONE_LABELS and detection.confidence >= CONFIDENCE_THRESHOLD:
                # Get bounding box in pixel coordinates
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                
                # Calculate center position and size
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                
                # Store drone data
                drone_info = {
                    "label": label,
                    "confidence": detection.confidence,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height,
                    "bbox": bbox
                }
                drone_data.append(drone_info)
                
                # Print detection data
                print(f"DRONE DETECTED: {label} ({detection.confidence:.2f}) at ({center_x}, {center_y}), size: {width}x{height}")
    
    return drone_data

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
        detections = []

        print("Starting object detection loop...")
        while True:
            # Process RGB frame if in debug mode
            if DEBUG_MODE and q_rgb is not None:
                in_rgb = q_rgb.tryGet()
                if in_rgb is not None:
                    frame = in_rgb.getCvFrame()

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
                        drone_data = process_drone_detections(dummy_frame, detections)
                    else:
                        drone_data = process_drone_detections(frame, detections)

            # If we have a valid frame and in debug mode, display it with current detections
            if DEBUG_MODE and frame is not None:
                display_frame("OAK-1 Object Detection", frame, detections)
            
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