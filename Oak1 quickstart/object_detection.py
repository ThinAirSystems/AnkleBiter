import cv2
import depthai as dai
import numpy as np
import time # Added for potential delay check

print("Attempting to initialize DepthAI for Object Detection...")

# --- Configuration ---
NN_MODEL = "mobilenet-ssd" # Uses a pre-compiled model from DepthAI's cache
NN_INPUT_SIZE = 300
CONFIDENCE_THRESHOLD = 0.5

# Mobilenet SSD label mapping
label_map = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Helper to convert normalized NN coordinates to pixel coordinates
def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# Helper function to draw detections
def display_frame(name, frame, detections):
    color = (0, 255, 0) # Green
    if frame is None:
        print("Received None frame, cannot display.")
        return
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
                 label = f"Label {label_index}" # Fallback if label index is out of bounds
            cv2.putText(frame, label, (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            # Get confidence text
            confidence = f"{int(detection.confidence * 100)}%"
            cv2.putText(frame, confidence, (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
        except Exception as e:
            print(f"Error drawing detection: {e}, Detection data: {detection}")


    # Show the frame
    cv2.imshow(name, frame)

try:
    # --- Define the Pipeline ---
    pipeline = dai.Pipeline()

    # 1. Define sources
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(NN_INPUT_SIZE, NN_INPUT_SIZE) # Match NN input
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # 2. Define Neural Network node
    print(f"Loading Neural Network model: {NN_MODEL}")
    detection_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

    # Define the MODEL_PATH first
    from pathlib import Path
    MODEL_PATH = str(Path("C:/Users/armer/OneDrive/ThinAir_Main/ThinAirSystems_Root/5 - Active projects/CUAS/Oak1 quickstart/models/mobilenet-ssd_openvino_2021.2_6shave.blob"))
    detection_nn.setBlobPath(MODEL_PATH)  # Use setBlobPath instead
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.input.setBlocking(False)

    # 3. Define Output nodes
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("nn")

    # 4. Linking
    cam_rgb.preview.link(detection_nn.input)
    detection_nn.passthrough.link(xout_rgb.input) # Use passthrough for synchronized frame
    detection_nn.out.link(xout_nn.input)

    print("Pipeline defined. Attempting to connect to device...")

    # --- Connect and Run ---
    # Adding timeout and trying to list devices if connection fails
    found, device_info = dai.Device.getFirstAvailableDevice()
    if not found:
        print("No OAK device found. Make sure it's connected.")
        exit()

    with dai.Device(pipeline, device_info) as device:
        print(f"Connected to OAK device: {device.getDeviceInfo().name}")
        print("Downloading/Compiling model blob if necessary (this may take a moment)...")

        # Get output queues
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        frame = None
        detections = []

        print("Starting object detection loop...")
        while True:
            # Try to get packets from both queues
            in_rgb = q_rgb.tryGet()
            in_nn = q_nn.tryGet()

            if in_rgb is not None:
                frame = in_rgb.getCvFrame()

            if in_nn is not None:
                detections = in_nn.detections

            # If we have a valid frame, display it with current detections
            if frame is not None:
                 display_frame("OAK-1 Object Detection", frame, detections)
            else:
                # Optional small delay if no frame yet
                time.sleep(0.001)


            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == -1: # No key pressed
                pass
            else: # Any other key pressed
                print(f"Key {key} pressed")


except Exception as e:
    print(f"An error occurred during object detection: {e}")
    import traceback
    traceback.print_exc() # Print detailed traceback
    print("Make sure the OAK device is connected, libraries are installed, and the model is accessible.")

finally:
    # Close OpenCV windows
    cv2.destroyAllWindows()
    print("Detection stopped or failed.")