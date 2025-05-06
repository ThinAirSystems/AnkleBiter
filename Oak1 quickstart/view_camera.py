import cv2
import depthai as dai
import time # Added for potential delay check if needed

print("Attempting to initialize DepthAI...")

try:
    # 1. Create a DepthAI pipeline
    pipeline = dai.Pipeline()

    # 2. Define a source node - Color Camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480) # Set preview size
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR) # OpenCV expects BGR

    # 3. Define an output node - XLinkOut
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb_preview")

    # 4. Link the nodes
    cam_rgb.preview.link(xout_rgb.input)

    print("Pipeline created. Attempting to connect to device...")

    # 5. Connect to the OAK device and start the pipeline
    # Adding timeout and trying to list devices if connection fails
    found, device_info = dai.Device.getFirstAvailableDevice()
    if not found:
        print("No OAK device found. Make sure it's connected.")
        exit()

    with dai.Device(pipeline, device_info) as device:
        print(f"Connected to OAK device: {device.getDeviceInfo().name}")
        # Get the output queue
        q_rgb = device.getOutputQueue(name="rgb_preview", maxSize=4, blocking=False)

        print("Streaming RGB preview...")
        while True:
            # Get the RGB frame
            in_rgb = q_rgb.tryGet() # Non-blocking call

            if in_rgb is not None:
                # Retrieve the frame data and display it
                frame = in_rgb.getCvFrame()
                cv2.imshow("OAK-1 RGB Preview", frame)
            else:
                # Optional: add a small delay if no frame received, prevents high CPU usage in loop
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
    print(f"An error occurred: {e}")
    print("Make sure the OAK device is connected and drivers/udev rules are set up.")

finally:
    # Close OpenCV windows
    cv2.destroyAllWindows()
    print("Stream stopped or failed.")