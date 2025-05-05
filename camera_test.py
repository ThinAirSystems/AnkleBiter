import cv2
import time
import platform
import subprocess

def list_available_cameras_windows():
    """Attempt to list available camera devices on Windows"""
    try:
        # This only works if you have ffmpeg installed and in PATH
        result = subprocess.run(["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"], 
                               capture_output=True, text=True, check=False)
        print("\nAvailable camera devices (via ffmpeg):")
        print(result.stderr)  # ffmpeg outputs to stderr
    except Exception as e:
        print(f"Couldn't list cameras via ffmpeg: {e}")
        print("To get a list of cameras, you can install ffmpeg or check Device Manager")

def try_open_camera(index, api=None, description=""):
    """Try to open a camera with specific index and API"""
    try:
        if api is None:
            cap = cv2.VideoCapture(index)
        else:
            cap = cv2.VideoCapture(index, api)
        
        # Check if opened successfully
        if cap.isOpened():
            # Try to read a frame to confirm it's working
            ret, frame = cap.read()
            if ret:
                # Save the frame as an image file
                filename = f"camera_{index}{'_' + str(api) if api else ''}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ SUCCESS: {description} - Camera at index {index}{' with API ' + str(api) if api else ''} works!")
                print(f"   Frame captured and saved as {filename}")
                
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"   Resolution: {width}x{height}, FPS: {fps}")
                
                cap.release()
                return True
            else:
                print(f"❌ {description} - Camera at index {index}{' with API ' + str(api) if api else ''} opened but couldn't read frame")
                cap.release()
                return False
        else:
            print(f"❌ {description} - Couldn't open camera at index {index}{' with API ' + str(api) if api else ''}")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {str(e)}")
        return False

def main():
    print("OpenCV Camera Diagnostic Tool")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print("Testing various camera access methods...\n")
    
    # Try different camera indices
    success = False
    
    # Try default access method
    for i in range(3):  # Try first 3 indices
        if try_open_camera(i, description=f"Method 1: Default (index {i})"):
            success = True
    
    # Try DirectShow backend (Windows)
    if platform.system() == 'Windows':
        for i in range(2):
            if try_open_camera(i, cv2.CAP_DSHOW, f"Method 2: DirectShow (index {i})"):
                success = True
                
        # Try Media Foundation backend (Windows 10+)
        for i in range(2):
            if try_open_camera(i, cv2.CAP_MSMF, f"Method 3: Media Foundation (index {i})"):
                success = True
    
    # Try Video for Linux on Linux
    elif platform.system() == 'Linux':
        for i in range(2):
            if try_open_camera(i, cv2.CAP_V4L2, f"Method 4: V4L2 (index {i})"):
                success = True
    
    # Try AVFoundation on macOS
    elif platform.system() == 'Darwin':  # macOS
        for i in range(2):
            if try_open_camera(i, cv2.CAP_AVFOUNDATION, f"Method 5: AVFoundation (index {i})"):
                success = True
    
    # Print summary
    print("\n--- Summary ---")
    if success:
        print("At least one camera access method was successful! ✅")
        print("Check the saved image files to verify the camera is working correctly.")
    else:
        print("No camera could be accessed. 😢")
        print("Possible causes:")
        print("1. No camera is connected to your system")
        print("2. Camera is being used by another application")
        print("3. Camera drivers are not installed correctly")
        print("4. Camera hardware might be disabled or malfunctioning")
        
        # Try to list available cameras on Windows
        if platform.system() == 'Windows':
            list_available_cameras_windows()
            
            print("\nAdditional Windows troubleshooting steps:")
            print("1. Check Device Manager for any camera devices with warnings")
            print("2. Try the built-in Windows Camera app to test if the camera works at all")
            print("3. Update or reinstall your camera drivers")
            print("4. Check if camera access is enabled in Windows Privacy Settings")

if __name__ == "__main__":
    main()