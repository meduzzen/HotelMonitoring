from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig
import time

def main():
    """Main function to run the tracking system."""
    # Configuration
    reid_model_path = 'models/model.pth.tar-30'

    camera_configs = [
        CameraConfig(camera_id=1, stream_url="rtsp://admin:tEsTgfhjkm1729@10.0.0.21:554/cam/realmonitor?channel=1&subtype=1"),
    ]

    # Initialize and run tracker
    tracker = MultiCameraTracker(reid_model_path, camera_configs)
    start = time.time()
    tracker.run()
    end = time.time()
    elapsed = end-start
    print(f'needed {elapsed:.2f}')
if __name__ == "__main__":
    main()