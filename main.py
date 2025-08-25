from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig
import time
def main():
    """Main function to run the tracking system."""
    # Configuration
    reid_model_path = 'models/model.pth.tar-40'

    camera_configs = [
    #    CameraConfig(camera_id=1, stream_url="", output_url=""),
        CameraConfig(camera_id=2, stream_url="rtsp://admin:tEsTgfhjkm1729@10.0.0.21:554/cam/realmonitor?channel=1&subtype=1", output_url="rtsp://mediamtx:8554/cam/elev"),
        # CameraConfig(camera_id=3, stream_url="rtsp://admin:tEsTgfhjkm1729@192.168.12.167:554/cam/realmonitor?channel=1&subtype=1", output_url="rtsp://mediamtx:8554/cam/enter"),
        # CameraConfig(camera_id=4, stream_url="rtsp://admin:tEsTgfhjkm1729@192.168.12.20:554/cam/realmonitor?channel=1&subtype=1", output_url="rtsp://mediamtx:8554/cam/reception"),
    ]

    start_time = time.time()
    # Initialize and run tracker
    tracker = MultiCameraTracker(reid_model_path, camera_configs)
    tracker.run()
    end_time = time.time()
    print(f'result time {end_time-start_time:.2f}')


if __name__ == "__main__":
    main()