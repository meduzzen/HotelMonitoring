from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig
import time
def main():
    """Main function to run the tracking system."""
    # Configuration
    reid_model_path = 'models/model.pth.tar-40'

    camera_configs = [
       CameraConfig(camera_id=1, stream_url="", output_url=""),
        #CameraConfig(camera_id=2, video_path="videos/elevator_2m.mp4"),
        #CameraConfig(camera_id=3, video_path="videos/enter_corect.mp4"),
        #CameraConfig(camera_id=4, video_path="videos/reception_1m.mp4"),
    ]

    start_time = time.time()
    # Initialize and run tracker
    tracker = MultiCameraTracker(reid_model_path, camera_configs)
    tracker.run()
    end_time = time.time()
    print(f'result time {end_time-start_time:.2f}')


if __name__ == "__main__":
    main()