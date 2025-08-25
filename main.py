from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig
import time


def main():
    """Main function to run the tracking system."""
    # Configuration
    reid_model_path = "models/model.pth.tar-30"

    camera_configs = [
        # CameraConfig(camera_id=1, video_path="videos/enter_corect.mp4"), #1 камера це вхід в готель
        # CameraConfig(camera_id=2, video_path="videos/arka_corect.mp4"),
        CameraConfig(
            camera_id=3, video_path="videos/elevator_2m.mp4"
        ),  # 3 камера це ліфт
    ]

    # Initialize and run tracker
    tracker = MultiCameraTracker(reid_model_path, camera_configs)
    start = time.time()
    tracker.run()
    end = time.time()
    elapsed = end - start
    print(f"needed {elapsed:.2f}")


if __name__ == "__main__":
    main()
