from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig

def main():
    """Main function to run the tracking system."""
    # Configuration

    camera_configs = [
        CameraConfig(camera_id=0, video_path="videos/cam1.mp4"),
        CameraConfig(camera_id=1, video_path="videos/arch_7am.mp4"),
    ]


    # Initialize and run tracker
    tracker = MultiCameraTracker(camera_configs)
    tracker.run()


if __name__ == "__main__":
    main()