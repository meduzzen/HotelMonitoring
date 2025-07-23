from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig

def main():
    """Main function to run the tracking system."""
    # Configuration
    reid_model_path = 'models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'

    camera_configs = [
        CameraConfig(name="cam1", video_path="videos/cam1.mp4"),
        CameraConfig(name="cam2", video_path="videos/cam6.mp4"),
    ]


    # Initialize and run tracker
    tracker = MultiCameraTracker(reid_model_path, camera_configs)
    tracker.run()


if __name__ == "__main__":
    main()