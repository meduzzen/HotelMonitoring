from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig
import time

def main():
    """Main function to run the tracking system."""
    # Configuration
    reid_model_path = 'models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth'

    camera_configs = [
        CameraConfig(name="elevator", stream_url="rtsp://admin:tEsTgfhjkm1729@10.0.0.21:554/cam/realmonitor?channel=1&subtype=1"),
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