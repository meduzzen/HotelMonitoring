import time

from ai_services.multicamera import MultiCameraTracker
from config.camera import CameraConfig


def main():
    """Main function to run the tracking system."""
    reid_model_path = "models/model.pth.tar-30"

    camera_configs = [
        # CameraConfig(camera_id=1, video_path="videos/enter_corect.mp4",    min_box_area=5000),   # entrance — people far, small boxes
        # CameraConfig(camera_id=2, video_path="videos/arka_corect.mp4",  min_box_area=5000),   # arch — same
        # CameraConfig(camera_id=3, video_path="videos/elevator_2m.mp4", min_box_area=20000),  # elevator — people close, large boxes
        CameraConfig(
            camera_id=4, video_path="videos/1084464550-preview.mp4", min_box_area=1
        ),
        CameraConfig(
            camera_id=6, video_path="videos/1092645527-preview.mp4", min_box_area=1
        ),
    ]

    tracker = MultiCameraTracker(reid_model_path, camera_configs)
    start = time.time()
    tracker.run()
    elapsed = time.time() - start
    print(f"Total processing time: {elapsed:.2f}s")

    # Print summary from DB
    db = tracker.analytics_db
    print("\n--- Analytics summary ---")
    print(f"Unique persons seen : {len(db.get_all_persons())}")
    print(f"Total entries       : {db.get_entries_count()}")
    print(f"Total exits         : {db.get_exits_count()}")


if __name__ == "__main__":
    main()
