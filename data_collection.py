from ultralytics import YOLO
import torch

from config.camera import CameraConfig
from ai_services.collection import CropSaver


def main():
    # Example: video input and folder output
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    output_folder = "person_crops"

    config = CameraConfig(
        name="videos",
        video_path="videos/elevator_new_multiple.mp4",
    )

    detector = YOLO("models/yolov8s.pt").to(device)
    crop_saver = CropSaver(config, output_folder)
    crop_saver.process_and_save_crops(detector)


if __name__ == "__main__":
    main()
