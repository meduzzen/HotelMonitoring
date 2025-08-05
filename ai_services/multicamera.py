import os
from typing import List

import cv2

from ai_services.process_camera import CameraProcessor, SharedMultiCameraTracker
from ai_services.face_recognition import FaceRecognition
from config.camera import CameraConfig
import threading
import torch

device = torch.device("cpu")

face_recog = FaceRecognition()
def get_fairmot_opt(load_model_path, input_w, input_h, fps):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    reid_dim = 128
    opt = Struct(
        task='mot',
        dataset='jde',
        arch='dla_34',
        load_model=load_model_path,
        reid_dim=reid_dim,
        heads={'hm': 1, 'wh': 4, 'id': reid_dim, 'reg': 2},
        head_conv=256,
        input_w=input_w,
        input_h=input_h,
        output_w=input_w // 4,
        output_h=input_h // 4,
        down_ratio=4,
        mean=[0.408, 0.447, 0.470],
        std=[0.289, 0.274, 0.278],
        nID=14455,
        conf_thres=0.4,
        det_thres=0.3,
        nms_thres=0.4,
        track_buffer=30,
        min_box_area=100,
        fix_res=True,
        keep_res=False,
        not_reg_offset=False,
        reg_offset=True,
        pad=31,
        num_stacks=1,
        root_dir=os.getcwd(),
        vis_thresh=0.5,
        gpus=[0],
        gpus_str='0',
        K=500,
        ltrb=True,
        num_classes=1,
    )
    opt.frame_rate = fps
    return opt
opt = get_fairmot_opt(
    load_model_path='FairMOT/models/fairmot_dla34.pth',
    input_w=1088,
    input_h=608,
    fps=30
)
class MultiCameraTracker:
    """Main class for multi-camera person tracking system with threading."""

    def __init__(self, camera_configs: List[CameraConfig]):
        tracker=SharedMultiCameraTracker(opt, face_recog)
        self.cameras = {
            config.camera_id: CameraProcessor(config, tracker=tracker)
            for config in camera_configs
        }
        self.threads = []

    def run(self) -> None:
        """Run multi-camera tracking in parallel threads."""
        try:
            for cam_name, camera in self.cameras.items():
                t = threading.Thread(target=self._process_camera_loop, args=(cam_name, camera))
                t.start()
                self.threads.append(t)

            # Wait for all threads to finish
            for t in self.threads:
                t.join()

        finally:
            self._cleanup()

    def _process_camera_loop(self, cam_name, camera):
        """Process one camera in a loop inside a thread."""
        while True:
            frame = camera.process_frame()
            if frame is None:
                break
            annotated_frame = camera.process_tracks(frame)
            camera.write_frame(annotated_frame)
            # Optionally show frame:
            #cv2.imshow(str(cam_name), annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        camera.cleanup()

    def _cleanup(self):
        for camera in self.cameras.values():
            camera.cleanup()
        cv2.destroyAllWindows()
