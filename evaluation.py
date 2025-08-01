import cv2
import os
import numpy as np
from ai_services.face_recognition import FaceRecognition
import uuid
import sys
import time
sys.stdout = open('evaluation_log_facenet1.txt', 'w')
def process_video(video_path: str, output_dir: str = "faces_1/crops"):
    face_recog = FaceRecognition()
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(fps * 60 * 10)  # 10 minutes max

    print(f"[Info] FPS: {fps}, Max frames to process: {max_frames}")
    saved_ids = set()
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % 100 != 0:
            continue

        print(f"\n[Frame {frame_idx}] Processing...")

        # Extract and save face crops
        

        embedding = face_recog.extract_face_embedding(frame)
        if embedding is None:
            continue
        
        for face_embedding in embedding:

        # Compare with known embeddings
            matched_id = face_recog.find_matching_face_id(face_embedding)
            if matched_id:
                print(f"[Result] Existing ID matched: {matched_id}")
                faces = face_recog.extract_and_save_crop(frame, matched_id)
            else:
                new_id = str(uuid.uuid4())[:8]
                face_recog.save_face_embedding(new_id, face_embedding, frame)
                faces = face_recog.extract_and_save_crop(frame, new_id)
                print(f"[Result] New ID assigned: {new_id}")

    cap.release()
    print("[Done] Video processing completed.")

if __name__ == "__main__":
    start_time = time.time()
    elevator_video = "videos/elevator_cut.mp4"  # Replace with your actual path
    process_video(elevator_video)
    end_time = time.time()
    result_time = end_time-start_time
    print(f'Time needed {result_time:.2f}')
