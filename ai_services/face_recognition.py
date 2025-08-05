
import numpy as np
import uuid
from deepface import DeepFace
import sys
import cv2
import os
from scipy.spatial.distance import cosine
import time
class FaceRecognition:
    def __init__(self, threshold: float = 0.3):
        self.known_face_encodings: list[np.ndarray] = []
        self.known_face_ids: list[str] = []
        self.known_faces: dict[str, np.ndarray] = {}
        self.threshold = threshold
        self.metric = 'cosine'
        self.model_name = 'Facenet512'


    def extract_face_embedding(self, image: np.ndarray) :
        try:
            # Get face embeddings
                
            results = DeepFace.represent(
                    img_path=image,
                    model_name=self.model_name,
                    detector_backend='fastmtcnn', 
                    enforce_detection=True,
                    align= True
                )
            if results and isinstance(results, list):
                    embeddings = [np.array(res['embedding']) for res in results]
                    return embeddings
        except Exception as e:
            print(f"[FaceEmbedding] Error: {e}")
            return None

        
        else:
            print("No face embedding returned.")
            return None


    def extract_and_save_crop(self, image: np.ndarray, face_id) -> np.ndarray | None:
        try:
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend='fastmtcnn',
                enforce_detection=True,
                align=True
                )
        except ValueError as e:
            print(f"[FaceEmbedding] No face detected in frame. Skipping. ({e})")
            return None
        except Exception as e:
            print(f"[FaceEmbedding] Unexpected error: {e}")
            return None
        
        if not faces:
            print("[FaceEmbedding] No faces detected.")
            return None
        os.makedirs('faces/crops', exist_ok=True)

        for idx, face in enumerate(faces):
            face_crop = face["face"]
            face_crop_uint8 = (face_crop * 255).astype(np.uint8)
                
            timing = time.time()
            # Save the face crop for inspections
            save_path = os.path.join('faces/crops', f"{face_id}_face_{idx}_{timing}.jpg")
            # Convert RGB to BGR for cv2.imwrite
            face_crop_bgr = cv2.cvtColor(face_crop_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, face_crop_bgr)
            print(f"[FaceEmbedding] Saved face crop to {save_path}")
    
    def find_matching_face_id(self, embedding: np.ndarray) -> str | None:
        if not self.known_faces:
            print('no known faces')
            return None

        best_id = None
        best_distance = float("inf")

        for face_id, known_embedding in self.known_faces.items():
            distance = cosine(known_embedding, embedding)
            print(f"Compared to {face_id} -> distance = {distance:.4f}")
            if distance < self.threshold and distance < best_distance:
                best_id = face_id
                best_distance = distance

        if best_id:
            print(f"[FaceRec] Found matching ID: {best_id} (distance: {best_distance:.4f})")
        else:
            print("[FaceRec] No matching face found.")

        return best_id
    
    def save_face_embedding(self, face_id: str, embedding: np.ndarray, face_image):
        if face_id in self.known_faces:
            print(f"[FaceRec] ID {face_id} already has embedding. Skipping save.")
            return
        print(f"[FaceRec] Saving new face: {face_id}")
        self.known_faces[face_id] = embedding
