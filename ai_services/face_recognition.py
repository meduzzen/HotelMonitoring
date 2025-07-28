
import numpy as np
import uuid
from deepface import DeepFace
import sys
import cv2
import os
from scipy.spatial.distance import cosine
class FaceRecognition:
    def __init__(self, threshold: float = 0.3):
        self.known_face_encodings: list[np.ndarray] = []
        self.known_face_ids: list[str] = []
        self.known_faces: dict[str, np.ndarray] = {}
        self.threshold = threshold
        self.metric = 'cosine'
        self.model_name = 'Facenet512'


    def extract_face_embedding(self, image: np.ndarray) -> np.ndarray | None:
        try:
            # Ensure image is RGB
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend='mediapipe',
            enforce_detection=False,
            align=True
            )
            if not faces:
                print("[FaceEmbedding] No faces detected.")
                return None
            
            for face in faces:
                face_crop = face["face"]
            # Get face embeddings
                results = DeepFace.represent(
                    img_path=face_crop,
                    model_name=self.model_name,
                    detector_backend='mediapipe',  # or 'retinaface', 'mtcnn'
                    enforce_detection=True
                )
                if results and isinstance(results, list):
                    embedding = results[0]['embedding']
                    return np.array(embedding)
        except Exception as e:
            print(f"[FaceEmbedding] Error: {e}")
            return None

        
        else:
            print("No face embedding returned.")
            return None
        
    '''def extract_face_crop(self, image: np.ndarray) -> np.ndarray | None:
        try:
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend='opencv',
                enforce_detection=False,
                align=True
            )

            if not faces:
                return None
            return faces[0]['face']
        except Exception as e:
            print(f"[FaceCrop] Error: {e}")
            return None'''


    
    def find_matching_face_id(self, embedding: np.ndarray) -> str | None:
        if not self.known_faces:
            print('no known faces')
            return None

        best_id = None
        best_distance = float("inf")

        for face_id, known_embedding in self.known_faces.items():
            distance = cosine(known_embedding, embedding)
            if distance < self.threshold and distance < best_distance:
                best_id = face_id
                best_distance = distance

        if best_id:
            print(f"[FaceRec] Found matching ID: {best_id} (distance: {best_distance:.4f})")
        else:
            print("[FaceRec] No matching face found.")

        return best_id
    
    def save_face_embedding(self, face_id: str, embedding: np.ndarray, face_image):
        print(f"[FaceRec] Saving new face: {face_id}")
        self.known_faces[face_id] = embedding

        if face_image is not None:
            os.makedirs("./faces", exist_ok=True)
            face_path = os.path.join("faces", f"{face_id}.jpg")
            if face_image.shape[2] == 3:
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(face_path, face_image)