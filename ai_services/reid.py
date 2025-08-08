import uuid
import torch
import numpy as np
from typing import Deque
from collections import deque

from scipy.spatial.distance import cosine
from torchvision import transforms
import torchreid
from torchvision.transforms import InterpolationMode
from ai_services.face_recognition import FaceRecognition


from config.tracking import TrackingConfig

tracking_config=TrackingConfig()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ReIDModel:

    def __init__(self, model_path: str, face_recognition: FaceRecognition):
        self.model_path = model_path
        self.model = self._load_model()
        self.transform = self._create_transform()
        self.embedding_db: dict[str, Deque[np.ndarray]] = {}
        self.threshold = tracking_config.reid_threshold
        self.buffer_size = tracking_config.embedding_buffer_size
        self.face_recognition = face_recognition
        self.failed_face_recognition_attempts = set()

    def _load_model(self) -> torch.nn.Module:
        model = torchreid.models.build_model(
            name='osnet_x1_0',
            num_classes=1000,
            pretrained=True
        )
        torchreid.utils.load_pretrained_weights(model, self.model_path)
        model.eval()
        model.to(device)
        return model

    def _create_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            input_tensor = self.transform(crop).unsqueeze(0).to(device)
            feature = self.model(input_tensor)
            normalized_feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            return normalized_feature.cpu().numpy().flatten()

    def assign_global_id(self, embedding: np.ndarray, camera_name: str, frame, frame_number) -> str:
        """Assign global ID based on embedding similarity."""
        if not hasattr(self, 'failed_face_recognition_attempts'):
            self.failed_face_recognition_attempts = set()

        best_gid = self._find_best_match(embedding)
        print(f"Processing frame_number={frame_number} for best_gid={best_gid}")

        try:
            face_emb = self.face_recognition.extract_face_embedding(frame)
        except Exception as e:
            face_emb = None
            print('No face detected')
        face_crops = None
        if best_gid:
            self._update_embedding_buffer(best_gid, embedding)
            key = (best_gid, frame_number)
            if key in self.failed_face_recognition_attempts:
                print(f"Skipping face recognition for ID {best_gid} at frame {frame_number} due to previous failure.")
            else:
                try:
                    face_crops = self.face_recognition.extract_and_save_crop(frame, best_gid)
                    if face_emb is not None:
                        self.face_recognition.save_face_embedding(best_gid, face_emb, frame)
                        print(f"Face saved for existing ReID match: {best_gid}")
                    else:
                        print(f"No face detected for existing ReID match: {best_gid}")
                        self.failed_face_recognition_attempts.add(key)
                except Exception as e:
                    print(f"Face extraction failed for {best_gid} at frame {frame_number}: {e}")
                    self.failed_face_recognition_attempts.add(key)
            return best_gid, face_crops

        if face_emb is not None:
            matching_face_id = self.face_recognition.find_matching_face_id(face_emb)
            if matching_face_id:
                print(f"[FaceRec] Face match found → Assigning existing face ID: {matching_face_id}")
                if matching_face_id in self.embedding_db:
                    self._update_embedding_buffer(matching_face_id, embedding)
                key = (matching_face_id, frame_number)
                if key in self.failed_face_recognition_attempts:
                    print(f"Skipping face recognition for ID {matching_face_id} at frame {frame_number} due to previous failure.")
                else:
                    try:
                        self.face_recognition.save_face_embedding(matching_face_id, face_emb, frame)
                        face_crops = self.face_recognition.extract_and_save_crop(frame, face_id=matching_face_id)
                    except Exception as e:
                        print(f"Failed to save face for ID {matching_face_id} at frame {frame_number}: {e}")
                        self.failed_face_recognition_attempts.add(key)
                return matching_face_id, face_crops
            else:
                new_gid = self._create_new_identity(embedding)
                self.embedding_db[new_gid] = deque([embedding], maxlen=self.buffer_size)
                key = (new_gid, frame_number)
                try:
                    self.face_recognition.save_face_embedding(new_gid, face_emb, frame)
                    face_crops = self.face_recognition.extract_and_save_crop(frame, face_id=new_gid)
                except Exception as e:
                    print(f"Failed to save face for new ID {new_gid} at frame {frame_number}: {e}")
                    self.failed_face_recognition_attempts.add(key)
                print(f"[FaceRec] No face match → Created and assigned new ID: {new_gid}")
                return new_gid, face_crops

        # ReID failed and no face fallback
        new_gid = self._create_new_identity(embedding)
        self.embedding_db[new_gid] = deque([embedding], maxlen=self.buffer_size)
        print(f"[ReID+FaceRec] No match → Assigned new ID: {new_gid}")
        return new_gid, face_crops




    def _find_best_match(self, embedding: np.ndarray) -> str | None:
        """Find the best matching global ID for the given embedding."""
        best_gid = None
        best_distance = float('inf')

        for gid, embedding_buffer in self.embedding_db.items():
            avg_embedding = np.mean(embedding_buffer, axis=0)
            distance = cosine(avg_embedding, embedding)

            if distance < self.threshold and distance < best_distance:
                best_gid = gid
                best_distance = distance

        return best_gid

    def _update_embedding_buffer(self, gid: str, embedding: np.ndarray) -> None:
        """Update the embedding buffer for a given global ID."""
        if len(self.embedding_db[gid]) >= self.buffer_size:
            self.embedding_db[gid].popleft()
        self.embedding_db[gid].append(embedding)

    def _create_new_identity(self, embedding: np.ndarray) -> str:
        """Create a new global identity."""
        new_gid = str(uuid.uuid4())[:8]
        self.embedding_db[new_gid] = deque([embedding], maxlen=self.buffer_size)
        return new_gid
