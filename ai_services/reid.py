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

    def assign_global_id(self, embedding: np.ndarray, camera_name: str, frame) -> str:
        """Assign global ID based on embedding similarity."""
        best_gid = self._find_best_match(embedding)

        is_elevator = 'elevator' in camera_name.lower()
        if is_elevator:
            face_emb = self.face_recognition.extract_face_embedding(frame)
            if face_emb is None:
                print('note detected face')
            if face_emb is not None:
                face_id = self.face_recognition.find_matching_face_id(face_emb)
                if face_id:
                    print(f"[FaceRec] Existing face match found → Assigned ID: ")
                    return face_id
                else:
                    new_gid = self._create_new_identity(embedding)
                    self.face_recognition.save_face_embedding(new_gid, face_emb, frame)
                    print(f"[FaceRec] New face → Assigned new ID: {new_gid}")
                    return new_gid
            else:
                print("[FaceRec] No face detected in elevator frame.")
        if best_gid:
            self._update_embedding_buffer(best_gid, embedding)
            '''if 'elevator' in camera_name.lower():
                face_emb = self.face_recognition.extract_face_embedding(frame)
                print('note detected face')
                if face_emb is not None:
                    self.face_recognition.save_face_embedding(best_gid, face_emb)
                    print(f"[FaceRec] Face embedding saved for existing body ID: {best_gid}")'''
            # if camera is elevator
            # make face recognition
            # add face embedding(or something needed to recognize the persons face again) and assign to this face embedding the same face id as to the body
            return best_gid
        print(f"[ReID] No match found → Assigned new ID: {new_gid}")
        return self._create_new_identity(embedding)

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
