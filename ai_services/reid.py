import uuid
import torch
import numpy as np
from typing import Deque
from collections import deque

from scipy.spatial.distance import cosine
from torchvision import transforms
import torchreid
from torchvision.transforms import InterpolationMode

from config.tracking import TrackingConfig
import time

from schema.embedding import EmbeddingEntry

tracking_config = TrackingConfig()

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class ReIDModel:
    """Improved ReID model with consistent ID assignment and threshold management."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()
        self.transform = self._create_transform()
        self.embedding_db: dict[str, Deque[EmbeddingEntry]] = {}
        self.threshold = tracking_config.reid_threshold
        self.buffer_size = tracking_config.embedding_buffer_size

    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the ReID model."""
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
        """Create image transformation pipeline."""
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

    def assign_global_id(self, embedding: np.ndarray, camera_id: int, current_id: str) -> str:
        best_gid = self._find_best_match(embedding, camera_id, current_id)

        if best_gid:
            self._update_embedding_buffer(best_gid, embedding, camera_id)
            return best_gid
        else:
            return self._create_new_identity(embedding, camera_id)

    def _find_best_match(self, embedding: np.ndarray, camera_id: int, current_id : str) -> str | None:
        # if camera_id == 2:
        #     threshold = self.threshold * 1.1
        # else:
        #     threshold = self.threshold
        threshold = self.threshold
        best_gid = None
        this_id_gid = None
        best_distance = float('inf')
        now = time.time()
        match = False

        for gid, embedding_buffer in self.embedding_db.items():

            avg_embedding = np.mean([e.embedding for e in embedding_buffer], axis=0)

            distance = cosine(avg_embedding, embedding)

            last_entry = embedding_buffer[-1]
            if last_entry.camera_id != camera_id: #якшо айді знаходиться на камері умовно 2 то
                time_diff = now - last_entry.timestamp #він може бути присутній на камері 1 якщо не пройшов якийсь час (зараз 1 секунда)
                if time_diff < 1.0:
                    continue
            else:
                threshold = self.threshold*0.9 #для тієї ж камери трешхолд опускаємо

            if gid == current_id: #окремо зберігаємо айді який зараз обробляється, якщо ніхто не перевершить відстань то
                this_id_gid = gid #він і далі назначається треку, хоча відстань може і не проходити трешхолд
                if distance < best_distance:
                    best_distance = distance
                match = True
                #print(f"Camera {camera_id}, last entry: {last_entry.camera_id}, gid: {gid}, current_id: {current_id}, distance: {distance}")

            if distance < threshold:
                best_gid = gid
                best_distance = distance
            # print(f"Camera {camera_id}, last entry: {last_entry.camera_id}, gid: {gid}, current_id: {current_id}, distance: {distance}")
        if best_gid is None and match:
            best_gid = this_id_gid

        print(f"Camera id: {camera_id}, best distance: {best_distance}, best gid: {best_gid}, current id: {current_id}")
        return best_gid

    def _update_embedding_buffer(self, gid: str, embedding: np.ndarray, camera_id: int) -> None:
        """Update the embedding buffer for a given global ID."""
        entry = EmbeddingEntry(
            embedding=embedding,
            timestamp=time.time(),
            camera_id=camera_id
        )
        if len(self.embedding_db[gid]) >= self.buffer_size:
            self.embedding_db[gid].popleft()
        self.embedding_db[gid].append(entry)

    def _create_new_identity(self, embedding: np.ndarray, camera_id: int) -> str:
        """Create a new global identity."""
        new_gid = str(uuid.uuid4())[:8]
        entry = EmbeddingEntry(
            embedding=embedding,
            timestamp=time.time(),
            camera_id=camera_id
        )
        self.embedding_db[new_gid] = deque([entry], maxlen=self.buffer_size)
        return new_gid
