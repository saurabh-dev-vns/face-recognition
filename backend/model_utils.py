import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple
from keras_facenet import FaceNet

embedder = FaceNet()

# Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
FACE_DETECTOR = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# Mediapipe FaceMesh for liveness
mp_face_mesh = mp.solutions.face_mesh
FACE_MESH = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def detect_primary_face(image_bgr: np.ndarray) -> Tuple[Tuple[int,int,int,int], np.ndarray]:
    """
    Detect first face in image.
    Returns (x, y, w, h), face_rgb or (None, None) if no face found.
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = FACE_DETECTOR.process(img_rgb)
    if not results.detections:
        return None, None

    det = results.detections[0]
    h, w, _ = image_bgr.shape
    bbox = det.location_data.relative_bounding_box
    x = max(0, int(bbox.xmin * w))
    y = max(0, int(bbox.ymin * h))
    bw = min(w - x, int(bbox.width * w))
    bh = min(h - y, int(bbox.height * h))

    face_rgb = img_rgb[y:y+bh, x:x+bw]
    return (x, y, bw, bh), face_rgb


def get_embedding(face_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a normalized FaceNet embedding for a given RGB face image.
    Returns a 512-dim numpy array or None if extraction fails.
    """
    if face_rgb is None or face_rgb.size == 0:
        return None

    results = embedder.extract(face_rgb)
    if not results:
        return None

    emb = results[0]['embedding']
    # Normalize
    emb = emb / np.linalg.norm(emb)
    return emb


# -----------------------
# 🟢 NEW: Liveness Detection
# -----------------------

# Indices of left & right eye landmarks (Mediapipe FaceMesh)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_idx):
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)

def is_live_face(image_bgr: np.ndarray) -> bool:
    """
    Returns True if blink detected (indicating liveness).
    """
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(img_rgb)
    if not results.multi_face_landmarks:
        return False

    lm = results.multi_face_landmarks[0].landmark
    left_ear = eye_aspect_ratio(lm, LEFT_EYE_IDX)
    right_ear = eye_aspect_ratio(lm, RIGHT_EYE_IDX)
    avg_ear = (left_ear + right_ear) / 2.0

    # EAR threshold (lower -> eyes closed)
    return avg_ear < 0.20
