# backend/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import base64
import cv2
import numpy as np
from .model_utils import detect_primary_face, get_embedding, is_live_face
from .database import SessionLocal, init_db, get_all_embeddings
from numpy.linalg import norm

init_db()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

def load_known():
    s = SessionLocal()
    return get_all_embeddings(s)  # list of (user_id, name, emb)

@app.post('/recognize')
async def recognize(payload: dict):
    KNOWN = load_known()

    frames = payload.get('frames')
    if not frames:
        return {'status': 'error', 'message': 'no frames provided'}

    blink_detected = False
    emb = None

    for dataurl in frames:
        try:
            header, encoded = dataurl.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            continue

        box, face = detect_primary_face(img)
        if face is None:
            continue

        if is_live_face(img):
            blink_detected = True

        if emb is None:
            emb = get_embedding(face)

    if emb is None:
        return {'status': 'error', 'message': 'No face detected'}

    if not blink_detected:
        return {'status': 'error', 'message': 'Liveness check failed (please blink)'}

    # face recognition
    best = None
    best_score = -1
    for user_id, name, known_emb in KNOWN:
        sim = np.dot(emb, known_emb) / (norm(emb) * norm(known_emb))
        if sim > best_score:
            best_score = sim
            best = (user_id, name)

    threshold = 0.55
    if best_score >= threshold:
        from .database import Attendance, SessionLocal
        s = SessionLocal()
        att = Attendance(user_id=best[0])
        s.add(att)
        s.commit()
        return {
            'status': 'ok',
            'message': f'Attendance marked for {best[1]}',
            'score': float(best_score)
        }
    else:
        return {
            'status': 'error',
            'message': 'Face not recognized',
            'score': float(best_score)
        }
