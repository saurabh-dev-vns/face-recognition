import cv2
import numpy as np
from model_utils import detect_primary_face, get_embedding
from database import SessionLocal, init_db, save_embedding

init_db()

def capture_from_webcam(n=5):
    cap = cv2.VideoCapture(0)
    collected = []
    print('Press SPACE to capture, ESC to quit')
    while len(collected) < n:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Enroll - press SPACE to capture', frame)
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC
            break
        elif k % 256 == 32:  # SPACE
            collected.append(frame.copy())
            print(f'Captured {len(collected)}/{n}')
    cap.release()
    cv2.destroyAllWindows()
    return collected

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()

    imgs = capture_from_webcam(args.count)
    embeddings = []
    for img in imgs:
        _, face = detect_primary_face(img)
        if face is None:
            print('[WARN] No face detected, skipping...')
            continue
        emb = get_embedding(face)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        print('[ERROR] No embeddings generated. Try again with better lighting.')
        exit(1)

    avg = np.mean(np.vstack(embeddings), axis=0)
    session = SessionLocal()
    save_embedding(session, args.user_id, args.name, avg)
    session.close()  # ✅ GOOD PRACTICE
    print(f'[SUCCESS] Enrollment complete for {args.name}')
