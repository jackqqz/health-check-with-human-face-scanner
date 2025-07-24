# face_detect.py

import os
import cv2
import mediapipe as mp
import joblib

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

def detect_face_and_save(frame, save_dir, frame_index):
    """
    Detects face in the frame using MediaPipe and saves bounding box as .pkl file.
    save_dir: folder where to store frame_index.pkl
    frame_index: int
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    if results.detections:
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape

        x1 = int(bboxC.xmin * iw)
        y1 = int(bboxC.ymin * ih)
        x2 = x1 + int(bboxC.width * iw)
        y2 = y1 + int(bboxC.height * ih)
        score = detection.score[0]

        # Format: [x1, y1, x2, y2, score]
        bbox = [x1, y1, x2, y2, float(score)]

        # Save as .pkl
        pkl_path = os.path.join(save_dir, f"{frame_index}.pkl")
        joblib.dump(bbox, pkl_path)
        return bbox
    return None
