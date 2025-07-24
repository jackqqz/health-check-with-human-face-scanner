# import cv2
# import numpy as np
# import torch
# import mediapipe as mp

# mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)

# def extract_roi(frame):
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = mp_face_detection.process(rgb)

#     rois = []
#     face_coords = []

#     if results.detections:
#         h, w, _ = frame.shape
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             x_min = int(bboxC.xmin * w)
#             y_min = int(bboxC.ymin * h)
#             x_max = int((bboxC.xmin + bboxC.width) * w)
#             y_max = int((bboxC.ymin + bboxC.height) * h)

#             # Ensure bounding box is within frame boundaries
#             x_min = max(0, x_min)
#             y_min = max(0, y_min)
#             x_max = min(w, x_max)
#             y_max = min(h, y_max)

#             # Extract ROI
#             roi = frame[y_min:y_max, x_min:x_max]
#             if roi.shape[0] > 0 and roi.shape[1] > 0:
#                 rois.append(roi)
#                 face_coords.append((x_min, y_min, x_max - x_min, y_max - y_min))  # (x, y, width, height)

#     return rois, face_coords

# def extract_roi_age_gender(frame):
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = mp_face_detection.process(rgb)

#     rois = []
#     face_coords = []

#     if results.detections:
#         h, w, _ = frame.shape
#         for detection in results.detections:
#             bboxC = detection.location_data.relative_bounding_box
#             x_min = int(bboxC.xmin * w)
#             y_min = int(bboxC.ymin * h)
#             x_max = int((bboxC.xmin + bboxC.width) * w)
#             y_max = int((bboxC.ymin + bboxC.height) * h)

#             # Expand box by 30% margin
#             margin = 0.3
#             dx = int((x_max - x_min) * margin)
#             dy = int((y_max - y_min) * margin)
#             x_min = max(0, x_min - dx)
#             y_min = max(0, y_min - dy)
#             x_max = min(w, x_max + dx)
#             y_max = min(h, y_max + dy)

#             # Create square crop around center
#             box_w = x_max - x_min
#             box_h = y_max - y_min
#             box_size = max(box_w, box_h)
#             center_x = x_min + box_w // 2
#             center_y = y_min + box_h // 2

#             # Compute new square box
#             new_x_min = max(0, center_x - box_size // 2)
#             new_y_min = max(0, center_y - box_size // 2)
#             new_x_max = min(w, center_x + box_size // 2)
#             new_y_max = min(h, center_y + box_size // 2)

#             roi = frame[new_y_min:new_y_max, new_x_min:new_x_max]
#             if roi.shape[0] > 0 and roi.shape[1] > 0:
#                 rois.append(roi)
#                 face_coords.append((new_x_min, new_y_min, new_x_max - new_x_min, new_y_max - new_y_min))

#     return rois, face_coords

# def preprocess_frame(roi):
#     roi_resized = cv2.resize(roi, (64, 64))  # match input size of model
#     roi_norm = roi_resized / 255.0
#     roi_tensor = torch.tensor(roi_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
#     return roi_tensor

import cv2
import numpy as np
import torch
import mediapipe as mp
import joblib
import os 

# Increase confidence threshold
mp_face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.3)

# Load a pre-trained human face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def verify_human_face(roi):
    """Secondary verification to ensure the detected face is human"""
    if roi.shape[0] < 20 or roi.shape[1] < 20:
        return False
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def extract_roi_1(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(rgb)

    rois = []
    face_coords = []

    if not results.detections:
        print("No faces detected by MediaPipe.")

    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            x_max = int((bboxC.xmin + bboxC.width) * w)
            y_max = int((bboxC.ymin + bboxC.height) * h)

            # Ensure bounding box is within frame boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Extract ROI
            roi = frame[y_min:y_max, x_min:x_max]
            
            # Secondary verification to confirm this is a human face
            if roi.shape[0] > 0 and roi.shape[1] > 0 and verify_human_face(roi):
                rois.append(roi)
                face_coords.append((x_min, y_min, x_max - x_min, y_max - y_min))  # (x, y, width, height)

    return rois, face_coords

def extract_roi(frame, detect_save_dir=None, frame_index=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(rgb)

    rois = []
    face_coords = []

    if results.detections:
        h, w, _ = frame.shape
        detection = results.detections[0]  # Use only the most confident detection
        bboxC = detection.location_data.relative_bounding_box
        x_min = int(bboxC.xmin * w)
        y_min = int(bboxC.ymin * h)
        x_max = int((bboxC.xmin + bboxC.width) * w)
        y_max = int((bboxC.ymin + bboxC.height) * h)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            rois.append(roi)
            face_coords.append((x_min, y_min, x_max - x_min, y_max - y_min))

            # === Save .pkl detection if enabled ===
            if detect_save_dir and frame_index is not None:
                os.makedirs(detect_save_dir, exist_ok=True)
                score = detection.score[0]
                bbox_data = [x_min, y_min, x_max, y_max, float(score)]
                save_path = os.path.join(detect_save_dir, f"{frame_index}.pkl")
                joblib.dump(bbox_data, save_path)

    return rois, face_coords

def extract_roi_age_gender_1(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_detection.process(rgb)

    rois = []
    face_coords = []

    if results.detections:
        h, w, _ = frame.shape
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x_min = int(bboxC.xmin * w)
            y_min = int(bboxC.ymin * h)
            x_max = int((bboxC.xmin + bboxC.width) * w)
            y_max = int((bboxC.ymin + bboxC.height) * h)

            # Create ROI for verification
            verify_roi = frame[y_min:y_max, x_min:x_max]
            
            # Only proceed if this is confirmed to be a human face
            if verify_roi.shape[0] > 0 and verify_roi.shape[1] > 0 and verify_human_face(verify_roi):
                # Expand box by 30% margin for age/gender detection
                margin = 0.3
                dx = int((x_max - x_min) * margin)
                dy = int((y_max - y_min) * margin)
                x_min = max(0, x_min - dx)
                y_min = max(0, y_min - dy)
                x_max = min(w, x_max + dx)
                y_max = min(h, y_max + dy)

                # Create square crop around center
                box_w = x_max - x_min
                box_h = y_max - y_min
                box_size = max(box_w, box_h)
                center_x = x_min + box_w // 2
                center_y = y_min + box_h // 2

                # Compute new square box
                new_x_min = max(0, center_x - box_size // 2)
                new_y_min = max(0, center_y - box_size // 2)
                new_x_max = min(w, center_x + box_size // 2)
                new_y_max = min(h, center_y + box_size // 2)

                roi = frame[new_y_min:new_y_max, new_x_min:new_x_max]
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    rois.append(roi)
                    face_coords.append((new_x_min, new_y_min, new_x_max - new_x_min, new_y_max - new_y_min))

    return rois, face_coords

def extract_roi_age_gender(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Adjust minSize and scaleFactor for better robustness
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),  # Ignore tiny detections
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return [], []  # No faces found

    # Take the largest face by area (most likely the subject)
    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

    # Add margin to ensure context
    margin = 20
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)

    roi = frame[y1:y2, x1:x2]
    return [roi], [(x1, y1, x2 - x1, y2 - y1)]