import os
import joblib
import gc
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from tools.io_tools import mkdir_if_missing
from preprocess.gen_STmap import image_to_STmap_68


def get_landmarks_list(num_frames, load_video_dir):
    landmarks_list = []
    for i in range(num_frames):
        path = os.path.join(load_video_dir, f"{i}.pkl")
        if os.path.isfile(path):
            landmarks = joblib.load(open(path, 'rb'))
        else:
            landmarks = None
        landmarks_list.append(landmarks)
    return landmarks_list


def process_frame(index, frame, landmarks, output_dir, skin_threshold):
    if landmarks is None:
        print(f"[SKIP] No landmark for frame {index}")
        return 0

    landmarks = landmarks.astype('int32')
    flag_segment = skin_threshold > 0

    bgr_map = image_to_STmap_68(
        image=frame,
        landmarks=landmarks,
        roi_x=5,
        roi_y=5,
        flag_plot=False,
        flag_segment=flag_segment,
        skin_threshold=skin_threshold,
    )

    data = {"5x5": bgr_map[:, ::-1]}  # RGB to BGR
    output_path = os.path.join(output_dir, f"{index}_{skin_threshold}.pkl")
    joblib.dump(data, output_path)

    return 1


def process_segment(segment_dir, output_base_dir, landmark_dir, skin_threshold=0, num_workers=4):
    video_path = None
    for f in os.listdir(segment_dir):
        if f.endswith(".mp4"):
            video_path = os.path.join(segment_dir, f)
            break
    if not video_path:
        raise FileNotFoundError(f"No video found in {segment_dir}")

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    mkdir_if_missing(output_base_dir)
    num_frames = len(frames)
    landmarks_list = get_landmarks_list(num_frames, landmark_dir)

    print(f"[INFO] Cropping video: {video_path}")
    print(f"[INFO] Using landmarks from: {landmark_dir}")
    print(f"[INFO] Saving 5x5 crops to: {output_base_dir}")

    # === Parallel Execution ===
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(
            partial(process_frame, output_dir=output_base_dir, skin_threshold=skin_threshold),
            range(num_frames),
            frames,
            landmarks_list
        )

    del frames
    gc.collect()
    print(f"[DONE] 5x5 crops saved to: {output_base_dir}")
