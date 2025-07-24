import os
import joblib
import numpy as np
import cv2
from tools.io_tools import mkdir_if_missing

# === Directories ===
CROP_DIR = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\output\crop"  # TODO: Set your actual output root
DATA_DIR = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\output\data"

# === POS Algorithm ===
def pos(RGB, fps):
    WinSec = 1.6
    N = RGB.shape[0]
    H = np.zeros(N)
    l_win = int(np.ceil(WinSec * fps))
    for n in range(l_win - 1, N):
        m = n - l_win + 1
        if m >= 0:
            Cn = RGB[m:n+1, :] / np.mean(RGB[m:n+1, :], axis=0)
            # No need transpose anymore
            h = Cn[:, 0]  # Only one channel
            h -= np.mean(h)
            H[m:n+1] += h
    return H

# === Dataset Generation ===
def gen_dataset(which_skin_threshold, segment_id):

    segment_crop_dir = os.path.join(CROP_DIR, segment_id)
    output_video_dir = os.path.join(DATA_DIR, segment_id)
    mkdir_if_missing(output_video_dir)

    print(f'Processing segment folder: {segment_crop_dir}')
    print(f'Output will be saved to: {output_video_dir}')

    frame_files = sorted([
        f for f in os.listdir(segment_crop_dir)
        if f.endswith(f"_{which_skin_threshold}.pkl")
    ])
    if not frame_files:
        print(f"No ROI data found for segment folder {segment_id}")
        return

    # === Load per-frame ROI RGB data ===
    rgb_data = []
    for frame_file in frame_files:
        frame_path = os.path.join(segment_crop_dir, frame_file)
        data = joblib.load(frame_path)
        rgb_data.append(data["5x5"])  # shape per frame: (n_ROIs, 3)
    rgb_data = np.array(rgb_data)  # (num_frames, n_ROIs, 3)
    rgb_data = np.transpose(rgb_data, (1, 0, 2))  # (n_ROIs, num_frames, 3)

    # === Apply POS on each ROI ===
    pos_signals = []
    for roi_signal in rgb_data:
        roi_signal = roi_signal[:, 1:2]  # Keep green channel only
        pos_sig = pos(roi_signal, fps=30.0)
        pos_signals.append(pos_sig)
    pos_stmap = np.stack(pos_signals, axis=0)  # (n_ROIs, num_frames)

    # === Normalize & Convert to color map ===
    stmap_img = cv2.normalize(pos_stmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stmap_img = cv2.applyColorMap(stmap_img, cv2.COLORMAP_JET)

    # === Save STMap ===
    joblib.dump(pos_stmap, os.path.join(output_video_dir, "POS_STMap.pkl"))
    cv2.imwrite(os.path.join(output_video_dir, "POS_STMap.png"), stmap_img)
    print(f"Saved POS STMap to {output_video_dir}")

# Now your gen_data can be:

def gen_data(segment_id):
    gen_dataset(which_skin_threshold=0, segment_id=segment_id)
