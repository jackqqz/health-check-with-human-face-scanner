import os
import cv2
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader, Dataset

from attachment.pytorch_face_landmark.common.utils import BBox
from attachment.pytorch_face_landmark.models.mobilefacenet import MobileFaceNet
from tools.io_tools import mkdir_if_missing

# Normalization constants
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

# === Config ===
which_model = 'MobileFaceNet'


# === Model Loader ===
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileFaceNet([112, 112], 136).to(device)
    model_path = os.path.join("attachment", "pytorch_face_landmark", "checkpoint", "mobilefacenet_model_best.pth.tar")
    state = torch.load(model_path, map_location=device)

    # Handle "module." prefixes
    if "state_dict" in state:
        state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    else:
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state)
    model.eval()
    return model


# === Face & BBox Loader ===
def get_faces_and_bboxes(height, width, num_frames, load_video_dir):
    new_bboxes = []
    faces = []
    borders = []

    for f_idx in range(num_frames):
        frame_path = os.path.join(load_video_dir, f"{f_idx}.pkl")
        if not os.path.isfile(frame_path):
            new_bboxes.append(None)
            faces.append([0, 0, 0, 0, 0])
            borders.append([0, 0, 0, 0])
        else:
            face = joblib.load(frame_path)
            x1, y1, x2, y2 = face[:4]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min(w, h) * 1.2)
            cx = x1 + w // 2
            cy = y1 + h // 2
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            new_bbox = BBox([x1, x2, y1, y2])
            faces.append(face)
            borders.append([dx, dy, edx, edy])
            new_bboxes.append(new_bbox)

    return faces, borders, new_bboxes


# === Dataset Wrapper ===
class LandmarkDataset(Dataset):
    def __init__(self, video_, bboxes_, borders_):
        self.video = video_
        self.bboxes = bboxes_
        self.borders = borders_
        self.num_frames = len(video_)
        self.out_size = 112
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return self.num_frames

    def __getitem__(self, frame_idx):
        img = self.video[frame_idx]
        bbox = self.bboxes[frame_idx]
        if bbox is None:
            return torch.zeros((3, self.out_size, self.out_size), dtype=torch.float32, device=self.device)

        dx, dy, edx, edy = self.borders[frame_idx]
        cropped = img[bbox.top:bbox.bottom, bbox.left:bbox.right]

        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        face = cv2.resize(cropped, (self.out_size, self.out_size)) / 255.0
        face = (face - mean) / std
        face = face.transpose(2, 0, 1)

        tensor = torch.tensor(face).float().to(self.device)
        return tensor


# === Main Landmark Processing Function ===
def landmark_from_segment(segment_dir, output_dir):
    """
    Args:
        segment_dir: folder containing .mp4 and detect/*.pkl
        output_dir: where to save landmark .pkl and vis images
    """
    # Load video
    video_path = None
    for f in os.listdir(segment_dir):
        if f.endswith(".mp4"):
            video_path = os.path.join(segment_dir, f)
            break
    if not video_path:
        raise FileNotFoundError("No video found.")

    detect_dir = os.path.join(segment_dir, "detect")
    if not os.path.isdir(detect_dir):
        raise FileNotFoundError(f"Missing detection folder: {detect_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Load video frames
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    model = load_model()
    h, w, _ = frames[0].shape
    num_frames = len(frames)

    faces, borders, new_bboxes = get_faces_and_bboxes(h, w, num_frames, detect_dir)
    dataset = LandmarkDataset(frames, new_bboxes, borders)
    loader = DataLoader(dataset, batch_size=512, shuffle=False)

    start_idx = 0
    for batch in loader:
        preds = model(batch)[0].detach().cpu().numpy().reshape(batch.shape[0], -1, 2)
        for i in range(batch.shape[0]):
            bbox = new_bboxes[start_idx]
            if bbox is None:
                start_idx += 1
                continue

            landmark = preds[i]
            landmark = bbox.reprojectLandmark(landmark)

            # Save landmark
            joblib.dump(landmark, os.path.join(output_dir, f"{start_idx}.pkl"))

            # Save visual
            vis_img = frames[start_idx].copy()
            for (x, y) in landmark:
                cv2.circle(vis_img, (int(x), int(y)), 2, (0, 0, 255), -1)
            vis_path = os.path.join(output_dir, f"vis_{start_idx}.jpg")
            cv2.imwrite(vis_path, vis_img)

            print(f"[Saved] Frame {start_idx}")
            start_idx += 1
