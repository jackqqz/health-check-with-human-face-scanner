import cv2
import numpy as np

AGE_PROTO = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\model\age_deploy.prototxt"
AGE_MODEL = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\model\age_net.caffemodel"
GENDER_PROTO = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\model\gender_deploy.prototxt"
GENDER_MODEL = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\model\gender_net.caffemodel"

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

def detect_age_gender(face_roi):
    print("Face ROI shape:", face_roi.shape if face_roi is not None else "None")

    if face_roi is None or not isinstance(face_roi, np.ndarray):
        return "Unknown", 0.0, "Unknown", 0.0

    try:
        face_resized = cv2.resize(face_roi, (227, 227))
    except Exception as e:
        print("Resize failed:", e)
        return "Unknown", 0.0, "Unknown", 0.0

    blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227),
                                 (78.426, 87.768, 114.896), swapRB=False)

    # Gender prediction
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender_idx = gender_preds[0].argmax()
    gender = GENDER_LIST[gender_idx]
    gender_conf = float(gender_preds[0][gender_idx]) * 100

    # Age prediction
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx = age_preds[0].argmax()
    age = AGE_LIST[age_idx]
    age_conf = float(age_preds[0][age_idx]) * 100

    return age, age_conf, gender, gender_conf


#######################################
# Model 2: FairFace
#######################################

# import os
# import cv2
# import torch
# import torchvision.transforms as transforms
# import torch.nn.functional as F
# from torchvision.models import resnet34
# from facenet_pytorch import MTCNN
# from PIL import Image
# import torch.nn as nn

# # ========== Model Path ==========
# FAIRFACE_MODEL_PATH = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\model\res34_fair_align_multi_7_20190809.pt"

# # ========== Set device ==========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ========== Age/Gender labels ==========
# age_list = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
# gender_list = ['Male', 'Female']

# # ========== Face detector ==========
# mtcnn = MTCNN(keep_all=False, device=device)

# # ========== Preprocessing ==========
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # ========== Define FairFace headless model ==========
# class FairFaceBase(nn.Module):
#     def __init__(self):
#         super(FairFaceBase, self).__init__()
#         self.resnet = resnet34(pretrained=False)
#         in_features = self.resnet.fc.in_features
#         self.resnet.fc = nn.Identity()  # remove final FC
#         self.gender_head = nn.Linear(in_features, 2)
#         self.age_head = nn.Linear(in_features, 9)

#     def forward(self, x):
#         features = self.resnet(x)
#         gender_out = self.gender_head(features)
#         age_out = self.age_head(features)
#         return gender_out, age_out

# # ========== Load pretrained model ==========
# model = FairFaceBase().to(device)
# state_dict = torch.load(FAIRFACE_MODEL_PATH, map_location=device)
# model.resnet.load_state_dict(state_dict, strict=False)
# model.eval()

# # ========== Detection Function ==========
# def detect_age_gender(image_bgr):
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     image_pil = Image.fromarray(image_rgb)

#     # Detect face
#     face = mtcnn(image_pil)
#     if face is None:
#         return "Unknown", "Unknown"

#     face_tensor = transform(image_pil).unsqueeze(0).to(device)

#     with torch.no_grad():
#         gender_out, age_out = model(face_tensor)
#         gender = gender_list[torch.argmax(F.softmax(gender_out, dim=1)).item()]
#         age_group = age_list[torch.argmax(F.softmax(age_out, dim=1)).item()]

#     return age_group, gender
