import streamlit as st
import sys 
import importlib
import os 
import shutil

from utils.model import load_model
from tabs.webcam import webcam
from tabs.upload_video import upload_video
from tabs.help import display_help_content

# Add full absolute path
OUTPUT_DIR = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\frontend\output"

sys.path.append(
    r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search_and_Train")

print("PYTHON PATH:", sys.path)  # DEBUG: confirm path is added

genotypes_path = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search_and_Train\genotypes.py"

spec = importlib.util.spec_from_file_location("genotypes", genotypes_path)
genotypes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(genotypes)
sys.modules["genotypes"] = genotypes
sys.modules["models.genotypes"] = genotypes
sys.modules["Search_and_Train.genotypes"] = genotypes
NASGenotype = genotypes.Genotype

# ABSOLUTE PATH TO augment_cnn.py
augment_path = r"C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search_and_Train\models\augment_cnn.py"

spec2 = importlib.util.spec_from_file_location("augment_cnn", augment_path)
augment_cnn = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(augment_cnn)
sys.modules["models.augment_cnn"] = augment_cnn  # <-- make it available as expected
AugmentCNN = augment_cnn.AugmentCNN


st.set_page_config(
    page_title="Health Check with Human Face Scanner", 
    layout="centered")

st.title("💓 Health Check with Human Face Scanner")

webcam_tab, upload_tab, help_tab = st.tabs(["📷 Check Webcam (Live)", "📹 Upload Video", "🆘 Help"])

model = load_model("model/g_5x5_model_best.pth.tar")
device = next(model.parameters()).device
model.eval()

# ====== TAB 1: Webcam Live HR ======
with webcam_tab:
    webcam(model, device)

# ====== TAB 2: Upload Video HR ======
with upload_tab:
    upload_video(model, device)

# ====== TAB 3: Help ======
with help_tab:
    display_help_content()