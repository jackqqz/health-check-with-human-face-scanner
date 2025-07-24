# model_performance.py

import os
import json
import torch
import numpy as np
from torchvision import transforms
import MyDataset
from genotypes import Genotype
from models.augment_cnn import AugmentCNN

# ─── Paths ──────────────────────────────────────────────────────────────
fileRoot      = r"C:/Users/User/Documents/Monash/FYP/PURE_Full"
saveRoot      = r"C:/Users/User/Documents/Monash/FYP/pure_gb_4x4_50"
model_dir     = r"C:/Users/User/Documents/Monash/FYP/health-check-with-human-face-scanner/NAS-HR/Search_and_Train/augments/pure_gb_4x4"
model_path    = os.path.join(model_dir, "model_best.pth.tar")
stats_path    = os.path.join(model_dir, "error_stats.json")

# Log file (optional)
# log_file_path = os.path.join(model_dir, "test_results.txt")

# ─── Hyperparameters ────────────────────────────────────────────────────
batch_size  = 32
num_workers = 0

# ─── Preprocessing (must match training) ───────────────────────────────
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225])
toTensor  = transforms.ToTensor()
resize    = transforms.Resize((64, 300))
tf        = transforms.Compose([resize, toTensor, normalize])

# ─── Prepare test DataLoader ───────────────────────────────────────────
test_ds = MyDataset.Data_STMap(
    root_dir = saveRoot + "_Test",
    frames_num = 300,
    transform = tf
)
test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# ─── Load the heteroscedastic model ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reconstruct your AugmentCNN exactly as you did at train time
input_size     = [64, 300]
input_channels = 3
init_channels  = 36
n_classes      = 1
layers         = 6
use_aux        = False  # or True if you trained with auxiliary
genotype =  Genotype(normal=[[('dil_conv_5x5', 1), ('max_pool_3x3', 0)], [('max_pool_3x3', 0), ('dil_conv_5x5', 1)], [('max_pool_3x3', 0), ('skip_connect', 2)]], normal_concat=range(2, 5), reduce=[[('max_pool_3x3', 1), ('sep_conv_5x5', 0)], [('max_pool_3x3', 2), ('sep_conv_5x5', 0)], [('max_pool_3x3', 3), ('max_pool_3x3', 2)]], reduce_concat=range(2, 5))


model = AugmentCNN(input_size, input_channels,
                   init_channels, n_classes,
                   layers, use_aux, genotype)
model = model.to(device)

# Load checkpoint and strip "module." if needed
ckpt = torch.load(model_path, map_location=device)
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        new_k = k.replace("module.", "")
        new_sd[new_k] = v
    model.load_state_dict(new_sd)
else:
    model = ckpt  # if you saved the whole model

model.eval()

# ─── Inference & collect μ predictions ────────────────────────────────
HR_pred = []
HR_true = []

with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        # ** two-headed output **
        mu, logvar = model(X)  
        # we only care about the mean μ for error stats
        HR_pred.extend(mu.cpu().numpy())
        HR_true.extend(y.view(-1).cpu().numpy())

HR_pred = np.array(HR_pred)
HR_true = np.array(HR_true)

# ─── Compute residuals & statistics ────────────────────────────────────
residuals  = np.abs(HR_pred - HR_true)
mean_error = float(np.mean(residuals))
std_error  = float(np.std(residuals, ddof=1))
p95_error  = float(np.percentile(residuals, 95))

# Option: override to a “nice” round number for your UI
# p95_error = 8.0

stats = {
    "mean_error": mean_error,
    "std_error":  std_error,
    "p95_error":  p95_error
}

# ─── Save JSON for Streamlit app to consume ────────────────────────────
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

# ─── Print summary ─────────────────────────────────────────────────────
print("Evaluation complete. Error stats:")
print(json.dumps(stats, indent=2))
