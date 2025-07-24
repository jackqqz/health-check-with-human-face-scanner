# augment.py
""" Training augmented model with heteroscedastic Gaussian NLL loss """
import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import AugmentConfig
import MyDataset
from torchvision import transforms
import utils
from models.augment_cnn import AugmentCNN
from torch.utils.data import DataLoader
import scipy.io as io

# ─── Parse command-line config ───────────────────────────────────────
config = AugmentConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── TensorBoard & logger ────────────────────────────────────────────
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)
logger = utils.get_logger(os.path.join(config.path, f"{config.name}.log"))
config.print_params(logger.info)

# ─── Data transforms ─────────────────────────────────────────────────
resize    = transforms.Resize((64, 300))
toTensor  = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
transform = transforms.Compose([resize, toTensor, normalize])

# ─── Paths for raw & preprocessed STMaps ─────────────────────────────
fileRoot = r"C:/Users/User/Desktop/Backup Lenovo/D Drive/Testing FYP/PURE_G_5x5"
saveRoot = (
    r"C:/Users/User/Desktop/Backup Lenovo/D Drive/Testing FYP/"
    + "G_5x5_PURE_STMap"
    + str(config.fold_num)
    + str(config.fold_index)
)

# ─── If requested, regenerate train/val/test splits ────────────────
if config.reData == 1:
    train_idx, val_idx, test_idx = MyDataset.SplitDataset(
        fileRoot, train_ratio=0.7, val_ratio=0.2
    )
    PIC_NAME = 'STMap_YUV_Align_CSI_POS.png'
    STEP     = 15
    FRAMES   = 300

    MyDataset.getIndex(fileRoot, train_idx,
                       saveRoot + '_Train', PIC_NAME, STEP, FRAMES)
    MyDataset.getIndex(fileRoot, val_idx,
                       saveRoot + '_Val', PIC_NAME, STEP, FRAMES)
    MyDataset.getIndex(fileRoot, test_idx,
                       saveRoot + '_Test', PIC_NAME, STEP, FRAMES)

# ─── Sanity-check contents ────────────────────────────────────────────
for suffix in ['_Train','_Val','_Test']:
    folder = saveRoot + suffix
    logger.info(f"Contents of {folder}: {os.listdir(folder)}")

# ─── Build Datasets & DataLoaders ───────────────────────────────────
train_ds = MyDataset.Data_STMap(
    root_dir=saveRoot + '_Train',
    frames_num=300,
    transform=transform
)
val_ds = MyDataset.Data_STMap(
    root_dir=saveRoot + '_Val',
    frames_num=300,
    transform=transform
)

train_loader = DataLoader(
    train_ds, batch_size=config.batch_size, shuffle=True,
    num_workers=config.workers, pin_memory=True
)
valid_loader = DataLoader(
    val_ds, batch_size=config.batch_size, shuffle=False,
    num_workers=config.workers, pin_memory=True
)

# ─── Build model with two-headed (μ, log σ²) output ─────────────────
use_aux = config.aux_weight > 0.0
model = AugmentCNN(
    input_size=np.array([64,300]),
    C_in=3,
    C=config.init_channels,
    n_classes=1,
    n_layers=config.layers,
    auxiliary=use_aux,
    genotype=config.genotype
)
model = nn.DataParallel(model, device_ids=config.gpus).to(device)
model._init_weight()

# ─── Optimizer ───────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

# ─── Heteroscedastic Gaussian NLL loss (two-headed) ────────────────
def gaussian_nll(mu, logvar, target):
    var = torch.exp(logvar).clamp(min=1e-6)
    # 0.5 * [ (y−μ)²/σ² + logσ² ]
    nll = 0.5 * ((target - mu)**2 / var + logvar)
    return nll.mean()

# ─── Training iteration ──────────────────────────────────────────────
def train(loader, model, optimizer, epoch):
    model.train()
    meter = utils.AverageMeter()
    step0 = epoch * len(loader)
    lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch+1} start, LR={lr:.2e}")
    writer.add_scalar('train/lr', lr, step0)

    for i, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        # ** Two-headed forward **  
        mu, logvar = model(X)

        # ** Apply NLL loss **  
        loss = gaussian_nll(mu, logvar, y.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        meter.update(loss.item(), X.size(0))
        if i % config.print_freq == 0 or i == len(loader)-1:
            logger.info(
                f"Train [{epoch+1}/{config.epochs}] Step [{i}/{len(loader)-1}] "
                f"Loss {meter.avg:.4f}"
            )
        writer.add_scalar('train/loss', loss.item(), step0 + i)

# ─── Validation iteration ────────────────────────────────────────────
def validate(loader, model, epoch, step, best):
    model.eval()
    meter = utils.AverageMeter()
    HR_pr, HR_gt = [], []

    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            mu, logvar = model(X)
            loss = gaussian_nll(mu, logvar, y.view(-1))

            meter.update(loss.item(), X.size(0))
            HR_pr.extend(mu.cpu().numpy())
            HR_gt.extend(y.view(-1).cpu().numpy())

            if i % config.print_freq == 0 or i == len(loader)-1:
                logger.info(
                    f"Valid [{epoch+1}/{config.epochs}] Step [{i}/{len(loader)-1}] "
                    f"Loss {meter.avg:.4f}"
                )

    # compute traditional metrics for logging
    me, std, mae, rmse, mer, p = utils.MyEval(HR_pr, HR_gt)
    logger.info(
        f"Epoch {epoch+1} Eval → me={me:.4f}, std={std:.4f}, "
        f"mae={mae:.4f}, rmse={rmse:.4f}"
    )
    writer.add_scalar('val/loss', meter.avg, step)

    is_best = meter.avg < best
    utils.save_checkpoint(model, config.path, is_best)
    return meter.avg if is_best else best

# ─── Main training loop ─────────────────────────────────────────────
if __name__ == "__main__":
    best_loss = float('inf')
    for epoch in range(config.epochs):
        train(train_loader, model, optimizer, epoch)
        step = (epoch+1) * len(train_loader)
        best_loss = validate(valid_loader, model, epoch, step, best_loss)

    logger.info(f"Training completed. Best validation loss: {best_loss:.4f}")
