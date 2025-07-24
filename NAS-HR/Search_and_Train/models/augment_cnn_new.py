""" CNN for network augmentation with heteroscedastic uncertainty heads """
import torch
import torch.nn as nn
from models.augment_cells import AugmentCell
from models import ops

class AugmentCNN(nn.Module):
    """ Augmented CNN model with two output heads: mean (µ) and log-variance (log σ²) """
    def __init__(self, input_size, C_in, C, n_classes, n_layers, auxiliary, genotype,
                 stem_multiplier=3):
        super().__init__()
        # ─── Save config ────────────────────────────────────────────────
        self.C_in      = C_in
        self.C         = C
        self.n_classes = n_classes
        self.n_layers  = n_layers
        self.genotype  = genotype

        # ─── Stem network ───────────────────────────────────────────────
        C_cur = 32
        self.stem = nn.Sequential(
            nn.BatchNorm2d(C_in),
            nn.Conv2d(C_in, C_cur, 5, 2, 2, bias=False),
            nn.BatchNorm2d(C_cur),
            nn.ReLU(),
            nn.Conv2d(C_cur, C_cur, 3, 2, 1, bias=False),
            nn.BatchNorm2d(C_cur),
            nn.ReLU(),
        )

        # ─── Build DARTS cells ──────────────────────────────────────────
        C_pp, C_p = C_cur, C_cur
        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # reduction every ~1/6th of the network
            if i in [n_layers//6, 3*n_layers//6, 5*n_layers//6]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)

            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

        # ─── Global pooling ─────────────────────────────────────────────
        self.gap = nn.Sequential(
            nn.Conv2d(C_p, 512, 3, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(1),
        )

        # ─── **Heteroscedastic heads** ─────────────────────────────────
        # Instead of a single linear, we now have:
        # 1) head_mu     → predicts the mean µ(x)
        # 2) head_logvar → predicts log σ²(x)
        self.head_mu     = nn.Linear(512, n_classes)
        self.head_logvar = nn.Linear(512, 1)

    def _init_weight(self):
        """Initialize conv and linear weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, x):
        # 1) stem
        s0 = s1 = self.stem(x)
        # 2) cells
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        # 3) global pooling & flatten
        out = self.gap(s1)                  # → [B, 512, 1, 1]
        out = out.view(out.size(0), -1)     # → [B, 512]
        # 4) two heads
        mu     = self.head_mu(out)         # → [B, 1]
        logvar = self.head_logvar(out)     # → [B, 1]
        # Return 1D tensors of length B
        return mu.view(-1), logvar.view(-1)

    def drop_path_prob(self, p):
        """Set drop path probability on all DropPath_ modules."""
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
