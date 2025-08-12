# -*- coding: utf-8 -*-
"""multimodal_align_mask_fixed.py – contrastive alignment with MASK‑token for NaN

 ▶ 仅改动 2 处逻辑（Encoder + 训练循环）；其余保持原样。
"""

from pathlib import Path
import random, logging, os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import models_vit           # RETFound ViT backbone
from loss import NTXentLoss  # 你的 NT‑Xent 实现

# -----------------------------------------------------------------------------
# 1. 表格数据增强 (SCARF‑style)
# -----------------------------------------------------------------------------
class TabularAugmentation:
    def __init__(self, mask_prob: float = 0.3, noise_std: float = 0.01):
        self.mask_prob = mask_prob; self.noise_std = noise_std
    def __call__(self, x: torch.Tensor):
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        m = torch.rand_like(x).lt(self.mask_prob)
        x = x.clone(); x[m] = 0.0
        return x + torch.randn_like(x) * self.noise_std

# -----------------------------------------------------------------------------
# 2. MASK‑token 表格嵌入
# -----------------------------------------------------------------------------
class TabularEmbedding(nn.Module):
    def __init__(self, feat_dim: int, hid_dim: int, proj_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hid_dim), nn.ReLU(True), nn.Linear(hid_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False)   # ← 新增
        )
        self.mask_token = nn.Parameter(torch.randn(proj_dim) * 0.02)
    def forward(self, x):                 # x:[B,feat_dim] 可能含 NaN
        isnan = torch.isnan(x)
        z = self.proj(torch.nan_to_num(x, nan=0.0))
        z = z + isnan.any(1, keepdim=True).float() * self.mask_token
        return z                          # [B,proj_dim]

# -----------------------------------------------------------------------------
# 3. RETFound 图像 encoder  + projector  (改动①)
# -----------------------------------------------------------------------------
class RETFoundEncoder(nn.Module):
    """
    ViT backbone  + 2‑layer SimCLR projector.
    返回 **未归一化** 的投影向量；是否做归一化交给上层处理。
    """
    def __init__(self, checkpoint_path: str = "", arch: str = "vit_large_patch16"):
        super().__init__()
        self.vit = getattr(models_vit, arch)(
            img_size=224, num_classes=0, drop_path_rate=0, global_pool=False
        )
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt)
            self.vit.load_state_dict(state, strict=False)

        # projector
        self.proj = nn.Sequential(
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024, affine=False)   # 等价于 L2‑Norm，但先不归一化
        )

    # timm.forward_features（删去无关分支）
    def _forward_tokens(self, x: torch.Tensor):
        B = x.size(0)
        x = self.vit.patch_embed(x)           # (B,196,1024)
        cls_tok = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tok, x), dim=1)    # (B,197,1024)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        return self.vit.norm(x)               # (B,197,1024)

    def forward(self, x, use_gpa: bool = True):
        tokens  = self._forward_tokens(x)
        feat    = tokens[:, 1:].mean(1) if use_gpa else tokens[:, 0]  # (B,1024)
        z_raw   = self.proj(feat)                                     # **未归一化**
        return z_raw

# -----------------------------------------------------------------------------
# 4. Dataset
# -----------------------------------------------------------------------------
class OCT2DContrastDataset(Dataset):
    def __init__(self, img_dir: Path, csv_path: Path, transform, tab_aug=None, mode="train"):
        self.transform = transform
        self.tab_aug  = tab_aug if mode == "train" else None
        df            = pd.read_csv(csv_path)
        self.bio_cols = [c for c in df.columns if c != "eid"]
        self.samples  = []
        for _, r in df.iterrows():
            imgs = list(img_dir.glob(f"{int(r['eid'])}_*.png"))
            if not imgs: continue
            bio = r[self.bio_cols].values.astype(np.float32)
            for p in imgs:
                self.samples.append((p, bio))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, bio = self.samples[idx]
        img = self.transform(Image.open(p).convert("RGB"))
        bio = torch.tensor(bio)
        if self.tab_aug: bio = self.tab_aug(bio)
        return img, bio

# -----------------------------------------------------------------------------
# 5. Dual Encoder
# -----------------------------------------------------------------------------
class DualEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_enc = RETFoundEncoder(cfg["ckpt"], cfg["arch"])
        self.tab_enc = TabularEmbedding(cfg["feat_dim"], cfg["tab_hid"], cfg["proj_dim"])
    def forward(self, img, tab):
        if tab.ndim == 1: tab = tab.unsqueeze(0)
        return self.img_enc(img), self.tab_enc(tab)

# -----------------------------------------------------------------------------
# 6. 训练循环 (改动②)
# -----------------------------------------------------------------------------
LOG_PATH = "/media/16T/3D_OCT_2d/tablur/train.log"

def setup_logger(path=LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logging.basicConfig(filename=path, filemode="w", level=logging.INFO,
                        format="%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler())

@torch.no_grad()
def cosine_eval(loader, model, device):
    model.eval(); sims = []
    for img, tab in loader:
        zi_raw, zt_raw = model(img.to(device), tab.to(device))
        sims.append(torch.cosine_similarity(F.normalize(zi_raw, dim=1),
                                            F.normalize(zt_raw, dim=1)).mean().item())
    return float(np.mean(sims))


def train_align(train_ld, val_ld, model, device, epochs=50):
    setup_logger(); logger = logging.getLogger(__name__)
    loss_fn = NTXentLoss(0.5, 0)
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best    = -1e9

    for ep in range(1, epochs + 1):
        model.train(); running = 0.0
        pbar = tqdm(train_ld, desc=f"Epoch {ep}/{epochs}", ncols=90, mininterval=0.5)
        for img, tab in pbar:
            img, tab = img.to(device), tab.to(device)
            zi_raw, zt_raw = model(img, tab)                 # 未归一化

            # 先统计方差
            var_img = zi_raw.float().std(dim=0).mean()
            var_tab = zt_raw.float().std(dim=0).mean()

            # 再归一化后算对比损失
            zi = F.normalize(zi_raw, dim=1)
            zt = F.normalize(zt_raw, dim=1)
            loss, _, _ = loss_fn(zi, zt)

            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", v_img=f"{var_img:.2e}", v_tab=f"{var_tab:.2e}")
            logger.info(f"ep {ep:02d} | step {pbar.n}/{len(train_ld)} | loss {loss.item():.4f} | v_img {var_img:.2e} | v_tab {var_tab:.2e}")

        # epoch 结束
        epoch_loss = running / len(train_ld)
        val_sim    = cosine_eval(val_ld, model, device)
        logger.info(f"Epoch {ep} done | train_loss {epoch_loss:.4f} | val_cos {val_sim:.4f}")
        if val_sim > best:
            best = val_sim
            torch.save({"model": model.img_enc.vit.state_dict()}, f"retfound_backbone_ep{ep:02d}_cos{val_sim:.4f}.pth")
            logger.info("✓ Saved improved backbone")

# -----------------------------------------------------------------------------
# 7. 主程序
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    root = Path("/media/16T/3D_OCT_2d/data/2D_OCT")
    csv_train, csv_test = root / "train.csv", root / "test.csv"
    print(f"Loading datasets from {csv_train} and {csv_test}")

    _MEAN = [0.485, 0.456, 0.406]; _STD = [0.229, 0.224, 0.225]
    train_tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.Lambda(lambda im: im.convert("RGB")), T.ToTensor(), T.Normalize(_MEAN, _STD)])
    val_tf   = T.Compose([T.Resize(224), T.CenterCrop(224), T.Lambda(lambda im: im.convert("RGB")), T.ToTensor(), T.Normalize(_MEAN, _STD)])

    ds_train = OCT2DContrastDataset(root / "train", csv_train, train_tf, None, "train")
    ds_val   = OCT2DContrastDataset(root / "test",  csv_test,  val_tf,   None,                        "val")

    cfg = dict(
        ckpt="/media/16T/3D_OCT_2d/Weight/RETFound_oct_weights.pth",
        arch="vit_large_patch16",
        img_sz=224,
        proj_dim=1024,
        tab_hid=128,
        feat_dim=len(ds_train.bio_cols),
    )

    dl_train = DataLoader(ds_train, batch_size=32, shuffle=True,  num_workers=4)
    dl_val   = DataLoader(ds_val,   batch_size=32, shuffle=False, num_workers=4)

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DualEncoder(cfg).to(device)
    train_align(dl_train, dl_val, model, device, epochs=50)
