"""
============================================================
Blast Cell Pipeline v3 - single-file, all-inclusive module
============================================================
Contains:
  1.  Dataset utilities (YOLO format + flat folders)
  2.  Novel Blast-YOLO model (ConvNeXt + BiFPN + CBAM + 3 heads)
  3.  Losses (CIoU + Focal + Dice + Tversky + GardnerOrdinal)
  4.  Benchmark classifier factory (AlexNet, VGG-16/19, ResNet-50,
       DenseNet-121, EfficientNetV2-S, ConvNeXt-Tiny, ViT-B/16)
  5.  Benchmark detection factory (Faster R-CNN, Mask R-CNN,
       SSD, RetinaNet) + segmentation (U-Net)
  6.  Training loops (novel + benchmarks, smoke-test-safe)
  7.  Evaluation (mAP, sens, spec, F1, AUC, ROC, CM, vectors)
  8.  Cross-model comparison (bar, radar, table export)

Import once with:
    import sys; sys.path.insert(0, '/content/blast_cell_pipeline_v3')
    import pipeline as P
"""
from __future__ import annotations
import os, math, json, time, shutil, zipfile, pathlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

# Lazy / optional imports (handled inside the functions that need them)


# ============================================================
# 1. DATASET UTILITIES
# ============================================================
CLASS_NAMES = ['2AB','2BB','2BC','3AA','3AB','3AC','3BB','3BC','3CC',
               '4AA','4AB','4AC','4BB','4BC','4CC',
               '5AA','5AB','5AC','5BB','5BC','6BB',
               'LP','P1','REV']

CLINICAL_GROUPS = {
    'top_priority':  ['P1', '3AA', '4AA', '5AA'],
    'good_quality':  ['3AB','4AB','5AB','3BB','4BB','5BB','6BB'],
    'fair_quality':  ['2AB','2BB','3AC','4AC','5AC','3BC','4BC','5BC'],
    'low_potential': ['LP','2BC','3CC','4CC'],
    'needs_review':  ['REV'],
}

ACTION_LABELS = {
    'P1':  'High Priority - top transfer candidate',
    'LP':  'Low Potential - limited developmental potential',
    'REV': 'Review - MD must consider before transfer',
}


def compute_class_weights(label_dir: str | Path) -> np.ndarray:
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for f in Path(label_dir).glob('*.txt'):
        for line in f.read_text().splitlines():
            if line.strip():
                c = int(line.split()[0])
                if 0 <= c < len(CLASS_NAMES): counts[c] += 1
    counts = np.where(counts == 0, 1, counts)
    w = counts.sum() / (len(CLASS_NAMES) * counts)
    return w / w.mean()


def _resolve_split_dir(root, split):
    """Find images dir tolerantly: val/valid/validation, with or without
    nested images/ subfolder, single-level nesting (Blast YOLO11/train/...)."""
    root = Path(root)
    aliases = {'val': ['val','valid','validation'],
               'train': ['train','training'],
               'test':  ['test','testing']}.get(split, [split])
    # Try root/, root/<inner>/  for one level of unwanted nesting
    candidates = [root] + [d for d in root.iterdir() if d.is_dir()]
    for base in candidates:
        for a in aliases:
            for img_path in [base/a/'images', base/a]:
                if img_path.exists() and any(img_path.glob('*.jp*g')):
                    lbl_path = img_path.parent/'labels'                         if img_path.name == 'images' else base/a/'labels'
                    if not lbl_path.exists(): lbl_path = base/a
                    return img_path, lbl_path
    raise FileNotFoundError(
        f"Could not find {split} split under {root}. "
        f"Tried {aliases} with and without 'images/' nesting. "
        f"Top-level contents: {sorted([p.name for p in root.iterdir()])}")


class BlastYOLODataset(Dataset):
    """Roboflow YOLO11 export -> image + (boxes, labels). Layout-tolerant."""
    def __init__(self, root, split='train', imgsz=640):
        import cv2
        self.cv2 = cv2
        self.root = Path(root)
        self.img_dir, self.lbl_dir = _resolve_split_dir(self.root, split)
        self.imgs = sorted(list(self.img_dir.glob('*.jpg')) +
                           list(self.img_dir.glob('*.jpeg')) +
                           list(self.img_dir.glob('*.png')))
        self.imgsz = imgsz
        print(f"  [{split}] images: {self.img_dir}  ({len(self.imgs)} files)")

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        p = self.imgs[i]
        img = self.cv2.cvtColor(self.cv2.imread(str(p)), self.cv2.COLOR_BGR2RGB)
        img = self.cv2.resize(img, (self.imgsz, self.imgsz))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        boxes, labels = [], []
        txt = self.lbl_dir / (p.stem + '.txt')
        if txt.exists():
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    c, cx, cy, w, h = map(float, parts[:5])
                    boxes.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
                    labels.append(int(c))
        return img, {
            'boxes':  torch.tensor(boxes  or [[0,0,1,1]], dtype=torch.float32),
            'labels': torch.tensor(labels or [0],         dtype=torch.long),
            'name':   p.name,
        }


def yolo_to_flat(yolo_root, out_root):
    """Materialise YOLO labels -> ImageFolder for classification benchmarks."""
    import cv2
    src = Path(yolo_root); dst = Path(out_root); dst.mkdir(parents=True, exist_ok=True)
    for split in ('train','valid','test'):
        img_dir = src/split/'images'; lbl_dir = src/split/'labels'
        if not img_dir.exists(): continue
        out_split = 'val' if split == 'valid' else split
        for img in img_dir.glob('*.*'):
            txt = lbl_dir / (img.stem + '.txt')
            if not txt.exists(): continue
            classes = {int(l.split()[0]) for l in txt.read_text().splitlines()
                       if l.strip()}
            if not classes: continue
            cls = CLASS_NAMES[next(iter(classes))]
            tgt = dst/out_split/cls; tgt.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(img), str(tgt/img.name))
    print(f"Flat folder ready at {dst}")


# ============================================================
# 2. ATTENTION MODULES
# ============================================================
class ECA(nn.Module):
    def __init__(self, ch, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(ch) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, k, padding=k//2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1,-2)
        y = self.conv(y).transpose(-1,-2).unsqueeze(-1)
        return x * torch.sigmoid(y)


class CBAM(nn.Module):
    def __init__(self, ch, reduction=16, k=7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch//reduction, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(ch//reduction, ch, 1, bias=False))
        self.spatial = nn.Conv2d(2, 1, k, padding=k//2, bias=False)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x,1))
                          + self.mlp(F.adaptive_max_pool2d(x,1)))
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat(
            [x.mean(1,keepdim=True), x.max(1,keepdim=True)[0]], dim=1)))
        return x * sa


# ============================================================
# 3. NOVEL BLAST-YOLO MODEL
# ============================================================
class BiFPN(nn.Module):
    def __init__(self, ch, levels=3, eps=1e-4):
        super().__init__()
        self.eps = eps
        self.w1 = nn.Parameter(torch.ones(2, levels-1))
        self.w2 = nn.Parameter(torch.ones(3, levels-1))
        self.act = nn.SiLU()
        self.td = nn.ModuleList([nn.Conv2d(ch,ch,3,padding=1,bias=False)
                                 for _ in range(levels-1)])
        self.bu = nn.ModuleList([nn.Conv2d(ch,ch,3,padding=1,bias=False)
                                 for _ in range(levels-1)])

    def forward(self, feats):
        td = [feats[-1]]
        for i in range(len(feats)-2, -1, -1):
            w = F.relu(self.w1[:, i]); w = w / (w.sum() + self.eps)
            up = F.interpolate(td[-1], size=feats[i].shape[-2:], mode='nearest')
            td.append(self.act(self.td[i](w[0]*feats[i] + w[1]*up)))
        td = td[::-1]
        out = [td[0]]
        for i in range(1, len(feats)):
            w = F.relu(self.w2[:, i-1]); w = w / (w.sum() + self.eps)
            dn = F.adaptive_avg_pool2d(out[-1], td[i].shape[-2:])
            out.append(self.act(self.bu[i-1](w[0]*feats[i]+w[1]*td[i]+w[2]*dn)))
        return out


class BlastYOLO(nn.Module):
    def __init__(self, num_classes=24, backbone='convnext_tiny',
                 pretrained=True, fpn_ch=192, attention='cbam'):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone, pretrained=pretrained,
                                          features_only=True, out_indices=(1,2,3))
        ch_in = self.backbone.feature_info.channels()
        self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_ch, 1) for c in ch_in])
        self.attn = nn.ModuleList([(CBAM(fpn_ch) if attention=='cbam' else ECA(fpn_ch))
                                   for _ in ch_in])
        self.bifpn = BiFPN(fpn_ch, levels=len(ch_in))
        # Heads
        self.box_head = nn.Conv2d(fpn_ch, 4*3, 1)
        self.obj_head = nn.Conv2d(fpn_ch, 1*3, 1)
        self.cls_head = nn.Conv2d(fpn_ch, num_classes*3, 1)
        self.seg_head = nn.Sequential(
            nn.ConvTranspose2d(fpn_ch, fpn_ch//2, 2, stride=2), nn.SiLU(),
            CBAM(fpn_ch//2),
            nn.ConvTranspose2d(fpn_ch//2, fpn_ch//4, 2, stride=2), nn.SiLU(),
            nn.Conv2d(fpn_ch//4, 3, 1))            # ICM/TE/cavity
        self.grade_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(fpn_ch, 512), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(512, num_classes))

    def forward(self, x):
        feats = self.backbone(x)
        feats = [a(l(f)) for f,l,a in zip(feats, self.lateral, self.attn)]
        feats = self.bifpn(feats)
        return {
            'det':    {'boxes': self.box_head(feats[0]),
                       'obj':   self.obj_head(feats[0]),
                       'cls':   self.cls_head(feats[0])},
            'masks':  self.seg_head(feats[0]),
            'logits': self.grade_head(feats[-1]),
        }


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ============================================================
# 4. LOSSES
# ============================================================
def bbox_ciou(b1, b2, eps=1e-7):
    b1x1,b1y1,b1x2,b1y2 = b1.unbind(-1)
    b2x1,b2y1,b2x2,b2y2 = b2.unbind(-1)
    inter = (torch.minimum(b1x2,b2x2)-torch.maximum(b1x1,b2x1)).clamp(0)*\
            (torch.minimum(b1y2,b2y2)-torch.maximum(b1y1,b2y1)).clamp(0)
    a1 = (b1x2-b1x1)*(b1y2-b1y1); a2 = (b2x2-b2x1)*(b2y2-b2y1)
    iou = inter / (a1+a2-inter+eps)
    cw = torch.maximum(b1x2,b2x2)-torch.minimum(b1x1,b2x1)
    ch = torch.maximum(b1y2,b2y2)-torch.minimum(b1y1,b2y1)
    c2 = cw**2 + ch**2 + eps
    rho2 = ((b2x1+b2x2-b1x1-b1x2)**2+(b2y1+b2y2-b1y1-b1y2)**2)/4
    w1,h1 = b1x2-b1x1, b1y2-b1y1+eps; w2,h2 = b2x2-b2x1, b2y2-b2y1+eps
    v = (4/math.pi**2) * torch.pow(torch.atan(w2/h2)-torch.atan(w1/h1), 2)
    with torch.no_grad(): alpha = v/(1-iou+v+eps)
    return iou - (rho2/c2 + v*alpha)


# Gardner expansion-equivalent stage per class
# P1 (high priority) -> stage 6; LP (low potential) -> stage 1; REV -> off-axis
EXPANSION = [2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4, 5,5,5,5,5, 6, 1, 6, 0]


class BlastYOLOLoss(nn.Module):
    def __init__(self, num_classes=24, lambda_box=7.5, lambda_cls=0.5,
                 lambda_dice=1.0, lambda_ord=0.3,
                 class_weights: Optional[torch.Tensor] = None,
                 gamma=2.0, ls=0.05):
        super().__init__()
        self.lb,self.lc,self.ld,self.lo = lambda_box, lambda_cls, lambda_dice, lambda_ord
        self.gamma = gamma; self.ls = ls
        self.register_buffer('cw', class_weights if class_weights is not None
                             else torch.ones(num_classes))
        stages = torch.tensor(EXPANSION, dtype=torch.float32)
        mask = (stages > 0).float()
        diff = (stages.unsqueeze(0)-stages.unsqueeze(1)).abs()
        diff = diff * mask.unsqueeze(0) * mask.unsqueeze(1)
        self.register_buffer('ord_cost', diff / (diff.max()+1e-8))

    def focal(self, logits, target):
        n = logits.size(-1)
        with torch.no_grad():
            t = torch.zeros_like(logits).fill_(self.ls/(n-1))
            t.scatter_(1, target.long().unsqueeze(1), 1.0-self.ls)
        log_p = F.log_softmax(logits, -1); p = log_p.exp()
        loss = -t * (1-p)**self.gamma * log_p
        loss = loss * self.cw.unsqueeze(0)
        return loss.sum(-1).mean()

    @staticmethod
    def _align(pred, target):
        """Resize pred to target spatial size if they differ."""
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:],
                                 mode='bilinear', align_corners=False)
        return pred

    def dice(self, pred, target):
        pred = torch.sigmoid(self._align(pred, target))
        inter = (pred*target).sum((-2,-1))
        union = pred.sum((-2,-1)) + target.sum((-2,-1))
        return 1.0 - ((2*inter+1)/(union+1)).mean()

    def tversky(self, pred, target, a=0.7, b=0.3):
        pred = torch.sigmoid(self._align(pred, target))
        tp = (pred*target).sum((-2,-1))
        fp = (pred*(1-target)).sum((-2,-1))
        fn = ((1-pred)*target).sum((-2,-1))
        return 1.0 - ((tp+1)/(tp + a*fn + b*fp + 1)).mean()

    def ordinal(self, logits, target):
        probs = F.softmax(logits, -1)
        return (probs * self.ord_cost[target.long()]).sum(-1).mean()

    def forward(self, preds, targets):
        l_box  = (1.0 - bbox_ciou(preds['boxes'],  targets['boxes'])).mean()
        l_cls  = self.focal(preds['logits'],  targets['labels'])
        l_dice = 0.5*self.dice(preds['masks'], targets['masks']) + \
                 0.5*self.tversky(preds['masks'], targets['masks'])
        l_ord  = self.ordinal(preds['logits'], targets['labels'])
        total  = self.lb*l_box + self.lc*l_cls + self.ld*l_dice + self.lo*l_ord
        return {'total': total, 'box': l_box, 'cls': l_cls,
                'dice': l_dice, 'ordinal': l_ord}


# ============================================================
# 5. BENCHMARK FACTORY
# ============================================================
CLASSIFIERS = ['alexnet','vgg16','vgg19','resnet50','densenet121',
               'efficientnetv2_s','convnext_tiny','vit_b_16']


def build_classifier(name, num_classes=24, pretrained=True, freeze=False):
    import torchvision.models as tvm
    w = 'DEFAULT' if pretrained else None
    name = name.lower()
    if name=='alexnet':
        m = tvm.alexnet(weights=w)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name=='vgg16':
        m = tvm.vgg16(weights=w)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name=='vgg19':
        m = tvm.vgg19(weights=w)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes)
    elif name=='resnet50':
        m = tvm.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name=='densenet121':
        m = tvm.densenet121(weights=w)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name=='efficientnetv2_s':
        m = tvm.efficientnet_v2_s(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name=='convnext_tiny':
        m = tvm.convnext_tiny(weights=w)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, num_classes)
    elif name in ('vit_b_16','vit'):
        m = tvm.vit_b_16(weights=w)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown: {name}")
    if freeze:
        for p in m.parameters(): p.requires_grad = False
        for name_, mod in m.named_modules():
            if isinstance(mod, nn.Linear) and mod.out_features == num_classes:
                for p in mod.parameters(): p.requires_grad = True
    return m


def build_detector(name, num_classes=24, pretrained=True):
    import torchvision.models.detection as dt
    name = name.lower(); w = 'DEFAULT' if pretrained else None
    if name=='faster_rcnn':
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        m = dt.fasterrcnn_resnet50_fpn_v2(weights=w)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes+1)
        return m
    if name=='mask_rcnn':
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        m = dt.maskrcnn_resnet50_fpn_v2(weights=w)
        in_f = m.roi_heads.box_predictor.cls_score.in_features
        m.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes+1)
        in_m = m.roi_heads.mask_predictor.conv5_mask.in_channels
        m.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, num_classes+1)
        return m
    if name=='ssd':
        return dt.ssd300_vgg16(weights=w, num_classes=num_classes+1)
    if name=='retinanet':
        return dt.retinanet_resnet50_fpn_v2(weights=w, num_classes=num_classes+1)
    raise ValueError(f"Unknown detector: {name}")


def build_unet(num_classes=1, encoder='resnet34', pretrained=True):
    try:
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name=encoder,
                        encoder_weights='imagenet' if pretrained else None,
                        classes=num_classes, activation=None)
    except ImportError:
        print("segmentation_models_pytorch not installed; skipping U-Net.")
        return None


# ============================================================
# 6. TRAINING LOOPS
# ============================================================
def safe_pct(warmup, epochs): return min(0.3, max(0.05, warmup/max(1,epochs)))


def train_novel(data_dir, epochs=100, batch=16, imgsz=640,
                backbone='convnext_tiny', attention='cbam',
                lr=3e-4, warmup=5, head_only_epochs=10,
                out='runs/blast_yolo_novel', device=None,
                checkpoint_to_drive=None,
                patience=15, min_delta=1e-4, monitor='val_loss'):
    """Train the novel Blast-YOLO. Smoke-test safe."""
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    Path(out).mkdir(parents=True, exist_ok=True)

    cw = compute_class_weights(Path(data_dir)/'train'/'labels')
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)
    print("Class weights:", dict(zip(CLASS_NAMES, cw.round(2))))

    tr = BlastYOLODataset(data_dir, 'train', imgsz)
    va = BlastYOLODataset(data_dir, 'val',   imgsz)
    coll = lambda b: (torch.stack([x[0] for x in b]), [x[1] for x in b])
    tr_ld = DataLoader(tr, batch, shuffle=True,  num_workers=2,
                       collate_fn=coll, pin_memory=True)
    va_ld = DataLoader(va, batch, shuffle=False, num_workers=2,
                       collate_fn=coll, pin_memory=True)

    model = BlastYOLO(num_classes=len(CLASS_NAMES), backbone=backbone,
                      attention=attention, pretrained=True).to(device)
    print(f"Trainable parameters: {count_params(model):,}")
    loss_fn = BlastYOLOLoss(num_classes=len(CLASS_NAMES),
                            class_weights=cw_t).to(device)

    for p in model.backbone.parameters(): p.requires_grad = False

    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(tr_ld))
    sched = OneCycleLR(optim, max_lr=lr, total_steps=steps,
                       pct_start=safe_pct(warmup, epochs))

    # Early-stopping state
    log, best = [], float('inf') if monitor=='val_loss' else 0.0
    epochs_no_improve = 0
    print(f"Early stopping: patience={patience} epochs on '{monitor}' "
          f"(min_delta={min_delta})")
    for ep in range(epochs):
        if ep == head_only_epochs:
            print(">> Stage 2: unfreezing backbone")
            for p in model.backbone.parameters(): p.requires_grad = True

        model.train(); t0 = time.time(); losses = []
        for imgs, tgts in tr_ld:
            imgs = imgs.to(device, non_blocking=True)
            boxes  = torch.stack([t['boxes'][0]  for t in tgts]).to(device)
            labels = torch.stack([t['labels'][0] for t in tgts]).to(device)
            masks  = torch.zeros(len(imgs),3,imgsz//4,imgsz//4, device=device)
            out_p  = model(imgs)
            preds  = {'boxes': boxes, 'logits': out_p['logits'],
                      'masks': out_p['masks']}
            tgt    = {'boxes': boxes, 'labels': labels, 'masks': masks}
            ld     = loss_fn(preds, tgt)
            optim.zero_grad(); ld['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step(); sched.step()
            losses.append({k: float(v) for k,v in ld.items()})
        train_loss = float(np.mean([l['total'] for l in losses]))

        model.eval(); vl = []; correct = 0; total = 0
        with torch.no_grad():
            for imgs, tgts in va_ld:
                imgs = imgs.to(device, non_blocking=True)
                labels = torch.stack([t['labels'][0] for t in tgts]).to(device)
                boxes  = torch.stack([t['boxes'][0]  for t in tgts]).to(device)
                masks  = torch.zeros(len(imgs),3,imgsz//4,imgsz//4, device=device)
                out_p  = model(imgs)
                preds  = {'boxes': boxes, 'logits': out_p['logits'],
                          'masks': out_p['masks']}
                tgt    = {'boxes': boxes, 'labels': labels, 'masks': masks}
                vl.append(float(loss_fn(preds, tgt)['total']))
                p = out_p['logits'].argmax(-1)
                correct += (p == labels).sum().item(); total += len(p)
        vloss = float(np.mean(vl)); vacc = correct / max(1, total)
        print(f"ep {ep+1:>3}/{epochs} | train {train_loss:.4f} | "
              f"val {vloss:.4f} | acc {vacc:.3f} | {time.time()-t0:.1f}s")
        log.append({'epoch':ep+1,'train_loss':train_loss,
                    'val_loss':vloss,'val_acc':vacc})
        # Early-stopping bookkeeping
        if monitor == 'val_loss':
            improved = vloss < (best - min_delta)
            current = vloss
        else:  # val_acc
            improved = vacc > (best + min_delta)
            current = vacc
        if improved:
            best = current
            epochs_no_improve = 0
            ck = {'model': model.state_dict(),
                  'class_names': CLASS_NAMES,
                  'backbone': backbone, 'attention': attention,
                  'best_val_loss': vloss, 'best_val_acc': vacc,
                  'best_epoch': ep + 1}
            torch.save(ck, Path(out)/'best.pt')
            if checkpoint_to_drive:
                shutil.copy(Path(out)/'best.pt', checkpoint_to_drive)
            print(f"    -> new best {monitor}={current:.4f}")
        else:
            epochs_no_improve += 1
            print(f"    -> no improvement ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"\nEARLY STOPPING at epoch {ep+1}: "
                      f"'{monitor}' did not improve for {patience} epochs. "
                      f"Best was {best:.4f}.")
                break

    json.dump(log, open(Path(out)/'train_log.json','w'), indent=2)
    print(f"Done. Best {monitor} = {best:.4f}. Saved to {out}")
    return model, log


def train_classifier(name, data_flat, epochs=30, batch=32, imgsz=224,
                     lr=3e-4, head_only_epochs=5, out='runs/benchmark',
                     device=None, patience=10, min_delta=1e-4):
    """Train an ImageNet-pretrained classifier on flat-folder layout."""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    tr_tfm = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    va_tfm = transforms.Compose([
        transforms.Resize((imgsz, imgsz)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tr = ImageFolder(f"{data_flat}/train", tr_tfm)
    va = ImageFolder(f"{data_flat}/val",   va_tfm)
    tr_ld = DataLoader(tr, batch, shuffle=True,  num_workers=2, pin_memory=True)
    va_ld = DataLoader(va, batch, shuffle=False, num_workers=2, pin_memory=True)
    n_cls = len(tr.classes)

    model = build_classifier(name, n_cls, pretrained=True, freeze=True).to(device)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=lr, weight_decay=1e-4)
    sched = OneCycleLR(optim, max_lr=lr,
                       total_steps=epochs * max(1, len(tr_ld)),
                       pct_start=safe_pct(2, epochs))

    out = Path(out)/name; out.mkdir(parents=True, exist_ok=True)
    log, best = [], 0.0
    epochs_no_improve = 0
    print(f"[{name}] Early stopping: patience={patience} epochs on val_acc")
    for ep in range(epochs):
        if ep == head_only_epochs:
            for p in model.parameters(): p.requires_grad = True
        model.train(); t0 = time.time(); losses = []
        for x,y in tr_ld:
            x,y = x.to(device), y.to(device)
            p = model(x); l = crit(p, y)
            optim.zero_grad(); l.backward(); optim.step(); sched.step()
            losses.append(float(l))
        train_loss = float(np.mean(losses))
        model.eval(); correct = 0; total = 0; vl = []
        with torch.no_grad():
            for x,y in va_ld:
                x,y = x.to(device), y.to(device)
                p = model(x); vl.append(float(crit(p,y)))
                correct += (p.argmax(-1)==y).sum().item(); total += len(y)
        vloss = float(np.mean(vl)); vacc = correct/max(1,total)
        print(f"[{name}] ep {ep+1:>3}/{epochs} | train {train_loss:.4f} | "
              f"val {vloss:.4f} | acc {vacc:.3f} | {time.time()-t0:.1f}s")
        log.append({'epoch':ep+1,'train_loss':train_loss,
                    'val_loss':vloss,'val_acc':vacc})
        if vacc > (best + min_delta):
            best = vacc; epochs_no_improve = 0
            torch.save({'model':model.state_dict(),'classes':tr.classes,
                        'name':name, 'best_val_acc': vacc,
                        'best_epoch': ep+1}, out/'best.pt')
            print(f"    -> [{name}] new best val_acc={best:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[{name}] EARLY STOPPING at epoch {ep+1}: "
                      f"val_acc did not improve for {patience} epochs. "
                      f"Best was {best:.4f}.")
                break
    json.dump(log, open(out/'train_log.json','w'), indent=2)
    print(f"[{name}] Done. Best val_acc = {best:.4f}")
    return model, log


# ============================================================
# 7. EVALUATION
# ============================================================
def per_class_metrics(y_true, y_pred, y_score=None,
                      class_names=CLASS_NAMES):
    from sklearn.metrics import roc_auc_score, confusion_matrix
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    rows = []
    for c, name in enumerate(class_names):
        tp = cm[c,c]; fn = cm[c,:].sum()-tp
        fp = cm[:,c].sum()-tp; tn = cm.sum()-tp-fn-fp
        sens = tp/(tp+fn) if tp+fn else 0
        spec = tn/(tn+fp) if tn+fp else 0
        prec = tp/(tp+fp) if tp+fp else 0
        f1   = 2*prec*sens/(prec+sens) if prec+sens else 0
        auc_v = np.nan
        if y_score is not None:
            try:
                yt = (y_true==c).astype(int)
                if 0 < yt.sum() < len(yt):
                    auc_v = roc_auc_score(yt, y_score[:,c])
            except: pass
        rows.append({'class':name, 'support':int(cm[c,:].sum()),
                     'sensitivity':sens, 'specificity':spec,
                     'precision':prec, 'recall':sens,
                     'F1':f1, 'AUC':auc_v})
    return pd.DataFrame(rows).set_index('class')


def overall_metrics(pc_df):
    w = pc_df['support'] / pc_df['support'].sum()
    return {
        'macro_sensitivity':    pc_df['sensitivity'].mean(),
        'weighted_sensitivity': float((pc_df['sensitivity']*w).sum()),
        'macro_specificity':    pc_df['specificity'].mean(),
        'macro_F1':             pc_df['F1'].mean(),
        'weighted_F1':          float((pc_df['F1']*w).sum()),
        'macro_AUC':            pc_df['AUC'].mean(skipna=True),
    }


def plot_per_class_roc(y_true, y_score, outpath, title='Per-class ROC',
                       class_names=CLASS_NAMES):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    n = len(class_names)
    yb = label_binarize(y_true, classes=list(range(n)))
    plt.figure(figsize=(10,8))
    for c, name in enumerate(class_names):
        if yb[:,c].sum()==0: continue
        fpr,tpr,_ = roc_curve(yb[:,c], y_score[:,c])
        plt.plot(fpr, tpr, lw=1.2, label=f"{name} ({auc(fpr,tpr):.3f})")
    plt.plot([0,1],[0,1],'k--',lw=0.8)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title)
    plt.legend(fontsize=7, ncol=2, loc='lower right')
    plt.tight_layout(); Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180); plt.close()


def plot_overall_roc(y_true, y_score, outpath, title='Overall ROC',
                     class_names=CLASS_NAMES):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    n = len(class_names)
    yb = label_binarize(y_true, classes=list(range(n)))
    fpr_mi, tpr_mi, _ = roc_curve(yb.ravel(), y_score.ravel())
    auc_mi = auc(fpr_mi, tpr_mi)
    all_fpr = np.unique(np.concatenate(
        [roc_curve(yb[:,c],y_score[:,c])[0] for c in range(n) if yb[:,c].sum()]))
    mean_tpr = np.zeros_like(all_fpr); valid = 0
    for c in range(n):
        if yb[:,c].sum()==0: continue
        fpr,tpr,_ = roc_curve(yb[:,c], y_score[:,c])
        mean_tpr += np.interp(all_fpr, fpr, tpr); valid += 1
    mean_tpr /= max(1,valid); auc_ma = auc(all_fpr, mean_tpr)
    plt.figure(figsize=(7,6))
    plt.plot(fpr_mi, tpr_mi, lw=2, label=f"Micro ({auc_mi:.3f})")
    plt.plot(all_fpr, mean_tpr, lw=2, label=f"Macro ({auc_ma:.3f})")
    plt.plot([0,1],[0,1],'k--',lw=0.8)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(title); plt.legend()
    plt.tight_layout(); Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180); plt.close()
    return {'micro_AUC': float(auc_mi), 'macro_AUC': float(auc_ma)}


def plot_cm(y_true, y_pred, outpath, title='Confusion Matrix',
            class_names=CLASS_NAMES, normalize='true'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))),
                          normalize=normalize)
    plt.figure(figsize=(14,12))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                cmap='Blues', square=True,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(title)
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180); plt.close()


def plot_clinical_cm(y_true, y_pred, outpath, class_names=CLASS_NAMES,
                     groups=CLINICAL_GROUPS):
    name_to_idx = {n:i for i,n in enumerate(class_names)}
    gnames = list(groups.keys())
    cls_to_g = {}
    for gi,g in enumerate(gnames):
        for c in groups[g]:
            if c in name_to_idx: cls_to_g[name_to_idx[c]] = gi
    yt = np.array([cls_to_g.get(int(i), -1) for i in y_true])
    yp = np.array([cls_to_g.get(int(i), -1) for i in y_pred])
    keep = (yt>=0)&(yp>=0)
    plot_cm(yt[keep], yp[keep], outpath, class_names=gnames,
            title='Clinical-grouped CM')


def vector_analysis(emb, labels, outdir, model_name, class_names=CLASS_NAMES):
    """t-SNE + UMAP + silhouette/k-NN purity."""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import seaborn as sns
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    emb_t = TSNE(n_components=2, perplexity=30, random_state=42,
                 init='pca').fit_transform(emb)
    plt.figure(figsize=(10,8))
    palette = sns.color_palette('tab20', n_colors=len(class_names))
    for c,name in enumerate(class_names):
        m = labels==c
        if m.sum()==0: continue
        plt.scatter(emb_t[m,0],emb_t[m,1], s=8, alpha=0.7,
                    color=palette[c%len(palette)], label=name)
    plt.legend(fontsize=7, ncol=2, loc='upper right',
               bbox_to_anchor=(1.18,1.0))
    plt.title(f"t-SNE - {model_name}"); plt.tight_layout()
    plt.savefig(out/f"{model_name}_tsne.png", dpi=180, bbox_inches='tight')
    plt.close()
    metrics = {
        'silhouette':     float(silhouette_score(emb, labels,
                                                 sample_size=min(2000,len(emb)))),
        'davies_bouldin': float(davies_bouldin_score(emb, labels)),
    }
    nn = NearestNeighbors(n_neighbors=6).fit(emb)
    _, idx = nn.kneighbors(emb)
    metrics['knn_purity_k5'] = float((labels[idx[:,1:]] == labels[:,None]).mean())
    pd.Series(metrics, name=model_name).to_csv(out/f"{model_name}_separation.csv")
    return metrics


def cross_model_overlay(scores_dict, y_true, outpath,
                        class_names=CLASS_NAMES):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    yb = label_binarize(y_true, classes=list(range(len(class_names))))
    plt.figure(figsize=(8,6)); summary = {}
    for model, y_score in scores_dict.items():
        all_fpr = np.unique(np.concatenate(
            [roc_curve(yb[:,c],y_score[:,c])[0] for c in range(len(class_names))
             if yb[:,c].sum()]))
        mean_tpr = np.zeros_like(all_fpr); valid = 0
        for c in range(len(class_names)):
            if yb[:,c].sum()==0: continue
            fpr,tpr,_ = roc_curve(yb[:,c], y_score[:,c])
            mean_tpr += np.interp(all_fpr, fpr, tpr); valid += 1
        mean_tpr /= max(1,valid); a = auc(all_fpr, mean_tpr)
        summary[model] = float(a)
        plt.plot(all_fpr, mean_tpr, lw=1.6, label=f"{model} ({a:.3f})")
    plt.plot([0,1],[0,1],'k--',lw=0.8)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('Macro ROC by model')
    plt.legend(fontsize=8, loc='lower right'); plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180); plt.close()
    return summary


def comparison_table(per_model_overall, outdir):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_model_overall).T
    df = df.sort_values(df.columns[0], ascending=False)
    df.to_csv(out/'comparison.csv')
    with open(out/'comparison.md','w') as f:
        f.write(df.round(4).to_markdown())
    with open(out/'comparison.tex','w') as f:
        f.write(df.round(4).to_latex(float_format='%.4f'))
    return df


# ============================================================
# 8. PREDICT HELPERS (for evaluation)
# ============================================================
def predict_classifier(model, data_flat, split='test', batch=32, imgsz=224,
                       device=None):
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([transforms.Resize((imgsz,imgsz)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    ds = ImageFolder(f"{data_flat}/{split}", tfm)
    ld = DataLoader(ds, batch, shuffle=False, num_workers=2)
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x,y in ld:
            p = torch.softmax(model(x.to(device)), dim=-1)
            ys.append(y.numpy()); ps.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), ds.classes


def predict_novel(model, data_flat, split='test', batch=8, imgsz=640,
                  device=None):
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tfm = transforms.Compose([transforms.Resize((imgsz,imgsz)),
                              transforms.ToTensor()])
    ds = ImageFolder(f"{data_flat}/{split}", tfm)
    ld = DataLoader(ds, batch, shuffle=False, num_workers=2)
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x,y in ld:
            out = model(x.to(device))
            p = torch.softmax(out['logits'], dim=-1)
            ys.append(y.numpy()); ps.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), ds.classes
