"""Inference wrapper - identical to v3 streamlit_app/inference.py."""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import pipeline as P

_NORM = T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])

ACTION_LABELS = P.ACTION_LABELS

def _clinical_group(grade):
    for g, members in P.CLINICAL_GROUPS.items():
        if grade in members: return g.replace('_',' ')
    return 'unknown'

def _overlay(img, masks):
    if masks is None: return img
    a = np.asarray(img).astype(np.float32)
    m = torch.nn.functional.interpolate(
        masks.unsqueeze(0), size=img.size[::-1], mode='bilinear',
        align_corners=False).squeeze(0).cpu().numpy()
    m = 1/(1+np.exp(-m))
    a[...,0] = np.clip(a[...,0]+80*m[0], 0, 255)
    a[...,1] = np.clip(a[...,1]+80*m[1], 0, 255)
    a[...,2] = np.clip(a[...,2]+80*m[2], 0, 255)
    return Image.fromarray(a.astype(np.uint8))


class BlastocystGrader:
    def __init__(self, weights, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        ck = torch.load(weights, map_location=self.device)
        self.class_names = ck.get('class_names', ck.get('classes', P.CLASS_NAMES))
        if 'backbone' in ck:
            self.model_name = 'Blast-YOLO (novel)'
            self.model = P.BlastYOLO(num_classes=len(self.class_names),
                                     backbone=ck['backbone'],
                                     attention=ck.get('attention','cbam'),
                                     pretrained=False).to(self.device)
            self.model.load_state_dict(ck['model'])
            self.imgsz=640; self.has_seg=True
            self.tfm = T.Compose([T.Resize((640,640)), T.ToTensor()])
        else:
            self.model_name = ck.get('name','classifier')
            self.model = P.build_classifier(self.model_name,
                                            len(self.class_names),
                                            pretrained=False).to(self.device)
            self.model.load_state_dict(ck['model'])
            self.imgsz=224; self.has_seg=False
            self.tfm = T.Compose([T.Resize((224,224)), T.ToTensor(), _NORM])
        self.model.eval()

    @torch.no_grad()
    def predict(self, img):
        x = self.tfm(img).unsqueeze(0).to(self.device)
        if self.has_seg:
            out = self.model(x)
            logits = out['logits']; masks = out['masks'][0]
        else:
            logits = self.model(x); masks = None
        probs = torch.softmax(logits, -1)[0].cpu().numpy()
        idx = int(np.argmax(probs)); order = np.argsort(probs)[::-1][:3]
        grade = self.class_names[idx]
        return {
            'grade': grade, 'confidence': float(probs[idx]),
            'clinical_group': _clinical_group(grade),
            'action': ACTION_LABELS.get(grade, ''),
            'probs': probs.tolist(),
            'top3': [(self.class_names[i], float(probs[i])) for i in order],
            'overlay': _overlay(img, masks),
        }
