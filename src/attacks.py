import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from .models import cnn_proba


def _bounds_from_data(X):
    # Standardized feature values can be perturbed, but clipping prevents
    # unbounded adversarial drift. These bounds are intentionally conservative.
    return np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0)


def fgsm(model, X, y, epsilon=0.03, device=None, clip=True):
    device = device or next(model.parameters()).device; model.eval()
    xb = torch.from_numpy(X.copy()).float().to(device).unsqueeze(1)
    yb = torch.from_numpy(y).long().to(device); xb.requires_grad_(True)
    loss = CrossEntropyLoss()(model(xb), yb); model.zero_grad(); loss.backward()
    adv = (xb + epsilon * xb.grad.sign()).detach().squeeze(1).cpu().numpy().astype('float32')
    if clip:
        lo, hi = _bounds_from_data(X)
        adv = np.clip(adv, lo, hi).astype('float32')
    return adv


def pgd(model, X, y, epsilon=0.03, alpha=0.0075, steps=5, device=None, clip=True):
    device = device or next(model.parameters()).device; model.eval()
    x0 = torch.from_numpy(X.copy()).float().to(device).unsqueeze(1)
    yb = torch.from_numpy(y).long().to(device)
    adv = x0.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        loss = CrossEntropyLoss()(model(adv), yb); model.zero_grad(); loss.backward()
        adv = adv.detach() + alpha * adv.grad.sign()
        adv = torch.max(torch.min(adv, x0 + epsilon), x0 - epsilon)
    out = adv.squeeze(1).cpu().numpy().astype('float32')
    if clip:
        lo, hi = _bounds_from_data(X)
        out = np.clip(out, lo, hi).astype('float32')
    return out


def hybrid_proba(rf, cnn, X, device=None, rf_weight=0.5):
    rp = rf.predict_proba(X)[:, 1]; cp = cnn_proba(cnn, X, device)
    return rf_weight * rp + (1 - rf_weight) * cp
