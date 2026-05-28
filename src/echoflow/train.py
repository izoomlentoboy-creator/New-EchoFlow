"""Train EchoFlowNet (hierarchical multi-task) on synthetic or real data.

    python -m echoflow.train                          # synthetic data
    python -m echoflow.train --data-root data/real     # real corpus
    python -m echoflow.train --epochs 40 --samples-per-class 800
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .config import Config, CLASSES, NUM_CLASSES
from .data.dataset import (build_synthetic_dataset, load_real_dataset,
                           VoiceDataset, Corpus)
from .models.fusion import EchoFlowNet


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_split(labels, val, test, seed):
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in range(NUM_CLASSES):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        nt, nv = int(n * test), int(n * val)
        te += idx[:nt].tolist()
        va += idx[nt:nt + nv].tolist()
        tr += idx[nt + nv:].tolist()
    rng.shuffle(tr)
    return tr, va, te


def macro_f1(cm: np.ndarray) -> float:
    f1s = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def multitask_loss(model, logb, logs, sev, yb, ys, yv, w_bin):
    """Kendall homoscedastic-uncertainty weighted multi-task loss."""
    lb = F.cross_entropy(logb, yb, weight=w_bin)
    # subtype loss only over pathological samples
    mask_sub = ys >= 0
    ls = (F.cross_entropy(logs[mask_sub], ys[mask_sub])
          if mask_sub.any() else logb.new_zeros(()))
    # severity loss only over labelled samples
    mask_sev = yv >= 0
    lv = (F.mse_loss(sev[mask_sev].squeeze(-1), yv[mask_sev])
          if mask_sev.any() else logb.new_zeros(()))
    s = model.log_var
    total = (0.5 * torch.exp(-s[0]) * lb + 0.5 * s[0]
             + 0.5 * torch.exp(-s[1]) * ls + 0.5 * s[1]
             + 0.5 * torch.exp(-s[2]) * lv + 0.5 * s[2])
    return total, float(lb.detach()), float(ls.detach()), float(lv.detach())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    bin_cm = np.zeros((2, 2), dtype=int)
    sev_err, sev_n = 0.0, 0
    for mel, bio, y5, yb, ys, yv in loader:
        mel, bio = mel.to(device), bio.to(device)
        probs, sev = model.class_probs(mel, bio)
        pred = probs.argmax(1).cpu().numpy()
        for t, p in zip(y5.numpy(), pred):
            cm[t, p] += 1
        pb = (probs[:, 1:].sum(1) > probs[:, 0]).long().cpu().numpy()
        for t, p in zip(yb.numpy(), pb):
            bin_cm[t, p] += 1
        m = yv >= 0
        if m.any():
            sev_err += float((sev.squeeze(-1).cpu()[m] - yv[m]).abs().sum())
            sev_n += int(m.sum())
    acc = np.trace(cm) / max(cm.sum(), 1)
    sens = bin_cm[1, 1] / max(bin_cm[1].sum(), 1)     # pathology recall
    spec = bin_cm[0, 0] / max(bin_cm[0].sum(), 1)     # healthy recall
    return dict(acc=acc, macro_f1=macro_f1(cm), cm=cm,
                sensitivity=sens, specificity=spec,
                sev_mae=(sev_err / sev_n if sev_n else float("nan")))


def fit_temperature(model, loader, device):
    """Temperature scaling on validation for calibrated probabilities.

    A single scalar T divides both head logits; optimized to minimize NLL of
    the assembled 5-class distribution.
    """
    model.eval()
    logb_all, logs_all, y_all = [], [], []
    with torch.no_grad():
        for mel, bio, y5, yb, ys, yv in loader:
            lb, ls, _ = model(mel.to(device), bio.to(device))
            logb_all.append(lb.cpu()); logs_all.append(ls.cpu())
            y_all.append(y5)
    logb = torch.cat(logb_all); logs = torch.cat(logs_all)
    y = torch.cat(y_all)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def assemble(T):
        pb = torch.softmax(logb / T, 1)
        ps = torch.softmax(logs / T, 1)
        return torch.cat([pb[:, :1], pb[:, 1:2] * ps], 1).clamp_min(1e-8)

    def closure():
        opt.zero_grad()
        probs = assemble(torch.exp(logT))
        loss = F.nll_loss(torch.log(probs), y)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(logT).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--samples-per-class", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs: cfg.train.epochs = args.epochs
    if args.samples_per_class: cfg.train.samples_per_class = args.samples_per_class
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.lr: cfg.train.lr = args.lr

    set_seed(cfg.train.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    if args.data_root:
        print(f"Loading real corpus from {args.data_root} ...")
        corpus = load_real_dataset(cfg, args.data_root)
    else:
        print(f"Synthetic corpus ({cfg.train.samples_per_class}/class) with "
              "domain randomization...")
        corpus = build_synthetic_dataset(cfg)
    print(f"Features ready in {time.time()-t0:.1f}s. mels={corpus.mels.shape}")

    tr, va, te = stratified_split(corpus.labels, cfg.train.val_split,
                                  cfg.train.test_split, cfg.train.seed)
    bio_mean = corpus.bios[tr].mean(0)
    bio_std = corpus.bios[tr].std(0)

    full = VoiceDataset(corpus, train=False)
    train_ds = VoiceDataset(corpus, train=True, seed=cfg.train.seed)
    train_loader = DataLoader(Subset(train_ds, tr), batch_size=cfg.train.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(full, va), batch_size=cfg.train.batch_size)
    test_loader = DataLoader(Subset(full, te), batch_size=cfg.train.batch_size)

    model = EchoFlowNet(cfg.features.n_acoustic, NUM_CLASSES).to(device)
    model.set_bio_stats(bio_mean, bio_std)

    # binary class weights (1 healthy : 4 pathology ⇒ up-weight healthy)
    yb_tr = np.array([0 if corpus.labels[i] == 0 else 1 for i in tr])
    cnt = np.bincount(yb_tr, minlength=2).astype(np.float32)
    w_bin = torch.tensor(cnt.sum() / (2 * np.clip(cnt, 1, None)),
                         dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    best_f1, best_state, patience = -1.0, None, 0
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        run = 0.0; n = 0
        for mel, bio, y5, yb, ys, yv in train_loader:
            mel, bio = mel.to(device), bio.to(device)
            yb, ys, yv = yb.to(device), ys.to(device), yv.to(device)
            opt.zero_grad()
            logb, logs, sev = model(mel, bio)
            loss, _, _, _ = multitask_loss(model, logb, logs, sev,
                                           yb, ys, yv, w_bin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * len(yb); n += len(yb)
        sched.step()
        m = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d}/{cfg.train.epochs} | loss {run/max(n,1):7.4f} | "
              f"val_acc {m['acc']:.4f} | macroF1 {m['macro_f1']:.4f} | "
              f"sens {m['sensitivity']:.3f} spec {m['specificity']:.3f}")
        if m["macro_f1"] > best_f1:
            best_f1 = m["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.early_stop_patience:
                print(f"Early stop @ {epoch} (best macroF1 {best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    temperature = fit_temperature(model, val_loader, device)
    print(f"Calibration temperature: {temperature:.3f}")

    m = evaluate(model, test_loader, device)
    cm = m["cm"]
    print("\n=== TEST ===")
    print(f"accuracy {m['acc']:.4f} | macroF1 {m['macro_f1']:.4f} | "
          f"sensitivity {m['sensitivity']:.4f} | specificity {m['specificity']:.4f} "
          f"| severity MAE {m['sev_mae']:.4f}")
    print("confusion matrix (rows=true, cols=pred):")
    print("        " + " ".join(f"{c[:6]:>7}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"{CLASSES[i][:7]:>7} " + " ".join(f"{v:7d}" for v in row))

    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[c] = dict(precision=round(float(p), 4), recall=round(float(r), 4),
                            f1=round(2 * p * r / (p + r), 4) if p + r else 0.0,
                            support=int(cm[i].sum()))

    ckpt = os.path.join(args.out, "echoflow.pt")
    torch.save({"model_state": model.state_dict(),
                "bio_mean": bio_mean, "bio_std": bio_std,
                "temperature": temperature,
                "config": json.loads(cfg.to_json()), "classes": CLASSES}, ckpt)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"test_accuracy": round(float(m["acc"]), 4),
                   "test_macro_f1": round(m["macro_f1"], 4),
                   "screening_sensitivity": round(float(m["sensitivity"]), 4),
                   "screening_specificity": round(float(m["specificity"]), 4),
                   "severity_mae": round(float(m["sev_mae"]), 4),
                   "calibration_temperature": round(temperature, 4),
                   "per_class": per_class,
                   "data_source": args.data_root or "synthetic"},
                  f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        f.write(cfg.to_json())
    print(f"\nSaved checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
