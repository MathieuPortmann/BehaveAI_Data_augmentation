#!/usr/bin/env python3
"""
BehaveAI Phase 2b - Self-supervised fine-tuning of the Re-ID embedding.

The intra-video tracker already gives us a free identity signal: within ONE
video, every distinct ``track_id`` is, by construction, ONE individual, and two
tracks that are alive in the SAME frame are necessarily DIFFERENT individuals.
We turn that into a metric-learning problem with NO manual identity labels:

  * each ``(video, track_id)`` pair becomes one class,
  * crops of that track (read back from the source video) are its examples,
  * a MegaDescriptor backbone is fine-tuned with an ArcFace margin head so that
    same-track crops cluster and different-track crops separate.

The result is a domain-adapted appearance embedding that
``BehaveAI_reid.ReIDRegistry`` loads at inference (see ``reid_checkpoint``),
which is exactly what helps the monochrome-herd case the histogram cannot.

This module is self-contained: it needs only ``timm`` (MegaDescriptor backbone)
and ``torch`` - it does NOT require the ``wildlife-tools`` package. The ArcFace
head below is a standard re-implementation.

Usage
-----
    python BehaveAI_reid_finetune.py <project_dir> [options]

Common options:
    --export-only           mine + export crops, skip training (inspect first)
    --epochs N              training epochs (default 30)
    --backbone T-224|L-224|L-384|T-CNN-288
    --max-per-track N       cap crops exported per track (default 40)
    --min-crops N           drop tracks with fewer than N crops (default 6)
    --batch N               batch size (default 32)
    --device cuda|cpu       (default: auto)
"""

import os
import sys
import csv
import glob
import math
import argparse
import configparser

import numpy as np
import cv2


# ----------------------------------------------------------------------------
# Project / path helpers
# ----------------------------------------------------------------------------

def _read_project_dirs(project_dir):
    """Return (clips_dir, output_dir, reid_model_dir) for a project."""
    ini = os.path.join(project_dir, "BehaveAI_settings.ini")
    clips_dir = os.path.join(project_dir, "clips")
    output_dir = os.path.join(project_dir, "output")
    if os.path.isfile(ini):
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        d = cfg["DEFAULT"]
        clips_dir = _resolve(project_dir, d.get("clips_dir", clips_dir))
        output_dir = _resolve(project_dir, d.get("output_dir", output_dir))
    reid_model_dir = os.path.join(project_dir, "model_reid")
    return clips_dir, output_dir, reid_model_dir


def _resolve(project_dir, p):
    if not p:
        return p
    p = p.strip()
    return p if os.path.isabs(p) else os.path.join(project_dir, p)


_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _index_videos(clips_dir):
    """Map video stem -> absolute path (recursive scan of clips_dir)."""
    index = {}
    for root, _dirs, files in os.walk(clips_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in _VIDEO_EXTS:
                index.setdefault(os.path.splitext(f)[0], os.path.join(root, f))
    return index


# ----------------------------------------------------------------------------
# Mining: parse tracking CSVs into per-class frame/bbox requests
# ----------------------------------------------------------------------------

def parse_tracking_csv(csv_path):
    """Yield (track_id, frame, bbox_or_None, (cx, cy)) for valid rows of one CSV.

    Columns 0-3 are frame,id,x,y (x,y = centroid). Newer CSVs additionally append
    x1,y1,x2,y2 as the LAST 4 columns; when present and valid they are returned as
    the bbox, otherwise bbox is None and the caller falls back to a fixed-size box
    around the centroid (keeps legacy 12-column CSVs usable).
    """
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return
        for row in reader:
            if len(row) < 4:
                continue
            try:
                frame = int(float(row[0]))
                tid = int(float(row[1]))
                cx, cy = float(row[2]), float(row[3])
            except (ValueError, IndexError):
                continue
            bbox = None
            if len(row) >= 16:
                try:
                    x1, y1, x2, y2 = (int(float(row[-4])), int(float(row[-3])),
                                      int(float(row[-2])), int(float(row[-1])))
                    if x2 > x1 and y2 > y1:
                        bbox = (x1, y1, x2, y2)
                except (ValueError, IndexError):
                    bbox = None
            yield tid, frame, bbox, (cx, cy)


def mine_classes(output_dir, video_index, max_per_track):
    """Build {class_label: [(video_path, frame, bbox_or_None, centroid), ...]}.

    class_label = "<video_stem>#<track_id>". For each class the requested crops
    are subsampled evenly to at most ``max_per_track`` so long tracks do not
    swamp the dataset.
    """
    requests = {}
    csvs = glob.glob(os.path.join(output_dir, "**", "*_tracking.csv"), recursive=True)
    for csv_path in csvs:
        stem = os.path.basename(csv_path)[:-len("_tracking.csv")]
        video = video_index.get(stem)
        if video is None:
            print(f"  ! no source video for {stem} (skipped)")
            continue
        per_track = {}
        for tid, frame, bbox, centroid in parse_tracking_csv(csv_path):
            per_track.setdefault(tid, []).append((frame, bbox, centroid))
        for tid, items in per_track.items():
            items = _subsample(items, max_per_track)
            label = f"{stem}#{tid}"
            requests[label] = [(video, fr, bb, ce) for fr, bb, ce in items]
    return requests


def _subsample(items, k):
    """Evenly pick at most k items, preserving temporal spread."""
    n = len(items)
    if n <= k:
        return items
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx.tolist()))]


def export_crops(requests, crops_dir, min_crops, img_size=224, pad=0.08,
                 box_size=160):
    """Read frames from videos and write per-class crop JPGs (ImageFolder layout).

    Returns the number of classes and crops actually written. Frames are read
    sequentially per video (sorted by frame index) to keep VideoCapture seeks
    cheap. A class is only kept if it yields >= min_crops crops. When a row has
    no bbox, a ``box_size`` square centred on the centroid is used instead.
    """
    # Regroup requests by video so each file is opened once.
    by_video = {}
    for label, items in requests.items():
        for (video, frame, bbox, centroid) in items:
            by_video.setdefault(video, []).append((label, frame, bbox, centroid))

    written = {}
    for video, items in by_video.items():
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"  ! cannot open {video}")
            continue
        items.sort(key=lambda t: t[1])
        for (label, frame, bbox, centroid) in items:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            h, w = img.shape[:2]
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
                xa, ya = x1 - dx, y1 - dy
                xb, yb = x2 + dx, y2 + dy
            else:
                cx, cy = int(centroid[0]), int(centroid[1])
                half = box_size // 2
                xa, ya, xb, yb = cx - half, cy - half, cx + half, cy + half
            xa, ya = max(0, xa), max(0, ya)
            xb, yb = min(w, xb), min(h, yb)
            crop = img[ya:yb, xa:xb]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (img_size, img_size))
            written.setdefault(label, []).append(crop)
        cap.release()

    n_classes = n_crops = 0
    for label, crops in written.items():
        if len(crops) < min_crops:
            continue
        cls_dir = os.path.join(crops_dir, _safe(label))
        os.makedirs(cls_dir, exist_ok=True)
        for i, crop in enumerate(crops):
            cv2.imwrite(os.path.join(cls_dir, f"{i:04d}.jpg"), crop)
        n_classes += 1
        n_crops += len(crops)
    return n_classes, n_crops


def _safe(label):
    return "".join(c if c.isalnum() or c in "-._#" else "_" for c in label)


# ----------------------------------------------------------------------------
# ArcFace head + training (lazy torch/timm import)
# ----------------------------------------------------------------------------

_MEGADESCRIPTOR_TAGS = {
    "T-224": "BVRA/MegaDescriptor-T-224",
    "L-224": "BVRA/MegaDescriptor-L-224",
    "L-384": "BVRA/MegaDescriptor-L-384",
    "T-CNN-288": "BVRA/MegaDescriptor-T-CNN-288",
}


def _build_arcface(torch):
    import torch.nn as nn
    import torch.nn.functional as F

    class ArcFace(nn.Module):
        """Additive angular margin head (Deng et al., 2019)."""

        def __init__(self, in_features, num_classes, scale=30.0, margin=0.30):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(num_classes, in_features))
            nn.init.xavier_uniform_(self.weight)
            self.scale, self.margin = scale, margin

        def forward(self, feats, labels):
            cos = F.linear(F.normalize(feats), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
            theta = torch.acos(cos)
            target = torch.cos(theta + self.margin)
            onehot = F.one_hot(labels, num_classes=self.weight.shape[0]).float()
            logits = self.scale * (onehot * target + (1.0 - onehot) * cos)
            return logits

    return ArcFace


def train_embedding(crops_dir, out_path, backbone="T-224", epochs=30,
                    batch=32, lr=1e-4, device=None, img_size=224):
    """Fine-tune a MegaDescriptor backbone with ArcFace; save the backbone weights.

    Saves a dict {'state_dict', 'backbone', 'embed_dim', 'num_classes'} to
    ``out_path`` so ``BehaveAI_reid`` can rebuild and load it.
    """
    import torch
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tag = _MEGADESCRIPTOR_TAGS.get(backbone, _MEGADESCRIPTOR_TAGS["T-224"])

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ds = ImageFolder(crops_dir, transform=tfm)
    if len(ds.classes) < 2:
        raise RuntimeError(
            f"Need >=2 track-identities to fine-tune; found {len(ds.classes)} "
            f"in {crops_dir}. Run the tracker on more video first.")
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)

    backbone_model = timm.create_model(f"hf-hub:{tag}", pretrained=True, num_classes=0).to(device)
    embed_dim = backbone_model.num_features
    head = _build_arcface(torch)(embed_dim, len(ds.classes)).to(device)

    params = list(backbone_model.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    ce = torch.nn.CrossEntropyLoss()

    print(f"Fine-tuning {tag} on {len(ds)} crops / {len(ds.classes)} identities "
          f"({device}, {epochs} epochs)...")
    backbone_model.train(); head.train()
    for ep in range(epochs):
        tot = correct = 0
        run_loss = 0.0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = backbone_model(imgs)
            logits = head(feats, labels)
            loss = ce(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            run_loss += loss.item() * imgs.size(0)
            correct += int((logits.argmax(1) == labels).sum())
            tot += imgs.size(0)
        sched.step()
        print(f"  epoch {ep + 1:3d}/{epochs}  loss={run_loss / max(1, tot):.4f}  "
              f"train_acc={correct / max(1, tot):.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "state_dict": backbone_model.state_dict(),
        "backbone": backbone,
        "embed_dim": int(embed_dim),
        "num_classes": int(len(ds.classes)),
    }, out_path)
    print(f"Saved fine-tuned embedding -> {out_path}")
    return out_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Self-supervised Re-ID embedding fine-tuning")
    ap.add_argument("project_dir")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--backbone", default="T-224", choices=list(_MEGADESCRIPTOR_TAGS))
    ap.add_argument("--max-per-track", type=int, default=40)
    ap.add_argument("--min-crops", type=int, default=6)
    ap.add_argument("--box-size", type=int, default=160,
                    help="square crop side (px) around the centroid when the CSV "
                         "has no bbox columns (legacy 12-column tracking CSVs)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    clips_dir, output_dir, reid_model_dir = _read_project_dirs(args.project_dir)
    crops_dir = os.path.join(reid_model_dir, "reid_crops")
    out_path = os.path.join(reid_model_dir, "megadescriptor_finetuned.pt")

    print(f"clips:  {clips_dir}")
    print(f"output: {output_dir}")
    if not os.path.isdir(output_dir):
        sys.exit(f"No output dir with tracking CSVs at {output_dir}. Run tracking first.")

    video_index = _index_videos(clips_dir)
    requests = mine_classes(output_dir, video_index, args.max_per_track)
    print(f"Mined {len(requests)} candidate track-identities; exporting crops...")
    n_classes, n_crops = export_crops(requests, crops_dir, args.min_crops,
                                      img_size=args.img_size, box_size=args.box_size)
    print(f"Exported {n_crops} crops across {n_classes} identities -> {crops_dir}")
    if n_classes < 2:
        sys.exit("Fewer than 2 usable identities; need more tracked video before fine-tuning.")

    if args.export_only:
        print("--export-only set; stopping before training.")
        return
    train_embedding(crops_dir, out_path, backbone=args.backbone, epochs=args.epochs,
                    batch=args.batch, device=args.device, img_size=args.img_size)


if __name__ == "__main__":
    main()
