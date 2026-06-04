#!/usr/bin/env python3
"""
BehaveAI Phase 2a - Auto-trained segmentation (whole horse + body parts).

Two complementary segmentation models, both obtained with NO manual mask
annotation, then domain-adapted to the user's overhead drone footage:

  STAGE 1 - whole horse (silhouette)
    Sample frames from the project clips, run ultralytics ``auto_annotate``
    (the project's detector proposes boxes, SAM2 turns each into a mask) to get
    YOLO-seg labels ON THE REAL DRONE FRAMES, then train a small YOLO11-seg
    "horse" model. Segmentation transfers to top-down views far better than
    side-view pose models, so this gives a robust foreground/orientation source
    for the Re-ID grid descriptor (BehaveAI_reid.py, reid_foreground=yoloseg).

  STAGE 2 - body parts (head / neck / torso / tail / leg)
    1. SEED (mandatory): convert PASCAL-Part horse part masks to YOLO-seg and
       pre-train a parts model -> it learns the notion of a "part" (side view).
    2. ADAPT (mandatory): self-train on the user's own horse crops - the seed
       model pseudo-labels them, confident predictions are kept, and training
       continues from the seed weights -> a parts model adapted to drone view.

Everything is a thin wrapper over ultralytics (already a dependency). Heavy
steps (SAM2 download, training) run on the user's machine/GPU.

Usage
-----
    python BehaveAI_segmentation.py <subcommand> [options]

Subcommands:
    sample-frames     extract frames from clips into a working folder
    autolabel-horse   auto_annotate sampled frames (detector + SAM2) -> YOLO-seg
    train-horse       train YOLO11-seg on the auto-labelled horse dataset
    pascal-to-yolo    convert PASCAL-Part horse parts -> YOLO-seg dataset (seed)
    train-parts       pre-train the parts model on the PASCAL-Part seed
    pseudolabel-parts run the parts model on drone horse-crops -> YOLO-seg labels
    finetune-parts    continue training the parts model on the pseudo-labels
"""

import os
import sys
import glob
import shutil
import random
import argparse
import configparser

import numpy as np
import cv2


PART_CLASSES = ["head", "neck", "torso", "tail", "leg"]
PART_INDEX = {name: i for i, name in enumerate(PART_CLASSES)}
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")
_IMG_EXTS = (".jpg", ".jpeg", ".png")


# ----------------------------------------------------------------------------
# Project helpers
# ----------------------------------------------------------------------------

def _resolve(project_dir, p):
    if not p:
        return p
    p = p.strip()
    return p if os.path.isabs(p) else os.path.join(project_dir, p)


def _project_paths(project_dir):
    ini = os.path.join(project_dir, "BehaveAI_settings.ini")
    clips_dir = os.path.join(project_dir, "clips")
    if os.path.isfile(ini):
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        clips_dir = _resolve(project_dir, cfg["DEFAULT"].get("clips_dir", clips_dir))
    seg_dir = os.path.join(project_dir, "model_segmentation")
    default_det = os.path.join(project_dir, "model_primary_motion", "train",
                               "weights", "best.pt")
    return clips_dir, seg_dir, default_det


def _scan_videos(clips_dir):
    out = []
    for root, _d, files in os.walk(clips_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in _VIDEO_EXTS:
                out.append(os.path.join(root, f))
    return out


# ----------------------------------------------------------------------------
# STAGE 1 - whole horse
# ----------------------------------------------------------------------------

def sample_frames(project_dir, n_frames=300, every=0, out_dir=None):
    """Extract frames from project clips into a flat folder for auto-labelling."""
    clips_dir, seg_dir, _ = _project_paths(project_dir)
    out_dir = out_dir or os.path.join(seg_dir, "horse", "frames")
    os.makedirs(out_dir, exist_ok=True)
    videos = _scan_videos(clips_dir)
    if not videos:
        sys.exit(f"No videos under {clips_dir}")
    per_video = max(1, n_frames // len(videos)) if not every else None
    saved = 0
    for v in videos:
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if every:
            idxs = list(range(0, total, every))
        else:
            idxs = sorted(random.sample(range(total), min(per_video, total))) if total else []
        stem = os.path.splitext(os.path.basename(v))[0]
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            cv2.imwrite(os.path.join(out_dir, f"{stem}_{fi:06d}.jpg"), img)
            saved += 1
        cap.release()
    print(f"Sampled {saved} frames -> {out_dir}")
    return out_dir


def autolabel_horse(project_dir, det_model=None, sam_model="sam2_b.pt",
                    frames_dir=None, device=""):
    """Auto-annotate sampled frames into YOLO-seg labels via detector + SAM2."""
    from ultralytics.data.annotator import auto_annotate
    clips_dir, seg_dir, default_det = _project_paths(project_dir)
    det_model = det_model or default_det
    frames_dir = frames_dir or os.path.join(seg_dir, "horse", "frames")
    if not os.path.isfile(det_model):
        sys.exit(f"Detector weights not found: {det_model} (train the primary model first)")
    if not glob.glob(os.path.join(frames_dir, "*")):
        sys.exit(f"No frames in {frames_dir}; run 'sample-frames' first.")
    labels_dir = os.path.join(seg_dir, "horse", "labels_auto")
    print(f"Auto-annotating {frames_dir} with det={det_model}, sam={sam_model} ...")
    auto_annotate(data=frames_dir, det_model=det_model, sam_model=sam_model,
                  device=device, output_dir=labels_dir)
    print(f"YOLO-seg labels -> {labels_dir}")
    return _assemble_yolo_dataset(frames_dir, labels_dir,
                                  os.path.join(seg_dir, "horse", "dataset"),
                                  names=["horse"])


def _assemble_yolo_dataset(images_src, labels_src, dataset_dir, names, val_ratio=0.2):
    """Lay out images/labels into train/val splits and write a data.yaml."""
    imgs = [f for f in glob.glob(os.path.join(images_src, "*"))
            if os.path.splitext(f)[1].lower() in _IMG_EXTS]
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio)) if imgs else 0
    for split, subset in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
        idir = os.path.join(dataset_dir, "images", split)
        ldir = os.path.join(dataset_dir, "labels", split)
        os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
        for img in subset:
            stem = os.path.splitext(os.path.basename(img))[0]
            lab = os.path.join(labels_src, stem + ".txt")
            if not os.path.isfile(lab):
                continue  # no detection on this frame -> skip
            shutil.copy(img, os.path.join(idir, os.path.basename(img)))
            shutil.copy(lab, os.path.join(ldir, stem + ".txt"))
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    _write_yaml(yaml_path, dataset_dir, names)
    print(f"Dataset ready -> {yaml_path}")
    return yaml_path


def _write_yaml(path, dataset_dir, names):
    with open(path, "w") as fh:
        fh.write(f"path: {dataset_dir}\n")
        fh.write("train: images/train\nval: images/val\n")
        fh.write(f"nc: {len(names)}\n")
        fh.write("names: [" + ", ".join(names) + "]\n")


def train_seg(data_yaml, out_dir, weights="yolo11n-seg.pt", epochs=100,
              imgsz=640, device="", name="train"):
    """Train/continue-train a YOLO-seg model on a dataset.yaml."""
    from ultralytics import YOLO
    if not os.path.isfile(data_yaml):
        sys.exit(f"data.yaml not found: {data_yaml}")
    model = YOLO(weights)
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, device=device,
                project=out_dir, name=name)
    best = os.path.join(out_dir, name, "weights", "best.pt")
    print(f"Trained -> {best}")
    return best


# ----------------------------------------------------------------------------
# STAGE 2 - body parts: PASCAL-Part seed
# ----------------------------------------------------------------------------

def mask_to_polygons(mask, img_w, img_h, min_area=30):
    """Binary mask -> list of normalised YOLO-seg polygons (flat x,y lists)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area or len(cnt) < 3:
            continue
        eps = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        flat = []
        for x, y in approx:
            flat.append(min(1.0, max(0.0, float(x) / img_w)))
            flat.append(min(1.0, max(0.0, float(y) / img_h)))
        polys.append(flat)
    return polys


def _part_to_class(part_name):
    """Map a PASCAL-Part horse part name to a reduced class index, or None."""
    p = str(part_name).lower()
    if p in ("torso",):
        return PART_INDEX["torso"]
    if p in ("neck",):
        return PART_INDEX["neck"]
    if p in ("tail",):
        return PART_INDEX["tail"]
    if p == "head" or p in ("leye", "reye", "lear", "rear", "muzzle", "lhorn", "rhorn"):
        return PART_INDEX["head"]
    if "leg" in p or p.endswith("ho"):  # lfuleg/lflleg/lfho ...
        return PART_INDEX["leg"]
    return None


def pascal_to_yolo(pascal_root, out_dir):
    """Convert PASCAL-Part horse part annotations into a YOLO-seg dataset.

    Expects the standard layout:
      <pascal_root>/Annotations_Part/*.mat   (part masks, from PASCAL-Part)
      <pascal_root>/JPEGImages/*.jpg         (VOC2010 images)
    """
    from scipy.io import loadmat
    anno_dir = os.path.join(pascal_root, "Annotations_Part")
    img_dir = os.path.join(pascal_root, "JPEGImages")
    if not os.path.isdir(anno_dir):
        sys.exit(f"Missing {anno_dir} (download PASCAL-Part Annotations_Part).")

    images_out = os.path.join(out_dir, "images", "all")
    labels_out = os.path.join(out_dir, "labels", "all")
    os.makedirs(images_out, exist_ok=True); os.makedirs(labels_out, exist_ok=True)

    n_imgs = 0
    for mat_path in glob.glob(os.path.join(anno_dir, "*.mat")):
        stem = os.path.splitext(os.path.basename(mat_path))[0]
        img_path = os.path.join(img_dir, stem + ".jpg")
        if not os.path.isfile(img_path):
            continue
        mat = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        anno = mat.get("anno")
        if anno is None:
            continue
        objects = _as_list(getattr(anno, "objects", []))
        lines = []
        H = W = None
        for obj in objects:
            if str(getattr(obj, "class", "")).lower() != "horse":
                continue
            parts = _as_list(getattr(obj, "parts", []))
            for part in parts:
                pmask = getattr(part, "mask", None)
                if pmask is None:
                    continue
                if H is None:
                    H, W = np.asarray(pmask).shape[:2]
                cls = _part_to_class(getattr(part, "part_name", ""))
                if cls is None:
                    continue
                for poly in mask_to_polygons(pmask, W, H):
                    lines.append(str(cls) + " " + " ".join(f"{v:.6f}" for v in poly))
        if not lines:
            continue
        shutil.copy(img_path, os.path.join(images_out, stem + ".jpg"))
        with open(os.path.join(labels_out, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        n_imgs += 1

    # Split into train/val and write data.yaml.
    yaml_path = _split_existing(out_dir, PART_CLASSES)
    print(f"PASCAL-Part horse parts: {n_imgs} images -> {yaml_path}")
    return yaml_path


def _split_existing(dataset_dir, names, val_ratio=0.2):
    """Move images/labels/all into train/val splits and write data.yaml."""
    images_all = os.path.join(dataset_dir, "images", "all")
    labels_all = os.path.join(dataset_dir, "labels", "all")
    imgs = [f for f in glob.glob(os.path.join(images_all, "*"))
            if os.path.splitext(f)[1].lower() in _IMG_EXTS]
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio)) if imgs else 0
    for split, subset in (("val", imgs[:n_val]), ("train", imgs[n_val:])):
        idir = os.path.join(dataset_dir, "images", split)
        ldir = os.path.join(dataset_dir, "labels", split)
        os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
        for img in subset:
            stem = os.path.splitext(os.path.basename(img))[0]
            lab = os.path.join(labels_all, stem + ".txt")
            shutil.move(img, os.path.join(idir, os.path.basename(img)))
            if os.path.isfile(lab):
                shutil.move(lab, os.path.join(ldir, stem + ".txt"))
    shutil.rmtree(images_all, ignore_errors=True)
    shutil.rmtree(labels_all, ignore_errors=True)
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    _write_yaml(yaml_path, dataset_dir, names)
    return yaml_path


def _as_list(x):
    """PASCAL structs squeeze single elements to scalars; normalise to a list."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return list(x.ravel())
    return [x]


# ----------------------------------------------------------------------------
# STAGE 2 - body parts: self-training on the user's drone crops
# ----------------------------------------------------------------------------

def pseudolabel_parts(crops_dir, parts_weights, out_dir, conf=0.45, device=""):
    """Run the seed parts model on horse crops; keep confident YOLO-seg labels.

    ``crops_dir`` should hold cropped horse images (e.g. model_reid/reid_crops
    produced by BehaveAI_reid_finetune.py, or any folder of horse crops).
    """
    from ultralytics import YOLO
    if not os.path.isfile(parts_weights):
        sys.exit(f"Seed parts weights not found: {parts_weights} (run train-parts first)")
    imgs = [f for f in glob.glob(os.path.join(crops_dir, "**", "*"), recursive=True)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS]
    if not imgs:
        sys.exit(f"No crop images under {crops_dir}")
    images_out = os.path.join(out_dir, "images", "all")
    labels_out = os.path.join(out_dir, "labels", "all")
    os.makedirs(images_out, exist_ok=True); os.makedirs(labels_out, exist_ok=True)

    model = YOLO(parts_weights)
    kept = 0
    for img_path in imgs:
        res = model.predict(img_path, conf=conf, verbose=False, device=device)[0]
        if res.masks is None or len(res.masks) == 0:
            continue
        h, w = res.orig_shape
        lines = []
        for cls_t, xy in zip(res.boxes.cls.tolist(), res.masks.xy):
            poly = []
            for x, y in xy:
                poly.append(min(1.0, max(0.0, float(x) / w)))
                poly.append(min(1.0, max(0.0, float(y) / h)))
            if len(poly) >= 6:
                lines.append(str(int(cls_t)) + " " + " ".join(f"{v:.6f}" for v in poly))
        if not lines:
            continue
        stem = f"pl_{kept:06d}"
        shutil.copy(img_path, os.path.join(images_out, stem + ".jpg"))
        with open(os.path.join(labels_out, stem + ".txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        kept += 1
    yaml_path = _split_existing(out_dir, PART_CLASSES)
    print(f"Pseudo-labelled {kept} crops -> {yaml_path}")
    return yaml_path


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample-frames"); p.add_argument("project_dir")
    p.add_argument("--n", type=int, default=300); p.add_argument("--every", type=int, default=0)

    p = sub.add_parser("autolabel-horse"); p.add_argument("project_dir")
    p.add_argument("--det-model", default=None); p.add_argument("--sam-model", default="sam2_b.pt")
    p.add_argument("--device", default="")

    p = sub.add_parser("train-horse"); p.add_argument("project_dir")
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--weights", default="yolo11n-seg.pt")
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="")

    p = sub.add_parser("pascal-to-yolo")
    p.add_argument("--pascal-root", required=True); p.add_argument("--out", required=True)

    p = sub.add_parser("train-parts")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=100); p.add_argument("--weights", default="yolo11n-seg.pt")
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="")

    p = sub.add_parser("pseudolabel-parts")
    p.add_argument("--crops-dir", required=True); p.add_argument("--parts-weights", required=True)
    p.add_argument("--out", required=True); p.add_argument("--conf", type=float, default=0.45)
    p.add_argument("--device", default="")

    p = sub.add_parser("finetune-parts")
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed-weights", required=True); p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640); p.add_argument("--device", default="")

    args = ap.parse_args(argv)

    if args.cmd == "sample-frames":
        sample_frames(args.project_dir, n_frames=args.n, every=args.every)
    elif args.cmd == "autolabel-horse":
        autolabel_horse(args.project_dir, det_model=args.det_model,
                        sam_model=args.sam_model, device=args.device)
    elif args.cmd == "train-horse":
        _, seg_dir, _ = _project_paths(args.project_dir)
        data_yaml = os.path.join(seg_dir, "horse", "dataset", "data.yaml")
        train_seg(data_yaml, os.path.join(seg_dir, "horse"), weights=args.weights,
                  epochs=args.epochs, imgsz=args.imgsz, device=args.device)
    elif args.cmd == "pascal-to-yolo":
        pascal_to_yolo(args.pascal_root, args.out)
    elif args.cmd == "train-parts":
        train_seg(args.data, args.out, weights=args.weights, epochs=args.epochs,
                  imgsz=args.imgsz, device=args.device, name="parts_seed")
    elif args.cmd == "pseudolabel-parts":
        pseudolabel_parts(args.crops_dir, args.parts_weights, args.out,
                          conf=args.conf, device=args.device)
    elif args.cmd == "finetune-parts":
        # Continue training from the PASCAL-Part seed weights on drone pseudo-labels.
        train_seg(args.data, args.out, weights=args.seed_weights, epochs=args.epochs,
                  imgsz=args.imgsz, device=args.device, name="parts_finetuned")


if __name__ == "__main__":
    main()
