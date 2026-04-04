#!/usr/bin/env python3
"""
BehaveAI Data Augmentation

Applies augmentation to all existing annotations in the project.
Each triggered augmentation parameter produces one independent copy of the original.
The original annotation is never modified.

Usage: called programmatically from the GUI or directly via CLI.
"""

import os
import sys
import configparser
import random
import shutil
import cv2
import numpy as np
from pathlib import Path

try:
    from PIL import Image as PilImage, ImageEnhance, ImageFilter
except ImportError:
    PilImage = None


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_augmentation_config(config_path):
    """
    Read augmentation parameters from BehaveAI_settings.ini.
    Returns the AUGMENTATION_CONFIG dict used by the augmentation functions,
    or None if global_probability is 0 (augmentation disabled).
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)
    d = config['DEFAULT']

    global_prob = float(d.get('aug_global_probability', '0'))

    # If global probability is 0, augmentation is disabled — return None immediately
    if global_prob == 0:
        return None

    aug_config = {
        'global_probability': global_prob,
        'params': {
            'brightness': {
                'range': [float(x) for x in d.get('aug_brightness_range', '0.8,1.2').split(',')],
                'type': 'float',
                'probability': float(d.get('aug_brightness_probability', '0')),
            },
            'contrast': {
                'range': [float(x) for x in d.get('aug_contrast_range', '0.8,1.2').split(',')],
                'type': 'float',
                'probability': float(d.get('aug_contrast_probability', '0')),
            },
            'saturation': {
                'range': [float(x) for x in d.get('aug_saturation_range', '0.8,1.2').split(',')],
                'type': 'float',
                'probability': float(d.get('aug_saturation_probability', '0')),
            },
            'hue': {
                'range': [float(x) for x in d.get('aug_hue_range', '-15,15').split(',')],
                'type': 'int',
                'probability': float(d.get('aug_hue_probability', '0')),
            },
            'sharpness': {
                'range': [float(x) for x in d.get('aug_sharpness_range', '0.8,1.5').split(',')],
                'type': 'float',
                'probability': float(d.get('aug_sharpness_probability', '0')),
            },
            'blur': {
                'range': [float(x) for x in d.get('aug_blur_range', '1,3').split(',')],
                'type': 'int',
                'probability': float(d.get('aug_blur_probability', '0')),
            },
            'noise': {
                'range': [float(x) for x in d.get('aug_noise_range', '0,25').split(',')],
                'type': 'int',
                'probability': float(d.get('aug_noise_probability', '0')),
            },
            'shear': {
                'range': [float(x) for x in d.get('aug_shear_range', '-0.1,0.1').split(',')],
                'type': 'float',
                'probability': float(d.get('aug_shear_probability', '0')),
            },
            'flip_h': {
                'options': [x.strip().lower() == 'true'
                            for x in d.get('aug_flip_h_options', 'True,False').split(',')],
                'probability': float(d.get('aug_flip_h_probability', '0')),
            },
            'flip_v': {
                'options': [x.strip().lower() == 'true'
                            for x in d.get('aug_flip_v_options', 'True,False').split(',')],
                'probability': float(d.get('aug_flip_v_probability', '0')),
            },
            'temperature': {
                'range': [float(x) for x in d.get('aug_temperature_range', '0,10').split(',')],
                'type': 'int',
                'probability': float(d.get('aug_temperature_probability', '0')),
            },
        }
    }
    return aug_config


# ---------------------------------------------------------------------------
# Parameter sampling — one dict per triggered parameter
# ---------------------------------------------------------------------------

def sample_augmentation_list(aug_config):
    """
    Randomly decide which augmentation parameters apply to one image.

    Returns a list of single-key dicts — one per triggered parameter.
    Each dict contains exactly one key-value pair, e.g. {'brightness': 1.3}.

    Returns an empty list if the image is not selected for augmentation
    (global probability check) or if no individual parameter fires.

    This guarantees that each triggered parameter produces one independent
    copy of the original annotation.
    """
    # Global gate: with probability (1 - global_probability) skip entirely
    if random.random() >= aug_config['global_probability']:
        return []

    triggered = []

    for param_name, param_cfg in aug_config['params'].items():
        # Each parameter has its own independent probability
        if random.random() >= param_cfg['probability']:
            continue

        # Sample a value for this parameter
        if 'options' in param_cfg:
            value = random.choice(param_cfg['options'])
        elif param_cfg['type'] == 'float':
            value = round(random.uniform(param_cfg['range'][0], param_cfg['range'][1]), 4)
        elif param_cfg['type'] == 'int':
            value = random.randint(int(param_cfg['range'][0]), int(param_cfg['range'][1]))
            if param_name == 'blur' and value % 2 == 0:
                value = max(1, value - 1)
        else:
            continue

        # One dict per parameter — this is the key design rule
        triggered.append({param_name: value})

    return triggered


# ---------------------------------------------------------------------------
# Image transformation — one parameter at a time
# ---------------------------------------------------------------------------

def apply_single_augmentation(bgr_image, param_name, value):
    """
    Apply exactly one augmentation parameter to a BGR numpy image (OpenCV format).
    Returns the augmented BGR numpy image.
    The input image is not modified.
    """
    if PilImage is None:
        raise ImportError("Pillow is required for augmentation: pip install Pillow")

    # Convert BGR (OpenCV) -> RGB (PIL)
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    img = PilImage.fromarray(rgb)

    if param_name == 'brightness':
        img = ImageEnhance.Brightness(img).enhance(value)

    elif param_name == 'contrast':
        img = ImageEnhance.Contrast(img).enhance(value)

    elif param_name == 'saturation':
        img = ImageEnhance.Color(img).enhance(value)

    elif param_name == 'sharpness':
        img = ImageEnhance.Sharpness(img).enhance(value)

    elif param_name == 'blur':
        k = max(1, int(value))
        if k % 2 == 0:
            k = max(1, k - 1)
        img = img.filter(ImageFilter.GaussianBlur(radius=k))

    elif param_name == 'noise':
        arr = np.array(img, dtype=np.int16)
        noise = np.random.randint(-value, value + 1, arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = PilImage.fromarray(arr)

    elif param_name == 'hue':
        # Shift hue channel in HSV space
        hsv = img.convert('HSV')
        arr = np.array(hsv, dtype=np.int16)
        arr[:, :, 0] = (arr[:, :, 0] + int(value)) % 256
        img = PilImage.fromarray(arr.astype(np.uint8), 'HSV').convert('RGB')

    elif param_name == 'temperature':
        # Warm/cool shift: add to red, subtract from blue
        arr = np.array(img, dtype=np.int16)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + int(value), 0, 255)  # red
        arr[:, :, 2] = np.clip(arr[:, :, 2] - int(value), 0, 255)  # blue
        img = PilImage.fromarray(arr.astype(np.uint8))

    elif param_name == 'flip_h':
        if value:  # value is True/False
            img = img.transpose(PilImage.FLIP_LEFT_RIGHT)

    elif param_name == 'flip_v':
        if value:
            img = img.transpose(PilImage.FLIP_TOP_BOTTOM)

    elif param_name == 'shear':
        # Affine shear along x axis
        w, h = img.size
        shear_x = float(value)
        # Affine transform matrix for horizontal shear
        transform_matrix = (1, shear_x, -shear_x * h / 2,
                             0, 1, 0)
        img = img.transform((w, h), PilImage.AFFINE, transform_matrix,
                             resample=PilImage.BILINEAR)

    # Convert back RGB -> BGR
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Label transformation for geometric augmentations
# ---------------------------------------------------------------------------

def transform_labels(label_path, param_name, value):
    """
    Read a YOLO label file and return transformed label lines as a string.
    Only flip_h and flip_v require coordinate adjustments.
    All other augmentations are colour/filter changes — labels are unchanged.

    YOLO format: <class> <xc> <yc> <w> <h>  (all normalised 0..1)
    flip_h: xc -> 1 - xc
    flip_v: yc -> 1 - yc
    """
    if not os.path.exists(label_path):
        return ''

    with open(label_path, 'r') as f:
        lines = f.readlines()

    if param_name not in ('flip_h', 'flip_v') or not value:
        # No geometric change — return labels unchanged
        return ''.join(lines)

    out_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            out_lines.append(line)
            continue
        cls = parts[0]
        xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        if param_name == 'flip_h':
            xc = 1.0 - xc
        elif param_name == 'flip_v':
            yc = 1.0 - yc

        out_lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    return ''.join(out_lines)


# ---------------------------------------------------------------------------
# Core augmentation loop
# ---------------------------------------------------------------------------

def apply_augmentation_to_all_annotations(config_path, progress_callback=None):
    """
    Main entry point called by the GUI button.

    Scans all annotation images in the project, applies augmentation, and writes
    augmented copies alongside the originals.

    Rules:
    - Original files are never modified.
    - Files whose basename already contains '_aug_' are skipped (no re-augmentation).
    - Each triggered parameter produces one independent copy named
      <basename>_aug_<param_name>.<ext>.
    - If a copy with that name already exists it is overwritten (idempotent).

    Args:
        config_path (str): Path to BehaveAI_settings.ini.
        progress_callback (callable): Optional function(current, total, message)
            called periodically so the GUI can update a progress bar.

    Returns:
        (int, int): (copies_created, originals_processed)
    """
    aug_config = load_augmentation_config(config_path)
    if aug_config is None:
        print("Augmentation disabled (aug_global_probability = 0). Nothing to do.")
        return 0, 0

    project_dir = os.path.dirname(os.path.abspath(config_path))

    # Directories that hold annotation images (train and val for both streams)
    image_dirs = [
        os.path.join(project_dir, 'annot_static',  'images', 'train'),
        os.path.join(project_dir, 'annot_static',  'images', 'val'),
        os.path.join(project_dir, 'annot_motion',  'images', 'train'),
        os.path.join(project_dir, 'annot_motion',  'images', 'val'),
    ]

    # Collect all original images (skip augmented copies from previous runs)
    image_ext = ('.jpg', '.jpeg', '.png')
    originals = []
    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(image_ext):
                continue
            basename = os.path.splitext(fname)[0]
            # Skip files that are themselves augmented copies
            if '_aug_' in basename:
                continue
            originals.append(os.path.join(img_dir, fname))

    total = len(originals)
    copies_created = 0

    print(f"Augmentation: {total} original annotation images found.")

    for i, img_path in enumerate(originals):
        if progress_callback:
            progress_callback(i, total, os.path.basename(img_path))

        img_dir    = os.path.dirname(img_path)
        fname      = os.path.basename(img_path)
        basename   = os.path.splitext(fname)[0]
        ext        = os.path.splitext(fname)[1]

        # Derive the corresponding label directory
        # image path:  .../annot_static/images/train/foo.jpg
        # label path:  .../annot_static/labels/train/foo.txt
        label_dir  = img_dir.replace('images', 'labels')
        label_path = os.path.join(label_dir, basename + '.txt')

        # Load the original image
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  Warning: could not read {img_path}, skipping.")
            continue

        # Sample which augmentations apply — returns list of single-key dicts
        augmentation_list = sample_augmentation_list(aug_config)

        if not augmentation_list:
            continue  # this image was not selected for augmentation this run

        for aug_dict in augmentation_list:
            # Each dict has exactly one key
            param_name, value = next(iter(aug_dict.items()))

            # Build output paths
            aug_basename   = f"{basename}_aug_{param_name}"
            aug_img_path   = os.path.join(img_dir,   aug_basename + ext)
            aug_label_path = os.path.join(label_dir, aug_basename + '.txt')

            # Apply image transformation
            try:
                aug_bgr = apply_single_augmentation(bgr, param_name, value)
            except Exception as e:
                print(f"  Warning: augmentation '{param_name}' failed on {fname}: {e}")
                continue

            # Write augmented image
            cv2.imwrite(aug_img_path, aug_bgr)

            # Write (potentially transformed) labels
            transformed_labels = transform_labels(label_path, param_name, value)
            with open(aug_label_path, 'w') as lf:
                lf.write(transformed_labels)

            copies_created += 1

    if progress_callback:
        progress_callback(total, total, "Done")

    print(f"Augmentation complete: {copies_created} copies created from {total} originals.")
    return copies_created, total


# ---------------------------------------------------------------------------
# CLI entry point (optional — for testing outside the GUI)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python BehaveAI_augmentation.py <project_dir | BehaveAI_settings.ini>")
        sys.exit(1)

    arg = os.path.abspath(sys.argv[1])
    ini = os.path.join(arg, 'BehaveAI_settings.ini') if os.path.isdir(arg) else arg

    if not os.path.exists(ini):
        print(f"Settings file not found: {ini}")
        sys.exit(1)

    apply_augmentation_to_all_annotations(ini)
