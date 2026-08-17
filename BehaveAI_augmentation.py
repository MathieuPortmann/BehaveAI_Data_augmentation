#!/usr/bin/env python3
"""
BehaveAI Data Augmentation

Applies augmentation to all existing annotations in the project.
Each triggered augmentation parameter produces one independent copy of the original.
The original annotation is never modified.

New features:
  - aug_target_classes: comma-separated list of class names to augment.
    Only images containing at least one annotation of a target class are augmented.
    Leave empty (or omit) to augment all classes.

  - Multi-segment range syntax using '|' as separator:
      Single range   : 0.8,1.2
      Two ranges     : 0.5,0.8 | 1.2,1.6
      Discrete value : 0.5
      Mix            : 0.5,0.8 | 1.0 | 1.2,1.6
    When multiple segments are provided each segment produces ONE independent
    augmented copy (same behaviour as having multiple parameters, but for one
    parameter). A single segment keeps the original random-sample behaviour.

Usage: called programmatically from the GUI or directly via CLI.
"""

import os
import sys
import configparser
import random
import cv2
import numpy as np

try:
    from PIL import Image as PilImage, ImageEnhance, ImageFilter
except ImportError:
    PilImage = None


# ---------------------------------------------------------------------------
# Progress bar helper — prints to stdout so the BehaveAI launcher picks it up
# ---------------------------------------------------------------------------

def _print_progress(current, total, label='Augmenting'):
    """
    Print a tqdm-style progress line that the BehaveAI launcher recognises
    and overwrites in-place (matches the is_progress_line regex: 'XX% |').
    Uses only ASCII characters to stay compatible with Windows cp1252 encoding.
    """
    if total <= 0:
        return
    pct = int(100 * current / total)
    bar_width = 30
    filled = int(bar_width * current / total)
    bar = '#' * filled + '-' * (bar_width - filled)
    line = f"{pct}% |{bar}| {current}/{total}  {label}"
    print(line, end='\r', flush=True)
    if current >= total:
        print(flush=True)


# ---------------------------------------------------------------------------
# Multi-segment range parser  (NEW)
# ---------------------------------------------------------------------------

def _parse_segments(raw_str):
    """
    Parse a range/discrete string that may contain multiple segments separated
    by '|'.

    Each segment is either:
      - A pair  'lo,hi'  -> interpreted as a continuous range [lo, hi]
      - A single value 'v' -> interpreted as the discrete value v

    Returns a list of segments, where each segment is one of:
      ('range',    lo, hi)   — sample uniformly from [lo, hi]
      ('discrete', value)    — use this exact value

    Examples
    --------
    '0.8,1.2'          -> [('range', 0.8, 1.2)]
    '0.5,0.8 | 1.2,1.6' -> [('range', 0.5, 0.8), ('range', 1.2, 1.6)]
    '0.5 | 1.0 | 1.5'  -> [('discrete', 0.5), ('discrete', 1.0), ('discrete', 1.5)]
    '0.5,0.8 | 1.0 | 1.2,1.6' -> [('range', 0.5, 0.8), ('discrete', 1.0), ('range', 1.2, 1.6)]
    """
    segments = []
    for part in raw_str.split('|'):
        part = part.strip()
        if not part:
            continue
        if ',' in part:
            lo_str, hi_str = part.split(',', 1)
            segments.append(('range', float(lo_str.strip()), float(hi_str.strip())))
        else:
            segments.append(('discrete', float(part)))
    if not segments:
        # Fallback: treat whole string as a single range/discrete
        segments = _parse_segments('0,0')
    return segments


def _sample_segment(segment, value_type):
    """
    Sample one value from a parsed segment.

    Parameters
    ----------
    segment    : tuple returned by _parse_segments
    value_type : 'float' or 'int'

    Returns a float or int depending on value_type.
    """
    kind = segment[0]
    if kind == 'discrete':
        v = segment[1]
        return int(round(v)) if value_type == 'int' else float(v)
    else:  # 'range'
        lo, hi = segment[1], segment[2]
        if value_type == 'float':
            return round(random.uniform(lo, hi), 4)
        else:
            return random.randint(int(lo), int(hi))


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_augmentation_config(config_path):
    """
    Read augmentation parameters from BehaveAI_settings.ini.
    Returns the AUGMENTATION_CONFIG dict used by the augmentation functions,
    or None if global_probability is 0 (augmentation disabled).

    New INI keys
    ------------
    aug_target_classes : comma-separated class names to augment (empty = all)
    motion_disable_color_aug : keep colour transforms off the motion stream
    All range keys now accept the multi-segment syntax described in _parse_segments.
    """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)
    d = config['DEFAULT']

    global_prob = float(d.get('aug_global_probability', '0'))
    if global_prob == 0:
        return None

    # Same key the online (Ultralytics) HSV augmentation honours in
    # BehaveAI_classify_track.py, so the two paths cannot disagree about
    # whether the false-colour motion encoding may be recoloured.
    motion_disable_color_aug = str(
        d.get('motion_disable_color_aug', 'true')).lower() == 'true'

    # --- target classes filter (NEW) ---
    raw_target = d.get('aug_target_classes', '').strip()
    target_classes = [c.strip() for c in raw_target.split(',') if c.strip()] \
        if raw_target else []

    # Helper: parse a range key into its list of segments
    def _segs(key, default):
        return _parse_segments(d.get(key, default))

    aug_config = {
        'global_probability': global_prob,
        'target_classes': target_classes,           # NEW
        'motion_disable_color_aug': motion_disable_color_aug,
        'params': {
            'brightness': {
                'segments': _segs('aug_brightness_range', '0.8,1.2'),
                'type': 'float',
                'probability': float(d.get('aug_brightness_probability', '0')),
            },
            'contrast': {
                'segments': _segs('aug_contrast_range', '0.8,1.2'),
                'type': 'float',
                'probability': float(d.get('aug_contrast_probability', '0')),
            },
            'saturation': {
                'segments': _segs('aug_saturation_range', '0.8,1.2'),
                'type': 'float',
                'probability': float(d.get('aug_saturation_probability', '0')),
            },
            'hue': {
                'segments': _segs('aug_hue_range', '-15,15'),
                'type': 'int',
                'probability': float(d.get('aug_hue_probability', '0')),
            },
            'sharpness': {
                'segments': _segs('aug_sharpness_range', '0.8,1.5'),
                'type': 'float',
                'probability': float(d.get('aug_sharpness_probability', '0')),
            },
            'blur': {
                'segments': _segs('aug_blur_range', '1,3'),
                'type': 'int',
                'probability': float(d.get('aug_blur_probability', '0')),
            },
            'noise': {
                'segments': _segs('aug_noise_range', '0,25'),
                'type': 'int',
                'probability': float(d.get('aug_noise_probability', '0')),
            },
            'shear': {
                'segments': _segs('aug_shear_range', '-0.1,0.1'),
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
                'segments': _segs('aug_temperature_range', '0,10'),
                'type': 'int',
                'probability': float(d.get('aug_temperature_probability', '0')),
            },
        }
    }
    return aug_config


# ---------------------------------------------------------------------------
# Class-filter helper  (NEW)
# ---------------------------------------------------------------------------

def _label_contains_target_class(label_path, target_class_indices):
    """
    Return True if the YOLO label file at label_path contains at least one
    annotation whose class index is in target_class_indices.

    If target_class_indices is empty (no filter configured) always returns True.
    If the label file does not exist OR is empty (save_empty_frames = true
    produces empty label files for background frames), returns True so the
    image is included — empty frames are valid training data.
    """
    if not target_class_indices:
        return True                  # no filter — augment everything
    if not os.path.exists(label_path):
        return True                  # missing label = empty frame, keep it

    with open(label_path, 'r') as f:
        lines = [l for l in f if l.strip()]

    if not lines:
        return True                  # empty label file = background frame, keep it

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            if int(parts[0]) in target_class_indices:
                return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Parameter sampling — one dict per (parameter, segment) pair  (UPDATED)
# ---------------------------------------------------------------------------

# One-shot flag so the shear refusal is printed once, not once per image.
_SHEAR_WARNED = []


def sample_augmentation_list(aug_config):
    """
    Decide which augmentation parameters apply to one image and produce the
    list of (param_name, value, segment_index) tuples to apply.

    Multi-segment behaviour
    -----------------------
    If a parameter has N segments and fires (probability check passes):
      - Single segment  : one copy, value sampled from that segment  (original behaviour)
      - Multiple segments: N copies, one per segment, each with its own sampled value

    The returned list contains one dict per copy to be written:
      {'param_name': value, '_seg_idx': i}
    The '_seg_idx' key is used to build a unique output filename when N > 1.

    Returns an empty list if the global gate rejects this image or no
    individual parameter fires.
    """
    if random.random() >= aug_config['global_probability']:
        return []

    triggered = []

    for param_name, param_cfg in aug_config['params'].items():
        # Shear moves the pixels but transform_labels() only rewrites the box
        # coordinates for flip_h/flip_v, so a sheared copy would carry its
        # ORIGINAL boxes -- silently mislabelled training data. Refuse it here
        # rather than produce it: a missing augmentation costs far less than a
        # corrupt one, and the failure would otherwise be invisible.
        if param_name == 'shear' and param_cfg['probability'] > 0:
            if not _SHEAR_WARNED:
                print("Augmentation: ignoring 'shear' -- the label file is not "
                      "transformed for it, so the copies would be mislabelled. "
                      "Set aug_shear_probability = 0 to silence this.")
                _SHEAR_WARNED.append(True)
            continue
        if random.random() >= param_cfg['probability']:
            continue

        # --- flip_h / flip_v keep their existing 'options' logic ---
        if 'options' in param_cfg:
            value = random.choice(param_cfg['options'])
            triggered.append({param_name: value, '_seg_idx': 0})
            continue

        # --- range/discrete parameters ---
        segments = param_cfg.get('segments', [])
        vtype    = param_cfg.get('type', 'float')

        if len(segments) <= 1:
            # Single segment — original behaviour: one copy, random sample
            seg   = segments[0] if segments else ('range', 0.0, 1.0)
            value = _sample_segment(seg, vtype)
            if param_name == 'blur' and vtype == 'int' and value % 2 == 0:
                value = max(1, value - 1)
            triggered.append({param_name: value, '_seg_idx': 0})
        else:
            # Multiple segments — one copy per segment
            for seg_idx, seg in enumerate(segments):
                value = _sample_segment(seg, vtype)
                if param_name == 'blur' and vtype == 'int' and value % 2 == 0:
                    value = max(1, value - 1)
                triggered.append({param_name: value, '_seg_idx': seg_idx})

    return triggered


# ---------------------------------------------------------------------------
# Image transformation — one parameter at a time  (unchanged)
# ---------------------------------------------------------------------------

def apply_single_augmentation(bgr_image, param_name, value):
    """
    Apply exactly one augmentation parameter to a BGR numpy image (OpenCV format).
    Returns the augmented BGR numpy image.
    The input image is not modified.
    """
    if PilImage is None:
        raise ImportError("Pillow is required for augmentation: pip install Pillow")

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
        # Shift hue channel in HSV space using OpenCV (avoids deprecated Pillow HSV mode)
        bgr_tmp  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        hsv_tmp  = cv2.cvtColor(bgr_tmp, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv_tmp[:, :, 0] = (hsv_tmp[:, :, 0] + int(value)) % 180
        hsv_tmp  = np.clip(hsv_tmp, 0, 255).astype(np.uint8)
        rgb_tmp  = cv2.cvtColor(cv2.cvtColor(hsv_tmp, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
        img = PilImage.fromarray(rgb_tmp)

    elif param_name == 'temperature':
        arr = np.array(img, dtype=np.int16)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + int(value), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - int(value), 0, 255)
        img = PilImage.fromarray(arr.astype(np.uint8))

    elif param_name == 'flip_h':
        if value:
            img = img.transpose(PilImage.FLIP_LEFT_RIGHT)

    elif param_name == 'flip_v':
        if value:
            img = img.transpose(PilImage.FLIP_TOP_BOTTOM)

    elif param_name == 'shear':
        w, h = img.size
        shear_x = float(value)
        transform_matrix = (1, shear_x, -shear_x * h / 2, 0, 1, 0)
        img = img.transform((w, h), PilImage.AFFINE, transform_matrix,
                             resample=PilImage.BILINEAR)

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Label transformation for geometric augmentations  (unchanged)
# ---------------------------------------------------------------------------

def transform_labels(label_path, param_name, value):
    """
    Read a YOLO label file and return transformed label lines as a string.
    Only flip_h and flip_v require coordinate adjustments.
    """
    if not os.path.exists(label_path):
        return ''

    with open(label_path, 'r') as f:
        lines = f.readlines()

    if param_name not in ('flip_h', 'flip_v') or not value:
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

# Parameters EXCLUDED from motion images (colour augmentations that would
# corrupt the false-colour motion encoding) when motion_disable_color_aug is on.
# Everything NOT in this set is allowed on motion images.
MOTION_EXCLUDED_PARAMS = {'brightness', 'contrast', 'saturation', 'hue', 'temperature'}

# For convenience: the full set of allowed params on motion images
MOTION_ALLOWED_PARAMS = {'blur', 'noise', 'sharpness', 'shear', 'flip_h', 'flip_v'}


def apply_augmentation_to_all_annotations(config_path, progress_callback=None):
    """
    Main entry point called by the GUI button or CLI.

    For each original annotation image:
      1. Skip if no label file contains a target class (when aug_target_classes
         is configured).
      2. Sample the augmentation list; multi-segment parameters produce one
         copy per segment.
      3. Write each copy as <basename>_aug_<param>[_<seg_idx>].<ext>.

    Returns (copies_created, originals_processed).
    """
    aug_config = load_augmentation_config(config_path)
    if aug_config is None:
        print("Augmentation disabled (aug_global_probability = 0). Nothing to do.")
        return 0, 0

    project_dir = os.path.dirname(os.path.abspath(config_path))

    # --- Build target-class index set (NEW) ---
    # We need the YOLO integer indices that correspond to the configured class names.
    # The class names are read from the annotation YAML files (static + motion).
    # If target_classes is empty, both sets stay empty (= no filter, augment all).
    # Resolved PER STREAM. The two streams number their classes independently
    # (static Stand..Excrete = 0..8, motion Walk..Balk = 0..14), so a single
    # merged set tests a static label file against motion-derived indices: with
    # 'Trot' (motion 1) and 'Roll' (motion 3) selected, static indices 1 and 3 --
    # Recumbent and Graze -- matched too. Graze is in most frames, so the filter
    # silently degraded to "augment everything" exactly when it was asked to
    # target the rare classes.
    target_idx_by_stream = {'static': set(), 'motion': set()}
    target_classes = aug_config.get('target_classes', [])
    if target_classes:
        import yaml
        for stream, yaml_name in (('static', 'static_annotations.yaml'),
                                  ('motion', 'motion_annotations.yaml')):
            yaml_path = os.path.join(project_dir, yaml_name)
            if not os.path.exists(yaml_path):
                continue
            try:
                with open(yaml_path, 'r') as yf:
                    ydata = yaml.safe_load(yf)
                names = ydata.get('names', [])
                if isinstance(names, dict):          # YOLO also allows {idx: name}
                    names = [names[k] for k in sorted(names)]
                for idx, name in enumerate(names):
                    if name in target_classes:
                        target_idx_by_stream[stream].add(idx)
            except Exception as e:
                print(f"  Warning: could not read {yaml_path}: {e}")
        if any(target_idx_by_stream.values()):
            print(f"Class filter active — augmenting only classes: {target_classes}")
            for stream in ('static', 'motion'):
                print(f"    {stream}: indices {sorted(target_idx_by_stream[stream]) or '(none)'}")
        else:
            print(f"  Warning: aug_target_classes={target_classes} but none were found "
                  f"in YAML files. All classes will be augmented.")

    # Annotation image directories -- TRAIN ONLY.
    # Copies are written next to their original, so augmenting val used to put
    # synthetic images in the set that decides early stopping and reports
    # val_loss/mAP. That measures the model on data it will never meet and, worse,
    # changes the stopping decision itself. The validation set stays real.
    image_dirs = [
        os.path.join(project_dir, 'annot_static',  'images', 'train'),
        os.path.join(project_dir, 'annot_motion',  'images', 'train'),
    ]

    image_ext = ('.jpg', '.jpeg', '.png')
    originals = []
    for img_dir in image_dirs:
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            if not fname.lower().endswith(image_ext):
                continue
            basename = os.path.splitext(fname)[0]
            if '_aug_' in basename:
                continue
            originals.append(os.path.join(img_dir, fname))

    total        = len(originals)
    copies_created = 0
    skipped_class  = 0

    print(f"Augmentation: {total} original annotation images found.")
    _print_progress(0, total, label='Augmenting')

    for i, img_path in enumerate(originals):

        _print_progress(i + 1, total, label=os.path.basename(img_path))

        if progress_callback:
            progress_callback(i, total, os.path.basename(img_path))

        img_dir  = os.path.dirname(img_path)
        fname    = os.path.basename(img_path)
        basename = os.path.splitext(fname)[0]
        ext      = os.path.splitext(fname)[1]

        label_dir  = img_dir.replace('images', 'labels')
        label_path = os.path.join(label_dir, basename + '.txt')

        is_motion = 'annot_motion' in img_path.replace('\\', '/')

        # --- Class filter: against THIS stream's index space (see above) ---
        if not _label_contains_target_class(
                label_path, target_idx_by_stream['motion' if is_motion else 'static']):
            skipped_class += 1
            continue

        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  Warning: could not read {img_path}, skipping.")
            continue

        augmentation_list = sample_augmentation_list(aug_config)

        if is_motion and aug_config.get('motion_disable_color_aug', True):
            # Colour augmentations are excluded from motion images because the
            # false-colour encoding carries motion information — altering
            # hue/brightness/etc. would corrupt that signal. Gated on the same
            # setting as the online HSV augmentation, so turning it off in the
            # settings really does re-enable colour transforms here.
            augmentation_list = [
                aug for aug in augmentation_list
                if next(k for k in aug if k != '_seg_idx') not in MOTION_EXCLUDED_PARAMS
            ]

        if not augmentation_list:
            continue

        for aug_dict in augmentation_list:
            # Extract param name (the only key that is not '_seg_idx')
            param_name = next(k for k in aug_dict if k != '_seg_idx')
            value      = aug_dict[param_name]
            seg_idx    = aug_dict.get('_seg_idx', 0)

            # Build unique output filename:
            # single-segment  -> <basename>_aug_<param>.<ext>   (unchanged)
            # multi-segment   -> <basename>_aug_<param>_<idx>.<ext>
            num_segs = len(aug_config['params'][param_name].get('segments', [None]))
            if num_segs > 1:
                aug_basename = f"{basename}_aug_{param_name}_{seg_idx}"
            else:
                aug_basename = f"{basename}_aug_{param_name}"

            aug_img_path   = os.path.join(img_dir,   aug_basename + ext)
            aug_label_path = os.path.join(label_dir, aug_basename + '.txt')

            try:
                aug_bgr = apply_single_augmentation(bgr, param_name, value)
            except Exception as e:
                print(f"  Warning: augmentation '{param_name}' failed on {fname}: {e}")
                continue

            cv2.imwrite(aug_img_path, aug_bgr)

            transformed_labels = transform_labels(label_path, param_name, value)
            with open(aug_label_path, 'w') as lf:
                lf.write(transformed_labels)

            copies_created += 1

    if progress_callback:
        progress_callback(total, total, "Done")

    if skipped_class:
        print(f"  Skipped {skipped_class} images (no target class found).")
    print(f"Augmentation complete: {copies_created} copies created from {total} originals.")
    return copies_created, total


# ---------------------------------------------------------------------------
# CLI entry point
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
