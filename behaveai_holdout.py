# behaveai_holdout.py
# Deterministic whole-video train/holdout assignment shared by the annotation
# tool (per-video YOLO train/val split), the crop classifiers (secondary,
# species, age -- see build_classification_split) and the complex-behaviour
# model (permanent held-out evaluation set).
#
# A video's status is derived purely from a hash of its filename stem, so it
# never needs a stored list: an existing video's status never changes, and a
# new video is classified automatically the first time it is seen.
#
# Every model in the pipeline splits on whole videos, never on individual
# frames or crops: frames a few hundredths of a second apart are near
# duplicates, so a per-frame (or per-crop) split validates the model on images
# it has effectively already seen.

import hashlib
import os
import shutil

_SALT = "behaveai-holdout-v1"


def is_holdout_video(stem, fraction, salt=_SALT):
    """Return True if `stem` falls in the holdout bucket for `fraction`.

    Deterministic and stable: same stem + fraction + salt always returns the
    same result. Not cryptographic — just a stable, roughly-uniform partition.
    """
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    h = hashlib.sha256(f"{salt}:{stem}".encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def holdout_status(stem, fraction, salt=_SALT):
    return "holdout" if is_holdout_video(stem, fraction, salt) else "train"


def split_groups(groups, fraction, salt=_SALT):
    """Partition an iterable of video stems into (train_stems, holdout_stems),
    each de-duplicated and sorted."""
    train, holdout = set(), set()
    for stem in groups:
        (holdout if is_holdout_video(stem, fraction, salt) else train).add(stem)
    return sorted(train), sorted(holdout)


def video_label_for_annotation(basename):
    """Recover the source video stem from an annotation basename, which is
    always <video_label>_<frame_number>[_aug_<param>[_<idx>]] (see
    BehaveAI_annotation.py:save_annotation and BehaveAI_augmentation.py)."""
    name = basename.split('_aug_', 1)[0]
    stem, sep, tail = name.rpartition('_')
    return stem if sep and tail.isdigit() else name


def video_label_for_crop(filename):
    """Recover the source video stem from a crop filename, which is
    <video_label>_<frame_number>_<x1>_<y1>.<ext> (see
    BehaveAI_annotation.py:save_annotation and the _parse_crop_filename helper
    in BehaveAI_inspect_dataset.py). Falls back to the frame parser when the
    name does not carry the three trailing numeric fields.

    A video stem that itself ends in '_<digits>' groups several videos under one
    label. That is safe: it only makes the grouping coarser, and a single video
    can still never straddle the train/val boundary."""
    name = os.path.splitext(filename)[0].split('_aug_', 1)[0]
    parts = name.split('_')
    if len(parts) >= 4 and all(p.isdigit() for p in parts[-3:]):
        return '_'.join(parts[:-3])
    return video_label_for_annotation(name)


# ---------------------------------------------------------------------------
# Classification (crop) datasets
# ---------------------------------------------------------------------------

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
_SPLIT_MARKER = '.holdout_split'


def _pool_contents(pool_dir):
    """{class folder name: sorted list of image filenames} for a crop pool."""
    contents = {}
    for name in sorted(os.listdir(pool_dir)):
        class_dir = os.path.join(pool_dir, name)
        if not os.path.isdir(class_dir):
            continue
        contents[name] = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith(_IMG_EXTS)
            and os.path.isfile(os.path.join(class_dir, f))
        )
    return contents


def build_classification_split(pool_dir, fraction, salt=_SALT, force=False):
    """Materialise `<pool_dir>_split/{train,val}/<class>/` from the flat
    class-folder pool `<pool_dir>/<class>/`, assigning whole videos to one side
    with is_holdout_video -- the same partition the detectors use.

    Ultralytics splits a bare class-folder dataset itself
    (check_cls_dataset -> split_classify_dataset) when it finds no `train/`
    subdirectory, but that split is per crop, randomly shuffled without a seed,
    and copied into a `_split` directory it creates with exist_ok and never
    cleans. Successive retrains therefore reshuffle into a directory that still
    holds the previous run's files, so the same crop ends up on both sides and
    classes deleted from the ethogram survive. Handing Ultralytics the directory
    built here keeps it from ever taking that path.

    Returns the split directory to train on, or `pool_dir` unchanged when no
    honest split is possible (no crops, or an empty validation side)."""
    if not pool_dir or not os.path.isdir(pool_dir):
        return pool_dir

    split_dir = pool_dir.rstrip('/\\') + '_split'
    contents = _pool_contents(pool_dir)
    if not any(contents.values()):
        return pool_dir

    # Freshness: rebuild only when the pool, the fraction or the salt changed.
    # The fingerprint covers every filename, so a deleted class or crop forces a
    # rebuild instead of lingering in the split.
    h = hashlib.sha256(f"{salt}|{fraction}".encode('utf-8'))
    for class_name, files in contents.items():
        h.update(f"\n{class_name}:".encode('utf-8'))
        for f in files:
            h.update(f"\n{f}".encode('utf-8'))
    fingerprint = h.hexdigest()
    marker = os.path.join(split_dir, _SPLIT_MARKER)
    if not force and os.path.isdir(split_dir) and os.path.exists(marker):
        try:
            with open(marker, 'r', encoding='utf-8') as f:
                if f.read().strip() == fingerprint:
                    print(f"Holdout split: up-to-date -> {split_dir}")
                    return split_dir
        except OSError:
            pass

    # Rebuild from scratch: copying into a surviving directory is exactly the
    # leak this function exists to prevent.
    if os.path.isdir(split_dir):
        shutil.rmtree(split_dir)

    n_train = n_val = 0
    empty_train, empty_val = [], []
    for class_name, files in contents.items():
        # Both sides get every class folder, even when empty, so that the
        # train and val ImageFolder scans agree on the class indices.
        train_class = os.path.join(split_dir, 'train', class_name)
        val_class = os.path.join(split_dir, 'val', class_name)
        os.makedirs(train_class, exist_ok=True)
        os.makedirs(val_class, exist_ok=True)
        c_train = c_val = 0
        for f in files:
            stem = video_label_for_crop(f)
            if is_holdout_video(stem, fraction, salt):
                shutil.copy2(os.path.join(pool_dir, class_name, f),
                             os.path.join(val_class, f))
                c_val += 1
            else:
                shutil.copy2(os.path.join(pool_dir, class_name, f),
                             os.path.join(train_class, f))
                c_train += 1
        n_train += c_train
        n_val += c_val
        if files and c_train == 0:
            empty_train.append(f"{class_name} ({c_val})")
        if files and c_val == 0:
            empty_val.append(f"{class_name} ({c_train})")

    total = n_train + n_val
    print(f"Holdout split: {pool_dir} -> {split_dir}  "
          f"{n_train} train / {n_val} val "
          f"({100.0 * n_val / total:.1f}% val, target {fraction * 100:.0f}%), "
          f"whole videos, no crop on both sides")

    if n_val == 0:
        # Nothing to validate on. Fall back to the pool so training still runs,
        # but say plainly that the split did not happen.
        print(f"WARNING: no video of '{pool_dir}' falls in the holdout at "
              f"val_frequency={fraction}. Falling back to Ultralytics' own "
              f"per-crop random split, which leaks between train and val. "
              f"Raise val_frequency or annotate more videos.")
        shutil.rmtree(split_dir, ignore_errors=True)
        return pool_dir

    # A per-video split cannot balance a class that lives in only a few videos.
    # Report it rather than move crops across the boundary, which would put the
    # leak back. Either case also makes Ultralytics log its own "found N images
    # in X classes (requires Y classes)" ERROR for the side holding the empty
    # folder -- cosmetic, since the class folders exist on both sides so the
    # indices still line up and training runs, but say so or it reads as fatal.
    if empty_train:
        print(f"WARNING: {pool_dir}: class(es) with no training crop at all - "
              f"{', '.join(empty_train)} (count = crops stranded in val). "
              f"The model cannot learn them; annotate them in more videos. "
              f"Ultralytics will log an ERROR about the train class count; "
              f"training still proceeds.")
    if empty_val:
        print(f"NOTE: {pool_dir}: class(es) absent from the holdout - "
              f"{', '.join(empty_val)} (count = training crops). They are "
              f"trained but never validated. Ultralytics will log an ERROR "
              f"about the val class count; training still proceeds.")

    with open(marker, 'w', encoding='utf-8') as f:
        f.write(fingerprint)
    return split_dir
