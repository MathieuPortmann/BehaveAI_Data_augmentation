# behaveai_holdout.py
# Deterministic whole-video train/holdout assignment shared by the annotation
# tool (per-video YOLO train/val split) and the complex-behaviour model
# (permanent held-out evaluation set).
#
# A video's status is derived purely from a hash of its filename stem, so it
# never needs a stored list: an existing video's status never changes, and a
# new video is classified automatically the first time it is seen.

import hashlib

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
