#!/usr/bin/env python3
"""
Re-split existing annot_static / annot_motion train/val images (+ labels) BY
WHOLE VIDEO, using the same deterministic assignment as BehaveAI_annotation.py
and BehaveAI_complex_model.py (see behaveai_holdout.is_holdout_video).

Annotations saved before that change were split per-frame at random, so
frames from the same video can currently sit on both sides of train/val. This
script regroups every already-annotated frame (and its augmented copies +
label file) by source video and moves it to the correct side, so
val_frequency is honoured at the video level going forward.

An annotation image follows the convention <video_label>_<frame_number>.jpg,
with augmented copies as <video_label>_<frame_number>_aug_<param>[_<idx>].jpg
(see BehaveAI_annotation.py:save_annotation and BehaveAI_augmentation.py).

Usage (run from, or pointed at, a project directory that contains
BehaveAI_settings.ini):
    python behaveai_rebalance_holdout.py <project_dir>            # dry-run
    python behaveai_rebalance_holdout.py <project_dir> --apply     # move files
"""
import os
import sys
import argparse
import configparser

from behaveai_holdout import is_holdout_video

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def _video_label_for(basename):
    """Recover the source video stem from an annotation basename, which is
    always <video_label>_<frame_number>[_aug_<param>[_<idx>]]."""
    name = basename.split('_aug_', 1)[0]
    stem, sep, tail = name.rpartition('_')
    return stem if sep and tail.isdigit() else name


def _rebalance_stream(stream_dir, fraction, apply):
    train_img = os.path.join(stream_dir, 'images', 'train')
    val_img = os.path.join(stream_dir, 'images', 'val')
    train_lbl = os.path.join(stream_dir, 'labels', 'train')
    val_lbl = os.path.join(stream_dir, 'labels', 'val')
    if not os.path.isdir(train_img) and not os.path.isdir(val_img):
        return 0, 0

    moved = kept = 0
    for side_img, side_lbl in ((train_img, train_lbl), (val_img, val_lbl)):
        if not os.path.isdir(side_img):
            continue
        currently_holdout = (side_img == val_img)
        for fname in sorted(os.listdir(side_img)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            basename = os.path.splitext(fname)[0]
            stem = _video_label_for(basename)
            wants_holdout = is_holdout_video(stem, fraction)
            if wants_holdout == currently_holdout:
                kept += 1
                continue

            dest_img_dir = val_img if wants_holdout else train_img
            dest_lbl_dir = val_lbl if wants_holdout else train_lbl
            src_img = os.path.join(side_img, fname)
            dst_img = os.path.join(dest_img_dir, fname)
            src_lbl = os.path.join(side_lbl, basename + '.txt')
            dst_lbl = os.path.join(dest_lbl_dir, basename + '.txt')

            direction = 'train->holdout' if wants_holdout else 'holdout->train'
            print(f"  {'MOVE' if apply else 'would move'}: {fname} ({direction}, video={stem})")
            if apply:
                os.makedirs(dest_img_dir, exist_ok=True)
                os.makedirs(dest_lbl_dir, exist_ok=True)
                os.replace(src_img, dst_img)
                if os.path.exists(src_lbl):
                    os.replace(src_lbl, dst_lbl)
            moved += 1
    return moved, kept


def rebalance_project(project_dir, apply):
    ini_path = os.path.join(project_dir, 'BehaveAI_settings.ini')
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(ini_path)
    fraction = float(cfg['DEFAULT'].get('val_frequency', '0.1'))
    print(f"Project: {project_dir}  (val_frequency = {fraction})")

    total_moved = total_kept = 0
    for stream in ('annot_static', 'annot_motion'):
        moved, kept = _rebalance_stream(os.path.join(project_dir, stream), fraction, apply)
        print(f"  {stream}: {moved} moved, {kept} already correct")
        total_moved += moved
        total_kept += kept

    if not apply and total_moved:
        print(f"  ({total_moved} file(s) would move — re-run with --apply)")
    return total_moved, total_kept


def _main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project', help="Project directory (contains BehaveAI_settings.ini).")
    parser.add_argument('--apply', action='store_true',
                         help="Actually move files (default: dry-run, prints planned moves).")
    args = parser.parse_args()
    rebalance_project(os.path.abspath(args.project), args.apply)


if __name__ == '__main__':
    _main()
