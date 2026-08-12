#!/usr/bin/env python3
"""
Re-split existing annot_static / annot_motion train/val images (+ labels) BY
WHOLE VIDEO, using the same deterministic assignment as BehaveAI_annotation.py
and BehaveAI_complex_model.py (see behaveai_holdout.is_holdout_video).

Annotations saved before that change were split per-frame at random, so
frames from the same video can currently sit on both sides of train/val. This
script regroups every already-annotated frame (and its augmented copies, label
file and grey-region mask) by source video and moves it to the correct side, so
val_frequency is honoured at the video level going forward.

Also run it after anything that changes a video's *name*: the side is a hash of
the filename stem, so a rename reassigns videos between train and holdout, and
leaving the files where they were would train the detector on frames that
BehaveAI_evaluate_detection.py (which filters by the hash, not by the folder)
then scores as holdout.

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

from behaveai_holdout import is_holdout_video, video_label_for_annotation

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def _rebalance_stream(stream_dir, fraction, apply):
    train_img = os.path.join(stream_dir, 'images', 'train')
    val_img = os.path.join(stream_dir, 'images', 'val')
    if not os.path.isdir(train_img) and not os.path.isdir(val_img):
        return 0, 0

    # The grey-region sidecar travels with its frame: BehaveAI_inspect_dataset.py
    # looks for it beside the label (lbl_dir.replace('labels', 'masks')), so a
    # mask left on the other side is silently invisible and re-editing the frame
    # drops its grey boxes.
    subs = (('images', lambda b, f: f),
            ('labels', lambda b, f: b + '.txt'),
            ('masks', lambda b, f: b + '.mask.txt'))

    moved = kept = 0
    for side in ('train', 'val'):
        side_img = os.path.join(stream_dir, 'images', side)
        if not os.path.isdir(side_img):
            continue
        currently_holdout = (side == 'val')
        for fname in sorted(os.listdir(side_img)):
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            basename = os.path.splitext(fname)[0]
            stem = video_label_for_annotation(basename)
            wants_holdout = is_holdout_video(stem, fraction)
            if wants_holdout == currently_holdout:
                kept += 1
                continue

            dest = 'val' if wants_holdout else 'train'
            direction = 'train->holdout' if wants_holdout else 'holdout->train'
            print(f"  {'MOVE' if apply else 'would move'}: {fname} ({direction}, video={stem})")
            if apply:
                for sub, name_of in subs:
                    src = os.path.join(stream_dir, sub, side, name_of(basename, fname))
                    if not os.path.exists(src):
                        continue
                    dst_dir = os.path.join(stream_dir, sub, dest)
                    os.makedirs(dst_dir, exist_ok=True)
                    os.replace(src, os.path.join(dst_dir, os.path.basename(src)))
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
