#!/usr/bin/env python3
"""
BehaveAI_bootstrap_species.py — ONE-SHOT tool, self-deletes after a successful --apply.

Backfills the species (model 0) training set for a project whose existing
primary annotations (annot_static/, annot_motion/) predate the species feature.
Every box in those annotations is, today, unambiguously the project's first
species (species_list[0]) — this script crops each box out of its already-saved
annotation image and copies it into annot_species_crop/<species_list[0]>/, using
the exact same crop-filename convention (<video_label>_<frame>_<x1>_<y1>.jpg) the
annotation tool itself uses, so BehaveAI_annotation.py picks the crops back up
transparently when a frame is reopened.

It NEVER writes into annot_static/annot_motion/ (read-only) and NEVER moves or
renames anything - purely additive copies into a new folder.

Usage:
    python BehaveAI_bootstrap_species.py <project_dir>            # dry run (default)
    python BehaveAI_bootstrap_species.py <project_dir> --apply    # write crops, then
                                                                   # delete this script

Once annot_species_crop/<species_list[0]>/ exists, this script has done its job:
any *future* species is annotated directly with the "Espèce" button in
BehaveAI_annotation.py, not via a migration script like this one.
"""

import os
import sys
import argparse
import configparser

import cv2

from behaveai_config import get_species_list, species_folder


def _norm_to_pixels(xc, yc, bw, bh, w, h):
	cx = float(xc) * w
	cy = float(yc) * h
	bw_p = float(bw) * w
	bh_p = float(bh) * h
	x1 = int(cx - bw_p / 2); y1 = int(cy - bh_p / 2)
	x2 = int(cx + bw_p / 2); y2 = int(cy + bh_p / 2)
	x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
	x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
	return x1, y1, x2, y2


def _iter_boxes(images_dir, labels_dir):
	"""Yield (image_path, image_stem, x1, y1, x2, y2) for every box in every
	annotated image under images_dir/labels_dir. Read-only: never touches
	images_dir/labels_dir."""
	if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
		return
	for fname in sorted(os.listdir(images_dir)):
		if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
			continue
		stem = os.path.splitext(fname)[0]
		label_path = os.path.join(labels_dir, stem + '.txt')
		if not os.path.exists(label_path):
			continue
		image_path = os.path.join(images_dir, fname)
		img = cv2.imread(image_path)
		if img is None:
			continue
		h, w = img.shape[:2]
		with open(label_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) < 5:
					continue
				try:
					xc, yc, bw, bh = parts[1:5]
					x1, y1, x2, y2 = _norm_to_pixels(xc, yc, bw, bh, w, h)
				except Exception:
					continue
				yield image_path, stem, x1, y1, x2, y2


def bootstrap_species(project_dir, apply=False):
	ini_path = os.path.join(project_dir, 'BehaveAI_settings.ini')
	if not os.path.exists(ini_path):
		print(f"No BehaveAI_settings.ini found in {project_dir}")
		return 1

	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(ini_path)
	species_list = get_species_list(config)
	target_species = species_list[0]

	annot_static_dir = os.path.join(project_dir, species_folder('annot_static', target_species, species_list))
	annot_motion_dir = os.path.join(project_dir, species_folder('annot_motion', target_species, species_list))
	species_crop_dir = os.path.join(project_dir, 'annot_species_crop', target_species)

	sources = [
		(os.path.join(annot_static_dir, 'images', 'train'), os.path.join(annot_static_dir, 'labels', 'train')),
		(os.path.join(annot_static_dir, 'images', 'val'), os.path.join(annot_static_dir, 'labels', 'val')),
		(os.path.join(annot_motion_dir, 'images', 'train'), os.path.join(annot_motion_dir, 'labels', 'train')),
		(os.path.join(annot_motion_dir, 'images', 'val'), os.path.join(annot_motion_dir, 'labels', 'val')),
	]

	# Crops already backfilled (idempotent re-run / resume after a partial --apply).
	existing = set()
	if os.path.isdir(species_crop_dir):
		existing = set(os.listdir(species_crop_dir))

	planned = 0
	written = 0
	if apply:
		os.makedirs(species_crop_dir, exist_ok=True)

	for images_dir, labels_dir in sources:
		for image_path, stem, x1, y1, x2, y2 in _iter_boxes(images_dir, labels_dir):
			crop_name = f"{stem}_{x1}_{y1}.jpg"
			if crop_name in existing:
				continue
			planned += 1
			if not apply:
				continue
			img = cv2.imread(image_path)
			if img is None:
				continue
			crop = img[y1:y2, x1:x2]
			if crop is None or crop.size == 0:
				continue
			cv2.imwrite(os.path.join(species_crop_dir, crop_name), crop)
			written += 1

	if not apply:
		print(f"[dry run] Would create {planned} crop(s) in "
			  f"{os.path.relpath(species_crop_dir, project_dir)}/ (species: {target_species}).")
		print("Re-run with --apply to write them for real. This script deletes itself "
			  "after a successful --apply, so back up the repo if you want to keep it.")
		return 0

	print(f"Wrote {written} crop(s) to {os.path.relpath(species_crop_dir, project_dir)}/ (species: {target_species}).")
	print("Bootstrap done. You can now train the species model (model_species) the same "
		  "way the secondary models are trained.")

	try:
		os.remove(__file__)
		print(f"Removed {os.path.basename(__file__)} - this was a one-shot migration tool.")
	except Exception as e:
		print(f"Could not remove {__file__} automatically ({e}); delete it manually.")

	return 0


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('project_dir', help='Path to the BehaveAI project directory')
	parser.add_argument('--apply', action='store_true',
						 help='Actually write the crops (default is dry-run) and self-delete on success')
	args = parser.parse_args()
	sys.exit(bootstrap_species(os.path.abspath(args.project_dir), apply=args.apply))
