#!/usr/bin/env python3
"""BehaveAI dataset tiling (for SAHI-consistent training)

When SAHI sliced inference is enabled, the detector runs on native-resolution
tiles (a ~60px horse stays ~60px) instead of the whole 4K frame resized to 640
(the horse shrinks to ~10px). A model trained the usual way -- on whole frames
resized to 640 -- has only ever seen ~10px horses, so it does NOT recognise the
~60px horses SAHI feeds it, and detection collapses. This module removes that
train/inference scale mismatch: it slices the annotated dataset into the SAME
tiles used at inference, remapping every YOLO label into each tile, so training
sees horses at the tile scale.

It reads a YOLO data.yaml (train/val image dirs + nc/names), tiles every image,
and writes a parallel tiled dataset (`<src>_tiled/`) plus a tiled data.yaml that
`maybe_retrain` can train on directly.

Usage as a library:
    tiled_yaml = tile_dataset(src_yaml, slice_h=640, slice_w=640,
                              overlap_h=0.2, overlap_w=0.2)

CLI (for validating the tiling before a real train run):
    python BehaveAI_tiling.py static_annotations.yaml --limit 3 --out /tmp/tiled
"""

import os
import glob
import argparse

import cv2
import numpy as np
import yaml as _yaml


_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')


def _slice_origins(length, tile, overlap):
	"""Top-left origins covering `length` with `tile`-sized windows at `overlap`
	fraction. The last window is clamped to the far edge so the border is covered
	exactly once even when length is not a multiple of the step."""
	if length <= tile:
		return [0]
	step = max(1, int(round(tile * (1.0 - overlap))))
	origins = list(range(0, length - tile + 1, step))
	if not origins or origins[-1] != length - tile:
		origins.append(length - tile)
	return origins


def _labels_path_for(image_path):
	"""YOLO convention: .../images/<split>/x.jpg -> .../labels/<split>/x.txt."""
	base = os.path.splitext(image_path)[0] + '.txt'
	head, tail = os.path.split(base)
	parent, split = os.path.split(head)
	root = os.path.dirname(parent)
	cand = os.path.join(root, 'labels', split, tail)
	if os.path.exists(cand):
		return cand
	# Fallback: sibling labels/ next to images/ without a split level.
	cand2 = base.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)
	return cand2


def _read_labels(label_path):
	"""Return list of (cls, cx, cy, w, h) in normalised coords, or []."""
	out = []
	if not label_path or not os.path.exists(label_path):
		return out
	with open(label_path, 'r', encoding='utf-8', errors='replace') as f:
		for line in f:
			p = line.split()
			if len(p) < 5:
				continue
			try:
				out.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
			except ValueError:
				continue
	return out


def _remap_labels(labels, W, H, x0, y0, tw, th, min_visibility):
	"""Remap normalised whole-image labels into one tile at (x0, y0, tw, th).

	Keeps a box only if the visible fraction of its area inside the tile is at
	least `min_visibility` (so a horse clipped in half at a tile seam is dropped
	rather than taught as a tiny truncated box). Returns tile-normalised lines."""
	lines = []
	for cls, cx, cy, w, h in labels:
		# whole-image absolute box
		ax1 = (cx - w / 2.0) * W
		ay1 = (cy - h / 2.0) * H
		ax2 = (cx + w / 2.0) * W
		ay2 = (cy + h / 2.0) * H
		area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
		if area <= 0:
			continue
		# intersection with the tile
		ix1 = max(ax1, x0)
		iy1 = max(ay1, y0)
		ix2 = min(ax2, x0 + tw)
		iy2 = min(ay2, y0 + th)
		iw = ix2 - ix1
		ih = iy2 - iy1
		if iw <= 1 or ih <= 1:
			continue
		if (iw * ih) / area < min_visibility:
			continue
		# to tile-normalised
		ncx = ((ix1 + ix2) / 2.0 - x0) / tw
		ncy = ((iy1 + iy2) / 2.0 - y0) / th
		nw = iw / tw
		nh = ih / th
		lines.append(f"{cls} {ncx:.6f} {ncy:.6f} {nw:.6f} {nh:.6f}")
	return lines


def _tile_one_split(img_dir, out_img_dir, out_lbl_dir, slice_h, slice_w,
					overlap_h, overlap_w, min_visibility, keep_empty, limit):
	os.makedirs(out_img_dir, exist_ok=True)
	os.makedirs(out_lbl_dir, exist_ok=True)
	images = []
	for ext in _IMG_EXTS:
		images.extend(glob.glob(os.path.join(img_dir, '*' + ext)))
	images = sorted(set(images))
	if limit:
		images = images[:limit]
	n_tiles = n_kept = 0
	for img_path in images:
		img = cv2.imread(img_path)
		if img is None:
			continue
		H, W = img.shape[:2]
		labels = _read_labels(_labels_path_for(img_path))
		stem = os.path.splitext(os.path.basename(img_path))[0]
		for y0 in _slice_origins(H, slice_h, overlap_h):
			for x0 in _slice_origins(W, slice_w, overlap_w):
				tw = min(slice_w, W)
				th = min(slice_h, H)
				tile = img[y0:y0 + th, x0:x0 + tw]
				lines = _remap_labels(labels, W, H, x0, y0, tw, th, min_visibility)
				n_tiles += 1
				if not lines and not keep_empty:
					continue
				tag = f"{stem}_t{x0}_{y0}"
				cv2.imwrite(os.path.join(out_img_dir, tag + '.jpg'), tile)
				with open(os.path.join(out_lbl_dir, tag + '.txt'), 'w', encoding='utf-8') as f:
					f.write('\n'.join(lines) + ('\n' if lines else ''))
				n_kept += 1
	return len(images), n_tiles, n_kept


def tile_dataset(src_yaml, out_dir=None, slice_h=640, slice_w=640,
				 overlap_h=0.2, overlap_w=0.2, min_visibility=0.3,
				 keep_empty=False, limit=0, force=False):
	"""Slice a YOLO dataset into tiles and return the path of a tiled data.yaml.

	Skips work when a tiled set already exists for the same source image count
	(freshness marker), unless force=True. Set limit>0 to tile only the first N
	images per split (validation)."""
	src_yaml = os.path.abspath(src_yaml)
	with open(src_yaml, 'r', encoding='utf-8') as f:
		spec = _yaml.safe_load(f)
	if out_dir is None:
		out_dir = os.path.splitext(src_yaml)[0] + '_tiled'
	out_dir = os.path.abspath(out_dir)
	out_yaml = os.path.join(out_dir, 'data.yaml')

	# Freshness: only re-tile when the source train image count changed.
	train_dir = spec.get('train')
	src_count = 0
	if train_dir and os.path.isdir(train_dir):
		for ext in _IMG_EXTS:
			src_count += len(glob.glob(os.path.join(train_dir, '*' + ext)))
	marker = os.path.join(out_dir, '.source_count')
	if not force and not limit and os.path.exists(out_yaml) and os.path.exists(marker):
		try:
			if int(open(marker).read().strip()) == src_count:
				print(f"Tiling: up-to-date ({src_count} source images) -> {out_yaml}")
				return out_yaml
		except Exception:
			pass

	print(f"Tiling {src_yaml} -> {out_dir} "
		  f"(tile {slice_w}x{slice_h}, overlap {overlap_w}/{overlap_h}, "
		  f"min_vis {min_visibility})")
	tiled = {'nc': spec.get('nc'), 'names': spec.get('names')}
	for split in ('train', 'val'):
		img_dir = spec.get(split)
		if not img_dir or not os.path.isdir(img_dir):
			continue
		out_img_dir = os.path.join(out_dir, 'images', split)
		out_lbl_dir = os.path.join(out_dir, 'labels', split)
		# Clear a previous full run (not for limited validation runs).
		if not limit:
			for d in (out_img_dir, out_lbl_dir):
				if os.path.isdir(d):
					for fn in glob.glob(os.path.join(d, '*')):
						try:
							os.remove(fn)
						except OSError:
							pass
		n_img, n_tiles, n_kept = _tile_one_split(
			img_dir, out_img_dir, out_lbl_dir, slice_h, slice_w,
			overlap_h, overlap_w, min_visibility, keep_empty, limit)
		tiled[split] = out_img_dir
		print(f"  {split}: {n_img} images -> {n_tiles} tiles, kept {n_kept} "
			  f"(with {'>=1 box' if not keep_empty else 'all'})")

	os.makedirs(out_dir, exist_ok=True)
	with open(out_yaml, 'w', encoding='utf-8') as f:
		_yaml.safe_dump(tiled, f, sort_keys=False, allow_unicode=True)
	if not limit:
		with open(marker, 'w') as f:
			f.write(str(src_count))
	print(f"Tiled data.yaml written: {out_yaml}")
	return out_yaml


if __name__ == '__main__':
	ap = argparse.ArgumentParser(description="Tile a YOLO dataset for SAHI-consistent training.")
	ap.add_argument("src_yaml")
	ap.add_argument("--out", default=None)
	ap.add_argument("--slice", type=int, default=640)
	ap.add_argument("--overlap", type=float, default=0.2)
	ap.add_argument("--min-vis", type=float, default=0.3)
	ap.add_argument("--keep-empty", action="store_true")
	ap.add_argument("--limit", type=int, default=0, help="tile only first N images/split (validation)")
	ap.add_argument("--force", action="store_true")
	a = ap.parse_args()
	tile_dataset(a.src_yaml, a.out, a.slice, a.slice, a.overlap, a.overlap,
				 a.min_vis, a.keep_empty, a.limit, a.force)
