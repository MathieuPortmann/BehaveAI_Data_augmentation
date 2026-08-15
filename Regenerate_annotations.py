#!/usr/bin/env python3
"""
Regenerate motion annotation images for a BehaveAI project.

Usage:
	python regenerate_motion_dataset.py <path/to/BehaveAI_settings.ini>
or:
	python regenerate_motion_dataset.py		# will prompt for INI via file dialog

This script:
 - reads the settings INI (and resolves relative paths relative to the INI's directory)
 - rebuilds motion images (annot_motion/images/{train,val}) using the same processing
   as the annotation tool (sampling a small window of frames, computing diffs, chromatic tail, etc.)
 - applies masks and blocking boxes in the same way as your annotator

Optional dataset conversions, used by the settings GUI when one of the two
cross-stream training switches changes (see behaveai_config.load_stream_training_config):

	--relabel merge    every box goes into BOTH label trees under the shared global
					   index space (primary_train_both_streams turned ON)
	--relabel split    the shared space is split back into per-stream label files
					   (primary_train_both_streams turned OFF)
	--recrop           rebuild annot_<stream>_crop/ so each secondary crop sits in
					   the stream(s) the current switches call for

Both conversions are lossless: the global index carries the class either way, and
the crops are re-cut from the regenerated images, so nothing you annotated is lost
and the switches can be flipped back and forth.
"""
import cv2
import os
import numpy as np
import configparser

from behaveai_config import resolve_project_dir, parse_class_list, load_stream_training_config
import glob
import sys
import time
from collections import deque

# optional GUI prompt if INI not supplied
try:
	import tkinter as tk
	from tkinter import filedialog, messagebox
	_HAS_TK = True
except Exception:
	_HAS_TK = False

# -----------------------
# Helpers: path resolve / config loader
# -----------------------

def load_config(config_path):
	"""
	Read configuration from config_path and return (params_dict, clips_dir_resolved).
	params contains numeric / strategy settings used by the image generation pipeline.
	clips_dir_resolved is an absolute (or normalized) path to the clips directory resolved
	relative to the INI's project directory.
	"""
	config = configparser.ConfigParser()
	config.optionxform = str  # preserve case
	config.read(config_path)

	project_dir = os.path.dirname(os.path.abspath(config_path))

	params = {}
	try:
		# Read parameters (same names as your previous implementation)
		params['scale_factor'] = float(config['DEFAULT'].get('scale_factor', '1.0'))
		params['expA'] = float(config['DEFAULT'].get('expA', '0.5'))
		params['expB'] = float(config['DEFAULT'].get('expB', '0.8'))
		params['strategy'] = config['DEFAULT'].get('strategy', 'exponential')
		params['chromatic_tail_only'] = config['DEFAULT'].get('chromatic_tail_only', 'false').lower()
		params['lum_weight'] = float(config['DEFAULT'].get('lum_weight', '0.7'))
		params['rgb_multipliers'] = [float(x) for x in config['DEFAULT'].get('rgb_multipliers', '2,2,2').split(',')]
		params['frame_skip'] = int(config['DEFAULT'].get('frame_skip', '0'))
		params['motion_threshold'] = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))
		params['motion_blocks_static'] = config['DEFAULT'].get('motion_blocks_static', 'false').lower()
		params['static_blocks_motion'] = config['DEFAULT'].get('static_blocks_motion', 'false').lower()
		# 'true' matches what the settings GUI writes and what the annotation tool
		# and the dataset inspector require: those two treat the key as mandatory
		# (KeyError with a clear message when absent), so a divergent fallback here
		# would silently regenerate under a different policy than the one that
		# created the images in the first place.
		params['save_empty_frames'] = config['DEFAULT'].get('save_empty_frames', 'true').lower()

		# Cross-stream training switches + the split point of the global primary
		# index space (everything below n_static is a static class). Both are
		# needed to convert label files between the two layouts.
		_stream_cfg = load_stream_training_config(config)
		params['primary_both_streams'] = _stream_cfg['primary_both_streams']
		params['secondary_both_streams'] = _stream_cfg['secondary_both_streams']
		params['n_static'] = len(parse_class_list(config['DEFAULT'].get('primary_static_classes', '')))
		params['n_primary'] = params['n_static'] + len(
			parse_class_list(config['DEFAULT'].get('primary_motion_classes', '')))

		# Blocking is meaningless once both detectors are trained on every box, and
		# honouring it here would grey out the targets they are meant to learn.
		if params['primary_both_streams']:
			params['motion_blocks_static'] = 'false'
			params['static_blocks_motion'] = 'false'

		# Compute base frame window size (number of sampled frames)
		base_window = 4
		if params['strategy'] == 'exponential':
			if params['expA'] > 0.2 or params['expB'] > 0.2:
				base_window = 5
			if params['expA'] > 0.5 or params['expB'] > 0.5:
				base_window = 10
			if params['expA'] > 0.7 or params['expB'] > 0.7:
				base_window = 15
			if params['expA'] > 0.8 or params['expB'] > 0.8:
				base_window = 20
			if params['expA'] > 0.9 or params['expB'] > 0.9:
				base_window = 45

		params['base_frame_window'] = base_window
		params['frame_window'] = base_window * (params['frame_skip'] + 1)

	except KeyError as e:
		raise KeyError(f"Missing configuration parameter: {e}")

	# Resolve clips_dir relative to project_dir (fallback 'clips')
	clips_dir = resolve_project_dir(config, project_dir, 'clips')

	return params, clips_dir


# -----------------------
# Image processing helpers (unchanged logic besides small improvements)
# -----------------------

def generate_base_images(video_path, frame_num, params):
	"""
	Generate static and motion images for a specific video frame.
	frame_num is interpreted as the LAST frame of the motion window to mimic the annotator.
	Returns (static_img_bgr, motion_img_bgr) or (None, None) on failure.
	"""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"Error opening video: {video_path}")
		return None, None

	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if total_frames <= 0:
		print(f"Video appears empty or unreadable: {video_path}")
		cap.release()
		return None, None

	step = params['frame_skip'] + 1
	base_N = params.get('base_frame_window', 4)

	# compute start so last appended index should equal frame_num
	start_frame = int(frame_num - (base_N - 1) * step)
	start_frame = max(0, start_frame)
	if start_frame > total_frames - 1:
		print(f"Start frame {start_frame} beyond video length ({total_frames}) for {video_path}")
		cap.release()
		return None, None

	cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
	collected = []
	read_count = 0
	idx = start_frame
	# safety limit: don't try more than frame_window + some slack
	max_reads = params['frame_window'] + 10

	while len(collected) < base_N and idx <= total_frames - 1 and read_count <= max_reads:
		ret, frame = cap.read()
		if not ret:
			break
		if (read_count % step) == 0:
			if params['scale_factor'] != 1.0:
				frame = cv2.resize(frame, None, fx=params['scale_factor'], fy=params['scale_factor'])
			collected.append(frame.copy())
		read_count += 1
		idx += 1

	if not collected:
		cap.release()
		print(f"Could not collect frames for target {frame_num} (start {start_frame}) in {video_path}")
		return None, None

	# Process collected frames to produce diffs for the last frame
	prev_frames = [None] * 3
	static_img = None
	diffs = None
	gray = None

	for i, f in enumerate(collected):
		if f is None:
			continue
		frame_bgr = f
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

		if static_img is None:
			static_img = frame_bgr.copy()
			prev_frames = [gray.copy()] * 3
			continue

		current_diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

		if params['strategy'] == 'exponential':
			prev_frames[0] = gray
			prev_frames[1] = cv2.addWeighted(prev_frames[1], params['expA'], gray, 1 - params['expA'], 0)
			prev_frames[2] = cv2.addWeighted(prev_frames[2], params['expB'], gray, 1 - params['expB'], 0)
		elif params['strategy'] == 'sequential':
			prev_frames[2] = prev_frames[1]
			prev_frames[1] = prev_frames[0]
			prev_frames[0] = gray

		static_img = frame_bgr.copy()
		diffs = current_diffs

	if diffs is None or gray is None:
		cap.release()
		print(f"Insufficient frames to compute diffs for {frame_num} (collected {len(collected)} frames)")
		return None, None

	# Build motion image (chromatic tail or normal)
	if params['chromatic_tail_only'] == 'true':
		tb = cv2.subtract(diffs[0], diffs[1])
		tr = cv2.subtract(diffs[2], diffs[1])
		tg = cv2.subtract(diffs[1], diffs[0])

		blue = cv2.addWeighted(gray, params['lum_weight'], tb, params['rgb_multipliers'][2], params['motion_threshold'])
		green = cv2.addWeighted(gray, params['lum_weight'], tg, params['rgb_multipliers'][1], params['motion_threshold'])
		red = cv2.addWeighted(gray, params['lum_weight'], tr, params['rgb_multipliers'][0], params['motion_threshold'])
	else:
		blue = cv2.addWeighted(gray, params['lum_weight'], diffs[0], params['rgb_multipliers'][2], params['motion_threshold'])
		green = cv2.addWeighted(gray, params['lum_weight'], diffs[1], params['rgb_multipliers'][1], params['motion_threshold'])
		red = cv2.addWeighted(gray, params['lum_weight'], diffs[2], params['rgb_multipliers'][0], params['motion_threshold'])

	motion_img = cv2.merge([blue, green, red]).astype(np.uint8)

	cap.release()
	return static_img, motion_img


def read_mask_file(mask_path):
	boxes = []
	if os.path.exists(mask_path):
		with open(mask_path, 'r') as f:
			for line in f:
				parts = line.strip().split()
				if len(parts) == 4:
					try:
						boxes.append(tuple(map(int, parts)))
					except Exception:
						pass
	return boxes


def apply_grey_boxes(image, boxes):
	result = image.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result


def apply_blocking_boxes(image, boxes):
	result = image.copy()
	for (x1, y1, x2, y2) in boxes:
		cv2.rectangle(result, (x1, y1), (x2, y2), (128, 128, 128), -1)
	return result


def get_blocking_boxes(label_path, img_w, img_h):
	boxes = []
	if os.path.exists(label_path):
		with open(label_path, 'r') as f:
			for line in f:
				parts = line.split()
				if len(parts) < 5:
					continue
				try:
					xc = float(parts[1]); yc = float(parts[2])
					w = float(parts[3]); h = float(parts[4])
				except Exception:
					continue
				x1 = int((xc - w/2) * img_w)
				y1 = int((yc - h/2) * img_h)
				x2 = int((xc + w/2) * img_w)
				y2 = int((yc + h/2) * img_h)
				boxes.append((x1, y1, x2, y2))
	return boxes


# -----------------------
# Label-space conversion (primary_train_both_streams on <-> off)
# -----------------------

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
STREAM_DIRS = ('annot_static', 'annot_motion')


def _read_label_file(path):
	"""[(cls, xc, yc, bw, bh)] from a YOLO label file; missing file -> []."""
	rows = []
	if not os.path.exists(path):
		return rows
	with open(path, 'r') as f:
		for line in f:
			parts = line.split()
			if len(parts) < 5:
				continue
			try:
				rows.append((int(parts[0]), parts[1], parts[2], parts[3], parts[4]))
			except ValueError:
				continue
	return rows


def _write_label_file(path, rows):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w') as f:
		for cls, xc, yc, bw, bh in rows:
			f.write(f"{cls} {xc} {yc} {bw} {bh}\n")


def _index_label_trees():
	"""{base_name: {tree: split}} over every label file in both annotation trees."""
	frames = {}
	for tree in STREAM_DIRS:
		for split in ('train', 'val'):
			label_dir = os.path.join(tree, 'labels', split)
			if not os.path.isdir(label_dir):
				continue
			for label_file in glob.glob(os.path.join(label_dir, '*.txt')):
				if label_file.endswith('.mask.txt'):
					continue
				base = os.path.splitext(os.path.basename(label_file))[0]
				frames.setdefault(base, {})[tree] = split
	return frames


def _mirror_mask(base, src_tree, src_split, dst_tree, dst_split):
	"""Copy a frame's grey-region mask into the other tree when it has none.

	The two trees carry the same mask (the annotator draws it once, on the frame),
	so a frame that only now gains a label file in the other tree must gain its
	mask too - otherwise the regenerated image would lose its grey regions."""
	src = os.path.join(src_tree, 'masks', src_split, f"{base}.mask.txt")
	dst = os.path.join(dst_tree, 'masks', dst_split, f"{base}.mask.txt")
	if os.path.exists(src) and not os.path.exists(dst):
		os.makedirs(os.path.dirname(dst), exist_ok=True)
		with open(src, 'r') as fi, open(dst, 'w') as fo:
			fo.write(fi.read())


def _drop_frame_from_tree(base, tree, split):
	"""Remove a frame's label + image from one tree (it has no box there anymore)."""
	for path in (os.path.join(tree, 'labels', split, f"{base}.txt"),
				 *(os.path.join(tree, 'images', split, base + ext) for ext in IMAGE_EXTS)):
		if os.path.exists(path):
			try:
				os.remove(path)
			except OSError as e:
				print(f"  could not remove {path}: {e}")


def relabel_datasets(mode, n_static, save_empty_frames):
	"""Convert both label trees between the per-stream and the shared index space.

	merge: every box lands in both trees under the global index (static classes
	       0..n_static-1, motion classes after them), which is what both detectors
	       are trained on when primary_train_both_streams is on.
	split: the reverse - each box goes back to the tree of its own class, and the
	       motion tree's indices are re-based onto the motion list.

	Lossless in both directions, because the global index identifies the class
	regardless of which tree the line sits in.
	"""
	frames = _index_label_trees()
	print(f"Relabelling {len(frames)} annotated frame(s): {mode}")
	converted = 0
	for base, trees in sorted(frames.items()):
		static_split = trees.get('annot_static')
		motion_split = trees.get('annot_motion')
		split = static_split or motion_split
		static_path = os.path.join('annot_static', 'labels', static_split or split, f"{base}.txt")
		motion_path = os.path.join('annot_motion', 'labels', motion_split or split, f"{base}.txt")

		if mode == 'merge':
			static_rows = _read_label_file(static_path)
			motion_rows = _read_label_file(motion_path)
			# Two identical trees mean this frame is already in the shared space
			# (a re-run, or a rebuild that was interrupted part-way). Offsetting the
			# motion rows again would shift every class, so leave it alone: a merged
			# motion file legitimately contains indices below n_static.
			already_merged = bool(static_rows) and sorted(static_rows) == sorted(motion_rows)
			rows = list(static_rows)
			if not already_merged:
				seen = set(rows)
				for cls, xc, yc, bw, bh in motion_rows:
					entry = (cls + n_static, xc, yc, bw, bh)
					if entry not in seen:
						seen.add(entry)
						rows.append(entry)
			if not rows and save_empty_frames != 'true':
				continue
			_write_label_file(static_path, rows)
			_write_label_file(motion_path, rows)
			if static_split and not motion_split:
				_mirror_mask(base, 'annot_static', static_split, 'annot_motion', static_split)
			elif motion_split and not static_split:
				_mirror_mask(base, 'annot_motion', motion_split, 'annot_static', motion_split)
		else:  # split
			# Both trees hold the same global-index rows; read whichever exists.
			rows = _read_label_file(static_path) or _read_label_file(motion_path)
			static_rows = [(c, xc, yc, bw, bh) for c, xc, yc, bw, bh in rows if c < n_static]
			motion_rows = [(c - n_static, xc, yc, bw, bh) for c, xc, yc, bw, bh in rows if c >= n_static]
			for tree, tree_rows, tree_split in (('annot_static', static_rows, static_split or split),
												('annot_motion', motion_rows, motion_split or split)):
				path = os.path.join(tree, 'labels', tree_split, f"{base}.txt")
				if tree_rows or save_empty_frames == 'true':
					_write_label_file(path, tree_rows)
				else:
					# No box of this stream left: the annotation tool would not have
					# written this frame into this tree at all.
					_drop_frame_from_tree(base, tree, tree_split)
		converted += 1
	print(f"Relabelled {converted} frame(s).")


# -----------------------
# Secondary crop rebuild (secondary_train_both_streams, and any primary change)
# -----------------------

def _parse_crop_filename(fn):
	"""'<video>_<frame>_<x1>_<y1>.jpg' -> (video_label, frame, x1, y1) or None."""
	parts = os.path.splitext(fn)[0].split('_')
	if len(parts) < 4:
		return None
	try:
		return '_'.join(parts[:-3]), int(parts[-3]), int(parts[-2]), int(parts[-1])
	except ValueError:
		return None


def index_secondary_crops():
	"""{(video_label, frame): {(x1, y1): secondary_name}} over both crop trees.

	The crops themselves are the only record of which secondary a box was given
	(the label files hold the primary only), so they are read before anything is
	deleted and re-cut from the regenerated images afterwards."""
	index = {}
	for base_crop_dir in ('annot_static_crop', 'annot_motion_crop'):
		if not os.path.isdir(base_crop_dir):
			continue
		for secondary_name in sorted(os.listdir(base_crop_dir)):
			sec_dir = os.path.join(base_crop_dir, secondary_name)
			if not os.path.isdir(sec_dir):
				continue
			for fn in os.listdir(sec_dir):
				if not fn.lower().endswith(IMAGE_EXTS):
					continue
				parsed = _parse_crop_filename(fn)
				if parsed is None:
					continue
				video_label, frame, x1, y1 = parsed
				index.setdefault((video_label, frame), {})[(x1, y1)] = secondary_name
	return index


def _norm_to_pixels(xc, yc, bw, bh, w, h):
	"""Same truncation as index_annotations._norm_to_pixels, so the pixel boxes
	recovered here match the ones the annotation tool reloads."""
	cx = float(xc) * w
	cy = float(yc) * h
	bw_p = float(bw) * w
	bh_p = float(bh) * h
	x1 = int(cx - bw_p / 2); y1 = int(cy - bh_p / 2)
	x2 = int(cx + bw_p / 2); y2 = int(cy + bh_p / 2)
	x1 = max(0, min(w - 1, x1)); y1 = max(0, min(h - 1, y1))
	x2 = max(0, min(w - 1, x2)); y2 = max(0, min(h - 1, y2))
	return x1, y1, x2, y2


def _delete_frame_crops(video_label, frame):
	for base_crop_dir in ('annot_static_crop', 'annot_motion_crop'):
		if not os.path.isdir(base_crop_dir):
			continue
		for secondary_name in os.listdir(base_crop_dir):
			sec_dir = os.path.join(base_crop_dir, secondary_name)
			if not os.path.isdir(sec_dir):
				continue
			for fn in list(os.listdir(sec_dir)):
				if not fn.lower().endswith(IMAGE_EXTS):
					continue
				parsed = _parse_crop_filename(fn)
				if parsed and parsed[0] == video_label and parsed[1] == frame:
					try:
						os.remove(os.path.join(sec_dir, fn))
					except OSError:
						pass


def recrop_frame(base_name, boxes, crops_for_frame, static_final, motion_final, params):
	"""Re-cut one frame's secondary crops into the stream(s) the switches ask for.

	`boxes` is [(global_primary_idx, x1, y1, x2, y2)] recovered from the label
	file, `crops_for_frame` the {(x1, y1): secondary_name} recorded before the old
	crops were deleted. Matching is by top-left corner with the same +/-2 px
	tolerance index_annotations uses, because the coordinates make a round trip
	through the normalised label format."""
	if not crops_for_frame:
		return 0
	written = 0
	for (cx1, cy1), secondary_name in crops_for_frame.items():
		match = None
		for gidx, x1, y1, x2, y2 in boxes:
			if abs(x1 - cx1) <= 2 and abs(y1 - cy1) <= 2:
				match = (gidx, x1, y1, x2, y2)
				break
		if match is None:
			print(f"  {base_name}: no box under crop at ({cx1},{cy1}) — crop dropped")
			continue
		gidx, x1, y1, x2, y2 = match
		if params['secondary_both_streams']:
			targets = (('annot_static_crop', static_final), ('annot_motion_crop', motion_final))
		elif gidx < params['n_static']:
			targets = (('annot_static_crop', static_final),)
		else:
			targets = (('annot_motion_crop', motion_final),)
		for base_crop_dir, source_img in targets:
			if source_img is None:
				continue
			crop = source_img[y1:y2, x1:x2]
			if crop is None or crop.size == 0:
				continue
			class_dir = os.path.join(base_crop_dir, secondary_name)
			os.makedirs(class_dir, exist_ok=True)
			cv2.imwrite(os.path.join(class_dir, f"{base_name}_{x1}_{y1}.jpg"), crop)
			written += 1
	return written


def boxes_from_labels(base_name, trees, params, img_w, img_h):
	"""[(global_primary_idx, x1, y1, x2, y2)] for a frame, from whichever label
	tree holds it, with the motion tree's indices lifted into the global space."""
	boxes = []
	seen = set()
	for tree, split in trees.items():
		path = os.path.join(tree, 'labels', split, f"{base_name}.txt")
		offset = 0
		if tree == 'annot_motion' and not params['primary_both_streams']:
			offset = params['n_static']
		for cls, xc, yc, bw, bh in _read_label_file(path):
			x1, y1, x2, y2 = _norm_to_pixels(xc, yc, bw, bh, img_w, img_h)
			entry = (cls + offset, x1, y1, x2, y2)
			if entry not in seen:
				seen.add(entry)
				boxes.append(entry)
	return boxes


# -----------------------
# Main regeneration function
# -----------------------

def regenerate_annotations(config_path, relabel=None, recrop=False):
	"""Regenerate motion and static images using parameters & clips_dir from config_path.

	`relabel` ('merge' | 'split') converts the label trees between the per-stream and
	the shared index space first; `recrop` rebuilds the secondary crop folders from
	the regenerated images. Both are driven by the settings GUI when one of the two
	cross-stream training switches changes.
	"""
	params, clips_dir = load_config(config_path)

	# Ensure we operate with project_dir as cwd to keep relative paths consistent
	project_dir = os.path.dirname(os.path.abspath(config_path))
	os.chdir(project_dir)

	print(f"Regenerating using INI: {config_path}")
	print(f"Using clips directory: {clips_dir}")

	# Label conversion runs first: everything below (which frames belong to which
	# tree, which boxes a crop can attach to) reads the converted files.
	if relabel:
		relabel_datasets(relabel, params['n_static'], params['save_empty_frames'])

	# The crops are the only record of each box's secondary class, so they are read
	# before the rebuild deletes them.
	crop_index = index_secondary_crops() if recrop else {}
	if recrop:
		print(f"Rebuilding secondary crops for {len(crop_index)} annotated frame(s).")

	# One entry per annotated frame, listing which tree(s) hold it and in which
	# split. Frames present in both trees used to be decoded twice.
	frames = _index_label_trees()

	print(f"Found {len(frames)} annotated frames to process (motion + static).")

	# extensions to search for video files
	exts = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
	crops_written = 0

	# process each unique frame
	for base_name, trees in sorted(frames.items()):
		parts = base_name.split('_')
		try:
			frame_num = int(parts[-1])
		except ValueError:
			print(f"Skipping {base_name}: trailing token is not an integer")
			continue
		video_name = '_'.join(parts[:-1])

		# find video in clips_dir
		video_path = None
		for ext in exts:
			test_path = os.path.join(clips_dir, video_name + ext)
			if os.path.exists(test_path):
				video_path = test_path
				break

		if not video_path:
			print(f"Video not found for {base_name}: looking in {clips_dir} for files named {video_name}.*")
			continue

		static_img, motion_img = generate_base_images(video_path, frame_num, params)
		if static_img is None and motion_img is None:
			print(f"  Could not generate images for {base_name}")
			continue

		# image dims (prefer static_img if available else motion_img)
		ref_img = static_img if static_img is not None else motion_img
		img_h, img_w = ref_img.shape[:2]

		# Each tree keeps its own split for this frame (normally the same one).
		static_split = trees.get('annot_static')
		motion_split = trees.get('annot_motion')
		split = static_split or motion_split

		# mask & label paths for both static and motion (may or may not exist)
		static_mask_path = os.path.join('annot_static', 'masks', static_split or split, f"{base_name}.mask.txt")
		motion_mask_path = os.path.join('annot_motion', 'masks', motion_split or split, f"{base_name}.mask.txt")

		static_mask_boxes = read_mask_file(static_mask_path)
		motion_mask_boxes = read_mask_file(motion_mask_path)

		static_label_path = os.path.join('annot_static', 'labels', static_split or split, f"{base_name}.txt")
		motion_label_path = os.path.join('annot_motion', 'labels', motion_split or split, f"{base_name}.txt")

		static_final = None
		motion_final = None

		# -----------------------
		# Process static images (save into annot_static/images/<split>/)
		# -----------------------
		# We regenerate static images when:
		#  - the frame has a static label file, OR
		#  - save_empty_frames == 'true' (keep parity with motion logic)
		if static_split or params['save_empty_frames'] == 'true':
			if static_img is None:
				print(f"  No static image for {base_name}")
			else:
				static_final = static_img.copy()

				# Apply grey boxes defined for static
				static_final = apply_grey_boxes(static_final, static_mask_boxes)

				# If motion_blocks_static enabled, block regions where motion labels exist
				if params.get('motion_blocks_static', 'false') == 'true':
					# blocking boxes come from motion labels (if present)
					static_block_boxes = get_blocking_boxes(motion_label_path, img_w, img_h)
					static_final = apply_blocking_boxes(static_final, static_block_boxes)

				static_img_path = os.path.join('annot_static', 'images', static_split or split, f"{base_name}.jpg")
				os.makedirs(os.path.dirname(static_img_path), exist_ok=True)
				cv2.imwrite(static_img_path, static_final)
				print(f"Regenerated static: {static_img_path}")

		# -----------------------
		# Process motion images (save into annot_motion/images/<split>/)
		# -----------------------
		# Keep original behaviour: regenerate motion when the frame has a motion
		# label file or when save_empty_frames is enabled.
		if motion_split or params['save_empty_frames'] == 'true':
			if motion_img is None:
				print(f"  No motion image for {base_name}")
			else:
				motion_final = motion_img.copy()

				# Apply grey boxes defined for motion
				motion_final = apply_grey_boxes(motion_final, motion_mask_boxes)

				# Apply static blocking if enabled (static annotations can block motion image)
				if params.get('static_blocks_motion', 'false') == 'true':
					static_boxes = get_blocking_boxes(static_label_path, img_w, img_h)
					motion_final = apply_blocking_boxes(motion_final, static_boxes)

				motion_img_path = os.path.join('annot_motion', 'images', motion_split or split, f"{base_name}.jpg")
				os.makedirs(os.path.dirname(motion_img_path), exist_ok=True)
				cv2.imwrite(motion_img_path, motion_final)
				print(f"Regenerated motion: {motion_img_path}")

		# -----------------------
		# Rebuild this frame's secondary crops from the images just regenerated,
		# so each crop ends up in the stream(s) the current switches call for.
		# -----------------------
		if recrop:
			crops_for_frame = crop_index.get((video_name, frame_num))
			if crops_for_frame:
				boxes = boxes_from_labels(base_name, trees, params, img_w, img_h)
				_delete_frame_crops(video_name, frame_num)
				crops_written += recrop_frame(base_name, boxes, crops_for_frame,
											  static_final, motion_final, params)

	if recrop:
		print(f"Rebuilt {crops_written} secondary crop file(s).")
	print("Regeneration loop complete.")



# -----------------------
# CLI & prompt logic
# -----------------------

def choose_ini_path_via_dialog():
	if not _HAS_TK:
		return None
	root = tk.Tk()
	root.withdraw()
	path = filedialog.askopenfilename(title="Select BehaveAI settings INI", filetypes=[("INI files", "*.ini"), ("All files", "*.*")])
	root.destroy()
	return path

def parse_cli(argv):
	"""(positional_path_or_None, relabel_mode_or_None, recrop_bool) from argv[1:]."""
	positional = None
	relabel = None
	recrop = False
	i = 0
	while i < len(argv):
		a = argv[i]
		if a == '--recrop':
			recrop = True
		elif a == '--relabel':
			i += 1
			if i >= len(argv) or argv[i] not in ('merge', 'split'):
				raise SystemExit("--relabel takes 'merge' or 'split'")
			relabel = argv[i]
		elif a.startswith('--relabel='):
			relabel = a.split('=', 1)[1]
			if relabel not in ('merge', 'split'):
				raise SystemExit("--relabel takes 'merge' or 'split'")
		elif a.startswith('--'):
			raise SystemExit(f"Unknown option: {a}")
		elif positional is None:
			positional = a
		else:
			raise SystemExit(f"Unexpected argument: {a}")
		i += 1
	return positional, relabel, recrop


if __name__ == "__main__":
	positional, relabel, recrop = parse_cli(sys.argv[1:])

	# Determine config_path from command-line or prompt
	if positional:
		arg = os.path.abspath(positional)
		if os.path.isdir(arg):
			config_path = os.path.join(arg, "BehaveAI_settings.ini")
		else:
			config_path = arg
	else:
		config_path = choose_ini_path_via_dialog()
		if not config_path:
			# no selection: report and exit
			print("No settings INI selected — exiting.")
			sys.exit(0)

	config_path = os.path.abspath(config_path)
	if not os.path.exists(config_path):
		print(f"Config file not found: {config_path}")
		sys.exit(1)

	# Run regeneration
	start_t = time.time()
	regenerate_annotations(config_path, relabel=relabel, recrop=recrop)
	elapsed = time.time() - start_t
	print(f"Regeneration complete! Elapsed {elapsed:.1f} s")
