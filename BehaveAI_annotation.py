#!/usr/bin/env python3


import os
import sys
import time
import cv2
import numpy as np
import configparser
import random
import csv
import re
from collections import deque
import threading
import copy

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from index_annotations import AnnotationIndex
from behaveai_config import (
	load_secondary_config, NONE_LABEL,
	get_species_list, species_folder, load_ethogram_for_species, load_age_classes,
)
from behaveai_holdout import is_holdout_video


# Try to import YOLO
try:
	from ultralytics import YOLO
except Exception:
	YOLO = None

#--- Configuration parsing ----------
def choose_ini_path_from_dialog():
	root = tk.Tk(); root.withdraw()
	ini_path = filedialog.askopenfilename(
		title="Select BehaveAI settings INI",
		filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
	)
	root.destroy()
	return ini_path

if len(sys.argv) > 1:
	arg = os.path.abspath(sys.argv[1])
	if os.path.isdir(arg):
		config_path = os.path.join(arg, "BehaveAI_settings.ini")
	else:
		config_path = arg
else:
	config_path = choose_ini_path_from_dialog()
	if not config_path:
		tk.messagebox.showinfo("No settings file", "No settings INI selected — exiting.")
		sys.exit(0)

config_path = os.path.abspath(config_path)
if not os.path.exists(config_path):
	try:
		root = tk.Tk(); root.withdraw()
		messagebox.showerror("Missing settings", f"Configuration file not found: {config_path}")
		root.destroy()
	except Exception:
		print(f"Configuration file not found: {config_path}")
	sys.exit(1)

project_dir = os.path.dirname(config_path)
os.chdir(project_dir)
config = configparser.ConfigParser()
config.optionxform = str
config.read(config_path)

def resolve_project_path(value, fallback):
	if value is None or str(value).strip() == '':
		value = fallback
	value = str(value)
	if os.path.isabs(value):
		return os.path.normpath(value)
	return os.path.normpath(os.path.join(project_dir, value))

clips_dir_ini = config['DEFAULT'].get('clips_dir', 'clips')
clips_dir = resolve_project_path(clips_dir_ini, 'clips')

# Second annotation source: input_dir (e.g. Activity_budget folder).
# If the key is missing or empty the source is simply ignored.
input_dir_ini = config['DEFAULT'].get('input_dir', '')
input_dir_for_annotation = resolve_project_path(input_dir_ini, '') if input_dir_ini.strip() else ''




# ---- Species (model 0) + per-species ethogram/age (model 0.5) ----
# The first species keeps every bare/legacy key and folder name below byte-for-byte
# (species_key/species_folder resolve to the unscoped name for species_list[0]), so
# an existing single-species project is completely unaffected until a 2nd species
# is added via the settings GUI.
species_list = get_species_list(config)
_sp_cols = [c.strip() for c in config['DEFAULT'].get('species_colors', '').split(';') if c.strip()]
species_colors = [tuple(map(int, c.split(',')))[::-1] for c in _sp_cols]
species_hotkeys = [key.strip() for key in config['DEFAULT'].get('species_hotkeys', '').split(',')]
species_classes_info = list(zip(species_hotkeys, species_list))
species_class_dict = {ord(key): idx for idx, (key, _) in enumerate(species_classes_info) if key}

# Secondary sentinels: -1 = "not applicable" (primary has no allowed secondary, no crop);
# -2 = explicit "none" (eligible primary, no real secondary -> a __none__ negative crop).
NONE_SEC = -2

# Species/age crops are pooled project-wide (not scoped per active species): they are
# model 0/0.5's own training data, keyed by the crop's own predicted class name -
# exactly like the secondary pool is keyed by secondary name, not by primary.
species_cropped_base_dir = 'annot_species_crop'
age_cropped_base_dir = 'annot_age_crop'


def _load_ethogram_globals(species):
	"""(Re)compute every species-scoped global (primary/secondary/age classes,
	colors, hotkeys, model + dataset folder paths) for `species`. Called once at
	startup for species_list[0] and again whenever the user switches species in
	the new Espèce button group."""
	global primary_static_classes, primary_motion_classes
	global primary_static_colors, primary_motion_colors
	global primary_static_hotkeys, primary_motion_hotkeys
	global primary_classes, primary_colors, primary_hotkeys
	global primary_classes_info, primary_class_dict
	global secondary_classes, secondary_colors, secondary_hotkeys, secondary_map
	global allowed_secondary_idx, hierarchical_mode
	global secondary_classes_info, secondary_class_dict, ignore_secondary
	global age_classes, age_colors, age_hotkeys, age_classes_info, age_class_dict
	global primary_static_project_path, primary_static_model_path, primary_static_yaml_path
	global primary_motion_project_path, primary_motion_model_path, primary_motion_yaml_path
	global secondary_static_project_path, secondary_static_data_path, secondary_static_model_path
	global secondary_motion_project_path, secondary_motion_data_path, secondary_motion_model_path
	global motion_cropped_base_dir, static_cropped_base_dir
	global static_train_images_dir, static_val_images_dir
	global static_train_labels_dir, static_val_labels_dir
	global motion_train_images_dir, motion_val_images_dir
	global motion_train_labels_dir, motion_val_labels_dir

	eth = load_ethogram_for_species(config, species, species_list)
	primary_static_classes = eth['primary_static_classes']
	primary_motion_classes = eth['primary_motion_classes']
	primary_static_colors = eth['primary_static_colors']
	primary_motion_colors = eth['primary_motion_colors']
	primary_static_hotkeys = eth['primary_static_hotkeys']
	primary_motion_hotkeys = eth['primary_motion_hotkeys']

	primary_classes = primary_static_classes + primary_motion_classes
	primary_colors = primary_static_colors + primary_motion_colors
	primary_hotkeys = primary_static_hotkeys + primary_motion_hotkeys
	primary_classes_info = list(zip(primary_hotkeys, primary_classes))
	primary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(primary_classes_info) if key}

	secondary_classes = eth['secondary_classes']
	secondary_colors = eth['secondary_colors']
	secondary_hotkeys = eth['secondary_hotkeys']
	secondary_map = eth['secondary_map']
	allowed_secondary_idx = eth['allowed_secondary_idx']
	hierarchical_mode = eth['hierarchical_mode']
	secondary_classes_info = list(zip(secondary_hotkeys, secondary_classes))
	secondary_class_dict = {ord(key): idx for idx, (key, _) in enumerate(secondary_classes_info) if key}
	ignore_secondary = [primary_classes[i] for i in range(len(primary_classes))
						if i >= len(allowed_secondary_idx) or not allowed_secondary_idx[i]]

	age = load_age_classes(config, species, species_list)
	age_classes = age['age_classes']
	age_colors = age['age_colors']
	age_hotkeys = age['age_hotkeys']
	age_classes_info = list(zip(age_hotkeys, age_classes))
	age_class_dict = {ord(key): idx for idx, (key, _) in enumerate(age_classes_info) if key}

	primary_static_project_path = species_folder('model_primary_static', species, species_list)
	primary_static_model_path = os.path.join(primary_static_project_path, "train", "weights", "best.pt")
	primary_static_yaml_path = species_folder('static_annotations', species, species_list) + '.yaml'

	primary_motion_project_path = species_folder('model_primary_motion', species, species_list)
	primary_motion_model_path = os.path.join(primary_motion_project_path, "train", "weights", "best.pt")
	primary_motion_yaml_path = species_folder('motion_annotations', species, species_list) + '.yaml'

	motion_cropped_base_dir = species_folder('annot_motion_crop', species, species_list)
	static_cropped_base_dir = species_folder('annot_static_crop', species, species_list)

	secondary_static_project_path = None
	secondary_static_data_path = None
	secondary_static_model_path = None
	secondary_motion_project_path = None
	secondary_motion_data_path = None
	secondary_motion_model_path = None
	if hierarchical_mode:
		secondary_static_project_path = species_folder('model_secondary_static', species, species_list)
		secondary_static_data_path = static_cropped_base_dir
		secondary_static_model_path = os.path.join(secondary_static_project_path, "train", "weights", "best.pt")

		secondary_motion_project_path = species_folder('model_secondary_motion', species, species_list)
		secondary_motion_data_path = motion_cropped_base_dir
		secondary_motion_model_path = os.path.join(secondary_motion_project_path, "train", "weights", "best.pt")

	annot_static_dir = species_folder('annot_static', species, species_list)
	annot_motion_dir = species_folder('annot_motion', species, species_list)
	static_train_images_dir = f'{annot_static_dir}/images/train'
	static_val_images_dir = f'{annot_static_dir}/images/val'
	static_train_labels_dir = f'{annot_static_dir}/labels/train'
	static_val_labels_dir = f'{annot_static_dir}/labels/val'

	motion_train_images_dir = f'{annot_motion_dir}/images/train'
	motion_val_images_dir = f'{annot_motion_dir}/images/val'
	motion_train_labels_dir = f'{annot_motion_dir}/labels/train'
	motion_val_labels_dir = f'{annot_motion_dir}/labels/val'


# Read parameters
try:

	dominant_source = config['DEFAULT']['dominant_source'].lower()

	_load_ethogram_globals(species_list[0])

	# Common parameters
	scale_factor = float(config['DEFAULT'].get('scale_factor', '1.0'))
	expA = float(config['DEFAULT'].get('expA', '0.5'))
	expB = float(config['DEFAULT'].get('expB', '0.8'))
	val_frequency = float(config['DEFAULT'].get('val_frequency', '0.1'))

	lum_weight = float(config['DEFAULT'].get('lum_weight', '0.7'))
	strategy = config['DEFAULT'].get('strategy', 'exponential')
	chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
	primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh', '0.5'))
	secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh', '0.5'))
	rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
	line_thickness = int(config['DEFAULT'].get('line_thickness', '1'))
	font_size = float(config['DEFAULT'].get('font_size', '0.5'))
	box_line_scale = float(config['DEFAULT'].get('box_line_scale', '0.5'))
	box_font_scale = float(config['DEFAULT'].get('box_font_scale', '0.35'))
	buttons_per_row = int(config['DEFAULT'].get('buttons_per_row', '8'))
	# ~ cross_blocking = config['DEFAULT']['cross_blocking'].lower()
	iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
	motion_blocks_static = config['DEFAULT']['motion_blocks_static'].lower()
	static_blocks_motion = config['DEFAULT']['static_blocks_motion'].lower()
	save_empty_frames = config['DEFAULT']['save_empty_frames'].lower()
	frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")



if motion_blocks_static not in ('true', 'false'):
	raise ValueError("motion_blocks_static must be 'true' or 'false'")
if static_blocks_motion not in ('true', 'false'):
	raise ValueError("static_blocks_motion must be 'true' or 'false'")
if save_empty_frames not in ('true', 'false'):
	raise ValueError("save_empty_frames must be 'true' or 'false'")

def default_secondary_for(primary_idx):
	"""Default secondary for a primary: -2 ('none') when the primary is secondary-eligible,
	else -1 (not applicable). Making 'none' the default collects negatives automatically."""
	if primary_idx is not None and 0 <= primary_idx < len(allowed_secondary_idx) and allowed_secondary_idx[primary_idx]:
		return NONE_SEC
	return -1

# Espèce (model 0) / Âge (model 0.5): always-valid sticky defaults, unlike
# active_primary/active_secondary which start "pending" (box-first workflow).
# They default to the first configured entry and simply ride along with every
# new box until the annotator clicks a different one - see _apply_classes_to_selected_boxes.
active_species = 0
active_age = 0 if age_classes else -1

# initial selections
active_primary = 0
if len(primary_static_classes) <= 1:
	active_primary = 1
active_secondary = default_secondary_for(active_primary)


def _build_annotation_index():
	return AnnotationIndex(
		static_train_images_dir,
		static_val_images_dir,
		static_train_labels_dir,
		static_val_labels_dir,
		motion_train_images_dir,
		motion_val_images_dir,
		motion_train_labels_dir,
		motion_val_labels_dir,
		motion_cropped_base_dir,
		static_cropped_base_dir,
		clips_dir,
		primary_static_classes,
		primary_classes,
		secondary_classes,
		hierarchical_mode,
		ignore_secondary=ignore_secondary,
		species_list=species_list,
		age_classes=age_classes,
		species_cropped_base_dir=species_cropped_base_dir,
		age_cropped_base_dir=age_cropped_base_dir,
	)


annotation_index = _build_annotation_index()

items = annotation_index.list_images_labels_and_masks()

# Build quick lookup: video_label -> set(of frame numbers that have annotations)
def build_annot_index_map(items_list):
	m = {}
	for it in items_list:
		base = it.get('basename', '')
		if '_' not in base:
			continue
		vlabel, tail = base.rsplit('_', 1)
		try:
			frm = int(tail)
		except Exception:
			continue
		m.setdefault(vlabel, set()).add(frm)
	return m

# initial annotated frames map (used to draw ticks on seek)
annotated_frames_map = build_annot_index_map(items)


# ---- Build the pool of all annotatable frames across all video sources ----

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def _scan_videos_recursive(root_directory):
	"""
	Walk root_directory recursively and return a sorted list of all video
	file paths found at any depth.  Works with any folder structure.
	Returns an empty list if root_directory does not exist.
	"""
	found = []
	if not os.path.isdir(root_directory):
		return found
	for dirpath, _dirnames, filenames in os.walk(root_directory):
		for fname in sorted(filenames):
			if fname.lower().endswith(VIDEO_EXTENSIONS):
				found.append(os.path.join(dirpath, fname))
	return found


def build_frame_pool(clips_directory, frame_window, extra_directories=None):
	"""
	Scan all video files found recursively inside clips_directory (and any
	paths listed in extra_directories) and return a list of annotatable
	(video_path, frame_number) pairs.

	A frame is annotatable if its index >= frame_window - 1, because the
	motion engine needs frame_window preceding frames to compute the diff.
	The last 2 frames are skipped as a safety margin for capture.read().

	Parameters
	----------
	clips_directory : str
		Primary video source (e.g. Training folder).  Scanned recursively.
	frame_window : int
		Minimum number of frames required before a frame is annotatable.
	extra_directories : list[str] | None
		Additional directories to scan recursively (e.g. Activity_budget
		folder).  Each directory is scanned independently; duplicate video
		paths are silently ignored.
	"""
	pool = []
	seen_paths = set()

	# Collect all source directories, filtering out empty / non-existent ones
	sources = [clips_directory]
	if extra_directories:
		for d in extra_directories:
			if d and os.path.isdir(d):
				sources.append(d)

	for source in sources:
		video_files = _scan_videos_recursive(source)
		for vpath in video_files:
			# Avoid adding the same file twice (e.g. if dirs overlap)
			norm = os.path.normpath(vpath)
			if norm in seen_paths:
				continue
			seen_paths.add(norm)

			cap = cv2.VideoCapture(vpath)
			if not cap.isOpened():
				print(f"Warning: could not open {vpath}, skipping.")
				cap.release()
				continue
			n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			cap.release()

			first_valid = frame_window - 1
			last_valid  = n_frames - 3  # safety margin

			if first_valid > last_valid:
				print(f"Warning: {os.path.basename(vpath)} is too short "
					  f"for the current frameWindow ({n_frames} frames), skipping.")
				continue

			for fn in range(first_valid, last_valid + 1):
				pool.append((vpath, fn))

	return pool


def get_unannotated_pool(full_pool, annotation_index):
	"""
	Filter full_pool by removing frames that already have an annotation on disk.

	The annotation filenames follow the convention  <video_label>_<frame_number>.jpg
	so we rebuild the annotated set from the index and exclude matching entries.
	"""
	# Build a set of (video_label, frame_number) pairs that are already annotated
	existing_items = annotation_index.list_images_labels_and_masks()
	annotated_set = set()
	for item in existing_items:
		base = item.get('basename', '')
		if '_' not in base:
			continue
		vlabel, tail = base.rsplit('_', 1)
		try:
			annotated_set.add((vlabel, int(tail)))
		except ValueError:
			continue

	# Filter the pool: keep only entries not yet annotated
	unannotated = []
	for vpath, fn in full_pool:
		vlabel = os.path.splitext(os.path.basename(vpath))[0]
		if (vlabel, fn) not in annotated_set:
			unannotated.append((vpath, fn))
	print(f"Frame pool built: {len(annotated_set)}  annotated frames on {len(full_pool)} annotatable frames across {len(set(p for p,_ in full_pool))} video(s).")
	return unannotated


def pick_random_frame(unannotated_pool):
	"""
	Pick a random (video_path, frame_number) from the unannotated pool.
	Returns (None, None) if the pool is empty.
	"""
	if not unannotated_pool:
		return None, None
	return random.choice(unannotated_pool)


# ---------- CSV time-code navigation ----------

_TIMECODE_RE = re.compile(r'^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$')


def _parse_frame_value(raw, fps):
	"""
	Convert a raw CSV time-code cell into a frame number.

	Accepts either a plain integer frame index ("1530") or an mm:ss /
	hh:mm:ss time-code ("02:15", "01:02:03") which is turned into a frame
	using the video fps.  Returns an int frame number, or None if the value
	cannot be understood.
	"""
	if raw is None:
		return None
	s = str(raw).strip()
	if not s:
		return None

	# Plain integer frame index
	try:
		return int(s)
	except ValueError:
		pass

	# mm:ss or hh:mm:ss time-code
	m = _TIMECODE_RE.match(s)
	if m and fps and fps > 0:
		hours   = int(m.group(1)) if m.group(1) else 0
		minutes = int(m.group(2))
		seconds = float(m.group(3))
		total_seconds = hours * 3600 + minutes * 60 + seconds
		return int(round(total_seconds * fps))

	return None


def parse_timecode_csv(csv_path, full_pool, frame_window):
	"""
	Parse a CSV listing frames / time-codes to annotate and return an ordered
	list of (video_path, frame_number) targets.

	Expected columns (case-insensitive, first match wins):
	  - video : 'video_filename' | 'video' | 'filename'
	  - time  : 'frame' | 'timecode' | 'time' | 'start_frame'

	Each time-code cell may be a plain frame index or an mm:ss / hh:mm:ss
	value (converted via the video fps).  Video names are matched against the
	pool on their stem (filename without extension), case-insensitively.
	Out-of-range frames are clamped; duplicates are removed while preserving
	CSV order.  Returns [] on any structural problem.
	"""
	# Map  stem(lower) -> video_path  from the known pool
	stem_to_path = {}
	for vpath, _fn in full_pool:
		stem = os.path.splitext(os.path.basename(vpath))[0].lower()
		stem_to_path.setdefault(stem, vpath)

	# Per-video (fps, total_frames) memo so we open each file at most once
	video_meta = {}

	def _meta(vpath):
		if vpath not in video_meta:
			cap = cv2.VideoCapture(vpath)
			fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
			nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
			cap.release()
			video_meta[vpath] = (fps, nfr)
		return video_meta[vpath]

	targets = []
	seen = set()

	try:
		with open(csv_path, 'r', newline='', encoding='utf-8-sig') as f:
			# Skip blank lines and comment lines starting with '#'
			rows = [ln for ln in f if ln.strip() and not ln.lstrip().startswith('#')]
		if not rows:
			print("CSV time-codes: file is empty.")
			return []

		reader = csv.DictReader(rows)
		if not reader.fieldnames:
			print("CSV time-codes: no header row found.")
			return []

		# Resolve column names (case-insensitive)
		lower_map = {name.lower().strip(): name for name in reader.fieldnames}

		def _pick(*candidates):
			for c in candidates:
				if c in lower_map:
					return lower_map[c]
			return None

		video_col = _pick('video_filename', 'video', 'filename')
		time_col  = _pick('frame', 'timecode', 'time', 'start_frame')

		if video_col is None or time_col is None:
			print("CSV time-codes: could not find a video column "
				  "(video_filename/video/filename) and a time column "
				  "(frame/timecode/time/start_frame).")
			return []

		for row in reader:
			vname = (row.get(video_col) or '').strip()
			if not vname:
				continue
			stem = os.path.splitext(os.path.basename(vname))[0].lower()
			vpath = stem_to_path.get(stem)
			if vpath is None:
				print(f"CSV time-codes: video '{vname}' not found in project, skipping.")
				continue

			fps, total = _meta(vpath)
			fnum = _parse_frame_value(row.get(time_col), fps)
			if fnum is None:
				print(f"CSV time-codes: unreadable time value "
					  f"'{row.get(time_col)}' for {vname}, skipping.")
				continue

			# Clamp into the annotatable range for this video
			lo = max(0, frame_window - 1)
			hi = max(lo, total - 1) if total > 0 else fnum
			fnum = min(max(fnum, lo), hi)

			key = (vpath, fnum)
			if key in seen:
				continue
			seen.add(key)
			targets.append(key)

	except Exception as e:
		print(f"CSV time-codes: failed to parse '{csv_path}': {e}")
		return []

	print(f"CSV time-codes: loaded {len(targets)} target frame(s) from {os.path.basename(csv_path)}.")
	return targets

# frameWindow logic
frameWindow = 4
if strategy == 'exponential':
	if expA > 0.2 or expB > 0.2:
		frameWindow = 5
	if expA > 0.5 or expB > 0.5:
		frameWindow = 10
	if expA > 0.7 or expB > 0.7:
		frameWindow = 15
	if expA > 0.8 or expB > 0.8:
		frameWindow = 20
	if expA > 0.9 or expB > 0.9:
		frameWindow = 45

raw_buf = deque(maxlen=frameWindow)
frameWindow = frameWindow * (frame_skip + 1)


# ---- Initial random frame selection ----

# Build pool from clips_dir (Training) AND input_dir (Activity_budget) when available
_extra_dirs = [input_dir_for_annotation] if input_dir_for_annotation else []
full_frame_pool = build_frame_pool(clips_dir, frameWindow, extra_directories=_extra_dirs)

if not full_frame_pool:
	_searched = clips_dir
	if _extra_dirs:
		_searched += '\n' + '\n'.join(_extra_dirs)
	root_tmp = tk.Tk(); root_tmp.withdraw()
	messagebox.showinfo("No videos found",
		f"No valid video files found in:\n{_searched}\n\nAdd videos and relaunch.")
	root_tmp.destroy()
	sys.exit(0)

unannotated_pool = get_unannotated_pool(full_frame_pool, annotation_index)

if not unannotated_pool:
	root_tmp = tk.Tk(); root_tmp.withdraw()
	messagebox.showinfo("All annotated",
		"All frames in the clips folder have already been annotated!")
	root_tmp.destroy()
	sys.exit(0)

video_path, _initial_frame = pick_random_frame(unannotated_pool)


capture = cv2.VideoCapture(video_path)
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
video_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Apply the randomly chosen frame (overrides the default frameWindow-based value set below)
# We store it here; frame_number is set a few lines later after frameWindow is computed.
_random_start_frame = _initial_frame

right_frame_width = max(96, int(video_height / 3))
frame_number = min(max(_random_start_frame, frameWindow - 1), total_frames - 1)
frame_updated = True

# state
boxes = []
grey_boxes = []
original_frame = None
fr = None
motion_image = None

# Frame-source navigation state.
#   nav_mode    : 'random' (default behaviour) or 'csv' (walk through CSV time-codes)
#   csv_targets : ordered list of (video_path, frame_number) parsed from a CSV
#   csv_cursor  : index of the next CSV target to present
nav_mode = 'random'
csv_targets = []
csv_cursor = 0

video_label = os.path.splitext(os.path.basename(video_path))[0]

bottom_bar_height = int(20 + font_size * 20)
grey_mode = False
annot_count = 1
auto_ann_switch = 1
show_mode = 1  # 1 = motion false color, -1 = static RGB
zoom_hide = 0
disp_scale_factor = 1.0

last_mouse_move = 0.0
ANIM_STILL_THRESHOLD = 0.5
ANIM_FPS = 8
last_anim_draw = 0.0
ANIM_DT = 1.0 / ANIM_FPS

# Load models
model_static = None
model_motion = None
# Two per-stream secondary classifiers (shared pool of secondary labels).
# Crops are routed by the primary's stream; each model's classes are the union
# of secondaries mapped to that stream's primaries.
secondary_static_model = None
secondary_motion_model = None


def _load_models_for_active_species():
	"""(Re)load the primary/secondary models for whichever species is currently
	active (species_static_model_path etc. are already species-scoped by
	_load_ethogram_globals). Called once at startup and again on species switch."""
	global model_static, model_motion, secondary_static_model, secondary_motion_model
	model_static = None
	model_motion = None
	secondary_static_model = None
	secondary_motion_model = None
	if YOLO is None:
		return
	if os.path.exists(primary_static_model_path):
		try:
			model_static = YOLO(primary_static_model_path)
		except Exception as e:
			print("Failed to load primary static model:", e)
	if os.path.exists(primary_motion_model_path):
		try:
			model_motion = YOLO(primary_motion_model_path)
		except Exception as e:
			print("Failed to load primary motion model:", e)

	if hierarchical_mode:
		if secondary_static_model_path and os.path.exists(secondary_static_model_path):
			try:
				secondary_static_model = YOLO(secondary_static_model_path)
				print('Secondary static model found')
			except Exception as e:
				print("Failed to load secondary static model:", e)
		else:
			print('Secondary static model not found')

		if secondary_motion_model_path and os.path.exists(secondary_motion_model_path):
			try:
				secondary_motion_model = YOLO(secondary_motion_model_path)
				print('Secondary motion model found')
			except Exception as e:
				print("Failed to load secondary motion model:", e)
		else:
			print('Secondary motion model not found')


_load_models_for_active_species()

# Species (model 0) / age (model 0.5) classifiers: single models, project-wide
# (not species-scoped - species_list[0] IS what model 0 discriminates between),
# loaded once. Absent until the user has annotated enough to train them.
model_species = None
model_age = None
if YOLO is not None:
	_species_path = os.path.join('model_species', "train", "weights", "best.pt")
	if os.path.exists(_species_path):
		try:
			model_species = YOLO(_species_path)
			print('Species model found')
		except Exception as e:
			print("Failed to load species model:", e)

	_age_path = os.path.join('model_age', "train", "weights", "best.pt")
	if os.path.exists(_age_path):
		try:
			model_age = YOLO(_age_path)
			print('Age model found')
		except Exception as e:
			print("Failed to load age model:", e)


def classify_secondary(primary_global_idx, x1, y1, x2, y2, static_img, motion_img):
	"""Run the stream-appropriate secondary classifier on a crop and return
	(pool_index, confidence), restricted to the secondaries allowed for this
	primary. Returns (-1, -1.0) when no secondary applies/available."""
	if not hierarchical_mode:
		return -1, -1.0
	if primary_global_idx is None or primary_global_idx < 0 or primary_global_idx >= len(allowed_secondary_idx):
		return -1, -1.0
	allowed = allowed_secondary_idx[primary_global_idx]
	if not allowed:
		return -1, -1.0
	if primary_global_idx < len(primary_static_classes):
		model = secondary_static_model
		crop_src = static_img
	else:
		model = secondary_motion_model
		crop_src = motion_img
	if model is None or crop_src is None:
		return -1, -1.0
	crop = crop_src[y1:y2, x1:x2]
	if crop is None or crop.size == 0:
		return -1, -1.0
	try:
		res = model.predict(crop, verbose=False)
	except Exception:
		return -1, -1.0
	if not res or res[0].probs is None:
		return -1, -1.0
	allowed_names = set(secondary_classes[i] for i in allowed)
	allowed_names.add(NONE_LABEL)   # the classifier may vote "no secondary"
	probs = res[0].probs.data
	best_name, best_conf = None, -1.0
	for m_idx, nm in model.names.items():
		if nm not in allowed_names:
			continue
		try:
			c = float(probs[m_idx])
		except Exception:
			continue
		if c > best_conf:
			best_conf = c
			best_name = nm
	if best_name == NONE_LABEL:
		return NONE_SEC, best_conf   # explicit "none"
	if best_name is None:
		return -1, -1.0
	return secondary_classes.index(best_name), best_conf


def _classify_pooled(model, class_names, x1, y1, x2, y2, crop_src):
	"""Run a plain (non-hierarchical) crop classifier - used for the species
	(model 0) and age (model 0.5) classifiers, which have no allowed-subset
	restriction and no '__none__' sentinel (both are mandatory per box)."""
	if model is None or crop_src is None or not class_names:
		return -1, -1.0
	crop = crop_src[y1:y2, x1:x2]
	if crop is None or crop.size == 0:
		return -1, -1.0
	try:
		res = model.predict(crop, verbose=False)
	except Exception:
		return -1, -1.0
	if not res or res[0].probs is None:
		return -1, -1.0
	probs = res[0].probs.data
	best_name, best_conf = None, -1.0
	for m_idx, nm in model.names.items():
		if nm not in class_names:
			continue
		try:
			c = float(probs[m_idx])
		except Exception:
			continue
		if c > best_conf:
			best_conf = c
			best_name = nm
	if best_name is None:
		return -1, -1.0
	return class_names.index(best_name), best_conf


def classify_species(x1, y1, x2, y2, static_img, motion_img, source):
	"""Run the species classifier (model 0) on a crop. Pooled (not per-stream);
	crop is taken from whichever stream the primary detection came from."""
	crop_src = static_img if source == 'static' else motion_img
	return _classify_pooled(model_species, species_list, x1, y1, x2, y2, crop_src)


def classify_age(x1, y1, x2, y2, static_img, motion_img, source):
	"""Run the age classifier (model 0.5) on a crop, restricted to the active
	species' age classes. Pooled (not per-stream)."""
	crop_src = static_img if source == 'static' else motion_img
	return _classify_pooled(model_age, age_classes, x1, y1, x2, y2, crop_src)


# Helper: convert BGR -> PhotoImage

def cv2_to_photoimage(bgr_img):
	rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
	pil = Image.fromarray(rgb)
	return ImageTk.PhotoImage(pil)

## remove overlapping detections
def non_max_suppression(box_list):
	"""Remove overlapping detections keeping highest confidence box."""
	if len(box_list) == 0:
		return []

	# Calculate overall confidence for each box
	confidences = []
	for box in box_list:
		if hierarchical_mode:
			conf = box[6]
		else:
			conf = box[5]

		confidences.append(conf)

	# Sort by confidence (descending)
	sorted_indices = sorted(range(len(box_list)), key=lambda i: confidences[i], reverse=True)
	suppressed = [False] * len(box_list)
	keep = []

	for i in range(len(sorted_indices)):
		idx_i = sorted_indices[i]
		if suppressed[idx_i]:
			continue

		keep.append(box_list[idx_i])
		box_i = box_list[idx_i]
		coords_i = (box_i[0], box_i[1], box_i[2], box_i[3])

		for j in range(i + 1, len(sorted_indices)):
			idx_j = sorted_indices[j]
			if suppressed[idx_j]:
				continue

			box_j = box_list[idx_j]
			coords_j = (box_j[0], box_j[1], box_j[2], box_j[3])

			if iou(coords_i, coords_j) > iou_thresh:
				suppressed[idx_j] = True

	return keep



def iou(box1, box2):
	xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
	xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
	inter = max(0, xb-xa) * max(0, yb-ya)
	area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
	area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
	prop1 = inter/area1
	prop2 = inter/area2
	if prop1 > prop2:
		return prop1 if prop1 > 0 else 0
	else:
		return prop2 if prop2 > 0 else 0



# ------------------------------
# Load saved labels for a given base (video_label_frame)
# ------------------------------
def norm_to_pixels(xc, yc, bw, bh, w, h):
	cx = float(xc) * w
	cy = float(yc) * h
	bw_p = float(bw) * w
	bh_p = float(bh) * h
	x1 = int(cx - bw_p/2); y1 = int(cy - bh_p/2)
	x2 = int(cx + bw_p/2); y2 = int(cy + bh_p/2)
	x1 = max(0, min(w-1, x1)); y1 = max(0, min(h-1, y1)); x2 = max(0, min(w-1, x2)); y2 = max(0, min(h-1, y2))
	return x1, y1, x2, y2


# Auto-annotate: uses model_static / model_motion and per-primary secondary models
def _classify_species_and_age(x1, y1, x2, y2, source):
	"""Model-assisted species/age suggestion for an auto-detected box: use the
	trained model 0/0.5 when available, else fall back to the sticky active_species/
	active_age default (same rationale as classify_secondary falling back to -1)."""
	species_idx, species_conf = classify_species(x1, y1, x2, y2, fr, motion_image, source)
	if species_idx < 0:
		species_idx, species_conf = active_species, -1.0
	age_idx, age_conf = classify_age(x1, y1, x2, y2, fr, motion_image, source)
	if age_idx < 0:
		age_idx, age_conf = active_age, -1.0
	return species_idx, species_conf, age_idx, age_conf


def auto_annotate_local():
	# Collect all primary detections
	# ~ all_detections = []
	global boxes

	# Primary static detection
	if primary_static_classes and model_static != None:
		results_static = model_static.predict(fr, conf=primary_conf_thresh, verbose=False)
		for box in results_static[0].boxes:
			# ~ coords = tuple(map(int, box.xyxy[0].tolist()))
			class_idx = int(box.cls[0])
			primary_class = primary_static_classes[class_idx]
			conf = float(box.conf[0])
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			sp_idx, sp_conf, ag_idx, ag_conf = _classify_species_and_age(x1, y1, x2, y2, 'static')

			if hierarchical_mode:
				secondary_class_idx, secondary_conf = classify_secondary(
					class_idx, x1, y1, x2, y2, fr, motion_image)
				boxes.append((x1, y1, x2, y2, class_idx, secondary_class_idx, conf, secondary_conf,
							  sp_idx, sp_conf, ag_idx, ag_conf))


			else:
				boxes.append((x1, y1, x2, y2, class_idx, conf, sp_idx, sp_conf, ag_idx, ag_conf))


	# Primary motion detection
	if primary_motion_classes and model_motion != None:
		results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
		for box in results_motion[0].boxes:
			class_idx = int(box.cls[0])
			primary_class = primary_motion_classes[class_idx]
			conf = float(box.conf[0])
			x1, y1, x2, y2 = map(int, box.xyxy[0])
			sp_idx, sp_conf, ag_idx, ag_conf = _classify_species_and_age(x1, y1, x2, y2, 'motion')

			if hierarchical_mode:
				global_primary_idx = class_idx + len(primary_static_classes)
				secondary_class_idx, secondary_conf = classify_secondary(
					global_primary_idx, x1, y1, x2, y2, fr, motion_image)
				boxes.append((x1, y1, x2, y2, global_primary_idx, secondary_class_idx, conf, secondary_conf,
							  sp_idx, sp_conf, ag_idx, ag_conf))

			else:
				boxes.append((x1, y1, x2, y2, class_idx + len(primary_static_classes), conf,
							  sp_idx, sp_conf, ag_idx, ag_conf))

	if boxes:
		boxes = non_max_suppression(boxes)



# draw boxes onto a frame copy
def draw_boxes_on_image(base_img, selected_set=None, thickness=None, font_scale=None,
						 to_screen=None, bounds=None):
	"""
	Draw hierarchical boxes onto a *copy* of base_img.
	- Outer rectangle uses primary color (slightly thicker)
	- Inner rectangle uses secondary color (if present)
	- Label shows PRIMARY conf SECONDARY conf (primary uppercased)
	- Unclassified boxes (primary_cls < 0) are drawn dashed-red as "pending".
	- The selected box gets a bright highlight.

	`thickness`/`font_scale` default to the configured line_thickness/font_size.
	`to_screen(vx, vy) -> (sx, sy)` maps native video coordinates into base_img's
	coordinate space (identity by default); pass the final on-screen scaled
	composite as base_img plus a to_screen that folds in zoom + canvas-fit
	scale so lines/text are drawn crisp, at their real final pixel size,
	rather than baked onto the native frame and blurred by a later resize.
	`bounds` (max_x, max_y), if given, skips boxes entirely outside it — so
	they don't bleed into an adjacent region of a shared canvas (e.g. the
	zoom column to the right of the main display).
	"""
	th = line_thickness if thickness is None else thickness
	fs = font_size if font_scale is None else font_scale
	xf = to_screen if to_screen is not None else (lambda vx, vy: (vx, vy))
	out = base_img.copy()

	def _species_age_label(box):
		"""Species/age always occupy the box's last 4 slots - see
		_load_ethogram_globals / index_annotations._attach_species_and_age_crops.
		The 10-element floor is the shortest real box shape (non-hierarchical);
		anything shorter has no species/age fields at all. Returned as separate
		lines (not joined with '/') so each stacks on its own row."""
		species_idx = box[-4] if len(box) >= 10 else -1
		age_idx = box[-2] if len(box) >= 10 else -1
		parts = []
		if species_idx is not None and 0 <= species_idx < len(species_list):
			parts.append(species_list[species_idx])
		if age_idx is not None and 0 <= age_idx < len(age_classes):
			parts.append(age_classes[age_idx])
		return parts

	def _draw_stacked_label(lines, lx, ly, color):
		"""Draw `lines` (top to bottom) stacked immediately above (lx, ly),
		each on its own row, with a single shared background rectangle."""
		lines = [l for l in lines if l]
		if not lines:
			return
		sizes = [cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for l in lines]
		total_h = sum(h for _w, h in sizes) + th * 4 * len(lines)
		max_w = max(w for w, _h in sizes)
		cv2.rectangle(out, (lx - th, ly - total_h), (lx + max_w + th*2, ly), (0, 0, 0), -1)
		y = ly - th * 2
		for line, (_w, h) in zip(reversed(lines), reversed(sizes)):
			cv2.putText(out, line, (lx, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
			y -= h + th * 4

	for bi, box in enumerate(boxes):

		primary_cls = box[4] if len(box) > 4 else -1
		sx1, sy1 = xf(box[0], box[1])
		sx2, sy2 = xf(box[2], box[3])
		x1, y1, x2, y2 = int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2))
		if bounds is not None:
			max_x, max_y = bounds
			if x2 < 0 or y2 < 0 or x1 > max_x or y1 > max_y:
				continue
		is_selected = (selected_set is not None and bi in selected_set)
		species_age_lines = _species_age_label(box)

		# --- Pending / unclassified box (no primary chosen yet) ---
		if primary_cls is None or primary_cls < 0 or primary_cls >= len(primary_classes):
			cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), max(1, th))
			cv2.putText(out, "? choose primary", (x1, max(y1 - 6, 10)),
						cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 255), th, cv2.LINE_AA)
			if is_selected:
				cv2.rectangle(out, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 1)
			continue

		if hierarchical_mode:
			secondary_cls = box[5] if len(box) > 5 else -1
			conf = box[6] if len(box) > 6 else -1
			sec_conf = box[7] if len(box) > 7 else -1
			pcol = primary_colors[primary_cls] if primary_cls < len(primary_colors) else (255,255,255)
			has_secondary = (secondary_cls is not None and secondary_cls >= 0 and secondary_cls < len(secondary_colors))
			scol = secondary_colors[secondary_cls] if has_secondary else pcol

			# draw outer box (primary) slightly thicker
			outer_th = max(1, th + 2)
			cv2.rectangle(out, (x1-outer_th, y1-outer_th), (x2+outer_th, y2+outer_th), pcol, outer_th)

			# draw inner box (secondary) only when a secondary is set
			if has_secondary:
				cv2.rectangle(out, (x1, y1), (x2, y2), scol, th)

			# compose label: PRIMARY (upper) [+ conf], then secondary [+ conf]
			label = f"{primary_classes[primary_cls].upper()}"
			if conf != -1 and conf is not None:
				try:
					label = label + f" {conf:.2f}"
				except Exception:
					pass

			if has_secondary and secondary_cls < len(secondary_classes):
				label2 = f"{secondary_classes[secondary_cls]}"
				if sec_conf != -1 and sec_conf is not None:
					try:
						label2 = label2 + f" {sec_conf:.2f}"
					except Exception:
						pass
				label = label + " " + label2

			# draw label background and text - species/age line above primary/secondary
			_draw_stacked_label([*species_age_lines, label], x1, y1, pcol)
			if is_selected:
				cv2.rectangle(out, (x1-outer_th-2, y1-outer_th-2), (x2+outer_th+2, y2+outer_th+2), (0, 255, 255), 1)

		else:
			conf = box[5] if len(box) > 5 else -1
			pcol = primary_colors[primary_cls] if primary_cls < len(primary_colors) else (255,255,255)
			cv2.rectangle(out, (x1, y1), (x2, y2), pcol, th)
			label = f"{primary_classes[primary_cls]}"
			if conf != -1 and conf is not None:
				try:
					label = label + f" {conf:.2f}"
				except Exception:
					pass
			_draw_stacked_label([*species_age_lines, label], x1, y1, pcol)
			if is_selected:
				cv2.rectangle(out, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 1)

	# grey masks (as previously)
	for gx1, gy1, gx2, gy2 in grey_boxes:
		gsx1, gsy1 = xf(gx1, gy1)
		gsx2, gsy2 = xf(gx2, gy2)
		overlay = out.copy()
		cv2.rectangle(overlay, (int(round(gsx1)), int(round(gsy1))), (int(round(gsx2)), int(round(gsy2))), (128,128,128), -1)
		cv2.addWeighted(overlay, 0.5, out, 0.5, 0, out)

	return out


def save_annotation():
	global annot_count, boxes
	if original_frame is None or (not boxes and not grey_boxes) and save_empty_frames == 'false':
		return
	# Drop unclassified (pending) boxes — they have no primary class yet.
	pending = [b for b in boxes if (len(b) <= 4 or b[4] is None or b[4] < 0)]
	if pending:
		print(f"Skipping {len(pending)} unclassified box(es) — assign a primary class first.")
		boxes = [b for b in boxes if not (len(b) <= 4 or b[4] is None or b[4] < 0)]
	# Whole-video assignment to train/validation (deterministic, stable — see
	# behaveai_holdout.is_holdout_video). Every frame from this video lands in
	# the same split, avoiding cross-split leakage of near-duplicate frames.
	is_val = is_holdout_video(video_label, val_frequency)

	motion_target_img_dir = motion_val_images_dir if is_val else motion_train_images_dir
	motion_target_lbl_dir = motion_val_labels_dir if is_val else motion_train_labels_dir

	static_target_img_dir = static_val_images_dir if is_val else static_train_images_dir
	static_target_lbl_dir = static_val_labels_dir if is_val else static_train_labels_dir

	annot_type = 'validation' if is_val else 'training'


	# Save image with grey overlays
	motion_ann_frame = original_frame.copy()
	for gx1, gy1, gx2, gy2 in grey_boxes:
		cv2.rectangle(motion_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)

	static_ann_frame = fr.copy()
	for gx1, gy1, gx2, gy2 in grey_boxes:
		cv2.rectangle(static_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)


	# fill static boxes with grey (to avoid cross-training on similar motion & static things)
	static_count = 0
	motion_count = 0

	for box in boxes:
		if hierarchical_mode:
			x1, y1, x2, y2, primary_cls = box[0], box[1], box[2], box[3], box[4]
		else:
			x1, y1, x2, y2, primary_cls = box[0], box[1], box[2], box[3], box[4]
		if primary_cls < len(primary_static_classes): # primary class is static
			static_count +=1
			if static_blocks_motion == 'true':
				cv2.rectangle(motion_ann_frame, (x1, y1), (x2, y2), (128, 128, 128), -line_thickness)
		else:  # primary class is motion
			motion_count +=1
			if motion_blocks_static == 'true':
				cv2.rectangle(static_ann_frame, (x1, y1), (x2, y2), (128, 128, 128), -line_thickness)



	h, w = original_frame.shape[:2]
	base_filename = f"{video_label}_{frame_number}"

	## delete any existing annotations for this frame
	deleted = annotation_index.delete_frame(base_filename)
	if deleted:
		print("Overwriting existing annotation")

	if static_count > 0 or save_empty_frames == 'true': # don't save blank images
		static_img_path = os.path.join(static_target_img_dir, f"{base_filename}.jpg")
		cv2.imwrite(static_img_path, static_ann_frame)


		# Save static labels
		static_ann_path = os.path.join(static_target_lbl_dir, f"{base_filename}.txt")
		with open(static_ann_path, 'w') as f:
			for box in boxes:
				if hierarchical_mode:
					x1, y1, x2, y2, primary_cls = box[0], box[1], box[2], box[3], box[4]
				else:
					x1, y1, x2, y2, primary_cls = box[0], box[1], box[2], box[3], box[4]
				if primary_cls < len(primary_static_classes):
					# ~ if y1 < button_height:
						# ~ continue
					xc = (x1 + x2) / 2 / w
					yc = (y1 + y2) / 2 / h
					bw = abs(x2 - x1) / w
					bh = abs(y2 - y1) / h
					f.write(f"{primary_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

	if motion_count > 0 or save_empty_frames == 'true': # don't save blank images
		img_path = os.path.join(motion_target_img_dir, f"{base_filename}.jpg")
		cv2.imwrite(img_path, motion_ann_frame)

		# Save motion labels
		motion_ann_path = os.path.join(motion_target_lbl_dir, f"{base_filename}.txt")
		with open(motion_ann_path, 'w') as f:
			for box in boxes:
				if hierarchical_mode:
					x1, y1, x2, y2, primary_cls = box[0], box[1], box[2], box[3], box[4]
				else:
					x1, y1, x2, y2, primary_cls = box[0], box[1], box[2], box[3], box[4]
				if primary_cls >= len(primary_static_classes):
					# ~ if y1 < button_height:
						# ~ continue
					xc = (x1 + x2) / 2 / w
					yc = (y1 + y2) / 2 / h
					bw = abs(x2 - x1) / w
					bh = abs(y2 - y1) / h
					f.write(f"{primary_cls - len(primary_static_classes) } {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

	if static_blocks_motion == 'true':
		# ann_frames need re-making after greying out the static for the above primary training
		motion_ann_frame = original_frame.copy()
		for gx1, gy1, gx2, gy2 in grey_boxes:
			cv2.rectangle(motion_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)

	if motion_blocks_static == 'true':
		static_ann_frame = fr.copy()
		for gx1, gy1, gx2, gy2 in grey_boxes:
			cv2.rectangle(static_ann_frame, (gx1, gy1), (gx2, gy2), (128, 128, 128), -line_thickness)


	if hierarchical_mode:

		# Remove stale secondary crops for THIS frame before rewriting them. A box relabeled to a
		# different (or no) secondary must not keep its old crop: crop-to-box matching on reload is
		# primary-agnostic (by x/y only), so a leftover crop would mis-attach (e.g. Graze + sternal).
		for _bcd in (static_cropped_base_dir, motion_cropped_base_dir):
			if not _bcd or not os.path.isdir(_bcd):
				continue
			for _sec in os.listdir(_bcd):
				_sdir = os.path.join(_bcd, _sec)
				if not os.path.isdir(_sdir):
					continue
				for _fn in list(os.listdir(_sdir)):
					if not _fn.lower().endswith(('.jpg', '.jpeg', '.png')):
						continue
					_parts = os.path.splitext(_fn)[0].split('_')
					if len(_parts) < 4:
						continue
					try:
						_fr = int(_parts[-3]); _vl = '_'.join(_parts[:-3])
					except Exception:
						continue
					if _vl == video_label and _fr == frame_number:
						try:
							os.remove(os.path.join(_sdir, _fn))
						except Exception:
							pass

		for box in boxes:
			x1, y1, x2, y2, primary_cls, secondary_cls = box[0], box[1], box[2], box[3], box[4], box[5]
			if primary_cls is None or primary_cls < 0 or primary_cls >= len(primary_classes):
				continue

			# Determine the crop class folder: a real secondary, or the explicit
			# "none" negative (__none__) for a secondary-eligible primary. Untouched
			# boxes (-1) / primaries with no allowed secondary produce no crop.
			if secondary_cls is not None and 0 <= secondary_cls < len(secondary_classes):
				secondary_class_name = secondary_classes[secondary_cls]
			elif secondary_cls == NONE_SEC and primary_cls < len(allowed_secondary_idx) and allowed_secondary_idx[primary_cls]:
				secondary_class_name = NONE_LABEL
			else:
				continue

			# Route the crop to the stream of the primary only; pooled layout
			# annot_<stream>_crop/<secondary | __none__>/ (no per-primary level).
			if primary_cls < len(primary_static_classes):
				base_crop_dir = static_cropped_base_dir
				crop_src = static_ann_frame
			else:
				base_crop_dir = motion_cropped_base_dir
				crop_src = motion_ann_frame

			crop = crop_src[y1:y2, x1:x2]
			if crop is None or crop.size == 0:
				continue

			class_dir = os.path.join(base_crop_dir, secondary_class_name)
			os.makedirs(class_dir, exist_ok=True)
			crop_path = os.path.join(class_dir, f"{video_label}_{frame_number}_{x1}_{y1}.jpg")
			cv2.imwrite(crop_path, crop)

	# Species (model 0) / age (model 0.5) crops - pooled project-wide, unconditional
	# (mandatory per box, unlike the optional/hierarchical-only secondary above).
	for box in boxes:
		x1, y1, x2, y2, primary_cls = box[0], box[1], box[2], box[3], box[4]
		if primary_cls is None or primary_cls < 0 or primary_cls >= len(primary_classes):
			continue
		species_idx, age_idx = box[-4], box[-2]

		if primary_cls < len(primary_static_classes):
			crop_src = static_ann_frame
		else:
			crop_src = motion_ann_frame
		crop = crop_src[y1:y2, x1:x2]
		if crop is None or crop.size == 0:
			continue

		if species_idx is not None and 0 <= species_idx < len(species_list):
			species_class_dir = os.path.join(species_cropped_base_dir, species_list[species_idx])
			os.makedirs(species_class_dir, exist_ok=True)
			cv2.imwrite(os.path.join(species_class_dir, f"{video_label}_{frame_number}_{x1}_{y1}.jpg"), crop)

		if age_idx is not None and 0 <= age_idx < len(age_classes):
			age_class_dir = os.path.join(age_cropped_base_dir, age_classes[age_idx])
			os.makedirs(age_class_dir, exist_ok=True)
			cv2.imwrite(os.path.join(age_class_dir, f"{video_label}_{frame_number}_{x1}_{y1}.jpg"), crop)


	# Create mask directories
	static_mask_dir = static_target_lbl_dir.replace('labels', 'masks')
	motion_mask_dir = motion_target_lbl_dir.replace('labels', 'masks')
	os.makedirs(static_mask_dir, exist_ok=True)
	os.makedirs(motion_mask_dir, exist_ok=True)

	# Save grey box coordinates to mask files
	mask_content = ""
	for gx1, gy1, gx2, gy2 in grey_boxes:
		mask_content += f"{gx1} {gy1} {gx2} {gy2}\n"

	# Write mask files
	mask_filename = f"{base_filename}.mask.txt"
	static_mask_path = os.path.join(static_mask_dir, mask_filename)
	motion_mask_path = os.path.join(motion_mask_dir, mask_filename)

	with open(static_mask_path, 'w') as f:
		f.write(mask_content)
	with open(motion_mask_path, 'w') as f:
		f.write(mask_content)

	print(f"Saved #{annot_count} frame {frame_number} -> {annot_type}")

	annot_count += 1

# ---- Prefetch cache ----
# Stores the precomputed result for the next frame so that transitioning is instant.
# Structure: {
#   'video_path': str,
#   'frame_number': int,
#   'fr': np.ndarray,           # raw static frame
#   'motion_image': np.ndarray, # motion false-colour frame
#   'original_frame': np.ndarray,
#   'raw_buf': list,            # list of frames for animation buffer
#   'video_width': int,
#   'video_height': int,
#   'total_frames': int,
# }
_prefetch_cache = {}
_prefetch_lock  = threading.Lock()
_prefetch_thread = None

# ---- Last annotated frame cache (for Shift+Enter to go back) ----
# Stores the same structure as _prefetch_cache for the frame just saved.
_last_annotated_cache = {}
# The next frame that has already been chosen and is being prefetched.
# Set by load_next_random_frame so the cache lookup is deterministic.
_prefetched_target = (None, None)  # (video_path, frame_number)
# Target display size used when computing the prefetch composite.
# Updated by loop() to match the current canvas size.
_display_size_hint = (800, 600)

def _compute_frame_data(vpath, fnum):
	"""
	Open vpath, seek to fnum, compute static frame + motion image + raw_buf.
	Returns a dict ready to be stored in _prefetch_cache, or None on failure.
	This function is designed to run in a background thread — it uses only
	local variables and does not touch any global state.
	"""
	try:
		cap = cv2.VideoCapture(vpath)
		if not cap.isOpened():
			return None

		n_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		vid_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		start_frame = fnum - frameWindow + 1
		if start_frame < 0:
			cap.release()
			return None

		cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

		prev_frames = [None] * 3
		local_raw_buf = []
		local_fr = None
		local_motion = None
		local_original = None
		gray = None
		diffs = None
		frame_count = 0

		for i in range(frameWindow):
			ret, raw_frame = cap.read()
			if not ret:
				break
			if frame_count == 0:
				local_fr = raw_frame.copy()
				if scale_factor != 1.0:
					local_fr = cv2.resize(local_fr, (0, 0), fx=scale_factor, fy=scale_factor)
				local_raw_buf.append(local_fr.copy())
				gray = cv2.cvtColor(local_fr, cv2.COLOR_BGR2GRAY)
				if i == 0:
					prev_frames = [gray.copy()] * 3
					frame_count += 1
					if frame_count > frame_skip:
						frame_count = 0
					continue
				diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
				if strategy == 'exponential':
					prev_frames[0] = gray
					prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, 1 - expA, 0)
					prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, 1 - expB, 0)
				else:
					prev_frames[2] = prev_frames[1]
					prev_frames[1] = prev_frames[0]
					prev_frames[0] = gray
			frame_count += 1
			if frame_count > frame_skip:
				frame_count = 0

		cap.release()

		if diffs is None or local_fr is None:
			return None

		if chromatic_tail_only == 'true':
			tb = cv2.subtract(diffs[0], diffs[1])
			tr = cv2.subtract(diffs[2], diffs[1])
			tg = cv2.subtract(diffs[1], diffs[0])
			blue  = cv2.addWeighted(gray, lum_weight, tb, rgb_multipliers[2], motion_threshold)
			green = cv2.addWeighted(gray, lum_weight, tg, rgb_multipliers[1], motion_threshold)
			red   = cv2.addWeighted(gray, lum_weight, tr, rgb_multipliers[0], motion_threshold)
		else:
			blue  = cv2.addWeighted(gray, lum_weight, diffs[0], rgb_multipliers[2], motion_threshold)
			green = cv2.addWeighted(gray, lum_weight, diffs[1], rgb_multipliers[1], motion_threshold)
			red   = cv2.addWeighted(gray, lum_weight, diffs[2], rgb_multipliers[0], motion_threshold)

		local_motion   = cv2.merge((blue, green, red)).astype(np.uint8)
		local_original = local_motion.copy()

		# ---- Precompute the scaled composite for instant display ----
		# We compute both the motion view and the static view at display size
		# so redraw() can use them directly without any resize work.
		disp_w, disp_h = _display_size_hint
		disp_w = max(1, disp_w)
		disp_h = max(1, disp_h)

		# Resize both views to display size
		motion_disp  = cv2.resize(local_motion, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
		static_disp  = cv2.resize(local_fr,     (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

		# Build the zoom column (3 panes stacked vertically)
		widget_size  = max(32, disp_h // 3)
		zoom_col     = np.zeros((widget_size * 3, widget_size, 3), dtype=np.uint8)

		cx = vid_width  // 2
		cy = vid_height // 2

		def _quick_crop(src, cx, cy, out_size):
			"""Centre crop at native resolution then resize — no padding needed for preview."""
			h, w = src.shape[:2]
			half = out_size // 2
			x1 = max(0, cx - half); y1 = max(0, cy - half)
			x2 = min(w, cx + half); y2 = min(h, cy + half)
			crop = src[y1:y2, x1:x2]
			if crop.size == 0:
				return np.zeros((out_size, out_size, 3), dtype=np.uint8)
			return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)

		# Top pane: static zoom
		zoom_col[0:widget_size, 0:widget_size]               = _quick_crop(local_fr,     cx, cy, widget_size)
		# Mid pane: motion zoom
		zoom_col[widget_size:widget_size*2, 0:widget_size]   = _quick_crop(local_motion, cx, cy, widget_size)
		# Bottom pane: blank (animation — cannot be precomputed, changes over time)
		zoom_col[widget_size*2:widget_size*3, 0:widget_size] = np.zeros((widget_size, widget_size, 3), dtype=np.uint8)

		# Assemble composite (motion view by default — user can toggle with Space)
		composite_w = disp_w + widget_size
		composite_h = max(disp_h, widget_size * 3)
		composite   = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)
		composite[0:disp_h,      0:disp_w]    = motion_disp
		composite[0:widget_size*3, disp_w:]   = zoom_col

		# Scale composite to a safe canvas size
		# (actual canvas size may differ slightly — a final cheap resize in redraw handles it)
		c_w = composite_w
		c_h = composite_h
		scale_w = float(c_w) / float(max(1, composite_w))
		scale_h = float(c_h) / float(max(1, composite_h))
		scale   = min(scale_w, scale_h) if scale_w > 0 and scale_h > 0 else 1.0

		scaled_w = max(1, int(round(composite_w * scale)))
		scaled_h = max(1, int(round(composite_h * scale)))
		scaled_composite = cv2.resize(composite, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

		# Convert to PhotoImage-ready format (RGB numpy array)
		# We store as numpy RGB rather than PhotoImage because PhotoImage must be
		# created on the main thread in Tkinter.
		composite_rgb = cv2.cvtColor(scaled_composite, cv2.COLOR_BGR2RGB)

		return {
			'video_path':      vpath,
			'frame_number':    fnum,
			'fr':              local_fr,
			'motion_image':    local_motion,
			'original_frame':  local_original,
			'raw_buf':         local_raw_buf,
			'video_width':     vid_width,
			'video_height':    vid_height,
			'total_frames':    n_frames,
			'composite_rgb':   composite_rgb,
			'composite_scale': scale,
			'display_size':    (disp_w, disp_h),
			'widget_size':     widget_size,
		}

	except Exception as e:
		print(f"Prefetch error for {vpath} frame {fnum}: {e}")
		return None

def _start_prefetch(vpath, fnum):
	"""
	Launch a background thread to precompute the next frame.
	Only one prefetch thread runs at a time — if one is already running it is
	left to finish (we do not cancel it, just let the new result overwrite).
	"""
	global _prefetch_thread, _prefetch_cache

	def _worker():
		result = _compute_frame_data(vpath, fnum)
		with _prefetch_lock:
			_prefetch_cache.clear()
			if result is not None:
				_prefetch_cache.update(result)

	# Clear stale cache immediately so we don't accidentally use old data
	with _prefetch_lock:
		_prefetch_cache.clear()

	_prefetch_thread = threading.Thread(target=_worker, daemon=True)
	_prefetch_thread.start()


def go_to_frame(app_instance, target_vpath, target_fnum):
	"""
	Jump to an explicit (video_path, frame_number) target, switching video if
	needed.  Used by the CSV time-code navigation mode.  Loads synchronously
	(no prefetch) and refreshes the UI / seek bar / annotation ticks.
	"""
	global video_path, capture, frame_number, video_label
	global total_frames, video_width, video_height, frame_updated
	global annotated_frames_map, items
	global fr, original_frame, motion_image, raw_buf
	global _last_annotated_cache

	# Save current frame into the "last annotated" cache before moving on
	if original_frame is not None and fr is not None:
		_last_annotated_cache = {
			'video_path':     video_path,
			'frame_number':   frame_number,
			'fr':             fr.copy(),
			'motion_image':   motion_image.copy() if motion_image is not None else None,
			'original_frame': original_frame.copy(),
			'raw_buf':        list(raw_buf),
			'video_width':    video_width,
			'video_height':   video_height,
			'total_frames':   total_frames,
		}

	# Switch video if the target lives in a different file
	if target_vpath != video_path:
		capture.release()
		capture      = cv2.VideoCapture(target_vpath)
		total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
		video_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
		video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		video_path   = target_vpath
		video_label  = os.path.splitext(os.path.basename(video_path))[0]
		app_instance.root.title(f"BehaveAI — {os.path.basename(video_path)}")
		app_instance.seek.configure(to=max(0, total_frames - 1))

	frame_number  = min(max(target_fnum, frameWindow - 1), total_frames - 1)
	frame_updated = True

	# Refresh UI
	items = annotation_index.list_images_labels_and_masks()
	annotated_frames_map = build_annot_index_map(items)
	app_instance.seek.set(frame_number)
	try:
		app_instance.frame_var.set(f'Frame {frame_number}')
	except Exception:
		pass
	app_instance.draw_seek_ticks()


def load_next_target(app_instance):
	"""
	Advance to the next frame according to the active navigation mode.

	In 'csv' mode, walk the parsed CSV targets in order via go_to_frame();
	when the list is exhausted, fall back to random navigation.  In 'random'
	mode (the default), defer to load_next_random_frame().
	"""
	global csv_cursor

	if nav_mode == 'csv' and csv_targets:
		if csv_cursor >= len(csv_targets):
			messagebox.showinfo("CSV finished",
				"All time-codes from the CSV have been visited. Switching back to random mode.")
			app_instance.set_nav_mode('random')
			load_next_random_frame(app_instance)
			return
		vpath, fnum = csv_targets[csv_cursor]
		csv_cursor += 1
		go_to_frame(app_instance, vpath, fnum)
		# Update the button label so progress (i/N) stays in sync
		app_instance.update_source_button()
	else:
		load_next_random_frame(app_instance)


def load_next_random_frame(app_instance):
	"""
	Transition to the next random unannotated frame.
	If _prefetched_target matches a ready cache entry, the transition is instant.
	Otherwise falls back to synchronous loading.
	After loading, picks the NEXT frame deterministically and starts prefetching it.
	"""
	global video_path, capture, frame_number, video_label
	global total_frames, video_width, video_height, frame_updated
	global annotated_frames_map, items
	global fr, original_frame, motion_image, raw_buf
	global _last_annotated_cache, _prefetched_target

	# Save current frame into the "last annotated" cache before moving on
	if original_frame is not None and fr is not None:
		_last_annotated_cache = {
			'video_path':     video_path,
			'frame_number':   frame_number,
			'fr':             fr.copy(),
			'motion_image':   motion_image.copy() if motion_image is not None else None,
			'original_frame': original_frame.copy(),
			'raw_buf':        list(raw_buf),
			'video_width':    video_width,
			'video_height':   video_height,
			'total_frames':   total_frames,
		}

	# Refresh the annotated pool
	new_unannotated = get_unannotated_pool(full_frame_pool, annotation_index)
	if not new_unannotated:
		messagebox.showinfo("All annotated",
			"Congratulations! All frames have been annotated.")
		app_instance.root.destroy()
		return

	# Use the pre-chosen target if it exists, otherwise pick randomly now
	target_vpath, target_fnum = _prefetched_target
	if target_vpath is None or (target_vpath, target_fnum) not in set(new_unannotated):
		# No valid pre-chosen target — pick randomly
		target_vpath, target_fnum = pick_random_frame(new_unannotated)

	new_video_path  = target_vpath
	new_frame_number = target_fnum

	# ---- Try to use prefetch cache ----
	cache_hit = False
	with _prefetch_lock:
		cached_vpath  = _prefetch_cache.get('video_path')
		cached_fnum   = _prefetch_cache.get('frame_number')
		if cached_vpath == new_video_path and cached_fnum == new_frame_number:
			fr             = _prefetch_cache['fr']
			motion_image   = _prefetch_cache['motion_image']
			original_frame = _prefetch_cache['original_frame']
			raw_buf.clear()
			for f in _prefetch_cache['raw_buf']:
				raw_buf.append(f)
			cache_hit = True
			print(f"Cache hit — {os.path.basename(new_video_path)} frame {new_frame_number}")

	if cache_hit:
		if new_video_path != video_path:
			capture.release()
			capture      = cv2.VideoCapture(new_video_path)
			video_path   = new_video_path
			video_label  = os.path.splitext(os.path.basename(video_path))[0]
			app_instance.root.title(f"BehaveAI — {os.path.basename(video_path)}")

		# Apply cached video metadata (use cache values, avoid re-reading from capture)
		with _prefetch_lock:
			video_width  = _prefetch_cache['video_width']
			video_height = _prefetch_cache['video_height']
			total_frames = _prefetch_cache['total_frames']

		app_instance.seek.configure(to=max(0, total_frames - 1))
		frame_number  = new_frame_number
		frame_updated = False

		# ---- Instant display using the pre-rendered composite ----
		with _prefetch_lock:
			composite_rgb   = _prefetch_cache.get('composite_rgb')
			cached_scale    = _prefetch_cache.get('composite_scale', 1.0)
			cached_disp_sz  = _prefetch_cache.get('display_size', (800, 600))
			cached_widget_sz = _prefetch_cache.get('widget_size', 200)

		if composite_rgb is not None:
			try:
				from PIL import Image as PilImage
				pil_img  = PilImage.fromarray(composite_rgb)
				tk_photo = ImageTk.PhotoImage(pil_img)

				# Store on app so Tkinter doesn't garbage-collect it
				app_instance.tk_img = tk_photo
				app_instance.composite_scale  = cached_scale
				app_instance.display_size     = cached_disp_sz

				h, w = composite_rgb.shape[:2]
				app_instance.canvas.config(scrollregion=(0, 0, w, h))
				app_instance.canvas.delete('all')
				app_instance.canvas.create_image(0, 0, image=tk_photo, anchor='nw')
				# Full redraw will follow on the next loop() tick with animation etc.
			except Exception as e:
				print(f"Fast display failed: {e}")
				app_instance.redraw()
			else:
				app_instance.redraw()

	else:
		print(f"Cache miss — loading {os.path.basename(new_video_path)} frame {new_frame_number} synchronously")
		if new_video_path != video_path:
			capture.release()
			capture      = cv2.VideoCapture(new_video_path)
			total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
			video_width  = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
			video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
			video_path   = new_video_path
			video_label  = os.path.splitext(os.path.basename(video_path))[0]
			app_instance.root.title(f"BehaveAI — {os.path.basename(video_path)}")
			app_instance.seek.configure(to=max(0, total_frames - 1))

		frame_number  = min(max(new_frame_number, frameWindow - 1), total_frames - 1)
		frame_updated = True

	# Refresh UI
	items = annotation_index.list_images_labels_and_masks()
	annotated_frames_map = build_annot_index_map(items)
	app_instance.seek.set(frame_number)
	try:
		app_instance.frame_var.set(f'Frame {frame_number}')
	except Exception:
		pass
	app_instance.draw_seek_ticks()

	# ---- Pick the NEXT target and schedule prefetch with a short delay ----
	# The delay lets the main thread finish rendering the current frame
	# before the prefetch thread starts competing for the video decoder.
	pool_after = get_unannotated_pool(full_frame_pool, annotation_index)
	pool_after = [(vp, fn) for vp, fn in pool_after
				  if not (vp == new_video_path and fn == new_frame_number)]
	if pool_after:
		next_vpath, next_fnum = pick_random_frame(pool_after)
		_prefetched_target = (next_vpath, next_fnum)
		# Delay prefetch by 2 seconds so the UI is responsive first
		app_instance.root.after(2000, lambda vp=next_vpath, fn=next_fnum: _start_prefetch(vp, fn))
	else:
		_prefetched_target = (None, None)


# ---------- Tk UI (composite single-image display) ----------
class AnnotatorTk:
	def __init__(self, root):
		self.root = root
		root.title(f"BehaveAI — {os.path.basename(video_path)}")

		# Size the window to fit the current screen (any size); keep it resizable so
		# the composite display scales down to whatever space is available.
		sw = root.winfo_screenwidth()
		sh = root.winfo_screenheight()
		default_w = max(1000, int(video_width * 1.2))
		default_h = max(700, int(video_height * 1.2))
		win_w = min(default_w, sw)
		win_h = min(default_h, max(400, sh - 60))  # leave room for the taskbar
		root.geometry(f"{win_w}x{win_h}+0+0")
		root.minsize(min(900, max(400, sw - 40)), min(600, max(300, sh - 80)))

		# F11 toggles fullscreen for users who prefer it.
		self._fullscreen = False
		def _toggle_fullscreen(event=None):
			self._fullscreen = not self._fullscreen
			try:
				root.attributes('-fullscreen', self._fullscreen)
			except Exception:
				pass
		root.bind_all('<F11>', _toggle_fullscreen)

		# main layout
		self.main = tk.Frame(root)
		self.main.pack(fill='both', expand=True)

		# left container which holds the single composite canvas
		self.left = tk.Frame(self.main)
		self.left.pack(side='left', fill='both', expand=True)
		self.left.pack_propagate(False)

		# conservative initial canvas size to avoid early thrash
		self.canvas = tk.Canvas(self.left, bg='black', highlightthickness=0,
								width=min(800, video_width), height=min(600, video_height))
		self.canvas.pack(fill='both', expand=True)

		# --- bottom control bar (seek + grey toggle) ---
		self.controls = tk.Frame(self.left)
		self.controls.pack(fill='x', pady=(4, 2))

		# grey toggle (left)
		self.grey_btn = tk.Button(self.controls, text="Grey (g)", width=10, command=self.toggle_grey)
		self.grey_btn.pack(side='left', padx=4)

		# frame-source selector (random frames vs CSV time-codes)
		self.source_btn = tk.Button(self.controls, text="Source: Random", width=18,
									command=self.open_source_menu)
		self.source_btn.pack(side='left', padx=4)

		# frame number label (shows current frame number)
		self.frame_var = tk.StringVar(value=str(frame_number))
		self.frame_label = tk.Label(self.controls, textvariable=self.frame_var, width=8, anchor='w')
		self.frame_label.pack(side='left', padx=(0,6))

		# container for tickline + seek scale so ticks sit *above* the slider
		self.seek_container = tk.Frame(self.controls)
		self.seek_container.pack(side='left', fill='x', expand=True, padx=4)

		# small tick canvas sitting above the actual scale (height can be tuned)
		self.seek_ticks = tk.Canvas(self.seek_container, height=8, bg=self.controls.cget('bg'), highlightthickness=0)
		self.seek_ticks.pack(fill='x', padx=0, pady=(0,1))

		self.seek_ticks.bind('<Configure>', lambda e: self.draw_seek_ticks())


		# the real seek scale below the tick rail
		self.seek = ttk.Scale(
			self.seek_container,
			from_=0,
			to=max(0, total_frames - 1),
			orient='horizontal',
			command=self.on_seek
		)
		self.seek.pack(fill='x', expand=True)

		self.buttons_frame = tk.Frame(self.left)
		self.buttons_frame.pack(side='bottom', fill='x', pady=(4,4))

		# Status banner guiding the box-first workflow (set in update_button_states)
		self.status_label = tk.Label(self.buttons_frame, text='', anchor='w', fg='#444')
		self.status_label.pack(side='top', fill='x', padx=2)

		self.primary_buttons = []     # (btn, color_hex, global_primary_idx)
		self.secondary_buttons = []   # (btn, color_hex, pool_idx)
		self.species_buttons = []     # (btn, color_hex, species_idx)
		self.age_buttons = []         # (btn, color_hex, age_idx)

		# Number of buttons per row — controlled via Settings > Display Settings
		BUTTONS_PER_ROW = buttons_per_row
		n_static = len(primary_static_classes)

		# --- Species group (model 0): always-valid sticky selection, pre-selected
		# to the first configured species. Hidden when only one species is defined
		# (today's single-species projects look exactly as before). Live reload of
		# the primary/secondary/age ethogram on species switch is a follow-up TODO
		# for when a 2nd species actually exists - selecting the (only) species
		# today is a no-op beyond keeping active_species in sync.
		self.species_frame = tk.LabelFrame(self.buttons_frame, text='Espèce')
		if len(species_list) > 1:
			for idx, name in enumerate(species_list):
				color_hex = None
				if idx < len(species_colors):
					bgr = species_colors[idx]
					color_hex = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
				key = species_classes_info[idx][0] if idx < len(species_classes_info) else ''
				label = "{} ({})".format(name, key) if key else name
				btn = tk.Button(self.species_frame, text=label, width=14, relief='raised',
								command=lambda i=idx: self.select_species(i))
				btn.grid(row=0, column=idx, padx=2, pady=2)
				self.species_buttons.append((btn, color_hex, idx))
			self.species_frame.pack(fill='x', pady=(0,2))

		# --- Age group (model 0.5): same always-valid sticky pattern, scoped to
		# the active species' age_classes. Hidden when no age classes are defined.
		self.age_frame = tk.LabelFrame(self.buttons_frame, text='Âge')
		if age_classes:
			for idx, name in enumerate(age_classes):
				color_hex = None
				if idx < len(age_colors):
					bgr = age_colors[idx]
					color_hex = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
				key = age_classes_info[idx][0] if idx < len(age_classes_info) else ''
				label = "{} ({})".format(name, key) if key else name
				btn = tk.Button(self.age_frame, text=label, width=12, relief='raised',
								command=lambda i=idx: self.select_age(i))
				btn.grid(row=0, column=idx, padx=2, pady=2)
				self.age_buttons.append((btn, color_hex, idx))
			self.age_frame.pack(fill='x', pady=(0,2))

		# --- Primary groups: two titled frames (Static / Motion) ---
		self.primary_static_frame = tk.LabelFrame(self.buttons_frame, text='Primary — Static')
		self.primary_motion_frame = tk.LabelFrame(self.buttons_frame, text='Primary — Motion')
		self.primary_static_frame.pack(fill='x', pady=(0,2))
		self.primary_motion_frame.pack(fill='x', pady=(0,2))

		static_pos = 0
		motion_pos = 0
		for idx, name in enumerate(primary_classes):
			if name == '0':
				continue
			color_hex = None
			if idx < len(primary_colors):
				bgr = primary_colors[idx]
				color_hex = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
			key = primary_classes_info[idx][0] if idx < len(primary_classes_info) else ''
			label = "{} ({})".format(name, key) if key else name
			if idx < n_static:
				parent = self.primary_static_frame
				grid_row, grid_col = static_pos // BUTTONS_PER_ROW, static_pos % BUTTONS_PER_ROW
				static_pos += 1
			else:
				parent = self.primary_motion_frame
				grid_row, grid_col = motion_pos // BUTTONS_PER_ROW, motion_pos % BUTTONS_PER_ROW
				motion_pos += 1
			btn = tk.Button(parent, text=label, width=12, relief='raised',
							command=lambda i=idx: self.select_primary(i))
			btn.grid(row=grid_row, column=grid_col, padx=2, pady=2)
			self.primary_buttons.append((btn, color_hex, idx))

		# Hide empty primary group frames
		if static_pos == 0:
			self.primary_static_frame.pack_forget()
		if motion_pos == 0:
			self.primary_motion_frame.pack_forget()

		# --- Secondary group: one titled frame, buttons pre-created, shown on demand ---
		self.secondary_frame = tk.LabelFrame(self.buttons_frame, text='Secondary (optional)')
		self._secondary_per_row = BUTTONS_PER_ROW
		if hierarchical_mode:
			for idx, name in enumerate(secondary_classes):
				color_hex = None
				if idx < len(secondary_colors):
					bgr = secondary_colors[idx]
					color_hex = '#%02x%02x%02x' % (bgr[2], bgr[1], bgr[0])
				key = secondary_classes_info[idx][0] if idx < len(secondary_classes_info) else ''
				label = "{} ({})".format(name, key) if key else name
				btn = tk.Button(self.secondary_frame, text=label, width=12, relief='raised',
								command=lambda i=idx: self.select_secondary(i))
				# Pre-created but not gridded; refresh_secondary_buttons() shows the allowed subset.
				self.secondary_buttons.append((btn, color_hex, idx))
				# Explicit "none" option (pool index -2). Shown first for eligible primaries;
				# selected by default so untouched eligible boxes become __none__ negatives.
			none_btn = tk.Button(self.secondary_frame, text="none (n)", width=12, relief='raised',
								 command=self.select_none)
			self.secondary_buttons.append((none_btn, None, NONE_SEC))


		# bind events
		self.canvas.bind('<ButtonPress-1>', self.on_mouse_down)
		self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
		self.canvas.bind('<ButtonRelease-1>', self.on_mouse_up)
		self.canvas.bind('<Button-3>', self.on_right_click)
		self.canvas.bind('<Motion>', self.on_motion)
		# Mouse-wheel zoom (Windows/macOS use <MouseWheel>; X11 uses Button-4/5).
		self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)
		self.canvas.bind('<Button-4>', self.on_mouse_wheel)
		self.canvas.bind('<Button-5>', self.on_mouse_wheel)
		# Middle mouse button (wheel click) resets the zoom.
		self.canvas.bind('<Button-2>', self.on_reset_zoom)

		root.bind_all('<Key>', self.on_key_all)
		# ~ root.bind_all('<Left>', lambda e: self.key_step(-1))
		# ~ root.bind_all('<Right>', lambda e: self.key_step(1))
		root.bind_all('<space>', lambda e: self.toggle_show_mode())
		root.bind_all('<Return>', self._on_return_key)
		root.bind_all('<Escape>', self.reset_selection)

		# drawing/display state
		# ~ self.display_size = (video_width, video_height)
		self.display_size = (min(800, video_width), min(600, video_height))
		self.tk_img = None
		self.last_mouse = None
		self.drawing = False
		self.start_canvas_xy = None
		# Indices into `boxes` of the boxes currently selected for (re)labelling.
		# Multiple boxes can be selected so the same behaviour is assigned to all.
		self.selected_boxes = set()

		# Mouse-wheel zoom state (1.0 = no zoom). zoom_cx/zoom_cy are the zoom
		# centre in video coordinates; _zoom_crop is the visible video rectangle.
		self.zoom = 1.0
		self.zoom_cx = video_width / 2
		self.zoom_cy = video_height / 2
		self._zoom_crop = (0, 0, video_width, video_height)

		# Keep the display size hint in sync for the prefetch thread
		global _display_size_hint
		_display_size_hint = self.display_size

		# small layout tuning: padding between main and zoom column when composing
		self._composite_gap = 8

		# schedule loop
		self.root.after(30, self.loop)
		self.refresh_secondary_buttons()
		self.update_button_states()


	# button handlers
	def select_primary(self, class_idx):
		global active_primary, active_secondary, grey_mode, show_mode
		active_primary = class_idx
		grey_mode = False
		show_mode = -1 if active_primary < len(primary_static_classes) else 1
		# Drop the current secondary if it is not allowed for this primary; fall back
		# to the primary's default ('none' when eligible, else not-applicable).
		allowed = allowed_secondary_idx[class_idx] if 0 <= class_idx < len(allowed_secondary_idx) else []
		if active_secondary not in allowed:
			active_secondary = default_secondary_for(class_idx)
		self._apply_classes_to_selected_boxes()
		self.refresh_secondary_buttons()
		self.update_button_states()
		self.redraw()

	def select_secondary(self, class_idx):
		global active_secondary, grey_mode
		grey_mode = False
		# Toggle: clicking the active secondary again returns to 'none' (-2), not
		# 'untouched' (-1), so the box still produces a __none__ negative crop.
		active_secondary = default_secondary_for(active_primary) if active_secondary == class_idx else class_idx
		self._apply_classes_to_selected_boxes()
		self.update_button_states()
		self.redraw()

	def select_none(self):
		"""Explicitly mark the selected box(es) as having no secondary (-2)."""
		global active_secondary, grey_mode
		grey_mode = False
		active_secondary = NONE_SEC
		self._apply_classes_to_selected_boxes()
		self.update_button_states()
		self.redraw()

	def select_species(self, idx):
		"""Espèce (model 0): always-valid sticky selection - takes effect on
		selected/pending boxes immediately, independent of primary/secondary state."""
		global active_species
		active_species = idx
		self._apply_classes_to_selected_boxes()
		self.update_button_states()
		self.redraw()

	def select_age(self, idx):
		"""Âge (model 0.5): same always-valid sticky pattern as select_species."""
		global active_age
		active_age = idx
		self._apply_classes_to_selected_boxes()
		self.update_button_states()
		self.redraw()

	def reset_selection(self, event=None):
		"""Escape: deselect all boxes and go back to step 1 (no model selected)."""
		global active_primary, active_secondary
		active_primary = -1
		active_secondary = -1
		self.selected_boxes = set()
		self.refresh_secondary_buttons()
		self.update_button_states()
		self.redraw()

	def _apply_classes_to_selected_boxes(self):
		"""Re-label every selected/pending box with the active classes. Species/age
		are always-valid stickies and are rewritten unconditionally (so clicking
		Espèce/Âge takes effect even before a primary has been chosen); primary/
		secondary are only rewritten once a primary is actually active (box-first
		workflow) - otherwise the box's existing primary/secondary is preserved."""
		global boxes
		has_primary = active_primary is not None and active_primary >= 0
		for idx in list(getattr(self, 'selected_boxes', set())):
			if idx is None or idx < 0 or idx >= len(boxes):
				continue
			b = boxes[idx]
			x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
			if hierarchical_mode:
				primary_cls = active_primary if has_primary else b[4]
				secondary_cls = active_secondary if has_primary else (b[5] if len(b) > 5 else -1)
				c1 = b[6] if len(b) > 6 else -1
				c2 = b[7] if len(b) > 7 else -1
				sp_conf = b[9] if len(b) > 9 else -1.0
				ag_conf = b[11] if len(b) > 11 else -1.0
				boxes[idx] = (x1, y1, x2, y2, primary_cls, secondary_cls, c1, c2,
							  active_species, sp_conf, active_age, ag_conf)
			else:
				primary_cls = active_primary if has_primary else b[4]
				c1 = b[5] if len(b) > 5 else -1
				sp_conf = b[7] if len(b) > 7 else -1.0
				ag_conf = b[9] if len(b) > 9 else -1.0
				boxes[idx] = (x1, y1, x2, y2, primary_cls, c1,
							  active_species, sp_conf, active_age, ag_conf)

	def refresh_secondary_buttons(self):
		"""Show only the secondaries allowed for the active primary (fast grid/forget)."""
		if not hierarchical_mode or not self.secondary_buttons:
			return
		allowed = []
		if active_primary is not None and 0 <= active_primary < len(allowed_secondary_idx):
			allowed = allowed_secondary_idx[active_primary]
		for btn, _col, _idx in self.secondary_buttons:
			btn.grid_forget()
		if not allowed:
			self.secondary_frame.pack_forget()
			return
		# 'none' (NONE_SEC) shown first, then the secondaries allowed for this primary.
		display = [NONE_SEC] + list(allowed)
		pool_to_btn = {idx: btn for (btn, _c, idx) in self.secondary_buttons}
		per_row = self._secondary_per_row
		pos = 0
		for pool_idx in display:
			btn = pool_to_btn.get(pool_idx)
			if btn is None:
				continue
			btn.grid(row=pos // per_row, column=pos % per_row, padx=2, pady=2)
			pos += 1
		if not self.secondary_frame.winfo_ismapped():
			self.secondary_frame.pack(fill='x', pady=(0,2))

	def toggle_grey(self):
		global grey_mode
		grey_mode = not grey_mode
		self.update_button_states()

	# ---- Frame-source selection (random vs CSV time-codes) ----
	def update_source_button(self):
		"""Refresh the Source button label to reflect the active nav mode."""
		try:
			if nav_mode == 'csv' and csv_targets:
				self.source_btn.config(
					text=f"Source: CSV ({min(csv_cursor, len(csv_targets))}/{len(csv_targets)})")
			else:
				self.source_btn.config(text="Source: Random")
		except Exception:
			pass

	def set_nav_mode(self, mode):
		global nav_mode
		nav_mode = mode if mode in ('random', 'csv') else 'random'
		self.update_source_button()

	def _project_timecodes_dir(self):
		"""
		Return the project's timecodes/ directory (creating it and a starter
		example CSV on first use), derived from clips_dir's parent.
		"""
		try:
			project_dir = os.path.dirname(os.path.normpath(clips_dir))
			tc_dir = os.path.join(project_dir, 'timecodes')
			os.makedirs(tc_dir, exist_ok=True)
			example = os.path.join(tc_dir, 'example_timecodes.csv')
			if not os.path.exists(example):
				with open(example, 'w', newline='', encoding='utf-8') as f:
					f.write(
						"# BehaveAI - list of time-codes to annotate.\n"
						"# video_filename : name of the video file in clips/ (with or without extension).\n"
						"# timecode       : integer frame index (e.g. 1530) OR mm:ss (e.g. 01:02) or hh:mm:ss.\n"
						"# behaviour      : optional memo column (ignored by the tool).\n"
						"video_filename,timecode,behaviour\n"
						"my_video_01.mp4,1530,grazing\n"
						"my_video_01.mp4,02:15,walking\n"
						"my_video_02.mp4,00:42,fighting\n"
					)
			return tc_dir
		except Exception as e:
			print(f"Could not prepare timecodes folder: {e}")
			return None

	def open_source_menu(self):
		"""Popup letting the user pick the frame source: random or a CSV of time-codes."""
		global csv_targets, csv_cursor
		menu = tk.Menu(self.root, tearoff=0)
		menu.add_command(label="Random frames",
						 command=lambda: self.set_nav_mode('random'))
		menu.add_command(label="Load a time-code CSV…",
						 command=self._load_csv_source)
		try:
			x = self.source_btn.winfo_rootx()
			y = self.source_btn.winfo_rooty() + self.source_btn.winfo_height()
			menu.tk_popup(x, y)
		finally:
			menu.grab_release()

	def _load_csv_source(self):
		global csv_targets, csv_cursor
		initial = self._project_timecodes_dir() or os.path.dirname(os.path.normpath(clips_dir))
		path = filedialog.askopenfilename(
			title="Choose a time-code CSV",
			initialdir=initial,
			filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
		if not path:
			return
		targets = parse_timecode_csv(path, full_frame_pool, frameWindow)
		if not targets:
			messagebox.showerror(
				"Invalid CSV",
				"No usable time-code was found in this CSV.\n"
				"Check the columns (video_filename + frame/timecode) and that the "
				"video names match the project's files.")
			return
		csv_targets = targets
		csv_cursor = 0
		self.set_nav_mode('csv')
		# Jump straight to the first target
		load_next_target(self)

	def update_button_states(self):
		for btn, col, cls in self.species_buttons:
			if cls == active_species:
				btn.config(relief='sunken')
				if col:
					try:
						btn.config(bg=col)
					except Exception:
						pass
			else:
				btn.config(relief='raised', bg='#888888')
		for btn, col, cls in self.age_buttons:
			if cls == active_age:
				btn.config(relief='sunken')
				if col:
					try:
						btn.config(bg=col)
					except Exception:
						pass
			else:
				btn.config(relief='raised', bg='#888888')
		for btn, col, cls in self.primary_buttons:
			if cls == active_primary:
				btn.config(relief='sunken')
				if col:
					try:
						btn.config(bg=col)
					except Exception:
						pass
			else:
				btn.config(relief='raised', bg='#888888')
		for btn, col, cls in self.secondary_buttons:
			if cls == active_secondary:
				btn.config(relief='sunken')
				if col:
					try:
						btn.config(bg=col)
					except Exception:
						pass
			else:
				btn.config(relief='raised', bg='#888888')
		self.grey_btn.config(relief='sunken' if grey_mode else 'raised')
		self._update_status_banner()

	def _update_status_banner(self):
		try:
			prefix_parts = []
			if 0 <= active_species < len(species_list):
				prefix_parts.append(species_list[active_species])
			if age_classes and 0 <= active_age < len(age_classes):
				prefix_parts.append(age_classes[active_age])
			prefix = "  |  ".join(prefix_parts)
			if prefix:
				prefix = prefix + "  ||  "

			if active_primary is None or active_primary < 0:
				txt = prefix + 'Step 1 — Draw a box, then choose a primary behaviour'
			else:
				pname = primary_classes[active_primary] if active_primary < len(primary_classes) else '?'
				if hierarchical_mode and active_primary < len(allowed_secondary_idx) and allowed_secondary_idx[active_primary]:
					if active_secondary is not None and active_secondary >= 0 and active_secondary < len(secondary_classes):
						sname = secondary_classes[active_secondary]
						txt = prefix + "Primary: {}  |  Secondary: {}  (Esc = reset)".format(pname, sname)
					else:
						txt = prefix + "Primary: {}  |  Secondary: none (optional)  (Esc = reset)".format(pname)
				else:
					txt = prefix + "Primary: {}  (Esc = reset)".format(pname)
			self.status_label.config(text=txt)
		except Exception:
			pass

	def draw_seek_ticks(self):
		"""Draw small ticks for annotated frames and a red cursor for current frame."""
		try:
			self.seek_ticks.delete('all')
		except Exception:
			return

		w = self.seek_ticks.winfo_width()
		if w <= 2:
			# widget not yet realised — try again shortly
			self.root.after(100, self.draw_seek_ticks)
			return

		# get annotated frames for this video's video_label
		ann_set = annotated_frames_map.get(video_label, set())
		if not ann_set:
			return

		# draw ticks (color / height are adjustable)
		for frm in ann_set:
			if frm < 0 or frm >= max(1, total_frames):
				continue
			x = int(round((frm / float(max(1, total_frames - 1))) * (w - 1)))
			# short yellow tick (top-down)
			# ~ self.seek_ticks.create_line(x, 0, x, 6, fill='yellow', width=1)
			self.seek_ticks.create_line(x, 0, x, 10, fill='red', width=2)

		# draw current-frame cursor
		cur_x = int(round((frame_number / float(max(1, total_frames - 1))) * (w - 1)))
		# ~ self.seek_ticks.create_line(cur_x, 0, cur_x, 7, fill='red', width=2)
		self.seek_ticks.create_line(cur_x, 0, cur_x, 10, fill='black', width=2)

	def refresh_annotation_index_map(self):
		"""Rebuild global `items` and annotated_frames_map from the shared index."""
		try:
			global items, annotated_frames_map
			items = annotation_index.list_images_labels_and_masks()
			annotated_frames_map = build_annot_index_map(items)
		except Exception:
			pass

	def jump_to_annotated(self, direction):
		"""Jump to previous (direction=-1) or next (direction=+1) annotated frame for current video_label.
		   If none found, do nothing.
		"""
		try:
			ann_set = sorted(annotated_frames_map.get(video_label, []))
			if not ann_set:
				return
			cur = int(frame_number)
			if direction > 0:
				# next annotated frame strictly greater than cur
				for frm in ann_set:
					if frm > cur:
						self.seek.set(frm)
						self.on_seek(str(frm))
						return
				# wrap to first
				self.seek.set(ann_set[0])
				self.on_seek(str(ann_set[0]))
			else:
				# previous annotated frame strictly less than cur
				for frm in reversed(ann_set):
					if frm < cur:
						self.seek.set(frm)
						self.on_seek(str(frm))
						return
				# wrap to last
				self.seek.set(ann_set[-1])
				self.on_seek(str(ann_set[-1]))
		except Exception:
			pass


	def on_seek(self, val):
		global frame_number, frame_updated
		try:
			frame_number = int(float(val))
		except Exception:
			frame_number = 0
		frame_updated = True
		try:
			self.frame_var.set(f'Frame {str(frame_number)}')
		except Exception:
			pass
		# redraw ticks to show current cursor
		try:
			self.draw_seek_ticks()
		except Exception:
			pass



	def canvas_to_video(self, canvas_point):
		"""
		Map a canvas (x,y) into video coordinates (vx, vy).
		Accounts for the composite image being uniformly scaled to fit the canvas.
		Top-left anchored (composite drawn at 0,0).
		"""
		cx, cy = canvas_point
		c_w = self.canvas.winfo_width() or 1
		c_h = self.canvas.winfo_height() or 1

		# fallback values if redraw hasn't set them yet
		disp_w, disp_h = getattr(self, 'display_size', (video_width, video_height))
		scale = getattr(self, 'composite_scale', 1.0)

		# scaled displayed video region (left part of composite)
		scaled_disp_w = max(1, int(round(disp_w * scale)))
		scaled_disp_h = max(1, int(round(disp_h * scale)))

		# if click is outside scaled main display, clamp to nearest edge
		if cx < 0: cx = 0
		if cy < 0: cy = 0

		# only map if inside scaled main display; if outside we still return nearest edge point
		# map back to display coords then to video coords
		display_x = min(cx, scaled_disp_w - 1) / scale
		display_y = min(cy, scaled_disp_h - 1) / scale

		# Account for the active mouse-wheel zoom crop (x0,y0 origin, cw,ch size).
		x0, y0, cw, ch = getattr(self, '_zoom_crop', (0, 0, video_width, video_height))
		vx = x0 + display_x * (cw / float(max(1, disp_w)))
		vy = y0 + display_y * (ch / float(max(1, disp_h)))
		return (vx, vy)



	def video_to_canvas(self, vx, vy):
		disp_w, disp_h = self.display_size
		cx = int(round((vx * disp_w / float(video_width))))
		cy = int(round((vy * disp_h / float(video_height))))
		return (cx, cy)

	# drawing handlers
	def on_mouse_down(self, event):
		self.drawing = True
		self.start_canvas_xy = (event.x, event.y)
		self.last_mouse = (event.x, event.y)

	def on_mouse_drag(self, event):
		if not self.drawing:
			return
		self.last_mouse = (event.x, event.y)
		self.redraw(temp_rect=(self.start_canvas_xy, (event.x, event.y)))

	def on_mouse_up(self, event):
		global active_primary, active_secondary
		if not self.drawing:
			return
		self.drawing = False
		start_v = self.canvas_to_video(self.start_canvas_xy)
		end_v = self.canvas_to_video((event.x, event.y))
		x1, x2 = sorted([int(round(start_v[0])), int(round(end_v[0]))])
		y1, y2 = sorted([int(round(start_v[1])), int(round(end_v[1]))])
		x1 = max(0, min(video_width-1, x1)); x2 = max(0, min(video_width-1, x2))
		y1 = max(0, min(video_height-1, y1)); y2 = max(0, min(video_height-1, y2))
		if abs(x2-x1) > 5 and abs(y2-y1) > 5:
			if grey_mode:
				grey_boxes.append((x1, y1, x2, y2))
			else:
				# Once the current selection has been classified, drawing a new box
				# starts a fresh selection; otherwise keep accumulating so several
				# boxes can be drawn and assigned the same behaviour at once.
				if active_primary is not None and active_primary >= 0:
					self.selected_boxes = set()
				active_primary = -1
				active_secondary = -1
				# Species/age are always-valid stickies (unlike primary/secondary,
				# a new box is born already carrying the current active_species/age).
				if hierarchical_mode:
					boxes.append((x1, y1, x2, y2, -1, -1, -1, -1, active_species, -1.0, active_age, -1.0))
				else:
					boxes.append((x1, y1, x2, y2, -1, -1, active_species, -1.0, active_age, -1.0))
				self.selected_boxes.add(len(boxes) - 1)
				self.refresh_secondary_buttons()
				self.update_button_states()
		elif not grey_mode:
			# A click (no real drag): select the box under the cursor to relabel it.
			self._select_box_at(x1, y1)
		self.redraw()

	def on_mouse_wheel(self, event):
		"""Zoom the annotation view in/out around the cursor with the mouse wheel."""
		# Determine scroll direction across platforms.
		direction = 0
		if getattr(event, 'delta', 0):
			direction = 1 if event.delta > 0 else -1
		elif getattr(event, 'num', None) == 4:
			direction = 1
		elif getattr(event, 'num', None) == 5:
			direction = -1
		if direction == 0:
			return
		# Zoom towards the video point currently under the cursor.
		try:
			vx, vy = self.canvas_to_video((event.x, event.y))
		except Exception:
			vx, vy = video_width / 2, video_height / 2
		factor = 1.25 if direction > 0 else (1.0 / 1.25)
		self.zoom = max(1.0, min(8.0, self.zoom * factor))
		if self.zoom <= 1.0:
			self.zoom = 1.0
			self.zoom_cx, self.zoom_cy = video_width / 2, video_height / 2
		else:
			self.zoom_cx, self.zoom_cy = vx, vy
		self.redraw()

	def on_reset_zoom(self, event=None):
		"""Middle mouse button (wheel click): reset the zoom to fit."""
		self.zoom = 1.0
		self.zoom_cx, self.zoom_cy = video_width / 2, video_height / 2
		self.redraw()

	def _select_box_at(self, x, y):
		"""Toggle the topmost box under (x, y) in the multi-selection.

		Clicking a box adds it to the selection (or removes it if already
		selected); the active classes follow the clicked box so the buttons
		reflect it. Escape clears the whole selection.
		"""
		global active_primary, active_secondary, show_mode
		for i in range(len(boxes)-1, -1, -1):
			bx1, by1, bx2, by2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
			if bx1 <= x <= bx2 and by1 <= y <= by2:
				if i in self.selected_boxes:
					self.selected_boxes.discard(i)
				else:
					self.selected_boxes.add(i)
					b = boxes[i]
					active_primary = b[4] if len(b) > 4 else -1
					active_secondary = b[5] if (hierarchical_mode and len(b) > 5) else -1
					if active_primary is not None and 0 <= active_primary < len(primary_classes):
						show_mode = -1 if active_primary < len(primary_static_classes) else 1
				self.refresh_secondary_buttons()
				self.update_button_states()
				return

	def on_right_click(self, event):
		v = self.canvas_to_video((event.x, event.y))
		x, y = int(v[0]), int(v[1])
		removed = False
		for i in range(len(boxes)-1, -1, -1):
			bx1, by1, bx2, by2 = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
			if bx1 <= x <= bx2 and by1 <= y <= by2:
				del boxes[i]; removed = True
				# Keep the selection indices consistent after deletion.
				self.selected_boxes = {(j - 1 if j > i else j) for j in self.selected_boxes if j != i}
				break
		if not removed:
			for i in range(len(grey_boxes)-1, -1, -1):
				gx1, gy1, gx2, gy2 = grey_boxes[i]
				if gx1 <= x <= gx2 and gy1 <= y <= gy2:
					del grey_boxes[i]; break
		self.redraw()

	def on_motion(self, event):
		self.last_mouse = (event.x, event.y)
		self.redraw()

	# keyboard
	def on_key_all(self, event):
		global active_primary, active_secondary, grey_mode, boxes, grey_boxes, frame_number, frame_updated, show_mode

		ch = event.char
		ks = event.keysym

		# Frame step - step larger when Shift is held (event.state & 0x1 tests Shift mask)
		# Support CTRL + Left/Right to jump to previous/next annotated frame (event.state & 0x4 tests CTRL mask on X11)
		if ks == 'Left':
			# CTRL jump to previous annotated frame
			if (event.state & 0x4):
				self.jump_to_annotated(-1)
				return
			step = -10 if (event.state & 0x1) else -1
			self.key_step(step)
			return
		if ks == 'Right':
			# CTRL jump to next annotated frame
			if (event.state & 0x4):
				self.jump_to_annotated(+1)
				return
			step = 10 if (event.state & 0x1) else 1
			self.key_step(step)
			return


		if ch and ch != '0':
			c_ord = ord(ch)
			# 'n' -> explicit "none" secondary (only for secondary-eligible primaries).
			if ch in ('n', 'N'):
				if active_primary is not None and 0 <= active_primary < len(allowed_secondary_idx) and allowed_secondary_idx[active_primary]:
					self.select_none()
				return
			# Espèce/Âge hotkeys take priority over primary/secondary.
			if c_ord in species_class_dict:
				self.select_species(species_class_dict[c_ord])
				return
			if c_ord in age_class_dict:
				self.select_age(age_class_dict[c_ord])
				return
			# Primary hotkey takes priority; reuse select_primary (sticky + relabel + refresh).
			if c_ord in primary_class_dict:
				self.select_primary(primary_class_dict[c_ord])
				return
			# Secondary hotkey: only when allowed for the active primary (toggle on repeat).
			if c_ord in secondary_class_dict:
				pool_idx = secondary_class_dict[c_ord]
				allowed = allowed_secondary_idx[active_primary] if (active_primary is not None and 0 <= active_primary < len(allowed_secondary_idx)) else []
				if pool_idx in allowed:
					self.select_secondary(pool_idx)
				return

		# Ctrl+P — skip current frame without saving, advance to next frame
		# (event.state & 0x4 tests the Control mask.)
		if ks in ('p', 'P') and (event.state & 0x4):
			boxes.clear()
			grey_boxes.clear()
			self.selected_boxes = set()
			load_next_target(self)
			return

		if ch == 'u':
			if grey_mode:
				if grey_boxes: grey_boxes.pop()
			elif boxes:
				boxes.pop()
			self.redraw()
			return

		if ch == 'g':
			self.toggle_grey()
			return

		# Shift+Enter — go back to the last annotated frame
		if ks == 'Return' and (event.state & 0x1):
			self._restore_last_annotated()
			return

		if ks == 'Return':
			save_annotation()
			boxes.clear()
			grey_boxes.clear()
			load_next_target(self)
			return

		if ks == 'Delete':
			print("\nWARNING: This will delete ALL files for this frame!")
			print("Press ENTER to confirm, any other key to cancel...")
			# Wait for confirmation using a simple key binding approach
			self.root.bind('<Return>', self.confirm_delete)
			self.root.bind('<Escape>', self.cancel_delete)
			self.delete_pending = True
			return

	def confirm_delete(self, event=None):
		if hasattr(self, 'delete_pending') and self.delete_pending:
			base_filename = f"{video_label}_{frame_number}"
			# ~ if delete_frame_data(base_filename):
			deleted = annotation_index.delete_frame(base_filename)
			if deleted:
				# Clear the current display
				boxes.clear()
				grey_boxes.clear()
				print(f"All files for frame {frame_number} have been deleted")
				frame_updated = True
				try:
					# refresh index and redraw ticks immediately
					self.refresh_annotation_index_map()
					self.draw_seek_ticks()
				except Exception:
					pass
				self.redraw()
			self.delete_pending = False
			# Remove the temporary key bindings
			self.root.unbind('<Return>')
			self.root.unbind('<Escape>')
			# Prevent the save function from being called
			return "break"

	def cancel_delete(self, event=None):
		if hasattr(self, 'delete_pending') and self.delete_pending:
			print("Deletion cancelled")
			self.delete_pending = False
			# Remove the temporary key bindings
			self.root.unbind('<Return>')
			self.root.unbind('<Escape>')
			# Prevent the save function from being called
			return "break"

	def key_step(self, delta):
		global frame_number, frame_updated
		frame_number = min(max(0, frame_number + delta), total_frames - 1)
		frame_updated = True
		self.seek.set(frame_number)

	def toggle_show_mode(self):
		global show_mode
		show_mode *= -1

	def _on_return_key(self, event):
			"""
			Route Return and Shift+Return to the correct handlers.
			Shift is detected via event.state bit 0x1.
			This method is bound globally so it must handle both cases explicitly
			rather than relying on on_key_all, which is bypassed by bind_all.
			"""
			# Ignore Return while a delete confirmation is pending — confirm_delete handles it
			if getattr(self, 'delete_pending', False):
				return

			if event.state & 0x1:  # Shift is held
				self._restore_last_annotated()
			else:
				self.key_save()

	def key_save(self):
		save_annotation()
		boxes.clear()
		grey_boxes.clear()
		load_next_target(self)


	def redraw(self, temp_rect=None):
		"""
		Compose main display + three zoom panes into a single composite image,
		scale that composite uniformly to fit the available canvas width/height,
		then display it anchored top-left. Draw crosshair and temp rect on the
		scaled composite so canvas coords match.
		"""
		global original_frame, fr, motion_image, last_mouse_move

		if original_frame is None:
			return

		# pick base image depending on current view mode (native video pixels)
		if show_mode == -1:
			base = fr.copy() if fr is not None else np.zeros((video_height, video_width, 3), dtype=np.uint8)
		else:
			base = motion_image.copy() if motion_image is not None else np.zeros((video_height, video_width, 3), dtype=np.uint8)

		# Boxes/labels are drawn later, directly onto the final on-screen image
		# (see below, after `scaled` is composed) rather than baked onto this
		# native frame — baking them here and letting the crop+resize below
		# stretch them via bilinear interpolation blurs thin lines instead of
		# crisply thickening them, so the zoom didn't visibly affect box
		# borders even though it did affect freshly-rasterised text.
		display = base

		# Apply mouse-wheel zoom: crop a sub-region of the frame around the
		# zoom centre; it is then scaled up to the display size.
		zoom = getattr(self, 'zoom', 1.0)
		if zoom and zoom > 1.0:
			cw = max(1, int(round(video_width / zoom)))
			ch = max(1, int(round(video_height / zoom)))
			zcx = int(getattr(self, 'zoom_cx', video_width / 2))
			zcy = int(getattr(self, 'zoom_cy', video_height / 2))
			x0 = max(0, min(video_width - cw, zcx - cw // 2))
			y0 = max(0, min(video_height - ch, zcy - ch // 2))
			display = display[y0:y0 + ch, x0:x0 + cw]
			self._zoom_crop = (x0, y0, cw, ch)
		else:
			self._zoom_crop = (0, 0, video_width, video_height)

		# initial desired main display size (before final uniform scaling)
		disp_w = max(1, int(self.display_size[0]))
		disp_h = max(1, int(self.display_size[1]))

		# resize main display (native composite size)
		disp_resized = cv2.resize(display, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

		# --- prepare zoom panes (native size) ---
		MAG = 2.0
		MAG_ANIM = 1.0

		widget_size = max(32, int(disp_h / 3))

		display_scale = float(video_width) / float(max(1, disp_w))
		crop_vid = max(2, int(round(widget_size * display_scale / MAG)))
		crop_vid_anim = max(2, int(round(widget_size * display_scale / MAG_ANIM)))

		# prevent absurdly large crop sizes (memory blowouts).
		MAX_ZOOM_CROP = 2048
		crop_vid = min(crop_vid, MAX_ZOOM_CROP)
		crop_vid_anim = min(crop_vid_anim, MAX_ZOOM_CROP)

		# ~ def padded_crop(src, cx, cy, crop_size):
			# ~ h, w = src.shape[:2]
			# ~ x1 = cx - crop_size // 2
			# ~ y1 = cy - crop_size // 2
			# ~ x2 = x1 + crop_size
			# ~ y2 = y1 + crop_size
			# ~ sx1 = max(0, x1); sy1 = max(0, y1)
			# ~ sx2 = min(w, x2); sy2 = min(h, y2)
			# ~ out = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
			# ~ if sx2 > sx1 and sy2 > sy1:
				# ~ dst_x1 = sx1 - x1
				# ~ dst_y1 = sy1 - y1
				# ~ dst_x2 = dst_x1 + (sx2 - sx1)
				# ~ dst_y2 = dst_y1 + (sy2 - sy1)
				# ~ out[dst_y1:dst_y2, dst_x1:dst_x2] = src[sy1:sy2, sx1:sx2]
			# ~ return out, (x1, y1, x2, y2)

		def padded_crop(src, cx, cy, crop_size):
			h, w = src.shape[:2]

			# Defensive clamp in case upstream computed a large crop_size.
			MAX_PADDDED_CROP = 2048
			use_crop = int(min(crop_size, MAX_PADDDED_CROP))

			x1 = cx - crop_size // 2
			y1 = cy - crop_size // 2
			x2 = x1 + crop_size
			y2 = y1 + crop_size
			sx1 = max(0, x1); sy1 = max(0, y1)
			sx2 = min(w, x2); sy2 = min(h, y2)

			# create output at the clamped size but still compute box using original crop coords
			out = np.zeros((use_crop, use_crop, 3), dtype=np.uint8)
			if sx2 > sx1 and sy2 > sy1:
				# destination offsets must respect the difference between original and clamped size
				# compute offsets relative to the clamped output
				dst_x1 = sx1 - x1
				dst_y1 = sy1 - y1
				dst_x2 = dst_x1 + (sx2 - sx1)
				dst_y2 = dst_y1 + (sy2 - sy1)

				# If we clamped use_crop < crop_size, we may need to shift the destination region
				# ensure indices fit inside out array
				dst_x1 = max(0, dst_x1)
				dst_y1 = max(0, dst_y1)
				dst_x2 = min(use_crop, dst_x2)
				dst_y2 = min(use_crop, dst_y2)

				out[dst_y1:dst_y2, dst_x1:dst_x2] = src[sy1:sy2, sx1:sx2]
			return out, (x1, y1, x2, y2)

		# center of interest in video coords
		if self.last_mouse is not None:
			try:
				vx, vy = self.canvas_to_video(self.last_mouse)
				cx = int(min(max(0, vx), video_width - 1))
				cy = int(min(max(0, vy), video_height - 1))
			except Exception:
				cx, cy = video_width // 2, video_height // 2
		else:
			cx, cy = video_width // 2, video_height // 2

		# ~ # top zoom (static)
		z_top = None
		if fr is not None:
			crop_img, crop_box = padded_crop(fr, cx, cy, crop_vid)
			z_top = cv2.resize(crop_img, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
			rel_x = cx - crop_box[0]; rel_y = cy - crop_box[1]
			if 0 <= rel_x < crop_vid and 0 <= rel_y < crop_vid:
				zx = int(round(rel_x * widget_size / crop_vid))
				zy = int(round(rel_y * widget_size / crop_vid))
				cv2.line(z_top, (0, zy), (widget_size-1, zy), (255,255,255), 1)
				cv2.line(z_top, (zx, 0), (zx, widget_size-1), (255,255,255), 1)
			cv2.rectangle(z_top, (0, 0), (widget_size-1, widget_size-1), (0, 0, 0), 1)

		# mid zoom (motion)
		z_mid = None
		if original_frame is not None:
			crop_img, crop_box = padded_crop(original_frame, cx, cy, crop_vid)
			z_mid = cv2.resize(crop_img, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
			rel_x = cx - crop_box[0]; rel_y = cy - crop_box[1]
			if 0 <= rel_x < crop_vid and 0 <= rel_y < crop_vid:
				zx = int(round(rel_x * widget_size / crop_vid))
				zy = int(round(rel_y * widget_size / crop_vid))
				cv2.line(z_mid, (0, zy), (widget_size-1, zy), (255,255,255), 1)
				cv2.line(z_mid, (zx, 0), (zx, widget_size-1), (255,255,255), 1)
			cv2.rectangle(z_mid, (0, 0), (widget_size-1, widget_size-1), (0, 0, 0), 1)

		# bottom zoom (animation)
		z_bot = None
		if len(raw_buf) == raw_buf.maxlen:
			idx = int(((time.time() - last_mouse_move) * ANIM_FPS) % raw_buf.maxlen)
			small = raw_buf[idx]
			small_crop, crop_box = padded_crop(small, cx, cy, crop_vid_anim)
			z_bot = cv2.resize(small_crop, (widget_size, widget_size), interpolation=cv2.INTER_LINEAR)
		else:
			z_bot = np.zeros((widget_size, widget_size, 3), dtype=np.uint8)
		# add single-pixel black border to bottom pane as well
		cv2.rectangle(z_bot, (0, 0), (widget_size-1, widget_size-1), (0, 0, 0), 1)

		gap = 0
		right_col_w = widget_size
		right_col_h = widget_size * 3  # no extra gap used here

		# composite size: main display + immediate right column
		composite_h = max(disp_h, right_col_h)
		composite_w = disp_w + right_col_w
		composite = np.zeros((composite_h, composite_w, 3), dtype=np.uint8)

		# place main display at top-left (no horizontal gap)
		composite[0:disp_h, 0:disp_w] = disp_resized

		# zoom column starts immediately after the main display
		zoom_x_off = disp_w
		zoom_y_off = 0



		def place_zoom(zi, x_off, y_off):
			if zi is None:
				return
			h_rem = composite.shape[0] - y_off
			w_rem = composite.shape[1] - x_off
			if h_rem <= 0 or w_rem <= 0:
				return
			zi_h, zi_w = zi.shape[:2]
			use_h = min(zi_h, h_rem)
			use_w = min(zi_w, w_rem)
			zi_crop = zi[0:use_h, 0:use_w]
			composite[y_off:y_off+use_h, x_off:x_off+use_w] = zi_crop

		place_zoom(z_top, zoom_x_off, zoom_y_off)
		place_zoom(z_mid, zoom_x_off, zoom_y_off + widget_size + gap)
		place_zoom(z_bot, zoom_x_off, zoom_y_off + 2 * (widget_size + gap))

		# --- scale composite to fit canvas ---
		c_w = self.canvas.winfo_width() or 1
		c_h = self.canvas.winfo_height() or 1
		scale_w = float(c_w) / float(max(1, composite_w))
		scale_h = float(c_h) / float(max(1, composite_h))
		scale = min(scale_w, scale_h) if (scale_w > 0 and scale_h > 0) else 1.0
		# store for mapping functions
		self.composite_scale = scale

		scaled_w = max(1, int(round(composite_w * scale)))
		scaled_h = max(1, int(round(composite_h * scale)))
		scaled = cv2.resize(composite, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)


		# draw crosshair — but *limit* it to the main display area so it doesn't cross into the zoom column
		scaled_disp_w = max(1, int(round(disp_w * scale)))
		scaled_disp_h = max(1, int(round(disp_h * scale)))

		# Draw boxes/labels directly onto the final on-screen image, in real
		# screen pixels, so they are crisp (no bilinear-resize blur) and their
		# size explicitly grows with mouse-wheel zoom and the canvas-fit scale
		# — exactly like the crosshair/temp-rect overlays below.
		x0, y0, cw, ch = self._zoom_crop
		crop_to_disp = float(disp_w) / float(max(1, cw))
		def _box_to_screen(vx, vy, _x0=x0, _y0=y0, _s=crop_to_disp * scale):
			return (vx - _x0) * _s, (vy - _y0) * _s
		# box_line_scale/box_font_scale (user-configurable, see Display Settings)
		# apply extra reduction on top of line_thickness/font_size before
		# scaling with zoom, so borders/labels can be tuned thinner/smaller
		# than the base config values used elsewhere (e.g. saved crop masks).
		box_thickness = max(1, int(round(line_thickness * zoom * scale * box_line_scale)))
		box_font_size = max(0.15, font_size * zoom * scale * box_font_scale)
		scaled = draw_boxes_on_image(scaled, selected_set=getattr(self, 'selected_boxes', None),
									  thickness=box_thickness, font_scale=box_font_size,
									  to_screen=_box_to_screen, bounds=(scaled_disp_w, scaled_disp_h))

		# These overlays are drawn directly onto the final screen-space image
		# (after the zoom crop/resize and canvas-fit scale already applied), so
		# scale their thickness by the same factors to match the placed boxes'
		# on-screen size and grow with mouse-wheel zoom instead of staying fixed.
		overlay_thickness = max(1, int(round(line_thickness * zoom * scale)))

		if self.last_mouse is not None:
			mx, my = self.last_mouse
			# only draw if the mouse is inside the scaled main-display region
			if 0 <= mx < scaled_disp_w and 0 <= my < scaled_disp_h:
				cv2.line(scaled, (int(mx), 0), (int(mx), scaled_disp_h), (255,255,255), overlay_thickness)
				cv2.line(scaled, (0, int(my)), (scaled_disp_w, int(my)), (255,255,255), overlay_thickness)


		# determine temporary rectangle to draw (if drawing and no explicit temp_rect provided)
		if temp_rect is None and getattr(self, 'drawing', False):
			if self.start_canvas_xy is not None and self.last_mouse is not None:
				temp_rect = (self.start_canvas_xy, self.last_mouse)

		# draw temporary rect (coordinates are canvas coords; draw onto scaled image)
		if temp_rect is not None:
			(sx, sy), (ex, ey) = temp_rect
			# clip to scaled image
			rx1 = max(0, min(sx, scaled_w-1)); ry1 = max(0, min(sy, scaled_h-1))
			rx2 = max(0, min(ex, scaled_w-1)); ry2 = max(0, min(ey, scaled_h-1))
			cv2.rectangle(scaled, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (255,255,255), overlay_thickness)

		# convert and display
		self.tk_img = cv2_to_photoimage(scaled)
		try:
			self.canvas.config(scrollregion=(0, 0, scaled_w, scaled_h))
		except Exception:
			pass
		self.canvas.delete('all')
		self.canvas.create_image(0, 0, image=self.tk_img, anchor='nw')

		try:
			self.draw_seek_ticks()
		except Exception:
			pass



	def loop(self):
		global frame_updated, fr, original_frame, motion_image, raw_buf, last_mouse_move, last_anim_draw, boxes, grey_boxes

		need_ = False
		now = time.time()

		if frame_updated:
			frame_updated = False
			boxes.clear(); grey_boxes.clear()
			last_frame = frame_number
			start_frame = last_frame - frameWindow + 1
			if start_frame < 0:
				original_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
				fr = original_frame.copy()
				motion_image = original_frame.copy()
				raw_buf.clear()
				need_ = True
			else:
				capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
				prev_frames = [None] * 3
				motion_image = None
				frame_count = 0
				raw_buf.clear()
				for i in range(frameWindow):
					ret, raw_frame = capture.read()
					if not ret:
						break
					if frame_count == 0:
						fr = raw_frame.copy()
						if scale_factor != 1.0:
							fr = cv2.resize(fr, (0,0), fx=scale_factor, fy=scale_factor)
						raw_buf.append(fr.copy())
						gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
						if i == 0:
							prev_frames = [gray.copy()] * 3
							frame_count += 1
							if frame_count > frame_skip:
								frame_count = 0
							continue
						diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
						if strategy == 'exponential':
							prev_frames[0] = gray
							prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, 1-expA, 0)
							prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, 1-expB, 0)
						else:
							prev_frames[2] = prev_frames[1]
							prev_frames[1] = prev_frames[0]
							prev_frames[0] = gray
					frame_count += 1
					if frame_count > frame_skip:
						frame_count = 0
				if 'diffs' in locals():
					if chromatic_tail_only == 'true':
						tb = cv2.subtract(diffs[0], diffs[1])
						tr = cv2.subtract(diffs[2], diffs[1])
						tg = cv2.subtract(diffs[1], diffs[0])
						blue = cv2.addWeighted(gray, lum_weight, tb, rgb_multipliers[2], motion_threshold)
						green = cv2.addWeighted(gray, lum_weight, tg, rgb_multipliers[1], motion_threshold)
						red = cv2.addWeighted(gray, lum_weight, tr, rgb_multipliers[0], motion_threshold)
					else:
						blue = cv2.addWeighted(gray, lum_weight, diffs[0], rgb_multipliers[2], motion_threshold)
						green = cv2.addWeighted(gray, lum_weight, diffs[1], rgb_multipliers[1], motion_threshold)
						red = cv2.addWeighted(gray, lum_weight, diffs[2], rgb_multipliers[0], motion_threshold)
					motion_image = cv2.merge((blue, green, red)).astype(np.uint8)
					original_frame = motion_image.copy()


					try:
						base = f"{video_label}_{frame_number}"
						boxes, grey_boxes = annotation_index.load_labels_for_basename(base, fr, original_frame)
					except Exception as e:
						print("Error loading saved annotations for", f"{video_label}_{frame_number},", e)

					if boxes or grey_boxes:
						# ~ print('Annotations found')
						pass
					else:
						if auto_ann_switch == 1:
							auto_annotate_local()
						need_ = True
						last_anim_draw = time.time()

		else:
			if (now - last_mouse_move) > ANIM_STILL_THRESHOLD and (now - last_anim_draw) >= ANIM_DT:
				last_anim_draw = now
				need_ = True

		# recompute display size preserving aspect ratio (main video area only)
		c_w = self.canvas.winfo_width() or 400
		c_h = self.canvas.winfo_height() or 300
		aspect = video_width / video_height
		if c_w / aspect <= c_h:
			disp_w = c_w
			disp_h = int(c_w / aspect)
		else:
			disp_h = c_h
			disp_w = int(c_h * aspect)
		self.display_size = (max(1, int(disp_w)), max(1, int(disp_h)))


		# ensure the temporary rectangle remains visible while mouse is held
		if getattr(self, 'drawing', False):
			need_ = True

		if need_:
			self.redraw()


		self.root.after(30, self.loop)

	def _restore_last_annotated(self):
			"""
			Restore the frame that was displayed just before the last save.
			Allows the user to go back and correct a mistake (Shift+Enter).
			"""
			global video_path, capture, frame_number, video_label
			global total_frames, video_width, video_height, frame_updated
			global fr, original_frame, motion_image, raw_buf

			if not _last_annotated_cache:
				print("No previous frame in cache to restore.")
				return

			cache = _last_annotated_cache

			# Switch video if needed
			if cache['video_path'] != video_path:
				capture.release()
				capture        = cv2.VideoCapture(cache['video_path'])
				video_path     = cache['video_path']
				video_label    = os.path.splitext(os.path.basename(video_path))[0]
				total_frames   = cache['total_frames']
				video_width    = cache['video_width']
				video_height   = cache['video_height']
				self.root.title(f"BehaveAI — {os.path.basename(video_path)}")
				self.seek.configure(to=max(0, total_frames - 1))

			# Restore frame data from cache
			fr             = cache['fr'].copy()
			motion_image   = cache['motion_image'].copy() if cache['motion_image'] is not None else None
			original_frame = cache['original_frame'].copy()
			raw_buf.clear()
			for f in cache['raw_buf']:
				raw_buf.append(f)

			frame_number  = cache['frame_number']
			frame_updated = False  # data already loaded, no need to recompute

			# Clear any boxes drawn on the new frame — user is back on the old one
			boxes.clear()
			grey_boxes.clear()

			# Load existing annotations for this frame if they exist
			try:
				base = f"{video_label}_{frame_number}"
				loaded_boxes, loaded_grey = annotation_index.load_labels_for_basename(
					base, fr, original_frame)
				boxes.extend(loaded_boxes)
				grey_boxes.extend(loaded_grey)
			except Exception as e:
				print(f"Could not reload annotations for restored frame: {e}")

			self.seek.set(frame_number)
			try:
				self.frame_var.set(f'Frame {frame_number}')
			except Exception:
				pass
			self.draw_seek_ticks()
			self.redraw()
			print(f"Restored frame {frame_number} from {os.path.basename(video_path)}")

# Launch app
root = tk.Tk()
app = AnnotatorTk(root)

# Pick the first prefetch target deterministically and remember it
_initial_unannotated = get_unannotated_pool(full_frame_pool, annotation_index)
_initial_unannotated = [(vp, fn) for vp, fn in _initial_unannotated
						if not (vp == video_path and fn == frame_number)]
if _initial_unannotated:
	_next_vpath, _next_fnum = pick_random_frame(_initial_unannotated)
	_prefetched_target = (_next_vpath, _next_fnum)
	# Delay so the first frame loads without competing with the prefetch thread
	root.after(3000, lambda vp=_next_vpath, fn=_next_fnum: _start_prefetch(vp, fn))

root.mainloop()

capture.release()
print("Done annotating video.")
