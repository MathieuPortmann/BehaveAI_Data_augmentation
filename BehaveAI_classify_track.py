import cv2
import numpy as np
import csv
import os
import glob
import random
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
import configparser
from behaveai_config import (
	load_secondary_config, NONE_LABEL,
	get_species_list, species_folder, load_ethogram_for_species, load_age_classes,
	resolve_project_dirs, parse_train_overrides,
)
from behaveai_render import (
	load_render_style, draw_labeled_detection, draw_frame_number,
)
from behaveai_holdout import build_classification_split
import time
import shutil
import gc
import json
import tempfile
import tkinter as tk
from tkinter import messagebox, filedialog
import subprocess
# ~ import config_watcher
import sys


# --- NCNN helper utilities -----------------------


def ncnn_dir_for_weights(weights_path):
	"""Return the expected NCNN export directory for a given .pt path."""
	base, ext = os.path.splitext(weights_path)
	# Ultralytics export typically creates a folder named like "<base>_ncnn_model"
	return base + "_ncnn_model"

def ncnn_files_exist(ncnn_dir):
	"""Return True if NCNN param+bin appear to exist in the export dir."""
	if not os.path.isdir(ncnn_dir):
		return False
	# Look for .param and .bin files (ncnn export creates *.param and *.bin)
	has_param = any(f.endswith(".param") for f in os.listdir(ncnn_dir))
	has_bin = any(f.endswith(".bin") for f in os.listdir(ncnn_dir))
	return has_param and has_bin

def ensure_ncnn_export(weights_path, task, timeout=300):
	"""
	Ensure an NCNN conversion exists for weights_path.
	Returns the ncnn_dir on success, None on failure (falls back to .pt).
	This will skip conversion if the ncnn folder already exists.
	"""
	ncnn_dir = ncnn_dir_for_weights(weights_path)
	if ncnn_files_exist(ncnn_dir):
		return ncnn_dir

	try:
		print(f"Exporting {weights_path} -> NCNN (this may take a while)...")
		model = YOLO(weights_path, task=task)
		# Use Ultralytics export API. This creates the folder "<base>_ncnn_model".
		# Some installs can be slow; we try and catch errors below.
		model.export(format="ncnn")
		# Wait a short time for files to appear (export is synchronous in most versions).
		start = time.time()
		while time.time() - start < timeout:
			if ncnn_files_exist(ncnn_dir):
				print(f"NCNN export complete: {ncnn_dir}")
				return ncnn_dir
			time.sleep(0.5)
		# timed out
		print(f"NCNN export timeout for {weights_path}")
		return None
	except Exception as e:
		# Don't crash — export can fail on some systems; print useful debugging info and return None
		print(f"Warning: NCNN export failed for {weights_path}: {e}")
		return None

def load_model_with_ncnn_preference(weights_path, task):
	"""
	Attempt to use NCNN if available (or convert it). If conversion or loading fails,
	fall back to the original PyTorch .pt path.
	Returns a YOLO model instance (which may wrap NCNN or .pt).
	"""
	# If a .pt was not provided (maybe already a folder), just try loading directly
	if not weights_path.endswith(".pt"):
		try:
			return YOLO(weights_path, task=task)
		except Exception as e:
			print(f"Error loading model {weights_path}: {e}")
			raise

	ncnn_dir = ncnn_dir_for_weights(weights_path)
	# prefer existing NCNN dir if present
	if ncnn_files_exist(ncnn_dir):
		try:
			print(f"Loading NCNN model from {ncnn_dir}")
			return YOLO(ncnn_dir, task=task)
		except Exception as e:
			print(f"Failed to load NCNN model at {ncnn_dir}: {e} (falling back to .pt)")

	# Otherwise attempt conversion (one-time). If it fails, fall back to .pt.
	exported = ensure_ncnn_export(weights_path, task)
	if exported:
		try:
			return YOLO(exported, task=task)
		except Exception as e:
			print(f"Failed to load NCNN-exported model {exported}: {e} (falling back to .pt)")

	# Finally, fallback to direct .pt load
	print(f"Using original weights (PyTorch) at {weights_path}")
	return YOLO(weights_path, task=task)
# --------------------------------------------------------------------


# --- SAHI sliced-inference helpers -----------------------------------
# All lazy-imported so this module still loads when sahi is not installed
# (sahi is only needed once a sahi_enabled_* flag is turned on).


def build_sahi_model(yolo_model, conf, image_size=640):
	"""Wrap an ALREADY-LOADED ultralytics YOLO object in a SAHI detection model.

	Passing the in-memory object (model=...) rather than a path means SAHI never
	re-loads from disk, so it is indifferent to a .pt vs an NCNN-exported backend.
	Returns None on any failure so the caller transparently falls back to plain
	whole-frame .predict(). Newer sahi uses model_type="ultralytics"; older
	releases only know "yolov8" — try both.
	"""
	try:
		from sahi import AutoDetectionModel
	except Exception as e:
		print(f"SAHI unavailable ({e}); using whole-frame detection.")
		return None
	try:
		import torch
		device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
	except Exception:
		device = 'cpu'
	for model_type in ('ultralytics', 'yolov8'):
		try:
			return AutoDetectionModel.from_pretrained(
				model_type=model_type, model=yolo_model,
				confidence_threshold=conf, device=device, image_size=image_size)
		except Exception as e:
			last_err = e
	print(f"SAHI wrap failed ({last_err}); using whole-frame detection.")
	return None


def sahi_detect(sahi_model, image, class_names, source,
				slice_h, slice_w, overlap_h, overlap_w,
				pp_type, pp_metric, pp_thresh, standard_pred):
	"""Run SAHI sliced prediction on one image and return detections in the same
	dict shape the whole-frame path produces. SAHI already remaps tile-local
	boxes to full-image coordinates and runs cross-tile NMS, so coords are
	full-frame xyxy exactly like model.predict() gives."""
	from sahi.predict import get_sliced_prediction
	result = get_sliced_prediction(
		image, sahi_model,
		slice_height=slice_h, slice_width=slice_w,
		overlap_height_ratio=overlap_h, overlap_width_ratio=overlap_w,
		postprocess_type=pp_type, postprocess_match_metric=pp_metric,
		postprocess_match_threshold=pp_thresh,
		perform_standard_pred=standard_pred, verbose=0)
	dets = []
	for op in result.object_prediction_list:
		x1, y1, x2, y2 = op.bbox.to_xyxy()
		dets.append({
			'coords': (int(x1), int(y1), int(x2), int(y2)),
			'primary_class': class_names[int(op.category.id)],
			'primary_conf': float(op.score.value),
			'source': source,
		})
	return dets


def record_stream_verdict(md, det):
	"""Keep the best verdict of EACH stream on a merged detection.

	primary_static_* / primary_motion_* are read downstream (activity budget,
	complex features, and the frame miner) as "what the static / motion detector
	said about this animal", so they have to be filled by source. The merge below
	is winner-takes-all: it overwrites primary_class and only remembers the single
	detection it displaced, whatever that one's source was. Two same-source
	detections merging -- the common case under dominant_source='confidence' --
	would therefore file a static verdict under primary_motion_class, and every
	reader downstream would trust it.
	"""
	if det['source'] == 'static':
		key_cls, key_conf = 'static_class', 'static_conf'
	else:
		key_cls, key_conf = 'motion_class', 'motion_conf'
	if det['primary_conf'] > md.get(key_conf, 0.0):
		md[key_cls] = det['primary_class']
		md[key_conf] = det['primary_conf']
# --------------------------------------------------------------------


# --- BoT-SORT / ByteTrack integration helpers ------------------------
# The Ultralytics trackers expect a results-like object exposing .xywh/.xyxy/
# .conf/.cls plus __len__/__getitem__ (they do results[bool_mask] internally).
# This minimal wrapper provides exactly that from our detection dicts. Validated
# by an isolated probe (see plan) before wiring it in here.


class _DetResultsForTracker:
	def __init__(self, xywh, xyxy, conf, cls):
		self.xywh, self.xyxy, self.conf, self.cls = xywh, xyxy, conf, cls

	def __len__(self):
		return len(self.conf)

	def __getitem__(self, mask):
		return _DetResultsForTracker(self.xywh[mask], self.xyxy[mask],
									 self.conf[mask], self.cls[mask])


def bot_track_update(bot_tracker, processed_detections, frame):
	"""Feed processed_detections to an Ultralytics BoT-SORT/ByteTrack tracker and
	return {detection_index: track_id}.

	The per-frame index into processed_detections is stashed in the tracker's
	`cls` field (which BoT-SORT transports untouched from input detection to the
	output row that updated a track that frame) and read back from output column
	6, so each track row maps to the exact detection that produced it -- letting
	the caller re-attach class/secondary metadata. Column 7 (`idx`) is NOT used:
	the tracker resets it relative to its internal high/low-score subsets.
	Output row layout: [x1, y1, x2, y2, track_id, score, cls, idx].
	"""
	n = len(processed_detections)
	if n == 0:
		empty = _DetResultsForTracker(np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32),
									  np.zeros((0,), np.float32), np.zeros((0,), np.float32))
		bot_tracker.update(empty, img=frame)
		return {}
	xyxy = np.array([d['coords'] for d in processed_detections], dtype=np.float32)
	xywh = np.empty_like(xyxy)
	xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
	xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
	xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
	xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
	conf = np.array([float(d.get('primary_conf', 0.0)) for d in processed_detections], dtype=np.float32)
	cls = np.arange(n, dtype=np.float32)
	tracks = bot_tracker.update(_DetResultsForTracker(xywh, xyxy, conf, cls), img=frame)
	return {int(round(row[6])): int(row[4]) for row in tracks}
# --------------------------------------------------------------------



def move_to_expected(project_path, run_name="train", runs_root="runs"):
    """
    If a YOLOv2x-style run was just created under runs/.../<run_name>,
    move that run/<run_name> directory into project_path/<run_name>.
    Returns the destination path on success, or None on failure / nothing found.

    This is a FALLBACK. BehaveAI_train_worker.py passes project=project_path and
    name=run_name, so Ultralytics normally writes straight to the destination and
    there is nothing under runs/ to rescue. It only earns its keep when a run
    lands elsewhere.

    Both guards below exist because this function used to destroy the model it
    was meant to install: it deleted the destination up front and then moved in
    whatever the glob found, so a single stale runs/**/train left by an earlier
    interrupted run wiped the weights Ultralytics had just written. The move
    failure is caught and only warned about, so the pipeline carried on and blew
    up much later on YOLO(<model>/train/weights/best.pt) with a bare
    FileNotFoundError -- a long way from the cause.
    """
    dst_train = os.path.join(project_path, run_name)  # e.g. model_primary_motion/train

    # Guard 1: the training already landed where it belongs. Importing a stray
    # run over it would replace a fresh model with an older, unrelated one.
    if os.path.exists(os.path.join(dst_train, "weights", "best.pt")):
        return dst_train

    # look first in runs/detect/**/train then in runs/**/train
    candidates = glob.glob(os.path.join(runs_root, "detect", "**", run_name), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(runs_root, "**", run_name), recursive=True)

    # Guard 2: keep only directories that actually hold trained weights. An
    # empty or half-written run folder is not worth deleting the destination for.
    candidates = [p for p in candidates
                  if os.path.isdir(p) and os.path.exists(os.path.join(p, "weights", "best.pt"))]
    if not candidates:
        return None

    # pick most recently modified candidate
    candidates = sorted(candidates, key=os.path.getmtime, reverse=True)
    src_train = candidates[0]                     # e.g. runs/detect/2026-02-24_train

    try:
        # remove existing destination so the move yields the expected layout
        if os.path.exists(dst_train):
            try:
                shutil.rmtree(dst_train)
            except Exception:
                pass

        shutil.move(src_train, dst_train)

        # best-effort: remove any now-empty ancestor dirs under runs_root
        runs_root_abs = os.path.abspath(runs_root)
        parent = os.path.abspath(os.path.dirname(src_train))
        # remove upward until we hit runs_root or a non-empty dir
        while parent.startswith(runs_root_abs):
            try:
                if os.path.isdir(parent) and not os.listdir(parent):
                    shutil.rmtree(parent)
                    parent = os.path.dirname(parent)
                else:
                    break
            except Exception:
                break

        print(f"Moved YOLO training output: '{src_train}' -> '{dst_train}'")
        return dst_train
    except Exception as e:
        print(f"Warning: failed to move YOLO run folder '{src_train}' -> '{dst_train}': {e}")
        return None



# ---------- Project-aware configuration loading --------------------------

def pick_ini_via_dialog():
	root = tk.Tk()
	root.withdraw()
	path = filedialog.askopenfilename(
		title="Select BehaveAI settings INI",
		filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
	)
	root.destroy()
	return path

# Determine config_path (accept project dir or direct INI path)
if len(sys.argv) > 1:
	arg = os.path.abspath(sys.argv[1])
	if os.path.isdir(arg):
		config_path = os.path.join(arg, "BehaveAI_settings.ini")
	else:
		config_path = arg
else:
	config_path = pick_ini_via_dialog()
	if not config_path:
		tk.messagebox.showinfo("No settings file", "No settings INI selected — exiting.")
		sys.exit(0)

config_path = os.path.abspath(config_path)
if not os.path.exists(config_path):
	tk.messagebox.showerror("Missing settings", f"Configuration file not found: {config_path}")
	sys.exit(1)

# Set project directory to the INI parent and make it the working directory
project_dir = os.path.dirname(config_path)
os.chdir(project_dir)
print(f"Working directory set to project dir: {project_dir}")
print(f"Using settings file: {config_path}")

# Load configuration
config = configparser.ConfigParser()
config.optionxform = str  # keep case
config.read(config_path)

# Directory keys from the INI: absolute as given, relative to the project
# otherwise, defaults clips/ input/ output/ inside the project.
clips_dir, input_folder, output_folder = resolve_project_dirs(config, project_dir)


# ---- Species (model 0) + per-species ethogram/age (model 0.5) ----
# The first species keeps every bare/legacy key and folder name below byte-for-byte
# (species_key/species_folder resolve to the unscoped name for species_list[0]), so
# an existing single-species project is completely unaffected until a 2nd species
# is added via the settings GUI.
#
# NOTE (scope of this increment): the primary static/motion DETECTORS below still
# run once, for species_list[0] only - a true per-species multi-detector loop
# (one full detection pass per species, merged) is future work for when a 2nd
# species actually has its own annotated primary behaviours (untestable today with
# only one species; see BehaveAI_features_and_settings.md). Model 0 (species) and
# model 0.5 (age) below DO run per detection today, exactly as scoped.
species_list = get_species_list(config)
species_cropped_base_dir = 'annot_species_crop'
age_cropped_base_dir = 'annot_age_crop'
species_model_path = os.path.join('model_species', "train", "weights", "best.pt")
age_model_path = os.path.join('model_age', "train", "weights", "best.pt")

# Read parameters
try:

	_eth = load_ethogram_for_species(config, species_list[0], species_list)
	primary_static_classes = _eth['primary_static_classes']
	primary_motion_classes = _eth['primary_motion_classes']
	primary_static_colors = _eth['primary_static_colors']
	primary_motion_colors = _eth['primary_motion_colors']
	primary_static_hotkeys = _eth['primary_static_hotkeys']
	primary_motion_hotkeys = _eth['primary_motion_hotkeys']

	motion_cropped_base_dir = species_folder('annot_motion_crop', species_list[0], species_list)
	static_cropped_base_dir = species_folder('annot_static_crop', species_list[0], species_list)

	primary_classes = primary_static_classes + primary_motion_classes
	primary_colors = primary_static_colors + primary_motion_colors

	secondary_classes = _eth['secondary_classes']
	secondary_colors = _eth['secondary_colors']
	secondary_hotkeys = _eth['secondary_hotkeys']
	secondary_map = _eth['secondary_map']
	allowed_secondary_idx = _eth['allowed_secondary_idx']
	hierarchical_mode = _eth['hierarchical_mode']

	age_classes = load_age_classes(config, species_list[0], species_list)['age_classes']

	# Derived: primaries that have no secondary at all (kept for legacy guards).
	ignore_secondary = [primary_classes[i] for i in range(len(primary_classes))
						if i >= len(allowed_secondary_idx) or not allowed_secondary_idx[i]]

	primary_static_project_path = species_folder('model_primary_static', species_list[0], species_list)
	primary_static_model_path = os.path.join(primary_static_project_path, "train", "weights", "best.pt")
	primary_static_yaml_path = species_folder('static_annotations', species_list[0], species_list) + '.yaml'

	primary_motion_project_path = species_folder('model_primary_motion', species_list[0], species_list)
	primary_motion_model_path = os.path.join(primary_motion_project_path, "train", "weights", "best.pt")
	primary_motion_yaml_path = species_folder('motion_annotations', species_list[0], species_list) + '.yaml'

	dominant_source = config['DEFAULT']['dominant_source'].lower()

	primary_classifier = config['DEFAULT'].get('primary_classifier', 'yolo11s.pt')
	primary_epochs = int(config['DEFAULT'].get('primary_epochs', '50'))
	secondary_classifier = config['DEFAULT'].get('secondary_classifier', 'yolo11s-cls.pt')
	secondary_epochs = int(config['DEFAULT'].get('secondary_epochs', '50'))
	# Early stopping: primary_epochs/secondary_epochs above are the CAP; training
	# stops after `train_patience` epochs with no val improvement (Ultralytics
	# default is 100, which effectively never triggers on short runs).
	train_patience = int(config['DEFAULT'].get('train_patience', '30'))
	# Training/inference resolution. Ultralytics stores imgsz in the checkpoint and
	# predict() reads it back, so a single value governs both ends and they cannot
	# drift apart. 640 on a 3840px drone frame shrinks every animal by 6x, which is
	# what limits recall on the small ones -- see the primary_imgsz help entry.
	# YOLO requires a multiple of 32; round rather than fail deep inside training.
	def _imgsz(key, default):
		v = int(config['DEFAULT'].get(key, default))
		snapped = max(32, int(round(v / 32.0)) * 32)
		if snapped != v:
			print(f"{key}={v} is not a multiple of 32; using {snapped}.")
		return snapped
	primary_imgsz = _imgsz('primary_imgsz', '640')
	secondary_imgsz = _imgsz('secondary_imgsz', '224')
	# Share of whole videos held out. The annotation tool already routes frames
	# to annot_<stream>/images/{train,val} with it; the crop classifiers below
	# get the same partition through build_classification_split.
	val_frequency = float(config['DEFAULT'].get('val_frequency', '0.1'))
	# The motion stream is a false-colour encoding where COLOUR carries the motion
	# signal, so YOLO's online HSV augmentation would corrupt it. When enabled
	# (default), the motion detector and motion secondary classifier are trained
	# with hsv_h/hsv_s/hsv_v = 0; geometric/occlusion augs (mosaic, erasing, flip,
	# translate, scale) stay on. The static stream keeps the Ultralytics defaults.
	# auto_augment=None matters for the CLASSIFICATION task only (the secondary
	# crop classifiers): Ultralytics defaults it to 'randaugment', whose op pool
	# includes Color/Brightness/Contrast/Sharpness/Posterize/Solarize — i.e. it
	# re-introduces through the back door exactly the colour jitter hsv_*=0 just
	# removed. Detection training ignores the key.
	motion_disable_color_aug = config['DEFAULT'].get('motion_disable_color_aug', 'true').lower() == 'true'

	# User-supplied model.train() kwargs, one set for the detectors and one for
	# the crop classifiers. They are kept separate because the two tasks read
	# different keys: `mosaic` is detection-only, while `scale` and
	# `auto_augment` also drive classification's RandomResizedCrop, so a single
	# shared list would change the crop models by accident.
	primary_train_overrides = parse_train_overrides(
		config['DEFAULT'].get('primary_train_overrides', ''), 'primary_train_overrides')
	secondary_train_overrides = parse_train_overrides(
		config['DEFAULT'].get('secondary_train_overrides', ''), 'secondary_train_overrides')

	def _with_motion_color_rules(base):
		"""Layer the motion stream's colour rules on top of the user overrides.

		The motion images are false colour -- the hue IS the movement signal --
		so hsv_* jitter and RandomAugment's colour ops destroy the very thing
		being learned. That is a correctness constraint, not a preference, so it
		wins over anything set in the INI."""
		if not motion_disable_color_aug:
			return dict(base) if base else None
		merged = dict(base) if base else {}
		clobbered = [k for k in ('hsv_h', 'hsv_s', 'hsv_v', 'auto_augment') if k in merged]
		if clobbered:
			print(f"motion stream: ignoring {', '.join(clobbered)} from the INI overrides -- "
				  f"motion_disable_color_aug is on and the motion images encode movement as colour.")
		merged.update({'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0, 'auto_augment': None})
		return merged

	primary_motion_train_overrides = _with_motion_color_rules(primary_train_overrides)
	secondary_motion_train_overrides = _with_motion_color_rules(secondary_train_overrides)

	if hierarchical_mode:
		secondary_static_project_path = species_folder('model_secondary_static', species_list[0], species_list)
		secondary_static_data_path = static_cropped_base_dir
		secondary_static_model_path = os.path.join(secondary_static_project_path, "train", "weights", "best.pt")

		secondary_motion_project_path = species_folder('model_secondary_motion', species_list[0], species_list)
		secondary_motion_data_path = motion_cropped_base_dir
		secondary_motion_model_path = os.path.join(secondary_motion_project_path, "train", "weights", "best.pt")

	# Common parameters
	scale_factor = float(config['DEFAULT'].get('scale_factor', '1.0'))
	expA = float(config['DEFAULT'].get('expA', '0.5'))
	expB = float(config['DEFAULT'].get('expB', '0.8'))
	lum_weight = float(config['DEFAULT'].get('lum_weight', '0.7'))
	strategy = config['DEFAULT'].get('strategy', 'exponential')
	chromatic_tail_only = config['DEFAULT']['chromatic_tail_only'].lower()
	rgb_multipliers = [float(x) for x in config['DEFAULT']['rgb_multipliers'].split(',')]
	use_ncnn = config['DEFAULT']['use_ncnn'].lower()
	primary_conf_thresh = float(config['DEFAULT'].get('primary_conf_thresh', '0.5'))
	secondary_conf_thresh = float(config['DEFAULT'].get('secondary_conf_thresh', '0.5'))

	# --- SAHI sliced inference (small-object detection) ---
	# Optional, per-stream, OFF by default: a project whose INI predates these
	# keys (or leaves them false) detects exactly as before. When enabled, the
	# full 4K frame is diced into native-resolution tiles so a ~60px horse is
	# fed to the model at ~60px instead of being shrunk to ~10px by the 640
	# whole-frame resize. Slicing every tile is 20-40x slower, hence per-stream
	# switches plus an auto-skip when the frame is not meaningfully larger than a
	# tile. Detection confidence stays primary_conf_thresh here (the decoupled
	# low threshold arrives with the BoT-SORT tracker, which filters after
	# tracking); SAHI only changes HOW the frame is fed to the detector.
	sahi_enabled_static = config['DEFAULT'].get('sahi_enabled_static', 'false').lower() == 'true'
	sahi_enabled_motion = config['DEFAULT'].get('sahi_enabled_motion', 'false').lower() == 'true'
	sahi_slice_height = int(config['DEFAULT'].get('sahi_slice_height', '640'))
	sahi_slice_width = int(config['DEFAULT'].get('sahi_slice_width', '640'))
	sahi_overlap_height_ratio = float(config['DEFAULT'].get('sahi_overlap_height_ratio', '0.2'))
	sahi_overlap_width_ratio = float(config['DEFAULT'].get('sahi_overlap_width_ratio', '0.2'))
	sahi_postprocess_type = config['DEFAULT'].get('sahi_postprocess_type', 'NMS')
	sahi_postprocess_match_metric = config['DEFAULT'].get('sahi_postprocess_match_metric', 'IOS')
	sahi_postprocess_match_threshold = float(config['DEFAULT'].get('sahi_postprocess_match_threshold', '0.5'))
	sahi_perform_standard_pred = config['DEFAULT'].get('sahi_perform_standard_pred', 'false').lower() == 'true'
	sahi_min_dim_factor = float(config['DEFAULT'].get('sahi_min_dim_factor', '1.5'))

	# SAHI mode is a coherent train+infer switch: when a stream is tiled, its
	# detector is retrained on the sliced dataset AND inference loads that tiled
	# model. The tiled model lives in a separate *_tiled project so the existing
	# whole-frame model is never overwritten -- flip the flag off to fall back to
	# it instantly. Redirect the project/model paths here (defined above) so both
	# the training calls and the inference model-load below use the tiled model.
	if sahi_enabled_static and primary_static_classes:
		primary_static_project_path = primary_static_project_path + '_tiled'
		primary_static_model_path = os.path.join(primary_static_project_path, 'train', 'weights', 'best.pt')
	if sahi_enabled_motion and primary_motion_classes:
		primary_motion_project_path = primary_motion_project_path + '_tiled'
		primary_motion_model_path = os.path.join(primary_motion_project_path, 'train', 'weights', 'best.pt')

	# --- Tracker: BoT-SORT/ByteTrack (default) or the legacy Kalman tracker ---
	# 'botsort' brings camera-motion compensation (GMC) inside the association
	# loop and ByteTrack's two-tier matching; association is kinematic only.
	# 'kalman' keeps the old homemade tracker for comparison. Long-gap identity
	# recovery is deferred to the offline stitching pass, so track_buffer stays
	# short here.
	tracker_type = config['DEFAULT'].get('tracker_type', 'botsort').lower()
	tracker_track_high_thresh = float(config['DEFAULT'].get('tracker_track_high_thresh', '0.5'))
	tracker_track_low_thresh = float(config['DEFAULT'].get('tracker_track_low_thresh', '0.1'))
	tracker_new_track_thresh = float(config['DEFAULT'].get('tracker_new_track_thresh', '0.6'))
	tracker_track_buffer = int(config['DEFAULT'].get('tracker_track_buffer', '30'))
	tracker_match_thresh = float(config['DEFAULT'].get('tracker_match_thresh', '0.8'))
	tracker_gmc_method = config['DEFAULT'].get('tracker_gmc_method', 'sparseOptFlow')

	match_distance_thresh = float(config['DEFAULT'].get('match_distance_thresh', '200'))
	delete_after_missed = float(config['DEFAULT'].get('delete_after_missed', '5'))

	centroid_merge_thresh = float(config['DEFAULT'].get('centroid_merge_thresh', '50'))
	iou_thresh = float(config['DEFAULT'].get('iou_thresh', '0.95'))
	line_thickness = int(config['DEFAULT'].get('line_thickness', '1'))
	font_size = float(config['DEFAULT'].get('font_size', '0.5'))
	# Box/label styling shared with the annotation tool (see behaveai_render.py).
	render_style = load_render_style(config)
	# Whether to write <video>_detected.mp4 alongside the tracking CSV. Off by
	# default: the CSVs are the result of a run, the annotated copy is a visual
	# check that costs roughly the size of the source clip per video.
	detection_video_enabled = config['DEFAULT'].get('detection_video_enabled', 'false').lower() == 'true'
	frame_skip = int(config['DEFAULT'].get('frame_skip', '0'))
	ab_analysis_duration_s = float(config['DEFAULT'].get('ab_analysis_duration_s', '0'))

	process_noise_pos = float(config['kalman'].get('process_noise_pos', '0.01'))
	process_noise_vel = float(config['kalman'].get('process_noise_vel', '0.1'))
	measurement_noise = float(config['kalman'].get('measurement_noise', '0.1'))
	motion_threshold = -1 * int(config['DEFAULT'].get('motion_threshold', '0'))

except KeyError as e:
	raise KeyError(f"Missing configuration parameter: {e}")


# Validate configuration

if len(primary_motion_classes) != len(primary_motion_colors) or len(primary_motion_classes) != len(primary_motion_hotkeys):
	raise ValueError("Primary motion classes, colors and hotkeys must match in configuration.")
if len(primary_static_classes) != len(primary_static_colors) or len(primary_static_classes) != len(primary_static_hotkeys):
	raise ValueError("Primary static classes, colors and hotkeys must match in configuration.")
if dominant_source != 'motion' and dominant_source != 'static' and dominant_source != 'confidence':
	raise ValueError("dominant_source must be motion, static, or confidence")

if len(primary_static_classes) > 0:
	if not os.path.exists(primary_static_yaml_path):
		print(f"Error: Primary static YAML file not found. Run the Annotation script once to fix this")
		sys.exit(1)

if len(primary_motion_classes) > 0:
	if not os.path.exists(primary_motion_yaml_path):
		print(f"Error: Primary motion YAML file not found. Run the Annotation script once to fix this")
		sys.exit(1)


# ~ # check whether settings have been changed, and motion annotation library needs rebuilding
# ~ settings_changed = config_watcher.check_settings_changed(current_config_path=config_path, saved_config_path=None, model_dirs=['model_primary_motion'])
# ~ # Globals for prompting/behaviour inside maybe_retrain
# ~ regen_prompt_shown = False
# ~ force_rebuild_motion = False


global_response = 0 # if 'yes' is selected for any model re-training, retraining should be perfoemd for all models

def count_images_in_dataset(path):
	## Count images in a dataset, handling both YAML-based and directory-based datasets
	# If path is a YAML file (primary models)
	if path.endswith('.yaml'):
		try:
			import yaml
			with open(path, 'r') as f:
				data = yaml.safe_load(f)

			# Get the path to the training images
			train_path = data['train']
			base_dir = os.path.dirname(path)
			abs_train_path = os.path.join(base_dir, train_path)

			# Handle different dataset formats
			if abs_train_path.endswith('.txt'):
				# Text file with image paths
				with open(abs_train_path, 'r') as f:
					return len(f.readlines())
			else:
				# Directory with images
				image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
				return len([f for f in os.listdir(abs_train_path)
							if os.path.splitext(f)[1].lower() in image_exts])
		except Exception as e:
			print(f"Error counting images: {e}")
			return 0

	# If path is a directory (secondary models)
	elif os.path.isdir(path):
		total_count = 0
		image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

		# Walk through all subdirectories
		for root, dirs, files in os.walk(path):
			# Only count files in leaf directories (class directories)
			if not dirs:  # This is a leaf directory (no subdirectories)
				count = sum(1 for f in files
						   if os.path.splitext(f)[1].lower() in image_exts)
				total_count += count

		return total_count

	else:
		print(f"Unsupported dataset format: {path}")
		return 0


def free_gpu_memory(*objs):
	"""Release model references and clear the CUDA cache between training runs.

	The pipeline trains several YOLO models sequentially in one process. On a
	small (8 GB) GPU the allocator cache from a finished model is not returned,
	so the next model.train() can hit CUDA out-of-memory. Dropping the model
	references and emptying the cache avoids that.
	"""
	for o in objs:
		try:
			del o
		except Exception:
			pass
	gc.collect()
	try:
		import torch
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()
	except Exception:
		pass


def train_in_subprocess(weights, data, epochs, imgsz, project, workers=4, patience=None, train_overrides=None):
	"""Train one YOLO model in an isolated subprocess.

	The pipeline trains several models back-to-back. Doing so in a single
	process intermittently crashed with `CUDA error: resource already mapped`
	in the dataloader's pin-memory thread (cu130 / Blackwell GPUs): the second
	training inherited a corrupted CUDA context from the first. Running each
	training in its own process gives every model a fresh CUDA context, so the
	carry-over can no longer happen. The parent process never initialises CUDA
	for training, keeping it clean for later inference/tracking too.

	Raises RuntimeError if the worker process exits non-zero.
	"""
	worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BehaveAI_train_worker.py")
	cfg = {
		"cwd": os.getcwd(),
		"weights": weights,
		"data": data,
		"epochs": epochs,
		"imgsz": imgsz,
		"project": project,
		"workers": workers,
		"patience": patience,
		"train_overrides": train_overrides or {},
	}
	with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
		json.dump(cfg, tf)
		cfg_path = tf.name
	try:
		result = subprocess.run([sys.executable, worker, cfg_path])
		if result.returncode != 0:
			raise RuntimeError(
				f"Training subprocess failed (exit code {result.returncode}) for project '{project}'"
			)
	finally:
		try:
			os.remove(cfg_path)
		except OSError:
			pass


def _assert_weights(model_type, project_path, model_path):
	"""Fail here, with the context, when a training run left no weights behind.

	The caller loads model_path with YOLO() a few lines later. Without this the
	only symptom is a bare FileNotFoundError from inside torch.serialization,
	which says nothing about which model failed or why, and arrives after the
	training log has scrolled away.
	"""
	if os.path.exists(model_path):
		return
	found = []
	if os.path.isdir(project_path):
		for root, _dirs, files in os.walk(project_path):
			for f in files:
				if f.endswith('.pt'):
					found.append(os.path.relpath(os.path.join(root, f), project_path))
	# Where Ultralytics would have written the run had it ignored our absolute
	# project path (machine-global runs_dir + task + project). Naming it turns a
	# "no weights" dead end into a one-line fix for whoever reads the traceback.
	elsewhere = ''
	try:
		from ultralytics.utils import SETTINGS as _ULTRA_SETTINGS
		_runs = str(_ULTRA_SETTINGS.get('runs_dir', ''))
		if _runs:
			_stray = os.path.join(_runs, 'detect', os.path.basename(project_path), 'train')
			_hit = ' <-- weights are here' if os.path.exists(os.path.join(_stray, 'weights', 'best.pt')) else ''
			elsewhere = f"\n  Ultralytics runs_dir is '{_runs}'; check {_stray}{_hit}"
	except Exception:
		pass
	raise RuntimeError(
		f"Training reported success for '{model_type}' but produced no weights at "
		f"{model_path}.\n"
		f"  .pt files under {project_path}: {found or 'none'}\n"
		f"  Check the training log above for an Ultralytics error, and for a stray "
		f"'runs/' directory next to the project -- a leftover run there used to be "
		f"moved over the freshly trained model."
		f"{elsewhere}"
	)


def repair_dataset_yaml(yaml_path):
	"""Point a project's dataset yaml back at its own annotation folders.

	The yaml used to carry ABSOLUTE train/val paths, so a project copied to
	another machine (or a renamed project folder) still pointed at the annotation
	folders of the machine where the settings were last saved, and training died
	with "Dataset 'static_annotations.yaml' images not found, missing path
	C:\\...\\annot_static\\images\\val". Three cases: an absolute entry that does
	point into this project is rewritten relative (so the next copy works), a
	broken entry is redirected to the matching folder inside this project, and an
	entry resolving outside the project is only redirected when the in-project
	folder actually holds images. Re-saving the settings in the GUI rewrites the
	yaml too, but nothing forces the user to do that after copying a project over.
	"""
	if not str(yaml_path).endswith('.yaml') or not os.path.exists(yaml_path):
		return
	import yaml as _yaml
	base_dir = os.path.dirname(os.path.abspath(yaml_path))
	try:
		with open(yaml_path, 'r') as f:
			data = _yaml.safe_load(f) or {}
	except Exception as e:
		print(f"Warning: could not read {yaml_path}: {e}")
		return

	def _has_images(d):
		exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
		try:
			return any(f.lower().endswith(exts) for f in os.listdir(d))
		except OSError:
			return False

	fixed = {}
	# A stale dataset root would be used ahead of the yaml's own directory, so
	# drop it and let Ultralytics fall back to that directory.
	root = data.get('path')
	if isinstance(root, str) and not os.path.isdir(os.path.join(base_dir, root)):
		data.pop('path')
		fixed['path'] = ('(removed)', 'stale dataset root')

	for key in ('train', 'val'):
		entry = data.get(key)
		if not isinstance(entry, str):
			continue
		resolved = os.path.abspath(os.path.join(base_dir, entry))
		exists = os.path.isdir(resolved)
		# normcase: on Windows the same folder is often spelled with a different
		# case (BehaveAI_Herdwise vs BehaveAI_HERDWISE) in these yaml files.
		inside = os.path.normcase(resolved).startswith(os.path.normcase(base_dir) + os.sep)
		if exists and inside and not os.path.isabs(entry):
			continue
		if exists and inside:
			# Right folder, absolute spelling: rewriting it relative is what makes
			# this project survive being copied to another machine.
			data[key] = os.path.relpath(resolved, base_dir).replace('\\', '/')
			fixed[key] = (data[key], 'was an absolute path')
			continue
		# Keep the tail of the recorded path ('annot_static/images/val'): that
		# part survives moving the project, the drive and parents do not.
		parts = entry.replace('\\', '/').strip('/').split('/')
		for n in (3, 2, 1):
			if len(parts) < n:
				continue
			candidate = '/'.join(parts[-n:])
			cand_dir = os.path.join(base_dir, candidate)
			if not os.path.isdir(cand_dir):
				continue
			# An entry that resolves but points outside the project is only
			# redirected when the in-project folder actually holds images: the
			# empty placeholders created on every settings save must not silently
			# replace a dataset someone deliberately kept elsewhere.
			if exists and not _has_images(cand_dir):
				break
			data[key] = candidate
			fixed[key] = (candidate, 'pointed outside this project' if exists
						  else 'pointed at a folder that does not exist here')
			break
	if not fixed:
		return

	try:
		with open(yaml_path, 'w') as f:
			_yaml.safe_dump(data, f, sort_keys=False)
	except Exception as e:
		print(f"Warning: could not rewrite {yaml_path}: {e}")
		return
	for key, (val, why) in fixed.items():
		print(f"Repaired {os.path.basename(yaml_path)}: '{key}' -> '{val}' ({why})")


def maybe_retrain(model_type, yaml_path, project_path, model_path, classifier, epochs, imgsz, patience=None, train_overrides=None):
	"""
	Decide whether to (re)train a model based on existence and image counts.
	- If model_path exists and the recorded train_count differs from the current dataset,
	  prompt the user to retrain (Yes/No).
	- If model_path does not exist, perform first-time training.
	Returns True if a training run was performed, False otherwise.
	"""

	# Determine whether this is a motion model by naming
	is_motion_model = ('motion' in model_type.lower()) or ('secondary_motion' in project_path.lower()) or ('primary_motion' in project_path.lower())

	# If model exists: compare recorded image count (train_count.txt) with current dataset
	if os.path.exists(model_path):
		if os.path.exists(os.path.join(project_path, 'train_count.txt')):
			try:
				with open(os.path.join(project_path, 'train_count.txt'), 'r') as f:
					last_count = int(f.read().strip())
			except Exception:
				last_count = -1
		else:
			last_count = -1

		current_count = count_images_in_dataset(yaml_path)

		# Counts match -> annotations unchanged, nothing to do.
		if current_count == last_count:
			return False

		# New annotations detected -> retrain automatically, no confirmation
		# dialog. Models whose annotation count is unchanged are skipped above, so
		# pressing "Train" trains static/motion/secondary back-to-back with no
		# intermediate pop-ups.
		print(
			f"New annotations detected for '{model_type}' model: image count "
			f"changed from {last_count} to {current_count}. Re-training..."
		)
		# Backup existing model dir/project and retrain from its weights
		backup_dir = project_path + "_backup"
		i = 1
		while os.path.exists(f"{backup_dir}{i}"):
			i += 1
		final_backup = f"{backup_dir}{i}"
		try:
			shutil.copytree(project_path, final_backup)
			print(f"Existing model copied to {final_backup}")
		except Exception as e:
			print(f"Warning: failed to backup {project_path}: {e}")

		start_weights = os.path.join(final_backup, "train", "weights", "best.pt")
		print(f'Training new {model_type} model using existing weights...')
		train_in_subprocess(start_weights, yaml_path, epochs, imgsz, project_path, patience=patience, train_overrides=train_overrides)
		move_to_expected(project_path, run_name="train", runs_root="runs")
		_assert_weights(model_type, project_path, model_path)
		print(f'Done training {model_type} model')
		# Update saved train count
		with open(os.path.join(project_path, 'train_count.txt'), 'w') as f:
			f.write(str(current_count))
		# copy existing settings ini file for reference (so you know which settings were used for each model)
		os.makedirs(project_path, exist_ok=True)
		# ~ dst = os.path.join(project_path, os.path.basename(config_path))
		dst = os.path.join(project_path, 'saved_settings.ini')
		try:
			shutil.copy2(config_path, dst)
			print(f"Saved settings snapshot to {dst}")
		except Exception as e:
			print(f"Warning: could not copy settings to model dir: {e}")
		return True

	else:
		# Model missing -> do first-time training
		print(f'{model_type} model not found, building it...')
		train_in_subprocess(classifier, yaml_path, epochs, imgsz, project_path, patience=patience, train_overrides=train_overrides)
		move_to_expected(project_path, run_name="train", runs_root="runs")
		_assert_weights(model_type, project_path, model_path)
		print(f'Done training {model_type} model')

		current_count = count_images_in_dataset(yaml_path)
		os.makedirs(project_path, exist_ok=True)
		with open(os.path.join(project_path, 'train_count.txt'), 'w') as f:
			f.write(str(current_count))

		# copy existing settings ini file for reference (so you know which settings were used for each model)
		os.makedirs(project_path, exist_ok=True)
		# ~ dst = os.path.join(project_path, os.path.basename(config_path))
		dst = os.path.join(project_path, 'saved_settings.ini')
		try:
			shutil.copy2(config_path, dst)
			print(f"Saved settings snapshot to {dst}")
		except Exception as e:
			print(f"Warning: could not copy settings to model dir: {e}")

		return True


# Two per-stream secondary classifiers, trained on the pooled crop folders
# annot_<stream>_crop/<secondary>/ (one model per stream, shared label pool).
secondary_static_model = None
secondary_motion_model = None

# Species (model 0) / age (model 0.5) classifiers: single models, project-wide
# (not species-scoped), loaded once. Absent until enough species/age crops have
# been annotated to train them (see _count_class_subdirs gate in __main__ below).
model_species = None
model_age = None


def _count_class_subdirs(d):
	"""Count immediate subfolders of `d` that contain at least one image."""
	if not d or not os.path.isdir(d):
		return 0
	n = 0
	for name in os.listdir(d):
		sub = os.path.join(d, name)
		try:
			if os.path.isdir(sub) and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(sub)):
				n += 1
		except Exception:
			continue
	return n


def _classify_pooled(model, class_names, crop):
	"""Run a plain (non-hierarchical) crop classifier - used for the species
	(model 0) and age (model 0.5) classifiers, which have no allowed-subset
	restriction and no '__none__' sentinel (unlike the per-primary secondary
	classifiers above). Returns (class_name, confidence) or ('', 0.0)."""
	if model is None or crop is None or crop.size == 0 or not class_names:
		return '', 0.0
	try:
		res = model.predict(crop, verbose=False)
	except Exception:
		return '', 0.0
	if not res or res[0].probs is None:
		return '', 0.0
	probs = res[0].probs.data
	best_name, best_conf = '', -1.0
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
	return (best_name, best_conf) if best_name else ('', 0.0)


if __name__ == '__main__':
	#-------CHECK PRIMARY MODEL EXISTS----------
	if primary_static_classes:
		repair_dataset_yaml(primary_static_yaml_path)
		_static_train_yaml = primary_static_yaml_path
		if sahi_enabled_static:
			# Tile the annotated dataset so training sees horses at the same
			# (tile) scale SAHI feeds at inference. Freshness-cached: only
			# re-tiles when the source annotation count changed.
			from BehaveAI_tiling import tile_dataset
			_static_train_yaml = tile_dataset(
				primary_static_yaml_path,
				slice_h=sahi_slice_height, slice_w=sahi_slice_width,
				overlap_h=sahi_overlap_height_ratio, overlap_w=sahi_overlap_width_ratio)
		maybe_retrain('primary static', _static_train_yaml, primary_static_project_path,
			primary_static_model_path, primary_classifier, primary_epochs, primary_imgsz, patience=train_patience, train_overrides=primary_train_overrides)


	if primary_motion_classes:
		repair_dataset_yaml(primary_motion_yaml_path)
		_motion_train_yaml = primary_motion_yaml_path
		if sahi_enabled_motion:
			from BehaveAI_tiling import tile_dataset
			_motion_train_yaml = tile_dataset(
				primary_motion_yaml_path,
				slice_h=sahi_slice_height, slice_w=sahi_slice_width,
				overlap_h=sahi_overlap_height_ratio, overlap_w=sahi_overlap_width_ratio)
		maybe_retrain('primary motion', _motion_train_yaml, primary_motion_project_path,
			primary_motion_model_path, primary_classifier, primary_epochs, primary_imgsz, patience=train_patience, train_overrides=primary_motion_train_overrides)

	if hierarchical_mode:
		# Static stream (pooled over all static primaries)
		_static_class_count = _count_class_subdirs(secondary_static_data_path)
		if _static_class_count >= 2:
			# Whole-video train/val split, same partition as the detectors above.
			# Without it Ultralytics splits the crop pool itself, per crop and
			# unseeded (see behaveai_holdout.build_classification_split).
			_static_crop_data = build_classification_split(secondary_static_data_path, val_frequency)
			maybe_retrain('secondary_static', _static_crop_data, secondary_static_project_path,
				secondary_static_model_path, secondary_classifier, secondary_epochs, secondary_imgsz, patience=train_patience, train_overrides=secondary_train_overrides)
			if use_ncnn == 'true':
				secondary_static_model = load_model_with_ncnn_preference(secondary_static_model_path, "classify")
			else:
				secondary_static_model = YOLO(secondary_static_model_path)
		else:
			print(f"WARNING: Secondary static classifier skipped: only {_static_class_count} class "
				f"folder(s) with crops in '{secondary_static_data_path}', need >=2. "
				f"Static secondary behaviours will not be predicted until you annotate "
				f"at least 2 distinct secondary classes.")

		# Motion stream (pooled over all motion primaries)
		_motion_class_count = _count_class_subdirs(secondary_motion_data_path)
		if _motion_class_count >= 2:
			_motion_crop_data = build_classification_split(secondary_motion_data_path, val_frequency)
			maybe_retrain('secondary_motion', _motion_crop_data, secondary_motion_project_path,
				secondary_motion_model_path, secondary_classifier, secondary_epochs, secondary_imgsz, patience=train_patience, train_overrides=secondary_motion_train_overrides)
			if use_ncnn == 'true':
				secondary_motion_model = load_model_with_ncnn_preference(secondary_motion_model_path, "classify")
			else:
				secondary_motion_model = YOLO(secondary_motion_model_path)
		else:
			print(f"WARNING: Secondary motion classifier skipped: only {_motion_class_count} class "
				f"folder(s) with crops in '{secondary_motion_data_path}', need >=2. "
				f"Motion secondary behaviours will not be predicted until you annotate "
				f"at least 2 distinct secondary classes.")

	# Species (model 0) / age (model 0.5): same pooled-classifier pattern as the
	# secondary models above, gated on >=2 annotated classes (can't train a
	# classifier to distinguish only one class - e.g. model 0 needs a 2nd species
	# actually annotated before it can do anything).
	_species_class_count = _count_class_subdirs(species_cropped_base_dir)
	if _species_class_count >= 2:
		_species_crop_data = build_classification_split(species_cropped_base_dir, val_frequency)
		maybe_retrain('species', _species_crop_data, 'model_species',
			species_model_path, secondary_classifier, secondary_epochs, secondary_imgsz, patience=train_patience, train_overrides=secondary_train_overrides)
		if use_ncnn == 'true':
			model_species = load_model_with_ncnn_preference(species_model_path, "classify")
		else:
			model_species = YOLO(species_model_path)
	else:
		print(f"WARNING: Species classifier (model 0) skipped: only {_species_class_count} class "
			f"folder(s) with crops in '{species_cropped_base_dir}', need >=2. Species will not be "
			f"predicted until you annotate at least 2 distinct species.")

	_age_class_count = _count_class_subdirs(age_cropped_base_dir)
	if _age_class_count >= 2:
		_age_crop_data = build_classification_split(age_cropped_base_dir, val_frequency)
		maybe_retrain('age', _age_crop_data, 'model_age',
			age_model_path, secondary_classifier, secondary_epochs, secondary_imgsz, patience=train_patience, train_overrides=secondary_train_overrides)
		if use_ncnn == 'true':
			model_age = load_model_with_ncnn_preference(age_model_path, "classify")
		else:
			model_age = YOLO(age_model_path)
	else:
		print(f"WARNING: Age classifier (model 0.5) skipped: only {_age_class_count} class "
			f"folder(s) with crops in '{age_cropped_base_dir}', need >=2. Age will not be "
			f"predicted until you annotate at least 2 distinct age classes.")


	# --- PARAMETERS -----------------------------------------------------------

	expA2 = 1 - expA
	expB2 = 1 - expB

	# input_folder / output_folder come from the INI (resolved at module level).
	# They used to be overwritten here with <project_dir>/input and
	# <project_dir>/output, which silently ignored input_dir/output_dir: a
	# project pointing its videos elsewhere found nothing to process and said
	# nothing about it.
	print(f"Reading videos from: {input_folder}")
	print(f"Writing results to:  {output_folder}")

	progress_update = 10 # print progress every n frames

	def iou(box1, box2):
		xa = max(box1[0], box2[0]); ya = max(box1[1], box2[1])
		xb = min(box1[2], box2[2]); yb = min(box1[3], box2[3])
		inter = max(0, xb-xa) * max(0, yb-ya)
		area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
		area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
		# A box can be degenerate (zero width or height): YOLO occasionally
		# returns one thinner than a pixel and int() collapses it. There is no
		# area to express the intersection as a proportion of, and inter is 0
		# anyway, so report no overlap instead of dividing by zero -- that crash
		# used to kill the run 88% into a video.
		if area1 <= 0 or area2 <= 0:
			return 0
		prop1 = inter/area1
		prop2 = inter/area2
		# return the larger proportional overlap - e.g. if one box is entirely inside another, this will return a 1.0, whereas the previous wouldn't
		if prop1 > prop2:
			return prop1 if prop1 > 0 else 0
		else:
			return prop2 if prop2 > 0 else 0
		# ~ union = area1 + area2 - inter
		# ~ return inter/union if union > 0 else 0


	# --- TRACKER CLASS -------------------------------------------------------
	class KalmanTracker:
		def __init__(self, dist_thresh, max_missed):
			self.next_id = 1
			self.tracks = {}  # tid -> {'kf': KalmanFilter, 'missed': int}
			self.prev_positions = {}  # Track previous positions
			self.dist_thresh = dist_thresh
			self.max_missed = max_missed

		def _create_kf(self, initial_pt):

			#Create a 4D state (x, y, vx, vy) Kalman Filter measuring (x, y).
			kf = cv2.KalmanFilter(4, 2)
			# State transition: x' = x + vx, y' = y + vy
			kf.transitionMatrix = np.array([[1, 0, 1, 0],
											[0, 1, 0, 1],
											[0, 0, 1, 0],
											[0, 0, 0, 1]], dtype=np.float32)
			# Measurement: we only observe x, y
			kf.measurementMatrix = np.array([[1, 0, 0, 0],
											 [0, 1, 0, 0]], dtype=np.float32)
			# Tune these covariances to your scene

			kf.processNoiseCov = np.diag([process_noise_pos, process_noise_pos, process_noise_vel, process_noise_vel]).astype(np.float32)
			kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
			# Initialize state
			kf.statePre  = np.array([[initial_pt[0]],
									 [initial_pt[1]],
									 [0.],
									 [0.]], dtype=np.float32)
			kf.statePost = kf.statePre.copy()
			return kf

			self._prune_duplicate_tracks()

		def predict_all(self):
			"""
			Predict the next position for every track.
			Returns list of (tid, predicted_pt).
			"""
			preds = []
			for tid, tr in self.tracks.items():
				pred = tr['kf'].predict()
				preds.append((tid, (float(pred[0, 0]), float(pred[1, 0]))))
			return preds



		def _prune_duplicate_tracks(self):
			"""
			Merge any two tracks whose current posteriors are very close.
			Call this at the end of update().
			"""
			tids = list(self.tracks.keys())
			posts = {}
			for tid in tids:
				sp = self.tracks[tid]['kf'].statePost
				posts[tid] = (float(sp[0,0]), float(sp[1,0]))
			to_drop = set()
			for i, t1 in enumerate(tids):
				x1, y1 = posts[t1]
				for t2 in tids[i+1:]:
					x2, y2 = posts[t2]
					if np.hypot(x1-x2, y1-y2) < self.dist_thresh * 0.5:
						# mark the higher ID for deletion
						to_drop.add(max(t1, t2))
			for tid in to_drop:
				del self.tracks[tid]

		def update(self, detections):

			#detections: list of (x, y) centroids
			#Returns a dict: detection_index -> track_id

			# 1) Predict all tracks forward one step
			preds = self.predict_all()  # list of (tid, (px, py))
			track_ids = [t[0] for t in preds]
			pred_pts   = [t[1] for t in preds]

			# 2) Build cost matrix = Euclidean distance
			if pred_pts and detections:
				cost = np.zeros((len(pred_pts), len(detections)), dtype=np.float32)
				for i, p in enumerate(pred_pts):
					for j, d in enumerate(detections):
						cost[i, j] = np.hypot(p[0] - d[0], p[1] - d[1])
				row_idx, col_idx = linear_sum_assignment(cost)
			else:
				row_idx = np.array([], dtype=int)
				col_idx = np.array([], dtype=int)

			assigned_detects = {}
			matched_tracks = set()
			matched_dets   = set()

			# 3) Associate tracks ↔ detections
			for r, c in zip(row_idx, col_idx):
				if cost[r, c] < self.dist_thresh:
					tid = track_ids[r]
					matched_tracks.add(tid)
					matched_dets.add(c)
					assigned_detects[c] = tid

					# Get the measurement point
					dpt = detections[c]
					meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])

					# Correct KF with the detection measurement
					self.tracks[tid]['kf'].correct(meas)
					self.tracks[tid]['missed'] = 0

					# Update previous position
					self.prev_positions[tid] = (dpt[0], dpt[1])

			# 4) Process unassigned detections
			for i, dpt in enumerate(detections):
				if i in matched_dets:
					continue

				# try to find an existing track under the threshold
				best_tid, best_dist = None, float('inf')
				for tid, (px, py) in preds:
					d = np.hypot(dpt[0]-px, dpt[1]-py)
					if d < best_dist:
						best_dist, best_tid = d, tid

				if best_dist < self.dist_thresh:
					assigned_detects[i] = best_tid
					self.tracks[best_tid]['missed'] = 0
					meas = np.array([[np.float32(dpt[0])], [np.float32(dpt[1])]])
					self.tracks[best_tid]['kf'].correct(meas)

					# Update previous position
					self.prev_positions[best_tid] = (dpt[0], dpt[1])
					matched_tracks.add(best_tid)  # Add to matched tracks

				else:
					# New track
					tid = self.next_id
					kf = self._create_kf(dpt)
					self.tracks[tid] = {'kf': kf, 'missed': 0}
					assigned_detects[i] = tid
					self.prev_positions[tid] = (dpt[0], dpt[1])  # Initialize position
					matched_tracks.add(tid)  # Add to matched tracks
					self.next_id += 1

			# 5) Handle unmatched tracks
			for tid in list(self.tracks.keys()):
				if tid not in matched_tracks:
					self.tracks[tid]['missed'] += 1
					# Increase uncertainty when missing detections
					noise_scale = min(2.0, 1.0 + self.tracks[tid]['missed'] * 0.2)

					# FIXED: Preserve matrix type and structure
					kf = self.tracks[tid]['kf']
					new_noise = kf.processNoiseCov.copy()
					new_noise *= noise_scale
					kf.processNoiseCov = new_noise

					# Remove track if missed too many times
					if self.tracks[tid]['missed'] > self.max_missed:
						del self.tracks[tid]
						if tid in self.prev_positions:
							del self.prev_positions[tid]

			return assigned_detects


	# --- MAIN PROCESSING -----------------------------------------------------
	def process_video(file):
		os.makedirs(output_folder, exist_ok=True)
		base = os.path.splitext(os.path.basename(file))[0]
		cap = cv2.VideoCapture(file)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		if not cap.isOpened(): return
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*scale_factor)
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*scale_factor)
		fps = cap.get(cv2.CAP_PROP_FPS)

		# Compute frame limit from analysis duration setting (0 = no limit / full video)
		if ab_analysis_duration_s and ab_analysis_duration_s > 0:
			max_frame_limit = int(ab_analysis_duration_s * fps)
			print(f"Analysis window: first {ab_analysis_duration_s:.0f}s "
			      f"({max_frame_limit} frames at {fps:.1f} fps) — "
			      f"full video is {total_frames} frames "
			      f"({total_frames / fps:.0f}s)")
		else:
			max_frame_limit = None

		writer = None
		if detection_video_enabled:
			writer = cv2.VideoWriter(
				os.path.join(output_folder, base + "_detected.mp4"),
				cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
			)

		if primary_static_classes:
			if use_ncnn == 'true':
				model_static = load_model_with_ncnn_preference(primary_static_model_path, "detect")
			else:
				model_static = YOLO(primary_static_model_path)

		if primary_motion_classes:
			if use_ncnn == 'true':
				model_motion = load_model_with_ncnn_preference(primary_motion_model_path, "detect")
			else:
				model_motion = YOLO(primary_motion_model_path)

		# Optional SAHI sliced-inference wrappers, built once per video. Only when
		# the stream flag is on AND the (post-scale) frame is meaningfully larger
		# than a tile — otherwise tiling is pure overhead. Any failure leaves the
		# wrapper None and the detection block falls back to whole-frame .predict.
		sahi_model_static = None
		sahi_model_motion = None
		_sahi_worth_it = max(w, h) > max(sahi_slice_width, sahi_slice_height) * sahi_min_dim_factor
		if (sahi_enabled_static or sahi_enabled_motion) and not _sahi_worth_it:
			print(f"SAHI: frame {w}x{h} not larger than tile x {sahi_min_dim_factor:g}"
				  f" — tiling skipped, using whole-frame detection.")
		if sahi_enabled_static and _sahi_worth_it and primary_static_classes:
			sahi_model_static = build_sahi_model(model_static, primary_conf_thresh)
		if sahi_enabled_motion and _sahi_worth_it and primary_motion_classes:
			sahi_model_motion = build_sahi_model(model_motion, primary_conf_thresh)


		# Tracker: BoT-SORT/ByteTrack (default) or the legacy Kalman tracker.
		bot_tracker = None
		if tracker_type in ('botsort', 'bytetrack'):
			from ultralytics.trackers.bot_sort import BOTSORT
			from ultralytics.trackers.byte_tracker import BYTETracker
			from ultralytics.utils import IterableSimpleNamespace, YAML
			from ultralytics.utils.checks import check_yaml
			_tcfg = IterableSimpleNamespace(**YAML.load(check_yaml(
				'botsort.yaml' if tracker_type == 'botsort' else 'bytetrack.yaml')))
			_tcfg.track_high_thresh = tracker_track_high_thresh
			_tcfg.track_low_thresh = tracker_track_low_thresh
			_tcfg.new_track_thresh = tracker_new_track_thresh
			_tcfg.track_buffer = tracker_track_buffer
			_tcfg.match_thresh = tracker_match_thresh
			if tracker_type == 'botsort':
				_tcfg.gmc_method = tracker_gmc_method
			_fps_int = int(fps) if fps and fps > 0 else 30
			bot_tracker = (BOTSORT(args=_tcfg, frame_rate=_fps_int) if tracker_type == 'botsort'
						   else BYTETracker(args=_tcfg, frame_rate=_fps_int))
			tracker = None
			print(f"Tracker: {tracker_type} (Ultralytics).")
		else:
			tracker = KalmanTracker(match_distance_thresh, delete_after_missed)

		# Previous centroid per track id, for the tracker-agnostic motion vector.
		prev_centroid = {}

		prev_frames, frame_idx = None, 0
		csv_file = open(os.path.join(output_folder, base + "_tracking.csv"), 'w', newline='')
		csv_writer = csv.writer(csv_file)
		# Updated CSV header with four streams.
		# The first 12 columns keep their original order/meaning; the bounding-box
		# geometry (x1, y1, x2, y2) is appended additively so downstream tools that
		# only read the first 12 columns (e.g. activity budget) are unaffected.
		# species_class/age_class (model 0/0.5) are appended after x1,y1,x2,y2 -
		# same additive convention, existing readers of the first 16 columns unaffected.
		# `source` (last column, same convention) names the stream whose detection
		# won the box, which the per-stream class columns no longer imply now that
		# they are filled by source: it says which image the crop classifiers ran on.
		csv_writer.writerow([
			"frame", "id", "x", "y",
			"primary_static_class", "primary_static_conf",
			"primary_motion_class", "primary_motion_conf",
			"secondary_static_class", "secondary_static_conf",
			"secondary_motion_class", "secondary_motion_conf",
			"x1", "y1", "x2", "y2",
			"species_class", "species_conf", "age_class", "age_conf",
			"source"
		])

		print(f"Processing video: {file}")
		print('Initialising')
		current_frame = 0
		print_tick = 0
		start_time = time.time()

		frame_count = 0

		while True:
			ret, raw_frame = cap.read()
			if not ret: break
			frame_idx += 1

			# Stop processing once the analysis window is reached
			if max_frame_limit is not None and frame_idx > max_frame_limit:
				break

			if frame_count == 0:
				if scale_factor != 1.0:
					raw_frame = cv2.resize(raw_frame, None, fx=scale_factor, fy=scale_factor)
				gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
				frame = raw_frame.copy()
				if prev_frames is None:
					prev_frames = [gray.copy() for _ in range(3)]
					continue

				# only process motion information if necessary
				if primary_motion_classes:

					diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]

					if strategy == 'exponential':
						prev_frames[0] = gray
						prev_frames[1] = cv2.addWeighted(prev_frames[1], expA, gray, expA2, 0)
						prev_frames[2] = cv2.addWeighted(prev_frames[2], expB, gray, expB2, 0)
					elif strategy == 'sequential':
						prev_frames[2] = prev_frames[1]
						prev_frames[1] = prev_frames[0]
						prev_frames[0] = gray


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

				# Collect all primary detections
				all_detections = []

				# Primary static detection (SAHI-tiled when enabled, else whole-frame)
				if primary_static_classes:
					if sahi_model_static is not None:
						all_detections.extend(sahi_detect(
							sahi_model_static, frame, primary_static_classes, 'static',
							sahi_slice_height, sahi_slice_width,
							sahi_overlap_height_ratio, sahi_overlap_width_ratio,
							sahi_postprocess_type, sahi_postprocess_match_metric,
							sahi_postprocess_match_threshold, sahi_perform_standard_pred))
					else:
						results_static = model_static.predict(frame, conf=primary_conf_thresh, verbose=False)
						for box in results_static[0].boxes:
							coords = tuple(map(int, box.xyxy[0].tolist()))
							class_idx = int(box.cls[0])
							class_name = primary_static_classes[class_idx]
							conf = float(box.conf[0])
							all_detections.append({
								'coords': coords,
								'primary_class': class_name,
								'primary_conf': conf,
								'source': 'static',
							})

				# Primary motion detection (SAHI-tiled when enabled, else whole-frame)
				if primary_motion_classes:
					if sahi_model_motion is not None:
						all_detections.extend(sahi_detect(
							sahi_model_motion, motion_image, primary_motion_classes, 'motion',
							sahi_slice_height, sahi_slice_width,
							sahi_overlap_height_ratio, sahi_overlap_width_ratio,
							sahi_postprocess_type, sahi_postprocess_match_metric,
							sahi_postprocess_match_threshold, sahi_perform_standard_pred))
					else:
						results_motion = model_motion.predict(motion_image, conf=primary_conf_thresh, verbose=False)
						for box in results_motion[0].boxes:
							coords = tuple(map(int, box.xyxy[0].tolist()))
							class_idx = int(box.cls[0])
							class_name = primary_motion_classes[class_idx]
							conf = float(box.conf[0])
							all_detections.append({
								'coords': coords,
								'primary_class': class_name,
								'primary_conf': conf,
								'source': 'motion',
							})


				# Drop degenerate boxes before anything consumes them. A detection
				# thinner than a pixel collapses to zero width or height once the
				# corners are cast to int, and such a box has no crop to classify
				# and no area to compare -- it is junk, not a detection.
				_usable = [d for d in all_detections
						   if d['coords'][2] > d['coords'][0] and d['coords'][3] > d['coords'][1]]
				if len(_usable) != len(all_detections):
					print(f"\n  frame {frame_idx}: dropped "
						  f"{len(all_detections) - len(_usable)} zero-area detection(s)")
				all_detections = _usable

				# Merge detections based on proximity
				merged_detections = []
				for det in all_detections:
					x1, y1, x2, y2 = det['coords']
					cx, cy = (x1+x2)//2, (y1+y2)//2

					# Find matching existing detection
					matched = False
					for md in merged_detections:
						md_cx, md_cy = md['centroid']
						dist = np.hypot(cx - md_cx, cy - md_cy)

						# Calculate IOU
						md_x1, md_y1, md_x2, md_y2 = md['coords']
						overlap = iou((x1, y1, x2, y2), (md_x1, md_y1, md_x2, md_y2))
						ms_source = md['source']

						if dist < centroid_merge_thresh or overlap > iou_thresh:
							# Record what each stream said BEFORE the winner-takes-all
							# arbitration below overwrites primary_class.
							record_stream_verdict(md, det)

							# Merge classes - keep highest confidence detection for each source
							if det['source'] == ms_source or dominant_source == 'confidence': # mathcing sources so select best, or confidence strategy used
								if det['source'] == 'static':
									# Keep highest confidence static detection
									if 'primary_conf' not in md or det['primary_conf'] > md['primary_conf']:
										md['primary_class'] = det['primary_class']
										md['primary_conf'] = det['primary_conf']
										md['coords'] = det['coords']  # Update to higher conf box
										md['centroid'] = (cx, cy)
										md['source'] = det['source']
								else:  # motion source
									# Keep highest confidence motion detection
									if 'primary_conf' not in md or det['primary_conf'] > md['primary_conf']:
										md['primary_class'] = det['primary_class']
										md['primary_conf'] = det['primary_conf']
										md['coords'] = det['coords']  # Update to higher conf box
										md['centroid'] = (cx, cy)
										md['source'] = det['source']
							elif det['source'] == 'static' and dominant_source == 'static':
									# Keep static detection
									md['primary_class'] = det['primary_class']
									md['primary_conf'] = det['primary_conf']
									md['coords'] = det['coords']  # Update to higher conf box
									md['centroid'] = (cx, cy)
									md['source'] = det['source']
							elif det['source'] == 'motion' and dominant_source == 'motion':
									# Keep motion detection
									md['primary_class'] = det['primary_class']
									md['primary_conf'] = det['primary_conf']
									md['coords'] = det['coords']  # Update to higher conf box
									md['centroid'] = (cx, cy)
									md['source'] = det['source']


							matched = True
							break

					if not matched:
						# Add as new detection
						new_det = {
							'coords': det['coords'],
							'centroid': (cx, cy),
							'source': det['source'],
							'static_class': '',
							'static_conf': 0.0,
							'motion_class': '',
							'motion_conf': 0.0,
						}
						record_stream_verdict(new_det, det)
						if det['source'] == 'static':
							new_det['primary_class'] = det['primary_class']
							new_det['primary_conf'] = det['primary_conf']
							# ~ if 'secondary_static_class' in det:
								# ~ new_det['secondary_static_class'] = det['secondary_static_class']
								# ~ new_det['secondary_static_conf'] = det['secondary_static_conf']
						else:  # motion source
							new_det['primary_class'] = det['primary_class']
							new_det['primary_conf'] = det['primary_conf']
							# ~ if 'secondary_motion_class' in det:
								# ~ new_det['secondary_motion_class'] = det['secondary_motion_class']
								# ~ new_det['secondary_motion_conf'] = det['secondary_motion_conf']
						merged_detections.append(new_det)



				# Run secondary classification on each primary detection
				processed_detections = []
				for det in merged_detections:
					coords = det['coords']
					primary_class = det['primary_class']
					source = det['source']

					# One column pair per stream, filled by source (see
					# record_stream_verdict). An empty class means that detector did
					# not see this animal at all -- which is the signal the frame
					# miner reads to find what one stream misses.
					det['primary_static_class'] = det['static_class']
					det['primary_static_conf'] = det['static_conf']
					det['primary_motion_class'] = det['motion_class']
					det['primary_motion_conf'] = det['motion_conf']

					# Species (model 0) / age (model 0.5): run on the same crop the
					# secondary classifiers use below, unconditionally (mandatory per
					# detection, unlike the optional/hierarchical-only secondary).
					x1, y1, x2, y2 = coords
					crop_img_sa = frame if source == 'static' else motion_image
					crop_sa = crop_img_sa[y1:y2, x1:x2] if crop_img_sa is not None else None
					species_class, species_conf = _classify_pooled(model_species, species_list, crop_sa)
					age_class, age_conf = _classify_pooled(model_age, age_classes, crop_sa)
					det['species_class'] = species_class
					det['species_conf'] = species_conf
					det['age_class'] = age_class
					det['age_conf'] = age_conf

					if hierarchical_mode:
						x1, y1, x2, y2 = coords

						# Per-stream secondary model + crop source, and the global
						# primary index (to restrict output to allowed secondaries).
						if source == 'static':
							sec_model = secondary_static_model
							crop_img = frame
							try:
								g_idx = primary_static_classes.index(primary_class)
							except ValueError:
								g_idx = -1
						else:  # motion source
							sec_model = secondary_motion_model
							crop_img = motion_image
							try:
								g_idx = len(primary_static_classes) + primary_motion_classes.index(primary_class)
							except ValueError:
								g_idx = -1

						allowed = allowed_secondary_idx[g_idx] if 0 <= g_idx < len(allowed_secondary_idx) else []

						secondary_class = ''
						secondary_conf = 0.0

						# Classify only when a model exists and the primary allows secondaries;
						# restrict the prediction to the allowed secondaries of this primary.
						if sec_model is not None and allowed and crop_img is not None:
							crop = crop_img[y1:y2, x1:x2]
							if crop is not None and crop.size > 0:
								try:
									sec_results = sec_model.predict(crop, verbose=False)
								except Exception:
									sec_results = None
								if sec_results and sec_results[0].probs is not None:
									allowed_names = set(secondary_classes[i] for i in allowed)
									allowed_names.add(NONE_LABEL)   # model may vote "no secondary"
									probs = sec_results[0].probs.data
									best_conf = -1.0
									best_name = ''
									for m_idx, nm in sec_model.names.items():
										if nm not in allowed_names:
											continue
										try:
											c = float(probs[m_idx])
										except Exception:
											continue
										if c > best_conf:
											best_conf = c
											best_name = nm
									# The explicit __none__ class winning means "no secondary".
									# secondary_conf_thresh is then a second, stricter floor on
									# the winner: a sub-label that beats __none__ but scores
									# below it is left undecided. 0 keeps the __none__ vote alone.
									if (best_name and best_name != NONE_LABEL
											and best_conf >= secondary_conf_thresh):
										secondary_class = best_name
										secondary_conf = best_conf

						# Add secondary results to detection
						if source == 'static':
							det['secondary_static_class'] = secondary_class
							det['secondary_static_conf'] = secondary_conf
						else:  # motion source
							det['secondary_motion_class'] = secondary_class
							det['secondary_motion_conf'] = secondary_conf

					processed_detections.append(det)


				# Prepare for tracking
				cents = [d['centroid'] for d in processed_detections]

				if bot_tracker is not None:
					assignment = bot_track_update(bot_tracker, processed_detections, frame)
				else:
					assignment = tracker.update(cents)

				# ~ frame = motion_image ## enable this line ot save the motion video instead of static

				# Process tracked objects
				for idx, det in enumerate(processed_detections):
					tid = assignment.get(idx, None)
					if tid is None:
						continue
					if tracker is not None and tid not in tracker.tracks:
						continue

					x1, y1, x2, y2 = det['coords']
					cx, cy = det['centroid']

					# Get all class info with default values
					ps_class = det.get('primary_static_class', '')
					ps_conf = det.get('primary_static_conf', 0)
					pm_class = det.get('primary_motion_class', '')
					pm_conf = det.get('primary_motion_conf', 0)
					ss_class = det.get('secondary_static_class', '')
					ss_conf = det.get('secondary_static_conf', 0)
					sm_class = det.get('secondary_motion_class', '')
					sm_conf = det.get('secondary_motion_conf', 0)
					p_source = det.get('source', '')
					sp_class = det.get('species_class', '')
					sp_conf = det.get('species_conf', 0)
					ag_class = det.get('age_class', '')
					ag_conf = det.get('age_conf', 0)
					# Display-only fallback: the species/age classifiers only exist when the
					# project defines >=2 classes (see the model-loading gate above), so a
					# single-species/single-age project would otherwise never show them.
					# Deliberately not written back to det/CSV - those keep the model's own
					# (possibly empty) verdict.
					sp_display = sp_class or (species_list[0] if len(species_list) == 1 else '')
					ag_display = ag_class or (age_classes[0] if len(age_classes) == 1 else '')
					top_lines = []
					if render_style.show_species and sp_display:
						top_lines.append(sp_display)
					if render_style.show_age and ag_display:
						top_lines.append(ag_display)


					# Create display label
					label_parts = []
					# ~ if ps_class:
					if p_source == 'static':
						label_parts.append(f"{ps_class.upper()}")
						primary_cls = ps_class
					else:
						label_parts.append(f"{pm_class.upper()}")
						primary_cls = pm_class

					primary_col = primary_colors[primary_classes.index(primary_cls)]
					secondary_col = (255, 255, 255)
					secondary_cls = ''      # reset per detection to avoid leaking the previous box's secondary

					if hierarchical_mode:

						if sm_class != '' and sm_class != primary_cls:
							secondary_cls = sm_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]
						if ss_class != '' and ss_class != primary_cls:
							secondary_cls = ss_class
							secondary_col = secondary_colors[secondary_classes.index(secondary_cls)]


						# Primary-only label when this primary forbids secondaries OR the
						# secondary was rejected (below threshold) -> secondary_cls stayed ''.
						if primary_cls in ignore_secondary or secondary_cls == '':
							label = f"{tid} {primary_cls.upper()}"
							draw_labeled_detection(frame, x1, y1, x2, y2, x2-x1, y2-y1,
												 render_style, primary_color=primary_col,
												 top_lines=top_lines, label=label,
												 bounds=(frame.shape[1], frame.shape[0]))
						else:
							label = f"{tid} {primary_cls.upper()} {secondary_cls}"
							draw_labeled_detection(frame, x1, y1, x2, y2, x2-x1, y2-y1,
												 render_style, primary_color=primary_col,
												 secondary_color=secondary_col, hierarchical=True,
												 top_lines=top_lines, label=label,
												 bounds=(frame.shape[1], frame.shape[0]))
					else:
						label = f"{tid} {primary_cls}"
						draw_labeled_detection(frame, x1, y1, x2, y2, x2-x1, y2-y1,
											 render_style, primary_color=primary_col,
											 top_lines=top_lines, label=label,
											 bounds=(frame.shape[1], frame.shape[0]))




					# Draw motion vector: finite difference of this id's centroid
					# between frames (tracker-agnostic; no Kalman internals).
					prev = prev_centroid.get(tid)
					if prev is not None:
						vx, vy = cx - prev[0], cy - prev[1]
						next_x, next_y = cx + vx, cy + vy
						if all(np.isfinite(v) for v in (cx, cy, next_x, next_y)):
							light_color = tuple(int(0.8 * ch + 0.2 * 255) for ch in primary_col)
							cv2.line(frame, (int(cx), int(cy)), (int(next_x), int(next_y)), primary_col, line_thickness)
							cv2.circle(frame, (int(next_x), int(next_y)), 3, light_color, -line_thickness)
							cv2.circle(frame, (int(cx), int(cy)), 3, primary_col, -line_thickness)
					prev_centroid[tid] = (cx, cy)

					# Write to CSV. The bounding box (x1, y1, x2, y2) comes straight
					# from det['coords'] (unpacked above) and is appended after the
					# original 12 columns so the legacy column layout is preserved.
					csv_writer.writerow([
						frame_idx, tid, cx, cy,
						ps_class, f"{ps_conf:.3f}",
						pm_class, f"{pm_conf:.3f}",
						ss_class, f"{ss_conf:.3f}",
						sm_class, f"{sm_conf:.3f}",
						x1, y1, x2, y2,
						sp_class, f"{sp_conf:.3f}", ag_class, f"{ag_conf:.3f}",
						p_source
					])


				# ~ # print frame number
				draw_frame_number(frame, str(current_frame), render_style)

				if writer is not None:
					writer.write(frame)

				if print_tick > progress_update:
					elapsed = time.time() - start_time
					current_fps = current_frame / elapsed if elapsed > 0 else 0
					frames_total_for_progress = max_frame_limit if max_frame_limit is not None else total_frames
					pc_done = 100 * (frame_skip+1) * current_frame / frames_total_for_progress
					print(f"Progress: {pc_done:.2f}% | {current_fps:.1f} FPS", end='\r', flush=True)
					print_tick = 0
				current_frame += 1
				print_tick += 1

			frame_count += 1

			if frame_count > frame_skip:
				frame_count = 0

		cap.release()
		if writer is not None:
			writer.release()
		csv_file.close()
		print(f"Done processing {base} | {current_fps:.1f} FPS")


	# Collect all video files recursively from input_folder (any subfolder depth)
	_video_exts = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
	_all_input_videos = []
	for _dirpath, _dirs, _files in os.walk(input_folder):
		for _fname in sorted(_files):
			if _fname.lower().endswith(_video_exts):
				_all_input_videos.append(os.path.join(_dirpath, _fname))

	# Say why nothing will be processed. os.walk() on a missing directory yields
	# nothing at all, so an input_dir left over from another machine used to end
	# the run in silence, looking like a success.
	if not _all_input_videos:
		if not os.path.isdir(input_folder):
			print(f"WARNING: input directory does not exist: {input_folder}\n"
				  f"         Fix 'input_dir' in {os.path.basename(config_path)} "
				  f"(empty = <project>/input). No video will be processed.")
		else:
			print(f"No video found under {input_folder} — nothing to process.")

	# Processing order. Name order is chronological here (the clips are named from
	# their capture timestamp) and therefore also groups by site and by herd, so a
	# run that is interrupted -- or simply read before it finishes -- leaves a
	# partial set covering the first sites only. Everything downstream inherits
	# that: the frame miner scores whatever tracking CSVs exist, and the activity
	# budget aggregates them. Shuffling makes any prefix of the run a
	# representative sample of the corpus instead of its first chapter.
	_order = str(config['DEFAULT'].get('video_order', 'name')).strip().lower()
	if _order == 'random':
		# Seed 0 means "a different order every run", which is what you want when
		# re-running to widen coverage. A non-zero seed pins the order, so an
		# interrupted run can be resumed in the same sequence and a paper can
		# state exactly which videos a partial pass covered.
		_seed = int(float(config['DEFAULT'].get('video_order_seed', '0') or 0))
		_rng = random.Random(_seed) if _seed else random.Random()
		_rng.shuffle(_all_input_videos)
		print(f"Processing {len(_all_input_videos)} video(s) in random order"
			  + (f" (seed {_seed})" if _seed else " (unseeded)"))
	elif _all_input_videos:
		print(f"Processing {len(_all_input_videos)} video(s) in name order")

	for _n, vid in enumerate(_all_input_videos, 1):
			print(f"\n[{_n}/{len(_all_input_videos)}] {os.path.basename(vid)}")
			process_video(vid)

	# Auto-launch drone motion correction when enabled (disabled -> no behaviour change)
	try:
		if config['DEFAULT'].get('drone_correction_enabled', 'false').lower() == 'true':
			from behaveai_drone import run_drone_correction
			print("\nLaunching drone motion correction...")
			run_drone_correction(config_path)
	except Exception as e:
		import traceback
		print(f"Drone motion correction failed: {e}")
		traceback.print_exc()

	# Auto-launch offline tracklet stitching when enabled. Runs AFTER drone
	# correction so it stitches on the stabilised (x_corrected) coordinates,
	# which are comparable across the whole clip; disabled -> no behaviour change.
	try:
		if config['DEFAULT'].get('stitch_enabled', 'false').lower() == 'true':
			from BehaveAI_stitch_tracklets import run_stitch_project
			print("\nLaunching offline tracklet stitching...")
			run_stitch_project(config_path)
	except Exception as e:
		import traceback
		print(f"Tracklet stitching failed: {e}")
		traceback.print_exc()

	# Auto-launch metric geometry when enabled (disabled -> no behaviour change).
	# Runs after drone correction so it can consume the corrected CSV if present.
	try:
		if config['DEFAULT'].get('metric_enabled', 'false').lower() == 'true':
			from behaveai_drone import run_metric_geometry
			print("\nLaunching metric geometry (flight-log -> metres)...")
			run_metric_geometry(config_path)
	except Exception as e:
		import traceback
		print(f"Metric geometry failed: {e}")
		traceback.print_exc()

	# Auto-launch the interaction graph (deterministic dyadic/group features ->
	# edges/nodes). REQUIRES metric geometry: the graph works in metres and skips
	# any video without *_tracking_metric.csv. Must run BEFORE complex
	# classification, which back-fills interaction_type into the edges file.
	try:
		if config['DEFAULT'].get('interaction_graph_enabled', 'true').lower() == 'true':
			if config['DEFAULT'].get('metric_enabled', 'false').lower() != 'true':
				print("\nInteraction graph: SKIPPED — it measures distances and speeds in "
					  "metres, which needs metric_enabled = true (plus a .flightlog.csv "
					  "beside each video). Enable it in the settings, or set "
					  "interaction_graph_enabled = false to stop seeing this notice.")
			else:
				from BehaveAI_complex_features import run_complex_features
				print("\nLaunching interaction graph...")
				run_complex_features(config_path)
	except Exception as e:
		import traceback
		print(f"Interaction graph failed: {e}")
		traceback.print_exc()

	# Auto-launch the complex-behaviour stage: (re)train on new annotations when
	# needed -- exactly like the YOLO models' maybe_retrain (automatic, keyed off
	# model_complex/train_count.txt) -- then classify every video, writing
	# *_complex_predictions.csv and populating interaction_type in the edges file.
	# End-to-end NO-OP for projects with no complex annotations and no trained model.
	try:
		if config['DEFAULT'].get('complex_classify_enabled', 'true').lower() == 'true':
			from BehaveAI_complex_model import run_complex_stage
			print("\nLaunching complex-behaviour stage (train-if-new + classify)...")
			run_complex_stage(config_path)
	except Exception as e:
		import traceback
		print(f"Complex-behaviour stage failed: {e}")
		traceback.print_exc()

	# Auto-launch the activity budget LAST, so it runs on the most-processed CSV
	# per video (stitched identities + any metric columns) AND after the interaction
	# graph + complex classification, whose outputs it now folds in per individual.
	try:
		from BehaveAI_activity_budget import run_activity_budget
		print("\nLaunching activity budget analysis...")
		run_activity_budget(config_path)
	except Exception as e:
		import traceback
		print(f"Activity budget analysis failed: {e}")
		traceback.print_exc()
