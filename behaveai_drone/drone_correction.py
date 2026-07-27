#!/usr/bin/env python3
"""
BehaveAI Drone Motion Correction

Post-processing step for the HERDWISE multi-individual pipeline. The drone is
normally held still while filming, but it may pan/zoom to follow the herd, which
adds apparent (background) motion to every tracked centroid and corrupts the
velocity of slow-moving horses.

This module removes that drone-induced motion. For each consecutive pair of
processed frames it:
  1. masks out the tracked horses (their bounding boxes, dilated) so optical flow
     is computed ONLY on the static background;
  2. estimates the global background motion with sparse optical flow
     (goodFeaturesToTrack + calcOpticalFlowPyrLK) and a RANSAC-fitted global
     transform (affine partial 2D by default, or homography);
  3. chains those transforms and maps every centroid into one stabilised
     reference frame (relative to the first processed frame);
  4. recomputes velocities from the corrected, smoothed positions using the real
     frame gap between rows (frame_skip aware).

Conceptually, "correcting the position" is identical to "subtracting the drone's
apparent motion at that point from the horse's apparent velocity" — it is the
same model, applied to positions.

Everything stays in image space: there is NO telemetry and NO metric (m/s, GPS)
conversion. The motion false-colour stream is NOT used here; kinematics come from
finite differences on the tracked centroids only.

Input:  original video + its *_tracking.csv (ideally with x1,y1,x2,y2 columns).
Output: <videoname>_tracking_corrected.csv = the original CSV with five appended
        columns: x_corrected, y_corrected, vx_corrected, vy_corrected,
        correction_quality (one of 'ok', 'uncertain', 'none').

Usage (batch over a project's output folder):
  python BehaveAI_drone_correction.py <project_dir | BehaveAI_settings.ini>
"""

import os
import sys
import csv
import glob
import argparse
import configparser
from collections import defaultdict

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Tunable detector / tracker constants (not exposed in the INI on purpose —
# the INI keys cover the scientifically meaningful knobs).
# ---------------------------------------------------------------------------

_MAX_CORNERS   = 500
_QUALITY_LEVEL = 0.01
_MIN_DISTANCE  = 8
_BLOCK_SIZE    = 7
_LK_WIN        = (21, 21)
_LK_MAXLEVEL   = 3
_LK_CRITERIA   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
_RANSAC_REPROJ = 3.0

# Fraction of feature-poor steps above which we consider the background
# persistently unusable and (when enabled) switch to smoothing-only fallback.
_FALLBACK_POOR_FRACTION = 0.5

# Relative change in the cumulative uniform-scale factor that marks a new
# stable segment (altitude / zoom change). Exposed via the run log so TASK 3/4
# can recompute body_len_ref per segment when body_len_ref_scope = segment.
_SCALE_SEGMENT_THRESH = 0.10


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_drone_correction_config(config_path):
	"""Read the drone-correction parameters from a BehaveAI INI (DEFAULT section).

	Every key is read with a fallback so older INIs without these keys still work.
	Returns a plain dict.
	"""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	return {
		'enabled':            str(d.get('drone_correction_enabled', 'false')).lower() == 'true',
		'model':              d.get('drone_correction_model', 'affine'),
		'box_dilation':       float(d.get('drone_correction_box_dilation', '0.20')),
		'min_features':       int(float(d.get('drone_correction_min_features', '30'))),
		'uncertain_std':      float(d.get('drone_correction_uncertain_std', '8.0')),
		'smoothing':          d.get('drone_correction_smoothing', 'savgol'),
		'smoothing_window':   int(float(d.get('drone_correction_smoothing_window', '7'))),
		'fallback_smoothing': str(d.get('drone_correction_fallback_smoothing', 'true')).lower() == 'true',
	}


# ---------------------------------------------------------------------------
# Tracking CSV I/O
# ---------------------------------------------------------------------------

def _read_tracking_csv(csv_path):
	"""Read a tracking CSV preserving all original columns/values.

	Returns (fieldnames, rows, has_bbox) where rows is a list of dicts and
	has_bbox is True when the x1,y1,x2,y2 columns are present.
	"""
	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fieldnames = list(reader.fieldnames or [])
		rows = [dict(r) for r in reader]
	has_bbox = all(c in fieldnames for c in ('x1', 'y1', 'x2', 'y2'))
	return fieldnames, rows, has_bbox


def _boxes_by_frame(rows, has_bbox):
	"""Build {frame:int -> [(x1,y1,x2,y2), ...]} from rows that carry a valid box."""
	boxes = defaultdict(list)
	if not has_bbox:
		return boxes
	for r in rows:
		try:
			frame = int(r['frame'])
			x1, y1, x2, y2 = int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2'])
		except (ValueError, KeyError, TypeError):
			continue
		if x2 > x1 and y2 > y1:
			boxes[frame].append((x1, y1, x2, y2))
	return boxes


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _build_background_mask(height, width, boxes, dilation):
	"""Return a uint8 mask (255 = background usable for flow, 0 = masked horse).

	Each box is dilated by `dilation` (a fraction of its own width/height) before
	being zeroed out, so feathered horse edges are excluded too.
	"""
	mask = np.full((height, width), 255, dtype=np.uint8)
	for (x1, y1, x2, y2) in boxes:
		bw = x2 - x1
		bh = y2 - y1
		dx = int(round(bw * dilation))
		dy = int(round(bh * dilation))
		mx1 = max(0, x1 - dx); my1 = max(0, y1 - dy)
		mx2 = min(width, x2 + dx); my2 = min(height, y2 + dy)
		if mx2 > mx1 and my2 > my1:
			mask[my1:my2, mx1:mx2] = 0
	return mask


def _to_3x3(M, model):
	"""Promote an estimated transform to a 3x3 homogeneous matrix."""
	if model == 'homography':
		return np.asarray(M, dtype=np.float64)
	# affine partial 2D is 2x3
	out = np.eye(3, dtype=np.float64)
	out[:2, :] = np.asarray(M, dtype=np.float64)
	return out


def _apply_transform_points(M3x3, pts):
	"""Apply a 3x3 homogeneous transform to an Nx2 array of points."""
	n = pts.shape[0]
	homo = np.hstack([pts, np.ones((n, 1))])
	proj = (M3x3 @ homo.T).T
	w = proj[:, 2:3]
	w[np.abs(w) < 1e-12] = 1e-12
	return proj[:, :2] / w


def _cumulative_scale(C):
	"""Uniform-scale factor of a (cumulative) 3x3 transform."""
	return float(np.hypot(C[0, 0], C[1, 0]))


def _dedup_points(pts):
	"""Remove near-duplicate points (rounded to the integer grid)."""
	if len(pts) == 0:
		return pts
	keys = np.round(pts).astype(np.int32)
	_, idx = np.unique(keys, axis=0, return_index=True)
	return pts[np.sort(idx)]


# ---------------------------------------------------------------------------
# Per-step transform estimation
# ---------------------------------------------------------------------------

def _estimate_step_transform(prev_gray, cur_gray, bg_mask, model,
							 min_features, uncertain_std, carried_pts):
	"""Estimate the background transform mapping prev-frame coords -> cur-frame coords.

	Detects corners on the masked background of prev_gray (merged with carried
	landmark points from the previous step so persistent features are preferred),
	tracks them with pyramidal Lucas-Kanade, then fits a global model with RANSAC.

	Returns (M3x3 or None, info) where info holds n_features, n_inliers,
	residual_std and the boolean flags few_features / high_residual. The carried
	landmark points for the next step (inlier positions in cur-frame coords) are
	returned in info['landmarks'].
	"""
	info = {'n_features': 0, 'n_inliers': 0, 'residual_std': float('inf'),
			'few_features': True, 'high_residual': False, 'landmarks': None}

	fresh = cv2.goodFeaturesToTrack(
		prev_gray, maxCorners=_MAX_CORNERS, qualityLevel=_QUALITY_LEVEL,
		minDistance=_MIN_DISTANCE, mask=bg_mask, blockSize=_BLOCK_SIZE)

	pts_list = []
	if fresh is not None:
		pts_list.append(fresh.reshape(-1, 2).astype(np.float32))
	if carried_pts is not None and len(carried_pts):
		pts_list.append(np.asarray(carried_pts, dtype=np.float32).reshape(-1, 2))
	if not pts_list:
		return None, info

	pts = _dedup_points(np.vstack(pts_list).astype(np.float32))
	if len(pts) < 3:
		return None, info

	p0 = pts.reshape(-1, 1, 2)
	p1, status, _ = cv2.calcOpticalFlowPyrLK(
		prev_gray, cur_gray, p0, None,
		winSize=_LK_WIN, maxLevel=_LK_MAXLEVEL, criteria=_LK_CRITERIA)
	if p1 is None or status is None:
		return None, info

	status = status.reshape(-1)
	good_prev = p0.reshape(-1, 2)[status == 1]
	good_cur = p1.reshape(-1, 2)[status == 1]
	info['n_features'] = int(len(good_prev))
	if len(good_prev) < 3:
		return None, info

	if model == 'homography':
		M, inliers = cv2.findHomography(good_prev, good_cur, cv2.RANSAC, _RANSAC_REPROJ)
	else:
		M, inliers = cv2.estimateAffinePartial2D(
			good_prev, good_cur, method=cv2.RANSAC, ransacReprojThreshold=_RANSAC_REPROJ)
	if M is None:
		return None, info

	M3 = _to_3x3(M, model)
	inlier_mask = (inliers.ravel().astype(bool)
				   if inliers is not None else np.ones(len(good_prev), dtype=bool))
	n_inliers = int(inlier_mask.sum())
	info['n_inliers'] = n_inliers

	# Residual flow std on the inliers: how well the global model explains them.
	if n_inliers >= 1:
		src = good_prev[inlier_mask]
		dst = good_cur[inlier_mask]
		proj = _apply_transform_points(M3, src)
		resid = np.linalg.norm(proj - dst, axis=1)
		info['residual_std'] = float(resid.std()) if len(resid) > 1 else 0.0
		# Persist the inlier positions (in cur-frame coords) as landmarks.
		info['landmarks'] = good_cur[inlier_mask].copy()

	info['few_features'] = n_inliers < min_features
	info['high_residual'] = info['residual_std'] > uncertain_std
	return M3, info


# ---------------------------------------------------------------------------
# Smoothing & kinematics
# ---------------------------------------------------------------------------

def _smooth_series(values, method, window):
	"""Smooth a 1-D series. method = savgol | moving_average | none.

	Gracefully reduces the window to fit short series and falls back to the raw
	values when smoothing is not possible (or scipy is unavailable for savgol).
	"""
	arr = np.asarray(values, dtype=np.float64)
	n = len(arr)
	if method == 'none' or n < 3 or window < 3:
		return arr

	win = min(window, n)
	if win % 2 == 0:
		win -= 1
	if win < 3:
		return arr

	if method == 'savgol':
		try:
			from scipy.signal import savgol_filter
			poly = min(2, win - 1)
			return savgol_filter(arr, win, poly)
		except Exception:
			method = 'moving_average'  # fall through to the simple smoother

	if method == 'moving_average':
		kernel = np.ones(win) / win
		padded = np.pad(arr, win // 2, mode='edge')
		return np.convolve(padded, kernel, mode='valid')[:n]

	return arr


def _velocity(frames, xs, ys):
	"""Finite-difference velocity (per frame) using the real frame spacing.

	np.gradient with the frame coordinate array handles non-uniform spacing, so
	frame_skip gaps are respected automatically.
	"""
	frames = np.asarray(frames, dtype=np.float64)
	xs = np.asarray(xs, dtype=np.float64)
	ys = np.asarray(ys, dtype=np.float64)
	if len(frames) < 2:
		return np.zeros(len(frames)), np.zeros(len(frames))
	# np.gradient needs strictly increasing coordinates; guard against duplicates.
	if np.any(np.diff(frames) <= 0):
		vx = np.zeros(len(frames))
		vy = np.zeros(len(frames))
		for i in range(len(frames)):
			a = max(0, i - 1)
			b = min(len(frames) - 1, i + 1)
			dt = frames[b] - frames[a]
			if dt > 0:
				vx[i] = (xs[b] - xs[a]) / dt
				vy[i] = (ys[b] - ys[a]) / dt
		return vx, vy
	return np.gradient(xs, frames), np.gradient(ys, frames)


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def correct_video_tracking(
	video_path, tracking_csv_path, output_csv_path,
	box_dilation=0.20, transform_model="affine",
	min_background_features=30, uncertain_std_thresh=8.0,
	smoothing="savgol", smoothing_window=7,
	fallback_smoothing=True,
):
	"""Estimate per-frame drone motion from optical flow on the animal-masked
	background, correct tracked centroids into a stabilised reference frame,
	recompute velocities, and write <videoname>_tracking_corrected.csv. Operates on
	centroids only; the motion false-colour stream is NOT used here. No
	telemetry/metric conversion.
	"""
	fieldnames, rows, has_bbox = _read_tracking_csv(tracking_csv_path)
	if not rows:
		print(f"  {os.path.basename(tracking_csv_path)}: no rows, skipping.")
		return

	if not has_bbox:
		# Graceful degradation: without boxes we cannot mask the horses, so flow is
		# computed on the whole frame and the result is never better than 'uncertain'.
		print(f"  WARNING: {os.path.basename(tracking_csv_path)} has no x1,y1,x2,y2 "
			  f"columns — masking disabled, results flagged 'uncertain'.")

	boxes_by_frame = _boxes_by_frame(rows, has_bbox)

	# Ordered list of processed frames actually present in the CSV.
	processed_frames = sorted({int(r['frame']) for r in rows})
	processed_set = set(processed_frames)
	if not processed_frames:
		return

	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print(f"  ERROR: could not open video {video_path}; copying raw positions.")
		_write_fallback_copy(fieldnames, rows, output_csv_path,
							 smoothing, smoothing_window, quality='none')
		return

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Per-frame cumulative transform (frame0 -> frame f) and quality flag.
	frame_cumulative = {}
	frame_quality = {}
	frame_scale = {}
	frame_resid = {}                          # per-step residual flow std (px)
	frame_ninl = {}                           # per-step RANSAC inlier count

	C = np.eye(3, dtype=np.float64)          # cumulative transform
	prev_gray = None
	prev_boxes = None
	carried_pts = None
	n_steps = 0
	n_poor = 0

	vid_idx = -1
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		vid_idx += 1
		csv_frame = vid_idx + 1               # classify_track writes frame_idx = video_idx + 1
		if csv_frame not in processed_set:
			continue

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cur_boxes = boxes_by_frame.get(csv_frame, [])

		if prev_gray is None:
			# First processed frame is the reference frame.
			frame_cumulative[csv_frame] = C.copy()
			frame_quality[csv_frame] = 'ok'
			frame_scale[csv_frame] = 1.0
			frame_resid[csv_frame] = 0.0
			frame_ninl[csv_frame] = 0
		else:
			bg_mask = (_build_background_mask(height, width, prev_boxes, box_dilation)
					   if has_bbox else None)
			M3, info = _estimate_step_transform(
				prev_gray, gray, bg_mask, transform_model,
				min_background_features, uncertain_std_thresh, carried_pts)
			n_steps += 1
			if info['few_features']:
				n_poor += 1
			frame_resid[csv_frame] = info['residual_std']
			frame_ninl[csv_frame] = info['n_inliers']

			if M3 is None:
				# No usable transform for this step: carry the cumulative forward.
				frame_quality[csv_frame] = 'none'
				carried_pts = None
			else:
				C = M3 @ C
				if info['few_features'] or info['high_residual']:
					frame_quality[csv_frame] = 'uncertain'
				elif not has_bbox:
					frame_quality[csv_frame] = 'uncertain'
				else:
					frame_quality[csv_frame] = 'ok'
				carried_pts = info['landmarks']

			frame_cumulative[csv_frame] = C.copy()
			frame_scale[csv_frame] = _cumulative_scale(C)

		prev_gray = gray
		prev_boxes = cur_boxes

	cap.release()

	# Decide whether to fall back to smoothing-only correction.
	fallback_mode = False
	if n_steps > 0 and (n_poor / n_steps) >= _FALLBACK_POOR_FRACTION:
		if fallback_smoothing:
			fallback_mode = True
			print(f"  Background persistently feature-poor "
				  f"({n_poor}/{n_steps} steps) — using smoothing-only fallback "
				  f"(quality 'uncertain').")
		else:
			print(f"  Background persistently feature-poor "
				  f"({n_poor}/{n_steps} steps) and fallback disabled — "
				  f"positions flagged 'none'/'uncertain'.")

	# Report cumulative-scale drift so TASK 3/4 can segment body_len_ref.
	_report_scale_segments(frame_scale, processed_frames)

	# Diagnostic sidecar for evaluation: per-frame continuous residual flow std,
	# inlier count and cumulative scale (the categorical correction_quality alone
	# hides the underlying magnitudes). Additive — does not affect the corrected CSV.
	_write_correction_diag(output_csv_path, processed_frames,
						   frame_resid, frame_ninl, frame_scale, frame_quality)

	# ---- Build corrected positions per row, then per-id smoothing + velocity ----
	# Stage 1: corrected (pre-smoothing) position + quality for every row.
	by_id = defaultdict(list)
	for r in rows:
		f = int(r['frame'])
		x = float(r['x']); y = float(r['y'])
		if fallback_mode:
			cx, cy = x, y
			q = 'uncertain'
		else:
			q = frame_quality.get(f, 'none')
			if q == 'none':
				cx, cy = x, y                  # copy raw
			else:
				C_f = frame_cumulative.get(f)
				if C_f is None:
					cx, cy = x, y
					q = 'none'
				else:
					try:
						p = _apply_transform_points(np.linalg.inv(C_f),
													np.array([[x, y]], dtype=np.float64))[0]
						cx, cy = float(p[0]), float(p[1])
					except np.linalg.LinAlgError:
						cx, cy = x, y
						q = 'none'
		by_id[r['id']].append((f, cx, cy, q, r))

	# Stage 2: smooth each id's corrected track and differentiate.
	for tid, recs in by_id.items():
		recs.sort(key=lambda t: t[0])
		frames = [t[0] for t in recs]
		xs = _smooth_series([t[1] for t in recs], smoothing, smoothing_window)
		ys = _smooth_series([t[2] for t in recs], smoothing, smoothing_window)
		vx, vy = _velocity(frames, xs, ys)
		for i, (f, _cx, _cy, q, r) in enumerate(recs):
			r['x_corrected'] = f"{xs[i]:.3f}"
			r['y_corrected'] = f"{ys[i]:.3f}"
			r['vx_corrected'] = f"{vx[i]:.4f}"
			r['vy_corrected'] = f"{vy[i]:.4f}"
			r['correction_quality'] = q

	_write_corrected_csv(fieldnames, rows, output_csv_path)

	n_ok = sum(1 for r in rows if r.get('correction_quality') == 'ok')
	n_unc = sum(1 for r in rows if r.get('correction_quality') == 'uncertain')
	n_none = sum(1 for r in rows if r.get('correction_quality') == 'none')
	print(f"  Wrote {os.path.basename(output_csv_path)}: "
		  f"{len(rows)} rows (ok={n_ok}, uncertain={n_unc}, none={n_none}).")


def _write_correction_diag(output_csv_path, processed_frames,
						   frame_resid, frame_ninl, frame_scale, frame_quality):
	"""Write <stem>_correction_diag.csv next to the corrected CSV: per processed
	frame, the continuous residual flow std (px), RANSAC inlier count, cumulative
	uniform scale and the quality flag. Read by BehaveAI_evaluate_geometry.py.
	Non-finite residuals (no inliers) are written blank."""
	if output_csv_path.endswith('_tracking_corrected.csv'):
		diag_path = output_csv_path[:-len('_tracking_corrected.csv')] + '_correction_diag.csv'
	else:
		diag_path = output_csv_path + '.diag.csv'
	try:
		with open(diag_path, 'w', newline='', encoding='utf-8') as f:
			w = csv.writer(f)
			w.writerow(['frame', 'residual_std', 'n_inliers', 'cumulative_scale', 'quality'])
			for fr in processed_frames:
				resid = frame_resid.get(fr)
				resid_str = f"{resid:.4f}" if (resid is not None and np.isfinite(resid)) else ''
				scale = frame_scale.get(fr)
				scale_str = f"{scale:.5f}" if scale is not None else ''
				w.writerow([fr, resid_str, frame_ninl.get(fr, ''), scale_str,
							frame_quality.get(fr, '')])
	except OSError as e:
		print(f"  (could not write correction diagnostics: {e})")


def _report_scale_segments(frame_scale, processed_frames):
	"""Print stable-scale segment boundaries based on cumulative-scale drift.

	A new segment starts whenever the cumulative uniform scale has drifted more
	than _SCALE_SEGMENT_THRESH (relative) from the scale at the segment start.
	This exposes altitude/zoom changes for TASK 3/4 (body_len_ref_scope=segment).
	"""
	if not frame_scale:
		return
	segments = []
	seg_start = processed_frames[0]
	ref_scale = frame_scale.get(seg_start, 1.0) or 1.0
	smin = smax = ref_scale
	for f in processed_frames:
		s = frame_scale.get(f, ref_scale)
		smin = min(smin, s); smax = max(smax, s)
		if ref_scale > 0 and abs(s - ref_scale) / ref_scale > _SCALE_SEGMENT_THRESH:
			segments.append((seg_start, f))
			seg_start = f
			ref_scale = s
	segments.append((seg_start, processed_frames[-1]))
	drift = (smax / smin) if smin > 0 else 1.0
	if len(segments) > 1 or drift > (1.0 + _SCALE_SEGMENT_THRESH):
		print(f"  Scale drift: min={smin:.3f} max={smax:.3f} (ratio {drift:.3f}); "
			  f"{len(segments)} stable segment(s): "
			  f"{', '.join(f'{a}-{b}' for a, b in segments)}")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

_NEW_COLS = ['x_corrected', 'y_corrected', 'vx_corrected', 'vy_corrected', 'correction_quality']


def _write_corrected_csv(fieldnames, rows, output_csv_path):
	"""Write the corrected CSV = original columns + the five appended columns."""
	out_fields = list(fieldnames) + [c for c in _NEW_COLS if c not in fieldnames]
	os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
	with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
		writer.writeheader()
		writer.writerows(rows)


def _write_fallback_copy(fieldnames, rows, output_csv_path,
						 smoothing, smoothing_window, quality='none'):
	"""When the video cannot be opened: copy raw positions, smooth, mark quality."""
	by_id = defaultdict(list)
	for r in rows:
		by_id[r['id']].append(r)
	for tid, recs in by_id.items():
		recs.sort(key=lambda r: int(r['frame']))
		frames = [int(r['frame']) for r in recs]
		xs = _smooth_series([float(r['x']) for r in recs], smoothing, smoothing_window)
		ys = _smooth_series([float(r['y']) for r in recs], smoothing, smoothing_window)
		vx, vy = _velocity(frames, xs, ys)
		for i, r in enumerate(recs):
			r['x_corrected'] = f"{xs[i]:.3f}"
			r['y_corrected'] = f"{ys[i]:.3f}"
			r['vx_corrected'] = f"{vx[i]:.4f}"
			r['vy_corrected'] = f"{vy[i]:.4f}"
			r['correction_quality'] = quality
	_write_corrected_csv(fieldnames, rows, output_csv_path)


# ---------------------------------------------------------------------------
# Batch / project entry points
# ---------------------------------------------------------------------------

def _build_video_index(root):
	"""Return {filename -> absolute path} for all videos under root (recursive)."""
	index = {}
	if not root or not os.path.isdir(root):
		return index
	video_exts = ('.mp4', '.avi', '.mov', '.mkv')
	for dirpath, _, files in os.walk(root):
		for fname in files:
			if fname.lower().endswith(video_exts):
				index.setdefault(fname, os.path.join(dirpath, fname))
	return index


def _find_video_for_csv(csv_path, video_index):
	"""Locate the source video for a *_tracking.csv via its stem."""
	stem = os.path.basename(csv_path).replace('_tracking.csv', '')
	for ext in ('.MP4', '.mp4', '.avi', '.mov', '.mkv', '.AVI', '.MOV', '.MKV'):
		cand = stem + ext
		if cand in video_index:
			return video_index[cand]
	return None


def run_drone_correction(config_path):
	"""Batch-correct every *_tracking.csv in the project's output folder.

	Reads parameters from the INI, resolves each CSV's source video from the
	input/clips folders, and writes <video>_tracking_corrected.csv next to it.
	"""
	config_path = os.path.abspath(config_path)
	project_dir = os.path.dirname(config_path)
	params = load_drone_correction_config(config_path)

	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']

	def _resolve(key, default):
		v = d.get(key, default)
		return v if os.path.isabs(v) else os.path.join(project_dir, v)

	output_dir = _resolve('output_dir', 'output')
	input_dir = _resolve('input_dir', 'input')
	clips_dir = _resolve('clips_dir', 'clips')

	video_index = _build_video_index(input_dir)
	for fname, path in _build_video_index(clips_dir).items():
		video_index.setdefault(fname, path)

	csv_files = sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv')))
	if not csv_files:
		print(f"Drone correction: no tracking CSVs found in {output_dir}")
		return

	print(f"Drone correction: processing {len(csv_files)} tracking CSV(s) "
		  f"(model={params['model']}, smoothing={params['smoothing']})...")

	for csv_path in csv_files:
		video_path = _find_video_for_csv(csv_path, video_index)
		out_path = csv_path.replace('_tracking.csv', '_tracking_corrected.csv')
		if video_path is None:
			print(f"  {os.path.basename(csv_path)}: source video not found — skipped.")
			continue
		try:
			correct_video_tracking(
				video_path, csv_path, out_path,
				box_dilation=params['box_dilation'],
				transform_model=params['model'],
				min_background_features=params['min_features'],
				uncertain_std_thresh=params['uncertain_std'],
				smoothing=params['smoothing'],
				smoothing_window=params['smoothing_window'],
				fallback_smoothing=params['fallback_smoothing'],
			)
		except Exception as e:
			import traceback
			print(f"  ERROR correcting {os.path.basename(csv_path)}: {e}")
			traceback.print_exc()

	print("Drone correction complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
	parser = argparse.ArgumentParser(
		description="Remove drone-induced motion from BehaveAI tracking CSVs.")
	parser.add_argument('target',
						help="Project directory or BehaveAI_settings.ini (batch mode), "
							 "or a video file when --csv is given (single-file mode).")
	parser.add_argument('--csv', default=None,
						help="Single-file mode: the *_tracking.csv for the given video.")
	parser.add_argument('--out', default=None,
						help="Single-file mode: output CSV path "
							 "(default: <video>_tracking_corrected.csv next to the input CSV).")
	args = parser.parse_args()

	# Single-file mode
	if args.csv is not None:
		out = args.out or args.csv.replace('_tracking.csv', '_tracking_corrected.csv')
		correct_video_tracking(args.target, args.csv, out)
		return

	# Batch / project mode
	target = os.path.abspath(args.target)
	ini = os.path.join(target, 'BehaveAI_settings.ini') if os.path.isdir(target) else target
	if not os.path.exists(ini):
		print(f"Settings file not found: {ini}")
		sys.exit(1)
	run_drone_correction(ini)


if __name__ == '__main__':
	_main()
