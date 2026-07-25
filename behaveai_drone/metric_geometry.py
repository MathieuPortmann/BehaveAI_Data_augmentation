#!/usr/bin/env python3
"""
BehaveAI Metric Geometry (pixels -> real-world metres, from flight-log telemetry)

Post-processing step for the HERDWISE multi-individual pipeline, mirroring the
TASK 1 drone-correction pattern. It turns each tracked animal's image position
into ground-plane coordinates in METRES, so downstream inter-individual
distances and speeds are real-world (m, m/s) rather than pixels.

The metric chain is  metres/pixel = f(h, theta, f_px, image row)  with:
  - h      = camera height above ground, read from the flight log (rel_alt_m);
  - theta  = camera pitch, read from the flight log (gb_pitch) — the tilt DJI
             SRT files lack and that the horizon fit otherwise has to estimate;
  - f_px   = pixel focal length, from the camera spec by default
             (focal_len_mm * frame_width / sensor_width_mm), or a checkerboard
             calibration if one is provided per drone;
  - the ground point of each detection is its box bottom-centre (the feet).

Assumptions (flagged, not assumed silently): flat ground and zero roll. gb_roll
is checked per frame; frames near the horizon (where a pixel maps to many
metres) are flagged. Videos without a flight log get metric_quality='none' and
keep only their pixel coordinates — the activity budget itself is unaffected.

Input:  <video>_tracking(_corrected).csv  +  <video>.flightlog.csv (same stem).
Output: <video>_tracking_metric.csv = the input CSV with three appended columns
        X_m, Y_m (ground coords, metres; X forward, Y lateral) and metric_quality
        (one of 'ok', 'uncertain', 'none').

Usage (batch over a project's output folder):
  python BehaveAI_metric_geometry.py <project_dir | BehaveAI_settings.ini>
"""

import os
import sys
import csv
import glob
import argparse
import configparser

import numpy as np

try:  # as a package member
	from .flightlog import load_flightlog, sample_flightlog, flightlog_summary
	from .horizon_geometry import pixel_focal_length, horizon_row, ground_point_from_pixel
except ImportError:  # run directly from inside the package dir
	from flightlog import load_flightlog, sample_flightlog, flightlog_summary
	from horizon_geometry import pixel_focal_length, horizon_row, ground_point_from_pixel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_metric_config(config_path):
	"""Read metric-geometry parameters from a BehaveAI INI (DEFAULT section).

	Every key has a fallback so older INIs still work. Per-drone focal overrides
	(from a checkerboard) are read as metric_fpx_<DroneToken> when present.
	"""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	fpx_overrides = {}
	for key in d:
		if key.startswith('metric_fpx_'):
			try:
				fpx_overrides[key[len('metric_fpx_'):]] = float(d.get(key))
			except (ValueError, TypeError):
				pass
	return {
		'enabled':          str(d.get('metric_enabled', 'false')).lower() == 'true',
		'focal_len_mm':     float(d.get('metric_focal_len_mm', '24.0')),
		'sensor_width_mm':  float(d.get('metric_sensor_width_mm', '36.0')),
		'roll_max_deg':     float(d.get('metric_roll_max_deg', '3.0')),
		'horizon_margin_px': float(d.get('metric_horizon_margin_px', '50')),
		'fpx_overrides':    fpx_overrides,
	}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def drone_token(name):
	"""Return the drone token that prefixes a BehaveAI/HERDWISE filename
	(e.g. 'Mini4Pro', 'Mini3Pro', 'Mini3'). Falls back to the first '_' field."""
	base = os.path.basename(name)
	first = base.split('_', 1)[0]
	# The 'DJI_Mini_3_...' legacy naming needs the first three underscore fields.
	if first.upper() == 'DJI' and base.lower().startswith('dji_mini'):
		parts = base.split('_')
		return '_'.join(parts[:3])          # e.g. 'DJI_Mini_3'
	return first


def resolve_fpx(stem, frame_width, focal_len_mm, sensor_width_mm, fpx_overrides):
	"""Pick f_px for a video: a per-drone checkerboard override if configured,
	else the spec-derived value. Returns (f_px, source_str)."""
	tok = drone_token(stem)
	if tok in fpx_overrides:
		return fpx_overrides[tok], f'checkerboard:{tok}'
	return pixel_focal_length(focal_len_mm, frame_width, sensor_width_mm), 'spec'


def _read_tracking_csv(csv_path):
	"""Read a tracking CSV, preserving all columns. Returns (fieldnames, rows,
	has_bbox)."""
	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fieldnames = list(reader.fieldnames or [])
		rows = [dict(r) for r in reader]
	has_bbox = all(c in fieldnames for c in ('x1', 'y1', 'x2', 'y2'))
	return fieldnames, rows, has_bbox


def _video_dims_fps(video_path):
	"""Read (width, height, fps) from a video header without decoding frames."""
	import cv2
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		return None
	w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
	cap.release()
	return w, h, float(fps)


def _ground_contact_pixel(row, has_bbox):
	"""Feet of the animal = bottom-centre of its box; falls back to the centroid
	(worse, so the caller downgrades quality). Returns (u, v, from_bbox)."""
	if has_bbox:
		try:
			x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
			if x2 > x1 and y2 > y1:
				return (x1 + x2) / 2.0, float(y2), True
		except (ValueError, KeyError, TypeError):
			pass
	try:
		return float(row['x']), float(row['y']), False
	except (ValueError, KeyError, TypeError):
		return None, None, False


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

_NEW_COLS = ['X_m', 'Y_m', 'metric_quality']


def metric_video_tracking(video_path, tracking_csv_path, flightlog_path, output_csv_path,
						  focal_len_mm=24.0, sensor_width_mm=36.0,
						  roll_max_deg=3.0, horizon_margin_px=50.0, fpx_overrides=None):
	"""Project every tracked detection to ground-plane metres using the flight
	log, and write <video>_tracking_metric.csv. Rows without usable telemetry
	keep empty X_m/Y_m and metric_quality='none'."""
	fpx_overrides = fpx_overrides or {}
	fieldnames, rows, has_bbox = _read_tracking_csv(tracking_csv_path)
	if not rows:
		print(f"  {os.path.basename(tracking_csv_path)}: no rows, skipping.")
		return

	dims = _video_dims_fps(video_path) if video_path else None
	if dims is None:
		print(f"  {os.path.basename(tracking_csv_path)}: could not read video dims — "
			  f"metric skipped (quality 'none').")
		_write_all_none(fieldnames, rows, output_csv_path)
		return
	width, height, fps = dims
	cx, cy = width / 2.0, height / 2.0

	stem = os.path.basename(tracking_csv_path)
	stem = stem.replace('_tracking_corrected.csv', '').replace('_tracking.csv', '')
	f_px, fpx_source = resolve_fpx(stem, width, focal_len_mm, sensor_width_mm, fpx_overrides)

	flightlog = load_flightlog(flightlog_path) if flightlog_path and os.path.exists(flightlog_path) else {}
	if not flightlog:
		print(f"  {stem}: no flight log — metric quality 'none' (pixels kept).")
		_write_all_none(fieldnames, rows, output_csv_path)
		return

	summ = flightlog_summary(flightlog)
	roll_med = (summ.get('gb_roll') or {}).get('median')
	print(f"  {stem}: f_px={f_px:.0f} ({fpx_source}), "
		  f"pitch median={(summ.get('gb_pitch') or {}).get('median')}, "
		  f"roll median={roll_med}, {summ.get('n_rows')} log rows.")

	n_ok = n_unc = n_none = 0
	for r in rows:
		try:
			frame = int(r['frame'])
		except (ValueError, KeyError, TypeError):
			r['X_m'] = ''; r['Y_m'] = ''; r['metric_quality'] = 'none'; n_none += 1
			continue
		t_s = (frame - 1) / fps                       # csv frame N -> video frame N-1
		h_m, gb_pitch, gb_roll = sample_flightlog(flightlog, t_s)

		u, v, from_bbox = _ground_contact_pixel(r, has_bbox)
		if (u is None or np.isnan(h_m) or np.isnan(gb_pitch)):
			r['X_m'] = ''; r['Y_m'] = ''; r['metric_quality'] = 'none'; n_none += 1
			continue

		pitch_deg = -gb_pitch                          # gb_pitch<0 looking down -> theta>0
		pt = ground_point_from_pixel(u, v, h_m, pitch_deg, f_px, cx, cy)
		if pt is None:                                 # at/above horizon
			r['X_m'] = ''; r['Y_m'] = ''; r['metric_quality'] = 'none'; n_none += 1
			continue

		X_m, Y_m = pt
		# Quality: downgrade for bad roll, missing bbox, or near-horizon margin.
		v_horizon = horizon_row(pitch_deg, f_px, cy)
		margin = v - v_horizon
		quality = 'ok'
		if (not np.isnan(gb_roll) and abs(gb_roll) > roll_max_deg) \
				or not from_bbox or margin < horizon_margin_px:
			quality = 'uncertain'
		r['X_m'] = f"{X_m:.3f}"; r['Y_m'] = f"{Y_m:.3f}"; r['metric_quality'] = quality
		if quality == 'ok':
			n_ok += 1
		else:
			n_unc += 1

	_write_metric_csv(fieldnames, rows, output_csv_path)
	print(f"  Wrote {os.path.basename(output_csv_path)}: "
		  f"{len(rows)} rows (ok={n_ok}, uncertain={n_unc}, none={n_none}).")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _write_metric_csv(fieldnames, rows, output_csv_path):
	out_fields = list(fieldnames) + [c for c in _NEW_COLS if c not in fieldnames]
	os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
	with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction='ignore')
		writer.writeheader()
		writer.writerows(rows)


def _write_all_none(fieldnames, rows, output_csv_path):
	"""Emit the CSV with empty metric columns (quality 'none') — used when the
	video or flight log is unavailable, so downstream steps still find the file."""
	for r in rows:
		r['X_m'] = ''; r['Y_m'] = ''; r['metric_quality'] = 'none'
	_write_metric_csv(fieldnames, rows, output_csv_path)


# ---------------------------------------------------------------------------
# Batch / project entry points
# ---------------------------------------------------------------------------

def _build_index(root, exts):
	index = {}
	if not root or not os.path.isdir(root):
		return index
	for dirpath, _, files in os.walk(root):
		for fname in files:
			if fname.lower().endswith(exts):
				index.setdefault(fname, os.path.join(dirpath, fname))
	return index


def _find_by_stem(stem, index, exts):
	for ext in exts:
		cand = stem + ext
		if cand in index:
			return index[cand]
	return None


def run_metric_geometry(config_path):
	"""Batch-project every *_tracking(_corrected).csv in the project's output
	folder to ground-plane metres, writing <video>_tracking_metric.csv."""
	config_path = os.path.abspath(config_path)
	project_dir = os.path.dirname(config_path)
	params = load_metric_config(config_path)

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

	video_exts = ('.mp4', '.avi', '.mov', '.mkv')
	video_index = _build_index(input_dir, video_exts)
	for fname, path in _build_index(clips_dir, video_exts).items():
		video_index.setdefault(fname, path)
	log_index = _build_index(input_dir, ('.flightlog.csv',))
	for fname, path in _build_index(clips_dir, ('.flightlog.csv',)).items():
		log_index.setdefault(fname, path)

	# Pick, per video, the most-processed CSV so X_m/Y_m are attached to the final
	# (stitched) identities: stitched > drone-corrected > raw.
	jobs = {}
	for suf in ('_tracking_stitched.csv', '_tracking_corrected.csv', '_tracking.csv'):
		for p in sorted(glob.glob(os.path.join(output_dir, '*' + suf))):
			jobs.setdefault(os.path.basename(p)[:-len(suf)], p)

	if not jobs:
		print(f"Metric geometry: no tracking CSVs found in {output_dir}")
		return

	print(f"Metric geometry: processing {len(jobs)} video(s) "
		  f"(focal={params['focal_len_mm']}mm, sensor={params['sensor_width_mm']}mm)...")

	vid_exts_cased = ('.MP4', '.mp4', '.avi', '.mov', '.mkv', '.AVI', '.MOV', '.MKV')
	for stem, csv_path in sorted(jobs.items()):
		video_path = _find_by_stem(stem, video_index, vid_exts_cased)
		log_path = _find_by_stem(stem, log_index, ('.flightlog.csv',))
		out_path = os.path.join(output_dir, stem + '_tracking_metric.csv')
		try:
			metric_video_tracking(
				video_path, csv_path, log_path, out_path,
				focal_len_mm=params['focal_len_mm'],
				sensor_width_mm=params['sensor_width_mm'],
				roll_max_deg=params['roll_max_deg'],
				horizon_margin_px=params['horizon_margin_px'],
				fpx_overrides=params['fpx_overrides'])
		except Exception as e:
			import traceback
			print(f"  ERROR on {os.path.basename(csv_path)}: {e}")
			traceback.print_exc()

	print("Metric geometry complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
	parser = argparse.ArgumentParser(
		description="Project BehaveAI tracking CSVs to ground-plane metres using flight-log telemetry.")
	parser.add_argument('target', help="Project directory or BehaveAI_settings.ini (batch), "
										"or a video file when --csv is given (single-file).")
	parser.add_argument('--csv', default=None, help="Single-file mode: the *_tracking.csv.")
	parser.add_argument('--flightlog', default=None, help="Single-file mode: the .flightlog.csv.")
	parser.add_argument('--out', default=None, help="Single-file mode: output CSV path.")
	parser.add_argument('--focal-len-mm', type=float, default=24.0)
	parser.add_argument('--sensor-width-mm', type=float, default=36.0)
	args = parser.parse_args()

	if args.csv is not None:
		out = args.out or (args.csv
			.replace('_tracking_stitched.csv', '_tracking_metric.csv')
			.replace('_tracking_corrected.csv', '_tracking_metric.csv')
			.replace('_tracking.csv', '_tracking_metric.csv'))
		metric_video_tracking(args.target, args.csv, args.flightlog, out,
							  focal_len_mm=args.focal_len_mm, sensor_width_mm=args.sensor_width_mm)
		return

	target = os.path.abspath(args.target)
	ini = os.path.join(target, 'BehaveAI_settings.ini') if os.path.isdir(target) else target
	if not os.path.exists(ini):
		print(f"Settings file not found: {ini}")
		sys.exit(1)
	run_metric_geometry(ini)


if __name__ == '__main__':
	_main()
