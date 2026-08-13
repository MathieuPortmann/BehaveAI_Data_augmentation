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

TWO ground frames are emitted, because they answer different questions:

  - X_m, Y_m       : per-frame, CAMERA-RELATIVE (origin under the drone, Y along
                     the camera heading), from the RAW feet pixel and that
                     frame's own height/pitch. Two animals in the SAME frame
                     share this origin, so their separation is a true ground
                     distance. Differencing these over TIME is meaningless as
                     soon as the drone moves — the origin moves with it.
  - Xs_m, Ys_m     : STABILISED, from the DRONE-CORRECTED feet pixel projected
                     with ONE reference height/pitch. The drone correction has
                     already re-anchored every centroid into the first processed
                     frame's image frame, so a single projection yields a ground
                     frame that stands still for the whole clip. This is the
                     frame in which speeds, headings and approach rates are
                     real. Valid only while the drone holds its station — see
                     telemetry_drift().

Input:  <video>_tracking(_corrected).csv  +  <video>.flightlog.csv (same stem).
Output: <video>_tracking_metric.csv = the input CSV with five appended columns
        X_m, Y_m, Xs_m, Ys_m (ground coords, metres) and metric_quality
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

try:
	from behaveai_config import resolve_project_dir
except ImportError:  # run directly: the repo root is not on sys.path
	sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from behaveai_config import resolve_project_dir

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')


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
		# Cross-check of the metric scale against the animals' apparent size.
		# Tolerance 0.25 means the two independent estimates of camera height must
		# agree within a factor 1.25 either way — the band the horizon CLI has
		# always used to call a calibration plausible.
		'assumed_body_length_m': float(d.get('metric_assumed_body_length_m', '2.2')),
		'scale_tolerance':  float(d.get('metric_scale_tolerance', '0.25')),
		# Ground-plane stability check (report-only, see geometry_drift): how far
		# y_horizon may move between sliding windows, as a fraction of frame
		# height, before the clip is called non-flat; and the window length.
		'geometry_drift_frac': float(d.get('metric_geometry_drift_frac', '0.25')),
		'geometry_window_s': float(d.get('metric_geometry_window_s', '10.0')),
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


def video_dims_fps(video_path):
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


def _stabilised_contact_pixel(row, has_bbox):
	"""Feet of the animal expressed in the drone-STABILISED image frame.

	The drone correction maps each centroid from its own frame into the reference
	frame's image coordinates; that displacement is a global re-anchoring, so the
	same offset applies to the feet at first order:
	    feet_stab = feet_raw + (corrected_centroid - raw_centroid)
	Returns (u, v, from_bbox), or (None, None, False) when the row carries no
	usable correction."""
	try:
		xc = (row.get('x_corrected', '') or '').strip()
		yc = (row.get('y_corrected', '') or '').strip()
		if not xc or not yc or row.get('correction_quality', '') == 'none':
			return None, None, False
		dx = float(xc) - float(row['x'])
		dy = float(yc) - float(row['y'])
	except (ValueError, KeyError, TypeError):
		return None, None, False
	u, v, from_bbox = _ground_contact_pixel(row, has_bbox)
	if u is None:
		return None, None, False
	return u + dx, v + dy, from_bbox


def _angular_range_deg(values):
	"""Peak-to-peak range of an angle series, in degrees, wrap-safe (a heading
	crossing 360 must not read as a 360-degree turn)."""
	v = np.asarray(list(values) if values is not None else [], dtype=float)
	v = v[~np.isnan(v)]
	if v.size < 2:
		return 0.0
	unwrapped = np.degrees(np.unwrap(np.radians(v)))
	return float(unwrapped.max() - unwrapped.min())


def scale_disagreement(tracking_csv_path, flightlog, f_px, width, height,
					   focal_len_mm, assumed_body_length_m, tolerance):
	"""Cross-check the metric scale against a second, independent source.

	Every ground distance scales LINEARLY with the camera height taken from the
	flight log, so a biased rel_alt biases every distance by the same percentage
	— a systematic error that no amount of averaging removes. rel_alt is
	barometric and referenced to the TAKEOFF point, so a takeoff spot 2 m above
	the animals' ground already costs ~7%, and sloped terrain costs more.

	The independent check: fit the animals' apparent size against image row
	(horizon_geometry.estimate_ground_plane) and ask what real body length that
	implies given the telemetry height. If the two disagree, one of the
	assumptions behind the metric conversion is violated — flat ground, an
	unbiased rel_alt, or the focal length.

	Returns (reason_or_None, ratio_or_None). reason is a human-readable string
	when the check FAILS; None when it passes or cannot be run.
	"""
	try:
		from .horizon_geometry import (load_boxes_csv, foal_flags,
									   estimate_ground_plane, estimate_metric_scale)
	except ImportError:
		from horizon_geometry import (load_boxes_csv, foal_flags,
									  estimate_ground_plane, estimate_metric_scale)

	alt = flightlog.get('rel_alt_m') if flightlog else None
	if alt is None:
		return None, None
	a = np.asarray(alt, dtype=float)
	a = a[~np.isnan(a)]
	if a.size == 0:
		return None, None
	rel_alt_mean = float(a.mean())

	try:
		td = load_boxes_csv(tracking_csv_path)
		if not td['has_bbox']:
			return None, None
		fit = estimate_ground_plane(td, foal_flags(td))
	except (ValueError, OSError, KeyError):
		# Too few adult boxes to fit: no second opinion available, which is not
		# itself evidence of a problem.
		return None, None

	calib = estimate_metric_scale(fit, rel_alt_mean, focal_len_mm, width, height,
								  assumed_body_length_m=assumed_body_length_m, f_px=f_px)
	ratio = calib['height_agreement_ratio']
	if not np.isfinite(ratio) or ratio <= 0:
		return None, None
	lo, hi = 1.0 / (1.0 + tolerance), 1.0 + tolerance
	if lo <= ratio <= hi:
		return None, float(ratio)
	return (f"the two independent scale sources disagree by {abs(ratio - 1.0):.0%} "
			f"(telemetry says the camera was {rel_alt_mean:.1f} m up; the animals' "
			f"apparent size implies {calib['implied_height_m']:.1f} m, i.e. a real "
			f"body length of {calib['implied_body_length_m']:.2f} m vs the assumed "
			f"{assumed_body_length_m:.2f} m)"), float(ratio)


def geometry_drift(tracking_csv_path, frame_height, fps, window_seconds=10.0,
				   drift_thresh_frac=0.25, ill_spread_frac=0.25):
	"""Is ONE ground-plane fit valid for the whole clip? Asks the ANIMALS, not the log.

	This is the third guard, and the only one that can see the case the other two
	miss: sloped terrain under a perfectly steady gimbal. telemetry_drift() reads
	the flight log, which reports a rock-steady drone; scale_disagreement() fits
	the whole clip at once, so a geometry that changes over the clip can average
	back into the tolerance band. Fitting the size-vs-image-row relation in
	sliding windows exposes it: y_horizon moves when the ground plane does.

	REPORT-ONLY by design — it prints and returns a reason, but the caller does
	NOT downgrade metric_quality on it. Known false-alarm mode: on herd scenes
	('Multi-Harems') the animals cluster at one depth inside a 10 s window, which
	destabilises the LOCAL fit while the global geometry is fine — one such clip
	scored 97.3% inliers and a coherent metric calibration while this check called
	it 69% UNSTABLE.

	It therefore reports TWO numbers and thresholds on the first:
	  - the peak-to-peak RANGE of y_horizon across windows, which is what decides;
	  - the robust IQR, which says whether that range is a steady drift (IQR large
	    too) or a handful of bad windows (IQR small).
	Thresholding the IQR instead was tried and rejected: on the one clip known to
	be genuinely sloped (Dovns Klint, user-confirmed cliff site) the range is 47%
	of frame height while the IQR is only 14%, so an IQR gate at any plausible
	tolerance MISSES the very case this guard exists for. The range keeps the
	sensitivity; the IQR qualifies the alarm instead of replacing it.

	Windows below ill_spread_frac of the frame height are dropped first — a window
	can clear the degeneracy floor and still be too ill-conditioned to vote.

	Returns (reason_or_None, range_frac_or_None); reason is set when it FAILS.
	"""
	try:
		from .horizon_geometry import (load_boxes_csv, foal_flags,
									   estimate_ground_plane_windowed, summarize_drift)
	except ImportError:
		from horizon_geometry import (load_boxes_csv, foal_flags,
									  estimate_ground_plane_windowed, summarize_drift)
	if not frame_height or not fps or fps <= 0:
		return None, None
	try:
		td = load_boxes_csv(tracking_csv_path)
		if not td['has_bbox']:
			return None, None
		windows = estimate_ground_plane_windowed(
			td, foal_flags(td), frame_height,
			window_frames=max(60, int(round(window_seconds * fps))))
	except (ValueError, OSError, KeyError):
		return None, None

	drift = summarize_drift(windows, min_spread_rows=ill_spread_frac * frame_height)
	if drift is None:
		# Too few well-conditioned windows to have an opinion, which is not itself
		# evidence of a problem (short clip, sparse herd, or animals at one depth).
		return None, None
	range_frac = drift['y_horizon_range'] / float(frame_height)
	iqr_frac = drift['y_horizon_iqr'] / float(frame_height)
	if range_frac <= drift_thresh_frac:
		return None, range_frac
	shape = ("a steady drift" if iqr_frac > drift_thresh_frac / 2.0
			 else f"a few outlying windows (robust IQR only {iqr_frac:.0%})")
	return (f"the ground-plane fit is not constant across the clip: y_horizon moves by "
			f"{range_frac:.0%} of frame height across {drift['n_windows']} window(s) "
			f"({drift['n_skipped']} skipped, {drift['n_degenerate']} degenerate, "
			f"{drift['n_ill_conditioned']} ill-conditioned), and the shape of it is "
			f"{shape}"), range_frac


def telemetry_drift(flightlog, alt_frac_max=0.15, pitch_deg_max=2.0, yaw_deg_max=5.0):
	"""Reasons why ONE fixed camera geometry cannot represent the whole clip.

	The stabilised ground frame (Xs_m/Ys_m) projects every drone-corrected pixel
	with a single (height, pitch) pair, which only holds while the drone keeps
	its station. Returns a list of human-readable reasons; empty means stable.
	The 15% altitude threshold matches the one the horizon-fit CLI already warns
	on, so both paths flag the same clips."""
	if not flightlog:
		return ['no flight log']
	reasons = []
	alt = flightlog.get('rel_alt_m')
	if alt is not None:
		a = np.asarray(alt, dtype=float)
		a = a[~np.isnan(a)]
		if a.size >= 2 and a.mean() > 0:
			frac = float(a.max() - a.min()) / float(a.mean())
			if frac > alt_frac_max:
				reasons.append(f"rel_alt varies by {frac:.0%} of its mean (> {alt_frac_max:.0%})")
	pitch_range = _angular_range_deg(flightlog.get('gb_pitch'))
	if pitch_range > pitch_deg_max:
		reasons.append(f"gb_pitch drifts {pitch_range:.1f} deg (> {pitch_deg_max} deg)")
	yaw_range = _angular_range_deg(flightlog.get('gb_yaw'))
	if yaw_range > yaw_deg_max:
		reasons.append(f"gb_yaw turns {yaw_range:.1f} deg (> {yaw_deg_max} deg)")
	return reasons


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

_NEW_COLS = ['X_m', 'Y_m', 'Xs_m', 'Ys_m', 'metric_quality']


def _anchor_frame(rows):
	"""Lowest frame number carrying a usable drone correction — the frame whose
	image coordinates the corrected columns are anchored to."""
	best = None
	for r in rows:
		if not (r.get('x_corrected', '') or '').strip():
			continue
		if r.get('correction_quality', '') == 'none':
			continue
		try:
			f = int(r['frame'])
		except (ValueError, KeyError, TypeError):
			continue
		if best is None or f < best:
			best = f
	return best


def metric_video_tracking(video_path, tracking_csv_path, flightlog_path, output_csv_path,
						  focal_len_mm=24.0, sensor_width_mm=36.0,
						  roll_max_deg=3.0, horizon_margin_px=50.0, fpx_overrides=None,
						  assumed_body_length_m=2.2, scale_tolerance=0.25,
						  geometry_drift_frac=0.25, geometry_window_s=10.0):
	"""Project every tracked detection to ground-plane metres using the flight
	log, and write <video>_tracking_metric.csv. Rows without usable telemetry
	keep empty X_m/Y_m and metric_quality='none'."""
	fpx_overrides = fpx_overrides or {}
	fieldnames, rows, has_bbox = _read_tracking_csv(tracking_csv_path)
	if not rows:
		print(f"  {os.path.basename(tracking_csv_path)}: no rows, skipping.")
		return

	dims = video_dims_fps(video_path) if video_path else None
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

	# --- Reference geometry for the stabilised ground frame ---------------
	# The drone-corrected coordinates all live in the anchor frame's image frame,
	# so ONE (height, pitch) pair projects the whole clip into a ground frame
	# that stands still. That only holds while the drone keeps its station.
	anchor = _anchor_frame(rows)
	h0 = pitch0_deg = float('nan')
	if anchor is not None:
		h0, gb_pitch0, _ = sample_flightlog(flightlog, (anchor - 1) / fps)
		pitch0_deg = -gb_pitch0
		print(f"  {stem}: stabilised frame anchored on frame {anchor} "
			  f"(h={h0:.1f}m, pitch={pitch0_deg:.1f} deg below horizontal).")
	else:
		print(f"  {stem}: no drone-corrected rows — stabilised frame (Xs_m/Ys_m) "
			  f"unavailable; run drone correction first if you need speeds.")

	drift_reasons = telemetry_drift(flightlog)
	if drift_reasons and anchor is not None:
		print(f"  {stem}: WARNING — the drone did not hold its station "
			  f"({'; '.join(drift_reasons)}). A single reference geometry no longer "
			  f"describes the clip, so every row is downgraded to metric_quality="
			  f"'uncertain' and Xs_m/Ys_m-derived speeds must not be trusted.")

	# --- Scale cross-check: is the telemetry height believable? --------------
	scale_reason, scale_ratio = scale_disagreement(
		tracking_csv_path, flightlog, f_px, width, height,
		focal_len_mm, assumed_body_length_m, scale_tolerance)
	if scale_reason:
		print(f"  {stem}: WARNING — {scale_reason}. Ground distances scale linearly "
			  f"with camera height, so every distance in this clip is likely off by "
			  f"a similar factor. Usual causes: sloped terrain, a takeoff point at a "
			  f"different elevation than the animals, or a wrong focal length. Rows "
			  f"downgraded to metric_quality='uncertain'.")
	elif scale_ratio is not None:
		print(f"  {stem}: scale cross-check OK (independent height estimates agree "
			  f"within {abs(scale_ratio - 1.0):.0%}).")

	# --- Ground-plane stability: the only guard that can see terrain slope ----
	# REPORT-ONLY (see geometry_drift's docstring): it prints, it does not feed
	# the quality downgrade below, because its false-alarm rate on herd scenes is
	# not yet measured. To promote it, add `or geom_reason` to the quality test.
	geom_reason, geom_frac = geometry_drift(
		tracking_csv_path, height, fps,
		window_seconds=geometry_window_s, drift_thresh_frac=geometry_drift_frac)
	if geom_reason:
		print(f"  {stem}: WARNING — {geom_reason}. Neither the flight log nor the scale "
			  f"cross-check can see this: both are consistent with a steady drone over "
			  f"SLOPED ground. Distances stay in the CSV at their current quality, but "
			  f"treat this clip's metres as indicative until you check it by hand with "
			  f"`horizon_geometry.py <csv> --check-drift`.")
	elif geom_frac is not None:
		print(f"  {stem}: ground-plane stability OK (y_horizon moves {geom_frac:.0%} of frame height "
			  f"across windows).")

	n_ok = n_unc = n_none = n_stab = 0
	for r in rows:
		try:
			frame = int(r['frame'])
		except (ValueError, KeyError, TypeError):
			_blank_metric(r); n_none += 1
			continue
		t_s = (frame - 1) / fps                       # csv frame N -> video frame N-1
		h_m, gb_pitch, gb_roll = sample_flightlog(flightlog, t_s)

		u, v, from_bbox = _ground_contact_pixel(r, has_bbox)
		if (u is None or np.isnan(h_m) or np.isnan(gb_pitch)):
			_blank_metric(r); n_none += 1
			continue

		pitch_deg = -gb_pitch                          # gb_pitch<0 looking down -> theta>0
		pt = ground_point_from_pixel(u, v, h_m, pitch_deg, f_px, cx, cy)
		if pt is None:                                 # at/above horizon
			_blank_metric(r); n_none += 1
			continue

		X_m, Y_m = pt
		# Quality: downgrade for bad roll, missing bbox, or near-horizon margin.
		v_horizon = horizon_row(pitch_deg, f_px, cy)
		margin = v - v_horizon
		quality = 'ok'
		if (not np.isnan(gb_roll) and abs(gb_roll) > roll_max_deg) \
				or not from_bbox or margin < horizon_margin_px \
				or drift_reasons or scale_reason:
			quality = 'uncertain'
		r['X_m'] = f"{X_m:.3f}"; r['Y_m'] = f"{Y_m:.3f}"; r['metric_quality'] = quality

		# Stabilised ground frame: corrected feet, reference geometry.
		r['Xs_m'] = ''; r['Ys_m'] = ''
		if anchor is not None and not np.isnan(h0) and not np.isnan(pitch0_deg):
			us, vs, _ = _stabilised_contact_pixel(r, has_bbox)
			if us is not None:
				pts = ground_point_from_pixel(us, vs, h0, pitch0_deg, f_px, cx, cy)
				if pts is not None:
					r['Xs_m'] = f"{pts[0]:.3f}"; r['Ys_m'] = f"{pts[1]:.3f}"
					n_stab += 1

		if quality == 'ok':
			n_ok += 1
		else:
			n_unc += 1

	_write_metric_csv(fieldnames, rows, output_csv_path)
	print(f"  Wrote {os.path.basename(output_csv_path)}: "
		  f"{len(rows)} rows (ok={n_ok}, uncertain={n_unc}, none={n_none}; "
		  f"{n_stab} with stabilised coords).")


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def _blank_metric(row):
	"""Mark a row as carrying no usable metric position, in either ground frame."""
	row['X_m'] = ''; row['Y_m'] = ''
	row['Xs_m'] = ''; row['Ys_m'] = ''
	row['metric_quality'] = 'none'


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
		_blank_metric(r)
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


def _media_dirs(config_path):
	"""(input_dir, clips_dir) resolved against the project directory."""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	project_dir = os.path.dirname(os.path.abspath(config_path))
	return (resolve_project_dir(cfg, project_dir, 'input'),
			resolve_project_dir(cfg, project_dir, 'clips'))


def build_video_index(config_path):
	"""Index every source video of a project by filename (input/ then clips/).
	Public so other stages can locate a video without re-deriving the layout."""
	input_dir, clips_dir = _media_dirs(config_path)
	index = _build_index(input_dir, tuple(e.lower() for e in VIDEO_EXTS))
	for fname, path in _build_index(clips_dir, tuple(e.lower() for e in VIDEO_EXTS)).items():
		index.setdefault(fname, path)
	return index


def find_video_for_stem(stem, video_index):
	"""Path of the source video for a tracking-CSV stem, or None."""
	return _find_by_stem(stem, video_index, VIDEO_EXTS)


def video_fps_for_stem(config_path, stem, default=30.0, video_index=None):
	"""Frames per second of a video, by tracking-CSV stem.

	Every stage that converts frames to seconds needs this; resolving it here
	keeps one definition of where a project's videos live. Falls back to
	`default` when the video cannot be found or read."""
	index = build_video_index(config_path) if video_index is None else video_index
	path = find_video_for_stem(stem, index)
	if not path:
		return float(default)
	dims = video_dims_fps(path)
	if not dims or dims[2] <= 0:
		return float(default)
	return float(dims[2])


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

	output_dir = resolve_project_dir(d, project_dir, 'output')
	input_dir, clips_dir = _media_dirs(config_path)

	video_index = build_video_index(config_path)
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

	for stem, csv_path in sorted(jobs.items()):
		video_path = find_video_for_stem(stem, video_index)
		log_path = _find_by_stem(stem, log_index, ('.flightlog.csv',))
		out_path = os.path.join(output_dir, stem + '_tracking_metric.csv')
		try:
			metric_video_tracking(
				video_path, csv_path, log_path, out_path,
				focal_len_mm=params['focal_len_mm'],
				sensor_width_mm=params['sensor_width_mm'],
				roll_max_deg=params['roll_max_deg'],
				horizon_margin_px=params['horizon_margin_px'],
				fpx_overrides=params['fpx_overrides'],
				assumed_body_length_m=params['assumed_body_length_m'],
				scale_tolerance=params['scale_tolerance'],
				geometry_drift_frac=params['geometry_drift_frac'],
				geometry_window_s=params['geometry_window_s'])
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
