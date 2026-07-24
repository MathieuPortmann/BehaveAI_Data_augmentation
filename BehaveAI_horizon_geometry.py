#!/usr/bin/env python3
"""
BehaveAI Ground-Plane Geometry (horizon estimation from horse apparent size)

Estimates camera pitch and the image scale field directly from the tracked
horses, with no gimbal telemetry. On a flat ground plane with zero camera
roll, an object's apparent size in pixels is proportional to the distance (in
image rows) between its base and the horizon line:

    size_px = slope * (y_base - y_horizon)

which is a plain linear regression size_px = slope * y_base + intercept, so
y_horizon = -intercept / slope. Fitting this over every adult horse in a
video (foals are excluded — they bias the reference size) gives both the
camera pitch (from y_horizon) and the scale field (from slope) in one pass.
Roll is assumed zero, which the gimbal enforces; the drone is assumed static
for the duration of the fit (one geometry per video).

This is the horizon-estimation step (ground-plane rectification for full 2D
distances is a follow-on: once y_horizon/slope are trusted, they parametrise
the homography that unwarps the ground plane before Kalman tracking).
estimate_metric_scale() covers the LATERAL (tangential) direction only —
exact for a zero-roll camera over flat ground — which is enough to convert
body-length-normalised sizes/distances into metres; the DEPTH (radial)
direction compresses nonlinearly and still needs the full rectification.

Inputs: a HERDWISE tracking CSV (<video>_tracking.csv or
_tracking_corrected.csv) with the TASK 0 bbox columns (x1, y1, x2, y2), plus
(for metric calibration) rel_alt telemetry from the matching .SRT — see
BehaveAI_srt_telemetry.py.

CLI: python BehaveAI_horizon_geometry.py <tracking_csv> [--frame-height 2160]
"""

import argparse

import numpy as np

from BehaveAI_complex_features import load_tracking_csv, compute_body_len_ref


def _collect_adult_boxes(track_data, is_foal):
	"""Return (frames, y_base, size) arrays for every adult horse box.

	y_base = bottom of the box (y2, the feet). size = the larger box side
	(orientation-robust: a horse's apparent length dominates whether it is
	lying broadside or facing the camera).
	"""
	box = track_data['box']
	frames, y_base, size = [], [], []
	for (frame, tid), (x1, y1, x2, y2) in box.items():
		if is_foal.get(tid, False):
			continue
		frames.append(frame)
		y_base.append(y2)
		size.append(max(x2 - x1, y2 - y1))
	return (np.asarray(frames, dtype=int),
			np.asarray(y_base, dtype=float),
			np.asarray(size, dtype=float))


def _fit_ground_plane(y_base, size, min_boxes=20, min_spread=None):
	"""Core MAD-robust linear fit of size_px = slope * y_base + intercept.

	Outliers (bad detections, box glitches) are rejected via one MAD-based
	pass, then refit. Returns None (not raises) when there aren't enough
	points — callers decide whether that's fatal (global fit) or just a
	skipped window (windowed fit).

	When min_spread is given, a fit is refused (returned as
	degenerate=True) if the y_base spread is too small to constrain the
	regression (small windows where horses sit at similar depth), or if the
	fitted slope comes out <= 0 (physically impossible — apparent size
	cannot shrink as an object gets closer to the bottom of the frame).
	Both are signs of an ill-conditioned fit, not a real geometry result.
	"""
	n_total = len(y_base)
	if n_total < min_boxes:
		return None

	spread = float(np.percentile(y_base, 95) - np.percentile(y_base, 5))
	if min_spread is not None and spread < min_spread:
		return {'degenerate': True, 'reason': 'insufficient_depth_spread',
				'n_total': n_total, 'spread_rows': spread}

	slope, intercept = np.polyfit(y_base, size, 1)
	residuals = size - (slope * y_base + intercept)
	mad = float(np.median(np.abs(residuals - np.median(residuals))))
	keep = np.abs(residuals) < 3 * 1.4826 * mad if mad > 0 else np.ones(n_total, dtype=bool)
	slope, intercept = np.polyfit(y_base[keep], size[keep], 1)

	if min_spread is not None and slope <= 0:
		return {'degenerate': True, 'reason': 'non_positive_slope',
				'n_total': n_total, 'spread_rows': spread,
				'slope': float(slope), 'intercept': float(intercept)}

	y_horizon = -intercept / slope if slope != 0 else float('nan')

	return {
		'slope': float(slope),
		'intercept': float(intercept),
		'y_horizon': float(y_horizon),
		'n_total': n_total,
		'n_kept': int(keep.sum()),
		'spread_rows': spread,
		'degenerate': False,
	}


def estimate_ground_plane(track_data, is_foal, min_boxes=20):
	"""Single video-wide fit. Valid when the camera pitch is constant for the
	whole clip (guaranteed by a hovering drone; NOT guaranteed handheld —
	use estimate_ground_plane_windowed()/summarize_drift() to check first).

	Raises ValueError if there are too few adult boxes to fit reliably.
	"""
	_, y_base, size = _collect_adult_boxes(track_data, is_foal)
	result = _fit_ground_plane(y_base, size, min_boxes=min_boxes)
	if result is None:
		raise ValueError(
			f"Only {len(y_base)} adult box(es) available (need >= {min_boxes}) — "
			f"cannot fit the ground plane reliably.")
	return result


def estimate_ground_plane_windowed(track_data, is_foal, frame_height, window_frames=300, step_frames=None,
									min_boxes=20, min_spread_frac=0.15):
	"""Fit the ground plane in overlapping sliding windows over the clip.

	Use this to detect camera pitch drift (e.g. handheld footage where the
	operator tilts while tracking horses) instead of trusting one global fit.
	window_frames/step_frames default to 300/150 (10s/5s at 30fps).

	A short window can have plenty of boxes yet still be ill-conditioned if
	the horses in it all sit at similar depth (small y_base spread) — the
	regression is then numerically unstable and can produce impossible
	results (e.g. a negative slope). min_spread_frac (fraction of
	frame_height) guards against that: windows below it are marked
	degenerate rather than fit. Lower than the global fit's 0.25 warning
	threshold since a single window covers far less scene/time.

	Returns a list of dicts (chronological). Each is one of:
	  - {window_start, window_end, n_total, skipped=True} — fewer than
	    min_boxes adult boxes in the window.
	  - {window_start, window_end, degenerate=True, reason, ...} —
	    insufficient depth spread or a non-positive fitted slope.
	  - {window_start, window_end, degenerate=False, slope, intercept,
	    y_horizon, n_total, n_kept, spread_rows} — a valid fit.
	"""
	if step_frames is None:
		step_frames = window_frames // 2
	min_spread = min_spread_frac * frame_height

	frames, y_base, size = _collect_adult_boxes(track_data, is_foal)
	if len(frames) == 0:
		return []

	f_min, f_max = int(frames.min()), int(frames.max())
	windows = []
	start = f_min
	while start <= f_max:
		end = start + window_frames
		mask = (frames >= start) & (frames < end)
		n = int(mask.sum())
		if n < min_boxes:
			windows.append({'window_start': start, 'window_end': end, 'n_total': n, 'skipped': True})
		else:
			fit = _fit_ground_plane(y_base[mask], size[mask], min_boxes=min_boxes, min_spread=min_spread)
			fit['window_start'] = start
			fit['window_end'] = end
			fit['skipped'] = False
			windows.append(fit)
		start += step_frames
	return windows


def summarize_drift(windows):
	"""Summarise pitch stability from estimate_ground_plane_windowed() output.

	Returns None if fewer than 2 valid (non-skipped, non-degenerate)
	windows (can't say anything about drift), else a dict with the
	window-to-window y_horizon range. Whether that range is "too much"
	depends on scene depth (frame_height is a reasonable yardstick) — left
	to the caller/CLI to threshold.
	"""
	valid = [w for w in windows if not w.get('skipped') and not w.get('degenerate')]
	n_skipped = sum(1 for w in windows if w.get('skipped'))
	n_degenerate = sum(1 for w in windows if w.get('degenerate'))
	if len(valid) < 2:
		return None
	horizons = np.array([w['y_horizon'] for w in valid])
	return {
		'n_windows': len(valid),
		'n_skipped': n_skipped,
		'n_degenerate': n_degenerate,
		'y_horizon_min': float(horizons.min()),
		'y_horizon_max': float(horizons.max()),
		'y_horizon_median': float(np.median(horizons)),
		'y_horizon_range': float(horizons.max() - horizons.min()),
	}


def _describe_horizon(y_horizon, frame_height):
	if 0 <= y_horizon <= frame_height:
		return "horizon INSIDE the frame: camera is close to horizontal."
	elif y_horizon < 0:
		return f"horizon {abs(y_horizon):.0f} rows above the frame: camera is tilted down."
	else:
		return f"horizon {y_horizon - frame_height:.0f} rows below the frame: camera is tilted up."


# ---------------------------------------------------------------------------
# Metric calibration (pixels -> metres, using altitude telemetry)
# ---------------------------------------------------------------------------

def pixel_focal_length(focal_len_mm, frame_width_px, sensor_width_mm=36.0):
	"""Convert a 35mm-equivalent focal length to a pixel focal length.

	Assumes the video's horizontal field of view matches what focal_len_mm
	would produce on a sensor_width_mm-wide sensor (the standard "35mm
	equivalent" convention, referenced to a 36mm-wide full-frame sensor).
	This is an approximation — DJI's equivalent-focal-length spec is
	usually diagonal-FOV-matched, not width-matched, and the video crop may
	not equal the full sensor width — good to a few percent, not
	survey-grade. Cross-validate with estimate_metric_scale() rather than
	trusting it blindly.
	"""
	return frame_width_px * focal_len_mm / sensor_width_mm


def camera_pitch_rad(y_horizon, frame_height_px, f_px):
	"""Camera pitch below horizontal (radians) from the fitted horizon row.
	Assumes the principal point is at the image vertical centre and roll=0
	(the gimbal's job, still true handheld — see module docstring)."""
	cy = frame_height_px / 2.0
	return float(np.arctan2(cy - y_horizon, f_px))


def estimate_metric_scale(fit_result, rel_alt_m, focal_len_mm, frame_width_px, frame_height_px,
						   assumed_body_length_m=2.2):
	"""Calibrate a ground-plane fit into real metres using drone altitude
	telemetry (rel_alt), cross-checked against an assumed real horse body
	length — never trust one scale source alone (see plan notes).

	Exact result for a zero-roll pinhole camera over flat ground, derived
	from first principles (matches the empirical linear fit exactly):
	given camera height h above ground and pitch theta below horizontal,

	    slope = S * cos(theta) / h

	where S is an object's real lateral size (m) and slope is this
	module's fitted size_px/row slope. Two independent ways to use it:
	  - telemetry-anchored: implied_body_length_m = slope * h / cos(theta),
	    using h = rel_alt_m — compare to a real horse's plausible size.
	  - body-length-anchored: implied_height_m = assumed_body_length_m *
	    cos(theta) / slope — compare to the measured rel_alt_m.
	Large disagreement between the two (height_agreement_ratio far from 1)
	flags a violated assumption: sloped terrain, rel_alt biased by a
	takeoff point at a different elevation than the horses, or
	body_len_ref not measuring the same thing as assumed_body_length_m.

	Also returns gsd_lateral_mpp(y): metres-per-pixel in the LATERAL
	(tangential) direction only at image row y — exact under the same
	assumptions. The DEPTH (radial, toward/away from camera) direction
	compresses nonlinearly and is NOT covered here; converting depth
	distances needs the full ground-plane rectification (module docstring).
	"""
	f_px = pixel_focal_length(focal_len_mm, frame_width_px)
	theta = camera_pitch_rad(fit_result['y_horizon'], frame_height_px, f_px)
	cos_theta = float(np.cos(theta))
	slope = fit_result['slope']

	implied_body_length_m = slope * rel_alt_m / cos_theta
	implied_height_m = assumed_body_length_m * cos_theta / slope if slope != 0 else float('nan')
	agreement = implied_height_m / rel_alt_m if rel_alt_m else float('nan')

	def gsd_lateral_mpp(y):
		dy = y - fit_result['y_horizon']
		if dy <= 0:
			raise ValueError("y must be below the fitted horizon (y > y_horizon) to convert to ground scale.")
		return rel_alt_m / (dy * cos_theta)

	return {
		'f_px': f_px,
		'pitch_deg': float(np.degrees(theta)),
		'rel_alt_m': rel_alt_m,
		'implied_body_length_m': implied_body_length_m,
		'assumed_body_length_m': assumed_body_length_m,
		'implied_height_m': implied_height_m,
		'height_agreement_ratio': agreement,
		'gsd_lateral_mpp': gsd_lateral_mpp,
	}


# ---------------------------------------------------------------------------
# Ground-plane projection (inverse perspective mapping) — full 2D metres
# ---------------------------------------------------------------------------
#
# Exact model for a zero-roll pinhole camera at height h above flat ground,
# pitched pitch_deg below horizontal (pitch_deg > 0 = looking down). Principal
# point (cx, cy); pixel focal length f_px. A ground point (X forward, Y
# lateral, in metres, origin under the camera) projects to pixel (u, v):
#     u = cx + f_px * Y / D ,  v = cy + f_px * (h*cosθ - X*sinθ) / D
#     with depth D = X*cosθ + h*sinθ  and θ = radians(pitch_deg).
# Inverting gives the ground coordinates of the pixel where an animal's feet
# touch the ground — this is what turns pixel distances into real metres in
# ANY direction (unlike estimate_metric_scale's lateral-only scale).


def horizon_row(pitch_deg, f_px, cy):
	"""Image row of the horizon line (where ground depth -> infinity)."""
	theta = np.radians(pitch_deg)
	return cy - f_px * np.tan(theta)


def ground_point_from_pixel(u, v, h_m, pitch_deg, f_px, cx, cy):
	"""Inverse-project an image pixel to ground-plane metres (X forward, Y lateral).

	Returns (X_m, Y_m) or None when the pixel is at/above the horizon or maps
	behind the camera (no valid ground intersection). Assumes zero roll and
	flat ground; the caller supplies h/pitch from telemetry (gb pitch is
	negative when looking down, so pass pitch_deg = -gb_pitch).
	"""
	theta = float(np.radians(pitch_deg))
	ct, st = np.cos(theta), np.sin(theta)
	a = (u - cx) / f_px
	b = (v - cy) / f_px
	denom = b * ct + st
	if denom <= 1e-9:                       # at/above the horizon line
		return None
	X = h_m * (ct - b * st) / denom
	depth = X * ct + h_m * st
	if depth <= 1e-9:                       # behind the camera
		return None
	Y = a * depth
	return float(X), float(Y)


def ground_distance_m(u1, v1, u2, v2, h_m, pitch_deg, f_px, cx, cy):
	"""Euclidean ground distance (m) between two image pixels' ground points.
	Returns None if either pixel fails to project (at/above horizon)."""
	p1 = ground_point_from_pixel(u1, v1, h_m, pitch_deg, f_px, cx, cy)
	p2 = ground_point_from_pixel(u2, v2, h_m, pitch_deg, f_px, cx, cy)
	if p1 is None or p2 is None:
		return None
	return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def _main():
	parser = argparse.ArgumentParser(
		description="Estimate camera pitch (horizon line) and scale field from tracked horse sizes.")
	parser.add_argument('tracking_csv', help="<video>_tracking.csv or _tracking_corrected.csv")
	parser.add_argument('--frame-height', type=int, default=2160, help="Video frame height in pixels (default: 2160 for 4K).")
	parser.add_argument('--frame-width', type=int, default=3840, help="Video frame width in pixels (default: 3840 for 4K).")
	parser.add_argument('--foal-ratio', type=float, default=0.7, help="Size ratio below which a track is treated as a foal (default: 0.7).")
	parser.add_argument('--srt', default=None,
						 help="Matching .SRT telemetry file: calibrate the global fit into real metres using its "
							  "rel_alt (mean over the clip) and focal_len. Only applies to the global fit "
							  "(not --check-drift) — check drift stability first.")
	parser.add_argument('--assumed-body-length-m', type=float, default=2.2,
						 help="Real adult horse body length (m) used as the independent cross-check "
							  "against rel_alt-based calibration (default: 2.2).")
	parser.add_argument('--check-drift', action='store_true',
						 help="Handheld/uncertain footage: fit in sliding windows instead of trusting one global fit, "
							  "and report whether y_horizon drifts (camera pitch not constant).")
	parser.add_argument('--window-frames', type=int, default=300, help="Sliding-window size in frames (default: 300 = 10s @30fps).")
	parser.add_argument('--step-frames', type=int, default=None, help="Sliding-window step in frames (default: window-frames/2).")
	parser.add_argument('--drift-thresh-frac', type=float, default=0.25,
						 help="y_horizon range across windows, as a fraction of frame height, above which pitch is "
							  "flagged unstable (default: 0.25).")
	args = parser.parse_args()

	track_data = load_tracking_csv(args.tracking_csv)
	if not track_data['has_bbox']:
		print("ERROR: this tracking CSV has no bbox columns (x1,y1,x2,y2) — "
			  "re-run classify_track (TASK 0 bbox output) to regenerate it.")
		return

	ref, size_ratio, is_foal = compute_body_len_ref(track_data, foal_ratio=args.foal_ratio)
	n_foal = sum(1 for v in is_foal.values() if v)
	print(f"body_len_ref={ref:.1f}px across {len(track_data['id_frames'])} track(s) ({n_foal} likely foal(s), excluded).")

	if args.check_drift:
		windows = estimate_ground_plane_windowed(
			track_data, is_foal, args.frame_height, window_frames=args.window_frames, step_frames=args.step_frames)
		print(f"\n{'window':>16}{'n_boxes':>10}{'y_horizon':>12}{'slope':>10}")
		for w in windows:
			if w.get('skipped'):
				print(f"{w['window_start']:>7}-{w['window_end']:<7}{w['n_total']:>10}{'SKIPPED (too few boxes)':>26}")
			elif w.get('degenerate'):
				print(f"{w['window_start']:>7}-{w['window_end']:<7}{w['n_total']:>10}{'DEGENERATE (' + w['reason'] + ')':>36}")
			else:
				print(f"{w['window_start']:>7}-{w['window_end']:<7}{w['n_total']:>10}{w['y_horizon']:>12.0f}{w['slope']:>10.4f}")

		drift = summarize_drift(windows)
		if drift is None:
			print("\nNot enough valid windows to assess drift — clip too short, too few adult boxes per window, "
				  "or all windows were degenerate (insufficient depth spread per window).")
			return
		print(f"\ny_horizon across {drift['n_windows']} window(s) ({drift['n_skipped']} skipped, "
			  f"{drift['n_degenerate']} degenerate): "
			  f"median={drift['y_horizon_median']:.0f}, range={drift['y_horizon_range']:.0f} rows")
		range_frac = drift['y_horizon_range'] / args.frame_height
		if range_frac > args.drift_thresh_frac:
			print(f"-> UNSTABLE: y_horizon drifts by {range_frac:.0%} of frame height across windows. "
				  f"Camera pitch is not constant — do NOT trust a single global fit for this clip; "
				  f"segment it first or use the per-window geometry.")
		else:
			print(f"-> STABLE: y_horizon drift ({range_frac:.0%} of frame height) is small. "
				  f"A single global fit is a reasonable approximation for this clip.")
		print(_describe_horizon(drift['y_horizon_median'], args.frame_height))
		return

	result = estimate_ground_plane(track_data, is_foal)
	spread_frac = result['spread_rows'] / args.frame_height
	print(f"y_base spread = {result['spread_rows']:.0f} rows ({spread_frac:.0%} of frame height)")
	if spread_frac < 0.25:
		print("WARNING: horses too concentrated in depth - fit will be unreliable.")

	kept_frac = result['n_kept'] / result['n_total']
	print(f"slope = {result['slope']:.4f} px/row | y_horizon = {result['y_horizon']:.0f} "
		  f"| kept {result['n_kept']}/{result['n_total']} ({kept_frac:.1%})")
	print(_describe_horizon(result['y_horizon'], args.frame_height))

	if args.srt:
		from BehaveAI_srt_telemetry import parse_srt, focal_len_mm
		telemetry = parse_srt(args.srt)
		rel_alts = [float(r['rel_alt']) for r in telemetry if 'rel_alt' in r]
		focals = [focal_len_mm(r['focal_len']) for r in telemetry if 'focal_len' in r]
		if not rel_alts or not focals:
			print(f"\nERROR: no rel_alt/focal_len telemetry parsed from {args.srt} — cannot calibrate.")
			return
		rel_alt_mean = float(np.mean(rel_alts))
		rel_alt_range = max(rel_alts) - min(rel_alts)
		focal_mean = float(np.mean(focals))
		print(f"\nSRT telemetry: rel_alt mean={rel_alt_mean:.1f}m (range {min(rel_alts):.1f}-{max(rel_alts):.1f}m), "
			  f"focal_len={focal_mean:.1f}mm")
		if rel_alt_range > 0.15 * rel_alt_mean:
			print(f"WARNING: rel_alt varies by {rel_alt_range:.1f}m ({rel_alt_range/rel_alt_mean:.0%} of the mean) "
				  f"over the clip — camera height is not constant, the metric calibration below (which assumes a "
				  f"single height) is only a rough average.")

		calib = estimate_metric_scale(result, rel_alt_mean, focal_mean, args.frame_width, args.frame_height,
									   assumed_body_length_m=args.assumed_body_length_m)
		print(f"\npitch = {calib['pitch_deg']:.1f} deg below horizontal")
		print(f"telemetry-anchored: implied real horse body length = {calib['implied_body_length_m']:.2f}m "
			  f"(using rel_alt={rel_alt_mean:.1f}m)")
		print(f"body-length-anchored: implied camera height = {calib['implied_height_m']:.2f}m "
			  f"(assuming body length={args.assumed_body_length_m}m) vs measured rel_alt={rel_alt_mean:.1f}m")
		ratio = calib['height_agreement_ratio']
		if 0.8 <= ratio <= 1.25:
			print(f"-> AGREEMENT good (ratio={ratio:.2f}): the two independent scale sources roughly agree — "
				  f"the metric calibration is plausible.")
		else:
			print(f"-> AGREEMENT poor (ratio={ratio:.2f}): the two independent scale sources disagree substantially — "
				  f"do not trust the metric calibration without investigating (sloped terrain, rel_alt bias, "
				  f"assumed body length, or detection-box quality).")


if __name__ == '__main__':
	_main()
