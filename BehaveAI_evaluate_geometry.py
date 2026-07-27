#!/usr/bin/env python3
"""BehaveAI geometry validation -- drone motion correction (+ metric scaffold).

The complex-behaviour features (approach rates, velocities, group cohesion) are
built on drone-corrected, and optionally metre-scaled, trajectories. That
correction is a real algorithm, not an aggregation, so its accuracy must be
validated before any downstream kinematic claim.

Three checks:

  1. Synthetic recovery -- warp a textured frame by KNOWN similarity transforms
     (translation / rotation / zoom / combined) and let the pipeline's own
     estimator recover them (``drone_correction._estimate_step_transform``).
     Reports the reprojection error; near-zero means the estimator is sound.

  2. Correction-quality summary -- over the existing
     ``output/*_tracking_corrected.csv`` (and the ``*_correction_diag.csv``
     sidecars, when present): the ok/uncertain/none breakdown plus the
     continuous residual-flow-std distribution (median / p95 / max).

  3. Metric distance error (optional, scaffold) -- with a
     ``known_distances.csv`` (video, frame, u1,v1,u2,v2, true_m) and a flight log,
     computes ground-plane distances and their error. Skips gracefully when no
     telemetry is configured (the HERDWISE project has no flight logs yet).

Usage:
  python BehaveAI_evaluate_geometry.py <project_dir | BehaveAI_settings.ini>
      [--known-distances path.csv] [--tol 0.5]

Outputs: <project>/evaluation/geometry_report.txt
"""

import os
import csv
import glob
import math
import argparse

import numpy as np
import cv2

import behaveai_eval_common as ec
from behaveai_drone import drone_correction as dc


# ---------------------------------------------------------------------------
# 1. Synthetic recovery of a known transform
# ---------------------------------------------------------------------------

def _make_texture(h=720, w=1280, seed=0):
	"""A deterministic textured grey image with many trackable corners."""
	rng = np.random.default_rng(seed)
	img = np.full((h, w), 110, dtype=np.uint8)
	for _ in range(400):
		x = int(rng.integers(0, w)); y = int(rng.integers(0, h))
		bw = int(rng.integers(8, 46)); bh = int(rng.integers(8, 46))
		val = int(rng.integers(0, 256))
		cv2.rectangle(img, (x, y), (min(x + bw, w - 1), min(y + bh, h - 1)), val, -1)
	return cv2.GaussianBlur(img, (3, 3), 0)


def _similarity_2x3(s, theta_deg, tx, ty):
	"""Build a 2x3 similarity (scale+rotation+translation) transform."""
	th = math.radians(theta_deg)
	a = s * math.cos(th)
	b = s * math.sin(th)
	return np.array([[a, -b, tx], [b, a, ty]], dtype=np.float64)


def synthetic_recovery(tol=0.5):
	"""Recover known transforms with the pipeline estimator. Returns (rows, ok)."""
	prev = _make_texture()
	h, w = prev.shape
	# Grid of evaluation points (interior, away from the warp border).
	gx, gy = np.meshgrid(np.linspace(w * 0.15, w * 0.85, 8),
						 np.linspace(h * 0.15, h * 0.85, 8))
	grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

	cases = [
		("translation", _similarity_2x3(1.0, 0.0, 14.0, -9.0)),
		("rotation_3deg", _similarity_2x3(1.0, 3.0, 0.0, 0.0)),
		("zoom_1.05", _similarity_2x3(1.05, 0.0, 0.0, 0.0)),
		("combined", _similarity_2x3(1.03, 2.0, 8.0, 5.0)),
	]
	rows = []
	all_ok = True
	for name, M_true in cases:
		cur = cv2.warpAffine(prev, M_true, (w, h), flags=cv2.INTER_LINEAR,
							 borderMode=cv2.BORDER_REFLECT)
		M3_est, info = dc._estimate_step_transform(
			prev, cur, None, 'affine', 30, 8.0, None)
		if M3_est is None:
			rows.append({'case': name, 'n_inliers': info['n_inliers'],
						 'mean_err_px': float('nan'), 'max_err_px': float('nan'), 'pass': False})
			all_ok = False
			continue
		M_true_3 = dc._to_3x3(M_true, 'affine')
		proj_true = dc._apply_transform_points(M_true_3, grid)
		proj_est = dc._apply_transform_points(M3_est, grid)
		err = np.linalg.norm(proj_true - proj_est, axis=1)
		mean_err, max_err = float(err.mean()), float(err.max())
		ok = mean_err <= tol
		all_ok = all_ok and ok
		rows.append({'case': name, 'n_inliers': info['n_inliers'],
					 'mean_err_px': mean_err, 'max_err_px': max_err, 'pass': ok})
	return rows, all_ok


# ---------------------------------------------------------------------------
# 2. Correction-quality summary from existing outputs
# ---------------------------------------------------------------------------

def quality_summary(output_dir):
	"""Summarise correction_quality (and residual std where diag sidecars exist)
	across output/*_tracking_corrected.csv. Returns list of per-file dicts."""
	files = sorted(glob.glob(os.path.join(output_dir, '*_tracking_corrected.csv')))
	out = []
	for path in files:
		counts = {'ok': 0, 'uncertain': 0, 'none': 0}
		with open(path, newline='', encoding='utf-8', errors='replace') as f:
			for r in csv.DictReader(f):
				q = (r.get('correction_quality') or '').strip()
				if q in counts:
					counts[q] += 1
		rec = {'file': os.path.basename(path), 'total': sum(counts.values()), **counts}

		diag = path[:-len('_tracking_corrected.csv')] + '_correction_diag.csv'
		resid = []
		if os.path.exists(diag):
			with open(diag, newline='', encoding='utf-8', errors='replace') as f:
				for r in csv.DictReader(f):
					v = (r.get('residual_std') or '').strip()
					if v:
						try:
							resid.append(float(v))
						except ValueError:
							pass
		if resid:
			a = np.asarray(resid)
			rec['resid_median'] = float(np.median(a))
			rec['resid_p95'] = float(np.percentile(a, 95))
			rec['resid_max'] = float(a.max())
		out.append(rec)
	return out


# ---------------------------------------------------------------------------
# 3. Metric distance error (optional scaffold)
# ---------------------------------------------------------------------------

def metric_distance_error(project_dir, config, known_csv):
	"""Compute ground-plane distance error against known references. Returns
	(rows, note). Skips rows whose video has no flight log."""
	rows = []
	try:
		from behaveai_drone import horizon_geometry as hg
		from behaveai_drone import flightlog as fl
		from behaveai_drone import metric_geometry as mg
	except Exception as e:
		return rows, f"metric modules unavailable ({e})"

	mcfg = mg.load_metric_config(ec.resolve_config_path(project_dir)) \
		if hasattr(mg, 'load_metric_config') else {}
	focal_mm = float(mcfg.get('focal_len_mm', 24.0)) if isinstance(mcfg, dict) else 24.0
	sensor_mm = float(mcfg.get('sensor_width_mm', 36.0)) if isinstance(mcfg, dict) else 36.0
	input_dir = ec.resolve_dir(project_dir, config, 'input_dir', 'input')

	refs = []
	with open(known_csv, newline='', encoding='utf-8', errors='replace') as f:
		for r in csv.DictReader(f):
			refs.append(r)

	# Cache per-video telemetry + dims.
	vid_cache = {}

	def _load_video(stem):
		if stem in vid_cache:
			return vid_cache[stem]
		flightlog_path = None
		vpath = None
		for root, _dirs, fnames in os.walk(input_dir):
			for fn in fnames:
				base, ext = os.path.splitext(fn)
				if base == stem and ext.lower() in ('.mp4', '.mov', '.avi', '.mkv'):
					vpath = os.path.join(root, fn)
				if fn == stem + '.flightlog.csv':
					flightlog_path = os.path.join(root, fn)
		info = None
		if vpath and flightlog_path:
			cap = cv2.VideoCapture(vpath)
			if cap.isOpened():
				w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
				h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
				fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
				cap.release()
				flog = fl.load_flightlog(flightlog_path)
				f_px = hg.pixel_focal_length(focal_mm, w, sensor_mm)
				info = {'w': w, 'h': h, 'fps': fps, 'flog': flog, 'f_px': f_px}
		vid_cache[stem] = info
		return info

	skipped = 0
	for r in refs:
		stem = os.path.splitext(str(r.get('video', '')).strip())[0]
		info = _load_video(stem)
		if not info or not info['flog']:
			skipped += 1
			continue
		try:
			frame = int(float(r['frame']))
			u1, v1 = float(r['u1']), float(r['v1'])
			u2, v2 = float(r['u2']), float(r['v2'])
			true_m = float(r['true_m'])
		except (KeyError, ValueError):
			continue
		t_s = (frame - 1) / info['fps']
		rel_alt, gb_pitch, _gb_roll = fl.sample_flightlog(info['flog'], t_s)
		cx, cy = info['w'] / 2.0, info['h'] / 2.0
		est = hg.ground_distance_m(u1, v1, u2, v2, rel_alt, -gb_pitch, info['f_px'], cx, cy)
		if est is None:
			continue
		rows.append({'video': stem, 'frame': frame, 'true_m': true_m,
					 'est_m': est, 'abs_err_m': abs(est - true_m)})
	note = f"{skipped} reference(s) skipped (no flight log for their video)" if skipped else ""
	return rows, note


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	ap = argparse.ArgumentParser(description="Validate BehaveAI drone/metric geometry.")
	ap.add_argument('project', help="project directory or BehaveAI_settings.ini path")
	ap.add_argument('--known-distances', default=None,
					help="optional CSV (video,frame,u1,v1,u2,v2,true_m) for metric error")
	ap.add_argument('--tol', type=float, default=0.5, help="synthetic recovery tolerance (px)")
	a = ap.parse_args()

	config_path = ec.resolve_config_path(a.project)
	if not os.path.exists(config_path):
		ap.error(f"settings not found: {config_path}")
	project_dir = os.path.dirname(config_path)
	config = ec.load_config(config_path)
	output_dir = ec.resolve_dir(project_dir, config, 'output_dir', 'output')

	lines = []
	lines.append("BehaveAI geometry validation")
	lines.append("=" * 60)
	lines.append(f"project : {project_dir}")
	lines.append("")

	# 1. Synthetic recovery
	syn_rows, syn_ok = synthetic_recovery(tol=a.tol)
	lines.append(f"1. Synthetic transform recovery (tol = {a.tol} px): "
				 f"{'PASS' if syn_ok else 'FAIL'}")
	lines.append(f"   {'case':<16} {'inliers':>7} {'mean_err_px':>12} {'max_err_px':>11} {'pass':>5}")
	for r in syn_rows:
		lines.append(f"   {r['case']:<16} {r['n_inliers']:>7} "
					 f"{r['mean_err_px']:>12.4f} {r['max_err_px']:>11.4f} "
					 f"{str(r['pass']):>5}")
	lines.append("")

	# 2. Correction-quality summary
	q_rows = quality_summary(output_dir)
	lines.append("2. Drone-correction quality (from output/*_tracking_corrected.csv):")
	if not q_rows:
		lines.append("   (no corrected CSVs found -- run the pipeline with "
					 "drone_correction_enabled first)")
	else:
		agg = {'ok': 0, 'uncertain': 0, 'none': 0, 'total': 0}
		for r in q_rows:
			pct = (100.0 * r['ok'] / r['total']) if r['total'] else 0.0
			extra = ""
			if 'resid_median' in r:
				extra = (f"  resid px median/p95/max = "
						 f"{r['resid_median']:.2f}/{r['resid_p95']:.2f}/{r['resid_max']:.2f}")
			lines.append(f"   {r['file']}: ok={r['ok']} unc={r['uncertain']} "
						 f"none={r['none']} ({pct:.1f}% ok){extra}")
			for k in ('ok', 'uncertain', 'none', 'total'):
				agg[k] += r[k]
		pct = (100.0 * agg['ok'] / agg['total']) if agg['total'] else 0.0
		lines.append(f"   TOTAL: ok={agg['ok']} uncertain={agg['uncertain']} "
					 f"none={agg['none']} ({pct:.1f}% ok)")
	lines.append("")

	# 3. Metric distance error (optional)
	lines.append("3. Metric distance error:")
	if a.known_distances:
		m_rows, note = metric_distance_error(project_dir, config, a.known_distances)
		if m_rows:
			errs = np.array([r['abs_err_m'] for r in m_rows])
			lines.append(f"   references scored: {len(m_rows)}  "
						 f"MAE = {errs.mean():.3f} m  max = {errs.max():.3f} m")
			for r in m_rows:
				lines.append(f"   {r['video']} f{r['frame']}: "
							 f"true={r['true_m']:.2f} est={r['est_m']:.2f} "
							 f"|err|={r['abs_err_m']:.3f} m")
		else:
			lines.append("   (no reference could be scored)")
		if note:
			lines.append(f"   note: {note}")
	else:
		lines.append("   (skipped -- no --known-distances CSV; metric geometry is not "
					 "configured for this project)")
	lines.append("")

	report = "\n".join(lines) + "\n"
	eval_dir = ec.ensure_eval_dir(project_dir)
	with open(os.path.join(eval_dir, 'geometry_report.txt'), 'w', encoding='utf-8') as f:
		f.write(report)
	print(report)
	print(f"Wrote geometry_report.txt -> {eval_dir}")


if __name__ == '__main__':
	main()
