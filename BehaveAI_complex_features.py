#!/usr/bin/env python3
"""
BehaveAI Complex Features + Interaction Graph

Deterministic layer that turns a (drone-corrected) tracking CSV into per-frame
DYADIC and GROUP features, and aggregates them into the INTERACTION GRAPH — the
primary analysis output of the HERDWISE multi-individual pipeline, directly
importable into R igraph.

Group features are computed over the WHOLE co-present herd per frame (no
spatial sub-grouping): every individual present in a frame counts as one group
for that frame.

EVERYTHING IS IN METRES. Distances are real ground distances (m), speeds are
real ground speeds (m/s), areas are m^2. There is no body-length normalisation
and no pixel-space fallback: a video without usable metric geometry is skipped
outright rather than silently measured in a different unit, because mixing
body-lengths and metres across videos makes the resulting features
incomparable — which is the whole point of the graph.

Two ground frames come out of the metric stage and are used for what each is
valid for (see behaveai_drone/metric_geometry.py):
  - X_m, Y_m   : per-frame, camera-relative -> DISTANCES between animals seen
                 in the same frame, contact, group area/cohesion.
  - Xs_m, Ys_m : stabilised, one reference geometry -> SPEEDS, headings,
                 approach rates. Differencing X_m instead would measure the
                 drone's motion, not the horse's.

Foal/adult comes from the trained age classifier (model_age, "model 0.5") via
the age_class/age_conf columns — never re-derived from box size here. An
individual the age model did not label is 'unknown', not guessed.

Inputs (per video, from the project output folder):
  - <video>_tracking_metric.csv  (REQUIRED: carries X_m/Y_m/Xs_m/Ys_m)

Outputs (overwritten on every run / parameter change):
  - <video>_interaction_edges.csv   (one row per ordered dyad, episode, or frame)
  - <video>_interaction_nodes.csv   (per track_id node attributes)

Dependencies: numpy, scipy, networkx. pandas is intentionally NOT required —
the project uses the stdlib csv module and the deliverable is CSV for R igraph,
so the "df" returns here are lists of dicts written with csv.DictWriter (R reads
them identically). networkx is required for graph metrics; if it is missing the
edges/nodes files are still written and only centrality/community summaries are
skipped (with a clear message).

igraph note: the edges columns follow the spec order (frame_start, frame_end,
source_id, target_id, ...). In R, build the graph by pointing igraph at the
id columns, e.g.:
    e <- read.csv("..._interaction_edges.csv")
    v <- read.csv("..._interaction_nodes.csv")
    g <- igraph::graph_from_data_frame(
            e[, c("source_id","target_id",
                  setdiff(names(e), c("source_id","target_id")))],
            directed = TRUE, vertices = v)
"""

import os
import sys
import csv
import glob
import math
import argparse
import configparser
from collections import defaultdict, Counter

import numpy as np

try:
	import networkx as nx
	_NX_AVAILABLE = True
except Exception:
	_NX_AVAILABLE = False


# Speed below which motion cannot be told from detection jitter. This is measured
# per clip by estimate_speed_noise_floor(); the constant below is only the
# fallback for clips with too little data to measure. A fixed value cannot be
# right in general — the floor scales with altitude, box size and detector
# quality. On 4K footage at ~45 m, the 1.8 px of box jitter measured in the
# HERDWISE corpus already yields ~0.4 m/s of apparent speed on a standing horse,
# so the few-cm/s value one would naively pick filters nothing at all.
_FALLBACK_SPEED_FLOOR_MS = 0.4

# Quantile of the simulated noise-only speed distribution taken as the floor.
_SPEED_FLOOR_QUANTILE = 0.90


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Keys that measured things in pixels / body lengths, and what replaced them when
# the interaction layer moved to metres. Silently ignoring a stale key would let a
# project believe it is gating interactions at 800 (px) while the code reads a
# default of 15 (m), so a leftover key is an error, not a warning. No automatic
# conversion: turning 800 px into metres needs a scale that does not exist
# retroactively.
_RETIRED_KEYS = {
	'complex_max_interaction_distance': 'complex_max_interaction_distance_m (now in METRES)',
	'complex_contact_dist_bodylen':     'complex_contact_dist_m (now in METRES)',
	'complex_speed_low_bodylen':        'complex_speed_low_ms (now in m/s)',
	'complex_speed_high_bodylen':       'complex_speed_high_ms (now in m/s)',
	'foal_size_ratio_thresh':           'nothing — foal/adult now comes from the age classifier (model_age)',
	'body_len_ref_scope':               'nothing — it never had any effect and body lengths are gone',
}

# Granularity spellings: 'per_interaction' described one edge per EPISODE, which
# is not what it produced (one edge per dyad for the whole clip). Renamed, old
# spelling still accepted so existing INIs keep working.
_GRANULARITY_ALIASES = {'per_interaction': 'per_dyad'}


def reject_retired_keys(d, config_path=''):
	"""Raise if an INI still carries a pixel/body-length key from before the
	metric migration. Shared with the model + candidate stages so a stale project
	fails the same way whichever entry point is used."""
	stale = [k for k in _RETIRED_KEYS if k in d]
	if not stale:
		return
	where = f" in {os.path.basename(config_path)}" if config_path else ""
	lines = "\n".join(f"    {k}  ->  {_RETIRED_KEYS[k]}" for k in stale)
	raise ValueError(
		f"The interaction layer now works in metres, but{where} these retired "
		f"settings are still present:\n{lines}\n"
		f"Remove them (and set the replacement where there is one) — they are not "
		f"converted automatically, because pixels cannot be turned into metres "
		f"after the fact.")


def normalise_granularity(value):
	"""Canonical granularity name, accepting the pre-rename spelling."""
	value = (value or '').strip() or 'per_dyad'
	return _GRANULARITY_ALIASES.get(value, value)


def load_complex_config(config_path):
	"""Read the TASK 4 parameters from a BehaveAI INI (DEFAULT section)."""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	reject_retired_keys(d, config_path)
	return {
		'max_interaction_distance_m': float(d.get('complex_max_interaction_distance_m', '15')),
		'min_duration_frames':        int(float(d.get('complex_min_duration_frames', '10'))),
		'contact_iou_thresh':         float(d.get('complex_contact_iou_thresh', '0.05')),
		'contact_dist_m':             float(d.get('complex_contact_dist_m', '2.5')),
		'window_frames':              int(float(d.get('complex_window_frames', '30'))),
		'edge_granularity':           normalise_granularity(d.get('interaction_edge_granularity', 'per_dyad')),
		'weight_metric':              d.get('interaction_weight_metric', 'sri'),
	}


# ---------------------------------------------------------------------------
# Loading a tracking CSV
# ---------------------------------------------------------------------------

def _best_class(row, kind):
	"""Pick the primary or secondary YOLO class label from a tracking row.

	kind = 'primary' -> motion class preferred over static; 'secondary' likewise.
	Returns '' when none is present.
	"""
	if kind == 'primary':
		m = (row.get('primary_motion_class', '') or '').strip()
		s = (row.get('primary_static_class', '') or '').strip()
	else:
		m = (row.get('secondary_motion_class', '') or '').strip()
		s = (row.get('secondary_static_class', '') or '').strip()
	if m:
		return m
	if s:
		return s
	return ''


class MissingMetricError(Exception):
	"""A tracking CSV carries no usable ground-plane coordinates.

	Raised instead of falling back to pixels: a video measured in a different
	unit than its neighbours would poison every cross-video comparison the
	interaction graph exists to support."""


def load_tracking_csv(csv_path, fps=30.0):
	"""Load a metric tracking CSV into a track_data dict, in METRES.

	Requires the columns the metric stage appends (X_m/Y_m and, for speeds,
	Xs_m/Ys_m). Rows whose metric_quality is 'none' carry no ground position and
	are dropped. Raises MissingMetricError when nothing usable is left.

	pos  : per-frame camera-relative ground metres  -> distances within a frame
	pos_s: stabilised ground metres                 -> displacement over time
	vel  : m/s in the stabilised frame, from smoothed finite differences
	"""
	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fieldnames = list(reader.fieldnames or [])
		rows = [dict(r) for r in reader]

	name = os.path.basename(csv_path)
	if not ('X_m' in fieldnames and 'Y_m' in fieldnames):
		raise MissingMetricError(
			f"{name} has no X_m/Y_m columns — run the metric geometry stage "
			f"(metric_enabled = true) before the interaction graph.")
	has_stab = 'Xs_m' in fieldnames and 'Ys_m' in fieldnames
	has_bbox = all(c in fieldnames for c in ('x1', 'y1', 'x2', 'y2'))
	has_age = 'age_class' in fieldnames
	if not has_bbox:
		print(f"  WARNING: {name} has no bbox columns — overlap features disabled.")
	if not has_stab:
		print(f"  WARNING: {name} has no Xs_m/Ys_m columns — speeds unavailable "
			  f"(re-run the metric stage after drone correction).")

	present = defaultdict(list)
	pos, pos_s, box, prim, sec = {}, {}, {}, {}, {}
	age, age_conf = {}, {}
	id_frames = defaultdict(list)
	n_dropped = 0

	for r in rows:
		try:
			frame = int(r['frame'])
			tid = str(r['id'])
		except (ValueError, KeyError, TypeError):
			continue
		if (frame, tid) in pos:
			continue
		if r.get('metric_quality', '') == 'none' or (r.get('X_m', '') or '').strip() == '':
			n_dropped += 1
			continue
		try:
			x = float(r['X_m']); y = float(r['Y_m'])
		except (ValueError, TypeError):
			n_dropped += 1
			continue

		present[frame].append(tid)
		pos[(frame, tid)] = (x, y)
		id_frames[tid].append(frame)
		prim[(frame, tid)] = _best_class(r, 'primary')
		sec[(frame, tid)] = _best_class(r, 'secondary')
		if has_age:
			age[(frame, tid)] = (r.get('age_class', '') or '').strip()
			try:
				age_conf[(frame, tid)] = float(r.get('age_conf', '') or 0.0)
			except (ValueError, TypeError):
				age_conf[(frame, tid)] = 0.0
		if has_stab and (r.get('Xs_m', '') or '').strip() != '':
			try:
				pos_s[(frame, tid)] = (float(r['Xs_m']), float(r['Ys_m']))
			except (ValueError, TypeError):
				pass
		if has_bbox:
			try:
				x1, y1, x2, y2 = int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2'])
				if x2 > x1 and y2 > y1:
					box[(frame, tid)] = (x1, y1, x2, y2)
			except (ValueError, TypeError):
				pass

	if not present:
		raise MissingMetricError(
			f"{name}: every row has metric_quality='none' — no usable ground "
			f"positions (missing flight log, or the camera never saw the ground "
			f"below the horizon).")
	if n_dropped:
		print(f"  {name}: dropped {n_dropped} row(s) without a ground position.")

	# Speeds live in the STABILISED frame: differencing the per-frame camera
	# frame would return the drone's own motion. Metres per second, from the real
	# frame gap (frame_skip aware).
	vel = _stabilised_velocities(pos_s, id_frames, fps)

	# What counts as "moving" in THIS clip, measured from its own noise.
	speed_floor, sigma_pos, n_resid = estimate_speed_noise_floor(pos_s, id_frames, fps)

	return {
		'frames': sorted(present.keys()),
		'present': present,
		'pos': pos, 'pos_s': pos_s, 'vel': vel, 'box': box,
		'prim': prim, 'sec': sec,
		'age': age, 'age_conf': age_conf,
		'id_frames': {t: sorted(fs) for t, fs in id_frames.items()},
		'has_bbox': has_bbox, 'has_stab': bool(pos_s), 'has_age': has_age,
		'fps': float(fps),
		'speed_floor_ms': speed_floor,
		'pos_noise_m': sigma_pos,
		'pos_noise_samples': n_resid,
	}


def _stabilised_velocities(pos_s, id_frames, fps):
	"""Per-track ground velocity in m/s, from smoothed stabilised positions.

	Tracks with no stabilised coordinates get zero velocity rather than a
	pixel-space substitute — a wrong speed is worse than a missing one."""
	vel = {}
	fps = float(fps) if fps and fps > 0 else 30.0
	for tid, frames in id_frames.items():
		fs = [f for f in sorted(frames) if (f, tid) in pos_s]
		if len(fs) < 2:
			for f in frames:
				vel[(f, tid)] = (0.0, 0.0)
			continue
		xs = _smooth(np.array([pos_s[(f, tid)][0] for f in fs], dtype=float))
		ys = _smooth(np.array([pos_s[(f, tid)][1] for f in fs], dtype=float))
		# t in SECONDS -> gradient is already m/s.
		ts = np.array(fs, dtype=float) / fps
		if np.all(np.diff(ts) > 0):
			vx = np.gradient(xs, ts); vy = np.gradient(ys, ts)
		else:
			vx = np.zeros(len(fs)); vy = np.zeros(len(fs))
		for i, f in enumerate(fs):
			vel[(f, tid)] = (float(vx[i]), float(vy[i]))
		for f in frames:
			vel.setdefault((f, tid), (0.0, 0.0))
	return vel


def estimate_speed_noise_floor(pos_s, id_frames, fps, win=5,
							   quantile=_SPEED_FLOOR_QUANTILE, min_samples=60):
	"""Ground speed (m/s) below which motion cannot be told from detection noise.

	Differentiating position amplifies noise enormously: at 30 fps a 4 cm error on
	a single frame is over 1 m/s of apparent speed before smoothing. So the same
	box jitter that leaves distances accurate to a few centimetres makes a
	standing horse look like it is walking. That threshold has to be MEASURED,
	not guessed, because it scales with altitude, box size and detector quality.

	Two steps:
	  1. Position-noise scale: the high-frequency residual pos - smooth(pos),
	     which real movement (slow relative to the smoothing window) does not
	     reach. A robust MAD-based scale keeps genuine manoeuvres from inflating
	     it, and the residual is de-biased for the variance a w-point moving
	     average absorbs (var(resid) = sigma^2 * (1 - 1/w)).
	  2. Propagate through the EXACT smoothing + gradient path the pipeline uses,
	     by Monte-Carlo on a synthetic stationary track, and take `quantile` of
	     the resulting speeds. Simulating beats an analytic formula here because
	     it automatically tracks any change to _smooth() or the differencing.

	Returns (floor_ms, sigma_pos_m, n_samples); floor falls back to
	_FALLBACK_SPEED_FLOOR_MS when there is too little data to measure.
	"""
	fps = float(fps) if fps and fps > 0 else 30.0
	resid = []
	for tid, frames in id_frames.items():
		fs = [f for f in sorted(frames) if (f, tid) in pos_s]
		if len(fs) < 3 * win:
			continue
		xs = np.array([pos_s[(f, tid)][0] for f in fs], dtype=float)
		ys = np.array([pos_s[(f, tid)][1] for f in fs], dtype=float)
		resid.extend(np.hypot(xs - _smooth(xs, win), ys - _smooth(ys, win)).tolist())

	if len(resid) < min_samples:
		return _FALLBACK_SPEED_FLOOR_MS, float('nan'), len(resid)

	r = np.asarray(resid, dtype=float)
	# Robust scale of a 2-D residual magnitude: median / 1.1774 is the Rayleigh
	# equivalent of the MAD, and is unmoved by the heavy tail that real movement
	# and tracking glitches put on this distribution.
	sigma = float(np.median(r)) / 1.1774
	absorbed = math.sqrt(max(1.0 - 1.0 / win, 1e-6))
	sigma /= absorbed
	if not np.isfinite(sigma) or sigma <= 0:
		return _FALLBACK_SPEED_FLOOR_MS, float('nan'), len(resid)

	# Monte-Carlo the noise-only speed distribution through the real code path.
	rng = np.random.default_rng(0)          # fixed seed: same data -> same floor
	n_steps, n_tracks = 120, 40
	t = np.arange(n_steps) / fps
	speeds = []
	for _ in range(n_tracks):
		nx = _smooth(rng.normal(0.0, sigma, n_steps), win)
		ny = _smooth(rng.normal(0.0, sigma, n_steps), win)
		speeds.extend(np.hypot(np.gradient(nx, t), np.gradient(ny, t)).tolist())
	floor = float(np.quantile(speeds, quantile))
	return floor, sigma, len(resid)


def _smooth(arr, win=5):
	"""Light moving-average smoothing (edge-padded) for velocity estimation."""
	arr = np.asarray(arr, dtype=float)
	n = len(arr)
	if n < 3:
		return arr
	win = min(win, n)
	if win % 2 == 0:
		win -= 1
	if win < 3:
		return arr
	kernel = np.ones(win) / win
	return np.convolve(np.pad(arr, win // 2, mode='edge'), kernel, mode='valid')[:n]


# ---------------------------------------------------------------------------
# Age (foal / adult) from the trained age classifier
# ---------------------------------------------------------------------------

UNKNOWN_AGE = 'unknown'


def track_age_classes(track_data):
	"""Return (age_by_track, mean_conf_by_track) from the age classifier's
	per-detection output.

	The pipeline already runs a dedicated age model (model_age, "model 0.5")
	whose verdict rides in the age_class/age_conf columns; a per-track
	CONFIDENCE-WEIGHTED majority vote turns those per-detection labels into one
	label per individual. A track the model never labelled is UNKNOWN_AGE — it
	is not guessed from box size, because apparent size at 15-50 m confounds age
	with distance, posture and orientation.
	"""
	votes = defaultdict(lambda: defaultdict(float))
	confs = defaultdict(list)
	for (f, tid), label in track_data.get('age', {}).items():
		if not label:
			continue
		w = track_data.get('age_conf', {}).get((f, tid), 0.0)
		votes[tid][label] += max(float(w), 1e-6)   # an unconfident vote still counts
		confs[tid].append(float(w))

	age_by_track, conf_by_track = {}, {}
	for tid in track_data['id_frames']:
		v = votes.get(tid)
		if v:
			age_by_track[tid] = max(v, key=v.get)
			conf_by_track[tid] = float(np.mean(confs[tid])) if confs.get(tid) else 0.0
		else:
			age_by_track[tid] = UNKNOWN_AGE
			conf_by_track[tid] = 0.0
	return age_by_track, conf_by_track


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _iou(a, b):
	"""IoU of two (x1,y1,x2,y2) boxes; 0 when either is None."""
	if a is None or b is None:
		return 0.0
	xa = max(a[0], b[0]); ya = max(a[1], b[1])
	xb = min(a[2], b[2]); yb = min(a[3], b[3])
	inter = max(0, xb - xa) * max(0, yb - ya)
	if inter <= 0:
		return 0.0
	area_a = (a[2] - a[0]) * (a[3] - a[1])
	area_b = (b[2] - b[0]) * (b[3] - b[1])
	union = area_a + area_b - inter
	return inter / union if union > 0 else 0.0


def _cosine(u, v, min_norm=_FALLBACK_SPEED_FLOOR_MS):
	"""Cosine similarity of two velocity vectors; 0 when either is too slow for
	its direction to mean anything."""
	nu = math.hypot(u[0], u[1]); nv = math.hypot(v[0], v[1])
	if nu < min_norm or nv < min_norm:
		return 0.0
	return (u[0] * v[0] + u[1] * v[1]) / (nu * nv)


def _heading_diff(u, v, min_norm=_FALLBACK_SPEED_FLOOR_MS):
	"""Absolute angular difference of two movement directions, in [0, pi]; 0 when
	either animal is too slow to have a heading."""
	nu = math.hypot(u[0], u[1]); nv = math.hypot(v[0], v[1])
	if nu < min_norm or nv < min_norm:
		return 0.0
	a = math.atan2(u[1], u[0]); b = math.atan2(v[1], v[0])
	d = abs(a - b) % (2 * math.pi)
	return d if d <= math.pi else (2 * math.pi - d)


# ---------------------------------------------------------------------------
# 4.1 Dyadic (pairwise) features
# ---------------------------------------------------------------------------

def compute_pairwise_features(track_data, age_by_track,
							  max_distance_m=15.0, contact_iou_thresh=0.05,
							  contact_dist_m=2.5):
	"""Per-frame dyadic features, in metres, for ordered pairs within
	max_distance_m of each other on the ground.

	Returns a list of dicts (one per ordered (source, target) pair per frame).
	The ordering encodes role (e.g. chaser first); both (A,B) and (B,A) are
	emitted. approach_rate is filled in a second pass per pair (time derivative
	of distance in m/s; negative = approaching).

	The distance gate is in METRES, so the same physical separation qualifies
	whether the drone flew at 15 m or 50 m — a pixel gate silently meant two
	different things at two altitudes.
	"""
	fps = track_data.get('fps', 30.0) or 30.0
	# Direction features are meaningless below this clip's own noise floor.
	floor = track_data.get('speed_floor_ms', _FALLBACK_SPEED_FLOOR_MS)
	rows = []
	for f in track_data['frames']:
		ids = track_data['present'][f]
		n = len(ids)
		if n < 2:
			continue
		for i in range(n):
			for j in range(n):
				if i == j:
					continue
				a, b = ids[i], ids[j]
				pa = track_data['pos'][(f, a)]; pb = track_data['pos'][(f, b)]
				distance_m = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
				if distance_m > max_distance_m:
					continue
				va = track_data['vel'].get((f, a), (0.0, 0.0))
				vb = track_data['vel'].get((f, b), (0.0, 0.0))
				speed_a = math.hypot(va[0], va[1])
				speed_b = math.hypot(vb[0], vb[1])
				box_a = track_data['box'].get((f, a))
				box_b = track_data['box'].get((f, b))
				iou = _iou(box_a, box_b) if track_data['has_bbox'] else 0.0
				in_contact = (iou > contact_iou_thresh) or (distance_m < contact_dist_m)
				rows.append({
					'frame': f,
					'source_id': a, 'target_id': b,
					'distance_m': distance_m,
					'speed_A': speed_a, 'speed_B': speed_b,
					'rel_speed': speed_a - speed_b,
					'speed_similarity': _cosine(va, vb, floor),
					'approach_rate': 0.0,  # filled below
					'heading_diff': _heading_diff(va, vb, floor),
					'bbox_iou': iou,
					'in_contact': bool(in_contact),
					'age_A': age_by_track.get(a, UNKNOWN_AGE),
					'age_B': age_by_track.get(b, UNKNOWN_AGE),
					'class_A_primary': track_data['prim'].get((f, a), ''),
					'class_A_secondary': track_data['sec'].get((f, a), ''),
					'class_B_primary': track_data['prim'].get((f, b), ''),
					'class_B_secondary': track_data['sec'].get((f, b), ''),
				})

	# Second pass: approach_rate = d(distance)/dt in m/s over consecutive
	# observed frames (frame gaps converted to seconds, so frame_skip and fps
	# never leak into the value).
	by_pair = defaultdict(list)
	for idx, r in enumerate(rows):
		by_pair[(r['source_id'], r['target_id'])].append(idx)
	for pair, idxs in by_pair.items():
		idxs.sort(key=lambda k: rows[k]['frame'])
		for k in range(1, len(idxs)):
			cur, prev = rows[idxs[k]], rows[idxs[k - 1]]
			dt = (cur['frame'] - prev['frame']) / fps
			if dt > 0:
				cur['approach_rate'] = (cur['distance_m'] - prev['distance_m']) / dt
	return rows


# ---------------------------------------------------------------------------
# 4.2 Group (whole co-present herd) features
# ---------------------------------------------------------------------------

def whole_herd_groups(track_data):
	"""Trivial grouping: every individual present in a frame is one group ('herd')
	for that frame — no spatial partitioning. Same {frame -> [(group_id, ids)]}
	shape as any other grouping passed to compute_group_features."""
	return {f: [('herd', track_data['present'][f])] for f in track_data['frames']}


def whole_herd_window_candidates(track_data, win, min_members=3):
	"""Non-overlapping windows of length `win` where at least `min_members` of
	the whole co-present herd were seen; members = most frequent per-frame
	presence within the window. Returns [(wstart, wend, ids), ...]."""
	if not track_data['frames']:
		return []
	counts_by_window = defaultdict(Counter)
	for f in track_data['frames']:
		wstart = (f // win) * win
		for t in track_data['present'][f]:
			counts_by_window[wstart][t] += 1
	out = []
	for wstart, cnt in counts_by_window.items():
		ids = [t for t, _ in cnt.most_common()]
		if len(ids) >= min_members:
			out.append((wstart, wstart + win - 1, ids))
	return out


def compute_group_features(track_data, groups):
	"""Per-frame, per-group fixed-size feature vector, in metres (see module
	docstring).

	`groups` maps frame -> [(group_id, [track_ids]), ...]; use whole_herd_groups()
	for the whole co-present herd, or an ad-hoc single-group mapping (e.g. a
	labelled segment's own ids) for other callers.

	Returns a list of dicts. Independent of N (the number of members). Shape
	statistics (cohesion, area, elongation) use the per-frame ground frame;
	centroid speed uses the STABILISED one, so a moving drone cannot make a
	standing herd look like it is travelling.
	"""
	fps = track_data.get('fps', 30.0) or 30.0
	floor = track_data.get('speed_floor_ms', _FALLBACK_SPEED_FLOOR_MS)
	pos_s = track_data.get('pos_s', {})
	# Pre-compute per-(frame, group) barycentre in the stabilised frame for
	# centroid-speed differencing.
	bary = {}
	for f in track_data['frames']:
		for gid, ids in groups.get(f, []):
			pts = [pos_s[(f, t)] for t in ids if (f, t) in pos_s]
			if pts:
				bary[(f, gid)] = np.mean(np.array(pts, dtype=float), axis=0)

	# Barycentre speed per group lineage over its observed frames.
	g_frames = defaultdict(list)
	for (f, gid) in bary:
		g_frames[gid].append(f)
	centroid_speed = {}
	for gid, fs in g_frames.items():
		fs = sorted(fs)
		for k, f in enumerate(fs):
			if k == 0:
				centroid_speed[(f, gid)] = 0.0
			else:
				dt = (f - fs[k - 1]) / fps
				if dt > 0:
					d = np.linalg.norm(bary[(f, gid)] - bary[(fs[k - 1], gid)])
					centroid_speed[(f, gid)] = float(d / dt)
				else:
					centroid_speed[(f, gid)] = 0.0

	rows = []
	for f in track_data['frames']:
		for gid, ids in groups.get(f, []):
			ids = [t for t in ids if (f, t) in track_data['pos']]
			n = len(ids)
			if n == 0:
				continue
			pts = np.array([track_data['pos'][(f, t)] for t in ids], dtype=float)
			vels = np.array([track_data['vel'].get((f, t), (0.0, 0.0)) for t in ids], dtype=float)
			speeds = np.linalg.norm(vels, axis=1)

			# Polarisation: magnitude of the mean unit velocity, over the members
			# that are actually moving. Counting standing animals would turn
			# detection jitter into a confident collective heading — a grazing
			# herd scored 0.87 before this floor.
			norms = np.linalg.norm(vels, axis=1)
			moving = norms > floor
			if moving.any():
				units = vels[moving] / norms[moving][:, None]
				polarisation = float(np.linalg.norm(units.mean(axis=0)))
			else:
				polarisation = 0.0

			centroid = pts.mean(axis=0)
			disp = np.linalg.norm(pts - centroid, axis=1)
			cohesion = float(disp.std()) if n > 1 else 0.0

			area = _hull_area(pts)

			# Behavioural synchrony: fraction sharing the modal primary class.
			classes = [track_data['prim'].get((f, t), '') for t in ids]
			classes = [c for c in classes if c]
			if classes:
				modal = Counter(classes).most_common(1)[0][1]
				synchrony = modal / len(ids)
			else:
				synchrony = 0.0

			elongation = _elongation(pts)

			rows.append({
				'frame': f, 'group_id': gid, 'n_members': n,
				'mean_speed': float(speeds.mean()),
				'polarisation': polarisation,
				'cohesion': cohesion,
				'area': area,
				'centroid_speed': centroid_speed.get((f, gid), 0.0),
				'synchrony': synchrony,
				'elongation': elongation,
			})
	return rows


def _hull_area(pts):
	"""Convex-hull area of a set of 2-D points (0 for <3 or degenerate)."""
	if len(pts) < 3:
		return 0.0
	try:
		from scipy.spatial import ConvexHull
		return float(ConvexHull(pts).volume)  # 'volume' is area in 2-D
	except Exception:
		return 0.0


def _elongation(pts):
	"""Ratio of principal axes (PCA of points). 1 = round, >1 = elongated/column."""
	if len(pts) < 2:
		return 1.0
	# Cap the ratio so a (near-)collinear formation yields a large but FINITE
	# value rather than inf, which would break the feature vector / CSV import.
	_ELONG_CAP = 100.0
	cov = np.cov(pts.T)
	try:
		ev = np.linalg.eigvalsh(cov)
		ev = np.sort(ev)[::-1]
		if ev[1] <= 1e-9:
			return _ELONG_CAP if ev[0] > 1e-9 else 1.0
		return float(min(math.sqrt(ev[0] / ev[1]), _ELONG_CAP))
	except Exception:
		return 1.0


# ---------------------------------------------------------------------------
# 4.3 Window aggregation (fixed-size vector for the TASK 7 model)
# ---------------------------------------------------------------------------

_SCALAR_DYADIC = ['distance_m', 'speed_A', 'speed_B', 'rel_speed',
				  'speed_similarity', 'approach_rate', 'heading_diff', 'bbox_iou']
_CLASS_KEYS = ['class_A_primary', 'class_A_secondary',
			   'class_B_primary', 'class_B_secondary']


def aggregate_window(features, start_frame, end_frame):
	"""Aggregate per-frame feature dicts over [start_frame, end_frame] into a
	fixed-size vector (mean/std/min/max of scalars + a normalised bag of YOLO
	primary AND secondary labels). The YOLO labels are kept (not dropped) so the
	TASK 7 model receives them alongside the kinematic features.
	"""
	win = [r for r in features if start_frame <= r['frame'] <= end_frame]
	out = {'n_frames': len(win)}
	scalar_keys = [k for k in _SCALAR_DYADIC if win and k in win[0]]
	for k in scalar_keys:
		vals = np.array([float(r.get(k, 0.0)) for r in win], dtype=float) if win else np.array([])
		if vals.size:
			out[f'{k}_mean'] = float(vals.mean())
			out[f'{k}_std'] = float(vals.std())
			out[f'{k}_min'] = float(vals.min())
			out[f'{k}_max'] = float(vals.max())
		else:
			out[f'{k}_mean'] = out[f'{k}_std'] = out[f'{k}_min'] = out[f'{k}_max'] = 0.0
	if win and 'in_contact' in win[0]:
		out['contact_fraction'] = float(np.mean([1.0 if r['in_contact'] else 0.0 for r in win]))
	# Bag-of-labels for the YOLO primary + secondary classes (normalised).
	bag = Counter()
	for r in win:
		for ck in _CLASS_KEYS:
			c = r.get(ck, '')
			if c:
				bag[f'{ck}={c}'] += 1
	total = sum(bag.values())
	for key, cnt in bag.items():
		out[f'label_{key}'] = cnt / total if total else 0.0
	return out


# ---------------------------------------------------------------------------
# 4.4 Interaction graph
# ---------------------------------------------------------------------------

def _dominant(values):
	"""Most common non-empty value, or ''."""
	vals = [v for v in values if v]
	return Counter(vals).most_common(1)[0][0] if vals else ''


def _typical_step(frames):
	"""Median gap between consecutive processed frames (handles frame_skip)."""
	if len(frames) < 2:
		return 1
	diffs = np.diff(sorted(set(frames)))
	return int(np.median(diffs)) if len(diffs) else 1


def _id_key(tid):
	"""Sort key giving numeric-looking track ids their natural order."""
	return (len(str(tid)), str(tid))


def _canonical_pair(a, b):
	"""Unordered pair as an ordered tuple, so (A,B) and (B,A) collapse to one."""
	return (a, b) if _id_key(a) <= _id_key(b) else (b, a)


def build_interaction_graph(pairwise, nodes, track_data, granularity="per_dyad",
							weight_metric="sri", min_duration_frames=10):
	"""Aggregate per-frame dyadic features into interaction-graph edges.

	The deterministic layer is UNDIRECTED. Every metric it can compute
	(separation, contact, speed similarity, approach rate) is symmetric, so
	emitting both (A,B) and (B,A) doubled the file while duplicating each value
	and dressing the result up as a directed graph it is not. Direction only
	becomes meaningful once the complex model names an actor, which it does by
	filling interaction_type/actor_id afterwards.

	pairwise   : per-frame ORDERED-pair feature dicts (the model needs role, so
	             compute_pairwise_features still emits both directions).
	nodes      : node-attribute dicts (one per track_id).
	track_data : used for co-presence (SRI denominator) and fps.
	Returns (edges_rows, nodes_rows).
	"""
	granularity = normalise_granularity(granularity)
	step = _typical_step(track_data['frames'])
	fps = track_data.get('fps', 30.0) or 30.0
	min_obs = max(1, min_duration_frames)

	# Collapse the mirrored orderings onto one canonical unordered pair.
	by_pair = defaultdict(list)
	for r in pairwise:
		key = _canonical_pair(r['source_id'], r['target_id'])
		if (r['source_id'], r['target_id']) == key:
			by_pair[key].append(r)
	for k in by_pair:
		by_pair[k].sort(key=lambda r: r['frame'])

	co_present = _co_presence_counts(track_data, by_pair.keys())

	if granularity == 'per_frame':
		# min_duration_frames applies here too: a dyad seen for three frames is
		# noise at every granularity, not only when episodes are cut.
		edges = []
		for key, rs in by_pair.items():
			if len(rs) < min_obs:
				continue
			for r in rs:
				edges.append({
					'frame': r['frame'], 'source_id': key[0], 'target_id': key[1],
					'directed': 'false', 'weight': 1.0,
					'distance_m': round(r['distance_m'], 3),
					'in_contact': int(bool(r['in_contact'])),
					'speed_similarity': round(r['speed_similarity'], 4),
					'approach_rate': round(r['approach_rate'], 4),
					'class_source_primary': r['class_A_primary'],
					'class_source_secondary': r['class_A_secondary'],
					'class_target_primary': r['class_B_primary'],
					'class_target_secondary': r['class_B_secondary'],
				})
	else:
		episodes = []  # (pair, [rows])
		for key, rs in by_pair.items():
			if granularity == 'per_segment':
				cur = [rs[0]]
				for r in rs[1:]:
					if r['frame'] - cur[-1]['frame'] > 2 * step:
						episodes.append((key, cur)); cur = [r]
					else:
						cur.append(r)
				episodes.append((key, cur))
			else:  # per_dyad: one row per dyad, spanning the whole clip
				episodes.append((key, rs))

		edges = []
		for (key, rs) in episodes:
			n_obs = len(rs)
			if n_obs < min_obs:
				continue
			dm = np.array([r['distance_m'] for r in rs], dtype=float)
			n_co = co_present.get(key, n_obs)
			edges.append({
				'frame_start': rs[0]['frame'], 'frame_end': rs[-1]['frame'],
				'source_id': key[0], 'target_id': key[1], 'directed': 'false',
				'interaction_type': '', 'actor_id': '',
				'weight': 0.0,  # filled below
				'n_frames_observed': n_obs,
				'n_frames_co_present': n_co,
				'duration_s': round(n_obs / fps, 3),
				'mean_distance_m': round(float(dm.mean()), 3),
				'min_distance_m': round(float(dm.min()), 3),
				'contact_fraction': round(float(np.mean([1.0 if r['in_contact'] else 0.0 for r in rs])), 4),
				'mean_speed_similarity': round(float(np.mean([r['speed_similarity'] for r in rs])), 4),
				'mean_approach_rate': round(float(np.mean([r['approach_rate'] for r in rs])), 4),
				'class_source_primary': _dominant([r['class_A_primary'] for r in rs]),
				'class_source_secondary': _dominant([r['class_A_secondary'] for r in rs]),
				'class_target_primary': _dominant([r['class_B_primary'] for r in rs]),
				'class_target_secondary': _dominant([r['class_B_secondary'] for r in rs]),
			})
		_assign_weights(edges, weight_metric)

	_graph_summary(edges, granularity)
	return edges, nodes


def _co_presence_counts(track_data, pairs):
	"""Frames in which BOTH members of a pair were tracked — the denominator that
	makes an association index comparable between clips of different lengths."""
	frame_sets = {tid: set(fs) for tid, fs in track_data['id_frames'].items()}
	out = {}
	for (a, b) in pairs:
		out[(a, b)] = len(frame_sets.get(a, set()) & frame_sets.get(b, set()))
	return out


def _assign_weights(edges, weight_metric):
	"""Set the 'weight' column: sri | duration_s | proximity_m.

	All three are ABSOLUTE quantities. The previous 'combined' metric min-max
	scaled the weights WITHIN each video, so an edge weight of 0.8 meant
	"strong for this clip" and nothing else — two videos could not be compared,
	which defeats the point of building the network per video.
	"""
	if not edges:
		return
	for e in edges:
		if weight_metric == 'duration_s':
			w = e['duration_s']
		elif weight_metric == 'proximity_m':
			w = 1.0 / e['mean_distance_m'] if e['mean_distance_m'] > 1e-9 else 0.0
		else:  # sri (default): simple ratio index, bounded [0, 1]
			n_co = e.get('n_frames_co_present') or 0
			w = (e['n_frames_observed'] / n_co) if n_co > 0 else 0.0
			w = min(w, 1.0)
		e['weight'] = round(float(w), 4)


def _graph_summary(edges, granularity):
	"""Print connected-components / community / centrality summaries (networkx)."""
	if granularity == 'per_frame' or not edges:
		return
	if not _NX_AVAILABLE:
		print("  networkx not available — skipping centrality/community summary "
			  "(edges/nodes still written). Install with: pip install networkx")
		return
	g = nx.Graph()
	for e in edges:
		g.add_edge(e['source_id'], e['target_id'], weight=e.get('weight', 1.0))
	if g.number_of_nodes() == 0:
		return
	ug = g
	n_comp = nx.number_connected_components(ug)
	deg = nx.degree_centrality(g)
	try:
		bet = nx.betweenness_centrality(g, weight='weight')
	except Exception:
		bet = {}
	top_deg = max(deg, key=deg.get) if deg else None
	top_bet = max(bet, key=bet.get) if bet else None
	try:
		communities = list(nx.algorithms.community.greedy_modularity_communities(ug))
		n_comm = len(communities)
	except Exception:
		n_comm = None
	msg = (f"  Interaction graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, "
		   f"{n_comp} connected component(s)")
	if n_comm is not None:
		msg += f", {n_comm} communit(y/ies)"
	if top_deg is not None:
		msg += f"; most central (degree)={top_deg}"
	if top_bet is not None:
		msg += f", (betweenness)={top_bet}"
	print(msg)


# ---------------------------------------------------------------------------
# Node attributes
# ---------------------------------------------------------------------------

def build_nodes(track_data, age_by_track, age_conf_by_track):
	"""Build per-track node-attribute rows for the nodes CSV."""
	total_span = (max(track_data['frames']) - min(track_data['frames']) + 1) \
		if track_data['frames'] else 1

	nodes = []
	for tid, frames in track_data['id_frames'].items():
		prim = [track_data['prim'].get((f, tid), '') for f in frames]
		sec = [track_data['sec'].get((f, tid), '') for f in frames]
		nodes.append({
			'track_id': tid,
			'first_frame': min(frames),
			'last_frame': max(frames),
			'age_class': age_by_track.get(tid, UNKNOWN_AGE),
			'age_conf_mean': round(age_conf_by_track.get(tid, 0.0), 4),
			'primary_class_dominant': _dominant(prim),
			'secondary_class_dominant': _dominant(sec),
			'presence_ratio': round(len(frames) / total_span, 4) if total_span else 0.0,
		})
	nodes.sort(key=lambda n: _id_key(n['track_id']))
	return nodes


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

_EPISODE_COLS = ['frame_start', 'frame_end', 'source_id', 'target_id', 'directed',
				 'interaction_type', 'actor_id', 'weight',
				 'n_frames_observed', 'n_frames_co_present', 'duration_s',
				 'mean_distance_m', 'min_distance_m', 'contact_fraction',
				 'mean_speed_similarity', 'mean_approach_rate',
				 'class_source_primary', 'class_source_secondary',
				 'class_target_primary', 'class_target_secondary']
# n_frames_observed / n_frames_co_present / duration_s are written whatever the
# chosen weight metric, so any association index can be recomputed in R without
# re-running the pipeline.
_EDGE_COLS = {
	'per_dyad': _EPISODE_COLS,
	'per_segment': _EPISODE_COLS,
	'per_frame': ['frame', 'source_id', 'target_id', 'directed', 'weight',
				  'distance_m',
				  'in_contact', 'speed_similarity', 'approach_rate',
				  'class_source_primary', 'class_source_secondary',
				  'class_target_primary', 'class_target_secondary'],
}
_NODE_COLS = ['track_id', 'first_frame', 'last_frame',
			  'age_class', 'age_conf_mean', 'primary_class_dominant',
			  'secondary_class_dominant', 'presence_ratio']


def write_interaction_graph(edges_rows, nodes_rows, edges_path, nodes_path,
							granularity="per_dyad"):
	"""Write (overwrite) the edges and nodes CSVs. Always truncates the file so a
	changed granularity/weight never leaves stale rows behind."""
	os.makedirs(os.path.dirname(os.path.abspath(edges_path)), exist_ok=True)
	granularity = normalise_granularity(granularity)
	edge_cols = _EDGE_COLS.get(granularity, _EDGE_COLS['per_dyad'])
	with open(edges_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=edge_cols, extrasaction='ignore')
		w.writeheader()
		w.writerows(edges_rows)
	with open(nodes_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=_NODE_COLS, extrasaction='ignore')
		w.writeheader()
		w.writerows(nodes_rows)


# ---------------------------------------------------------------------------
# Batch / project entry point
# ---------------------------------------------------------------------------

def resolve_output_dir(config_path):
	"""Absolute output folder of a project, from its INI."""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	project_dir = os.path.dirname(os.path.abspath(config_path))
	raw = cfg['DEFAULT'].get('output_dir', 'output')
	return raw if os.path.isabs(raw) else os.path.join(project_dir, raw)


def find_metric_csv(output_dir, stem):
	"""The metric tracking CSV for a video stem, or None. This is the ONLY input
	the interaction layer accepts — see the module docstring on why there is no
	pixel-space fallback."""
	path = os.path.join(output_dir, stem + '_tracking_metric.csv')
	return path if os.path.exists(path) else None


def run_complex_features(config_path):
	"""Build the interaction graph for every metric tracking CSV in the output
	folder.

	Requires <video>_tracking_metric.csv; writes <video>_interaction_edges.csv
	and <video>_interaction_nodes.csv (overwritten each run / parameter change).
	The metric stage itself already consumes the most-processed CSV (stitched >
	drone-corrected > raw), so the metric file carries the best identities
	available — there is no separate fallback chain to maintain here.
	"""
	config_path = os.path.abspath(config_path)
	params = load_complex_config(config_path)
	output_dir = resolve_output_dir(config_path)

	jobs = {os.path.basename(p)[:-len('_tracking_metric.csv')]: p
			for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking_metric.csv')))}

	if not jobs:
		# Name what IS there, so "nothing happened" is never a mystery.
		other = glob.glob(os.path.join(output_dir, '*_tracking*.csv'))
		if other:
			print(f"Complex features: found {len(other)} tracking CSV(s) in {output_dir} but "
				  f"none with metric geometry. The interaction graph works in metres — set "
				  f"metric_enabled = true, give each video its .flightlog.csv, and re-run so "
				  f"*_tracking_metric.csv is produced.")
		else:
			print(f"Complex features: no tracking CSVs found in {output_dir}")
		return

	video_index = None
	try:
		from behaveai_drone.metric_geometry import build_video_index
		video_index = build_video_index(config_path)
	except Exception:
		pass

	print(f"Complex features: processing {len(jobs)} video(s) "
		  f"(granularity={params['edge_granularity']}, weight={params['weight_metric']})...")
	for stem, csv_path in sorted(jobs.items()):
		try:
			_process_one(stem, csv_path, output_dir, params, config_path, video_index)
		except MissingMetricError as e:
			print(f"  SKIPPED {stem}: {e}")
		except Exception as e:
			import traceback
			print(f"  ERROR on {os.path.basename(csv_path)}: {e}")
			traceback.print_exc()
	print("Complex features complete.")


def video_fps(config_path, stem, video_index=None):
	"""Frames per second for a video stem; 30.0 when it cannot be determined.
	Delegates to the metric stage so one module owns where a project's videos
	live."""
	try:
		from behaveai_drone.metric_geometry import video_fps_for_stem
		return video_fps_for_stem(config_path, stem, default=30.0, video_index=video_index)
	except Exception:
		return 30.0


def _process_one(stem, csv_path, output_dir, params, config_path, video_index=None):
	"""Run the full TASK 4 pipeline for a single video."""
	fps = video_fps(config_path, stem, video_index)
	track_data = load_tracking_csv(csv_path, fps=fps)
	if not track_data['frames']:
		print(f"  {stem}: no rows, skipping.")
		return

	age_by_track, age_conf_by_track = track_age_classes(track_data)
	counts = Counter(age_by_track.values())
	print(f"  {stem}: {len(track_data['id_frames'])} track(s) at {fps:.2f} fps "
		  f"(age: {', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}).")
	# Report the measured noise floor: it is a property of THIS clip's footage and
	# belongs in the methods section, not buried in a constant.
	sigma = track_data.get('pos_noise_m', float('nan'))
	if np.isfinite(sigma):
		print(f"  {stem}: position noise {sigma * 100:.1f} cm -> speeds below "
			  f"{track_data['speed_floor_ms']:.2f} m/s are indistinguishable from "
			  f"jitter and are treated as stationary.")
	else:
		print(f"  {stem}: too few contiguous tracks to measure the noise floor — "
			  f"using the {_FALLBACK_SPEED_FLOOR_MS:.2f} m/s fallback.")
	if counts.get(UNKNOWN_AGE) == len(age_by_track):
		print(f"  {stem}: NOTE — no individual carries an age label. Train the age "
			  f"classifier (annot_age_crop/) if foal/adult matters for this analysis.")

	pairwise = compute_pairwise_features(
		track_data, age_by_track,
		max_distance_m=params['max_interaction_distance_m'],
		contact_iou_thresh=params['contact_iou_thresh'],
		contact_dist_m=params['contact_dist_m'])

	# Group features (whole co-present herd per frame) computed for downstream
	# use (TASK 7) and summarised here.
	groups = whole_herd_groups(track_data)
	gfeat = compute_group_features(track_data, groups)
	if gfeat:
		pol = np.mean([g['polarisation'] for g in gfeat])
		print(f"  {stem}: {len(gfeat)} group-feature rows (mean polarisation={pol:.2f}).")

	nodes = build_nodes(track_data, age_by_track, age_conf_by_track)
	edges, nodes = build_interaction_graph(
		pairwise, nodes, track_data,
		granularity=params['edge_granularity'],
		weight_metric=params['weight_metric'],
		min_duration_frames=params['min_duration_frames'])

	edges_path = os.path.join(output_dir, stem + '_interaction_edges.csv')
	nodes_path = os.path.join(output_dir, stem + '_interaction_nodes.csv')
	write_interaction_graph(edges, nodes, edges_path, nodes_path,
							granularity=params['edge_granularity'])
	print(f"  {stem}: wrote {len(edges)} edge row(s), {len(nodes)} node(s).")


def _main():
	parser = argparse.ArgumentParser(
		description="Compute dyadic/group features and the networkx interaction graph.")
	parser.add_argument('target', help="Project directory or BehaveAI_settings.ini.")
	args = parser.parse_args()
	target = os.path.abspath(args.target)
	ini = os.path.join(target, 'BehaveAI_settings.ini') if os.path.isdir(target) else target
	if not os.path.exists(ini):
		print(f"Settings file not found: {ini}")
		sys.exit(1)
	run_complex_features(ini)


if __name__ == '__main__':
	_main()
