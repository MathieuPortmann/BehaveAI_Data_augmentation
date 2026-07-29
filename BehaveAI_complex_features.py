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

Everything stays in image space (no metric conversion). All sizes/speeds are
normalised by a reference body length body_len_ref = robust median box diagonal
of ADULT-sized horses (clarification 3), never each horse's own size.

Inputs (per video, from the project output folder):
  - <video>_tracking_corrected.csv  (preferred; falls back to <video>_tracking.csv)

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_complex_config(config_path):
	"""Read the TASK 4 parameters from a BehaveAI INI (DEFAULT section)."""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	return {
		'max_interaction_distance': float(d.get('complex_max_interaction_distance', '400')),
		'min_duration_frames':      int(float(d.get('complex_min_duration_frames', '10'))),
		'contact_iou_thresh':       float(d.get('complex_contact_iou_thresh', '0.05')),
		'contact_dist_bodylen':     float(d.get('complex_contact_dist_bodylen', '1.5')),
		'window_frames':            int(float(d.get('complex_window_frames', '30'))),
		'edge_granularity':         d.get('interaction_edge_granularity', 'per_interaction'),
		'weight_metric':            d.get('interaction_weight_metric', 'duration'),
		'body_len_ref_scope':       d.get('body_len_ref_scope', 'video'),
		'foal_size_ratio':          float(d.get('foal_size_ratio_thresh', '0.7')),
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


def load_tracking_csv(csv_path):
	"""Load a (corrected) tracking CSV into a track_data dict.

	Prefers the drone-corrected columns (x_corrected, y_corrected and, when
	present, vx_corrected/vy_corrected); otherwise falls back to raw (x, y) with
	a one-time warning and velocities computed by finite differences. Detects the
	bbox columns; when absent, box/overlap features are disabled (warn once).
	"""
	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fieldnames = list(reader.fieldnames or [])
		rows = [dict(r) for r in reader]

	used_corrected = 'x_corrected' in fieldnames and 'y_corrected' in fieldnames
	has_vel = 'vx_corrected' in fieldnames and 'vy_corrected' in fieldnames
	has_bbox = all(c in fieldnames for c in ('x1', 'y1', 'x2', 'y2'))
	has_metric = 'X_m' in fieldnames and 'Y_m' in fieldnames
	if not used_corrected:
		print(f"  NOTE: {os.path.basename(csv_path)} has no corrected columns — using raw (x, y).")
	if not has_bbox:
		print(f"  WARNING: {os.path.basename(csv_path)} has no bbox columns — "
			  f"overlap/size features disabled.")

	present = defaultdict(list)
	pos, vel, box, diag, prim, sec = {}, {}, {}, {}, {}, {}
	pos_m = {}                                        # metric ground coords (X_m, Y_m) when present
	id_frames = defaultdict(list)

	for r in rows:
		try:
			frame = int(r['frame'])
			tid = str(r['id'])
			if used_corrected and r.get('x_corrected', '') != '':
				x = float(r['x_corrected']); y = float(r['y_corrected'])
			else:
				x = float(r['x']); y = float(r['y'])
		except (ValueError, KeyError, TypeError):
			continue
		if (frame, tid) in pos:
			continue
		present[frame].append(tid)
		pos[(frame, tid)] = (x, y)
		id_frames[tid].append(frame)
		prim[(frame, tid)] = _best_class(r, 'primary')
		sec[(frame, tid)] = _best_class(r, 'secondary')
		if has_vel and r.get('vx_corrected', '') != '':
			try:
				vel[(frame, tid)] = (float(r['vx_corrected']), float(r['vy_corrected']))
			except (ValueError, TypeError):
				pass
		if has_bbox:
			try:
				x1, y1, x2, y2 = int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2'])
				if x2 > x1 and y2 > y1:
					box[(frame, tid)] = (x1, y1, x2, y2)
					diag[(frame, tid)] = float(np.hypot(x2 - x1, y2 - y1))
			except (ValueError, TypeError):
				pass
		if has_metric and r.get('X_m', '') != '' and r.get('metric_quality', '') != 'none':
			try:
				pos_m[(frame, tid)] = (float(r['X_m']), float(r['Y_m']))
			except (ValueError, TypeError):
				pass

	# Compute velocities by finite differences when corrected velocities are absent.
	if not has_vel:
		for tid, frames in id_frames.items():
			fs = sorted(frames)
			xs = np.array([pos[(f, tid)][0] for f in fs], dtype=float)
			ys = np.array([pos[(f, tid)][1] for f in fs], dtype=float)
			xs = _smooth(xs); ys = _smooth(ys)
			fa = np.array(fs, dtype=float)
			if len(fs) >= 2 and np.all(np.diff(fa) > 0):
				vx = np.gradient(xs, fa); vy = np.gradient(ys, fa)
			else:
				vx = np.zeros(len(fs)); vy = np.zeros(len(fs))
			for i, f in enumerate(fs):
				vel[(f, tid)] = (float(vx[i]), float(vy[i]))

	return {
		'frames': sorted(present.keys()),
		'present': present,
		'pos': pos, 'vel': vel, 'box': box, 'diag': diag,
		'prim': prim, 'sec': sec,
		'pos_m': pos_m,
		'id_frames': {t: sorted(fs) for t, fs in id_frames.items()},
		'has_bbox': has_bbox, 'used_corrected': used_corrected,
		'has_metric': has_metric and bool(pos_m),
	}


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
# Body-length reference (robust adult median) + per-id size ratio / foal flag
# ---------------------------------------------------------------------------

def compute_body_len_ref(track_data, scope="video", foal_ratio=0.7):
	"""Return (body_len_ref, size_ratio, is_foal).

	body_len_ref is the robust median box diagonal of adult-sized horses; foals
	(per-track median diagonal below foal_ratio * initial median) are excluded so
	they do not shrink the reference. size_ratio[id] = body_len_i / body_len_ref;
	is_foal[id] = size_ratio < foal_ratio. When boxes are unavailable, falls back
	to a nearest-neighbour scene scale (sizes then default to 1.0, no foals).
	"""
	diag = track_data['diag']
	per_track = defaultdict(list)
	for (f, tid), d in diag.items():
		if d and d > 0:
			per_track[tid].append(d)
	track_med = {tid: float(np.median(ds)) for tid, ds in per_track.items() if ds}

	if not track_med:
		# No boxes at all: use a nearest-neighbour length scale; no foal info.
		ref = _nn_scale(track_data)
		return ref, {tid: 1.0 for tid in track_data['id_frames']}, \
			{tid: False for tid in track_data['id_frames']}

	arr = np.array(list(track_med.values()), dtype=float)
	initial = float(np.median(arr))
	adults = arr[arr >= foal_ratio * initial]
	ref = float(np.median(adults)) if adults.size else float(np.median(arr))
	if ref <= 0:
		ref = _nn_scale(track_data)

	size_ratio = {}
	is_foal = {}
	for tid in track_data['id_frames']:
		d = track_med.get(tid)
		if d and ref > 0:
			size_ratio[tid] = d / ref
			is_foal[tid] = (d / ref) < foal_ratio
		else:
			size_ratio[tid] = 1.0
			is_foal[tid] = False
	return ref, size_ratio, is_foal


def _nn_scale(track_data):
	"""Median nearest-neighbour distance across frames (fallback length unit)."""
	from scipy.spatial import cKDTree
	nn = []
	for f in track_data['frames']:
		ids = track_data['present'][f]
		if len(ids) >= 2:
			pts = np.array([track_data['pos'][(f, t)] for t in ids], dtype=float)
			tree = cKDTree(pts)
			dd, _ = tree.query(pts, k=2)
			nn.extend(dd[:, 1].tolist())
	return float(np.median(nn)) if nn else 50.0


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


def _cosine(u, v):
	"""Cosine similarity of two 2-vectors (0 if either is ~zero)."""
	nu = math.hypot(u[0], u[1]); nv = math.hypot(v[0], v[1])
	if nu < 1e-9 or nv < 1e-9:
		return 0.0
	return (u[0] * v[0] + u[1] * v[1]) / (nu * nv)


def _heading_diff(u, v):
	"""Absolute angular difference of two movement directions, in [0, pi]."""
	nu = math.hypot(u[0], u[1]); nv = math.hypot(v[0], v[1])
	if nu < 1e-9 or nv < 1e-9:
		return 0.0
	a = math.atan2(u[1], u[0]); b = math.atan2(v[1], v[0])
	d = abs(a - b) % (2 * math.pi)
	return d if d <= math.pi else (2 * math.pi - d)


# ---------------------------------------------------------------------------
# 4.1 Dyadic (pairwise) features
# ---------------------------------------------------------------------------

def compute_pairwise_features(track_data, body_len_ref, size_ratio,
							  max_distance=400.0, contact_iou_thresh=0.05,
							  contact_dist_bodylen=1.5):
	"""Per-frame dyadic features for ordered pairs within max_distance.

	Returns a list of dicts (one per ordered (source, target) pair per frame).
	The ordering encodes role (e.g. chaser first); both (A,B) and (B,A) are
	emitted. approach_rate is filled in a second pass per pair (time derivative
	of distance; negative = approaching). Distances are normalised by
	body_len_ref (NOT the pair mean), so mare-foal contact is handled correctly.
	"""
	ref = body_len_ref if body_len_ref and body_len_ref > 0 else 1.0
	pos_m = track_data.get('pos_m', {})
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
				dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
				if dist > max_distance:
					continue
				# Real-world ground distance (metres) when both endpoints have a
				# metric position; None otherwise (pixel features are unaffected).
				pma, pmb = pos_m.get((f, a)), pos_m.get((f, b))
				distance_m = (math.hypot(pma[0] - pmb[0], pma[1] - pmb[1])
							  if pma is not None and pmb is not None else None)
				va = track_data['vel'].get((f, a), (0.0, 0.0))
				vb = track_data['vel'].get((f, b), (0.0, 0.0))
				speed_a = math.hypot(va[0], va[1]) / ref
				speed_b = math.hypot(vb[0], vb[1]) / ref
				box_a = track_data['box'].get((f, a))
				box_b = track_data['box'].get((f, b))
				iou = _iou(box_a, box_b) if track_data['has_bbox'] else 0.0
				dist_bl = dist / ref
				in_contact = (iou > contact_iou_thresh) or (dist_bl < contact_dist_bodylen)
				rows.append({
					'frame': f,
					'source_id': a, 'target_id': b,
					'distance': dist,
					'distance_bodylen': dist_bl,
					'distance_m': distance_m,
					'speed_A': speed_a, 'speed_B': speed_b,
					'rel_speed': speed_a - speed_b,
					'speed_similarity': _cosine(va, vb),
					'approach_rate': 0.0,  # filled below
					'heading_diff': _heading_diff(va, vb),
					'bbox_iou': iou,
					'in_contact': bool(in_contact),
					'size_ratio_A': size_ratio.get(a, 1.0),
					'size_ratio_B': size_ratio.get(b, 1.0),
					'class_A_primary': track_data['prim'].get((f, a), ''),
					'class_A_secondary': track_data['sec'].get((f, a), ''),
					'class_B_primary': track_data['prim'].get((f, b), ''),
					'class_B_secondary': track_data['sec'].get((f, b), ''),
				})

	# Second pass: approach_rate = d(distance)/dt over consecutive observed frames.
	by_pair = defaultdict(list)
	for idx, r in enumerate(rows):
		by_pair[(r['source_id'], r['target_id'])].append(idx)
	for pair, idxs in by_pair.items():
		idxs.sort(key=lambda k: rows[k]['frame'])
		for k in range(1, len(idxs)):
			cur, prev = rows[idxs[k]], rows[idxs[k - 1]]
			df = cur['frame'] - prev['frame']
			if df > 0:
				cur['approach_rate'] = (cur['distance'] - prev['distance']) / df
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


def compute_group_features(track_data, groups, body_len_ref):
	"""Per-frame, per-group fixed-size feature vector (see module docstring).

	`groups` maps frame -> [(group_id, [track_ids]), ...]; use whole_herd_groups()
	for the whole co-present herd, or an ad-hoc single-group mapping (e.g. a
	labelled segment's own ids) for other callers.

	Returns a list of dicts. Independent of N (the number of members). Centroid
	speed is derived from the barycentre trajectory over consecutive frames.
	"""
	ref = body_len_ref if body_len_ref and body_len_ref > 0 else 1.0
	# Pre-compute per-(frame, group) barycentre for centroid-speed differencing.
	bary = {}
	for f in track_data['frames']:
		for gid, ids in groups.get(f, []):
			pts = [track_data['pos'][(f, t)] for t in ids if (f, t) in track_data['pos']]
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
				dfr = f - fs[k - 1]
				if dfr > 0:
					d = np.linalg.norm(bary[(f, gid)] - bary[(fs[k - 1], gid)])
					centroid_speed[(f, gid)] = float(d / dfr / ref)
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
			speeds = np.linalg.norm(vels, axis=1) / ref

			# Polarisation: magnitude of the mean unit velocity.
			norms = np.linalg.norm(vels, axis=1)
			moving = norms > 1e-9
			if moving.any():
				units = vels[moving] / norms[moving][:, None]
				polarisation = float(np.linalg.norm(units.mean(axis=0)))
			else:
				polarisation = 0.0

			centroid = pts.mean(axis=0)
			disp = np.linalg.norm(pts - centroid, axis=1) / ref
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

_SCALAR_DYADIC = ['distance_bodylen', 'speed_A', 'speed_B', 'rel_speed',
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


def build_interaction_graph(pairwise, nodes, granularity="per_interaction",
							weight_metric="duration", min_duration_frames=10,
							all_frames=None):
	"""Aggregate per-frame dyadic features into interaction-graph edges.

	pairwise : list of per-frame ordered-pair feature dicts (compute_pairwise_features).
	nodes    : list of node-attribute dicts (one per track_id).
	Returns (edges_rows, nodes_rows). networkx (when present) is used to print
	connected-components / community / centrality summaries; node columns follow
	the spec and do not include centralities.
	"""
	step = _typical_step(all_frames if all_frames is not None
						  else [r['frame'] for r in pairwise])

	by_pair = defaultdict(list)
	for r in pairwise:
		by_pair[(r['source_id'], r['target_id'])].append(r)
	for k in by_pair:
		by_pair[k].sort(key=lambda r: r['frame'])

	if granularity == 'per_frame':
		edges = [{
			'frame': r['frame'], 'source_id': r['source_id'], 'target_id': r['target_id'],
			'weight': 1.0,
			'distance_bodylen': round(r['distance_bodylen'], 4),
			'distance_m': ('' if r.get('distance_m') is None else round(r['distance_m'], 3)),
			'in_contact': int(bool(r['in_contact'])),
			'speed_similarity': round(r['speed_similarity'], 4),
			'approach_rate': round(r['approach_rate'], 4),
			'class_source_primary': r['class_A_primary'],
			'class_source_secondary': r['class_A_secondary'],
			'class_target_primary': r['class_B_primary'],
			'class_target_secondary': r['class_B_secondary'],
		} for r in pairwise]
	else:
		episodes = []  # (source, target, [rows])
		for (s, t), rs in by_pair.items():
			if granularity == 'per_segment':
				cur = [rs[0]]
				for r in rs[1:]:
					if r['frame'] - cur[-1]['frame'] > 2 * step:
						episodes.append((s, t, cur)); cur = [r]
					else:
						cur.append(r)
				episodes.append((s, t, cur))
			else:  # per_interaction: one episode spanning the whole video
				episodes.append((s, t, rs))

		edges = []
		for (s, t, rs) in episodes:
			n_obs = len(rs)
			if granularity == 'per_segment' and n_obs < max(1, min_duration_frames):
				continue
			dbl = np.array([r['distance_bodylen'] for r in rs], dtype=float)
			edges.append({
				'frame_start': rs[0]['frame'], 'frame_end': rs[-1]['frame'],
				'source_id': s, 'target_id': t, 'directed': 'true',
				'interaction_type': '',
				'weight': 0.0,  # filled after normalisation
				'n_frames_observed': n_obs,
				'mean_distance_bodylen': round(float(dbl.mean()), 4),
				'min_distance_bodylen': round(float(dbl.min()), 4),
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


def _assign_weights(edges, weight_metric):
	"""Set the 'weight' column per the chosen metric (duration | proximity | combined)."""
	if not edges:
		return
	durations = np.array([e['n_frames_observed'] for e in edges], dtype=float)
	prox = np.array([1.0 / e['mean_distance_bodylen'] if e['mean_distance_bodylen'] > 1e-9 else 0.0
					 for e in edges], dtype=float)

	def _norm(a):
		lo, hi = a.min(), a.max()
		return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

	if weight_metric == 'proximity':
		w = prox
	elif weight_metric == 'combined':
		w = 0.5 * _norm(durations) + 0.5 * _norm(prox)
	else:  # duration (default)
		w = durations
	for e, wi in zip(edges, w):
		e['weight'] = round(float(wi), 4)


def _graph_summary(edges, granularity):
	"""Print connected-components / community / centrality summaries (networkx)."""
	if granularity == 'per_frame' or not edges:
		return
	if not _NX_AVAILABLE:
		print("  networkx not available — skipping centrality/community summary "
			  "(edges/nodes still written). Install with: pip install networkx")
		return
	g = nx.DiGraph()
	for e in edges:
		g.add_edge(e['source_id'], e['target_id'], weight=e.get('weight', 1.0))
	if g.number_of_nodes() == 0:
		return
	ug = g.to_undirected()
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

def build_nodes(track_data, size_ratio, is_foal):
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
			'size_ratio': round(size_ratio.get(tid, 1.0), 4),
			'is_foal': int(bool(is_foal.get(tid, False))),
			'primary_class_dominant': _dominant(prim),
			'secondary_class_dominant': _dominant(sec),
			'presence_ratio': round(len(frames) / total_span, 4) if total_span else 0.0,
		})
	nodes.sort(key=lambda n: (len(str(n['track_id'])), str(n['track_id'])))
	return nodes


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

_EDGE_COLS = {
	'per_interaction': ['frame_start', 'frame_end', 'source_id', 'target_id', 'directed',
						'interaction_type', 'weight', 'n_frames_observed',
						'mean_distance_bodylen', 'min_distance_bodylen', 'contact_fraction',
						'mean_speed_similarity', 'mean_approach_rate',
						'class_source_primary', 'class_source_secondary',
						'class_target_primary', 'class_target_secondary'],
	'per_segment': ['frame_start', 'frame_end', 'source_id', 'target_id', 'directed',
					'interaction_type', 'weight', 'n_frames_observed',
					'mean_distance_bodylen', 'min_distance_bodylen', 'contact_fraction',
					'mean_speed_similarity', 'mean_approach_rate',
					'class_source_primary', 'class_source_secondary',
					'class_target_primary', 'class_target_secondary'],
	'per_frame': ['frame', 'source_id', 'target_id', 'weight', 'distance_bodylen',
				  'distance_m',
				  'in_contact', 'speed_similarity', 'approach_rate',
				  'class_source_primary', 'class_source_secondary',
				  'class_target_primary', 'class_target_secondary'],
}
_NODE_COLS = ['track_id', 'first_frame', 'last_frame',
			  'size_ratio', 'is_foal', 'primary_class_dominant',
			  'secondary_class_dominant', 'presence_ratio']


def write_interaction_graph(edges_rows, nodes_rows, edges_path, nodes_path,
							granularity="per_interaction"):
	"""Write (overwrite) the edges and nodes CSVs. Always truncates the file so a
	changed granularity/weight never leaves stale rows behind."""
	os.makedirs(os.path.dirname(os.path.abspath(edges_path)), exist_ok=True)
	edge_cols = _EDGE_COLS.get(granularity, _EDGE_COLS['per_interaction'])
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

def run_complex_features(config_path):
	"""Build the interaction graph for every tracking CSV in the output folder.

	Prefers <video>_tracking_corrected.csv; writes <video>_interaction_edges.csv
	and <video>_interaction_nodes.csv (overwritten each run / parameter change).
	"""
	config_path = os.path.abspath(config_path)
	project_dir = os.path.dirname(config_path)
	params = load_complex_config(config_path)

	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	output_dir_raw = d.get('output_dir', 'output')
	output_dir = output_dir_raw if os.path.isabs(output_dir_raw) \
		else os.path.join(project_dir, output_dir_raw)

	# Preference: metric CSV (superset with X_m/Y_m) > stitched (best identities) >
	# drone-corrected > raw. metric geometry itself consumes the stitched CSV, so
	# metric already carries the stitched identities; the stitched fallback matters
	# when stitching is on but metric is off -- without it the graph (and, through it,
	# the activity budget) would silently run on the pre-stitch identities.
	jobs = {}
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking_metric.csv'))):
		jobs[os.path.basename(p).replace('_tracking_metric.csv', '')] = p
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking_stitched.csv'))):
		jobs.setdefault(os.path.basename(p).replace('_tracking_stitched.csv', ''), p)
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking_corrected.csv'))):
		jobs.setdefault(os.path.basename(p).replace('_tracking_corrected.csv', ''), p)
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv'))):
		jobs.setdefault(os.path.basename(p).replace('_tracking.csv', ''), p)

	if not jobs:
		print(f"Complex features: no tracking CSVs found in {output_dir}")
		return

	print(f"Complex features: processing {len(jobs)} video(s) "
		  f"(granularity={params['edge_granularity']}, weight={params['weight_metric']})...")
	for stem, csv_path in sorted(jobs.items()):
		try:
			_process_one(stem, csv_path, output_dir, params)
		except Exception as e:
			import traceback
			print(f"  ERROR on {os.path.basename(csv_path)}: {e}")
			traceback.print_exc()
	print("Complex features complete.")


def _process_one(stem, csv_path, output_dir, params):
	"""Run the full TASK 4 pipeline for a single video."""
	track_data = load_tracking_csv(csv_path)
	if not track_data['frames']:
		print(f"  {stem}: no rows, skipping.")
		return

	ref, size_ratio, is_foal = compute_body_len_ref(
		track_data, scope=params['body_len_ref_scope'], foal_ratio=params['foal_size_ratio'])
	n_foal = sum(1 for v in is_foal.values() if v)
	print(f"  {stem}: body_len_ref={ref:.1f}px, {len(track_data['id_frames'])} tracks "
		  f"({n_foal} likely foal(s)).")

	pairwise = compute_pairwise_features(
		track_data, ref, size_ratio,
		max_distance=params['max_interaction_distance'],
		contact_iou_thresh=params['contact_iou_thresh'],
		contact_dist_bodylen=params['contact_dist_bodylen'])

	# Group features (whole co-present herd per frame) computed for downstream
	# use (TASK 7) and summarised here.
	groups = whole_herd_groups(track_data)
	gfeat = compute_group_features(track_data, groups, ref)
	if gfeat:
		pol = np.mean([g['polarisation'] for g in gfeat])
		print(f"  {stem}: {len(gfeat)} group-feature rows (mean polarisation={pol:.2f}).")

	nodes = build_nodes(track_data, size_ratio, is_foal)
	edges, nodes = build_interaction_graph(
		pairwise, nodes,
		granularity=params['edge_granularity'],
		weight_metric=params['weight_metric'],
		min_duration_frames=params['min_duration_frames'],
		all_frames=track_data['frames'])

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
