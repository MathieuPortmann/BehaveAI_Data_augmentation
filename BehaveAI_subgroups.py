#!/usr/bin/env python3
"""
BehaveAI Sub-grouping (fission-fusion)

Partitions the co-present horses, per frame, into spatially coherent SUB-GROUPS
that are stable in time, so group-level behaviours (TASK 4) can be computed per
sub-group (e.g. one band trekking while another grazes).

A sub-group here is an OBSERVED SPATIAL CLUSTER, not the named social band.

Method:
  * Per-frame clustering with DBSCAN(eps, min_samples=1). With min_samples=1 every
    point is a core point, so DBSCAN reduces EXACTLY to the connected components of
    the eps-neighbourhood graph (isolated horses become singletons). We implement
    that connected-components form directly with scipy.cKDTree + union-find, which
    is identical to sklearn's DBSCAN(min_samples=1) but needs no scikit-learn.
  * eps is expressed in REFERENCE body lengths (subgroup_eps_bodylen *
    body_len_ref). body_len_ref is the robust median of the per-individual box
    diagonals of ADULT-sized horses (clarification 3) — never each horse's own
    size — computed at video level by default, or per stable-scale segment when
    body_len_ref_scope = segment (altitude/zoom change).
  * Temporal stability: a horse only changes sub-group once the change has
    persisted for >= subgroup_min_stable_frames (debounce, removes flicker).
  * Sub-group identity over time is tracked by membership overlap; merges and
    splits are detected and logged.

Positions come from the drone-corrected columns (x_corrected, y_corrected) when
present, otherwise the raw (x, y) with a one-time warning. Everything is in image
space; no metric conversion.

Output <videoname>_subgroups.csv columns:
  frame, subgroup_id, track_ids, n_members, centroid_x, centroid_y
(track_ids is ';'-separated.)

Usage:
  python BehaveAI_subgroups.py <project_dir | BehaveAI_settings.ini>
"""

import os
import sys
import csv
import glob
import argparse
import configparser
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree


# Relative change in the per-frame median box size that starts a new stable
# segment when body_len_ref_scope = segment (proxy for an altitude/zoom change).
_SEGMENT_REL_THRESH = 0.15
# Window for smoothing the per-frame median box size before segmenting.
_SEGMENT_SMOOTH_WIN = 15


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_subgroup_config(config_path):
	"""Read sub-grouping parameters from a BehaveAI INI (DEFAULT section)."""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	return {
		'eps_bodylen':       float(d.get('subgroup_eps_bodylen', '4.0')),
		'min_stable_frames': int(float(d.get('subgroup_min_stable_frames', '10'))),
		'foal_size_ratio':   float(d.get('foal_size_ratio_thresh', '0.7')),
		'scope':             d.get('body_len_ref_scope', 'video'),
	}


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _read_positions_and_boxes(csv_path):
	"""Read a (corrected) tracking CSV.

	Returns (frames_sorted, present_ids, positions, row_diag, has_bbox) where:
	  - present_ids[frame]  -> list of track-id strings present that frame
	  - positions[(frame, tid)] -> (x, y) using corrected columns when available
	  - row_diag[(frame, tid)]  -> box diagonal in px, or None when no bbox
	"""
	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fieldnames = list(reader.fieldnames or [])
		rows = [dict(r) for r in reader]

	use_corrected = 'x_corrected' in fieldnames and 'y_corrected' in fieldnames
	has_bbox = all(c in fieldnames for c in ('x1', 'y1', 'x2', 'y2'))

	present_ids = defaultdict(list)
	positions = {}
	row_diag = {}
	for r in rows:
		try:
			frame = int(r['frame'])
			tid = str(r['id'])
			if use_corrected and r.get('x_corrected', '') != '':
				x = float(r['x_corrected']); y = float(r['y_corrected'])
			else:
				x = float(r['x']); y = float(r['y'])
		except (ValueError, KeyError, TypeError):
			continue
		# A track id should appear once per frame; ignore accidental duplicates.
		if (frame, tid) in positions:
			continue
		present_ids[frame].append(tid)
		positions[(frame, tid)] = (x, y)
		if has_bbox:
			try:
				diag = float(np.hypot(int(r['x2']) - int(r['x1']),
									  int(r['y2']) - int(r['y1'])))
				row_diag[(frame, tid)] = diag if diag > 0 else None
			except (ValueError, TypeError):
				row_diag[(frame, tid)] = None

	frames_sorted = sorted(present_ids.keys())
	return frames_sorted, present_ids, positions, row_diag, has_bbox, use_corrected


# ---------------------------------------------------------------------------
# Body-length reference
# ---------------------------------------------------------------------------

def _robust_body_len_ref(track_diags, foal_ratio):
	"""Robust median box diagonal of adult-sized tracks.

	track_diags: list of per-track median diagonals. Foals (diag below
	foal_ratio * initial median) are excluded so they do not shrink the
	adult reference.
	"""
	arr = np.array([d for d in track_diags if d and d > 0], dtype=float)
	if arr.size == 0:
		return 0.0
	initial = float(np.median(arr))
	adults = arr[arr >= foal_ratio * initial]
	if adults.size == 0:
		adults = arr
	return float(np.median(adults))


def _per_track_median_diag(frames, present_ids, row_diag, frame_filter=None):
	"""Median box diagonal per track (optionally restricted to some frames)."""
	by_track = defaultdict(list)
	for f in frames:
		if frame_filter is not None and not frame_filter(f):
			continue
		for tid in present_ids[f]:
			d = row_diag.get((f, tid))
			if d:
				by_track[tid].append(d)
	return {tid: float(np.median(ds)) for tid, ds in by_track.items() if ds}


def _fallback_length(frames, present_ids, positions):
	"""Scene length scale when boxes are unavailable: median nearest-neighbour
	distance between co-present horses across the video."""
	nn = []
	for f in frames:
		ids = present_ids[f]
		if len(ids) >= 2:
			pts = np.array([positions[(f, t)] for t in ids], dtype=float)
			tree = cKDTree(pts)
			d, _ = tree.query(pts, k=2)
			nn.extend(d[:, 1].tolist())
	return float(np.median(nn)) if nn else 50.0


def _segment_frames(frames, present_ids, row_diag):
	"""Split the video into stable-scale segments by change-points in the
	smoothed per-frame median box diagonal (proxy for altitude/zoom)."""
	med = []
	for f in frames:
		ds = [row_diag.get((f, t)) for t in present_ids[f]]
		ds = [d for d in ds if d]
		med.append(np.median(ds) if ds else np.nan)
	med = np.array(med, dtype=float)
	# Forward/backward fill NaNs, then smooth.
	if np.all(np.isnan(med)):
		return [(frames[0], frames[-1])]
	idx = np.where(~np.isnan(med))[0]
	med = np.interp(np.arange(len(med)), idx, med[idx])
	win = min(_SEGMENT_SMOOTH_WIN, len(med))
	if win >= 3:
		if win % 2 == 0:
			win -= 1
		kernel = np.ones(win) / win
		med = np.convolve(np.pad(med, win // 2, mode='edge'), kernel, mode='valid')[:len(frames)]

	segments = []
	seg_start_i = 0
	ref = med[0]
	for i in range(len(frames)):
		if ref > 0 and abs(med[i] - ref) / ref > _SEGMENT_REL_THRESH:
			segments.append((frames[seg_start_i], frames[i - 1] if i > 0 else frames[i]))
			seg_start_i = i
			ref = med[i]
	segments.append((frames[seg_start_i], frames[-1]))
	return segments


def _build_body_len_lookup(frames, present_ids, row_diag, positions,
						   has_bbox, foal_ratio, scope):
	"""Return (body_len_at(frame) -> px, info) for the requested scope."""
	if not has_bbox:
		ref = _fallback_length(frames, present_ids, positions)
		print("  WARNING: no bbox columns — body_len_ref approximated from "
			  f"nearest-neighbour spacing (~{ref:.1f}px); results are approximate.")
		return (lambda f: ref), {'scope': 'fallback', 'body_len_ref': ref, 'segments': 1}

	if scope == 'segment':
		segments = _segment_frames(frames, present_ids, row_diag)
		seg_refs = []
		for (a, b) in segments:
			diags = _per_track_median_diag(
				frames, present_ids, row_diag,
				frame_filter=lambda f, a=a, b=b: a <= f <= b)
			seg_refs.append(_robust_body_len_ref(list(diags.values()), foal_ratio))
		# Map each frame to its segment's reference.
		frame_ref = {}
		for (a, b), ref in zip(segments, seg_refs):
			for f in frames:
				if a <= f <= b:
					frame_ref[f] = ref if ref > 0 else None
		# Fill any zero refs with the global one.
		global_ref = _robust_body_len_ref(
			list(_per_track_median_diag(frames, present_ids, row_diag).values()), foal_ratio)
		lookup = lambda f: (frame_ref.get(f) or global_ref or 1.0)
		if len(segments) > 1:
			print(f"  body_len_ref per segment: "
				  f"{', '.join(f'{a}-{b}:{r:.1f}px' for (a, b), r in zip(segments, seg_refs))}")
		return lookup, {'scope': 'segment', 'segments': len(segments)}

	# Default: one reference for the whole video.
	diags = _per_track_median_diag(frames, present_ids, row_diag)
	ref = _robust_body_len_ref(list(diags.values()), foal_ratio)
	if ref <= 0:
		ref = _fallback_length(frames, present_ids, positions)
	# Flag likely foals from the bimodal size distribution.
	foals = [tid for tid, d in diags.items() if ref > 0 and d / ref < foal_ratio]
	if foals:
		print(f"  Likely foals (body_len/ref < {foal_ratio}): {len(foals)} "
			  f"({', '.join(sorted(foals, key=lambda t: (len(t), t))[:8])}"
			  f"{'...' if len(foals) > 8 else ''})")
	return (lambda f: ref), {'scope': 'video', 'body_len_ref': ref, 'segments': 1}


# ---------------------------------------------------------------------------
# Clustering (DBSCAN min_samples=1 == connected components of the eps-graph)
# ---------------------------------------------------------------------------

def _cluster_labels(points, eps):
	"""Connected components of the eps-neighbourhood graph (union-find).

	Identical to sklearn DBSCAN(eps=eps, min_samples=1): isolated points form
	singleton clusters; no noise label.
	"""
	n = len(points)
	if n == 0:
		return []
	if n == 1:
		return [0]
	parent = list(range(n))

	def find(a):
		while parent[a] != a:
			parent[a] = parent[parent[a]]
			a = parent[a]
		return a

	pts = np.asarray(points, dtype=float)
	tree = cKDTree(pts)
	for i, j in tree.query_pairs(r=eps):
		ra, rb = find(i), find(j)
		if ra != rb:
			parent[ra] = rb

	roots = [find(i) for i in range(n)]
	remap = {}
	out = []
	for r in roots:
		if r not in remap:
			remap[r] = len(remap)
		out.append(remap[r])
	return out


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def compute_subgroups(csv_path, output_csv_path,
					  eps_bodylen=4.0, min_stable_frames=10,
					  foal_size_ratio=0.7, scope="video"):
	"""Cluster co-present horses per frame into temporally stable sub-groups and
	write <videoname>_subgroups.csv. See module docstring for the method."""
	frames, present_ids, positions, row_diag, has_bbox, use_corrected = \
		_read_positions_and_boxes(csv_path)
	if not frames:
		print(f"  {os.path.basename(csv_path)}: no rows, skipping.")
		return

	if not use_corrected:
		print(f"  NOTE: {os.path.basename(csv_path)} has no corrected columns — "
			  f"using raw (x, y).")

	body_len_at, info = _build_body_len_lookup(
		frames, present_ids, row_diag, positions, has_bbox, foal_size_ratio, scope)

	# --- Per-frame raw clusters (as frozensets of track ids) ---
	raw_clusters = {}
	for f in frames:
		ids = present_ids[f]
		pts = [positions[(f, t)] for t in ids]
		ref = body_len_at(f) or 1.0
		eps = eps_bodylen * ref
		labels = _cluster_labels(pts, eps)
		groups = defaultdict(list)
		for tid, lab in zip(ids, labels):
			groups[lab].append(tid)
		raw_clusters[f] = [frozenset(g) for g in groups.values()]

	# --- Lineage tracking (consistent sub-group ids; merge/split events) ---
	frame_assign, merges, splits = _track_lineages(frames, raw_clusters)

	# --- Temporal stability debounce (per individual) ---
	final_assign = _debounce(frames, frame_assign, min_stable_frames)

	# --- Write output rows ---
	_write_subgroups(frames, final_assign, positions, output_csv_path)

	n_rows = sum(len(set(a.values())) for a in final_assign.values())
	print(f"  Wrote {os.path.basename(output_csv_path)}: {n_rows} sub-group rows "
		  f"over {len(frames)} frames (merges={merges}, splits={splits}, "
		  f"scope={info['scope']}).")


def _track_lineages(frames, raw_clusters):
	"""Assign a persistent lineage id to each cluster by membership overlap with
	the previous frame. Returns (frame_assign, n_merges, n_splits) where
	frame_assign[frame] = {track_id: lineage_id}."""
	next_lid = 1
	prev = []  # list of (frozenset members, lid)
	frame_assign = {}
	n_merges = 0
	n_splits = 0

	for f in frames:
		clusters = raw_clusters[f]
		# Best previous lineage (by overlap) for each new cluster.
		cluster_best = []
		for cl in clusters:
			best_lid, best_ov = None, 0
			for members, lid in prev:
				ov = len(cl & members)
				if ov > best_ov:
					best_ov, best_lid = ov, lid
			cluster_best.append((best_lid, best_ov))

		# Group clusters that claim the same lineage -> the rest are splits.
		lid_to_clusters = defaultdict(list)
		for idx, (lid, ov) in enumerate(cluster_best):
			if lid is not None and ov > 0:
				lid_to_clusters[lid].append(idx)

		assigned = [None] * len(clusters)
		for lid, idxs in lid_to_clusters.items():
			if len(idxs) == 1:
				assigned[idxs[0]] = lid
			else:
				idxs_sorted = sorted(idxs, key=lambda i: cluster_best[i][1], reverse=True)
				assigned[idxs_sorted[0]] = lid
				for i in idxs_sorted[1:]:
					assigned[i] = next_lid
					next_lid += 1
					n_splits += 1
		for idx in range(len(clusters)):
			if assigned[idx] is None:
				assigned[idx] = next_lid
				next_lid += 1

		# Count merges: a new cluster overlapping >1 previous lineage.
		for cl in clusters:
			overlapping = {lid for members, lid in prev if cl & members}
			if len(overlapping) > 1:
				n_merges += 1

		assign = {}
		new_prev = []
		for cl, lid in zip(clusters, assigned):
			for tid in cl:
				assign[tid] = lid
			new_prev.append((cl, lid))
		frame_assign[f] = assign
		prev = new_prev

	return frame_assign, n_merges, n_splits


def _debounce(frames, frame_assign, min_stable_frames):
	"""Commit a horse's sub-group change only after it persists for
	min_stable_frames consecutive frames (removes flicker)."""
	committed = {}
	pending = {}
	final_assign = {}
	for f in frames:
		raw = frame_assign[f]
		fa = {}
		for tid, lid in raw.items():
			if tid not in committed:
				committed[tid] = lid
				pending.pop(tid, None)
			elif lid == committed[tid]:
				pending.pop(tid, None)
			else:
				cand, cnt = pending.get(tid, (None, 0))
				if cand == lid:
					cnt += 1
				else:
					cand, cnt = lid, 1
				if cnt >= max(1, min_stable_frames):
					committed[tid] = lid
					pending.pop(tid, None)
				else:
					pending[tid] = (cand, cnt)
			fa[tid] = committed[tid]
		final_assign[f] = fa
	return final_assign


def _write_subgroups(frames, final_assign, positions, output_csv_path):
	"""Write the sub-groups CSV: one row per (frame, sub-group)."""
	os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

	def _id_key(t):
		try:
			return (0, int(t))
		except ValueError:
			return (1, t)

	with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(['frame', 'subgroup_id', 'track_ids',
						 'n_members', 'centroid_x', 'centroid_y'])
		for fr in frames:
			groups = defaultdict(list)
			for tid, lid in final_assign[fr].items():
				groups[lid].append(tid)
			for lid in sorted(groups.keys()):
				ids = sorted(groups[lid], key=_id_key)
				pts = np.array([positions[(fr, t)] for t in ids], dtype=float)
				cx, cy = pts.mean(axis=0)
				writer.writerow([fr, lid, ';'.join(ids), len(ids),
								 f"{cx:.2f}", f"{cy:.2f}"])


# ---------------------------------------------------------------------------
# Batch / project entry points
# ---------------------------------------------------------------------------

def run_subgrouping(config_path):
	"""Compute sub-groups for every tracking CSV in the project's output folder.

	Prefers <video>_tracking_corrected.csv (TASK 1); falls back to
	<video>_tracking.csv with a warning. Writes <video>_subgroups.csv.
	"""
	config_path = os.path.abspath(config_path)
	project_dir = os.path.dirname(config_path)
	params = load_subgroup_config(config_path)

	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	output_dir_raw = d.get('output_dir', 'output')
	output_dir = output_dir_raw if os.path.isabs(output_dir_raw) \
		else os.path.join(project_dir, output_dir_raw)

	# One job per video: prefer the corrected CSV, else the raw tracking CSV.
	corrected = sorted(glob.glob(os.path.join(output_dir, '*_tracking_corrected.csv')))
	jobs = {}
	for p in corrected:
		stem = os.path.basename(p).replace('_tracking_corrected.csv', '')
		jobs[stem] = p
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv'))):
		stem = os.path.basename(p).replace('_tracking.csv', '')
		jobs.setdefault(stem, p)  # only if no corrected version exists

	if not jobs:
		print(f"Sub-grouping: no tracking CSVs found in {output_dir}")
		return

	print(f"Sub-grouping: processing {len(jobs)} video(s) "
		  f"(eps={params['eps_bodylen']} body lengths, scope={params['scope']})...")
	for stem, csv_path in sorted(jobs.items()):
		out_path = os.path.join(output_dir, stem + '_subgroups.csv')
		try:
			compute_subgroups(
				csv_path, out_path,
				eps_bodylen=params['eps_bodylen'],
				min_stable_frames=params['min_stable_frames'],
				foal_size_ratio=params['foal_size_ratio'],
				scope=params['scope'])
		except Exception as e:
			import traceback
			print(f"  ERROR sub-grouping {os.path.basename(csv_path)}: {e}")
			traceback.print_exc()
	print("Sub-grouping complete.")


def _main():
	parser = argparse.ArgumentParser(
		description="Partition co-present horses into temporally stable sub-groups.")
	parser.add_argument('target',
						help="Project directory or BehaveAI_settings.ini (batch mode), "
							 "or a tracking CSV when --out is given (single-file mode).")
	parser.add_argument('--out', default=None,
						help="Single-file mode: output sub-groups CSV path.")
	args = parser.parse_args()

	if args.out is not None:
		compute_subgroups(args.target, args.out)
		return

	target = os.path.abspath(args.target)
	ini = os.path.join(target, 'BehaveAI_settings.ini') if os.path.isdir(target) else target
	if not os.path.exists(ini):
		print(f"Settings file not found: {ini}")
		sys.exit(1)
	run_subgrouping(ini)


if __name__ == '__main__':
	_main()
