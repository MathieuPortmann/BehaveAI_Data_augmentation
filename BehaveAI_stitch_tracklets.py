#!/usr/bin/env python3
"""BehaveAI offline tracklet stitching

The online tracker (BoT-SORT/ByteTrack, or the legacy Kalman) is causal: once it
cuts a track or swaps an id it never revisits the decision. But the video is on
disk, so we can read it whole and re-link the short, reliable tracklets it
produced into longer identities -- purely on kinematics, no appearance.

Reads a tracking CSV (prefers the drone-corrected one, whose x_corrected/
y_corrected live in a single stabilised reference frame comparable across the
whole clip), groups rows into tracklets by id, and links compatible end->start
pairs. Two hard constraints make the problem easy and safe:
  * two tracklets overlapping in TIME are different animals (never linked);
  * an implied speed above the physical gate is impossible (never linked).
The soft cost is the normalised gap between where tracklet A is heading and
where B starts, in [0, 1]; a link is kept only if it costs less than doing
nothing (the dummy cost). K (group size) is NOT a constraint -- only a reported
diagnostic. Writes <video>_tracking_stitched.csv (id remapped) and
<video>_stitch_report.txt.

Scale note: the gate is in pixels/frame here. On a stabilised frame at fixed
altitude that is fine; when a flight log (rel_alt + pitch + focal) is available
the physical m/s gate should be converted to a per-frame px gate -- see
resolve_speed_gate() for the extension point.

Usage:
  python BehaveAI_stitch_tracklets.py <project_dir | settings.ini>
  python BehaveAI_stitch_tracklets.py --csv <one_tracking_corrected.csv>
"""

import os
import csv
import glob
import argparse
import configparser
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9  # never np.inf: scipy's linear_sum_assignment rejects an all-inf matrix


# ---------------------------------------------------------------------------
# Reading / tracklets
# ---------------------------------------------------------------------------

def _read_rows(csv_path):
	with open(csv_path, newline='', encoding='utf-8', errors='replace') as f:
		reader = csv.DictReader(f)
		return list(reader), list(reader.fieldnames or [])


def _pos_keys(fieldnames):
	"""Prefer the drone-corrected, globally-comparable coordinates."""
	if 'x_corrected' in fieldnames and 'y_corrected' in fieldnames:
		return 'x_corrected', 'y_corrected', True
	return 'x', 'y', False


def extract_tracklets(rows, xk, yk, min_len=2):
	"""Group rows by id -> tracklet dict. Endpoint velocities are finite
	differences over up to the last/first 5 samples."""
	by_id = defaultdict(list)
	for r in rows:
		try:
			fr = int(r['frame'])
			x = float(r[xk]); y = float(r[yk])
		except (ValueError, KeyError, TypeError):
			continue
		q = r.get('correction_quality', 'ok')
		by_id[str(r['id'])].append((fr, x, y, q))

	tracklets = {}
	for tid, pts in by_id.items():
		pts.sort(key=lambda p: p[0])
		if len(pts) < min_len:
			# keep singletons too -- they can still be a link target/source,
			# just with zero velocity. Only drop empties.
			if not pts:
				continue
		frames = np.array([p[0] for p in pts], dtype=float)
		xs = np.array([p[1] for p in pts], dtype=float)
		ys = np.array([p[2] for p in pts], dtype=float)

		def _vel(idx_slice, sign):
			if len(frames) < 2:
				return np.array([0.0, 0.0])
			fsub = frames[idx_slice]; xsub = xs[idx_slice]; ysub = ys[idx_slice]
			df = fsub[-1] - fsub[0]
			if df == 0:
				return np.array([0.0, 0.0])
			return np.array([(xsub[-1] - xsub[0]) / df, (ysub[-1] - ysub[0]) / df])

		end_vel = _vel(slice(max(0, len(frames) - 5), len(frames)), +1)
		start_vel = _vel(slice(0, min(5, len(frames))), -1)
		tracklets[tid] = {
			'id': tid,
			'start_frame': int(frames[0]), 'end_frame': int(frames[-1]),
			'start_pos': np.array([xs[0], ys[0]]), 'end_pos': np.array([xs[-1], ys[-1]]),
			'start_vel': start_vel, 'end_vel': end_vel,
			'start_q': pts[0][3], 'end_q': pts[-1][3],
			'n': len(pts),
		}
	return tracklets


# ---------------------------------------------------------------------------
# Cost + solve
# ---------------------------------------------------------------------------

def resolve_speed_gate(default_px_per_frame, flightlog=None):
	"""Extension point: with a flight log (rel_alt + pitch + focal) a physical
	m/s gate can be converted to a per-frame pixel gate that tracks altitude.
	Not wired yet (test videos have no flight log) -- returns the configured
	pixel gate, which is correct on a stabilised frame at roughly fixed altitude."""
	return default_px_per_frame


def build_cost(order, tracklets, max_speed_px, quality_gate):
	"""Cost matrix rows=tracklet ENDS, cols=tracklet STARTS (same tracklet list).
	Entry = normalised gap cost in [0,1]; BIG for forbidden pairs (temporal
	overlap, implied speed over the gate, or a 'none'-quality endpoint)."""
	n = len(order)
	C = np.full((n, n), BIG, dtype=float)
	for i, a_id in enumerate(order):          # A ends
		A = tracklets[a_id]
		for j, b_id in enumerate(order):      # B starts
			if i == j:
				continue
			B = tracklets[b_id]
			gap = B['start_frame'] - A['end_frame']
			if gap <= 0:                      # temporal overlap / not after -> forbidden
				continue
			if quality_gate and (A['end_q'] == 'none' or B['start_q'] == 'none'):
				continue
			pred = A['end_pos'] + A['end_vel'] * gap
			dist = float(np.hypot(*(pred - B['start_pos'])))
			denom = max_speed_px * gap        # radius the animal COULD reach
			cost = dist / denom if denom > 0 else BIG
			if cost <= 1.0:                   # within the physical envelope
				C[i, j] = cost
	return C


def stitch(cost, max_link_cost):
	"""Dummy-augmented linear assignment with an opt-out. A real end<->start link
	is taken only when it is cheaper than 'leave unmatched' (cost max_link_cost).
	Returns list of (end_index, start_index)."""
	n = cost.shape[0]
	if n == 0:
		return []
	C = np.full((2 * n, 2 * n), BIG, dtype=float)
	C[:n, :n] = np.where(np.isfinite(cost), cost, BIG)
	np.fill_diagonal(C[:n, n:], max_link_cost)   # end i unmatched
	np.fill_diagonal(C[n:, :n], max_link_cost)   # start j unmatched
	C[n:, n:] = 0.0                              # dummy<->dummy free
	rows, cols = linear_sum_assignment(C)
	return [(r, c) for r, c in zip(rows, cols)
			if r < n and c < n and cost[r, c] < max_link_cost]


class _UnionFind:
	def __init__(self, items):
		self.p = {x: x for x in items}

	def find(self, x):
		while self.p[x] != x:
			self.p[x] = self.p[self.p[x]]
			x = self.p[x]
		return x

	def union(self, a, b):
		ra, rb = self.find(a), self.find(b)
		if ra != rb:
			self.p[rb] = ra


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _concurrency_stats(tracklets):
	"""Distribution of simultaneously-active tracklets per frame."""
	if not tracklets:
		return 0, 0, 0
	fmin = min(t['start_frame'] for t in tracklets.values())
	fmax = max(t['end_frame'] for t in tracklets.values())
	counts = np.zeros(fmax - fmin + 1, dtype=int)
	for t in tracklets.values():
		counts[t['start_frame'] - fmin: t['end_frame'] - fmin + 1] += 1
	return int(np.median(counts)), int(np.percentile(counts, 95)), int(counts.max())


def _write_report(path, tracklets, id_map, expected_group_size):
	med, p95, mx = _concurrency_stats(tracklets)
	n_chains = len(set(id_map.values()))
	lines = [
		"BehaveAI tracklet-stitch report",
		f"tracklets in: {len(tracklets)}",
		f"chains out (distinct stitched ids): {n_chains}",
		"concurrent tracklets/frame: "
		f"median={med} p95={p95} max={mx}",
		f"expected_group_size: {expected_group_size if expected_group_size else 'unknown'}",
		"",
		"Reading guide:",
		f"  - n_chains ({n_chains}) < max concurrent ({mx}) would be logically "
		"impossible (two simultaneous tracklets are different animals).",
	]
	warn = n_chains < mx
	if warn:
		lines.append("  !! WARNING: n_chains < max concurrent -> non-overlap constraint bug.")
	if mx and n_chains > 3 * mx:
		lines.append(f"  - n_chains >> concurrent: residual fragmentation (unrecovered splits).")
	if expected_group_size and mx > expected_group_size:
		lines.append("  - max concurrent > expected_group_size: field count wrong, an outside "
					 "individual is present, or detection false positives.")
	with open(path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines) + '\n')
	return warn, (med, p95, mx), n_chains


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_stitch(csv_path, out_path=None, max_speed_px=60.0, max_link_cost=0.5,
			   min_len=2, expected_group_size=0, quality_gate=True, verbose=True):
	"""Stitch one tracking CSV. Returns (out_csv, report_path, id_map)."""
	rows, fieldnames = _read_rows(csv_path)
	if not rows:
		if verbose:
			print(f"  {os.path.basename(csv_path)}: no rows, skipping.")
		return None, None, {}
	xk, yk, used_corr = _pos_keys(fieldnames)
	tracklets = extract_tracklets(rows, xk, yk, min_len=min_len)
	order = list(tracklets.keys())

	max_speed_px = resolve_speed_gate(max_speed_px)
	cost = build_cost(order, tracklets, max_speed_px, quality_gate)
	links = stitch(cost, max_link_cost)

	uf = _UnionFind(order)
	for i, j in links:
		uf.union(order[i], order[j])   # chain A-end -> B-start
	# Map each original id -> compact chain id (1..K), stable by first appearance.
	rep_to_new = {}
	id_map = {}
	next_id = 1
	for tid in sorted(order, key=lambda t: tracklets[t]['start_frame']):
		rep = uf.find(tid)
		if rep not in rep_to_new:
			rep_to_new[rep] = next_id
			next_id += 1
		id_map[tid] = rep_to_new[rep]

	if out_path is None:
		out_path = csv_path.replace('_tracking_corrected.csv', '_tracking_stitched.csv')
		out_path = out_path.replace('_tracking.csv', '_tracking_stitched.csv')
	for r in rows:
		if str(r['id']) in id_map:
			r['id'] = id_map[str(r['id'])]
	with open(out_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
		w.writeheader()
		w.writerows(rows)

	report_path = out_path.replace('_tracking_stitched.csv', '_stitch_report.txt')
	warn, dist, n_chains = _write_report(report_path, tracklets, id_map, expected_group_size)
	if verbose:
		src = 'corrected' if used_corr else 'raw (WARNING: not drone-stabilised)'
		print(f"  {os.path.basename(csv_path)} [{src}]: {len(tracklets)} tracklets, "
			  f"{len(links)} links -> {n_chains} chains "
			  f"(concurrent med/p95/max = {dist[0]}/{dist[1]}/{dist[2]})"
			  + ("  !!WARN" if warn else ""))
	return out_path, report_path, id_map


def _load_cfg(config_path):
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	return {
		'enabled': str(d.get('stitch_enabled', 'false')).lower() == 'true',
		'max_speed_px': float(d.get('stitch_max_speed_px_per_frame', '60')),
		'max_link_cost': float(d.get('stitch_max_link_cost', '0.5')),
		'min_len': int(d.get('stitch_min_tracklet_len', '2')),
		'expected_group_size': int(d.get('expected_group_size', '0') or '0'),
		'quality_gate': str(d.get('stitch_quality_gate', 'true')).lower() == 'true',
	}


def run_stitch_project(config_path):
	"""Batch every *_tracking_corrected.csv (falling back to *_tracking.csv) in
	the project's output folder."""
	config_path = os.path.abspath(config_path)
	project_dir = os.path.dirname(config_path)
	p = _load_cfg(config_path)
	cfg = configparser.ConfigParser(); cfg.optionxform = str; cfg.read(config_path)
	out_raw = cfg['DEFAULT'].get('output_folder', 'output')
	output_dir = out_raw if os.path.isabs(out_raw) else os.path.join(project_dir, out_raw)

	jobs = {}
	for c in sorted(glob.glob(os.path.join(output_dir, '*_tracking_corrected.csv'))):
		jobs[os.path.basename(c).replace('_tracking_corrected.csv', '')] = c
	for c in sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv'))):
		jobs.setdefault(os.path.basename(c).replace('_tracking.csv', ''), c)
	if not jobs:
		print(f"Stitch: no tracking CSVs in {output_dir}")
		return
	print(f"Stitch: processing {len(jobs)} video(s)...")
	for stem, c in jobs.items():
		run_stitch(c, max_speed_px=p['max_speed_px'], max_link_cost=p['max_link_cost'],
				   min_len=p['min_len'], expected_group_size=p['expected_group_size'],
				   quality_gate=p['quality_gate'])


if __name__ == '__main__':
	ap = argparse.ArgumentParser(description="Offline kinematic tracklet stitching.")
	ap.add_argument("target", nargs='?', help="project dir or BehaveAI_settings.ini")
	ap.add_argument("--csv", help="stitch a single tracking CSV instead")
	ap.add_argument("--max-speed-px", type=float, default=60.0)
	ap.add_argument("--max-link-cost", type=float, default=0.5)
	ap.add_argument("--expected-group-size", type=int, default=0)
	a = ap.parse_args()
	if a.csv:
		run_stitch(a.csv, max_speed_px=a.max_speed_px, max_link_cost=a.max_link_cost,
				   expected_group_size=a.expected_group_size)
	elif a.target:
		ini = a.target if a.target.endswith('.ini') else os.path.join(a.target, 'BehaveAI_settings.ini')
		run_stitch_project(ini)
	else:
		ap.error("give a project dir / .ini, or --csv <file>")
