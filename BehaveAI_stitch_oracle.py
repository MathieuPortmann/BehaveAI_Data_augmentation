#!/usr/bin/env python3
"""BehaveAI stitching oracle -- controlled fragmentation benchmark

Purpose. `stitch_max_gap_seconds` (and how permissive the link prior should be)
must not be chosen by taste. This harness measures what the stitcher actually
does as a function of gap length, on real trajectories, and produces the curve
those settings should be read off.

Method. Take the long tracks of a real tracking CSV as pseudo-truth, cut each one
into fragments separated by a gap of G frames (the frames inside the gap are
deleted, exactly as an occlusion would), give every fragment a fresh id, and run
the real linking machinery on the result. Because we know which fragments belong
together, every link can be scored:

  recovery     = correct adjacent re-links / cuts made          (higher is better)
  contamination= links joining two DIFFERENT animals / links taken (lower is better)
  chain purity = chains containing exactly one animal / chains

Sweeping G gives the operating curve: the gap beyond which recovery collapses or
contamination takes off is where `stitch_max_gap_seconds` belongs.

What this does and does not establish. It isolates the ASSOCIATION step: the
input trajectories are the tracker's own output, so a tracker id-swap inside a
"true" track is inherited as truth, and only fragmentation is simulated, never a
swap. It therefore validates the geometry and the decision rule, NOT the whole
detect-track-stitch chain -- that needs hand-annotated MOT ground truth
(EVALUATION_PLAN §C). Read it as a calibration and a sanity bound, not as an
accuracy claim.

Asymmetry to keep in mind when reading the curve: a missed re-link costs
statistical power (one animal counted as two), a contaminated chain costs
validity (two animals counted as one, both budgets wrong). They are not
interchangeable -- prefer under-linking.

Usage:
  python BehaveAI_stitch_oracle.py --csv output/CLIP_tracking_corrected.csv \
      --fps 30 --gaps 0.5,1,2,3,5,8,12 --segment-s 10 --repeats 3
"""

import os
import csv
import sys
import math
import argparse
from collections import defaultdict

import numpy as np

from BehaveAI_stitch_tracklets import (_read_rows, _pos_keys, extract_tracklets,
									   estimate_motion_noise, occupied_area_px2,
									   build_cost, stitch, _UnionFind)


def load_true_tracks(csv_path, min_len_frames):
	"""Pseudo-truth: the tracker's own tracks, kept when long enough to be cut."""
	rows, fieldnames = _read_rows(csv_path)
	xk, yk, used_corr = _pos_keys(fieldnames)
	by_id = defaultdict(lambda: defaultdict(list))
	for r in rows:
		try:
			by_id[str(r['id'])][int(r['frame'])].append((float(r[xk]), float(r[yk])))
		except (ValueError, KeyError, TypeError):
			continue
	tracks = {}
	for tid, per_frame in by_id.items():
		# Collapse the static/motion duplicate rows the same way the stitcher
		# does, otherwise one animal looks like two detections 100 px apart and
		# every spacing statistic below is nonsense.
		pts = [(fr,) + tuple(np.median(np.array(v, dtype=float), axis=0))
			   for fr, v in sorted(per_frame.items())]
		if pts and (pts[-1][0] - pts[0][0]) >= min_len_frames:
			tracks[tid] = pts
	return tracks, used_corr


def fragment(tracks, segment_frames, gap_frames, rng):
	"""Cut every track into fragments of ~segment_frames separated by gap_frames.

	Returns (rows, frag_owner): synthetic tracking rows with fresh per-fragment
	ids, and the map fragment_id -> (true_id, index_within_track).
	"""
	rows, frag_owner, next_frag = [], {}, 1
	for tid, pts in tracks.items():
		f0, f1 = pts[0][0], pts[-1][0]
		cursor, k = f0, 0
		while cursor <= f1:
			# jitter the segment length so cut points are not all in phase
			seg = int(segment_frames * rng.uniform(0.7, 1.3))
			lo, hi = cursor, cursor + seg
			chunk = [p for p in pts if lo <= p[0] < hi]
			if len(chunk) >= 4:
				fid = str(next_frag); next_frag += 1
				frag_owner[fid] = (tid, k); k += 1
				for fr, x, y in chunk:
					rows.append({'frame': str(fr), 'id': fid, 'x': f'{x:.3f}', 'y': f'{y:.3f}'})
			cursor = hi + gap_frames        # the gap frames are simply deleted
	return rows, frag_owner


def overlapping_pairs(tracks):
	"""Pairs of pseudo-true tracks that are active at the same time, i.e. that are
	PROVABLY different animals.

	This matters: the pseudo-truth is the tracker's own output, so two tracks that
	never coexist may well be one animal the tracker had already split. Charging a
	link between them as contamination would blame the stitcher for repairing a
	genuine fragmentation. Only time-overlapping pairs are counted as errors; the
	rest are reported separately as 'ambiguous'."""
	spans = {tid: (pts[0][0], pts[-1][0]) for tid, pts in tracks.items()}
	over = set()
	ids = sorted(spans)
	for i, a in enumerate(ids):
		a0, a1 = spans[a]
		for b in ids[i + 1:]:
			b0, b1 = spans[b]
			if a0 <= b1 and b0 <= a1:
				over.add((a, b)); over.add((b, a))
	return over


def score(links, order, frag_owner, id_map, overlap):
	"""Classify every link taken and every resulting chain."""
	correct, nonadjacent, cross, ambiguous = 0, 0, 0, 0
	for i, j in links:
		a, b = order[i], order[j]
		ta, ka = frag_owner[a]
		tb, kb = frag_owner[b]
		if ta != tb:
			if (ta, tb) in overlap:
				cross += 1            # provably two different animals: an error
			else:
				ambiguous += 1        # possibly the same animal, split by the tracker
		elif kb == ka + 1:
			correct += 1
		else:
			nonadjacent += 1
	chains = defaultdict(set)
	for fid, cid in id_map.items():
		if fid in frag_owner:
			chains[cid].add(frag_owner[fid][0])
	impure = 0
	for s in chains.values():
		ss = sorted(s)
		if any((x, y) in overlap for i, x in enumerate(ss) for y in ss[i + 1:]):
			impure += 1
	return {'correct': correct, 'nonadjacent': nonadjacent, 'cross': cross,
			'ambiguous': ambiguous, 'n_chains': len(chains),
			'pure_chains': len(chains) - impure}


def run_trial(tracks, segment_frames, gap_frames, fps, rng, max_gap_s, gap_prior_s,
			  extrap_horizon_s, max_speed_px, prior_log_odds, overlap):
	rows, frag_owner = fragment(tracks, segment_frames, gap_frames, rng)
	if len(frag_owner) < 2:
		return None
	vel_window = max(2, int(round(0.5 * fps)))
	tracklets, _dropped = extract_tracklets(rows, 'x', 'y', min_len=1,
											vel_window_frames=vel_window)
	order = list(tracklets.keys())
	horizon = max(1.0, extrap_horizon_s * fps)
	noise = estimate_motion_noise(tracklets, horizon=horizon, vel_window_frames=vel_window)
	cost, diag = build_cost(order, tracklets, max_speed_px,
							max(1, int(round(max_gap_s * fps))),
							noise['m0_px'], noise['alpha'], noise['beta'], noise['nu'],
							max(1.0, gap_prior_s * fps), occupied_area_px2(tracklets),
							horizon, quality_gate=False, prior_log_odds=prior_log_odds)
	links = stitch(cost)

	uf = _UnionFind(order)
	for i, j in links:
		uf.union(order[i], order[j])
	rep_to_new, id_map, nxt = {}, {}, 1
	for tid in sorted(order, key=lambda t: (tracklets[t]['start_frame'], t)):
		rep = uf.find(tid)
		if rep not in rep_to_new:
			rep_to_new[rep] = nxt; nxt += 1
		id_map[tid] = rep_to_new[rep]

	s = score(links, order, frag_owner, id_map, overlap)
	# One cut per fragment beyond the first, per true track.
	cuts = len(frag_owner) - len(set(t for t, _ in frag_owner.values()))
	s.update({'fragments': len(frag_owner), 'cuts': cuts, 'links': len(links),
			  'candidates': diag['candidates'], 'beta': noise['beta'],
			  'nu': noise['nu']})
	return s


def neighbour_spacing_px(tracks, n_frames=300, rng=None):
	"""Median nearest-neighbour distance between animals visible at the same time.

	This is the scale that decides whether kinematic linking is identifiable at
	all: the prediction spread sigma(gap) has to stay well under it, otherwise the
	right continuation and its neighbours are indistinguishable and no threshold
	can separate them. Reporting it turns "the stitcher did badly on this clip"
	into "this clip is too tightly packed for kinematics".
	"""
	rng = rng or np.random.default_rng(0)
	pos_by_frame = defaultdict(list)
	for pts in tracks.values():
		for fr, x, y in pts:
			pos_by_frame[fr].append((x, y))
	frames = [f for f, v in pos_by_frame.items() if len(v) >= 2]
	if not frames:
		return float('nan')
	if len(frames) > n_frames:
		frames = list(rng.choice(np.array(frames), n_frames, replace=False))
	dists = []
	for f in frames:
		p = np.array(pos_by_frame[f], dtype=float)
		d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
		np.fill_diagonal(d, np.inf)
		dists.extend(d.min(axis=1).tolist())
	return float(np.median(dists)) if dists else float('nan')


def main():
	ap = argparse.ArgumentParser(description="Controlled-fragmentation benchmark for the stitcher.")
	ap.add_argument('--csv', required=True, help="a real *_tracking(_corrected).csv")
	ap.add_argument('--fps', type=float, default=30.0)
	ap.add_argument('--gaps', default="0.5,1,2,3,5,8,12",
					help="gap durations to sweep, in seconds")
	ap.add_argument('--segment-s', type=float, default=10.0,
					help="mean fragment length in seconds")
	ap.add_argument('--repeats', type=int, default=3, help="trials per gap (different cut points)")
	ap.add_argument('--max-gap-s', type=float, default=None,
					help="the stitcher's own max_gap; default = widest swept gap x2, "
						 "so the sweep is not truncated by the setting under test")
	ap.add_argument('--gap-prior-s', type=float, default=5.0)
	ap.add_argument('--extrap-horizon-s', type=float, default=1.0)
	ap.add_argument('--max-speed-px', type=float, default=60.0)
	ap.add_argument('--prior-log-odds', type=float, default=0.0)
	ap.add_argument('--out', default=None, help="write the table as CSV here")
	a = ap.parse_args()

	gaps_s = [float(g) for g in a.gaps.split(',') if g.strip()]
	max_gap_s = a.max_gap_s if a.max_gap_s else max(gaps_s) * 2
	segment_frames = max(8, int(round(a.segment_s * a.fps)))

	tracks, used_corr = load_true_tracks(a.csv, min_len_frames=3 * segment_frames)
	if len(tracks) < 2:
		print(f"Not enough long tracks in {a.csv} to run the benchmark "
			  f"(need >= 2 tracks of {3 * a.segment_s:.0f} s).")
		return 1

	overlap = overlapping_pairs(tracks)
	spacing = neighbour_spacing_px(tracks)

	# A short note that belongs on every printout, not just in the docstring.
	print(f"Oracle benchmark on {os.path.basename(a.csv)}")
	print(f"  pseudo-truth: {len(tracks)} tracker tracks longer than "
		  f"{3 * a.segment_s:.0f} s, coordinates = "
		  f"{'stabilised x_corrected' if used_corr else 'RAW x/y (not drone-stabilised)'}")
	print(f"  median nearest-neighbour spacing: {spacing:.0f} px "
		  f"(the scale sigma(gap) must stay well below to be identifiable)")
	print(f"  fragments of ~{a.segment_s:.0f} s, {a.repeats} trial(s) per gap, "
		  f"stitcher max_gap = {max_gap_s:.0f} s, prior log-odds = {a.prior_log_odds:+.1f}")
	print("  NOTE: this scores the ASSOCIATION step only. The tracker's own id "
		  "swaps are inherited as truth; it is not a substitute for MOT ground "
		  "truth. 'contam' counts only links between tracks that were visible at "
		  "the same time (provably two animals); links between never-simultaneous "
		  "tracks are 'ambig' -- they may be a genuine repair.\n")

	hdr = f"{'gap_s':>6} {'frags':>6} {'cuts':>6} {'cand':>6} {'links':>6} " \
		  f"{'recovery':>9} {'contam':>8} {'ambig':>7} {'nonadj':>7} {'purity':>7}"
	print(hdr); print('-' * len(hdr))
	table = []
	for gs in gaps_s:
		gf = max(1, int(round(gs * a.fps)))
		acc = defaultdict(float)
		n_ok = 0
		for rep in range(a.repeats):
			rng = np.random.default_rng(1000 * rep + gf)
			s = run_trial(tracks, segment_frames, gf, a.fps, rng, max_gap_s,
						  a.gap_prior_s, a.extrap_horizon_s, a.max_speed_px,
						  a.prior_log_odds, overlap)
			if s is None:
				continue
			n_ok += 1
			for k, v in s.items():
				acc[k] += v
		if not n_ok:
			continue
		m = {k: v / n_ok for k, v in acc.items()}
		recovery = m['correct'] / m['cuts'] if m['cuts'] else float('nan')
		contam = m['cross'] / m['links'] if m['links'] else 0.0
		ambig = m['ambiguous'] / m['links'] if m['links'] else 0.0
		nonadj = m['nonadjacent'] / m['links'] if m['links'] else 0.0
		purity = m['pure_chains'] / m['n_chains'] if m['n_chains'] else float('nan')
		row = {'gap_s': gs, 'fragments': m['fragments'], 'cuts': m['cuts'],
			   'candidates': m['candidates'], 'links': m['links'],
			   'recovery': recovery, 'contamination': contam, 'ambiguous': ambig,
			   'nonadjacent_frac': nonadj, 'chain_purity': purity,
			   'beta': m['beta'], 'nu': m['nu'], 'nn_spacing_px': spacing}
		table.append(row)
		print(f"{gs:6.1f} {m['fragments']:6.0f} {m['cuts']:6.0f} {m['candidates']:6.0f} "
			  f"{m['links']:6.0f} {recovery:9.3f} {contam:8.3f} {ambig:7.3f} "
			  f"{nonadj:7.3f} {purity:7.3f}")

	print("\nHow to read it: pick the largest gap where recovery is still worth "
		  "having AND contamination is acceptably low, and set "
		  "stitch_max_gap_seconds there. A missed link costs power; a contaminated "
		  "chain costs validity -- they are not equally bad.")

	if a.out and table:
		with open(a.out, 'w', newline='', encoding='utf-8') as f:
			w = csv.DictWriter(f, fieldnames=list(table[0]))
			w.writeheader(); w.writerows(table)
		print(f"\nTable written to {a.out}")
	return 0


if __name__ == '__main__':
	sys.exit(main())
