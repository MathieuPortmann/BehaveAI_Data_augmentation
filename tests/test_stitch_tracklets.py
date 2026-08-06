#!/usr/bin/env python3
"""Regression tests for the offline tracklet stitcher.

Runnable two ways, so nobody needs a test runner installed to check a change:
    python tests/test_stitch_tracklets.py
    pytest tests/test_stitch_tracklets.py

Each test pins a property that was broken at some point, or that the decision
rule depends on. The first one is the reason this file exists: the original cost
`dist / (max_speed * gap)` shrank as the gap grew, so the linker preferred the
longest, least justified links and merged animals that were minutes and thousands
of pixels apart.
"""

import os
import sys
import csv
import math
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BehaveAI_stitch_tracklets import (extract_tracklets, build_cost, stitch,
									   estimate_motion_noise, occupied_area_px2,
									   check_chains_disjoint, kappa, run_stitch,
									   neighbour_spacing_px)

FPS = 30.0
VEL_WINDOW = 15


# --- helpers ---------------------------------------------------------------

def walk(tid, f0, n, x0, y0, vx, vy, noise=3.0, seed=0):
	"""A tracklet: constant velocity plus localisation noise, one row per frame."""
	rng = np.random.default_rng(seed)
	return [{'frame': str(f0 + i), 'id': tid,
			 'x': f'{x0 + vx * i + rng.normal(0, noise):.2f}',
			 'y': f'{y0 + vy * i + rng.normal(0, noise):.2f}'}
			for i in range(n)]


def link_ids(rows, max_gap_s=60.0, prior_log_odds=0.0, max_speed_px=60.0):
	"""Run the real decision path and return the set of (from_id, to_id) links."""
	tracklets, dropped = extract_tracklets(rows, 'x', 'y', min_len=1,
										   vel_window_frames=VEL_WINDOW)
	order = list(tracklets.keys())
	noise = estimate_motion_noise(tracklets, horizon=FPS, vel_window_frames=VEL_WINDOW)
	cost, diag = build_cost(order, tracklets, max_speed_px,
							int(max_gap_s * FPS), noise['m0_px'], noise['alpha'],
							noise['beta'], noise['nu'], 5.0 * FPS,
							occupied_area_px2(tracklets), FPS,
							quality_gate=False, prior_log_odds=prior_log_odds)
	return {(order[i], order[j]) for i, j in stitch(cost)}, tracklets, order, cost


def _fixed_cost(gap, r=40.0, m0=2.0, alpha=0.5, beta=1.6, nu=2.0, area=1.5e7,
				tau=150.0):
	"""The cost formula alone, for pairs that differ only in gap."""
	s = math.sqrt(m0 ** 2 + alpha * gap ** beta) / kappa(nu)
	return (math.log(2 * math.pi * s * s)
			+ (nu + 2) / 2.0 * math.log1p(r * r / (nu * s * s))
			+ gap / tau - math.log(area))


# --- tests -----------------------------------------------------------------

def test_distant_animal_is_not_linked():
	"""THE regression. A separate animal, 900 frames later and 1300 px away, must
	not be absorbed into the first animal's identity."""
	rows = (walk('1', 0, 60, 100, 200, 8, 0, seed=1)
			+ walk('2', 70, 60, 700, 205, 8, 0, seed=2)      # true continuation
			+ walk('3', 1000, 60, 1800, 900, 0, 0, seed=3))  # different animal
	links, *_ = link_ids(rows)
	assert ('2', '3') not in links and ('1', '3') not in links, links


def test_true_continuation_is_linked():
	"""The counterpart: a genuine short-gap continuation must survive. Without
	this, 'link nothing' would pass every other test in the file."""
	vx, vy, n, gap = 6.0, 1.0, 90, 11
	end_x, end_y = 100 + vx * (n - 1), 200 + vy * (n - 1)
	rows = (walk('1', 0, n, 100, 200, vx, vy, seed=1)
			+ walk('2', n - 1 + gap, n, end_x + vx * gap, end_y + vy * gap,
				   vx, vy, seed=2))
	links, *_ = link_ids(rows)
	assert ('1', '2') in links, links


def test_long_gaps_are_never_free():
	"""The property the old formula violated. Its cost dist/(v_max*gap) fell
	monotonically to zero, so the longest gap was always the cheapest link.

	The correct behaviour is not 'cost always rises with the gap': for a FIXED
	residual a longer gap is genuinely more forgiving, since the animal had more
	time to drift, so the cost dips first. What must hold is that the dip is
	bounded and reverses -- beyond it the cost rises without limit and crosses
	zero, so a long enough gap is always rejected."""
	gaps = [5, 15, 30, 60, 120, 300, 900, 1800, 3600]
	costs = [_fixed_cost(g) for g in gaps]
	k = costs.index(min(costs))
	assert all(b > a for a, b in zip(costs[k:], costs[k + 1:])), costs
	assert costs[-1] > 0, costs          # a very long gap is rejected outright
	assert costs[-1] > costs[0], costs   # and never preferred to a short one


def test_temporal_overlap_is_never_linked():
	rows = (walk('1', 0, 120, 100, 100, 2, 0, seed=1)
			+ walk('2', 60, 120, 110, 105, 2, 0, seed=2))    # overlaps in time
	links, _t, _o, _c = link_ids(rows)
	assert links == set(), links


def test_speed_gate_rejects_the_impossible():
	"""A 5000 px jump in 10 frames is 500 px/frame: beyond any physical gate."""
	rows = (walk('1', 0, 60, 100, 100, 1, 0, seed=1)
			+ walk('2', 70, 60, 5100, 100, 1, 0, seed=2))
	links, *_ = link_ids(rows, max_speed_px=60.0)
	assert links == set(), links


def test_chains_stay_temporally_ordered():
	"""A chain must be a sequence, never two animals seen at once."""
	rows = []
	for k in range(4):
		rows += walk(str(k + 1), k * 200, 150, 100 + 3 * k * 200, 300, 3, 0, seed=k)
	links, tracklets, order, _c = link_ids(rows)
	id_map = {}
	for n, t in enumerate(order):
		id_map[t] = 1                       # deliberately put everything in one chain
	bad = check_chains_disjoint(id_map, tracklets)
	assert bad == [], bad
	# ...and a chain that really does overlap must be caught
	rows2 = (walk('1', 0, 120, 100, 100, 2, 0, seed=1)
			 + walk('2', 60, 120, 900, 900, 2, 0, seed=2))
	tk, _d = extract_tracklets(rows2, 'x', 'y', min_len=1, vel_window_frames=VEL_WINDOW)
	assert check_chains_disjoint({'1': 1, '2': 1}, tk) != []


def test_duplicate_rows_per_frame_are_collapsed():
	"""The static and motion detection streams both write a row for the same
	(frame, id); a tracklet must still hold one sample per frame."""
	rows = walk('1', 0, 40, 100, 100, 2, 0, seed=1)
	rows += [dict(r, x=str(float(r['x']) + 90)) for r in rows]   # motion-stream copy
	tk, _d = extract_tracklets(rows, 'x', 'y', min_len=1, vel_window_frames=VEL_WINDOW)
	assert tk['1']['n'] == 40
	assert len(tk['1']['frames']) == len(set(tk['1']['frames'].tolist()))


def test_short_tracklets_keep_their_rows_with_unique_ids():
	"""Dropping a tracklet from the linking problem must not delete its rows, and
	must not leave it holding a raw id that collides with the compact 1..K ids."""
	rows = (walk('1', 0, 120, 100, 100, 2, 0, seed=1)
			+ walk('2', 200, 120, 500, 100, 2, 0, seed=2)
			+ walk('7', 400, 3, 900, 900, 0, 0, seed=3))        # too short to link
	path = os.path.join(tempfile.mkdtemp(), 'clip_tracking.csv')
	with open(path, 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=['frame', 'id', 'x', 'y'])
		w.writeheader(); w.writerows(rows)
	out, _rep, id_map = run_stitch(path, fps=FPS, min_len=10, verbose=False)
	assert set(id_map) == {'1', '2', '7'}
	assert len(set(id_map.values())) == len(id_map), id_map    # no collision
	with open(out, newline='', encoding='utf-8') as f:
		assert len(list(csv.DictReader(f))) == len(rows)       # nothing deleted


def test_gaussian_limit_recovers_the_gaussian_cost():
	"""nu -> infinity must reduce to the Gaussian negative log-likelihood, so the
	heavy-tailed model can only ever relax the old assumption, never distort it."""
	r, s = 40.0, 25.0
	nu = 1e7
	t = math.log(2 * math.pi * s * s) + (nu + 2) / 2.0 * math.log1p(r * r / (nu * s * s))
	g = math.log(2 * math.pi * s * s) + r * r / (2 * s * s)
	assert abs(t - g) < 1e-3, (t, g)


def test_kappa_matches_the_student_t_median():
	"""kappa(nu) converts a measured median residual into the t scale; check it
	against the closed-form CDF rather than trusting the algebra."""
	for nu in (0.5, 1.0, 2.0, 5.0, 50.0):
		s, m = 1.0, kappa(nu)
		cdf = 1.0 - (1.0 + m * m / (nu * s * s)) ** (-nu / 2.0)
		assert abs(cdf - 0.5) < 1e-9, (nu, cdf)


def test_run_is_deterministic():
	rows = []
	for k in range(6):
		rows += walk(str(k + 1), k * 90, 80, 100 + 40 * k, 100 + 30 * k, 2, 1, seed=k)
	a, *_ = link_ids(rows)
	b, *_ = link_ids(rows)
	assert a == b


def test_neighbour_spacing_is_measured_between_animals():
	"""Two animals 500 px apart, tracked simultaneously."""
	rows = walk('1', 0, 60, 0, 0, 0, 0, noise=0.0, seed=1) + \
		walk('2', 0, 60, 500, 0, 0, 0, noise=0.0, seed=2)
	tk, _d = extract_tracklets(rows, 'x', 'y', min_len=1, vel_window_frames=VEL_WINDOW)
	sp = neighbour_spacing_px((t['frames'], t['xs'], t['ys']) for t in tk.values())
	assert abs(sp - 500.0) < 1.0, sp


if __name__ == '__main__':
	fails = 0
	for name, fn in sorted(globals().items()):
		if name.startswith('test_') and callable(fn):
			try:
				fn()
				print(f"  PASS  {name}")
			except AssertionError as e:
				fails += 1
				print(f"  FAIL  {name}: {e}")
	print(f"\n{'all tests passed' if not fails else f'{fails} test(s) failed'}")
	sys.exit(1 if fails else 0)
