#!/usr/bin/env python3
"""BehaveAI offline tracklet stitching

The online tracker (BoT-SORT/ByteTrack, or the legacy Kalman) is causal: once it
cuts a track or swaps an id it never revisits the decision. But the video is on
disk, so we can read it whole and re-link the short, reliable tracklets it
produced into longer identities -- purely on kinematics, no appearance.

Reads a tracking CSV (prefers the drone-corrected one, whose x_corrected/
y_corrected live in a single stabilised reference frame comparable across the
whole clip), groups rows into tracklets by id, and links compatible end->start
pairs.

Decision rule (v2)
------------------
Two HARD gates make the problem safe. They are physics, not tuning:
  * two tracklets overlapping in TIME are different animals (never linked);
  * a displacement above `max_speed` * gap is impossible (never linked). With a
    flight log the gate is a real m/s bound converted to pixels (see
    resolve_speed_gate); without one it falls back to a pixel cap.
  * a gap longer than `max_gap` is not considered at all (see below).

The SOFT decision is a likelihood ratio between two explicit hypotheses, not a
tuned threshold:

  H_link : B is the continuation of A. The residual r between A's constant-velocity
           prediction and B's start follows a 2-D Student-t of scale s(gap) and
           nu degrees of freedom, and the occlusion duration has an exponential
           prior with time constant tau.
  H_new  : B is an unrelated track that appeared anywhere in the region the herd
           occupies, i.e. uniform over that area.

  cost = -log P(H_link) + log P(H_new)
       = log(2 pi s^2) + (nu+2)/2 * log(1 + r^2/(nu s^2)) + gap/tau
         - log(area) - prior_log_odds

and a link is taken when cost < 0, i.e. only when the continuation hypothesis is
actually more likely than "a new animal showed up". There is no arbitrary
`max_link_cost` any more, and -- crucially -- the cost now GROWS with the gap
(through log s^2 and gap/tau), where the v1 cost dist/(v_max*gap) shrank with it
and therefore preferred the longest, least justified links.

Nothing in that formula is hand-set. `estimate_motion_noise()` measures s(gap)
AND nu per clip from the tracker's own tracklets, by running the very predictor
used for linking at several lags and fitting the observed error distribution. The
tail matters: the residual is strongly non-Gaussian (median 3 px at half a second
against a 90th percentile of 160 px), and a Gaussian fitted to that core rejects
42 % of genuine re-links. Everything the run used is written to the JSON report,
so a result can be audited.

Known limit, measured not assumed. The oracle benchmark below shows the decisive
factor is how tightly packed the herd is: 1-3 % contamination on a clip whose
animals sit 441 px apart, 29 % on one at 198 px, at every setting tried. When the
spacing is small, kinematics alone cannot identify the continuation and no
threshold fixes it -- the report prints the spacing so the failure mode is
visible rather than silent.

K (group size) is NOT a constraint -- only a reported diagnostic.

Outputs: <video>_tracking_stitched.csv (id remapped), <video>_stitch_report.txt
and <video>_stitch_report.json.

Calibrating max_gap: run BehaveAI_stitch_oracle.py, which cuts real trajectories
at controlled gap lengths and measures recovery vs contamination as a function of
gap. The setting should come from that curve, not from taste.

Usage:
  python BehaveAI_stitch_tracklets.py <project_dir | settings.ini>
  python BehaveAI_stitch_tracklets.py --csv <one_tracking_corrected.csv>
"""

import os
import csv
import sys
import glob
import json
import math
import hashlib
import argparse
import configparser

from behaveai_config import resolve_project_dir
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

BIG = 1e9  # never np.inf: scipy's linear_sum_assignment rejects an all-inf matrix

# Maximum sustained speed of a galloping horse, used as the physical gate when a
# flight log gives the pixel scale. Deliberately generous: the gate only has to
# exclude the impossible, the likelihood ratio does the discrimination.
HORSE_MAX_SPEED_M_S = 17.0

# Fallbacks for the motion-noise model when a clip has too little data to measure
# it (short tracklets only). Order of magnitude from the two HERDWISE drone clips;
# the report always states whether the values were measured or defaulted.
DEFAULT_M0_PX = 1.0           # median residual at lag 0
DEFAULT_ALPHA = 0.3           # px^2 per frame^beta
DEFAULT_BETA = 1.7            # measured growth exponent, NOT the 3 of a free CV model
DEFAULT_NU = 1.0              # Student-t d.o.f.; heavy-tailed, as measured


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


def _lsq_velocity(frames, xs, ys):
	"""Least-squares slope (px per frame) over the given samples. More robust than
	an endpoint difference, which a single bad box can dominate."""
	if len(frames) < 2:
		return np.array([0.0, 0.0])
	f = frames - frames.mean()
	denom = float((f * f).sum())
	if denom <= 0:
		return np.array([0.0, 0.0])
	return np.array([float((f * (xs - xs.mean())).sum() / denom),
					 float((f * (ys - ys.mean())).sum() / denom)])


def _window_slice(frames, at_start, window_frames):
	"""Index slice covering `window_frames` frames from the start or the end."""
	if at_start:
		lim = frames[0] + window_frames
		n = int(np.searchsorted(frames, lim, side='right'))
		return slice(0, max(2, n))
	lim = frames[-1] - window_frames
	i = int(np.searchsorted(frames, lim, side='left'))
	return slice(min(i, max(0, len(frames) - 2)), len(frames))


def extract_tracklets(rows, xk, yk, min_len=1, vel_window_frames=15):
	"""Group rows by id -> tracklet dict.

	Two rows can share a (frame, id): the static and the motion detection stream
	both feed the same track and their box centres differ by tens of pixels. They
	are collapsed to the per-frame median position, so a tracklet holds exactly
	one sample per frame and its velocity is well defined. The jitter this
	introduces is not swept under the rug -- it is part of what
	estimate_motion_noise() measures as part of the residual scale.

	Returns (tracklets, dropped_ids). Tracklets shorter than min_len samples are
	dropped from the linking problem; their rows are kept in the output and given
	their own fresh identity, never silently deleted.
	"""
	by_id = defaultdict(lambda: defaultdict(list))
	quality = defaultdict(dict)
	for r in rows:
		try:
			fr = int(r['frame'])
			x = float(r[xk]); y = float(r[yk])
		except (ValueError, KeyError, TypeError):
			continue
		tid = str(r['id'])
		by_id[tid][fr].append((x, y))
		quality[tid][fr] = r.get('correction_quality', 'ok')

	tracklets, dropped = {}, []
	for tid, per_frame in by_id.items():
		fr_sorted = sorted(per_frame)
		if len(fr_sorted) < max(1, min_len):
			dropped.append(tid)
			continue
		frames = np.array(fr_sorted, dtype=float)
		pts = np.array([np.median(np.array(per_frame[f], dtype=float), axis=0)
						for f in fr_sorted], dtype=float)
		xs, ys = pts[:, 0], pts[:, 1]
		tracklets[tid] = {
			'id': tid,
			'frames': frames, 'xs': xs, 'ys': ys,
			'start_frame': int(frames[0]), 'end_frame': int(frames[-1]),
			'start_pos': np.array([xs[0], ys[0]]),
			'end_pos': np.array([xs[-1], ys[-1]]),
			# Both velocities point FORWARD in time: start_vel is used to run B
			# backwards towards A (start_pos - start_vel * h).
			'start_vel': _lsq_velocity(*_slice3(frames, xs, ys,
												_window_slice(frames, True, vel_window_frames))),
			'end_vel': _lsq_velocity(*_slice3(frames, xs, ys,
											  _window_slice(frames, False, vel_window_frames))),
			'start_q': quality[tid][fr_sorted[0]],
			'end_q': quality[tid][fr_sorted[-1]],
			'n': len(fr_sorted),
		}
	return tracklets, dropped


def _slice3(frames, xs, ys, sl):
	return frames[sl], xs[sl], ys[sl]


# ---------------------------------------------------------------------------
# Motion-noise model, measured per clip
# ---------------------------------------------------------------------------

def _predict(frames, xs, ys, i, lag, horizon, vel_window_frames):
	"""Constant-velocity prediction `lag` frames past sample i, using exactly the
	predictor build_cost() uses (same window, same damped horizon)."""
	f0 = frames[i]
	lo = int(np.searchsorted(frames, f0 - vel_window_frames, side='left'))
	if i - lo < 1:
		return None
	v = _lsq_velocity(frames[lo:i + 1], xs[lo:i + 1], ys[lo:i + 1])
	h = min(lag, horizon)
	return np.array([xs[i], ys[i]]) + v * h


def estimate_motion_noise(tracklets, lags=(1, 2, 5, 10, 20, 50, 100, 200),
						  horizon=30.0, vel_window_frames=15,
						  max_samples_per_lag=4000, min_samples_per_lag=30):
	"""Measure how fast the constant-velocity prediction degrades, in this clip.

	For each lag L we run the linking predictor inside tracklets, where the answer
	is known, and record the per-axis squared error. Because the measurement uses
	the same predictor as the decision, velocity-estimation noise and per-frame
	localisation noise are included by construction -- no correction factor.

	Two things are fitted, both from the data rather than assumed:

	1. SCALE. m(L), the median residual, follows m^2 = m0^2 + alpha * L^beta,
	   fitted in log space so every decade of lag counts equally. beta is fitted,
	   not fixed at the 3 that a constant-velocity model driven by white-noise
	   acceleration predicts: on real herd clips it measures ~1.6-1.8, because
	   grazing horses' displacement is bounded rather than a velocity random walk.
	   Forcing beta = 3 would inflate the spread at long gaps and make long links
	   look far more plausible than they are.

	2. TAIL. The residual is NOT Gaussian -- on the HERDWISE clips the median is
	   3 px at half a second while the 90th percentile is 160 px, three orders of
	   magnitude of spread. A Gaussian fitted to that core is wildly overconfident
	   and rejects most genuine re-links (measured: 42 % of true continuations
	   scored as impossible). The residual is therefore modelled as a 2-D Student-t
	       p(x) = 1/(2 pi s^2) * (1 + |x|^2/(nu s^2))^(-(nu+2)/2)
	   whose degrees of freedom nu are fitted by pooling scale-normalised residuals
	   across lags. nu -> infinity recovers the Gaussian, so this only ever relaxes
	   an assumption; the fitted value is reported.

	Returns a dict with m0_px/alpha/beta (scale law), nu, the per-lag measurements
	and the provenance. Lags beyond the fitted range are an extrapolation, which
	the report states.
	"""
	per_lag, pooled_z = [], []
	rng = np.random.default_rng(0)   # deterministic subsampling
	for L in lags:
		errs = []
		for t in tracklets.values():
			frames, xs, ys = t['frames'], t['xs'], t['ys']
			if len(frames) < 4:
				continue
			targets = np.searchsorted(frames, frames + L, side='left')
			cand = np.nonzero((targets < len(frames)) &
							  (frames[np.minimum(targets, len(frames) - 1)] == frames + L))[0]
			if cand.size == 0:
				continue
			if cand.size > 200:
				cand = rng.choice(cand, 200, replace=False)
			for i in cand:
				p = _predict(frames, xs, ys, int(i), L, horizon, vel_window_frames)
				if p is None:
					continue
				j = int(targets[i])
				errs.append(float(np.hypot(p[0] - xs[j], p[1] - ys[j])))
			if len(errs) >= max_samples_per_lag:
				break
		if len(errs) >= min_samples_per_lag:
			e = np.asarray(errs)
			m = float(np.median(e))
			if m > 0:
				pooled_z.append(e / m)
			per_lag.append({'lag': int(L), 'n': len(errs), 'median_px': m,
							'p90_px': float(np.percentile(e, 90))})

	if len(per_lag) < 3:
		return {'m0_px': DEFAULT_M0_PX, 'alpha': DEFAULT_ALPHA, 'beta': DEFAULT_BETA,
				'nu': DEFAULT_NU, 'rms_log_residual': None,
				'source': 'default (too little within-tracklet data to measure)',
				'per_lag': per_lag, 'fitted_lag_max': 0}

	L = np.array([p['lag'] for p in per_lag], dtype=float)
	y = np.array([p['median_px'] ** 2 for p in per_lag], dtype=float)

	def _law(m0sq, alpha, beta, LL):
		return m0sq + alpha * LL ** beta

	try:
		from scipy.optimize import curve_fit

		def _f(LL, log_m0sq, log_alpha, beta):
			return np.log(np.exp(log_m0sq) + np.exp(log_alpha) * LL ** beta)

		p0 = [math.log(max(y[0], 0.1)), math.log(max(y[-1] / L[-1] ** 1.5, 1e-6)), 1.5]
		popt, _ = curve_fit(_f, L, np.log(y), p0=p0,
							bounds=([-10.0, -30.0, 0.5], [12.0, 10.0, 3.0]),
							maxfev=20000)
		m0sq, alpha, beta = math.exp(popt[0]), math.exp(popt[1]), float(popt[2])
		rms = float(np.sqrt(((np.log(y) - _f(L, *popt)) ** 2).mean()))
		source = 'measured from this clip'
	except Exception:
		A = np.column_stack([np.ones_like(L), L ** DEFAULT_BETA])
		coef, *_ = np.linalg.lstsq(A, y, rcond=None)
		m0sq, alpha, beta = max(float(coef[0]), 0.01), max(float(coef[1]), 1e-9), DEFAULT_BETA
		rms = float(np.sqrt(((np.log(y) - np.log(_law(m0sq, alpha, beta, L))) ** 2).mean()))
		source = f'measured from this clip (fixed beta={DEFAULT_BETA})'

	nu = fit_tail_nu(np.concatenate(pooled_z)) if pooled_z else DEFAULT_NU
	return {'m0_px': math.sqrt(max(m0sq, 1e-6)), 'alpha': alpha, 'beta': beta,
			'nu': nu, 'rms_log_residual': rms, 'source': source,
			'per_lag': per_lag, 'fitted_lag_max': int(L.max())}


def kappa(nu):
	"""Median radius of a 2-D Student-t of unit scale: r_median = s * kappa(nu).
	Lets the robust median of the residuals set the scale s directly."""
	return math.sqrt(nu * (2.0 ** (2.0 / nu) - 1.0))


def fit_tail_nu(z, grid=(0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 200.0)):
	"""Degrees of freedom of the 2-D Student-t, by grid-search maximum likelihood
	on residuals normalised by their own per-lag median (so every lag contributes
	to the SHAPE and none to the scale). nu = 200 is Gaussian for all practical
	purposes, so the grid contains the null hypothesis it might have to accept."""
	z = np.asarray(z, dtype=float)
	z = z[np.isfinite(z) & (z > 0)]
	if z.size < 100:
		return DEFAULT_NU
	if z.size > 200000:
		z = np.random.default_rng(0).choice(z, 200000, replace=False)
	best, best_ll = DEFAULT_NU, -np.inf
	for nu in grid:
		s = 1.0 / kappa(nu)                       # unit median by construction
		# radial log-density: log[(r/s^2) (1+r^2/(nu s^2))^(-(nu+2)/2)]
		ll = float(np.sum(np.log(z) - 2 * math.log(s)
						  - (nu + 2) / 2.0 * np.log1p(z ** 2 / (nu * s * s))))
		if ll > best_ll:
			best_ll, best = ll, nu
	return best


# ---------------------------------------------------------------------------
# Physical speed gate
# ---------------------------------------------------------------------------

def resolve_speed_gate(default_px_per_frame, flightlog=None, f_px=None, fps=None,
					   max_speed_m_s=HORSE_MAX_SPEED_M_S, margin=1.5):
	"""Convert a physical m/s bound into the px/frame gate, when the geometry is
	known; otherwise return the configured pixel cap.

	For a ground target the image scale is largest at nadir and smaller
	everywhere else (it falls as sin(depression angle)), so px_per_m <= f_px / H
	over the whole frame. Taking the MINIMUM camera height of the clip therefore
	gives a bound that is conservative in the safe direction: the gate can only
	be too permissive, never wrongly reject a real link -- which is what a hard
	gate should be, since the likelihood ratio does the discrimination.

	`margin` covers the known bias of barometric rel_alt (referenced to the
	take-off point, so sloped terrain or a raised take-off spot inflates H).

	Returns (px_per_frame, provenance_string).
	"""
	if not flightlog or f_px is None or not fps:
		return default_px_per_frame, f'configured pixel cap ({default_px_per_frame} px/frame)'
	alt = flightlog.get('rel_alt_m')
	if alt is None:
		return default_px_per_frame, f'configured pixel cap ({default_px_per_frame} px/frame)'
	a = np.asarray(alt, dtype=float)
	a = a[np.isfinite(a) & (a > 0)]
	if a.size == 0:
		return default_px_per_frame, f'configured pixel cap ({default_px_per_frame} px/frame)'
	h_min = float(np.percentile(a, 5))     # robust minimum height
	px_per_m = f_px / h_min
	gate = max_speed_m_s * px_per_m / float(fps) * margin
	return gate, (f'physical: {max_speed_m_s:.0f} m/s at {px_per_m:.1f} px/m '
				  f'(f_px={f_px:.0f}, H={h_min:.1f} m, fps={fps:.1f}, margin x{margin:g})')


# ---------------------------------------------------------------------------
# Cost + solve
# ---------------------------------------------------------------------------

def neighbour_spacing_px(series, n_frames=300, rng=None):
	"""Median nearest-neighbour distance between animals visible at the same time.

	`series` is an iterable of (frames, xs, ys) arrays. This is the scale that
	decides whether kinematic linking can work at all: the prediction residual has
	to stay well under it, or the right continuation is indistinguishable from its
	neighbours and no threshold repairs that. On the two HERDWISE clips the
	oracle benchmark measured 1-3 % contamination at 441 px spacing and 29 % at
	198 px -- so a low value here is a warning sign, not a detail.
	"""
	rng = rng or np.random.default_rng(0)
	pos_by_frame = defaultdict(list)
	for frames, xs, ys in series:
		for f, x, y in zip(frames, xs, ys):
			pos_by_frame[int(f)].append((x, y))
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


def occupied_area_px2(tracklets):
	"""Area of the region the herd actually occupies, used as the alternative
	hypothesis' uniform density. The bounding box of all observed positions is
	both more honest and more conservative than the full frame: a new animal is
	assumed to appear where animals are, which makes 'new track' a stronger
	competitor and linking harder."""
	if not tracklets:
		return 1.0
	xs = np.concatenate([t['xs'] for t in tracklets.values()])
	ys = np.concatenate([t['ys'] for t in tracklets.values()])
	w = max(float(xs.max() - xs.min()), 1.0)
	h = max(float(ys.max() - ys.min()), 1.0)
	return w * h


def build_cost(order, tracklets, max_speed_px, max_gap_frames, m0_px, alpha,
			   beta, nu, tau_frames, area_px2, horizon_frames, quality_gate=True,
			   prior_log_odds=0.0):
	"""Cost matrix rows=tracklet ENDS, cols=tracklet STARTS (same tracklet list).

	Entry = -log[ P(continuation) / P(new track) ]; BIG for pairs excluded by a
	hard gate. Negative means the continuation hypothesis wins.

	Also returns a diagnostic dict: how many pairs each gate rejected, and the
	cost distribution of the survivors. A run that links nothing should say
	whether it saw no candidates at all or judged them all implausible.
	"""
	n = len(order)
	C = np.full((n, n), BIG, dtype=float)
	log_area = math.log(max(area_px2, 1.0))
	k_nu = kappa(nu)                  # median residual -> Student-t scale
	rej = {'temporal_overlap': 0, 'gap_too_long': 0, 'quality': 0, 'unreachable': 0}
	costs = []
	for i, a_id in enumerate(order):          # A ends
		A = tracklets[a_id]
		for j, b_id in enumerate(order):      # B starts
			if i == j:
				continue
			B = tracklets[b_id]
			gap = B['start_frame'] - A['end_frame']
			if gap <= 0:
				rej['temporal_overlap'] += 1
				continue
			if gap > max_gap_frames:
				rej['gap_too_long'] += 1
				continue
			if quality_gate and (A['end_q'] == 'none' or B['start_q'] == 'none'):
				rej['quality'] += 1
				continue
			step = float(np.hypot(*(B['start_pos'] - A['end_pos'])))
			if step > max_speed_px * gap:     # physically unreachable
				rej['unreachable'] += 1
				continue
			h = min(float(gap), horizon_frames)
			r_fwd = np.hypot(*(A['end_pos'] + A['end_vel'] * h - B['start_pos']))
			r_bwd = np.hypot(*(B['start_pos'] - B['start_vel'] * h - A['end_pos']))
			r = 0.5 * (float(r_fwd) + float(r_bwd))
			# 2-D Student-t: -log p(r) = log(2 pi s^2) + (nu+2)/2 * log(1 + r^2/(nu s^2))
			s = math.sqrt(m0_px ** 2 + alpha * float(gap) ** beta) / k_nu
			nll_link = (math.log(2.0 * math.pi * s * s)
						+ (nu + 2.0) / 2.0 * math.log1p(r * r / (nu * s * s))
						+ gap / tau_frames)
			C[i, j] = nll_link - log_area - prior_log_odds
			costs.append(C[i, j])
	diag = {'pairs_total': n * (n - 1), 'rejected': rej, 'candidates': len(costs),
			'candidates_favourable': int(sum(1 for c in costs if c < 0)),
			'best_cost': round(float(min(costs)), 3) if costs else None,
			'median_cost': round(float(np.median(costs)), 3) if costs else None}
	return C, diag


def stitch(cost):
	"""Dummy-augmented linear assignment. The opt-out costs 0 because the cost
	matrix is already a log-likelihood RATIO against 'this is a new track': a link
	is taken exactly when it beats that alternative. Returns [(end_i, start_j)]."""
	n = cost.shape[0]
	if n == 0:
		return []
	C = np.full((2 * n, 2 * n), BIG, dtype=float)
	C[:n, :n] = np.where(np.isfinite(cost), cost, BIG)
	np.fill_diagonal(C[:n, n:], 0.0)   # end i unmatched
	np.fill_diagonal(C[n:, :n], 0.0)   # start j unmatched
	C[n:, n:] = 0.0                    # dummy<->dummy free
	rows, cols = linear_sum_assignment(C)
	return [(r, c) for r, c in zip(rows, cols) if r < n and c < n and cost[r, c] < 0.0]


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


def check_chains_disjoint(id_map, tracklets):
	"""A chain must be a temporally ordered sequence: no two tracklets sharing a
	chain may overlap in time. The assignment cannot produce such a chain, so a
	violation means a bug -- assert it rather than trust it."""
	by_chain = defaultdict(list)
	for tid, cid in id_map.items():
		if tid in tracklets:
			by_chain[cid].append(tracklets[tid])
	bad = []
	for cid, ts in by_chain.items():
		ts = sorted(ts, key=lambda t: t['start_frame'])
		for a, b in zip(ts, ts[1:]):
			if b['start_frame'] <= a['end_frame']:
				bad.append((cid, a['id'], b['id']))
	return bad


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


def _write_report(path, tracklets, id_map, expected_group_size, meta, links_info,
				  dropped, overlaps):
	med, p95, mx = _concurrency_stats(tracklets)
	n_chains = len(set(id_map.values()))
	gaps = [l['gap'] for l in links_info]
	g = meta['gates']
	lines = [
		"BehaveAI tracklet-stitch report",
		f"tracklets in: {len(tracklets)}"
		+ (f"  (+{len(dropped)} below min length, kept as their own identities)" if dropped else ""),
		f"links taken: {len(links_info)}",
		f"chains out (distinct stitched ids): {n_chains}",
		f"concurrent tracklets/frame: median={med} p95={p95} max={mx}",
		f"expected_group_size: {expected_group_size if expected_group_size else 'unknown'}",
		"",
		"Decision model:",
		f"  speed gate: {meta['speed_gate_source']}",
		f"  max gap: {meta['max_gap_frames']} frames ({meta['max_gap_s']:.1f} s)",
		f"  residual model: median^2(gap) = {meta['m0_px']:.1f}^2 + "
		f"{meta['alpha']:.3g} * gap^{meta['beta']:.2f}, Student-t nu={meta['nu']:g} "
		f"[{meta['noise_source']}]",
		f"  occlusion prior tau: {meta['tau_frames']:.0f} frames ({meta['tau_s']:.1f} s)",
		f"  new-track area: {meta['area_px2']:.3g} px^2",
		f"  prior log-odds: {meta['prior_log_odds']:+.2f}",
		"",
		"Identifiability of this clip:",
		f"  median nearest-neighbour spacing: {meta['nn_spacing_px']:.0f} px",
		f"  median residual at max gap:       {meta['median_residual_at_max_gap_px']:.0f} px",
		"  For reference, the oracle benchmark measured 1-3 % contamination on a "
		"clip with 441 px spacing and 29 % on one with 198 px. Tight packing, not "
		"the settings, is what limits kinematic linking -- run "
		"BehaveAI_stitch_oracle.py on this clip before trusting its links.",
		"",
		"Candidate pairs (why links were or were not taken):",
		f"  {g['pairs_total']} ordered pairs -> {g['candidates']} passed the hard gates "
		f"({g['candidates_favourable']} of them favourable, cost < 0)",
		"  rejected by: " + ", ".join(f"{k}={v}" for k, v in g['rejected'].items()),
	]
	if g['candidates']:
		lines.append(f"  cost of the candidates: best={g['best_cost']} median={g['median_cost']} "
					 "(negative = continuation beats 'a new animal appeared')")
	elif len(tracklets) > 1:
		lines.append("  no candidate at all: the tracklets do not overlap in a way that could "
					 "be linked (check max_gap, or the tracker simply did not fragment).")
	if meta['noise_fitted_lag_max'] and meta['max_gap_frames'] > meta['noise_fitted_lag_max']:
		lines.append(f"  NOTE: gaps beyond {meta['noise_fitted_lag_max']} frames extrapolate the "
					 "fitted noise model (no within-tracklet data that long).")
	if gaps:
		lines.append(f"  gaps of the links taken: median={np.median(gaps):.0f} "
					 f"max={max(gaps)} frames")
	lines += [
		"",
		"Reading guide:",
		f"  - n_chains ({n_chains}) < max concurrent ({mx}) would be logically "
		"impossible (two simultaneous tracklets are different animals).",
	]
	warn = bool(overlaps)
	if overlaps:
		lines.append(f"  !! BUG: {len(overlaps)} chain(s) contain temporally overlapping "
					 f"tracklets, e.g. {overlaps[0]}.")
	if n_chains < mx:
		warn = True
		lines.append("  !! WARNING: n_chains < max concurrent -> non-overlap constraint bug.")
	if mx and n_chains > 3 * mx:
		lines.append("  - n_chains >> concurrent: residual fragmentation (unrecovered splits).")
	if expected_group_size and mx > expected_group_size:
		lines.append("  - max concurrent > expected_group_size: field count wrong, an outside "
					 "individual is present, or detection false positives.")
	with open(path, 'w', encoding='utf-8') as f:
		f.write('\n'.join(lines) + '\n')

	with open(path.replace('.txt', '.json'), 'w', encoding='utf-8') as f:
		json.dump({'counts': {'tracklets': len(tracklets), 'dropped': len(dropped),
							  'links': len(links_info), 'chains': n_chains},
				   'concurrency': {'median': med, 'p95': p95, 'max': mx},
				   'expected_group_size': expected_group_size or None,
				   'params': meta, 'links': links_info,
				   'overlap_violations': overlaps}, f, indent=2)
	return warn, (med, p95, mx), n_chains


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _sha1(path):
	h = hashlib.sha1()
	with open(path, 'rb') as f:
		for chunk in iter(lambda: f.read(1 << 20), b''):
			h.update(chunk)
	return h.hexdigest()[:12]


def run_stitch(csv_path, out_path=None, max_speed_px=60.0, fps=30.0,
			   max_gap_s=5.0, gap_prior_s=5.0, extrap_horizon_s=1.0,
			   min_len=1, expected_group_size=0, quality_gate=True,
			   prior_log_odds=-5.0, flightlog=None, f_px=None,
			   max_speed_m_s=HORSE_MAX_SPEED_M_S, speed_gate_margin=1.5,
			   verbose=True):
	"""Stitch one tracking CSV. Returns (out_csv, report_path, id_map)."""
	rows, fieldnames = _read_rows(csv_path)
	if not rows:
		if verbose:
			print(f"  {os.path.basename(csv_path)}: no rows, skipping.")
		return None, None, {}
	xk, yk, used_corr = _pos_keys(fieldnames)
	fps = float(fps) if fps and fps > 0 else 30.0
	vel_window = max(2, int(round(0.5 * fps)))
	tracklets, dropped = extract_tracklets(rows, xk, yk, min_len=min_len,
										   vel_window_frames=vel_window)
	order = list(tracklets.keys())

	max_gap_frames = max(1, int(round(max_gap_s * fps)))
	tau_frames = max(1.0, gap_prior_s * fps)
	horizon_frames = max(1.0, extrap_horizon_s * fps)
	gate_px, gate_src = resolve_speed_gate(max_speed_px, flightlog, f_px, fps,
										   max_speed_m_s, speed_gate_margin)
	noise = estimate_motion_noise(tracklets, horizon=horizon_frames,
								  vel_window_frames=vel_window)
	area = occupied_area_px2(tracklets)
	spacing = neighbour_spacing_px((t['frames'], t['xs'], t['ys'])
								   for t in tracklets.values())
	median_resid_at_max_gap = math.sqrt(noise['m0_px'] ** 2 +
										noise['alpha'] * float(max_gap_frames) ** noise['beta'])

	cost, gate_diag = build_cost(order, tracklets, gate_px, max_gap_frames,
								 noise['m0_px'], noise['alpha'], noise['beta'],
								 noise['nu'], tau_frames, area, horizon_frames,
								 quality_gate, prior_log_odds)
	links = stitch(cost)

	uf = _UnionFind(order)
	links_info = []
	for i, j in links:
		a, b = order[i], order[j]
		uf.union(a, b)   # chain A-end -> B-start
		links_info.append({'from': a, 'to': b,
						   'gap': int(tracklets[b]['start_frame'] - tracklets[a]['end_frame']),
						   'cost': round(float(cost[i, j]), 3)})
	# Map each original id -> compact chain id (1..K), stable by first appearance.
	rep_to_new, id_map, next_id = {}, {}, 1
	for tid in sorted(order, key=lambda t: (tracklets[t]['start_frame'], t)):
		rep = uf.find(tid)
		if rep not in rep_to_new:
			rep_to_new[rep] = next_id
			next_id += 1
		id_map[tid] = rep_to_new[rep]
	# Tracklets excluded from linking keep their rows but get fresh ids ABOVE the
	# chains: reusing their original id would collide with the compact 1..K space.
	for tid in sorted(dropped):
		id_map[tid] = next_id
		next_id += 1

	overlaps = check_chains_disjoint(id_map, tracklets)

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

	meta = {
		'source_csv': os.path.basename(csv_path), 'source_sha1': _sha1(csv_path),
		'coordinates': 'x_corrected/y_corrected' if used_corr else 'raw x/y (NOT drone-stabilised)',
		'fps': fps, 'speed_gate_px_per_frame': gate_px, 'speed_gate_source': gate_src,
		'max_gap_frames': max_gap_frames, 'max_gap_s': max_gap_s,
		'tau_frames': tau_frames, 'tau_s': gap_prior_s,
		'extrap_horizon_frames': horizon_frames, 'extrap_horizon_s': extrap_horizon_s,
		'm0_px': noise['m0_px'], 'alpha': noise['alpha'], 'beta': noise['beta'],
		'nu': noise['nu'],
		'noise_source': noise['source'], 'noise_per_lag': noise['per_lag'],
		'noise_fitted_lag_max': noise['fitted_lag_max'],
		'noise_rms_log_residual': noise['rms_log_residual'],
		'area_px2': area, 'prior_log_odds': prior_log_odds,
		'min_tracklet_len': min_len, 'quality_gate': quality_gate,
		'nn_spacing_px': spacing,
		'median_residual_at_max_gap_px': median_resid_at_max_gap,
		'gates': gate_diag,
	}
	report_path = out_path.replace('_tracking_stitched.csv', '_stitch_report.txt')
	warn, dist, n_chains = _write_report(report_path, tracklets, id_map,
										 expected_group_size, meta, links_info,
										 dropped, overlaps)
	if verbose:
		src = 'corrected' if used_corr else 'raw (WARNING: not drone-stabilised)'
		print(f"  {os.path.basename(csv_path)} [{src}]: {len(tracklets)} tracklets, "
			  f"{len(links)} links -> {n_chains} chains "
			  f"(concurrent med/p95/max = {dist[0]}/{dist[1]}/{dist[2]})"
			  + ("  !!WARN" if warn else ""))
		print(f"    gate {gate_px:.1f} px/frame [{gate_src}]; "
			  f"median^2(gap) = {noise['m0_px']:.1f}^2 + {noise['alpha']:.3g}"
			  f"*gap^{noise['beta']:.2f}, nu={noise['nu']:g} [{noise['source']}]; "
			  f"{gate_diag['candidates']} candidate pairs, "
			  f"{gate_diag['candidates_favourable']} favourable")
	return out_path, report_path, id_map


def _load_cfg(config_path):
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	if d.get('stitch_max_link_cost') is not None:
		print("  NOTE: stitch_max_link_cost is obsolete and ignored. The link "
			  "decision is now a likelihood ratio against 'this is a new track' "
			  "(threshold 0); use stitch_link_prior_log_odds to make it stricter "
			  "(negative) or looser (positive).")
	return {
		'enabled': str(d.get('stitch_enabled', 'false')).lower() == 'true',
		'max_speed_px': float(d.get('stitch_max_speed_px_per_frame', '60')),
		'max_speed_m_s': float(d.get('stitch_max_speed_m_per_s', str(HORSE_MAX_SPEED_M_S))),
		'speed_gate_margin': float(d.get('stitch_speed_gate_margin', '1.5')),
		'max_gap_s': float(d.get('stitch_max_gap_seconds', '5')),
		'gap_prior_s': float(d.get('stitch_gap_prior_seconds', '5')),
		'extrap_horizon_s': float(d.get('stitch_extrapolation_horizon_seconds', '1')),
		'prior_log_odds': float(d.get('stitch_link_prior_log_odds', '-5')),
		'min_len': int(d.get('stitch_min_tracklet_len', '1')),
		'expected_group_size': int(d.get('expected_group_size', '0') or '0'),
		'quality_gate': str(d.get('stitch_quality_gate', 'true')).lower() == 'true',
	}


def _clip_geometry(config_path, stem):
	"""(fps, f_px, flightlog) for a stem, or (30.0, None, None) when the metric
	stack cannot resolve them. Never fatal: without a flight log the stitcher
	falls back to the configured pixel gate."""
	try:
		from behaveai_drone.metric_geometry import (load_metric_config, resolve_fpx,
													build_video_index, find_video_for_stem,
													video_fps_for_stem, video_dims_fps,
													_media_dirs, _build_index, _find_by_stem)
		from behaveai_drone.flightlog import load_flightlog
	except ImportError:
		return 30.0, None, None
	try:
		index = build_video_index(config_path)
		fps = video_fps_for_stem(config_path, stem, default=30.0, video_index=index)
		video = find_video_for_stem(stem, index)
		if not video:
			return fps, None, None
		# Same lookup as run_metric_geometry: <stem>.flightlog.csv beside the
		# video, in the input dir or the offline-clips dir.
		input_dir, clips_dir = _media_dirs(config_path)
		log_index = _build_index(input_dir, ('.flightlog.csv',))
		for fname, path in _build_index(clips_dir, ('.flightlog.csv',)).items():
			log_index.setdefault(fname, path)
		fl_path = _find_by_stem(stem, log_index, ('.flightlog.csv',))
		if not fl_path:
			return fps, None, None
		flightlog = load_flightlog(fl_path)
		dims = video_dims_fps(video)
		if not dims:
			return fps, None, None
		width = dims[0]
		params = load_metric_config(config_path)
		f_px, _src = resolve_fpx(stem, width, params['focal_len_mm'],
								 params['sensor_width_mm'], params['fpx_overrides'])
		return fps, f_px, flightlog
	except Exception as e:                      # geometry is a bonus, never a blocker
		print(f"    (speed gate stays in pixels: {type(e).__name__}: {e})")
		return 30.0, None, None


def run_stitch_project(config_path):
	"""Batch every *_tracking_corrected.csv (falling back to *_tracking.csv) in
	the project's output folder."""
	config_path = os.path.abspath(config_path)
	project_dir = os.path.dirname(config_path)
	p = _load_cfg(config_path)
	cfg = configparser.ConfigParser(); cfg.optionxform = str; cfg.read(config_path)
	# resolve_project_dir also honours the legacy 'output_folder' spelling, so an
	# INI using it does not silently stitch nothing.
	output_dir = resolve_project_dir(cfg, project_dir, 'output')

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
		fps, f_px, flightlog = _clip_geometry(config_path, stem)
		run_stitch(c, max_speed_px=p['max_speed_px'], fps=fps,
				   max_gap_s=p['max_gap_s'], gap_prior_s=p['gap_prior_s'],
				   extrap_horizon_s=p['extrap_horizon_s'], min_len=p['min_len'],
				   expected_group_size=p['expected_group_size'],
				   quality_gate=p['quality_gate'], prior_log_odds=p['prior_log_odds'],
				   flightlog=flightlog, f_px=f_px, max_speed_m_s=p['max_speed_m_s'],
				   speed_gate_margin=p['speed_gate_margin'])


if __name__ == '__main__':
	ap = argparse.ArgumentParser(description="Offline kinematic tracklet stitching.")
	ap.add_argument("target", nargs='?', help="project dir or BehaveAI_settings.ini")
	ap.add_argument("--csv", help="stitch a single tracking CSV instead")
	ap.add_argument("--fps", type=float, default=30.0)
	ap.add_argument("--max-speed-px", type=float, default=60.0)
	ap.add_argument("--max-gap-s", type=float, default=5.0)
	ap.add_argument("--gap-prior-s", type=float, default=5.0)
	ap.add_argument("--prior-log-odds", type=float, default=-5.0)
	ap.add_argument("--expected-group-size", type=int, default=0)
	a = ap.parse_args()
	if a.csv:
		run_stitch(a.csv, fps=a.fps, max_speed_px=a.max_speed_px,
				   max_gap_s=a.max_gap_s, gap_prior_s=a.gap_prior_s,
				   prior_log_odds=a.prior_log_odds,
				   expected_group_size=a.expected_group_size)
	elif a.target:
		ini = a.target if a.target.endswith('.ini') else os.path.join(a.target, 'BehaveAI_settings.ini')
		run_stitch_project(ini)
	else:
		ap.error("give a project dir / .ini, or --csv <file>")
