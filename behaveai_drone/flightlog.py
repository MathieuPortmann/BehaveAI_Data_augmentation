#!/usr/bin/env python3
"""
BehaveAI Flight-log Reader

Reads the `<video>.flightlog.csv` sidecars produced by the WP1 telemetry
recovery (see W:\\WP1_Activity-budget\\Renamer\\herdwise_flightlog.py). These
carry the gimbal attitude that DJI SRT files lack — in particular `gb_pitch`,
which is exactly the camera tilt the metric geometry needs and which the
horizon fit otherwise has to estimate from the animals.

CRITICAL: two column SCHEMAS coexist in the corpus and differ by more than
order — Mini4Pro sidecars have 21 columns (extra `utc`, `isGPSUsed`,
`flycState`), Mini3/Mini3Pro have 18. Reading by fixed index therefore shifts
the gimbal fields by one column on Mini4Pro (you read `ac_yaw` thinking it is
`gb_pitch`). This reader ALWAYS resolves columns BY NAME. The data itself is
correct for all models (gb_pitch ~ -30, gb_roll ~ 0).

The `video_t_s` column is already aligned to the video timeline (t = 0 at the
first video frame; negative values are pre-roll), so no wall-clock matching is
needed — just interpolate at t = frame_time.

Usage as a library:
    fl = load_flightlog(path)
    h, pitch, roll = sample_flightlog(fl, t_seconds)
"""

import csv

import numpy as np

# Canonical field names we consume (resolved by name, never by index).
_NUMERIC_FIELDS = (
	'video_t_s', 'flyTime_s', 'rel_alt_m', 'abs_alt_m', 'vps_m',
	'gb_pitch', 'gb_roll', 'gb_yaw', 'ac_pitch', 'ac_roll', 'ac_yaw',
	'latitude', 'longitude',
)


def load_flightlog(path):
	"""Load a `.flightlog.csv` into a dict of float numpy arrays, keyed by the
	canonical column NAMES present in the file, sorted by `video_t_s`.

	Missing/blank cells become NaN. Returns {} if the file has no `video_t_s`
	column (cannot be aligned to a video) or no rows.
	"""
	with open(path, newline='', encoding='utf-8', errors='replace') as f:
		reader = csv.DictReader(f)
		field_set = set(reader.fieldnames or [])
		if 'video_t_s' not in field_set:
			return {}
		cols = [c for c in _NUMERIC_FIELDS if c in field_set]
		acc = {c: [] for c in cols}
		for row in reader:
			for c in cols:
				v = (row.get(c, '') or '').strip()
				try:
					acc[c].append(float(v))
				except ValueError:
					acc[c].append(float('nan'))

	if not acc.get('video_t_s'):
		return {}

	out = {c: np.asarray(vals, dtype=float) for c, vals in acc.items()}
	order = np.argsort(out['video_t_s'])
	return {c: v[order] for c, v in out.items()}


def sample_at(flightlog, t_s, key):
	"""Linear-interpolate `key` at video time `t_s` (seconds). NaNs in the
	series are dropped before interpolation; returns NaN if unavailable or the
	key is absent. Outside the log's time span the nearest endpoint is held."""
	if not flightlog or key not in flightlog or 'video_t_s' not in flightlog:
		return float('nan')
	t = flightlog['video_t_s']
	y = flightlog[key]
	good = ~(np.isnan(t) | np.isnan(y))
	if good.sum() == 0:
		return float('nan')
	return float(np.interp(t_s, t[good], y[good]))


def sample_flightlog(flightlog, t_s):
	"""Convenience: return (rel_alt_m, gb_pitch_deg, gb_roll_deg) at video time
	`t_s`. Any component may be NaN if its column is missing."""
	return (sample_at(flightlog, t_s, 'rel_alt_m'),
			sample_at(flightlog, t_s, 'gb_pitch'),
			sample_at(flightlog, t_s, 'gb_roll'))


def flightlog_summary(flightlog):
	"""Quick stats used for plausibility checks / logging: pitch & roll
	range and medians, altitude range. Returns {} for an empty log."""
	if not flightlog or 'gb_pitch' not in flightlog:
		return {}

	def _stat(key):
		if key not in flightlog:
			return None
		v = flightlog[key]
		v = v[~np.isnan(v)]
		if v.size == 0:
			return None
		return {'min': float(v.min()), 'max': float(v.max()), 'median': float(np.median(v))}

	return {'gb_pitch': _stat('gb_pitch'), 'gb_roll': _stat('gb_roll'),
			'rel_alt_m': _stat('rel_alt_m'), 'n_rows': int(len(flightlog['video_t_s']))}
