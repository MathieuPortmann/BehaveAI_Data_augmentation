#!/usr/bin/env python3
"""
BehaveAI Complex-Behaviour Candidate Proposal (heuristic + active learning)

Run AFTER the first hand-annotations + model exist (PHASE 8/9) to accelerate
further annotation. Produces <video>_complex_candidates.csv using the TASK 6
schema with annotator_confidence='auto'; track_ids lists all involved
individuals (ordered for role). The complex annotation tool's "Load candidates"
action reads this file so the annotator only has to confirm/correct.

Two complementary sources:

  1. Heuristic rules over the TASK 4 feature streams (thresholds from the INI,
     meant to be calibrated on the first annotations — not invented from zero):
       - allogrooming           : in_contact AND both speeds ~0, sustained.
       - chase                  : high speed_similarity AND ~constant distance AND
                                  both speeds high, sustained.
       - stampede               : high whole-herd mean speed AND high polarisation.
       - trek                   : high polarisation AND moderate speed AND non-zero
                                  centroid speed (high elongation reinforces).
       - synchronised_rest_graze: speed ~0 AND high synchrony AND low dispersion.
     Only rules whose behaviour name is in the configured complex_behaviours list
     are proposed (so candidate labels match the ethogram).

  2. Active learning with the trained TASK 7 model: the most UNCERTAIN windows
     (lowest max-probability) are surfaced as the next segments to annotate, with
     the model's current best guess as the suggested behaviour.

All thresholds are in real units (metres, m/s) and read the metric tracking CSV,
so a rule means the same thing at 15 m and at 50 m of altitude.
"""

import os
import sys
import csv
import glob
import argparse
import configparser
from collections import defaultdict, Counter

import numpy as np

import BehaveAI_complex_features as CF
import BehaveAI_complex_model as CM

# Candidates are written with the TASK 6 complex-behaviour schema.
TASK6_COLUMNS = [
	'video_filename', 'start_frame', 'end_frame', 'behaviour', 'track_ids',
	'annotator_confidence', 'fps', 'frame_width', 'frame_height',
]

# Internal constant: |approach_rate| (metres / second) below this is treated as
# "roughly constant distance" for the chase rule — a chaser holds its gap.
_CHASE_CONST_DIST_MS = 0.5

# Internal constant: standard deviation (metres) of members' distance to the herd
# centroid below which the group counts as tightly clustered.
_TIGHT_GROUP_SPREAD_M = 5.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_candidate_config(config_path):
	"""Heuristic thresholds + active-learning settings (all read with fallbacks)."""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	base = CM.load_model_config(config_path)
	base.update({
		# In METRES PER SECOND. The old body-lengths-per-frame thresholds were
		# never usable: 0.05 body-len/frame is ~3.3 m/s at 30 fps — a trot — yet
		# the rules used it to mean "standing still", so allogrooming and
		# synchronised rest could fire on moving animals and chase almost never
		# fired. Metric features make an honest threshold expressible.
		'speed_low_ms':       float(d.get('complex_speed_low_ms', '0.2')),
		'speed_high_ms':      float(d.get('complex_speed_high_ms', '3.0')),
		'polarisation_high':  float(d.get('complex_polarisation_high', '0.7')),
		'synchrony_high':     float(d.get('complex_synchrony_high', '0.7')),
		'candidate_topk':     int(float(d.get('complex_candidate_topk', '50'))),
		'behaviours':         [b.strip() for b in d.get('complex_behaviours', '').split(',') if b.strip()],
	})
	return base


def candidate_csv_path(output_dir, video_stem):
	return os.path.join(output_dir, video_stem + '_complex_candidates.csv')


# ---------------------------------------------------------------------------
# Run finder
# ---------------------------------------------------------------------------

def _typical_step(frames):
	if len(frames) < 2:
		return 1
	diffs = np.diff(sorted(set(frames)))
	return int(np.median(diffs)) if len(diffs) else 1


def _find_runs(frames_ok, min_span, tol):
	"""Maximal runs of ok==True with inter-frame gaps <= tol and span >= min_span.

	frames_ok: list of (frame, ok_bool) sorted by frame. Returns [(start, end), ...].
	"""
	runs = []
	start = last = None
	for f, ok in frames_ok:
		if ok and start is None:
			start = last = f
		elif ok and f - last <= tol:
			last = f
		elif ok:
			if last - start + 1 >= min_span:
				runs.append((start, last))
			start = last = f
		else:
			if start is not None and last - start + 1 >= min_span:
				runs.append((start, last))
			start = last = None
	if start is not None and last - start + 1 >= min_span:
		runs.append((start, last))
	return runs


# ---------------------------------------------------------------------------
# Heuristic candidates
# ---------------------------------------------------------------------------

def _heuristic_candidates(cache, params):
	"""Apply the heuristic rules; return [(start, end, ids, behaviour), ...].

	Only behaviours present in the configured complex_behaviours list are emitted.
	"""
	allowed = set(params['behaviours'])
	td = cache['track_data']
	tol = 2 * _typical_step(td['frames'])
	min_dur = params['min_duration_frames']
	# A "standing still" threshold set below the clip's own noise floor would call
	# every grazing horse a mover, so the configured value is raised to the floor
	# when the footage is noisier than the setting assumes.
	floor = td.get('speed_floor_ms', 0.0)
	lo = max(params['speed_low_ms'], floor)
	hi = max(params['speed_high_ms'], lo * 1.5)
	if lo > params['speed_low_ms']:
		print(f"  {cache['stem']}: complex_speed_low_ms={params['speed_low_ms']:.2f} is below "
			  f"this clip's noise floor; using {lo:.2f} m/s instead.")
	out = []

	# --- Dyadic rules per ordered pair ---
	for (a, b), rows in cache['pair_index'].items():
		rows = sorted(rows, key=lambda r: r['frame'])
		if 'allogrooming' in allowed:
			seq = [(r['frame'], bool(r['in_contact']) and r['speed_A'] < lo and r['speed_B'] < lo)
				   for r in rows]
			for (s, e) in _find_runs(seq, min_dur, tol):
				out.append((s, e, [a, b], 'allogrooming'))
		if 'chase' in allowed:
			seq = [(r['frame'],
					r['speed_similarity'] > 0.6 and r['speed_A'] > hi and r['speed_B'] > hi
					and abs(r['approach_rate']) < _CHASE_CONST_DIST_MS)
				   for r in rows]
			for (s, e) in _find_runs(seq, min_dur, tol):
				out.append((s, e, [a, b], 'chase'))

	# --- Group rules over the whole co-present herd ---
	group_rules = {'stampede', 'trek', 'synchronised_rest_graze'} & allowed
	if group_rules:
		out += _group_heuristics(cache, params, tol, min_dur, lo, hi, group_rules)
	return out


def _group_heuristics(cache, params, tol, min_dur, lo, hi, group_rules):
	"""Group-level heuristic candidates from the whole co-present herd's feature
	stream (no spatial sub-grouping — every co-present individual counts)."""
	out = []
	td = cache['track_data']
	gfeat = CF.compute_group_features(td, CF.whole_herd_groups(td))
	if not gfeat:
		return out
	gfeat.sort(key=lambda g: g['frame'])
	pol_hi, syn_hi = params['polarisation_high'], params['synchrony_high']

	rules = {
		'stampede': lambda g: g['mean_speed'] > hi and g['polarisation'] > pol_hi,
		'trek': lambda g: (g['polarisation'] > pol_hi and lo <= g['mean_speed'] <= hi
						   and g['centroid_speed'] > lo),
		# cohesion is now the spread of members around the centroid IN METRES, so
		# the "tight group" bound is a real distance rather than a body-length count.
		'synchronised_rest_graze': lambda g: (g['mean_speed'] < lo and g['synchrony'] > syn_hi
											  and g['cohesion'] < _TIGHT_GROUP_SPREAD_M),
	}
	for beh in group_rules:
		cond = rules[beh]
		seq = [(g['frame'], cond(g)) for g in gfeat]
		for (s, e) in _find_runs(seq, min_dur, tol):
			# Use the most-present members within the run for track_ids.
			ids = _dominant_members(td, s, e)
			if ids:
				out.append((s, e, ids, beh))
	return out


def _dominant_members(track_data, s, e):
	"""Most frequent co-present members (ordered) across [s, e]."""
	cnt = Counter()
	span = 0
	for f in track_data['frames']:
		if s <= f <= e:
			span += 1
			for t in track_data['present'][f]:
				cnt[t] += 1
	if not cnt:
		return []
	# Keep members present in at least half the run's observed frames.
	chosen = [t for t, c in cnt.most_common() if c >= max(1, span // 2)]
	return chosen or [t for t, _ in cnt.most_common()]


# ---------------------------------------------------------------------------
# Active-learning candidates
# ---------------------------------------------------------------------------

def _active_learning_candidates(cache, params, bundle):
	"""Surface the most-uncertain windows using the trained model.

	Returns [(start, end, ids, behaviour), ...] for the top-K windows by
	uncertainty (1 - max predicted probability).
	"""
	pipe, classes = bundle['pipeline'], bundle['classes']
	win = params['window_frames']
	cand = []  # (start, end, ids)

	for (a, b), rows in cache['pair_index'].items():
		frames = sorted(r['frame'] for r in rows)
		if len(frames) < params['min_duration_frames']:
			continue
		s = frames[0]
		while s <= frames[-1]:
			cand.append((s, s + win - 1, [a, b]))
			s += win

	for (wstart, wend, ids) in CF.whole_herd_window_candidates(cache['track_data'], win, min_members=3):
		cand.append((wstart, wend, sorted(ids, key=lambda t: (len(t), t))))

	if not cand:
		return []
	feats = [CM.segment_feature_dict(cache, ids, s, e)[0] for (s, e, ids) in cand]
	proba = pipe.predict_proba(feats)
	uncertainty = 1.0 - proba.max(axis=1)
	order = np.argsort(uncertainty)[::-1][:params['candidate_topk']]
	out = []
	for i in order:
		s, e, ids = cand[i]
		j = int(np.argmax(proba[i]))
		out.append((s, e, ids, classes[j]))
	return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _video_metadata(output_dir, stem, config_path=None):
	"""Return (fps, width, height) by reading the source video if found, else ('', '', '').

	Needs the INI to know where the videos are: input/clips used to be guessed
	as siblings of the output directory, which found nothing as soon as a
	project pointed either key elsewhere. Without a config_path there is nothing
	to consult, so the metadata columns are simply left empty.
	"""
	if not config_path:
		return ('', '', '')
	try:
		import cv2
		from BehaveAI_annotation_complex import find_annotatable_videos, resolve_dirs
		_pdir, _odir, input_dir, clips_dir = resolve_dirs(config_path)
		vids = find_annotatable_videos(output_dir, input_dir, clips_dir)
		path = next((v['video_path'] for v in vids if v['stem'] == stem and v['video_path']), None)
		if path and os.path.exists(path):
			cap = cv2.VideoCapture(path)
			fps = cap.get(cv2.CAP_PROP_FPS)
			w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			cap.release()
			return (f"{fps:.6g}" if fps else '', w, h)
	except Exception:
		pass
	return ('', '', '')


def _dedup(cands):
	"""Drop duplicate/overlapping candidates with the same behaviour and track_ids."""
	by_key = defaultdict(list)
	for (s, e, ids, beh) in cands:
		by_key[(beh, ';'.join(ids))].append((s, e, ids, beh))
	out = []
	for key, items in by_key.items():
		items.sort(key=lambda c: c[0])
		cur = list(items[0])
		for (s, e, ids, beh) in items[1:]:
			if s <= cur[1] + 1:
				cur[1] = max(cur[1], e)
			else:
				out.append(tuple(cur)); cur = [s, e, ids, beh]
		out.append(tuple(cur))
	out.sort(key=lambda c: (c[0], c[3]))
	return out


def generate_candidates_for_video(stem, csv_path, output_dir, params, bundle,
								  config_path=None, video_index=None):
	"""Write <stem>_complex_candidates.csv from heuristic + active-learning sources."""
	cache = CM._build_video_cache(stem, csv_path, params, config_path, video_index)
	cache['output_dir'] = output_dir
	cache['stem'] = stem

	cands = _heuristic_candidates(cache, params)
	n_heur = len(cands)
	n_al = 0
	if bundle is not None:
		al = _active_learning_candidates(cache, params, bundle)
		n_al = len(al)
		cands += al

	cands = _dedup(cands)
	fps, w, h = _video_metadata(output_dir, stem, config_path)

	video_filename = stem
	out_path = candidate_csv_path(output_dir, stem)
	with open(out_path, 'w', newline='', encoding='utf-8') as f:
		writer = csv.DictWriter(f, fieldnames=TASK6_COLUMNS, extrasaction='ignore')
		writer.writeheader()
		for (s, e, ids, beh) in cands:
			writer.writerow({
				'video_filename': video_filename, 'start_frame': s, 'end_frame': e,
				'behaviour': beh, 'track_ids': ';'.join(ids),
				'annotator_confidence': 'auto', 'fps': fps,
				'frame_width': w, 'frame_height': h,
			})
	print(f"  {stem}: {len(cands)} candidate(s) "
		  f"(heuristic={n_heur}, active-learning={n_al}) -> {os.path.basename(out_path)}")


def generate_candidates(project_path):
	"""Generate complex-behaviour candidates for every tracking CSV in the project."""
	config_path = CM._config_path_for(project_path)
	params = load_candidate_config(config_path)
	project_dir, output_dir = CM._resolve_output_dir(config_path)

	# Load the trained model for active learning (optional).
	bundle = None
	model_path = os.path.join(project_dir, CM.MODEL_DIR_NAME, 'pipeline.joblib')
	if CM._SKLEARN_AVAILABLE and os.path.exists(model_path):
		try:
			import joblib
			bundle = joblib.load(model_path)
		except Exception as e:
			print(f"Could not load complex model ({e}); heuristics only.")
	else:
		print("No trained complex model found — generating heuristic candidates only "
			  "(train the model to enable active learning).")

	suffix = '_tracking_metric.csv'
	jobs = {os.path.basename(p)[:-len(suffix)]: p
			for p in sorted(glob.glob(os.path.join(output_dir, '*' + suffix)))}
	if not jobs:
		print(f"Candidates: no metric tracking CSVs in {output_dir} — the heuristics "
			  f"are expressed in m and m/s, so the metric stage must run first.")
		return

	video_index = CM._video_index(config_path)
	print(f"Candidates: processing {len(jobs)} video(s)...")
	for stem, csv_path in sorted(jobs.items()):
		try:
			generate_candidates_for_video(stem, csv_path, output_dir, params, bundle,
										  config_path, video_index)
		except CF.MissingMetricError as e:
			print(f"  SKIPPED {stem}: {e}")
		except Exception as e:
			import traceback
			print(f"  ERROR on {stem}: {e}")
			traceback.print_exc()
	print("Candidate proposal complete.")


def _main():
	parser = argparse.ArgumentParser(
		description="Propose complex-behaviour candidates (heuristic + active learning).")
	parser.add_argument('target', help="Project directory or BehaveAI_settings.ini.")
	args = parser.parse_args()
	target = os.path.abspath(args.target)
	ini = os.path.join(target, 'BehaveAI_settings.ini') if os.path.isdir(target) else target
	if not os.path.exists(ini):
		print(f"Settings file not found: {ini}")
		sys.exit(1)
	generate_candidates(ini)


if __name__ == '__main__':
	_main()
