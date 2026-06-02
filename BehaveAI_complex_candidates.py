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
       - stampede               : high sub-group mean speed AND high polarisation.
       - trek                   : high polarisation AND moderate speed AND non-zero
                                  centroid speed (high elongation reinforces).
       - synchronised_rest_graze: speed ~0 AND high synchrony AND low dispersion.
     Only rules whose behaviour name is in the configured complex_behaviours list
     are proposed (so candidate labels match the ethogram).

  2. Active learning with the trained TASK 7 model: the most UNCERTAIN windows
     (lowest max-probability) are surfaced as the next segments to annotate, with
     the model's current best guess as the suggested behaviour.

Everything is image-space; no metric conversion.
"""

import os
import sys
import csv
import glob
import argparse
import configparser
from collections import defaultdict

import numpy as np

import BehaveAI_complex_features as CF
import BehaveAI_complex_model as CM

# Candidates are written with the TASK 6 complex-behaviour schema.
TASK6_COLUMNS = [
	'video_filename', 'start_frame', 'end_frame', 'behaviour', 'track_ids',
	'annotator_confidence', 'fps', 'frame_width', 'frame_height',
]

# Internal constant: |approach_rate| (body lengths / frame) below this is treated
# as "roughly constant distance" for the chase rule.
_CHASE_CONST_DIST_BODYLEN = 0.05


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
		'speed_low_bodylen':  float(d.get('complex_speed_low_bodylen', '0.05')),
		'speed_high_bodylen': float(d.get('complex_speed_high_bodylen', '0.25')),
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
	ref = cache['ref'] if cache['ref'] and cache['ref'] > 0 else 1.0
	tol = 2 * _typical_step(td['frames'])
	min_dur = params['min_duration_frames']
	lo, hi = params['speed_low_bodylen'], params['speed_high_bodylen']
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
					and abs(r['approach_rate']) / ref < _CHASE_CONST_DIST_BODYLEN)
				   for r in rows]
			for (s, e) in _find_runs(seq, min_dur, tol):
				out.append((s, e, [a, b], 'chase'))

	# --- Group rules per sub-group ---
	group_rules = {'stampede', 'trek', 'synchronised_rest_graze'} & allowed
	if group_rules:
		out += _group_heuristics(cache, params, tol, min_dur, lo, hi, group_rules)
	return out


def _group_heuristics(cache, params, tol, min_dur, lo, hi, group_rules):
	"""Group-level heuristic candidates from the sub-group feature streams."""
	out = []
	sg_path = os.path.join(cache['output_dir'], cache['stem'] + '_subgroups.csv')
	if not os.path.exists(sg_path):
		return out
	subgroups = CF.load_subgroups_csv(sg_path)
	if not subgroups:
		return out
	gfeat = CF.compute_group_features(cache['track_data'], subgroups, cache['ref'])
	# Member set per (frame, subgroup) for emitting track_ids.
	members = {}
	for f, groups in subgroups.items():
		for sgid, ids in groups:
			members[(f, sgid)] = ids

	by_sg = defaultdict(list)
	for g in gfeat:
		by_sg[g['subgroup_id']].append(g)
	pol_hi, syn_hi = params['polarisation_high'], params['synchrony_high']

	for sgid, rows in by_sg.items():
		rows.sort(key=lambda g: g['frame'])
		rules = {
			'stampede': lambda g: g['mean_speed'] > hi and g['polarisation'] > pol_hi,
			'trek': lambda g: (g['polarisation'] > pol_hi and lo <= g['mean_speed'] <= hi
							   and g['centroid_speed'] > lo),
			'synchronised_rest_graze': lambda g: (g['mean_speed'] < lo and g['synchrony'] > syn_hi
												  and g['cohesion'] < 1.0),
		}
		for beh in group_rules:
			cond = rules[beh]
			seq = [(g['frame'], cond(g)) for g in rows]
			for (s, e) in _find_runs(seq, min_dur, tol):
				# Use the most-populated member set within the run for track_ids.
				ids = _dominant_members(members, sgid, s, e)
				if ids:
					out.append((s, e, ids, beh))
	return out


def _dominant_members(members, sgid, s, e):
	"""Most frequent member set (ordered) of a sub-group across [s, e]."""
	from collections import Counter
	cnt = Counter()
	chosen = []
	for (f, g), ids in members.items():
		if g == sgid and s <= f <= e:
			for t in ids:
				cnt[t] += 1
	if not cnt:
		return []
	# Keep members present in at least half the run's observed frames.
	span = sum(1 for (f, g) in members if g == sgid and s <= f <= e)
	for t, c in cnt.most_common():
		if c >= max(1, span // 2):
			chosen.append(t)
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

	sg_path = os.path.join(cache['output_dir'], cache['stem'] + '_subgroups.csv')
	if os.path.exists(sg_path):
		subgroups = CF.load_subgroups_csv(sg_path)
		wins = defaultdict(lambda: defaultdict(set))
		for f, groups in subgroups.items():
			wstart = (f // win) * win
			for sgid, ids in groups:
				if len(ids) >= 3:
					wins[sgid][wstart].update(ids)
		for sgid, ws in wins.items():
			for wstart, ids in ws.items():
				if len(ids) >= 3:
					cand.append((wstart, wstart + win - 1, sorted(ids, key=lambda t: (len(t), t))))

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

def _video_metadata(output_dir, stem):
	"""Return (fps, width, height) by reading the source video if found, else ('', '', '')."""
	try:
		import cv2
		from BehaveAI_annotation_complex import find_annotatable_videos
		# input/clips are resolved relative to the project (output_dir parent).
		project_dir = os.path.dirname(os.path.abspath(output_dir))
		vids = find_annotatable_videos(output_dir,
									   os.path.join(project_dir, 'input'),
									   os.path.join(project_dir, 'clips'))
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


def generate_candidates_for_video(stem, csv_path, output_dir, params, bundle):
	"""Write <stem>_complex_candidates.csv from heuristic + active-learning sources."""
	cache = CM._build_video_cache(stem, csv_path, params)
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
	fps, w, h = _video_metadata(output_dir, stem)

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

	jobs = {}
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking_corrected.csv'))):
		jobs[os.path.basename(p).replace('_tracking_corrected.csv', '')] = p
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv'))):
		jobs.setdefault(os.path.basename(p).replace('_tracking.csv', ''), p)
	if not jobs:
		print(f"Candidates: no tracking CSVs found in {output_dir}")
		return

	print(f"Candidates: processing {len(jobs)} video(s)...")
	for stem, csv_path in sorted(jobs.items()):
		try:
			generate_candidates_for_video(stem, csv_path, output_dir, params, bundle)
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
