#!/usr/bin/env python3
"""
BehaveAI Complex-Behaviour Model (supervised baseline)

Trains a classifier on the human complex-behaviour annotations
(<video>_complex_behaviours.csv) using the windowed tabular features from TASK 4,
and predicts complex behaviours over a video (<video>_complex_predictions.csv).

The DEFAULT model is a scikit-learn baseline on fixed-size window feature vectors
(robust with little data, interpretable). The lstm/transformer models train a real
torch sequence network over the per-timestep feature sequence (each segment is
sliced into complex_seq_steps sub-windows); they reuse the same features and the
same by-video evaluation. Asking for a sequence model without torch is an ERROR,
not a quiet downgrade: substituting the baseline used to produce a saved model
whose own saved_settings.ini claimed a different architecture.

All features are in METRES / m/s and come from <video>_tracking_metric.csv (see
BehaveAI_complex_features). A video without metric geometry is skipped rather
than measured in some other unit.

First-class inputs: the per-individual YOLO simple-behaviour labels (primary AND,
when present, secondary) for every involved individual, plus the group synchrony
feature, are encoded (bag-of-labels over the window) ALONGSIDE the
kinematic/graph features. The complex model therefore consumes BOTH the
geometry/graph features and YOLO's simple-behaviour outputs. Missing secondary
labels are handled defensively (their one-hot is simply absent / all-zero).

Evaluation splits BY VIDEO (never by segment) to avoid leaking the same
individuals across train/val; reports per-class F1, macro-F1 and a confusion
matrix; handles class imbalance via balanced sample weights; and down-weights
windows whose drone-correction quality is not 'ok'.

A permanent, deterministic holdout set of whole videos (see
behaveai_holdout.is_holdout_video, keyed off val_frequency) is excluded from
ALL training — the final model is fit on the train pool only. metrics.txt
reports two distinct numbers: "By-video CV macro-F1 (train pool)" (diagnostic
cross-validation among training videos) and "Held-out video macro-F1 (never
trained on)" (true generalization to videos the deployed model has never
seen). Holdout membership is stable across runs and adapts automatically as
videos are added — no list is stored.

Public functions: build_dataset, train_model, classify_video, analyse_confusion.

Outputs under model_complex/: pipeline.joblib, train_count.txt,
saved_settings.ini, metrics.txt, feature_importances.txt, merge_suggestions.txt.
"""

import os
import sys
import csv
import glob
import argparse
import configparser
import shutil
from collections import defaultdict, Counter

import numpy as np

from behaveai_config import resolve_project_dir
import BehaveAI_complex_features as CF
from behaveai_holdout import is_holdout_video, split_groups

# scikit-learn is required for the baseline; degrade with a clear message.
try:
	from sklearn.feature_extraction import DictVectorizer
	from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
	from sklearn.pipeline import Pipeline
	from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_predict
	from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
	from sklearn.dummy import DummyClassifier
	from sklearn.utils.class_weight import compute_sample_weight
	import joblib
	_SKLEARN_AVAILABLE = True
except Exception:
	_SKLEARN_AVAILABLE = False


MODEL_DIR_NAME = 'model_complex'
PREDICTION_COLUMNS = ['start_frame', 'end_frame', 'track_ids', 'behaviour', 'probability']

_GROUP_SCALARS = ['mean_speed', 'polarisation', 'cohesion', 'area',
				  'centroid_speed', 'synchrony', 'elongation']


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_model_config(config_path):
	"""Read the TASK 7 + relevant TASK 4 parameters from a BehaveAI INI."""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	CF.reject_retired_keys(d, config_path)
	return {
		'model_type':          d.get('complex_model_type', 'baseline'),
		'baseline_classifier': d.get('complex_baseline_classifier', 'random_forest'),
		'window_frames':       int(float(d.get('complex_window_frames', '30'))),
		'max_interaction_distance_m': float(d.get('complex_max_interaction_distance_m', '15')),
		'contact_iou_thresh':  float(d.get('complex_contact_iou_thresh', '0.05')),
		'contact_dist_m':      float(d.get('complex_contact_dist_m', '2.5')),
		'min_duration_frames': int(float(d.get('complex_min_duration_frames', '10'))),
		'edge_granularity':    CF.normalise_granularity(d.get('interaction_edge_granularity', 'per_dyad')),
		'weight_metric':       d.get('interaction_weight_metric', 'sri'),
		'confusion_merge_rate': float(d.get('complex_confusion_merge_rate', '0.20')),
		'predict_min_proba':   float(d.get('complex_predict_min_proba', '0.5')),
		'holdout_fraction':    float(d.get('val_frequency', '0.1')),
		# --- Deep sequence model (lstm/transformer) hyper-parameters ---
		# All read with safe defaults so the deep path works even if these keys
		# are absent from an older INI.
		'seq_steps':           int(float(d.get('complex_seq_steps', '8'))),
		'deep_epochs':         int(float(d.get('complex_deep_epochs', '60'))),
		'deep_hidden':         int(float(d.get('complex_deep_hidden', '64'))),
		'deep_layers':         int(float(d.get('complex_deep_layers', '1'))),
		'deep_heads':          int(float(d.get('complex_deep_heads', '4'))),
		'deep_dropout':        float(d.get('complex_deep_dropout', '0.2')),
		'deep_lr':             float(d.get('complex_deep_lr', '0.001')),
		'deep_batch':          int(float(d.get('complex_deep_batch', '16'))),
	}


def _resolve_output_dir(config_path):
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	project_dir = os.path.dirname(os.path.abspath(config_path))
	return project_dir, resolve_project_dir(cfg, project_dir, 'output')


def _config_path_for(project_path):
	p = os.path.abspath(project_path)
	return os.path.join(p, 'BehaveAI_settings.ini') if os.path.isdir(p) else p


def _video_index(config_path):
	"""Project video index (for fps lookups), or None if unavailable."""
	try:
		from behaveai_drone.metric_geometry import build_video_index
		return build_video_index(config_path)
	except Exception:
		return None


# ---------------------------------------------------------------------------
# Per-video cache (track data, features, quality)
# ---------------------------------------------------------------------------

def _read_quality(csv_path):
	"""Return {(frame, id) -> correction_quality} (empty if no such column)."""
	q = {}
	with open(csv_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		if 'correction_quality' not in (reader.fieldnames or []):
			return q
		for r in reader:
			try:
				q[(int(r['frame']), str(r['id']))] = r.get('correction_quality', '')
			except (ValueError, KeyError, TypeError):
				continue
	return q


def _build_video_cache(stem, csv_path, params, config_path=None, video_index=None):
	"""Load + precompute everything needed to extract features for one video."""
	fps = CF.video_fps(config_path, stem, video_index) if config_path else 30.0
	track_data = CF.load_tracking_csv(csv_path, fps=fps)
	age_by_track, age_conf_by_track = CF.track_age_classes(track_data)
	pairwise = CF.compute_pairwise_features(
		track_data, age_by_track,
		max_distance_m=params['max_interaction_distance_m'],
		contact_iou_thresh=params['contact_iou_thresh'],
		contact_dist_m=params['contact_dist_m'])
	pair_index = defaultdict(list)
	for r in pairwise:
		pair_index[(r['source_id'], r['target_id'])].append(r)
	return {
		'stem': stem, 'csv_path': csv_path, 'track_data': track_data,
		'age': age_by_track, 'age_conf': age_conf_by_track,
		'pairwise': pairwise, 'pair_index': pair_index,
		'quality': _read_quality(csv_path),
	}


# ---------------------------------------------------------------------------
# Feature extraction for a labelled segment
# ---------------------------------------------------------------------------

def _aggregate_scalars(dicts, keys, prefix):
	"""mean/std/min/max of selected scalar keys across a list of dicts."""
	out = {}
	if not dicts:
		for k in keys:
			out[f'{prefix}{k}_mean'] = out[f'{prefix}{k}_std'] = 0.0
			out[f'{prefix}{k}_min'] = out[f'{prefix}{k}_max'] = 0.0
		return out
	for k in keys:
		vals = np.array([float(dd.get(k, 0.0)) for dd in dicts], dtype=float)
		out[f'{prefix}{k}_mean'] = float(vals.mean())
		out[f'{prefix}{k}_std'] = float(vals.std())
		out[f'{prefix}{k}_min'] = float(vals.min())
		out[f'{prefix}{k}_max'] = float(vals.max())
	return out


def segment_feature_dict(cache, ids, start, end):
	"""Build one fixed-schema feature dict for an annotated/candidate segment.

	Pools the dyadic features of all ordered pairs among `ids` over the window
	(aggregate_window -> scalars + YOLO label bag), adds aggregated group features
	for the ad-hoc group of `ids`, plus age/individual-count meta.
	Returns (feature_dict, quality_weight).
	"""
	ids = [str(t) for t in ids]
	id_set = set(ids)
	td = cache['track_data']

	# --- Dyadic: pool all ordered-pair rows among ids within the window ---
	pooled = []
	for a in ids:
		for b in ids:
			if a == b:
				continue
			for r in cache['pair_index'].get((a, b), []):
				if start <= r['frame'] <= end:
					pooled.append(r)
	feat = {f'dy_{k}': v for k, v in CF.aggregate_window(pooled, start, end).items()}

	# --- Group: treat ids as an ad-hoc group over the window ---
	sub = {}
	for f in td['frames']:
		if start <= f <= end:
			present = [t for t in ids if (f, t) in td['pos']]
			if present:
				sub[f] = [('seg', present)]
	gfeat_rows = CF.compute_group_features(td, sub) if sub else []
	feat.update(_aggregate_scalars(gfeat_rows, _GROUP_SCALARS, 'gp_'))

	# --- Meta: individual count + age composition (from the age classifier) ---
	# Counting each age class separately keeps 'unknown' visible to the model
	# instead of folding it into "not a foal", which a single foal_count would.
	ages = [cache['age'].get(t, CF.UNKNOWN_AGE) for t in ids]
	feat['meta_n_individuals'] = float(len(ids))
	for label, n in Counter(ages).items():
		feat[f'meta_n_age={label}'] = float(n)
	feat['meta_age_conf_mean'] = float(
		np.mean([cache['age_conf'].get(t, 0.0) for t in ids])) if ids else 0.0

	# --- Quality weight: fraction of involved (frame,id) rows that are 'ok' ---
	q = cache['quality']
	if q:
		tot = ok = 0
		for f in td['frames']:
			if start <= f <= end:
				for t in ids:
					if (f, t) in td['pos']:
						tot += 1
						qq = q.get((f, t), 'ok')
						if qq == 'ok':
							ok += 1
						elif qq == 'none':
							ok += 0  # excluded
						else:
							ok += 0.5  # 'uncertain' down-weighted
		quality_weight = (ok / tot) if tot else 1.0
	else:
		quality_weight = 1.0
	return feat, max(0.05, quality_weight)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(project_path, include_holdout='train'):
	"""Gather all *_complex_behaviours.csv, extract one window feature vector per
	annotation, and split metadata BY VIDEO.

	`include_holdout` selects which videos to include, using the same
	deterministic per-video assignment as the YOLO split (see
	behaveai_holdout.is_holdout_video): 'train' (default) skips holdout videos,
	'holdout' returns only holdout videos, 'all' applies no filter.

	Returns (X, y, groups, weights) where X is a list of feature dicts, y the
	behaviour labels, groups the video stems (for by-video CV), weights the
	per-sample quality weights. Caches are reused across a video's annotations.
	"""
	config_path = _config_path_for(project_path)
	params = load_model_config(config_path)
	_, output_dir = _resolve_output_dir(config_path)

	ann_files = sorted(glob.glob(os.path.join(output_dir, '*_complex_behaviours.csv')))
	X, y, groups, weights = [], [], [], []
	if not ann_files:
		return X, y, groups, weights
	video_index = _video_index(config_path)

	for ann_path in ann_files:
		stem = os.path.basename(ann_path).replace('_complex_behaviours.csv', '')
		if include_holdout != 'all':
			is_ho = is_holdout_video(stem, params['holdout_fraction'])
			if (include_holdout == 'train') == is_ho:
				continue
		csv_path = CF.find_metric_csv(output_dir, stem)
		if csv_path is None:
			print(f"  {stem}: no metric tracking CSV — annotations skipped "
				  f"(the model works in metres; run the metric stage first).")
			continue
		try:
			cache = _build_video_cache(stem, csv_path, params, config_path, video_index)
		except CF.MissingMetricError as e:
			print(f"  {stem}: {e} — annotations skipped.")
			continue

		with open(ann_path, newline='', encoding='utf-8') as f:
			for r in csv.DictReader(f):
				try:
					s = int(r['start_frame']); e = int(r['end_frame'])
					ids = [t for t in str(r['track_ids']).split(';') if t]
					beh = (r.get('behaviour', '') or '').strip()
				except (ValueError, KeyError, TypeError):
					continue
				if not beh or not ids or s >= e:
					continue
				feat, qw = segment_feature_dict(cache, ids, s, e)
				X.append(feat); y.append(beh); groups.append(stem); weights.append(qw)

	return X, y, groups, weights


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_classifier(params):
	if params['baseline_classifier'] == 'hist_gradient_boosting':
		return HistGradientBoostingClassifier(random_state=0)
	return RandomForestClassifier(n_estimators=300, class_weight='balanced',
								  random_state=0, n_jobs=-1)


def _build_pipeline(params):
	return Pipeline([
		('vec', DictVectorizer(sparse=False)),
		('clf', _make_classifier(params)),
	])


def train_model(project_path, config=None):
	"""Train the complex-behaviour model and save it to model_complex/.

	Splits by video for honest evaluation; writes metrics (per-class F1, macro-F1,
	confusion matrix), feature importances and the merge-suggestion report.
	"""
	if not _SKLEARN_AVAILABLE:
		print("scikit-learn is required to train the complex-behaviour model. "
			  "Install it with: pip install scikit-learn")
		return None

	config_path = _config_path_for(project_path)
	params = load_model_config(config_path)
	project_dir = os.path.dirname(config_path)

	if params['model_type'] in ('lstm', 'transformer'):
		return _train_deep_model(project_path, params, project_dir, config_path)
	return _train_baseline(project_path, params, project_dir, config_path)


def _train_baseline(project_path, params, project_dir, config_path):
	"""Train and save the scikit-learn baseline (the supported default path)."""
	X, y, groups, weights = build_dataset(project_path, include_holdout='train')
	if len(X) < 2:
		print(f"Not enough annotations to train ({len(X)} found). "
			  "Annotate more segments first.")
		return None

	classes = sorted(set(y))
	print(f"Complex model: {len(X)} annotated segments, {len(classes)} class(es) "
		  f"across {len(set(groups))} video(s): {classes}")

	pipe = _build_pipeline(params)
	sample_weight = compute_sample_weight('balanced', y) * np.array(weights, dtype=float)

	# --- By-video cross-validated predictions for honest metrics ---
	n_groups = len(set(groups))
	metrics_lines = []
	if n_groups >= 2:
		splitter = LeaveOneGroupOut() if n_groups <= 10 else GroupKFold(n_splits=5)
		try:
			y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=splitter,
									   params={'clf__sample_weight': sample_weight})
		except TypeError:
			# Older sklearn signature uses fit_params=
			y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=splitter,
									   fit_params={'clf__sample_weight': sample_weight})
		macro = f1_score(y, y_pred, average='macro', labels=classes, zero_division=0)
		metrics_lines.append(f"By-video CV macro-F1 (train pool): {macro:.3f}\n")
		# Chance baselines under the SAME by-video splits, so the macro-F1 above is
		# read against what a label-only guesser would score (a fair paper needs this).
		for strat in ('most_frequent', 'stratified'):
			dpipe = Pipeline([('vec', DictVectorizer(sparse=False)),
							  ('clf', DummyClassifier(strategy=strat, random_state=0))])
			y_base = cross_val_predict(dpipe, X, y, groups=groups, cv=splitter)
			mb = f1_score(y, y_base, average='macro', labels=classes, zero_division=0)
			ab = accuracy_score(y, y_base)
			metrics_lines.append(f"Chance baseline ({strat}) by-video CV: "
								 f"macro-F1={mb:.3f}, accuracy={ab:.3f}\n")
		metrics_lines.append(classification_report(y, y_pred, labels=classes, zero_division=0))
		cm = confusion_matrix(y, y_pred, labels=classes)
		metrics_lines.append("\nConfusion matrix (rows=true, cols=pred):\n")
		metrics_lines.append("labels: " + ", ".join(classes) + "\n")
		metrics_lines.append(np.array2string(cm))
		_write_merge_suggestions(project_dir, classes, cm, params['confusion_merge_rate'])
		print(f"  By-video CV macro-F1 (train pool) = {macro:.3f}")
	else:
		metrics_lines.append("Only one annotated video in the train pool — no by-video "
							 "held-out CV possible. Metrics below are TRAIN-ONLY "
							 "(optimistic); annotate a second video for honest scores.\n")
		pipe.fit(X, y, clf__sample_weight=sample_weight)
		y_pred = pipe.predict(X)
		macro = f1_score(y, y_pred, average='macro', labels=classes, zero_division=0)
		metrics_lines.append(f"TRAIN-ONLY macro-F1: {macro:.3f}\n")
		metrics_lines.append(classification_report(y, y_pred, labels=classes, zero_division=0))
		print(f"  TRAIN-ONLY macro-F1 = {macro:.3f} (single video; not held-out)")

	# --- Fit the final model on the train pool ONLY (holdout videos never seen) ---
	pipe.fit(X, y, clf__sample_weight=sample_weight)

	# --- Evaluate on the permanent holdout set (videos never trained on) ---
	X_ho, y_ho, groups_ho, _ = build_dataset(project_path, include_holdout='holdout')
	if X_ho:
		y_pred_ho = pipe.predict(X_ho)
		macro_ho = f1_score(y_ho, y_pred_ho, average='macro', labels=classes, zero_division=0)
		metrics_lines.append(f"\nHeld-out video macro-F1 (never trained on): {macro_ho:.3f}\n")
		metrics_lines.append(classification_report(y_ho, y_pred_ho, labels=classes, zero_division=0))
		metrics_lines.append("Held-out videos: " + ", ".join(sorted(set(groups_ho))) + "\n")
		print(f"  Held-out video macro-F1 = {macro_ho:.3f} ({len(set(groups_ho))} video(s))")
	else:
		metrics_lines.append("\nNo videos currently fall in the holdout set (small corpus) — "
							 "no never-trained-on evaluation available yet.\n")

	model_dir = os.path.join(project_dir, MODEL_DIR_NAME)
	os.makedirs(model_dir, exist_ok=True)
	joblib.dump({'model_kind': 'baseline', 'pipeline': pipe,
				 'classes': classes, 'params': params},
				os.path.join(model_dir, 'pipeline.joblib'))
	with open(os.path.join(model_dir, 'train_count.txt'), 'w') as f:
		f.write(str(len(X)))
	try:
		shutil.copy2(config_path, os.path.join(model_dir, 'saved_settings.ini'))
	except Exception:
		pass
	with open(os.path.join(model_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
		f.write("\n".join(metrics_lines) + "\n")
	_write_feature_importances(model_dir, pipe, params)
	print(f"  Saved complex model -> {model_dir}")
	return pipe


def _write_feature_importances(model_dir, pipe, params):
	"""Write feature importances (RandomForest) for interpretability."""
	try:
		vec = pipe.named_steps['vec']
		clf = pipe.named_steps['clf']
		names = vec.get_feature_names_out()
		if hasattr(clf, 'feature_importances_'):
			imp = clf.feature_importances_
			order = np.argsort(imp)[::-1][:30]
			with open(os.path.join(model_dir, 'feature_importances.txt'), 'w', encoding='utf-8') as f:
				f.write("Top feature importances:\n")
				for i in order:
					f.write(f"  {names[i]:<40} {imp[i]:.4f}\n")
		else:
			with open(os.path.join(model_dir, 'feature_importances.txt'), 'w', encoding='utf-8') as f:
				f.write(f"{params['baseline_classifier']} does not expose "
						"feature_importances_; use permutation importance if needed.\n")
	except Exception as e:
		print(f"  (feature importances unavailable: {e})")


def _write_merge_suggestions(project_dir, classes, cm, rate):
	"""Flag class pairs systematically confused above `rate` as MERGE candidates."""
	model_dir = os.path.join(project_dir, MODEL_DIR_NAME)
	os.makedirs(model_dir, exist_ok=True)
	suggestions = []
	support = cm.sum(axis=1)
	for i, ci in enumerate(classes):
		for j, cj in enumerate(classes):
			if i == j or support[i] == 0:
				continue
			confused = cm[i, j] / support[i]
			if confused >= rate:
				suggestions.append((ci, cj, confused))
	path = os.path.join(model_dir, 'merge_suggestions.txt')
	with open(path, 'w', encoding='utf-8') as f:
		f.write("Confusion-driven MERGE SUGGESTIONS (review manually; nothing is "
				"merged automatically).\n")
		f.write(f"Threshold: a true class confused as another >= {rate:.0%} of the time.\n\n")
		if not suggestions:
			f.write("No systematically-confused class pairs above the threshold.\n")
		else:
			for ci, cj, r in sorted(suggestions, key=lambda t: -t[2]):
				f.write(f"  '{ci}' is predicted as '{cj}' {r:.0%} of the time "
						f"-> consider merging.\n")
	if suggestions:
		print(f"  {len(suggestions)} merge suggestion(s) written to merge_suggestions.txt")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def classify_video(project_path, video_name):
	"""Predict complex behaviours over a video using sliding windows on each
	interacting ordered pair and the whole co-present herd; write
	<video>_complex_predictions.csv and populate interaction_type in the edges
	file where possible."""
	if not _SKLEARN_AVAILABLE:
		print("scikit-learn is required to classify; install it first.")
		return
	config_path = _config_path_for(project_path)
	params = load_model_config(config_path)
	project_dir, output_dir = _resolve_output_dir(config_path)

	model_path = os.path.join(project_dir, MODEL_DIR_NAME, 'pipeline.joblib')
	if not os.path.exists(model_path):
		print(f"No trained model at {model_path}; run train_model first.")
		return
	bundle = joblib.load(model_path)
	classes = bundle['classes']
	model_kind = bundle.get('model_kind', 'baseline')
	deep_predict = None
	if model_kind == 'deep':
		deep_predict = _load_deep_predictor(os.path.join(project_dir, MODEL_DIR_NAME), bundle)
		if deep_predict is None:
			return
	else:
		pipe = bundle['pipeline']

	stem = video_name
	for suf in ('_tracking_metric.csv', '_tracking_stitched.csv',
				'_tracking_corrected.csv', '_tracking.csv'):
		if os.path.basename(video_name).endswith(suf):
			stem = os.path.basename(video_name)[:-len(suf)]
			break
	# Training and prediction now read the SAME metric CSV. They used to differ
	# (train on corrected/raw, predict on metric/stitched), which silently put the
	# two in different id spaces whenever stitching was on.
	csv_path = CF.find_metric_csv(output_dir, stem)
	if csv_path is None:
		print(f"{stem}: no metric tracking CSV — nothing to classify (the model "
			  f"works in metres; run the metric stage first).")
		return

	try:
		cache = _build_video_cache(stem, csv_path, params, config_path, _video_index(config_path))
	except CF.MissingMetricError as e:
		print(f"{stem}: {e}")
		return
	win = params['window_frames']
	thr = params['predict_min_proba']

	candidates = []  # (start, end, ids)
	# Pairs: slide non-overlapping windows over each interacting ordered pair.
	for (a, b), rows in cache['pair_index'].items():
		frames = sorted(r['frame'] for r in rows)
		if len(frames) < params['min_duration_frames']:
			continue
		f0, f1 = frames[0], frames[-1]
		s = f0
		while s <= f1:
			candidates.append((s, s + win - 1, [a, b]))
			s += win
	# Whole co-present herd (group behaviours): membership per window.
	for (wstart, wend, ids) in CF.whole_herd_window_candidates(cache['track_data'], win, min_members=3):
		candidates.append((wstart, wend, ids))

	# Predict each candidate; keep those above the probability threshold.
	preds = []
	meta = []
	if deep_predict is not None:
		seqs = []
		for (s, e, ids) in candidates:
			seq, _ = _segment_sequence_dicts(cache, ids, s, e, params['seq_steps'])
			seqs.append(seq); meta.append((s, e, ids))
		proba = deep_predict(seqs) if seqs else np.zeros((0, len(classes)))
	else:
		feats = []
		for (s, e, ids) in candidates:
			feat, _ = segment_feature_dict(cache, ids, s, e)
			feats.append(feat); meta.append((s, e, ids))
		proba = pipe.predict_proba(feats) if feats else np.zeros((0, len(classes)))
	for (s, e, ids), pr in zip(meta, proba):
		j = int(np.argmax(pr))
		if pr[j] >= thr:
			preds.append({'start_frame': s, 'end_frame': e,
						  'track_ids': ';'.join(ids),
						  'behaviour': classes[j], 'probability': round(float(pr[j]), 4)})

	preds = _merge_adjacent_predictions(preds)
	out_path = os.path.join(output_dir, stem + '_complex_predictions.csv')
	with open(out_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS, extrasaction='ignore')
		w.writeheader()
		w.writerows(preds)
	print(f"  {stem}: wrote {len(preds)} prediction(s) -> {os.path.basename(out_path)}")

	_populate_edge_interaction_types(output_dir, stem, preds)


def _merge_adjacent_predictions(preds):
	"""Merge consecutive windows with the same track_ids + behaviour into segments."""
	by_key = defaultdict(list)
	for p in preds:
		by_key[(p['track_ids'], p['behaviour'])].append(p)
	merged = []
	for key, items in by_key.items():
		items.sort(key=lambda p: p['start_frame'])
		cur = dict(items[0])
		for p in items[1:]:
			if p['start_frame'] <= cur['end_frame'] + 1:
				cur['end_frame'] = max(cur['end_frame'], p['end_frame'])
				cur['probability'] = max(cur['probability'], p['probability'])
			else:
				merged.append(cur); cur = dict(p)
		merged.append(cur)
	merged.sort(key=lambda p: (p['start_frame'], p['track_ids']))
	return merged


def _populate_edge_interaction_types(output_dir, stem, preds):
	"""Fill interaction_type (and actor_id) in the edges file from predictions.

	The deterministic edges are UNDIRECTED — every metric it computes is
	symmetric — so direction enters here and only here: the prediction's
	track_ids are ordered by role, and its first id is recorded as actor_id.
	Without that column the direction the model inferred would be lost when the
	mirrored row was dropped.
	"""
	edges_path = os.path.join(output_dir, stem + '_interaction_edges.csv')
	if not os.path.exists(edges_path) or not preds:
		return
	# Best (highest-probability) behaviour per UNORDERED pair, keeping the actor.
	best = {}
	for p in preds:
		ids = p['track_ids'].split(';')
		if len(ids) == 2:
			key = tuple(sorted(ids, key=lambda t: (len(str(t)), str(t))))
			if key not in best or p['probability'] > best[key][2]:
				best[key] = (p['behaviour'], ids[0], p['probability'])
	with open(edges_path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fields = reader.fieldnames or []
		if 'interaction_type' not in fields or 'source_id' not in fields:
			return
		rows = list(reader)
	for r in rows:
		key = tuple(sorted((r.get('source_id'), r.get('target_id')),
						   key=lambda t: (len(str(t)), str(t))))
		if key in best:
			r['interaction_type'] = best[key][0]
			if 'actor_id' in fields:
				r['actor_id'] = best[key][1]
	with open(edges_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
		w.writeheader()
		w.writerows(rows)
	print(f"  {stem}: populated interaction_type for {len(best)} edge pair(s).")


# ---------------------------------------------------------------------------
# Confusion analysis (merge support)
# ---------------------------------------------------------------------------

def analyse_confusion(project_path):
	"""Re-run by-video CV and write the confusion matrix + merge suggestions.

	Does NOT merge anything; the user edits the behaviour list and re-trains.
	"""
	if not _SKLEARN_AVAILABLE:
		print("scikit-learn is required for confusion analysis.")
		return None
	config_path = _config_path_for(project_path)
	params = load_model_config(config_path)
	project_dir = os.path.dirname(config_path)
	X, y, groups, weights = build_dataset(project_path, include_holdout='train')
	if len(X) < 2 or len(set(groups)) < 2:
		print("Confusion analysis needs >=2 annotated videos for by-video CV.")
		return None
	classes = sorted(set(y))
	pipe = _build_pipeline(params)
	sw = compute_sample_weight('balanced', y) * np.array(weights, dtype=float)
	splitter = LeaveOneGroupOut() if len(set(groups)) <= 10 else GroupKFold(n_splits=5)
	try:
		y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=splitter,
								   params={'clf__sample_weight': sw})
	except TypeError:
		y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=splitter,
								   fit_params={'clf__sample_weight': sw})
	cm = confusion_matrix(y, y_pred, labels=classes)
	_write_merge_suggestions(project_dir, classes, cm, params['confusion_merge_rate'])
	print("Confusion matrix (rows=true, cols=pred):")
	print("labels:", classes)
	print(cm)
	return cm


# ---------------------------------------------------------------------------
# Optional deep models (torch): LSTM / Transformer over feature sequences
# ---------------------------------------------------------------------------
#
# The deep path consumes a *temporal sequence* of the SAME features the baseline
# pools: a labelled segment [start, end] is sliced into `seq_steps` contiguous
# sub-windows and segment_feature_dict() is computed on each, yielding one feature
# vector per timestep. The sequence model can therefore capture temporal dynamics
# the baseline aggregates away, while reusing the tested feature code, the same
# YOLO-label-bag + kinematic/graph features, and the same by-video evaluation.
#
# Saved bundle (model_complex/pipeline.joblib) carries model_kind='deep' plus the
# fitted DictVectorizer, the standardisation stats and the architecture; the torch
# weights live alongside in deep_model.pt. When torch is unavailable the trainer
# falls back to the dependency-light baseline so the user always gets a model.

DEEP_WEIGHTS_NAME = 'deep_model.pt'


def _segment_sequence_dicts(cache, ids, start, end, n_steps):
	"""Slice [start, end] into up to n_steps contiguous sub-windows and return a
	list of per-timestep feature dicts (reusing segment_feature_dict), plus the
	whole-segment quality weight."""
	start, end = int(start), int(end)
	total = max(1, end - start + 1)
	n_steps = max(1, int(n_steps))
	step = max(1, -(-total // n_steps))  # ceil division
	seq = []
	cs = start
	while cs <= end:
		ce = min(end, cs + step - 1)
		feat, _ = segment_feature_dict(cache, ids, cs, ce)
		seq.append(feat)
		cs = ce + 1
	# Whole-segment quality weight (consistent with the baseline's per-window one).
	_, qw = segment_feature_dict(cache, ids, start, end)
	return seq, qw


def build_sequence_dataset(project_path, params=None, include_holdout='train'):
	"""Like build_dataset, but each sample is a *sequence* of per-timestep feature
	dicts. Returns (X_seq, y, groups, weights). See build_dataset for
	`include_holdout` semantics."""
	config_path = _config_path_for(project_path)
	if params is None:
		params = load_model_config(config_path)
	_, output_dir = _resolve_output_dir(config_path)

	ann_files = sorted(glob.glob(os.path.join(output_dir, '*_complex_behaviours.csv')))
	X_seq, y, groups, weights = [], [], [], []
	if not ann_files:
		return X_seq, y, groups, weights

	n_steps = params['seq_steps']
	video_index = _video_index(config_path)
	for ann_path in ann_files:
		stem = os.path.basename(ann_path).replace('_complex_behaviours.csv', '')
		if include_holdout != 'all':
			is_ho = is_holdout_video(stem, params['holdout_fraction'])
			if (include_holdout == 'train') == is_ho:
				continue
		csv_path = CF.find_metric_csv(output_dir, stem)
		if csv_path is None:
			print(f"  {stem}: no metric tracking CSV — annotations skipped "
				  f"(the model works in metres; run the metric stage first).")
			continue
		try:
			cache = _build_video_cache(stem, csv_path, params, config_path, video_index)
		except CF.MissingMetricError as e:
			print(f"  {stem}: {e} — annotations skipped.")
			continue
		with open(ann_path, newline='', encoding='utf-8') as f:
			for r in csv.DictReader(f):
				try:
					s = int(r['start_frame']); e = int(r['end_frame'])
					ids = [t for t in str(r['track_ids']).split(';') if t]
					beh = (r.get('behaviour', '') or '').strip()
				except (ValueError, KeyError, TypeError):
					continue
				if not beh or not ids or s >= e:
					continue
				seq, qw = _segment_sequence_dicts(cache, ids, s, e, n_steps)
				if not seq:
					continue
				X_seq.append(seq); y.append(beh); groups.append(stem); weights.append(qw)
	return X_seq, y, groups, weights


def _vectorize_sequences(seqs, vec, mean, std):
	"""Turn a list of dict-sequences into a list of standardised float32 arrays
	[Ti, D] using a fitted DictVectorizer and per-feature mean/std."""
	out = []
	for seq in seqs:
		mat = vec.transform(seq).astype('float32')   # [Ti, D]
		mat = (mat - mean) / std
		out.append(mat)
	return out


def _make_deep_components():
	"""Build the torch-dependent pieces lazily (models + train/predict helpers).

	Returns None if torch is unavailable so callers can degrade gracefully.
	"""
	try:
		import torch
		import torch.nn as nn
	except Exception:
		return None

	class _SeqDataset(torch.utils.data.Dataset):
		def __init__(self, X_list, y_idx=None, sw=None):
			self.X = X_list
			self.y = y_idx
			self.sw = sw

		def __len__(self):
			return len(self.X)

		def __getitem__(self, i):
			x = torch.from_numpy(self.X[i])
			y = -1 if self.y is None else int(self.y[i])
			w = 1.0 if self.sw is None else float(self.sw[i])
			return x, y, w

	def _collate(batch):
		xs, ys, ws = zip(*batch)
		lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
		maxlen = int(lengths.max())
		dim = xs[0].shape[1]
		padded = torch.zeros(len(xs), maxlen, dim, dtype=torch.float32)
		for i, x in enumerate(xs):
			padded[i, :x.shape[0]] = x
		return (padded, lengths,
				torch.tensor(ys, dtype=torch.long),
				torch.tensor(ws, dtype=torch.float32))

	def _masked_mean(seq, lengths):
		# seq: [B, T, H]; mean over valid timesteps.
		mask = (torch.arange(seq.shape[1], device=seq.device)[None, :]
				< lengths[:, None].to(seq.device)).unsqueeze(-1).float()
		summed = (seq * mask).sum(dim=1)
		return summed / mask.sum(dim=1).clamp(min=1.0)

	class LSTMClassifier(nn.Module):
		def __init__(self, input_dim, hidden, n_layers, n_classes, dropout):
			super().__init__()
			self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers,
								batch_first=True, bidirectional=True,
								dropout=dropout if n_layers > 1 else 0.0)
			self.head = nn.Sequential(nn.Dropout(dropout),
									  nn.Linear(2 * hidden, n_classes))

		def forward(self, x, lengths):
			packed = nn.utils.rnn.pack_padded_sequence(
				x, lengths.cpu(), batch_first=True, enforce_sorted=False)
			out, _ = self.lstm(packed)
			out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
			return self.head(_masked_mean(out, lengths))

	class TransformerClassifier(nn.Module):
		def __init__(self, input_dim, hidden, n_layers, n_heads, n_classes, dropout):
			super().__init__()
			# d_model must be divisible by n_heads.
			d_model = max(n_heads, (hidden // n_heads) * n_heads)
			self.proj = nn.Linear(input_dim, d_model)
			self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
			layer = nn.TransformerEncoderLayer(
				d_model=d_model, nhead=n_heads, dim_feedforward=2 * d_model,
				dropout=dropout, batch_first=True)
			self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, n_layers))
			self.head = nn.Sequential(nn.Dropout(dropout),
									  nn.Linear(d_model, n_classes))
			self._d_model = d_model

		def forward(self, x, lengths):
			h = self.proj(x)
			t = h.shape[1]
			h = h + self.pos[:, :t]
			key_padding = (torch.arange(t, device=x.device)[None, :]
						   >= lengths[:, None].to(x.device))
			h = self.encoder(h, src_key_padding_mask=key_padding)
			return self.head(_masked_mean(h, lengths))

	def build_model(arch):
		if arch['model_type'] == 'transformer':
			m = TransformerClassifier(arch['input_dim'], arch['hidden'],
									  arch['n_layers'], arch['n_heads'],
									  arch['n_classes'], arch['dropout'])
		else:
			m = LSTMClassifier(arch['input_dim'], arch['hidden'],
							   arch['n_layers'], arch['n_classes'], arch['dropout'])
		return m

	def train(model, X_list, y_idx, sample_w, class_w, epochs, lr, batch):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		model.to(device)
		ds = _SeqDataset(X_list, y_idx, sample_w)
		loader = torch.utils.data.DataLoader(
			ds, batch_size=max(1, batch), shuffle=True, collate_fn=_collate)
		opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
		cw = torch.tensor(class_w, dtype=torch.float32, device=device)
		crit = nn.CrossEntropyLoss(weight=cw, reduction='none')
		model.train()
		for _ in range(max(1, epochs)):
			for padded, lengths, ys, ws in loader:
				padded, ys, ws = padded.to(device), ys.to(device), ws.to(device)
				opt.zero_grad()
				logits = model(padded, lengths)
				loss = (crit(logits, ys) * ws).mean()
				loss.backward()
				opt.step()
		return model

	def predict_proba(model, X_list, batch):
		if not X_list:
			return np.zeros((0, 0), dtype=float)
		device = next(model.parameters()).device
		ds = _SeqDataset(X_list)
		loader = torch.utils.data.DataLoader(
			ds, batch_size=max(1, batch), shuffle=False, collate_fn=_collate)
		model.eval()
		out = []
		with torch.no_grad():
			for padded, lengths, _ys, _ws in loader:
				logits = model(padded.to(device), lengths)
				out.append(torch.softmax(logits, dim=1).cpu().numpy())
		return np.concatenate(out, axis=0)

	return {
		'torch': torch, 'build_model': build_model,
		'train': train, 'predict_proba': predict_proba,
	}


def _fit_vectorizer(seqs):
	"""Fit a DictVectorizer on every timestep dict and compute standardisation
	stats. Returns (vec, mean, std)."""
	flat = [d for seq in seqs for d in seq]
	vec = DictVectorizer(sparse=False)
	mat = vec.fit_transform(flat).astype('float32')
	mean = mat.mean(axis=0)
	std = mat.std(axis=0)
	std[std < 1e-6] = 1.0
	return vec, mean, std


def _class_weights(y_idx, n_classes):
	counts = np.bincount(np.asarray(y_idx, dtype=int), minlength=n_classes).astype(float)
	counts[counts == 0] = 1.0
	w = counts.sum() / (n_classes * counts)
	return w


def _train_deep_fold(comp, X_seq, y_idx, weights, n_classes, arch, params):
	"""Fit vectorizer + a fresh model on the given (already-selected) data."""
	vec, mean, std = _fit_vectorizer(X_seq)
	X_list = _vectorize_sequences(X_seq, vec, mean, std)
	fold_arch = dict(arch, input_dim=X_list[0].shape[1], n_classes=n_classes)
	model = comp['build_model'](fold_arch)
	class_w = _class_weights(y_idx, n_classes)
	comp['train'](model, X_list, y_idx, weights, class_w,
				  params['deep_epochs'], params['deep_lr'], params['deep_batch'])
	return model, vec, mean, std, fold_arch


def _train_deep_model(project_path, params, project_dir, config_path):
	"""Train and save an LSTM/Transformer over feature sequences (needs torch).

	Falls back to the scikit-learn baseline when torch is unavailable so the user
	always gets a usable, saved model.
	"""
	if not _SKLEARN_AVAILABLE:
		print("scikit-learn is required (DictVectorizer) for the deep model too. "
			  "Install it with: pip install scikit-learn")
		return None

	comp = _make_deep_components()
	if comp is None:
		# Deliberately NOT falling back to the baseline. Doing so wrote a bundle
		# with model_kind='baseline' next to a saved_settings.ini still claiming
		# 'lstm', so the artefact contradicted its own provenance record — the one
		# thing a saved model must never do. torch is absent only when the
		# environment is broken (the launcher installs it per CPU/CUDA; ultralytics
		# needs it too), so this is a setup error, not a supported mode.
		print(f"ERROR: complex_model_type='{params['model_type']}' needs torch, which "
			  f"is not importable in this environment.\n"
			  f"  Nothing was written to {MODEL_DIR_NAME}/ — the previously saved "
			  f"model (if any) is untouched.\n"
			  f"  Either repair the environment (re-run the launcher, which installs "
			  f"the right torch build for this machine) or set "
			  f"complex_model_type = baseline in the INI if you actually want the "
			  f"scikit-learn model.")
		return None

	X_seq, y, groups, weights = build_sequence_dataset(project_path, params, include_holdout='train')
	if len(X_seq) < 2:
		print(f"Not enough annotations to train ({len(X_seq)} found). "
			  "Annotate more segments first.")
		return None

	classes = sorted(set(y))
	cls_to_idx = {c: i for i, c in enumerate(classes)}
	y_idx = [cls_to_idx[c] for c in y]
	weights = np.asarray(weights, dtype=float)
	n_classes = len(classes)
	print(f"Complex {params['model_type']} model: {len(X_seq)} annotated segments, "
		  f"{n_classes} class(es) across {len(set(groups))} video(s): {classes}")

	arch = {
		'model_type': params['model_type'],
		'hidden':   params['deep_hidden'],
		'n_layers': params['deep_layers'],
		'n_heads':  params['deep_heads'],
		'dropout':  params['deep_dropout'],
		'seq_steps': params['seq_steps'],
	}

	# --- By-video CV for honest metrics (train a fresh model per fold) ---
	metrics_lines = []
	uniq_groups = sorted(set(groups))
	if len(uniq_groups) >= 2:
		if len(uniq_groups) <= 8:
			folds = [[g] for g in uniq_groups]            # leave-one-video-out
		else:
			rng = np.random.RandomState(0)
			shuffled = list(uniq_groups); rng.shuffle(shuffled)
			folds = [shuffled[i::5] for i in range(5)]    # 5 grouped folds
		groups_arr = np.asarray(groups)
		y_true_cv, y_pred_cv = [], []
		for held in folds:
			held = set(held)
			tr = [i for i in range(len(X_seq)) if groups[i] not in held]
			te = [i for i in range(len(X_seq)) if groups[i] in held]
			if not tr or not te or len(set(y_idx[i] for i in tr)) < 2:
				continue
			model, vec, mean, std, fa = _train_deep_fold(
				comp, [X_seq[i] for i in tr], [y_idx[i] for i in tr],
				weights[tr], n_classes, arch, params)
			te_list = _vectorize_sequences([X_seq[i] for i in te], vec, mean, std)
			proba = comp['predict_proba'](model, te_list, params['deep_batch'])
			pred_idx = proba.argmax(axis=1)
			y_true_cv.extend(classes[y_idx[i]] for i in te)
			y_pred_cv.extend(classes[int(j)] for j in pred_idx)
		if y_true_cv:
			macro = f1_score(y_true_cv, y_pred_cv, average='macro',
							 labels=classes, zero_division=0)
			metrics_lines.append(f"By-video CV macro-F1 (train pool): {macro:.3f}\n")
			metrics_lines.append(classification_report(
				y_true_cv, y_pred_cv, labels=classes, zero_division=0))
			cm = confusion_matrix(y_true_cv, y_pred_cv, labels=classes)
			metrics_lines.append("\nConfusion matrix (rows=true, cols=pred):\n")
			metrics_lines.append("labels: " + ", ".join(classes) + "\n")
			metrics_lines.append(np.array2string(cm))
			_write_merge_suggestions(project_dir, classes, cm, params['confusion_merge_rate'])
			print(f"  By-video CV macro-F1 (train pool) = {macro:.3f}")
	else:
		metrics_lines.append("Only one annotated video in the train pool — no by-video "
							 "held-out CV possible. Annotate a second video for "
							 "honest scores.\n")
		print("  Single video; metrics not held-out.")

	# --- Fit the final model on the train pool ONLY (holdout videos never seen) ---
	model, vec, mean, std, final_arch = _train_deep_fold(
		comp, X_seq, y_idx, weights, n_classes, arch, params)

	# --- Evaluate on the permanent holdout set (videos never trained on) ---
	X_seq_ho, y_ho, groups_ho, _ = build_sequence_dataset(project_path, params, include_holdout='holdout')
	if X_seq_ho:
		te_list_ho = _vectorize_sequences(X_seq_ho, vec, mean, std)
		proba_ho = comp['predict_proba'](model, te_list_ho, params['deep_batch'])
		pred_idx_ho = proba_ho.argmax(axis=1)
		y_pred_ho = [classes[int(j)] for j in pred_idx_ho]
		macro_ho = f1_score(y_ho, y_pred_ho, average='macro', labels=classes, zero_division=0)
		metrics_lines.append(f"\nHeld-out video macro-F1 (never trained on): {macro_ho:.3f}\n")
		metrics_lines.append(classification_report(y_ho, y_pred_ho, labels=classes, zero_division=0))
		metrics_lines.append("Held-out videos: " + ", ".join(sorted(set(groups_ho))) + "\n")
		print(f"  Held-out video macro-F1 = {macro_ho:.3f} ({len(set(groups_ho))} video(s))")
	else:
		metrics_lines.append("\nNo videos currently fall in the holdout set (small corpus) — "
							 "no never-trained-on evaluation available yet.\n")

	model_dir = os.path.join(project_dir, MODEL_DIR_NAME)
	os.makedirs(model_dir, exist_ok=True)
	comp['torch'].save(model.state_dict(), os.path.join(model_dir, DEEP_WEIGHTS_NAME))
	joblib.dump({'model_kind': 'deep', 'classes': classes, 'params': params,
				 'vectorizer': vec, 'feat_mean': mean, 'feat_std': std,
				 'arch': final_arch},
				os.path.join(model_dir, 'pipeline.joblib'))
	with open(os.path.join(model_dir, 'train_count.txt'), 'w') as f:
		f.write(str(len(X_seq)))
	try:
		shutil.copy2(config_path, os.path.join(model_dir, 'saved_settings.ini'))
	except Exception:
		pass
	with open(os.path.join(model_dir, 'metrics.txt'), 'w', encoding='utf-8') as f:
		f.write("\n".join(metrics_lines) + "\n")
	print(f"  Saved complex {params['model_type']} model -> {model_dir}")
	return model


def _load_deep_predictor(model_dir, bundle):
	"""Reconstruct a trained deep model from a saved bundle. Returns a callable
	predict_proba(seqs)->[N,C] over dict-sequences, or None if torch is missing."""
	comp = _make_deep_components()
	if comp is None:
		print("This project's complex model is a deep (lstm/transformer) model but "
			  "torch is not installed; cannot classify. Install torch or retrain "
			  "with complex_model_type=baseline.")
		return None
	arch = bundle['arch']
	model = comp['build_model'](arch)
	state = comp['torch'].load(os.path.join(model_dir, DEEP_WEIGHTS_NAME),
							   map_location='cpu')
	model.load_state_dict(state)
	vec, mean, std = bundle['vectorizer'], bundle['feat_mean'], bundle['feat_std']
	batch = int(bundle['params'].get('deep_batch', 16))

	def predict_proba(seqs):
		X_list = _vectorize_sequences(seqs, vec, mean, std)
		return comp['predict_proba'](model, X_list, batch)

	return predict_proba


# ---------------------------------------------------------------------------
# Batch / project entry point
# ---------------------------------------------------------------------------

def run_complex_classify(project_path):
	"""Batch-classify complex behaviours for every tracking CSV in the output folder.

	Auto-pipeline entry point (mirrors run_complex_features / run_activity_budget).
	It is a deliberate NO-OP — with a clear message, never an error — when no trained
	model exists yet (model_complex/pipeline.joblib absent), which is the normal case
	for projects without complex-behaviour annotations.

	Requires <video>_tracking_metric.csv (the model works in metres) and
	classifies each distinct video stem once.
	"""
	config_path = _config_path_for(project_path)
	project_dir, output_dir = _resolve_output_dir(config_path)

	model_path = os.path.join(project_dir, MODEL_DIR_NAME, 'pipeline.joblib')
	if not os.path.exists(model_path):
		print("  No trained complex-behaviour model (model_complex/pipeline.joblib) — "
			  "skipping complex classification.")
		return

	suffix = '_tracking_metric.csv'
	stems = [os.path.basename(p)[:-len(suffix)]
			 for p in sorted(glob.glob(os.path.join(output_dir, '*' + suffix)))]

	if not stems:
		print(f"  No metric tracking CSVs in {output_dir} — nothing to classify "
			  f"(the model works in metres; run the metric stage first).")
		return

	print(f"Complex classification: processing {len(stems)} video(s)...")
	for stem in stems:
		classify_video(config_path, stem)


def count_complex_annotations(project_path, include_holdout='train'):
	"""Count the valid complex-behaviour annotation segments, mirroring EXACTLY the
	row filters and holdout/tracking-CSV skips of build_dataset (without extracting
	features). So count_complex_annotations(include_holdout='train') equals
	len(build_dataset(include_holdout='train')[0]) — i.e. the number train_model
	records in model_complex/train_count.txt. That lets us detect new annotations by
	a plain integer compare, the same way maybe_retrain does for the YOLO models.
	"""
	config_path = _config_path_for(project_path)
	params = load_model_config(config_path)
	_, output_dir = _resolve_output_dir(config_path)

	ann_files = sorted(glob.glob(os.path.join(output_dir, '*_complex_behaviours.csv')))
	n = 0
	for ann_path in ann_files:
		stem = os.path.basename(ann_path).replace('_complex_behaviours.csv', '')
		if include_holdout != 'all':
			is_ho = is_holdout_video(stem, params['holdout_fraction'])
			if (include_holdout == 'train') == is_ho:
				continue
		# build_dataset skips annotation files whose metric tracking CSV is missing.
		if CF.find_metric_csv(output_dir, stem) is None:
			continue
		with open(ann_path, newline='', encoding='utf-8') as f:
			for r in csv.DictReader(f):
				try:
					s = int(r['start_frame']); e = int(r['end_frame'])
					ids = [t for t in str(r['track_ids']).split(';') if t]
					beh = (r.get('behaviour', '') or '').strip()
				except (ValueError, KeyError, TypeError):
					continue
				if not beh or not ids or s >= e:
					continue
				n += 1
	return n


def maybe_train_complex(project_path):
	"""(Re)train the complex-behaviour model iff new annotations warrant it — the
	direct analogue of maybe_retrain() for the YOLO models (automatic, no pop-up).

	Decision, keyed off model_complex/train_count.txt exactly like the simple models:
	  - model present, current train-pool annotation count == recorded count -> no-op;
	  - model present, counts differ (and >= 2 usable segments)             -> retrain;
	  - model absent, >= 2 usable annotations                               -> first train;
	  - too few annotations to train, or none                              -> no-op.
	Returns True if a training run was performed.
	"""
	config_path = _config_path_for(project_path)
	project_dir, _ = _resolve_output_dir(config_path)
	model_path = os.path.join(project_dir, MODEL_DIR_NAME, 'pipeline.joblib')
	count_path = os.path.join(project_dir, MODEL_DIR_NAME, 'train_count.txt')

	current = count_complex_annotations(config_path, include_holdout='train')

	if os.path.exists(model_path):
		last = -1
		if os.path.exists(count_path):
			try:
				with open(count_path) as f:
					last = int(f.read().strip())
			except Exception:
				last = -1
		if current == last:
			return False  # annotations unchanged -> nothing to do
		if current < 2:
			print(f"  Complex annotations changed ({last} -> {current}) but too few to "
				  "train (need >= 2); keeping the existing model.")
			return False
		print(f"  New complex annotations detected: image/segment count changed from "
			  f"{last} to {current}. Re-training complex-behaviour model...")
		return train_model(config_path) is not None

	# No model yet: first-time train only if there is enough to learn from.
	if current < 2:
		return False
	print(f"  No complex model yet and {current} annotation segment(s) present — "
		  "training complex-behaviour model...")
	return train_model(config_path) is not None


def run_complex_stage(project_path):
	"""Full complex-behaviour stage for the auto-pipeline: (re)train on new
	annotations when needed (like the YOLO models), then classify every video.

	End-to-end NO-OP when a project has no complex annotations and no trained model,
	so behaviour is unchanged for projects that don't use complex behaviours.
	"""
	maybe_train_complex(project_path)
	run_complex_classify(project_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
	parser = argparse.ArgumentParser(
		description="Train / evaluate / run the complex-behaviour model.")
	parser.add_argument('target', help="Project directory or BehaveAI_settings.ini.")
	parser.add_argument('--action', choices=['train', 'classify', 'confusion', 'stage'],
						default='train')
	parser.add_argument('--video', default=None,
						help="Video stem for --action classify (default: all videos).")
	args = parser.parse_args()

	target = os.path.abspath(args.target)
	ini = os.path.join(target, 'BehaveAI_settings.ini') if os.path.isdir(target) else target
	if not os.path.exists(ini):
		print(f"Settings file not found: {ini}")
		sys.exit(1)

	if args.action == 'train':
		train_model(ini)
	elif args.action == 'confusion':
		analyse_confusion(ini)
	elif args.action == 'stage':
		# Same as the auto-pipeline: (re)train on new annotations, then classify all.
		run_complex_stage(ini)
	elif args.action == 'classify':
		if args.video:
			classify_video(ini, args.video)
		else:
			run_complex_classify(ini)


if __name__ == '__main__':
	_main()
