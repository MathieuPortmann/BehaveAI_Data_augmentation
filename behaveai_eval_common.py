#!/usr/bin/env python3
"""Shared helpers for the BehaveAI evaluation scripts.

Small, dependency-light utilities reused by BehaveAI_evaluate_detection.py and
BehaveAI_evaluate_geometry.py: project/INI resolution (mirroring the pipeline
post-processing scripts), directory resolution, box geometry, greedy IoU
matching, and the per-project ``evaluation/`` output directory.

Stdlib ``csv`` + ``numpy`` only -- no pandas (repo convention).
"""

import os
import math
import configparser

import numpy as np

# Printed wherever a metric is undefined rather than zero (no held-out instance
# to recall, or no prediction to be precise about).
UNDEFINED = '—'


# ---------------------------------------------------------------------------
# Project / config resolution (mirrors metric_geometry / activity_budget)
# ---------------------------------------------------------------------------

def resolve_config_path(arg):
	"""Resolve a CLI argument (a project directory OR a BehaveAI_settings.ini
	path) to the settings INI path, exactly like the pipeline scripts do."""
	arg = os.path.abspath(arg)
	if os.path.isdir(arg):
		return os.path.join(arg, "BehaveAI_settings.ini")
	return arg


def load_config(config_path):
	"""Read an INI with case-sensitive keys (optionxform = str)."""
	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(config_path)
	return config


def resolve_dir(project_dir, config, key, default):
	"""Resolve a possibly-relative directory from the INI against project_dir."""
	raw = config['DEFAULT'].get(key, default) or default
	return raw if os.path.isabs(raw) else os.path.join(project_dir, raw)


def ensure_eval_dir(project_dir):
	"""Create and return <project_dir>/evaluation (generated outputs live here,
	never committed -- see .gitignore)."""
	d = os.path.join(project_dir, "evaluation")
	os.makedirs(d, exist_ok=True)
	return d


# ---------------------------------------------------------------------------
# Box geometry
# ---------------------------------------------------------------------------

def iou_xyxy(a, b):
	"""Standard intersection-over-union of two (x1,y1,x2,y2) boxes."""
	xa = max(a[0], b[0]); ya = max(a[1], b[1])
	xb = min(a[2], b[2]); yb = min(a[3], b[3])
	inter = max(0.0, xb - xa) * max(0.0, yb - ya)
	area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
	area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
	union = area_a + area_b - inter
	return inter / union if union > 0 else 0.0


def containment(a, b):
	"""Max proportional overlap -- the SAME quantity the pipeline's stream-merge
	calls ``iou`` (BehaveAI_classify_track.py:887-901): 1.0 when one box is fully
	inside the other. Replicated here so the merged-detection evaluation matches
	the deployed merge rule exactly."""
	xa = max(a[0], b[0]); ya = max(a[1], b[1])
	xb = min(a[2], b[2]); yb = min(a[3], b[3])
	inter = max(0.0, xb - xa) * max(0.0, yb - ya)
	area_a = max(1e-12, (a[2] - a[0]) * (a[3] - a[1]))
	area_b = max(1e-12, (b[2] - b[0]) * (b[3] - b[1]))
	prop_a = inter / area_a
	prop_b = inter / area_b
	return max(prop_a, prop_b) if inter > 0 else 0.0


def yolo_to_xyxy(cx, cy, w, h):
	"""Normalized YOLO (cx,cy,w,h) -> normalized (x1,y1,x2,y2)."""
	return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def dedup_boxes(boxes, thr=0.5):
	"""Collapse near-duplicate boxes (IoU >= thr), keeping the first of each
	group. Used to build a class-agnostic 'union of animals' ground truth from
	the two per-stream label sets."""
	kept = []
	for b in boxes:
		if all(iou_xyxy(b, k) < thr for k in kept):
			kept.append(b)
	return kept


# ---------------------------------------------------------------------------
# Greedy matching
# ---------------------------------------------------------------------------

def greedy_match(gt, pred, thr=0.5, iou_fn=iou_xyxy):
	"""Greedy one-to-one matching of predicted boxes to GT boxes by descending
	IoU. Returns (matches, used_gt, used_pred) where matches is a list of
	(gt_idx, pred_idx, iou). TP = len(matches); FP = len(pred) - |used_pred|;
	FN = len(gt) - |used_gt|."""
	pairs = []
	for gi, g in enumerate(gt):
		for pi, p in enumerate(pred):
			v = iou_fn(g, p)
			if v >= thr:
				pairs.append((v, gi, pi))
	pairs.sort(reverse=True)
	used_gt, used_pred, matches = set(), set(), []
	for v, gi, pi in pairs:
		if gi in used_gt or pi in used_pred:
			continue
		used_gt.add(gi); used_pred.add(pi)
		matches.append((gi, pi, v))
	return matches, used_gt, used_pred


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def prf(tp, fp, fn):
	"""Precision, recall, F1 from counts (0 when undefined)."""
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
	return precision, recall, f1


def wilson_ci(k, n, z=1.959963985):
	"""Wilson score 95% confidence interval for the proportion k/n.

	The per-class tables are dominated by rare behaviours with 1-5 held-out
	instances, where a bare point estimate invites over-reading ("recall 0.20").
	Wilson rather than the normal approximation because it stays inside [0,1] and
	still returns a usable interval at k=0, k=n and n<10 -- exactly the regime
	those classes live in. Returns (lo, hi), or (None, None) when n == 0.
	"""
	if n <= 0:
		return (None, None)
	p = k / n
	denom = 1.0 + z * z / n
	centre = (p + z * z / (2 * n)) / denom
	half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
	return (max(0.0, centre - half), min(1.0, centre + half))


def fmt_ci(lo, hi, undefined=None):
	"""Render a (lo, hi) interval for a text table, or the undefined marker.

	Shared so the detection and classifier tables print intervals identically."""
	if lo is None:
		return UNDEFINED if undefined is None else undefined
	return f"[{lo:.3f}-{hi:.3f}]"


def confusion_block(conf_dict, classes, indent='  '):
	"""Compact text confusion matrix (rows=true, cols=pred index).

	Shared by the detection and classifier evaluations so both reports read the
	same way; Ultralytics only ever saves this as a PNG, which cannot be quoted
	in a table."""
	if not any(conf_dict.values()):
		return indent + "(no predictions)"
	idx = {c: i for i, c in enumerate(classes)}
	out = [indent + "confusion (rows=true, cols=pred; col j = class index j):",
		   indent + "labels: " + ", ".join(f"{i}:{c}" for i, c in enumerate(classes))]
	for c in classes:
		row = [conf_dict.get((c, p), 0) for p in classes]
		out.append(f"{indent}{idx[c]:>2} " + " ".join(f"{n:>4}" for n in row))
	return "\n".join(out)


def average_precision(scored, n_gt):
	"""VOC-style AP from a list of (confidence, is_tp) over ALL frames of a
	variant, against n_gt ground-truth boxes. Integrates precision over recall.

	scored: list of (conf, bool_is_tp), one entry per predicted box (already
	greedily matched at the eval IoU so each GT is a TP at most once).
	"""
	if n_gt == 0:
		return 0.0
	order = sorted(scored, key=lambda s: s[0], reverse=True)
	tp = 0
	fp = 0
	rec_prev = 0.0
	ap = 0.0
	for _, is_tp in order:
		if is_tp:
			tp += 1
		else:
			fp += 1
		recall = tp / n_gt
		precision = tp / (tp + fp)
		ap += precision * (recall - rec_prev)
		rec_prev = recall
	return ap
