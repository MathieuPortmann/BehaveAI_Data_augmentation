#!/usr/bin/env python3
"""BehaveAI detection evaluation -- dual-stream ablation on the frozen holdout.

An article that claims a *dual-stream* (motion + static) detector has to show
what the second stream buys. This script scores three detection variants on the
held-out videos:

  * static-only  -- the static (RGB) primary detector alone
  * motion-only  -- the motion (false-colour) primary detector alone
  * merged       -- the two streams combined by the pipeline's own merge rule

against a class-agnostic "union of annotated animals" ground truth (detection
and classification are separated by design, so animal-finding is scored first).
It also reports the per-stream behaviour *classification* quality (confusion
matrix + per-class F1) over matched detections -- the per-frame behaviour metric.

Why it runs on the saved annotation images: the images under
``annot_static/images`` and ``annot_motion/images`` are EXACTLY what the two
detectors were trained on and what auto-annotation runs them on, so no video /
motion re-generation is needed. The holdout is the deterministic per-video split
``behaveai_holdout.is_holdout_video(stem, val_frequency)`` -- the same frozen
test set the annotation tool and the complex model use.

The merge is replicated from BehaveAI_classify_track.py:887-1430 (containment
overlap + centroid distance + dominant_source resolution) so the merged metric
reflects the deployed system. Everything is computed in normalized [0,1] box
coordinates so the two streams' images can differ in size (motion is often
``scale_factor``-downscaled).

Usage:
  python BehaveAI_evaluate_detection.py <project_dir | BehaveAI_settings.ini>
      [--split holdout|val|train|all] [--conf C] [--iou 0.5]

Outputs (under <project>/evaluation/):
  detection_report.txt          -- human-readable summary
  detection_stream_ablation.csv -- static/motion/merged P/R/F1/AP (the headline)
  detection_by_class.csv        -- per-stream, per-class classification metrics
"""

import os
import sys
import csv
import math
import argparse

import behaveai_eval_common as ec
from behaveai_config import get_species_list, load_ethogram_for_species
from behaveai_holdout import is_holdout_video, video_label_for_annotation

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


# ---------------------------------------------------------------------------
# Ground truth / prediction I/O
# ---------------------------------------------------------------------------

def read_yolo_label(path, class_names):
	"""Read a YOLO label file -> list of (class_name, (x1,y1,x2,y2) normalized).
	Missing/empty files yield an empty list."""
	out = []
	if not path or not os.path.exists(path):
		return out
	with open(path, encoding='utf-8', errors='replace') as f:
		for line in f:
			p = line.split()
			if len(p) < 5:
				continue
			try:
				ci = int(float(p[0]))
				cx, cy, w, h = (float(p[1]), float(p[2]), float(p[3]), float(p[4]))
			except ValueError:
				continue
			name = class_names[ci] if 0 <= ci < len(class_names) else str(ci)
			out.append((name, ec.yolo_to_xyxy(cx, cy, w, h)))
	return out


def predict_stream(model, img_path, conf, class_names):
	"""Run a YOLO detector on one image -> (dets, (w,h)). Each det is a dict with
	normalized 'coords', 'primary_conf', 'primary_class'."""
	res = model.predict(img_path, conf=conf, verbose=False)
	r = res[0]
	h, w = r.orig_shape
	dets = []
	for box in r.boxes:
		x1, y1, x2, y2 = box.xyxy[0].tolist()
		ci = int(box.cls[0])
		name = class_names[ci] if 0 <= ci < len(class_names) else str(ci)
		dets.append({
			'coords': (x1 / w, y1 / h, x2 / w, y2 / h),
			'primary_conf': float(box.conf[0]),
			'primary_class': name,
		})
	return dets, (w, h)


# ---------------------------------------------------------------------------
# Stream merge -- faithful replica of BehaveAI_classify_track.py:1346-1430
# ---------------------------------------------------------------------------

def merge_detections(dets, centroid_thresh, iou_thresh, dominant_source):
	"""Merge per-stream detections into one set, keeping (for a matched pair) the
	winner per the dominant_source rule. Input dets must be static-first then
	motion (the pipeline's append order), each carrying 'source'."""
	merged = []
	for det in dets:
		x1, y1, x2, y2 = det['coords']
		cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
		matched = False
		for md in merged:
			mcx, mcy = md['centroid']
			dist = math.hypot(cx - mcx, cy - mcy)
			overlap = ec.containment(det['coords'], md['coords'])
			if dist < centroid_thresh or overlap > iou_thresh:
				same = (det['source'] == md['source'])
				if same or dominant_source == 'confidence':
					if det['primary_conf'] > md['primary_conf']:
						_take(md, det, cx, cy)
				elif det['source'] == 'static' and dominant_source == 'static':
					_take(md, det, cx, cy)
				elif det['source'] == 'motion' and dominant_source == 'motion':
					_take(md, det, cx, cy)
				matched = True
				break
		if not matched:
			merged.append({
				'coords': det['coords'], 'centroid': (cx, cy),
				'primary_conf': det['primary_conf'],
				'primary_class': det['primary_class'], 'source': det['source'],
			})
	return merged


def _take(md, det, cx, cy):
	"""md adopts det's box/class/conf/source (the merge 'winner' branch)."""
	md['coords'] = det['coords']
	md['centroid'] = (cx, cy)
	md['primary_conf'] = det['primary_conf']
	md['primary_class'] = det['primary_class']
	md['source'] = det['source']


# ---------------------------------------------------------------------------
# Frame gathering
# ---------------------------------------------------------------------------

def gather_frames(project_dir, stream, keep_stem):
	"""Index original (non-augmented) annotation frames for one stream, filtered
	by keep_stem(video_stem). Returns {basename_no_ext: {'image','label'}}."""
	base = os.path.join(project_dir, 'annot_' + stream)
	out = {}
	for sub in ('train', 'val'):
		img_dir = os.path.join(base, 'images', sub)
		lbl_dir = os.path.join(base, 'labels', sub)
		if not os.path.isdir(img_dir):
			continue
		for fn in sorted(os.listdir(img_dir)):
			stem_noext, ext = os.path.splitext(fn)
			if ext.lower() not in IMG_EXTS or '_aug_' in stem_noext:
				continue
			if not keep_stem(video_label_for_annotation(stem_noext)):
				continue
			out[stem_noext] = {
				'image': os.path.join(img_dir, fn),
				'label': os.path.join(lbl_dir, stem_noext + '.txt'),
			}
	return out


def _boxes(dets):
	return [d['coords'] for d in dets]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
	ap = argparse.ArgumentParser(description="Evaluate BehaveAI dual-stream detection on the holdout.")
	ap.add_argument('project', help="project directory or BehaveAI_settings.ini path")
	ap.add_argument('--split', default='holdout', choices=('holdout', 'val', 'train', 'all'),
					help="which videos to score (default: holdout = the frozen test set)")
	ap.add_argument('--conf', type=float, default=None, help="override primary_conf_thresh")
	ap.add_argument('--iou', type=float, default=0.5, help="IoU threshold for GT matching (default 0.5)")
	a = ap.parse_args()

	config_path = ec.resolve_config_path(a.project)
	if not os.path.exists(config_path):
		ap.error(f"settings not found: {config_path}")
	project_dir = os.path.dirname(config_path)
	config = ec.load_config(config_path)
	d = config['DEFAULT']

	# Ethogram / class lists (first species uses the bare keys, matching the
	# pipeline's index->name mapping and the model_primary_* folders).
	species_list = get_species_list(config)
	etho = load_ethogram_for_species(config, species_list[0], species_list)
	static_classes = etho['primary_static_classes']
	motion_classes = etho['primary_motion_classes']

	conf_thresh = a.conf if a.conf is not None else float(d.get('primary_conf_thresh', '0.5'))
	centroid_merge_thresh = float(d.get('centroid_merge_thresh', '50'))
	iou_thresh = float(d.get('iou_thresh', '0.95'))
	dominant_source = d.get('dominant_source', 'confidence').lower()
	val_frequency = float(d.get('val_frequency', '0.1'))

	if a.split == 'train':
		keep = lambda s: not is_holdout_video(s, val_frequency)
	elif a.split == 'all':
		keep = lambda s: True
	else:  # holdout or val (val == holdout by construction of the per-video split)
		keep = lambda s: is_holdout_video(s, val_frequency)

	static_frames = gather_frames(project_dir, 'static', keep) if static_classes else {}
	motion_frames = gather_frames(project_dir, 'motion', keep) if motion_classes else {}
	all_keys = sorted(set(static_frames) | set(motion_frames))
	if not all_keys:
		print(f"No frames for split '{a.split}'. Nothing to evaluate.")
		return

	# Models (lazy import so the module stays importable without torch).
	from ultralytics import YOLO
	model_static = model_motion = None
	sp = os.path.join(project_dir, 'model_primary_static', 'train', 'weights', 'best.pt')
	mp = os.path.join(project_dir, 'model_primary_motion', 'train', 'weights', 'best.pt')
	if static_classes and os.path.exists(sp):
		model_static = YOLO(sp)
	if motion_classes and os.path.exists(mp):
		model_motion = YOLO(mp)
	if model_static is None and model_motion is None:
		print("No trained primary detector found (model_primary_static/motion). "
			  "Train the detectors first.")
		return

	# Accumulators
	variants = ('static', 'motion', 'merged')
	det = {v: {'tp': 0, 'fp': 0, 'fn': 0, 'scored': []} for v in variants}
	n_gt_union = 0
	# per-stream classification confusion: {(gt_class, pred_class): count}
	cls_conf = {'static': {}, 'motion': {}}
	cls_gt_support = {'static': {}, 'motion': {}}
	n_frames = 0

	for key in all_keys:
		s_entry = static_frames.get(key)
		m_entry = motion_frames.get(key)

		static_gt = read_yolo_label(s_entry['label'], static_classes) if s_entry else []
		motion_gt = read_yolo_label(m_entry['label'], motion_classes) if m_entry else []
		gt_union = ec.dedup_boxes([b for _, b in static_gt] + [b for _, b in motion_gt], thr=a.iou)

		static_dets = []
		ref_w = None
		if model_static is not None and s_entry:
			static_dets, (ref_w, _) = predict_stream(model_static, s_entry['image'], conf_thresh, static_classes)
		motion_dets = []
		if model_motion is not None and m_entry:
			motion_dets, (mw, _) = predict_stream(model_motion, m_entry['image'], conf_thresh, motion_classes)
			if ref_w is None:
				ref_w = mw
		for de in static_dets:
			de['source'] = 'static'
		for de in motion_dets:
			de['source'] = 'motion'

		centroid_thresh_norm = centroid_merge_thresh / ref_w if ref_w else 0.0
		merged = merge_detections(static_dets + motion_dets, centroid_thresh_norm, iou_thresh, dominant_source)

		variant_dets = {'static': static_dets, 'motion': motion_dets, 'merged': merged}
		n_gt_union += len(gt_union)
		for v in variants:
			preds = variant_dets[v]
			pboxes = _boxes(preds)
			matches, used_gt, used_pred = ec.greedy_match(gt_union, pboxes, thr=a.iou)
			det[v]['tp'] += len(matches)
			det[v]['fp'] += len(pboxes) - len(used_pred)
			det[v]['fn'] += len(gt_union) - len(used_gt)
			for pi, pd in enumerate(preds):
				det[v]['scored'].append((pd['primary_conf'], pi in used_pred))

		# Per-stream classification over matched detections
		_accumulate_classification(static_gt, static_dets, a.iou, cls_conf['static'], cls_gt_support['static'])
		_accumulate_classification(motion_gt, motion_dets, a.iou, cls_conf['motion'], cls_gt_support['motion'])
		n_frames += 1

	_write_outputs(project_dir, config_path, a, conf_thresh, centroid_merge_thresh, iou_thresh,
				   dominant_source, val_frequency, n_frames, n_gt_union, det,
				   cls_conf, cls_gt_support, static_classes, motion_classes)


def _accumulate_classification(gt, preds, iou_thr, conf_dict, support):
	"""Match preds to same-stream GT boxes; record (gt_class, pred_class) pairs.
	Classification is scored only over matched detections (detection recall is
	reported separately in the ablation table)."""
	gt_boxes = [b for _, b in gt]
	gt_cls = [c for c, _ in gt]
	pred_boxes = [p['coords'] for p in preds]
	matches, _, _ = ec.greedy_match(gt_boxes, pred_boxes, thr=iou_thr)
	for gi, pi, _ in matches:
		gc = gt_cls[gi]
		pc = preds[pi]['primary_class']
		conf_dict[(gc, pc)] = conf_dict.get((gc, pc), 0) + 1
		support[gc] = support.get(gc, 0) + 1


def _class_metrics(conf_dict, classes):
	"""Per-class precision/recall/F1 from a confusion dict (rows=gt, cols=pred),
	computed over matched detections only."""
	rows = {}
	for c in classes:
		tp = conf_dict.get((c, c), 0)
		gt_total = sum(v for (g, _), v in conf_dict.items() if g == c)
		pred_total = sum(v for (_, p), v in conf_dict.items() if p == c)
		precision, recall, f1 = ec.prf(tp, pred_total - tp, gt_total - tp)
		rows[c] = {'support': gt_total, 'tp': tp, 'precision': precision, 'recall': recall, 'f1': f1}
	return rows


def _write_outputs(project_dir, config_path, args, conf_thresh, centroid_merge_thresh, iou_thresh,
				   dominant_source, val_frequency, n_frames, n_gt_union, det,
				   cls_conf, cls_gt_support, static_classes, motion_classes):
	eval_dir = ec.ensure_eval_dir(project_dir)
	lines = []
	lines.append("BehaveAI detection evaluation")
	lines.append("=" * 60)
	lines.append(f"project            : {project_dir}")
	lines.append(f"split              : {args.split}  (val_frequency = {val_frequency})")
	lines.append(f"frames scored      : {n_frames}")
	lines.append(f"GT animals (union) : {n_gt_union}")
	lines.append(f"primary_conf_thresh: {conf_thresh}")
	lines.append(f"merge              : centroid_merge_thresh={centroid_merge_thresh}px, "
				 f"iou_thresh(containment)={iou_thresh}, dominant_source={dominant_source}")
	lines.append(f"GT match IoU       : {args.iou}")
	lines.append("")
	lines.append("Detection ablation (class-agnostic animal-finding):")
	lines.append(f"  {'variant':<10} {'TP':>6} {'FP':>6} {'FN':>6} {'P':>7} {'R':>7} {'F1':>7} {'AP@IoU':>7}")
	ablation_rows = []
	for v in ('static', 'motion', 'merged'):
		tp, fp, fn = det[v]['tp'], det[v]['fp'], det[v]['fn']
		p, r, f1 = ec.prf(tp, fp, fn)
		ap = ec.average_precision(det[v]['scored'], n_gt_union)
		lines.append(f"  {v:<10} {tp:>6} {fp:>6} {fn:>6} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {ap:>7.3f}")
		ablation_rows.append({'variant': v, 'tp': tp, 'fp': fp, 'fn': fn,
							  'precision': round(p, 4), 'recall': round(r, 4),
							  'f1': round(f1, 4), 'ap': round(ap, 4)})
	lines.append("")

	by_class_rows = []
	for stream, classes in (('static', static_classes), ('motion', motion_classes)):
		if not classes:
			continue
		rows = _class_metrics(cls_conf[stream], classes)
		macro = [rows[c]['f1'] for c in classes if rows[c]['support'] > 0]
		macro_f1 = sum(macro) / len(macro) if macro else 0.0
		lines.append(f"Behaviour classification -- {stream} stream "
					 f"(matched detections only), macro-F1 = {macro_f1:.3f}:")
		lines.append(f"  {'class':<16} {'support':>7} {'P':>7} {'R':>7} {'F1':>7}")
		for c in classes:
			row = rows[c]
			lines.append(f"  {c:<16} {row['support']:>7} {row['precision']:>7.3f} "
						 f"{row['recall']:>7.3f} {row['f1']:>7.3f}")
			by_class_rows.append({'stream': stream, 'class': c, 'support': row['support'],
								  'tp': row['tp'], 'precision': round(row['precision'], 4),
								  'recall': round(row['recall'], 4), 'f1': round(row['f1'], 4)})
		lines.append(_confusion_block(cls_conf[stream], classes))
		lines.append("")

	report = "\n".join(lines) + "\n"
	with open(os.path.join(eval_dir, 'detection_report.txt'), 'w', encoding='utf-8') as f:
		f.write(report)
	with open(os.path.join(eval_dir, 'detection_stream_ablation.csv'), 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=['variant', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'ap'])
		w.writeheader()
		w.writerows(ablation_rows)
	with open(os.path.join(eval_dir, 'detection_by_class.csv'), 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=['stream', 'class', 'support', 'tp', 'precision', 'recall', 'f1'])
		w.writeheader()
		w.writerows(by_class_rows)

	print(report)
	print(f"Wrote detection_report.txt, detection_stream_ablation.csv, detection_by_class.csv -> {eval_dir}")


def _confusion_block(conf_dict, classes):
	"""Compact text confusion matrix (rows=true, cols=pred index)."""
	if not any(conf_dict.values()):
		return "  (no matched detections)"
	idx = {c: i for i, c in enumerate(classes)}
	header = "  confusion (rows=true, cols=pred; col j = class index j):"
	out = [header, "  labels: " + ", ".join(f"{i}:{c}" for i, c in enumerate(classes))]
	for c in classes:
		row = [conf_dict.get((c, p), 0) for p in classes]
		out.append(f"  {idx[c]:>2} " + " ".join(f"{n:>4}" for n in row))
	return "\n".join(out)


if __name__ == '__main__':
	main()
