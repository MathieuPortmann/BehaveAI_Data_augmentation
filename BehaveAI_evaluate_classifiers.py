#!/usr/bin/env python3
"""BehaveAI crop-classifier evaluation -- per-class metrics for the models that
so far only ever reported a top-1 accuracy.

The secondary (static / motion), age and species models are Ultralytics
*classification* models trained on the crop pools. Training reports a single
number, ``metrics/accuracy_top1``, and on these pools that number is close to
uninformative: the held-out motion crops are 94% ``__none__``, so "94% accurate"
is exactly what a model answering ``__none__`` every time would score. Reporting
it alone invites a reader to compute the baseline and conclude the classifier
does nothing.

So, per model, this script reports:

  * per-class support / precision / recall (with a Wilson 95% CI) / F1
  * the majority-class baseline right next to top-1, so the two can be compared
  * macro-F1 over the classes that actually have held-out crops, plus balanced
    accuracy (mean per-class recall)
  * for the secondary models, the two questions a pooled accuracy conflates:
    (a) is there a secondary behaviour at all (``__none__`` vs the rest), and
    (b) which one, among the crops that have one
  * the confusion matrix as numbers -- Ultralytics only saves it as a PNG

Ground truth is the class *folder name* under ``<pool>_split/val/``, so the
metrics never depend on Ultralytics' internal class indexing; the model's own
``names`` only decode its prediction. Classes whose folder is empty on the
scored side stay in the table with support 0 rather than being dropped: "we
could not measure this class" and "this class scored 0" are different claims.

Scope, to state in the Methods: this scores the classifiers on ground-truth
crops -- oracle detection. It is a component metric, not an end-to-end one; the
detector's own errors are scored by BehaveAI_evaluate_detection.py. The val side
is the frozen per-video holdout (behaveai_holdout.build_classification_split),
the same partition the detectors use, so crops of one video never straddle it.

Usage:
  python BehaveAI_evaluate_classifiers.py <project_dir | BehaveAI_settings.ini>
      [--split val|train|all] [--models secondary_static,secondary_motion,age,species]
      [--imgsz N] [--device cpu|0] [--batch N]

Outputs (under <project>/evaluation/):
  classification_report.txt              -- human-readable, one block per model
  classification_summary.csv             -- top-1 vs baseline, macro-F1, ...
  classification_by_class.csv            -- the per-class table (the headline)
  classification_confusion_<model>.csv   -- one full matrix per model
"""

import os
import sys
import csv
import argparse

import behaveai_eval_common as ec
from behaveai_config import get_species_list, species_folder

NONE_CLASS = '__none__'
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')
UNDEFINED = ec.UNDEFINED


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_models(project_dir, config):
	"""Locate every crop classifier of the project: name, crop pool, weights.

	Mirrors the paths BehaveAI_classify_track.py builds -- the secondary pools
	are species-scoped (species_folder), species/age are not -- so a model is
	found here exactly when the pipeline would have trained and loaded it."""
	species_list = get_species_list(config)
	first = species_list[0] if species_list else None
	specs = [
		('secondary_static', species_folder('annot_static_crop', first, species_list),
		 species_folder('model_secondary_static', first, species_list)),
		('secondary_motion', species_folder('annot_motion_crop', first, species_list),
		 species_folder('model_secondary_motion', first, species_list)),
		('species', 'annot_species_crop', 'model_species'),
		('age', 'annot_age_crop', 'model_age'),
	]
	found = []
	for name, pool, model_dir in specs:
		split_dir = os.path.join(project_dir, pool + '_split')
		weights = os.path.join(project_dir, model_dir, 'train', 'weights', 'best.pt')
		if os.path.isdir(split_dir) and os.path.isfile(weights):
			found.append({'name': name, 'split_dir': split_dir, 'weights': weights,
						  'model_dir': os.path.join(project_dir, model_dir)})
	return found


def training_imgsz(model_dir, default):
	"""The imgsz the model was actually trained at, from the run's args.yaml.

	Predicting at a different size than training silently costs accuracy, and
	the INI may have moved on since the model was trained -- the run's own
	args.yaml is the only record of what this checkpoint saw."""
	path = os.path.join(model_dir, 'train', 'args.yaml')
	try:
		import yaml
		with open(path, 'r', encoding='utf-8') as f:
			return int(yaml.safe_load(f).get('imgsz', default))
	except Exception:
		return default


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def gather_crops(split_dir, split):
	"""[(image_path, true_class)] plus the full class list of the split.

	Every class folder is listed even when empty: build_classification_split
	materialises both sides for every class precisely so the two ImageFolder
	scans agree, and a class with no held-out crop is a finding, not a gap."""
	sides = ('train', 'val') if split == 'all' else (split,)
	classes, items = set(), []
	for side in sides:
		side_dir = os.path.join(split_dir, side)
		if not os.path.isdir(side_dir):
			continue
		for class_name in sorted(os.listdir(side_dir)):
			class_dir = os.path.join(side_dir, class_name)
			if not os.path.isdir(class_dir):
				continue
			classes.add(class_name)
			for fn in sorted(os.listdir(class_dir)):
				if fn.lower().endswith(IMAGE_EXTS):
					items.append((os.path.join(class_dir, fn), class_name))
	return items, sorted(classes)


def predict_crops(weights, items, imgsz, device, batch):
	"""Run the classifier over the crops -> ({(true, pred): count}, model classes).

	Predictions are decoded through the model's own ``names`` while ground truth
	comes from the folder, so a model whose class order differs from the folder
	listing is still scored correctly."""
	from ultralytics import YOLO
	model = YOLO(weights)
	names = model.names
	conf = {}
	for i in range(0, len(items), batch):
		chunk = items[i:i + batch]
		results = model.predict([p for p, _ in chunk], imgsz=imgsz, device=device,
								verbose=False)
		for (_, truth), res in zip(chunk, results):
			pred = names[int(res.probs.top1)]
			conf[(truth, pred)] = conf.get((truth, pred), 0) + 1
	return conf, [names[i] for i in sorted(names)]


def class_metrics(conf, classes):
	"""Per-class support/tp/fp/fn/P/R/F1 + Wilson CI on recall.

	precision is None when the class was never predicted, recall is None when it
	has no held-out instance: 0.0 would read as "the model got it wrong", a
	different claim from "there was nothing to get right"."""
	rows = {}
	for c in classes:
		tp = conf.get((c, c), 0)
		support = sum(v for (t, _), v in conf.items() if t == c)
		predicted = sum(v for (_, p), v in conf.items() if p == c)
		precision = tp / predicted if predicted else None
		recall = tp / support if support else None
		if precision is None and recall is None:
			f1 = None
		else:
			p_, r_ = precision or 0.0, recall or 0.0
			f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0
		lo, hi = ec.wilson_ci(tp, support)
		rows[c] = {'support': support, 'tp': tp, 'fp': predicted - tp, 'fn': support - tp,
				   'precision': precision, 'recall': recall, 'f1': f1,
				   'recall_lo': lo, 'recall_hi': hi}
	return rows


def summarise(conf, classes, rows):
	"""Model-level numbers, including the baseline that makes top-1 readable."""
	n = sum(conf.values())
	correct = sum(conf.get((c, c), 0) for c in classes)
	supports = {c: rows[c]['support'] for c in classes}
	scored = [c for c in classes if supports[c] > 0]
	majority = max(supports, key=lambda c: supports[c]) if supports else None
	f1s = [rows[c]['f1'] or 0.0 for c in scored]
	recalls = [rows[c]['recall'] or 0.0 for c in scored]
	weighted = sum((rows[c]['f1'] or 0.0) * supports[c] for c in scored)
	return {
		'n': n,
		'top1': correct / n if n else 0.0,
		'majority_class': majority,
		'majority_baseline': supports[majority] / n if n and majority else 0.0,
		'classes_with_support': len(scored),
		'classes_total': len(classes),
		'macro_f1': sum(f1s) / len(f1s) if f1s else 0.0,
		'weighted_f1': weighted / n if n else 0.0,
		'balanced_accuracy': sum(recalls) / len(recalls) if recalls else 0.0,
	}


def none_split(conf, classes):
	"""Split a secondary model's score into the two questions it conflates.

	(a) presence: ``__none__`` vs anything else -- a binary detection metric.
	(b) identity: among crops that DO have a secondary, is the right one picked.
	A pooled accuracy over a 9:1 pool answers neither, because (a) alone can
	carry it. Returns None for models with no ``__none__`` class."""
	if NONE_CLASS not in classes:
		return None
	tp = fp = fn = tn = 0
	for (truth, pred), v in conf.items():
		t_pos, p_pos = truth != NONE_CLASS, pred != NONE_CLASS
		if t_pos and p_pos:
			tp += v
		elif not t_pos and p_pos:
			fp += v
		elif t_pos and not p_pos:
			fn += v
		else:
			tn += v
	p, r, f1 = ec.prf(tp, fp, fn)
	lo, hi = ec.wilson_ci(tp, tp + fn)
	n_pos = sum(v for (t, _), v in conf.items() if t != NONE_CLASS)
	right = sum(v for (t, pr), v in conf.items() if t != NONE_CLASS and t == pr)
	acc_lo, acc_hi = ec.wilson_ci(right, n_pos)
	return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
			'precision': p, 'recall': r, 'f1': f1, 'recall_lo': lo, 'recall_hi': hi,
			'n_positive': n_pos, 'identity_correct': right,
			'identity_accuracy': right / n_pos if n_pos else 0.0,
			'identity_lo': acc_lo, 'identity_hi': acc_hi}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
	return UNDEFINED if x is None else f"{x:.{nd}f}"


def _csv_num(x, nd=4):
	return '' if x is None else round(x, nd)


def build_report(project_dir, args, results):
	lines = ["BehaveAI crop-classifier evaluation",
			 "=" * 72,
			 f"project        : {project_dir}",
			 f"split scored   : {args.split}  (frozen per-video holdout)",
			 "ground truth   : the class folder of each crop (oracle detection --",
			 "                 detector errors are scored separately, see",
			 "                 BehaveAI_evaluate_detection.py)",
			 f"undefined cells: '{UNDEFINED}' = precision with no prediction, or recall",
			 "                 with no held-out instance; 0.000 is always a real miss",
			 "CI             : Wilson score interval, 95%, on recall",
			 ""]
	for r in results:
		s, rows = r['summary'], r['rows']
		lines.append("-" * 72)
		lines.append(f"MODEL {r['name']}   ({os.path.relpath(r['weights'], project_dir)})")
		lines.append(f"  crops scored      : {s['n']}   imgsz={r['imgsz']}")
		lines.append(f"  top-1 accuracy    : {s['top1']:.4f}")
		lines.append(f"  majority baseline : {s['majority_baseline']:.4f}  "
					 f"(always predict '{s['majority_class']}')")
		lines.append(f"  gain over baseline: {s['top1'] - s['majority_baseline']:+.4f}")
		lines.append(f"  macro-F1          : {s['macro_f1']:.4f}  "
					 f"(over the {s['classes_with_support']}/{s['classes_total']} "
					 f"classes with held-out crops)")
		lines.append(f"  weighted F1       : {s['weighted_f1']:.4f}")
		lines.append(f"  balanced accuracy : {s['balanced_accuracy']:.4f}  "
					 f"(mean per-class recall)")
		lines.append("")
		lines.append(f"  {'class':<14}{'support':>8}{'P':>8}{'R':>8}  {'R 95% CI':<16}{'F1':>7}")
		for c in r['classes']:
			row = rows[c]
			lines.append(f"  {c:<14}{row['support']:>8}{_fmt(row['precision']):>8}"
						 f"{_fmt(row['recall']):>8}  "
						 f"{ec.fmt_ci(row['recall_lo'], row['recall_hi']):<16}"
						 f"{_fmt(row['f1']):>7}")
		lines.append("")
		if r['none']:
			ns = r['none']
			lines.append("  The pooled accuracy above answers two questions at once; split:")
			lines.append(f"    (a) presence of a secondary ('{NONE_CLASS}' vs rest): "
						 f"P={ns['precision']:.3f} R={ns['recall']:.3f} "
						 f"{ec.fmt_ci(ns['recall_lo'], ns['recall_hi'])} F1={ns['f1']:.3f}")
			lines.append(f"    (b) identity among the {ns['n_positive']} crops that have "
						 f"one: {ns['identity_accuracy']:.3f} "
						 f"{ec.fmt_ci(ns['identity_lo'], ns['identity_hi'])}")
			lines.append("")
		lines.append(ec.confusion_block(r['conf'], r['classes']))
		lines.append("")
	return "\n".join(lines) + "\n"


def write_outputs(project_dir, args, results):
	eval_dir = ec.ensure_eval_dir(project_dir)
	report = build_report(project_dir, args, results)

	by_class, summary = [], []
	for r in results:
		for c in r['classes']:
			row = r['rows'][c]
			by_class.append({'model': r['name'], 'class': c, 'support': row['support'],
							 'tp': row['tp'], 'fp': row['fp'], 'fn': row['fn'],
							 'precision': _csv_num(row['precision']),
							 'recall': _csv_num(row['recall']),
							 'recall_ci_low': _csv_num(row['recall_lo']),
							 'recall_ci_high': _csv_num(row['recall_hi']),
							 'f1': _csv_num(row['f1'])})
		s, ns = r['summary'], r['none']
		summary.append({'model': r['name'], 'crops': s['n'], 'imgsz': r['imgsz'],
						'top1': round(s['top1'], 4),
						'majority_baseline': round(s['majority_baseline'], 4),
						'majority_class': s['majority_class'],
						'gain_over_baseline': round(s['top1'] - s['majority_baseline'], 4),
						'macro_f1': round(s['macro_f1'], 4),
						'weighted_f1': round(s['weighted_f1'], 4),
						'balanced_accuracy': round(s['balanced_accuracy'], 4),
						'classes_with_support': s['classes_with_support'],
						'classes_total': s['classes_total'],
						'presence_precision': _csv_num(ns['precision']) if ns else '',
						'presence_recall': _csv_num(ns['recall']) if ns else '',
						'presence_f1': _csv_num(ns['f1']) if ns else '',
						'identity_n': ns['n_positive'] if ns else '',
						'identity_accuracy': _csv_num(ns['identity_accuracy']) if ns else ''})

	names = ['classification_report.txt', 'classification_summary.csv',
			 'classification_by_class.csv']
	with open(os.path.join(eval_dir, names[0]), 'w', encoding='utf-8') as f:
		f.write(report)
	with open(os.path.join(eval_dir, names[1]), 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
		w.writeheader()
		w.writerows(summary)
	with open(os.path.join(eval_dir, names[2]), 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=list(by_class[0].keys()))
		w.writeheader()
		w.writerows(by_class)
	for r in results:
		fn = f"classification_confusion_{r['name']}.csv"
		names.append(fn)
		with open(os.path.join(eval_dir, fn), 'w', newline='', encoding='utf-8') as f:
			w = csv.writer(f)
			w.writerow(['true/pred'] + r['classes'])
			for c in r['classes']:
				w.writerow([c] + [r['conf'].get((c, p), 0) for p in r['classes']])

	print(report)
	print(f"Wrote {', '.join(names)} -> {eval_dir}")


def main():
	ap = argparse.ArgumentParser(
		description="Per-class metrics for the BehaveAI crop classifiers "
					"(secondary static/motion, species, age).")
	ap.add_argument('project', help="project directory or BehaveAI_settings.ini path")
	ap.add_argument('--split', default='val', choices=('val', 'train', 'all'),
					help="which side of the frozen split to score (default: val)")
	ap.add_argument('--models', default='all',
					help="comma-separated subset of the discovered models (default: all)")
	ap.add_argument('--imgsz', type=int, default=None,
					help="override the imgsz read from each model's args.yaml")
	ap.add_argument('--device', default=None, help="ultralytics device, e.g. 0 or cpu")
	ap.add_argument('--batch', type=int, default=32, help="crops per predict call")
	args = ap.parse_args()

	config_path = ec.resolve_config_path(args.project)
	if not os.path.isfile(config_path):
		sys.exit(f"No BehaveAI_settings.ini at {config_path}")
	project_dir = os.path.dirname(os.path.abspath(config_path))
	config = ec.load_config(config_path)

	models = discover_models(project_dir, config)
	if args.models != 'all':
		wanted = {m.strip() for m in args.models.split(',') if m.strip()}
		models = [m for m in models if m['name'] in wanted]
	if not models:
		sys.exit("No trained crop classifier with a materialised _split directory was "
				 "found in this project -- nothing to evaluate.")

	ini_imgsz = int(config['DEFAULT'].get('secondary_imgsz', '224') or 224)
	results = []
	for spec in models:
		items, classes = gather_crops(spec['split_dir'], args.split)
		if not items:
			print(f"SKIP {spec['name']}: no crops on the '{args.split}' side of "
				  f"{spec['split_dir']}")
			continue
		imgsz = args.imgsz or training_imgsz(spec['model_dir'], ini_imgsz)
		print(f"Scoring {spec['name']}: {len(items)} crops, {len(classes)} classes, "
			  f"imgsz={imgsz} ...")
		conf, model_classes = predict_crops(spec['weights'], items, imgsz,
											args.device, args.batch)
		# Model order first (it is the reference), then any folder class the model
		# does not know -- a mismatch has to stay visible, not be silently dropped.
		ordered = list(model_classes) + [c for c in classes if c not in model_classes]
		extra = [c for c in classes if c not in model_classes]
		if extra:
			print(f"  WARNING: {spec['name']} has no output for annotated class(es) "
				  f"{', '.join(extra)} -- the model predates them, retrain before "
				  f"quoting these rows.")
		rows = class_metrics(conf, ordered)
		results.append({'name': spec['name'], 'weights': spec['weights'], 'imgsz': imgsz,
						'classes': ordered, 'conf': conf, 'rows': rows,
						'summary': summarise(conf, ordered, rows),
						'none': none_split(conf, ordered)})

	if not results:
		sys.exit("Nothing scored.")
	write_outputs(project_dir, args, results)


if __name__ == '__main__':
	main()
