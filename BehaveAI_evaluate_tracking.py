#!/usr/bin/env python3
"""BehaveAI tracking evaluation (HOTA / DetA / AssA / IDF1 / MOTA / IDSW)

An article that *evaluates a tracking pipeline* has to report standard MOT
metrics against a hand-annotated ground truth. This wraps TrackEval (the HOTA
reference implementation, which also gives CLEAR/MOTA and Identity/IDF1 in one
run) so you can score any BehaveAI tracking CSV.

Crucially it reports DetA and AssA SEPARATELY (HOTA = sqrt(DetA x AssA)): the
detection/SAHI work moves DetA, the tracker/stitching work moves AssA, so an
ablation can only *attribute* a gain if the two are split.

What you provide: a per-sequence ground-truth file in MOT-Challenge format
(frame,id,bb_left,bb_top,w,h,conf,class,vis), made in an external tool
(CVAT/DarkLabel) and annotated at 1-2 fps (TrackEval scores only the frames
present in the GT -- ~360 boxes for 3 min x 10 horses, not 54,000). Point this
script at those GT files and the matching BehaveAI CSVs.

Usage:
  # one sequence
  python BehaveAI_evaluate_tracking.py --seq clipA \
      --gt clipA_gt.txt --pred output/clipA_tracking_stitched.csv
  # several (repeat --seq/--gt/--pred as triples, same order)
"""

import os
import csv
import tempfile
import argparse

# TrackEval 1.0.dev1 still uses the numpy aliases np.float/np.int/... that numpy
# 1.24+ removed. Restore them before TrackEval runs (harmless no-op on old numpy).
import numpy as _np
for _alias, _t in (('float', float), ('int', int), ('bool', bool),
				   ('object', object), ('str', str)):
	if not hasattr(_np, _alias):
		setattr(_np, _alias, _t)

# BehaveAI tracking CSV -> MOT-Challenge lines ------------------------------

def csv_to_mot(csv_path, out_txt):
	"""Convert a BehaveAI *_tracking(_corrected/_stitched/_metric).csv to a
	MOT-Challenge results file: frame,id,bb_left,bb_top,w,h,conf,-1,-1,-1.
	Uses the x1,y1,x2,y2 box columns. Rows without a usable box are skipped."""
	n = 0
	with open(csv_path, newline='', encoding='utf-8', errors='replace') as f, \
		 open(out_txt, 'w', encoding='utf-8') as o:
		reader = csv.DictReader(f)
		for r in reader:
			try:
				fr = int(r['frame']); tid = int(float(r['id']))
				x1 = float(r['x1']); y1 = float(r['y1'])
				x2 = float(r['x2']); y2 = float(r['y2'])
			except (ValueError, KeyError, TypeError):
				continue
			w = x2 - x1; h = y2 - y1
			if w <= 0 or h <= 0:
				continue
			conf = r.get('primary_static_conf') or r.get('primary_motion_conf') or '1'
			try:
				conf = float(conf)
			except ValueError:
				conf = 1.0
			o.write(f"{fr},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1\n")
			n += 1
	return n


def _seq_length(mot_txt):
	m = 0
	with open(mot_txt, encoding='utf-8', errors='replace') as f:
		for line in f:
			p = line.split(',')
			if p and p[0].strip().isdigit():
				m = max(m, int(p[0]))
	return m


# TrackEval driver ----------------------------------------------------------

def evaluate(seqs, work_dir=None, verbose=True):
	"""seqs: list of (name, gt_txt, pred_source). pred_source is a MOT .txt or a
	BehaveAI CSV (auto-converted). Returns a dict name -> metrics; also a
	'COMBINED_SEQ' entry. Prints a compact table."""
	import trackeval

	cleanup = work_dir is None
	work_dir = work_dir or tempfile.mkdtemp(prefix='behaveai_eval_')
	gt_root = os.path.join(work_dir, 'gt', 'mot_challenge')
	tr_root = os.path.join(work_dir, 'trackers', 'mot_challenge')
	seq_info = {}
	for name, gt_txt, pred in seqs:
		gt_dir = os.path.join(gt_root, name, 'gt')
		os.makedirs(gt_dir, exist_ok=True)
		# copy GT verbatim (already MOT format)
		with open(gt_txt, encoding='utf-8', errors='replace') as s, \
			 open(os.path.join(gt_dir, 'gt.txt'), 'w', encoding='utf-8') as d:
			d.write(s.read())
		# predictions: convert CSV or copy MOT txt
		data_dir = os.path.join(tr_root, 'behaveai', 'data')
		os.makedirs(data_dir, exist_ok=True)
		pred_txt = os.path.join(data_dir, name + '.txt')
		if pred.lower().endswith('.csv'):
			csv_to_mot(pred, pred_txt)
		else:
			with open(pred, encoding='utf-8', errors='replace') as s, \
				 open(pred_txt, 'w', encoding='utf-8') as d:
				d.write(s.read())
		seq_info[name] = max(_seq_length(os.path.join(gt_dir, 'gt.txt')),
							  _seq_length(pred_txt))

	eval_config = trackeval.Evaluator.get_default_eval_config()
	for k in ('PRINT_CONFIG', 'PRINT_RESULTS', 'OUTPUT_SUMMARY',
			  'OUTPUT_DETAILED', 'PLOT_CURVES', 'TIME_PROGRESS', 'DISPLAY_LESS_PROGRESS'):
		if k in eval_config:
			eval_config[k] = False
	ds = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
	ds.update({
		'GT_FOLDER': gt_root, 'TRACKERS_FOLDER': tr_root,
		'BENCHMARK': 'BehaveAI', 'SPLIT_TO_EVAL': 'all',
		'DO_PREPROC': False, 'SKIP_SPLIT_FOL': True,
		'TRACKERS_TO_EVAL': ['behaveai'], 'CLASSES_TO_EVAL': ['pedestrian'],
		'SEQ_INFO': seq_info, 'PRINT_CONFIG': False,
	})
	evaluator = trackeval.Evaluator(eval_config)
	dataset = trackeval.datasets.MotChallenge2DBox(ds)
	metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
	output, _ = evaluator.evaluate([dataset], metrics)

	res = output['MotChallenge2DBox']['behaveai']
	summary = {}
	for name, per_seq in res.items():
		ped = per_seq.get('pedestrian', {})
		import numpy as np
		hota = ped.get('HOTA', {})
		clear = ped.get('CLEAR', {})
		ident = ped.get('Identity', {})

		def _mean(x):
			return float(np.mean(x)) if hasattr(x, '__len__') else float(x)
		summary[name] = {
			'HOTA': _mean(hota.get('HOTA', 0.0)),
			'DetA': _mean(hota.get('DetA', 0.0)),
			'AssA': _mean(hota.get('AssA', 0.0)),
			'MOTA': _mean(clear.get('MOTA', 0.0)),
			'IDF1': _mean(ident.get('IDF1', 0.0)),
			'IDSW': int(clear.get('IDSW', 0)),
			'Frag': int(clear.get('Frag', 0)),
		}
	if verbose:
		cols = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'Frag']
		print(f"{'sequence':<24} " + ' '.join(f"{c:>7}" for c in cols))
		for name in sorted(summary):
			s = summary[name]
			print(f"{name:<24} " + ' '.join(
				(f"{s[c]:7.3f}" if c in ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA')
				 else f"{s[c]:7d}") for c in cols))
	if cleanup:
		import shutil
		shutil.rmtree(work_dir, ignore_errors=True)
	return summary


def _main():
	ap = argparse.ArgumentParser(description="Evaluate BehaveAI tracking against MOT-format GT.")
	ap.add_argument('--seq', action='append', default=[], help="sequence name (repeatable)")
	ap.add_argument('--gt', action='append', default=[], help="GT MOT .txt (repeatable, same order)")
	ap.add_argument('--pred', action='append', default=[], help="BehaveAI CSV or MOT .txt (same order)")
	ap.add_argument('--work-dir', default=None, help="keep the built layout here instead of a temp dir")
	a = ap.parse_args()
	if not (len(a.seq) == len(a.gt) == len(a.pred)) or not a.seq:
		ap.error("give equal numbers of --seq/--gt/--pred (at least one each)")
	evaluate(list(zip(a.seq, a.gt, a.pred)), work_dir=a.work_dir)


if __name__ == '__main__':
	_main()
