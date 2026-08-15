#!/usr/bin/env python3
"""Tests for the frame miner and for the per-stream columns it reads.

Runnable two ways, like the stitcher tests:
    python tests/test_mine_frames.py
    pytest tests/test_mine_frames.py

Each test pins a property the selection depends on. Several of them exist
because the obvious implementation is wrong in a way that only shows up after a
day of annotating: summing scores instead of taking the max makes crowded frames
win regardless of difficulty, and ranking without a spacing rule returns the same
confused second over and over.
"""

import os
import re
import ast
import sys
import csv
import random
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BehaveAI_mine_frames import (
	signal_det_gap, signal_det_missed, signal_det_lowconf, signal_pair_unseen,
	signal_flicker, signal_rare_class, collapse, select, interleave,
	observed_pairs, annotated_frames, tracking_csvs, load_tracking_csv,
	parse_quota_overrides, load_project, spread_by_video,
	DEFAULT_QUOTAS, PAIR_MIN_SUPPORT,
)


# --- helpers ---------------------------------------------------------------

def row(frame, tid, static=('', 0.0), motion=('', 0.0), age=('', 0.0), species=0.0):
	return {'frame': frame, 'id': tid,
			'static_class': static[0], 'static_conf': static[1],
			'motion_class': motion[0], 'motion_conf': motion[1],
			'species_conf': species, 'age_class': age[0], 'age_conf': age[1],
			'source': 'static' if static[0] else 'motion'}


def cand(stem, frame, score, hits=1, detail=''):
	return (stem, frame, score, detail, hits)


def load_from_module(filename, names):
	"""Execute a few top-level definitions of a script module in isolation.

	BehaveAI_classify_track and BehaveAI_annotation both open a settings dialog
	and chdir at import time, so they cannot be imported from a test. The
	functions under test are pure, so their source is extracted and run alone.
	"""
	path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), filename)
	src = open(path, encoding='utf-8').read()
	tree = ast.parse(src)
	wanted = []
	for node in tree.body:
		if isinstance(node, (ast.FunctionDef, ast.Assign)):
			target = (node.name if isinstance(node, ast.FunctionDef)
					  else getattr(node.targets[0], 'id', None))
			if target in names:
				wanted.append(node)
	ns = {'os': os, 're': re, 'csv': csv}
	exec(compile(ast.Module(body=wanted, type_ignores=[]), path, 'exec'), ns)
	missing = [n for n in names if n not in ns]
	assert not missing, f"{missing} not found in {filename}"
	return ns


def load_record_stream_verdict():
	"""Pull record_stream_verdict out of BehaveAI_classify_track without importing it.

	That module opens a settings dialog and chdirs at import time, so it cannot be
	imported from a test; the function itself is pure, so its source is extracted
	and executed on its own.
	"""
	path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
						'BehaveAI_classify_track.py')
	src = open(path, encoding='utf-8').read()
	tree = ast.parse(src)
	for node in tree.body:
		if isinstance(node, ast.FunctionDef) and node.name == 'record_stream_verdict':
			ns = {}
			exec(compile(ast.Module(body=[node], type_ignores=[]), path, 'exec'), ns)
			return ns['record_stream_verdict']
	raise AssertionError('record_stream_verdict not found in BehaveAI_classify_track.py')


# --- the CSV fix the miner depends on --------------------------------------

def test_stream_verdict_files_each_source_separately():
	"""The bug: two static detections merging filed a static class as motion."""
	rsv = load_record_stream_verdict()
	md = {'static_class': '', 'static_conf': 0.0, 'motion_class': '', 'motion_conf': 0.0}
	rsv(md, {'source': 'static', 'primary_class': 'Graze', 'primary_conf': 0.4})
	rsv(md, {'source': 'static', 'primary_class': 'Stand', 'primary_conf': 0.9})
	assert md['static_class'] == 'Stand', md
	assert md['motion_class'] == '', md          # nothing motion ever saw it
	assert md['motion_conf'] == 0.0, md


def test_stream_verdict_keeps_the_best_of_each_stream():
	rsv = load_record_stream_verdict()
	md = {'static_class': '', 'static_conf': 0.0, 'motion_class': '', 'motion_conf': 0.0}
	rsv(md, {'source': 'static', 'primary_class': 'Graze', 'primary_conf': 0.7})
	rsv(md, {'source': 'motion', 'primary_class': 'Walk', 'primary_conf': 0.3})
	rsv(md, {'source': 'motion', 'primary_class': 'Trot', 'primary_conf': 0.8})
	assert (md['static_class'], md['static_conf']) == ('Graze', 0.7), md
	assert (md['motion_class'], md['motion_conf']) == ('Trot', 0.8), md


# --- signals ---------------------------------------------------------------

def test_gap_is_found_and_reported_at_its_middle():
	rows = [row(10, 1), row(11, 1), row(16, 1), row(17, 1)]
	out = signal_det_gap(rows, track_buffer=30)
	assert len(out) == 1, out
	frame, score, _detail = out[0]
	assert score == 4.0, out          # frames 12..15 missing
	assert 11 < frame < 16, out


def test_gap_longer_than_the_track_buffer_is_not_a_gap():
	"""Past track_buffer the tracker has dropped the identity: the same id
	reappearing is a coincidence of numbering, not a missed detection."""
	rows = [row(10, 1), row(100, 1)]
	assert signal_det_gap(rows, track_buffer=30) == []


def test_only_motion_only_detections_count_as_missed():
	rows = [row(1, 1, static=('Graze', 0.9)),                    # static-only: the norm
			row(2, 2, motion=('Walk', 0.6)),                     # motion-only: a miss
			row(3, 3, static=('Stand', 0.8), motion=('Walk', 0.5))]
	out = signal_det_missed(rows)
	assert [f for f, _s, _d in out] == [2], out


def test_lowconf_band_excludes_confident_and_ranks_the_least_sure_first():
	rows = [row(1, 1, static=('Graze', 0.95)),
			row(2, 2, static=('Graze', 0.30)),
			row(3, 3, static=('Graze', 0.12))]
	out = signal_det_lowconf(rows, band=(0.10, 0.35))
	frames = [f for f, _s, _d in out]
	assert frames == [2, 3], out
	scores = {f: s for f, s, _d in out}
	assert scores[3] > scores[2], out


def test_pair_unseen_needs_the_static_class_to_be_well_observed():
	"""Without the support gate this stratum reports label sparsity, not error."""
	rows = [row(1, 1, static=('Recumbent', 0.8), motion=('Gallop', 0.7))]
	pairs = {('Stand', 'Walk')}
	assert signal_pair_unseen(rows, pairs, {'Recumbent': PAIR_MIN_SUPPORT - 1}) == []
	out = signal_pair_unseen(rows, pairs, {'Recumbent': PAIR_MIN_SUPPORT})
	assert len(out) == 1, out
	assert out[0][1] == 0.7, out      # scored by the LOWER of the two confidences


def test_pair_that_humans_did_annotate_is_not_flagged():
	rows = [row(1, 1, static=('Graze', 0.8), motion=('Walk', 0.7))]
	out = signal_pair_unseen(rows, {('Graze', 'Walk')}, {'Graze': 500})
	assert out == []


def test_flicker_counts_class_changes_inside_the_window():
	rows = [row(f, 1, static=(cls, 0.9)) for f, cls in
			[(0, 'Stand'), (2, 'Graze'), (4, 'Stand'), (6, 'Graze')]]
	out = signal_flicker(rows, window=15, min_changes=2)
	assert out, out
	assert max(s for _f, s, _d in out) >= 3


def test_flicker_stays_linear_on_a_full_length_video():
	"""The first version rescanned the rest of the track for every row, which on a
	ten-minute clip (~18 000 rows per track) took minutes instead of a fraction of
	a second. The bound is deliberately loose — it only has to catch quadratic."""
	import time
	rows = [row(f, tid, static=('Graze' if (f // 7) % 2 else 'Stand', 0.9))
			for f in range(18000) for tid in (1, 2, 3, 4, 5)]
	t0 = time.time()
	signal_flicker(rows)
	elapsed = time.time() - t0
	assert elapsed < 10.0, f"signal_flicker took {elapsed:.1f}s on 90k rows"


def test_a_stable_track_never_flickers():
	rows = [row(f, 1, static=('Graze', 0.9)) for f in range(0, 30, 2)]
	assert signal_flicker(rows, window=15, min_changes=2) == []


def test_rare_class_ignores_classes_absent_from_the_rarity_map():
	rows = [row(1, 1, static=('Graze', 0.9)), row(2, 2, static=('Nurse', 0.2))]
	out = signal_rare_class(rows, {'Nurse': 0.8})
	assert [f for f, _s, _d in out] == [2], out


# --- aggregation -----------------------------------------------------------

def test_frame_score_is_the_max_not_the_sum():
	"""Otherwise a crowd of easy animals outranks a single genuinely hard one."""
	raw = [(7, 0.2, 'a'), (7, 0.2, 'b'), (7, 0.2, 'c'), (9, 0.5, 'd')]
	got = {f: score for f, score, _detail, _hits in collapse(raw)}
	assert got[7] == 0.2, got
	assert got[9] == 0.5, got


def test_collapse_keeps_the_hit_count_as_tiebreak():
	raw = [(7, 0.5, 'a'), (7, 0.3, 'b'), (9, 0.5, 'c')]
	hits = {f: h for f, _s, _d, h in collapse(raw)}
	assert hits == {7: 2, 9: 1}, hits


# --- selection -------------------------------------------------------------

def test_quotas_are_respected():
	candidates = {
		'det_gap': [cand('v', f, 1.0) for f in range(0, 2000, 100)],
		'flicker': [cand('w', f, 1.0) for f in range(0, 2000, 100)],
	}
	out = select(candidates, {'det_gap': 0.5, 'flicker': 0.5}, budget=10,
				 min_spacing=1, max_per_video=100, excluded=set())
	assert len(out['det_gap']) == 5, out
	assert len(out['flicker']) == 5, out


def test_spacing_rejects_near_duplicates_of_the_same_second():
	candidates = {'det_gap': [cand('v', f, 1.0) for f in (100, 101, 102, 400)]}
	out = select(candidates, {'det_gap': 1.0}, budget=10,
				 min_spacing=90, max_per_video=100, excluded=set())
	frames = sorted(f for _s, f, _sc, _d in out['det_gap'])
	assert frames == [100, 400], frames


def test_per_video_cap_keeps_other_videos_in_the_mix():
	candidates = {'det_gap': [cand('busy', f, 5.0) for f in range(0, 5000, 200)] +
							 [cand('quiet', 10, 0.1)]}
	out = select(candidates, {'det_gap': 1.0}, budget=10,
				 min_spacing=1, max_per_video=3, excluded=set())
	stems = [s for s, _f, _sc, _d in out['det_gap']]
	assert stems.count('busy') == 3, stems
	assert 'quiet' in stems, stems


def test_already_annotated_frames_are_never_proposed():
	candidates = {'det_gap': [cand('v', 10, 9.0), cand('v', 500, 1.0)]}
	out = select(candidates, {'det_gap': 1.0}, budget=10,
				 min_spacing=1, max_per_video=100, excluded={('v', 10)})
	frames = [f for _s, f, _sc, _d in out['det_gap']]
	assert frames == [500], frames


def test_unfilled_quota_is_handed_to_the_other_strata():
	candidates = {
		'det_gap': [cand('v', 100, 1.0)],                       # only one to give
		'flicker': [cand('w', f, 1.0) for f in range(0, 3000, 100)],
	}
	out = select(candidates, {'det_gap': 0.5, 'flicker': 0.5}, budget=10,
				 min_spacing=1, max_per_video=100, excluded=set())
	assert len(out['det_gap']) == 1, out
	assert sum(len(v) for v in out.values()) == 10, out


def test_leftover_budget_is_shared_not_won_by_the_biggest_numbers():
	"""det_gap scores are frame counts (tens), lowconf scores are margins (<0.35).
	Sorting the leftover pool by raw score therefore handed the whole remainder to
	whichever stratum happened to use the largest unit."""
	candidates = {
		'det_gap':     [cand('v', f, 25.0) for f in range(0, 4000, 100)],
		'det_lowconf': [cand('w', f, 0.2) for f in range(0, 4000, 100)],
		'random':      [cand('x', f, 0.99) for f in range(0, 4000, 100)],
	}
	quotas = {'det_gap': 0.1, 'det_lowconf': 0.1, 'random': 0.1}
	out = select(candidates, quotas, budget=30, min_spacing=1,
				 max_per_video=100, excluded=set())
	counts = {s: len(v) for s, v in out.items()}
	assert sum(counts.values()) == 30, counts
	# Nobody may run away with it: 30 frames over 3 strata, evenly.
	assert max(counts.values()) - min(counts.values()) <= 1, counts


def test_a_frame_is_never_proposed_twice_across_strata():
	shared = [cand('v', 10, 1.0)]
	out = select({'det_gap': list(shared), 'flicker': list(shared)},
				 {'det_gap': 0.5, 'flicker': 0.5}, budget=10,
				 min_spacing=1, max_per_video=100, excluded=set())
	total = [(s, f) for items in out.values() for s, f, _sc, _d in items]
	assert len(total) == len(set(total)) == 1, out


def test_interleaving_keeps_any_prefix_balanced():
	selected = {'a': [('v', f, 1.0, '') for f in range(5)],
				'b': [('w', f, 1.0, '') for f in range(5)]}
	order = interleave(selected)
	strata = [s for _stem, _f, s, _sc, _d in order[:4]]
	assert strata.count('a') == 2 and strata.count('b') == 2, strata


# --- disk-facing helpers ---------------------------------------------------

def _write_label(path, rows):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, 'w', encoding='utf-8') as f:
		for r in rows:
			f.write(' '.join(str(x) for x in r) + '\n')


def test_observed_pairs_matches_the_two_streams_by_geometry():
	with tempfile.TemporaryDirectory() as tmp:
		s_dir = os.path.join(tmp, 'annot_static')
		m_dir = os.path.join(tmp, 'annot_motion')
		# Same animal in both trees (index 3 = Graze, index 0 = Walk).
		_write_label(os.path.join(s_dir, 'labels', 'train', 'clip_100.txt'),
					 [(3, 0.5, 0.5, 0.1, 0.1)])
		_write_label(os.path.join(m_dir, 'labels', 'train', 'clip_100.txt'),
					 [(0, 0.5, 0.5, 0.1, 0.1)])
		# A second frame where the boxes are far apart: not the same animal.
		_write_label(os.path.join(s_dir, 'labels', 'train', 'clip_200.txt'),
					 [(0, 0.1, 0.1, 0.05, 0.05)])
		_write_label(os.path.join(m_dir, 'labels', 'train', 'clip_200.txt'),
					 [(1, 0.9, 0.9, 0.05, 0.05)])

		statics = ['Stand', 'Recumbent', 'Drink', 'Graze']
		motions = ['Walk', 'Trot']
		pairs, totals = observed_pairs(s_dir, m_dir, statics, motions)
		assert ('Graze', 'Walk') in pairs, pairs
		assert ('Stand', 'Trot') not in pairs, pairs
		assert totals['Graze'] == 1 and totals['Stand'] == 1, totals


def test_annotated_frames_reads_the_tool_naming_convention():
	with tempfile.TemporaryDirectory() as tmp:
		s_dir = os.path.join(tmp, 'annot_static')
		d = os.path.join(s_dir, 'images', 'train')
		os.makedirs(d)
		for name in ('Mini3Pro_2026-04-27_12-33-45_Harems_Site_6263.jpg', 'clip_10.jpg'):
			open(os.path.join(d, name), 'w').close()
		got = annotated_frames(s_dir, os.path.join(tmp, 'annot_motion'))
		assert ('clip', 10) in got, got
		assert ('Mini3Pro_2026-04-27_12-33-45_Harems_Site', 6263) in got, got


def test_derived_tracking_csvs_are_not_mined_twice():
	with tempfile.TemporaryDirectory() as tmp:
		for name in ('a_tracking.csv', 'a_tracking_corrected.csv',
					 'a_tracking_stitched.csv', 'b_tracking.csv'):
			open(os.path.join(tmp, name), 'w').close()
		got = sorted(os.path.basename(p) for p in tracking_csvs(tmp))
		assert got == ['a_tracking.csv', 'b_tracking.csv'], got


def test_a_csv_without_the_source_column_is_flagged_as_stale():
	with tempfile.TemporaryDirectory() as tmp:
		path = os.path.join(tmp, 'old_tracking.csv')
		with open(path, 'w', newline='', encoding='utf-8') as f:
			w = csv.writer(f)
			w.writerow(['frame', 'id', 'primary_static_class', 'primary_static_conf'])
			w.writerow([1, 1, 'Graze', '0.900'])
		rows, has_source = load_tracking_csv(path)
		assert has_source is False
		assert rows[0]['static_class'] == 'Graze', rows


# --- the contract between the miner and the annotation tool ----------------

class _FakeCapture:
	"""Stands in for a video the parser probes for fps.

	Stubbed rather than imported so the suite keeps running without OpenCV, and
	so the fps is fixed here instead of depending on a file on disk.
	"""
	def isOpened(self):
		return True

	def get(self, prop):
		return 30.0 if prop == 'fps' else 9000.0

	def release(self):
		pass


class _FakeCv2:
	CAP_PROP_FPS = 'fps'
	CAP_PROP_FRAME_COUNT = 'count'

	@staticmethod
	def VideoCapture(_path):
		return _FakeCapture()


def _parse_targets(csv_text, stem='clipB'):
	"""Run the annotation tool's CSV parser over some text, without importing it."""
	ns = load_from_module('BehaveAI_annotation.py',
						  {'_TIMECODE_RE', '_parse_frame_value', 'parse_timecode_csv'})
	ns['cv2'] = _FakeCv2
	with tempfile.TemporaryDirectory() as tmp:
		path = os.path.join(tmp, 'targets.csv')
		with open(path, 'w', encoding='utf-8', newline='') as f:
			f.write(csv_text)
		pool = [(os.path.join(tmp, f'{stem}.mp4'), 0)]
		meta = {}
		targets = ns['parse_timecode_csv'](path, pool, 1, {}, meta)
		return targets, meta


def test_the_tool_reads_what_the_miner_writes():
	"""The miner's own header, straight out of BehaveAI_mine_frames.mine()."""
	targets, meta = _parse_targets(
		"video_filename,frame,timecode,reason,score,detail\n"
		"clipB,405,00:13,det_gap,11.0000,track 2 lost for 11 frame(s)\n"
		"clipB,603,00:20,flicker,2.0000,track 1: 2 static changes in 15 frames\n")
	assert len(targets) == 2, targets
	assert [f for _v, f in targets] == [405, 603], targets
	memo = meta[targets[0]]
	assert 'det_gap' in memo and 'track 2 lost' in memo, memo


def test_a_plain_timecode_csv_still_works_and_shows_its_memo():
	"""The human-written format predates the miner and must keep working; its
	`behaviour` column is the same kind of memo, so it is shown too."""
	targets, meta = _parse_targets(
		"video_filename,timecode,behaviour\n"
		"clipB,02:15,walking\n")
	assert len(targets) == 1, targets
	assert meta[targets[0]] == 'walking', meta


def test_a_csv_without_memo_columns_leaves_the_title_alone():
	targets, meta = _parse_targets("video_filename,frame\nclipB,120\n")
	assert len(targets) == 1, targets
	assert meta == {}, meta


# --- CLI -------------------------------------------------------------------

def test_quota_overrides_renormalise():
	out = parse_quota_overrides('flicker=1.0', DEFAULT_QUOTAS)
	assert abs(sum(out.values()) - 1.0) < 1e-9, out
	assert out['flicker'] > DEFAULT_QUOTAS['flicker'], out


def test_default_quotas_sum_to_one():
	assert abs(sum(DEFAULT_QUOTAS.values()) - 1.0) < 1e-9, DEFAULT_QUOTAS


def test_rare_class_holds_the_largest_share():
	"""The imbalance is the project's binding constraint, so the stratum that
	can move a starved class outranks the ones that refine a well-fed one."""
	assert DEFAULT_QUOTAS['rare_class'] == max(DEFAULT_QUOTAS.values()), DEFAULT_QUOTAS
	assert DEFAULT_QUOTAS['rare_class'] > DEFAULT_QUOTAS['flicker'], DEFAULT_QUOTAS


def test_spread_by_video_breaks_one_clip_owning_a_stratum():
	"""Score alone lets the hardest video own the top of the list: its frames all
	score high, so the budget buys one herd on one afternoon."""
	# clipA has the six best scores; clipB and clipC trail.
	items = ([cand('clipA', 100 * i, 10.0 - i) for i in range(6)]
			 + [cand('clipB', 100 * i, 3.0 - i * 0.1) for i in range(3)]
			 + [cand('clipC', 100 * i, 2.0 - i * 0.1) for i in range(3)])
	ranked = sorted(items, key=lambda c: (-c[2], -c[4], c[0], c[1]))
	spread = spread_by_video(ranked)

	assert len(spread) == len(items), "no candidate may be dropped"
	assert sorted(spread) == sorted(items), "only the order may change"
	# The first round covers every video before any video gets a second frame.
	assert {c[0] for c in spread[:3]} == {'clipA', 'clipB', 'clipC'}, spread[:3]
	# Score still decides the order inside a video.
	a_frames = [c[1] for c in spread if c[0] == 'clipA']
	assert a_frames == [0, 100, 200, 300, 400, 500], a_frames
	# The richest video still leads each round.
	assert spread[0][0] == 'clipA', spread[0]


def test_selection_spreads_across_videos_within_a_stratum():
	"""The end-to-end consequence: a 6-frame budget must not come from one clip
	when three have candidates."""
	candidates = {'det_gap': (
		[cand('clipA', 1000 * i, 10.0 - i) for i in range(8)]
		+ [cand('clipB', 1000 * i, 3.0) for i in range(4)]
		+ [cand('clipC', 1000 * i, 2.0) for i in range(4)])}
	out = select(candidates, {'det_gap': 1.0}, 6, min_spacing=90,
				 max_per_video=20, excluded=set())
	videos = {stem for stem, _f, _s, _d in out['det_gap']}
	assert len(out['det_gap']) == 6, out
	assert videos == {'clipA', 'clipB', 'clipC'}, videos


def _minimal_project(tmp, extra=''):
	"""Smallest INI load_project accepts, so the budget wiring can be tested
	without a full project on disk."""
	ini = os.path.join(tmp, 'BehaveAI_settings.ini')
	with open(ini, 'w', encoding='utf-8') as f:
		f.write('[DEFAULT]\nspecies = Horse\noutput_dir = output\n' + extra)
	return tmp


def test_mining_budget_comes_from_the_ini():
	with tempfile.TemporaryDirectory() as tmp:
		proj = load_project(_minimal_project(tmp, 'mining_budget = 42\n'))
		assert proj['mining_budget'] == 42, proj['mining_budget']


def test_mining_budget_defaults_when_the_ini_predates_the_setting():
	with tempfile.TemporaryDirectory() as tmp:
		proj = load_project(_minimal_project(tmp))
		assert proj['mining_budget'] == 300, proj['mining_budget']


def _annotation_source():
	path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
						'BehaveAI_annotation.py')
	return open(path, encoding='utf-8').read()


def test_startup_picks_the_mined_frame_before_opening_a_capture():
	"""The point of the pre-cache is not paying a 4K seek interactively. Choosing
	the first frame *after* cv2.VideoCapture would pay exactly that, on a random
	clip, and then throw it away."""
	src = _annotation_source()
	choose = src.index('_autostart_targets:\n\tvideo_path, _initial_frame')
	open_cap = src.index('capture = cv2.VideoCapture(video_path)')
	assert choose < open_cap, "the startup frame must be chosen before the capture opens"


def test_startup_cursor_does_not_re_serve_the_frame_on_screen():
	"""csv_cursor is the index of the NEXT target; target[0] is already shown."""
	src = _annotation_source()
	block = src[src.index('if _autostart_targets:\n\t# Adopt the list'):]
	block = block[:block.index('else:')]
	assert 'csv_cursor = 1' in block, block[:400]


def test_startup_falls_back_to_random_without_a_list():
	src = _annotation_source()
	assert 'video_path, _initial_frame = pick_random_frame(unannotated_pool)' in src


def test_launcher_offers_the_miner_above_annotate():
	"""The launcher runs tools as `python <script> <project_dir>`, so the miner
	has to accept the project positionally; and the button is only useful before
	annotating, which is what its position in the stage list says."""
	# `stages` lives inside the launcher's __init__, so it is picked out of the
	# tree rather than executed: importing BehaveAI.py would build a window.
	path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BehaveAI.py')
	stages = None
	for node in ast.walk(ast.parse(open(path, encoding='utf-8').read())):
		if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'stages':
			stages = ast.literal_eval(node.value)
	assert stages, "no `stages` list found in BehaveAI.py"
	stage = dict(stages)['2 - Annotate']
	scripts = [s for _label, s in stage]
	assert 'BehaveAI_mine_frames.py' in scripts, scripts
	assert scripts.index('BehaveAI_mine_frames.py') < scripts.index('BehaveAI_annotation.py'), scripts


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
