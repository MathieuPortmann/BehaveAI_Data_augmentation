#!/usr/bin/env python3
"""BehaveAI frame miner — choose the next frames to annotate.

Uniform random sampling reproduces nature: in Horses_HERDWISE, Graze + Stand are
90 % of the static boxes while Head_threat, Stamp, Rear and Strike sit at zero.
Annotating twice as fast under uniform sampling just yields twice as much Graze,
so the bottleneck is *which* frames get annotated, not how quickly.

This reads the tracking CSVs a full inference pass leaves in <project>/output/,
scores every frame by the ways the pipeline is visibly struggling on it, and
writes a `mining_targets.csv` the annotation tool loads as a frame source (the
"Mining" nav mode; the plain CSV time-code mode reads it too, since the extra
columns are ignored there).

WHAT IT LOOKS FOR (each one a stratum with its own budget share)

  det_gap      A track exists at frame t-1 and t+1 but has no detection at t.
               BoT-SORT keeps a lost track alive for `track_buffer` frames, so
               the tracker itself tells us where the detector dropped an animal
               — the cheapest possible proxy for a miss, and it needs no second
               inference pass.
  det_missed   Only the motion stream saw the animal. Static-only is the norm
               (the motion stream fires on movers alone, recall ~0.08), so it
               carries no information; motion-only is the interesting direction,
               a moving animal the static detector missed.
  det_lowconf  Winning confidence inside a band (default 0.10-0.35): the model
               sees something without daring. Note this is bounded below by the
               project's own primary_conf_thresh — animals under that threshold
               are in no CSV at all and need a dedicated low-threshold pass.
  pair_unseen  The two streams report a (static, motion) combination the human
               annotations never contain. The static and motion taxonomies are
               different questions (posture vs gait), not competing answers, so
               "the classes differ" means nothing on its own; what is
               informative is a *combination* no annotator ever produced, e.g.
               Recumbent + Gallop. The reference is the project's own labels, so
               no plausibility table has to be invented.
  flicker      A track whose class keeps changing inside a short window —
               the ambiguous transitions, which for this project means mostly
               the Stand/Graze boundary that carries 90 % of the boxes.
  rare_class   Any detection of a class that is rare in the annotations, at any
               confidence. Useless for the four classes at zero examples: a
               detector cannot predict a class it has never seen. Those need
               either a human ethogram sweep or the geometric proposals of
               BehaveAI_complex_candidates.py.
  attr         Low-confidence age or species crop verdicts.
  random       Uniform frames, deliberately kept in the mix. It is the only
               unbiased stratum, it covers the blind spots where the model has
               no signal to emit at all, and it is the control that lets a paper
               state what mining actually bought (rare-class yield per frame,
               mined vs random).

WHY QUOTAS AND NOT ONE COMPOSITE SCORE

The signals share no unit — a confidence of 0.12 and "three class changes per
second" cannot be added without inventing weights, and those weights would have
to be defended in a methods section. Each stratum instead gets a share of the
annotation budget and is ranked internally by its own natural score. The output
is interleaved round-robin, so any prefix of the file is still a balanced sample
and stopping early costs nothing.

TWO EXCLUSIONS THAT MATTER

  * Holdout videos are skipped. Mined frames are harder than average by
    construction, so letting them into the validation side would silently make
    the test set harder than the phenomenon and the metrics incomparable with
    earlier runs. Use --include-holdout only if you know why you want that.
  * Frames already annotated are skipped, on the same <stem>_<frame> convention
    the annotation tool uses.

Usage:
  python BehaveAI_mine_frames.py --project projects/Horses_HERDWISE
  python BehaveAI_mine_frames.py --project <dir> --budget 300 --fps 30 \
      --quota flicker=0.30,rare_class=0.20
"""

import os
import re
import csv
import sys
import glob
import random
import argparse
import configparser
from collections import defaultdict

from behaveai_config import (
	get_species_list, species_folder, load_ethogram_for_species, load_age_classes,
	resolve_project_dir,
)
from behaveai_holdout import is_holdout_video
import behaveai_frame_cache as frame_cache

# Budget share per stratum. Detection (gap + missed + lowconf) takes 0.35 because
# an animal that is never detected can be neither tracked, classified nor
# counted: its error is not recoverable downstream, unlike a wrong class.
DEFAULT_QUOTAS = {
	'det_gap':     0.20,
	'det_missed':  0.08,
	'det_lowconf': 0.05,
	'pair_unseen': 0.20,
	'flicker':     0.10,
	'rare_class':  0.25,
	'attr':        0.04,
	'random':      0.08,
}
# rare_class carries the largest share on purpose. The imbalance is the
# project's binding constraint — Graze and Stand are 90 % of the static boxes
# and four classes sit at zero — and a rare-class frame is the only kind that
# can move a class the detector currently cannot predict at all. flicker and
# random gave up most of the room: flicker concentrates on the Stand/Graze
# boundary, which is precisely where examples are least scarce, and random is a
# control that only needs to stay measurable.
# Override per project with mining_quota_overrides, or per run with --quota.

# A stratum is only worth a share of the budget if the signal behind it is real;
# these are the knobs that decide "real".
LOWCONF_BAND = (0.10, 0.35)
FLICKER_WINDOW = 15          # frames (~0.5 s at 30 fps)
FLICKER_MIN_CHANGES = 2      # changes inside the window before it counts
ATTR_CONF_MAX = 0.60         # age/species verdicts under this are "unsure"
RARE_MAX_COUNT = 100         # a class with fewer annotated boxes than this is rare.
                             # Raised from 50 because that cut-off excluded the
                             # classes just above it (Groom at 63) while the
                             # detector is still visibly bad at them; the score
                             # scales with scarcity anyway, so a wider net costs
                             # ranking, not precision. Per project:
                             # mining_rare_max_count.
PAIR_MIN_SUPPORT = 30        # a static class needs this many observed boxes
                             # before "never seen with X" means anything
BOX_MATCH_IOU = 0.5


# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------

def load_project(project_arg):
	"""Resolve a project directory (or its INI) into everything the miner needs."""
	project_arg = os.path.abspath(project_arg)
	if os.path.isdir(project_arg):
		config_path = os.path.join(project_arg, 'BehaveAI_settings.ini')
	else:
		config_path = project_arg
	if not os.path.exists(config_path):
		raise SystemExit(f"No settings INI at {config_path}")
	project_dir = os.path.dirname(config_path)

	config = configparser.ConfigParser()
	config.optionxform = str
	config.read(config_path)
	d = config['DEFAULT']

	species_list = get_species_list(config)
	species = species_list[0]
	etho = load_ethogram_for_species(config, species, species_list)
	ages = load_age_classes(config, species, species_list)

	def _folder(base):
		return os.path.join(project_dir, species_folder(base, species, species_list))

	# The shared resolver owns the "blank means <project>/output" rule; this
	# used to re-implement it, which is how the variants drifted before.
	output_dir = resolve_project_dir(config, project_dir, 'output')

	return {
		'project_dir': project_dir,
		'output_dir': os.path.normpath(output_dir),
		'static_dir': _folder('annot_static'),
		'motion_dir': _folder('annot_motion'),
		'static_classes': etho['primary_static_classes'],
		'motion_classes': etho['primary_motion_classes'],
		'age_classes': ages['age_classes'],
		'val_frequency': float(d.get('val_frequency', '0.2') or 0.2),
		'conf_thresh': float(d.get('primary_conf_thresh', '0.1') or 0.1),
		'track_buffer': int(float(d.get('tracker_track_buffer', '30') or 30)),
		'mining_budget': int(float(d.get('mining_budget', '300') or 300)),
		'precache_workers': int(float(d.get('mining_precache_workers', '4') or 4)),
		'precache_animation': str(d.get('mining_precache_animation', 'true')).strip().lower() != 'false',
		'rare_max_count': int(float(d.get('mining_rare_max_count', str(RARE_MAX_COUNT))
									or RARE_MAX_COUNT)),
		'quota_overrides': str(d.get('mining_quota_overrides', '') or '').strip(),
		'max_per_video': int(float(d.get('mining_max_per_video', '20') or 20)),
		'clips_dir': resolve_project_dir(config, project_dir, 'clips'),
		'config': config,
	}


VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv')


def index_videos(clips_dir):
	"""Map video stem -> path, matching the annotation tool's convention.

	Lower-cased because a tracking CSV is named from the file on disk while the
	mined list is matched against the same pool the annotation tool builds, and
	a case difference between the two is not a real mismatch.
	"""
	index = {}
	if not os.path.isdir(clips_dir):
		return index
	for dirpath, _dirs, files in os.walk(clips_dir):
		for name in sorted(files):
			if name.lower().endswith(VIDEO_EXTENSIONS):
				index.setdefault(os.path.splitext(name)[0].lower(),
                                 os.path.join(dirpath, name))
	return index


# --------------------------------------------------------------------------
# What the human already labelled
# --------------------------------------------------------------------------

def _label_files(annot_dir):
	"""Every label file of an annot_* tree, train and val side."""
	out = []
	for split in ('train', 'val'):
		out.extend(glob.glob(os.path.join(annot_dir, 'labels', split, '*.txt')))
	return out


def _read_label_file(path):
	"""YOLO label file -> [(class_index, xc, yc, w, h)], skipping junk lines."""
	rows = []
	try:
		with open(path, 'r', encoding='utf-8', errors='replace') as f:
			for line in f:
				parts = line.split()
				if len(parts) < 5:
					continue
				try:
					rows.append((int(parts[0]), float(parts[1]), float(parts[2]),
								 float(parts[3]), float(parts[4])))
				except ValueError:
					continue
	except OSError:
		pass
	return rows


def annotated_frames(static_dir, motion_dir):
	"""(stem, frame) pairs already annotated, from the image filenames.

	Same <video_label>_<frame> convention the annotation tool writes and reads,
	so a frame this returns is one the tool would refuse to serve again.
	"""
	done = set()
	for base in (static_dir, motion_dir):
		for split in ('train', 'val'):
			for p in glob.glob(os.path.join(base, 'images', split, '*.jpg')):
				stem = os.path.splitext(os.path.basename(p))[0]
				if '_' not in stem:
					continue
				label, tail = stem.rsplit('_', 1)
				try:
					done.add((label, int(tail)))
				except ValueError:
					continue
	return done


def class_counts(annot_dir, classes):
	"""How many annotated boxes each class has. Drives the rare_class stratum."""
	counts = {c: 0 for c in classes}
	for path in _label_files(annot_dir):
		for cls_idx, _xc, _yc, _w, _h in _read_label_file(path):
			if 0 <= cls_idx < len(classes):
				counts[classes[cls_idx]] += 1
	return counts


def _iou_norm(a, b):
	"""IoU of two normalised (xc, yc, w, h) boxes."""
	ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
	ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
	bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
	bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
	ix1, iy1 = max(ax1, bx1), max(ay1, by1)
	ix2, iy2 = min(ax2, bx2), min(ay2, by2)
	inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
	if inter <= 0:
		return 0.0
	union = a[2] * a[3] + b[2] * b[3] - inter
	return inter / union if union > 0 else 0.0


def observed_pairs(static_dir, motion_dir, static_classes, motion_classes):
	"""Which (static class, motion class) combinations a human ever produced.

	Static and motion labels of one frame describe the same animals on the same
	geometry, so a box in each tree matching by IoU is one animal seen by both
	taxonomies. Returns (pairs, static_totals): the set of observed combinations
	and how many boxes back each static class, because "never seen with X" only
	means something for a class that has been seen often enough at all.
	"""
	pairs = set()
	static_totals = defaultdict(int)

	static_by_base = {}
	for path in _label_files(static_dir):
		static_by_base[os.path.splitext(os.path.basename(path))[0]] = path
	motion_by_base = {}
	for path in _label_files(motion_dir):
		motion_by_base[os.path.splitext(os.path.basename(path))[0]] = path

	for base, spath in static_by_base.items():
		s_rows = _read_label_file(spath)
		for s_idx, *s_box in s_rows:
			if 0 <= s_idx < len(static_classes):
				static_totals[static_classes[s_idx]] += 1

		mpath = motion_by_base.get(base)
		if not mpath:
			continue
		m_rows = _read_label_file(mpath)
		for s_idx, *s_box in s_rows:
			if not (0 <= s_idx < len(static_classes)):
				continue
			for m_idx, *m_box in m_rows:
				if not (0 <= m_idx < len(motion_classes)):
					continue
				if _iou_norm(s_box, m_box) >= BOX_MATCH_IOU:
					pairs.add((static_classes[s_idx], motion_classes[m_idx]))
	return pairs, dict(static_totals)


# --------------------------------------------------------------------------
# Tracking CSVs
# --------------------------------------------------------------------------

def tracking_csvs(output_dir):
	"""Raw per-frame detector output only.

	The _corrected / _stitched / _metric variants are the same detections after
	drone compensation, gap linking and ground-plane projection: mining wants
	what the detector actually produced, and counting a frame several times
	would skew every stratum.
	"""
	found = sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv')))
	return [p for p in found
			if not re.search(r'_(corrected|stitched|metric)\.csv$', os.path.basename(p))]


def load_tracking_csv(path):
	"""Read one tracking CSV into row dicts, keeping only usable rows.

	Returns (rows, has_source). `has_source` is False for CSVs produced before
	the per-stream columns were filled by source; the pair_unseen stratum is not
	trustworthy on those (primary_motion_class could hold a static verdict).
	"""
	rows = []
	has_source = False
	with open(path, newline='', encoding='utf-8', errors='replace') as f:
		reader = csv.DictReader(f)
		has_source = 'source' in (reader.fieldnames or [])
		for r in reader:
			try:
				frame = int(r['frame'])
				tid = int(float(r['id']))
			except (KeyError, TypeError, ValueError):
				continue

			def _f(key):
				try:
					return float(r.get(key) or 0.0)
				except (TypeError, ValueError):
					return 0.0

			rows.append({
				'frame': frame,
				'id': tid,
				'static_class': (r.get('primary_static_class') or '').strip(),
				'static_conf': _f('primary_static_conf'),
				'motion_class': (r.get('primary_motion_class') or '').strip(),
				'motion_conf': _f('primary_motion_conf'),
				'species_conf': _f('species_conf'),
				'age_class': (r.get('age_class') or '').strip(),
				'age_conf': _f('age_conf'),
				'source': (r.get('source') or '').strip(),
			})
	return rows, has_source


# --------------------------------------------------------------------------
# Signals — each returns [(frame, score, detail)]
# --------------------------------------------------------------------------

def signal_det_gap(rows, track_buffer):
	"""Frames where a track survives but its detection does not.

	Only the middle frame of each gap is proposed: the whole gap is one event,
	and serving five near-identical frames of it would spend the budget on a
	single animal.
	"""
	by_track = defaultdict(list)
	for r in rows:
		by_track[r['id']].append(r['frame'])

	out = []
	for tid, frames in by_track.items():
		frames = sorted(set(frames))
		for a, b in zip(frames, frames[1:]):
			gap = b - a - 1
			if gap <= 0 or gap > track_buffer:
				continue
			middle = a + (b - a) // 2
			out.append((middle, float(gap), f"track {tid} lost for {gap} frame(s)"))
	return out


def signal_det_missed(rows):
	"""Motion-only detections: a moving animal the static detector did not see.

	The mirror case (static-only) is left out on purpose. The motion stream only
	responds to animals that move, so static-only describes most of the corpus
	and would drown every other stratum without pointing at a defect.
	"""
	out = []
	for r in rows:
		if r['motion_class'] and not r['static_class']:
			out.append((r['frame'], r['motion_conf'],
						f"motion-only: {r['motion_class']} {r['motion_conf']:.2f}"))
	return out


def signal_det_lowconf(rows, band=LOWCONF_BAND):
	"""Detections whose winning confidence sits in the hesitation band."""
	lo, hi = band
	out = []
	for r in rows:
		conf = max(r['static_conf'], r['motion_conf'])
		if lo <= conf <= hi:
			cls = r['static_class'] or r['motion_class']
			# Lower inside the band is more interesting, hence hi - conf.
			out.append((r['frame'], hi - conf, f"{cls} at {conf:.2f}"))
	return out


def signal_pair_unseen(rows, pairs, static_totals):
	"""Stream combinations no annotator ever produced.

	Gated on the static class being well observed: with a few hundred annotated
	frames most legitimate combinations simply have not come up yet, and without
	the gate this stratum would mostly report the sparsity of the label set.
	"""
	out = []
	for r in rows:
		s, m = r['static_class'], r['motion_class']
		if not s or not m:
			continue
		if static_totals.get(s, 0) < PAIR_MIN_SUPPORT:
			continue
		if (s, m) in pairs:
			continue
		# Only confident nonsense is worth a frame; both streams must mean it.
		out.append((r['frame'], min(r['static_conf'], r['motion_conf']),
					f"{s}+{m} never annotated together"))
	return out


def signal_flicker(rows, window=FLICKER_WINDOW, min_changes=FLICKER_MIN_CHANGES):
	"""Tracks whose class will not settle inside a short window."""
	by_track = defaultdict(list)
	for r in rows:
		by_track[r['id']].append(r)

	out = []
	for tid, track in by_track.items():
		track.sort(key=lambda r: r['frame'])
		# Two pointers, not a rescan per row: a ten-minute video is ~18 000 rows
		# per track and the quadratic form of this loop takes minutes on it.
		end = 0
		for i, r in enumerate(track):
			if end < i + 1:
				end = i + 1
			while end < len(track) and track[end]['frame'] - r['frame'] < window:
				end += 1
			if end - i < 2:
				continue
			window_rows = track[i:end]
			for key in ('static_class', 'motion_class'):
				labels = [q[key] for q in window_rows if q[key]]
				changes = sum(1 for a, b in zip(labels, labels[1:]) if a != b)
				if changes >= min_changes:
					mid = window_rows[len(window_rows) // 2]['frame']
					out.append((mid, float(changes),
								f"track {tid}: {changes} {key.split('_')[0]} changes "
								f"in {window} frames"))
	return out


def signal_rare_class(rows, rarity):
	"""Any detection of a class the annotation set barely contains.

	`rarity` maps class name -> weight in (0, 1]; classes at or above
	RARE_MAX_COUNT boxes are absent from it.
	"""
	out = []
	for r in rows:
		for cls, conf in ((r['static_class'], r['static_conf']),
						  (r['motion_class'], r['motion_conf'])):
			if cls and cls in rarity:
				out.append((r['frame'], rarity[cls] * max(conf, 0.01),
							f"rare: {cls} at {conf:.2f}"))
	return out


def signal_attr(rows, age_classes):
	"""Unsure age or species verdicts on the crop classifiers."""
	out = []
	multi_age = len(age_classes) > 1
	for r in rows:
		if multi_age and r['age_class'] and 0 < r['age_conf'] < ATTR_CONF_MAX:
			out.append((r['frame'], ATTR_CONF_MAX - r['age_conf'],
						f"age {r['age_class']} at {r['age_conf']:.2f}"))
		elif 0 < r['species_conf'] < ATTR_CONF_MAX:
			out.append((r['frame'], ATTR_CONF_MAX - r['species_conf'],
						f"species at {r['species_conf']:.2f}"))
	return out


def signal_random(rows, rng, n):
	"""Uniform frames from the range this video actually covers."""
	if not rows:
		return []
	lo = min(r['frame'] for r in rows)
	hi = max(r['frame'] for r in rows)
	if hi <= lo:
		return []
	return [(rng.randint(lo, hi), rng.random(), 'random control') for _ in range(n)]


def collapse(raw):
	"""Per-detection hits -> per-frame candidates.

	The score is the max over the frame's detections, not the sum: one hard
	animal is what makes a frame worth opening. The count of hits is kept as the
	tie-break, because every animal in the frame has to be labelled anyway and a
	frame holding several hard cases returns more boxes for the same overhead.
	"""
	best = {}
	for frame, score, detail in raw:
		cur = best.get(frame)
		if cur is None or score > cur[0]:
			best[frame] = (score, detail, (cur[2] + 1) if cur else 1)
		else:
			best[frame] = (cur[0], cur[1], cur[2] + 1)
	return [(f, v[0], v[1], v[2]) for f, v in best.items()]


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def spread_by_video(ranked_items):
	"""Re-order an already-ranked stratum breadth-first across videos.

	Sorting by score alone lets one clip own a stratum: the videos the detector
	struggles most on produce the highest scores everywhere in the clip, so the
	top of the list is the same herd, the same light, the same afternoon. The
	per-video cap does not help when the budget is smaller than cap x videos,
	which is the normal case — 300 frames over 19 videos never reaches a cap of
	20.

	So: best frame of each video, then second-best of each, and so on. Score
	still decides the order *inside* a video, and no candidate is dropped —
	only the order changes, which is what the quota fill then reads down.
	"""
	by_video = defaultdict(list)
	for item in ranked_items:
		by_video[item[0]].append(item)
	# Videos with the most candidates go first within a round, so a clip that is
	# genuinely rich in signal still leads — it just cannot monopolise.
	order = sorted(by_video, key=lambda s: (-len(by_video[s]), s))
	out = []
	for i in range(max((len(v) for v in by_video.values()), default=0)):
		for stem in order:
			if i < len(by_video[stem]):
				out.append(by_video[stem][i])
	return out


def select(candidates, quotas, budget, min_spacing, max_per_video, excluded):
	"""Fill each stratum's share, then spend the remainder on what is left.

	candidates : {stratum: [(stem, frame, score, detail, hits)]}
	excluded   : set of (stem, frame) that must never be picked

	Anti-redundancy is the whole point of the spacing and per-video caps: ranking
	by score alone returns a hundred near-identical frames of the one sequence
	where the model is lost, which is a single lesson bought a hundred times.
	"""
	taken = set()
	per_video = defaultdict(int)
	chosen_frames = defaultdict(list)
	out = defaultdict(list)

	def _accept(stem, frame):
		if (stem, frame) in taken or (stem, frame) in excluded:
			return False
		if per_video[stem] >= max_per_video:
			return False
		for f in chosen_frames[stem]:
			if abs(f - frame) < min_spacing:
				return False
		return True

	def _commit(stratum, stem, frame, score, detail):
		taken.add((stem, frame))
		per_video[stem] += 1
		chosen_frames[stem].append(frame)
		out[stratum].append((stem, frame, score, detail))

	ranked = {}
	for stratum, items in candidates.items():
		ranked[stratum] = spread_by_video(
			sorted(items, key=lambda c: (-c[2], -c[4], c[0], c[1])))

	# Pass 1 — each stratum up to its share.
	for stratum, quota in quotas.items():
		target = int(round(budget * quota))
		for stem, frame, score, detail, _hits in ranked.get(stratum, []):
			if len(out[stratum]) >= target:
				break
			if _accept(stem, frame):
				_commit(stratum, stem, frame, score, detail)

	# Pass 2 — a stratum that could not fill its share (no gaps found, no rare
	# class predicted) hands the leftover budget to the others rather than
	# shrinking the day's work.
	#
	# Round-robin, not one sorted pool: the scores are a gap length, a change
	# count, a confidence margin and a uniform draw, so sorting them together
	# just ranks the strata by numeric scale. That is the very comparison the
	# quotas exist to avoid, and it let the random control take half the budget.
	remaining = budget - sum(len(v) for v in out.values())
	cursors = {s: 0 for s in ranked}
	while remaining > 0:
		progressed = False
		for stratum in list(quotas.keys()) + [s for s in ranked if s not in quotas]:
			if remaining <= 0:
				break
			items = ranked.get(stratum, [])
			i = cursors.get(stratum, 0)
			while i < len(items):
				stem, frame, score, detail, _hits = items[i]
				i += 1
				if _accept(stem, frame):
					_commit(stratum, stem, frame, score, detail)
					remaining -= 1
					progressed = True
					break
			cursors[stratum] = i
		if not progressed:
			break   # every stratum exhausted; the corpus simply has no more

	return out


def interleave(selected):
	"""Round-robin the strata so any prefix of the file stays balanced."""
	queues = {s: list(items) for s, items in selected.items() if items}
	order = []
	while queues:
		for stratum in list(queues.keys()):
			if not queues[stratum]:
				del queues[stratum]
				continue
			stem, frame, score, detail = queues[stratum].pop(0)
			order.append((stem, frame, stratum, score, detail))
	return order


def frame_to_timecode(frame, fps):
	if fps <= 0:
		return ''
	total = int(frame / fps)
	return f"{total // 60:02d}:{total % 60:02d}"


# --------------------------------------------------------------------------

def mine(project_arg, budget, fps, min_spacing_s, max_per_video, quotas,
		 include_holdout, seed, out_path=None, log=print):
	proj = load_project(project_arg)
	# None means "no explicit --budget", which is how the launcher button runs:
	# the INI is then the single place the size of a mining pass is set.
	if budget is None:
		budget = proj['mining_budget']
	csv_paths = tracking_csvs(proj['output_dir'])
	if not csv_paths:
		raise SystemExit(
			f"No *_tracking.csv in {proj['output_dir']}.\n"
			"Run BehaveAI_classify_track.py over the videos you want to mine first.")

	rarity_counts = class_counts(proj['static_dir'], proj['static_classes'])
	rarity_counts.update(class_counts(proj['motion_dir'], proj['motion_classes']))
	# A class at zero examples is left out on purpose: the detector cannot
	# predict what it has never seen, so no confidence of its will ever appear.
	rare_max = proj['rare_max_count']
	rarity = {c: 1.0 - (n / rare_max)
			  for c, n in rarity_counts.items() if 0 < n < rare_max}

	pairs, static_totals = observed_pairs(
		proj['static_dir'], proj['motion_dir'],
		proj['static_classes'], proj['motion_classes'])
	excluded = annotated_frames(proj['static_dir'], proj['motion_dir'])

	rng = random.Random(seed)
	min_spacing = max(1, int(round(min_spacing_s * fps)))
	candidates = defaultdict(list)
	skipped_holdout, stale_csvs, used = [], [], []

	log(f"Scoring {len(csv_paths)} tracking CSV(s) from {proj['output_dir']}")
	log(f"  {len(excluded)} frame(s) already annotated will be excluded")
	if rarity:
		log(f"  rare classes (< {rare_max} boxes): "
			+ ', '.join(f"{c} ({rarity_counts[c]})" for c in sorted(rarity, key=lambda c: rarity_counts[c])))
	log(f"  {len(pairs)} (static, motion) combination(s) observed in the annotations")

	for n, path in enumerate(csv_paths, 1):
		stem = os.path.basename(path)[:-len('_tracking.csv')]
		if not include_holdout and is_holdout_video(stem, proj['val_frequency']):
			skipped_holdout.append(stem)
			log(f"  [{n}/{len(csv_paths)}] {stem}: skipped (holdout video)")
			continue

		rows, has_source = load_tracking_csv(path)
		if not rows:
			log(f"  [{n}/{len(csv_paths)}] {stem}: skipped (no rows)")
			continue
		if not has_source:
			stale_csvs.append(stem)
		used.append(stem)

		signals = {
			'det_gap':     signal_det_gap(rows, proj['track_buffer']),
			'det_missed':  signal_det_missed(rows),
			'det_lowconf': signal_det_lowconf(rows),
			'flicker':     signal_flicker(rows),
			'rare_class':  signal_rare_class(rows, rarity),
			'attr':        signal_attr(rows, proj['age_classes']),
			'random':      signal_random(rows, rng, max_per_video * 2),
		}
		# A CSV written before the per-stream fix can file a static verdict under
		# primary_motion_class, which is exactly what this stratum reads.
		if has_source:
			signals['pair_unseen'] = signal_pair_unseen(rows, pairs, static_totals)

		found = {}
		for stratum, raw in signals.items():
			n_here = 0
			for frame, score, detail, hits in collapse(raw):
				candidates[stratum].append((stem, frame, score, detail, hits))
				n_here += 1
			if n_here:
				found[stratum] = n_here
		# Per video, because that is the grain at which something looks wrong:
		# a clip contributing nothing but `random` is one the detector either
		# handles perfectly or does not see at all, and the two are worth
		# telling apart before spending a budget on it.
		summary = ', '.join(f"{k} {v}" for k, v in sorted(found.items(), key=lambda kv: -kv[1]))
		log(f"  [{n}/{len(csv_paths)}] {stem}: {len(rows)} rows -> "
			+ (summary if summary else 'no candidate'))

	total_candidates = sum(len(v) for v in candidates.values())
	log(f"\nSelecting {budget} frame(s) from {total_candidates} candidate(s)"
		f" (min spacing {min_spacing_s:g} s, max {max_per_video} per video)")
	selected = select(candidates, quotas, budget, min_spacing, max_per_video, excluded)
	for stratum in sorted(candidates, key=lambda s: -len(candidates[s])):
		got, avail = len(selected.get(stratum, [])), len(candidates[stratum])
		want = int(round(budget * quotas.get(stratum, 0)))
		flag = '' if got >= want else f"  (quota {want}, short by {want - got})"
		log(f"  {stratum:<12} {got:>4} kept of {avail:>6} available{flag}")
	order = interleave(selected)

	if out_path is None:
		# Beside the cached frames, not in output/: the list and the frames it
		# points at are one artefact, produced and consumed together. Rows are
		# deleted as they get annotated, so this file is also the resume point.
		out_path = frame_cache.targets_path(frame_cache.cache_dir(proj['project_dir']))
	os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
	with open(out_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.writer(f)
		# video_filename + frame are what the annotation tool's CSV parser needs;
		# it ignores the rest, which exists so the annotator can see why a frame
		# was served and so the mix can be audited afterwards.
		w.writerow(['video_filename', 'frame', 'timecode', 'reason', 'score', 'detail'])
		for stem, frame, stratum, score, detail in order:
			w.writerow([stem, frame, frame_to_timecode(frame, fps),
						stratum, f"{score:.4f}", detail])

	report = {
		'out_path': out_path,
		'order': order,
		'videos_used': used,
		'skipped_holdout': skipped_holdout,
		'stale_csvs': stale_csvs,
		'per_stratum': {s: len(v) for s, v in selected.items()},
		'total': len(order),
		'rare_classes': sorted(rarity),
		'zero_classes': sorted(c for c, n in rarity_counts.items() if n == 0),
	}
	return report


def precache(proj, order, workers=4, animation=True, log=print):
	"""Decode every mined frame ahead of time into <project>/mined_frames/.

	The seek-plus-read of one 4K frame on a network mount is seconds of latency
	and almost no bandwidth, so the loop is I/O-bound and threads help: several
	requests in flight amortise the round trips that dominate. Threads (not
	processes) because the wait happens inside cv2, which releases the GIL.

	Frames already cached from an earlier run are skipped, which makes a second
	invocation after an interrupted one cheap.

	Returns a summary dict; never raises for a single unreadable frame, since
	one bad clip should not cost the whole batch.
	"""
	import time
	import threading
	from concurrent.futures import ThreadPoolExecutor

	cdir = frame_cache.cache_dir(proj['project_dir'])
	params = frame_cache.motion_params_from_config(proj['config'])
	videos = index_videos(proj['clips_dir'])

	todo, cached, unresolved = [], 0, set()
	for stem, frame, _stratum, _score, _detail in order:
		base = frame_cache.base_name(stem, frame)
		if frame_cache.has_entry(cdir, base):
			cached += 1
			continue
		vpath = videos.get(stem.lower())
		if vpath is None:
			unresolved.add(stem)
			continue
		todo.append((base, vpath, frame))

	log(f"\nPre-caching frames into {cdir}")
	log(f"  {len(todo)} to decode, {cached} already cached, "
		f"{len(unresolved)} target(s) whose video was not found")
	# Said up front rather than discovered afterwards: at 4K this runs to
	# gigabytes, and the animation buffer is most of it.
	per_frame_mb = 7.5 + (params.buf_len * 1.2 if animation else 0.0)
	log(f"  animation buffer {'on' if animation else 'off'}"
		f" ({params.buf_len} frame(s) per target)"
		f" -> roughly {per_frame_mb:.0f} MB each, {per_frame_mb * len(todo) / 1000:.1f} GB total")
	if unresolved:
		log(f"  videos not found under {proj['clips_dir']}: "
			+ ', '.join(sorted(unresolved)[:5])
			+ (' …' if len(unresolved) > 5 else ''))
	if not todo:
		log("  nothing to do.")
		return {'written': 0, 'cached': cached, 'failed': 0,
				'unresolved': sorted(unresolved), 'bytes': 0}

	# Sorting by video keeps each worker reading one file at a time; a network
	# mount serves sequential-ish access far better than a scattered one.
	todo.sort(key=lambda t: (t[1], t[2]))

	state = {'done': 0, 'written': 0, 'failed': 0, 'bytes': 0}
	lock = threading.Lock()
	started = time.time()
	# Report on a cadence, not per frame: a 300-frame run over SSHFS should say
	# enough to show it is alive and to let the annotator judge when to start,
	# without turning the log into 300 lines.
	step = max(1, len(todo) // 20)

	def _one(item):
		base, vpath, frame = item
		try:
			data = frame_cache.compute_frame_data(vpath, frame, params)
			if data is None:
				raise ValueError(f"frame {frame} unreadable "
								 f"(needs {params.frame_window} preceding frames)")
			written = frame_cache.write_entry(cdir, base, data, animation=animation)
		except Exception as e:
			with lock:
				state['failed'] += 1
				state['done'] += 1
				log(f"  ! {base}: {e}")
			return
		with lock:
			state['written'] += 1
			state['bytes'] += written
			state['done'] += 1
			done = state['done']
			if done % step == 0 or done == len(todo):
				elapsed = time.time() - started
				rate = done / elapsed if elapsed > 0 else 0
				eta = (len(todo) - done) / rate if rate > 0 else 0
				log(f"  {done:>5}/{len(todo)}  ({100.0 * done / len(todo):4.0f} %)  "
					f"{rate:4.1f} frame/s  eta {int(eta // 60):d}m{int(eta % 60):02d}s  "
					f"{state['bytes'] / 1e6:.0f} MB")

	with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
		list(pool.map(_one, todo))

	elapsed = time.time() - started
	log(f"  cached {state['written']} frame(s) in {int(elapsed // 60)}m{int(elapsed % 60):02d}s"
		f", {state['bytes'] / 1e6:.0f} MB written"
		+ (f", {state['failed']} failed" if state['failed'] else ""))
	return {'written': state['written'], 'cached': cached, 'failed': state['failed'],
			'unresolved': sorted(unresolved), 'bytes': state['bytes']}


def parse_quota_overrides(raw, quotas):
	"""`--quota flicker=0.3,rare_class=0.2` on top of the defaults, renormalised."""
	if not raw:
		return quotas
	out = dict(quotas)
	for chunk in raw.split(','):
		if '=' not in chunk:
			raise SystemExit(f"Bad --quota term '{chunk}', expected name=fraction")
		name, value = chunk.split('=', 1)
		name = name.strip()
		if name not in out:
			raise SystemExit(f"Unknown stratum '{name}'. Known: {', '.join(out)}")
		out[name] = float(value)
	total = sum(out.values())
	if total <= 0:
		raise SystemExit("Quotas sum to zero.")
	return {k: v / total for k, v in out.items()}


def _main():
	ap = argparse.ArgumentParser(description=__doc__,
								 formatter_class=argparse.RawDescriptionHelpFormatter)
	# The launcher runs every tool as `python <script>.py <project_dir>`, so the
	# project is accepted positionally as well as by flag. --project wins when
	# both are given; without either, the env var the launcher also exports.
	ap.add_argument('project_pos', nargs='?', default=None,
					help=argparse.SUPPRESS)
	ap.add_argument('--project', default=None,
					help='Project directory or its BehaveAI_settings.ini')
	ap.add_argument('--out', default=None,
					help='Output CSV (default <output_dir>/mining_targets.csv)')
	ap.add_argument('--budget', type=int, default=None,
					help='How many frames to propose (default: the mining_budget setting)')
	ap.add_argument('--fps', type=float, default=30.0,
					help='Frame rate, used for spacing and time-codes (default 30)')
	ap.add_argument('--min-spacing-s', type=float, default=3.0,
					help='Minimum seconds between two proposals of one video')
	ap.add_argument('--max-per-video', type=int, default=None,
					help='Cap on proposals per video (default: the mining_max_per_video setting)')
	ap.add_argument('--quota', default='',
					help='Override stratum shares, e.g. flicker=0.3,rare_class=0.2')
	ap.add_argument('--include-holdout', action='store_true',
					help='Also mine validation videos (biases the test set — see module docstring)')
	ap.add_argument('--seed', type=int, default=0,
					help='Seed of the random stratum, for reproducibility')
	ap.add_argument('--no-precache', action='store_true',
					help='Only write the target list; do not decode the frames ahead of time')
	ap.add_argument('--precache-workers', type=int, default=None,
					help='Parallel decoders (default: the mining_precache_workers setting)')
	ap.add_argument('--no-animation-cache', action='store_true',
					help='Skip the animation buffer (most of the disk cost; the zoom pane '
						 'then shows a still)')
	args = ap.parse_args()

	project = args.project or args.project_pos or os.environ.get('BEHAVEAI_PROJECT')
	if not project:
		ap.error('no project given: pass --project, a positional path, '
				 'or set BEHAVEAI_PROJECT')

	# INI first, then the flag, so a one-off run can lean further without
	# editing the project's own preference.
	_proj = load_project(project)
	quotas = parse_quota_overrides(_proj['quota_overrides'], DEFAULT_QUOTAS)
	quotas = parse_quota_overrides(args.quota, quotas)
	max_per_video = args.max_per_video if args.max_per_video is not None else _proj['max_per_video']
	report = mine(project, args.budget, args.fps, args.min_spacing_s,
				  max_per_video, quotas, args.include_holdout, args.seed,
				  args.out)

	print(f"\nWrote {report['total']} target frame(s) to {report['out_path']}")
	print(f"  from {len(report['videos_used'])} video(s)"
		  f", {len(report['skipped_holdout'])} holdout video(s) skipped")
	for stratum, n in sorted(report['per_stratum'].items(), key=lambda kv: -kv[1]):
		print(f"    {stratum:<12} {n}")
	if report['stale_csvs']:
		print(f"\n  {len(report['stale_csvs'])} CSV(s) predate the per-stream column fix; "
			  "the pair_unseen stratum was skipped for them.\n"
			  "  Re-run BehaveAI_classify_track.py to get it.")
	if report['zero_classes']:
		print(f"\n  No detector can propose these (zero annotated examples): "
			  f"{', '.join(report['zero_classes'])}.\n"
			  "  They need a human ethogram sweep or BehaveAI_complex_candidates.py.")

	if not args.no_precache and report['order']:
		proj = _proj
		workers = args.precache_workers or proj['precache_workers']
		animation = proj['precache_animation'] and not args.no_animation_cache
		precache(proj, report['order'], workers, animation)
		print("\nThe annotation tool will serve these from disk: "
			  "Source > Mined frames (model is unsure).")
	return 0


if __name__ == '__main__':
	sys.exit(_main())
