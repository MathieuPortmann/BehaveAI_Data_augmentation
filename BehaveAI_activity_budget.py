#!/usr/bin/env python3
"""
BehaveAI Activity Budget Analysis

Reads tracking CSVs produced by BehaveAI_classify_track.py and computes
per-individual activity budgets. Strangers (individuals not belonging to
the focal group) are flagged automatically and kept in the output with
their individual_type set to 'stranger'.

Outputs:
  - activity_budget_individual.csv  : one row per individual per video
  - activity_budget_suspects.csv    : one row per flagged individual with timecodes

Usage:
  Called automatically at the end of BehaveAI_classify_track.py, or directly:
  python BehaveAI_activity_budget.py <project_dir | BehaveAI_settings.ini>
"""

import os
import sys
import csv
import configparser
import glob

from behaveai_config import resolve_project_dir
from pathlib import Path
from collections import defaultdict


# Explicit sentinels. These are REPORTED categories, not silent gaps: a frame the
# detector never labelled becomes 'not_classified' and gets its own duration
# column, so per-behaviour seconds always add up to the tracked presence and a
# reader can tell measured absence from missing data.
NOT_CLASSIFIED = 'not_classified'
# A detection carrying a primary class but no secondary one — either because that
# primary has no secondary step in secondary_map, or because none was predicted
# (the reserved '__none__' class won, or the score stayed below the threshold).
NO_SECONDARY = 'none'
UNKNOWN_AGE = 'unknown'


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_activity_budget_config(config_path):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)
    d = config['DEFAULT']

    return {
        # Group-membership thresholds, in physical units (not ratios):
        #   min_presence_seconds -- seconds a subject must be tracked to be a member;
        #   edge_margin_px       -- isotropic pixel band at the frame edge, used only
        #                           to REPORT the side a short-presence subject entered
        #                           from (explains a flag, never decides it).
        'min_presence_seconds':  float(d.get('ab_min_presence_seconds', '30')),
        'edge_margin_px':        float(d.get('ab_edge_margin_px',       '100')),
        'group_type_separator':  d.get('ab_group_type_separator',  '_'),
        'group_type_field_index': int(d.get('ab_group_type_field_index', '4')),
        'analysis_duration_s':   float(d.get('ab_analysis_duration_s',  '0')),
        # Minimum number of classified (behavior != 'unknown') frames a track
        # needs to count as a group_member (0 = skip this criterion). The fallback
        # is 0 so projects whose INI predates this key behave exactly as before;
        # the new-project template / settings GUI default it to 5.
        'min_classified_frames': int(float(d.get('ab_min_classified_frames', '0'))),
    }


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def extract_group_type(filename, separator, field_index):
    """
    Extract the group type label from the video filename.

    Example:
      DJI_20260401080704_0162_D_only-mare_morning1.MP4
      separator='_', field_index=4  ->  'only-mare'
    """
    stem = Path(filename).stem
    parts = stem.split(separator)
    if field_index < len(parts):
        return parts[field_index]
    return 'unknown'


# ---------------------------------------------------------------------------
# Timecode helper
# ---------------------------------------------------------------------------

def frame_to_timecode(frame_number, fps):
    """Convert a frame number to MM:SS string."""
    if fps <= 0:
        return '00:00'
    total_seconds = int(frame_number / fps)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


# ---------------------------------------------------------------------------
# Read metadata file (optional manual overrides)
# ---------------------------------------------------------------------------

def load_groups_metadata(project_dir):
    """
    Read groups_metadata.csv if present.
    Returns a dict: filename -> {'group_id': str, 'group_type': str,
                                  'exclude_ids': set of int}
    """
    meta_path = os.path.join(project_dir, 'groups_metadata.csv')
    metadata = {}
    if not os.path.exists(meta_path):
        return metadata

    with open(meta_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('video_filename', '').strip()
            if not fname:
                continue
            exclude_raw = row.get('exclude_track_ids', '').strip()
            exclude_ids = set()
            if exclude_raw:
                for x in exclude_raw.split(';'):
                    x = x.strip()
                    if x.isdigit():
                        exclude_ids.add(int(x))
            metadata[fname] = {
                'group_id':   row.get('group_id',   '').strip(),
                'group_type': row.get('group_type', '').strip(),
                'exclude_ids': exclude_ids,
            }
    return metadata


# ---------------------------------------------------------------------------
# Read complex-behaviour predictions (optional; from BehaveAI_complex_model)
# ---------------------------------------------------------------------------

def load_complex_predictions(output_dir, video_stem, fps):
    """Read <stem>_complex_predictions.csv and attribute each episode's duration to
    every involved individual.

    Columns: start_frame, end_frame, track_ids (';'-separated), behaviour, probability.
    A complex behaviour involves several individuals, so its duration is credited to
    each track_id listed.

    Returns (by_track, labels):
      by_track : dict track_id(str) -> {behaviour -> [total_seconds, n_episodes]}
      labels   : set of complex-behaviour labels seen in this file.
    Missing file -> ({}, set()); purely additive, so projects without a complex model
    behave exactly as before.
    """
    path = os.path.join(output_dir, video_stem + '_complex_predictions.csv')
    labels = set()
    if not os.path.exists(path):
        return {}, labels

    by_track = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            try:
                s = int(r['start_frame'])
                e = int(r['end_frame'])
                beh = (r.get('behaviour', '') or '').strip()
                ids = [t.strip() for t in str(r.get('track_ids', '')).split(';') if t.strip()]
            except (ValueError, KeyError, TypeError):
                continue
            if not beh or not ids:
                continue
            dur_s = (e - s + 1) / fps if fps and fps > 0 else 0.0
            if dur_s < 0:
                dur_s = 0.0
            labels.add(beh)
            for tid in ids:
                by_track[tid][beh][0] += dur_s
                by_track[tid][beh][1] += 1

    return {tid: dict(behs) for tid, behs in by_track.items()}, labels


# ---------------------------------------------------------------------------
# Read interaction graph edges (optional; from BehaveAI_complex_features)
# ---------------------------------------------------------------------------

def load_interaction_metrics(output_dir, video_stem, fps):
    """Read <stem>_interaction_edges.csv and derive per-individual social metrics.

    Adapts to edge granularity (the header differs, see _EDGE_COLS in
    BehaveAI_complex_features): episode files (per_dyad / per_segment) carry
    n_frames_observed / contact_fraction / mean_distance_m / interaction_type /
    weight; per_frame files carry one row per observed frame (in_contact,
    distance_m).

    The graph is UNDIRECTED — one row per unordered pair — so each row is credited
    to BOTH of its endpoints, each with the other as partner. (It used to emit both
    orderings and credit only source_id; crediting one endpoint of a single
    undirected row would leave every target_id with no partners at all.)

    Returns dict track_id(str) -> social-metric dict. Missing file -> {}.
    """
    path = os.path.join(output_dir, video_stem + '_interaction_edges.csv')
    if not os.path.exists(path):
        return {}

    def _f(row, key):
        try:
            return float(row.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0.0

    acc = defaultdict(lambda: {
        'partners': set(),
        'weighted_degree': 0.0,
        'frames': 0.0,           # total observed frames across this individual's edges
        'contact_frames': 0.0,   # observed frames weighted by contact fraction
        'dist_weight_sum': 0.0,  # sum of (mean_distance * frames) for a weighted mean
        'type_frames': defaultdict(float),  # interaction_type -> observed frames
    })

    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        episode_mode = 'n_frames_observed' in fields
        for r in reader:
            s = str(r.get('source_id', '')).strip()
            t = str(r.get('target_id', '')).strip()
            if not s or not t:
                continue
            weight = _f(r, 'weight')
            if episode_mode:
                n_obs = _f(r, 'n_frames_observed')
                cfrac = _f(r, 'contact_fraction')
                mdist = _f(r, 'mean_distance_m')
                itype = (r.get('interaction_type', '') or '').strip()
            else:  # per_frame: one observed frame per row
                n_obs = 1.0
                cfrac = 1.0 if _f(r, 'in_contact') else 0.0
                mdist = _f(r, 'distance_m')
                itype = ''  # per_frame edges carry no interaction_type

            for me, partner in ((s, t), (t, s)):
                a = acc[me]
                a['partners'].add(partner)
                a['weighted_degree'] += weight
                a['frames'] += n_obs
                a['contact_frames'] += cfrac * n_obs
                a['dist_weight_sum'] += mdist * n_obs
                if itype:
                    a['type_frames'][itype] += n_obs

    out = {}
    for tid, a in acc.items():
        frames = a['frames']
        dominant_type = (max(a['type_frames'].items(), key=lambda kv: kv[1])[0]
                         if a['type_frames'] else '')
        out[tid] = {
            'interaction_n_partners':       len(a['partners']),
            'interaction_weighted_degree':  round(a['weighted_degree'], 4),
            'interaction_time_s':           round(frames / fps, 2) if fps and fps > 0 else 0.0,
            'contact_time_s':               round(a['contact_frames'] / fps, 2) if fps and fps > 0 else 0.0,
            'contact_fraction':             round(a['contact_frames'] / frames, 4) if frames > 0 else 0.0,
            'mean_interaction_distance_m':  round(a['dist_weight_sum'] / frames, 4) if frames > 0 else 0.0,
            # MODEL-DERIVED: interaction_type is back-filled by the complex model.
            # Kept separate downstream so the budget CSV stays deterministic.
            'dominant_interaction_type':    dominant_type,
        }
    return out


# ---------------------------------------------------------------------------
# Parse one tracking CSV
# ---------------------------------------------------------------------------

def parse_tracking_csv(csv_path, max_frame_limit=None):
    """
    Read a tracking CSV and return:
      - fps estimate (frames per second, derived from frame column)
      - video_width, video_height (if available, else None)
      - tracks: dict of track_id -> list of row dicts

    Each row dict has keys: frame (int), x (float), y (float),
    primary_class (str), primary_conf (float).

    If max_frame_limit is set, only frames <= max_frame_limit are kept,
    which restricts the analysis to a fixed duration from the start of the video.
    """
    tracks = defaultdict(list)
    fps = 0.0
    frame_numbers = []
    seen = set()   # (frame, tid) already taken -- keep the first row only

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame  = int(row['frame'])

                # Restrict to the analysis window if a frame limit is set
                if max_frame_limit is not None and frame > max_frame_limit:
                    continue

                tid    = int(row['id'])

                # Deduplicate on (frame, id): keep the first row for each pair so a
                # subject is never double-counted in presence/behaviour.
                if (frame, tid) in seen:
                    continue
                seen.add((frame, tid))

                x      = float(row['x'])
                y      = float(row['y'])

                # Bounding box (columns x1..y2). Present in current CSVs; fall back
                # to a zero-size box at the centroid for older files.
                try:
                    bx1 = float(row['x1']); by1 = float(row['y1'])
                    bx2 = float(row['x2']); by2 = float(row['y2'])
                except (KeyError, ValueError, TypeError):
                    bx1 = bx2 = x
                    by1 = by2 = y

                # Choose the best available class label. The motion stream wins
                # ties, matching the arbitration used everywhere else.
                pm_class = row.get('primary_motion_class', '').strip()
                ps_class = row.get('primary_static_class', '').strip()
                pm_conf  = float(row.get('primary_motion_conf', 0) or 0)
                ps_conf  = float(row.get('primary_static_conf', 0) or 0)

                sm_class = row.get('secondary_motion_class', '').strip()
                ss_class = row.get('secondary_static_class', '').strip()

                if pm_class and pm_conf >= ps_conf:
                    behavior  = pm_class
                    conf      = pm_conf
                    # The secondary was classified on the crop of the winning
                    # stream, so take that one first; the other stream is a
                    # fallback for a box only one detector saw.
                    secondary = sm_class or ss_class
                elif ps_class:
                    behavior  = ps_class
                    conf      = ps_conf
                    secondary = ss_class or sm_class
                else:
                    behavior  = NOT_CLASSIFIED
                    conf      = 0.0
                    secondary = ''

                # '__none__' is the reserved 'no secondary' answer of the
                # classifier; it is not a behaviour.
                if secondary in ('', '__none__'):
                    secondary = NO_SECONDARY

                age = (row.get('age_class', '') or '').strip() or UNKNOWN_AGE
                try:
                    age_conf = float(row.get('age_conf', 0) or 0)
                except (TypeError, ValueError):
                    age_conf = 0.0

                tracks[tid].append({
                    'frame':     frame,
                    'x':         x,
                    'y':         y,
                    'box':       (bx1, by1, bx2, by2),
                    'behavior':  behavior,
                    'conf':      conf,
                    'secondary': secondary,
                    'age':       age,
                    'age_conf':  age_conf,
                })
                frame_numbers.append(frame)
            except (ValueError, KeyError):
                continue

    # Estimate FPS: if total frames span is known we cannot derive real FPS
    # from the CSV alone. We store the max frame number so the caller can
    # combine it with real video metadata if available. Default assumption: 30 fps.
    # The caller can override by reading video metadata separately.
    max_frame = max(frame_numbers) if frame_numbers else 0

    return tracks, max_frame


# ---------------------------------------------------------------------------
# Flag strangers
# ---------------------------------------------------------------------------

def flag_strangers(tracks, max_frame, video_width, video_height, fps,
                   min_presence_seconds, edge_margin_px,
                   manual_exclude_ids, min_classified_frames=0):
    """
    Decide for each track_id whether it is a group_member or a stranger.

    A group_member must satisfy all of:
      1. tracked presence >= min_presence_seconds (a stranger enters AND leaves;
         a member that joins mid-clip still accrues plenty of seconds),
      2. at least min_classified_frames frames with behavior != 'unknown'
         (0 = skip this criterion).

    The side a subject FIRST appeared from (any frame edge within edge_margin_px,
    isotropic, using its bounding box) is REPORTED to explain a short-presence
    flag, but never excludes on its own -- a member joining from the top edge and
    staying is a group_member.

    presence_seconds = n_frames / fps: exact when the pipeline processes every
    frame (frame_skip = 0); with frame_skip > 0 it under-reports proportionally.

    This is intra-video only -- there is no inter-video / cross-session logic.

    Returns a dict: track_id -> {'individual_type', 'auto_flagged', 'flag_reason',
                                 'presence_seconds', 'first_frame', 'last_frame',
                                 'entry_side'}.
    """
    results = {}

    for tid, rows in tracks.items():
        n_frames        = len(rows)
        presence_seconds = n_frames / fps if fps and fps > 0 else 0.0

        first_frame  = min(r['frame'] for r in rows)
        last_frame   = max(r['frame'] for r in rows)

        # Bounding box at first appearance -> which frame edge(s) it entered from.
        first_row = next(r for r in rows if r['frame'] == first_frame)
        bx1, by1, bx2, by2 = first_row.get('box', (first_row['x'], first_row['y'],
                                                    first_row['x'], first_row['y']))
        sides = []
        if video_width and video_height and video_width > 0 and video_height > 0:
            if bx1 <= edge_margin_px:                sides.append('left')
            if bx2 >= video_width - edge_margin_px:  sides.append('right')
            if by1 <= edge_margin_px:                sides.append('top')
            if by2 >= video_height - edge_margin_px: sides.append('bottom')
        entry_side = ','.join(sides)
        entered_from_side = bool(sides)

        # Count classified frames (a real behaviour label) for criterion 2.
        n_classified = sum(1 for r in rows
                           if r.get('behavior') and r['behavior'] != NOT_CLASSIFIED)
        is_unclassified = (min_classified_frames > 0
                           and n_classified < min_classified_frames)

        # Determine flag reason
        is_manual     = tid in manual_exclude_ids
        is_short      = presence_seconds < min_presence_seconds
        is_side_entry = entered_from_side and is_short  # side alone is not enough

        if is_manual:
            reason = 'manual_exclude'
        elif is_side_entry:
            reason = f'short_presence+side_entry({entry_side})'
        elif is_short:
            reason = 'short_presence'
        elif is_unclassified:
            reason = 'insufficient_classified_frames'
        else:
            reason = ''

        is_stranger  = is_manual or is_short or is_unclassified
        auto_flagged = (is_short or is_unclassified) and not is_manual

        results[tid] = {
            'individual_type': 'stranger' if is_stranger else 'group_member',
            'auto_flagged':    auto_flagged,
            'flag_reason':     reason,
            'presence_seconds': round(presence_seconds, 2),
            'first_frame':     first_frame,
            'last_frame':      last_frame,
            'entry_side':      entry_side,
        }

    return results


# ---------------------------------------------------------------------------
# Compute per-individual activity budget
# ---------------------------------------------------------------------------

def compute_individual_budget(tracks, flag_info, fps, all_behaviors,
                              all_secondaries=()):
    """
    For each track, count frames per behaviour and derive durations.

    TWO PARALLEL decompositions of the same tracked time are written, never a
    single merged list: a secondary behaviour qualifies a primary one
    (Stand + alert is one frame, not two), so summing the two families together
    would double-count. Each family therefore sums to duration_s on its own:
        sum(behavior_*_s)  == duration_s   (includes not_classified)
        sum(secondary_*_s) == duration_s   (includes 'none')

    Returns list of dicts, one per individual.
    """
    records = []

    for tid, rows in tracks.items():
        info = flag_info[tid]
        n_frames_present = len(rows)
        duration_s = n_frames_present / fps if fps > 0 else 0.0

        # Count frames per behaviour, per secondary behaviour
        behavior_frames = defaultdict(int)
        secondary_frames = defaultdict(int)
        for r in rows:
            behavior_frames[r['behavior']] += 1
            secondary_frames[r.get('secondary', NO_SECONDARY)] += 1

        # Count transitions (label changes) for both families
        sorted_rows   = sorted(rows, key=lambda r: r['frame'])
        transitions   = defaultdict(int)
        sec_transitions = defaultdict(int)
        prev_behavior = None
        prev_secondary = None
        for r in sorted_rows:
            b = r['behavior']
            if prev_behavior is not None and b != prev_behavior:
                transitions[b] += 1
            prev_behavior = b
            s = r.get('secondary', NO_SECONDARY)
            if prev_secondary is not None and s != prev_secondary:
                sec_transitions[s] += 1
            prev_secondary = s

        # Age: confidence-weighted majority vote over the track. An individual the
        # age model never labelled stays 'unknown' rather than being guessed.
        age_votes = defaultdict(float)
        age_confs = []
        for r in rows:
            lab = r.get('age', UNKNOWN_AGE)
            if lab and lab != UNKNOWN_AGE:
                age_votes[lab] += max(float(r.get('age_conf', 0.0)), 1e-6)
                age_confs.append(float(r.get('age_conf', 0.0)))
        age_class = max(age_votes, key=age_votes.get) if age_votes else UNKNOWN_AGE
        age_conf_mean = (sum(age_confs) / len(age_confs)) if age_confs else 0.0

        # Build flat record
        rec = {
            'track_id':         tid,
            'individual_type':  info['individual_type'],
            'auto_flagged':     info['auto_flagged'],
            'n_frames_present': n_frames_present,
            'duration_s':       round(duration_s, 2),   # = tracked presence in seconds
            'age_class':        age_class,
            'age_conf_mean':    round(age_conf_mean, 4),
        }

        dominant_time = None
        dominant_count = None
        max_time = -1
        max_count = -1

        for b in all_behaviors:
            n  = behavior_frames.get(b, 0)
            s  = n / fps if fps > 0 else 0.0
            tr = transitions.get(b, 0)

            # Durations only: no percentage is derived here. A percentage hides
            # which denominator it used (tracked presence vs clip duration), and
            # both are written out so any ratio can be computed downstream.
            rec[f'behavior_{b}_s']   = round(s, 2)
            rec[f'behavior_{b}_n']   = tr

            # 'not_classified' gets its own duration column but is never reported
            # as the dominant behaviour: it is an absence of measurement, not one.
            if b == NOT_CLASSIFIED:
                continue
            if s > max_time:
                max_time     = s
                dominant_time = b
            if tr > max_count:
                max_count     = tr
                dominant_count = b

        rec['dominant_behavior_time']  = dominant_time  or ''
        rec['dominant_behavior_count'] = dominant_count or ''

        # Secondary behaviours: the parallel decomposition (see the docstring).
        for sb in all_secondaries:
            n  = secondary_frames.get(sb, 0)
            rec[f'secondary_{sb}_s'] = round(n / fps if fps > 0 else 0.0, 2)
            rec[f'secondary_{sb}_n'] = sec_transitions.get(sb, 0)

        records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_activity_budget(config_path, fps_default=30.0):
    """
    Process all tracking CSVs in the output folder and write the two result files.

    Args:
        config_path (str): Path to BehaveAI_settings.ini.
        fps_default (float): FPS to use when the real video FPS cannot be read.
    """
    config_path  = os.path.abspath(config_path)
    project_dir  = os.path.dirname(config_path)
    cfg          = load_activity_budget_config(config_path)

    # Resolve output folder
    ini          = configparser.ConfigParser()
    ini.optionxform = str
    ini.read(config_path)
    d            = ini['DEFAULT']
    output_dir   = resolve_project_dir(d, project_dir, 'output')

    metadata     = load_groups_metadata(project_dir)

    # Resolve input_dir: protocol videos live here (e.g. Activity_budget folder).
    # Used to locate video files for FPS / dimension reading.
    input_dir = resolve_project_dir(d, project_dir, 'input')

    # Helper: build a name -> full_path map by walking input_dir recursively.
    # This lets us find any video file regardless of how deep in the tree it lives.
    def _build_video_index(root):
        """Return dict: filename (with extension) -> absolute path."""
        index = {}
        if not os.path.isdir(root):
            return index
        video_exts = ('.mp4', '.avi', '.mov', '.mkv')
        for dirpath, _, files in os.walk(root):
            for fname in files:
                if fname.lower().endswith(video_exts):
                    index[fname] = os.path.join(dirpath, fname)
        return index

    video_file_index = _build_video_index(input_dir)

    # Pick, per video, the MOST-PROCESSED tracking CSV available. Each pipeline
    # stage appends to the previous file, so the most-advanced one is a superset
    # (final stitched identities, drone-corrected positions, metric columns). This
    # runs last in the pipeline, so the budget is computed on the final ids.
    _SUFFIXES = ('_tracking_metric.csv', '_tracking_stitched.csv',
                 '_tracking_corrected.csv', '_tracking.csv')

    def _video_stem(basename):
        for suf in _SUFFIXES:
            if basename.endswith(suf):
                return basename[:-len(suf)]
        return os.path.splitext(basename)[0]

    best = {}   # video stem -> path of its most-processed CSV (first suffix wins)
    for suf in _SUFFIXES:
        for p in sorted(glob.glob(os.path.join(output_dir, '*' + suf))):
            best.setdefault(_video_stem(os.path.basename(p)), p)

    # Exclude training-only footage. NOTE: test the stem AFTER stripping the full
    # suffix -- a naive '_tracking' strip left 'foo_Training_metric', which does
    # not end with '_Training', so Training clips silently leaked into the budget.
    csv_files = [p for stem, p in sorted(best.items()) if not stem.endswith('_Training')]
    skipped = len(best) - len(csv_files)
    if skipped:
        print(f"Activity budget: skipped {skipped} Training CSV(s).")
    if not csv_files:
        print(f"Activity budget: no tracking CSVs found in {output_dir}")
        return

    print(f"Activity budget: processing {len(csv_files)} tracking CSV(s)...")

    all_individual_rows = []
    all_suspect_rows    = []

    # Collect all behavior names across all files first
    all_behaviors_global = set()
    all_secondaries_global = set()
    all_complex_global   = set()   # complex-behaviour labels across all videos
    tracks_cache      = {}
    complex_cache     = {}   # csv_path -> {track_id(str) -> {beh -> [s, n]}}
    interaction_cache = {}   # csv_path -> {track_id(str) -> social-metric dict}
    for csv_path in csv_files:
        # Derive video filename from CSV name:  foo_tracking.csv -> foo.MP4 (best guess)
        csv_basename   = os.path.basename(csv_path)
        video_stem     = _video_stem(csv_basename)
        # Try to find matching video using the recursive index first,
        # then fall back to a direct path in input_dir or clips_dir.
        video_filename = video_stem  # fallback (no extension)
        video_full_path = None
        for ext in ('.MP4', '.mp4', '.avi', '.mov', '.mkv'):
            candidate = video_stem + ext
            if candidate in video_file_index:
                video_filename  = candidate
                video_full_path = video_file_index[candidate]
                break
        # Further fallback: check clips_dir (flat) for backward compatibility
        if video_full_path is None:
            clips_dir_ab = resolve_project_dir(d, project_dir, 'clips')
            for ext in ('.MP4', '.mp4', '.avi', '.mov', '.mkv'):
                candidate_path = os.path.join(clips_dir_ab, video_stem + ext)
                if os.path.exists(candidate_path):
                    video_filename  = video_stem + ext
                    video_full_path = candidate_path
                    break

        # FPS and clip length: read from the video file, else fall back.
        # The clip length is needed because a budget in seconds is only
        # interpretable against the duration actually analysed.
        fps = fps_default
        video_frames_total = 0
        try:
            import cv2
            if video_full_path and os.path.exists(video_full_path):
                cap = cv2.VideoCapture(video_full_path)
                fps_read = cap.get(cv2.CAP_PROP_FPS)
                if fps_read > 0:
                    fps = fps_read
                n_read = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if n_read and n_read > 0:
                    video_frames_total = int(n_read)
                cap.release()
        except Exception:
            pass

        # Compute frame limit from analysis duration setting (0 = no limit)
        analysis_duration_s = cfg.get('analysis_duration_s', 0)
        if analysis_duration_s and analysis_duration_s > 0:
            max_frame_limit = int(analysis_duration_s * fps)
            print(f"  Analysis window: first {analysis_duration_s:.0f}s "
                  f"({max_frame_limit} frames at {fps:.1f} fps)")
        else:
            max_frame_limit = None

        tracks, max_frame = parse_tracking_csv(csv_path, max_frame_limit=max_frame_limit)
        if max_frame_limit is not None:
            max_frame = min(max_frame, max_frame_limit)

        # Duration actually analysed, in seconds — the reference every per-track
        # duration is read against. Priority: the analysis window if one was set,
        # else the real clip length, else the last frame carrying a detection
        # (which under-estimates the clip when it ends with no animal in view).
        if max_frame_limit is not None:
            analysed_frames = max_frame_limit
        elif video_frames_total > 0:
            analysed_frames = video_frames_total
        else:
            analysed_frames = max_frame
        video_duration_s = round(analysed_frames / fps, 2) if fps > 0 else 0.0

        # Cache tracks together with pre-resolved video info
        tracks_cache[csv_path] = (tracks, max_frame, fps, video_filename,
                                  video_full_path, video_duration_s)

        # Collect every label that actually occurs, INCLUDING the explicit
        # 'not_classified' sentinel, so unlabelled time gets its own column
        # instead of vanishing from the report.
        for rows in tracks.values():
            for r in rows:
                if r['behavior']:
                    all_behaviors_global.add(r['behavior'])
                all_secondaries_global.add(r.get('secondary', NO_SECONDARY))

        # Optional: fold in the complex-behaviour predictions and interaction graph
        # for this video (read-if-present; absent files leave the extra columns empty).
        complex_by_track, complex_labels = load_complex_predictions(output_dir, video_stem, fps)
        complex_cache[csv_path] = complex_by_track
        all_complex_global |= complex_labels
        interaction_cache[csv_path] = load_interaction_metrics(output_dir, video_stem, fps)

    # Sort the real labels alphabetically, then pin the two sentinels last so the
    # CSV always ends with the "no measurement" columns rather than hiding them
    # in the middle of the behaviour block. Both are emitted UNCONDITIONALLY,
    # even when they are zero everywhere: a column that appears only when the
    # problem occurs is exactly the silent failure this is meant to remove, and
    # a fixed schema lets budgets from different videos be concatenated.
    all_behaviors = sorted(all_behaviors_global - {NOT_CLASSIFIED}) + [NOT_CLASSIFIED]
    all_secondaries = sorted(all_secondaries_global - {NO_SECONDARY}) + [NO_SECONDARY]
    all_complex_behaviours = sorted(all_complex_global)

    # Fixed schema for the optional social-metric columns. Added only when at least
    # one video actually has an interaction graph, so a project that never built one
    # produces a byte-for-byte identical file to before this feature.
    # DETERMINISTIC social metrics: geometry + tracking only, no model in the loop.
    _INTERACTION_FIELDS = ['interaction_n_partners', 'interaction_weighted_degree',
                           'interaction_time_s',
                           'contact_time_s', 'contact_fraction',
                           'mean_interaction_distance_m']
    # MODEL-DERIVED: everything below inherits the complex model's error rate and
    # therefore lives in its own file (see the second CSV written at the end).
    _PREDICTED_FIELDS = ['dominant_interaction_type']
    has_interactions = any(bool(v) for v in interaction_cache.values())

    for csv_path in csv_files:
        (tracks, max_frame, fps, video_filename, video_full_path,
         video_duration_s) = tracks_cache[csv_path]

        # Video dimensions for border detection
        video_width = video_height = None
        try:
            import cv2
            if video_full_path and os.path.exists(video_full_path):
                cap = cv2.VideoCapture(video_full_path)
                video_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        except Exception:
            pass

        # Metadata overrides
        file_meta      = metadata.get(video_filename, {})
        group_id       = file_meta.get('group_id', '')
        group_type_meta = file_meta.get('group_type', '')
        manual_exclude = file_meta.get('exclude_ids', set())

        # Extract group type from filename if not in metadata
        group_type = group_type_meta or extract_group_type(
            video_filename,
            cfg['group_type_separator'],
            cfg['group_type_field_index']
        )

        # Flag strangers
        flag_info = flag_strangers(
            tracks, max_frame, video_width, video_height, fps,
            cfg['min_presence_seconds'], cfg['edge_margin_px'],
            manual_exclude, min_classified_frames=cfg['min_classified_frames']
        )

        # Compute individual budgets
        individual_rows = compute_individual_budget(
            tracks, flag_info, fps, all_behaviors, all_secondaries)

        # Per-video complex-behaviour + interaction lookups (empty if no such files).
        complex_by_track     = complex_cache.get(csv_path, {})
        interaction_by_track = interaction_cache.get(csv_path, {})

        for rec in individual_rows:
            tid_str    = str(rec['track_id'])
            duration_s = rec.get('duration_s', 0.0)

            # --- Complex behaviours: seconds and number of episodes ---
            beh_map = complex_by_track.get(tid_str, {})
            for b in all_complex_behaviours:
                sec, n = beh_map.get(b, (0.0, 0))
                rec[f'complex_{b}_s']   = round(sec, 2)
                rec[f'complex_{b}_n']   = n

            # --- Interaction graph: social metrics (only when a graph exists) ---
            if has_interactions:
                soc = interaction_by_track.get(tid_str)
                if soc:
                    rec.update(soc)
                else:
                    for k in _INTERACTION_FIELDS:
                        rec[k] = 0.0
                    for k in _PREDICTED_FIELDS:
                        rec[k] = ''

            rec['video_filename']   = video_filename
            rec['group_id']         = group_id
            rec['group_type']       = group_type
            rec['video_duration_s'] = video_duration_s
            all_individual_rows.append(rec)

        # Build suspect rows (strangers only)
        for tid, info in flag_info.items():
            if info['individual_type'] != 'stranger':
                continue
            first_tc = frame_to_timecode(info['first_frame'], fps)
            last_tc  = frame_to_timecode(info['last_frame'],  fps)
            is_manual = tid in manual_exclude
            all_suspect_rows.append({
                'video_filename':        video_filename,
                'group_id':              group_id,
                'group_type':            group_type,
                'track_id':              tid,
                'flag_reason':           info['flag_reason'],
                'presence_seconds':      info['presence_seconds'],
                'first_seen_frame':      info['first_frame'],
                'last_seen_frame':       info['last_frame'],
                'first_seen_timecode':   first_tc,
                'last_seen_timecode':    last_tc,
                'entry_side':            info['entry_side'],
                'auto_flagged':          info['auto_flagged'],
                'manual_exclude':        is_manual,
            })

        print(f"  {video_filename}: {len(tracks)} tracks, "
              f"{sum(1 for i in flag_info.values() if i['individual_type']=='group_member')} members, "
              f"{sum(1 for i in flag_info.values() if i['individual_type']=='stranger')} strangers")

    # ── Write CSV 1: individual activity budgets ───────────────────────────
    if all_individual_rows:
        out1 = os.path.join(output_dir, 'activity_budget_individual.csv')

        # Build ordered fieldnames
        # Two reference durations are written side by side, so every quantity in
        # this file is a duration in seconds and any ratio can be formed
        # downstream against an explicit denominator:
        #   video_duration_s -- the clip duration actually analysed
        #   duration_s       -- how long this individual was tracked within it
        base_fields = ['video_filename', 'group_id', 'group_type',
                       'track_id', 'individual_type', 'auto_flagged',
                       'age_class', 'age_conf_mean',
                       'video_duration_s', 'n_frames_present', 'duration_s']
        # Two parallel families, each summing to duration_s. behavior_* ends with
        # not_classified; secondary_* ends with none.
        behavior_fields = []
        for b in all_behaviors:
            behavior_fields += [f'behavior_{b}_s',
                                f'behavior_{b}_n']
        for sb in all_secondaries:
            behavior_fields += [f'secondary_{sb}_s',
                                f'secondary_{sb}_n']
        extra_fields = ['dominant_behavior_time', 'dominant_behavior_count']
        interaction_fields = _INTERACTION_FIELDS if has_interactions else []
        # NOTE: complex_* and dominant_interaction_type are NOT here. They are
        # predictions of the complex-behaviour model, not measurements, and mixing
        # them into this file made the whole budget look like one deterministic
        # aggregation when part of it carried a classifier's error rate.
        fieldnames = (base_fields + behavior_fields
                      + interaction_fields + extra_fields)

        # Sort by video then individual_type (group_member first)
        all_individual_rows.sort(
            key=lambda r: (r['video_filename'],
                           0 if r['individual_type'] == 'group_member' else 1,
                           r['track_id']))

        with open(out1, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_individual_rows)

        print(f"Written: {out1}")

    # ── Write CSV 2: model-derived per-individual columns ─────────────────
    # Same keys (video_filename + track_id) so it joins back onto CSV 1 in one
    # line of R, but separate so a reader can never mistake a prediction for a
    # measurement.
    if all_individual_rows and (all_complex_behaviours or has_interactions):
        out2 = os.path.join(output_dir, 'activity_budget_predicted.csv')
        pred_key_fields = ['video_filename', 'group_id', 'group_type',
                           'track_id', 'individual_type',
                           'video_duration_s', 'duration_s']
        complex_fields = []
        for b in all_complex_behaviours:
            complex_fields += [f'complex_{b}_s',
                               f'complex_{b}_n']
        pred_fields = pred_key_fields + complex_fields + \
            (_PREDICTED_FIELDS if has_interactions else [])
        with open(out2, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=pred_fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_individual_rows)
        print(f"Written: {out2}  (model predictions — join on video_filename+track_id)")

    # ── Write CSV 4: suspects / strangers with timecodes ──────────────────
    if all_suspect_rows:
        out4 = os.path.join(output_dir, 'activity_budget_suspects.csv')
        fieldnames4 = [
            'video_filename', 'group_id', 'group_type', 'track_id',
            'flag_reason', 'presence_seconds',
            'first_seen_frame', 'last_seen_frame',
            'first_seen_timecode', 'last_seen_timecode',
            'entry_side', 'auto_flagged', 'manual_exclude',
        ]
        with open(out4, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames4, extrasaction='ignore')
            writer.writeheader()
            all_suspect_rows.sort(
                key=lambda r: (r['video_filename'], r['track_id']))
            writer.writerows(all_suspect_rows)

        print(f"Written: {out4}")
    else:
        print("No strangers detected — suspects file not written.")

    print("Activity budget analysis complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python BehaveAI_activity_budget.py "
              "<project_dir | BehaveAI_settings.ini>")
        sys.exit(1)

    arg = os.path.abspath(sys.argv[1])
    ini = os.path.join(arg, 'BehaveAI_settings.ini') if os.path.isdir(arg) else arg

    if not os.path.exists(ini):
        print(f"Settings file not found: {ini}")
        sys.exit(1)

    run_activity_budget(ini)
