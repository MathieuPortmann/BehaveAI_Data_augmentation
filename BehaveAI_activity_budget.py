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
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_activity_budget_config(config_path):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_path)
    d = config['DEFAULT']

    return {
        'min_presence_ratio':    float(d.get('ab_min_presence_ratio',    '0.10')),
        'border_zone_ratio':     float(d.get('ab_border_zone_ratio',     '0.15')),
        'group_type_separator':  d.get('ab_group_type_separator',  '_'),
        'group_type_field_index': int(d.get('ab_group_type_field_index', '4')),
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
# Parse one tracking CSV
# ---------------------------------------------------------------------------

def parse_tracking_csv(csv_path):
    """
    Read a tracking CSV and return:
      - fps estimate (frames per second, derived from frame column)
      - video_width, video_height (if available, else None)
      - tracks: dict of track_id -> list of row dicts

    Each row dict has keys: frame (int), x (float), y (float),
    primary_class (str), primary_conf (float).
    """
    tracks = defaultdict(list)
    fps = 0.0
    frame_numbers = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                frame  = int(row['frame'])
                tid    = int(row['id'])
                x      = float(row['x'])
                y      = float(row['y'])

                # Choose the best available class label
                # primary_motion_class takes precedence over primary_static_class
                pm_class = row.get('primary_motion_class', '').strip()
                ps_class = row.get('primary_static_class', '').strip()
                pm_conf  = float(row.get('primary_motion_conf', 0) or 0)
                ps_conf  = float(row.get('primary_static_conf', 0) or 0)

                if pm_class and pm_conf >= ps_conf:
                    behavior = pm_class
                    conf     = pm_conf
                elif ps_class:
                    behavior = ps_class
                    conf     = ps_conf
                else:
                    behavior = 'unknown'
                    conf     = 0.0

                tracks[tid].append({
                    'frame':    frame,
                    'x':        x,
                    'y':        y,
                    'behavior': behavior,
                    'conf':     conf,
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

def flag_strangers(tracks, max_frame, video_width, video_height,
                   min_presence_ratio, border_zone_ratio,
                   manual_exclude_ids):
    """
    Decide for each track_id whether it is a group_member or a stranger.

    Returns a dict: track_id -> {'individual_type': str,
                                  'auto_flagged': bool,
                                  'flag_reason': str}
    """
    total_frames = max(max_frame, 1)
    results = {}

    for tid, rows in tracks.items():
        n_frames     = len(rows)
        presence_ratio = n_frames / total_frames

        first_frame  = min(r['frame'] for r in rows)
        last_frame   = max(r['frame'] for r in rows)

        # Position at first appearance
        first_row    = next(r for r in rows if r['frame'] == first_frame)
        fx, fy       = first_row['x'], first_row['y']

        # Border zone check (requires video dimensions)
        in_border = False
        if video_width and video_height and video_width > 0 and video_height > 0:
            bx = border_zone_ratio * video_width
            by = border_zone_ratio * video_height
            in_border = (fx < bx or fx > video_width - bx or
                         fy < by or fy > video_height - by)

        # Determine flag reason
        is_manual   = tid in manual_exclude_ids
        is_short    = presence_ratio < min_presence_ratio
        is_border   = in_border and is_short  # border alone is not enough

        if is_manual:
            reason = 'manual_exclude'
        elif is_short and is_border:
            reason = 'short_presence+border_entry'
        elif is_short:
            reason = 'short_presence'
        else:
            reason = ''

        is_stranger  = is_manual or is_short
        auto_flagged = (is_short or is_border) and not is_manual

        results[tid] = {
            'individual_type': 'stranger' if is_stranger else 'group_member',
            'auto_flagged':    auto_flagged,
            'flag_reason':     reason,
            'presence_ratio':  round(presence_ratio, 4),
            'first_frame':     first_frame,
            'last_frame':      last_frame,
            'border_entry':    in_border,
        }

    return results


# ---------------------------------------------------------------------------
# Compute per-individual activity budget
# ---------------------------------------------------------------------------

def compute_individual_budget(tracks, flag_info, fps, all_behaviors):
    """
    For each track, count frames per behavior and derive durations.

    Returns list of dicts, one per individual.
    """
    records = []

    for tid, rows in tracks.items():
        info = flag_info[tid]
        n_frames_present = len(rows)
        duration_s = n_frames_present / fps if fps > 0 else 0.0

        # Count frames per behavior
        behavior_frames = defaultdict(int)
        for r in rows:
            behavior_frames[r['behavior']] += 1

        # Count transitions (behavior changes)
        sorted_rows   = sorted(rows, key=lambda r: r['frame'])
        transitions   = defaultdict(int)
        prev_behavior = None
        for r in sorted_rows:
            b = r['behavior']
            if prev_behavior is not None and b != prev_behavior:
                transitions[b] += 1
            prev_behavior = b

        # Build flat record
        rec = {
            'track_id':         tid,
            'individual_type':  info['individual_type'],
            'auto_flagged':     info['auto_flagged'],
            'n_frames_present': n_frames_present,
            'duration_s':       round(duration_s, 2),
            'presence_ratio':   info['presence_ratio'],
        }

        total_classified = sum(behavior_frames.values())
        dominant_time = None
        dominant_count = None
        max_time = -1
        max_count = -1

        for b in all_behaviors:
            n  = behavior_frames.get(b, 0)
            s  = n / fps if fps > 0 else 0.0
            pct = 100.0 * n / total_classified if total_classified > 0 else 0.0
            tr = transitions.get(b, 0)

            rec[f'behavior_{b}_s']   = round(s, 2)
            rec[f'behavior_{b}_n']   = tr
            rec[f'behavior_{b}_pct'] = round(pct, 2)

            if s > max_time:
                max_time     = s
                dominant_time = b
            if tr > max_count:
                max_count     = tr
                dominant_count = b

        rec['dominant_behavior_time']  = dominant_time  or ''
        rec['dominant_behavior_count'] = dominant_count or ''

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
    output_dir_raw = d.get('output_dir', 'output')
    if os.path.isabs(output_dir_raw):
        output_dir = output_dir_raw
    else:
        output_dir = os.path.join(project_dir, output_dir_raw)

    metadata     = load_groups_metadata(project_dir)

    # Collect all tracking CSVs
    csv_files = sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv')))
    if not csv_files:
        print(f"Activity budget: no tracking CSVs found in {output_dir}")
        return

    print(f"Activity budget: processing {len(csv_files)} tracking CSV(s)...")

    all_individual_rows = []
    all_suspect_rows    = []

    # Collect all behavior names across all files first
    all_behaviors_global = set()
    tracks_cache = {}
    for csv_path in csv_files:
        tracks, max_frame = parse_tracking_csv(csv_path)
        tracks_cache[csv_path] = (tracks, max_frame)
        for rows in tracks.values():
            for r in rows:
                if r['behavior'] and r['behavior'] != 'unknown':
                    all_behaviors_global.add(r['behavior'])

    all_behaviors = sorted(all_behaviors_global)

    for csv_path in csv_files:
        tracks, max_frame = tracks_cache[csv_path]

        # Derive video filename from CSV name:  foo_tracking.csv -> foo.MP4 (best guess)
        csv_basename   = os.path.basename(csv_path)
        video_stem     = csv_basename.replace('_tracking.csv', '')
        # Try to find matching video extension
        video_filename = video_stem  # fallback
        for ext in ('.MP4', '.mp4', '.avi', '.mov', '.mkv'):
            candidate = video_stem + ext
            candidate_path = os.path.join(
                d.get('clips_dir', os.path.join(project_dir, 'clips')),
                candidate)
            if os.path.exists(candidate_path):
                video_filename = candidate
                break

        # FPS: try to read from video file, else use default
        fps = fps_default
        try:
            import cv2
            clips_dir_raw = d.get('clips_dir', 'clips')
            clips_dir = clips_dir_raw if os.path.isabs(clips_dir_raw) \
                        else os.path.join(project_dir, clips_dir_raw)
            vpath = os.path.join(clips_dir, video_filename)
            if os.path.exists(vpath):
                cap = cv2.VideoCapture(vpath)
                fps_read = cap.get(cv2.CAP_PROP_FPS)
                if fps_read > 0:
                    fps = fps_read
                cap.release()
        except Exception:
            pass

        # Video dimensions for border detection
        video_width = video_height = None
        try:
            import cv2
            clips_dir_raw = d.get('clips_dir', 'clips')
            clips_dir = clips_dir_raw if os.path.isabs(clips_dir_raw) \
                        else os.path.join(project_dir, clips_dir_raw)
            vpath = os.path.join(clips_dir, video_filename)
            if os.path.exists(vpath):
                cap = cv2.VideoCapture(vpath)
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
            tracks, max_frame, video_width, video_height,
            cfg['min_presence_ratio'], cfg['border_zone_ratio'],
            manual_exclude
        )

        # Compute individual budgets
        individual_rows = compute_individual_budget(
            tracks, flag_info, fps, all_behaviors)

        for rec in individual_rows:
            rec['video_filename'] = video_filename
            rec['group_id']       = group_id
            rec['group_type']     = group_type
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
                'presence_ratio':        info['presence_ratio'],
                'first_seen_frame':      info['first_frame'],
                'last_seen_frame':       info['last_frame'],
                'first_seen_timecode':   first_tc,
                'last_seen_timecode':    last_tc,
                'border_entry':          info['border_entry'],
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
        base_fields = ['video_filename', 'group_id', 'group_type',
                       'track_id', 'individual_type', 'auto_flagged',
                       'n_frames_present', 'duration_s', 'presence_ratio']
        behavior_fields = []
        for b in all_behaviors:
            behavior_fields += [f'behavior_{b}_s',
                                f'behavior_{b}_n',
                                f'behavior_{b}_pct']
        extra_fields = ['dominant_behavior_time', 'dominant_behavior_count']
        fieldnames = base_fields + behavior_fields + extra_fields

        with open(out1, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            # Sort by video then individual_type (group_member first)
            all_individual_rows.sort(
                key=lambda r: (r['video_filename'],
                               0 if r['individual_type'] == 'group_member' else 1,
                               r['track_id']))
            writer.writerows(all_individual_rows)

        print(f"Written: {out1}")

    # ── Write CSV 4: suspects / strangers with timecodes ──────────────────
    if all_suspect_rows:
        out4 = os.path.join(output_dir, 'activity_budget_suspects.csv')
        fieldnames4 = [
            'video_filename', 'group_id', 'group_type', 'track_id',
            'flag_reason', 'presence_ratio',
            'first_seen_frame', 'last_seen_frame',
            'first_seen_timecode', 'last_seen_timecode',
            'border_entry', 'auto_flagged', 'manual_exclude',
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
