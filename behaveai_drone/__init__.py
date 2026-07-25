"""behaveai_drone -- drone/telemetry plugin for BehaveAI/HERDWISE.

Everything specific to *drone acquisition* lives here so it can be detached
later without touching the core pipeline:
  - drone_correction : stabilise tracks against apparent drone pan/zoom
                       (optical flow, no telemetry);
  - flightlog        : read <video>.flightlog.csv sidecars (rel_alt, gimbal);
  - srt_telemetry    : parse DJI SRT (rel_alt, focal length);
  - horizon_geometry : ground-plane geometry + inverse projection;
  - metric_geometry  : project image tracks to ground-plane metres.

The core pipeline (detection, tracker, offline stitching) does NOT depend on
this package: on a fixed camera it works without it; on a moving drone it uses
drone_correction's stabilised coordinates.
"""

from .drone_correction import run_drone_correction
from .metric_geometry import run_metric_geometry
from .flightlog import load_flightlog, sample_flightlog, flightlog_summary
from .srt_telemetry import parse_srt, focal_len_mm
from .horizon_geometry import (
	pixel_focal_length, camera_pitch_rad, horizon_row,
	ground_point_from_pixel, ground_distance_m)

__all__ = [
	'run_drone_correction', 'run_metric_geometry',
	'load_flightlog', 'sample_flightlog', 'flightlog_summary',
	'parse_srt', 'focal_len_mm',
	'pixel_focal_length', 'camera_pitch_rad', 'horizon_row',
	'ground_point_from_pixel', 'ground_distance_m',
]
