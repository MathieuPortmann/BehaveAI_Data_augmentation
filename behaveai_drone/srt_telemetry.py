#!/usr/bin/env python3
"""
BehaveAI SRT Telemetry Parser

Generic per-key parser for DJI .SRT sidecar telemetry, tolerant of the
field-format differences observed across DJI drone models/firmwares:
spacing before ':' ("iso : 100" vs "iso: 110"), field order, presence or
absence of dzoom_ratio, and "SrtCnt" vs "FrameCnt" block counters. Instead of
one fixed-order regex over the whole metadata line (brittle — breaks the
moment a field is missing, reordered, or spaced differently), it extracts
every "key: value" token generically, so new/missing/reordered fields don't
break parsing.

Known cross-firmware encoding quirks — left as RAW STRINGS here, callers
convert deliberately (see focal_len_mm() below for the one that matters for
the horizon-geometry pipeline):
  - focal_len: some firmwares write true mm ("24.00"), others write mm*10 as
    an int ("240") — disambiguated by the presence of a decimal point.
  - fnum: similarly either the raw f-number ("1.7") or f-number*100 ("170").
    Not used by this pipeline, left unconverted.
  - dzoom_ratio: present only on some models; its absence means "no zoom
    telemetry on this drone" (assume 1.0x), not a parse failure.
"""

import re
from pathlib import Path

_KV_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^\s,\]\[]+)')
_TIMESTAMP_RE = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)')


def parse_srt(path):
	"""Parse a DJI .SRT sidecar into a list of per-frame telemetry dicts.

	Each dict has whatever keys were present in that block's metadata line
	(all values as raw strings — numeric conversion/unit handling is the
	caller's job, since encoding varies by field, see module docstring),
	plus 'timestamp' when present. Blocks with no recognisable key:value
	telemetry are skipped (e.g. the leading bare frame-index block, or a
	malformed trailing block).
	"""
	text = Path(path).read_text(encoding="utf-8", errors="replace")
	rows = []
	for block in text.split("\n\n"):
		fields = dict(_KV_RE.findall(block))
		if not fields:
			continue
		tm = _TIMESTAMP_RE.search(block)
		if tm:
			fields['timestamp'] = tm.group(1)
		rows.append(fields)
	return rows


def focal_len_mm(raw):
	"""Disambiguate focal_len encoding: true mm ('24.00', has a decimal
	point) vs mm*10 as an int ('240', no decimal point). Returns float mm."""
	v = float(raw)
	return v if '.' in raw else v / 10.0


def difftime_ms(raw):
	"""Parse a DiffTime value like '33ms' into a float number of ms."""
	m = re.match(r'([\d.]+)', raw)
	return float(m.group(1)) if m else None
