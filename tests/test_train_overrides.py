#!/usr/bin/env python3
"""Regression tests for the INI -> model.train() override parser.

Runnable two ways, so nobody needs a test runner installed to check a change:
    python tests/test_train_overrides.py
    pytest tests/test_train_overrides.py

The point of the parser is that a mistake is reported instead of silently
ignored: before it existed, the only way to reach Ultralytics' augmentation
arguments was to edit the source. Each test pins one way of getting it wrong.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaveai_config import parse_train_overrides as P


def test_empty_is_none():
	# None (not {}) so the caller can pass it straight through as "use defaults".
	assert P('', 'x') is None
	assert P(None, 'x') is None
	assert P('   ', 'x') is None


def test_parses_floats_and_ints_by_default_type():
	got = P('mosaic=0.0, scale=0.2, close_mosaic=25', 'x')
	assert got == {'mosaic': 0.0, 'scale': 0.2, 'close_mosaic': 25}
	assert isinstance(got['close_mosaic'], int)     # int default -> int
	assert isinstance(got['mosaic'], float)


def test_none_and_bools():
	got = P('auto_augment=none, cos_lr=true, rect=False', 'x')
	assert got['auto_augment'] is None
	assert got['cos_lr'] is True
	assert got['rect'] is False


def test_unknown_key_is_dropped_not_passed_through():
	# A typo must never reach model.train(), where it would raise deep inside
	# training after the dataset scan -- or worse, be silently swallowed.
	assert P('mosiac=0.0', 'x') is None
	assert P('mosaic=0.0, mosiac=1.0', 'x') == {'mosaic': 0.0}


def test_reserved_keys_cannot_be_hijacked():
	# imgsz/epochs/patience have dedicated settings; an override would win the
	# dict update in the worker and silently contradict them.
	for key in ('imgsz=1280', 'epochs=10', 'patience=5', 'data=foo.yaml',
				'project=bar', 'model=yolo26x.pt'):
		assert P(key, 'x') is None, key


def test_bad_value_is_dropped_not_crashing():
	assert P('mosaic=abc', 'x') is None
	assert P('cos_lr=0.5', 'x') is None            # bool key, numeric value
	assert P('mosaic=abc, scale=0.2', 'x') == {'scale': 0.2}


def test_malformed_entries_are_skipped():
	assert P('mosaic', 'x') is None                 # no '='
	assert P('mosaic=0.0,,  , scale=0.2', 'x') == {'mosaic': 0.0, 'scale': 0.2}


def test_whitespace_tolerated():
	assert P('  mosaic = 0.0 ,  scale =0.2  ', 'x') == {'mosaic': 0.0, 'scale': 0.2}


def _main():
	fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
	failed = 0
	for fn in fns:
		try:
			fn()
			print(f"  PASS  {fn.__name__}")
		except AssertionError as e:
			failed += 1
			print(f"  FAIL  {fn.__name__}: {e}")
	print(f"\n{len(fns) - failed}/{len(fns)} passed")
	return 1 if failed else 0


if __name__ == '__main__':
	sys.exit(_main())
