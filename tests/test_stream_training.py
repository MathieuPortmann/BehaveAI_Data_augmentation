"""Cross-stream training switches: config helpers + the label/crop conversion.

The conversion is what makes the two checkboxes safe to flip: the settings GUI
runs it on save, and if it were not lossless every toggle would quietly corrupt
the dataset (a motion index read as a static one). These tests pin the round trip
merge -> split -> merge and the crop routing that goes with it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behaveai_config import load_stream_training_config, stream_label_classes
import Regenerate_annotations as RA


STATIC = ['Graze', 'Rest']
MOTION = ['Walk', 'Run']


def test_flags_default_to_legacy_behaviour():
    cfg = {'DEFAULT': {}}
    assert load_stream_training_config(cfg) == {
        'primary_both_streams': False, 'secondary_both_streams': False}


def test_flags_read_true():
    cfg = {'DEFAULT': {'primary_train_both_streams': 'True',
                       'secondary_train_both_streams': 'false'}}
    out = load_stream_training_config(cfg)
    assert out['primary_both_streams'] is True
    assert out['secondary_both_streams'] is False


def test_label_classes_off_are_per_stream():
    assert stream_label_classes(STATIC, MOTION, False) == (STATIC, MOTION)


def test_label_classes_on_are_the_union_in_global_order():
    static_names, motion_names = stream_label_classes(STATIC, MOTION, True)
    assert static_names == motion_names == STATIC + MOTION
    # The order is what makes a saved index mean the same class in either tree.
    assert static_names.index('Walk') == len(STATIC)


# ---------------------------------------------------------------- conversion

def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _read(path):
    with open(path) as f:
        return [l.split() for l in f.read().splitlines() if l.strip()]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project with one frame: one static box (Graze) and one motion box (Run)."""
    monkeypatch.chdir(tmp_path)
    _write(os.path.join('annot_static', 'labels', 'train', 'clip_100.txt'),
           "0 0.500000 0.500000 0.200000 0.200000\n")          # Graze  -> global 0
    _write(os.path.join('annot_motion', 'labels', 'train', 'clip_100.txt'),
           "1 0.800000 0.800000 0.100000 0.100000\n")          # Run    -> global 3
    return tmp_path


def test_merge_puts_every_box_in_both_trees_with_global_indices(project):
    RA.relabel_datasets('merge', n_static=len(STATIC), save_empty_frames='false')
    static = _read(os.path.join('annot_static', 'labels', 'train', 'clip_100.txt'))
    motion = _read(os.path.join('annot_motion', 'labels', 'train', 'clip_100.txt'))
    assert static == motion                       # both trees hold the same boxes
    assert sorted(int(r[0]) for r in static) == [0, 3]   # Graze, Run (global space)


def test_merge_then_split_round_trips(project):
    RA.relabel_datasets('merge', n_static=len(STATIC), save_empty_frames='false')
    RA.relabel_datasets('split', n_static=len(STATIC), save_empty_frames='false')
    assert _read(os.path.join('annot_static', 'labels', 'train', 'clip_100.txt')) == \
        [['0', '0.500000', '0.500000', '0.200000', '0.200000']]
    assert _read(os.path.join('annot_motion', 'labels', 'train', 'clip_100.txt')) == \
        [['1', '0.800000', '0.800000', '0.100000', '0.100000']]


def test_merge_is_idempotent(project):
    RA.relabel_datasets('merge', n_static=len(STATIC), save_empty_frames='false')
    first = _read(os.path.join('annot_static', 'labels', 'train', 'clip_100.txt'))
    RA.relabel_datasets('merge', n_static=len(STATIC), save_empty_frames='false')
    assert _read(os.path.join('annot_static', 'labels', 'train', 'clip_100.txt')) == first


def test_split_drops_the_tree_that_keeps_no_box(project, monkeypatch):
    # A frame whose boxes are all static must not leave an empty motion entry
    # behind when save_empty_frames is off - the annotation tool would not have
    # written one.
    _write(os.path.join('annot_static', 'labels', 'train', 'clip_100.txt'),
           "0 0.500000 0.500000 0.200000 0.200000\n")
    os.remove(os.path.join('annot_motion', 'labels', 'train', 'clip_100.txt'))
    _write(os.path.join('annot_motion', 'images', 'train', 'clip_100.jpg'), 'x')
    RA.relabel_datasets('merge', n_static=len(STATIC), save_empty_frames='false')
    RA.relabel_datasets('split', n_static=len(STATIC), save_empty_frames='false')
    assert not os.path.exists(os.path.join('annot_motion', 'labels', 'train', 'clip_100.txt'))
    assert not os.path.exists(os.path.join('annot_motion', 'images', 'train', 'clip_100.jpg'))


def test_merge_mirrors_the_mask_into_the_other_tree(project):
    _write(os.path.join('annot_static', 'masks', 'train', 'clip_100.mask.txt'), "1 2 3 4\n")
    os.remove(os.path.join('annot_motion', 'labels', 'train', 'clip_100.txt'))
    RA.relabel_datasets('merge', n_static=len(STATIC), save_empty_frames='false')
    assert os.path.exists(os.path.join('annot_motion', 'masks', 'train', 'clip_100.mask.txt'))


# ---------------------------------------------------------------- crops

def _params(secondary_both):
    return {'n_static': len(STATIC), 'secondary_both_streams': secondary_both,
            'primary_both_streams': True}


def test_recrop_routes_to_both_streams_when_enabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = [(0, 10, 10, 30, 30)]                       # global 0 -> a static class
    crops = {(10, 10): 'Recumbent'}
    n = RA.recrop_frame('clip_100', boxes, crops, img, img, _params(True))
    assert n == 2
    assert os.path.exists(os.path.join('annot_static_crop', 'Recumbent', 'clip_100_10_10.jpg'))
    assert os.path.exists(os.path.join('annot_motion_crop', 'Recumbent', 'clip_100_10_10.jpg'))


def test_recrop_routes_by_class_when_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = [(3, 10, 10, 30, 30)]                       # global 3 -> a motion class
    crops = {(10, 10): 'Recumbent'}
    n = RA.recrop_frame('clip_100', boxes, crops, img, img, _params(False))
    assert n == 1
    assert not os.path.isdir('annot_static_crop')
    assert os.path.exists(os.path.join('annot_motion_crop', 'Recumbent', 'clip_100_10_10.jpg'))


def test_recrop_tolerates_the_normalisation_round_trip(tmp_path, monkeypatch):
    # Crop filenames carry the box corner as the annotation tool saw it; the same
    # corner recovered from the normalised label can land a pixel off.
    monkeypatch.chdir(tmp_path)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    boxes = [(0, 11, 9, 30, 30)]
    crops = {(10, 10): 'Recumbent'}
    assert RA.recrop_frame('clip_100', boxes, crops, img, img, _params(False)) == 1


def test_recrop_drops_a_crop_with_no_box_left(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert RA.recrop_frame('clip_100', [(0, 60, 60, 80, 80)],
                           {(10, 10): 'Recumbent'}, img, img, _params(False)) == 0


def test_cli_parsing():
    assert RA.parse_cli(['proj', '--relabel', 'merge', '--recrop']) == ('proj', 'merge', True)
    assert RA.parse_cli(['--relabel=split', 'proj']) == ('proj', 'split', False)
    assert RA.parse_cli(['proj']) == ('proj', None, False)
