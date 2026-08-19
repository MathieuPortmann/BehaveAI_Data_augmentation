"""The two-stream ablation has to compare like with like.

merge_detections is not only a cross-stream arbiter: it also collapses duplicate
boxes *inside* one stream. Scoring raw static/motion output against a merged
variant that has had that clean-up would credit the fusion with a precision gain
that is really intra-stream de-duplication, so the evaluation runs every variant
through the same routine. These tests pin both halves of that claim.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BehaveAI_evaluate_detection import merge_detections

CENTROID = 0.02
IOU = 0.7
DOMINANT = 'confidence'


def _det(coords, conf, source):
    return {'coords': coords, 'primary_conf': conf,
            'primary_class': 'Graze', 'source': source}


def _merge(dets):
    return merge_detections(dets, CENTROID, IOU, DOMINANT)


def test_one_stream_alone_is_still_de_duplicated():
    # Two near-identical static boxes: the routine must return one, not two.
    dets = [_det((0.10, 0.10, 0.20, 0.20), 0.9, 'static'),
            _det((0.105, 0.105, 0.205, 0.205), 0.4, 'static')]
    assert len(_merge(dets)) == 1


def test_the_surviving_box_is_the_confident_one():
    dets = [_det((0.10, 0.10, 0.20, 0.20), 0.4, 'static'),
            _det((0.105, 0.105, 0.205, 0.205), 0.9, 'static')]
    out = _merge(dets)
    assert len(out) == 1
    assert out[0]['primary_conf'] == 0.9


def test_containment_collapses_a_nested_box_that_iou_would_keep():
    # A small box wholly inside a large one: IoU is low (~0.04) so the detector's
    # own NMS keeps both, but containment is 1.0 and the merge drops one. This is
    # the case that makes the intra-stream clean-up more than cosmetic.
    big = (0.10, 0.10, 0.50, 0.50)
    small = (0.30, 0.30, 0.38, 0.38)
    assert len(_merge([_det(big, 0.9, 'static'), _det(small, 0.5, 'static')])) == 1


def test_distinct_animals_in_one_stream_are_not_collapsed():
    dets = [_det((0.10, 0.10, 0.20, 0.20), 0.9, 'static'),
            _det((0.60, 0.60, 0.70, 0.70), 0.8, 'static')]
    assert len(_merge(dets)) == 2


def test_merging_a_stream_with_itself_is_idempotent():
    # What the ablation relies on: the single-stream variant is a fixed point, so
    # re-running the routine cannot keep shrinking the row it scores.
    dets = [_det((0.10, 0.10, 0.20, 0.20), 0.9, 'static'),
            _det((0.105, 0.105, 0.205, 0.205), 0.4, 'static'),
            _det((0.60, 0.60, 0.70, 0.70), 0.8, 'static')]
    once = _merge(dets)
    twice = _merge([_det(d['coords'], d['primary_conf'], d['source']) for d in once])
    assert len(once) == len(twice) == 2


def test_a_cross_stream_pair_still_collapses_to_one():
    # The arbitration the ablation is actually meant to measure.
    dets = [_det((0.10, 0.10, 0.20, 0.20), 0.6, 'static'),
            _det((0.105, 0.105, 0.205, 0.205), 0.9, 'motion')]
    out = _merge(dets)
    assert len(out) == 1
    assert out[0]['source'] == 'motion'
