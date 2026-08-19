"""What the crop-classifier table is allowed to claim.

Two things are easy to get wrong in a per-class table over a 9:1 pool, and both
would mislead a reader rather than merely be untidy:

  * "not measurable" printed as a measured zero -- a class with no held-out crop
    has no recall, and a class that was never predicted has no precision;
  * a top-1 accuracy quoted without the majority-class baseline that explains it.

These tests pin both, plus the Wilson interval the tables print.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import behaveai_eval_common as ec
from BehaveAI_evaluate_classifiers import class_metrics, summarise, none_split


def _r(x, nd=3):
    return None if x is None else round(x, nd)


# --- the interval ----------------------------------------------------------

def test_wilson_matches_published_values():
    # Reference values for the 95% Wilson score interval.
    assert tuple(_r(v) for v in ec.wilson_ci(130, 136)) == (0.907, 0.980)
    assert tuple(_r(v) for v in ec.wilson_ci(1, 5)) == (0.036, 0.624)


def test_wilson_stays_inside_the_unit_interval_at_the_edges():
    # k == 0 and k == n are exactly where the normal approximation degenerates.
    assert tuple(_r(v) for v in ec.wilson_ci(0, 1)) == (0.0, 0.793)
    assert tuple(_r(v) for v in ec.wilson_ci(2, 2)) == (0.342, 1.0)


def test_wilson_is_undefined_without_a_single_instance():
    assert ec.wilson_ci(0, 0) == (None, None)


# --- the "-" cells ---------------------------------------------------------

def test_a_class_with_no_held_out_crop_reports_nothing():
    # Roll/Scrape/Stamp: annotated in the ethogram, absent from the holdout.
    rows = class_metrics({('Walk', 'Walk'): 3}, ['Walk', 'Roll'])
    assert rows['Roll']['support'] == 0
    assert rows['Roll']['precision'] is None
    assert rows['Roll']['recall'] is None
    assert rows['Roll']['f1'] is None
    assert ec.fmt_ci(*ec.wilson_ci(0, 0)) == ec.UNDEFINED


def test_a_class_never_predicted_still_reports_a_real_zero_recall():
    # Drink: 2 held-out crops, never predicted. Recall 0.000 is a genuine miss
    # and must not be blanked out along with the undefined precision.
    rows = class_metrics({('Drink', 'Graze'): 2, ('Graze', 'Graze'): 5},
                         ['Graze', 'Drink'])
    assert rows['Drink']['precision'] is None
    assert rows['Drink']['recall'] == 0.0
    assert rows['Drink']['f1'] == 0.0


def test_a_support_zero_class_that_swallows_predictions_stays_visible():
    rows = class_metrics({('Graze', 'Browse'): 4}, ['Graze', 'Browse'])
    assert rows['Browse']['support'] == 0
    assert rows['Browse']['fp'] == 4
    assert rows['Browse']['precision'] == 0.0
    assert rows['Browse']['recall'] is None


# --- the baseline ----------------------------------------------------------

def test_top1_is_reported_against_the_majority_baseline():
    # The motion secondary pool in miniature: answering __none__ every time
    # scores 0.9 and must be shown as such next to the model's top-1.
    conf = {('__none__', '__none__'): 90, ('itself', '__none__'): 10}
    classes = ['__none__', 'itself']
    s = summarise(conf, classes, class_metrics(conf, classes))
    assert _r(s['top1']) == 0.9
    assert _r(s['majority_baseline']) == 0.9
    assert s['majority_class'] == '__none__'
    assert _r(s['balanced_accuracy']) == 0.5


def test_macro_f1_ignores_the_classes_that_could_not_be_measured():
    conf = {('a', 'a'): 4, ('b', 'b'): 1}
    classes = ['a', 'b', 'never_annotated']
    s = summarise(conf, classes, class_metrics(conf, classes))
    assert s['classes_with_support'] == 2
    assert s['classes_total'] == 3
    assert _r(s['macro_f1']) == 1.0


# --- the two questions a pooled accuracy conflates -------------------------

def test_presence_and_identity_are_scored_separately():
    conf = {('__none__', '__none__'): 90,
            ('itself', 'itself'): 6,
            ('itself', 'Tree'): 2,      # found, misnamed
            ('Tree', '__none__'): 2}    # missed entirely
    ns = none_split(conf, ['__none__', 'itself', 'Tree'])
    assert (ns['tp'], ns['fn'], ns['fp']) == (8, 2, 0)
    assert _r(ns['recall']) == 0.8          # presence: 8 of 10 found
    assert ns['n_positive'] == 10
    assert _r(ns['identity_accuracy']) == 0.6   # identity: only 6 named right


def test_models_without_a_none_class_have_no_split_to_report():
    assert none_split({('Adult', 'Adult'): 5}, ['Adult', 'Foal']) is None
