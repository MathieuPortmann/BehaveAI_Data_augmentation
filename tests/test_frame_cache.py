#!/usr/bin/env python3
"""Tests for the pre-cache of mined frames.

Runnable two ways:
    python tests/test_frame_cache.py
    pytest tests/test_frame_cache.py

The important test here is the first one. The motion false-colour image is not
a view of the video, it *is* what annot_motion/images holds and what the motion
detector trains on. Moving its computation into behaveai_frame_cache was only
safe if the pixels did not change: if they had, every motion annotation already
made would refer to an image the tool no longer produces, silently. So the new
path is compared against the implementation as it stood in git before the
refactor, on a real decoded video, pixel for pixel.
"""

import os
import ast
import sys
import shutil
import tempfile
import subprocess

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import behaveai_frame_cache as fc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The commit that introduced the shared module; its parent still has the inline
# implementation this refactor replaced.
BASELINE_REV = 'dd64885'


def _params(**over):
    base = dict(strategy='exponential', expA=0.5, expB=0.8, lum_weight=0.7,
                rgb_multipliers=[1.0, 1.0, 1.0], chromatic_tail_only='false',
                frame_skip=0, motion_threshold=0, scale_factor=1.0)
    base.update(over)
    fw, buf = fc.derive_frame_window(base['strategy'], base['expA'],
                                     base['expB'], base['frame_skip'])
    return fc.MotionParams(frame_window=fw, buf_len=buf, **base)


def _make_video(path, n_frames=60, w=160, h=120, seed=0):
    """A synthetic clip with a moving blob, so the diffs are non-trivial."""
    rng = np.random.RandomState(seed)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (w, h))
    assert writer.isOpened(), "no MJPG encoder available"
    for i in range(n_frames):
        frame = (rng.rand(h, w, 3) * 40).astype(np.uint8)
        cx = 20 + (i * 2) % (w - 40)
        cv2.rectangle(frame, (cx, 40), (cx + 20, 70), (200, 180, 160), -1)
        writer.write(frame)
    writer.release()


def _baseline_compute():
    """Pull the pre-refactor _compute_frame_data out of git and make it callable.

    Executed from source rather than imported: BehaveAI_annotation.py opens a
    settings dialog at import time. Returns None when the baseline revision is
    not reachable (shallow clone, exported tree), so the suite degrades to
    skipping this check rather than failing for the wrong reason.
    """
    try:
        src = subprocess.check_output(
            ['git', 'show', f'{BASELINE_REV}:BehaveAI_annotation.py'],
            cwd=REPO, stderr=subprocess.DEVNULL).decode('utf-8', 'replace')
    except Exception:
        return None
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == '_compute_frame_data'), None)
    if fn is None:
        return None
    p = _params()
    ns = {
        'cv2': cv2, 'np': np,
        'frameWindow': p.frame_window, 'scale_factor': p.scale_factor,
        'strategy': p.strategy, 'expA': p.expA, 'expB': p.expB,
        'frame_skip': p.frame_skip, 'chromatic_tail_only': p.chromatic_tail_only,
        'lum_weight': p.lum_weight, 'rgb_multipliers': p.rgb_multipliers,
        'motion_threshold': p.motion_threshold,
        '_display_size_hint': (800, 600),
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'baseline', 'exec'), ns)
    return ns['_compute_frame_data']


def test_motion_image_is_unchanged_by_the_refactor():
    baseline = _baseline_compute()
    if baseline is None:
        print("    (skipped: baseline revision not reachable)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, 'clip.avi')
        _make_video(vid)
        p = _params()
        for fnum in (20, 35, 50):
            old = baseline(vid, fnum)
            new = fc.compute_frame_data(vid, fnum, p)
            assert (old is None) == (new is None), f"frame {fnum}: one path gave None"
            if old is None:
                continue
            assert np.array_equal(old['motion_image'], new['motion_image']), \
                f"frame {fnum}: motion image differs from the pre-refactor code"
            assert np.array_equal(old['fr'], new['fr']), \
                f"frame {fnum}: static frame differs from the pre-refactor code"


def test_frame_window_matches_the_annotation_tool():
    """The thresholds are the annotation tool's; a mismatch would make the cache
    read a different number of frames than the live path and so produce a
    different motion image for the same frame."""
    # The thresholds are checked against max(expA, expB), not against expA: the
    # project default 0.5/0.8 clears the 0.7 step on expB alone and lands on 15.
    assert fc.derive_frame_window('exponential', 0.5, 0.8, 0) == (15, 15)
    # Strictly greater: 0.5/0.5 clears the 0.2 step only.
    assert fc.derive_frame_window('exponential', 0.5, 0.5, 0) == (5, 5)
    assert fc.derive_frame_window('exponential', 0.6, 0.6, 0) == (10, 10)
    assert fc.derive_frame_window('exponential', 0.95, 0.95, 0) == (45, 45)
    assert fc.derive_frame_window('rolling', 0.95, 0.95, 0) == (4, 4)
    # frame_skip stretches the source window without changing the buffer.
    assert fc.derive_frame_window('exponential', 0.5, 0.8, 2) == (45, 15)


def test_entry_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, 'clip.avi')
        _make_video(vid)
        cdir = fc.cache_dir(tmp)
        data = fc.compute_frame_data(vid, 30, _params())
        assert data is not None
        base = fc.base_name('clip', 30)
        assert not fc.has_entry(cdir, base)
        fc.write_entry(cdir, base, data)
        assert fc.has_entry(cdir, base)

        back = fc.read_entry(cdir, base)
        assert back is not None
        assert back['video_width'] == data['video_width']
        assert back['frame_number'] == 30
        assert back['from_cache'] is True
        assert len(back['raw_buf']) == len(data['raw_buf'])
        # JPEG, so not identical — but it must be the same picture, not a
        # different frame or a mis-ordered channel swap.
        diff = np.abs(back['motion_image'].astype(int) - data['motion_image'].astype(int))
        assert diff.mean() < 3.0, diff.mean()


def test_an_incomplete_entry_is_a_miss_not_a_truncated_hit():
    """A run killed mid-frame leaves images with no metadata. Treating that as a
    hit would serve the annotator half a frame."""
    with tempfile.TemporaryDirectory() as tmp:
        cdir = fc.cache_dir(tmp)
        os.makedirs(cdir)
        base = fc.base_name('clip', 30)
        cv2.imwrite(fc.static_path(cdir, base), np.zeros((8, 8, 3), np.uint8))
        cv2.imwrite(fc.motion_path(cdir, base), np.zeros((8, 8, 3), np.uint8))
        assert not fc.has_entry(cdir, base)
        assert fc.read_entry(cdir, base) is None


def test_take_image_moves_and_discard_clears_the_rest():
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, 'clip.avi')
        _make_video(vid)
        cdir = fc.cache_dir(tmp)
        base = fc.base_name('clip', 30)
        fc.write_entry(cdir, base, fc.compute_frame_data(vid, 30, _params()))

        dest = os.path.join(tmp, 'annot_static', 'images', 'train', base + '.jpg')
        assert fc.take_image(cdir, base, 'static', dest)
        assert os.path.exists(dest), "moved file is not at the destination"
        assert not os.path.exists(fc.static_path(cdir, base)), "source still there"
        # A move, not a copy: the entry is now incomplete and must read as a miss.
        assert not fc.has_entry(cdir, base)
        # Second call has nothing left to move and says so rather than raising.
        assert not fc.take_image(cdir, base, 'static', dest)

        fc.discard_entry(cdir, base)
        assert fc.entry_count(cdir) == 0
        assert not [n for n in os.listdir(cdir) if n.startswith(base + '.')]
        assert os.path.exists(dest), "discard must not touch the dataset"


def test_skipping_the_animation_buffer_is_a_valid_entry():
    """It is the bulk of the cache (10-45 frames per target against two images),
    so dropping it must stay a complete, readable entry -- not a broken one."""
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, 'clip.avi')
        _make_video(vid)
        cdir = fc.cache_dir(tmp)
        data = fc.compute_frame_data(vid, 30, _params())
        assert len(data['raw_buf']) > 1, "test needs a non-trivial buffer"

        small = fc.write_entry(cdir, fc.base_name('clip', 30), data, animation=False)
        big = fc.write_entry(cdir, fc.base_name('clip', 31), data, animation=True)
        assert small < big, (small, big)

        back = fc.read_entry(cdir, fc.base_name('clip', 30))
        assert back is not None, "an entry without the buffer must still read"
        assert back['raw_buf'] == []
        assert back['motion_image'] is not None
        assert not [n for n in os.listdir(cdir) if n.startswith('clip_30.buf')]


def test_discard_is_silent_when_there_is_no_entry():
    """Outside mining mode every save calls this and no cache exists."""
    with tempfile.TemporaryDirectory() as tmp:
        fc.discard_entry(fc.cache_dir(tmp), 'clip_30')
        fc.discard_entry(os.path.join(tmp, 'nope'), 'clip_30')


def test_a_frame_too_early_for_the_window_is_refused():
    """The motion image is defined by its run-up; there is no honest short one."""
    with tempfile.TemporaryDirectory() as tmp:
        vid = os.path.join(tmp, 'clip.avi')
        _make_video(vid)
        assert fc.compute_frame_data(vid, 2, _params()) is None


def test_cache_base_name_matches_the_dataset_naming():
    """This is what lets a cached image be moved instead of re-encoded: the
    annotation tool writes <video_label>_<frame>.jpg."""
    assert fc.base_name('Mini3Pro_2026-04-11_09-32-46_Geldings', 6263) == \
        'Mini3Pro_2026-04-11_09-32-46_Geldings_6263'


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
