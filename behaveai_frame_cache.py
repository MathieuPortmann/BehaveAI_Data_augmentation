# behaveai_frame_cache.py
# On-disk cache of decoded frames, shared by the frame miner (which fills it)
# and the annotation tool (which empties it).
#
# WHY THIS EXISTS
#   Serving one mined frame costs a seek plus `frameWindow` sequential reads on
#   a 4K clip living on an SSHFS mount. That is seconds of latency per frame,
#   paid interactively, one annotator sitting in front of it. The work is
#   perfectly predictable, though: the miner already knows the whole list of
#   frames it is about to propose. Doing it ahead of time, in parallel, over a
#   link that is throughput-bound rather than latency-bound turns a per-frame
#   stall into one batch wait.
#
#   Nothing here changes what gets annotated. It changes when the decoding
#   happens.
#
# THE MOTION FORMULA LIVES HERE, ONCE
#   The false-colour image is not a view of the video, it *is* the training
#   input: annot_motion/images holds these pixels, and the boxes an annotator
#   draws refer to them. So `strategy`, `expA`, `expB`, `lum_weight`,
#   `rgb_multipliers`, `chromatic_tail_only`, `frame_skip`, `motion_threshold`
#   and `scale_factor` are not display preferences -- changing one silently
#   invalidates every motion annotation already made.
#
#   That makes a second copy of the formula the worst possible kind of
#   duplication: a cache that drifts from the live path would feed the dataset
#   images that no longer match what the annotator was shown, with no error
#   anywhere. compute_frame_data() below is therefore the only implementation,
#   and BehaveAI_annotation.py calls it rather than keeping its own.
#
# LAYOUT
#   <project>/mined_frames/
#       <video_label>_<frame>.json        video path, dimensions, buffer length
#       <video_label>_<frame>.static.jpg  static RGB frame (after scale_factor)
#       <video_label>_<frame>.motion.jpg  motion false-colour frame
#       <video_label>_<frame>.buf<i>.jpg  animation buffer, oldest first
#
#   `<video_label>_<frame>` is exactly the `base_filename` the annotation tool
#   writes into the dataset, which is what lets a saved frame be moved rather
#   than re-encoded (see consume_entry and the caveat on it).

import os
import csv
import json
import shutil

import cv2
import numpy as np

CACHE_DIRNAME = 'mined_frames'

# Quality of the cached static/motion JPEGs. High on purpose: for the streams
# that can be moved into the dataset verbatim these ARE the dataset images, and
# for the others they are re-encoded once, so a low setting would compound.
JPEG_QUALITY = 95

# The animation buffer is different in kind. It never reaches the dataset -- it
# only feeds the small looping zoom pane that shows the annotator whether the
# animal is moving -- so compression artefacts there cost nothing, while its
# size dominates everything else: buf_len is 15 frames at the project's default
# 0.5/0.8 decay, against one static and one motion image. At 4K and q95 that is
# ~88 % of the cache, or 75 MB per mined frame; at q70 it is roughly a third of
# that, and it can be dropped entirely (see `animation` below) for the ~7 MB
# per frame that buys the whole latency win on its own.
BUF_JPEG_QUALITY = 70

_ENC = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
_BUF_ENC = [int(cv2.IMWRITE_JPEG_QUALITY), BUF_JPEG_QUALITY]


class MotionParams(object):
    """The motion-engine settings, resolved once from a project INI.

    Kept as an explicit object rather than read from module globals so the same
    computation can run in a worker thread, in the miner, and in the annotation
    tool without three different notions of the current settings.
    """

    __slots__ = ('strategy', 'expA', 'expB', 'lum_weight', 'rgb_multipliers',
                 'chromatic_tail_only', 'frame_skip', 'motion_threshold',
                 'scale_factor', 'frame_window', 'buf_len')

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


def derive_frame_window(strategy, expA, expB, frame_skip):
    """Return (frame_window, buf_len).

    An exponential decay needs a longer run-up before the trailing frames stop
    depending on whatever preceded the window, and the thresholds below are the
    annotation tool's own, reproduced so the cache reads exactly as many frames
    as the live path would. buf_len is the pre-skip count (the animation buffer
    holds decoded frames, not source frames).
    """
    window = 4
    if strategy == 'exponential':
        if expA > 0.2 or expB > 0.2:
            window = 5
        if expA > 0.5 or expB > 0.5:
            window = 10
        if expA > 0.7 or expB > 0.7:
            window = 15
        if expA > 0.8 or expB > 0.8:
            window = 20
        if expA > 0.9 or expB > 0.9:
            window = 45
    return window * (frame_skip + 1), window


def motion_params_from_config(config):
    """Build MotionParams from a configparser holding a BehaveAI_settings.ini."""
    d = config['DEFAULT']
    strategy = d.get('strategy', 'exponential')
    expA = float(d.get('expA', '0.5'))
    expB = float(d.get('expB', '0.8'))
    frame_skip = int(d.get('frame_skip', '0'))
    frame_window, buf_len = derive_frame_window(strategy, expA, expB, frame_skip)
    return MotionParams(
        strategy=strategy,
        expA=expA,
        expB=expB,
        lum_weight=float(d.get('lum_weight', '0.7')),
        rgb_multipliers=[float(x) for x in d.get('rgb_multipliers', '1,1,1').split(',')],
        chromatic_tail_only=str(d.get('chromatic_tail_only', 'false')).lower(),
        frame_skip=frame_skip,
        # Negated exactly as the annotation tool does: the INI states a noise
        # floor, cv2.addWeighted takes an offset.
        motion_threshold=-1 * int(d.get('motion_threshold', '0')),
        scale_factor=float(d.get('scale_factor', '1.0')),
        frame_window=frame_window,
        buf_len=buf_len,
    )


def compute_frame_data(vpath, fnum, p):
    """Decode `fnum` of `vpath` and build its static + motion pair.

    Returns a dict, or None when the frame cannot be produced -- notably when
    fewer than `frame_window` frames precede it, since the motion image is
    defined by the run-up and there is no honest way to fake a shorter one.

    Pure with respect to module state so it is safe in a thread pool.
    """
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        return None
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_frame = fnum - p.frame_window + 1
        if start_frame < 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        prev_frames = [None] * 3
        raw_buf = []
        local_fr = None
        gray = None
        diffs = None
        frame_count = 0

        for i in range(p.frame_window):
            ret, raw_frame = cap.read()
            if not ret:
                break
            if frame_count == 0:
                local_fr = raw_frame.copy()
                if p.scale_factor != 1.0:
                    local_fr = cv2.resize(local_fr, (0, 0),
                                          fx=p.scale_factor, fy=p.scale_factor)
                raw_buf.append(local_fr.copy())
                gray = cv2.cvtColor(local_fr, cv2.COLOR_BGR2GRAY)
                if i == 0:
                    prev_frames = [gray.copy()] * 3
                    frame_count += 1
                    if frame_count > p.frame_skip:
                        frame_count = 0
                    continue
                diffs = [cv2.absdiff(prev_frames[j], gray) for j in range(3)]
                if p.strategy == 'exponential':
                    prev_frames[0] = gray
                    prev_frames[1] = cv2.addWeighted(prev_frames[1], p.expA, gray, 1 - p.expA, 0)
                    prev_frames[2] = cv2.addWeighted(prev_frames[2], p.expB, gray, 1 - p.expB, 0)
                else:
                    prev_frames[2] = prev_frames[1]
                    prev_frames[1] = prev_frames[0]
                    prev_frames[0] = gray
            frame_count += 1
            if frame_count > p.frame_skip:
                frame_count = 0
    finally:
        cap.release()

    if diffs is None or local_fr is None:
        return None

    if p.chromatic_tail_only == 'true':
        tb = cv2.subtract(diffs[0], diffs[1])
        tr = cv2.subtract(diffs[2], diffs[1])
        tg = cv2.subtract(diffs[1], diffs[0])
        blue = cv2.addWeighted(gray, p.lum_weight, tb, p.rgb_multipliers[2], p.motion_threshold)
        green = cv2.addWeighted(gray, p.lum_weight, tg, p.rgb_multipliers[1], p.motion_threshold)
        red = cv2.addWeighted(gray, p.lum_weight, tr, p.rgb_multipliers[0], p.motion_threshold)
    else:
        blue = cv2.addWeighted(gray, p.lum_weight, diffs[0], p.rgb_multipliers[2], p.motion_threshold)
        green = cv2.addWeighted(gray, p.lum_weight, diffs[1], p.rgb_multipliers[1], p.motion_threshold)
        red = cv2.addWeighted(gray, p.lum_weight, diffs[2], p.rgb_multipliers[0], p.motion_threshold)

    motion = cv2.merge((blue, green, red)).astype(np.uint8)

    return {
        'video_path': vpath,
        'frame_number': fnum,
        'fr': local_fr,
        'motion_image': motion,
        'original_frame': motion.copy(),
        'raw_buf': raw_buf[-p.buf_len:],
        'video_width': vid_width,
        'video_height': vid_height,
        'total_frames': n_frames,
    }


# --------------------------------------------------------------------------
# Cache layout


def cache_dir(project_dir):
    return os.path.join(project_dir, CACHE_DIRNAME)


def base_name(video_label, frame):
    """The dataset's own naming, so a cached image can be moved into place."""
    return f"{video_label}_{frame}"


def _meta_path(cdir, base):
    return os.path.join(cdir, base + '.json')


def static_path(cdir, base):
    return os.path.join(cdir, base + '.static.jpg')


def motion_path(cdir, base):
    return os.path.join(cdir, base + '.motion.jpg')


def _buf_path(cdir, base, i):
    return os.path.join(cdir, f"{base}.buf{i}.jpg")


def has_entry(cdir, base):
    """A cache hit requires the metadata *and* both images.

    The metadata is written last, so a run interrupted mid-frame leaves images
    without it and is treated as a miss rather than as a truncated hit.
    """
    return (os.path.exists(_meta_path(cdir, base))
            and os.path.exists(static_path(cdir, base))
            and os.path.exists(motion_path(cdir, base)))


def write_entry(cdir, base, data, animation=True):
    """Write one frame's cache entry. Returns the bytes actually written.

    `animation=False` skips the buffer: the annotation tool then shows a still
    in the animation pane for that frame and everything else is unchanged. That
    is the difference between roughly 7 MB and 25 MB per 4K frame.
    """
    os.makedirs(cdir, exist_ok=True)
    paths = [static_path(cdir, base), motion_path(cdir, base)]
    cv2.imwrite(paths[0], data['fr'], _ENC)
    cv2.imwrite(paths[1], data['motion_image'], _ENC)
    n_buf = 0
    if animation:
        for i, img in enumerate(data['raw_buf']):
            p = _buf_path(cdir, base, i)
            cv2.imwrite(p, img, _BUF_ENC)
            paths.append(p)
        n_buf = len(data['raw_buf'])
    meta = {
        'video_path': data['video_path'],
        'frame_number': data['frame_number'],
        'video_width': data['video_width'],
        'video_height': data['video_height'],
        'total_frames': data['total_frames'],
        'n_buf': n_buf,
    }
    # Last, so it is the marker that the entry is complete.
    with open(_meta_path(cdir, base), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


def read_entry(cdir, base):
    """Load a cache entry back into the shape compute_frame_data returns, or
    None if it is absent or unreadable."""
    if not has_entry(cdir, base):
        return None
    try:
        with open(_meta_path(cdir, base), encoding='utf-8') as f:
            meta = json.load(f)
        fr = cv2.imread(static_path(cdir, base))
        motion = cv2.imread(motion_path(cdir, base))
        if fr is None or motion is None:
            return None
        raw_buf = []
        for i in range(meta.get('n_buf', 0)):
            img = cv2.imread(_buf_path(cdir, base, i))
            if img is not None:
                raw_buf.append(img)
    except Exception:
        return None
    return {
        'video_path': meta['video_path'],
        'frame_number': meta['frame_number'],
        'fr': fr,
        'motion_image': motion,
        'original_frame': motion.copy(),
        'raw_buf': raw_buf,
        'video_width': meta['video_width'],
        'video_height': meta['video_height'],
        'total_frames': meta['total_frames'],
        'from_cache': True,
    }


def take_image(cdir, base, stream, dest_path):
    """Move the cached `stream` ('static'/'motion') image to dest_path.

    Returns True when the file was moved. This is the no-re-encode path and it
    is only correct when the image the annotator saw is byte-for-byte the image
    the dataset should hold -- i.e. when no grey blocking rectangle applies to
    that stream. The caller decides; see save_annotation.
    """
    src = static_path(cdir, base) if stream == 'static' else motion_path(cdir, base)
    if not os.path.exists(src):
        return False
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.move(src, dest_path)
    return True


def discard_entry(cdir, base):
    """Drop whatever remains of an entry once its frame has been annotated.

    Tolerant of missing files: take_image may already have moved one or both
    images out, which is the normal case rather than an error.
    """
    if not os.path.isdir(cdir):
        return
    prefix = base + '.'
    for name in os.listdir(cdir):
        if name.startswith(prefix):
            try:
                os.remove(os.path.join(cdir, name))
            except OSError:
                pass


def entry_count(cdir):
    """How many complete entries the cache holds."""
    if not os.path.isdir(cdir):
        return 0
    return sum(1 for n in os.listdir(cdir) if n.endswith('.json'))


# --------------------------------------------------------------------------
# The target list
#
# It lives beside the frames it points at rather than in output/, because the
# two are one artefact: the list says what to annotate, the cache holds it
# decoded, and both are consumed together. Keeping them apart meant a stale
# list could outlive the frames, or survive a cache wipe pointing at nothing.
#
# Rows are removed as their frames get annotated, so the file doubles as the
# resume point: an interrupted session leaves exactly the work left to do, and
# reopening it picks up there rather than at the top.

TARGETS_NAME = 'mining_targets.csv'


def targets_path(cdir):
    return os.path.join(cdir, TARGETS_NAME)


def read_targets(path):
    """Return (fieldnames, rows). ([], []) when the file is absent or unreadable."""
    if not path or not os.path.exists(path):
        return [], []
    try:
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            return list(reader.fieldnames or []), list(reader)
    except Exception:
        return [], []


def _column(fieldnames, *names):
    """First matching column, case-insensitively — the parser in the annotation
    tool accepts several spellings and this has to agree with it."""
    lowered = {str(c).strip().lower(): c for c in fieldnames}
    for n in names:
        if n in lowered:
            return lowered[n]
    return None


def _write_targets(path, fieldnames, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # Replace in one step: a session killed mid-write must not leave a
    # half-written list, which would look like "almost everything is done".
    os.replace(tmp, path)


def prune_targets(path, is_annotated):
    """Drop every row whose frame is already annotated.

    `is_annotated(video_label, frame)` decides. Returns the number of rows
    removed, or 0 when the file has no usable video/frame columns (a
    hand-written time-code CSV, which must be left exactly as the user wrote
    it).
    """
    fieldnames, rows = read_targets(path)
    if not rows:
        return 0
    vcol = _column(fieldnames, 'video_filename', 'video', 'filename')
    fcol = _column(fieldnames, 'frame', 'start_frame')
    if not vcol or not fcol:
        return 0

    keep = []
    for r in rows:
        try:
            frame = int(float(str(r.get(fcol, '')).strip()))
        except (TypeError, ValueError):
            keep.append(r)      # unparseable: not ours to throw away
            continue
        label = os.path.splitext(str(r.get(vcol, '')).strip())[0]
        if not is_annotated(label, frame):
            keep.append(r)
    removed = len(rows) - len(keep)
    if removed:
        _write_targets(path, fieldnames, keep)
    return removed


def drop_target(path, video_label, frame):
    """Remove the row for one frame, once it has been annotated.

    Called on every save, so it is cheap and silent when the file does not
    exist or does not mention this frame.
    """
    return prune_targets(
        path, lambda lbl, f: lbl == video_label and f == int(frame)) > 0
