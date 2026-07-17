# behaveai_render.py
# Shared box/label drawing used by BOTH the classify/track pipeline (output
# videos) and the annotation tool, so the two render identically.
#
# Both callers previously carried their own near-identical _draw_stacked_label
# plus box-drawing branches; everything now goes through draw_labeled_detection.
#
# Sizing model
#   The base font/thickness are derived from the NATIVE bounding-box size, then
#   multiplied by `display_scale` (1.0 for output video; zoom * canvas-fit scale
#   for the annotation tool) and the per-tool multipliers box_line_scale /
#   box_font_scale. Deriving from native size before scaling keeps a given animal
#   visually stable while zooming.
#
# Setting adaptive_box_scaling = false restores the previous flat
# font_size/line_thickness behaviour.

import cv2
import numpy as np

from behaveai_config import _parse_colors

FONT = cv2.FONT_HERSHEY_SIMPLEX

PENDING_COLOR = (0, 0, 255)      # BGR red - box with no primary chosen yet
SELECTED_COLOR = (0, 255, 255)   # BGR cyan - selection highlight
PENDING_TEXT = "? choose primary"

# Extra pixels the hierarchical outer (primary) box extends beyond the inner one.
OUTER_EXTRA = 2

LABEL_BG_MODES = ('none', 'translucent', 'solid')


def _to_bool(raw, default=False):
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in ('1', 'true', 'yes', 'on'):
        return True
    if s in ('0', 'false', 'no', 'off'):
        return False
    return default


def _to_float(raw, default):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _to_int(raw, default):
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _parse_one_color(raw, default):
    """Parse a single 'R,G,B' INI value into a BGR tuple, via the same helper the
    rest of the codebase uses for colour lists."""
    cols = _parse_colors(raw)
    return cols[0] if cols else default


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _text_thickness(box_thickness):
    """Text stroke is deliberately decoupled from the box line: a thick border on a
    large animal must not turn its label into bold, blocky text. Keeps labels thin
    and legible while the box line alone carries the size cue."""
    return max(1, min(2, int(round(box_thickness / 2.0))))


class RenderStyle(object):
    """Display settings, read once from the INI [DEFAULT] section."""

    __slots__ = (
        'base_font', 'base_thickness', 'adaptive',
        'font_coeff', 'font_min', 'font_max',
        'thick_coeff', 'thick_min', 'thick_max',
        'label_bg_mode', 'label_bg_opacity', 'label_bg_color',
        'halo_thickness', 'halo_color',
        'show_species', 'show_age', 'outer_extra',
    )

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw.get(name))


def load_render_style(config):
    """Build a RenderStyle from a configparser instance. Every key is optional so
    projects whose INI predates these settings keep working unchanged."""
    d = config['DEFAULT']
    mode = str(d.get('label_bg_mode', 'translucent')).strip().lower()
    if mode not in LABEL_BG_MODES:
        mode = 'translucent'
    # A min above its max would otherwise pin the value to min and silently ignore
    # the box size, which reads as "the settings do nothing".
    font_min, font_max = sorted((_to_float(d.get('adaptive_font_min', '0.35'), 0.35),
                                 _to_float(d.get('adaptive_font_max', '0.9'), 0.9)))
    thick_min, thick_max = sorted((max(0.0, _to_float(d.get('adaptive_thickness_min', '1.0'), 1.0)),
                                   max(0.0, _to_float(d.get('adaptive_thickness_max', '3.0'), 3.0))))
    return RenderStyle(
        base_font=_to_float(d.get('font_size', '0.5'), 0.5),
        base_thickness=_to_int(d.get('line_thickness', '1'), 1),
        adaptive=_to_bool(d.get('adaptive_box_scaling', 'true'), True),
        font_coeff=_to_float(d.get('adaptive_font_coeff', '0.005'), 0.005),
        font_min=font_min,
        font_max=font_max,
        thick_coeff=_to_float(d.get('adaptive_thickness_coeff', '0.012'), 0.012),
        # Fractional: kept as floats through the whole computation and only rounded
        # to whole pixels at the cv2 call, so values like 0.75 / 2.25 still shift the
        # result (notably in the annotation tool, where zoom multiplies them).
        thick_min=thick_min,
        thick_max=thick_max,
        label_bg_mode=mode,
        label_bg_opacity=_clamp(_to_float(d.get('label_bg_opacity', '0.5'), 0.5), 0.0, 1.0),
        label_bg_color=_parse_one_color(d.get('label_bg_color', '0,0,0'), (0, 0, 0)),
        halo_thickness=max(0.0, _to_float(d.get('halo_thickness', '1.0'), 1.0)),
        halo_color=_parse_one_color(d.get('halo_color', '0,0,0'), (0, 0, 0)),
        show_species=_to_bool(d.get('show_species', 'true'), True),
        show_age=_to_bool(d.get('show_age', 'true'), True),
        outer_extra=OUTER_EXTRA,
    )


def adaptive_font_thickness(style, box_w, box_h, display_scale=1.0,
                            line_mult=1.0, font_mult=1.0):
    """Return (font_scale, thickness) for a box of NATIVE size box_w x box_h.

    `box_w`/`box_h` must be native (pre-zoom) pixels; `display_scale` folds in any
    later magnification (annotation tool zoom * canvas-fit scale)."""
    if style.adaptive:
        ref = min(abs(box_w), abs(box_h))
        font_native = _clamp(style.font_coeff * ref, style.font_min, style.font_max)
        # Stays fractional here on purpose - rounding before applying display_scale
        # / line_mult would throw away every sub-pixel setting.
        thick_native = _clamp(style.thick_coeff * ref, style.thick_min, style.thick_max)
    else:
        font_native = style.base_font
        thick_native = style.base_thickness
    font_scale = max(0.15, font_native * display_scale * font_mult)
    # OpenCV cannot stroke thinner than one whole pixel, so 1 is the hard floor.
    thickness = max(1, int(round(thick_native * display_scale * line_mult)))
    return font_scale, thickness


def _fill_rect(img, x_a, y_a, x_b, y_b, style):
    """Draw the label background, clipped to the image, honouring label_bg_mode."""
    if style.label_bg_mode == 'none':
        return
    h, w = img.shape[:2]
    x_a, x_b = _clamp(x_a, 0, w), _clamp(x_b, 0, w)
    y_a, y_b = _clamp(y_a, 0, h), _clamp(y_b, 0, h)
    if x_b <= x_a or y_b <= y_a:
        return
    if style.label_bg_mode == 'solid':
        cv2.rectangle(img, (x_a, y_a), (x_b, y_b), style.label_bg_color, -1)
        return
    # translucent: blend only the band's ROI, so cost stays proportional to the
    # label rather than the whole frame even with many detections.
    roi = img[y_a:y_b, x_a:x_b]
    overlay = np.full(roi.shape, style.label_bg_color, dtype=roi.dtype)
    img[y_a:y_b, x_a:x_b] = cv2.addWeighted(
        overlay, style.label_bg_opacity, roi, 1.0 - style.label_bg_opacity, 0.0)


def draw_box(img, p1, p2, color, thickness, style):
    """Rectangle with a thin dark halo underneath, so any colour stays readable on
    grass/soil/shadow. halo_thickness = 0 disables the halo.

    halo_thickness is in pixels per side and may be fractional. A value too small to
    add a whole pixel draws no halo at all, rather than being rounded up: that keeps
    the setting honest as a way to make the line as thin as possible (a halo always
    makes the box read thicker, since it adds width on both sides)."""
    if style.halo_thickness > 0:
        halo_th = int(round(thickness + 2 * style.halo_thickness))
        if halo_th > thickness:
            cv2.rectangle(img, p1, p2, style.halo_color, halo_th)
    cv2.rectangle(img, p1, p2, color, thickness)


def draw_stacked_label(img, lines, lx, ly, color, font_scale, thickness, style,
                       bounds=None):
    """Draw `lines` stacked above (lx, ly), sharing one background band.

    The band is kept inside the image (and inside `bounds`, when given, so the
    annotation tool's labels don't bleed into the zoom column)."""
    lines = [l for l in lines if l]
    if not lines:
        return
    # Band geometry follows the *text* stroke, not the box line, so a thick border
    # doesn't inflate the label's padding.
    th = _text_thickness(thickness)
    sizes = [cv2.getTextSize(l, FONT, font_scale, th)[0] for l in lines]
    total_h = sum(h for _w, h in sizes) + th * 4 * len(lines)
    max_w = max(w for w, _h in sizes)

    # Keep the band on-screen: push down when it would run off the top, and
    # shift sideways when it would run past the left/right edge.
    if ly < total_h:
        ly = total_h
    x_a = lx - th
    x_b = lx + max_w + th * 2
    limit_w = bounds[0] if bounds else img.shape[1]
    if x_b > limit_w:
        shift = x_b - limit_w
        x_a -= shift
        x_b -= shift
    if x_a < 0:
        x_b -= x_a
        x_a = 0
    text_x = x_a + th

    _fill_rect(img, x_a, ly - total_h, x_b, ly, style)
    y = ly - th * 2
    for line, (_w, h) in zip(reversed(lines), reversed(sizes)):
        cv2.putText(img, line, (text_x, y), FONT, font_scale, color, th, cv2.LINE_AA)
        y -= h + th * 4


def draw_labeled_detection(img, x1, y1, x2, y2, box_w, box_h, style, *,
                           display_scale=1.0, line_mult=1.0, font_mult=1.0,
                           primary_color=(255, 255, 255), secondary_color=None,
                           top_lines=(), label='', hierarchical=False,
                           pending=False, selected=False, bounds=None):
    """Draw one detection: box(es) + stacked label. Single entry point shared by
    the output-video renderer and the annotation tool.

    x1..y2 are drawing coordinates (already mapped to screen space by the caller);
    box_w/box_h are the NATIVE box size that drives adaptive sizing.
    """
    font_scale, th = adaptive_font_thickness(
        style, box_w, box_h, display_scale, line_mult, font_mult)

    if pending:
        draw_box(img, (x1, y1), (x2, y2), PENDING_COLOR, th, style)
        cv2.putText(img, PENDING_TEXT, (x1, max(y1 - 6, 10)), FONT, font_scale,
                    PENDING_COLOR, _text_thickness(th), cv2.LINE_AA)
        pad = 3
    elif hierarchical and secondary_color is not None:
        outer = max(1, th + style.outer_extra)
        draw_box(img, (x1 - outer, y1 - outer), (x2 + outer, y2 + outer),
                 primary_color, outer, style)
        draw_box(img, (x1, y1), (x2, y2), secondary_color, th, style)
        draw_stacked_label(img, [*top_lines, label], x1, y1, primary_color,
                           font_scale, th, style, bounds)
        pad = outer + 2
    else:
        draw_box(img, (x1, y1), (x2, y2), primary_color, th, style)
        draw_stacked_label(img, [*top_lines, label], x1, y1, primary_color,
                           font_scale, th, style, bounds)
        pad = 3

    if selected:
        cv2.rectangle(img, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                      SELECTED_COLOR, 1)


def draw_frame_number(img, text, style, color=(255, 255, 255)):
    """Frame counter in the top-left corner. Deliberately not box-adaptive - it is
    frame furniture, not a detection - but it honours label_bg_mode."""
    th = max(1, style.base_thickness)
    fs = style.base_font
    (label_w, label_h), _ = cv2.getTextSize(text, FONT, fs, th)
    _fill_rect(img, 0, 0, label_w + th * 4, label_h + th * 4, style)
    cv2.putText(img, text, (th * 2, label_h + th * 2), FONT, fs, color, th,
                cv2.LINE_AA)
