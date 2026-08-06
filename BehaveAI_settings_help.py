#!/usr/bin/env python3
"""
Shared help text, tooltips and theming for the BehaveAI GUIs.

This module is imported by both BehaveAI.py (launcher) and
BehaveAI_settings_gui.py (settings editor). It is purely presentational:
it adds explanations, hover tooltips and a consistent ttk theme. It does
NOT read or write any .ini key and changes no application behaviour.

Public API
----------
- PARAM_HELP : dict[str, dict] mapping a stable slug (~ the .ini key) to
      {"short": one-line hint, "what": role, "influence": effect on results}.
- BUTTON_HELP : dict[str, str] tooltip text for the launcher action buttons,
      keyed by the script filename.
- Tooltip(widget, text, ...) : lightweight hover tooltip (pure tkinter).
- apply_theme(root) : apply the shared ttk theme + named styles.
- tooltip_text(key) : build the full "what + influence" tooltip string.
- help_label(parent, text, key=None) : ttk.Label with a trailing 'ⓘ' glyph and
      a tooltip when the slug has help. Caller still grids/packs it.
- help_line(parent, key) : grey italic one-line hint (style 'Help.TLabel').
"""
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

INFO_GLYPH = "ⓘ"  # circled small letter i: ⓘ

# ----------------------------------------------------------------------------
# Tooltip
# ----------------------------------------------------------------------------

class Tooltip:
    """Show a small borderless popup with ``text`` while the pointer rests on
    ``widget``. Pure tkinter, no third-party dependency."""

    def __init__(self, widget, text, delay=450, wraplength=380):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id = None
        self._tip = None
        if not text:
            return
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 16
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        frame = tk.Frame(tw, background="#2b2b2b", borderwidth=1, relief="solid")
        frame.pack()
        label = tk.Label(
            frame, text=self.text, justify="left",
            background="#2b2b2b", foreground="#f0f0f0",
            wraplength=self.wraplength, padx=8, pady=6,
            font=("TkDefaultFont", 9),
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------

def apply_theme(root):
    """Apply the shared ttk theme and named styles. Safe to call once per
    top-level window. Returns the ttk.Style instance."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    base = tkfont.nametofont("TkDefaultFont")
    family = base.actual("family")
    size = base.actual("size")
    if size <= 0:
        size = 9

    help_font = (family, max(size - 1, 8), "italic")
    section_font = (family, size + 2, "bold")

    # Neutral palette — function first, no loud colours.
    bg = "#f4f5f7"
    accent = "#2f6fb3"

    try:
        root.configure(background=bg)
    except Exception:
        pass

    style.configure(".", background=bg)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg)
    style.configure("TLabelframe", background=bg)
    style.configure("TLabelframe.Label", background=bg, font=(family, size, "bold"))
    style.configure("TCheckbutton", background=bg)
    style.configure("TNotebook", background=bg)

    style.configure("Help.TLabel", foreground="#666666", font=help_font, background=bg)
    style.configure("Section.TLabel", font=section_font, background=bg)
    style.configure("Hint.TLabel", foreground="#555555", background=bg)
    style.configure("Info.TLabel", foreground=accent, background=bg)
    style.configure("Accent.TButton", font=(family, size, "bold"))

    return style


# ----------------------------------------------------------------------------
# Help-rendering helpers
# ----------------------------------------------------------------------------

def tooltip_text(key):
    """Build the full hover text (role + effect) for a slug."""
    entry = PARAM_HELP.get(key)
    if not entry:
        return ""
    parts = []
    what = entry.get("what")
    influence = entry.get("influence")
    if what:
        parts.append("What it does:\n" + what)
    if influence:
        parts.append("Effect on results:\n" + influence)
    return "\n\n".join(parts)


def help_label(parent, text, key=None):
    """Create (but do not place) a ttk.Label for a field caption. When ``key``
    has help, a trailing ⓘ glyph is appended and a hover tooltip is attached so
    novices see what the setting does and how it influences results."""
    has_help = bool(key) and key in PARAM_HELP
    caption = f"{text}  {INFO_GLYPH}" if has_help else text
    lbl = ttk.Label(parent, text=caption)
    if has_help:
        Tooltip(lbl, tooltip_text(key))
    return lbl


def help_line(parent, key, wraplength=560):
    """Create (but do not place) the grey one-line hint for a slug."""
    short = PARAM_HELP.get(key, {}).get("short", "")
    return ttk.Label(parent, text=short, style="Help.TLabel",
                     wraplength=wraplength, justify="left")


# ----------------------------------------------------------------------------
# Parameter help content
#
# Sourced from BehaveAI_features_and_settings.md (§11 parameter table, §2
# settings GUI, §14 methods summary). "short" is the always-visible grey line;
# "what" + "influence" are shown on hover over the ⓘ icon.
# ----------------------------------------------------------------------------

PARAM_HELP = {
    # ---------------- Model structure ----------------
    "primary_motion_classes": {
        "short": "Top-level behaviours detected on the motion (movement) image stream.",
        "what": "The list of primary classes the motion detector is trained to find "
                "(e.g. walk, run, graze). Each needs a name, a colour and a single-character hotkey.",
        "influence": "These become the classes of the primary motion YOLO model. Adding/removing "
                     "them changes the model structure and makes existing motion annotations and "
                     "trained models incompatible (a rebuild/retrain is required).",
    },
    "secondary_motion_classes": {
        "short": "Sub-classes refining each primary motion detection (hierarchical step).",
        "what": "Optional second-stage classes applied inside a primary motion box "
                "(e.g. a 'run' box further classified as fast/slow).",
        "influence": "Trains a secondary classifier. If used, you need at least 2 secondary "
                     "motion classes. Leaving empty disables the secondary motion stage.",
    },
    "primary_static_classes": {
        "short": "Top-level classes detected on the static (single-frame) image stream.",
        "what": "The list of primary classes the static detector is trained to find "
                "(postures/objects visible without motion encoding).",
        "influence": "Defines the primary static YOLO model. Changing the list alters the model "
                     "structure and invalidates existing static annotations/models.",
    },
    "secondary_static_classes": {
        "short": "Sub-classes refining each primary static detection (hierarchical step).",
        "what": "Optional second-stage classes applied inside a primary static box.",
        "influence": "Trains a secondary static classifier. If used, at least 2 secondary static "
                     "classes are required. Empty disables the secondary static stage.",
    },
    "class_colors": {
        "short": "Box/label colour for this class in annotation and output videos.",
        "what": "Purely a display colour (RGB) for the class.",
        "influence": "Visual only — does not affect detection or training. Pick distinct colours "
                     "so overlapping classes stay readable.",
    },
    "class_hotkeys": {
        "short": "Single-character keyboard shortcut to assign this class while annotating.",
        "what": "The key pressed in the annotation tool to label a box with this class.",
        "influence": "Must be a unique single character; 'u' and 'g' are reserved (undo / grey-out). "
                     "Speeds up annotation; no effect on the model.",
    },
    "ignore_secondary": {
        "short": "Legacy key. Superseded by the secondary map — not a setting any more.",
        "what": "In the old flat schema it listed the primary classes exempt from secondary "
                "classification. Under the shared-pool schema the exemption is DERIVED: a primary "
                "with no secondary mapped to it on the 'Model structure' tab is skipped "
                "automatically. The key is only still read when a project has no secondary pool at "
                "all, i.e. when the legacy fallback in behaveai_config.py kicks in.",
        "influence": "Editing it does nothing in a current project. To exempt a primary class, "
                     "clear its secondaries in the secondary-map editor instead.",
    },
    "motion_blocks_static": {
        "short": "Grey out motion boxes when generating the static annotation images.",
        "what": "When on, regions covered by motion boxes are greyed in the static stream so the "
                "static model does not learn from moving subjects.",
        "influence": "Affects how annotation images are built; changing it after annotating "
                     "triggers a recommended dataset rebuild.",
    },
    "static_blocks_motion": {
        "short": "Grey out static boxes when generating the motion annotation images.",
        "what": "When on, regions covered by static boxes are greyed in the motion stream.",
        "influence": "Affects how annotation images are built; changing it after annotating "
                     "triggers a recommended dataset rebuild.",
    },
    "dominant_source": {
        "short": "Which stream wins when motion and static disagree on a detection.",
        "what": "Selects the arbiter when a target is seen by both streams: confidence (highest "
                "score wins), motion, or static.",
        "influence": "Determines the final reported class on overlaps. 'confidence' is the "
                     "balanced default; force 'motion'/'static' if one stream is more reliable.",
    },

    # ---------------- Video paths ----------------
    "clips_dir": {
        "short": "Folder of training video clips used for annotation.",
        "what": "Where the annotation/augmentation tools look for source videos.",
        "influence": "Only the videos in this folder can be annotated and counted in project stats.",
    },
    "input_dir": {
        "short": "Folder of videos to process in batch inference.",
        "what": "Source folder for 'Train & batch classify' batch runs.",
        "influence": "Every video here is classified and tracked when you run batch inference.",
    },
    "output_dir": {
        "short": "Folder where batch results (videos, CSVs) are written.",
        "what": "Destination for annotated videos, tracking CSVs and downstream analyses.",
        "influence": "All pipeline outputs land here; ensure it has enough disk space.",
    },

    # ---------------- Data augmentation ----------------
    "aug_global_probability": {
        "short": "Master gate: chance any augmentation runs at all (0 disables augmentation).",
        "what": "Top-level probability applied before any per-transform probability.",
        "influence": "0 turns augmentation off entirely. Higher values expand the dataset with "
                     "synthetic variations — more robustness, but too much can distort the data.",
    },
    "aug_target_classes": {
        "short": "Restrict augmentation to these classes (empty = augment all).",
        "what": "Comma-separated class names; only frames containing them are augmented.",
        "influence": "Use it to boost under-represented classes without inflating common ones.",
    },
    "aug_brightness": {
        "short": "Random brightness scaling (range) applied with its own probability.",
        "what": "Multiplies image brightness by a factor sampled in the range.",
        "influence": "Helps the model cope with varying light. Range is min,max; probability is the "
                     "per-image chance to apply it.",
    },
    "aug_contrast": {
        "short": "Random contrast scaling (range) applied with its own probability.",
        "what": "Multiplies image contrast by a sampled factor.",
        "influence": "Improves robustness to exposure differences across cameras/conditions.",
    },
    "aug_saturation": {
        "short": "Random colour-saturation scaling (range) with its own probability.",
        "what": "Scales colour intensity by a sampled factor.",
        "influence": "Guards against camera colour differences; extreme values can hurt colour-"
                     "dependent motion cues.",
    },
    "aug_hue": {
        "short": "Random hue shift in degrees (range) with its own probability.",
        "what": "Rotates colours on the hue wheel by a sampled amount.",
        "influence": "Small shifts add colour robustness; large shifts can break the green-tail "
                     "motion encoding — keep modest.",
    },
    "aug_sharpness": {
        "short": "Random sharpness scaling (range) with its own probability.",
        "what": "Sharpens or softens the image by a sampled factor.",
        "influence": "Simulates focus/optics variation; helps generalisation.",
    },
    "aug_blur": {
        "short": "Random Gaussian blur radius (range, forced odd) with its own probability.",
        "what": "Applies a Gaussian blur of a sampled radius.",
        "influence": "Mimics motion blur / low quality footage; too much erases fine detail.",
    },
    "aug_noise": {
        "short": "Random additive noise magnitude (±range) with its own probability.",
        "what": "Adds pixel noise of a sampled magnitude.",
        "influence": "Improves robustness to grainy/low-light video; excessive noise degrades learning.",
    },
    "aug_shear": {
        "short": "Random horizontal shear factor (range) with its own probability.",
        "what": "Geometrically shears the image by a sampled factor.",
        "influence": "Adds viewpoint variation; boxes are transformed with the image.",
    },
    "aug_temperature": {
        "short": "Random colour-temperature shift (+red/-blue, range) with its own probability.",
        "what": "Warms or cools the image by a sampled amount.",
        "influence": "Simulates white-balance differences between cameras/times of day.",
    },
    "aug_flip_h": {
        "short": "Horizontal flip — possible values (e.g. True,False) and its probability.",
        "what": "Randomly mirrors the image left-right.",
        "influence": "Cheap, usually safe augmentation. Avoid if left/right matters to a behaviour.",
    },
    "aug_flip_v": {
        "short": "Vertical flip — possible values and its probability.",
        "what": "Randomly mirrors the image top-bottom.",
        "influence": "Rarely realistic for ground-level footage; sensible mainly for top-down "
                     "(drone) views.",
    },

    # ---------------- Motion strategy ----------------
    "strategy": {
        "short": "How motion is encoded from consecutive frames.",
        "what": "exponential = colour 'tail' fades with a decay rate; sequential = fixed-window "
                "frame stacking.",
        "influence": "Sets the visual appearance of the motion image the model sees. 'exponential' "
                     "is the default and pairs with expA/expB.",
    },
    "chromatic_tail_only": {
        "short": "Encode motion as the difference between the two decay rates only.",
        "what": "Uses differential colour encoding instead of absolute decay values.",
        "influence": "Emphasises direction/speed via colour difference; changes the motion image, "
                     "so a dataset rebuild is recommended after toggling.",
    },
    "expA": {
        "short": "Green-channel decay rate; also sets the motion frame-window size.",
        "what": "Decay of the green motion tail. It also maps to the window length "
                "(0.2->5, >0.5->10, >0.7->15, >0.8->20, >0.9->45 frames).",
        "influence": "Higher = longer memory / longer trails and bigger temporal window. Strongly "
                     "shapes the motion image; rebuild the dataset after changing.",
    },
    "expB": {
        "short": "Red-channel decay rate (typically a longer memory than green).",
        "what": "Decay of the red motion tail.",
        "influence": "Combined with expA it controls the two-colour motion trail. Larger gap "
                     "between expA and expB makes direction/speed more visible.",
    },
    "lum_weight": {
        "short": "Blend between pure colour motion (0) and greyscale luminance (1).",
        "what": "Weight of the greyscale luminance blended into the motion image.",
        "influence": "0 = pure colour encoding, 1 = pure grey. Mid values keep colour cues while "
                     "stabilising on textured backgrounds.",
    },
    "rgb_multipliers": {
        "short": "Per-channel amplifiers (r,g,b) for the motion signal.",
        "what": "Multiplies each motion colour channel before display/training.",
        "influence": "Boosts faint motion so it is visible; too high saturates and loses detail. "
                     "Default 4,4,4.",
    },
    "frame_skip": {
        "short": "Raw frames skipped between the frames actually sampled.",
        "what": "Subsamples the video before motion encoding.",
        "influence": "Higher = faster processing and longer apparent motion per step, but coarser "
                     "temporal detail. 0 keeps every frame.",
    },
    "motion_threshold": {
        "short": "Pixel-difference threshold below which inter-frame motion is ignored.",
        "what": "Noise floor for the motion image: differences smaller than this are treated as static.",
        "influence": "Higher suppresses sensor noise / lighting flicker but can erase subtle real "
                     "motion; 0 keeps all detected differences.",
    },

    # ---------------- Model type ----------------
    "val_frequency": {
        "short": "Fraction of whole videos permanently held out from training.",
        "what": "Share of videos (never partial videos) assigned to a held-out set. Every model "
                "uses this one partition: the primary detectors' train/val split, the crop "
                "classifiers (secondary, species, age) and the complex-behaviour model's honest "
                "evaluation. Assignment is deterministic by video name: an existing video's "
                "status never changes, and new videos are classified automatically.",
        "influence": "Bigger holdout = more reliable, honest metrics but fewer training videos. "
                     "~0.1–0.2 is typical.",
    },
    "primary_classifier": {
        "short": "Base YOLO detector for the primary models (size n/s/m/l).",
        "what": "Pretrained weights used as the starting point (yolo8/11/26).",
        "influence": "n is fastest/lightest, l is most accurate/heaviest. Bigger models need more "
                     "VRAM and time but usually detect better.",
    },
    "primary_epochs": {
        "short": "Number of training epochs for the primary models.",
        "what": "How many passes over the training set when training a primary model.",
        "influence": "More epochs can improve accuracy up to a point, then overfit and waste time.",
    },
    "secondary_classifier": {
        "short": "Base YOLO classifier for the secondary (sub-class) models.",
        "what": "Pretrained classification weights for the secondary stage.",
        "influence": "Same size/speed/accuracy trade-off as the primary classifier.",
    },
    "secondary_epochs": {
        "short": "Number of training epochs for the secondary models.",
        "what": "Training passes for each secondary classifier.",
        "influence": "More epochs = potentially better sub-classification, with the usual overfit risk.",
    },
    "use_ncnn": {
        "short": "Export and run inference with the NCNN backend.",
        "what": "Converts models to NCNN format for inference.",
        "influence": "Can speed up CPU/edge inference; results may differ slightly from the native "
                     "backend. Leave off unless you need it.",
    },
    "primary_conf_thresh": {
        "short": "Minimum confidence to keep a primary detection (0–1).",
        "what": "Detections below this score are discarded.",
        "influence": "Lower = more detections but more false positives; higher = cleaner but may "
                     "miss faint subjects.",
    },
    "secondary_conf_thresh": {
        "short": "Minimum confidence to accept a secondary classification. 0 disables the floor.",
        "what": "The secondary classifier first arbitrates against an explicit '__none__' class: if "
                "'no secondary' wins, the box gets no sub-label. This threshold is a second, "
                "stricter floor applied to the winner — a sub-label that beats '__none__' but scores "
                "below it is still dropped as undecided. Set 0 to keep the '__none__' vote alone.",
        "influence": "Higher avoids wrong sub-labels at the cost of leaving more boxes unclassified. "
                     "Boxes left undecided keep their primary class; only the secondary column is "
                     "empty.",
    },

    # ---------------- Tracking ----------------
    "tracker_type": {
        "short": "Association method: botsort, bytetrack, or the legacy kalman.",
        "what": "How detections are linked into tracks across frames. botsort adds "
                "camera-motion compensation (GMC) and ByteTrack's two-tier matching; "
                "bytetrack is the same without GMC; kalman is the original homemade tracker.",
        "influence": "botsort/bytetrack are far more robust to drone pan and occlusion. "
                     "The parameter block below changes with the choice.",
    },
    "tracker_track_high_thresh": {
        "short": "Confidence above which a detection starts the 1st association tier.",
        "what": "High-confidence detections are matched to tracks first (ByteTrack tier 1).",
        "influence": "Lower to admit more detections to the strong first match; too low lets "
                     "weak boxes seed tracks.",
    },
    "tracker_track_low_thresh": {
        "short": "Floor for the 2nd association tier (rescues weak detections).",
        "what": "Detections between low and high thresh get a second matching pass against "
                "unmatched tracks -- ByteTrack's key idea for keeping small/occluded subjects.",
        "influence": "Lower keeps more faint detections alive across gaps; too low adds noise.",
    },
    "tracker_new_track_thresh": {
        "short": "Confidence required to spawn a brand-new track.",
        "what": "A detection that matches nothing only creates a new id above this score.",
        "influence": "Higher = fewer spurious tracks from false positives, at the cost of a "
                     "slower start for genuine faint subjects.",
    },
    "tracker_track_buffer": {
        "short": "Frames a lost track is kept before deletion.",
        "what": "How long an unmatched track survives (as 'lost') awaiting re-association.",
        "influence": "Kept short on purpose: long occlusion recovery is left to the offline "
                     "stitching pass, not this causal buffer.",
    },
    "tracker_match_thresh": {
        "short": "IoU threshold for the first association step.",
        "what": "Minimum overlap to link a high-confidence detection to a predicted track box.",
        "influence": "Higher demands tighter overlap (fewer wrong links, more fragmentation).",
    },
    "tracker_gmc_method": {
        "short": "Camera-motion compensation for BoT-SORT (sparseOptFlow/orb/ecc/none).",
        "what": "Global motion estimation that cancels drone pan/zoom inside the association "
                "loop, so track prediction is done in a stabilised frame.",
        "influence": "sparseOptFlow is a good default; 'none' disables compensation (worse on "
                     "moving-drone footage). Ignored by bytetrack/kalman.",
    },
    "match_distance_thresh": {
        "short": "Max pixel distance to link a detection to an existing track.",
        "what": "Association gate between a new detection and a tracked object.",
        "influence": "Too small drops fast movers (ID switches); too large links unrelated subjects. "
                     "Scale with resolution and subject speed.",
    },
    "delete_after_missed": {
        "short": "Consecutive missed frames before a track is dropped.",
        "what": "How long a track survives without a matching detection.",
        "influence": "Higher tolerates occlusions but risks stale ghost tracks; lower deletes "
                     "quickly but fragments identities.",
    },
    "centroid_merge_thresh": {
        "short": "Max centroid distance to merge a static + motion detection.",
        "what": "Distance under which the two streams' detections are fused into one object.",
        "influence": "Too large merges distinct neighbours; too small leaves duplicate boxes.",
    },
    "iou_thresh": {
        "short": "Box overlap (IoU) required to merge/deduplicate detections.",
        "what": "Intersection-over-Union threshold for overlap handling.",
        "influence": "Higher only merges near-identical boxes; lower merges loosely overlapping "
                     "ones (risking over-merging).",
    },
    "kalman_process_noise_pos": {
        "short": "Assumed positional uncertainty injected per frame (Kalman).",
        "what": "Process noise on position in the tracking filter.",
        "influence": "Higher lets tracks react faster to sudden moves but jitter more; lower gives "
                     "smoother but laggier tracks.",
    },
    "kalman_process_noise_vel": {
        "short": "Assumed velocity uncertainty injected per frame (Kalman).",
        "what": "Process noise on velocity in the tracking filter.",
        "influence": "Higher adapts quickly to acceleration changes; lower assumes steadier motion.",
    },
    "kalman_measurement_noise": {
        "short": "Assumed noise in each measured centroid (Kalman).",
        "what": "How much the filter trusts raw detections vs. its prediction.",
        "influence": "Higher = smoother, prediction-led tracks; lower = follows detections closely "
                     "(more jitter).",
    },

    # ---------------- Drone motion correction ----------------
    "drone_enabled": {
        "short": "Compensate for camera/drone motion before tracking analysis.",
        "what": "Estimates global background motion and subtracts it from the tracking CSV.",
        "influence": "Off = no change. On = positions/speeds are expressed relative to the ground, "
                     "essential for moving-camera (drone) footage.",
    },
    "drone_model": {
        "short": "Background-motion model: affine or homography.",
        "what": "Transform fitted to background features each frame.",
        "influence": "affine is robust and cheap; homography handles perspective/tilt but needs more "
                     "good features and can be unstable on sparse scenes.",
    },
    "drone_box_dilation": {
        "short": "How much to expand subject boxes when masking them out of the background.",
        "what": "Fraction of box size added before excluding subjects from feature tracking.",
        "influence": "Larger removes more subject pixels (cleaner background estimate) but leaves "
                     "fewer features near subjects.",
    },
    "drone_min_features": {
        "short": "Minimum background features required to trust the estimated transform.",
        "what": "Below this count the frame's transform is considered unreliable.",
        "influence": "Higher demands richer scenes; if unmet, the fallback smoothing may take over.",
    },
    "drone_uncertain_std": {
        "short": "Residual-flow std (px) above which a frame is flagged uncertain.",
        "what": "Threshold on leftover motion after correction.",
        "influence": "Lower flags more frames as uncertain (conservative); higher trusts the "
                     "correction more.",
    },
    "drone_smoothing": {
        "short": "Centroid smoothing before computing speeds: savgol / moving_average / none.",
        "what": "Filter applied to trajectories before differentiation.",
        "influence": "Smoothing reduces speed noise; 'none' keeps raw detail but noisier velocities.",
    },
    "drone_smoothing_window": {
        "short": "Window length (odd) of the smoothing filter.",
        "what": "Number of frames in the smoothing window.",
        "influence": "Larger = smoother trajectories but more lag and rounded turns.",
    },
    "drone_fallback_smoothing": {
        "short": "When features are too scarce, smooth only (skip optical-flow correction).",
        "what": "Fallback behaviour on feature-poor frames.",
        "influence": "On = degrade gracefully to smoothing-only; off = leave those frames uncorrected.",
    },

    "ab_min_classified": {
        "short": "Min classified frames for an ID to count as a group member (0 = skip).",
        "what": "Activity-budget filter on how much behaviour an ID must have to be counted.",
        "influence": "Higher removes fleeting/spurious IDs from group statistics; 0 disables the filter.",
    },
    "ab_analysis_duration_s": {
        "short": "Seconds of each video used for the activity budget (0 = whole video).",
        "what": "Truncates the analysis window to a fixed duration from the start of the clip.",
        "influence": "A fixed window makes budgets comparable across clips of different lengths; 0 uses "
                     "the entire video.",
    },

    # ---------------- Interaction ----------------
    "complex_max_dist": {
        "short": "Pairs farther apart than this on the ground (m) are not interacting.",
        "what": "Maximum real-world ground distance for a dyad to be considered an interaction. "
                "Needs metric geometry; in metres it means the same thing at any flight altitude, "
                "which a pixel threshold could not.",
        "influence": "Larger captures loose proximity events (more, weaker edges); smaller keeps only "
                     "close encounters.",
    },
    "complex_min_duration": {
        "short": "Minimum observed frames for a dyad/episode to become an edge.",
        "what": "Pairs seen for fewer frames than this are dropped, at every granularity.",
        "influence": "Higher removes brief incidental contacts; lower keeps short interactions.",
    },
    "complex_contact_iou": {
        "short": "Box overlap (IoU) above which two subjects count as in contact.",
        "what": "Overlap-based contact criterion.",
        "influence": "Higher requires strong overlap for 'contact'; lower flags grazing touches.",
    },
    "complex_contact_dist": {
        "short": "Ground distance (m) below which subjects count as in contact.",
        "what": "Proximity-based contact criterion, complementing IoU. Measured centre to centre "
                "in real metres, so ~2-3 m is roughly touching for an adult horse.",
        "influence": "Larger labels near-misses as contact; smaller demands true closeness.",
    },
    "complex_window": {
        "short": "Window length (frames) over which features are aggregated for the model.",
        "what": "Temporal window for computing interaction/behaviour features.",
        "influence": "Longer captures slower behaviours but blurs quick ones; shorter is the reverse.",
    },
    "interaction_granularity": {
        "short": "Edge granularity in the interaction graph.",
        "what": "per_dyad = ONE row per pair summarising the whole clip (an association summary, "
                "not an episode); per_segment = one row per continuous episode; per_frame = one row "
                "per observed frame.",
        "influence": "Finer granularity yields larger, more detailed edge files. Changing this "
                     "regenerates and overwrites the edges file.",
    },
    "interaction_weight": {
        "short": "How interaction edge weights are computed (all comparable across videos).",
        "what": "sri = simple ratio index, time together / time both were visible, bounded 0-1; "
                "duration_s = seconds together; proximity_m = 1 / mean ground distance.",
        "influence": "Chooses what 'strength' means in the social graph. All three are absolute "
                     "quantities, so weights can be compared between clips and between sites.",
    },

    # ---------------- Complex behaviours ----------------
    "complex_behaviours": {
        "short": "User-defined list of dyadic AND group behaviours to model.",
        "what": "Names (each with a unique single-char hotkey) for complex behaviours you annotate "
                "and train a model on.",
        "influence": "Defines the target classes of the complex-behaviour model. Hotkeys must be "
                     "unique and not reserved.",
    },
    "complex_model_type": {
        "short": "Model family for complex behaviours: baseline / lstm / transformer.",
        "what": "baseline = scikit-learn classifier on aggregated features; lstm/transformer model "
                "the temporal sequence (require torch, and error out if it is missing rather than "
                "silently training the baseline instead).",
        "influence": "Sequence models can capture temporal dynamics better but need torch and more "
                     "data/compute; baseline is the dependency-light default.",
    },
    "complex_baseline_clf": {
        "short": "Baseline classifier: random_forest or hist_gradient_boosting.",
        "what": "The scikit-learn algorithm used when model type = baseline.",
        "influence": "Different speed/accuracy trade-offs; both work on the aggregated feature window.",
    },
    "complex_seq_steps": {
        "short": "How many sub-windows a labelled segment is sliced into (lstm/transformer).",
        "what": "Sequence length fed to the deep model: each segment becomes this many "
                "per-timestep feature vectors.",
        "influence": "More steps give finer temporal resolution but shorter sub-windows (noisier "
                     "per-step features) and more compute. Ignored by the baseline.",
    },
    "complex_deep_epochs": {
        "short": "Training epochs for the deep (lstm/transformer) model.",
        "what": "Number of passes over the annotated sequences during training.",
        "influence": "More epochs fit harder (risk of over-fitting on small data); fewer under-fit. "
                     "Ignored by the baseline.",
    },
    "complex_deep_hidden": {
        "short": "Hidden size of the LSTM / Transformer (d_model).",
        "what": "Width of the recurrent/encoder representation.",
        "influence": "Larger has more capacity but needs more data/compute. Ignored by the baseline.",
    },
    "complex_deep_layers": {
        "short": "Number of stacked recurrent / encoder layers.",
        "what": "Depth of the deep sequence model.",
        "influence": "Deeper can model more complex dynamics but is harder to train on little data. "
                     "Ignored by the baseline.",
    },
    "complex_deep_heads": {
        "short": "Attention heads (transformer only).",
        "what": "Number of self-attention heads; the effective d_model is rounded to a multiple.",
        "influence": "More heads capture more relation types; must divide d_model. LSTM ignores this.",
    },
    "complex_deep_dropout": {
        "short": "Dropout used in the deep model.",
        "what": "Regularisation probability applied during training.",
        "influence": "Higher reduces over-fitting but can under-fit small data. Ignored by the baseline.",
    },
    "complex_deep_lr": {
        "short": "Adam learning rate for the deep model.",
        "what": "Optimiser step size during deep-model training.",
        "influence": "Too high diverges; too low trains slowly. Ignored by the baseline.",
    },
    "complex_deep_batch": {
        "short": "Mini-batch size for the deep model.",
        "what": "Number of sequences per optimiser step.",
        "influence": "Larger is faster/steadier but uses more memory. Ignored by the baseline.",
    },
    "complex_confusion_merge_rate": {
        "short": "Confusion rate above which a class pair is suggested for merging.",
        "what": "Threshold on true-vs-predicted confusion between two behaviour classes.",
        "influence": "Lower flags more pairs as too-similar (suggest merging labels); higher flags "
                     "fewer.",
    },
    "complex_predict_min_proba": {
        "short": "Minimum probability to emit a complex-behaviour prediction.",
        "what": "Confidence gate on model output.",
        "influence": "Higher yields fewer but more confident predictions; lower labels more windows, "
                     "with more errors.",
    },
    "complex_speed_low": {
        "short": "Ground speed (m/s) below which a subject counts as ~still.",
        "what": "Lower bound used by the candidate-proposal heuristics. A grazing horse moves at "
                "well under 0.5 m/s, so ~0.2 is a reasonable 'standing' bound.",
        "influence": "Tunes what 'stationary' means when surfacing candidate windows.",
    },
    "complex_speed_high": {
        "short": "Ground speed (m/s) above which motion counts as fast (gallop/chase).",
        "what": "Upper bound used by the candidate-proposal heuristics. A horse trots at roughly "
                "3-4 m/s and canters faster.",
        "influence": "Tunes what 'fast' means when surfacing candidate windows.",
    },
    "complex_polarisation_high": {
        "short": "Group alignment above this suggests trek/stampede.",
        "what": "Polarisation threshold (how aligned headings are) for candidate proposals.",
        "influence": "Higher requires stronger collective alignment to flag directed group motion.",
    },
    "complex_synchrony_high": {
        "short": "Behavioural synchrony above this suggests synchronised rest/graze.",
        "what": "Synchrony threshold for candidate proposals.",
        "influence": "Higher requires more coordinated behaviour to flag a synchronised event.",
    },
    "complex_candidate_topk": {
        "short": "Number of most-uncertain windows surfaced as labelling candidates.",
        "what": "Active-learning budget per run.",
        "influence": "Higher proposes more windows to annotate (more work, faster coverage); lower "
                     "focuses on the very most uncertain.",
    },

    # ---------------- Display ----------------
    "line_thickness": {
        "short": "Thickness of bounding boxes and text in output overlays.",
        "what": "Pixel width of drawn boxes/labels.",
        "influence": "Display only. Thicker is readable on high-resolution video but can hide small "
                     "subjects.",
    },
    "font_size": {
        "short": "Scale of label text in output overlays.",
        "what": "Text size multiplier for drawn labels.",
        "influence": "Display only. Larger is readable but clutters crowded scenes.",
    },
    "box_line_scale": {
        "short": "Extra thinning of annotation-tool box borders. Ignored unless adaptive box "
                 "scaling is off.",
        "what": "Multiplier on line_thickness for the on-screen boxes only (zoomed with the "
                "mouse wheel); does not affect line_thickness itself or saved crop masks. "
                "Legacy knob: when adaptive_box_scaling is on, the annotation tool renders "
                "boxes exactly like the output video and this value is not applied.",
        "influence": "Display only, annotation tool, adaptive scaling off. Lower values keep "
                     "borders thinner at any zoom level; 1.0 matches line_thickness.",
    },
    "box_font_scale": {
        "short": "Extra shrinking of annotation-tool label text. Ignored unless adaptive box "
                 "scaling is off.",
        "what": "Multiplier on font_size for the on-screen labels only (zoomed with the mouse "
                "wheel); does not affect font_size itself or saved crop masks. Legacy knob: "
                "when adaptive_box_scaling is on, the annotation tool renders labels exactly "
                "like the output video and this value is not applied.",
        "influence": "Display only, annotation tool, adaptive scaling off. Lower values keep "
                     "labels smaller at any zoom level; 1.0 matches font_size.",
    },
    "buttons_per_row": {
        "short": "Number of class buttons shown per row in the annotation window.",
        "what": "Grid width of the class button panel in the annotation/inspect tools.",
        "influence": "Display only. More columns fit more classes per row but make each button "
                     "narrower; fewer columns give wider buttons but a taller panel.",
    },

    # ---------------- Box & label style (shared: output videos + annotation) ----------------
    "adaptive_box_scaling": {
        "short": "Scale label text and box lines to each animal's box size.",
        "what": "When on, font and line thickness are derived from the bounding box's native "
                "size (clamped by the min/max settings) instead of the flat font_size / "
                "line_thickness values. Applies to output videos and the annotation tool alike.",
        "influence": "Display only. On, small animals get small thin labels and large ones stay "
                     "readable. Off restores the previous flat font_size/line_thickness look.",
    },
    "adaptive_font_coeff": {
        "short": "Font size per pixel of box size, when adaptive scaling is on.",
        "what": "font = coeff x min(box_width, box_height), then clamped to the adaptive font "
                "min/max.",
        "influence": "Display only. Higher gives bigger text on the same animal; ignored when "
                     "adaptive_box_scaling is off.",
    },
    "adaptive_font_min": {
        "short": "Smallest font size adaptive scaling may produce.",
        "what": "Lower clamp on the box-derived font size.",
        "influence": "Display only. Raise it if labels on distant/small animals get unreadable.",
    },
    "adaptive_font_max": {
        "short": "Largest font size adaptive scaling may produce.",
        "what": "Upper clamp on the box-derived font size.",
        "influence": "Display only. Lower it if labels on close-up animals get oversized.",
    },
    "adaptive_thickness_coeff": {
        "short": "Box line thickness per pixel of box size, when adaptive scaling is on.",
        "what": "thickness = coeff x min(box_width, box_height), then clamped to the adaptive "
                "thickness min/max.",
        "influence": "Display only. Higher gives thicker borders; ignored when "
                     "adaptive_box_scaling is off.",
    },
    "adaptive_thickness_min": {
        "short": "Thinnest box line adaptive scaling may produce (decimals allowed).",
        "what": "Lower clamp, in pixels, on the box-derived line thickness. Fractional values "
                "such as 0.75 are kept through the maths and only rounded when the line is "
                "actually drawn.",
        "influence": "Display only. Drawn lines can never be thinner than one whole pixel, so "
                     "values below 1 mainly bite in the annotation tool, where zoom multiplies "
                     "them before rounding.",
    },
    "adaptive_thickness_max": {
        "short": "Thickest box line adaptive scaling may produce (decimals allowed).",
        "what": "Upper clamp, in pixels, on the box-derived line thickness. Fractional values "
                "such as 2.25 are honoured before rounding.",
        "influence": "Display only. Lower it if borders on large animals look heavy; this is the "
                     "setting that caps how thick a close-up animal's box can get.",
    },
    "label_bg_mode": {
        "short": "Background band behind box labels: none, translucent or solid.",
        "what": "'none' draws text straight onto the image, 'translucent' blends a band using "
                "label_bg_opacity, 'solid' fills it opaquely (the previous behaviour).",
        "influence": "Display only. 'translucent' keeps labels readable while hiding less of "
                     "neighbouring boxes and animals; 'none' hides nothing but can be hard to "
                     "read on busy backgrounds.",
    },
    "label_bg_opacity": {
        "short": "Opacity of the label background band when its mode is 'translucent'.",
        "what": "0.0 is fully see-through, 1.0 is fully opaque.",
        "influence": "Display only. Lower obscures less of what's behind the label but reduces "
                     "text contrast. Ignored unless label_bg_mode is 'translucent'.",
    },
    "label_bg_color": {
        "short": "Colour of the label background band, as R,G,B.",
        "what": "Single 'R,G,B' triple used for the band behind label text.",
        "influence": "Display only. Dark values suit the bright class colours used for text.",
    },
    "halo_thickness": {
        "short": "Thin dark outline drawn under box lines for contrast; 0 disables it "
                 "(decimals allowed).",
        "what": "Width in pixels of the darker border drawn just under each coloured box line, "
                "on each side. Fractional values are allowed; a value too small to add a whole "
                "pixel draws no halo at all.",
        "influence": "Display only. The halo adds width on both sides, so it is what makes a box "
                     "read thicker: 1 makes any class colour readable over grass, soil or shadow, "
                     "while 0 gives the thinnest possible box (a bare one-pixel line).",
    },
    "halo_color": {
        "short": "Colour of the outline drawn under box lines, as R,G,B.",
        "what": "Single 'R,G,B' triple used for the halo under each box line.",
        "influence": "Display only. Dark values give the strongest contrast on typical field "
                     "footage.",
    },
    "show_species": {
        "short": "Show the species name on each box.",
        "what": "Toggles the species line above the box label, in output videos and the "
                "annotation tool. When the project defines a single species, that name is shown "
                "even though no species classifier is trained.",
        "influence": "Display only. Turn off to reduce clutter when every animal is the same "
                     "species.",
    },
    "show_age": {
        "short": "Show the age class on each box.",
        "what": "Toggles the age line above the box label, in output videos and the annotation "
                "tool. When the project defines a single age class, that name is shown even "
                "though no age classifier is trained.",
        "influence": "Display only. Turn off to reduce clutter.",
    },

    # ---------------- Activity budget ----------------
    "ab_min_presence_seconds": {
        "short": "Min seconds a subject must be tracked to count as a group member.",
        "what": "Stranger/visitor threshold for the activity budget, in absolute seconds "
                "(presence = tracked frames / fps), not a fraction of the clip.",
        "influence": "Higher excludes brief visitors from group stats; lower includes transient IDs. "
                     "A member that joins mid-clip still accrues plenty of seconds.",
    },
    "ab_edge_margin_px": {
        "short": "Isotropic pixel band at the frame edge used to report side-of-entry.",
        "what": "How close (in pixels, same on all four sides) a subject's first-appearance box must "
                "be to a frame edge to be reported as having entered from that side.",
        "influence": "Only explains a short-presence flag (which side it entered from); it never makes "
                     "a subject a stranger on its own.",
    },
    "ab_group_type_separator": {
        "short": "Character separating fields in clip filenames.",
        "what": "Delimiter used to parse metadata encoded in filenames.",
        "influence": "Must match your naming convention or the group-type field cannot be read.",
    },
    "ab_group_type_field_index": {
        "short": "Zero-based filename field index that encodes the group type.",
        "what": "Which separated field of the filename holds the group type.",
        "influence": "Wrong index reads the wrong token; align it with your filename convention.",
    },

    # --- Offline tracklet stitching -----------------------------------------
    "stitch_enabled": {
        "short": "Re-link the tracker's tracklets into longer identities, offline.",
        "what": "Runs a non-causal pass over the whole clip after drone correction, joining "
                "tracklets that the causal tracker cut. Kinematics only — no appearance. A link "
                "is taken only when 'this is the same animal' is more likely than 'a new animal "
                "appeared here'.",
        "influence": "Fewer, longer identities, which is what per-individual budgets need. But a "
                     "wrong merge invents an animal and corrupts two budgets at once, so run "
                     "BehaveAI_stitch_oracle.py on your clips first: on a tightly packed herd the "
                     "contamination rate is high at every setting.",
    },
    "stitch_max_speed_m_per_s": {
        "short": "Physical speed an animal cannot exceed (m/s).",
        "what": "Hard gate: a displacement implying more than this cannot be the same animal. Used "
                "when a flight log gives the pixel scale; otherwise the pixel cap applies.",
        "influence": "Generous on purpose — it only has to exclude the impossible. The likelihood "
                     "ratio does the actual discrimination, so tightening this is not how you make "
                     "linking stricter.",
    },
    "stitch_max_speed_px_per_frame": {
        "short": "Fallback speed gate in pixels per frame, used without a flight log.",
        "what": "Same hard gate expressed in stabilised-frame pixels, for clips with no telemetry.",
        "influence": "Only valid at roughly constant altitude. Too low silently forbids real links.",
    },
    "stitch_speed_gate_margin": {
        "short": "Safety factor on the m/s → pixel conversion.",
        "what": "Covers the known bias of barometric rel_alt, which is referenced to the take-off "
                "point, so sloped terrain or a raised take-off spot mis-states the camera height.",
        "influence": "Larger is safer against wrongly rejecting real links; it does not make wrong "
                     "links more likely, since the gate is not what selects them.",
    },
    "stitch_max_gap_seconds": {
        "short": "Longest occlusion the stitcher will try to bridge.",
        "what": "Pairs separated by more than this are not even considered.",
        "influence": "The setting to derive from the oracle benchmark, not from taste: pick the "
                     "largest gap where recovery is still useful and contamination still low. "
                     "Longer gaps buy a few more links at a fast-rising error rate.",
    },
    "stitch_gap_prior_seconds": {
        "short": "Time constant of the prior on occlusion duration.",
        "what": "Sets how quickly a long gap is penalised: the cost carries a gap/tau term.",
        "influence": "Shorter makes long-gap links progressively harder. It shapes the penalty "
                     "smoothly, where max_gap_seconds cuts it off abruptly.",
    },
    "stitch_extrapolation_horizon_seconds": {
        "short": "How far a tracklet's velocity is trusted to predict forward.",
        "what": "Constant-velocity extrapolation is damped past this horizon, because beyond about "
                "a second an animal's instantaneous velocity no longer says where it went.",
        "influence": "Too long produces absurd predictions far outside the frame; too short throws "
                     "away real information about a moving animal.",
    },
    "stitch_link_prior_log_odds": {
        "short": "Deliberate bias on the link decision. Negative demands more evidence.",
        "what": "Added to the log-likelihood ratio. 0 = pure likelihood ratio; negative values make "
                "the stitcher conservative.",
        "influence": "The main tuning knob, and it is asymmetric by design: a missed link only "
                     "costs statistical power, a wrong merge costs validity. On the HERDWISE clips, "
                     "0 gave 90 % recovery at 3.0 % contamination and -5 gave 70 % at 1.7 %.",
    },
    "stitch_min_tracklet_len": {
        "short": "Tracklets shorter than this many samples are excluded from linking.",
        "what": "Very short tracklets carry no usable velocity. They are excluded from the linking "
                "problem but their rows are kept and given their own identity.",
        "influence": "Raising it removes noise from the linking problem; it never deletes data.",
    },
    "stitch_quality_gate": {
        "short": "Refuse links whose endpoint positions are flagged unreliable.",
        "what": "Skips pairs where either endpoint has correction_quality 'none', i.e. the drone "
                "correction could not place it in the stabilised frame.",
        "influence": "On by default. Turning it off lets the stitcher link on positions that are "
                     "known to be wrong.",
    },
    "expected_group_size": {
        "short": "Field-recorded herd size. A diagnostic only, never a constraint.",
        "what": "Written into the stitch report so you can compare it with the number of "
                "simultaneously tracked animals.",
        "influence": "Changes no decision anywhere in the pipeline. If the observed maximum exceeds "
                     "it, that is evidence about the count, the herd or the detector — the "
                     "stitcher will not force the result towards it.",
    },

    # ---------------- Training controls ----------------
    "train_patience": {
        "short": "Stop training after this many epochs with no validation improvement.",
        "what": "Early stopping. 'Primary epochs' / 'Secondary epochs' are only the CAP: training "
                "ends as soon as the validation metric has not improved for this many epochs. "
                "Ultralytics' own default is 100, which effectively never triggers on short runs, "
                "so BehaveAI sets 30.",
        "influence": "Lower ends runs sooner and wastes less GPU time, but can cut a model off "
                     "during a plateau it would have escaped. Higher trains longer for a possibly "
                     "marginal gain. It never changes the best weights that are kept.",
    },
    "motion_disable_color_aug": {
        "short": "Train the motion stream with YOLO's HSV colour augmentation switched off.",
        "what": "The motion stream is a FALSE-COLOUR encoding: the hue/saturation of a pixel carries "
                "the motion signal, not the animal's appearance. Ultralytics' online HSV jitter "
                "would scramble exactly that signal.",
        "influence": "Leave it on (the default) for any false-colour motion encoding. Turning it "
                     "off re-enables HSV jitter and generally degrades the motion detector. The "
                     "static stream is never affected.",
    },
    "save_empty_frames": {
        "short": "Keep annotation frames that contain no box at all (background negatives).",
        "what": "When a sampled frame ends up with zero boxes, this decides whether the image (plus "
                "an empty label file) is still written to annot_static/annot_motion. Those frames "
                "become pure-background negatives for the detector.",
        "influence": "True gives the detector explicit background examples, which usually reduces "
                     "false positives on empty terrain, at the cost of a larger dataset and a class "
                     "balance dominated by empty images. False keeps the dataset compact and "
                     "positive-only. Changing it only affects frames written from now on.",
    },

    # ---------------- End-of-pipeline auto-run switches ----------------
    "interaction_graph_enabled": {
        "short": "Build the interaction features + graph automatically after tracking.",
        "what": "Runs the dyadic/group feature extraction and writes the edges/nodes CSVs at the end "
                "of the pipeline, before the activity budget.",
        "influence": "Requires metric geometry: videos without it are skipped. Off means no "
                     "interaction outputs unless you run the step by hand; nothing else changes.",
    },
    "complex_classify_enabled": {
        "short": "Run complex-behaviour classification automatically after the interaction graph.",
        "what": "Applies the trained complex-behaviour model to the interaction features and writes "
                "its predictions before the activity budget runs.",
        "influence": "Off skips the step entirely (useful while no complex model is trained yet). "
                     "It never affects the primary/secondary behaviour columns.",
    },

    # ---------------- Metric geometry ----------------
    "metric_enabled": {
        "short": "Convert tracked pixel positions into real ground-plane metres.",
        "what": "Runs after drone correction and appends X_m, Y_m (camera-relative, per frame), "
                "Xs_m, Ys_m (stabilised) and metric_quality to each tracking CSV. The chain is "
                "metres/pixel = f(height, pitch, focal length, image row), with height and pitch "
                "read from the clip's flight log.",
        "influence": "Required by everything expressed in metres: interaction distances, speeds in "
                     "m/s, the stitcher's metric speed gate. Clips with no flight log get "
                     "metric_quality='none' and keep pixel coordinates only. Off changes nothing "
                     "else in the pipeline.",
    },
    "metric_focal_len_mm": {
        "short": "Camera focal length in mm, from the drone's spec sheet.",
        "what": "Used with the sensor width and the frame width to derive the pixel focal length "
                "f_px = focal_len_mm * frame_width / sensor_width_mm. A per-drone checkerboard "
                "calibration (metric_fpx_<DroneToken> in the INI) overrides it when present.",
        "influence": "Scales every distance linearly: a focal length 10 % too small makes all "
                     "measured distances 10 % too large. Use the value for the drone that actually "
                     "shot the clips, not a mixed-fleet average.",
    },
    "metric_sensor_width_mm": {
        "short": "Physical sensor width in mm, from the drone's spec sheet.",
        "what": "The other half of the pixel-focal-length formula. For a 1/1.3\" class drone sensor "
                "this is a few mm, NOT the 36 mm of a full-frame camera.",
        "influence": "Same linear effect as the focal length, in the opposite direction. Leaving "
                     "the 36 mm placeholder with a real drone focal length gives distances that are "
                     "wrong by roughly an order of magnitude.",
    },
    "metric_roll_max_deg": {
        "short": "Frames rolled more than this are flagged as unreliable.",
        "what": "The projection assumes zero roll. gb_roll is checked per frame against this limit "
                "and offending frames get metric_quality='uncertain'.",
        "influence": "Tighter flags more frames but keeps only geometry you can defend; looser "
                     "keeps more rows at the cost of letting tilted frames through.",
    },
    "metric_horizon_margin_px": {
        "short": "Reject detections closer than this many pixels to the horizon line.",
        "what": "Near the horizon one pixel maps to a huge and rapidly growing ground distance, so "
                "positions there are numerically meaningless.",
        "influence": "Larger is safer and discards more far-field animals; smaller keeps distant "
                     "animals whose metric positions may be wildly wrong. It flags rows, it does "
                     "not delete them.",
    },
    "metric_assumed_body_length_m": {
        "short": "Typical body length used to sanity-check the metric scale.",
        "what": "The apparent size of the animals gives a second, independent estimate of the "
                "camera height. It is compared against the flight-log height to catch a wrong "
                "focal length or sensor width before the numbers reach the analysis.",
        "influence": "Only affects the plausibility check and its warning, never the coordinates. "
                     "Set it to the species you actually filmed (≈2.2 m for an adult horse).",
    },
    "metric_scale_tolerance": {
        "short": "How far the two camera-height estimates may disagree before a warning.",
        "what": "0.25 means the apparent-size estimate and the flight-log height must agree within "
                "a factor 1.25 either way for the calibration to be called plausible.",
        "influence": "Tighter catches calibration errors earlier but cries wolf on mixed-age herds; "
                     "looser lets a genuinely wrong scale pass unnoticed.",
    },

    # ---------------- SAHI sliced inference ----------------
    "sahi_enabled_static": {
        "short": "Tile the static stream so small animals are seen at native resolution.",
        "what": "Instead of shrinking the whole 4K frame to 640 px, the frame is diced into "
                "native-resolution tiles and each is detected separately. A ~60 px horse reaches "
                "the model at ~60 px instead of ~10 px. Training follows: the stream is retrained "
                "on the sliced dataset into a separate *_tiled project, so the whole-frame model is "
                "never overwritten.",
        "influence": "Big recall gain on small/distant animals; 20-40x slower inference. Auto-skips "
                     "when the frame is not meaningfully larger than a tile. Turning it back off "
                     "instantly restores the whole-frame model.",
    },
    "sahi_enabled_motion": {
        "short": "Same tiled inference for the motion stream.",
        "what": "Independent of the static switch, because the two streams do not have the same "
                "small-object problem: motion blobs are often already large.",
        "influence": "Same recall/speed trade-off as the static switch, applied to the motion "
                     "detector and its own tiled model.",
    },
    "sahi_slice_height": {
        "short": "Tile height in pixels.",
        "what": "Height of each slice fed to the detector. 640 matches the model's native input, so "
                "no rescaling happens inside a tile.",
        "influence": "Smaller tiles magnify small animals further but multiply the tile count (and "
                     "the runtime) and cut more animals across tile borders.",
    },
    "sahi_slice_width": {
        "short": "Tile width in pixels.",
        "what": "Width of each slice. Keep it equal to the height unless the model was trained on "
                "a non-square input.",
        "influence": "Same trade-off as the tile height.",
    },
    "sahi_overlap_height_ratio": {
        "short": "Vertical overlap between neighbouring tiles, as a fraction of tile height.",
        "what": "Overlap is what stops an animal sitting exactly on a tile border from being cut in "
                "two and detected as nothing.",
        "influence": "Should exceed the apparent size of your animals relative to the tile. Higher "
                     "is safer and slower; 0 will lose animals on the seams.",
    },
    "sahi_overlap_width_ratio": {
        "short": "Horizontal overlap between neighbouring tiles.",
        "what": "Same role as the vertical overlap, along the other axis.",
        "influence": "Same trade-off: more overlap, more tiles, more runtime, fewer split animals.",
    },
    "sahi_postprocess_type": {
        "short": "How duplicate detections from overlapping tiles are merged.",
        "what": "NMS keeps the best box of a duplicate group; GREEDYNMM/NMM merge them into one box "
                "instead of discarding.",
        "influence": "NMS is the safe default for well-separated animals. The merging variants help "
                     "when an animal is genuinely split across tiles, but can fuse two animals "
                     "standing close together into one.",
    },
    "sahi_postprocess_match_metric": {
        "short": "Overlap measure used to decide two boxes are the same animal.",
        "what": "IOS (intersection over smaller) is designed for tiling: a box cut by a seam is a "
                "small fragment of the full one, so IoU would score it low and keep both. IOU is "
                "the classic measure.",
        "influence": "IOS removes seam duplicates far more reliably; IOU leaves more duplicates but "
                     "is less likely to merge two animals that genuinely overlap.",
    },
    "sahi_postprocess_match_threshold": {
        "short": "Overlap above which two boxes are treated as duplicates.",
        "what": "Applied to the match metric above.",
        "influence": "Lower merges more aggressively (fewer duplicates, more risk of swallowing a "
                     "neighbour); higher keeps more boxes, including duplicated ones.",
    },
    "sahi_perform_standard_pred": {
        "short": "Also run one whole-frame pass alongside the tiles.",
        "what": "Adds the ordinary resized full-frame detection to the tiled results before merging.",
        "influence": "Catches animals too large to fit in a tile, at the cost of one extra pass and "
                     "more duplicates for the merge step to resolve.",
    },
    "sahi_min_dim_factor": {
        "short": "Skip tiling unless the frame is this many times larger than a tile.",
        "what": "Guard against paying the 20-40x tiling cost on footage that gains nothing from it: "
                "tiling is skipped when max(frame) <= max(tile) * this factor.",
        "influence": "Higher makes the auto-skip more aggressive (tiling only on genuinely large "
                     "frames); 1.0 tiles almost anything bigger than a tile.",
    },

    # ---------------- Launcher: training metrics ----------------
    "show_metrics": {
        "short": "Which training metrics the launcher prints after a run.",
        "what": "Purely a display choice for the launcher's training report. (B) columns are the "
                "detection boxes, (M) columns the segmentation masks — the latter only exist for "
                "segmentation models. F1 is computed from precision and recall.",
        "influence": "Changes nothing in training or inference; only what you are shown.",
    },
}


# ----------------------------------------------------------------------------
# Launcher action-button help (keyed by script filename)
# ----------------------------------------------------------------------------

BUTTON_HELP = {
    "BehaveAI_settings_gui.py":
        "Configure the project: classes, motion encoding, training, tracking and analysis "
        "settings. Start here — most other steps need settings filled in first.",
    "BehaveAI_annotation.py":
        "Label primary behaviours on the motion and static image streams to build the "
        "training dataset.",
    "BehaveAI_annotation_complex.py":
        "Annotate complex dyadic / group behaviours (interactions) for the complex-behaviour model.",
    "BehaveAI_inspect_dataset.py":
        "Review and correct existing annotations and check class balance before training.",
    "BehaveAI_augmentation.py":
        "Generate augmented copies of annotated frames to enlarge and diversify the dataset. "
        "Enabled only when the global augmentation probability is > 0.",
    "BehaveAI_classify_track.py":
        "Train the primary/secondary models and run batch classification + tracking on the "
        "input videos.",
    "BehaveAI_complex_model.py":
        "Train the complex-behaviour model from the interaction features and complex annotations.",
    "BehaveAI_complex_candidates.py":
        "Propose the most informative windows to annotate next (active learning).",
    "BehaveAI_live.py":
        "Run detection/classification live from a camera feed.",
}
