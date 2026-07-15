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
        "short": "Primary classes that skip the secondary classification stage.",
        "what": "Marks a primary class as exempt from secondary classification.",
        "influence": "Detections of that class are never passed to the secondary model, saving "
                     "compute and avoiding meaningless sub-labels.",
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
        "what": "Share of videos (never partial videos) assigned to a held-out set, used for both "
                "the YOLO train/val split and the complex-behaviour model's honest evaluation. "
                "Assignment is deterministic by video name: an existing video's status never "
                "changes, and new videos are classified automatically.",
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
        "short": "Minimum confidence to accept a secondary classification (0–1).",
        "what": "Secondary predictions below this score are treated as undecided.",
        "influence": "Higher avoids wrong sub-labels at the cost of leaving some boxes unclassified.",
    },

    # ---------------- Tracking ----------------
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

    # ---------------- Re-Identification ----------------
    "reid_enabled": {
        "short": "Re-assign the same ID to a subject that reappears in the same video.",
        "what": "Master switch for intra-video re-identification.",
        "influence": "Off reproduces the original tracker exactly. On reduces ID switches after "
                     "occlusions/exits, at some extra compute.",
    },
    "reid_method": {
        "short": "Appearance backend used to recognise a subject that reappears.",
        "what": "How visual identity is compared: histogram (colour, no torch), embedding "
                "(MobileNetV3), or megadescriptor (timm).",
        "influence": "histogram is fast and dependency-free but weaker; embedding/megadescriptor are "
                     "stronger but need torch and fall back to histogram if it is unavailable.",
    },
    "reid_similarity_threshold": {
        "short": "Cosine similarity gate for the embedding / megadescriptor methods.",
        "what": "Minimum appearance similarity to accept a match (embedding methods).",
        "influence": "Higher = stricter, fewer wrong merges but more missed re-IDs; acts as a weak "
                     "tie-breaker alongside the spatial gate.",
    },
    "reid_histogram_min_similarity": {
        "short": "Minimum colour-histogram similarity to accept a match (histogram method only).",
        "what": "Appearance gate used only when method = histogram/grid.",
        "influence": "Below this, identity relies on position/time alone. Ignored when method = "
                     "embedding.",
    },
    "reid_max_disappeared": {
        "short": "How long (seconds) a vanished ID is kept before being pruned.",
        "what": "Registry pruning guard — NOT a hard match limit.",
        "influence": "Longer keeps identities available after long absences (more memory); it does "
                     "not by itself force or block a match.",
    },
    "reid_max_position": {
        "short": "Max pixel distance for the spatio-temporal recovery gate.",
        "what": "The dominant matching signal: how far a subject may have moved to still be the same.",
        "influence": "Too small misses fast/teleporting reappearances; too large risks swapping "
                     "nearby subjects.",
    },
    "ab_min_classified": {
        "short": "Min classified frames for an ID to count as a group member (0 = skip).",
        "what": "Activity-budget filter on how much behaviour an ID must have to be counted.",
        "influence": "Higher removes fleeting/spurious IDs from group statistics; 0 disables the filter.",
    },
    "reid_descriptor": {
        "short": "Spatial layout of the appearance descriptor: global or grid.",
        "what": "'global' = one masked HSV histogram of the box (legacy); 'grid' = one histogram per "
                "foreground-aware cell, concatenated (a coarse body-parts descriptor).",
        "influence": "'grid' captures local colour patterns (e.g. markings) for better discrimination "
                     "but is more sensitive to pose/mask quality.",
    },
    "reid_grid": {
        "short": "Grid layout 'RxC' for the grid descriptor (e.g. 3x3).",
        "what": "Number of rows x columns the foreground box is split into for the grid descriptor.",
        "influence": "Finer grids give more spatial detail but shorter per-cell histograms and more "
                     "noise. Ignored when descriptor = global.",
    },
    "reid_foreground": {
        "short": "Foreground masking method for the descriptor: hsv, sam2 or yoloseg.",
        "what": "How the subject is separated from the background before building the descriptor.",
        "influence": "hsv is fast and dependency-free; sam2/yoloseg are more accurate but need the "
                     "segmentation model (fall back to hsv on failure).",
    },
    "reid_orient": {
        "short": "Align the grid to the body's major axis (PCA on the mask).",
        "what": "Rotates the grid so cells track head/tail orientation rather than image axes.",
        "influence": "Improves cell-to-cell consistency across poses; adds a little cost and depends on "
                     "a usable foreground mask. Ignored when descriptor = global.",
    },
    "reid_backbone": {
        "short": "MegaDescriptor backbone tag for the embedding/megadescriptor method.",
        "what": "Which BVRA MegaDescriptor variant to load (T-224, L-224, L-384, T-CNN-288).",
        "influence": "Larger backbones (L-384) are more discriminative but slower and heavier; T-224 is "
                     "the light default. Only used by the embedding/megadescriptor method.",
    },
    "reid_checkpoint": {
        "short": "Path to a fine-tuned MegaDescriptor checkpoint (blank = auto-detect).",
        "what": "Optional .pt produced by BehaveAI_reid_finetune.py. Empty auto-detects "
                "model_reid/megadescriptor_finetuned.pt if present.",
        "influence": "A project-specific checkpoint improves re-ID on your animals; blank uses the "
                     "pretrained backbone.",
    },
    "ab_analysis_duration_s": {
        "short": "Seconds of each video used for the activity budget (0 = whole video).",
        "what": "Truncates the analysis window to a fixed duration from the start of the clip.",
        "influence": "A fixed window makes budgets comparable across clips of different lengths; 0 uses "
                     "the entire video.",
    },

    # ---------------- Reference body length ----------------
    "foal_size_ratio": {
        "short": "body_len / reference below this flags a likely foal.",
        "what": "Relative size threshold to tag small individuals as foals.",
        "influence": "Lower tags only the smallest; higher tags more individuals as foals.",
    },
    "body_len_ref_scope": {
        "short": "Scope of the reference body length: whole video or per segment.",
        "what": "video = one reference for the clip; segment = recompute on altitude/zoom drift.",
        "influence": "'segment' adapts to changing camera distance (drone) but is noisier; 'video' "
                     "is stable when the camera is fixed.",
    },

    # ---------------- Interaction ----------------
    "complex_max_dist": {
        "short": "Pairs farther apart than this (px) are not treated as interacting.",
        "what": "Maximum distance for a dyad to be considered an interaction.",
        "influence": "Larger captures loose proximity events (more, weaker edges); smaller keeps only "
                     "close encounters.",
    },
    "complex_min_duration": {
        "short": "Minimum length (frames) of an interaction episode.",
        "what": "Episodes shorter than this are dropped (per_segment granularity).",
        "influence": "Higher removes brief incidental contacts; lower keeps short interactions.",
    },
    "complex_contact_iou": {
        "short": "Box overlap (IoU) above which two subjects count as in contact.",
        "what": "Overlap-based contact criterion.",
        "influence": "Higher requires strong overlap for 'contact'; lower flags grazing touches.",
    },
    "complex_contact_dist": {
        "short": "Distance (body lengths) below which subjects count as in contact.",
        "what": "Proximity-based contact criterion, complementing IoU.",
        "influence": "Larger labels near-misses as contact; smaller demands true closeness.",
    },
    "complex_window": {
        "short": "Window length (frames) over which features are aggregated for the model.",
        "what": "Temporal window for computing interaction/behaviour features.",
        "influence": "Longer captures slower behaviours but blurs quick ones; shorter is the reverse.",
    },
    "interaction_granularity": {
        "short": "Edge granularity in the interaction graph.",
        "what": "per_interaction (one edge per episode), per_segment, or per_frame.",
        "influence": "Finer granularity yields larger, more detailed edge files. Changing this "
                     "regenerates and overwrites the edges file.",
    },
    "interaction_weight": {
        "short": "How interaction edge weights are computed.",
        "what": "duration, proximity, or combined.",
        "influence": "Chooses what 'strength' means in the social graph — time spent together, "
                     "closeness, or both.",
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
                "the temporal sequence (need torch).",
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
        "short": "Speed (body lengths/frame) below which a subject counts as ~still.",
        "what": "Lower bound used by the candidate-proposal heuristics.",
        "influence": "Tunes what 'stationary' means when surfacing candidate windows.",
    },
    "complex_speed_high": {
        "short": "Speed (body lengths/frame) above which motion counts as fast (gallop/chase).",
        "what": "Upper bound used by the candidate-proposal heuristics.",
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
    "buttons_per_row": {
        "short": "Number of class buttons shown per row in the annotation window.",
        "what": "Grid width of the class button panel in the annotation/inspect tools.",
        "influence": "Display only. More columns fit more classes per row but make each button "
                     "narrower; fewer columns give wider buttons but a taller panel.",
    },

    # ---------------- Activity budget ----------------
    "ab_min_presence_ratio": {
        "short": "Min fraction of frames a subject must be present to count as a group member.",
        "what": "Stranger/visitor threshold for the activity budget.",
        "influence": "Higher excludes brief visitors from group stats; lower includes transient IDs.",
    },
    "ab_border_zone_ratio": {
        "short": "Width of the image border zone, as a fraction of image size.",
        "what": "Edge band used to detect entering/leaving subjects.",
        "influence": "Larger treats more of the frame edge as transit zone, affecting presence and "
                     "stranger detection.",
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
