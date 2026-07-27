# BehaveAI / HERDWISE — Evaluation Plan

A per-layer protocol for evaluating the pipeline to publication standard, and a
map of the evaluation code that implements it. Written to be cited directly from
a Materials & Methods section.

## Guiding principle: evaluate the models, not the aggregates

The pipeline is a chain: **detection (dual-stream) → tracking → drone correction →
stitching → (metric geometry) → activity budget → interaction graph / complex
behaviours**. Evaluation targets the **learned models and the geometric
algorithms**, because those are where error is introduced. Two deliberate
exclusions:

- **The activity budget is not a separate evaluation target.** It is a
  deterministic aggregation — for each track, `time(behaviour) = frames(behaviour) / fps`
  — with no parameter and no inference. Its error is fully determined and bounded
  by (a) detection recall, (b) tracking identity consistency (IDF1 / fragmentation),
  and (c) per-frame behaviour-classification accuracy, each measured directly
  below. Re-scoring the budget against an independent human ethogram would conflate
  model error with protocol, temporal-resolution and behaviour-definition
  differences between two heterogeneous measurement systems, and adds no
  information beyond the component metrics.
- **Single annotator → no inter-annotator reliability (κ).** Where a metric is
  scored against human labels, an *intra*-annotator test–retest on a small subset
  is the only reliability check available, and is optional.

## 0. What each layer needs

| Layer | Tool | Ground truth | Headline metric |
|---|---|---|---|
| 1. Detection (dual-stream) | `BehaveAI_evaluate_detection.py` | held-out annotation frames | recall/F1 of **merged vs static-only vs motion-only** |
| 2. Tracking | `BehaveAI_evaluate_tracking.py` | hand-made MOT GT (external) | HOTA (DetA / AssA), IDF1, MOTA, IDSW, Frag |
| 3. Behaviour (per-frame) | `BehaveAI_evaluate_detection.py` | held-out annotation frames | per-class confusion / F1 per stream |
| 4. Complex behaviours | `BehaveAI_complex_model.py` | hand-annotated complex segments | by-video (LOVO) macro-F1 vs chance |
| Geometry (feeds layer 4) | `BehaveAI_evaluate_geometry.py` | synthetic + known-distance refs | reprojection error, residual-flow std |

All generated outputs are written under `<project>/evaluation/` (git-ignored).

## A. Common experimental frame

1. **Video-level splits, never frame-level.** Two frames from one clip are
   near-duplicates; a frame-level split inflates every temporal metric. The
   frozen test set is the deterministic per-video holdout
   `behaveai_holdout.is_holdout_video(stem, val_frequency)` — the same partition
   the annotation tool and the complex model already use. It never needs a stored
   list (a video's status is a hash of its filename stem).
2. **Ground truth is independent of the pipeline output.** MOT tracking GT is
   hand-annotated, not derived from the tracker; behaviour GT is the human YOLO
   labels, evaluated on holdout videos the detectors never trained on.
3. **Report uncertainty, not just a point estimate.** Prefer per-video mean ± SD
   (or a bootstrap CI) over a single decimal.
4. **Reproducibility.** Fix and cite the §11 settings, the model checkpoints
   (`saved_settings.ini`, `train_count.txt`, `*_backupN`), and the tool versions
   (ultralytics, TrackEval).

## B. Layer 1 — dual-stream detection

**Question.** Does the second (motion) stream improve animal detection over the
static stream alone?

**Tool.** `python BehaveAI_evaluate_detection.py <project> [--split holdout] [--conf C] [--iou 0.5]`

It runs each primary detector on its own **saved annotation images**
(`annot_static/images`, `annot_motion/images` — exactly what the detectors see),
replicates the pipeline's stream-merge (containment overlap + centroid distance +
`dominant_source`, from `BehaveAI_classify_track.py:1346-1430`) in normalized
coordinates, and scores three variants against a class-agnostic **union of
annotated animals**:

- `detection_stream_ablation.csv` — TP/FP/FN, precision, recall, F1, AP for
  **static-only / motion-only / merged**. The headline: merged recall should
  exceed each single stream, quantifying the dual-stream contribution.
- `detection_by_class.csv` + confusion matrices — per-stream behaviour
  classification over matched detections (this **is** the per-frame behaviour
  metric, layer 3). Static and motion taxonomies are scored separately.

**Ablations for the paper.** motion-only vs static-only vs merged; effect of
`dominant_source`; SAHI on/off; recall stratified by target size (px) and drone
altitude. **Limits.** Immobile animals are invisible to the motion stream (the
merge depends on static there); small/distant targets; class imbalance.

## C. Layer 2 — tracking

**Tool.** `BehaveAI_evaluate_tracking.py` (wraps TrackEval; reports DetA and AssA
**separately** so detection and association gains are attributable).

- Explicit: `--seq NAME --gt gt.txt --pred output/NAME_tracking_stitched.csv` (repeatable).
- Auto-match: `--project <project> --gt-dir <dir-of-STEM.txt>` pairs each GT file
  to the most-processed `output/STEM_tracking*.csv` (metric > stitched > corrected > raw).

**Ground truth.** Per-sequence MOT-Challenge files (`frame,id,bb_left,bb_top,w,h,conf,class,vis`)
hand-annotated at 1–2 fps in CVAT/DarkLabel on a few representative clips (varied
herd size, altitude, occlusion). TrackEval scores only the annotated frames.

**Ablations.** `tracker_type` (botsort/bytetrack/kalman); stitching on/off (expect
lower Frag/IDSW, higher AssA/IDF1); drone correction on/off. **Limits.** Track
crossings, long occlusions (appearance is deliberately unused at 15–50 m), and K
(herd size) is a reported diagnostic, never a constraint.

## D. Layer 3 — behaviour classification and the activity budget

Per-frame behaviour classification = the primary detectors' per-class metrics,
produced by `BehaveAI_evaluate_detection.py` (§B): confusion matrix, per-class and
macro F1 per stream, over held-out frames. The **activity budget** is then a
deterministic sum of those labels (see the guiding principle) — its accuracy is
inherited from §B + §C and is **not** scored against a separate human budget.

## E. Layer 4 — complex behaviours and the interaction graph

**Tool.** `BehaveAI_complex_model.py` already evaluates honestly by video:
LeaveOneGroupOut (≤10 videos) or GroupKFold(5), reporting per-class F1, macro-F1,
a confusion matrix, merge suggestions, and a never-trained-on holdout score. It
now also reports **chance baselines** (most-frequent and stratified) under the
same by-video splits, so the macro-F1 is read against what a label-only guesser
scores (`model_complex/metrics.txt`).

**What still gates validity.** The kinematic/graph features are built on
drone-corrected (and optionally metre-scaled) trajectories — validate those first
(§F). Baseline (RF/HGB) vs LSTM/Transformer should be compared with the small-corpus
overfitting risk in mind. Graph-edge validation against expert dyadic scoring, and
external cross-site validation, are perspectives. **Limits.** Small corpus, class
imbalance, subjectivity of group behaviours.

## F. Geometry validation (prerequisite for §E)

**Tool.** `python BehaveAI_evaluate_geometry.py <project> [--known-distances refs.csv]`

- **Synthetic recovery** — warps a textured frame by known similarity transforms
  (translation/rotation/zoom/combined) and recovers them with the pipeline's own
  estimator (`drone_correction._estimate_step_transform`); reports sub-pixel
  reprojection error. Directly validates the estimator.
- **Correction-quality summary** — the ok/uncertain/none breakdown over
  `output/*_tracking_corrected.csv`, plus the continuous residual-flow-std
  distribution (median / p95 / max) from the `*_correction_diag.csv` sidecars now
  emitted by `drone_correction.py`.
- **Metric distance error (scaffold)** — with a `known_distances.csv`
  (`video,frame,u1,v1,u2,v2,true_m`) and a flight log, reports ground-plane
  distance error. Currently skips (HERDWISE has no flight logs); a controlled,
  flat-terrain calibration clip with measured distances is needed before any
  metre/second claim is published.

## G. Systematic limits (for the Discussion)

Do not list limits anecdotally — **measure them**. Stratify each layer's metric by
altitude/target-size, occlusion density / herd size, terrain slope, lighting, and
camera motion; build a failure-mode taxonomy (identity fragmentation at crossings,
false negatives on immobile animals, specific behaviour confusions from the matrices,
metric bias on sloped terrain); and report cross-site / cross-season generalisation.

## H. Work order and minimum viable evaluation

1. **Ground truth first** (the bottleneck): a few MOT clips (§C) — the largest
   unblocker. Detection/behaviour GT already exists as the annotation labels.
2. **Validate geometry** (§F) before drawing any complex-behaviour conclusion.
3. **Detection ablation** (§B) — the signature result — then **tracking** (§C).
4. **Complex model** (§E) last, once §F holds.

**Minimum viable for a paper:** §B (motion/static/merged ablation) + §C (HOTA on
3–5 MOT clips) + §E (LOVO macro-F1 vs chance). Metric distances and the graph model
can be framed as perspectives if the corpus/calibration do not yet support them —
a defensible position, not a weakness.

## Appendix — producing MOT ground truth

Annotate a few whole clips at 1–2 fps in CVAT or DarkLabel, export MOT-Challenge
format (`frame,id,bb_left,bb_top,w,h,conf,class,vis`), name each file `<video_stem>.txt`,
and point `BehaveAI_evaluate_tracking.py --project <p> --gt-dir <dir>` at them. For
~3 min × 10 horses at 1 fps that is ~1800 boxes, not 54 000.
