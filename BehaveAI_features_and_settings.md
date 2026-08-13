# BehaveAI — Features, Settings & Methods Reference

**Version 5.0 — 2026-08-05.** This document describes the pipeline **as it currently stands**. It is a reference, not a change log: anything removed from the code is removed from here.

A combined **user guide**, **README**, and **Materials & Methods source** for BehaveAI: a project-based pipeline for annotating video, training YOLO detectors/classifiers on a dual **static (RGB)** \+ **motion (false-colour)** representation, tracking individuals, and computing per-individual activity budgets — plus the **HERDWISE** layer that turns those tracks into metric, multi-individual interaction data.

**How to read this document**

- Every feature is documented with a fixed template:  
  - **How it works** — mechanism and data flow  
  - **Purpose** — what the feature is for  
  - **Parameters** — settings that control it and their effect  
  - **Implementation** — functions \+ packages used  
- Each feature carries two separate descriptions:  
  - **\[Technical\]** — AI / computer-vision terminology  
  - **\[Plain\]** — no specialised vocabulary  
- Section 11 is the exhaustive parameter table for direct citation in a methods section.

**Software stack (packages):** ultralytics (YOLO detection + classification, BoT-SORT/ByteTrack trackers), opencv-python (cv2), numpy, scipy (scipy.optimize.linear\_sum\_assignment; scipy.signal for smoothing), lap (linear assignment inside the Ultralytics trackers), pillow (PIL), pyyaml (yaml), tkinter, picamera2 (Raspberry Pi only), ncnn / onnx / onnxslim / onnxruntime (optional inference & export backends), sahi (optional sliced inference), torch/torchvision (optional — the LSTM/Transformer complex-behaviour models; also the training backend), networkx (optional — interaction-graph metrics; the edges/nodes CSVs are still written without it), scikit-learn + joblib (the complex-behaviour baseline model), trackeval (tracking evaluation only), matplotlib + polars + moviepy/ffmpeg-python (auxiliary tooling), plus standard library (configparser, csv, glob, subprocess, threading, pathlib, collections, os, sys, re, math, time, random, shutil, base64, queue, platform, hashlib). **pandas is intentionally not used** — every CSV is read/written with the stdlib csv module, and the interaction-graph deliverable is plain CSV for R igraph.

---

## Pipeline Overview

**Per-individual pipeline**

- **Capture / collect** video clips → clips/  
- **Annotate** frames on two synchronised representations (static RGB \+ motion false-colour) → BehaveAI\_annotation.py  
- **Inspect / correct** the dataset → BehaveAI\_inspect\_dataset.py  
- **Augment** the dataset offline → BehaveAI\_augmentation.py  
- **Train** the species/age classifiers, the primary detectors (one per stream) and the secondary classifiers → BehaveAI\_classify\_track.py (each model in its own process via BehaveAI\_train\_worker.py)  
- **Detect → merge streams → classify → track** on batch videos → BehaveAI\_classify\_track.py  
- **Re-link tracklets offline** into longer identities, on kinematics → BehaveAI\_stitch\_tracklets.py  
- **Activity budget** per individual \+ stranger flagging → BehaveAI\_activity\_budget.py  

**HERDWISE multi-individual / complex-behaviour pipeline** (drone footage of horse herds; everything downstream of the tracker is in **real-world metres**, obtained from flight-log telemetry)

- **Drone motion correction** — remove the drone's apparent pan/zoom motion from every centroid via background optical flow → behaveai\_drone/drone\_correction.py (§6.8)  
- **Metric geometry** — project image positions onto the ground plane in metres, from the flight log → behaveai\_drone/metric\_geometry.py (§6.9)  
- **Interaction graph** — dyadic + group (whole co-present herd) features aggregated into an R-igraph-ready undirected graph (primary analysis output) → BehaveAI\_complex\_features.py (§6.10)  
- **Annotate complex behaviours** — multi-individual / dyadic / group annotation tool → BehaveAI\_annotation\_complex.py (§6.11)  
- **Train complex model** — supervised classifier (sklearn baseline or LSTM/Transformer), by-video evaluation → BehaveAI\_complex\_model.py (§6.12)  
- **Propose candidates** — heuristic + active-learning candidate segments to speed up annotation → BehaveAI\_complex\_candidates.py (§6.13)  

**Also**

- **Evaluate** detection, tracking and geometry against held-out / hand-annotated ground truth → BehaveAI\_evaluate\_\*.py (§7A)  
- **Live** acquisition \+ inference → BehaveAI\_live.py  
- All orchestrated by the **launcher** BehaveAI.py; configured by BehaveAI\_settings\_gui.py.

**Post-processing order.** BehaveAI\_classify\_track.py runs the whole chain itself, per project, in this order: `classify_track → drone_correction → tracklet stitching → metric_geometry → interaction graph → complex-behaviour stage → activity budget`. Every stage after the tracker is individually gated by an INI switch, and the activity budget runs **last**, on the most-processed CSV available per video.

---

## 1\. LAUNCHER (BehaveAI.py)

### 1.1 Project management

- **\[Technical\]**  
  - *How it works:* Projects live under a sibling projects/ directory. Each project is a self-contained workspace holding its settings, datasets, models, and I/O. Creating a project scaffolds clips/, input/, output/, timecodes/ (with an example\_timecodes.csv template) and a default BehaveAI\_settings.ini. A combobox enumerates project folders; selection triggers loading and statistics computation. Refresh rescans the directory.  
  - *Purpose:* Isolate independent experiments/datasets so settings, annotations, and weights never collide.  
  - *Parameters:* project\_path (absolute root, written into the INI); folder layout is fixed by convention.  
  - *Implementation:* pathlib.Path, os, configparser for INI scaffolding; tkinter.ttk.Combobox, simpledialog, messagebox for the UI.  
- **\[Plain\]**  
  - *How it works:* Each study is one folder under projects/, containing its own videos, labels, trained files, and a settings file. A menu lists those folders; picking one loads it. Creating one builds the empty folders and a default settings file automatically.  
  - *Purpose:* Keep separate studies fully apart so nothing from one mixes into another.  
  - *Parameters:* The project's location on disk; all sub-folders follow a fixed naming scheme.

### 1.2 Smart button enabling

- **\[Technical\]**  
  - *How it works:* Action buttons are grouped into labelled pipeline stages so the workflow order is obvious — **1 – Setup** (Settings); **2 – Annotate** (Annotate, *Annotate complex*, Inspect Dataset, Augment Dataset); **3 – Train** (Train & batch classify, *Train complex model*, *Propose candidates*); **4 – Run** (Live). Button availability is derived from INI state at selection time: *Settings* enables on any project; the class-dependent stages require ≥1 non-empty primary class list; *Augment Dataset* requires aug\_global\_probability \> 0. After any subprocess exits, states are re-evaluated.  
  - *Purpose:* Prevent launching stages whose preconditions (defined classes, enabled augmentation) are not met, and make the per-individual → multi-individual workflow self-explanatory.  
  - *Parameters:* primary\_motion\_classes / primary\_static\_classes (presence), aug\_global\_probability (\> 0).  
  - *Implementation:* configparser read; Tk widget state toggling.  
- **\[Plain\]**  
  - *How it works:* Buttons turn on only when their requirements exist — for example, the annotation button stays off until at least one class is defined, and the augmentation button stays off until augmentation is switched on. After a tool closes, the buttons are rechecked.  
  - *Purpose:* Stop you from starting a step that cannot yet run.  
  - *Parameters:* Whether classes are defined; whether the global augmentation value is above zero.

### 1.3 Integrated streaming terminal

- **\[Technical\]**  
  - *How it works:* Each tool is launched as a subprocess; stdout/stderr are read byte-by-byte and rendered live in a scrollable Tk text widget (stdout black, stderr blue). YOLO-style progress lines (N/M ... XX%) are detected and overwritten in place; OpenCV camera-probe warnings (\[ WARN:0@, \[ERROR:0@) are suppressed; ANSI escape codes are stripped.  
  - *Purpose:* Real-time, low-noise log of long-running training/inference jobs in one window.  
  - *Parameters:* none persisted; line-classification handled by is\_progress\_line() / strip\_ansi().  
  - *Implementation:* subprocess.Popen with PYTHONUNBUFFERED=1, threading, queue, re; TextRedirector class; Tk scrolledtext.  
- **\[Plain\]**  
  - *How it works:* Each tool runs as a separate program; its text output appears immediately in a scrolling panel, with normal messages in black and errors in blue. Repeating progress counters are rewritten on the same line instead of stacking up, and irrelevant camera-search warnings are hidden.  
  - *Purpose:* Watch what a long job is doing without clutter.  
  - *Parameters:* none.

### 1.4 Project statistics panel

- **\[Technical\]**  
  - *How it works:* On selection, a background thread scans annotation directories and model result files and reports: unique annotated frames, original vs augmented image counts, train/val split (actual vs target %), per-class bounding-box distribution with an ⚠ under-represented flag below 10 %, video coverage, per-augmentation breakdown, and per-model metrics parsed from results.csv (epochs, weights size/date, Precision/Recall/mAP@0.5/mAP@0.5:0.95 for box (B) and mask (M), box/cls/DFL losses, learning rates pg0–pg2, and a derived F1).  
  - *Purpose:* Dataset-balance and model-quality dashboard — the principal source of dataset and training statistics for a methods section.  
  - *Parameters:* val\_frequency (split target), augmentation parameters (breakdown), class lists (distribution).  
  - *Implementation:* os/glob directory scans, csv parsing of results.csv, threading for non-blocking compute; F1 \= 2·P·R/(P+R).  
- **\[Plain\]**  
  - *How it works:* Picking a project counts your labelled frames, how many are originals versus generated copies, the train/validation split, how balanced the classes are (flagging any under 10 %), how many videos have labels, and the quality scores of every trained file. It runs in the background so the window stays responsive.  
  - *Purpose:* A single readout of how big and balanced your data is and how well models scored.  
  - *Parameters:* The target validation fraction, the augmentation settings, and the class names.

### 1.5 Script launching context

- **\[Technical\]**  
  - *How it works:* Subprocesses inherit PYTHONUNBUFFERED=1, BEHAVEAI\_PROJECT (project path), working directory set to the project, and the project path as argv\[1\].  
  - *Purpose:* Guarantee every tool resolves the correct project regardless of invocation method.  
  - *Parameters:* BEHAVEAI\_PROJECT env var; argv.  
  - *Implementation:* subprocess, os.environ.  
- **\[Plain\]**  
  - *How it works:* Every tool is started with the project's location passed in three redundant ways so it always opens the right project.  
  - *Purpose:* Make sure tools never act on the wrong project.  
  - *Parameters:* The project path.

---

## 2\. SETTINGS GUI (BehaveAI\_settings\_gui.py)

*Every parameter carries an inline **ⓘ help marker** with a hover tooltip (text from BehaveAI\_settings\_help.py / PARAM\_HELP), a consistent **theme** is applied via apply\_theme(), and every config key the pipeline reads is editable here. Settings are organised into tabs: **Species, Model structure, Video paths, Data augmentation, Motion strategy, Model type, Tracking** (which also hosts the drone-motion-correction and offline-stitching groups), **Metric, Interaction, Complex Behaviours, Display Settings, Activity Budget**.*

*Two families of keys have no widget by design and are edited in the INI: the per-drone checkerboard focal lengths `metric_fpx_<DroneToken>` (one key per drone in the fleet, a list rather than a fixed field — a save carries them forward untouched), and `ignore_secondary`, a legacy key superseded by the secondary map (§2.2) and only consulted by behaveai\_config's fallback for projects with no secondary pool at all.*

### 2.1 Species & age classes (model 0 / 0.5)

- **\[Technical\]**  
  - *How it works:* A **Species** tab defines the species detected *before* the behaviour models (model 0), by scientific name (e.g. "Equus caballus"), each with a hotkey and colour; an **age** class list (model 0.5) is defined per species alongside the behaviour lists. The **first** species in species\_list keeps the bare, legacy key and folder names (primary\_static\_classes, annot\_static/, model\_primary\_static/…), so a single-species project needs no migration; every additional species gets suffixed keys `<base_key>__<slug>` and suffixed folders, resolved by behaveai\_config.species\_key() / species\_folder(). Crops for these two models are pooled project-wide in annot\_species\_crop/ and annot\_age\_crop/ and train into model\_species/ and model\_age/.  
  - *Purpose:* Let one project hold several species without their ethograms, datasets or weights colliding, and provide the age class (adult/foal) the metric interaction layer needs — measured by a classifier rather than guessed from apparent size.  
  - *Parameters:* species\_list, species\_hotkeys, species\_colors, age\_classes/age\_hotkeys/age\_colors (per species), show\_species, show\_age.  
  - *Implementation:* behaveai\_config.species\_key(), species\_folder(), species\_slug(), load\_age\_classes(), load\_species\_scoped\_config(); ClassListEditor in the GUI.  
- **\[Plain\]**  
  - *How it works:* One list names the animals the system should tell apart (by scientific name), and each of them has its own list of age categories and its own set of behaviour categories. The first species keeps the project's existing folders untouched; any species you add gets its own.  
  - *Purpose:* Study more than one animal in the same project, and know which individuals are young — a fact the herd analysis needs.  
  - *Parameters:* the species list and, per species, its age and behaviour categories.

### 2.2 Model structure editor (class groups & shared secondary pool)

- **\[Technical\]**  
  - *How it works:* Three editable groups per species: **Primary static**, **Primary motion**, and a single **Secondary (shared pool)** list reused across both streams. Each row has a label, an optional single-character hotkey, and an RGB colour (colour chooser). A separate **SecondaryMapEditor** assigns, per primary class, *which* of the shared secondaries are allowed (`secondary_map = Primary1:secA|secB; Primary2:secA`); a primary with no mapping entry simply has no secondary step. Validation blocks saving on duplicate/multi-char/reserved (u, g, n) hotkeys, on having zero primary classes, or on a shared secondary pool with exactly one class (YOLO classification needs ≥2). Adding/removing/clearing classes when annot\_motion/annot\_static already exist raises a structural-change warning. The loader also accepts per-stream secondary keys (secondary\_static\_classes / secondary\_motion\_classes) from older projects and reconstructs them into the shared pool + map. Shared by all tools via behaveai\_config.py (load\_secondary\_config, parse\_secondary\_map, format\_secondary\_map).  
  - *Purpose:* Define the two-tier label space — primary classes detected per stream and an optional, shared set of secondary sub-classes mapped per primary for hierarchical classification.  
  - *Parameters:* primary\_\*\_classes/colors/hotkeys, secondary\_classes/colors/hotkeys, secondary\_map.  
  - *Implementation:* ClassRow, ClassListEditor, SecondaryMapEditor (ttk.Frame); colorchooser; behaveai\_config.parse\_secondary\_map/format\_secondary\_map; parse\_list\_field/parse\_colors\_field/list\_to\_field/colors\_to\_field.  
- **\[Plain\]**  
  - *How it works:* You define your main categories in two lists (one per view) plus one shared list of optional finer categories, each with a name, a shortcut key, and a colour. A small table then says, for each main category, which of the finer categories are allowed for it. The tool refuses to save if two keys clash, a key uses a reserved letter, no main category exists, or the shared finer list has only one entry. It warns you if you change categories after labelling has begun.  
  - *Purpose:* Set up the categories the system will learn — main ones, plus an optional shared set of finer ones chosen per main category.  
  - *Parameters:* The category names, colours, shortcut keys, and the per-main-category list of allowed finer categories.

### 2.3 Cross-stream blocking (motion/static masking)

- **\[Technical\]**  
  - *How it works:* *Motion blocks static* paints motion-annotation box regions grey on static training images at save time; *Static blocks motion* does the reverse. This removes the other stream's targets from each image so a detector is not trained against objects it is not meant to learn in that stream.  
  - *Purpose:* Prevent cross-contamination between the two single-stream detectors.  
  - *Parameters:* motion\_blocks\_static, static\_blocks\_motion.  
  - *Implementation:* grey fill via cv2.rectangle(...,(128,128,128),-1) (applied in annotation/regeneration saving).  
- **\[Plain\]**  
  - *How it works:* Objects that belong to one view can be grey-covered in the other view's training images, so each detector only sees the objects it is supposed to learn.  
  - *Purpose:* Stop the two detectors from confusing each other's targets.  
  - *Parameters:* The two "blocks" switches.

### 2.4 Video paths

- **\[Technical\]** *How it works:* Three browse-able directory fields (clips\_dir, input\_dir, output\_dir) validated as non-empty at save. *Purpose:* declare data locations for annotation and batch I/O. *Parameters:* the three paths. *Implementation:* filedialog.askdirectory.  
- **\[Plain\]** *How it works:* Three folder pickers tell the system where training clips, batch inputs, and outputs live; saving warns on empty ones. *Purpose:* point the tools at the right folders. *Parameters:* the three folders.

### 2.5 Data-augmentation configuration

- **\[Technical\]**  
  - *How it works:* Exposes the master gate aug\_global\_probability plus a per-transform *range* and *probability* for brightness, contrast, saturation, hue, sharpness, blur, noise, shear, horizontal/vertical flip, and temperature, plus aug\_target\_classes (restrict augmentation to images containing given classes) and motion\_disable\_color\_aug. *Delete all augmented data* removes every file with \_aug\_ in its basename across image/label dirs (with confirmation).  
  - *Purpose:* Configure offline augmentation that enlarges and diversifies the dataset.  
  - *Parameters:* aug\_global\_probability, aug\_target\_classes, aug\_\<name\>\_range, aug\_\<name\>\_probability, aug\_flip\_\*\_options (see §5, §11).  
  - *Implementation:* INI read/write via configparser; deletion via glob/os.remove.  
- **\[Plain\]**  
  - *How it works:* You set one master chance plus, for each effect, a strength range and its own chance, and you can restrict augmentation to images containing particular categories. A button can delete all generated copies at once.  
  - *Purpose:* Control how the dataset is expanded with modified copies.  
  - *Parameters:* the master chance, the target categories, and each effect's range and chance.

### 2.6 Motion strategy configuration

- **\[Technical\]**  
  - *How it works:* Selects how the motion false-colour stream is built (see §3.3 for the algorithm): strategy (exponential/sequential), chromatic\_tail\_only, decay factors expA/expB (which also set the frame-window size), lum\_weight, rgb\_multipliers, frame\_skip, plus scale\_factor and motion\_threshold. The expA value maps to frameWindow (0.2→5, \>0.5→10, \>0.7→15, \>0.8→20, \>0.9→45).  
  - *Purpose:* Tune the temporal-difference encoding that makes movement visible to a detector.  
  - *Parameters:* all motion-strategy keys (see §11).  
  - *Implementation:* consumed by generate\_base\_images() and equivalents; cv2.absdiff, cv2.addWeighted, cv2.subtract, cv2.merge.  
- **\[Plain\]**  
  - *How it works:* You choose how movement is turned into colour: which method, how long the coloured trail lasts, how much grey image is mixed in, and how strongly each colour is boosted.  
  - *Purpose:* Adjust how clearly motion shows up.  
  - *Parameters:* the motion-strategy settings (see §11).

### 2.7 Model-type configuration

- **\[Technical\]**  
  - *How it works:* Sets val\_frequency, primary\_classifier (base detector, e.g. yolo11s.pt; yolo8/11/26 in n/s/m/l), primary\_epochs, secondary\_classifier (e.g. yolo11s-cls.pt), secondary\_epochs, use\_ncnn, primary\_conf\_thresh, secondary\_conf\_thresh (a floor applied *after* the `__none__` arbitration; 0 disables it) and dominant\_source (confidence/motion/static). A **Training controls** group adds train\_patience (early-stopping patience — the epoch counts are only the cap) and save\_empty\_frames (write box-free frames as background negatives); motion\_disable\_color\_aug sits on the Data augmentation tab (§2.5). The SAHI sliced-inference group (sahi\_enabled\_static/motion and the slice/overlap/post-process keys, §6.4) lives here too, and its eleven parameters stay hidden until one of the two stream switches is on.  
  - *Purpose:* Choose architecture sizes, training length, inference thresholds, the inter-stream tie-break rule, and whether detection runs on tiles.  
  - *Parameters:* the keys above (see §6, §11).  
  - *Implementation:* values consumed by BehaveAI\_classify\_track.py training/inference.  
- **\[Plain\]**  
  - *How it works:* You pick the base model files, how many training passes to run (and when to stop early), the minimum confidence to keep a result, which view wins when both views detect the same object, and whether large frames are cut into tiles before detection.  
  - *Purpose:* Set how the models are built and how strict detection is.  
  - *Parameters:* base models, epochs, confidence thresholds, tie-break rule, tiling.

### 2.8 Tracking configuration

- **\[Technical\]**  
  - *How it works:* The **Tracking** tab first selects tracker\_type (botsort | bytetrack | kalman) and then shows only the parameters that apply to the choice: for the Ultralytics trackers, tracker\_track\_high\_thresh, tracker\_track\_low\_thresh, tracker\_new\_track\_thresh, tracker\_track\_buffer, tracker\_match\_thresh and (BoT-SORT only) tracker\_gmc\_method; for the legacy tracker, match\_distance\_thresh, delete\_after\_missed and the \[kalman\] triplet process\_noise\_pos / process\_noise\_vel / measurement\_noise. Stream-merging (centroid\_merge\_thresh, iou\_thresh) applies to all choices. The same tab hosts the **drone motion correction** group (§6.8), the **offline stitching** group (§6.6) and the activity-budget membership gate ab\_min\_classified\_frames.  
  - *Purpose:* Govern detection-to-track association, stream merging, offline identity re-linking and camera-motion handling from one place.  
  - *Parameters:* the keys above (see §11).  
  - *Implementation:* read by BehaveAI\_classify\_track.py into the Ultralytics tracker config or KalmanTracker.  
- **\[Plain\]**  
  - *How it works:* You choose which tracker to use, and only that tracker's settings are shown. You also set how detections from the two views are combined, how the drone's own movement is compensated, and whether broken tracks are re-joined afterwards.  
  - *Purpose:* Control identity tracking end to end.  
  - *Parameters:* the tracker choice and its thresholds.

### 2.8b Metric-geometry configuration

- **\[Technical\]**  
  - *How it works:* A dedicated **Metric** tab holds metric\_enabled plus the camera parameters it derives the pixel focal length from (metric\_focal\_len\_mm, metric\_sensor\_width\_mm), the reliability flags (metric\_roll\_max\_deg, metric\_horizon\_margin\_px) and the scale cross-check (metric\_assumed\_body\_length\_m, metric\_scale\_tolerance). Per-drone checkerboard overrides remain INI-only (see the note in §2).  
  - *Purpose:* Turn the whole metric layer on and calibrate it from one place, since every metre-valued result downstream (§6.9–6.13, the stitcher's m/s gate) depends on it.  
  - *Parameters:* the metric\_\* keys (§11).  
  - *Implementation:* behaveai\_drone/metric\_geometry.load\_metric\_config().  
- **\[Plain\]**  
  - *How it works:* You switch on real-world measurement and tell the system which camera filmed the clips; it then reports distances in metres instead of pixels.  
  - *Purpose:* Make distances and speeds real, and comparable between videos.  
  - *Parameters:* the camera's focal length and sensor width, plus the checks that flag unreliable frames.

### 2.9 Display & Activity-Budget configuration

- **\[Technical\]**  
  - *How it works:* **Display:** detection\_video\_enabled (whether the pipeline writes the annotated video copy at all — off by default), base line\_thickness / font\_size plus the shared box-rendering style used by both the output video and the annotation tool (box\_line\_scale, box\_font\_scale, adaptive\_box\_scaling and its coefficient/min/max pairs, label\_bg\_mode / label\_bg\_opacity / label\_bg\_color, halo\_thickness / halo\_color, show\_species, show\_age, buttons\_per\_row), and a checkbox grid for the show\_metric\_\* keys that decide which training metrics the launcher prints after a run (display only). **Activity budget:** ab\_min\_presence\_seconds, ab\_min\_classified\_frames, ab\_edge\_margin\_px, ab\_analysis\_duration\_s, ab\_group\_type\_separator, ab\_group\_type\_field\_index (zero-based filename field for group type).  
  - *Purpose:* Overlay styling shared across tools, and the rules of the presence/stranger analysis (§7).  
  - *Parameters:* listed keys.  
  - *Implementation:* behaveai\_render.load\_render\_style() / draw\_labeled\_detection(); BehaveAI\_activity\_budget.py.  
- **\[Plain\]** *How it works:* Sets how boxes and labels are drawn (size, background, outline, whether species and age are shown) everywhere they appear, plus the rules that decide who counts as a group member versus a stranger and how the group type is read from the filename. *Purpose:* control labels on video and the presence analysis. *Parameters:* the display and budget values.

### 2.10 Save behaviour (with regeneration)

- **\[Technical\]**  
  - *How it works:* On save, writes BehaveAI\_settings.ini, generates static\_annotations.yaml and motion\_annotations.yaml (YOLO dataset configs), and creates missing annotation directory trees. If motion-strategy parameters changed and annotations exist, it offers to run Regenerate\_annotations.py; if accepted, it first backs up model\_primary\_motion, model\_primary\_static, and the secondary models to \*\_backupN.  
  - *Purpose:* Persist configuration, keep YOLO configs in sync, and keep motion images consistent with current settings.  
  - *Parameters:* all settings; motion keys trigger regeneration.  
  - *Implementation:* configparser, yaml.safe\_dump, os.makedirs, shutil backups, subprocess to launch regeneration.  
- **\[Plain\]**  
  - *How it works:* Saving writes the settings file and the two dataset description files, and builds any missing folders. If you changed how motion is computed and you already have labels, it offers to rebuild the motion images and safely backs up existing models first.  
  - *Purpose:* Save settings and keep everything consistent.  
  - *Parameters:* all settings; motion changes trigger the rebuild offer.

---

## 3\. ANNOTATION TOOL (BehaveAI\_annotation.py)

### 3.1 Frame-pool, random sampling, frame-by-frame stepping & CSV time-code navigation

- **\[Technical\]**  
  - *How it works:* Recursively scans clips\_dir and builds a pool of annotatable frames (those with enough preceding frames for the motion window). Already-annotated frames are excluded via the AnnotationIndex. The tool supports three navigation modes selectable at any time from a **Source** button in the control bar: **(1) Random** — launch and each save/skip draw a new random unannotated frame; exits when the pool is empty. **(2) Frame per frame** — stay inside the current video and advance by a fixed `seq_step` frames on every save/skip. The step is asked for when the mode is entered, and the dialog states the equivalent cadence at the clip's fps (1 = every frame, fps/2 = 2 fps, fps = 1 fps). Unlike Random it deliberately does **not** skip already-annotated frames: landing on one reloads its boxes for review, which is what continuous-stretch work needs. At the end of the clip it says so and stops advancing rather than jumping to another video. Pick the starting point with the seek bar (or arrive via Random/CSV) before switching the mode on. **(3) CSV time-codes** — load a CSV listing specific frames of interest; each save/skip then advances through those frames *in CSV order* (revisiting already-annotated frames is allowed, e.g. to verify a behaviour), automatically switching video as needed; when the list is exhausted the tool reverts to Random mode. The CSV may target many videos: a *video* column (video\_filename / video / filename) is matched against the pool on the filename stem (case-insensitive), and a *time* column (frame / timecode / time / start\_frame) is read either as an integer frame index or as an mm:ss / hh:mm:ss time-code converted via the video fps. Out-of-range frames are clamped; duplicates are dropped while preserving order. The Source button label shows the active mode and its progress (Source: Random / Source: Frame+15 / Source: CSV (i/N)). CSV files live in the project's timecodes/ folder; an example\_timecodes.csv template is created there on first use.  
  - *Purpose:* Spread annotation uniformly at random across all clips to avoid temporal bias (Random); label a continuous stretch of one clip at a controlled cadence, which is what behaviours only legible over time (Kick, Rear, Strike, Balk) and any per-clip dense pass require (Frame per frame); or steer annotation toward specific behaviours/events recorded elsewhere — e.g. tracker outputs or hand-made event lists (CSV time-codes).  
  - *Note:* Frame-per-frame produces temporally dense frames but **not** a MOT ground truth: annotation labels carry a class and a box, never a persistent track identity (see §Evaluation — tracking). Frames from one clip all land on the same side of the train/holdout split, so a dense pass cannot leak between train and val; it does, however, weight that clip heavily in the detector's training distribution.  
  - *Parameters:* frame\_window (derived from expA/strategy), frame\_skip, clips\_dir; Frame-per-frame adds seq\_step and CSV mode the timecodes/ CSV file (neither is an INI setting — both chosen at runtime).  
  - *Implementation:* build\_frame\_pool(), get\_unannotated\_pool(), pick\_random\_frame() (Random); load\_next\_sequential\_frame(), AnnotatorTk.\_enter\_sequential\_mode() (Frame per frame); parse\_timecode\_csv(), \_parse\_frame\_value() (CSV); go\_to\_frame(), load\_next\_target(), AnnotatorTk.open\_source\_menu()/set\_nav\_mode(), \_scan\_videos\_recursive(); random, csv, re, cv2, AnnotationIndex.  
- **\[Plain\]**  
  - *How it works:* The tool lists every frame that can be labelled across all clips and removes those already done. A **Source** button lets you choose how the next frame is picked: *Random frames* shows a random remaining one each time you save or skip; *Frame per frame* stays in the video you are on and simply moves forward by a fixed number of frames each time (you choose the number, and the dialog tells you what it means in frames per second); *Load a time-code CSV* instead walks through a list of specific moments you provide in a CSV file (by frame number or by mm:ss time), jumping to the right video automatically and showing how far through the list you are. CSV files are kept in the project's timecodes/ folder, where a ready-to-edit example is created for you.  
  - *Purpose:* Either label evenly at random, walk steadily through one clip when a behaviour only makes sense across several frames, or go straight to particular behaviours you already noted down.  
  - *Parameters:* the motion-window length, frame skipping, the clips folder, the step size (frame-by-frame mode) and the time-code CSV you load (CSV mode).

> **Google-Sheets time-code template (make\_timecodes\_template.py).** A helper script writes an `.xlsx` template (columns video\_filename, timecode, behaviour) ready to upload to Google Sheets, fill in, and export back to CSV. The timecode column is forced to TEXT format ("@") so that a cell typed as `02:15` is **not** auto-converted by Google Sheets to `02:15:00` (which BehaveAI would otherwise read as 2 h 15 m instead of 2 m 15 s). Requires openpyxl. `python make_timecodes_template.py [output.xlsx]`.

### 3.2 Prefetch system

- **\[Technical\]**  
  - *How it works:* After a frame is shown, a background thread (2 s delayed) precomputes the *next* frame's static image, motion image, raw buffer, and canvas-sized composite. On transition, a cache hit yields instant display; a miss falls back to synchronous load.  
  - *Purpose:* Hide motion-recomputation latency for fast labelling.  
  - *Parameters:* none persisted; depends on canvas size.  
  - *Implementation:* \_compute\_frame\_data(), \_start\_prefetch(); threading, copy.  
- **\[Plain\]**  
  - *How it works:* While you label one frame, the next is prepared in the background, so it usually appears with no wait.  
  - *Purpose:* Remove waiting between frames.  
  - *Parameters:* none.

### 3.3 Dual-stream motion false-colour generation (core algorithm)

- **\[Technical\]**  
  - *How it works:* For each target frame, base\_N greyscale frames (spaced by frame\_skip+1) are read. Three reference buffers are maintained. **Exponential:** buffer 0 \= previous grey; buffer 1 \= addWeighted(buf1, expA, gray, 1-expA); buffer 2 \= addWeighted(buf2, expB, gray, 1-expB) — three decay rates → a multi-timescale coloured trail. **Sequential:** the three buffers are the three preceding frames. Per-channel diffs \= absdiff(buffer\_j, gray). The false-colour image is merge(\[blue,green,red\]) where each channel \= addWeighted(gray, lum\_weight, diff\_j, rgb\_multiplier\_j, motion\_threshold). With chromatic\_tail\_only, channels use *differences between diffs* (subtract(diff\_i, diff\_j)) to isolate directional motion and suppress static luminance. scale\_factor downscales frames first.  
  - *Purpose:* Encode temporal motion into a single RGB image a standard YOLO detector can learn from, alongside the untouched static RGB frame.  
  - *Parameters:* strategy, expA, expB, lum\_weight, rgb\_multipliers, frame\_skip, scale\_factor, motion\_threshold, chromatic\_tail\_only.  
  - *Implementation:* cv2.VideoCapture, cv2.cvtColor, cv2.absdiff, cv2.addWeighted, cv2.subtract, cv2.merge, numpy; reference impl generate\_base\_images().  
- **\[Plain\]**  
  - *How it works:* Several recent frames are turned to grey and compared. Where things moved, the differences are turned into colour and laid over a faint grey copy of the scene, so a moving animal leaves a coloured trail whose colours show how recent each part of the movement was. The plain colour photo is kept separately and unchanged.  
  - *Purpose:* Make movement directly visible in one image, in addition to the normal photo.  
  - *Parameters:* the motion-strategy settings.

### 3.4 Composite display, zoom column & crosshair

- **\[Technical\]**  
  - *How it works:* A full-screen Tk canvas shows one composite: a main panel (motion *or* static, toggled with Space) and a right column of three panels — 2× static crop, 2× motion crop (both centred on the cursor), and an animation panel cycling the raw buffer at the annotated position. A white crosshair tracks the cursor; the composite is scaled uniformly to the canvas.  
  - *Purpose:* Provide synchronised magnified context plus real movement playback for precise labelling.  
  - *Parameters:* the shared box-rendering style (§2.9).  
  - *Implementation:* cv2 drawing, PIL.Image/ImageTk, cv2\_to\_photoimage(), behaveai\_render.draw\_labeled\_detection().  
- **\[Plain\]**  
  - *How it works:* One full-screen view shows either the motion or the photo (toggle with Space), plus three side panels: a zoomed photo, a zoomed motion image, and a small looping clip of the real movement at that spot.  
  - *Purpose:* See fine detail and the actual motion while placing boxes.  
  - *Parameters:* box and text size.

### 3.5 Seek bar with annotation ticks

- **\[Technical\]** *How it works:* A slider spans the current video; red ticks mark already-annotated frames, a black line marks the current position; dragging triggers full motion recomputation. *Purpose:* navigate within a clip and see coverage. *Parameters:* none. *Implementation:* Tk canvas/scale; AnnotationIndex for tick positions.  
- **\[Plain\]** *How it works:* A slider runs the whole clip; red marks show frames you already labelled and a black line shows where you are. *Purpose:* move through a clip and see what's done. *Parameters:* none.

### 3.6 Annotation controls

- **\[Technical\]**  
  - *How it works:* **Box-first workflow with sticky labels.** Left-drag draws a box; the active **primary** hotkey is sticky (selecting it relabels the current box and stays active for subsequent boxes); a **secondary** hotkey applies only when it is in the active primary's allowed set (per `secondary_map`) and **toggles off on repeat** — secondary is optional per box (sentinel −1 = none, reserved hotkey **`n`**). **Escape resets** the sticky primary+secondary selection (it does not skip). Right-click deletes the innermost box/mask under the cursor; g toggles grey-mask mode; u undoes; Space toggles view; **Enter** saves and advances; **Shift+Enter** returns to the last annotated frame; **Ctrl+P** skips the current frame without saving and advances; Delete deletes all files for the frame (Enter confirm / Escape cancel); arrows step ±1, Shift\+arrows ±10, Ctrl\+arrows jump between annotated frames. Enter/Ctrl+P advance to the next frame according to the active **Source** mode (random or CSV time-codes, §3.1).  
  - *Purpose:* Full keyboard-driven box/mask editing and navigation, with a quick box→primary→(optional)secondary labelling rhythm.  
  - *Parameters:* per-class primary/secondary \*\_hotkeys; reserved u, g, n.  
  - *Implementation:* Tk event bindings (on\_key\_all; Control mask = event.state & 0x4); AnnotatorTk select\_primary/select\_secondary/reset\_selection; allowed\_secondary\_idx from behaveai\_config; load\_next\_target() routes advancement by Source mode; non\_max\_suppression(), iou() for overlap handling.  
- **\[Plain\]**  
  - *How it works:* Draw a box, then press a main category's key (it stays selected for the next boxes); optionally add a finer category if it's allowed for that main one (press again to remove it). Escape clears the current selection, right-click removes a box, g for grey cover, u to undo, Space to switch views, Enter to save and move on, Ctrl+P to skip a frame, and arrow keys to step through frames.  
  - *Purpose:* Label and move around entirely from the keyboard.  
  - *Parameters:* your chosen shortcut keys.

### 3.7 Auto-annotation (model-assisted labelling)

- **\[Technical\]**  
  - *How it works:* If best.pt exists for primary static and/or motion, models load at startup and pre-populate boxes on unannotated frames; in hierarchical mode secondary classifiers run on each crop. Results are editable/clearable. Overlapping predictions are reduced with NMS.  
  - *Purpose:* Bootstrap labelling speed using the current models (active-learning loop).  
  - *Parameters:* primary\_conf\_thresh, secondary\_conf\_thresh.  
  - *Implementation:* ultralytics.YOLO, auto\_annotate\_local(), non\_max\_suppression(), iou().  
- **\[Plain\]**  
  - *How it works:* If trained files exist, the tool pre-draws its best guesses on new frames; you accept, fix, or clear them. In the two-tier mode it also guesses the finer category.  
  - *Purpose:* Speed up labelling by starting from the model's guesses.  
  - *Parameters:* the confidence thresholds.

### 3.8 Saving logic & on-disk layout

- **\[Technical\]**  
  - *How it works:* The frame is routed to the train or val side by the whole-video holdout of §3.9, then static and motion images are saved to their own dirs. motion\_blocks\_static/static\_blocks\_motion apply grey blocking before save. save\_empty\_frames keeps box-less frames as negatives. Grey-box coords go to .mask.txt; when a box carries a secondary label, its crop is saved to the **pooled** path annot\_\<stream\>\_crop/\<secondary\>/ (one shared crop dataset per stream, keyed by the shared secondary label); a box left without a secondary writes a negative crop to annot\_\<stream\>\_crop/\_\_none\_\_/ (§6.3). Species and age crops go to the project-wide annot\_species\_crop/ and annot\_age\_crop/. Crop filenames follow \<video\_label\>\_\<frame\>\_\<x1\>\_\<y1\>.jpg, which is how a crop's source video is recovered later. Existing annotations are overwritten; counts print to console.  
  - *Purpose:* Produce YOLO-format datasets for both streams (and pooled crop datasets feeding the two per-stream secondary classifiers plus the species/age models) with reproducible splits.  
  - *Parameters:* val\_frequency, save\_empty\_frames, motion\_blocks\_static, static\_blocks\_motion.  
  - *Implementation:* save\_annotation(); behaveai\_holdout.is\_holdout\_video(); cv2.imwrite, os; YOLO label format.  
- **\[Plain\]**  
  - *How it works:* Each labelled frame goes to training or validation according to which video it came from (§3.9), the photo and motion versions are saved separately, optional grey-covering is applied, empty frames can be kept as examples of "nothing here", and finer-category cut-outs are saved in their own folders.  
  - *Purpose:* Build the labelled datasets the models train on.  
  - *Parameters:* the validation fraction, keep-empty option, and the two blocking switches.

### 3.9 Train/validation partition: whole-video holdout (behaveai\_holdout.py)

- **\[Technical\]**  
  - *How it works:* **One partition governs every model in the pipeline, and its unit is the whole video, never the frame or the crop.** is\_holdout\_video(stem, val\_frequency) takes a SHA-256 of the video's filename stem (with a fixed salt), maps its first 32 bits to \[0,1\) and holds the video out when that value falls below val\_frequency. Nothing is stored: an existing video's side never changes, and a new video is classified the first time it is seen. Consumers: the **annotation tool** routes each saved frame to annot\_\<stream\>/images/{train,val} (§3.8); the **crop classifiers** (two per-stream secondary models, plus species and age) get the same partition at training time from build\_classification\_split(), which materialises annot\_\<pool\>\_split/{train,val}/\<class\>/ from the flat pooled crop folders — a crop's source video is recovered from its filename by video\_label\_for\_crop(); the **complex-behaviour model** excludes the same videos from training and scores on them (§6.12); and **BehaveAI\_evaluate\_detection.py** uses them as the frozen test split. build\_classification\_split() rebuilds from scratch behind a SHA-256 fingerprint of every pooled filename plus the fraction, so a class deleted from the ethogram or a removed crop cannot survive in the split; it creates every class folder on **both** sides even when empty, so the train and val ImageFolder scans agree on the class indices. It exists because Ultralytics, handed a bare class-folder dataset, runs its own split\_classify\_dataset(): **per crop, unseeded, and copied into a `_split` directory it creates with exist\_ok and never cleans** — successive retrains reshuffle into the previous run's files, so the same crop ends up on both sides. Two situations a per-video split cannot fix are reported rather than papered over by moving crops across the boundary: a class with no training crop (unlearnable) and a class absent from the holdout (trained but never validated).  
  - *Purpose:* Make validation honest. Frames a few hundredths of a second apart are near duplicates, so a per-frame or per-crop split scores the model on images it has effectively already trained on; holding out whole videos means every reported metric is measured on footage the model has never seen. Sharing one partition across detectors, crop classifiers and the complex model also lets their metrics be quoted together, and makes the whole split reproducible from val\_frequency alone.  
  - *Parameters:* val\_frequency (fraction of videos held out; the realised frame/crop percentage differs from it, since videos carry unequal numbers of annotations — quote the *video* split as the design and the frame/crop counts as what it yielded).  
  - *Implementation:* behaveai\_holdout.py — is\_holdout\_video(), holdout\_status(), split\_groups(), video\_label\_for\_annotation(), video\_label\_for\_crop(), build\_classification\_split(); hashlib, os, shutil. Called from BehaveAI\_annotation.py, BehaveAI\_classify\_track.py, BehaveAI\_complex\_model.py, BehaveAI\_evaluate\_detection.py, BehaveAI.py (project summary) and behaveai\_rebalance\_holdout.py.  
- **\[Plain\]**  
  - *How it works:* Whole videos — not individual frames — are set aside for testing. Which videos is decided from their file name, so the choice never drifts: a video already in the set stays in it, and a new video is sorted automatically the first time it appears. Everything the software trains uses that same set of held-back videos.  
  - *Purpose:* Two frames taken a fraction of a second apart are almost the same picture. If they were spread between the training and the testing pile, the software would be tested on pictures it had already studied, and would look far better than it is. Keeping whole videos aside means the scores come from footage it has genuinely never seen.  
  - *Parameters:* the fraction of videos to hold back.

---

## 4\. DATASET INSPECTOR (BehaveAI\_inspect\_dataset.py)

### 4.1 Library navigation & editing

- **\[Technical\]**  
  - *How it works:* Loads all annotated frames from the four directories (static/motion × train/val) into one ordered library; a slider spans it; arrows step ±1, Shift\+arrows ±10. The display reuses the annotation composite (main panel \+ zoom column \+ bottom bar; Space toggles view). Editing controls match the annotator (draw/delete/grey/u/g/Enter save/Delete remove). It loads the source video for the animation panel; if absent, repeats the static image. Secondary crops are rewritten on save in the same pooled annot\_\<stream\>\_crop/\<secondary|\_\_none\_\_\>/ layout, routed to the primary's own stream.  
  - *Purpose:* Review and correct the whole labelled corpus after the fact.  
  - *Parameters:* same overlay parameters; hierarchical mode toggles crop handling.  
  - *Implementation:* list\_images\_labels\_and\_masks(), load\_item(), save\_annotation\_and\_overwrite\_current(), draw\_boxes(), draw\_zoom(); AnnotationIndex, cv2, PIL.  
- **\[Plain\]**  
  - *How it works:* All your labelled frames are loaded into one list you scroll through with arrows. You can redraw, delete, and re-save boxes exactly as in the labelling tool, with the same zoom panels and motion playback.  
  - *Purpose:* Go back over everything you labelled and fix mistakes.  
  - *Parameters:* the same display options.

---

## 5\. DATA AUGMENTATION (BehaveAI\_augmentation.py)

### 5.1 Probabilistic offline augmentation

- **\[Technical\]**  
  - *How it works:* Scans original images in annot\_static/annot\_motion (train+val), skipping any with \_aug\_ in the name. When aug\_target\_classes is non-empty, only images containing at least one annotation of a listed class are considered. Each image first passes the global gate aug\_global\_probability; then each transform independently passes its own aug\_\<name\>\_probability. Every triggered transform writes one independent copy \<basename\>\_aug\_\<param\>.\<ext\> (originals untouched; re-runs are idempotent/overwriting). Parameter values are sampled from the configured range, which supports **multi-segment syntax** (`0.5,0.8 | 1.0 | 1.2,1.6`): each segment produces one independent copy.  
  - *Purpose:* Enlarge and diversify the dataset to improve generalisation and class balance, fully offline and inspectable.  
  - *Parameters:* aug\_global\_probability, aug\_target\_classes, aug\_\<name\>\_range, aug\_\<name\>\_probability, aug\_flip\_\*\_options.  
  - *Implementation:* apply\_augmentation\_to\_all\_annotations(), sample\_augmentation\_list(), \_parse\_segments(), \_sample\_segment(); random, numpy.  
- **\[Plain\]**  
  - *How it works:* It looks at your original labelled images and, image by image, decides by chance whether to make modified copies and which effects to apply. Each chosen effect makes its own separate copy; originals are never changed. You can restrict the whole process to images containing particular categories.  
  - *Purpose:* Add varied copies so the models cope better with real-world variation.  
  - *Parameters:* the master chance, the target categories, and each effect's chance and strength.

### 5.2 Transform set & per-transform behaviour

- **\[Technical\]**  
  - *How it works:* Photometric — brightness/contrast/saturation/sharpness via PIL.ImageEnhance; blur via PIL.ImageFilter.GaussianBlur (radius forced odd); hue via OpenCV HSV channel shift (mod 180); noise \= additive integer noise in ±value clipped to 0–255; temperature \= \+value red / −value blue. Geometric — flip\_h/flip\_v via PIL transpose; shear via PIL.Image.AFFINE (bilinear).  
  - *Purpose:* Cover colour, sharpness, sensor-noise, and geometric variability.  
  - *Parameters:* each transform's range; flip options.  
  - *Implementation:* apply\_single\_augmentation(); PIL (ImageEnhance, ImageFilter, Image), cv2, numpy.  
- **\[Plain\]**  
  - *How it works:* Effects include brighter/darker, more/less contrast or colour, sharper/blurrier, colour-shift, random speckle, warmer/cooler tint, mirror flips, and slanting.  
  - *Purpose:* Reproduce the range of looks real footage can have.  
  - *Parameters:* each effect's range.

### 5.3 Motion-image protection

- **\[Technical\]**  
  - *How it works:* For annot\_motion images, when motion\_disable\_color\_aug is on (default), only geometry/texture transforms (blur, noise, sharpness, shear, flip\_h, flip\_v) are allowed; colour transforms (brightness, contrast, saturation, hue, temperature) are excluded because the false-colour encoding carries the motion signal and altering colour would corrupt it.  
  - *Purpose:* Preserve the meaning of the motion stream under augmentation.  
  - *Parameters:* motion\_disable\_color\_aug (plus the source directory).  
  - *Implementation:* transform filtering in apply\_augmentation\_to\_all\_annotations().  
- **\[Plain\]**  
  - *How it works:* Motion images only get shape/texture effects, never colour changes, because their colours mean something specific.  
  - *Purpose:* Avoid destroying the motion information.  
  - *Parameters:* the colour-augmentation switch for motion images.

### 5.4 Label transformation & progress reporting

- **\[Technical\]** *How it works:* Only geometric transforms update YOLO labels — flip\_h: xc→1-xc; flip\_v: yc→1-yc; all others leave labels unchanged. Progress prints XX% |\#\#\#---| N/total filename, which the launcher overwrites in place. *Purpose:* keep labels valid after geometry; give live feedback. *Parameters:* none. *Implementation:* transform\_labels(), \_print\_progress().  
- **\[Plain\]** *How it works:* When an image is flipped, its box coordinates are flipped to match; other effects don't move boxes. A live progress bar is printed. *Purpose:* keep boxes correct and show progress. *Parameters:* none.

---

## 6\. TRAIN & BATCH CLASSIFY (BehaveAI\_classify\_track.py)

### 6.1 Conditional (re)training with transfer learning

- **\[Technical\]**  
  - *How it works:* maybe\_retrain() runs per model before processing. No model → train from scratch from the base classifier for the configured epochs. Existing model → compare train\_count.txt to the current dataset image count; if it changed, retrain automatically (no confirmation dialog, so headless runs never hang). On retrain → back up the model dir to \*\_backupN, fine-tune from the backed-up weights (transfer learning), update train\_count.txt. A copy of the INI (saved\_settings.ini) is stored per model for reproducibility. YOLO output is relocated from runs/detect/\*/train via move\_to\_expected(). Each model is trained **in its own process** (BehaveAI\_train\_worker.py, invoked with a JSON config): training several models sequentially in one process intermittently crashed with `CUDA error: resource already mapped` on recent GPUs, because the second training inherited a corrupted CUDA context from the first; a fresh process gives every model a clean context. A **secondary classifier whose pooled crop dataset has fewer than 2 sub-classes is skipped with a clear warning** (YOLO classification needs ≥2 classes) instead of erroring. Each **crop classifier** (secondary static/motion, species, age) is handed the whole-video split built by behaveai\_holdout.build\_classification\_split() rather than its flat crop pool, so Ultralytics never falls back to its own per-crop random split (§3.9); the detectors already receive train/val directories the annotation tool partitioned the same way. train\_patience sets early stopping.  
  - *Purpose:* Avoid redundant retraining, support incremental dataset growth, run unattended/headless, and guarantee reproducible model provenance.  
  - *Parameters:* primary\_classifier, primary\_epochs, secondary\_classifier, secondary\_epochs, train\_patience, val\_frequency (the split every model trains against). Detector image size is the model's native default; crop classifiers train at 224 px.  
  - *Implementation:* ultralytics.YOLO(...).train() inside BehaveAI\_train\_worker.py, count\_images\_in\_dataset(), move\_to\_expected(), behaveai\_holdout.build\_classification\_split(); subprocess, shutil, glob, configparser.  
- **\[Plain\]**  
  - *How it works:* Before processing, each model is checked: if it is missing it is trained; if the dataset changed it is retrained automatically, continuing from the old version, which is backed up first. Each model is trained in its own separate program run so one training cannot corrupt the next. A copy of the settings is saved next to each model.  
  - *Purpose:* Only retrain when needed and always keep a record of how each model was made.  
  - *Parameters:* base models, epochs, early-stopping patience.

### 6.2 NCNN export & loading

- **\[Technical\]** *How it works:* If use\_ncnn=true, checks for \<weights\>\_ncnn\_model/; if absent calls model.export(format="ncnn") (waits ≤300 s), then loads NCNN for inference, falling back to .pt on failure. Applied to all primary and secondary models. *Purpose:* faster CPU/ARM inference. *Parameters:* use\_ncnn. *Implementation:* ensure\_ncnn\_export(), load\_model\_with\_ncnn\_preference(), ncnn\_dir\_for\_weights(), ncnn\_files\_exist(); ultralytics, ncnn.  
- **\[Plain\]** *How it works:* Optionally converts models to a format that runs faster on CPUs and small devices, reusing an existing conversion if present and falling back to the original if conversion fails. *Purpose:* faster running on modest hardware. *Parameters:* the NCNN switch.

### 6.3 Two-stream detection \+ merging \+ hierarchical classification

> **Optional secondary via a `__none__` class.** The secondary classifier includes a reserved **`__none__`** class so it can answer *"no secondary"* instead of being forced to emit one. In the annotation tool, secondary-eligible primaries default to **none** (shown as the first *none* button; hotkey **`n`**, a reserved key); leaving a box untouched writes a negative crop to `annot_<stream>_crop/__none__/`. Training picks up `__none__` as an extra pooled class; at inference `__none__` competes in the arg-max and, when it wins, the box gets no secondary.

- **\[Technical\]**  
  - *How it works:* Per frame: compute the motion image (§3.3); run primary static YOLO on the raw frame and primary motion YOLO on the motion image (optionally tiled, §6.4). Merge across streams: two detections merge if centroid distance \< centroid\_merge\_thresh or box overlap \> iou\_thresh; the survivor is chosen by dominant\_source (confidence keeps higher score, else fixed motion/static preference). For each merged detection, crop the box and run the matching secondary classifier (static→static model, motion→motion model, with fallback to the other), keeping results above secondary\_conf\_thresh; the same crop also feeds the **species** (model 0) and **age** (model 0.5) classifiers when those are configured.  
  - *Purpose:* Combine the complementary strengths of appearance (static) and movement (motion) detection, then refine each detection with a fine-grained behaviour class plus species and age.  
  - *Parameters:* primary\_conf\_thresh, secondary\_conf\_thresh, centroid\_merge\_thresh, iou\_thresh, dominant\_source, secondary\_map.  
  - *Implementation:* ultralytics.YOLO, iou(), per-frame loop in process\_video(); cv2, numpy.  
- **\[Plain\]**  
  - *How it works:* Each frame is searched twice — once on the photo, once on the motion image. When both find the same object, the duplicates are merged and the better one is kept. Each kept object is then cut out and passed to further models that assign a finer behaviour, the species, and the age.  
  - *Purpose:* Use both appearance and movement to detect more reliably, then label each find more precisely.  
  - *Parameters:* the confidence thresholds, the merge distances, and the tie-break rule.

### 6.4 SAHI sliced inference & tiled training (BehaveAI\_tiling.py)

- **\[Technical\]**  
  - *How it works:* Optional. With sahi\_enabled\_static/motion, the frame is diced into sahi\_slice\_width × sahi\_slice\_height tiles overlapping by sahi\_overlap\_\*\_ratio; the detector runs on each tile at native resolution and the tile results are merged (sahi\_postprocess\_type, \_match\_metric, \_match\_threshold), optionally together with a whole-frame pass (sahi\_perform\_standard\_pred). Tiling is skipped — with a printed notice — when the frame is not larger than one tile × sahi\_min\_dim\_factor, since it would then be pure overhead. Because a detector trained the usual way (whole 4 K frame resized to 640, where a horse is ~10 px) has never seen the ~60 px horses SAHI feeds it, **the dataset must be tiled too**: BehaveAI\_tiling.py reads the YOLO data.yaml, slices every annotated image into the same tiles, remaps every label into each tile, and writes a parallel `<src>_tiled/` dataset plus its data.yaml; the pipeline then trains and infers from the `*_tiled` project. Both halves are needed — enabling sliced inference against a whole-frame-trained model collapses detection.  
  - *Purpose:* Recover small/distant subjects by presenting them to the detector near the scale it was trained on, without changing the model.  
  - *Parameters:* sahi\_enabled\_static, sahi\_enabled\_motion, sahi\_slice\_width, sahi\_slice\_height, sahi\_overlap\_width\_ratio, sahi\_overlap\_height\_ratio, sahi\_perform\_standard\_pred, sahi\_postprocess\_type, sahi\_postprocess\_match\_metric, sahi\_postprocess\_match\_threshold, sahi\_min\_dim\_factor (see §11).  
  - *Implementation:* sahi (AutoDetectionModel, get\_sliced\_prediction) via build\_sahi\_model(); BehaveAI\_tiling.tile\_dataset(); cv2, numpy. Imported lazily, so the pipeline runs without SAHI installed until a project turns tiling on.  
- **\[Plain\]**  
  - *How it works:* Instead of shrinking a huge frame to a small square (which turns a horse into a handful of pixels), the frame is cut into overlapping pieces and each piece is searched at full resolution; the results are then stitched back together. The training images are cut the same way, so the model learns at the size it will be used at.  
  - *Purpose:* Find animals that are too small to detect in a shrunken whole frame.  
  - *Parameters:* the tile size and overlap, and the rule for merging results between tiles.

### 6.5 Multi-object tracking

- **\[Technical\]**  
  - *How it works:* tracker\_type selects the tracker. **botsort** (default) and **bytetrack** are the Ultralytics implementations: their YAML defaults are loaded and then overridden with the project's tracker\_\* keys. Both use a Kalman prediction plus **two-tier association** — a first pass on high-score detections (tracker\_track\_high\_thresh) and a second on low-score ones (tracker\_track\_low\_thresh) to recover partially occluded animals — gated by tracker\_match\_thresh; a lost track survives tracker\_track\_buffer frames. BoT-SORT adds **global motion compensation** (tracker\_gmc\_method, sparseOptFlow by default), essential on a drone: without it the drone's own displacement reads as animal displacement. Association is **kinematic only** — no appearance descriptor is computed at any point, because appearance carries no usable signal on a ~60×35 px horse at 15–50 m with uniform coats. Long-gap identity recovery is instead the offline stitching pass's job (§6.6), so track\_buffer stays short here. **kalman** keeps the project's own tracker for comparison: one constant-velocity Kalman filter per individual (4D state x,y,vx,vy; 2D measurement x,y), a cost matrix of predicted-vs-detection Euclidean distances solved by the Hungarian algorithm (linear\_sum\_assignment) and gated by match\_distance\_thresh; unmatched detections start new tracks; unmatched tracks accrue missed frames and inflated process noise until delete\_after\_missed; near-coincident tracks are pruned (\_prune\_duplicate\_tracks, threshold \= ½ match\_distance\_thresh). Motion vectors are drawn as arrows in the output video for every tracker.  
  - *Purpose:* Maintain stable identities across frames and recover briefly missed detections.  
  - *Parameters:* tracker\_type; tracker\_track\_high\_thresh, tracker\_track\_low\_thresh, tracker\_new\_track\_thresh, tracker\_track\_buffer, tracker\_match\_thresh, tracker\_gmc\_method (Ultralytics trackers); match\_distance\_thresh, delete\_after\_missed, \[kalman\] process\_noise\_pos/process\_noise\_vel/measurement\_noise (legacy tracker).  
  - *Implementation:* ultralytics.trackers.bot\_sort.BOTSORT / byte\_tracker.BYTETracker (with lap for the assignment); cv2.KalmanFilter(4,2), scipy.optimize.linear\_sum\_assignment, numpy (np.hypot) for the legacy tracker.  
- **\[Plain\]**  
  - *How it works:* Each individual gets a predictor that estimates where it will be next; new detections are matched to the closest predictions, first the confident ones and then the doubtful ones. On drone footage the camera's own movement is measured and subtracted before matching. Unmatched detections become new individuals, and individuals that disappear for too long are dropped. The system never compares what the animals *look like* — only where they are and how they move.  
  - *Purpose:* Keep the same ID on the same individual over time.  
  - *Parameters:* the tracker choice and its matching thresholds.

### 6.6 Offline tracklet stitching (BehaveAI\_stitch\_tracklets.py)

- **\[Technical\]**  
  - *How it works:* The online tracker is causal: once it cuts a track or swaps an id it never revisits the decision. The video, however, is on disk, so this non-causal pass reads the whole clip and re-links the short, reliable tracklets into longer identities — **purely on kinematics, no appearance**. It reads a tracking CSV (preferring the drone-corrected one, whose x\_corrected/y\_corrected live in a single stabilised reference frame comparable across the whole clip), groups rows into tracklets by id (collapsing the duplicate static/motion rows that share a frame and id to their median position), and links compatible end→start pairs.  
    **Hard gates** — physics, not tuning: two tracklets overlapping in time are different animals; a displacement above the speed gate is impossible; a gap longer than stitch\_max\_gap\_seconds is not considered. With a flight log the gate is the physical stitch\_max\_speed\_m\_per\_s converted to pixels through the camera height (px/m ≤ f\_px/H everywhere on a ground plane, so the clip's minimum height gives a bound that can only ever be too permissive, never wrongly restrictive); without one, the stitch\_max\_speed\_px\_per\_frame cap applies.  
    **Soft decision** — a likelihood ratio between two explicit hypotheses, not a tuned threshold: *continuation* (the residual between A's damped constant-velocity prediction and B's start follows a 2-D Student-t of scale s(gap), with an exponential prior on occlusion duration) against *new track* (an unrelated animal appearing anywhere in the region the herd occupies, uniform over that area). A link is taken when the ratio favours continuation, i.e. cost \< 0, through a dummy-augmented linear assignment. stitch\_link\_prior\_log\_odds biases that decision deliberately; it is negative by default because a missed link costs statistical power while a wrong merge invents an animal and corrupts two budgets.  
    **The motion model is measured, not assumed.** estimate\_motion\_noise() runs the linking predictor *inside* tracklets, where the answer is known, at several lags, and fits both the scale law (median² = m0² \+ α·lag^β, β fitted — it measures ≈1.6–1.8, not the 3 an unconstrained constant-velocity model would give) and the tail (Student-t degrees of freedom). The tail matters: the residual is strongly non-Gaussian — median 3 px at half a second against a 90th percentile of 160 px — and a Gaussian fitted to that core rejects 42 % of genuine re-links. Everything used is written to a JSON report beside the text one, with the input file's hash, so any result can be audited.  
    The resulting group size K is **reported as a diagnostic, never imposed as a constraint**. Output: \<video\>\_tracking\_stitched.csv, \_stitch\_report.txt, \_stitch\_report.json.  
  - *Known limit, measured not assumed:* the deciding factor is how tightly packed the herd is. The oracle benchmark (below) measured 1–3 % contamination on a clip whose animals sit 441 px apart and **29 % on one at 198 px, at every setting tried** — when spacing is small, kinematics alone cannot identify the continuation and no threshold repairs it. The report prints the median nearest-neighbour spacing so this failure mode is visible rather than silent.  
  - *Calibration tool:* **BehaveAI\_stitch\_oracle.py** cuts real trajectories at controlled gap lengths and scores recovery, contamination and chain purity as a function of gap, which is where stitch\_max\_gap\_seconds and the prior should be read off. It isolates the association step only (the tracker's own id swaps are inherited as truth), so it calibrates and bounds — it is not a substitute for hand-annotated MOT ground truth (EVALUATION\_PLAN §C).  
  - *Purpose:* Recover the identity continuity a causal tracker cannot, without appearance and without assuming how many animals are present.  
  - *Parameters:* stitch\_enabled, stitch\_max\_speed\_m\_per\_s, stitch\_max\_speed\_px\_per\_frame, stitch\_speed\_gate\_margin, stitch\_max\_gap\_seconds, stitch\_gap\_prior\_seconds, stitch\_extrapolation\_horizon\_seconds, stitch\_link\_prior\_log\_odds, stitch\_min\_tracklet\_len, stitch\_quality\_gate (see §11). stitch\_max\_link\_cost is **obsolete**: an INI still carrying it gets a notice and the key is ignored.  
  - *Implementation:* scipy.optimize.linear\_sum\_assignment and curve\_fit, numpy; stdlib csv. Auto-launched at the end of classify\_track when stitch\_enabled. Regression tests in tests/test\_stitch\_tracklets.py.  
- **\[Plain\]**  
  - *How it works:* Once the whole video has been processed, the software looks back over it and re-joins pieces of track that belong to the same animal — never joining two pieces that exist at the same time, never joining two that would need an impossible speed, and never bridging a gap longer than the configured limit. For each candidate it asks a question with two answers: is this more likely to be the same animal coming back, or a different animal turning up here? It only joins when the first answer wins. How far an animal typically drifts while out of sight is measured from the video itself, not assumed. It also reports how many distinct animals it ended up with, as a check, without forcing that number.  
  - *Important caveat:* when animals are packed close together, this cannot work reliably — in testing, roughly one join in three was wrong on a tightly packed herd, whatever the settings. Run the calibration tool on your own clips before trusting it.  
  - *Purpose:* Give each animal one continuous identity over the clip instead of several fragments.  
  - *Parameters:* the maximum plausible speed, the longest gap to bridge, and how much evidence is demanded before joining.

### 6.7 Outputs & the post-processing chain

- **\[Technical\]**  
  - *How it works:* Per video, writes \<name\>\_tracking.csv and, when detection\_video\_enabled is on (off by default), \<name\>\_detected.mp4 (boxes, labels, track IDs, motion arrows, frame counter). The CSV carries the 12 base columns (frame,id,x,y,primary\_static\_class,primary\_static\_conf,primary\_motion\_class,primary\_motion\_conf,secondary\_static\_class,secondary\_static\_conf,secondary\_motion\_class,secondary\_motion\_conf), then the four bounding-box columns x1,y1,x2,y2 ((x,y) remains the box midpoint), then species\_class,species\_conf,age\_class,age\_conf. Columns are only ever appended, so a reader of the first 12 (or 16) columns keeps working. After all videos, the post-processing chain runs in order, each stage gated by its own switch: **drone motion correction** (§6.8, drone\_correction\_enabled) → **tracklet stitching** (§6.6, stitch\_enabled) → **metric geometry** (§6.9, metric\_enabled) → **interaction graph** (§6.10, interaction\_graph\_enabled; skipped with an explanatory message when metric\_enabled is off, because it measures in metres) → **complex-behaviour stage** (§6.12, complex\_classify\_enabled; a no-op for a project with no complex annotations and no trained model) → **activity budget** (§7, always). Each stage is wrapped so a failure is reported without aborting the rest.  
  - *Purpose:* Provide both a human-checkable video and machine-readable per-frame tracks, then chain into every downstream analysis in one run.  
  - *Parameters:* the display style (§2.9) for the overlay; the stage switches above.  
  - *Implementation:* cv2.VideoWriter, csv, behaveai\_render.draw\_labeled\_detection(); calls run\_drone\_correction(), run\_stitch\_project(), run\_metric\_geometry(), run\_complex\_features(), run\_complex\_stage(), run\_activity\_budget().  
- **\[Plain\]**  
  - *How it works:* For each video it saves an annotated copy plus a table of every detection per frame, then automatically runs, in order, the camera-motion correction, the track re-joining, the conversion to metres, the interaction analysis, the group-behaviour recognition and finally the activity budget. If one step fails the others still run.  
  - *Purpose:* Give you a viewable result, a data file, and the full analysis in a single run.  
  - *Parameters:* overlay style and the per-stage switches.

---

## 6A\. HERDWISE MULTI-INDIVIDUAL / COMPLEX-BEHAVIOUR PIPELINE

*The modules below build on the per-individual tracking CSV to analyse **interactions between horses** in drone footage of herds. Everything from §6.10 onwards is in **real-world units** — metres and m/s — obtained by projecting each animal onto the ground plane from flight-log telemetry (§6.9). They therefore **require** metric\_enabled = true and a `.flightlog.csv` beside each video; a clip without usable metric geometry is skipped rather than measured in some other unit, because pixel- or body-length-normalised features are not comparable between clips flown at different altitudes. Foal/adult status comes from the trained **age classifier**, not from apparent size. The interaction graph (§6.10) is the primary scientific output.*

### 6.8 Drone motion correction (behaveai\_drone/drone\_correction.py)

- **\[Technical\]**  
  - *How it works:* A post-processing step over each \<video\>\_tracking.csv. The drone is normally held still but may pan/zoom to follow the herd, adding apparent (background) motion to every centroid and corrupting slow-horse velocities. For each consecutive frame pair it (1) masks out the tracked horses (their bounding boxes, dilated by drone\_correction\_box\_dilation) so flow is measured on the **static background only**; (2) estimates global background motion with sparse optical flow (goodFeaturesToTrack + calcOpticalFlowPyrLK) and a RANSAC-fitted global transform (drone\_correction\_model = affine partial-2D or homography); (3) chains the transforms and maps every centroid into one stabilised reference frame (frame 0); (4) recomputes velocities from the corrected, smoothed positions over the **real frame gap** (frame\_skip-aware). Per-frame quality is flagged ok / uncertain (residual flow std \> drone\_correction\_uncertain\_std) / none; when background features are persistently too few (\< drone\_correction\_min\_features) it falls back to smoothing-only (drone\_correction\_fallback\_smoothing). Centroid smoothing is savgol / moving\_average / none over an odd window.  
  - *Purpose:* Remove drone-induced apparent motion so per-horse kinematics (and everything downstream) reflect real movement.  
  - *Parameters:* drone\_correction\_enabled, drone\_correction\_model, drone\_correction\_box\_dilation, drone\_correction\_min\_features, drone\_correction\_uncertain\_std, drone\_correction\_smoothing, drone\_correction\_smoothing\_window, drone\_correction\_fallback\_smoothing (see §11).  
  - *Implementation:* cv2 (goodFeaturesToTrack, calcOpticalFlowPyrLK, estimateAffinePartial2D / findHomography with RANSAC), numpy, scipy.signal (savgol); reads/writes via stdlib csv. Output \<video\>\_tracking\_corrected.csv = original columns + x\_corrected, y\_corrected, vx\_corrected, vy\_corrected, correction\_quality (plus a \*\_correction\_diag.csv sidecar).  
- **\[Plain\]**  
  - *How it works:* When the drone pans or zooms, everything in the picture seems to move. The tool looks only at the still background to work out how much the camera moved, then subtracts that from each horse's apparent path, so the saved positions and speeds reflect the horses' real movement. It marks each frame as reliable or not, and falls back to gentle smoothing when the background gives too little to work with.  
  - *Purpose:* Make sure later analysis measures the horses, not the drone.  
  - *Parameters:* the drone-correction switches and thresholds (see §11).

### 6.9 Metric geometry: pixels → ground-plane metres (behaveai\_drone/metric\_geometry.py)

- **\[Technical\]**  
  - *How it works:* Turns each tracked animal's image position into ground-plane coordinates in **metres**. The chain is metres-per-pixel \= f(h, θ, f\_px, image row), with **h** the camera height above ground read from the flight log (rel\_alt\_m), **θ** the camera pitch read from the flight log (gb\_pitch — the tilt DJI SRT files lack), and **f\_px** the pixel focal length derived from the camera spec (metric\_focal\_len\_mm × frame\_width / metric\_sensor\_width\_mm) or from a per-drone calibration. Flight-log sidecars are read **by column name**, never by index, because two schemas coexist in the corpus (Mini4Pro 21 columns, Mini3/Mini3Pro 18) and reading by position silently shifts the gimbal fields (behaveai\_drone/flightlog.py). Rows within metric\_horizon\_margin\_px of the horizon are rejected (the projection diverges there), as are clips whose roll exceeds metric\_roll\_max\_deg. A **scale cross-check** compares the flight-log camera height against the height implied by the animals' apparent size (metric\_assumed\_body\_length\_m); disagreement beyond metric\_scale\_tolerance flags the clip `uncertain`, which catches a biased rel\_alt or sloped terrain — errors that bias *every* distance in the clip by the same percentage. That cross-check fits the clip as a *whole*, and the flight log describes only the *drone*, so ground that slopes under a perfectly steady gimbal passes both; a third check therefore refits the ground plane in sliding windows (metric\_geometry\_window\_s) and reports how far the horizon row travels between them (metric\_geometry\_drift\_frac), alongside a robust spread that distinguishes a steady drift from a few ill-conditioned windows. It is **warning-only**: herd scenes where the animals bunch at one depth can trip it while the geometry is sound, so it never downgrades `metric_quality` — confirm a flagged clip with `horizon_geometry.py <csv> --check-drift`. Two ground frames are emitted because they are valid for different things: **X\_m/Y\_m** (per-frame, camera-relative) for **distances** between animals seen together, and **Xs\_m/Ys\_m** (stabilised on one reference geometry, from the drone-corrected pixel) for **speeds** — differencing the camera-relative frame would return the drone's own motion. Output: \<video\>\_tracking\_metric.csv. A telemetry-free fallback exists for geometry work (behaveai\_drone/horizon\_geometry.py) which fits the horizon line and image scale from the apparent size of adult horses alone (size\_px = slope × (y\_base − y\_horizon)), and behaveai\_drone/srt\_telemetry.py parses DJI SRT sidecars tolerantly across firmware variants.  
  - *Purpose:* Make inter-individual distances and speeds real-world quantities, comparable between clips flown at different altitudes — the precondition for every feature in §6.10.  
  - *Parameters:* metric\_enabled, metric\_focal\_len\_mm, metric\_sensor\_width\_mm, metric\_horizon\_margin\_px, metric\_roll\_max\_deg, metric\_scale\_tolerance, metric\_assumed\_body\_length\_m, metric\_geometry\_drift\_frac, metric\_geometry\_window\_s (see §11). Requires a `<video>.flightlog.csv` sidecar.  
  - *Implementation:* behaveai\_drone/metric\_geometry.py (run\_metric\_geometry), flightlog.py (name-based column resolution), horizon\_geometry.py, srt\_telemetry.py; numpy, stdlib csv.  
- **\[Plain\]**  
  - *How it works:* Knowing how high the drone was, how far the camera was tilted, and the lens characteristics, each animal's position on the picture is converted into a position on the ground measured in metres. The tool also double-checks that height against how big the horses actually look, and flags the clip if the two disagree — which is what happens on sloping ground or with a wrong altitude reading.  
  - *Purpose:* Measure real distances and real speeds, so two videos filmed at different heights can be compared.  
  - *Parameters:* the lens and tolerance settings, plus the flight log recorded with each video.

### 6.10 Interaction graph: dyadic & group features (BehaveAI\_complex\_features.py)

- **\[Technical\]**  
  - *How it works:* The deterministic layer that turns a **metric** tracking CSV (\<video\>\_tracking\_metric.csv — REQUIRED) into per-frame **dyadic** and **group** features and aggregates them into the **interaction graph** — the primary analysis output, directly importable into R igraph. **Everything is in real metres and m/s**; a video without usable metric geometry is skipped rather than silently measured in another unit. The two ground frames of §6.9 are used for what each is valid for: X\_m/Y\_m for **distances**, Xs\_m/Ys\_m for **speeds**. Dyadic features: distance\_m, approach\_rate (m/s, negative = approaching), speed\_similarity (cosine of velocity vectors), heading\_diff, in\_contact (box IoU \> complex\_contact\_iou\_thresh or ground distance \< complex\_contact\_dist\_m), each endpoint's age class, plus both endpoints' YOLO primary + secondary labels. Direction-based features are zeroed below 0.05 m/s: a standing animal has no heading, and detection jitter would otherwise fill that field with noise. The **speed noise floor is also measured per clip** from the high-frequency residual of its own trajectories — differentiating position amplifies noise, so ~2 px of box jitter yields ~0.4 m/s of apparent speed on a standing horse — and speeds below the measured floor are treated as stationary and reported in the run log. Group features are computed over the **whole co-present herd per frame** (no spatial partitioning): polarisation, cohesion (m), convex-hull area (m²), centroid speed (m/s), behavioural synchrony, PCA-based elongation (capped at 100 to avoid inf). Foal/adult comes from the **age classifier** (model\_age) via the age\_class/age\_conf columns, as a per-track confidence-weighted majority vote — never re-derived from apparent box size; an individual the age model never labelled is `unknown`, not guessed. aggregate\_window() pools features (mean/std/min/max + a normalised label bag kept for the model in §6.12). build\_interaction\_graph() builds an **undirected** networkx Graph at interaction\_edge\_granularity = per\_dyad | per\_segment | per\_frame — every metric this layer computes is symmetric, so it emits one row per unordered pair, and direction is supplied afterwards by the model (interaction\_type + actor\_id) — with edge weight = sri | duration\_s | proximity\_m, all absolute and therefore comparable between videos. If networkx is missing, the edges/nodes CSVs are still written and only the graph-metric summary is skipped.  
  - *Purpose:* Produce an analysis-ready, R-igraph-importable description of who interacts with whom, how, and for how long.  
  - *Parameters:* interaction\_graph\_enabled, interaction\_edge\_granularity, interaction\_weight\_metric, complex\_max\_interaction\_distance\_m, complex\_min\_duration\_frames, complex\_contact\_iou\_thresh, complex\_contact\_dist\_m, complex\_window\_frames (see §11). Requires metric\_enabled = true.  
  - *Implementation:* numpy, scipy, networkx; stdlib csv (lists-of-dicts, **no pandas**). Outputs \<video\>\_interaction\_edges.csv (columns ordered frame\_start, frame\_end, source\_id, target\_id, …) and \<video\>\_interaction\_nodes.csv. In R: `graph_from_data_frame(e[, c("source_id","target_id", ...)], directed=FALSE, vertices=v)`. Edge files always carry n\_frames\_observed, n\_frames\_co\_present and duration\_s, so any association index can be recomputed in R without re-running the pipeline.  
- **\[Plain\]**  
  - *How it works:* For every pair of nearby horses, and for the herd as a whole, the tool measures things like how far apart they are **in real metres**, whether they are approaching or moving apart, whether they move alike, whether they are touching, and what each is doing. It first works out how much of the apparent movement is just measurement noise, and ignores anything below that. It then condenses all this into a network ("who interacts with whom") that can be opened directly in standard network-analysis software.  
  - *Purpose:* Turn raw tracks into a clear map of the herd's interactions.  
  - *Parameters:* the graph granularity/weight and the interaction/contact thresholds (see §11).

### 6.11 Complex-behaviour annotation tool (BehaveAI\_annotation\_complex.py)

- **\[Technical\]**  
  - *How it works:* A **separate** Tk + OpenCV annotation tool for multi-individual (dyadic / group) behaviours; the per-individual YOLO annotation tool is untouched. Five-step workflow: (1) load the video's tracking CSV (corrected if present), draw per-id coloured boxes, seek bar + arrow navigation; (2) click boxes to build an **ordered** selection (the order encodes role), supporting N individuals; (3) Start/End buttons (or manual entry), validated start \< end; (4) pick a behaviour from the INI complex\_behaviours list (each with a hotkey) and an optional confidence (high/medium/low); (5) save — append a row, with list / edit / delete of existing rows. All data/IO/validation helpers are pure functions (no import-time side effects); the Tk app is built only under \_\_main\_\_, so helpers are headless-testable. CSV frame N ↔ video frame N−1. A **Load candidates** / **Review selected candidate** action reads \<video\>\_complex\_candidates.csv (§6.13) so proposals can be loaded into the editor for confirmation. Launched from the launcher's *Annotate complex* button.  
  - *Purpose:* Collect the human labels of dyadic/group behaviours that the complex model (§6.12) trains on.  
  - *Parameters:* complex\_behaviours, complex\_behaviours\_hotkeys (see §11).  
  - *Implementation:* tkinter, cv2 (lazy-imported in main()); stdlib csv. Output \<video\>\_complex\_behaviours.csv: video\_filename, start\_frame, end\_frame, behaviour, track\_ids (ordered ';'-separated), annotator\_confidence, fps, frame\_width, frame\_height.  
- **\[Plain\]**  
  - *How it works:* A second labelling tool, just for behaviours that involve several horses at once. You scrub to the moment, click the horses involved in the order that matters (e.g. who chases whom), set the start and end, choose the behaviour and how sure you are, and save. You can also load auto-suggested moments and simply confirm or fix them.  
  - *Purpose:* Record examples of group/interaction behaviours to teach the system.  
  - *Parameters:* the list of complex behaviours and their shortcut keys.

### 6.12 Complex-behaviour model (BehaveAI\_complex\_model.py)

- **\[Technical\]**  
  - *How it works:* Trains a classifier on the human complex-behaviour annotations using the windowed tabular features from §6.10, and predicts complex behaviours over a video. The **default** model (complex\_model\_type = baseline) is a scikit-learn DictVectorizer + RandomForest or HistGradientBoosting on fixed-size window feature vectors (robust with little data, interpretable). complex\_model\_type = **lstm / transformer** train a real **torch** sequence network over the per-timestep feature sequence (each labelled segment is sliced into complex\_seq\_steps sub-windows → one feature dict per timestep → DictVectorizer + per-feature standardisation → padded/masked bi-LSTM with masked-mean pool, or TransformerEncoder with learned positional embeddings + key-padding mask); they reuse the same features and by-video evaluation. **Asking for a sequence model without torch is an error, not a silent downgrade** to the baseline — substituting it would save an artefact contradicting its own saved\_settings.ini. The feature set deliberately fuses **geometry/graph features with YOLO's simple-behaviour labels** (primary and, when present, secondary, as a bag-of-labels over the window) for every involved individual, plus group synchrony. Evaluation splits **by video** (LeaveOneGroupOut, or GroupKFold for many videos) to avoid leaking individuals across train/val; reports per-class F1, macro-F1 and a confusion matrix; handles class imbalance with balanced sample weights; and down-weights windows whose drone-correction quality is not 'ok' (uncertain ×0.5). analyse\_confusion() turns by-video confusion into **merge suggestions** (class pairs confused above complex\_confusion\_merge\_rate) — it never auto-merges. classify\_video() slides windows over interacting pairs and the whole co-present herd, emits predictions above complex\_predict\_min\_proba, merges adjacent same-label runs, and populates interaction\_type in the edges file. Launched from the launcher's *Train complex model* button, and from the pipeline chain via run\_complex\_stage() (train-if-new + classify).  
  - *Purpose:* Learn to recognise and predict dyadic/group behaviours from the kinematic+graph+YOLO features, with honest by-video performance estimates.  
  - *Parameters:* complex\_classify\_enabled, complex\_model\_type, complex\_baseline\_classifier, complex\_seq\_steps, complex\_deep\_{epochs,hidden,layers,heads,dropout,lr,batch}, complex\_window\_frames, complex\_confusion\_merge\_rate, complex\_predict\_min\_proba (see §11).  
  - *Implementation:* scikit-learn (DictVectorizer, RandomForest / HistGradientBoosting, LeaveOneGroupOut / GroupKFold, cross\_val\_predict, f1\_score / confusion\_matrix, balanced sample weights), joblib; torch for the deep models. Saves model\_complex/: pipeline.joblib (+ deep\_model.pt for sequence models), train\_count.txt, saved\_settings.ini, metrics.txt, feature\_importances.txt, merge\_suggestions.txt. Output \<video\>\_complex\_predictions.csv: start\_frame, end\_frame, track\_ids, behaviour, probability.  
- **\[Plain\]**  
  - *How it works:* Using your labelled examples, the system learns to spot group behaviours from how the horses move and what the per-horse detector already says they're doing. The simple default learner works well even with few examples; deep-learning models are available when torch is installed, and asking for one without it is reported as an error rather than quietly replaced. Performance is always estimated by holding out whole videos, so the scores are honest, and the tool suggests which behaviour pairs it keeps confusing (you decide whether to merge them).  
  - *Purpose:* Automatically label group/interaction behaviours across whole videos, with trustworthy accuracy figures.  
  - *Parameters:* the model type, learner, and window/threshold settings (see §11).

### 6.13 Candidate proposal: heuristics + active learning (BehaveAI\_complex\_candidates.py)

- **\[Technical\]**  
  - *How it works:* Run after the first hand-annotations / a trained model exist, to accelerate further annotation. Two complementary sources: (1) **heuristic rules** over the §6.10 feature streams (frame-gap-tolerant run detection), thresholds from the INI and meant to be calibrated on the first annotations — allogrooming (in\_contact + both speeds ≈ 0, sustained), chase (high speed\_similarity + ≈ constant distance + both speeds high), stampede (high whole-herd mean speed + high polarisation), trek (high polarisation + moderate speed + non-zero centroid speed), synchronised\_rest\_graze (speed ≈ 0 + high synchrony + low dispersion); only behaviours present in the configured complex\_behaviours list are proposed. (2) **Active learning** with the trained §6.12 model: the most **uncertain** windows (lowest max-probability) over pairs and windows with ≥3 co-present individuals are surfaced, with the model's current best guess as the suggested label (top complex\_candidate\_topk). Heuristic-only when no model exists. Launched from the launcher's *Propose candidates* button; output is loaded by the annotation tool's *Load candidates* (§6.11) for confirmation — candidates are never consumed automatically.  
  - *Purpose:* Point the annotator at the segments most worth labelling next — both rule-flagged events and the model's least-certain windows.  
  - *Parameters:* complex\_speed\_low\_ms, complex\_speed\_high\_ms, complex\_polarisation\_high, complex\_synchrony\_high, complex\_candidate\_topk, plus the contact/distance thresholds from §6.10 (see §11).  
  - *Implementation:* numpy; reuses BehaveAI\_complex\_features and BehaveAI\_complex\_model; stdlib csv. Output \<video\>\_complex\_candidates.csv (same schema as §6.11, with annotator\_confidence = 'auto').  
- **\[Plain\]**  
  - *How it works:* The system points you at the moments most worth labelling next: events that match simple rules (like two still horses in contact = grooming, or a fast aligned group = stampede), and — once a model exists — the moments the model is least sure about. You then open these in the annotation tool and confirm or correct them.  
  - *Purpose:* Stop you hunting for examples by hand; label the most useful moments first.  
  - *Parameters:* the heuristic thresholds and how many uncertain moments to surface (see §11).

---

## 7\. ACTIVITY BUDGET ANALYSIS (BehaveAI\_activity\_budget.py)

### 7.1 Inputs & group-type extraction

- **\[Technical\]** *How it works:* Reads the most-processed tracking CSV available per video from output\_dir (metric → stitched → corrected → raw); optionally merges groups\_metadata.csv (manual group ID/type and excluded track IDs). Group type is parsed from the filename via ab\_group\_type\_separator \+ ab\_group\_type\_field\_index, overridable per video. ab\_analysis\_duration\_s optionally restricts the analysis to the first N seconds of each clip, so clips of unequal length stay comparable. *Purpose:* assemble per-video tracks and group metadata. *Parameters:* ab\_group\_type\_separator, ab\_group\_type\_field\_index, ab\_analysis\_duration\_s. *Implementation:* parse\_tracking\_csv(), extract\_group\_type(), load\_groups\_metadata(); csv, glob, configparser, collections.defaultdict.  
- **\[Plain\]** *How it works:* It reads the per-video tables (using the most refined version available) and works out each video's group type from its filename, with optional manual corrections, and can limit the analysis to the first part of each clip. *Purpose:* gather the data and group labels. *Parameters:* the filename separator and field position, and the analysis duration.

### 7.2 Stranger detection

- **\[Technical\]**  
  - *How it works:* Per track, presence is measured in **seconds**; tracks present for less than ab\_min\_presence\_seconds are flagged stranger. An entry point within ab\_edge\_margin\_px of the frame border combined with short presence adds short\_presence+border\_entry. A third criterion flags tracks with fewer than ab\_min\_classified\_frames classified frames (reason insufficient\_classified\_frames; 0 disables it). groups\_metadata.csv exclusions are flagged manual\_exclude. Strangers are kept with individual\_type \= stranger.  
  - *Purpose:* Separate resident group members from transient/edge intruders (and barely-classified noise tracks) without discarding their data.  
  - *Parameters:* ab\_min\_presence\_seconds, ab\_edge\_margin\_px, ab\_min\_classified\_frames.  
  - *Implementation:* flag\_strangers(); arithmetic over parsed tracks.  
- **\[Plain\]**  
  - *How it works:* An individual seen for too short a time is marked a stranger; if it also first appears at the edge of the image and stays briefly, that is noted; manual exclusions are honoured. Strangers are kept but labelled.  
  - *Purpose:* Tell regular members apart from passers-by.  
  - *Parameters:* the minimum presence in seconds and the edge-margin width.

### 7.3 Behaviour selection & per-individual budgets

- **\[Technical\]**  
  - *How it works:* Per track/frame, primary\_motion\_class is preferred when motion confidence ≥ static confidence, else primary\_static\_class (with empty-class fallback). compute\_individual\_budget() aggregates, per behaviour: behavior\_\<name\>\_s, \_n (entries), \_pct, plus dominant\_behavior\_time and dominant\_behavior\_count. Frames→seconds via frame\_to\_timecode() at the video FPS.  
  - *Purpose:* Quantify time and frequency of each behaviour per individual.  
  - *Parameters:* FPS (default 30), behaviour set (= class names).  
  - *Implementation:* compute\_individual\_budget(), frame\_to\_timecode().  
- **\[Plain\]**  
  - *How it works:* For each individual and frame it picks the most confident category (favouring the motion view on ties), then totals how long and how often each behaviour occurred and which dominated.  
  - *Purpose:* Measure each individual's time budget across behaviours.  
  - *Parameters:* the video frame rate and the behaviour list.

### 7.4 Output files

- **\[Technical\]**  
  - *How it works:* **activity\_budget\_individual.csv** — one row per individual per video (video\_filename, group\_id, group\_type, track\_id, individual\_type, auto\_flagged, n\_frames\_present, duration\_s, presence\_ratio, the behavior\_\* columns, dominant-time/-count). This file stays **deterministic**: it contains only what the per-individual pipeline measured. **activity\_budget\_predicted.csv** — the model-derived columns (complex\_\*, dominant\_interaction\_type) and the per-individual social metrics derived from the interaction edges, kept separate and joinable on video\_filename + track\_id, so a deterministic result is never mixed with a predicted one. **activity\_budget\_suspects.csv** — one row per flagged stranger (…flag\_reason, presence\_ratio, first/last\_seen\_frame, first/last\_seen\_timecode (MM:SS), border\_entry, auto\_flagged, manual\_exclude).  
  - *Purpose:* Deliver analysis-ready per-individual budgets, a separate predicted layer, and an audit list of flagged strangers.  
  - *Parameters:* inherited from §7.1–7.3.  
  - *Implementation:* run\_activity\_budget(), load\_complex\_predictions(), load\_interaction\_metrics(); csv.  
- **\[Plain\]**  
  - *How it works:* Three tables are written — one summarising every individual's measured behaviour times, one holding what the group-behaviour model predicted about them, and one listing every flagged stranger with when it was seen.  
  - *Purpose:* Provide the final summary, keep measured and predicted results apart, and give a checkable stranger list.  
  - *Parameters:* as above.

---

## 7A\. EVALUATION

*Three scripts score the pipeline against ground truth. They write into the project's `evaluation/` folder and are run manually — they are not part of the automatic chain. Shared helpers (project/INI resolution, box geometry, greedy IoU matching) live in behaveai\_eval\_common.py; stdlib csv + numpy only.*

### 7A.1 Detection: dual-stream ablation (BehaveAI\_evaluate\_detection.py)

- **\[Technical\]**  
  - *How it works:* A claim that a *dual-stream* detector helps has to show what the second stream buys, so three variants are scored on the **held-out videos** (§3.9): **static-only**, **motion-only**, and **merged** (the two combined by the pipeline's own merge rule), against a class-agnostic "union of annotated animals" ground truth — animal-finding is scored before classification, because detection and classification are separate stages by design. It also reports the per-stream behaviour **classification** quality (confusion matrix + per-class F1) over matched detections. It runs on the saved annotation images (annot\_\<stream\>/images), which are exactly the frames the models were evaluated against, so no re-extraction can introduce a mismatch.  
  - *Purpose:* Attribute detection performance to each stream, and separate finding an animal from labelling its behaviour.  
  - *Parameters:* primary\_conf\_thresh, val\_frequency (which videos are held out).  
  - *Implementation:* ultralytics.YOLO, behaveai\_holdout, behaveai\_eval\_common; numpy, csv.  
- **\[Plain\]** *How it works:* The two detectors are scored separately and then together, on videos the models have never seen, first on whether they find the animals at all and then on whether they name the behaviour correctly. *Purpose:* show what each of the two views actually contributes.

### 7A.2 Tracking: HOTA / DetA / AssA / IDF1 / MOTA / IDSW (BehaveAI\_evaluate\_tracking.py)

- **\[Technical\]**  
  - *How it works:* Wraps TrackEval (the HOTA reference implementation, which also yields CLEAR/MOTA and Identity/IDF1 in one run) to score any BehaveAI tracking CSV against a per-sequence ground truth in **MOT-Challenge format** (frame,id,bb\_left,bb\_top,w,h,conf,class,vis) made in an external tool (CVAT/DarkLabel). TrackEval scores only the frames present in the GT, so annotating at 1–2 fps suffices (~360 boxes for 3 min × 10 horses, not 54 000). Crucially it reports **DetA and AssA separately** (HOTA \= √(DetA × AssA)): detection/SAHI work moves DetA and tracker/stitching work moves AssA, so an ablation can only *attribute* a gain if the two are split.  
  - *Purpose:* Report standard MOT metrics, and make tracking improvements attributable to the component that produced them.  
  - *Parameters:* CLI — the ground-truth files and the tracking CSV(s) to score.  
  - *Implementation:* trackeval (installed from GitHub; a shim restores the removed numpy aliases it uses), stdlib csv.  
- **\[Plain\]** *How it works:* You hand-label a few frames per video in a standard format; the script compares the automatic tracks against them and reports the standard tracking scores, splitting "did it find the animals" from "did it keep their identities". *Purpose:* measure tracking quality with published, comparable metrics.

### 7A.3 Geometry: drone correction & metric scale (BehaveAI\_evaluate\_geometry.py)

- **\[Technical\]**  
  - *How it works:* Three checks. (1) **Synthetic recovery** — a textured frame is warped by *known* similarity transforms (translation / rotation / zoom / combined) and the pipeline's own estimator (drone\_correction.\_estimate\_step\_transform) must recover them; the reported reprojection error near zero means the estimator is sound. (2) **Correction-quality summary** over the existing \*\_tracking\_corrected.csv and \*\_correction\_diag.csv sidecars: the ok/uncertain/none breakdown plus the continuous residual distribution. (3) **Metric scale check** against the flight-log-derived geometry.  
  - *Purpose:* Validate the correction algorithm itself before any downstream kinematic claim rests on it — it is a real algorithm, not an aggregation, so its accuracy has to be shown.  
  - *Parameters:* the drone-correction and metric keys already in the INI.  
  - *Implementation:* behaveai\_drone.drone\_correction, behaveai\_eval\_common; cv2, numpy.  
- **\[Plain\]** *How it works:* The camera-motion correction is tested by moving a picture by a known amount and checking the software finds exactly that amount back, then by summarising how reliable it judged itself to be on the real videos. *Purpose:* prove the correction works before trusting any speed or distance built on it.

---

## 8\. LIVE CAMERA (BehaveAI\_live.py)

### 8.1 Camera backends

- **\[Technical\]**  
  - *How it works:* Enumerates OpenCV cameras by index (0–5). On Raspberry Pi OS, picamera2 is detected and exposed as picamera via Picamera2Wrapper (mimics cv2.VideoCapture; resolution/FPS via create\_video\_configuration(); up to 4 retry attempts; warm-up frames discarded).  
  - *Purpose:* Uniform capture API across USB/integrated webcams and the Pi camera.  
  - *Parameters:* selected camera index/picamera, resolution.  
  - *Implementation:* cv2.VideoCapture, Picamera2Wrapper, scan\_cameras(), is\_raspberry\_pi(); platform, picamera2.  
- **\[Plain\]**  
  - *How it works:* It finds connected cameras automatically; on a Raspberry Pi it also offers the Pi camera, made to behave like an ordinary webcam.  
  - *Purpose:* Use the same controls for any supported camera.  
  - *Parameters:* which camera and resolution.

### 8.2 Live controls, recording modes & indicator

- **\[Technical\]**  
  - *How it works:* Controls: camera/resolution selectors (locked after Start), *Enable classifier* (toggles inference live), *Manual recording* → clips/\<timestamp\>.mp4 (annotation-ready), *Record on detection* → output/det\_\<timestamp\>.mp4 (auto start/stop, stops after detection\_stop\_seconds\=3 s of no detections), *Show detections in recordings* (overlay vs clean), *Display stream* radio (Static/Motion/Disabled — Disabled saves CPU), live FPS (250 ms refresh), *Throttle FPS* (0=off), *Quit* (confirmed). On-screen REC icon labels REC M / REC D / REC M+D.  
  - *Purpose:* Real-time monitoring with flexible recording for both dataset capture and event archiving.  
  - *Parameters:* resolution, detection\_stop\_seconds, throttle FPS, display mode.  
  - *Implementation:* ControlGUI, CameraProcessor; threading, cv2, Tk ttk.  
- **\[Plain\]**  
  - *How it works:* You start a camera, optionally turn detection on, and choose recording: manual clips for later labelling, or automatic recording only while something is detected. You can show or hide boxes in recordings, pick which view to display (or none to save power), see live frame rate, and cap it.  
  - *Purpose:* Watch live and record either for data collection or to capture events.  
  - *Parameters:* resolution, the no-detection stop delay, the frame-rate cap, and the view.

### 8.3 Live detection/tracking & logging

- **\[Technical\]** *How it works:* Same detection path as batch (§6.3: motion image, dual primary detection, merging, secondary classification, NCNN with fallback); tracking uses the module's own constant-velocity **Kalman tracker** (the Ultralytics trackers and the offline stitching pass are batch-only), and arrows are drawn live. A session CSV det\_\<timestamp\>.csv (same columns as batch) is written **only while "Record on detection" is on**, to avoid logging empty frames. *Purpose:* live inference with selective, non-redundant logging. *Parameters:* as §6. *Implementation:* KalmanTracker + YOLO; csv.  
- **\[Plain\]** *How it works:* Live video is processed like batch videos (with the simpler built-in tracker), and detection rows are written to a file only while detection-recording is on. *Purpose:* run the analysis live without logging empty frames. *Parameters:* same as batch.

---

## 9\. ANNOTATION INDEX (index\_annotations.py)

### 9.1 Shared dataset-indexing helper

- **\[Technical\]**  
  - *How it works:* AnnotationIndex is the common dataset layer for the annotator, inspector, and launcher. list\_images\_labels\_and\_masks() scans all four dirs and returns sorted item dicts (paths to static/motion image, label, mask \+ origins). load\_labels\_and\_masks\_for\_item() loads YOLO labels and masks into box/grey-box lists; in hierarchical mode \_attach\_secondary\_crops() matches crops to boxes by exact key (x1,y1,primary\_name) with a ±2 px fallback. load\_labels\_for\_basename() builds an item from a basename. find\_video\_for\_item() resolves the source clip and guesses the frame number. delete\_frame() removes every file for a basename across all dirs (returns deleted paths). \_norm\_to\_pixels() converts YOLO coords to clamped pixels.  
  - *Purpose:* Single source of truth for locating, loading, matching, and deleting annotation artefacts.  
  - *Parameters:* none (operates on configured dirs).  
  - *Implementation:* os, cv2, numpy.  
- **\[Plain\]**  
  - *How it works:* One shared component knows where every labelled image, label, mask, and cut-out lives. It can list them, load a frame's boxes, match cut-outs to boxes, find the original video, and delete everything for one frame at once.  
  - *Purpose:* Keep all tools consistent about finding and handling labelled files.  
  - *Parameters:* none.

---

## 10\. ANNOTATION REGENERATION (Regenerate\_annotations.py)

### 10.1 Motion-image rebuild

- **\[Technical\]**  
  - *How it works:* Rebuilds static and motion annotation images from the original clips using the **current** motion-strategy settings, preserving labels. For each annotated frame (from label files): locate the source video, regenerate static/motion images via the same window/diff/chromatic logic (generate\_base\_images()), re-apply grey masks and blocking boxes, and overwrite the .jpg in place. Label .txt files are never modified. Launched automatically by the Settings GUI when motion parameters change and annotations exist; accepts a project dir or INI path (else file dialog).  
  - *Purpose:* Keep the labelled image set consistent with changed motion parameters without re-labelling.  
  - *Parameters:* all motion-strategy keys (§11); motion\_blocks\_static/static\_blocks\_motion.  
  - *Implementation:* regenerate\_annotations(), generate\_base\_images(), apply\_grey\_boxes(), apply\_blocking\_boxes(), get\_blocking\_boxes(), read\_mask\_file(); cv2, numpy, glob, configparser.  
- **\[Plain\]**  
  - *How it works:* If you change how motion is computed, it rebuilds the motion (and static) images for every already-labelled frame from the original videos, keeping the boxes exactly as they were and only replacing the images.  
  - *Purpose:* Update existing labelled images to new motion settings without redoing any labelling.  
  - *Parameters:* the motion-strategy settings and the blocking switches.

---

## 10A\. MAINTENANCE UTILITIES

Small one-purpose scripts, run by hand from a project directory.

| Script | What it does |
| :---- | :---- |
| behaveai\_rebalance\_holdout.py | Re-splits existing annot\_static / annot\_motion train/val images (+ labels) **by whole video** using behaveai\_holdout.is\_holdout\_video, for datasets annotated before the per-video split. |
| migrate\_secondary\_crops.py | Moves legacy per-primary crop folders (annot\_\<stream\>\_crop/\<primary\>/\<secondary\>/) into the pooled layout (annot\_\<stream\>\_crop/\<secondary\>/). Dry-run by default, `--apply` to move. |
| BehaveAI\_bootstrap\_species.py | One-shot: back-fills annot\_species\_crop/ for a project whose primary annotations predate the species model, by cropping every existing box. Purely additive; self-deletes after a successful `--apply`. |
| make\_timecodes\_template.py | Writes the Google-Sheets-ready .xlsx time-code template (§3.1). |

---

## 11\. SETTINGS FILE (BehaveAI\_settings.ini) — Complete Parameter Reference

*All keys live in \[DEFAULT\] except the \[kalman\] section. For a multi-species project, every class-configuration key exists once per species: the first species keeps the bare key, the others use `<key>__<species_slug>` (§2.1).*

### Species, class & label configuration

| Parameter | Description |
| :---- | :---- |
| species\_list | Comma-separated species (scientific names) detected by model 0 |
| species\_hotkeys / species\_colors | Single-character hotkeys / RGB triples for the species list |
| age\_classes / age\_hotkeys / age\_colors | Age classes (model 0.5) and their hotkeys/colours, per species |
| primary\_motion\_classes | Comma-separated class names for primary motion detection |
| primary\_motion\_colors | Semicolon-separated RGB triples (e.g. 255,0,0;0,255,0) |
| primary\_motion\_hotkeys | Comma-separated single-character hotkeys (optional per class) |
| primary\_static\_classes / \_colors / \_hotkeys | As above for primary static detection |
| secondary\_classes | **Shared pool** of secondary sub-class names, reused across both streams |
| secondary\_colors / secondary\_hotkeys | RGB triples / hotkeys for the shared secondary pool |
| secondary\_map | Per-primary allowed secondaries: `Primary1:secA\|secB; Primary2:secA` — a primary with no entry has no secondary step |
| dominant\_source | confidence, motion, or static — which stream wins a merge |

### Paths

| Parameter | Default | Description |
| :---- | :---- | :---- |
| clips\_dir | clips | Training video clips directory |
| input\_dir | input | Batch input directory |
| output\_dir | output | Batch output directory |
| project\_path | (project dir) | Absolute path to project root |

### Motion strategy

| Parameter | Default | Description |
| :---- | :---- | :---- |
| strategy | exponential | exponential or sequential |
| chromatic\_tail\_only | false | Use differential colour encoding (difference between decay rates only) |
| expA | 0.5 | Green-channel decay rate; also sets frame-window size (0.2→5, \>0.5→10, \>0.7→15, \>0.8→20, \>0.9→45) |
| expB | 0.8 | Red-channel decay rate (longer memory) |
| lum\_weight | 0.7 | Greyscale luminance blend weight (0=pure colour, 1=pure grey) |
| rgb\_multipliers | 4,4,4 | R,G,B motion-channel amplifiers |
| frame\_skip | 0 | Raw frames skipped between sampled frames |
| scale\_factor | 1.0 | Spatial downscaling factor applied before differencing |
| motion\_threshold | 0 | Additive offset to motion values (negative suppresses low motion) |

### Annotation behaviour

| Parameter | Default | Description |
| :---- | :---- | :---- |
| val\_frequency | 0.1 | Fraction of **whole videos** held out, for every model in the pipeline (§3.9). The realised share of frames/crops differs, since videos carry unequal numbers of annotations |
| save\_empty\_frames | true | Save frames with no annotations (negative examples) |
| motion\_blocks\_static | false | Grey out motion boxes in static images |
| static\_blocks\_motion | false | Grey out static boxes in motion images |

### Model training

| Parameter | Default | Description |
| :---- | :---- | :---- |
| primary\_classifier | yolo11s.pt | Base detector for primary models (yolo8/11/26, n/s/m/l) |
| primary\_epochs | 50 | Training epoch cap for primary models |
| secondary\_classifier | yolo11s-cls.pt | Base classifier for the crop models (secondary, species, age) |
| secondary\_epochs | 50 | Training epoch cap for the crop models |
| train\_patience | 30 | Early-stopping patience (epochs without improvement) |
| use\_ncnn | false | Export and use NCNN format for inference |

### Inference

| Parameter | Default | Description |
| :---- | :---- | :---- |
| primary\_conf\_thresh | 0.5 | Primary detection confidence threshold (0–1) |
| secondary\_conf\_thresh | 0.5 | Secondary classification confidence threshold (0–1) |

### SAHI sliced inference (§6.4)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| sahi\_enabled\_static | false | Slice the static stream into tiles before detection |
| sahi\_enabled\_motion | false | Same for the motion stream |
| sahi\_slice\_width | 640 | Tile width (px) |
| sahi\_slice\_height | 640 | Tile height (px) |
| sahi\_overlap\_width\_ratio | 0.2 | Horizontal tile overlap, as a fraction of tile width |
| sahi\_overlap\_height\_ratio | 0.2 | Vertical tile overlap |
| sahi\_perform\_standard\_pred | false | Also run one whole-frame pass and merge it with the tiles |
| sahi\_postprocess\_type | NMS | Tile-merge post-process (NMS / GREEDYNMM / …) |
| sahi\_postprocess\_match\_metric | IOS | Overlap metric used to merge boxes across tiles |
| sahi\_postprocess\_match\_threshold | 0.5 | Overlap above which two tile boxes are merged |
| sahi\_min\_dim\_factor | 1.5 | Tiling is skipped unless the frame is larger than one tile × this factor |

### Tracking (§6.5)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| tracker\_type | botsort | botsort, bytetrack, or kalman (legacy) |
| tracker\_track\_high\_thresh | 0.5 | First-stage association threshold (high-score detections) |
| tracker\_track\_low\_thresh | 0.1 | Second-stage threshold, for recovering low-score detections |
| tracker\_new\_track\_thresh | 0.6 | Score below which an unmatched detection does not start a track |
| tracker\_track\_buffer | 30 | Frames a lost track is kept alive |
| tracker\_match\_thresh | 0.8 | Association (IoU/cost) similarity threshold |
| tracker\_gmc\_method | sparseOptFlow | BoT-SORT global motion compensation: sparseOptFlow, orb, sift, ecc, none |
| match\_distance\_thresh | 200 | *(kalman)* Max pixel distance for detection-to-track association |
| delete\_after\_missed | 5 | *(kalman)* Consecutive missed frames before a track is deleted |
| centroid\_merge\_thresh | 50 | Max centroid distance to merge static+motion detections |
| iou\_thresh | 0.95 | IOU threshold for detection merging / overlap handling |

### Offline tracklet stitching (§6.6)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| stitch\_enabled | false | Master switch; off \= no behaviour change |
| stitch\_max\_speed\_m\_per\_s | 17 | Hard physical gate, converted to pixels when a flight log gives the camera height |
| stitch\_max\_speed\_px\_per\_frame | 60 | Fallback hard gate in stabilised-frame pixels, used without a flight log |
| stitch\_speed\_gate\_margin | 1.5 | Safety factor on that conversion, covering barometric rel\_alt bias |
| stitch\_max\_gap\_seconds | 5 | Longest occlusion considered at all; derive it from the oracle benchmark |
| stitch\_gap\_prior\_seconds | 5 | Time constant of the exponential prior penalising long gaps |
| stitch\_extrapolation\_horizon\_seconds | 1 | Velocity extrapolation is damped past this horizon |
| stitch\_link\_prior\_log\_odds | \-5 | Bias on the link decision; 0 \= pure likelihood ratio, negative demands more evidence |
| stitch\_min\_tracklet\_len | 1 | Tracklets shorter than this are excluded from linking; their rows are kept with their own identity |
| stitch\_quality\_gate | true | Refuse links across frames the drone correction flagged unreliable |
| expected\_group\_size | 0 | Field-recorded herd size; reported in the stitch report, never a constraint |
| ~~stitch\_max\_link\_cost~~ | — | **Obsolete.** The decision is now a likelihood ratio with threshold 0; use stitch\_link\_prior\_log\_odds |

### Kalman filter \[kalman\] section

| Parameter | Default | Description |
| :---- | :---- | :---- |
| process\_noise\_pos | 0.01 | Positional process noise injected per frame |
| process\_noise\_vel | 0.1 | Velocity process noise injected per frame |
| measurement\_noise | 0.1 | Assumed centroid measurement noise |

### Drone motion correction (§6.8)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| drone\_correction\_enabled | false | Master switch; off \= no behaviour change |
| drone\_correction\_model | affine | Global background-motion model: affine (partial-2D) or homography |
| drone\_correction\_box\_dilation | 0.20 | Box dilation (fraction of box size) when masking horses out of the background |
| drone\_correction\_min\_features | 30 | Minimum background features required to trust the estimated transform |
| drone\_correction\_uncertain\_std | 8.0 | Residual flow std (px) above which a frame is flagged 'uncertain' |
| drone\_correction\_smoothing | savgol | Centroid smoothing before differentiating: savgol, moving\_average, or none |
| drone\_correction\_smoothing\_window | 7 | Odd window length for the smoothing filter |
| drone\_correction\_fallback\_smoothing | true | When features are persistently too few, smooth-only (no optical-flow correction) |

### Metric geometry (§6.9)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| metric\_enabled | false | Project tracks to ground-plane metres; **required** by §6.10–6.13. Needs a `<video>.flightlog.csv` |
| metric\_focal\_len\_mm | 24.0 | Lens focal length (mm) used to derive the pixel focal length |
| metric\_sensor\_width\_mm | 36.0 | Sensor width (mm) for the same derivation |
| metric\_horizon\_margin\_px | 50 | Rows closer than this to the horizon are rejected (projection diverges) |
| metric\_roll\_max\_deg | 3.0 | Maximum camera roll tolerated before the clip is refused |
| metric\_scale\_tolerance | 0.25 | Allowed disagreement between flight-log height and animal-size-implied height before the clip is flagged 'uncertain' |
| metric\_assumed\_body\_length\_m | 2.2 | Reference adult body length used only by that cross-check |
| metric\_geometry\_drift\_frac | 0.25 | Ground-plane stability: how far the horizon row may travel between sliding windows, as a fraction of frame height, before the clip is reported non-flat (**warning only**, never changes metric\_quality) |
| metric\_geometry\_window\_s | 10.0 | Length of each sliding window used by that stability check |

### Interaction graph (§6.10)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| interaction\_graph\_enabled | true | Run the interaction-graph stage after metric geometry |
| interaction\_edge\_granularity | per\_dyad | Edge granularity: per\_dyad (one row per pair for the whole clip), per\_segment (one row per episode), or per\_frame. `per_interaction` is still accepted as the old name for per\_dyad |
| interaction\_weight\_metric | sri | Edge weight: sri (time together / time both visible, 0-1), duration\_s, or proximity\_m — all absolute, so comparable across videos |
| complex\_max\_interaction\_distance\_m | 15.0 | Pairs farther apart than this ON THE GROUND (metres) are ignored as interactions |
| complex\_min\_duration\_frames | 10 | Minimum observed frames for a dyad/episode to become an edge (all granularities) |
| complex\_contact\_iou\_thresh | 0.05 | Box IoU above this counts as contact |
| complex\_contact\_dist\_m | 2.5 | Ground distance (metres) below this counts as contact |
| complex\_window\_frames | 30 | Window length (frames) for aggregating features for the model |

### Complex-behaviour model (§6.12) & candidates (§6.13)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| complex\_classify\_enabled | true | Run the train-if-new + classify stage in the pipeline chain |
| complex\_behaviours | (empty) | Comma-separated, user-editable list of dyadic AND group behaviours |
| complex\_behaviours\_hotkeys | (empty) | Parallel comma-separated single-char hotkeys for the behaviours above |
| complex\_model\_type | baseline | Complex-behaviour model: baseline, lstm, or transformer (the last two require torch) |
| complex\_baseline\_classifier | random\_forest | Baseline classifier: random\_forest or hist\_gradient\_boosting |
| complex\_seq\_steps | 8 | Sub-windows a labelled segment is sliced into (deep-model sequence length) |
| complex\_deep\_epochs | 60 | Training epochs for the deep (lstm/transformer) model |
| complex\_deep\_hidden | 64 | Hidden size / d\_model of the deep model |
| complex\_deep\_layers | 1 | Stacked recurrent / encoder layers |
| complex\_deep\_heads | 4 | Attention heads (transformer only; must divide d\_model) |
| complex\_deep\_dropout | 0.2 | Dropout used in the deep model |
| complex\_deep\_lr | 0.001 | Adam learning rate for the deep model |
| complex\_deep\_batch | 16 | Mini-batch size for the deep model |
| complex\_confusion\_merge\_rate | 0.20 | Min true-vs-predicted confusion rate to flag a class pair as a merge suggestion |
| complex\_predict\_min\_proba | 0.5 | Minimum predicted probability to emit a complex-behaviour prediction |
| complex\_speed\_low\_ms | 0.2 | Candidate heuristics: ground speed (m/s) below this counts as ~still |
| complex\_speed\_high\_ms | 3.0 | Ground speed (m/s) above this counts as fast (gallop/chase) |
| complex\_polarisation\_high | 0.7 | Group polarisation above this counts as aligned (trek/stampede) |
| complex\_synchrony\_high | 0.7 | Behavioural synchrony above this counts as synchronised |
| complex\_candidate\_topk | 50 | Number of most-uncertain windows surfaced by active learning |

### Display (shared by the output video and the annotation tool)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| detection\_video\_enabled | false | Write \<video\>\_detected.mp4 alongside the tracking CSV. Off by default: the CSVs are the result of a run, the annotated copy is a visual check that costs roughly the size of the source clip |
| line\_thickness | 1 | Base bounding-box and text line thickness |
| font\_size | 0.5 | Base label text scale |
| box\_line\_scale | 1.0 | Per-tool multiplier on box line thickness |
| box\_font\_scale | 1.0 | Per-tool multiplier on label font size |
| adaptive\_box\_scaling | true | Derive font/thickness from the native box size, so a given animal looks the same at any zoom |
| adaptive\_font\_coeff / \_min / \_max | — | Coefficient and clamps of the adaptive font size |
| adaptive\_thickness\_coeff / \_min / \_max | — | Coefficient and clamps of the adaptive line thickness |
| label\_bg\_mode | — | Label background: none / solid / translucent |
| label\_bg\_opacity | — | Opacity of a translucent label background |
| label\_bg\_color | — | Label background RGB |
| halo\_thickness / halo\_color | — | Contrasting outline drawn behind text |
| show\_species / show\_age | — | Include the species / age label in the drawn label stack |
| buttons\_per\_row | — | Class-button rows in the annotation tool's control bar |

### Data augmentation

| Parameter | Default | Description |
| :---- | :---- | :---- |
| aug\_global\_probability | 0 | Master gate probability (0 disables augmentation) |
| aug\_target\_classes | (empty) | Only augment images containing at least one of these classes; empty = all |
| motion\_disable\_color\_aug | true | Forbid colour transforms on motion images (§5.3), offline *and* online: the motion detector and motion crop classifier are trained with hsv\_h/hsv\_s/hsv\_v \= 0 **and** auto\_augment \= None (Ultralytics defaults the latter to `randaugment`, whose op pool re-introduces Color/Brightness/Contrast/Sharpness through the back door on the classification task) |
| aug\_brightness\_range | 0.8,1.2 | Min,max brightness factor (multi-segment syntax allowed) |
| aug\_brightness\_probability | 0 | Per-image probability |
| aug\_contrast\_range | 0.8,1.2 | Min,max contrast factor |
| aug\_contrast\_probability | 0 | Per-image probability |
| aug\_saturation\_range | 0.8,1.2 | Min,max saturation factor |
| aug\_saturation\_probability | 0 | Per-image probability |
| aug\_hue\_range | \-15,15 | Min,max hue shift (degrees) |
| aug\_hue\_probability | 0 | Per-image probability |
| aug\_sharpness\_range | 0.8,1.5 | Min,max sharpness factor |
| aug\_sharpness\_probability | 0 | Per-image probability |
| aug\_blur\_range | 1,3 | Min,max Gaussian blur radius (forced odd) |
| aug\_blur\_probability | 0 | Per-image probability |
| aug\_noise\_range | 0,25 | Min,max additive noise magnitude (±value) |
| aug\_noise\_probability | 0 | Per-image probability |
| aug\_shear\_range | \-0.1,0.1 | Min,max horizontal shear factor |
| aug\_shear\_probability | 0 | Per-image probability |
| aug\_flip\_h\_options | True,False | Possible values for horizontal flip |
| aug\_flip\_h\_probability | 0 | Per-image probability |
| aug\_flip\_v\_options | True,False | Possible values for vertical flip |
| aug\_flip\_v\_probability | 0 | Per-image probability |
| aug\_temperature\_range | 0,10 | Min,max temperature shift (+red/−blue) |
| aug\_temperature\_probability | 0 | Per-image probability |

### Activity budget (§7)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| ab\_min\_presence\_seconds | 30 | Minimum presence (seconds) to count as a group member |
| ab\_min\_classified\_frames | 0 | Minimum classified frames to count as a group member (0 disables this criterion) |
| ab\_edge\_margin\_px | 100 | Border-zone width (px) used for the entry-point criterion |
| ab\_analysis\_duration\_s | 0 | Analyse only the first N seconds of each clip (0 = whole clip) |
| ab\_group\_type\_separator | \_ | Filename field separator |
| ab\_group\_type\_field\_index | 4 | Zero-based filename field index encoding group type |

---

## 12\. INSTALLERS & LAUNCHERS

### 12.1 Linux launcher (Linux\_Launcher.sh)

- **\[Technical\]**  
  - *How it works:* Self-bootstrapping bash script; bootstrap runs once. Installs apt packages (python3-venv, python3-pip, build-essential, ffmpeg, python3-opencv, …), creates the virtual environment with \--system-site-packages, pip-installs requirements.txt (the source of truth for dependencies, excluding opencv-python which comes from apt, and torch/torchvision which are installed **last** so the machine-specific build wins), then drops a .ultralytics\_ready marker. Launches BehaveAI.py if present, else a script path argument.  
  - *Purpose:* One-command environment setup and launch on Linux.  
  - *Parameters:* optional script path argument.  
  - *Implementation:* bash, apt, python3 \-m venv, pip.  
- **\[Plain\]**  
  - *How it works:* On first run it installs everything needed and creates an isolated Python environment, then starts the program; later runs skip setup.  
  - *Purpose:* Get running on Linux in one step.  
  - *Parameters:* an optional script to launch.

### 12.2 Windows launcher (Windows\_Launcher.bat \+ Windows\_Launcher\_ps.ps1)

- **\[Technical\]**  
  - *How it works:* The .bat opens PowerShell with execution-policy bypass. The script detects Python (py \-3/python/python3), offering to silently install Python 3.12 (64-bit); creates the venv; installs requirements.txt first; detects NVIDIA GPUs via nvidia-smi and WMI Win32\_VideoController; offers a PyTorch wheel choice (CPU, auto-CUDA, or manual) and installs it **last** so pip's transitive re-resolution cannot downgrade it back to CPU; verifies ultralytics/torch/cv2 imports; writes .ultralytics\_ready; logs to Windows\_Launcher\_ps.log.  
  - *Purpose:* Automated, GPU-aware Windows setup and launch.  
  - *Parameters:* PyTorch wheel choice; optional script path.  
  - *Implementation:* PowerShell, nvidia-smi, WMI, python \-m venv, pip.  
- **\[Plain\]**  
  - *How it works:* It finds or installs Python, builds an isolated environment, checks for an NVIDIA graphics card, lets you choose the matching acceleration, installs everything, verifies it, and starts the program — keeping a full log.  
  - *Purpose:* One-step Windows setup that uses your GPU if available.  
  - *Parameters:* the acceleration choice; an optional script to launch.

### 12.3 Windows uninstaller (Windows\_Uninstaller.bat \+ Windows\_Uninstaller\_ps.ps1)

- **\[Technical\]** *How it works:* Interactively removes the venv, marker, and logs; detects running venv Python processes and offers to kill them first; never removes working-directory scripts or system Python. *Purpose:* clean reversal of the environment only. *Parameters:* none. *Implementation:* PowerShell.  
- **\[Plain\]** *How it works:* Removes the installed environment and its logs, optionally stopping running processes first, while leaving your scripts and system Python untouched. *Purpose:* undo the install safely. *Parameters:* none.

---

## 13\. DIRECTORY STRUCTURE (per project)

*Folder names shown are those of the **first** species; additional species get the same names suffixed with their slug (§2.1).*

```
projects/
└── <project_name>/
    ├── BehaveAI_settings.ini
    ├── static_annotations.yaml
    ├── motion_annotations.yaml
    │
    ├── clips/                         ← training videos
    ├── timecodes/                     ← time-code CSVs for targeted annotation
    │                                     (example_timecodes.csv template)
    ├── input/                         ← batch input videos
    │                                     (+ <video>.flightlog.csv sidecars for metric geometry)
    ├── output/                        ← <video>_detected.mp4
    │                                     <video>_tracking.csv
    │                                     <video>_tracking_corrected.csv (+ _correction_diag.csv)
    │                                     <video>_tracking_stitched.csv
    │                                     <video>_tracking_metric.csv
    │                                     <video>_interaction_edges.csv / _interaction_nodes.csv
    │                                     <video>_complex_behaviours.csv   (annotations)
    │                                     <video>_complex_predictions.csv
    │                                     <video>_complex_candidates.csv
    │                                     activity_budget_individual.csv
    │                                     activity_budget_predicted.csv
    │                                     activity_budget_suspects.csv
    │                                     det_*.mp4 / det_*.csv            (live)
    ├── evaluation/                    ← detection / tracking / geometry evaluation reports
    │
    ├── annot_static/
    │   ├── images/{train,val}/
    │   ├── labels/{train,val}/
    │   └── masks/{train,val}/         ← grey-box coordinates (.mask.txt)
    ├── annot_motion/                  (same structure)
    │
    ├── annot_static_crop/             ← secondary crops (static), pooled by secondary label
    │   └── <secondary_class|__none__>/
    ├── annot_motion_crop/             ← secondary crops (motion), same layout
    ├── annot_species_crop/            ← species crops (model 0), pooled project-wide
    ├── annot_age_crop/                ← age crops (model 0.5)
    ├── annot_*_split/                 ← generated train/val split for the crop classifiers (§3.9)
    │
    ├── model_primary_static/
    │   ├── train/weights/best.pt
    │   ├── train/results.csv
    │   ├── train_count.txt
    │   └── saved_settings.ini
    ├── model_primary_motion/          (same structure)
    ├── model_secondary_static/        (one pooled per-stream model; same structure)
    ├── model_secondary_motion/        (same structure)
    ├── model_species/                 (same structure)
    ├── model_age/                     (same structure)
    ├── model_complex/                 ← pipeline.joblib (+ deep_model.pt), metrics.txt,
    │                                     merge_suggestions.txt, feature_importances.txt (§6.12)
    │
    └── groups_metadata.csv            ← optional manual overrides for activity budget
```

---

## 14\. METHODS SUMMARY (for a scientific Materials & Methods section)

- **\[Technical\]**  
  - *Acquisition:* video clips (drone footage, or captured live via OpenCV/picamera2) stored per project, with a per-video flight-log sidecar when metric analysis is required.  
  - *Representation:* each frame is encoded as (i) a static RGB image and (ii) a false-colour motion image from temporal frame differencing — exponential multi-timescale decay (expA, expB) or sequential differencing, blended with luminance (lum\_weight) and per-channel amplification (rgb\_multipliers), optionally differential (chromatic\_tail\_only).  
  - *Annotation:* random-frame sampling or CSV-driven time-code navigation (targeted annotation of specific frames/events, by integer frame or mm:ss), dual-stream YOLO-format boxes, optional hierarchical secondary crops plus species and age crops, optional cross-stream grey masking, whole-video train/val split \= val\_frequency (all models, detectors and crop classifiers alike; §3.9).  
  - *Augmentation:* offline, per-image probabilistic photometric/geometric transforms (PIL, cv2); motion images restricted to geometry/texture; geometric label correction.  
  - *Models:* separate primary YOLO detectors per stream (Ultralytics; configurable backbone/epochs, early stopping via train\_patience), pooled secondary YOLO classifiers per stream plus species and age classifiers (224 px crops), optional NCNN export; transfer learning on dataset growth, each model trained in an isolated process.  
  - *Detection & tracking:* optional SAHI sliced inference on native-resolution tiles (with a matching tiled dataset retrain) to recover small/distant subjects → dual detection → centroid/IOU merge (dominant\_source rule) → secondary/species/age classification → multi-object tracking by **BoT-SORT / ByteTrack** (Ultralytics; global camera-motion compensation + two-tier high/low-score association), selectable (tracker\_type) back to the legacy constant-velocity Kalman + Hungarian tracker. **Appearance is deliberately not used** at any stage of association — it is uninformative on a ~60×35 px horse at 15–50 m with uniform coats.  
  - *Identity (offline):* the causal tracker's short tracklets are re-linked over the whole clip by a non-causal **kinematic stitching** pass — hard gates of temporal non-overlap, a maximum plausible speed and a maximum occlusion length, then a likelihood ratio between "same animal returning" and "a different animal appearing here", solved as a dummy-augmented linear assignment. The motion model behind the ratio (drift scale and tail) is measured per clip from the tracker's own tracklets, not assumed; the known group size K is a reported diagnostic, never a constraint. Its usable envelope is set by how tightly packed the herd is (§6.6), which the controlled-fragmentation benchmark quantifies per clip.  
  - *Evaluation:* detection is scored as a three-way stream ablation (static-only / motion-only / merged) on the frozen whole-video holdout; tracking is scored against a hand-annotated MOT-format ground truth with **HOTA — reported as DetA and AssA separately** — plus IDF1, MOTA, ID switches and fragmentation (TrackEval), so detection and association gains can be attributed independently; the drone-correction estimator is validated on synthetic transforms of known magnitude.  
  - *Analysis (per individual):* per-individual activity budgets (time/frequency/percentage per behaviour) and rule-based stranger flagging (ab\_min\_presence\_seconds, ab\_min\_classified\_frames, ab\_edge\_margin\_px), kept separate from any model-predicted column.  
  - *Analysis (multi-individual, HERDWISE):* drone-induced apparent motion is removed by background optical-flow stabilisation (RANSAC affine/homography on horse-masked frames) and velocities recomputed from corrected, smoothed centroids; positions are then projected to **ground-plane metres** from flight-log telemetry (relative altitude + gimbal pitch + focal length; `behaveai_drone`), with an automatic scale cross-check against the animals' apparent size and a per-clip measured speed noise floor. Dyadic (distance m, approach rate m/s, velocity-cosine, contact) and group (polarisation, cohesion, hull area m², synchrony, elongation — computed over the whole co-present herd per frame) features are aggregated into an **undirected** interaction graph (networkx; R-igraph-ready edges/nodes CSVs, weights comparable across videos). Foal/adult status comes from the trained age classifier, never from apparent size. Multi-individual behaviours are hand-annotated, then classified by a by-video-evaluated model (scikit-learn baseline or LSTM/Transformer) that fuses the kinematic/graph features with the per-individual YOLO labels, and supplies the direction (interaction\_type + actor\_id) the symmetric deterministic layer cannot; heuristic + active-learning candidate proposal accelerates annotation.  
  - *Reproducibility:* per-model saved\_settings.ini, train\_count.txt, model backups (\*\_backupN), a filename-hash whole-video split reproducible from val\_frequency alone, Regenerate\_annotations.py for settings-consistent image rebuilds, and by-video (leave-one-video-out) evaluation for the complex-behaviour model. Cite exact values from §11.  
- **\[Plain\]**  
  - Videos are recorded and stored per study, with the drone's flight log kept alongside. Each frame is kept in two forms: the normal photo and a colour image that shows movement. Frames are labelled either at random across all clips or by following a list of specific moments supplied in a CSV (by frame number or mm:ss time), with optional finer categories plus species and age. The dataset can be enlarged with modified copies. Two detectors are trained — one per form — plus the finer classifiers, then used together: both forms are searched, duplicates are merged, finer categories are added, and each individual is tracked over time and re-joined afterwards where the tracking broke. The system never compares what animals look like, only where they are and how they move. Finally, it measures how long and how often each individual did each behaviour and flags likely strangers. For herd footage there is an extra layer: the drone's own camera movement is removed, positions are converted into real metres using the flight log, collective measures (how aligned, how tight, how in-sync the herd is) are computed for the whole group each moment, and a network of who-interacts-with-whom is built and can be opened in standard network software. Group behaviours (chasing, grooming, stampeding, trekking, resting together) are labelled by hand on a few examples and then recognised automatically across whole videos, with the system pointing you to the most useful moments to label next. Every score reported comes from videos the software has never trained on. Exact settings used should be reported from the table in §11.
