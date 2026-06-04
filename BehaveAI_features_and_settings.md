# BehaveAI — Features, Settings & Methods Reference

**Document version: 2.0** — updated 2026-06-05. Adds intra-video **Re-Identification** with a part-based appearance descriptor (foreground-masked, optionally body-axis-aligned grid), an optional **MegaDescriptor** backbone, and **self-supervised auto-training** (Re-ID embedding from tracker IDs + whole-horse/body-part segmentation), none of which require manual identity or part labels (§6.6–6.8).

A combined **user guide**, **README**, and **Materials & Methods source** for BehaveAI: a project-based pipeline for annotating video, training YOLO detectors/classifiers on a dual **static (RGB)** \+ **motion (false-colour)** representation, tracking individuals, and computing per-individual activity budgets, in batch or live.

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

**Software stack (packages):** ultralytics (YOLO; also SAM2 + auto\_annotate for segmentation), opencv-python (cv2), numpy, scipy (scipy.optimize.linear\_sum\_assignment; scipy.io for PASCAL-Part), pillow (PIL), pyyaml (yaml), tkinter, picamera2 (Raspberry Pi only), ncnn (optional inference backend), torch/torchvision (optional — Re-ID embedding & fine-tuning), timm (optional — MegaDescriptor Re-ID backbone), plus standard library (configparser, csv, glob, subprocess, threading, pathlib, collections, os, sys, re, time, random, shutil, base64, queue, platform).

---

## Pipeline Overview

- **Capture / collect** video clips → clips/  
- **Annotate** frames on two synchronised representations (static RGB \+ motion false-colour) → BehaveAI\_annotation.py  
- **Inspect / correct** the dataset → BehaveAI\_inspect\_dataset.py  
- **Augment** the dataset offline → BehaveAI\_augmentation.py  
- **Train** primary detectors (one per stream) and optional secondary classifiers → BehaveAI\_classify\_track.py  
- **Detect → merge streams → classify → track** on batch videos → BehaveAI\_classify\_track.py  
- **Re-identify** individuals within a video after occlusion (spatio-temporal gate \+ part-based appearance descriptor) → BehaveAI\_reid.py  
- *(optional)* **Auto-train** a domain-adapted Re-ID embedding from tracker IDs, and whole-horse / body-part **segmentation** (SAM2 \+ PASCAL-Part), with no manual labels → BehaveAI\_reid\_finetune.py, BehaveAI\_segmentation.py  
- **Activity budget** per individual \+ stranger flagging → BehaveAI\_activity\_budget.py  
- **Live** acquisition \+ inference → BehaveAI\_live.py  
- All orchestrated by the **launcher** BehaveAI.py; configured by BehaveAI\_settings\_gui.py.

---

## 1\. LAUNCHER (BehaveAI.py)

### 1.1 Project management

- **\[Technical\]**  
  - *How it works:* Projects live under a sibling projects/ directory. Each project is a self-contained workspace holding its settings, datasets, models, and I/O. Creating a project scaffolds clips/, input/, output/ and a default BehaveAI\_settings.ini. A combobox enumerates project folders; selection triggers loading and statistics computation. Refresh rescans the directory.  
  - *Purpose:* Isolate independent experiments/datasets so settings, annotations, and weights never collide.  
  - *Parameters:* project\_path (absolute root, written into the INI); folder layout is fixed by convention.  
  - *Implementation:* pathlib.Path, os, configparser for INI scaffolding; tkinter.ttk.Combobox, simpledialog, messagebox for the UI.  
- **\[Plain\]**  
  - *How it works:* Each study is one folder under projects/, containing its own videos, labels, trained files, and a settings file. A menu lists those folders; picking one loads it. Creating one builds the empty folders and a default settings file automatically.  
  - *Purpose:* Keep separate studies fully apart so nothing from one mixes into another.  
  - *Parameters:* The project's location on disk; all sub-folders follow a fixed naming scheme.

### 1.2 Smart button enabling

- **\[Technical\]**  
  - *How it works:* Button availability is derived from INI state at selection time. *Settings* enables on any project; *Annotate / Inspect / Train & batch / Live* require ≥1 non-empty primary class list; *Augment Dataset* requires aug\_global\_probability \> 0. After any subprocess exits, states are re-evaluated.  
  - *Purpose:* Prevent launching stages whose preconditions (defined classes, enabled augmentation) are not met.  
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

### 2.1 Model structure editor (class groups)

- **\[Technical\]**  
  - *How it works:* Four editable class groups — primary motion, secondary motion, primary static, secondary static. Each row has a label, a single-character hotkey, an RGB colour (colour chooser), and (primary only) an *Ignore secondary* flag. Validation blocks saving on duplicate/multi-char/reserved (u, g) hotkeys, on having zero primary classes, or on a secondary group with exactly one class (YOLO classification needs ≥2). Adding/removing/clearing classes when annot\_motion/annot\_static already exist raises a structural-change warning.  
  - *Purpose:* Define the two-tier label space — primary classes detected per stream and optional secondary sub-classes for hierarchical classification.  
  - *Parameters:* primary\_\*\_classes/colors/hotkeys, secondary\_\*\_classes/colors/hotkeys, ignore\_secondary.  
  - *Implementation:* ClassRow, ClassListEditor (ttk.Frame); colorchooser; parse\_list\_field/parse\_colors\_field/list\_to\_field/colors\_to\_field.  
- **\[Plain\]**  
  - *How it works:* You define your categories in four lists (two main, two optional sub-categories), each with a name, a shortcut key, and a colour. The tool refuses to save if two keys clash, a key uses a reserved letter, no main category exists, or a sub-category list has only one entry. It warns you if you change categories after labelling has begun.  
  - *Purpose:* Set up the categories the system will learn — main ones, plus optional finer ones.  
  - *Parameters:* The category names, colours, shortcut keys, and which main categories skip the finer step.

### 2.2 Cross-stream blocking (motion/static masking)

- **\[Technical\]**  
  - *How it works:* *Motion blocks static* paints motion-annotation box regions grey on static training images at save time; *Static blocks motion* does the reverse. This removes the other stream's targets from each image so a detector is not trained against objects it is not meant to learn in that stream.  
  - *Purpose:* Prevent cross-contamination between the two single-stream detectors.  
  - *Parameters:* motion\_blocks\_static, static\_blocks\_motion.  
  - *Implementation:* grey fill via cv2.rectangle(...,(128,128,128),-1) (applied in annotation/regeneration saving).  
- **\[Plain\]**  
  - *How it works:* Objects that belong to one view can be grey-covered in the other view's training images, so each detector only sees the objects it is supposed to learn.  
  - *Purpose:* Stop the two detectors from confusing each other's targets.  
  - *Parameters:* The two "blocks" switches.

### 2.3 Video paths

- **\[Technical\]** *How it works:* Three browse-able directory fields (clips\_dir, input\_dir, output\_dir) validated as non-empty at save. *Purpose:* declare data locations for annotation and batch I/O. *Parameters:* the three paths. *Implementation:* filedialog.askdirectory.  
- **\[Plain\]** *How it works:* Three folder pickers tell the system where training clips, batch inputs, and outputs live; saving warns on empty ones. *Purpose:* point the tools at the right folders. *Parameters:* the three folders.

### 2.4 Data-augmentation configuration

- **\[Technical\]**  
  - *How it works:* Exposes the master gate aug\_global\_probability plus a per-transform *range* and *probability* for brightness, contrast, saturation, hue, sharpness, blur, noise, shear, horizontal/vertical flip, and temperature. *Delete all augmented data* removes every file with \_aug\_ in its basename across image/label dirs (with confirmation).  
  - *Purpose:* Configure offline augmentation that enlarges and diversifies the dataset.  
  - *Parameters:* aug\_global\_probability, aug\_\<name\>\_range, aug\_\<name\>\_probability, aug\_flip\_\*\_options (see §5, §11).  
  - *Implementation:* INI read/write via configparser; deletion via glob/os.remove.  
- **\[Plain\]**  
  - *How it works:* You set one master chance plus, for each effect, a strength range and its own chance. A button can delete all generated copies at once.  
  - *Purpose:* Control how the dataset is expanded with modified copies.  
  - *Parameters:* the master chance and each effect's range and chance.

### 2.5 Motion strategy configuration

- **\[Technical\]**  
  - *How it works:* Selects how the motion false-colour stream is built (see §3.3 for the algorithm): strategy (exponential/sequential), chromatic\_tail\_only, decay factors expA/expB (which also set the frame-window size), lum\_weight, rgb\_multipliers, frame\_skip, plus scale\_factor and motion\_threshold. The expA value maps to frameWindow (0.2→5, \>0.5→10, \>0.7→15, \>0.8→20, \>0.9→45).  
  - *Purpose:* Tune the temporal-difference encoding that makes movement visible to a detector.  
  - *Parameters:* all motion-strategy keys (see §11).  
  - *Implementation:* consumed by generate\_base\_images() and equivalents; cv2.absdiff, cv2.addWeighted, cv2.subtract, cv2.merge.  
- **\[Plain\]**  
  - *How it works:* You choose how movement is turned into colour: which method, how long the coloured trail lasts, how much grey image is mixed in, and how strongly each colour is boosted.  
  - *Purpose:* Adjust how clearly motion shows up.  
  - *Parameters:* the motion-strategy settings (see §11).

### 2.6 Model-type configuration

- **\[Technical\]**  
  - *How it works:* Sets val\_frequency, primary\_classifier (base detector, e.g. yolo11s.pt; yolo8/11/26 in n/s/m/l), primary\_epochs, secondary\_classifier (e.g. yolo11s-cls.pt), secondary\_epochs, use\_ncnn, primary\_conf\_thresh, secondary\_conf\_thresh, and dominant\_source (confidence/motion/static).  
  - *Purpose:* Choose architecture sizes, training length, inference thresholds, and the inter-stream tie-break rule.  
  - *Parameters:* the keys above (see §6, §11).  
  - *Implementation:* values consumed by BehaveAI\_classify\_track.py training/inference.  
- **\[Plain\]**  
  - *How it works:* You pick the base model files, how many training passes to run, the minimum confidence to keep a result, and which view wins when both views detect the same object.  
  - *Purpose:* Set how the models are built and how strict detection is.  
  - *Parameters:* base models, epochs, confidence thresholds, tie-break rule.

### 2.7 Tracking & Kalman configuration

- **\[Technical\]** *How it works:* Exposes match\_distance\_thresh, delete\_after\_missed, centroid\_merge\_thresh, iou\_thresh, and the \[kalman\] triplet process\_noise\_pos/process\_noise\_vel/measurement\_noise. *Purpose:* govern detection-to-track association, stream merging, and Kalman uncertainty (see §6.4). *Parameters:* listed keys. *Implementation:* read into KalmanTracker.  
- **\[Plain\]** *How it works:* You set how far an object can move between frames and still count as the same individual, how long a missing individual is kept, and how detections from the two views are combined. *Purpose:* control identity tracking. *Parameters:* the tracking and Kalman values.

### 2.8 Display & Activity-Budget configuration

- **\[Technical\]** *How it works:* Display: line\_thickness, font\_size for overlays. Activity budget: ab\_min\_presence\_ratio, ab\_border\_zone\_ratio, ab\_group\_type\_separator, ab\_group\_type\_field\_index (zero-based filename field for group type). *Purpose:* overlay styling and post-hoc presence/stranger analysis (§7). *Parameters:* listed keys. *Implementation:* consumed by overlays and BehaveAI\_activity\_budget.py.  
- **\[Plain\]** *How it works:* Sets box/text size on output video, plus the rules that decide who counts as a group member versus a stranger and how the group type is read from the filename. *Purpose:* control labels on video and the presence analysis. *Parameters:* the display and budget values.

### 2.9 Save behaviour (with regeneration)

- **\[Technical\]**  
  - *How it works:* On save, writes BehaveAI\_settings.ini, generates static\_annotations.yaml and motion\_annotations.yaml (YOLO dataset configs), and creates missing annotation directory trees. If motion-strategy parameters changed and annotations exist, it offers to run Regenerate\_annotations.py; if accepted, it first backs up model\_primary\_motion, model\_primary\_static, and all model\_secondary\_motion\_\* to \*\_backupN.  
  - *Purpose:* Persist configuration, keep YOLO configs in sync, and keep motion images consistent with current settings.  
  - *Parameters:* all settings; motion keys trigger regeneration.  
  - *Implementation:* configparser, yaml.safe\_dump, os.makedirs, shutil backups, subprocess to launch regeneration.  
- **\[Plain\]**  
  - *How it works:* Saving writes the settings file and the two dataset description files, and builds any missing folders. If you changed how motion is computed and you already have labels, it offers to rebuild the motion images and safely backs up existing models first.  
  - *Purpose:* Save settings and keep everything consistent.  
  - *Parameters:* all settings; motion changes trigger the rebuild offer.

---

## 3\. ANNOTATION TOOL (BehaveAI\_annotation.py)

### 3.1 Frame-pool & random sampling

- **\[Technical\]**  
  - *How it works:* Recursively scans clips\_dir and builds a pool of annotatable frames (those with enough preceding frames for the motion window). Already-annotated frames are excluded via the AnnotationIndex. Launch and each save/skip draw a new random unannotated frame; exits when the pool is empty.  
  - *Purpose:* Spread annotation uniformly at random across all clips to avoid temporal bias.  
  - *Parameters:* frame\_window (derived from expA/strategy), frame\_skip, clips\_dir.  
  - *Implementation:* build\_frame\_pool(), get\_unannotated\_pool(), pick\_random\_frame(), \_scan\_videos\_recursive(); random, cv2, AnnotationIndex.  
- **\[Plain\]**  
  - *How it works:* The tool lists every frame that can be labelled across all clips, removes those already done, and shows you a random remaining one each time you save or skip.  
  - *Purpose:* Pick frames evenly at random rather than in order.  
  - *Parameters:* the motion-window length, frame skipping, and the clips folder.

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
  - *Parameters:* line\_thickness, font\_size (overlay rendering).  
  - *Implementation:* cv2 drawing, PIL.Image/ImageTk, cv2\_to\_photoimage(), draw\_boxes\_on\_image().  
- **\[Plain\]**  
  - *How it works:* One full-screen view shows either the motion or the photo (toggle with Space), plus three side panels: a zoomed photo, a zoomed motion image, and a small looping clip of the real movement at that spot.  
  - *Purpose:* See fine detail and the actual motion while placing boxes.  
  - *Parameters:* box and text size.

### 3.5 Seek bar with annotation ticks

- **\[Technical\]** *How it works:* A slider spans the current video; red ticks mark already-annotated frames, a black line marks the current position; dragging triggers full motion recomputation. *Purpose:* navigate within a clip and see coverage. *Parameters:* none. *Implementation:* Tk canvas/scale; AnnotationIndex for tick positions.  
- **\[Plain\]** *How it works:* A slider runs the whole clip; red marks show frames you already labelled and a black line shows where you are. *Purpose:* move through a clip and see what's done. *Parameters:* none.

### 3.6 Annotation controls

- **\[Technical\]**  
  - *How it works:* Left-drag draws a box; right-click deletes the innermost box/mask under the cursor; class hotkeys select primary/secondary class; g toggles grey-mask mode; u undoes; Space toggles view; Enter saves and advances; Shift+Enter returns to the previous frame; Escape skips; Delete deletes all files for the frame (Enter confirm / Escape cancel); arrows step ±1, Shift\+arrows ±10, Ctrl\+arrows jump between annotated frames.  
  - *Purpose:* Full keyboard-driven box/mask editing and navigation.  
  - *Parameters:* per-class \*\_hotkeys; reserved u, g.  
  - *Implementation:* Tk event bindings; AnnotatorTk; non\_max\_suppression(), iou() for overlap handling.  
- **\[Plain\]**  
  - *How it works:* Drag to draw a box, right-click to remove one, press a category's key to choose it, g for grey cover, u to undo, Space to switch views, Enter to save and move on, and arrow keys to step through frames.  
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
  - *How it works:* Each frame is assigned train/val by val\_frequency. Static and motion images are saved to their own dirs. motion\_blocks\_static/static\_blocks\_motion apply grey blocking before save. save\_empty\_frames keeps box-less frames as negatives. Grey-box coords go to .mask.txt; in hierarchical mode, per-box crops go to annot\_\*\_crop/\<primary\>/\<secondary\>/. Existing annotations are overwritten; counts print to console.  
  - *Purpose:* Produce YOLO-format datasets for both streams (and crop datasets for secondary classifiers) with reproducible splits.  
  - *Parameters:* val\_frequency, save\_empty\_frames, motion\_blocks\_static, static\_blocks\_motion.  
  - *Implementation:* save\_annotation(); cv2.imwrite, random, os; YOLO label format.  
- **\[Plain\]**  
  - *How it works:* Each labelled frame is randomly put in training or validation, the photo and motion versions are saved separately, optional grey-covering is applied, empty frames can be kept as examples of "nothing here", and finer-category cut-outs are saved in their own folders.  
  - *Purpose:* Build the labelled datasets the models train on.  
  - *Parameters:* the validation fraction, keep-empty option, and the two blocking switches.

---

## 4\. DATASET INSPECTOR (BehaveAI\_inspect\_dataset.py)

### 4.1 Library navigation & editing

- **\[Technical\]**  
  - *How it works:* Loads all annotated frames from the four directories (static/motion × train/val) into one ordered library; a slider spans it; arrows step ±1, Shift\+arrows ±10. The display reuses the annotation composite (main panel \+ zoom column \+ bottom bar; Space toggles view). Editing controls match the annotator (draw/delete/grey/u/g/Enter save/Delete remove). It loads the source video for the animation panel; if absent, repeats the static image. Hierarchical secondary crops are updated on save.  
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
  - *How it works:* Scans original images in annot\_static/annot\_motion (train+val), skipping any with \_aug\_ in the name. Each image first passes the global gate aug\_global\_probability; then each transform independently passes its own aug\_\<name\>\_probability. Every triggered transform writes one independent copy \<basename\>\_aug\_\<param\>.\<ext\> (originals untouched; re-runs are idempotent/overwriting). Parameter values are sampled from the configured range/segment.  
  - *Purpose:* Enlarge and diversify the dataset to improve generalisation and class balance, fully offline and inspectable.  
  - *Parameters:* aug\_global\_probability, aug\_\<name\>\_range, aug\_\<name\>\_probability, aug\_flip\_\*\_options.  
  - *Implementation:* apply\_augmentation\_to\_all\_annotations(), sample\_augmentation\_list(), \_parse\_segments(), \_sample\_segment(); random, numpy.  
- **\[Plain\]**  
  - *How it works:* It looks at your original labelled images and, image by image, decides by chance whether to make modified copies and which effects to apply. Each chosen effect makes its own separate copy; originals are never changed.  
  - *Purpose:* Add varied copies so the models cope better with real-world variation.  
  - *Parameters:* the master chance and each effect's chance and strength.

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
  - *How it works:* For annot\_motion images, only geometry/texture transforms (blur, noise, sharpness, shear, flip\_h, flip\_v) are allowed; colour transforms (brightness, contrast, saturation, hue, temperature) are excluded because the false-colour encoding carries the motion signal and altering colour would corrupt it.  
  - *Purpose:* Preserve the meaning of the motion stream under augmentation.  
  - *Parameters:* implicit (source directory).  
  - *Implementation:* transform filtering in apply\_augmentation\_to\_all\_annotations().  
- **\[Plain\]**  
  - *How it works:* Motion images only get shape/texture effects, never colour changes, because their colours mean something specific.  
  - *Purpose:* Avoid destroying the motion information.  
  - *Parameters:* none.

### 5.4 Label transformation & progress reporting

- **\[Technical\]** *How it works:* Only geometric transforms update YOLO labels — flip\_h: xc→1-xc; flip\_v: yc→1-yc; all others leave labels unchanged. Progress prints XX% |\#\#\#---| N/total filename, which the launcher overwrites in place. *Purpose:* keep labels valid after geometry; give live feedback. *Parameters:* none. *Implementation:* transform\_labels(), \_print\_progress().  
- **\[Plain\]** *How it works:* When an image is flipped, its box coordinates are flipped to match; other effects don't move boxes. A live progress bar is printed. *Purpose:* keep boxes correct and show progress. *Parameters:* none.

---

## 6\. TRAIN & BATCH CLASSIFY (BehaveAI\_classify\_track.py)

### 6.1 Conditional (re)training with transfer learning

- **\[Technical\]**  
  - *How it works:* maybe\_retrain() runs per model before processing. No model → train from scratch from the base classifier for the configured epochs. Existing model → compare train\_count.txt to current dataset image count; if changed, prompt to retrain. On retrain → back up the model dir to \*\_backupN, fine-tune from the backed-up weights (transfer learning), update train\_count.txt. A copy of the INI (saved\_settings.ini) is stored per model for reproducibility. YOLO output is relocated from runs/detect/\*/train via move\_to\_expected().  
  - *Purpose:* Avoid redundant retraining, support incremental dataset growth, and guarantee reproducible model provenance.  
  - *Parameters:* primary\_classifier, primary\_epochs, secondary\_classifier, secondary\_epochs, image sizes (primary native, secondary 224).  
  - *Implementation:* ultralytics.YOLO(...).train(), count\_images\_in\_dataset(), move\_to\_expected(); shutil, glob, configparser.  
- **\[Plain\]**  
  - *How it works:* Before processing, each model is checked: if missing it is trained; if the dataset grew it offers to retrain, continuing from the old version and backing it up first. A copy of the settings is saved next to each model.  
  - *Purpose:* Only retrain when needed and always keep a record of how each model was made.  
  - *Parameters:* base models, epochs, image sizes.

### 6.2 NCNN export & loading

- **\[Technical\]** *How it works:* If use\_ncnn=true, checks for \<weights\>\_ncnn\_model/; if absent calls model.export(format="ncnn") (waits ≤300 s), then loads NCNN for inference, falling back to .pt on failure. Applied to all primary and secondary models. *Purpose:* faster CPU/ARM inference. *Parameters:* use\_ncnn. *Implementation:* ensure\_ncnn\_export(), load\_model\_with\_ncnn\_preference(), ncnn\_dir\_for\_weights(), ncnn\_files\_exist(); ultralytics, ncnn.  
- **\[Plain\]** *How it works:* Optionally converts models to a format that runs faster on CPUs and small devices, reusing an existing conversion if present and falling back to the original if conversion fails. *Purpose:* faster running on modest hardware. *Parameters:* the NCNN switch.

### 6.3 Two-stream detection \+ merging \+ secondary classification

- **\[Technical\]**  
  - *How it works:* Per frame: compute the motion image (§3.3); run primary static YOLO on the raw frame and primary motion YOLO on the motion image. Merge across streams: two detections merge if centroid distance \< centroid\_merge\_thresh or box overlap \> iou\_thresh; the survivor is chosen by dominant\_source (confidence keeps higher score, else fixed motion/static preference). For each merged detection, crop the box and run the matching secondary classifier (static→static model, motion→motion model, with fallback to the other), keeping results above secondary\_conf\_thresh.  
  - *Purpose:* Combine the complementary strengths of appearance (static) and movement (motion) detection, then refine each detection with a fine-grained class.  
  - *Parameters:* primary\_conf\_thresh, secondary\_conf\_thresh, centroid\_merge\_thresh, iou\_thresh, dominant\_source, ignore\_secondary.  
  - *Implementation:* ultralytics.YOLO, iou(), per-frame loop in process\_video(); cv2, numpy.  
- **\[Plain\]**  
  - *How it works:* Each frame is searched twice — once on the photo, once on the motion image. When both find the same object, the duplicates are merged and the better one is kept. Each kept object is then cut out and passed to a second model that assigns a finer category.  
  - *Purpose:* Use both appearance and movement to detect more reliably, then label each find more precisely.  
  - *Parameters:* the confidence thresholds, the merge distances, and the tie-break rule.

### 6.4 Kalman multi-object tracking

- **\[Technical\]**  
  - *How it works:* KalmanTracker holds one constant-velocity Kalman filter per individual (4D state x,y,vx,vy; 2D measurement x,y). Each frame: predict all tracks, build a cost matrix of predicted-vs-detection Euclidean distances, solve optimal assignment with the Hungarian algorithm (linear\_sum\_assignment), gated by match\_distance\_thresh; unmatched detections start new tracks; unmatched tracks accrue missed and inflated process noise until delete\_after\_missed. The batch variant also prunes near-coincident tracks (\_prune\_duplicate\_tracks, threshold \= ½ match\_distance\_thresh). Motion vectors are drawn as arrows.  
  - *Purpose:* Maintain stable identities across frames and recover briefly missed detections.  
  - *Parameters:* match\_distance\_thresh, delete\_after\_missed, \[kalman\] process\_noise\_pos/process\_noise\_vel/measurement\_noise.  
  - *Implementation:* cv2.KalmanFilter(4,2) (predict/correct), scipy.optimize.linear\_sum\_assignment, numpy (np.hypot).  
- **\[Plain\]**  
  - *How it works:* Each individual gets a predictor that estimates where it will be next. New detections are matched to the closest predictions; unmatched detections become new individuals, and individuals that disappear for too long are dropped. Short gaps are bridged by the prediction.  
  - *Purpose:* Keep the same ID on the same individual over time.  
  - *Parameters:* the match distance, the drop-after-missed count, and the predictor's noise settings.

### 6.5 Outputs & post-processing

- **\[Technical\]**  
  - *How it works:* Per video writes \<name\>\_detected.mp4 (boxes, labels, track IDs, motion arrows, frame counter) and \<name\>\_tracking.csv (frame,id,x,y,primary\_static\_class,primary\_static\_conf,primary\_motion\_class,primary\_motion\_conf,secondary\_static\_class,secondary\_static\_conf,secondary\_motion\_class,secondary\_motion\_conf). After all videos, run\_activity\_budget() is called automatically.  
  - *Purpose:* Provide both a human-checkable video and machine-readable per-frame tracks, then chain into analysis.  
  - *Parameters:* line\_thickness, font\_size (overlay).  
  - *Implementation:* cv2.VideoWriter, csv; calls BehaveAI\_activity\_budget.run\_activity\_budget().  
- **\[Plain\]**  
  - *How it works:* For each video it saves an annotated copy plus a table of every detection per frame, then automatically runs the activity-budget step.  
  - *Purpose:* Give you a viewable result and a data file, then the summary analysis.  
  - *Parameters:* overlay size.

### 6.6 Intra-video Re-Identification & part-based appearance (BehaveAI\_reid.py)

- **\[Technical\]**  
  - *How it works:* When a track is deleted (occlusion or leaving the frame) it is registered with its last position and an appearance descriptor; when a new detection would otherwise open a new track, the registry tries to recover a lost id using (1) a **mandatory spatio-temporal gate** — Euclidean distance ≤ reid\_max\_position\_distance — and (2) **appearance** as a weak tie-breaker. Descriptor layouts: *global* \= one masked HSV histogram of the box centre (legacy); *grid* \= per-cell masked HSV histograms over a foreground-aware R×C grid — a coarse **body-part encoding** (front \= head/neck, centre \= body, rear \= croup) that can be aligned to the body's major axis (PCA on the foreground mask) when reid\_orient. Foreground comes from the green-hue rule (hsv) or a segmentation model (sam2 / yoloseg, via ultralytics). Embedding methods replace the histogram: *embedding* \= torchvision MobileNetV3; *megadescriptor* \= BVRA MegaDescriptor (timm), optionally a self-supervised **fine-tuned checkpoint** (reid\_checkpoint, §6.7). Matching is cosine similarity with per-method thresholds; in monochrome herds, below-threshold appearance falls back to the spatially closest plausible candidate. Descriptors are computed **only** on track loss/creation, never per-frame-per-track. Defaults (global \+ histogram) reproduce the pre-existing tracker behaviour exactly.  
  - *Purpose:* Keep a stable identity through occlusions and re-entries, with appearance that stays informative even in uniform-coat herds — without any manual identity labels.  
  - *Parameters:* reid\_enabled, reid\_method, reid\_descriptor, reid\_grid, reid\_foreground, reid\_orient, reid\_backbone, reid\_checkpoint, reid\_similarity\_threshold, reid\_histogram\_min\_similarity, reid\_max\_position\_distance, reid\_max\_disappeared\_seconds (see §11).  
  - *Implementation:* ReIDRegistry (extract\_descriptor, \_histogram\_descriptor, \_grid\_descriptor, \_foreground\_mask, \_seg\_mask, \_estimate\_orientation, find\_match, register\_lost\_track); cv2, numpy; optional torch / timm / ultralytics (SAM2, YOLO-seg).  
- **\[Plain\]**  
  - *How it works:* When an animal disappears and reappears, the system first checks it is near where it was last seen, then compares its look. The "look" can be a single colour summary of the whole box or, better, one summary per cell of a grid laid over the animal (top \= head, middle \= body, back \= rump), optionally turned to follow the body so the same cell always covers the same part. It can also use a powerful pretrained animal-recognition model. The comparison only runs when an individual is lost or newly appears, so it stays fast.  
  - *Purpose:* Put the same number back on the same animal after it was hidden, even when all animals look alike.  
  - *Parameters:* the Re-ID switches and thresholds (see §11).

### 6.7 Self-supervised Re-ID fine-tuning (BehaveAI\_reid\_finetune.py)

- **\[Technical\]**  
  - *How it works:* Treats the tracker's own output as free labels — within one video each track\_id is, by construction, one individual, and simultaneous tracks are different individuals. It mines \<video\>\#\<track\_id\> identity classes from output/\*\_tracking.csv, reads crops back from the source clips (real bbox when the CSV carries x1..y2; otherwise a fixed box\_size square around the centroid, so legacy 12-column CSVs stay usable), and fine-tunes a MegaDescriptor backbone (timm) with a self-contained **ArcFace** margin head. The result is saved to model\_reid/megadescriptor\_finetuned.pt and loaded automatically at inference (reid\_checkpoint).  
  - *Purpose:* Adapt the appearance embedding to the specific herd and site, which is what most improves Re-ID in uniform-coat groups — with no manual identity annotation.  
  - *Parameters (CLI):* \--backbone, \--epochs, \--max-per-track, \--min-crops, \--box-size, \--batch, \--export-only, \--device.  
  - *Implementation:* mine\_classes(), parse\_tracking\_csv(), export\_crops(), train\_embedding() with an ArcFace head; torch, timm, torchvision, cv2.  
- **\[Plain\]**  
  - *How it works:* Because the tracker already groups frames of the same animal under one number, those groups are used as free examples to teach a recognition model what each animal looks like here — no hand labelling. The improved model is saved and used automatically next time.  
  - *Purpose:* Make the system better at telling your specific animals apart over time, by itself.  
  - *Parameters:* the command-line training options above.

### 6.8 Whole-horse & body-part segmentation auto-training (BehaveAI\_segmentation.py)

- **\[Technical\]**  
  - *How it works:* Two stages, no manual masks. **Stage 1 (whole horse):** sample frames from clips, run ultralytics auto\_annotate (the project detector proposes boxes, SAM2 turns each into a mask) → YOLO-seg labels on the real drone frames → train a YOLO11-seg "horse" model, usable as the Re-ID foreground (reid\_foreground=yoloseg). **Stage 2 (parts: head/neck/torso/tail/leg):** a **mandatory PASCAL-Part seed** — convert PASCAL-Part horse part masks to YOLO-seg (pascal-to-yolo) and pre-train a parts model — then **self-train**: the seed model pseudo-labels the user's drone horse-crops (confidence-filtered) and training continues from the seed weights, adapting the parts to the overhead view. Segmentation transfers to top-down views far better than side-view pose models, which is why this route is preferred over keypoints.  
  - *Purpose:* Provide a robust silhouette (foreground/orientation) and coarse body-part regions for the Re-ID descriptor, domain-adapted to the user's footage, without manual annotation.  
  - *Parameters (subcommands):* sample-frames, autolabel-horse, train-horse, pascal-to-yolo, train-parts, pseudolabel-parts, finetune-parts; \--conf, \--epochs, \--imgsz, \--device, \--box… etc.  
  - *Implementation:* ultralytics auto\_annotate / SAM / YOLO(-seg); mask\_to\_polygons(), pascal\_to\_yolo(), \_part\_to\_class(); cv2, numpy, scipy.io.loadmat.  
- **\[Plain\]**  
  - *How it works:* The system outlines each horse automatically (using a "segment anything" model on its own detections), trains a fast outliner on your real footage, then learns body parts — first from a public horse-parts dataset, then refined on your own clips using its own guesses. No masks are drawn by hand.  
  - *Purpose:* Get clean outlines and rough body-part regions tuned to your videos, to feed the Re-ID step, without manual labelling.  
  - *Parameters:* the sub-step commands above.

---

## 7\. ACTIVITY BUDGET ANALYSIS (BehaveAI\_activity\_budget.py)

### 7.1 Inputs & group-type extraction

- **\[Technical\]** *How it works:* Reads all \*\_tracking.csv from output\_dir; optionally merges groups\_metadata.csv (manual group ID/type and excluded track IDs). Group type is parsed from the filename via ab\_group\_type\_separator \+ ab\_group\_type\_field\_index, overridable per video. *Purpose:* assemble per-video tracks and group metadata. *Parameters:* ab\_group\_type\_separator, ab\_group\_type\_field\_index. *Implementation:* parse\_tracking\_csv(), extract\_group\_type(), load\_groups\_metadata(); csv, glob, configparser, collections.defaultdict.  
- **\[Plain\]** *How it works:* It reads all the per-video tables and works out each video's group type from its filename, with optional manual corrections. *Purpose:* gather the data and group labels. *Parameters:* the filename separator and field position.

### 7.2 Stranger detection

- **\[Technical\]**  
  - *How it works:* Per track, presence\_ratio \= n\_frames\_present / total\_frames. Tracks below ab\_min\_presence\_ratio are flagged stranger. An entry point within the border zone (ab\_border\_zone\_ratio) combined with short presence adds short\_presence+border\_entry. groups\_metadata.csv exclusions are flagged manual\_exclude. Strangers are kept with individual\_type \= stranger.  
  - *Purpose:* Separate resident group members from transient/edge intruders without discarding their data.  
  - *Parameters:* ab\_min\_presence\_ratio, ab\_border\_zone\_ratio.  
  - *Implementation:* flag\_strangers(); arithmetic over parsed tracks.  
- **\[Plain\]**  
  - *How it works:* An individual seen in too few frames is marked a stranger; if it also first appears at the edge of the image and stays briefly, that is noted; manual exclusions are honoured. Strangers are kept but labelled.  
  - *Purpose:* Tell regular members apart from passers-by.  
  - *Parameters:* the minimum-presence fraction and the edge-zone width.

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
  - *How it works:* activity\_budget\_individual.csv — one row per individual per video (video\_filename, group\_id, group\_type, track\_id, individual\_type, auto\_flagged, n\_frames\_present, duration\_s, presence\_ratio, the behavior\_\* columns, dominant-time/-count). activity\_budget\_suspects.csv — one row per flagged stranger (...flag\_reason, presence\_ratio, first/last\_seen\_frame, first/last\_seen\_timecode (MM:SS), border\_entry, auto\_flagged, manual\_exclude).  
  - *Purpose:* Deliver analysis-ready per-individual budgets and a separate audit list of flagged strangers.  
  - *Parameters:* inherited from §7.1–7.3.  
  - *Implementation:* run\_activity\_budget(); csv.  
- **\[Plain\]**  
  - *How it works:* Two tables are written — one summarising every individual's behaviour times, one listing every flagged stranger with when it was seen.  
  - *Purpose:* Provide the final summary and a checkable stranger list.  
  - *Parameters:* as above.

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

- **\[Technical\]** *How it works:* Identical pipeline to batch (§6.3–6.4: motion image, dual primary detection, merging, secondary classification, Kalman tracking, NCNN with fallback); arrows drawn live. A session CSV det\_\<timestamp\>.csv (same columns as batch) is written **only while "Record on detection" is on**, to avoid logging empty frames. *Purpose:* live inference with selective, non-redundant logging. *Parameters:* as §6. *Implementation:* shared KalmanTracker/YOLO logic; csv.  
- **\[Plain\]** *How it works:* Live video is processed exactly like batch videos, and detection rows are written to a file only while detection-recording is on. *Purpose:* run the full analysis live without logging empty frames. *Parameters:* same as batch.

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

## 11\. SETTINGS FILE (BehaveAI\_settings.ini) — Complete Parameter Reference

### Class configuration

| Parameter | Description |
| :---- | :---- |
| primary\_motion\_classes | Comma-separated class names for primary motion detection |
| primary\_motion\_colors | Semicolon-separated RGB triples (e.g. 255,0,0;0,255,0) |
| primary\_motion\_hotkeys | Comma-separated single-character hotkeys |
| primary\_static\_classes | As above for primary static detection |
| primary\_static\_colors | As above |
| primary\_static\_hotkeys | As above |
| secondary\_motion\_classes | Sub-classes for hierarchical motion classification |
| secondary\_motion\_colors | As above |
| secondary\_motion\_hotkeys | As above |
| secondary\_static\_classes | Sub-classes for hierarchical static classification |
| secondary\_static\_colors | As above |
| secondary\_static\_hotkeys | As above |
| ignore\_secondary | Comma-separated primary class names exempt from secondary classification |
| dominant\_source | confidence, motion, or static |

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
| val\_frequency | 0.1 | Fraction of frames assigned to the validation set |
| save\_empty\_frames | true | Save frames with no annotations (negative examples) |
| motion\_blocks\_static | false | Grey out motion boxes in static images |
| static\_blocks\_motion | false | Grey out static boxes in motion images |

### Model training

| Parameter | Default | Description |
| :---- | :---- | :---- |
| primary\_classifier | yolo11s.pt | Base detector for primary models (yolo8/11/26, n/s/m/l) |
| primary\_epochs | 50 | Training epochs for primary models |
| secondary\_classifier | yolo11s-cls.pt | Base classifier for secondary models |
| secondary\_epochs | 50 | Training epochs for secondary models |
| use\_ncnn | false | Export and use NCNN format for inference |

### Inference

| Parameter | Default | Description |
| :---- | :---- | :---- |
| primary\_conf\_thresh | 0.5 | Primary detection confidence threshold (0–1) |
| secondary\_conf\_thresh | 0.5 | Secondary classification confidence threshold (0–1) |

### Tracking

| Parameter | Default | Description |
| :---- | :---- | :---- |
| match\_distance\_thresh | 200 | Max pixel distance for detection-to-track association |
| delete\_after\_missed | 5 | Consecutive missed frames before a track is deleted |
| centroid\_merge\_thresh | 50 | Max centroid distance to merge static+motion detections |
| iou\_thresh | 0.95 | IOU threshold for detection merging / overlap handling |

### Re-Identification (intra-video)

| Parameter | Default | Description |
| :---- | :---- | :---- |
| reid\_enabled | false | Enable intra-video Re-ID (false \= original tracker behaviour, exactly) |
| reid\_method | histogram | Appearance backend: histogram, embedding (MobileNetV3), or megadescriptor (timm) |
| reid\_descriptor | global | global (single histogram) or grid (per-cell body-part histograms) |
| reid\_grid | 3x3 | Grid size R×C for the grid descriptor |
| reid\_foreground | hsv | Foreground source for masking: hsv (green-hue rule), sam2, or yoloseg |
| reid\_orient | false | Align the grid to the body's major axis (PCA on the foreground mask) before binning |
| reid\_backbone | T-224 | MegaDescriptor variant: T-224, L-224, L-384, or T-CNN-288 |
| reid\_checkpoint | (auto) | Path to a fine-tuned MegaDescriptor checkpoint; empty \= auto-detect model\_reid/megadescriptor\_finetuned.pt |
| reid\_similarity\_threshold | 0.75 | Cosine threshold for the embedding / megadescriptor methods |
| reid\_histogram\_min\_similarity | 0.60 | Cosine threshold for the histogram / grid method |
| reid\_max\_position\_distance | 500 | Max pixel distance for the spatio-temporal recovery gate (dominant signal) |
| reid\_max\_disappeared\_seconds | 180 | Registry pruning guard in seconds (NOT a hard match limit) |

### Kalman filter \[kalman\] section

| Parameter | Default | Description |
| :---- | :---- | :---- |
| process\_noise\_pos | 0.01 | Positional process noise injected per frame |
| process\_noise\_vel | 0.1 | Velocity process noise injected per frame |
| measurement\_noise | 0.1 | Assumed centroid measurement noise |

### Display

| Parameter | Default | Description |
| :---- | :---- | :---- |
| line\_thickness | 1 | Bounding-box and text line thickness |
| font\_size | 0.5 | Label text scale |

### Data augmentation

| Parameter | Default | Description |
| :---- | :---- | :---- |
| aug\_global\_probability | 0 | Master gate probability (0 disables augmentation) |
| aug\_brightness\_range | 0.8,1.2 | Min,max brightness factor |
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

### Activity budget

| Parameter | Default | Description |
| :---- | :---- | :---- |
| ab\_min\_presence\_ratio | 0.10 | Minimum frame-fraction to count as a group member |
| ab\_border\_zone\_ratio | 0.15 | Border-zone width as fraction of image size |
| ab\_group\_type\_separator | \_ | Filename field separator |
| ab\_group\_type\_field\_index | 4 | Zero-based filename field index encoding group type |

---

## 12\. INSTALLERS & LAUNCHERS

### 12.1 Linux launcher (Linux\_Launcher.sh)

- **\[Technical\]**  
  - *How it works:* Self-bootstrapping bash script; bootstrap runs once. Installs apt packages (python3-venv, python3-pip, build-essential, ffmpeg, python3-opencv, …), creates \~/ultralytics-venv with \--system-site-packages, pip-installs ultralytics\[export\], numpy, tqdm, pillow, then drops a .ultralytics\_ready marker. Launches BehaveAI.py if present, else a script path argument.  
  - *Purpose:* One-command environment setup and launch on Linux.  
  - *Parameters:* optional script path argument.  
  - *Implementation:* bash, apt, python3 \-m venv, pip.  
- **\[Plain\]**  
  - *How it works:* On first run it installs everything needed and creates an isolated Python environment, then starts the program; later runs skip setup.  
  - *Purpose:* Get running on Linux in one step.  
  - *Parameters:* an optional script to launch.

### 12.2 Windows launcher (Windows\_Launcher.bat \+ Windows\_Launcher\_ps.ps1)

- **\[Technical\]**  
  - *How it works:* The .bat opens PowerShell with execution-policy bypass. The script detects Python (py \-3/python/python3), offering to silently install Python 3.12 (64-bit); creates %USERPROFILE%\\ultralytics-venv; detects NVIDIA GPUs via nvidia-smi and WMI Win32\_VideoController; offers a PyTorch wheel choice (CPU, auto-CUDA, or manual cu121/cu118/cu117/cu116); installs ultralytics\[export\], numpy, tqdm, pillow, opencv-python, ncnn, and the chosen PyTorch; verifies ultralytics/torch/cv2 imports; writes .ultralytics\_ready; logs to Windows\_Launcher\_ps.log.  
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

projects/

└── \<project\_name\>/

    ├── BehaveAI\_settings.ini

    ├── static\_annotations.yaml

    ├── motion\_annotations.yaml

    ├── clips/                          ← training videos

    ├── input/                          ← batch input videos

    ├── output/                         ← \*\_detected.mp4, \*\_tracking.csv,

    │                                      activity\_budget\_individual.csv,

    │                                      activity\_budget\_suspects.csv,

    │                                      det\_\*.mp4 / det\_\*.csv (live)

    ├── annot\_static/

    │   ├── images/{train,val}/

    │   ├── labels/{train,val}/

    │   └── masks/{train,val}/          ← grey-box coordinates (.mask.txt)

    ├── annot\_motion/

    │   ├── images/{train,val}/

    │   ├── labels/{train,val}/

    │   └── masks/{train,val}/

    ├── annot\_static\_crop/              ← secondary classification crops (static)

    │   └── \<primary\_class\>/\<secondary\_class\>/

    ├── annot\_motion\_crop/              ← secondary classification crops (motion)

    │   └── \<primary\_class\>/\<secondary\_class\>/

    ├── model\_primary\_static/

    │   ├── train/weights/best.pt

    │   ├── train/results.csv

    │   ├── train\_count.txt

    │   └── saved\_settings.ini

    ├── model\_primary\_motion/                       (same structure)

    ├── model\_secondary\_static\_\<primary\_class\>/     (same structure)

    ├── model\_secondary\_motion\_\<primary\_class\>/     (same structure)

    ├── model\_reid/                     ← reid\_crops/, megadescriptor\_finetuned.pt (§6.7)

    ├── model\_segmentation/             ← horse/ \+ parts YOLO-seg datasets & weights (§6.8)

    └── groups\_metadata.csv             ← optional manual overrides for activity budget

---

## 14\. METHODS SUMMARY (for a scientific Materials & Methods section)

- **\[Technical\]**  
  - *Acquisition:* video clips (optionally captured live via OpenCV/picamera2) stored per project.  
  - *Representation:* each frame is encoded as (i) a static RGB image and (ii) a false-colour motion image from temporal frame differencing — exponential multi-timescale decay (expA, expB) or sequential differencing, blended with luminance (lum\_weight) and per-channel amplification (rgb\_multipliers), optionally differential (chromatic\_tail\_only).  
  - *Annotation:* random-frame sampling, dual-stream YOLO-format boxes, optional hierarchical secondary crops, optional cross-stream grey masking, train/val split \= val\_frequency.  
  - *Augmentation:* offline, per-image probabilistic photometric/geometric transforms (PIL, cv2); motion images restricted to geometry/texture; geometric label correction.  
  - *Models:* separate primary YOLO detectors per stream (Ultralytics; configurable backbone/epochs), optional per-primary-class secondary YOLO classifiers (224 px), optional NCNN export; transfer learning on dataset growth.  
  - *Inference:* dual detection → centroid/IOU merge with dominant\_source rule → secondary classification → constant-velocity Kalman tracking with Hungarian assignment (scipy).  
  - *Re-identification:* intra-video identity recovery via a mandatory spatio-temporal gate plus a part-based appearance descriptor (foreground-masked, optionally body-axis-aligned grid of HSV histograms; optional MegaDescriptor embedding), with optional self-supervised fine-tuning from tracker IDs and SAM2 / PASCAL-Part segmentation auto-training — no manual identity or part labels.  
  - *Analysis:* per-individual activity budgets (time/frequency/percentage per behaviour) and rule-based stranger flagging (ab\_min\_presence\_ratio, ab\_border\_zone\_ratio).  
  - *Reproducibility:* per-model saved\_settings.ini, train\_count.txt, model backups (\*\_backupN), and Regenerate\_annotations.py for settings-consistent image rebuilds. Cite exact values from §11.  
- **\[Plain\]**  
  - Videos are recorded and stored per study. Each frame is kept in two forms: the normal photo and a colour image that shows movement. Frames are labelled at random across all clips, with optional finer categories. The dataset can be enlarged with modified copies. Two detectors are trained — one per form — plus optional finer classifiers, then used together: both forms are searched, duplicates are merged, finer categories are added, and each individual is tracked over time. Finally, the system measures how long and how often each individual did each behaviour and flags likely strangers. Exact settings used should be reported from the table in §11.

