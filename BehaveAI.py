#!/usr/bin/env python3
"""
BehaveAI Launcher (project-aware)

- Projects are stored in ./projects/
- Each project is a subdirectory containing:
	- BehaveAI_settings.ini
	- clips/ input/ output/ models/ yamls/  (created by the launcher)
- Launches scripts with the project's directory as cwd and passes the project path as argument

"""
import tkinter as tk
from tkinter import scrolledtext, Button, Frame, ttk, simpledialog, messagebox
import tkinter.font as tkfont
import subprocess
import threading
import queue
import sys
import os
import re
import csv
import base64
import time
from pathlib import Path
import configparser
from BehaveAI_augmentation import load_augmentation_config
from BehaveAI_settings_help import apply_theme, Tooltip, BUTTON_HELP
from behaveai_holdout import is_holdout_video, video_label_for_annotation


# -------------------------- utils --------------------------

def is_progress_line(s: str) -> bool:
	# legacy progress bar pattern (kept for older tools)
	if re.search(r"\d+% *\|", s):
		return True
	# Ultralytics-style epoch progress lines: start with "N/M" then later contain "XX%"
	# ~ if re.search(r"^\s*\d+/\d+\b.*\b\d{1,3}%\b", s):
	if re.search(r"^[ \t]+\d+/\d+\s", s):
		return True
	return False

ansi_escape = re.compile(r'\x1B\[[0-9;?]*[ -/]*[@-~]')

def strip_ansi(s: str) -> str:
	return ansi_escape.sub('', s)

# --------------------- text redirector ---------------------

class TextRedirector:
	def __init__(self, text_widget, tag):
		self.text = text_widget
		self.tag = tag
		self.last_tag = f"last_insert_{tag}"
		self.text.tag_configure(self.last_tag, background="")

	def _is_view_at_bottom(self) -> bool:
		try:
			top, bottom = self.text.yview()
			return bottom >= 0.995
		except Exception:
			return True

	def write(self, s):
		at_bottom = self._is_view_at_bottom()
		self.text.configure(state='normal')
		self.text.tag_remove(self.last_tag, "1.0", "end")
		self.text.insert("end", s, (self.tag, self.last_tag))
		if at_bottom:
			self.text.see("end")
		self.text.configure(state='disabled')

	def overwrite(self, s):
		at_bottom = self._is_view_at_bottom()
		self.text.configure(state='normal')
		ranges = self.text.tag_ranges(self.last_tag)
		if ranges:
			self.text.delete(ranges[0], ranges[1])
		self.text.insert("end", s, (self.tag, self.last_tag))
		if at_bottom:
			self.text.see("end")
		self.text.configure(state='disabled')


# --------------------- main app ---------------------

class ScriptRunnerApp:
	def __init__(self, root):
		self.root = root
		root.title("BehaveAI Launcher")
		apply_theme(root)
		root.geometry("1040x620")
		root.minsize(900, 540)

		# ===== Logo + action buttons row =====
		base64_image = "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"
		image_data = base64.b64decode(base64_image)
		self.logo_img = tk.PhotoImage(data=image_data)


		# Project storage dir (next to the launcher)
		self.base_dir = Path(os.getcwd())
		self.projects_dir = self.base_dir / "projects"
		self.projects_dir.mkdir(exist_ok=True)

		self.current_project = None  # Path object or None

		# ===== Top row: project controls =====
		self.project_frame = ttk.Frame(root)
		self.project_frame.pack(fill='x', padx=8, pady=(8, 0))

		# logo
		tk.Label(self.project_frame, image=self.logo_img, background='#f4f5f7').pack(side=tk.LEFT, padx=10)
		ttk.Label(self.project_frame, text="Project:").pack(side="left", padx=(4,6))

		# Combobox for existing projects
		self.project_var = tk.StringVar()
		self.project_combo = ttk.Combobox(self.project_frame, textvariable=self.project_var, state="readonly", width=30)
		self.project_combo.pack(side="left", padx=(0,6))
		self.project_combo.bind("<<ComboboxSelected>>", lambda e: self.select_project(self.project_var.get()))

		# New Project button
		ttk.Button(self.project_frame, text="New project", command=self.create_new_project, width=12).pack(side="left", padx=(0,6))

		# Refresh projects button
		ttk.Button(self.project_frame, text="Refresh", command=self.refresh_projects, width=8).pack(side="left", padx=(0,6))

		# Current project label
		self.current_label_var = tk.StringVar(value="(no project selected)")
		ttk.Label(self.project_frame, textvariable=self.current_label_var).pack(side="left", padx=(10,6))

		# ===== Action buttons, grouped by pipeline stage =====
		# Each stage is a labelled box so the workflow order is obvious to new users.
		self.button_frame = ttk.Frame(root)
		self.button_frame.pack(pady=8, padx=8, fill='x')

		stages = [
			("1 - Setup",    [("Settings", "BehaveAI_settings_gui.py")]),
			("2 - Annotate", [("Annotate", "BehaveAI_annotation.py"),
							   ("Annotate complex", "BehaveAI_annotation_complex.py"),
							   ("Inspect Dataset", "BehaveAI_inspect_dataset.py"),
							   ("Augment Dataset", "BehaveAI_augmentation.py")]),
			("3 - Train",    [("Train & batch classify", "BehaveAI_classify_track.py"),
							   ("Train complex model", "BehaveAI_complex_model.py"),
							   ("Propose candidates", "BehaveAI_complex_candidates.py")]),
			("4 - Run",      [("Live", "BehaveAI_live.py")]),
		]

		# action buttons (initially disabled: enable after project selection)
		self.buttons = {}
		for stage_title, items in stages:
			box = ttk.LabelFrame(self.button_frame, text=stage_title)
			box.pack(side=tk.LEFT, padx=(0, 8), pady=2, fill='y', anchor='n')
			for (label_text, script_name) in items:
				b = ttk.Button(box, text=label_text,
						   command=lambda s=script_name: self.run_script(s),
						   width=20, state="disabled")
				b.pack(side=tk.TOP, padx=6, pady=3, fill='x')
				Tooltip(b, BUTTON_HELP.get(script_name, ""))
				self.buttons[script_name] = b

		# ===== Integrated output / log =====
		ttk.Label(self.root, text="Output / log", style='Section.TLabel').pack(
			anchor='w', padx=10, pady=(4, 0))

		# Monospace so progress bars and the stats tables stay aligned.
		mono = tkfont.nametofont("TkFixedFont").copy()
		self.output_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='normal', height=22, font=mono)
		self.output_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
		self.output_area.tag_config('stdout', foreground='#1a1a1a')
		self.output_area.tag_config('stderr', foreground='#b35a00')

		self.stdout_rd = TextRedirector(self.output_area, 'stdout')
		self.stderr_rd = TextRedirector(self.output_area, 'stderr')

		self.output_queue = queue.Queue()
		self.output_buffer = {'stdout': b'', 'stderr': b''}
		self.last_progress_global = None

		# populate projects list
		self.refresh_projects()

		# update loop
		self.update_output()

	# --------------------- project management ---------------------

	def list_projects(self):
		projects = []
		for p in sorted(self.projects_dir.iterdir()):
			if p.is_dir():
				projects.append(p.name)
		return projects

	def is_settings_populated(self, project_path: Path) -> bool:
		"""Return True if project's BehaveAI_settings.ini exists and at least one primary
		class list is non-empty (not '0' or empty)."""
		ini = project_path / "BehaveAI_settings.ini"
		if not ini.exists():
			return False
		cfg = configparser.ConfigParser()
		try:
			cfg.read(ini)
			d = cfg['DEFAULT'] if 'DEFAULT' in cfg else cfg.defaults()
			def nonempty_list_field(s):
				if s is None:
					return False
				s = s.strip()
				if s == '' or s == '0':
					return False
				# if any non-empty token after splitting by comma that isn't '0'
				for token in s.split(','):
					if token.strip() and token.strip() != '0':
						return True
				return False
			return nonempty_list_field(d.get('primary_motion_classes', fallback='0')) \
				   or nonempty_list_field(d.get('primary_static_classes', fallback='0'))
		except Exception:
			return False

	def update_button_states(self):
		"""Enable/disable launcher buttons depending on current_project and settings file contents.

		Settings button is always enabled when a project is selected; other action buttons
		are enabled only when settings are present/populated.
		"""
		if self.current_project is None:
			# no project selected -> disable everything
			for btn in self.buttons.values():
				btn.config(state='disabled')
			return

		# Settings button should always be enabled for a selected project
		settings_script = "BehaveAI_settings_gui.py"
		for script_name, btn in self.buttons.items():
			if script_name == settings_script:
				btn.config(state='normal')
			elif script_name == "BehaveAI_augmentation.py":
				# Special handling for Augment Dataset button
				# Check aug_global_probability from settings file
				aug_config = load_augmentation_config(self.current_project / 'BehaveAI_settings.ini')
				if aug_config is None:
					# aug_global_probability is 0, so disable the button
					btn.config(state='disabled')
				else:
					btn.config(state='normal')
			else:
				# enable if settings are populated
				ok = self.is_settings_populated(self.current_project)
				btn.config(state='normal' if ok else 'disabled')

	def refresh_projects(self):
		projects = self.list_projects()
		self.project_combo['values'] = projects
		# keep selection if current project still present
		if self.current_project and self.current_project.name in projects:
			self.project_var.set(self.current_project.name)
			self.current_label_var.set(f"Project: {self.current_project.name}")
			# enable/disable action buttons depending on settings file
			self.update_button_states()
		else:
			self.project_var.set('')
			self.current_project = None
			self.current_label_var.set("(no project selected)")
			# no project -> disable everything
			self.update_button_states()

	def display_project_stats(self, project_path: Path):
	    """
	    Collect and display project statistics in the integrated terminal.
	    Runs in a background thread to avoid blocking the UI.
	    """
	    def _collect_and_display():
	        lines = []
	        sep = "─" * 60

	        lines.append(f"\n{sep}")
	        lines.append(f"  PROJECT: {project_path.name}")
	        lines.append(sep)

	        ini_path = project_path / "BehaveAI_settings.ini"
	        if not ini_path.exists():
	            lines.append("  No settings file found.")
	            self._write_stats_lines(lines)
	            return

	        cfg = configparser.ConfigParser()
	        cfg.optionxform = str
	        cfg.read(ini_path)
	        d = cfg['DEFAULT'] if 'DEFAULT' in cfg else cfg.defaults()

	        def parse_list(key):
	            v = d.get(key, '0').strip()
	            if not v or v == '0':
	                return []
	            return [x.strip() for x in v.split(',') if x.strip() and x.strip() != '0']

	        primary_static_classes  = parse_list('primary_static_classes')
	        primary_motion_classes  = parse_list('primary_motion_classes')
	        all_primary_classes     = primary_static_classes + primary_motion_classes
	        val_frequency           = float(d.get('val_frequency', '0.1'))

	        # ── Annotation directories ─────────────────────────────────
	        annot_dirs = {
	            'static_train_img':  project_path / 'annot_static'  / 'images' / 'train',
	            'static_val_img':    project_path / 'annot_static'  / 'images' / 'val',
	            'static_train_lbl':  project_path / 'annot_static'  / 'labels' / 'train',
	            'static_val_lbl':    project_path / 'annot_static'  / 'labels' / 'val',
	            'motion_train_img':  project_path / 'annot_motion'  / 'images' / 'train',
	            'motion_val_img':    project_path / 'annot_motion'  / 'images' / 'val',
	            'motion_train_lbl':  project_path / 'annot_motion'  / 'labels' / 'train',
	            'motion_val_lbl':    project_path / 'annot_motion'  / 'labels' / 'val',
	        }

	        img_ext = ('.jpg', '.jpeg', '.png')

	        def list_images(d):
	            if not d.is_dir():
	                return []
	            return [f for f in d.iterdir()
	                    if f.suffix.lower() in img_ext]

	        def is_augmented(fname):
	            return '_aug_' in fname

	        def count_originals_and_aug(files):
	            orig = [f for f in files if not is_augmented(f.stem)]
	            aug  = [f for f in files if is_augmented(f.stem)]
	            return orig, aug

	        # Gather all images (static + motion, train + val)
	        static_train_all = list_images(annot_dirs['static_train_img'])
	        static_val_all   = list_images(annot_dirs['static_val_img'])
	        motion_train_all = list_images(annot_dirs['motion_train_img'])
	        motion_val_all   = list_images(annot_dirs['motion_val_img'])

	        # Split originals / augmented
	        static_train_orig, static_train_aug = count_originals_and_aug(static_train_all)
	        static_val_orig,   static_val_aug   = count_originals_and_aug(static_val_all)
	        motion_train_orig, motion_train_aug = count_originals_and_aug(motion_train_all)
	        motion_val_orig,   motion_val_aug   = count_originals_and_aug(motion_val_all)

	        total_orig = len(static_train_orig) + len(static_val_orig) + \
	                     len(motion_train_orig) + len(motion_val_orig)
	        total_aug  = len(static_train_aug)  + len(static_val_aug)  + \
	                     len(motion_train_aug)  + len(motion_val_aug)

	        # Unique basenames (a frame can appear in both static and motion)
	        orig_basenames = set(
	            f.stem for f in static_train_orig + static_val_orig +
	                             motion_train_orig + motion_val_orig
	        )
	        total_unique_frames = len(orig_basenames)

	        lines.append(f"\n  ANNOTATIONS")
	        lines.append(f"  {'Unique annotated frames':<35} {total_unique_frames}")
	        lines.append(f"  {'Original images (static+motion)':<35} {total_orig}")
	        lines.append(f"  {'Augmented copies':<35} {total_aug}")

	        # Train / val split
	        orig_train = len(static_train_orig) + len(motion_train_orig)
	        orig_val   = len(static_val_orig)   + len(motion_val_orig)
	        if total_orig > 0:
	            actual_val_pct = 100.0 * orig_val / total_orig
	            lines.append(f"  {'Train / Val split':<35} "
	                          f"{orig_train} / {orig_val}  "
	                          f"({actual_val_pct:.1f}% val, target {val_frequency*100:.0f}%)")

	        # ── Holdout videos (permanent, deterministic, whole-video) ─────
	        # Scoped to videos that actually have annotated frames (i.e. can serve
	        # training) — NOT every raw file sitting in input/, which may include
	        # footage that hasn't been annotated (or even tracked) yet.
	        lines.append(f"\n  HOLDOUT VIDEOS")
	        stems = sorted({video_label_for_annotation(b) for b in orig_basenames})
	        if not stems:
	            lines.append("  No annotated videos yet.")
	        else:
	            holdout_stems = [s for s in stems if is_holdout_video(s, val_frequency)]
	            pct = 100.0 * len(holdout_stems) / len(stems)
	            lines.append(f"  {'Videos in holdout':<35} "
	                          f"{len(holdout_stems)} / {len(stems)}  "
	                          f"({pct:.1f}%, target {val_frequency*100:.0f}%)")

	        # ── Per-class annotation counts ────────────────────────────
	        if all_primary_classes:
	            lines.append(f"\n  CLASS DISTRIBUTION  (from label files)")

	            # Map class index -> name for static and motion
	            static_class_map = {i: name for i, name in enumerate(primary_static_classes)}
	            motion_class_map = {i: name for i, name in enumerate(primary_motion_classes)}

	            class_counts = {name: 0 for name in all_primary_classes}

	            def tally_labels(lbl_dir, class_map):
	                if not lbl_dir.is_dir():
	                    return
	                for lf in lbl_dir.iterdir():
	                    if lf.suffix != '.txt':
	                        continue
	                    # Skip augmented label files
	                    if is_augmented(lf.stem):
	                        continue
	                    try:
	                        for line in lf.read_text().splitlines():
	                            parts = line.strip().split()
	                            if not parts:
	                                continue
	                            cls_idx = int(parts[0])
	                            name = class_map.get(cls_idx)
	                            if name and name in class_counts:
	                                class_counts[name] += 1
	                    except Exception:
	                        pass

	            tally_labels(annot_dirs['static_train_lbl'], static_class_map)
	            tally_labels(annot_dirs['static_val_lbl'],   static_class_map)
	            tally_labels(annot_dirs['motion_train_lbl'], motion_class_map)
	            tally_labels(annot_dirs['motion_val_lbl'],   motion_class_map)

	            total_boxes = sum(class_counts.values())
	            for name in all_primary_classes:
	                count = class_counts[name]
	                pct   = 100.0 * count / total_boxes if total_boxes > 0 else 0.0
	                warn  = "  ⚠ under-represented" if pct < 10 and total_boxes > 0 else ""
	                lines.append(f"    {name:<28} {count:>5} boxes  ({pct:5.1f}%){warn}")

	        # ── Videos in clips dir ────────────────────────────────────
	        clips_dir_raw = d.get('clips_dir', 'clips')
	        clips_dir = Path(clips_dir_raw) if os.path.isabs(clips_dir_raw) \
	                    else project_path / clips_dir_raw

	        video_ext = ('.mp4', '.avi', '.mov', '.mkv')
	        if clips_dir.is_dir():
	            all_videos = [f for f in clips_dir.iterdir()
	                          if f.suffix.lower() in video_ext]
	            # Which videos have at least one annotated frame?
	            annotated_video_labels = set()
	            for stem in orig_basenames:
	                if '_' in stem:
	                    vlabel = stem.rsplit('_', 1)[0]
	                    annotated_video_labels.add(vlabel)

	            annotated_videos = [v for v in all_videos
	                                 if v.stem in annotated_video_labels]

	            lines.append(f"\n  VIDEOS")
	            lines.append(f"  {'Total videos in clips/':<35} {len(all_videos)}")
	            lines.append(f"  {'Videos with annotations':<35} "
	                          f"{len(annotated_videos)} / {len(all_videos)}")

	        # ── Augmented data breakdown (merged with parameters) ─────
	        all_aug_files = (static_train_aug + static_val_aug +
	                         motion_train_aug + motion_val_aug)
	        if all_aug_files:
	            # Count files per augmentation type
	            aug_type_counts = {}
	            for f in all_aug_files:
	                parts = f.stem.split('_aug_')
	                if len(parts) >= 2:
	                    aug_type = parts[-1]
	                    aug_type_counts[aug_type] = aug_type_counts.get(aug_type, 0) + 1

	            # Parameters defined in INI (canonical order)
	            aug_params = [
	                ('brightness',   'aug_brightness_range',   'aug_brightness_probability'),
	                ('contrast',     'aug_contrast_range',     'aug_contrast_probability'),
	                ('saturation',   'aug_saturation_range',   'aug_saturation_probability'),
	                ('hue',          'aug_hue_range',          'aug_hue_probability'),
	                ('sharpness',    'aug_sharpness_range',    'aug_sharpness_probability'),
	                ('blur',         'aug_blur_range',         'aug_blur_probability'),
	                ('noise',        'aug_noise_range',        'aug_noise_probability'),
	                ('shear',        'aug_shear_range',        'aug_shear_probability'),
	                ('flip_h',       'aug_flip_h_options',     'aug_flip_h_probability'),
	                ('flip_v',       'aug_flip_v_options',     'aug_flip_v_probability'),
	                ('temperature',  'aug_temperature_range',  'aug_temperature_probability'),
	            ]

	            global_prob = d.get('aug_global_probability', '0')
	            lines.append(f"\n  AUGMENTED DATA  ({total_aug} total files)")
	            lines.append(f"  {'Global probability':<35} {global_prob}")
	            # Unified header
	            lines.append(f"    {'Parameter':<14} {'Prob':>6}  {'Range':<20}  {'Files':>6}")
	            lines.append(f"    {'-'*14} {'-'*6}  {'-'*20}  {'-'*6}")
	            for param, range_key, prob_key in aug_params:
	                prob = float(d.get(prob_key, '0'))
	                if prob > 0:
	                    rng   = d.get(range_key, '—')
	                    count = aug_type_counts.get(param, 0)
	                    lines.append(f"    {param:<14} {prob:>6.2f}  {rng:<20}  {count:>5} files")

	        # ── Model metrics ──────────────────────────────────────────
	        model_dirs = [
	            ('Primary static',  project_path / 'model_primary_static'  / 'train'),
	            ('Primary motion',  project_path / 'model_primary_motion'  / 'train'),
	        ]
	        # Also pick up any secondary models
	        for p in sorted(project_path.iterdir()):
	            if p.is_dir() and p.name.startswith('model_secondary') \
	               and not re.search(r'_backup\d+$', p.name):
	                model_dirs.append((p.name.replace('model_', '').replace('_', ' '),
	                                   p / 'train'))

	        model_section_printed = False
	        for model_label, train_dir in model_dirs:
	            results_csv = train_dir / 'results.csv'
	            weights     = train_dir / 'weights' / 'best.pt'
	            if not results_csv.exists():
	                continue

	            if not model_section_printed:
	                lines.append(f"\n  MODEL METRICS  (last epoch)")
	                model_section_printed = True

	            try:
	                import csv as _csv
	                with open(results_csv, newline='') as fh:
	                    reader = _csv.DictReader(fh)
	                    rows = list(reader)
	                if not rows:
	                    continue
	                last = rows[-1]

	                # Strip whitespace from keys (YOLO sometimes pads them)
	                last = {k.strip(): v.strip() for k, v in last.items()}

	                # Common YOLO metrics keys (vary slightly by version)
	                def get_metric(keys):
	                    for k in keys:
	                        if k in last:
	                            try:
	                                return float(last[k])
	                            except ValueError:
	                                pass
	                    return None

	                epochs = len(rows)

	                size_str = ''
	                if weights.exists():
	                    size_mb  = weights.stat().st_size / 1_048_576
	                    mtime    = weights.stat().st_mtime
	                    mod_date = time.strftime('%Y-%m-%d', time.localtime(mtime))
	                    size_str = f"  weights: {size_mb:.1f} MB  ({mod_date})"

	                # ── Secondary models are YOLO *classifiers* (crops -> class):
	                # their results.csv has accuracy_top1/top5 + loss, not the
	                # detection mAP/precision columns handled below. ──────────
	                is_classifier = ('metrics/accuracy_top1' in last
	                                 or 'metrics/accuracy_top5' in last)
	                if is_classifier:
	                    lines.append(f"\n    [{model_label}]  (classifier)  {epochs} epochs{size_str}")
	                    cls_metrics = [
	                        ('Top-1 accuracy', ['metrics/accuracy_top1']),
	                        ('Top-5 accuracy', ['metrics/accuracy_top5']),
	                        ('Loss (train)',   ['train/loss']),
	                        ('Loss (val)',     ['val/loss']),
	                    ]
	                    for label, col_names in cls_metrics:
	                        val = get_metric(col_names)
	                        if val is not None:
	                            lines.append(f"    {label:<22} {val:.4f}")
	                    continue

	                lines.append(f"\n    [{model_label}]  {epochs} epochs{size_str}")

	                # ── Detection metrics (box) ────────────────────────────
	                # Each entry: (ini_key, display_label, csv_column_names)
	                metric_groups = [
	                    ('show_metric_precision_b',    'Precision (B)',     ['metrics/precision(B)', 'precision']),
	                    ('show_metric_recall_b',       'Recall (B)',        ['metrics/recall(B)',    'recall']),
	                    ('show_metric_map50_b',        'mAP@0.5 (B)',       ['metrics/mAP50(B)',     'mAP_0.5']),
	                    ('show_metric_map5095_b',      'mAP@0.5:0.95 (B)', ['metrics/mAP50-95(B)', 'mAP_0.5:0.95']),
	                    ('show_metric_precision_m',    'Precision (M)',     ['metrics/precision(M)']),
	                    ('show_metric_recall_m',       'Recall (M)',        ['metrics/recall(M)']),
	                    ('show_metric_map50_m',        'mAP@0.5 (M)',       ['metrics/mAP50(M)']),
	                    ('show_metric_map5095_m',      'mAP@0.5:0.95 (M)', ['metrics/mAP50-95(M)']),
	                    ('show_metric_box_loss_train', 'Box loss (train)',  ['train/box_loss',  'train/box_om']),
	                    ('show_metric_cls_loss_train', 'Cls loss (train)',  ['train/cls_loss',  'train/cls_om']),
	                    ('show_metric_dfl_loss_train', 'DFL loss (train)',  ['train/dfl_loss',  'train/dfl_om']),
	                    ('show_metric_box_loss_val',   'Box loss (val)',    ['val/box_loss',    'val/box_om']),
	                    ('show_metric_cls_loss_val',   'Cls loss (val)',    ['val/cls_loss',    'val/cls_om']),
	                    ('show_metric_dfl_loss_val',   'DFL loss (val)',    ['val/dfl_loss',    'val/dfl_om']),
	                    ('show_metric_lr_pg0',         'LR pg0',            ['x/lr0', 'lr/pg0']),
	                    ('show_metric_lr_pg1',         'LR pg1',            ['x/lr1', 'lr/pg1']),
	                    ('show_metric_lr_pg2',         'LR pg2',            ['x/lr2', 'lr/pg2']),
	                ]

	                # Default visibility (True if key absent from INI)
	                default_on = {
	                    'show_metric_precision_b':  True,
	                    'show_metric_recall_b':     True,
	                    'show_metric_f1_b':         True,
	                    'show_metric_map50_b':      True,
	                    'show_metric_map5095_b':    True,
	                }

	                for ini_key, label, col_names in metric_groups:
	                    show = d.get(ini_key, str(default_on.get(ini_key, False)).lower())
	                    if show.lower() != 'true':
	                        continue
	                    val = get_metric(col_names)
	                    if val is not None:
	                        lines.append(f"    {label:<22} {val:.4f}")

	                # F1-score (calculated from Precision + Recall)
	                if d.get('show_metric_f1_b', 'true').lower() == 'true':
	                    precision = get_metric(['metrics/precision(B)', 'precision'])
	                    recall    = get_metric(['metrics/recall(B)',    'recall'])
	                    if precision is not None and recall is not None and (precision + recall) > 0:
	                        f1 = 2 * (precision * recall) / (precision + recall)
	                        lines.append(f"    {'F1-score (B)':<22} {f1:.4f}")

	            except Exception as e:
	                lines.append(f"    [{model_label}]  could not read results: {e}")

	        # ── Complex-behaviour model (sklearn/torch, non-YOLO) ──────
	        # Not a YOLO run: no results.csv. Scores live in metrics.txt
	        # (macro-F1) and the fitted pipeline in pipeline.joblib. We parse
	        # the text file rather than loading the bundle (avoids pulling in
	        # torch/joblib on the launcher thread).
	        complex_dir     = project_path / 'model_complex'
	        complex_metrics = complex_dir / 'metrics.txt'
	        if complex_metrics.exists():
	            if not model_section_printed:
	                lines.append(f"\n  MODEL METRICS  (last epoch)")
	                model_section_printed = True
	            try:
	                bundle = complex_dir / 'pipeline.joblib'
	                size_str = ''
	                if bundle.exists():
	                    size_mb  = bundle.stat().st_size / 1_048_576
	                    mod_date = time.strftime('%Y-%m-%d',
	                                             time.localtime(bundle.stat().st_mtime))
	                    size_str = f"  model: {size_mb:.1f} MB  ({mod_date})"

	                tc_file = complex_dir / 'train_count.txt'
	                tc = tc_file.read_text().strip() if tc_file.exists() else '?'
	                lines.append(f"\n    [Complex behaviours]  {tc} annotated segments{size_str}")

	                text = complex_metrics.read_text(encoding='utf-8', errors='replace')
	                headline = [
	                    (r'By-video CV macro-F1 \(train pool\):\s*([0-9.]+)', 'By-video CV macro-F1'),
	                    (r'TRAIN-ONLY macro-F1:\s*([0-9.]+)',                 'Train-only macro-F1'),
	                    (r'Held-out video macro-F1[^:]*:\s*([0-9.]+)',        'Held-out macro-F1'),
	                ]
	                any_metric = False
	                for pat, label in headline:
	                    m = re.search(pat, text)
	                    if m:
	                        lines.append(f"    {label:<26} {float(m.group(1)):.3f}")
	                        any_metric = True
	                if not any_metric:
	                    lines.append(f"    (metrics.txt present but no macro-F1 score found)")
	            except Exception as e:
	                lines.append(f"    [Complex behaviours]  could not read metrics: {e}")

	        lines.append(f"\n{sep}\n")
	        self._write_stats_lines(lines)

	    threading.Thread(target=_collect_and_display, daemon=True).start()


	def _write_stats_lines(self, lines):
	    """Write a list of text lines to the integrated output terminal (thread-safe)."""
	    text = '\n'.join(lines) + '\n'
	    # Schedule on main thread — Tkinter widgets must only be touched from main thread
	    self.root.after(0, lambda: self._append_to_output(text))


	def _append_to_output(self, text):
	    """Append text to the output area on the main thread."""
	    self.output_area.configure(state='normal')
	    self.output_area.insert('end', text, ('stdout',))
	    self.output_area.see('end')
	    self.output_area.configure(state='disabled')

	def select_project(self, project_name):
		if not project_name:
			return
		proj_path = self.projects_dir / project_name
		if not proj_path.exists():
			messagebox.showerror("Project not found", f"Project '{project_name}' not found.")
			self.refresh_projects()
			return
		self.current_project = proj_path
		self.current_label_var.set(f"Project: {project_name}")
		# enable/disable buttons depending on settings
		self.update_button_states()
		self.display_project_stats(proj_path)

	def _enable_buttons(self, enable: bool):
		state = "normal" if enable else "disabled"
		for b in self.buttons.values():
			b.config(state=state)

	def create_new_project(self):
		name = simpledialog.askstring("New project", "Enter new project name (alphanumeric, - and _ allowed):", parent=self.root)
		if not name:
			return
		name = name.strip()
		# basic validation
		if any(c in name for c in r'\/:*?"<>|'):
			messagebox.showerror("Invalid name", "Project name contains invalid characters.")
			return
		target = self.projects_dir / name
		if target.exists():
			messagebox.showerror("Already exists", f"Project '{name}' already exists.")
			return

		try:
			# create structure
			target.mkdir(parents=True, exist_ok=False)
			(target / "clips").mkdir(exist_ok=True)
			(target / "input").mkdir(exist_ok=True)
			(target / "output").mkdir(exist_ok=True)
			(target / "timecodes").mkdir(exist_ok=True)
			# ~ (target / "models").mkdir(exist_ok=True)
			# ~ (target / "yamls").mkdir(exist_ok=True)

			# starter example CSV for the time-code navigation feature
			(target / "timecodes" / "example_timecodes.csv").write_text(
				"# BehaveAI - list of time-codes to annotate.\n"
				"# video_filename : name of the video file in clips/ (with or without extension).\n"
				"# timecode       : integer frame index (e.g. 1530) OR mm:ss (e.g. 01:02) or hh:mm:ss.\n"
				"# behaviour      : optional memo column (ignored by the tool).\n"
				"video_filename,timecode,behaviour\n"
				"my_video_01.mp4,1530,grazing\n"
				"my_video_01.mp4,02:15,walking\n"
				"my_video_02.mp4,00:42,fighting\n",
				encoding="utf-8")

			# create a starter BehaveAI_settings.ini in the project dir
			ini_path = target / "BehaveAI_settings.ini"
			ini_template = f"""[DEFAULT]
# Project settings for {name}
project_path = {str(target)}
clips = {str(target/'clips')}
input = {str(target/'input')}
output = {str(target/'output')}

# GUI defaults
primary_motion_classes = 0
primary_motion_colors = 0
primary_motion_hotkeys = 0
primary_static_classes = 0
primary_static_colors = 0
primary_static_hotkeys = 0

# Species detected before the primary/secondary models (model 0). The first entry
# keeps the bare primary_*/secondary_*/age_* keys above; additional species use
# suffixed keys (e.g. primary_static_classes__Bos_taurus) - see behaveai_config.py.
species_list = Equus caballus
species_colors = 0
species_hotkeys = e

# Age classes detected after species and before the primary behaviour (model 0.5).
# Predefined per species (same suffixing rule as the ethogram keys above).
age_classes = 0
age_colors = 0
age_hotkeys = 0

# Shared pool of secondary behaviours (reused across static/motion primaries).
secondary_classes = 0
secondary_colors = 0
secondary_hotkeys = 0
# Per-primary allowed secondaries: Primary1:secA|secB; Primary2:secA
secondary_map =

# === Drone motion correction ===
# Master switch; off = no behaviour change.
drone_correction_enabled = false
# Global background-motion model: affine | homography.
drone_correction_model = affine
# Box dilation (fraction of box size) when masking horses out of the background.
drone_correction_box_dilation = 0.20
# Minimum background features required to trust the estimated transform.
drone_correction_min_features = 30
# Residual flow std (px) above which a frame is flagged 'uncertain'.
drone_correction_uncertain_std = 8.0
# Centroid smoothing before differentiating: savgol | moving_average | none.
drone_correction_smoothing = savgol
# Odd window length for the smoothing filter.
drone_correction_smoothing_window = 7
# If static features are persistently too few, smooth-only (no optical-flow correction).
drone_correction_fallback_smoothing = true

# === Metric geometry (pixels -> metres, from flight-log telemetry) ===
# Master switch; off = no behaviour change. Needs a <video>.flightlog.csv sidecar.
metric_enabled = false
# Camera focal length (35mm-equivalent, mm) and reference sensor width (mm) used
# to derive the pixel focal length when no per-drone checkerboard override is set.
metric_focal_len_mm = 24.0
metric_sensor_width_mm = 36.0
# Max |gimbal roll| (deg) tolerated before a frame's metric result is flagged 'uncertain'.
metric_roll_max_deg = 3.0
# Min pixels below the horizon line required, else flagged 'uncertain' (near-horizon = huge error).
metric_horizon_margin_px = 50
# Optional per-drone pixel-focal overrides from a checkerboard calibration, e.g.:
# metric_fpx_Mini4Pro = 2560.0

# === Offline tracklet stitching (post-tracker, non-causal) ===
# Re-links the causal tracker's short tracklets into longer identities over the
# whole clip, on kinematics only. Off = no behaviour change. Runs after drone
# correction (stitches on the stabilised x_corrected coordinates).
stitch_enabled = false
# Physical speed gate in pixels/frame on the stabilised frame (an implied speed
# above this can't be the same animal). On fixed-altitude clips this is a plain
# px/frame cap; with a flight log it should track altitude (see resolve_speed_gate).
stitch_max_speed_px_per_frame = 60
# A link is kept only if its normalised gap cost (0..1) is below this -- i.e.
# cheaper than leaving the tracklets unlinked.
stitch_max_link_cost = 0.5
# Minimum samples for a tracklet to carry a velocity estimate (shorter still link).
stitch_min_tracklet_len = 2
# Refuse links whose endpoints have correction_quality 'none' (unreliable position).
stitch_quality_gate = true
# Field-recorded group size, if known. Purely a diagnostic in the stitch report;
# NEVER used as a constraint (0 / blank = unknown).
expected_group_size = 0

# Min classified frames for group_member in the activity budget (0 = skip).
ab_min_classified_frames = 5

# === Reference body length ===
# body_len_i / body_len_ref below this flags a likely foal.
foal_size_ratio_thresh = 0.7
# Reference body-length scope: video | segment (segment recomputes on scale drift).
body_len_ref_scope = video

# === Complex behaviours / interaction features ===
# Comma-separated, user-editable list of dyadic AND group behaviours.
complex_behaviours =
# Parallel comma-separated single-char hotkeys for the behaviours above.
complex_behaviours_hotkeys =
# Complex-behaviour model: baseline | lstm | transformer.
complex_model_type = baseline
# Baseline classifier: random_forest | hist_gradient_boosting.
complex_baseline_classifier = random_forest
# Deep sequence model (lstm/transformer) hyper-parameters; ignored by 'baseline'.
# Number of sub-windows a labelled segment is sliced into (sequence length).
complex_seq_steps = 8
# Training epochs for the deep model.
complex_deep_epochs = 60
# Hidden size of the LSTM / Transformer d_model.
complex_deep_hidden = 64
# Number of stacked recurrent / encoder layers.
complex_deep_layers = 1
# Attention heads (transformer only; must divide the effective d_model).
complex_deep_heads = 4
# Dropout used in the deep model.
complex_deep_dropout = 0.2
# Adam learning rate for the deep model.
complex_deep_lr = 0.001
# Mini-batch size for the deep model.
complex_deep_batch = 16
# Pairs farther apart than this (px) are ignored as interactions.
complex_max_interaction_distance = 400
# Minimum length (frames) of an interaction episode (per_segment granularity).
complex_min_duration_frames = 10
# Box IoU above this counts as contact.
complex_contact_iou_thresh = 0.05
# Distance (in body lengths) below this counts as contact.
complex_contact_dist_bodylen = 1.5
# Window length (frames) for aggregating features for the model.
complex_window_frames = 30
# Min true-vs-predicted confusion rate to flag a class pair as a merge suggestion.
complex_confusion_merge_rate = 0.20
# Minimum predicted probability to emit a complex-behaviour prediction.
complex_predict_min_proba = 0.5
# Candidate heuristics (body lengths / frame): speeds below 'low' count as ~still.
complex_speed_low_bodylen = 0.05
# Speeds above 'high' count as fast (gallop/chase).
complex_speed_high_bodylen = 0.25
# Group polarisation above this counts as aligned (trek/stampede).
complex_polarisation_high = 0.7
# Behavioural synchrony above this counts as synchronised.
complex_synchrony_high = 0.7
# Number of most-uncertain windows surfaced by active learning.
complex_candidate_topk = 50

# === Interaction graph (primary output) ===
# Edge granularity: per_interaction | per_segment | per_frame.
interaction_edge_granularity = per_interaction
# Edge weight metric: duration | proximity | combined.
interaction_weight_metric = duration

"""

			ini_path.write_text(ini_template)
		except Exception as e:
			messagebox.showerror("Error creating project", f"Failed to create project '{name}':\n{e}")
			return

		# refresh combobox and select
		self.refresh_projects()
		self.project_var.set(name)
		self.select_project(name)
		# show message
		self.output_area.configure(state='normal')
		self.output_area.insert("end", f"Created project: {name}\n")
		self.output_area.configure(state='disabled')

	# --------------------- script execution ---------------------

	def run_script(self, script_name):
		if self.current_project is None:
			messagebox.showwarning("No project", "Please select or create a project first.")
			return
		# run in background thread
		threading.Thread(target=self.execute_script, args=(script_name,), daemon=True).start()

	def execute_script(self, script_name):
		# script path is relative to this launcher file
		launcher_dir = Path(__file__).resolve().parent
		script_path = launcher_dir / script_name
		if not script_path.exists():
			# try without path (maybe it's in PATH)
			script_path = script_name

		env = os.environ.copy()
		env['PYTHONUNBUFFERED'] = '1'
		env['BEHAVEAI_PROJECT'] = str(self.current_project)

		# run with project's cwd so outputs and generation happen inside project folder
		try:
			proc = subprocess.Popen([sys.executable, '-u', str(script_path), str(self.current_project)],
									stdout=subprocess.PIPE,
									stderr=subprocess.PIPE,
									bufsize=0,
									env=env,
									cwd=str(self.current_project))
		except Exception as e:
			self.output_area.configure(state='normal')
			self.output_area.insert("end", f"Failed to start {script_name}: {e}\n")
			self.output_area.configure(state='disabled')
			return

		threading.Thread(target=self.read_stream, args=(proc.stdout, 'stdout'), daemon=True).start()
		threading.Thread(target=self.read_stream, args=(proc.stderr, 'stderr'), daemon=True).start()
		code = proc.wait()
		if code == 0:
			self.output_queue.put(('stdout', b"\nDone\n"))
		else:
			self.output_queue.put(('stdout', f"\nProcess exited with code: {code}\n".encode()))

		# Ensure button states are re-evaluated on the main thread (e.g. if the Settings GUI
		# changed or created BehaveAI_settings.ini)
		self.root.after(0, self.update_button_states)

	# --------------------- streaming output ---------------------

	def read_stream(self, stream, tag):
		while True:
			chunk = stream.read(1)
			if not chunk:
				break
			self.output_queue.put((tag, chunk))
		stream.close()

	def update_output(self):
		while not self.output_queue.empty():
			tag, data = self.output_queue.get()
			buf = self.output_buffer[tag] + data

			if buf.endswith(b'\r'):
				raw = buf[:-1]
				line_plain = strip_ansi(raw.decode('utf-8', errors='replace'))
				rd = self.stdout_rd if tag == 'stdout' else self.stderr_rd
				if is_progress_line(line_plain):
					rd.overwrite(line_plain)
					self.last_progress_global = line_plain.strip()
				else:
					rd.write(line_plain + '\n')
					self.last_progress_global = None
				buf = b''

			elif buf.endswith(b'\n'):
				raw = buf[:-1]
				line_plain = strip_ansi(raw.decode('utf-8', errors='replace'))
				rd = self.stdout_rd if tag == 'stdout' else self.stderr_rd
				line_stripped = line_plain.strip()

				if line_plain.startswith('[ WARN:0@') or line_plain.startswith('[ERROR:0@'): # don't print OpenCV camera searching errors
					continue
				else:
					if is_progress_line(line_plain):
					    # Overwrite the previous progress line instead of appending a new one.
					    rd.overwrite(line_plain + '\n')
					    self.last_progress_global = None
					# ~ if is_progress_line(line_plain):
						# ~ removed = False
						# ~ for last_tag in ('last_insert_stdout', 'last_insert_stderr'):
							# ~ ranges = self.output_area.tag_ranges(last_tag)
							# ~ if ranges:
								# ~ self.output_area.configure(state='normal')
								# ~ self.output_area.delete(ranges[0], ranges[1])
								# ~ self.output_area.tag_remove(last_tag, "1.0", "end")
								# ~ self.output_area.configure(state='disabled')
								# ~ removed = True
						# ~ rd.write(line_plain + '\n')
						# ~ self.last_progress_global = None
					else:
						if self.last_progress_global is not None and line_stripped == self.last_progress_global:
							self.last_progress_global = None
						else:
							rd.write(line_plain + '\n')
							self.last_progress_global = None
				buf = b''

			self.output_buffer[tag] = buf

		self.root.after(50, self.update_output)


# --------------------- main ---------------------

if __name__ == "__main__":
	root = tk.Tk()
	app = ScriptRunnerApp(root)
	root.mainloop()
