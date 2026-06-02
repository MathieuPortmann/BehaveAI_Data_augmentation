#!/usr/bin/env python3
"""
BehaveAI Settings Editor

This tool edits BehaveAI_settings.ini using the project dir direcotry passed to it
"""
import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import tkinter.font as tkfont
import configparser
import os
import sys
import yaml
import subprocess
import shutil
import glob
import time
from pathlib import Path
import re

INI_DEFAULT_PATH = os.path.join(os.getcwd(), 'BehaveAI_settings.ini')

CLASS_GROUPS = [
	('primary_motion', 'Primary motion'),
	('secondary_motion', 'Secondary motion'),
	('primary_static', 'Primary static'),
	('secondary_static', 'Secondary static'),
]

CLASSIFIER_OPTIONS = [
	'yolo26n.pt', 'yolo26s.pt', 'yolo26m.pt', 'yolo26l.pt',
	'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt',
	'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt',
]

RESERVED_HOTKEYS = {'u', 'g'}

DEFAULT_CLASS_COLORS = [
	(0, 220, 255),
	(0, 255, 97),
	(236, 255, 0),
	(255, 188, 0),
	(255, 97, 97),
	(255, 62, 190),
]

DEFAULT_FALLBACK_COLOR = (200, 200, 200)

# Global counter used to pick the next color / hotkey
NEW_ROW_COUNTER = 0

# ----------------------- Helpers for parsing/serialising -----------------------

def parse_list_field(value):
	"""Split comma-separated list, treat '0' or empty as empty list."""
	if value is None:
		return []
	s = value.strip()
	if s == '' or s == '0':
		return []
	return [x.strip() for x in s.split(',') if x.strip()]


def parse_colors_field(value):
	"""Colors may be a single triple 'r,g,b' or multiple separated by ';'
	Return list of (r,g,b) tuples of ints.
	Treat '0' or empty as empty list.
	"""
	if value is None:
		return []
	s = value.strip()
	if s == '' or s == '0':
		return []
	cols = []
	parts = s.split(';')
	for p in parts:
		p = p.strip()
		if not p or p == '0':
			continue
		comps = [c.strip() for c in p.split(',') if c.strip()]
		if len(comps) != 3:
			# ignore malformed
			continue
		try:
			cols.append(tuple(int(c) for c in comps))
		except ValueError:
			continue
	return cols


def colors_to_field(colors):
	"""Serialize list of (r,g,b) tuples to 'r,g,b;r,g,b' or '0' if empty"""
	if not colors:
		return '0'
	return ';'.join(','.join(str(int(v)) for v in triple) for triple in colors)


def list_to_field(lst):
	if not lst:
		return '0'
	return ','.join(lst)


# ----------------------- Class row widget -----------------------

class ClassRow(ttk.Frame):
	def __init__(self, master, label='', hotkey='', color=(200,200,200),
				 on_change=None, remove_callback=None,
				 show_ignore_secondary=False, initial_ignore=False, *args, **kwargs):
		super().__init__(master, *args, **kwargs)
		self.on_change = on_change
		self.remove_callback = remove_callback
		self.show_ignore_secondary = bool(show_ignore_secondary)

		self.label_var = tk.StringVar(value=label)
		self.hotkey_var = tk.StringVar(value=hotkey)
		# Store colour in vars so pick_color can update them
		self.r_var = tk.IntVar(value=color[0])
		self.g_var = tk.IntVar(value=color[1])
		self.b_var = tk.IntVar(value=color[2])
		self.ignore_secondary_var = tk.BooleanVar(value=bool(initial_ignore))

		# widgets
		self.label_entry = ttk.Entry(self, textvariable=self.label_var, width=20)
		self.hotkey_entry = ttk.Entry(self, textvariable=self.hotkey_var, width=4)

		# Colour chooser button (single control; RGB spinboxes removed)
		self.color_btn = tk.Button(self, text='Choose', command=self.pick_color, width=8)
		self._update_btn_color()

		self.ignore_secondary_cb = ttk.Checkbutton(self,
												   text='ignore secondary',
												   variable=self.ignore_secondary_var,
												   command=self._changed)

		self.remove_btn = ttk.Button(self, text='Remove', command=self._on_remove)

		# layout columns
		col = 0
		self.label_entry.grid(row=0, column=col, sticky='w', padx=(0,6)); col += 1
		self.hotkey_entry.grid(row=0, column=col, sticky='w', padx=(0,6)); col += 1
		self.color_btn.grid(row=0, column=col, sticky='w', padx=(0,6)); col += 1

		if self.show_ignore_secondary:
			self.ignore_secondary_cb.grid(row=0, column=col, padx=(6, 0))
			col += 1

		self.remove_btn.grid(row=0, column=col, sticky='w')

		# traces
		self.label_var.trace_add('write', self._changed)
		self.hotkey_var.trace_add('write', self._changed)
		# Note: we do not trace r/g/b vars because they are controlled by the chooser.

	def _update_btn_color(self):
		r, g, b = self.r_var.get(), self.g_var.get(), self.b_var.get()
		hexcol = f'#{r:02x}{g:02x}{b:02x}'
		# Use both bg and activebackground to make it visible on some platforms
		try:
			self.color_btn.configure(bg=hexcol, activebackground=hexcol)
		except Exception:
			# On some platforms ttk/button rendering differs; ignore failures gracefully
			try:
				self.color_btn.configure(background=hexcol)
			except Exception:
				pass

	def pick_color(self):
		# start color chooser with current colour
		try:
			initial = f'#{self.r_var.get():02x}{self.g_var.get():02x}{self.b_var.get():02x}'
		except Exception:
			initial = None
		rgb, hexcol = colorchooser.askcolor(color=initial)
		if rgb:
			r, g, b = [int(round(x)) for x in rgb]
			self.r_var.set(r)
			self.g_var.set(g)
			self.b_var.set(b)
			self._update_btn_color()
			self._changed()

	def _on_remove(self):
		# call removal callback provided by the parent editor so that the editor
		# can both remove the row from its list and destroy the widget.
		if callable(self.remove_callback):
			try:
				self.remove_callback(self)
			except Exception:
				try:
					self.destroy()
				except Exception:
					pass
		else:
			try:
				self.destroy()
			except Exception:
				pass

		if self.on_change:
			self.on_change()

	def _changed(self, *args):
		if self.on_change:
			self.on_change()

	def get(self):
		return (
			self.label_var.get().strip(),
			self.hotkey_var.get().strip(),
			(self.r_var.get(), self.g_var.get(), self.b_var.get()),
			bool(self.ignore_secondary_var.get())
		)


# ----------------------- Class list editor -----------------------

class ClassListEditor(ttk.Frame):
	"""
	Simple ClassListEditor that uses a global counter to choose the next default
	color and hotkey for newly-added rows.

	Behavior:
	  - If add_row() is called with color=None and/or hotkey=None, the editor
		will pick defaults based on NEW_ROW_COUNTER and then increment it.
	  - If caller provides explicit color or hotkey, those are used and the
		counter is NOT advanced (keeps behaviour simple).
	  - Supports suppression of confirm dialogs via set_suppress_confirm(True).
	  - confirm_modify callback is invoked before structural changes (add/clear/remove)
		unless suppressed.
	"""
	def __init__(self, master, title, on_change=None, initial=None, confirm_modify=None, *args, **kwargs):
		super().__init__(master, *args, **kwargs)
		self.on_change = on_change
		self.rows = []
		self.confirm_modify = confirm_modify
		self.suppress_confirm = False

		# Draw a bold title inside the editor (prevents duplicate titles)
		try:
			import tkinter.font as tkfont
			font_bold = tkfont.nametofont("TkDefaultFont").copy()
			font_bold.configure(weight="bold", size=11)
			ttk.Label(self, text=title, font=font_bold).grid(row=0, column=0, sticky='w')
		except Exception:
			ttk.Label(self, text=title).grid(row=0, column=0, sticky='w')

		btn_frame = ttk.Frame(self)
		btn_frame.grid(row=0, column=1, sticky='e')
		ttk.Button(btn_frame, text='Add', command=self.add_row).grid(row=0, column=0)
		ttk.Button(btn_frame, text='Clear', command=self.clear).grid(row=0, column=1, padx=(6,0))

		self.allow_ignore_secondary = title.lower().startswith('primary')
		self.rows_frame = ttk.Frame(self)
		self.rows_frame.grid(row=1, column=0, columnspan=2, sticky='we', pady=(6,0))

		if initial:
			for label, hotkey, color in initial:
				# Use provided values when loading initial content
				self._create_row(label, hotkey, color)

	def set_suppress_confirm(self, val: bool):
		self.suppress_confirm = bool(val)

	def _confirm_allowed(self):
		"""Helper to call confirm_modify when needed (respect suppress flag)."""
		if self.suppress_confirm:
			return True
		if callable(self.confirm_modify):
			try:
				return bool(self.confirm_modify())
			except Exception:
				return False
		return True

	def _pick_defaults_and_advance_counter(self):
		"""Pick color and hotkey from global counter and increment it."""
		global NEW_ROW_COUNTER
		idx = NEW_ROW_COUNTER
		# pick color from palette, fallback when palette exhausted
		if idx < len(DEFAULT_CLASS_COLORS):
			color = DEFAULT_CLASS_COLORS[idx]
		else:
			color = DEFAULT_FALLBACK_COLOR
		# pick hotkey: numeric 1..9 first, then a..z (single char)
		if idx < 9:
			hotkey = str(idx + 1)
		else:
			# letters after digits (wrap if > 9+26)
			letter_idx = (idx - 9) % 26
			hotkey = chr(ord('a') + letter_idx)
		NEW_ROW_COUNTER += 1
		return hotkey, color

	def add_row(self, label='', hotkey=None, color=None, ignore_secondary=False):
		"""
		Add a new ClassRow. If hotkey/color are None, choose defaults using
		the global counter (and advance it). If explicit values are provided,
		do not touch the counter.
		"""
		# Ask for confirmation if needed
		if not self._confirm_allowed():
			return

		# Auto-assign defaults if caller didn't provide them
		assigned_hotkey = hotkey
		assigned_color = color
		if hotkey is None or hotkey == '':
			assigned_hotkey, assigned_color_from_counter = self._pick_defaults_and_advance_counter()
			# If color was explicitly provided as None as well, use the color from the counter;
			# otherwise if caller provided color but not hotkey, use provided color and the counter's hotkey.
			if color is None:
				assigned_color = assigned_color_from_counter
		else:
			# caller provided a hotkey; only auto-assign color if color is None
			if color is None:
				# pick color using counter but don't increment the counter for simplicity:
				# reuse the current counter index but do not advance (keeps deterministic if user sets hotkeys manually)
				# We'll still use the counter value to pick a color so rows remain varied.
				global NEW_ROW_COUNTER
				idx = NEW_ROW_COUNTER
				assigned_color = DEFAULT_CLASS_COLORS[idx] if idx < len(DEFAULT_CLASS_COLORS) else DEFAULT_FALLBACK_COLOR

		# If both provided explicitly, we do not change NEW_ROW_COUNTER (simple behaviour)

		# Create the UI row
		self._create_row(label, assigned_hotkey or '', assigned_color or DEFAULT_FALLBACK_COLOR, ignore_secondary)
		if self.on_change:
			self.on_change()

	def _create_row(self, label, hotkey, color, ignore_secondary=False):
		# Remove callback for the row; respect confirm_modify unless suppressed
		def _remove_and_mark(row):
			if not self._confirm_allowed():
				return
			if row in self.rows:
				try:
					self.rows.remove(row)
				except ValueError:
					pass
			try:
				row.destroy()
			except Exception:
				pass
			if self.on_change:
				self.on_change()

		row = ClassRow(
			self.rows_frame,
			label=label,
			hotkey=hotkey,
			color=color if color is not None else DEFAULT_FALLBACK_COLOR,
			on_change=self.on_change,
			remove_callback=_remove_and_mark,
			show_ignore_secondary=self.allow_ignore_secondary,
			initial_ignore=ignore_secondary
		)
		row.pack(fill='x', pady=2, anchor='w')
		self.rows.append(row)

	def clear(self):
		if not self._confirm_allowed():
			return
		for r in list(self.rows):
			try:
				r.destroy()
			except Exception:
				pass
		self.rows = []
		if self.on_change:
			self.on_change()

	def get(self):
		"""Return list of (label, hotkey, (r,g,b), ignore_flag) skipping empty labels."""
		out = []
		for r in self.rows:
			try:
				label, hotkey, color, ignore = r.get()
			except Exception:
				continue
			if not label:
				continue
			out.append((label, hotkey, color, ignore))
		return out


# ----------------------- Complex-behaviour label editor -----------------------

class ComplexLabelEditor(ttk.Frame):
	"""Compact editor for the single user-editable complex-behaviour list.

	Each row is a behaviour name + a single-character hotkey (no colour/secondary,
	unlike ClassListEditor). get() returns a list of (name, hotkey) tuples.
	"""

	def __init__(self, master, on_change=None, *args, **kwargs):
		super().__init__(master, *args, **kwargs)
		self.on_change = on_change
		self.rows = []

		header = ttk.Frame(self)
		header.pack(fill='x')
		ttk.Label(header, text='Complex behaviours (dyadic & group)',
				  font=('TkDefaultFont', 9, 'bold')).pack(side='left')
		ttk.Button(header, text='Add', command=lambda: self.add_row()).pack(side='right')
		ttk.Button(header, text='Clear', command=self.clear).pack(side='right', padx=(0, 6))

		cols = ttk.Frame(self); cols.pack(fill='x', pady=(2, 0))
		ttk.Label(cols, text='name', width=24).pack(side='left')
		ttk.Label(cols, text='hotkey', width=8).pack(side='left')

		self.rows_frame = ttk.Frame(self)
		self.rows_frame.pack(fill='x')

	def add_row(self, name='', hotkey=''):
		row = ttk.Frame(self.rows_frame)
		name_var = tk.StringVar(value=name)
		hk_var = tk.StringVar(value=hotkey)
		name_e = ttk.Entry(row, textvariable=name_var, width=24)
		hk_e = ttk.Entry(row, textvariable=hk_var, width=8)
		name_e.pack(side='left', padx=(0, 4))
		hk_e.pack(side='left', padx=(0, 4))

		def _remove():
			if entry in self.rows:
				self.rows.remove(entry)
			row.destroy()
			if self.on_change:
				self.on_change()

		ttk.Button(row, text='Remove', command=_remove).pack(side='left')
		row.pack(fill='x', pady=1, anchor='w')
		entry = (name_var, hk_var)
		self.rows.append(entry)
		if self.on_change:
			self.on_change()

	def clear(self):
		for child in list(self.rows_frame.winfo_children()):
			child.destroy()
		self.rows = []
		if self.on_change:
			self.on_change()

	def get(self):
		"""Return [(name, hotkey), ...] skipping rows with an empty name."""
		out = []
		for name_var, hk_var in self.rows:
			name = name_var.get().strip()
			hk = hk_var.get().strip()
			if name:
				out.append((name, hk))
		return out


# ----------------------- Main app -----------------------

class SettingsEditorApp(tk.Tk):

	def __init__(self, ini_path=INI_DEFAULT_PATH):
		super().__init__()
		self.title('BehaveAI Settings Editor')
		self.geometry('700x600')
		self.ini_path = ini_path
		self.dirty = False

		self.project_dir = os.path.dirname(self.ini_path)

		self.clips_dir_var = tk.StringVar()
		self.input_dir_var = tk.StringVar()
		self.output_dir_var = tk.StringVar()

		# Data augmentation parameters
		self.aug_target_classes_var = tk.StringVar(value='')
		self.aug_global_prob_var = tk.DoubleVar(value=0)
		self.aug_brightness_range_var = tk.StringVar(value='0.8,1.2')
		self.aug_brightness_prob_var = tk.DoubleVar(value=0)
		self.aug_contrast_range_var = tk.StringVar(value='0.8,1.2')
		self.aug_contrast_prob_var = tk.DoubleVar(value=0)
		self.aug_saturation_range_var = tk.StringVar(value='0.8,1.2')
		self.aug_saturation_prob_var = tk.DoubleVar(value=0)
		self.aug_hue_range_var = tk.StringVar(value='-15,15')
		self.aug_hue_prob_var = tk.DoubleVar(value=0)
		self.aug_sharpness_range_var = tk.StringVar(value='0.8,1.5')
		self.aug_sharpness_prob_var = tk.DoubleVar(value=0)
		self.aug_blur_range_var = tk.StringVar(value='1,3')
		self.aug_blur_prob_var = tk.DoubleVar(value=0)
		self.aug_noise_range_var = tk.StringVar(value='0,25')
		self.aug_noise_prob_var = tk.DoubleVar(value=0)
		self.aug_shear_range_var = tk.StringVar(value='-0.1,0.1')
		self.aug_shear_prob_var = tk.DoubleVar(value=0)
		self.aug_flip_h_options_var = tk.StringVar(value='True,False')
		self.aug_flip_h_prob_var = tk.DoubleVar(value=0)
		self.aug_flip_v_options_var = tk.StringVar(value='True,False')
		self.aug_flip_v_prob_var = tk.DoubleVar(value=0.)
		self.aug_temperature_range_var = tk.StringVar(value='0,10')
		self.aug_temperature_prob_var = tk.DoubleVar(value=0)

		# Activity budget parameters
		self.ab_min_presence_ratio_var    = tk.DoubleVar(value=0.10)
		self.ab_border_zone_ratio_var     = tk.DoubleVar(value=0.15)
		self.ab_group_type_separator_var  = tk.StringVar(value='_')
		self.ab_group_type_field_index_var = tk.IntVar(value=4)

		# Drone motion correction parameters
		self.drone_enabled_var           = tk.BooleanVar(value=False)
		self.drone_model_var             = tk.StringVar(value='affine')
		self.drone_box_dilation_var      = tk.DoubleVar(value=0.20)
		self.drone_min_features_var      = tk.IntVar(value=30)
		self.drone_uncertain_std_var     = tk.DoubleVar(value=8.0)
		self.drone_smoothing_var         = tk.StringVar(value='savgol')
		self.drone_smoothing_window_var  = tk.IntVar(value=7)
		self.drone_fallback_smoothing_var = tk.BooleanVar(value=True)

		# Intra-video Re-Identification parameters
		self.reid_enabled_var          = tk.BooleanVar(value=True)
		self.reid_method_var           = tk.StringVar(value='histogram')
		self.reid_similarity_var       = tk.DoubleVar(value=0.75)
		self.reid_histogram_min_var    = tk.DoubleVar(value=0.60)
		self.reid_max_disappeared_var  = tk.DoubleVar(value=180.0)
		self.reid_max_position_var     = tk.DoubleVar(value=500.0)
		self.ab_min_classified_var     = tk.IntVar(value=5)

		# Sub-grouping (fission-fusion) parameters
		self.subgroup_eps_bodylen_var  = tk.DoubleVar(value=4.0)
		self.subgroup_min_stable_var   = tk.IntVar(value=10)
		self.foal_size_ratio_var       = tk.DoubleVar(value=0.7)
		self.body_len_ref_scope_var    = tk.StringVar(value='video')

		# Interaction features / graph parameters
		self.complex_max_dist_var       = tk.DoubleVar(value=400.0)
		self.complex_min_duration_var   = tk.IntVar(value=10)
		self.complex_contact_iou_var    = tk.DoubleVar(value=0.05)
		self.complex_contact_dist_var   = tk.DoubleVar(value=1.5)
		self.complex_window_var         = tk.IntVar(value=30)
		self.interaction_granularity_var = tk.StringVar(value='per_interaction')
		self.interaction_weight_var     = tk.StringVar(value='duration')

		# Complex-behaviour model selectors
		self.complex_model_type_var        = tk.StringVar(value='baseline')
		self.complex_baseline_clf_var      = tk.StringVar(value='random_forest')

		# Complex-behaviour model thresholds + candidate heuristics
		self.complex_confusion_merge_rate_var = tk.DoubleVar(value=0.20)
		self.complex_predict_min_proba_var    = tk.DoubleVar(value=0.5)
		self.complex_speed_low_var            = tk.DoubleVar(value=0.05)
		self.complex_speed_high_var           = tk.DoubleVar(value=0.25)
		self.complex_polarisation_high_var    = tk.DoubleVar(value=0.7)
		self.complex_synchrony_high_var       = tk.DoubleVar(value=0.7)
		self.complex_candidate_topk_var       = tk.IntVar(value=50)

		self.cfg = configparser.ConfigParser()
		self.cfg.optionxform = str  # preserve case

		# store loaded motion settings for later comparison
		self._loaded_motion_settings = {}

		self._build_ui()
		self.load_ini(self.ini_path)

	def _validate_paths(self):
		missing = []
		for name, var in [
			('Clips directory', self.clips_dir_var),
			('Input directory', self.input_dir_var),
			('Output directory', self.output_dir_var),
		]:
			if not var.get():
				missing.append(name)

		if missing:
			return "The following paths are missing:\n\n" + "\n".join(f"• {m}" for m in missing)
		return None

	def _validate_hotkeys(self):
		used = {}
		errors = []

		for key, editor in self.class_editors.items():
			for label, hotkey, _, _ in editor.get():

				if not hotkey:
					errors.append(
						f"Class '{label}' does not have a hotkey assigned."
					)
					continue

				if len(hotkey) != 1:
					errors.append(
						f"Hotkey '{hotkey}' for class '{label}' must be a single character."
					)
					continue

				hk = hotkey.lower()

				if hk in RESERVED_HOTKEYS:
					errors.append(
						f"Hotkey '{hotkey}' for class '{label}' is reserved "
						f"(undo / grey-out)."
					)
					continue

				if hk in used:
					errors.append(
						f"Hotkey '{hotkey}' is used by both "
						f"'{used[hk]}' and '{label}'."
					)
				else:
					used[hk] = label

		return errors


	def _validate_primary_classes(self):
		pm = self.class_editors['primary_motion'].get()
		ps = self.class_editors['primary_static'].get()

		if not pm and not ps:
			return (
				"You must define at least one PRIMARY class:\n\n"
				"• Primary motion OR\n"
				"• Primary static"
			)
		return None

	def _validate_secondary_classes(self):
		"""
		Ensure there are at least 2 secondary static and 2 secondary motion classes
		when they are being used (have at least one class defined).
		Returns (is_valid: bool, error_message: str)
		"""
		# Get secondary class counts and filter out empty labels
		sec_static = [x for x in self.class_editors['secondary_static'].get() if x[0]]
		sec_motion = [x for x in self.class_editors['secondary_motion'].get() if x[0]]

		# Only validate if there are any secondary classes defined
		needs_static_validation = len(sec_static) > 0
		needs_motion_validation = len(sec_motion) > 0

		errors = []

		if needs_static_validation and len(sec_static) < 2:
			errors.append(
				"At least 2 secondary static classes are required when using secondary static classes."
			)

		if needs_motion_validation and len(sec_motion) < 2:
			errors.append(
				"At least 2 secondary motion classes are required when using secondary motion classes."
			)

		if errors:
			return False, "\n\n".join(errors)

		return True, ""

	def _validate_complex_hotkeys(self):
		"""Validate complex-behaviour hotkeys: a provided hotkey must be a single
		character, unique among complex behaviours, and not reserved. An empty
		hotkey is allowed (auto-assigned by the annotation tool). Returns a list
		of error strings."""
		errors = []
		used = {}
		for name, hk in self.complex_editor.get():
			if not hk:
				continue  # empty -> auto-assigned later
			if len(hk) != 1:
				errors.append(f"Hotkey '{hk}' for complex behaviour '{name}' must be a single character.")
				continue
			lk = hk.lower()
			if lk in RESERVED_HOTKEYS:
				errors.append(f"Hotkey '{hk}' for complex behaviour '{name}' is reserved.")
				continue
			if lk in used:
				errors.append(f"Hotkey '{hk}' is used by both '{used[lk]}' and '{name}'.")
			else:
				used[lk] = name
		return errors

	def _confirm_modify_structure(self):
		"""
		Return True to allow structural changes (add/remove/clear), False to block.
		Show warning if annot_motion or annot_static exist in the project dir.
		"""
		annot_motion = os.path.join(self.project_dir, 'annot_motion')
		annot_static = os.path.join(self.project_dir, 'annot_static')
		if os.path.isdir(annot_motion) or os.path.isdir(annot_static):
			msg = (
				"Detected existing annotation directories in project:\n\n"
				f"  {annot_motion if os.path.isdir(annot_motion) else ''}\n"
				f"  {annot_static if os.path.isdir(annot_static) else ''}\n\n"
				"Modifying the model structure (adding/removing/clearing classes) may "
				"make existing annotations or trained models incompatible. Are you sure "
				"you want to proceed?"
			)
			return messagebox.askyesno("Warning: existing annotations detected", msg)
		return True

	def _build_ui(self):
		# top toolbar: load file
		toolbar = ttk.Frame(self)
		toolbar.pack(side='top', fill='x', padx=8, pady=6)

		notebook = ttk.Notebook(self)
		notebook.pack(fill='both', expand=True, padx=8, pady=6)

		# TAB 1: Model structure
		tab1 = ttk.Frame(notebook)
		notebook.add(tab1, text='Model structure')

		self.class_editors = {}
		for key, title in CLASS_GROUPS:
			# Create the ClassListEditor which draws its own (bold) title internally.
			editor = ClassListEditor(tab1, title=title, on_change=self._set_dirty, confirm_modify=self._confirm_modify_structure)
			editor.pack(fill='x', pady=(6,6), anchor='w')
			self.class_editors[key] = editor

		self.motion_blocks_static_var = tk.BooleanVar(value=False)
		ttk.Checkbutton(tab1, text='Motion blocks static', variable=self.motion_blocks_static_var, command=self._set_dirty).pack(anchor='w', pady=(8,0))
		self.static_blocks_motion_var = tk.BooleanVar(value=False)
		ttk.Checkbutton(tab1, text='Static blocks motion', variable=self.static_blocks_motion_var, command=self._set_dirty).pack(anchor='w')


		# TAB 1.2: Project paths
		tab_paths = ttk.Frame(notebook)
		notebook.add(tab_paths, text='Video paths')

		def _browse_dir(var):
			path = filedialog.askdirectory(
				initialdir=var.get() or self.project_dir,
				title='Select directory'
			)
			if path:
				var.set(path)
				self._set_dirty()

		def _path_row(parent, label, var, row):
			ttk.Label(parent, text=label).grid(row=row, column=0, sticky='w', padx=8, pady=6)
			ttk.Entry(parent, textvariable=var, width=60).grid(row=row, column=1, sticky='we', padx=8)
			ttk.Button(parent, text='Select…', command=lambda: _browse_dir(var)).grid(row=row, column=2, padx=8)

		tab_paths.columnconfigure(1, weight=1)

		_path_row(tab_paths, 'Training video clips directory',  self.clips_dir_var, 0)
		_path_row(tab_paths, 'Batch video input directory',  self.input_dir_var, 1)
		_path_row(tab_paths, 'Batch video output directory', self.output_dir_var, 2)

		# TAB 2: Video sampling parameters
		tab6 = ttk.Frame(notebook)
		notebook.add(tab6, text='Data augmentation')

		# Scrollable container so all rows fit without truncation
		aug_canvas = tk.Canvas(tab6, highlightthickness=0)
		aug_scroll = ttk.Scrollbar(tab6, orient='vertical', command=aug_canvas.yview)
		aug_inner  = ttk.Frame(aug_canvas)
		aug_inner.bind('<Configure>',
			lambda e: aug_canvas.configure(scrollregion=aug_canvas.bbox('all')))
		aug_canvas.create_window((0, 0), window=aug_inner, anchor='nw')
		aug_canvas.configure(yscrollcommand=aug_scroll.set)
		aug_canvas.pack(side='left', fill='both', expand=True)
		aug_scroll.pack(side='right', fill='y')

		r = 0  # running row counter

		# --- Target classes filter (NEW) ---
		ttk.Label(aug_inner, text='Target classes (leave empty = all)',
				  font=('TkDefaultFont', 9, 'bold')).grid(
			row=r, column=0, columnspan=4, sticky='w', padx=8, pady=(10, 0))
		r += 1
		ttk.Label(aug_inner, text='Classes to augment').grid(
			row=r, column=0, sticky='w', padx=8, pady=4)
		ttk.Entry(aug_inner, textvariable=self.aug_target_classes_var, width=40).grid(
			row=r, column=1, columnspan=3, sticky='w', padx=8)
		ttk.Label(aug_inner,
				  text='Comma-separated class names, e.g.  bird, tiger, filght, run   (empty = augment all classes)',
				  foreground='grey').grid(
			row=r+1, column=0, columnspan=4, sticky='w', padx=8, pady=(0, 6))
		r += 2

		# Separator
		ttk.Separator(aug_inner, orient='horizontal').grid(
			row=r, column=0, columnspan=4, sticky='ew', padx=8, pady=6)
		r += 1

		# --- Global probability ---
		ttk.Label(aug_inner, text='Global augmentation probability').grid(
			row=r, column=0, sticky='w', padx=8, pady=6)
		ttk.Spinbox(aug_inner, from_=0.0, to=1.0, increment=0.05,
					textvariable=self.aug_global_prob_var, width=8,
					command=self._set_dirty).grid(row=r, column=1, sticky='w', padx=8)
		r += 1

		# Helper: one parameter row  (label | range entry | prob label | prob spinbox)
		# Range entry is wider (width=35) to accommodate multi-segment syntax like
		#   0.5,0.8 | 1.0 | 1.2,1.6
		def _aug_row(parent, row, label, range_var, prob_var):
			ttk.Label(parent, text=f'{label} (range)').grid(
				row=row, column=0, sticky='w', padx=8, pady=4)
			ttk.Entry(parent, textvariable=range_var, width=35).grid(
				row=row, column=1, sticky='w', padx=8)
			ttk.Label(parent, text=f'{label} (probability)').grid(
				row=row, column=2, sticky='w', padx=8, pady=4)
			ttk.Spinbox(parent, from_=0.0, to=1.0, increment=0.05,
						textvariable=prob_var, width=8,
						command=self._set_dirty).grid(row=row, column=3, sticky='w', padx=8)

		# --- Multi-segment range syntax hint ---
		ttk.Label(aug_inner,
				  text='Range syntax:  single range: 0.8,1.2  |  '
				       'multi-segment: 0.5,0.8 | 1.2,1.6  |  '
				       'discrete value: 0.6  |  mix: 0.5,0.8 | 1.0 | 1.2,1.6',
				  foreground='grey').grid(
			row=r, column=0, columnspan=4, sticky='w', padx=8, pady=(0, 6))
		ttk.Label(aug_inner,
				  text='Each segment separated by | produces one independent augmented copy.',
				  foreground='grey').grid(
			row=r+1, column=0, columnspan=4, sticky='w', padx=8, pady=(0, 8))
		r += 2

		_aug_row(aug_inner, r, 'Brightness',   self.aug_brightness_range_var,   self.aug_brightness_prob_var);   r += 1
		_aug_row(aug_inner, r, 'Contrast',     self.aug_contrast_range_var,     self.aug_contrast_prob_var);     r += 1
		_aug_row(aug_inner, r, 'Saturation',   self.aug_saturation_range_var,   self.aug_saturation_prob_var);   r += 1
		_aug_row(aug_inner, r, 'Hue',          self.aug_hue_range_var,          self.aug_hue_prob_var);          r += 1
		_aug_row(aug_inner, r, 'Sharpness',    self.aug_sharpness_range_var,    self.aug_sharpness_prob_var);    r += 1
		_aug_row(aug_inner, r, 'Blur',         self.aug_blur_range_var,         self.aug_blur_prob_var);         r += 1
		_aug_row(aug_inner, r, 'Noise',        self.aug_noise_range_var,        self.aug_noise_prob_var);        r += 1
		_aug_row(aug_inner, r, 'Shear',        self.aug_shear_range_var,        self.aug_shear_prob_var);        r += 1
		_aug_row(aug_inner, r, 'Temperature',  self.aug_temperature_range_var,  self.aug_temperature_prob_var);  r += 1

		# Flip H — options field (not a range, kept as-is)
		ttk.Label(aug_inner, text='Horizontal Flip (options)').grid(
			row=r, column=0, sticky='w', padx=8, pady=4)
		ttk.Entry(aug_inner, textvariable=self.aug_flip_h_options_var, width=14).grid(
			row=r, column=1, sticky='w', padx=8)
		ttk.Label(aug_inner, text='Horizontal Flip (probability)').grid(
			row=r, column=2, sticky='w', padx=8, pady=4)
		ttk.Spinbox(aug_inner, from_=0.0, to=1.0, increment=0.05,
					textvariable=self.aug_flip_h_prob_var, width=8,
					command=self._set_dirty).grid(row=r, column=3, sticky='w', padx=8)
		r += 1

		# Flip V
		ttk.Label(aug_inner, text='Vertical Flip (options)').grid(
			row=r, column=0, sticky='w', padx=8, pady=4)
		ttk.Entry(aug_inner, textvariable=self.aug_flip_v_options_var, width=14).grid(
			row=r, column=1, sticky='w', padx=8)
		ttk.Label(aug_inner, text='Vertical Flip (probability)').grid(
			row=r, column=2, sticky='w', padx=8, pady=4)
		ttk.Spinbox(aug_inner, from_=0.0, to=1.0, increment=0.05,
					textvariable=self.aug_flip_v_prob_var, width=8,
					command=self._set_dirty).grid(row=r, column=3, sticky='w', padx=8)
		r += 1

		# Delete augmented data button
		ttk.Button(
			aug_inner,
			text='Delete all augmented data',
			command=self._delete_augmented_data
		).grid(row=r, column=0, columnspan=2, sticky='w', padx=8, pady=(16, 4))

		# TAB 2: Motion-from-colour strategy
		tab2 = ttk.Frame(notebook)
		notebook.add(tab2, text='Motion strategy')

		ttk.Label(tab2, text='Strategy').grid(row=0, column=0, sticky='w', padx=8, pady=(8,0))
		self.strategy_var = tk.StringVar(value='exponential')
		ttk.Combobox(tab2, values=['sequential', 'exponential'], textvariable=self.strategy_var, state='readonly').grid(row=0, column=1, sticky='w', padx=8, pady=(8,0))
		self.strategy_var.trace_add('write', lambda *a: self._set_dirty())

		self.chromatic_tail_only_var = tk.BooleanVar(value=False)
		ttk.Checkbutton(tab2, text='Chromatic tail only', variable=self.chromatic_tail_only_var, command=self._set_dirty).grid(row=1, column=0, sticky='w', padx=8, pady=(6,0))

		ttk.Label(tab2, text='Green decay (expA)').grid(row=2, column=0, sticky='w', padx=8, pady=(6,0))
		self.expA_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab2, from_=0.0, to=0.99, increment=0.01, textvariable=self.expA_var, width=6, command=self._set_dirty).grid(row=2, column=1, sticky='w', padx=8)

		ttk.Label(tab2, text='Red decay (expB)').grid(row=3, column=0, sticky='w', padx=8, pady=(6,0))
		self.expB_var = tk.DoubleVar(value=0.8)
		ttk.Spinbox(tab2, from_=0.0, to=0.99, increment=0.01, textvariable=self.expB_var, width=6, command=self._set_dirty).grid(row=3, column=1, sticky='w', padx=8)

		ttk.Label(tab2, text='Lum weight').grid(row=4, column=0, sticky='w', padx=8, pady=(6,0))
		self.lum_weight_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab2, from_=0.0, to=1.0, increment=0.01, textvariable=self.lum_weight_var, width=6, command=self._set_dirty).grid(row=4, column=1, sticky='w', padx=8)

		ttk.Label(tab2, text='RGB multipliers (r,g,b)').grid(row=5, column=0, sticky='w', padx=8, pady=(6,0))
		self.rgb_mult_var = tk.StringVar(value='4,4,4')
		ttk.Entry(tab2, textvariable=self.rgb_mult_var).grid(row=5, column=1, sticky='w', padx=8)
		self.rgb_mult_var.trace_add('write', lambda *a: self._set_dirty())

		ttk.Label(tab2, text='Frame skip').grid(row=6, column=0, sticky='w', padx=8, pady=(6,0))
		self.frame_skip_var = tk.IntVar(value=0)
		ttk.Spinbox(tab2, from_=0, to=10000, textvariable=self.frame_skip_var, width=8, command=self._set_dirty).grid(row=6, column=1, sticky='w', padx=8)

		# TAB 4: Model type
		tab3 = ttk.Frame(notebook)
		notebook.add(tab3, text='Model type')

		ttk.Label(tab3, text='Validation frequency').grid(row=0, column=0, sticky='w', padx=8, pady=(8,0))
		self.val_frequency_var = tk.DoubleVar(value=0.2)
		ttk.Spinbox(tab3, from_=0.0, to=1.0, increment=0.01, textvariable=self.val_frequency_var, width=6, command=self._set_dirty).grid(row=0, column=1, sticky='w', padx=8)

		ttk.Label(tab3, text='Primary classifier').grid(row=1, column=0, sticky='w', padx=8, pady=(8,0))
		self.primary_classifier_var = tk.StringVar(value='yolo26n.pt')
		ttk.Combobox(tab3, values=CLASSIFIER_OPTIONS, textvariable=self.primary_classifier_var).grid(row=1, column=1, sticky='w', padx=8, pady=(8,0))
		self.primary_classifier_var.trace_add('write', lambda *a: self._set_dirty())

		ttk.Label(tab3, text='Primary epochs').grid(row=2, column=0, sticky='w', padx=8, pady=(6,0))
		self.primary_epochs_var = tk.IntVar(value=100)
		ttk.Spinbox(tab3, from_=1, to=10000, textvariable=self.primary_epochs_var, width=8, command=self._set_dirty).grid(row=2, column=1, sticky='w', padx=8)

		ttk.Label(tab3, text='Secondary classifier').grid(row=3, column=0, sticky='w', padx=8, pady=(6,0))
		self.secondary_classifier_var = tk.StringVar(value='yolo26n-cls.pt')
		secondary_opts = [m.replace('.pt','-cls.pt') for m in CLASSIFIER_OPTIONS if m.startswith('yolo')]
		ttk.Combobox(tab3, values=secondary_opts, textvariable=self.secondary_classifier_var).grid(row=3, column=1, sticky='w', padx=8)
		self.secondary_classifier_var.trace_add('write', lambda *a: self._set_dirty())

		ttk.Label(tab3, text='Secondary epochs').grid(row=4, column=0, sticky='w', padx=8, pady=(6,0))
		self.secondary_epochs_var = tk.IntVar(value=100)
		ttk.Spinbox(tab3, from_=1, to=10000, textvariable=self.secondary_epochs_var, width=8, command=self._set_dirty).grid(row=4, column=1, sticky='w', padx=8)

		self.use_ncnn_var = tk.BooleanVar(value=False)
		ttk.Checkbutton(tab3, text='use_ncnn', variable=self.use_ncnn_var, command=self._set_dirty).grid(row=5, column=0, sticky='w', padx=8, pady=(8,0))

		ttk.Label(tab3, text='Primary confidence thresh').grid(row=6, column=0, sticky='w', padx=8, pady=(6,0))
		self.primary_conf_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab3, from_=0.0, to=1.0, increment=0.01, textvariable=self.primary_conf_var, width=6, command=self._set_dirty).grid(row=6, column=1, sticky='w', padx=8)

		ttk.Label(tab3, text='Secondary confidence thresh').grid(row=7, column=0, sticky='w', padx=8, pady=(6,0))
		self.secondary_conf_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab3, from_=0.0, to=1.0, increment=0.01, textvariable=self.secondary_conf_var, width=6, command=self._set_dirty).grid(row=7, column=1, sticky='w', padx=8)

		ttk.Label(tab3, text='Dominant source').grid(row=8, column=0, sticky='w', padx=8, pady=(8,0))
		self.dominant_source_var = tk.StringVar(value='confidence')
		ttk.Combobox(tab3, values=['confidence', 'motion', 'static'], textvariable=self.dominant_source_var, state='readonly').grid(row=8, column=1, sticky='w', padx=8, pady=(8,0))
		self.dominant_source_var.trace_add('write', lambda *a: self._set_dirty())

		# Activity budget parameters
		self.ab_min_presence_ratio_var = tk.DoubleVar(value=0.10)
		self.ab_border_zone_ratio_var  = tk.DoubleVar(value=0.15)
		self.ab_group_type_separator_var = tk.StringVar(value='_')
		self.ab_group_type_field_index_var = tk.IntVar(value=4)

		# TAB 5: Tracking
		tab4 = ttk.Frame(notebook)
		notebook.add(tab4, text='Tracking')

		ttk.Label(tab4, text='Match distance thresh').grid(row=0, column=0, sticky='w', padx=8, pady=(8,0))
		self.match_distance_var = tk.IntVar(value=200)
		ttk.Spinbox(tab4, from_=1, to=10000, textvariable=self.match_distance_var, width=8, command=self._set_dirty).grid(row=0, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='Delete after missed').grid(row=1, column=0, sticky='w', padx=8, pady=(6,0))
		self.delete_after_var = tk.IntVar(value=10)
		ttk.Spinbox(tab4, from_=1, to=10000, textvariable=self.delete_after_var, width=8, command=self._set_dirty).grid(row=1, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='Centroid merge thresh').grid(row=2, column=0, sticky='w', padx=8, pady=(6,0))
		self.centroid_merge_var = tk.IntVar(value=50)
		ttk.Spinbox(tab4, from_=1, to=10000, textvariable=self.centroid_merge_var, width=8, command=self._set_dirty).grid(row=2, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='IOU thresh (overlap required to merge)').grid(row=3, column=0, sticky='w', padx=8, pady=(6,0))
		self.iou_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab4, from_=0.0, to=1.0, increment=0.01, textvariable=self.iou_var, width=6, command=self._set_dirty).grid(row=3, column=1, sticky='w', padx=8)

		# Kalman subsection
		ttk.Label(tab4, text='Kalman filter').grid(row=4, column=0, sticky='w', padx=8, pady=(12,0))
		ttk.Label(tab4, text='Process noise position').grid(row=5, column=0, sticky='w', padx=8)
		self.kalman_pos_var = tk.DoubleVar(value=0.01)
		ttk.Entry(tab4, textvariable=self.kalman_pos_var).grid(row=5, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='Process noise velocity').grid(row=6, column=0, sticky='w', padx=8)
		self.kalman_vel_var = tk.DoubleVar(value=0.01)
		ttk.Entry(tab4, textvariable=self.kalman_vel_var).grid(row=6, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='Measurement noise').grid(row=7, column=0, sticky='w', padx=8)
		self.kalman_meas_var = tk.DoubleVar(value=0.2)
		ttk.Entry(tab4, textvariable=self.kalman_meas_var).grid(row=7, column=1, sticky='w', padx=8)

		# Drone motion correction subsection (post-processing on the tracking CSV)
		ttk.Separator(tab4, orient='horizontal').grid(
			row=8, column=0, columnspan=2, sticky='ew', padx=8, pady=(12, 6))
		ttk.Label(tab4, text='Drone motion correction').grid(row=9, column=0, sticky='w', padx=8)
		ttk.Checkbutton(tab4, text='Enable drone correction',
			variable=self.drone_enabled_var, command=self._set_dirty).grid(
			row=10, column=0, columnspan=2, sticky='w', padx=8, pady=(4, 0))

		ttk.Label(tab4, text='Transform model').grid(row=11, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab4, values=['affine', 'homography'],
			textvariable=self.drone_model_var, state='readonly', width=12).grid(
			row=11, column=1, sticky='w', padx=8)
		self.drone_model_var.trace_add('write', lambda *a: self._set_dirty())

		ttk.Label(tab4, text='Box dilation (fraction)').grid(row=12, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=0.0, to=1.0, increment=0.05,
			textvariable=self.drone_box_dilation_var, width=6, command=self._set_dirty).grid(
			row=12, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='Min background features').grid(row=13, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=1, to=100000,
			textvariable=self.drone_min_features_var, width=8, command=self._set_dirty).grid(
			row=13, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='Uncertain residual std (px)').grid(row=14, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=0.0, to=1000.0, increment=0.5,
			textvariable=self.drone_uncertain_std_var, width=8, command=self._set_dirty).grid(
			row=14, column=1, sticky='w', padx=8)

		ttk.Label(tab4, text='Smoothing').grid(row=15, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab4, values=['savgol', 'moving_average', 'none'],
			textvariable=self.drone_smoothing_var, state='readonly', width=14).grid(
			row=15, column=1, sticky='w', padx=8)
		self.drone_smoothing_var.trace_add('write', lambda *a: self._set_dirty())

		ttk.Label(tab4, text='Smoothing window (odd)').grid(row=16, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=3, to=99, increment=2,
			textvariable=self.drone_smoothing_window_var, width=6, command=self._set_dirty).grid(
			row=16, column=1, sticky='w', padx=8)

		ttk.Checkbutton(tab4, text='Fallback to smoothing-only when features scarce',
			variable=self.drone_fallback_smoothing_var, command=self._set_dirty).grid(
			row=17, column=0, columnspan=2, sticky='w', padx=8, pady=(4, 0))

		# TAB: Re-Identification (intra-video) — placed after Tracking, before Display
		tab_reid = ttk.Frame(notebook)
		notebook.add(tab_reid, text='Re-Identification')

		_help_font = ('TkDefaultFont', 8, 'italic')
		rr = 0

		ttk.Checkbutton(tab_reid, text='Enable intra-video Re-ID',
			variable=self.reid_enabled_var, command=self._set_dirty).grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 0)); rr += 1
		ttk.Label(tab_reid, text='Give a horse the same id after it reappears within the same video.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		ttk.Label(tab_reid, text='Appearance method').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_reid, values=['histogram', 'embedding'],
			textvariable=self.reid_method_var, state='readonly', width=12).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='histogram = colour, no torch; embedding needs torch (falls back to histogram).',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1
		self.reid_method_var.trace_add('write', lambda *a: self._set_dirty())

		ttk.Label(tab_reid, text='Similarity threshold').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=0.0, to=1.0, increment=0.01,
			textvariable=self.reid_similarity_var, width=6, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Embedding appearance similarity gate (cosine); only a weak tie-breaker.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		ttk.Label(tab_reid, text='Histogram min similarity').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=0.0, to=1.0, increment=0.01,
			textvariable=self.reid_histogram_min_var, width=6, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Histogram method only: minimum colour-histogram similarity (0..1) to accept an '
			'appearance match. Below this, identity relies on position/time only. Ignored when method = embedding.',
			font=_help_font, foreground='grey', wraplength=420, justify='left').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		ttk.Label(tab_reid, text='Max disappeared (seconds)').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=1.0, to=100000.0, increment=10.0,
			textvariable=self.reid_max_disappeared_var, width=10, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Registry pruning guard only — NOT a hard match limit.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		ttk.Label(tab_reid, text='Max position distance (px)').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=1.0, to=100000.0, increment=10.0,
			textvariable=self.reid_max_position_var, width=10, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Spatial plausibility gate — the primary matching signal.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		ttk.Label(tab_reid, text='Min classified frames (group member)').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=0, to=100000,
			textvariable=self.ab_min_classified_var, width=8, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Activity budget: min frames with a known behaviour to be a group_member (0 = skip).',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		# TAB: Sub-grouping (fission-fusion) — observed spatial clusters per frame
		tab_sg = ttk.Frame(notebook)
		notebook.add(tab_sg, text='Sub-grouping')
		sr = 0

		ttk.Label(tab_sg, text='Sub-grouping (fission-fusion)',
			font=('TkDefaultFont', 10, 'bold')).grid(
			row=sr, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 0)); sr += 1
		ttk.Label(tab_sg, text='Partition co-present horses into spatial sub-groups, stable in time.',
			font=_help_font, foreground='grey').grid(
			row=sr, column=0, columnspan=2, sticky='w', padx=8, pady=(0, 6)); sr += 1

		ttk.Label(tab_sg, text='DBSCAN radius (body lengths)').grid(row=sr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_sg, from_=0.5, to=50.0, increment=0.5,
			textvariable=self.subgroup_eps_bodylen_var, width=6, command=self._set_dirty).grid(
			row=sr, column=1, sticky='w', padx=8); sr += 1
		ttk.Label(tab_sg, text='Clustering radius in reference body lengths (not pixels).',
			font=_help_font, foreground='grey').grid(
			row=sr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); sr += 1

		ttk.Label(tab_sg, text='Min stable frames').grid(row=sr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_sg, from_=1, to=100000,
			textvariable=self.subgroup_min_stable_var, width=8, command=self._set_dirty).grid(
			row=sr, column=1, sticky='w', padx=8); sr += 1
		ttk.Label(tab_sg, text='A sub-group change must persist this many frames (anti-flicker).',
			font=_help_font, foreground='grey').grid(
			row=sr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); sr += 1

		ttk.Label(tab_sg, text='Foal size ratio threshold').grid(row=sr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_sg, from_=0.0, to=1.0, increment=0.05,
			textvariable=self.foal_size_ratio_var, width=6, command=self._set_dirty).grid(
			row=sr, column=1, sticky='w', padx=8); sr += 1
		ttk.Label(tab_sg, text='body_len / reference below this flags a likely foal.',
			font=_help_font, foreground='grey').grid(
			row=sr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); sr += 1

		ttk.Label(tab_sg, text='Body-length reference scope').grid(row=sr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_sg, values=['video', 'segment'],
			textvariable=self.body_len_ref_scope_var, state='readonly', width=12).grid(
			row=sr, column=1, sticky='w', padx=8); sr += 1
		ttk.Label(tab_sg, text='video = one reference; segment = recompute on altitude/zoom drift.',
			font=_help_font, foreground='grey').grid(
			row=sr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); sr += 1
		self.body_len_ref_scope_var.trace_add('write', lambda *a: self._set_dirty())

		# TAB: Interaction features / graph (TASK 4 primary output)
		tab_int = ttk.Frame(notebook)
		notebook.add(tab_int, text='Interaction')
		ir = 0
		ttk.Label(tab_int, text='Interaction features & graph',
			font=('TkDefaultFont', 10, 'bold')).grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 0)); ir += 1
		ttk.Label(tab_int, text='Per-frame dyadic/group features aggregated into the interaction graph (edges/nodes CSVs).',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=8, pady=(0, 6)); ir += 1

		ttk.Label(tab_int, text='Max interaction distance (px)').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=1.0, to=100000.0, increment=10.0,
			textvariable=self.complex_max_dist_var, width=8, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		ttk.Label(tab_int, text='Pairs farther apart than this are not treated as interacting.',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); ir += 1

		ttk.Label(tab_int, text='Min interaction duration (frames)').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=1, to=100000,
			textvariable=self.complex_min_duration_var, width=8, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		ttk.Label(tab_int, text='Contact IoU threshold').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=0.0, to=1.0, increment=0.01,
			textvariable=self.complex_contact_iou_var, width=6, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		ttk.Label(tab_int, text='Contact distance (body lengths)').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=0.0, to=50.0, increment=0.1,
			textvariable=self.complex_contact_dist_var, width=6, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		ttk.Label(tab_int, text='Window length (frames)').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=1, to=100000,
			textvariable=self.complex_window_var, width=8, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		ttk.Label(tab_int, text='Edge granularity').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_int, values=['per_interaction', 'per_segment', 'per_frame'],
			textvariable=self.interaction_granularity_var, state='readonly', width=16).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		ttk.Label(tab_int, text='Changing this regenerates and overwrites the edges file.',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); ir += 1
		self.interaction_granularity_var.trace_add('write', lambda *a: self._set_dirty())

		ttk.Label(tab_int, text='Edge weight metric').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_int, values=['duration', 'proximity', 'combined'],
			textvariable=self.interaction_weight_var, state='readonly', width=16).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		self.interaction_weight_var.trace_add('write', lambda *a: self._set_dirty())

		# TAB: Complex Behaviours (label list + model selectors)
		tab_cb = ttk.Frame(notebook)
		notebook.add(tab_cb, text='Complex Behaviours')

		self.complex_editor = ComplexLabelEditor(tab_cb, on_change=self._set_dirty)
		self.complex_editor.pack(fill='x', padx=8, pady=(10, 4), anchor='w')
		ttk.Label(tab_cb,
			text='Single user-editable list of dyadic AND group behaviours; each needs a '
				 'unique single-character hotkey.',
			font=_help_font, foreground='grey').pack(anchor='w', padx=8, pady=(0, 8))

		mdl = ttk.Frame(tab_cb); mdl.pack(fill='x', padx=8, pady=(4, 0))
		ttk.Label(mdl, text='Model type').grid(row=0, column=0, sticky='w', pady=(4, 0))
		ttk.Combobox(mdl, values=['baseline', 'lstm', 'transformer'],
			textvariable=self.complex_model_type_var, state='readonly', width=16).grid(
			row=0, column=1, sticky='w', padx=8)
		ttk.Label(mdl, text='Baseline classifier').grid(row=1, column=0, sticky='w', pady=(6, 0))
		ttk.Combobox(mdl, values=['random_forest', 'hist_gradient_boosting'],
			textvariable=self.complex_baseline_clf_var, state='readonly', width=20).grid(
			row=1, column=1, sticky='w', padx=8)
		self.complex_model_type_var.trace_add('write', lambda *a: self._set_dirty())
		self.complex_baseline_clf_var.trace_add('write', lambda *a: self._set_dirty())
		ttk.Label(tab_cb,
			text="baseline = scikit-learn classifier; lstm/transformer need torch. "
				 "Interaction & sub-grouping thresholds are on the 'Interaction' and "
				 "'Sub-grouping' tabs.",
			font=_help_font, foreground='grey').pack(anchor='w', padx=8, pady=(8, 0))

		# Model + candidate-heuristic thresholds
		thr = ttk.LabelFrame(tab_cb, text='Model & candidate thresholds')
		thr.pack(fill='x', padx=8, pady=(10, 4))
		def _thr_row(rownum, label, var, lo, hi, inc, helptext):
			ttk.Label(thr, text=label).grid(row=rownum * 2, column=0, sticky='w', padx=6, pady=(4, 0))
			ttk.Spinbox(thr, from_=lo, to=hi, increment=inc, textvariable=var, width=8,
				command=self._set_dirty).grid(row=rownum * 2, column=1, sticky='w', padx=6)
			ttk.Label(thr, text=helptext, font=_help_font, foreground='grey').grid(
				row=rownum * 2 + 1, column=0, columnspan=2, sticky='w', padx=24)
		_thr_row(0, 'Confusion merge rate', self.complex_confusion_merge_rate_var, 0.0, 1.0, 0.05,
				 'Confusion >= this flags a class pair as a merge suggestion.')
		_thr_row(1, 'Predict min probability', self.complex_predict_min_proba_var, 0.0, 1.0, 0.05,
				 'Minimum probability to emit a complex-behaviour prediction.')
		_thr_row(2, 'Speed ~still (body len/frame)', self.complex_speed_low_var, 0.0, 5.0, 0.01,
				 'Speeds below this count as ~stationary in the candidate heuristics.')
		_thr_row(3, 'Speed fast (body len/frame)', self.complex_speed_high_var, 0.0, 10.0, 0.05,
				 'Speeds above this count as fast (gallop / chase).')
		_thr_row(4, 'Polarisation high', self.complex_polarisation_high_var, 0.0, 1.0, 0.05,
				 'Sub-group alignment above this suggests trek/stampede.')
		_thr_row(5, 'Synchrony high', self.complex_synchrony_high_var, 0.0, 1.0, 0.05,
				 'Behavioural synchrony above this suggests synchronised rest/graze.')
		_thr_row(6, 'Active-learning top-K', self.complex_candidate_topk_var, 1, 100000, 1,
				 'Number of most-uncertain windows surfaced as candidates.')

		# TAB 6: Display
		tab5 = ttk.Frame(notebook)
		notebook.add(tab5, text='Display Settings')

		# viewing options
		ttk.Label(tab5, text='Viewing options').pack(anchor='w')
		self.line_thickness_var = tk.IntVar(value=1)
		ttk.Label(tab5, text='Line thickness').pack(anchor='w', pady=(6,0))
		ttk.Spinbox(tab5, from_=1, to=10, textvariable=self.line_thickness_var, width=6, command=self._set_dirty).pack(anchor='w')

		self.font_size_var = tk.DoubleVar(value=0.6)
		ttk.Label(tab5, text='Font size').pack(anchor='w', pady=(6,0))
		ttk.Spinbox(tab5, from_=0.1, to=5.0, increment=0.1, textvariable=self.font_size_var, width=6, command=self._set_dirty).pack(anchor='w')

		# TAB 7: Activity Budget
		tab_ab = ttk.Frame(notebook)
		notebook.add(tab_ab, text='Activity Budget')

		ttk.Label(tab_ab, text='Min presence ratio (stranger threshold)').grid(
			row=0, column=0, sticky='w', padx=8, pady=6)
		ttk.Spinbox(tab_ab, from_=0.01, to=1.0, increment=0.01,
			textvariable=self.ab_min_presence_ratio_var,
			width=8, command=self._set_dirty).grid(row=0, column=1, sticky='w', padx=8)

		ttk.Label(tab_ab, text='Border zone ratio (stranger threshold)').grid(
			row=1, column=0, sticky='w', padx=8, pady=6)
		ttk.Spinbox(tab_ab, from_=0.01, to=0.5, increment=0.01,
			textvariable=self.ab_border_zone_ratio_var,
			width=8, command=self._set_dirty).grid(row=1, column=1, sticky='w', padx=8)

		ttk.Label(tab_ab, text='Filename field separator').grid(
			row=2, column=0, sticky='w', padx=8, pady=6)
		ttk.Entry(tab_ab, textvariable=self.ab_group_type_separator_var,
			width=4).grid(row=2, column=1, sticky='w', padx=8)

		ttk.Label(tab_ab, text='Group type field index (0-based)').grid(
			row=3, column=0, sticky='w', padx=8, pady=6)
		ttk.Spinbox(tab_ab, from_=0, to=10, increment=1,
			textvariable=self.ab_group_type_field_index_var,
			width=6, command=self._set_dirty).grid(row=3, column=1, sticky='w', padx=8)

		# bottom save/cancel
		bottom = ttk.Frame(self)
		bottom.pack(side='bottom', fill='x', padx=8, pady=8)

		# Save button enabled at all times (per your fallback request)
		self.save_btn = ttk.Button(bottom, text='Save', command=self.on_save, state='normal')
		self.save_btn.pack(side='right', padx=(6,0))
		ttk.Button(bottom, text='Cancel', command=self.on_cancel).pack(side='right')

	# ----------------------- File I/O -----------------------

	def load_ini(self, path):
		if not os.path.exists(path):
			if messagebox.askyesno('Create new', f'{path} not found. Create a new settings file at this path?'):
				open(path,'w').close()
			else:
				return
		try:
			self.cfg.read(path)
		except Exception as e:
			messagebox.showerror('Error', f'Failed to read ini: {e}')
			return
		self.ini_path = path
		# populate fields
		d = self.cfg['DEFAULT'] if 'DEFAULT' in self.cfg else self.cfg.defaults()

		# project paths
		self.clips_dir_var.set(
			d.get('clips_dir', fallback=os.path.join(self.project_dir, 'clips'))
		)
		self.input_dir_var.set(
			d.get('input_dir', fallback=os.path.join(self.project_dir, 'input'))
		)
		self.output_dir_var.set(
			d.get('output_dir', fallback=os.path.join(self.project_dir, 'output'))
		)


		# classes
		# read global ignore_secondary list from the file (may be empty)
		ignore_list = parse_list_field(d.get('ignore_secondary', fallback=''))

		# Suppress confirmation dialogs while populating editors at startup
		for ed in self.class_editors.values():
			ed.set_suppress_confirm(True)

		for key, _title in CLASS_GROUPS:
			classes_s = d.get(f'{key}_classes', fallback='0')
			colors_s = d.get(f'{key}_colors', fallback='0')
			hotkeys_s = d.get(f'{key}_hotkeys', fallback='0')
			cls = parse_list_field(classes_s)
			cols = parse_colors_field(colors_s)
			hks = parse_list_field(hotkeys_s)
			editor = self.class_editors[key]
			editor.clear()
			for i, label in enumerate(cls):
				hot = hks[i] if i < len(hks) else ''
				col = cols[i] if i < len(cols) else (200,200,200)
				is_ignored = False
				# if this is a primary editor, check whether this label is in ignore_list
				if key.startswith('primary') and label in ignore_list:
					is_ignored = True
				# Use add_row (confirm suppressed) so saved origin ordering remains unchanged
				editor.add_row(label=label, hotkey=hot, color=col, ignore_secondary=is_ignored)

		# Re-enable confirmation dialogs after load
		for ed in self.class_editors.values():
			ed.set_suppress_confirm(False)


		# viewing
		self.line_thickness_var.set(int(d.get('line_thickness', fallback='1')))
		self.font_size_var.set(float(d.get('font_size', fallback='0.6')))
		self.motion_blocks_static_var.set(self._str_to_bool(d.get('motion_blocks_static', fallback='true')))
		self.static_blocks_motion_var.set(self._str_to_bool(d.get('static_blocks_motion', fallback='false')))

		# motion tab
		self.strategy_var.set(d.get('strategy', fallback='exponential'))
		self.chromatic_tail_only_var.set(self._str_to_bool(d.get('chromatic_tail_only', fallback='false')))
		self.expA_var.set(float(d.get('expA', fallback='0.5')))
		self.expB_var.set(float(d.get('expB', fallback='0.7')))
		self.lum_weight_var.set(float(d.get('lum_weight', fallback='0.5')))
		self.rgb_mult_var.set(d.get('rgb_multipliers', fallback='4,4,4'))
		self.frame_skip_var.set(int(d.get('frame_skip', fallback='0')))
		# ~ self.scale_factor_var.set(float(d.get('scale_factor', fallback='1.0')))

		# Save a snapshot of motion-related settings for later change detection
		self._loaded_motion_settings = {
			'strategy': str(d.get('strategy', fallback='sequential')),
			'chromatic_tail_only': str(d.get('chromatic_tail_only', fallback='false')).lower(),
			'expA': str(d.get('expA', fallback='0.5')),
			'expB': str(d.get('expB', fallback='0.7')),
			'lum_weight': str(d.get('lum_weight', fallback='0.5')),
			'rgb_multipliers': str(d.get('rgb_multipliers', fallback='4,4,4')).replace(' ', ''),
			'frame_skip': str(d.get('frame_skip', fallback='0')),
			'motion_blocks_static': str(d.get('motion_blocks_static', fallback='false')).lower(),
			'static_blocks_motion': str(d.get('static_blocks_motion', fallback='false')).lower(),
		}

		# model type
		self.val_frequency_var.set(float(d.get('val_frequency', fallback='0.2')))
		self.primary_classifier_var.set(d.get('primary_classifier', fallback='yolo26n.pt'))
		self.primary_epochs_var.set(int(d.get('primary_epochs', fallback='100')))
		self.secondary_classifier_var.set(d.get('secondary_classifier', fallback='yolo26n-cls.pt'))
		self.secondary_epochs_var.set(int(d.get('secondary_epochs', fallback='100')))
		self.use_ncnn_var.set(self._str_to_bool(d.get('use_ncnn', fallback='false')))
		self.primary_conf_var.set(float(d.get('primary_conf_thresh', fallback='0.5')))
		self.secondary_conf_var.set(float(d.get('secondary_conf_thresh', fallback='0.5')))
		self.dominant_source_var.set(d.get('dominant_source', fallback='confidence'))

		# tracking
		self.match_distance_var.set(int(d.get('match_distance_thresh', fallback='200')))
		self.delete_after_var.set(int(d.get('delete_after_missed', fallback='5')))
		self.centroid_merge_var.set(int(d.get('centroid_merge_thresh', fallback='50')))
		self.iou_var.set(float(d.get('iou_thresh', fallback='0.4')))

		# drone motion correction
		self.drone_enabled_var.set(self._str_to_bool(d.get('drone_correction_enabled', fallback='false')))
		self.drone_model_var.set(d.get('drone_correction_model', fallback='affine'))
		self.drone_box_dilation_var.set(float(d.get('drone_correction_box_dilation', fallback='0.20')))
		self.drone_min_features_var.set(int(float(d.get('drone_correction_min_features', fallback='30'))))
		self.drone_uncertain_std_var.set(float(d.get('drone_correction_uncertain_std', fallback='8.0')))
		self.drone_smoothing_var.set(d.get('drone_correction_smoothing', fallback='savgol'))
		self.drone_smoothing_window_var.set(int(float(d.get('drone_correction_smoothing_window', fallback='7'))))
		self.drone_fallback_smoothing_var.set(self._str_to_bool(d.get('drone_correction_fallback_smoothing', fallback='true')))

		# intra-video re-identification
		self.reid_enabled_var.set(self._str_to_bool(d.get('reid_enabled', fallback='true')))
		self.reid_method_var.set(d.get('reid_method', fallback='histogram'))
		self.reid_similarity_var.set(float(d.get('reid_similarity_threshold', fallback='0.75')))
		self.reid_histogram_min_var.set(float(d.get('reid_histogram_min_similarity', fallback='0.60')))
		self.reid_max_disappeared_var.set(float(d.get('reid_max_disappeared_seconds', fallback='180.0')))
		self.reid_max_position_var.set(float(d.get('reid_max_position_distance', fallback='500.0')))
		self.ab_min_classified_var.set(int(float(d.get('ab_min_classified_frames', fallback='5'))))

		# sub-grouping (fission-fusion)
		self.subgroup_eps_bodylen_var.set(float(d.get('subgroup_eps_bodylen', fallback='4.0')))
		self.subgroup_min_stable_var.set(int(float(d.get('subgroup_min_stable_frames', fallback='10'))))
		self.foal_size_ratio_var.set(float(d.get('foal_size_ratio_thresh', fallback='0.7')))
		self.body_len_ref_scope_var.set(d.get('body_len_ref_scope', fallback='video'))

		# interaction features / graph
		self.complex_max_dist_var.set(float(d.get('complex_max_interaction_distance', fallback='400')))
		self.complex_min_duration_var.set(int(float(d.get('complex_min_duration_frames', fallback='10'))))
		self.complex_contact_iou_var.set(float(d.get('complex_contact_iou_thresh', fallback='0.05')))
		self.complex_contact_dist_var.set(float(d.get('complex_contact_dist_bodylen', fallback='1.5')))
		self.complex_window_var.set(int(float(d.get('complex_window_frames', fallback='30'))))
		self.interaction_granularity_var.set(d.get('interaction_edge_granularity', fallback='per_interaction'))
		self.interaction_weight_var.set(d.get('interaction_weight_metric', fallback='duration'))

		# complex behaviours list + model selectors
		self.complex_model_type_var.set(d.get('complex_model_type', fallback='baseline'))
		self.complex_baseline_clf_var.set(d.get('complex_baseline_classifier', fallback='random_forest'))
		cb_names = parse_list_field(d.get('complex_behaviours', fallback=''))
		cb_hotkeys = parse_list_field(d.get('complex_behaviours_hotkeys', fallback=''))
		self.complex_editor.clear()
		for i, nm in enumerate(cb_names):
			hk = cb_hotkeys[i] if i < len(cb_hotkeys) else ''
			self.complex_editor.add_row(name=nm, hotkey=hk)
		self.complex_confusion_merge_rate_var.set(float(d.get('complex_confusion_merge_rate', fallback='0.20')))
		self.complex_predict_min_proba_var.set(float(d.get('complex_predict_min_proba', fallback='0.5')))
		self.complex_speed_low_var.set(float(d.get('complex_speed_low_bodylen', fallback='0.05')))
		self.complex_speed_high_var.set(float(d.get('complex_speed_high_bodylen', fallback='0.25')))
		self.complex_polarisation_high_var.set(float(d.get('complex_polarisation_high', fallback='0.7')))
		self.complex_synchrony_high_var.set(float(d.get('complex_synchrony_high', fallback='0.7')))
		self.complex_candidate_topk_var.set(int(float(d.get('complex_candidate_topk', fallback='50'))))

		if 'kalman' in self.cfg:
			ksec = self.cfg['kalman']
			self.kalman_pos_var.set(float(ksec.get('process_noise_pos', fallback='0.01')))
			self.kalman_vel_var.set(float(ksec.get('process_noise_vel', fallback='0.01')))
			self.kalman_meas_var.set(float(ksec.get('measurement_noise', fallback='0.2')))
		else:
			self.kalman_pos_var.set(0.01)
			self.kalman_vel_var.set(0.01)
			self.kalman_meas_var.set(0.2)

		self.aug_global_prob_var.set(float(d.get('aug_global_probability', fallback='0')))
		self.aug_target_classes_var.set(d.get('aug_target_classes', fallback=''))
		self.aug_brightness_range_var.set(d.get('aug_brightness_range', fallback='0.8,1.2'))
		self.aug_brightness_prob_var.set(float(d.get('aug_brightness_probability', fallback='0')))
		self.aug_contrast_range_var.set(d.get('aug_contrast_range', fallback='0.8,1.2'))
		self.aug_contrast_prob_var.set(float(d.get('aug_contrast_probability', fallback='0')))
		self.aug_saturation_range_var.set(d.get('aug_saturation_range', fallback='0.8,1.2'))
		self.aug_saturation_prob_var.set(float(d.get('aug_saturation_probability', fallback='0')))
		self.aug_hue_range_var.set(d.get('aug_hue_range', fallback='-15,15'))
		self.aug_hue_prob_var.set(float(d.get('aug_hue_probability', fallback='0')))
		self.aug_sharpness_range_var.set(d.get('aug_sharpness_range', fallback='0.8,1.5'))
		self.aug_sharpness_prob_var.set(float(d.get('aug_sharpness_probability', fallback='0')))
		self.aug_blur_range_var.set(d.get('aug_blur_range', fallback='1,3'))
		self.aug_blur_prob_var.set(float(d.get('aug_blur_probability', fallback='0')))
		self.aug_noise_range_var.set(d.get('aug_noise_range', fallback='0,25'))
		self.aug_noise_prob_var.set(float(d.get('aug_noise_probability', fallback='0')))
		self.aug_shear_range_var.set(d.get('aug_shear_range', fallback='-0.1,0.1'))
		self.aug_shear_prob_var.set(float(d.get('aug_shear_probability', fallback='0')))
		self.aug_flip_h_options_var.set(d.get('aug_flip_h_options', fallback='True,False'))
		self.aug_flip_h_prob_var.set(float(d.get('aug_flip_h_probability', fallback='0')))
		self.aug_flip_v_options_var.set(d.get('aug_flip_v_options', fallback='True,False'))
		self.aug_flip_v_prob_var.set(float(d.get('aug_flip_v_probability', fallback='0')))
		self.aug_temperature_range_var.set(d.get('aug_temperature_range', fallback='0,10'))
		self.aug_temperature_prob_var.set(float(d.get('aug_temperature_probability', fallback='0')))

		# activity budget
		self.ab_min_presence_ratio_var.set(float(d.get('ab_min_presence_ratio', fallback='0.10')))
		self.ab_border_zone_ratio_var.set(float(d.get('ab_border_zone_ratio', fallback='0.15')))
		self.ab_group_type_separator_var.set(d.get('ab_group_type_separator', fallback='_'))
		self.ab_group_type_field_index_var.set(int(d.get('ab_group_type_field_index', fallback='4')))


		self._set_dirty(False)

	def _str_to_bool(self, s):
		if isinstance(s, bool):
			return s
		if s is None:
			return False
		return str(s).lower() in ('1', 'true', 'yes', 'on')

	# ----------------------- YAML writer -----------------------

	def _write_yaml_configs(self):
		"""
		Write static_annotations.yaml and motion_annotations.yaml into the project (settings) directory.
		Create the expected annot_static/annot_motion directories if they don't exist.
		"""
		try:
			# directories relative to the project_dir (which is dirname(ini_path))
			static_train_images_dir = os.path.join(self.project_dir, 'annot_static', 'images', 'train')
			static_val_images_dir   = os.path.join(self.project_dir, 'annot_static', 'images', 'val')
			static_train_labels_dir = os.path.join(self.project_dir, 'annot_static', 'labels', 'train')
			static_val_labels_dir   = os.path.join(self.project_dir, 'annot_static', 'labels', 'val')

			motion_train_images_dir = os.path.join(self.project_dir, 'annot_motion', 'images', 'train')
			motion_val_images_dir   = os.path.join(self.project_dir, 'annot_motion', 'images', 'val')
			motion_train_labels_dir = os.path.join(self.project_dir, 'annot_motion', 'labels', 'train')
			motion_val_labels_dir   = os.path.join(self.project_dir, 'annot_motion', 'labels', 'val')

			# create directories (images + labels)
			for d in (static_train_images_dir, static_val_images_dir,
					  static_train_labels_dir, static_val_labels_dir,
					  motion_train_images_dir, motion_val_images_dir,
					  motion_train_labels_dir, motion_val_labels_dir):
				os.makedirs(d, exist_ok=True)

			# gather class names from GUI editors (use label entries only)
			try:
				primary_static_classes = [label for (label, _, _, _) in self.class_editors['primary_static'].get()]
				primary_motion_classes = [label for (label, _, _, _) in self.class_editors['primary_motion'].get()]
			except Exception:
				primary_static_classes = []
				primary_motion_classes = []

			# YAML dicts (train/val paths are absolute)
			static_yaml_dict = {
				'train': os.path.abspath(static_train_images_dir),
				'val':   os.path.abspath(static_val_images_dir),
				'nc':	len(primary_static_classes),
				'names': primary_static_classes,
			}
			motion_yaml_dict = {
				'train': os.path.abspath(motion_train_images_dir),
				'val':   os.path.abspath(motion_val_images_dir),
				'nc':	len(primary_motion_classes),
				'names': primary_motion_classes,
			}

			static_yaml_output = os.path.join(self.project_dir, 'static_annotations.yaml')
			motion_yaml_output = os.path.join(self.project_dir, 'motion_annotations.yaml')

			# write YAMLs, preserve order of names
			with open(static_yaml_output, 'w') as yf:
				yaml.safe_dump(static_yaml_dict, yf, sort_keys=False)
			with open(motion_yaml_output, 'w') as yf:
				yaml.safe_dump(motion_yaml_dict, yf, sort_keys=False)

			# Informational prints (visible if you run GUI from terminal)
			print(f"Written static YOLO dataset config to {static_yaml_output}")
			print(f"Written motion YOLO dataset config to {motion_yaml_output}")

		except Exception as e:
			# warn user but don't prevent INI saving
			messagebox.showwarning("YAML write error", f"Failed to write dataset YAMLs: {e}")


	# ----------------------- Helpers for regeneration/backups -----------------------

	def _has_existing_annotations(self):
		"""Return True if annot_motion or annot_static contain any images/labels (train or val)."""
		for base in ('annot_motion', 'annot_static'):
			for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
				path = os.path.join(self.project_dir, base, sub)
				if os.path.isdir(path):
					# check for any files
					try:
						for _, _, files in os.walk(path):
							for f in files:
								if f and not f.startswith('.'):
									return True
					except Exception:
						continue
		return False

	def _motion_settings_changed(self):
		"""Compare loaded motion settings with current GUI values; return True if any differ."""
		if not self._loaded_motion_settings:
			# no baseline loaded -> consider as changed to be conservative
			return True
		curr = {
			'strategy': str(self.strategy_var.get()),
			'chromatic_tail_only': str(self.chromatic_tail_only_var.get()).lower(),
			'expA': str(self.expA_var.get()),
			'expB': str(self.expB_var.get()),
			'lum_weight': str(self.lum_weight_var.get()),
			'rgb_multipliers': str(self.rgb_mult_var.get()).replace(' ', ''),
			'frame_skip': str(self.frame_skip_var.get()),
			'motion_blocks_static': str(self.motion_blocks_static_var.get()).lower(),
			'static_blocks_motion': str(self.static_blocks_motion_var.get()).lower(),
		}
		# strict string comparison is fine since we recorded strings originally
		for k, v in curr.items():
			if k not in self._loaded_motion_settings or str(self._loaded_motion_settings[k]) != str(v):
				return True
		return False

	def _backup_dir(self, orig_path):
		"""If orig_path exists, rename to orig_path_backupN where N is the next integer."""
		if not os.path.isdir(orig_path):
			return None
		parent = os.path.dirname(orig_path)
		base = os.path.basename(orig_path)
		# find next available suffix (lowest unused integer)
		n = 1
		while True:
			candidate = os.path.join(parent, f"{base}_backup{n}")
			if not os.path.exists(candidate):
				break
			n += 1
		try:
			os.rename(orig_path, candidate)
			return candidate
		except Exception as e:
			# return None on failure
			print(f"Failed to backup {orig_path}: {e}")
			return None


	def _backup_primary_and_secondary_motion_models(self):
		"""Rename model_primary_motion and model_secondary_motion* directories to backup names if they exist.

		Skip any directory that already ends with _backupN so backups are not re-backed-up.
		"""
		backed = []
		primary_dir = os.path.join(self.project_dir, 'model_primary_motion')
		if os.path.isdir(primary_dir):
			b = self._backup_dir(primary_dir)
			if b:
				backed.append(b)

		# also backup the primary static (in case motion blocks static has changed) - secondary static aren't changed
		primary_dir = os.path.join(self.project_dir, 'model_primary_static')
		if os.path.isdir(primary_dir):
			b = self._backup_dir(primary_dir)
			if b:
				backed.append(b)

		# secondary directories start with 'model_secondary_motion'
		# skip any names that already end with _backup<number>
		backup_suffix_re = re.compile(r'_backup\d+$')
		for name in sorted(os.listdir(self.project_dir)):
			if not name.startswith('model_secondary_motion'):
				continue
			# ignore directories that are already backuped (name ends with _backupN)
			if backup_suffix_re.search(name):
				continue
			path = os.path.join(self.project_dir, name)
			if os.path.isdir(path):
				b = self._backup_dir(path)
				if b:
					backed.append(b)
		return backed

	def _run_regenerate_script(self):
		"""
		Search for regenerate script and run it. Returns (success:bool, message:str).
		Search order:
		  1) sibling of this GUI script (same directory)
		  2) project_dir
		  3) current working directory
		"""

		script_name = "Regenerate_annotations.py"
		launcher_dir = Path(__file__).resolve().parent
		script_path = launcher_dir / script_name

		# Call script with current Python executable and pass the project INI path
		cmd = [sys.executable, script_path, self.ini_path]
		try:
			# run in project_dir so script relative path resolution is consistent
			proc = subprocess.run(cmd, cwd=self.project_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
			out = proc.stdout.strip()
			err = proc.stderr.strip()
			if proc.returncode != 0:
				msg = f"Regeneration script failed (exit {proc.returncode}).\n\nstdout:\n{out}\n\nstderr:\n{err}"
				return False, msg
			# success
			msg = f"Regeneration script finished successfully.\n\nstdout:\n{out}"
			return True, msg
		except Exception as e:
			return False, f"Failed to run regeneration script: {e}"


	# ----------------------- Delete augmented data  -----------------------

	def _delete_augmented_data(self):
		"""
		Delete all augmented annotation files (images and labels) from the project.
		Augmented files are identified by '_aug_' in their basename.
		Asks for confirmation before deleting.
		"""
		dirs_to_scan = [
			os.path.join(self.project_dir, 'annot_static',  'images', 'train'),
			os.path.join(self.project_dir, 'annot_static',  'images', 'val'),
			os.path.join(self.project_dir, 'annot_static',  'labels', 'train'),
			os.path.join(self.project_dir, 'annot_static',  'labels', 'val'),
			os.path.join(self.project_dir, 'annot_motion',  'images', 'train'),
			os.path.join(self.project_dir, 'annot_motion',  'images', 'val'),
			os.path.join(self.project_dir, 'annot_motion',  'labels', 'train'),
			os.path.join(self.project_dir, 'annot_motion',  'labels', 'val'),
		]

		# First pass: count files to delete so the confirmation dialog is informative
		to_delete = []
		for d in dirs_to_scan:
			if not os.path.isdir(d):
				continue
			for fname in os.listdir(d):
				basename = os.path.splitext(fname)[0]
				if '_aug_' in basename:
					to_delete.append(os.path.join(d, fname))

		if not to_delete:
			messagebox.showinfo("No augmented data",
				"No augmented files found in this project.")
			return

		confirmed = messagebox.askyesno(
			"Delete augmented data",
			f"This will permanently delete {len(to_delete)} augmented files.\n\n"
			"This cannot be undone. Continue?"
		)
		if not confirmed:
			return

		deleted = 0
		errors = 0
		for fpath in to_delete:
			try:
				os.remove(fpath)
				deleted += 1
			except Exception as e:
				print(f"Could not delete {fpath}: {e}")
				errors += 1

		msg = f"Deleted {deleted} augmented files."
		if errors:
			msg += f"\n{errors} files could not be deleted (see console)."
		messagebox.showinfo("Done", msg)


	# ----------------------- Save -----------------------

	def on_save(self):
		# ---- validation ----
		hotkey_errors = self._validate_hotkeys()
		if hotkey_errors:
			messagebox.showwarning(
				"Invalid hotkeys",
				"\n".join(hotkey_errors)
			)
			return

		primary_error = self._validate_primary_classes()
		if primary_error:
			messagebox.showwarning(
				"Missing primary class",
				primary_error
			)
			return

		# Add secondary class validation
		complex_hk_errors = self._validate_complex_hotkeys()
		if complex_hk_errors:
			messagebox.showwarning("Invalid complex-behaviour hotkeys",
				"\n".join(complex_hk_errors))
			return

		secondary_valid, secondary_error = self._validate_secondary_classes()
		if not secondary_valid:
			messagebox.showwarning(
				"Secondary Class Requirements",
				secondary_error + "\n\nPlease add more secondary classes before saving."
			)
			return

		# ---- build a fresh DEFAULT dict from the current GUI state ----
		new_default = {}

		ignore_secondary_labels = []

		for key, _title in CLASS_GROUPS:
			editor = self.class_editors[key]
			items = editor.get()  # now list of (label, hotkey, (r,g,b), ignore_flag)
			labels = []
			hks = []
			cols = []
			for label, hk, col, ignored in items:
				labels.append(label)
				hks.append(hk)
				cols.append(col)
				if key.startswith('primary') and ignored:
					ignore_secondary_labels.append(label)

			new_default[f'{key}_classes'] = list_to_field(labels)
			new_default[f'{key}_hotkeys'] = list_to_field(hks)
			new_default[f'{key}_colors'] = colors_to_field(cols)

		new_default['ignore_secondary'] = list_to_field(ignore_secondary_labels)

		# paths
		new_default['clips_dir'] = self.clips_dir_var.get()
		new_default['input_dir'] = self.input_dir_var.get()
		new_default['output_dir'] = self.output_dir_var.get()


		# viewing
		new_default['motion_blocks_static'] = str(self.motion_blocks_static_var.get()).lower()
		new_default['static_blocks_motion'] = str(self.static_blocks_motion_var.get()).lower()
		new_default['ignore_secondary'] = ''  # preserve empty default unless you expose it in GUI
		new_default['save_empty_frames'] = 'true'  # preserve default unless exposed
		new_default['dominant_source'] = self.dominant_source_var.get()
		new_default['scale_factor'] = '1.0'
		new_default['line_thickness'] = str(self.line_thickness_var.get())
		new_default['font_size'] = str(self.font_size_var.get())
		new_default['val_frequency'] = str(self.val_frequency_var.get())

		# Data augmentation parameters
		new_default['aug_global_probability'] = str(self.aug_global_prob_var.get())
		new_default['aug_target_classes'] = self.aug_target_classes_var.get()
		new_default['aug_brightness_range'] = self.aug_brightness_range_var.get()
		new_default['aug_brightness_probability'] = str(self.aug_brightness_prob_var.get())
		new_default['aug_contrast_range'] = self.aug_contrast_range_var.get()
		new_default['aug_contrast_probability'] = str(self.aug_contrast_prob_var.get())
		new_default['aug_saturation_range'] = self.aug_saturation_range_var.get()
		new_default['aug_saturation_probability'] = str(self.aug_saturation_prob_var.get())
		new_default['aug_hue_range'] = self.aug_hue_range_var.get()
		new_default['aug_hue_probability'] = str(self.aug_hue_prob_var.get())
		new_default['aug_sharpness_range'] = self.aug_sharpness_range_var.get()
		new_default['aug_sharpness_probability'] = str(self.aug_sharpness_prob_var.get())
		new_default['aug_blur_range'] = self.aug_blur_range_var.get()
		new_default['aug_blur_probability'] = str(self.aug_blur_prob_var.get())
		new_default['aug_noise_range'] = self.aug_noise_range_var.get()
		new_default['aug_noise_probability'] = str(self.aug_noise_prob_var.get())
		new_default['aug_shear_range'] = self.aug_shear_range_var.get()
		new_default['aug_shear_probability'] = str(self.aug_shear_prob_var.get())
		new_default['aug_flip_h_options'] = self.aug_flip_h_options_var.get()
		new_default['aug_flip_h_probability'] = str(self.aug_flip_h_prob_var.get())
		new_default['aug_flip_v_options'] = self.aug_flip_v_options_var.get()
		new_default['aug_flip_v_probability'] = str(self.aug_flip_v_prob_var.get())
		new_default['aug_temperature_range'] = self.aug_temperature_range_var.get()
		new_default['aug_temperature_probability'] = str(self.aug_temperature_prob_var.get())

		# activity budget
		new_default['ab_min_presence_ratio']    = str(self.ab_min_presence_ratio_var.get())
		new_default['ab_border_zone_ratio']     = str(self.ab_border_zone_ratio_var.get())
		new_default['ab_group_type_separator']  = self.ab_group_type_separator_var.get()
		new_default['ab_group_type_field_index'] = str(self.ab_group_type_field_index_var.get())

		# motion strategy
		new_default['strategy'] = self.strategy_var.get()
		new_default['chromatic_tail_only'] = str(self.chromatic_tail_only_var.get()).lower()
		new_default['expA'] = str(self.expA_var.get())
		new_default['expB'] = str(self.expB_var.get())
		new_default['lum_weight'] = str(self.lum_weight_var.get())
		new_default['rgb_multipliers'] = self.rgb_mult_var.get()
		new_default['frame_skip'] = str(self.frame_skip_var.get())
		# ~ new_default['scale_factor'] = str(self.scale_factor_var.get())

		# model type
		new_default['primary_classifier'] = self.primary_classifier_var.get()
		new_default['primary_epochs'] = str(self.primary_epochs_var.get())
		new_default['secondary_classifier'] = self.secondary_classifier_var.get()
		new_default['secondary_epochs'] = str(self.secondary_epochs_var.get())
		new_default['use_ncnn'] = str(self.use_ncnn_var.get()).lower()
		new_default['primary_conf_thresh'] = str(self.primary_conf_var.get())
		new_default['secondary_conf_thresh'] = str(self.secondary_conf_var.get())

		# tracking
		new_default['match_distance_thresh'] = str(self.match_distance_var.get())
		new_default['delete_after_missed'] = str(self.delete_after_var.get())
		new_default['centroid_merge_thresh'] = str(self.centroid_merge_var.get())
		new_default['iou_thresh'] = str(self.iou_var.get())

		# drone motion correction
		new_default['drone_correction_enabled'] = str(self.drone_enabled_var.get()).lower()
		new_default['drone_correction_model'] = self.drone_model_var.get()
		new_default['drone_correction_box_dilation'] = str(self.drone_box_dilation_var.get())
		new_default['drone_correction_min_features'] = str(self.drone_min_features_var.get())
		new_default['drone_correction_uncertain_std'] = str(self.drone_uncertain_std_var.get())
		new_default['drone_correction_smoothing'] = self.drone_smoothing_var.get()
		new_default['drone_correction_smoothing_window'] = str(self.drone_smoothing_window_var.get())
		new_default['drone_correction_fallback_smoothing'] = str(self.drone_fallback_smoothing_var.get()).lower()

		# intra-video re-identification
		new_default['reid_enabled'] = str(self.reid_enabled_var.get()).lower()
		new_default['reid_method'] = self.reid_method_var.get()
		new_default['reid_similarity_threshold'] = str(self.reid_similarity_var.get())
		new_default['reid_histogram_min_similarity'] = str(self.reid_histogram_min_var.get())
		new_default['reid_max_disappeared_seconds'] = str(self.reid_max_disappeared_var.get())
		new_default['reid_max_position_distance'] = str(self.reid_max_position_var.get())
		new_default['ab_min_classified_frames'] = str(self.ab_min_classified_var.get())

		# sub-grouping (fission-fusion)
		new_default['subgroup_eps_bodylen'] = str(self.subgroup_eps_bodylen_var.get())
		new_default['subgroup_min_stable_frames'] = str(self.subgroup_min_stable_var.get())
		new_default['foal_size_ratio_thresh'] = str(self.foal_size_ratio_var.get())
		new_default['body_len_ref_scope'] = self.body_len_ref_scope_var.get()

		# interaction features / graph
		new_default['complex_max_interaction_distance'] = str(self.complex_max_dist_var.get())
		new_default['complex_min_duration_frames'] = str(self.complex_min_duration_var.get())
		new_default['complex_contact_iou_thresh'] = str(self.complex_contact_iou_var.get())
		new_default['complex_contact_dist_bodylen'] = str(self.complex_contact_dist_var.get())
		new_default['complex_window_frames'] = str(self.complex_window_var.get())
		new_default['interaction_edge_granularity'] = self.interaction_granularity_var.get()
		new_default['interaction_weight_metric'] = self.interaction_weight_var.get()

		# complex behaviours list + model selectors
		cb_items = self.complex_editor.get()
		new_default['complex_behaviours'] = list_to_field([n for n, _ in cb_items])
		new_default['complex_behaviours_hotkeys'] = list_to_field([h for _, h in cb_items])
		new_default['complex_model_type'] = self.complex_model_type_var.get()
		new_default['complex_baseline_classifier'] = self.complex_baseline_clf_var.get()
		new_default['complex_confusion_merge_rate'] = str(self.complex_confusion_merge_rate_var.get())
		new_default['complex_predict_min_proba'] = str(self.complex_predict_min_proba_var.get())
		new_default['complex_speed_low_bodylen'] = str(self.complex_speed_low_var.get())
		new_default['complex_speed_high_bodylen'] = str(self.complex_speed_high_var.get())
		new_default['complex_polarisation_high'] = str(self.complex_polarisation_high_var.get())
		new_default['complex_synchrony_high'] = str(self.complex_synchrony_high_var.get())
		new_default['complex_candidate_topk'] = str(self.complex_candidate_topk_var.get())

		# ---- write kalman section (unchanged logic) ----
		if 'kalman' not in self.cfg:
			self.cfg['kalman'] = {}
		k = self.cfg['kalman']
		k['process_noise_pos'] = str(self.kalman_pos_var.get())
		k['process_noise_vel'] = str(self.kalman_vel_var.get())
		k['measurement_noise'] = str(self.kalman_meas_var.get())



		ignore_secondary = []

		for key, editor in self.class_editors.items():
			labels, hotkeys, colors = [], [], []

			for label, hk, col, ignore_sec in editor.get():
				if not label:
					continue

				labels.append(label)
				hotkeys.append(hk)
				colors.append(col)

				if key.startswith('primary') and ignore_sec:
					ignore_secondary.append(label)

		new_default['ignore_secondary'] = ','.join(ignore_secondary)

		path_error = self._validate_paths()
		if path_error:
			messagebox.showwarning("Invalid paths", path_error)
			return

		# Determine if we should prompt for regeneration AFTER saving:
		should_prompt_regen = False
		try:
			motion_changed = self._motion_settings_changed()
			existing_annotations = self._has_existing_annotations()
			if motion_changed and existing_annotations:
				should_prompt_regen = True
		except Exception:
			# conservative: prompt if unsure
			should_prompt_regen = True

		# ---- atomically replace defaults and write file ----
		try:
			# Replace defaults atomically
			self.cfg._defaults.clear()
			self.cfg._defaults.update(new_default)

			with open(self.ini_path, 'w') as f:
				self.cfg.write(f)

			# saved successfully
			self._set_dirty(False)

			# attempt to create the annot_*/ directories and write dataset YAMLs now
			try:
				self._write_yaml_configs()
			except Exception:
				# _write_yaml_configs already shows a messagebox on failure; keep going.
				pass

			# If we should prompt for regeneration, ask now (after save so regenerate uses new settings)
			if should_prompt_regen:
				ans = messagebox.askyesno(
					"Settings changed — rebuild annotation dataset?",
					("You have changed settings that affect how the annotation images are generated,\n"
					 "and the project already contains annotations. Rebuilding the annotation dataset\n"
					 "is recommended. This operation may take some time. Do you want to rebuild now?\n\n"
					 "If you choose Yes, existing primary and secondary motion model directories\n"
					 "will be renamed to force retraining (they will be moved to *_backupN).")
				)
				if ans:
					# 1) backup model_primary_motion and model_secondary_motion* directories
					backed = self._backup_primary_and_secondary_motion_models()
					if backed:
						print("Backed up model directories:")
						for b in backed:
							print("  ", b)
					else:
						print("No primary/secondary motion model directories found to back up.")

					# 2) run regeneration script
					ok, msg = self._run_regenerate_script()
					if ok:
						messagebox.showinfo("Regeneration finished", "Dataset regeneration finished successfully.")
					else:
						messagebox.showwarning("Regeneration failed or missing", msg)

			self.destroy()   # close the window on successful save

		except Exception as e:
			messagebox.showerror('Error', f'Failed to save ini: {e}')


	def on_cancel(self):
		if self.dirty:
			if not messagebox.askyesno('Discard changes?', 'There are unsaved changes. Discard and exit?'):
				return
		self.destroy()

	def _set_dirty(self, val=True):
		self.dirty = val
		# Save button left enabled at all times, keep dirty state for Cancel checks
		# If you later want to re-disable Save when no changes present, change below:
		if self.dirty:
			self.save_btn.config(state='normal')

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print(
			"Usage: python BehaveAI_settings_gui.py "
			"<project_dir | BehaveAI_settings.ini>"
		)
		sys.exit(1)

	arg = os.path.abspath(sys.argv[1])

	if os.path.isdir(arg):
		ini_path = os.path.join(arg, "BehaveAI_settings.ini")
	else:
		ini_path = arg

	if not os.path.isfile(ini_path):
		print(f"Settings file not found: {ini_path}")
		sys.exit(1)

	app = SettingsEditorApp(ini_path=ini_path)
	app.mainloop()
