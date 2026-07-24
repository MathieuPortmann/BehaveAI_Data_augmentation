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

from BehaveAI_settings_help import (
    PARAM_HELP, Tooltip, apply_theme, tooltip_text, help_label, help_line,
)
from behaveai_config import (
	parse_secondary_map, format_secondary_map, load_secondary_config,
	species_key, get_species_list, load_ethogram_for_species, load_age_classes,
	DEFAULT_SPECIES,
)

INI_DEFAULT_PATH = os.path.join(os.getcwd(), 'BehaveAI_settings.ini')

CLASS_GROUPS = [
	('primary_static', 'Primary static'),
	('primary_motion', 'Primary motion'),
	('age', 'Age'),
	('secondary', 'Secondary (shared pool)'),
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

		# 'ignore secondary' is obsolete: a primary simply has no entry in secondary_map.
		self.allow_ignore_secondary = False
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


# ----------------------- Secondary mapping editor -----------------------

class SecondaryMapEditor(ttk.Frame):
	"""Per-primary checkboxes choosing which shared secondaries are allowed.

	`get_primaries()` returns a list of (primary_name, 'static'|'motion').
	`get_pool()` returns the list of shared secondary names.
	Selections are preserved across rebuilds where the (primary, secondary)
	pair still exists.
	"""
	def __init__(self, master, get_primaries, get_pool, on_change=None, *args, **kwargs):
		super().__init__(master, *args, **kwargs)
		self.get_primaries = get_primaries
		self.get_pool = get_pool
		self.on_change = on_change
		self.vars = {}  # (primary_name, secondary_name) -> BooleanVar

		header = ttk.Frame(self)
		header.pack(fill='x')
		try:
			font_bold = tkfont.nametofont("TkDefaultFont").copy()
			font_bold.configure(weight="bold", size=11)
			ttk.Label(header, text='Secondary mapping (primary → allowed secondaries)', font=font_bold).pack(side='left')
		except Exception:
			ttk.Label(header, text='Secondary mapping (primary → allowed secondaries)').pack(side='left')
		ttk.Button(header, text='Refresh', command=self.rebuild).pack(side='right')

		ttk.Label(self,
			text=('Tick the secondaries available for each primary. Untick all to give a primary '
			      'no secondary. Click Refresh after changing primaries or the pool.'),
			style='Help.TLabel', wraplength=700, justify='left').pack(anchor='w', pady=(0, 4))

		self.body = ttk.Frame(self)
		self.body.pack(fill='x')
		self.rebuild()

	def _changed(self):
		if self.on_change:
			self.on_change()

	def rebuild(self):
		prev = {k: v.get() for k, v in self.vars.items()}
		for w in self.body.winfo_children():
			w.destroy()
		self.vars = {}

		pool = [p for p in self.get_pool() if p]
		primaries = [(n, s) for (n, s) in self.get_primaries() if n]
		if not pool or not primaries:
			ttk.Label(self.body,
				text='(Define primary classes and at least one shared secondary, then Refresh.)',
				style='Help.TLabel').grid(row=0, column=0, sticky='w')
			return

		row = 0
		last_stream = None
		for name, stream in primaries:
			if stream != last_stream:
				try:
					font_bold = tkfont.nametofont("TkDefaultFont").copy()
					font_bold.configure(weight="bold")
					ttk.Label(self.body, text=('Static' if stream == 'static' else 'Motion'),
							  font=font_bold).grid(row=row, column=0, sticky='w', pady=(6, 0))
				except Exception:
					ttk.Label(self.body, text=('Static' if stream == 'static' else 'Motion')).grid(row=row, column=0, sticky='w', pady=(6, 0))
				row += 1
				last_stream = stream
			ttk.Label(self.body, text=name).grid(row=row, column=0, sticky='w', padx=(12, 6))
			col = 1
			for sec in pool:
				var = tk.BooleanVar(value=prev.get((name, sec), False))
				cb = ttk.Checkbutton(self.body, text=sec, variable=var, command=self._changed)
				cb.grid(row=row, column=col, sticky='w', padx=4)
				self.vars[(name, sec)] = var
				col += 1
			row += 1

	def get(self):
		mapping = {}
		for (prim, sec), var in self.vars.items():
			if var.get():
				mapping.setdefault(prim, []).append(sec)
		return mapping

	def set(self, mapping):
		self.rebuild()
		for prim, secs in (mapping or {}).items():
			for sec in secs:
				if (prim, sec) in self.vars:
					self.vars[(prim, sec)].set(True)


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
		apply_theme(self)
		self.geometry('920x740')
		self.minsize(820, 640)
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

		# Reference body length parameters
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

		# Per-species CLASS_GROUPS + secondary-map state, keyed by species name.
		# Only holds species the user has actually switched to/edited this session;
		# untouched species are read straight from the ini on demand (see
		# _read_species_group_state). Flushed into new_default wholesale on save.
		self._species_group_cache = {}
		self._current_editing_species = None

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

	# ----------------------- Per-species state (Species tab + Model structure) -----------------------

	def _species_names(self):
		names = [lbl for lbl, _hk, _c, _i in self.species_editor.get() if lbl]
		return names if names else [DEFAULT_SPECIES]

	def _capture_species_group_state(self):
		"""Snapshot the on-screen CLASS_GROUPS editors + secondary map (whichever
		species is currently being edited)."""
		state = {key: self.class_editors[key].get() for key, _title in CLASS_GROUPS}
		state['secondary_map'] = self.secondary_map_editor.get()
		return state

	def _read_species_group_state(self, species, species_list=None):
		"""Read a species' CLASS_GROUPS + secondary map straight from the loaded
		ini (used for species not touched yet this session). Reuses
		load_ethogram_for_species/load_age_classes (behaveai_config.py) so the
		legacy secondary-schema fallback stays in one place."""
		if species_list is None:
			species_list = self._species_names()
		eth = load_ethogram_for_species(self.cfg, species, species_list)
		age = load_age_classes(self.cfg, species, species_list)

		def _rows(names, hotkeys, colors_bgr):
			rows = []
			for i, name in enumerate(names):
				hot = hotkeys[i] if i < len(hotkeys) else ''
				bgr = colors_bgr[i] if i < len(colors_bgr) else (200, 200, 200)
				rgb = (bgr[2], bgr[1], bgr[0])  # behaveai_config stores BGR; the GUI edits RGB
				rows.append((name, hot, rgb, False))
			return rows

		return {
			'primary_static': _rows(eth['primary_static_classes'], eth['primary_static_hotkeys'], eth['primary_static_colors']),
			'primary_motion': _rows(eth['primary_motion_classes'], eth['primary_motion_hotkeys'], eth['primary_motion_colors']),
			'age': _rows(age['age_classes'], age['age_hotkeys'], age['age_colors']),
			'secondary': _rows(eth['secondary_classes'], eth['secondary_hotkeys'], eth['secondary_colors']),
			'secondary_map': eth['secondary_map'],
		}

	def _apply_species_group_state(self, state):
		"""Populate the CLASS_GROUPS editors + secondary map editor from a state dict."""
		for ed in self.class_editors.values():
			ed.set_suppress_confirm(True)
		for key, _title in CLASS_GROUPS:
			editor = self.class_editors[key]
			editor.clear()
			for label, hotkey, color, _ignored in state.get(key, []):
				editor.add_row(label=label, hotkey=hotkey, color=color)
		for ed in self.class_editors.values():
			ed.set_suppress_confirm(False)
		self.secondary_map_editor.set(state.get('secondary_map', {}))

	def _all_species_states(self):
		"""Return ({species_name: state_dict}, species_list), flushing the
		currently-edited species' on-screen state into the cache first so nothing
		edited this session is lost when validating/saving."""
		if self._current_editing_species:
			self._species_group_cache[self._current_editing_species] = self._capture_species_group_state()
		species_list = self._species_names()
		out = {}
		for sp in species_list:
			out[sp] = self._species_group_cache.get(sp) or self._read_species_group_state(sp, species_list)
		return out, species_list

	def _on_species_list_changed(self):
		self._set_dirty()
		self._refresh_species_combo()

	def _refresh_species_combo(self):
		names = self._species_names()
		self._editing_species_combo['values'] = names
		if self._current_editing_species not in names:
			if self._current_editing_species:
				self._species_group_cache.pop(self._current_editing_species, None)
			self._editing_species_var.set(names[0])
			self._current_editing_species = names[0]
			state = self._species_group_cache.get(names[0]) or self._read_species_group_state(names[0], names)
			self._apply_species_group_state(state)

	def _on_editing_species_changed(self, event=None):
		new_species = self._editing_species_var.get()
		if new_species == self._current_editing_species:
			return
		if self._current_editing_species:
			self._species_group_cache[self._current_editing_species] = self._capture_species_group_state()
		species_list = self._species_names()
		state = self._species_group_cache.get(new_species) or self._read_species_group_state(new_species, species_list)
		self._apply_species_group_state(state)
		self._current_editing_species = new_species

	def _validate_species_list(self):
		names = [lbl for lbl, _hk, _c, _i in self.species_editor.get() if lbl]
		if not names:
			return "You must define at least one species."
		seen = set()
		dupes = set()
		for n in names:
			if n in seen:
				dupes.add(n)
			seen.add(n)
		if dupes:
			return "Duplicate species name(s): " + ", ".join(sorted(dupes)) + ". Species names must be unique."
		return None

	def _validate_hotkeys(self):
		errors = []

		# Species-selector hotkeys (a single top-level list, not itself species-scoped).
		used_species = {}
		for label, hotkey, _c, _i in self.species_editor.get():
			if not hotkey:
				continue  # empty hotkey allowed - mouse click only
			if len(hotkey) != 1:
				errors.append(f"Hotkey '{hotkey}' for species '{label}' must be a single character.")
				continue
			hk = hotkey.lower()
			if hk in RESERVED_HOTKEYS:
				errors.append(f"Hotkey '{hotkey}' for species '{label}' is reserved (undo / grey-out).")
				continue
			if hk in used_species:
				errors.append(f"Species hotkey '{hotkey}' is used by both '{used_species[hk]}' and '{label}'.")
			else:
				used_species[hk] = label

		states, species_list = self._all_species_states()
		multi = len(species_list) > 1
		for sp, state in states.items():
			used = {}
			for key, _title in CLASS_GROUPS:
				for label, hotkey, _c, _i in state.get(key, []):
					if not hotkey:
						continue  # empty hotkey allowed - mouse click only, no forced letter

					if len(hotkey) != 1:
						suffix = f" ({sp})" if multi else ""
						errors.append(f"Hotkey '{hotkey}' for class '{label}'{suffix} must be a single character.")
						continue

					hk = hotkey.lower()

					if hk in RESERVED_HOTKEYS:
						suffix = f" ({sp})" if multi else ""
						errors.append(f"Hotkey '{hotkey}' for class '{label}'{suffix} is reserved (undo / grey-out).")
						continue

					if hk in used:
						suffix = f" for species '{sp}'" if multi else ""
						errors.append(f"Hotkey '{hotkey}'{suffix} is used by both '{used[hk]}' and '{label}'.")
					else:
						used[hk] = label

		return errors


	def _validate_primary_classes(self):
		states, species_list = self._all_species_states()
		missing = [sp for sp, state in states.items()
		           if not state.get('primary_motion') and not state.get('primary_static')]
		if missing:
			if len(species_list) > 1:
				names = ", ".join(f"'{m}'" for m in missing)
				return (f"The following species have no PRIMARY class defined "
				        f"(need Primary motion OR Primary static): {names}")
			return (
				"You must define at least one PRIMARY class:\n\n"
				"• Primary motion OR\n"
				"• Primary static"
			)
		return None

	def _get_primaries_for_map(self):
		"""List of (primary_name, stream) for the secondary-mapping editor."""
		out = []
		for label, _hk, _col, _ig in self.class_editors['primary_static'].get():
			if label:
				out.append((label, 'static'))
		for label, _hk, _col, _ig in self.class_editors['primary_motion'].get():
			if label:
				out.append((label, 'motion'))
		return out

	def _get_pool_for_map(self):
		"""List of shared secondary names from the pool editor."""
		return [label for label, _hk, _col, _ig in self.class_editors['secondary'].get() if label]

	def _validate_secondary_classes(self):
		"""Validate the shared secondary pool + mapping, for every species.
		- every secondary referenced in the map exists in that species' pool
		- every primary referenced in the map exists for that species
		Returns (is_valid: bool, error_message: str)
		"""
		states, species_list = self._all_species_states()
		multi = len(species_list) > 1
		errors = []
		for sp, state in states.items():
			pool = set(lbl for lbl, _hk, _c, _i in state.get('secondary', []) if lbl)
			primaries = set(lbl for lbl, _hk, _c, _i in state.get('primary_static', []) if lbl)
			primaries |= set(lbl for lbl, _hk, _c, _i in state.get('primary_motion', []) if lbl)
			mapping = state.get('secondary_map', {})
			suffix = f" ({sp})" if multi else ""
			for prim, secs in mapping.items():
				if prim not in primaries:
					errors.append(f"Mapping refers to unknown primary '{prim}'{suffix}.")
				for s in secs:
					if s not in pool:
						errors.append(f"Mapping refers to unknown secondary '{s}' (not in the pool){suffix}.")

		if errors:
			# de-duplicate while preserving order
			seen = set()
			uniq = [e for e in errors if not (e in seen or seen.add(e))]
			return False, "\n\n".join(uniq)

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

	def _scroll_tab(self, notebook, title):
		"""Create a vertically-scrollable notebook tab and return the inner frame
		to populate. Used for tabs that grow tall once help lines are added, so
		content is never clipped."""
		parent = ttk.Frame(notebook)
		notebook.add(parent, text=title)
		canvas = tk.Canvas(parent, highlightthickness=0)
		vsb = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
		inner = ttk.Frame(canvas)
		inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')
		inner.bind('<Configure>',
			lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
		canvas.bind('<Configure>',
			lambda e: canvas.itemconfigure(inner_id, width=e.width))
		canvas.configure(yscrollcommand=vsb.set)
		canvas.pack(side='left', fill='both', expand=True)
		vsb.pack(side='right', fill='y')

		def _wheel(e):
			canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
		inner.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>', _wheel))
		inner.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))
		return inner

	def _build_ui(self):
		# top toolbar: load file
		toolbar = ttk.Frame(self)
		toolbar.pack(side='top', fill='x', padx=8, pady=(8, 0))

		ttk.Label(toolbar,
			text=('Hover the  ' + 'ⓘ' + '  icons for what each setting does and how it '
			      'affects results. The grey line under each field is a quick reminder.'),
			style='Hint.TLabel', wraplength=880, justify='left').pack(
			side='left', anchor='w')

		notebook = ttk.Notebook(self)
		notebook.pack(fill='both', expand=True, padx=8, pady=6)

		# TAB 0: Species (model 0, run before primary/secondary)
		tab_species = self._scroll_tab(notebook, 'Species')
		ttk.Label(tab_species, text='Species', style='Section.TLabel').pack(anchor='w', padx=8, pady=(10, 0))
		ttk.Label(tab_species,
			text=('Species detected before the behaviour models (model 0). Use scientific names '
			      '(e.g. "Equus caballus"). The first species keeps this project\'s existing '
			      'behaviour lists and model/annotation folders; additional species get their own '
			      '(configured per-species on the "Model structure" tab) without touching the '
			      'first species\' data.'),
			style='Help.TLabel', wraplength=700, justify='left').pack(anchor='w', padx=8, pady=(0, 6))
		self.species_editor = ClassListEditor(
			tab_species, title='Species', on_change=self._on_species_list_changed,
			confirm_modify=self._confirm_modify_structure)
		self.species_editor.pack(fill='x', pady=(6,6), anchor='w', padx=8)

		# TAB 1: Model structure
		tab1 = self._scroll_tab(notebook, 'Model structure')

		ttk.Label(tab1, text='Model structure', style='Section.TLabel').pack(anchor='w', padx=8, pady=(10, 0))
		ttk.Label(tab1,
			text=('Define the behaviour classes for each stream. Each class needs a name and a '
			      'colour (display only); a single-character hotkey is optional (classes without '
			      'one are still selectable, by mouse click only). Changing classes '
			      'after annotating may require rebuilding the dataset.'),
			style='Help.TLabel', wraplength=700, justify='left').pack(anchor='w', padx=8, pady=(0, 6))

		species_row = ttk.Frame(tab1)
		species_row.pack(fill='x', padx=8, pady=(0, 6), anchor='w')
		ttk.Label(species_row, text='Editing species:').pack(side='left')
		self._editing_species_var = tk.StringVar()
		self._editing_species_combo = ttk.Combobox(
			species_row, textvariable=self._editing_species_var, state='readonly', width=30)
		self._editing_species_combo.pack(side='left', padx=(6, 0))
		self._editing_species_combo.bind('<<ComboboxSelected>>', self._on_editing_species_changed)

		self.class_editors = {}
		for key, title in CLASS_GROUPS:
			# Create the ClassListEditor which draws its own (bold) title internally.
			editor = ClassListEditor(tab1, title=title, on_change=self._set_dirty, confirm_modify=self._confirm_modify_structure)
			editor.pack(fill='x', pady=(6,6), anchor='w', padx=8)
			self.class_editors[key] = editor

		# Secondary mapping editor (primary -> allowed shared secondaries)
		self.secondary_map_editor = SecondaryMapEditor(
			tab1, self._get_primaries_for_map, self._get_pool_for_map, on_change=self._set_dirty)
		self.secondary_map_editor.pack(fill='x', pady=(6,6), anchor='w', padx=8)

		self.motion_blocks_static_var = tk.BooleanVar(value=False)
		cb_mbs = ttk.Checkbutton(tab1, text='Motion blocks static  ' + 'ⓘ', variable=self.motion_blocks_static_var, command=self._set_dirty)
		cb_mbs.pack(anchor='w', padx=8, pady=(8,0))
		Tooltip(cb_mbs, tooltip_text('motion_blocks_static'))
		ttk.Label(tab1, text=PARAM_HELP['motion_blocks_static']['short'], style='Help.TLabel').pack(anchor='w', padx=24)
		self.static_blocks_motion_var = tk.BooleanVar(value=False)
		cb_sbm = ttk.Checkbutton(tab1, text='Static blocks motion  ' + 'ⓘ', variable=self.static_blocks_motion_var, command=self._set_dirty)
		cb_sbm.pack(anchor='w', padx=8)
		Tooltip(cb_sbm, tooltip_text('static_blocks_motion'))
		ttk.Label(tab1, text=PARAM_HELP['static_blocks_motion']['short'], style='Help.TLabel').pack(anchor='w', padx=24)


		# TAB 1.2: Project paths
		tab_paths = ttk.Frame(notebook)
		notebook.add(tab_paths, text='Video paths')

		ttk.Label(tab_paths, text='Video paths', style='Section.TLabel').grid(
			row=0, column=0, columnspan=3, sticky='w', padx=8, pady=(10, 6))

		def _browse_dir(var):
			path = filedialog.askdirectory(
				initialdir=var.get() or self.project_dir,
				title='Select directory'
			)
			if path:
				var.set(path)
				self._set_dirty()

		def _path_row(parent, label, var, row, key):
			help_label(parent, label, key).grid(row=row, column=0, sticky='w', padx=8, pady=(6, 0))
			ttk.Entry(parent, textvariable=var, width=60).grid(row=row, column=1, sticky='we', padx=8, pady=(6, 0))
			ttk.Button(parent, text='Select…', command=lambda: _browse_dir(var)).grid(row=row, column=2, padx=8, pady=(6, 0))
			help_line(parent, key).grid(row=row + 1, column=0, columnspan=3, sticky='w', padx=24, pady=(0, 4))

		tab_paths.columnconfigure(1, weight=1)

		_path_row(tab_paths, 'Training video clips directory',  self.clips_dir_var, 1, 'clips_dir')
		_path_row(tab_paths, 'Batch video input directory',  self.input_dir_var, 3, 'input_dir')
		_path_row(tab_paths, 'Batch video output directory', self.output_dir_var, 5, 'output_dir')

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
		help_label(aug_inner, 'Classes to augment', 'aug_target_classes').grid(
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
		help_label(aug_inner, 'Global augmentation probability', 'aug_global_probability').grid(
			row=r, column=0, sticky='w', padx=8, pady=6)
		ttk.Spinbox(aug_inner, from_=0.0, to=1.0, increment=0.05,
					textvariable=self.aug_global_prob_var, width=8,
					command=self._set_dirty).grid(row=r, column=1, sticky='w', padx=8)
		r += 1
		help_line(aug_inner, 'aug_global_probability').grid(
			row=r, column=0, columnspan=4, sticky='w', padx=24, pady=(0, 4))
		r += 1

		# Helper: one parameter row  (label | range entry | prob label | prob spinbox)
		# Range entry is wider (width=35) to accommodate multi-segment syntax like
		#   0.5,0.8 | 1.0 | 1.2,1.6
		def _aug_row(parent, row, label, range_var, prob_var, key=None):
			help_label(parent, f'{label} (range)', key).grid(
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

		_aug_row(aug_inner, r, 'Brightness',   self.aug_brightness_range_var,   self.aug_brightness_prob_var,   'aug_brightness');   r += 1
		_aug_row(aug_inner, r, 'Contrast',     self.aug_contrast_range_var,     self.aug_contrast_prob_var,     'aug_contrast');     r += 1
		_aug_row(aug_inner, r, 'Saturation',   self.aug_saturation_range_var,   self.aug_saturation_prob_var,   'aug_saturation');   r += 1
		_aug_row(aug_inner, r, 'Hue',          self.aug_hue_range_var,          self.aug_hue_prob_var,          'aug_hue');          r += 1
		_aug_row(aug_inner, r, 'Sharpness',    self.aug_sharpness_range_var,    self.aug_sharpness_prob_var,    'aug_sharpness');    r += 1
		_aug_row(aug_inner, r, 'Blur',         self.aug_blur_range_var,         self.aug_blur_prob_var,         'aug_blur');         r += 1
		_aug_row(aug_inner, r, 'Noise',        self.aug_noise_range_var,        self.aug_noise_prob_var,        'aug_noise');        r += 1
		_aug_row(aug_inner, r, 'Shear',        self.aug_shear_range_var,        self.aug_shear_prob_var,        'aug_shear');        r += 1
		_aug_row(aug_inner, r, 'Temperature',  self.aug_temperature_range_var,  self.aug_temperature_prob_var,  'aug_temperature');  r += 1

		# Flip H — options field (not a range, kept as-is)
		help_label(aug_inner, 'Horizontal Flip (options)', 'aug_flip_h').grid(
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
		help_label(aug_inner, 'Vertical Flip (options)', 'aug_flip_v').grid(
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
		tab2 = self._scroll_tab(notebook, 'Motion strategy')

		ttk.Label(tab2, text='Motion strategy', style='Section.TLabel').grid(
			row=0, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 0))
		ttk.Label(tab2,
			text='How movement is encoded into the colour image the model is trained on. '
			     'Changing these regenerates the motion images.',
			style='Help.TLabel', wraplength=640, justify='left').grid(
			row=1, column=0, columnspan=2, sticky='w', padx=8, pady=(0, 6))
		m = 2

		help_label(tab2, 'Strategy', 'strategy').grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0))
		self.strategy_var = tk.StringVar(value='exponential')
		ttk.Combobox(tab2, values=['sequential', 'exponential'], textvariable=self.strategy_var, state='readonly', width=14).grid(row=m, column=1, sticky='w', padx=8, pady=(6, 0)); m += 1
		self.strategy_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab2, 'strategy').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		self.chromatic_tail_only_var = tk.BooleanVar(value=False)
		cb_ct = ttk.Checkbutton(tab2, text='Chromatic tail only  ' + 'ⓘ', variable=self.chromatic_tail_only_var, command=self._set_dirty)
		cb_ct.grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0)); m += 1
		Tooltip(cb_ct, tooltip_text('chromatic_tail_only'))
		help_line(tab2, 'chromatic_tail_only').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		help_label(tab2, 'Green decay (expA)', 'expA').grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0))
		self.expA_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab2, from_=0.0, to=0.99, increment=0.01, textvariable=self.expA_var, width=6, command=self._set_dirty).grid(row=m, column=1, sticky='w', padx=8); m += 1
		help_line(tab2, 'expA').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		help_label(tab2, 'Red decay (expB)', 'expB').grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0))
		self.expB_var = tk.DoubleVar(value=0.8)
		ttk.Spinbox(tab2, from_=0.0, to=0.99, increment=0.01, textvariable=self.expB_var, width=6, command=self._set_dirty).grid(row=m, column=1, sticky='w', padx=8); m += 1
		help_line(tab2, 'expB').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		help_label(tab2, 'Lum weight', 'lum_weight').grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0))
		self.lum_weight_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab2, from_=0.0, to=1.0, increment=0.01, textvariable=self.lum_weight_var, width=6, command=self._set_dirty).grid(row=m, column=1, sticky='w', padx=8); m += 1
		help_line(tab2, 'lum_weight').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		help_label(tab2, 'RGB multipliers (r,g,b)', 'rgb_multipliers').grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0))
		self.rgb_mult_var = tk.StringVar(value='4,4,4')
		ttk.Entry(tab2, textvariable=self.rgb_mult_var).grid(row=m, column=1, sticky='w', padx=8); m += 1
		self.rgb_mult_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab2, 'rgb_multipliers').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		help_label(tab2, 'Frame skip', 'frame_skip').grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0))
		self.frame_skip_var = tk.IntVar(value=0)
		ttk.Spinbox(tab2, from_=0, to=10000, textvariable=self.frame_skip_var, width=8, command=self._set_dirty).grid(row=m, column=1, sticky='w', padx=8); m += 1
		help_line(tab2, 'frame_skip').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		help_label(tab2, 'Motion threshold', 'motion_threshold').grid(row=m, column=0, sticky='w', padx=8, pady=(6, 0))
		self.motion_threshold_var = tk.IntVar(value=0)
		ttk.Spinbox(tab2, from_=0, to=255, textvariable=self.motion_threshold_var, width=8, command=self._set_dirty).grid(row=m, column=1, sticky='w', padx=8); m += 1
		help_line(tab2, 'motion_threshold').grid(row=m, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); m += 1

		# TAB 4: Model type
		tab3 = self._scroll_tab(notebook, 'Model type')

		ttk.Label(tab3, text='Model type & training', style='Section.TLabel').grid(
			row=0, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 6))
		t = 1

		help_label(tab3, 'Validation frequency', 'val_frequency').grid(row=t, column=0, sticky='w', padx=8, pady=(6, 0))
		self.val_frequency_var = tk.DoubleVar(value=0.2)
		ttk.Spinbox(tab3, from_=0.0, to=1.0, increment=0.01, textvariable=self.val_frequency_var, width=6, command=self._set_dirty).grid(row=t, column=1, sticky='w', padx=8); t += 1
		help_line(tab3, 'val_frequency').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		help_label(tab3, 'Primary classifier', 'primary_classifier').grid(row=t, column=0, sticky='w', padx=8, pady=(6, 0))
		self.primary_classifier_var = tk.StringVar(value='yolo26n.pt')
		ttk.Combobox(tab3, values=CLASSIFIER_OPTIONS, textvariable=self.primary_classifier_var).grid(row=t, column=1, sticky='w', padx=8); t += 1
		self.primary_classifier_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab3, 'primary_classifier').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		help_label(tab3, 'Primary epochs', 'primary_epochs').grid(row=t, column=0, sticky='w', padx=8, pady=(6, 0))
		self.primary_epochs_var = tk.IntVar(value=100)
		ttk.Spinbox(tab3, from_=1, to=10000, textvariable=self.primary_epochs_var, width=8, command=self._set_dirty).grid(row=t, column=1, sticky='w', padx=8); t += 1
		help_line(tab3, 'primary_epochs').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		help_label(tab3, 'Secondary classifier', 'secondary_classifier').grid(row=t, column=0, sticky='w', padx=8, pady=(6, 0))
		self.secondary_classifier_var = tk.StringVar(value='yolo26n-cls.pt')
		secondary_opts = [mm.replace('.pt','-cls.pt') for mm in CLASSIFIER_OPTIONS if mm.startswith('yolo')]
		ttk.Combobox(tab3, values=secondary_opts, textvariable=self.secondary_classifier_var).grid(row=t, column=1, sticky='w', padx=8); t += 1
		self.secondary_classifier_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab3, 'secondary_classifier').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		help_label(tab3, 'Secondary epochs', 'secondary_epochs').grid(row=t, column=0, sticky='w', padx=8, pady=(6, 0))
		self.secondary_epochs_var = tk.IntVar(value=100)
		ttk.Spinbox(tab3, from_=1, to=10000, textvariable=self.secondary_epochs_var, width=8, command=self._set_dirty).grid(row=t, column=1, sticky='w', padx=8); t += 1
		help_line(tab3, 'secondary_epochs').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		self.use_ncnn_var = tk.BooleanVar(value=False)
		cb_ncnn = ttk.Checkbutton(tab3, text='use_ncnn  ' + 'ⓘ', variable=self.use_ncnn_var, command=self._set_dirty)
		cb_ncnn.grid(row=t, column=0, sticky='w', padx=8, pady=(8, 0)); t += 1
		Tooltip(cb_ncnn, tooltip_text('use_ncnn'))
		help_line(tab3, 'use_ncnn').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		help_label(tab3, 'Primary confidence thresh', 'primary_conf_thresh').grid(row=t, column=0, sticky='w', padx=8, pady=(6, 0))
		self.primary_conf_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab3, from_=0.0, to=1.0, increment=0.01, textvariable=self.primary_conf_var, width=6, command=self._set_dirty).grid(row=t, column=1, sticky='w', padx=8); t += 1
		help_line(tab3, 'primary_conf_thresh').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		help_label(tab3, 'Secondary confidence thresh', 'secondary_conf_thresh').grid(row=t, column=0, sticky='w', padx=8, pady=(6, 0))
		self.secondary_conf_var = tk.DoubleVar(value=0.5)
		ttk.Spinbox(tab3, from_=0.0, to=1.0, increment=0.01, textvariable=self.secondary_conf_var, width=6, command=self._set_dirty).grid(row=t, column=1, sticky='w', padx=8); t += 1
		help_line(tab3, 'secondary_conf_thresh').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		help_label(tab3, 'Dominant source', 'dominant_source').grid(row=t, column=0, sticky='w', padx=8, pady=(8, 0))
		self.dominant_source_var = tk.StringVar(value='confidence')
		ttk.Combobox(tab3, values=['confidence', 'motion', 'static'], textvariable=self.dominant_source_var, state='readonly').grid(row=t, column=1, sticky='w', padx=8); t += 1
		self.dominant_source_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab3, 'dominant_source').grid(row=t, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); t += 1

		# Activity budget parameters
		self.ab_min_presence_ratio_var = tk.DoubleVar(value=0.10)
		self.ab_border_zone_ratio_var  = tk.DoubleVar(value=0.15)
		self.ab_group_type_separator_var = tk.StringVar(value='_')
		self.ab_group_type_field_index_var = tk.IntVar(value=4)

		# TAB 5: Tracking
		tab4 = self._scroll_tab(notebook, 'Tracking')
		k = 0

		ttk.Label(tab4, text='Tracking & identity', style='Section.TLabel').grid(
			row=k, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 6)); k += 1

		def _track_spin(label, key, var, lo, hi, inc=None, width=8):
			nonlocal k
			help_label(tab4, label, key).grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
			kw = {} if inc is None else {'increment': inc}
			ttk.Spinbox(tab4, from_=lo, to=hi, textvariable=var, width=width,
				command=self._set_dirty, **kw).grid(row=k, column=1, sticky='w', padx=8); k += 1
			help_line(tab4, key).grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		self.match_distance_var = tk.IntVar(value=200)
		_track_spin('Match distance thresh', 'match_distance_thresh', self.match_distance_var, 1, 10000)
		self.delete_after_var = tk.IntVar(value=10)
		_track_spin('Delete after missed', 'delete_after_missed', self.delete_after_var, 1, 10000)
		self.centroid_merge_var = tk.IntVar(value=50)
		_track_spin('Centroid merge thresh', 'centroid_merge_thresh', self.centroid_merge_var, 1, 10000)
		self.iou_var = tk.DoubleVar(value=0.5)
		_track_spin('IOU thresh (overlap required to merge)', 'iou_thresh', self.iou_var, 0.0, 1.0, 0.01, width=6)

		# Kalman subsection
		ttk.Separator(tab4, orient='horizontal').grid(row=k, column=0, columnspan=2, sticky='ew', padx=8, pady=(10, 6)); k += 1
		ttk.Label(tab4, text='Kalman filter', style='Section.TLabel').grid(row=k, column=0, sticky='w', padx=8); k += 1

		help_label(tab4, 'Process noise position', 'kalman_process_noise_pos').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		self.kalman_pos_var = tk.DoubleVar(value=0.01)
		ttk.Entry(tab4, textvariable=self.kalman_pos_var, width=10).grid(row=k, column=1, sticky='w', padx=8); k += 1
		help_line(tab4, 'kalman_process_noise_pos').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Process noise velocity', 'kalman_process_noise_vel').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		self.kalman_vel_var = tk.DoubleVar(value=0.01)
		ttk.Entry(tab4, textvariable=self.kalman_vel_var, width=10).grid(row=k, column=1, sticky='w', padx=8); k += 1
		help_line(tab4, 'kalman_process_noise_vel').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Measurement noise', 'kalman_measurement_noise').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		self.kalman_meas_var = tk.DoubleVar(value=0.2)
		ttk.Entry(tab4, textvariable=self.kalman_meas_var, width=10).grid(row=k, column=1, sticky='w', padx=8); k += 1
		help_line(tab4, 'kalman_measurement_noise').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		# Drone motion correction subsection (post-processing on the tracking CSV)
		ttk.Separator(tab4, orient='horizontal').grid(row=k, column=0, columnspan=2, sticky='ew', padx=8, pady=(10, 6)); k += 1
		ttk.Label(tab4, text='Drone motion correction', style='Section.TLabel').grid(row=k, column=0, sticky='w', padx=8); k += 1

		cb_drone = ttk.Checkbutton(tab4, text='Enable drone correction  ' + 'ⓘ',
			variable=self.drone_enabled_var, command=self._set_dirty)
		cb_drone.grid(row=k, column=0, columnspan=2, sticky='w', padx=8, pady=(4, 0)); k += 1
		Tooltip(cb_drone, tooltip_text('drone_enabled'))
		help_line(tab4, 'drone_enabled').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Transform model', 'drone_model').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab4, values=['affine', 'homography'],
			textvariable=self.drone_model_var, state='readonly', width=12).grid(row=k, column=1, sticky='w', padx=8); k += 1
		self.drone_model_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab4, 'drone_model').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Box dilation (fraction)', 'drone_box_dilation').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=0.0, to=1.0, increment=0.05,
			textvariable=self.drone_box_dilation_var, width=6, command=self._set_dirty).grid(row=k, column=1, sticky='w', padx=8); k += 1
		help_line(tab4, 'drone_box_dilation').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Min background features', 'drone_min_features').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=1, to=100000,
			textvariable=self.drone_min_features_var, width=8, command=self._set_dirty).grid(row=k, column=1, sticky='w', padx=8); k += 1
		help_line(tab4, 'drone_min_features').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Uncertain residual std (px)', 'drone_uncertain_std').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=0.0, to=1000.0, increment=0.5,
			textvariable=self.drone_uncertain_std_var, width=8, command=self._set_dirty).grid(row=k, column=1, sticky='w', padx=8); k += 1
		help_line(tab4, 'drone_uncertain_std').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Smoothing', 'drone_smoothing').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab4, values=['savgol', 'moving_average', 'none'],
			textvariable=self.drone_smoothing_var, state='readonly', width=14).grid(row=k, column=1, sticky='w', padx=8); k += 1
		self.drone_smoothing_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab4, 'drone_smoothing').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		help_label(tab4, 'Smoothing window (odd)', 'drone_smoothing_window').grid(row=k, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab4, from_=3, to=99, increment=2,
			textvariable=self.drone_smoothing_window_var, width=6, command=self._set_dirty).grid(row=k, column=1, sticky='w', padx=8); k += 1
		help_line(tab4, 'drone_smoothing_window').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		cb_fb = ttk.Checkbutton(tab4, text='Fallback to smoothing-only when features scarce  ' + 'ⓘ',
			variable=self.drone_fallback_smoothing_var, command=self._set_dirty)
		cb_fb.grid(row=k, column=0, columnspan=2, sticky='w', padx=8, pady=(4, 0)); k += 1
		Tooltip(cb_fb, tooltip_text('drone_fallback_smoothing'))
		help_line(tab4, 'drone_fallback_smoothing').grid(row=k, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); k += 1

		# TAB: Re-Identification (intra-video) — placed after Tracking, before Display
		tab_reid = self._scroll_tab(notebook, 'Re-Identification')

		_help_font = ('TkDefaultFont', 8, 'italic')
		rr = 0

		cb_reid = ttk.Checkbutton(tab_reid, text='Enable intra-video Re-ID  ' + 'ⓘ',
			variable=self.reid_enabled_var, command=self._set_dirty)
		cb_reid.grid(row=rr, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 0)); rr += 1
		Tooltip(cb_reid, tooltip_text('reid_enabled'))
		ttk.Label(tab_reid, text='Give a horse the same id after it reappears within the same video.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		help_label(tab_reid, 'Appearance method', 'reid_method').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_reid, values=['histogram', 'embedding'],
			textvariable=self.reid_method_var, state='readonly', width=12).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='histogram = colour, no torch; embedding needs torch (falls back to histogram).',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1
		self.reid_method_var.trace_add('write', lambda *a: self._set_dirty())

		help_label(tab_reid, 'Similarity threshold', 'reid_similarity_threshold').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=0.0, to=1.0, increment=0.01,
			textvariable=self.reid_similarity_var, width=6, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Embedding appearance similarity gate (cosine); only a weak tie-breaker.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		help_label(tab_reid, 'Histogram min similarity', 'reid_histogram_min_similarity').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=0.0, to=1.0, increment=0.01,
			textvariable=self.reid_histogram_min_var, width=6, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Histogram method only: minimum colour-histogram similarity (0..1) to accept an '
			'appearance match. Below this, identity relies on position/time only. Ignored when method = embedding.',
			font=_help_font, foreground='grey', wraplength=420, justify='left').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		help_label(tab_reid, 'Max disappeared (seconds)', 'reid_max_disappeared').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=1.0, to=100000.0, increment=10.0,
			textvariable=self.reid_max_disappeared_var, width=10, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Registry pruning guard only — NOT a hard match limit.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		help_label(tab_reid, 'Max position distance (px)', 'reid_max_position').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=1.0, to=100000.0, increment=10.0,
			textvariable=self.reid_max_position_var, width=10, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Spatial plausibility gate — the primary matching signal.',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		help_label(tab_reid, 'Min classified frames (group member)', 'ab_min_classified').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_reid, from_=0, to=100000,
			textvariable=self.ab_min_classified_var, width=8, command=self._set_dirty).grid(
			row=rr, column=1, sticky='w', padx=8); rr += 1
		ttk.Label(tab_reid, text='Activity budget: min frames with a known behaviour to be a group_member (0 = skip).',
			font=_help_font, foreground='grey').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		# ---- Advanced ReID (appearance descriptor) ----
		ttk.Label(tab_reid, text='Advanced ReID', style='Section.TLabel').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=8, pady=(12, 2)); rr += 1
		ttk.Label(tab_reid, text='Shape of the appearance descriptor. Defaults reproduce the legacy '
			'single-histogram behaviour; change only if you know the ReID body-part pipeline.',
			font=_help_font, foreground='grey', wraplength=420, justify='left').grid(
			row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); rr += 1

		self.reid_descriptor_var = tk.StringVar(value='global')
		help_label(tab_reid, 'Descriptor layout', 'reid_descriptor').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_reid, values=['global', 'grid'], textvariable=self.reid_descriptor_var,
			state='readonly', width=12).grid(row=rr, column=1, sticky='w', padx=8); rr += 1
		self.reid_descriptor_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab_reid, 'reid_descriptor').grid(row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); rr += 1

		self.reid_grid_var = tk.StringVar(value='3x3')
		help_label(tab_reid, 'Grid (RxC)', 'reid_grid').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Entry(tab_reid, textvariable=self.reid_grid_var, width=8).grid(row=rr, column=1, sticky='w', padx=8); rr += 1
		self.reid_grid_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab_reid, 'reid_grid').grid(row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); rr += 1

		self.reid_foreground_var = tk.StringVar(value='hsv')
		help_label(tab_reid, 'Foreground masking', 'reid_foreground').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_reid, values=['hsv', 'sam2', 'yoloseg'], textvariable=self.reid_foreground_var,
			state='readonly', width=12).grid(row=rr, column=1, sticky='w', padx=8); rr += 1
		self.reid_foreground_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab_reid, 'reid_foreground').grid(row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); rr += 1

		self.reid_orient_var = tk.BooleanVar(value=False)
		cb_orient = ttk.Checkbutton(tab_reid, text='Orient grid to body axis  ' + 'ⓘ',
			variable=self.reid_orient_var, command=self._set_dirty)
		cb_orient.grid(row=rr, column=0, columnspan=2, sticky='w', padx=8, pady=(6, 0)); rr += 1
		Tooltip(cb_orient, tooltip_text('reid_orient'))
		help_line(tab_reid, 'reid_orient').grid(row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); rr += 1

		self.reid_backbone_var = tk.StringVar(value='T-224')
		help_label(tab_reid, 'MegaDescriptor backbone', 'reid_backbone').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_reid, values=['T-224', 'L-224', 'L-384', 'T-CNN-288'], textvariable=self.reid_backbone_var,
			state='readonly', width=12).grid(row=rr, column=1, sticky='w', padx=8); rr += 1
		self.reid_backbone_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab_reid, 'reid_backbone').grid(row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); rr += 1

		self.reid_checkpoint_var = tk.StringVar(value='')
		help_label(tab_reid, 'Fine-tuned checkpoint', 'reid_checkpoint').grid(row=rr, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Entry(tab_reid, textvariable=self.reid_checkpoint_var, width=40).grid(row=rr, column=1, sticky='w', padx=8); rr += 1
		self.reid_checkpoint_var.trace_add('write', lambda *a: self._set_dirty())
		help_line(tab_reid, 'reid_checkpoint').grid(row=rr, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4)); rr += 1

		# TAB: Interaction features / graph (TASK 4 primary output)
		tab_int = self._scroll_tab(notebook, 'Interaction')
		ir = 0
		ttk.Label(tab_int, text='Interaction features & graph',
			font=('TkDefaultFont', 10, 'bold')).grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 0)); ir += 1
		ttk.Label(tab_int, text='Per-frame dyadic/group features aggregated into the interaction graph (edges/nodes CSVs). Group features are computed over the whole co-present herd per frame.',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=8, pady=(0, 6)); ir += 1

		help_label(tab_int, 'Foal size ratio threshold', 'foal_size_ratio').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=0.0, to=1.0, increment=0.05,
			textvariable=self.foal_size_ratio_var, width=6, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		ttk.Label(tab_int, text='body_len / reference below this flags a likely foal.',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); ir += 1

		help_label(tab_int, 'Body-length reference scope', 'body_len_ref_scope').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_int, values=['video', 'segment'],
			textvariable=self.body_len_ref_scope_var, state='readonly', width=12).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		ttk.Label(tab_int, text='video = one reference; segment = recompute on altitude/zoom drift.',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); ir += 1
		self.body_len_ref_scope_var.trace_add('write', lambda *a: self._set_dirty())

		help_label(tab_int, 'Max interaction distance (px)', 'complex_max_dist').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=1.0, to=100000.0, increment=10.0,
			textvariable=self.complex_max_dist_var, width=8, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		ttk.Label(tab_int, text='Pairs farther apart than this are not treated as interacting.',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); ir += 1

		help_label(tab_int, 'Min interaction duration (frames)', 'complex_min_duration').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=1, to=100000,
			textvariable=self.complex_min_duration_var, width=8, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		help_label(tab_int, 'Contact IoU threshold', 'complex_contact_iou').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=0.0, to=1.0, increment=0.01,
			textvariable=self.complex_contact_iou_var, width=6, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		help_label(tab_int, 'Contact distance (body lengths)', 'complex_contact_dist').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=0.0, to=50.0, increment=0.1,
			textvariable=self.complex_contact_dist_var, width=6, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		help_label(tab_int, 'Window length (frames)', 'complex_window').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_int, from_=1, to=100000,
			textvariable=self.complex_window_var, width=8, command=self._set_dirty).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1

		help_label(tab_int, 'Edge granularity', 'interaction_granularity').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_int, values=['per_interaction', 'per_segment', 'per_frame'],
			textvariable=self.interaction_granularity_var, state='readonly', width=16).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		ttk.Label(tab_int, text='Changing this regenerates and overwrites the edges file.',
			font=_help_font, foreground='grey').grid(
			row=ir, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 6)); ir += 1
		self.interaction_granularity_var.trace_add('write', lambda *a: self._set_dirty())

		help_label(tab_int, 'Edge weight metric', 'interaction_weight').grid(row=ir, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Combobox(tab_int, values=['duration', 'proximity', 'combined'],
			textvariable=self.interaction_weight_var, state='readonly', width=16).grid(
			row=ir, column=1, sticky='w', padx=8); ir += 1
		self.interaction_weight_var.trace_add('write', lambda *a: self._set_dirty())

		# TAB: Complex Behaviours (label list + model selectors)
		tab_cb = self._scroll_tab(notebook, 'Complex Behaviours')

		self.complex_editor = ComplexLabelEditor(tab_cb, on_change=self._set_dirty)
		self.complex_editor.pack(fill='x', padx=8, pady=(10, 4), anchor='w')
		ttk.Label(tab_cb,
			text='Single user-editable list of dyadic AND group behaviours; each needs a '
				 'unique single-character hotkey.',
			font=_help_font, foreground='grey').pack(anchor='w', padx=8, pady=(0, 8))

		mdl = ttk.Frame(tab_cb); mdl.pack(fill='x', padx=8, pady=(4, 0))
		help_label(mdl, 'Model type', 'complex_model_type').grid(row=0, column=0, sticky='w', pady=(4, 0))
		ttk.Combobox(mdl, values=['baseline', 'lstm', 'transformer'],
			textvariable=self.complex_model_type_var, state='readonly', width=16).grid(
			row=0, column=1, sticky='w', padx=8)
		help_label(mdl, 'Baseline classifier', 'complex_baseline_clf').grid(row=1, column=0, sticky='w', pady=(6, 0))
		ttk.Combobox(mdl, values=['random_forest', 'hist_gradient_boosting'],
			textvariable=self.complex_baseline_clf_var, state='readonly', width=20).grid(
			row=1, column=1, sticky='w', padx=8)
		self.complex_model_type_var.trace_add('write', lambda *a: self._set_dirty())
		self.complex_baseline_clf_var.trace_add('write', lambda *a: self._set_dirty())
		ttk.Label(tab_cb,
			text="baseline = scikit-learn classifier; lstm/transformer need torch. "
				 "Interaction thresholds are on the 'Interaction' tab.",
			font=_help_font, foreground='grey').pack(anchor='w', padx=8, pady=(8, 0))

		# Model + candidate-heuristic thresholds
		thr = ttk.LabelFrame(tab_cb, text='Model & candidate thresholds')
		thr.pack(fill='x', padx=8, pady=(10, 4))
		def _thr_row(rownum, label, var, lo, hi, inc, helptext, key=None):
			help_label(thr, label, key).grid(row=rownum * 2, column=0, sticky='w', padx=6, pady=(4, 0))
			ttk.Spinbox(thr, from_=lo, to=hi, increment=inc, textvariable=var, width=8,
				command=self._set_dirty).grid(row=rownum * 2, column=1, sticky='w', padx=6)
			ttk.Label(thr, text=helptext, font=_help_font, foreground='grey').grid(
				row=rownum * 2 + 1, column=0, columnspan=2, sticky='w', padx=24)
		_thr_row(0, 'Confusion merge rate', self.complex_confusion_merge_rate_var, 0.0, 1.0, 0.05,
				 'Confusion >= this flags a class pair as a merge suggestion.', 'complex_confusion_merge_rate')
		_thr_row(1, 'Predict min probability', self.complex_predict_min_proba_var, 0.0, 1.0, 0.05,
				 'Minimum probability to emit a complex-behaviour prediction.', 'complex_predict_min_proba')
		_thr_row(2, 'Speed ~still (body len/frame)', self.complex_speed_low_var, 0.0, 5.0, 0.01,
				 'Speeds below this count as ~stationary in the candidate heuristics.', 'complex_speed_low')
		_thr_row(3, 'Speed fast (body len/frame)', self.complex_speed_high_var, 0.0, 10.0, 0.05,
				 'Speeds above this count as fast (gallop / chase).', 'complex_speed_high')
		_thr_row(4, 'Polarisation high', self.complex_polarisation_high_var, 0.0, 1.0, 0.05,
				 'Group alignment above this suggests trek/stampede.', 'complex_polarisation_high')
		_thr_row(5, 'Synchrony high', self.complex_synchrony_high_var, 0.0, 1.0, 0.05,
				 'Behavioural synchrony above this suggests synchronised rest/graze.', 'complex_synchrony_high')
		_thr_row(6, 'Active-learning top-K', self.complex_candidate_topk_var, 1, 100000, 1,
				 'Number of most-uncertain windows surfaced as candidates.', 'complex_candidate_topk')

		# TAB 6: Display
		# Scrollable: the box & label style section makes this tab taller than the window.
		tab5 = self._scroll_tab(notebook, 'Display Settings')

		# viewing options
		ttk.Label(tab5, text='Viewing options', style='Section.TLabel').pack(anchor='w', padx=8, pady=(10, 4))
		self.line_thickness_var = tk.IntVar(value=1)
		help_label(tab5, 'Line thickness', 'line_thickness').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=1, to=10, textvariable=self.line_thickness_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'line_thickness').pack(anchor='w', padx=24, pady=(0, 4))

		self.font_size_var = tk.DoubleVar(value=0.6)
		help_label(tab5, 'Font size', 'font_size').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.1, to=5.0, increment=0.1, textvariable=self.font_size_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'font_size').pack(anchor='w', padx=24, pady=(0, 4))

		self.box_line_scale_var = tk.DoubleVar(value=0.5)
		help_label(tab5, 'Annotation tool box line scale', 'box_line_scale').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.05, to=2.0, increment=0.05, textvariable=self.box_line_scale_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'box_line_scale').pack(anchor='w', padx=24, pady=(0, 4))

		self.box_font_scale_var = tk.DoubleVar(value=0.35)
		help_label(tab5, 'Annotation tool box font scale', 'box_font_scale').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.05, to=2.0, increment=0.05, textvariable=self.box_font_scale_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'box_font_scale').pack(anchor='w', padx=24, pady=(0, 4))

		self.buttons_per_row_var = tk.IntVar(value=8)
		help_label(tab5, 'Class buttons per row', 'buttons_per_row').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=1, to=20, textvariable=self.buttons_per_row_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'buttons_per_row').pack(anchor='w', padx=24, pady=(0, 4))

		# Box & label style - shared by the output videos and the annotation tool
		ttk.Label(tab5, text='Box & label style', style='Section.TLabel').pack(anchor='w', padx=8, pady=(14, 4))

		self.adaptive_box_scaling_var = tk.BooleanVar(value=True)
		cb_abs = ttk.Checkbutton(tab5, text='Scale font/lines to box size  ' + 'ⓘ',
			variable=self.adaptive_box_scaling_var, command=self._set_dirty)
		cb_abs.pack(anchor='w', padx=8, pady=(6, 0))
		Tooltip(cb_abs, tooltip_text('adaptive_box_scaling'))
		ttk.Label(tab5, text=PARAM_HELP['adaptive_box_scaling']['short'], style='Help.TLabel').pack(anchor='w', padx=24, pady=(0, 4))

		self.adaptive_font_coeff_var = tk.DoubleVar(value=0.005)
		help_label(tab5, 'Adaptive font coefficient', 'adaptive_font_coeff').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.001, to=0.05, increment=0.001, format='%.3f', textvariable=self.adaptive_font_coeff_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'adaptive_font_coeff').pack(anchor='w', padx=24, pady=(0, 4))

		self.adaptive_font_min_var = tk.DoubleVar(value=0.35)
		help_label(tab5, 'Adaptive font min', 'adaptive_font_min').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.1, to=5.0, increment=0.1, textvariable=self.adaptive_font_min_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'adaptive_font_min').pack(anchor='w', padx=24, pady=(0, 4))

		self.adaptive_font_max_var = tk.DoubleVar(value=0.9)
		help_label(tab5, 'Adaptive font max', 'adaptive_font_max').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.1, to=5.0, increment=0.1, textvariable=self.adaptive_font_max_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'adaptive_font_max').pack(anchor='w', padx=24, pady=(0, 4))

		self.adaptive_thickness_coeff_var = tk.DoubleVar(value=0.012)
		help_label(tab5, 'Adaptive thickness coefficient', 'adaptive_thickness_coeff').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.001, to=0.2, increment=0.001, format='%.3f', textvariable=self.adaptive_thickness_coeff_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'adaptive_thickness_coeff').pack(anchor='w', padx=24, pady=(0, 4))

		self.adaptive_thickness_min_var = tk.DoubleVar(value=1.0)
		help_label(tab5, 'Adaptive thickness min', 'adaptive_thickness_min').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.0, to=20.0, increment=0.05, format='%.2f', textvariable=self.adaptive_thickness_min_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'adaptive_thickness_min').pack(anchor='w', padx=24, pady=(0, 4))

		self.adaptive_thickness_max_var = tk.DoubleVar(value=3.0)
		help_label(tab5, 'Adaptive thickness max', 'adaptive_thickness_max').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.0, to=20.0, increment=0.05, format='%.2f', textvariable=self.adaptive_thickness_max_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'adaptive_thickness_max').pack(anchor='w', padx=24, pady=(0, 4))

		self.label_bg_mode_var = tk.StringVar(value='translucent')
		help_label(tab5, 'Label background', 'label_bg_mode').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Combobox(tab5, values=['none', 'translucent', 'solid'], textvariable=self.label_bg_mode_var, state='readonly', width=14).pack(anchor='w', padx=8)
		help_line(tab5, 'label_bg_mode').pack(anchor='w', padx=24, pady=(0, 4))

		self.label_bg_opacity_var = tk.DoubleVar(value=0.5)
		help_label(tab5, 'Label background opacity', 'label_bg_opacity').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.0, to=1.0, increment=0.05, textvariable=self.label_bg_opacity_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'label_bg_opacity').pack(anchor='w', padx=24, pady=(0, 4))

		self.label_bg_color_var = tk.StringVar(value='0,0,0')
		help_label(tab5, 'Label background colour (R,G,B)', 'label_bg_color').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Entry(tab5, textvariable=self.label_bg_color_var, width=12).pack(anchor='w', padx=8)
		help_line(tab5, 'label_bg_color').pack(anchor='w', padx=24, pady=(0, 4))

		self.halo_thickness_var = tk.DoubleVar(value=1.0)
		help_label(tab5, 'Box halo thickness', 'halo_thickness').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Spinbox(tab5, from_=0.0, to=5.0, increment=0.05, format='%.2f', textvariable=self.halo_thickness_var, width=6, command=self._set_dirty).pack(anchor='w', padx=8)
		help_line(tab5, 'halo_thickness').pack(anchor='w', padx=24, pady=(0, 4))

		self.halo_color_var = tk.StringVar(value='0,0,0')
		help_label(tab5, 'Box halo colour (R,G,B)', 'halo_color').pack(anchor='w', padx=8, pady=(6,0))
		ttk.Entry(tab5, textvariable=self.halo_color_var, width=12).pack(anchor='w', padx=8)
		help_line(tab5, 'halo_color').pack(anchor='w', padx=24, pady=(0, 4))

		self.show_species_var = tk.BooleanVar(value=True)
		cb_sp = ttk.Checkbutton(tab5, text='Show species on boxes  ' + 'ⓘ',
			variable=self.show_species_var, command=self._set_dirty)
		cb_sp.pack(anchor='w', padx=8, pady=(6, 0))
		Tooltip(cb_sp, tooltip_text('show_species'))
		ttk.Label(tab5, text=PARAM_HELP['show_species']['short'], style='Help.TLabel').pack(anchor='w', padx=24)

		self.show_age_var = tk.BooleanVar(value=True)
		cb_ag = ttk.Checkbutton(tab5, text='Show age on boxes  ' + 'ⓘ',
			variable=self.show_age_var, command=self._set_dirty)
		cb_ag.pack(anchor='w', padx=8)
		Tooltip(cb_ag, tooltip_text('show_age'))
		ttk.Label(tab5, text=PARAM_HELP['show_age']['short'], style='Help.TLabel').pack(anchor='w', padx=24, pady=(0, 4))

		# TAB 7: Activity Budget
		tab_ab = ttk.Frame(notebook)
		notebook.add(tab_ab, text='Activity Budget')

		ttk.Label(tab_ab, text='Activity budget', style='Section.TLabel').grid(
			row=0, column=0, columnspan=2, sticky='w', padx=8, pady=(10, 6))

		help_label(tab_ab, 'Min presence ratio (stranger threshold)', 'ab_min_presence_ratio').grid(
			row=1, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_ab, from_=0.01, to=1.0, increment=0.01,
			textvariable=self.ab_min_presence_ratio_var,
			width=8, command=self._set_dirty).grid(row=1, column=1, sticky='w', padx=8)
		help_line(tab_ab, 'ab_min_presence_ratio').grid(row=2, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4))

		help_label(tab_ab, 'Border zone ratio (stranger threshold)', 'ab_border_zone_ratio').grid(
			row=3, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_ab, from_=0.01, to=0.5, increment=0.01,
			textvariable=self.ab_border_zone_ratio_var,
			width=8, command=self._set_dirty).grid(row=3, column=1, sticky='w', padx=8)
		help_line(tab_ab, 'ab_border_zone_ratio').grid(row=4, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4))

		help_label(tab_ab, 'Filename field separator', 'ab_group_type_separator').grid(
			row=5, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Entry(tab_ab, textvariable=self.ab_group_type_separator_var,
			width=4).grid(row=5, column=1, sticky='w', padx=8)
		help_line(tab_ab, 'ab_group_type_separator').grid(row=6, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4))

		help_label(tab_ab, 'Group type field index (0-based)', 'ab_group_type_field_index').grid(
			row=7, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_ab, from_=0, to=10, increment=1,
			textvariable=self.ab_group_type_field_index_var,
			width=6, command=self._set_dirty).grid(row=7, column=1, sticky='w', padx=8)
		help_line(tab_ab, 'ab_group_type_field_index').grid(row=8, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4))

		self.ab_analysis_duration_var = tk.DoubleVar(value=0.0)
		help_label(tab_ab, 'Analysis duration (s, 0 = whole video)', 'ab_analysis_duration_s').grid(
			row=9, column=0, sticky='w', padx=8, pady=(6, 0))
		ttk.Spinbox(tab_ab, from_=0.0, to=100000.0, increment=1.0,
			textvariable=self.ab_analysis_duration_var,
			width=10, command=self._set_dirty).grid(row=9, column=1, sticky='w', padx=8)
		help_line(tab_ab, 'ab_analysis_duration_s').grid(row=10, column=0, columnspan=2, sticky='w', padx=24, pady=(0, 4))

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


		# species
		self.species_editor.set_suppress_confirm(True)
		self.species_editor.clear()
		species_list = get_species_list(self.cfg)
		sp_hotkeys = parse_list_field(d.get('species_hotkeys', fallback='0'))
		sp_colors = parse_colors_field(d.get('species_colors', fallback='0'))
		for i, name in enumerate(species_list):
			hot = sp_hotkeys[i] if i < len(sp_hotkeys) else ''
			col = sp_colors[i] if i < len(sp_colors) else (200, 200, 200)
			self.species_editor.add_row(label=name, hotkey=hot, color=col)
		self.species_editor.set_suppress_confirm(False)

		# classes, for the currently-selected species (species_list[0] on load).
		# Identical to the legacy unscoped behaviour for a single-species project,
		# since species_key() resolves to the bare key for species_list[0].
		self._species_group_cache = {}
		self._editing_species_combo['values'] = species_list
		self._editing_species_var.set(species_list[0])
		self._current_editing_species = species_list[0]
		state = self._read_species_group_state(species_list[0], species_list)
		self._apply_species_group_state(state)


		# viewing
		self.line_thickness_var.set(int(d.get('line_thickness', fallback='1')))
		self.font_size_var.set(float(d.get('font_size', fallback='0.6')))
		self.box_line_scale_var.set(float(d.get('box_line_scale', fallback='0.5')))
		self.box_font_scale_var.set(float(d.get('box_font_scale', fallback='0.35')))
		self.buttons_per_row_var.set(int(d.get('buttons_per_row', fallback='8')))

		# box & label style (see behaveai_render.py)
		self.adaptive_box_scaling_var.set(self._str_to_bool(d.get('adaptive_box_scaling', fallback='true')))
		self.adaptive_font_coeff_var.set(float(d.get('adaptive_font_coeff', fallback='0.005')))
		self.adaptive_font_min_var.set(float(d.get('adaptive_font_min', fallback='0.35')))
		self.adaptive_font_max_var.set(float(d.get('adaptive_font_max', fallback='0.9')))
		self.adaptive_thickness_coeff_var.set(float(d.get('adaptive_thickness_coeff', fallback='0.012')))
		self.adaptive_thickness_min_var.set(float(d.get('adaptive_thickness_min', fallback='1.0')))
		self.adaptive_thickness_max_var.set(float(d.get('adaptive_thickness_max', fallback='3.0')))
		self.label_bg_mode_var.set(d.get('label_bg_mode', fallback='translucent'))
		self.label_bg_opacity_var.set(float(d.get('label_bg_opacity', fallback='0.5')))
		self.label_bg_color_var.set(d.get('label_bg_color', fallback='0,0,0'))
		self.halo_thickness_var.set(float(d.get('halo_thickness', fallback='1.0')))
		self.halo_color_var.set(d.get('halo_color', fallback='0,0,0'))
		self.show_species_var.set(self._str_to_bool(d.get('show_species', fallback='true')))
		self.show_age_var.set(self._str_to_bool(d.get('show_age', fallback='true')))

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
		self.motion_threshold_var.set(int(float(d.get('motion_threshold', fallback='0'))))
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
		# advanced re-id appearance descriptor
		self.reid_descriptor_var.set(d.get('reid_descriptor', fallback='global'))
		self.reid_grid_var.set(d.get('reid_grid', fallback='3x3'))
		self.reid_foreground_var.set(d.get('reid_foreground', fallback='hsv'))
		self.reid_orient_var.set(self._str_to_bool(d.get('reid_orient', fallback='false')))
		self.reid_backbone_var.set(d.get('reid_backbone', fallback='T-224'))
		self.reid_checkpoint_var.set(d.get('reid_checkpoint', fallback=''))

		# reference body length
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
		self.ab_analysis_duration_var.set(float(d.get('ab_analysis_duration_s', fallback='0')))


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

		# secondary model directories (per-stream): 'model_secondary_motion' and
		# 'model_secondary_static'. Skip any names that already end with _backup<number>.
		backup_suffix_re = re.compile(r'_backup\d+$')
		for name in sorted(os.listdir(self.project_dir)):
			if not (name.startswith('model_secondary_motion') or name.startswith('model_secondary_static')):
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
		species_error = self._validate_species_list()
		if species_error:
			messagebox.showwarning("Invalid species list", species_error)
			return

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

		# Species list itself (top-level, not species-scoped).
		species_items = self.species_editor.get()
		new_default['species_list'] = list_to_field([lbl for lbl, _hk, _c, _i in species_items])
		new_default['species_hotkeys'] = list_to_field([hk for _l, hk, _c, _i in species_items])
		new_default['species_colors'] = colors_to_field([c for _l, _hk, c, _i in species_items])

		# CLASS_GROUPS (primary_static/primary_motion/age/secondary) + secondary_map,
		# written out for every species using species-scoped key names (species_key
		# resolves to the bare key for the first species, so an existing
		# single-species project's keys are byte-identical to before).
		states, species_list = self._all_species_states()
		for sp in species_list:
			state = states[sp]
			for key, _title in CLASS_GROUPS:
				labels, hks, cols = [], [], []
				for label, hk, col, _ignored in state.get(key, []):
					labels.append(label)
					hks.append(hk)
					cols.append(col)
				new_default[species_key(f'{key}_classes', sp, species_list)] = list_to_field(labels)
				new_default[species_key(f'{key}_hotkeys', sp, species_list)] = list_to_field(hks)
				new_default[species_key(f'{key}_colors', sp, species_list)] = colors_to_field(cols)

			new_default[species_key('secondary_map', sp, species_list)] = format_secondary_map(
				state.get('secondary_map', {}))

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
		new_default['box_line_scale'] = str(self.box_line_scale_var.get())
		new_default['box_font_scale'] = str(self.box_font_scale_var.get())
		new_default['buttons_per_row'] = str(self.buttons_per_row_var.get())

		# box & label style (see behaveai_render.py)
		new_default['adaptive_box_scaling'] = str(self.adaptive_box_scaling_var.get()).lower()
		new_default['adaptive_font_coeff'] = str(self.adaptive_font_coeff_var.get())
		new_default['adaptive_font_min'] = str(self.adaptive_font_min_var.get())
		new_default['adaptive_font_max'] = str(self.adaptive_font_max_var.get())
		new_default['adaptive_thickness_coeff'] = str(self.adaptive_thickness_coeff_var.get())
		new_default['adaptive_thickness_min'] = str(self.adaptive_thickness_min_var.get())
		new_default['adaptive_thickness_max'] = str(self.adaptive_thickness_max_var.get())
		new_default['label_bg_mode'] = self.label_bg_mode_var.get()
		new_default['label_bg_opacity'] = str(self.label_bg_opacity_var.get())
		new_default['label_bg_color'] = self.label_bg_color_var.get()
		new_default['halo_thickness'] = str(self.halo_thickness_var.get())
		new_default['halo_color'] = self.halo_color_var.get()
		new_default['show_species'] = str(self.show_species_var.get()).lower()
		new_default['show_age'] = str(self.show_age_var.get()).lower()

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
		new_default['ab_analysis_duration_s']   = str(self.ab_analysis_duration_var.get())

		# motion strategy
		new_default['strategy'] = self.strategy_var.get()
		new_default['chromatic_tail_only'] = str(self.chromatic_tail_only_var.get()).lower()
		new_default['expA'] = str(self.expA_var.get())
		new_default['expB'] = str(self.expB_var.get())
		new_default['lum_weight'] = str(self.lum_weight_var.get())
		new_default['rgb_multipliers'] = self.rgb_mult_var.get()
		new_default['frame_skip'] = str(self.frame_skip_var.get())
		new_default['motion_threshold'] = str(self.motion_threshold_var.get())
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
		new_default['reid_descriptor'] = self.reid_descriptor_var.get()
		new_default['reid_grid'] = self.reid_grid_var.get()
		new_default['reid_foreground'] = self.reid_foreground_var.get()
		new_default['reid_orient'] = str(self.reid_orient_var.get()).lower()
		new_default['reid_backbone'] = self.reid_backbone_var.get()
		new_default['reid_checkpoint'] = self.reid_checkpoint_var.get().strip()

		# reference body length
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

		# ---- preserve metric-geometry keys (not yet exposed as widgets) ----
		# on_save rebuilds DEFAULT from scratch, so carry forward any metric_*
		# keys already on disk (incl. per-drone metric_fpx_* checkerboard
		# overrides) so a settings save never silently drops them.
		try:
			prev_default = dict(self.cfg['DEFAULT']) if 'DEFAULT' in self.cfg else {}
		except Exception:
			prev_default = {}
		_metric_defaults = {
			'metric_enabled': 'false',
			'metric_focal_len_mm': '24.0',
			'metric_sensor_width_mm': '36.0',
			'metric_roll_max_deg': '3.0',
			'metric_horizon_margin_px': '50',
		}
		for mk, mv in _metric_defaults.items():
			new_default[mk] = prev_default.get(mk, mv)
		for mk, mv in prev_default.items():
			if mk.startswith('metric_fpx_'):
				new_default[mk] = mv

		# ---- write kalman section (unchanged logic) ----
		if 'kalman' not in self.cfg:
			self.cfg['kalman'] = {}
		k = self.cfg['kalman']
		k['process_noise_pos'] = str(self.kalman_pos_var.get())
		k['process_noise_vel'] = str(self.kalman_vel_var.get())
		k['measurement_noise'] = str(self.kalman_meas_var.get())

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
