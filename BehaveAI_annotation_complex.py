#!/usr/bin/env python3
"""
BehaveAI Complex-Behaviour Annotation

A separate annotation tool for MULTI-INDIVIDUAL (dyadic / group) behaviours. The
existing per-individual YOLO annotation tool (BehaveAI_annotation.py) is left
completely unchanged; this module is launched on its own.

It reuses the visual conventions of the per-individual tool (Tkinter + OpenCV
canvas, seek bar, per-id coloured boxes) and follows a five-step workflow:
  1. Temporal navigation  — load the video's tracking CSV (corrected if present),
     draw per-id coloured boxes, seek bar + arrow-key navigation.
  2. Individual selection  — click boxes to build an ORDERED selection (role order),
     supporting N individuals (group play, herding, ...).
  3. Time range            — Start/End buttons (or manual entry), validated start<end.
  4. Label                 — dropdown of complex behaviours from the INI, each with a
     hotkey; optional confidence (high/medium/low).
  5. Save                  — append a row; list / edit / delete existing rows.

At pilot start annotation is fully MANUAL. A "Load candidates" button is present
but inactive; it will read <video>_complex_candidates.csv in PHASE 10 (TASK 5).

Saved file <video>_complex_behaviours.csv columns:
  video_filename, start_frame, end_frame, behaviour, track_ids,
  annotator_confidence, fps, frame_width, frame_height
(track_ids is ordered and ';'-separated; the order encodes role.)

The data/IO/validation helpers below are pure functions (no import-time side
effects); the Tk application is built only under __main__, so the helpers can be
imported and tested headlessly.
"""

import os
import sys
import csv
import argparse
import configparser

# Tk / OpenCV are only needed for the interactive app (imported lazily in main()).

COMPLEX_CSV_COLUMNS = [
	'video_filename', 'start_frame', 'end_frame', 'behaviour', 'track_ids',
	'annotator_confidence', 'fps', 'frame_width', 'frame_height',
]

CONFIDENCE_LEVELS = ['high', 'medium', 'low']

# Per-id colour palette (BGR), cycled deterministically by track id.
_PALETTE = [
	(0, 220, 255), (0, 255, 97), (236, 255, 0), (255, 188, 0),
	(255, 97, 97), (255, 62, 190), (0, 165, 255), (180, 105, 255),
	(0, 255, 255), (128, 255, 0),
]


# ---------------------------------------------------------------------------
# Pure helpers (headless-testable)
# ---------------------------------------------------------------------------

def id_color(track_id):
	"""Deterministic BGR colour for a track id (stable across runs)."""
	try:
		key = int(track_id)
	except (ValueError, TypeError):
		key = sum(ord(c) for c in str(track_id))
	return _PALETTE[key % len(_PALETTE)]


def load_complex_behaviours(config_path):
	"""Return (names, hotkeys) for the complex behaviours from the INI.

	Names come from `complex_behaviours` (comma-separated); hotkeys from
	`complex_behaviours_hotkeys` (parallel comma-separated). Missing hotkeys are
	auto-filled with 1..9 then a..z so the tool always has a usable key.
	"""
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']
	names = [n.strip() for n in d.get('complex_behaviours', '').split(',') if n.strip()]
	hk_raw = [h.strip() for h in d.get('complex_behaviours_hotkeys', '').split(',') if h.strip()]
	hotkeys = []
	for i, _name in enumerate(names):
		if i < len(hk_raw) and hk_raw[i]:
			hotkeys.append(hk_raw[i][0])
		elif i < 9:
			hotkeys.append(str(i + 1))
		else:
			hotkeys.append(chr(ord('a') + (i - 9) % 26))
	return names, hotkeys


def resolve_dirs(config_path):
	"""Return (project_dir, output_dir, input_dir, clips_dir) resolved from the INI."""
	project_dir = os.path.dirname(os.path.abspath(config_path))
	cfg = configparser.ConfigParser()
	cfg.optionxform = str
	cfg.read(config_path)
	d = cfg['DEFAULT']

	def _res(key, default):
		v = d.get(key, default)
		return v if os.path.isabs(v) else os.path.join(project_dir, v)

	return (project_dir, _res('output_dir', 'output'),
			_res('input_dir', 'input'), _res('clips_dir', 'clips'))


def find_annotatable_videos(output_dir, input_dir, clips_dir):
	"""List videos that have a tracking CSV in output_dir.

	Returns a list of dicts: {stem, video_path (or None), tracking_csv}. The
	corrected CSV is preferred over the raw tracking CSV.
	"""
	import glob
	video_index = {}
	for root in (input_dir, clips_dir):
		if root and os.path.isdir(root):
			for dirpath, _, files in os.walk(root):
				for fn in files:
					if fn.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
						video_index.setdefault(os.path.splitext(fn)[0], os.path.join(dirpath, fn))

	jobs = {}
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking_corrected.csv'))):
		jobs[os.path.basename(p).replace('_tracking_corrected.csv', '')] = p
	for p in sorted(glob.glob(os.path.join(output_dir, '*_tracking.csv'))):
		jobs.setdefault(os.path.basename(p).replace('_tracking.csv', ''), p)

	out = []
	for stem, csv_path in sorted(jobs.items()):
		out.append({'stem': stem, 'tracking_csv': csv_path,
					'video_path': video_index.get(stem)})
	return out


def build_frame_boxes(tracking_csv):
	"""Read a tracking CSV into {video_frame_index -> [(track_id, (x1,y1,x2,y2))]}.

	The classifier writes CSV frame = video_frame_index + 1, so we map back by
	subtracting 1. Rows without a usable bbox are skipped. Returns
	(frame_boxes, all_ids, has_bbox).
	"""
	frame_boxes = {}
	all_ids = set()
	has_bbox = False
	with open(tracking_csv, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		fields = reader.fieldnames or []
		has_bbox = all(c in fields for c in ('x1', 'y1', 'x2', 'y2'))
		for r in reader:
			try:
				vframe = int(r['frame']) - 1
				tid = str(r['id'])
			except (ValueError, KeyError, TypeError):
				continue
			all_ids.add(tid)
			box = None
			if has_bbox:
				try:
					x1, y1, x2, y2 = int(r['x1']), int(r['y1']), int(r['x2']), int(r['y2'])
					if x2 > x1 and y2 > y1:
						box = (x1, y1, x2, y2)
				except (ValueError, TypeError):
					box = None
			frame_boxes.setdefault(vframe, []).append((tid, box))
	return frame_boxes, all_ids, has_bbox


def validate_segment(start_frame, end_frame):
	"""Return (ok, message). A valid segment needs integer frames with start < end."""
	try:
		s = int(start_frame); e = int(end_frame)
	except (ValueError, TypeError):
		return False, "Start and end frames must be integers."
	if s < 0 or e < 0:
		return False, "Frames must be non-negative."
	if s >= e:
		return False, "Start frame must be strictly before end frame."
	return True, ""


def complex_csv_path(output_dir, video_stem):
	"""Path of the complex-behaviours CSV for a video stem."""
	return os.path.join(output_dir, video_stem + '_complex_behaviours.csv')


def read_complex_rows(csv_path):
	"""Read existing complex-behaviour rows (list of dicts); [] if absent."""
	if not os.path.exists(csv_path):
		return []
	with open(csv_path, newline='', encoding='utf-8') as f:
		return [dict(r) for r in csv.DictReader(f)]


def write_complex_rows(csv_path, rows):
	"""Write (overwrite) all complex-behaviour rows."""
	os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
	with open(csv_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=COMPLEX_CSV_COLUMNS, extrasaction='ignore')
		w.writeheader()
		w.writerows(rows)


def make_complex_row(video_filename, start_frame, end_frame, behaviour, track_ids,
					 confidence, fps, frame_width, frame_height):
	"""Build a complex-behaviour row dict. track_ids is an ordered list (role order)."""
	return {
		'video_filename': video_filename,
		'start_frame': int(start_frame),
		'end_frame': int(end_frame),
		'behaviour': behaviour,
		'track_ids': ';'.join(str(t) for t in track_ids),
		'annotator_confidence': confidence,
		'fps': fps,
		'frame_width': frame_width,
		'frame_height': frame_height,
	}


def append_complex_row(csv_path, row):
	"""Append a single complex-behaviour row, creating the file with a header if new."""
	rows = read_complex_rows(csv_path)
	rows.append(row)
	write_complex_rows(csv_path, rows)
	return rows


# ---------------------------------------------------------------------------
# Interactive Tk application (built only when run as a script)
# ---------------------------------------------------------------------------

def main():
	import tkinter as tk
	from tkinter import ttk, messagebox, filedialog
	import cv2
	from PIL import Image, ImageTk

	# ---- Resolve project / settings ----
	if len(sys.argv) > 1:
		arg = os.path.abspath(sys.argv[1])
		config_path = os.path.join(arg, 'BehaveAI_settings.ini') if os.path.isdir(arg) else arg
	else:
		root = tk.Tk(); root.withdraw()
		config_path = filedialog.askopenfilename(
			title="Select BehaveAI settings INI",
			filetypes=[("INI files", "*.ini"), ("All files", "*.*")])
		root.destroy()
		if not config_path:
			return
	config_path = os.path.abspath(config_path)
	if not os.path.exists(config_path):
		print(f"Settings file not found: {config_path}")
		return

	project_dir, output_dir, input_dir, clips_dir = resolve_dirs(config_path)
	behaviours, hotkeys = load_complex_behaviours(config_path)
	if not behaviours:
		r = tk.Tk(); r.withdraw()
		messagebox.showwarning("No complex behaviours",
			"No complex behaviours are defined.\n\nAdd some in Settings → "
			"Complex Behaviours before annotating.")
		r.destroy()
		return

	videos = find_annotatable_videos(output_dir, input_dir, clips_dir)
	videos = [v for v in videos if v['video_path']]
	if not videos:
		r = tk.Tk(); r.withdraw()
		messagebox.showinfo("No videos",
			"No videos with a tracking CSV were found.\n\nRun 'Train & batch "
			"classify' first so tracking CSVs exist in the output folder.")
		r.destroy()
		return

	# ---- Video chooser ----
	chooser = tk.Tk()
	chooser.title("Choose a video to annotate")
	chooser.geometry("520x320")
	tk.Label(chooser, text="Videos with a tracking CSV:").pack(anchor='w', padx=10, pady=(10, 4))
	lb = tk.Listbox(chooser)
	for v in videos:
		lb.insert('end', v['stem'])
	lb.pack(fill='both', expand=True, padx=10, pady=4)
	lb.selection_set(0)
	chosen = {'video': None}

	def _choose():
		sel = lb.curselection()
		if sel:
			chosen['video'] = videos[sel[0]]
			chooser.destroy()

	tk.Button(chooser, text="Annotate", command=_choose).pack(side='right', padx=10, pady=10)
	tk.Button(chooser, text="Cancel", command=chooser.destroy).pack(side='right', pady=10)
	chooser.mainloop()
	if chosen['video'] is None:
		return

	app_root = tk.Tk()
	ComplexAnnotator(app_root, chosen['video'], output_dir, behaviours, hotkeys,
					 tk, ttk, messagebox, cv2, Image, ImageTk)
	app_root.mainloop()


class ComplexAnnotator:
	"""Tkinter + OpenCV UI for annotating multi-individual behaviours."""

	def __init__(self, root, video, output_dir, behaviours, hotkeys,
				 tk, ttk, messagebox, cv2, Image, ImageTk):
		self.root = root
		self.tk = tk; self.ttk = ttk; self.messagebox = messagebox
		self.cv2 = cv2; self.Image = Image; self.ImageTk = ImageTk

		self.video = video
		self.output_dir = output_dir
		self.behaviours = behaviours
		self.hotkeys = hotkeys
		self.video_filename = os.path.basename(video['video_path'])
		self.csv_path = complex_csv_path(output_dir, video['stem'])

		# Tracking data
		self.frame_boxes, self.all_ids, self.has_bbox = build_frame_boxes(video['tracking_csv'])

		# Video
		self.cap = cv2.VideoCapture(video['video_path'])
		self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
		self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		# State
		self.frame_number = 0
		self.selection = []          # ordered list of selected track ids (role order)
		self.start_frame = None
		self.end_frame = None
		self.confidence = tk.StringVar(value='high')
		self.behaviour_var = tk.StringVar(value=behaviours[0])
		self.disp_scale = 1.0
		self.disp_offset = (0, 0)
		self._frame_cache = (None, None)  # (frame_number, bgr)

		root.title(f"BehaveAI complex — {self.video_filename}")
		root.geometry("1200x760")
		self._build_ui()
		self._refresh_rows_list()
		self.redraw()
		self._bind_keys()

	# ---- UI construction ----
	def _build_ui(self):
		tk, ttk = self.tk, self.ttk
		main = tk.Frame(self.root); main.pack(fill='both', expand=True)

		# Left: canvas + seek
		left = tk.Frame(main); left.pack(side='left', fill='both', expand=True)
		self.canvas = tk.Canvas(left, bg='black', highlightthickness=0)
		self.canvas.pack(fill='both', expand=True)
		self.canvas.bind('<Button-1>', self._on_click)
		self.canvas.bind('<Configure>', lambda e: self.redraw())

		ctrl = tk.Frame(left); ctrl.pack(fill='x', pady=4)
		self.frame_var = tk.StringVar(value="Frame 0")
		tk.Label(ctrl, textvariable=self.frame_var, width=12, anchor='w').pack(side='left', padx=4)
		self.seek = ttk.Scale(ctrl, from_=0, to=max(0, self.total_frames - 1),
							   orient='horizontal', command=self._on_seek)
		self.seek.pack(side='left', fill='x', expand=True, padx=4)

		# Right: control panel
		right = tk.Frame(main, width=320); right.pack(side='right', fill='y')
		right.pack_propagate(False)

		tk.Label(right, text="Selected individuals (role order)",
				 font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=8, pady=(8, 0))
		self.sel_list = tk.Listbox(right, height=5)
		self.sel_list.pack(fill='x', padx=8)
		selbtns = tk.Frame(right); selbtns.pack(fill='x', padx=8, pady=2)
		tk.Button(selbtns, text="Clear selection", command=self._clear_selection).pack(side='left')

		tk.Label(right, text="Time range", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=8, pady=(10, 0))
		rng = tk.Frame(right); rng.pack(fill='x', padx=8)
		tk.Button(rng, text="Set Start", command=self._set_start).pack(side='left')
		tk.Button(rng, text="Set End", command=self._set_end).pack(side='left', padx=4)
		self.range_var = tk.StringVar(value="start=–  end=–")
		tk.Label(right, textvariable=self.range_var).pack(anchor='w', padx=8)
		man = tk.Frame(right); man.pack(fill='x', padx=8, pady=2)
		tk.Label(man, text="start").pack(side='left')
		self.start_entry = tk.Entry(man, width=7); self.start_entry.pack(side='left', padx=(2, 8))
		tk.Label(man, text="end").pack(side='left')
		self.end_entry = tk.Entry(man, width=7); self.end_entry.pack(side='left', padx=2)
		tk.Button(man, text="Apply", command=self._apply_manual_range).pack(side='left', padx=4)

		tk.Label(right, text="Behaviour", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=8, pady=(10, 0))
		opts = [f"{n} ({hk})" for n, hk in zip(self.behaviours, self.hotkeys)]
		self._opt_to_name = {o: n for o, n in zip(opts, self.behaviours)}
		self.behaviour_combo = ttk.Combobox(right, values=opts, state='readonly')
		self.behaviour_combo.current(0)
		self.behaviour_combo.pack(fill='x', padx=8)

		tk.Label(right, text="Confidence", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=8, pady=(10, 0))
		conf = tk.Frame(right); conf.pack(fill='x', padx=8)
		for lvl in CONFIDENCE_LEVELS:
			tk.Radiobutton(conf, text=lvl, value=lvl, variable=self.confidence).pack(side='left')

		tk.Button(right, text="Save segment", command=self._save_segment).pack(fill='x', padx=8, pady=(12, 2))
		# Deferred to PHASE 10 (TASK 5): pre-populated reviewable candidates.
		self.load_candidates_btn = tk.Button(right, text="Load candidates (coming soon)",
											  state='disabled')
		self.load_candidates_btn.pack(fill='x', padx=8, pady=2)

		tk.Label(right, text="Saved segments", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', padx=8, pady=(10, 0))
		self.rows_list = tk.Listbox(right, height=8)
		self.rows_list.pack(fill='both', expand=True, padx=8)
		self.rows_list.bind('<<ListboxSelect>>', self._on_row_select)
		rowbtns = tk.Frame(right); rowbtns.pack(fill='x', padx=8, pady=4)
		tk.Button(rowbtns, text="Delete", command=self._delete_row).pack(side='left')
		tk.Button(rowbtns, text="Edit (load)", command=self._edit_row).pack(side='left', padx=4)

	def _bind_keys(self):
		self.root.bind('<Left>', lambda e: self._step(-1))
		self.root.bind('<Right>', lambda e: self._step(1))
		self.root.bind('<Shift-Left>', lambda e: self._step(-10))
		self.root.bind('<Shift-Right>', lambda e: self._step(10))
		# Behaviour hotkeys select the matching dropdown entry.
		for i, hk in enumerate(self.hotkeys):
			self.root.bind(hk, lambda e, idx=i: self._select_behaviour(idx))

	# ---- Navigation ----
	def _step(self, delta):
		self.frame_number = min(max(0, self.frame_number + delta), max(0, self.total_frames - 1))
		self.seek.set(self.frame_number)
		self.redraw()

	def _on_seek(self, val):
		try:
			self.frame_number = int(float(val))
		except (ValueError, TypeError):
			self.frame_number = 0
		self.redraw()

	def _select_behaviour(self, idx):
		if 0 <= idx < len(self.behaviours):
			self.behaviour_combo.current(idx)

	# ---- Selection ----
	def _on_click(self, event):
		vx, vy = self._canvas_to_video(event.x, event.y)
		hit = None
		for tid, box in self.frame_boxes.get(self.frame_number, []):
			if box and box[0] <= vx <= box[2] and box[1] <= vy <= box[3]:
				hit = tid
				break
		if hit is not None:
			if hit in self.selection:
				self.selection.remove(hit)   # click again to deselect
			else:
				self.selection.append(hit)   # preserve role order
			self._refresh_selection_list()
			self.redraw()

	def _clear_selection(self):
		self.selection = []
		self._refresh_selection_list()
		self.redraw()

	def _refresh_selection_list(self):
		self.sel_list.delete(0, 'end')
		for i, tid in enumerate(self.selection):
			self.sel_list.insert('end', f"{i + 1}. id {tid}")

	# ---- Time range ----
	def _set_start(self):
		self.start_frame = self.frame_number
		self._refresh_range()

	def _set_end(self):
		self.end_frame = self.frame_number
		self._refresh_range()

	def _apply_manual_range(self):
		try:
			self.start_frame = int(self.start_entry.get())
			self.end_frame = int(self.end_entry.get())
		except ValueError:
			self.messagebox.showerror("Invalid range", "Start/end must be integers.")
			return
		self._refresh_range()

	def _refresh_range(self):
		s = '–' if self.start_frame is None else self.start_frame
		e = '–' if self.end_frame is None else self.end_frame
		self.range_var.set(f"start={s}  end={e}")

	# ---- Save / edit / delete ----
	def _save_segment(self):
		if not self.selection:
			self.messagebox.showwarning("No selection", "Select at least one individual.")
			return
		ok, msg = validate_segment(self.start_frame, self.end_frame)
		if not ok:
			self.messagebox.showwarning("Invalid time range", msg)
			return
		behaviour = self._opt_to_name.get(self.behaviour_combo.get(), self.behaviour_combo.get())
		row = make_complex_row(
			self.video_filename, self.start_frame, self.end_frame, behaviour,
			list(self.selection), self.confidence.get(),
			f"{self.fps:.6g}", self.frame_w, self.frame_h)
		append_complex_row(self.csv_path, row)
		self._refresh_rows_list()
		print(f"Saved: {behaviour} [{self.start_frame}-{self.end_frame}] ids={row['track_ids']}")

	def _refresh_rows_list(self):
		self.rows = read_complex_rows(self.csv_path)
		self.rows_list.delete(0, 'end')
		for r in self.rows:
			self.rows_list.insert(
				'end', f"{r.get('behaviour','')} [{r.get('start_frame','')}-{r.get('end_frame','')}] "
				f"ids={r.get('track_ids','')} ({r.get('annotator_confidence','')})")

	def _on_row_select(self, event):
		pass  # selection handled on explicit Edit/Delete

	def _delete_row(self):
		sel = self.rows_list.curselection()
		if not sel:
			return
		del self.rows[sel[0]]
		write_complex_rows(self.csv_path, self.rows)
		self._refresh_rows_list()

	def _edit_row(self):
		"""Load a saved row back into the editor (selection, range, behaviour),
		then delete it so re-saving stores the corrected version."""
		sel = self.rows_list.curselection()
		if not sel:
			return
		r = self.rows[sel[0]]
		self.selection = [t for t in str(r.get('track_ids', '')).split(';') if t]
		try:
			self.start_frame = int(r.get('start_frame'))
			self.end_frame = int(r.get('end_frame'))
		except (ValueError, TypeError):
			self.start_frame = self.end_frame = None
		beh = r.get('behaviour', '')
		if beh in self.behaviours:
			self.behaviour_combo.current(self.behaviours.index(beh))
		if r.get('annotator_confidence') in CONFIDENCE_LEVELS:
			self.confidence.set(r['annotator_confidence'])
		# Remove the original; the corrected version is saved fresh.
		del self.rows[sel[0]]
		write_complex_rows(self.csv_path, self.rows)
		self._refresh_rows_list()
		self._refresh_selection_list()
		self._refresh_range()
		if self.start_frame is not None:
			self.frame_number = self.start_frame
			self.seek.set(self.frame_number)
		self.redraw()

	# ---- Coordinate mapping ----
	def _canvas_to_video(self, cx, cy):
		ox, oy = self.disp_offset
		s = self.disp_scale if self.disp_scale > 0 else 1.0
		return (cx - ox) / s, (cy - oy) / s

	# ---- Rendering ----
	def _read_frame(self):
		if self._frame_cache[0] == self.frame_number and self._frame_cache[1] is not None:
			return self._frame_cache[1]
		self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self.frame_number)
		ret, frame = self.cap.read()
		if not ret:
			import numpy as np
			frame = np.zeros((self.frame_h, self.frame_w, 3), dtype='uint8')
		self._frame_cache = (self.frame_number, frame)
		return frame

	def redraw(self):
		cv2 = self.cv2
		frame = self._read_frame().copy()
		# Draw per-id boxes; selected ids are highlighted (thicker + order index).
		for tid, box in self.frame_boxes.get(self.frame_number, []):
			if not box:
				continue
			col = id_color(tid)
			selected = tid in self.selection
			thick = 3 if selected else 1
			cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), col, thick)
			label = f"{tid}"
			if selected:
				label = f"#{self.selection.index(tid) + 1} id{tid}"
			cv2.putText(frame, label, (box[0], max(0, box[1] - 5)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, thick, cv2.LINE_AA)

		# Fit-to-canvas scaling (letterboxed, top-left anchored).
		cw = self.canvas.winfo_width() or 800
		ch = self.canvas.winfo_height() or 600
		fh, fw = frame.shape[:2]
		s = min(cw / fw, ch / fh) if fw and fh else 1.0
		self.disp_scale = s
		self.disp_offset = (0, 0)
		disp = cv2.resize(frame, (max(1, int(fw * s)), max(1, int(fh * s))))
		rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
		self._photo = self.ImageTk.PhotoImage(self.Image.fromarray(rgb))
		self.canvas.delete('all')
		self.canvas.create_image(0, 0, image=self._photo, anchor='nw')
		self.frame_var.set(f"Frame {self.frame_number}")


if __name__ == '__main__':
	# argparse keeps parity with the launcher (project path positional).
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('project_path', nargs='?', default=None)
	parser.parse_known_args()
	main()
