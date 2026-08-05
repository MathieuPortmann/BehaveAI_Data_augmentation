#!/usr/bin/env python3
"""
BehaveAI Intra-video Re-Identification

Gives a horse the SAME track id after it reappears within the SAME video (e.g.
after an occlusion or after leaving and re-entering the frame). This is purely
INTRA-video: there is no inter-video / cross-session logic here.

Re-identification works WITHOUT any labelled data by combining two signals:

  * Spatial / temporal plausibility (PRIMARY, mandatory). A returning detection
    may only match a recently-lost track if it reappears within a generous
    Euclidean distance of where that track was lost. This gate dominates the
    decision; disappearances can last from seconds to minutes, so there is NO
    hard maximum-disappearance in the matching logic (max_disappeared_frames is
    only a registry-pruning guard).

  * Appearance similarity (SECONDARY, weak tie-breaker). A colour histogram
    (default, no torch) or an optional CNN embedding ranks the gated candidates.
    IMPORTANT — in monochrome herds (e.g. Exmoor ponies, all one brown, often
    against same-coloured ground) appearance is essentially UNINFORMATIVE; when
    the best appearance score is below threshold the match falls back to the
    spatially closest plausible candidate, so ids are still recovered using the
    spatio-temporal gate alone.

Descriptors are computed ONLY when a track is lost (registered) and when a new
track would otherwise be created — never per-frame-per-track.
"""

import os

import numpy as np
import cv2

# torch / torchvision are optional and only used by reid_method == 'embedding'.
try:
	import torch
	import torchvision
	_TORCH_AVAILABLE = True
except Exception:
	_TORCH_AVAILABLE = False


# OpenCV hue range (0..179) considered "green" — grass / foliage — and excluded
# from the colour histogram so the background does not dominate the descriptor.
_GREEN_HUE_LOW = 35
_GREEN_HUE_HIGH = 85


class ReIDRegistry:
	"""Registry of recently-lost tracks used to re-identify returning horses.

	The matching decision is driven by spatio-temporal plausibility; appearance
	similarity is only a weak tie-breaker and is uninformative in monochrome
	herds (see module docstring).
	"""

	def __init__(self, method="histogram", similarity_threshold=0.75,
				 histogram_min_similarity=0.60,
				 max_position_distance=500.0, max_disappeared_frames=900,
				 descriptor="global", grid="3x3", foreground="hsv",
				 orient=False, backbone="T-224", checkpoint=None):
		self.method = method
		# similarity_threshold gates the embedding (cosine) method; histogram scores
		# live on a different scale, so the histogram method has its own gate.
		self.similarity_threshold = float(similarity_threshold)
		self.histogram_min_similarity = float(histogram_min_similarity)
		self.max_position_distance = float(max_position_distance)
		self.max_disappeared_frames = int(max_disappeared_frames)

		# Spatial layout of the appearance descriptor:
		#   'global' -> single masked HSV histogram of the central box (legacy).
		#   'grid'   -> one masked HSV histogram per cell of a foreground-aware grid,
		#               concatenated (a coarse "body parts" descriptor).
		self.descriptor = str(descriptor).lower()
		self._grid_rows, self._grid_cols = self._parse_grid(grid)
		# Foreground source used to ignore background before binning:
		#   'hsv'     -> exclude green hues (legacy, zero extra dependency).
		#   'sam2'/'yoloseg' -> per-crop silhouette via ultralytics, hsv on failure.
		self.foreground = str(foreground).lower()
		# Align the grid to the body's major axis (PCA on the foreground mask) so the
		# same cell maps to the same body region regardless of heading. Off by default.
		self.orient = bool(orient)
		self.backbone = str(backbone)
		# Optional path to a self-supervised fine-tuned MegaDescriptor checkpoint
		# (produced by BehaveAI_reid_finetune.py). Loaded if it exists.
		self.checkpoint = checkpoint or None

		# track_id -> {'descriptor': np.ndarray|None, 'position': (x, y), 'frame': int}
		self.lost = {}
		# Appearance score of the most recent successful match (for logging).
		self.last_match_score = 0.0
		self._monochrome_noted = False

		# Optional segmentation model for foreground masking, loaded lazily ONCE.
		self._seg_model = None
		self._seg_failed = False

		# Optional embedding model, loaded ONCE here. 'embedding' uses a torchvision
		# backbone (ImageNet); 'megadescriptor' uses the BVRA animal-ReID foundation
		# model via timm. Both fall back to the colour histogram if unavailable.
		self._embed_model = None
		if self.method in ("embedding", "megadescriptor"):
			if _TORCH_AVAILABLE:
				try:
					self._embed_model = self._load_embed_model()
					print(f"Re-ID: using {self.method} embedding descriptor.")
				except Exception as e:
					print(f"Re-ID: embedding unavailable ({e}); "
						  f"falling back to colour histogram.")
					self.method = "histogram"
			else:
				print("Re-ID: torch/torchvision not found; "
					  "falling back to colour histogram.")
				self.method = "histogram"

	@staticmethod
	def _parse_grid(grid):
		"""Parse a 'RxC' grid spec into (rows, cols); default 3x3 on bad input."""
		try:
			r, c = str(grid).lower().split("x")
			r, c = int(r), int(c)
			if r >= 1 and c >= 1:
				return r, c
		except Exception:
			pass
		return 3, 3

	# ------------------------------------------------------------------
	# Embedding model (optional)
	# ------------------------------------------------------------------

	def _load_embed_model(self):
		"""Load the appearance backbone once, ready to return a feature vector.

		'megadescriptor' loads the BVRA animal-ReID foundation model via timm
		(input 224, normalise 0.5); 'embedding' loads a torchvision MobileNetV3
		(input 128, ImageNet stats) with its classifier head removed.
		"""
		if self.method == "megadescriptor":
			import timm
			# A fine-tuned checkpoint records which backbone it was trained on; use
			# it so the architecture matches the saved weights.
			ckpt = None
			backbone = self.backbone
			if self.checkpoint and os.path.isfile(self.checkpoint):
				ckpt = torch.load(self.checkpoint, map_location="cpu")
				backbone = ckpt.get("backbone", backbone)
			tag = {"T-224": "BVRA/MegaDescriptor-T-224",
				   "L-224": "BVRA/MegaDescriptor-L-224",
				   "L-384": "BVRA/MegaDescriptor-L-384",
				   "T-CNN-288": "BVRA/MegaDescriptor-T-CNN-288"}.get(
					   backbone, "BVRA/MegaDescriptor-T-224")
			# pretrained=False when loading our own weights avoids a needless download.
			model = timm.create_model(f"hf-hub:{tag}", pretrained=(ckpt is None),
									  num_classes=0)
			if ckpt is not None:
				model.load_state_dict(ckpt["state_dict"], strict=False)
				print(f"Re-ID: loaded fine-tuned MegaDescriptor ({self.checkpoint}).")
			model.eval()
			self._embed_size = 384 if "384" in tag else (288 if "288" in tag else 224)
			self._embed_mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
			self._embed_std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
			return model

		from torchvision import models
		try:
			weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
			model = models.mobilenet_v3_small(weights=weights)
		except Exception:
			model = models.mobilenet_v3_small(pretrained=True)
		# Drop the classifier so the penultimate feature vector is returned.
		model.classifier = torch.nn.Identity()
		model.eval()
		self._embed_size = 128
		self._embed_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
		self._embed_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
		return model

	def _embedding_descriptor(self, bgr_crop):
		"""Backbone embedding of a BGR crop (L2-normalised float32).

		The crop is converted BGR->RGB so the channel order matches the weights
		the backbone was trained on.
		"""
		s = getattr(self, "_embed_size", 128)
		crop = cv2.resize(bgr_crop, (s, s))
		crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
		x = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
		x = (x - self._embed_mean) / self._embed_std
		x = x.unsqueeze(0)
		with torch.no_grad():
			feat = self._embed_model(x).flatten().cpu().numpy().astype(np.float32)
		norm = np.linalg.norm(feat)
		return feat / norm if norm > 0 else feat

	# ------------------------------------------------------------------
	# Colour-histogram descriptor (default)
	# ------------------------------------------------------------------

	def _histogram_descriptor(self, bgr_crop):
		"""HSV colour histogram of the central ~60% of the box.

		64 bins per channel, green hues excluded (grass/foliage), L2-normalised.
		"""
		h, w = bgr_crop.shape[:2]
		# Central ~60% of the box to avoid corner background.
		y0, y1 = int(h * 0.2), int(h * 0.8)
		x0, x1 = int(w * 0.2), int(w * 0.8)
		central = bgr_crop[y0:y1, x0:x1]
		if central.size == 0:
			central = bgr_crop

		hsv = cv2.cvtColor(central, cv2.COLOR_BGR2HSV)
		hue = hsv[:, :, 0]
		# Mask out green pixels (grass / foliage) before binning.
		non_green = ~((hue >= _GREEN_HUE_LOW) & (hue <= _GREEN_HUE_HIGH))
		mask = (non_green.astype(np.uint8)) * 255
		if int(mask.sum()) == 0:
			mask = None  # all-green crop -> use everything rather than nothing

		ranges = [(0, 180), (0, 256), (0, 256)]  # H, S, V
		hists = []
		for ch in range(3):
			hist = cv2.calcHist([hsv], [ch], mask, [64], list(ranges[ch]))
			hists.append(hist.flatten())
		desc = np.concatenate(hists).astype(np.float32)
		norm = np.linalg.norm(desc)
		return desc / norm if norm > 0 else desc

	# ------------------------------------------------------------------
	# Foreground masking (shared by the grid descriptor)
	# ------------------------------------------------------------------

	def _foreground_mask(self, bgr_crop):
		"""Return a uint8 0/255 foreground mask for the whole crop, or None.

		'hsv' excludes green hues (grass/foliage); 'sam2'/'yoloseg' delegate to a
		segmentation model and fall back to the hsv rule on any failure. None means
		"use every pixel" (e.g. an all-green crop), so callers must tolerate it.
		"""
		if self.foreground in ("sam2", "yoloseg"):
			m = self._seg_mask(bgr_crop)
			if m is not None:
				return m
			# fall through to the hsv rule on segmentation failure
		hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
		hue = hsv[:, :, 0]
		non_green = ~((hue >= _GREEN_HUE_LOW) & (hue <= _GREEN_HUE_HIGH))
		mask = (non_green.astype(np.uint8)) * 255
		return mask if int(mask.sum()) > 0 else None

	def _seg_mask(self, bgr_crop):
		"""Silhouette of the dominant object in the crop via ultralytics, or None.

		The segmentation model is loaded once on first use; if it is unavailable
		the registry permanently falls back to the hsv rule (self._seg_failed).
		"""
		if self._seg_failed:
			return None
		try:
			if self._seg_model is None:
				from ultralytics import SAM, YOLO
				if self.foreground == "sam2":
					self._seg_model = SAM("sam2_b.pt")
				else:
					self._seg_model = YOLO("yolo11n-seg.pt")
			res = self._seg_model.predict(bgr_crop, verbose=False)
			if not res or res[0].masks is None or len(res[0].masks.data) == 0:
				return None
			# Largest mask = the animal filling most of its own detection box.
			data = res[0].masks.data.cpu().numpy()
			areas = data.reshape(data.shape[0], -1).sum(axis=1)
			m = data[int(np.argmax(areas))]
			m = cv2.resize(m, (bgr_crop.shape[1], bgr_crop.shape[0]))
			return (m > 0.5).astype(np.uint8) * 255
		except Exception as e:
			print(f"Re-ID: segmentation foreground unavailable ({e}); using hsv rule.")
			self._seg_failed = True
			return None

	# ------------------------------------------------------------------
	# Orientation + grid (coarse "body parts") descriptor
	# ------------------------------------------------------------------

	@staticmethod
	def _estimate_orientation(mask):
		"""Angle (deg) of the foreground's major axis via PCA, or None.

		The axis is direction-ambiguous (180-degree flip) — it aligns the grid to
		the body but does not by itself disambiguate head from tail.
		"""
		if mask is None:
			return None
		ys, xs = np.nonzero(mask)
		if xs.size < 20:
			return None
		pts = np.column_stack([xs, ys]).astype(np.float32)
		pts -= pts.mean(axis=0)
		cov = np.cov(pts.T)
		evals, evecs = np.linalg.eigh(cov)
		major = evecs[:, int(np.argmax(evals))]
		return float(np.degrees(np.arctan2(major[1], major[0])))

	@staticmethod
	def _rotate(img, angle, flags):
		h, w = img.shape[:2]
		M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
		return cv2.warpAffine(img, M, (w, h), flags=flags,
							   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

	def _grid_descriptor(self, bgr_crop):
		"""Concatenated per-cell masked HSV histograms over a foreground grid.

		With self.orient the grid is rotated onto the body's major axis so a cell
		maps to the same body region regardless of heading. Empty cells contribute
		zeros, so the vector length is fixed (rows*cols*3*32) and stays comparable
		across detections in find_match.
		"""
		mask = self._foreground_mask(bgr_crop)
		crop = bgr_crop
		if self.orient:
			angle = self._estimate_orientation(mask)
			if angle is not None:
				crop = self._rotate(bgr_crop, angle, cv2.INTER_LINEAR)
				mask = self._rotate(mask, angle, cv2.INTER_NEAREST) \
					if mask is not None else None

		hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
		h, w = hsv.shape[:2]
		rows, cols = self._grid_rows, self._grid_cols
		ranges = [(0, 180), (0, 256), (0, 256)]  # H, S, V
		bins = 32
		cells = []
		for r in range(rows):
			y0, y1 = int(h * r / rows), int(h * (r + 1) / rows)
			for c in range(cols):
				x0, x1 = int(w * c / cols), int(w * (c + 1) / cols)
				cell_hsv = hsv[y0:y1, x0:x1]
				cell_mask = None if mask is None else mask[y0:y1, x0:x1]
				if cell_mask is not None and int(cell_mask.sum()) == 0:
					cells.append(np.zeros(bins * 3, dtype=np.float32))
					continue
				if cell_hsv.size == 0:
					cells.append(np.zeros(bins * 3, dtype=np.float32))
					continue
				ch_hists = [cv2.calcHist([cell_hsv], [ch], cell_mask, [bins],
										 list(ranges[ch])).flatten() for ch in range(3)]
				cells.append(np.concatenate(ch_hists).astype(np.float32))
		desc = np.concatenate(cells).astype(np.float32)
		norm = np.linalg.norm(desc)
		return desc / norm if norm > 0 else desc

	# ------------------------------------------------------------------
	# Public descriptor entry point
	# ------------------------------------------------------------------

	def extract_descriptor(self, bgr_crop):
		"""Return an L2-normalised appearance descriptor for a BGR crop, or None.

		Crops smaller than 10x10 px (or empty) yield None; matching then relies on
		the spatio-temporal gate alone.
		"""
		if bgr_crop is None or getattr(bgr_crop, 'size', 0) == 0:
			return None
		h, w = bgr_crop.shape[:2]
		if h < 10 or w < 10:
			return None
		try:
			if self.method in ("embedding", "megadescriptor") and self._embed_model is not None:
				return self._embedding_descriptor(bgr_crop)
			if self.descriptor == "grid":
				return self._grid_descriptor(bgr_crop)
			return self._histogram_descriptor(bgr_crop)
		except Exception:
			return None

	# ------------------------------------------------------------------
	# Registry operations
	# ------------------------------------------------------------------

	def register_lost_track(self, track_id, descriptor, position, frame_number):
		"""Remember a track that has just been deleted so it can be recovered."""
		self.lost[track_id] = {
			'descriptor': descriptor,
			'position': (float(position[0]), float(position[1])),
			'frame': int(frame_number),
		}

	def find_match(self, descriptor, position, frame_number):
		"""Return the id of a plausible lost track for a new detection, or None.

		The spatial gate is applied first; among gated candidates the best
		appearance match above the similarity threshold wins, otherwise the
		spatially closest plausible candidate is accepted (monochrome-herd
		recovery). The chosen entry is removed (one-to-one matching).
		"""
		self.last_match_score = 0.0
		if not self.lost:
			return None

		px, py = float(position[0]), float(position[1])

		# 1) Spatial/temporal plausibility gate (mandatory, dominant signal).
		gated = []
		for tid, e in self.lost.items():
			ex, ey = e['position']
			dist = float(np.hypot(px - ex, py - ey))
			if dist <= self.max_position_distance:
				gated.append((tid, dist, e['descriptor']))
		if not gated:
			return None

		# 2) Appearance ranking among gated candidates (weak tie-breaker).
		best_tid, best_sim = None, -1.0
		if descriptor is not None:
			for tid, _dist, desc in gated:
				if desc is not None and desc.shape == descriptor.shape:
					sim = float(np.dot(descriptor, desc))  # cosine (both L2-normalised)
					if sim > best_sim:
						best_sim, best_tid = sim, tid

		# Pick the appearance threshold for the active descriptor type. Histogram
		# scores and embedding cosine scores are not on the same scale, so each
		# method has its own calibrated gate.
		if self.method == "histogram":
			appearance_threshold = self.histogram_min_similarity
		else:  # "embedding"
			appearance_threshold = self.similarity_threshold

		if best_tid is not None and best_sim >= appearance_threshold:
			chosen = best_tid
			self.last_match_score = best_sim
		else:
			# Spatially closest plausible candidate (recovers monochrome herds).
			chosen = min(gated, key=lambda c: c[1])[0]
			self.last_match_score = max(best_sim, 0.0)
			if not self._monochrome_noted:
				print("Re-ID note: appearance is a weak tie-breaker and is "
					  "uninformative in monochrome herds; relying on "
					  "spatio-temporal plausibility.")
				self._monochrome_noted = True

		del self.lost[chosen]
		return chosen

	def prune_old_entries(self, current_frame):
		"""Drop registry entries older than the pruning guard (NOT a match limit)."""
		drop = [tid for tid, e in self.lost.items()
				if current_frame - e['frame'] > self.max_disappeared_frames]
		for tid in drop:
			del self.lost[tid]
