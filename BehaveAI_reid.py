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
				 max_position_distance=500.0, max_disappeared_frames=900):
		self.method = method
		self.similarity_threshold = float(similarity_threshold)
		self.max_position_distance = float(max_position_distance)
		self.max_disappeared_frames = int(max_disappeared_frames)

		# track_id -> {'descriptor': np.ndarray|None, 'position': (x, y), 'frame': int}
		self.lost = {}
		# Appearance score of the most recent successful match (for logging).
		self.last_match_score = 0.0
		self._monochrome_noted = False

		# Optional embedding model, loaded ONCE here.
		self._embed_model = None
		if self.method == "embedding":
			if _TORCH_AVAILABLE:
				try:
					self._embed_model = self._load_embed_model()
					print("Re-ID: using torchvision embedding descriptor.")
				except Exception as e:
					print(f"Re-ID: embedding unavailable ({e}); "
						  f"falling back to colour histogram.")
					self.method = "histogram"
			else:
				print("Re-ID: torch/torchvision not found; "
					  "falling back to colour histogram.")
				self.method = "histogram"

	# ------------------------------------------------------------------
	# Embedding model (optional)
	# ------------------------------------------------------------------

	def _load_embed_model(self):
		"""Load a torchvision backbone once, classifier head removed."""
		from torchvision import models
		try:
			weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
			model = models.mobilenet_v3_small(weights=weights)
		except Exception:
			model = models.mobilenet_v3_small(pretrained=True)
		# Drop the classifier so the penultimate feature vector is returned.
		model.classifier = torch.nn.Identity()
		model.eval()
		self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
		self._imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
		return model

	def _embedding_descriptor(self, bgr_crop):
		"""Penultimate-layer embedding of a BGR crop (L2-normalised float32)."""
		crop = cv2.resize(bgr_crop, (128, 128))
		x = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
		x = (x - self._imagenet_mean) / self._imagenet_std
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
			if self.method == "embedding" and self._embed_model is not None:
				return self._embedding_descriptor(bgr_crop)
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

		if best_tid is not None and best_sim >= self.similarity_threshold:
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
