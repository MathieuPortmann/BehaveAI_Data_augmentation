"""
Standalone training worker for BehaveAI.

Trains a SINGLE YOLO model in its own process. The main pipeline
(BehaveAI_classify_track.py) trains several models sequentially; doing that in
one process intermittently crashed with

    CUDA error: resource already mapped  (cudaErrorAlreadyMapped)
    ... in pin memory thread for device 0

on cu130 / Blackwell GPUs, because the second training inherited a corrupted
CUDA context from the first. Running each training in a fresh process gives
every model a clean CUDA context, so the carry-over can no longer occur.

Invoked as:  python BehaveAI_train_worker.py <config.json>

The JSON config holds:
    cwd      project directory (relative dataset paths resolve against it)
    weights  pretrained weights / start checkpoint passed to YOLO(...)
    data     dataset yaml (detection) or folder (classification)
    epochs   number of epochs (cap; early stopping may end sooner)
    imgsz    training image size
    project  output project directory
    workers  dataloader workers (optional, default 4)
    patience early-stopping patience (optional; None -> Ultralytics default)
    train_overrides  dict of extra model.train() kwargs (optional), e.g.
             {"hsv_h": 0, "hsv_s": 0, "hsv_v": 0} to disable colour
             augmentation for the motion false-colour stream
"""
import os
import sys
import json


def main():
	if len(sys.argv) < 2:
		print("train worker: missing config path", file=sys.stderr)
		return 2

	with open(sys.argv[1], "r") as f:
		cfg = json.load(f)

	# Run from the project directory so relative dataset paths resolve.
	if cfg.get("cwd"):
		os.chdir(cfg["cwd"])

	from ultralytics import YOLO

	# Disable the pinned-memory dataloader thread. That thread is what raised
	# `CUDA error: resource already mapped` on this GPU/driver combo. This
	# Ultralytics version hardcodes pin_memory=True in build_dataloader and has
	# no setting for it, so we wrap the function to force it off. Cost is a
	# slightly slower host->device copy; reliability is the priority here.
	import ultralytics.data.build as _build
	import ultralytics.data as _data

	_orig_build_dataloader = _build.build_dataloader

	def _no_pin_build_dataloader(*args, **kwargs):
		kwargs["pin_memory"] = False
		return _orig_build_dataloader(*args, **kwargs)

	# Patch the source module and the package re-export. The trainers do
	# `from ultralytics.data import build_dataloader` at their own (lazy) import
	# time, so patching the `ultralytics.data` namespace before they load is what
	# makes them pick up the no-pin version. Any already-loaded trainer module is
	# patched explicitly as well.
	_build.build_dataloader = _no_pin_build_dataloader
	_data.build_dataloader = _no_pin_build_dataloader
	for _modname in (
		"ultralytics.models.yolo.detect.train",
		"ultralytics.models.yolo.classify.train",
	):
		_mod = sys.modules.get(_modname)
		if _mod is not None and hasattr(_mod, "build_dataloader"):
			_mod.build_dataloader = _no_pin_build_dataloader

	model = YOLO(cfg["weights"])
	train_kwargs = dict(
		data=cfg["data"],
		epochs=cfg["epochs"],
		imgsz=cfg["imgsz"],
		project=cfg["project"],
		name="train",
		exist_ok=True,
		workers=cfg.get("workers", 4),
	)
	# Early stopping (epochs above is the cap). None -> Ultralytics default.
	if cfg.get("patience") is not None:
		train_kwargs["patience"] = cfg["patience"]
	# Per-stream augmentation overrides (e.g. hsv=0 for the motion false-colour
	# stream, whose colour encodes the motion signal). Empty -> Ultralytics defaults.
	train_kwargs.update(cfg.get("train_overrides") or {})
	model.train(**train_kwargs)
	return 0


if __name__ == "__main__":
	sys.exit(main())
