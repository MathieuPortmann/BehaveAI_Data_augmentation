# trash/ — reversible removals

This directory is **git-ignored**. Nothing here is committed. It is a staging
area for the "tracking overhaul" work (SAHI + BoT-SORT + offline stitching):
retired files and extracted code blocks are moved here instead of being deleted,
so any removal is reversible.

## Convention
- **Whole files** are moved here unchanged (e.g. `BehaveAI_reid.py`).
- **Blocks removed from a file that stays** are copied into
  `trash/<original_file>__<block>.txt` **before** being deleted from the source.

## Catalogue

- **BehaveAI_reid.py** — whole file — intra-video appearance Re-ID registry
  (histogram / MobileNetV3 / MegaDescriptor descriptors, spatio-temporal recovery
  gate). Removed: appearance is uninformative on a ~60x35px horse at 15-50m; the
  offline kinematic stitching pass now does long-gap identity recovery.
- **BehaveAI_reid_finetune.py** — whole file — self-supervised fine-tuning of the
  MegaDescriptor embedding from tracker ids (ArcFace head, needs timm). Removed
  with the Re-ID it fed.
- **BehaveAI_segmentation.py** — whole file — SAM2 / YOLO-seg whole-horse & body-part
  segmentation auto-training. Its only consumer was the Re-ID foreground mask
  (reid_foreground=yoloseg/sam2). Kept here (not deleted) because segmentation
  still has independent value for BEHAVIOUR classification (posture: head-down =
  graze, elongated = recumbent) — resurrect from here if that need materialises.
- **BehaveAI_classify_track__reid_hooks.txt** — extracted blocks — every line of
  Re-ID wiring that used to live in BehaveAI_classify_track.py: the `with_reid`
  flag on the BoT-SORT config, the `reid_registry` plumbing, the four Re-ID
  blocks inside `KalmanTracker.update` (activation guard, crop cache, recovery
  attempt before minting a new id, gallery register/prune) and the crop
  extraction at the call site. With this pass, no Re-ID code remains anywhere
  outside `trash/`.
