#!/usr/bin/env python3
"""
Migrate legacy per-primary secondary crop folders to the pooled layout.

Old layout : annot_<stream>_crop/<primary>/<secondary>/*.jpg
New layout : annot_<stream>_crop/<secondary>/*.jpg   (one folder per secondary)

The two per-stream secondary classifiers train on the pooled folders, so crops
of the same secondary coming from different primaries are merged together.

Usage (run from the project directory that contains annot_static_crop / annot_motion_crop):
    python migrate_secondary_crops.py            # dry-run, prints what would happen
    python migrate_secondary_crops.py --apply     # actually move the files
    python migrate_secondary_crops.py --apply /path/to/project
"""
import os
import sys
import shutil

IMAGE_EXTS = ('.jpg', '.jpeg', '.png')


def _has_images(d):
    try:
        return any(f.lower().endswith(IMAGE_EXTS) for f in os.listdir(d))
    except Exception:
        return False


def migrate_base(base_dir, apply):
    """Migrate one annot_<stream>_crop directory. Returns (moved, skipped)."""
    if not os.path.isdir(base_dir):
        return 0, 0
    moved = 0
    skipped = 0
    for primary_name in sorted(os.listdir(base_dir)):
        primary_dir = os.path.join(base_dir, primary_name)
        if not os.path.isdir(primary_dir):
            continue
        # Already-pooled folders contain images directly -> nothing to do.
        if _has_images(primary_dir):
            skipped += 1
            continue
        # Otherwise treat children as <secondary> subfolders to lift up one level.
        for secondary_name in sorted(os.listdir(primary_dir)):
            sec_dir = os.path.join(primary_dir, secondary_name)
            if not os.path.isdir(sec_dir):
                continue
            dest_dir = os.path.join(base_dir, secondary_name)
            for fn in sorted(os.listdir(sec_dir)):
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                src = os.path.join(sec_dir, fn)
                dst = os.path.join(dest_dir, fn)
                # Avoid collisions across primaries by prefixing the primary name.
                if os.path.exists(dst):
                    dst = os.path.join(dest_dir, f"{primary_name}__{fn}")
                print(f"  {'MOVE' if apply else 'would move'}: {src} -> {dst}")
                if apply:
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.move(src, dst)
                moved += 1
        if apply:
            # Remove the now-empty primary folder tree.
            try:
                shutil.rmtree(primary_dir, ignore_errors=True)
            except Exception:
                pass
    return moved, skipped


def main():
    args = [a for a in sys.argv[1:]]
    apply = '--apply' in args
    args = [a for a in args if a != '--apply']
    project = args[0] if args else os.getcwd()
    project = os.path.abspath(project)
    print(f"Project: {project}")
    print(f"Mode   : {'APPLY (files will be moved)' if apply else 'DRY-RUN (no changes)'}")

    total_moved = 0
    for stream in ('annot_static_crop', 'annot_motion_crop'):
        base = os.path.join(project, stream)
        print(f"\n[{stream}]")
        if not os.path.isdir(base):
            print("  (not present)")
            continue
        moved, skipped = migrate_base(base, apply)
        print(f"  files {'moved' if apply else 'to move'}: {moved}; already-pooled folders: {skipped}")
        total_moved += moved

    print(f"\nDone. Total files {'moved' if apply else 'to move'}: {total_moved}")
    if not apply and total_moved:
        print("Re-run with --apply to perform the migration.")


if __name__ == '__main__':
    main()
