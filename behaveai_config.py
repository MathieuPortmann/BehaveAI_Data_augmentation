# behaveai_config.py
# Shared loader for the secondary-behaviour configuration used by the annotation
# tool, the settings GUI and the classify/track pipeline.
#
# New schema (BehaveAI_settings.ini, [DEFAULT]):
#   secondary_classes = Recumbent,Water          # shared pool (single set, INI order)
#   secondary_hotkeys = r,w
#   secondary_colors  = 143,85,27;60,56,126      # "R,G,B" entries separated by ';'
#   secondary_map     = Graze:Recumbent|Water; Rest:Recumbent|Water
#
# The stream (static/motion) of a secondary is inferred from the primary it is
# attached to (a primary listed in primary_static_classes -> static crop/model,
# otherwise -> motion). A primary absent from secondary_map (or mapped to an
# empty list) simply has no secondary.
#
# Backward compatibility: if `secondary_classes`/`secondary_map` are absent, the
# loader falls back to the legacy keys (secondary_static_classes,
# secondary_motion_classes, ignore_secondary) and reconstructs an equivalent
# pool + map so existing projects keep working.

DEFAULT_SECONDARY_COLOR = (200, 200, 200)  # BGR, used when a colour is missing


def parse_class_list(raw):
    """Split a comma-separated INI list, dropping blanks and the '0' sentinel."""
    if raw is None:
        return []
    items = [x.strip() for x in str(raw).split(',')]
    return [x for x in items if x and x != '0']


def _parse_colors(raw):
    """Parse ';'-separated 'R,G,B' colours into BGR tuples (matching the rest of
    the codebase, which stores colours reversed for OpenCV)."""
    out = []
    for c in str(raw or '').split(';'):
        c = c.strip()
        if not c or c == '0':
            continue
        try:
            rgb = tuple(int(v) for v in c.split(','))
            out.append(rgb[::-1])  # RGB -> BGR
        except Exception:
            out.append(DEFAULT_SECONDARY_COLOR)
    return out


def parse_secondary_map(raw):
    """Parse 'Primary:secA|secB; Primary2:secC' into {primary: [secondaries]}."""
    mapping = {}
    if not raw:
        return mapping
    for entry in str(raw).split(';'):
        entry = entry.strip()
        if not entry or ':' not in entry:
            continue
        prim, secs = entry.split(':', 1)
        prim = prim.strip()
        if not prim or prim == '0':
            continue
        sec_list = [s.strip() for s in secs.split('|') if s.strip() and s.strip() != '0']
        mapping[prim] = sec_list
    return mapping


def format_secondary_map(mapping):
    """Inverse of parse_secondary_map, for writing the INI from the settings GUI."""
    parts = []
    for prim, secs in mapping.items():
        secs = [s for s in secs if s]
        if not prim or not secs:
            continue
        parts.append("{}:{}".format(prim, '|'.join(secs)))
    return '; '.join(parts)


def _align(seq, n, default):
    """Pad/truncate a list to length n using `default` for missing entries."""
    seq = list(seq)
    if len(seq) < n:
        seq = seq + [default] * (n - len(seq))
    return seq[:n]


def load_secondary_config(config, primary_static_classes, primary_motion_classes):
    """Load the shared secondary pool + mapping from a configparser object.

    Returns a dict with:
      secondary_classes      list[str]                 (pool, stable INI order)
      secondary_hotkeys      list[str]                 (aligned to pool length)
      secondary_colors       list[tuple]               (BGR, aligned to pool length)
      secondary_map          dict[str, list[str]]      (primary name -> secondaries)
      allowed_secondary_idx  list[list[int]]           (indexed by primary index in
                                                         primary_static + primary_motion)
      hierarchical_mode      bool
    """
    d = config['DEFAULT']
    primary_static_classes = parse_class_list(','.join(primary_static_classes)) if isinstance(primary_static_classes, (list, tuple)) else parse_class_list(primary_static_classes)
    primary_motion_classes = parse_class_list(','.join(primary_motion_classes)) if isinstance(primary_motion_classes, (list, tuple)) else parse_class_list(primary_motion_classes)
    primary_classes = list(primary_static_classes) + list(primary_motion_classes)
    static_set = set(primary_static_classes)

    pool = parse_class_list(d.get('secondary_classes', ''))

    if pool:
        # ---- New schema ----
        secondary_classes = pool
        secondary_hotkeys = _align(
            [h.strip() for h in str(d.get('secondary_hotkeys', '')).split(',')],
            len(pool), '')
        secondary_colors = _align(_parse_colors(d.get('secondary_colors', '')),
                                  len(pool), DEFAULT_SECONDARY_COLOR)
        secondary_map = parse_secondary_map(d.get('secondary_map', ''))
        if not secondary_map:
            # No explicit map: allow every pool secondary on every primary.
            secondary_map = {p: list(secondary_classes) for p in primary_classes}
    else:
        # ---- Legacy fallback ----
        legacy_static = parse_class_list(d.get('secondary_static_classes', ''))
        legacy_motion = parse_class_list(d.get('secondary_motion_classes', ''))
        static_cols = _parse_colors(d.get('secondary_static_colors', ''))
        motion_cols = _parse_colors(d.get('secondary_motion_colors', ''))
        static_keys = [h.strip() for h in str(d.get('secondary_static_hotkeys', '')).split(',')]
        motion_keys = [h.strip() for h in str(d.get('secondary_motion_hotkeys', '')).split(',')]

        # Build a pool that is the union (static first, then motion-only entries).
        secondary_classes = list(legacy_static)
        secondary_colors = _align(static_cols, len(legacy_static), DEFAULT_SECONDARY_COLOR)
        secondary_hotkeys = _align(static_keys, len(legacy_static), '')
        for i, name in enumerate(legacy_motion):
            if name in secondary_classes:
                continue
            secondary_classes.append(name)
            secondary_colors.append(motion_cols[i] if i < len(motion_cols) else DEFAULT_SECONDARY_COLOR)
            secondary_hotkeys.append(motion_keys[i] if i < len(motion_keys) else '')

        ignore = set(parse_class_list(d.get('ignore_secondary', '')))
        secondary_map = {}
        for p in primary_classes:
            if p in ignore:
                continue
            allowed = legacy_static if p in static_set else legacy_motion
            if allowed:
                secondary_map[p] = list(allowed)

    # Build per-primary allowed index lists (indices into the shared pool).
    name_to_idx = {name: i for i, name in enumerate(secondary_classes)}
    allowed_secondary_idx = []
    for p in primary_classes:
        idxs = [name_to_idx[s] for s in secondary_map.get(p, []) if s in name_to_idx]
        allowed_secondary_idx.append(idxs)

    hierarchical_mode = bool(secondary_classes) and any(allowed_secondary_idx)

    return {
        'secondary_classes': secondary_classes,
        'secondary_hotkeys': secondary_hotkeys,
        'secondary_colors': secondary_colors,
        'secondary_map': secondary_map,
        'allowed_secondary_idx': allowed_secondary_idx,
        'hierarchical_mode': hierarchical_mode,
    }
