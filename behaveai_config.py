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

import os

DEFAULT_SECONDARY_COLOR = (200, 200, 200)  # BGR, used when a colour is missing

# Reserved class name for the "no secondary" negative in the secondary classifier.
# Boxes of a secondary-eligible primary that carry no real secondary are cropped into
# annot_<stream>_crop/__none__/ so the classifier learns an explicit "none" class and is
# no longer forced to emit a secondary. Cannot collide with a real secondary name.
NONE_LABEL = "__none__"

# Default species when a project predates the species feature (species_list absent).
# Scientific name, matching the convention used for every entry in species_list.
DEFAULT_SPECIES = "Equus caballus"

# INI keys for the three project directories, newest key first. The `*_folder`
# spellings are legacy and only ever appear in old settings files.
PROJECT_DIR_KEYS = {
    'clips':  ('clips_dir', 'clips_folder'),
    'input':  ('input_dir', 'input_folder'),
    'output': ('output_dir', 'output_folder'),
}


def resolve_project_path(value, fallback, project_dir):
    """Resolve one directory setting: absolute as given, relative to the project.

    Empty or missing falls back to `fallback` (pass '' to mean "this source is
    disabled", as the annotation tool does for input_dir).
    """
    if value is None or str(value).strip() == '':
        value = fallback
    value = str(value).strip()
    if value == '':
        return ''
    if os.path.isabs(value):
        return os.path.normpath(value)
    return os.path.normpath(os.path.join(project_dir, value))


def resolve_project_dir(config, project_dir, which, fallback=None):
    """Resolve 'clips' | 'input' | 'output' for a project from its INI.

    The single definition of the rule every stage must follow, because they used
    to each re-implement it and drift: BehaveAI_classify_track resolved the INI
    values and then overwrote them with a hardcoded <project_dir>/input just
    before use (a project pointing elsewhere silently processed nothing), the
    complex-candidate helper rebuilt input/clips from the output directory's
    parent, and the variants reading the key with a plain .get() sent output to
    the project root when the key existed but was empty. `config` accepts either
    a ConfigParser or its [DEFAULT] section.
    """
    section = config['DEFAULT'] if hasattr(config, 'sections') else config
    raw = ''
    for key in PROJECT_DIR_KEYS[which]:
        raw = section.get(key, '') or ''
        if str(raw).strip():
            break
    return resolve_project_path(raw, which if fallback is None else fallback, project_dir)


def resolve_project_dirs(config, project_dir):
    """(clips, input, output) for a project, resolved from its INI."""
    return tuple(resolve_project_dir(config, project_dir, w)
                 for w in ('clips', 'input', 'output'))


def species_slug(name):
    """Filesystem-safe version of a scientific species name (space -> underscore).
    Used only for folder/file naming; the raw name (with space) is what's stored in
    the INI, shown in the GUI/annotation tool and written to CSVs."""
    return str(name or '').strip().replace(' ', '_')


def get_species_list(config):
    """Parse `species_list` from [DEFAULT]; defaults to a single species so projects
    that predate this feature keep working unchanged."""
    raw = config['DEFAULT'].get('species_list', '')
    species = parse_class_list(raw)
    return species if species else [DEFAULT_SPECIES]


def species_key(base_key, species, species_list):
    """Resolve the INI key to use for `base_key` under `species`.

    The first species in species_list keeps the bare, legacy key name (so an
    existing single-species project's ini needs zero migration). Every other
    species gets a `<base_key>__<slug>` key."""
    if not species_list or species == species_list[0]:
        return base_key
    return f"{base_key}__{species_slug(species)}"


def species_folder(base_name, species, species_list):
    """Resolve the folder name to use for `base_name` under `species` (same
    first-species-is-bare rule as species_key, applied to directory names)."""
    if not species_list or species == species_list[0]:
        return base_name
    return f"{base_name}__{species_slug(species)}"


def load_ethogram_for_species(config, species, species_list):
    """Load primary static/motion classes+hotkeys+colors and the secondary pool
    for a given species, using species-scoped keys (species_key). Reuses
    parse_class_list/_parse_colors/load_secondary_config as-is."""
    d = config['DEFAULT']

    def _classes(base):
        return parse_class_list(d.get(species_key(base, species, species_list), ''))

    def _hotkeys(base, n):
        raw = d.get(species_key(f'{base}_hotkeys', species, species_list), '')
        return _align([h.strip() for h in str(raw).split(',')], n, '')

    def _colors(base, n):
        raw = d.get(species_key(f'{base}_colors', species, species_list), '')
        return _align(_parse_colors(raw), n, DEFAULT_SECONDARY_COLOR)

    primary_static_classes = _classes('primary_static_classes')
    primary_motion_classes = _classes('primary_motion_classes')
    primary_static_hotkeys = _hotkeys('primary_static', len(primary_static_classes))
    primary_motion_hotkeys = _hotkeys('primary_motion', len(primary_motion_classes))
    primary_static_colors = _colors('primary_static', len(primary_static_classes))
    primary_motion_colors = _colors('primary_motion', len(primary_motion_classes))

    # load_secondary_config reads secondary_* keys straight off config['DEFAULT'];
    # build a scoped shim so it picks up this species' secondary_* keys.
    secondary_keys = ('secondary_classes', 'secondary_hotkeys', 'secondary_colors',
                      'secondary_map', 'secondary_static_classes', 'secondary_motion_classes',
                      'secondary_static_colors', 'secondary_motion_colors',
                      'secondary_static_hotkeys', 'secondary_motion_hotkeys', 'ignore_secondary')
    shim = {k: d.get(species_key(k, species, species_list), '') for k in secondary_keys}
    secondary_cfg = load_secondary_config({'DEFAULT': shim}, primary_static_classes, primary_motion_classes)

    return {
        'primary_static_classes': primary_static_classes,
        'primary_motion_classes': primary_motion_classes,
        'primary_static_hotkeys': primary_static_hotkeys,
        'primary_motion_hotkeys': primary_motion_hotkeys,
        'primary_static_colors': primary_static_colors,
        'primary_motion_colors': primary_motion_colors,
        **secondary_cfg,
    }


def load_age_classes(config, species, species_list):
    """Load the age classes+hotkeys+colors defined for a given species (same
    scoping/pattern as the ethogram groups)."""
    d = config['DEFAULT']
    classes = parse_class_list(d.get(species_key('age_classes', species, species_list), ''))
    hotkeys = _align(
        [h.strip() for h in str(d.get(species_key('age_hotkeys', species, species_list), '')).split(',')],
        len(classes), '')
    colors = _align(_parse_colors(d.get(species_key('age_colors', species, species_list), '')),
                     len(classes), DEFAULT_SECONDARY_COLOR)
    return {'age_classes': classes, 'age_hotkeys': hotkeys, 'age_colors': colors}


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
