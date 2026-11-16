r"""
File Name: geometry.py
Author: Connor Sullivan
Date: 2026.08.010
Revision: v12

============================================================
PURPOSE
============================================================
Defines the DeviceArray class: the geometry generator for a periodic
artificial spin ice (ASI) device array, and the class the rest of this
pipeline's superposition machinery (geometry.py's own caller,
superposition_operations.py, mumax3_parent_superposition.py) is built
around. Given a per-sublattice device size, spacing, rotation, and
translation, DeviceArray builds every device's (x, y) center, geometry
angle, and magnetization angle, re-buildable on demand whenever any of
those parameters change.

Also contains compute_magnetization_angles, a module-level function with
the same magnetization-pattern computation DeviceArray.set_magnetization_
pattern uses internally, callable on its own (for example, to prototype a
new pattern, or to compute angles for devices that do not come from a
DeviceArray at all) without needing a DeviceArray instance.

============================================================
WHAT THIS CLASS CAN AND CANNOT DO (read this before building an array)
============================================================
DeviceArray can build any PERIODIC array: any number of sublattices, each
with its own device size, lattice spacing, lattice rotation, device
(island) rotation, and translation offset, combined into one array and
automatically centered on (0, 0). This covers square ASI, pinwheel ASI
(square ASI with the lattice rotated 45 degrees), kagome ASI (three
directionally-unique sublattices), and any other periodic geometry
expressible as "N sublattices, each a rotated/translated rectangular grid."

The tradeoff for that generality is that the array is defined ENTIRELY by
its per-sublattice setup parameters, regenerated from scratch by
update_array() every time ANY of them changes (directly via
update_parameters(), or indirectly via set_magnetization_pattern(),
sublattice_dims changes, etc.) -- there is no notion of "nudge this one
sublattice's rotation angle slightly and keep everything else exactly
where it was." Three specific consequences of this are worth understanding
up front, since they are the most common source of "the array does not
look like what I expected" surprises:

1. CHANGING A SUBLATTICE'S geometry angle (rotation_angles), device
   (island) rotation (device_rotations), or translation (translations)
   moves that sublattice's devices RELATIVE TO THE ARRAY'S ORIGIN, not in
   place. Every device position is rebuilt from that sublattice's own
   local lattice mesh, rotated about that mesh's OWN center, translated,
   and only then does the WHOLE array (every sublattice combined) get
   re-centered on (0, 0) by its new overall center of mass. Changing one
   sublattice's rotation or translation shifts where that sublattice's
   devices land relative to every other sublattice, which shifts the
   array's overall center of mass, which shifts EVERY device's final
   position (including devices in sublattices you did not touch) once the
   whole array is re-centered. A device index is not a stable handle on
   "this physical device" across such a change -- after update_array()
   reruns, index 8 may correspond to a different lattice site than it did
   before, and no device is guaranteed to have kept its old (x, y)
   position, even in an untouched sublattice.

2. CHANGING THE ARRAY SIZE (sublattice_dims, or a termination setting that
   changes how many devices a sublattice actually gets built with) changes
   the same center-of-mass-recentering chain described above, for the same
   reason: a different device count changes each sublattice's own local
   mesh center, which changes that sublattice's placement, which changes
   the overall center of mass every device gets re-centered against.
   Growing or shrinking the array is not equivalent to adding or removing
   devices at the edges of an otherwise-unchanged layout -- the entire
   array shifts.

3. Because of (1) and (2), device indices and absolute (x, y) positions
   should be treated as valid only for ONE specific, fully-built
   configuration. Do not cache a device's index or position and assume it
   still refers to the same physical device after ANY call that
   regenerates the array (update_parameters(); set_magnetization_pattern()
   changing sublattice/array-level structure is fine since it does not
   call update_array() -- see that method's own documentation -- but
   sublattice_dims, rotation_angles, device_rotations, translations,
   nn_x/nn_y, device_length/device_width, or sublattice_termination changes
   all do). Re-read device_centers/device_data/get_device_array() fresh
   after any such change rather than relying on previously read values.

============================================================
LIMITATIONS
============================================================
- This class was affected by two real bugs, both fixed in this revision
  (see the parent revision if either fix needs to be reviewed against the
  original behavior):
    1. set_device_property's column mapping for the "angle" and
       "sublattice" keys was swapped relative to every other part of this
       class's own device_array column convention (column 2 is always
       sublattice ID, column 3 is always geometry angle -- see
       update_array/_refresh_data/get_device_array, all consistent with
       each other). set_device_property's col_map had these reversed, so
       calling set_device_property(index, "angle", value) actually
       overwrote that device's SUBLATTICE ID with `value`, leaving its
       real angle untouched -- and set_device_property(index, "sublattice",
       value) actually overwrote its angle instead. This affected this
       file's own bottom-of-file worked example
       (array.set_device_property(8, "angle", 25), etc.), which had been
       silently corrupting sublattice IDs rather than rotating devices.
    2. update_from_array's fallback branch (used only when no device_data
       is given AND the instance does not already have a device_data
       attribute -- a narrow case that does not occur during ordinary use,
       since __init__ always populates device_data first) read a
       caller-supplied array's angle/sublattice columns in the OPPOSITE
       order from this class's own convention (row[2] treated as angle,
       row[3] as sublattice, instead of the correct row[2]=sublattice,
       row[3]=angle). Fixed to match the convention used everywhere else.
    3. compute_magnetization_angles/_lattice_coords needs the array's
       RAW center of mass (its center of mass BEFORE update_array's
       whole-array re-centering step) to undo that re-centering and
       recover each device's true integer lattice coordinates -- the
       checkerboard parity that 'opposite'/'row_opposite'/'col_opposite'/
       magnetic_sublattice="stripe" all key off of. set_magnetization_
       pattern instead passed self.device_centers -- already re-centered
       on (0, 0) by that same update_array step -- so np.mean(device_
       centers) inside compute_magnetization_angles recovered ~(0, 0)
       instead of the real offset, on every call. This is silently wrong
       for any array whose true pre-centering center of mass is not
       already (0, 0) itself (translations, boundary-fill, uneven
       termination, kagome, etc. -- most arrays besides a single centered
       symmetric sublattice), shifting every recovered lattice coordinate
       by that missed offset and corrupting checkerboard parity, generally
       without raising any error. Fixed by having update_array persist
       that raw offset as self._raw_center_of_mass, and set_magnetization_
       pattern now passes it into compute_magnetization_angles's new
       `center_of_mass` parameter explicitly instead of leaving that
       function to (mis)recover it from already-centered data.
- check_for_overlaps requires the `shapely` package, imported inside that
  method (not at module level) specifically so the rest of this class
  remains usable in an environment without shapely installed, at the cost
  of that one method failing at call time rather than at import time if
  shapely is missing.
- sublattice_termination_mode="rotated_boundary_fill" assumes the target
  footprint is well-approximated by a rectangular box -- exactly right for
  square/pinwheel-style arrays. For a hexagonal footprint (for example,
  kagome), the rectangular box is only an outer bound, and this mode will
  overfill the corners -- see _apply_boundary_fill's own documentation, and
  remove_devices() for the recommended way to hand-trim such a footprint.
"""

import numpy as np  # array operations used throughout this file
import matplotlib.pyplot as plt  # layout plotting
import matplotlib.patches as patches  # device ellipse rendering


# --------------------------
# ---- Standalone magnetization-pattern computation (usable without a DeviceArray) ----
def compute_magnetization_angles(
    device_data,  # dict {idx: {"x":..., "y":..., "sublattice":..., ...}} -- for example, DeviceArray.device_data, or any dict with the same shape
    device_centers,  # Nx2 array of ALL device centers, used as a fallback ONLY (see center_of_mass below) -- kept for backward-compatible standalone calls
    translations,  # dict {sublattice_id: [tx, ty]}, in the SAME units as device_centers (meters)
    nn_x, nn_y,  # nearest-neighbor spacing, in the SAME units as device_centers (meters); a per-sublattice dict {sublattice_id: spacing} matching DeviceArray.nn_x/.nn_y, or a single scalar applied to every sublattice
    mode="identical",  # 'identical' | 'opposite' | 'row_opposite' | 'col_opposite' | 'random'
    base_angle=0,  # base magnetization angle, degrees (0 = +x, 90 = +y, 180 = -x, 270 = -y)
    sublattice=None,  # optional sublattice filter -- only devices in this sublattice are included in the returned dict
    magnetic_sublattice=None,  # set to "stripe" for a spin-ice geometry whose magnetic sublattice differs from its geometric sublattice
    parity_term_fix=1,  # 0 or 1, shifts which lattice sites count as "even"/"odd" for the checkerboard-style modes
    random_seed=None,  # optional random seed, for reproducible output when mode="random"
    center_of_mass=None,  # (com_x, com_y), the array's RAW center of mass BEFORE the whole-array re-centering update_array() applies (see DeviceArray.update_array, where this value is computed and persisted as self._raw_center_of_mass). This is what _lattice_coords actually needs to undo the re-centering step -- device_centers passed in from a live DeviceArray is already POST-centering (mean essentially 0 by construction), so recomputing "center of mass" from it (the old, buggy fallback below) recovers ~(0, 0) instead of the real offset, silently shifting every recovered lattice coordinate by the true (generally nonzero) center-of-mass whenever a sublattice's own placement is not already symmetric about the array origin (translations, boundary-fill, uneven termination, kagome, etc.). Always supply this explicitly when calling from a DeviceArray instance (set_magnetization_pattern does). Only omitted for a fully standalone call with device data that never went through DeviceArray's own centering step.
    verbose=False,  # print the number of devices computed when True
):
    """
    GOAL:
        Compute a magnetization angle for a set of devices, without
        mutating anything -- the same computation
        DeviceArray.set_magnetization_pattern uses internally, factored out
        to module level so it can be called on its own (to prototype a new
        pattern, or to compute angles for devices that do not come from a
        DeviceArray at all).

    HOW IT TALKS TO OTHER FILES:
        Called by DeviceArray.set_magnetization_pattern (this file), which
        supplies the instance's own geometry as inputs and writes the
        result back into self.device_data. Not called by any file outside
        this one directly, but designed to be usable that way.

    PARAMETERS:
        See the inline comments on each parameter above.

        MODES:
            identical:    every device gets `base_angle`.
            opposite:     checkerboard alternation between `base_angle` and
                `base_angle + 180`, by the parity of each device's own
                lattice coordinates (or by magnetic_sublattice="stripe"'s
                own parity map, if that is set).
            row_opposite:  same alternation, but by row (y lattice
                coordinate) parity only.
            col_opposite:  same alternation, but by column (x lattice
                coordinate) parity only.
            random:        each device independently, randomly gets either
                `base_angle` or `base_angle + 180`.

    RETURNS:
        Dict[int, float]: {device_index: magnetization_angle_degrees}, one
        entry for every device considered (respecting the `sublattice`
        filter -- a device outside the filter is simply absent from the
        returned dict, not present with an unchanged value).

    LIMITATIONS:
        Raises ValueError if `mode` is not one of the five supported
        values.
        If `center_of_mass` is not supplied, this falls back to
        np.mean(device_centers, axis=0) -- correct ONLY if the supplied
        device_centers has not already been re-centered on (0, 0) (see
        `center_of_mass`'s own parameter documentation above). Any call
        using a live DeviceArray's own device_centers/device_data (which
        set_magnetization_pattern always does) MUST supply
        `center_of_mass` explicitly, or checkerboard-style modes
        ('opposite', 'row_opposite', 'col_opposite', magnetic_sublattice=
        "stripe") will silently compute the wrong parity for any array
        whose true pre-centering center of mass is not already (0, 0).
    """
    if center_of_mass is not None:  # the caller supplied the array's real pre-centering offset (DeviceArray.set_magnetization_pattern always does) -- use it directly rather than re-deriving (and getting ~(0, 0)) from already-centered device_centers
        com_x, com_y = center_of_mass
    else:  # standalone-call fallback: only correct if device_centers has NOT already been re-centered on (0, 0) -- see this parameter's own documentation above
        com_x = np.mean(device_centers[:, 0])  # array-wide center of mass, x
        com_y = np.mean(device_centers[:, 1])  # array-wide center of mass, y

    def _spacing(nn, sub):
        # This sublattice's own spacing: a per-sublattice dict lookup, or the flat
        # scalar itself if nn_x/nn_y were given as a single shared value.
        return nn[sub] if isinstance(nn, dict) else nn

    def _lattice_coords(dev):
        # Recovers this device's INTEGER lattice coordinates (which row/column of its
        # own sublattice's mesh it came from) by undoing centering and translation,
        # then dividing by that sublattice's own spacing -- needed to determine
        # checkerboard parity for the "opposite"-style modes below.
        sub = dev["sublattice"]  # this device's sublattice ID
        tx, ty = translations.get(sub, [0, 0])  # this sublattice's translation offset
        x = dev["x"] + com_x - tx  # undo the array-wide centering, then undo this sublattice's own translation
        y = dev["y"] + com_y - ty
        nx, ny = _spacing(nn_x, sub), _spacing(nn_y, sub)  # this sublattice's own spacing -- may differ between sublattices (for example, kagome)
        ix = int(round((x - nx / 2) / nx))  # integer lattice column
        iy = int(round((y - ny / 2) / ny))  # integer lattice row
        return ix, iy

    mag_sub_map = {}  # device index -> magnetic sublattice (0 or 1), only populated when magnetic_sublattice="stripe" is in use
    if magnetic_sublattice == "stripe" and sublattice is None:
        for idx, dev in device_data.items():  # precompute every device's magnetic-sublattice parity once, up front
            ix, iy = _lattice_coords(dev)
            mag_sub_map[idx] = (ix + iy) % 2  # stripe pattern: magnetic sublattice determined by checkerboard parity of the underlying (untranslated, uncentered) lattice coordinates

    if random_seed is not None:  # seed the random generator for reproducible "random" mode output
        np.random.seed(random_seed)

    result = {}  # accumulated across every considered device below
    for idx, dev in device_data.items():  # compute one device's magnetization angle at a time
        if sublattice is not None and dev["sublattice"] != sublattice:  # this device is outside the requested sublattice filter
            continue
        ix, iy = _lattice_coords(dev)  # this device's own integer lattice coordinates
        effective_sub = mag_sub_map.get(idx) if magnetic_sublattice == "stripe" else None  # this device's precomputed magnetic-sublattice parity, if applicable

        if mode == "identical":
            m_angle = base_angle
        elif mode == "opposite":
            parity = effective_sub if magnetic_sublattice == "stripe" else (ix + iy + parity_term_fix) % 2  # checkerboard parity: magnetic-sublattice-based, or lattice-coordinate-based
            m_angle = base_angle if parity == 0 else (base_angle + 180) % 360
        elif mode == "row_opposite":
            parity = effective_sub if magnetic_sublattice == "stripe" else (iy + parity_term_fix) % 2  # row-only parity
            m_angle = base_angle if parity == 0 else (base_angle + 180) % 360
        elif mode == "col_opposite":
            parity = effective_sub if magnetic_sublattice == "stripe" else (ix + parity_term_fix) % 2  # column-only parity
            m_angle = base_angle if parity == 0 else (base_angle + 180) % 360
        elif mode == "random":
            m_angle = np.random.choice([base_angle, (base_angle + 180) % 360])  # independent random choice per device
        else:
            raise ValueError("mode must be: identical, opposite, row_opposite, col_opposite, random")
        result[idx] = m_angle
    if verbose:
        print(f"[DEBUG] compute_magnetization_angles: computed {len(result)} device angle(s), mode='{mode}'.")
    return result


# --------------------------
# ---- Periodic ASI device array geometry generator (see module-level PURPOSE / LIMITATIONS above) ----
class DeviceArray:
    # --------------------------
    # ---- Initialization ----
    def __init__(self, device_length, device_width, nn_x, nn_y, sublattice_dims,
                 rotation_angles=None, translations=None, device_rotations=None,
                 sublattice_termination=None, sublattice_termination_mode=None, verbose=False):
        """
        GOAL:
            Build a new periodic device array from per-sublattice setup
            parameters, generating every device's position and angle via
            update_array() (see that method for the actual geometry
            construction).

        HOW IT TALKS TO OTHER FILES:
            Called by mumax3_parent_superposition.py to build the ASI
            geometry a superposition run is based on. Calls
            _normalize_sublattice_param (this class) to resolve each
            per-sublattice parameter, and update_array (this class) at the
            end to perform the actual construction.

        PARAMETERS:
            device_length, device_width: device size, in nm. A single
                scalar (applied to every sublattice) or a dict
                {sublattice_id: value} -- every sublattice in
                `sublattice_dims` must end up with a value; there is no
                default, unlike an angle defaulting to 0, since there is no
                sensible universal default for a physical size.
            nn_x, nn_y: nearest-neighbor lattice spacing, in nm. Same
                scalar-or-dict convention and same no-default requirement
                as device_length/device_width.
            sublattice_dims (Dict[int, Tuple[int, int]]): {sublattice_id:
                (num_x, num_y)} -- the SETUP/REQUESTED size of each
                sublattice's rectangular grid, before any termination is
                applied (see sublattice_dims_xy, populated by
                update_array(), for the ACTUAL built size). Sublattice IDs
                may be any hashable value and need not be contiguous
                (for example, kagome's three directionally-unique
                sublattices).
            rotation_angles: per-sublattice LATTICE rotation, in degrees --
                rotates where that sublattice's devices sit, not their own
                orientation (see device_rotations for that). A single
                scalar, a dict {sublattice_id: angle}, or None (defaults
                every sublattice to 0).
            translations: per-sublattice offset, in nm, applied after
                lattice rotation and before the whole array is re-centered.
                A dict {sublattice_id: [tx, ty]}, or None (defaults every
                sublattice to [0, 0]).
            device_rotations: per-sublattice DEVICE (island) orientation, in
                degrees -- independent of rotation_angles, so a
                sublattice's lattice positions and its devices' own
                orientation can be set separately. Same scalar/dict/None
                convention as rotation_angles.
            sublattice_termination: per-sublattice edge-extension spec --
                {sublattice_id: {"pos_x": N, "neg_x": N, "pos_y": N,
                "neg_y": N}} adds N extra rows/columns of devices in that
                direction, in that sublattice's own pre-rotation local
                lattice frame. The older boolean form
                {"x": True/False, "y": True/False, "neg_x": ..., "neg_y":
                ...} (True meaning "add exactly 1") is still accepted for
                backward compatibility, with a deprecation warning.
            sublattice_termination_mode: per-sublattice choice of HOW
                termination is applied -- "fixed_row_col" (the default) or
                "rotated_boundary_fill" (see the class-level module
                docstring's LIMITATIONS section, and _apply_boundary_fill's
                own documentation, for the difference). A single scalar, a
                dict {sublattice_id: mode}, or None (defaults every
                sublattice to "fixed_row_col").
            verbose (bool): forwarded to update_array and the methods it
                calls, so their own progress messages appear when True.

        RETURNS:
            None. Populates every DeviceArray attribute via update_array().

        LIMITATIONS:
            See this class's module-level PURPOSE/LIMITATIONS sections
            regarding what changing any of these parameters after
            construction (via update_parameters()) does to already-built
            device positions and indices.
        """
        self.sublattice_dims = sublattice_dims  # {sublattice_id: (num_x, num_y)} -- the SETUP/REQUEST size (see sublattice_dims_xy for the actual built size after termination)
        self.sublattices = len(sublattice_dims)  # number of sublattices (kept for backward compatibility; every iteration in this class goes over sublattice_dims.keys() directly, so this supports arbitrary/non-contiguous sublattice IDs)

        # nn_x/nn_y/device_length/device_width are per-sublattice dicts, so different
        # sublattices can have different spacings/sizes (for example, kagome's
        # directionally-unique sublattices genuinely differ). A single scalar still
        # works and broadcasts to every sublattice. Every sublattice in
        # sublattice_dims MUST end up with a value here -- there is no sensible
        # universal default for a physical spacing/size (unlike an angle defaulting
        # to 0), so a dict missing an entry raises rather than silently guessing.
        self.nn_x = {k: v * 1e-9 for k, v in self._normalize_sublattice_param(nn_x, default=None, name="nn_x").items()}  # nm -> m
        self.nn_y = {k: v * 1e-9 for k, v in self._normalize_sublattice_param(nn_y, default=None, name="nn_y").items()}  # nm -> m
        self.device_length = {k: v * 1e-9 for k, v in self._normalize_sublattice_param(device_length, default=None, name="device_length").items()}  # nm -> m
        self.device_width = {k: v * 1e-9 for k, v in self._normalize_sublattice_param(device_width, default=None, name="device_width").items()}  # nm -> m
        for _d, _name in [(self.nn_x, "nn_x"), (self.nn_y, "nn_y"), (self.device_length, "device_length"), (self.device_width, "device_width")]:  # verify every sublattice got a value for each of these four required parameters
            _missing = set(self.sublattice_dims.keys()) - set(_d.keys())
            if _missing:
                raise ValueError(f"{_name} is missing required sublattice ID(s) {sorted(_missing)}; provide a value for every sublattice in sublattice_dims (or a single scalar to apply to all).")

        # Translations are applied before centering, so they must be in the same
        # units as the device centers (meters) internally.
        if translations:
            self.translations = {k: [v[0] * 1e-9, v[1] * 1e-9] for k, v in translations.items()}  # nm -> m
        else:
            self.translations = {sid: [0.0, 0.0] for sid in self.sublattice_dims}  # default: no translation for every sublattice

        # Every per-sublattice parameter is a dict explicitly keyed by sublattice ID
        # (matching sublattice_dims/translations), rather than a positional list --
        # this removes the old ambiguity where a list's position had to silently line
        # up with however sublattice_dims happened to iterate.
        self.rotation_angles = self._normalize_sublattice_param(rotation_angles, default=0, name="rotation_angles")  # per-sublattice LATTICE rotation, degrees
        self.device_rotations = self._normalize_sublattice_param(device_rotations, default=0, name="device_rotations")  # per-sublattice DEVICE (island) orientation, degrees -- independent of rotation_angles
        self.sublattice_termination = sublattice_termination if sublattice_termination else {}  # {sublattice_id: {edge: count}} -- see _normalize_termination for the accepted forms

        self.sublattice_termination_mode = self._normalize_sublattice_param(sublattice_termination_mode, default="fixed_row_col", name="sublattice_termination_mode")  # per-sublattice choice of HOW termination is applied
        invalid_modes = {m for m in self.sublattice_termination_mode.values() if m not in ("fixed_row_col", "rotated_boundary_fill")}
        if invalid_modes:  # guard against an unrecognized termination mode
            raise ValueError(f"sublattice_termination_mode must be 'fixed_row_col' or 'rotated_boundary_fill', got {invalid_modes}")

        self.sublattice_dims_xy = {}  # populated by update_array(): the ACTUAL built (num_x, num_y) per sublattice after termination is applied -- distinct from sublattice_dims, which is just the original setup request
        self.update_array(verbose=verbose)  # generate device centers/angles from the parameters set above

    # --------------------------
    # ---- Normalize a per-sublattice parameter into a dict keyed by sublattice ID ----
    def _normalize_sublattice_param(self, value, default, name):
        """
        GOAL:
            Normalize any of the accepted input shapes for a per-sublattice
            parameter into the one canonical shape every other method in
            this class expects: a plain dict {sublattice_id: value}.

        HOW IT TALKS TO OTHER FILES:
            Called by __init__ and update_parameters (this class), for
            every per-sublattice parameter (nn_x, nn_y, device_length,
            device_width, rotation_angles, device_rotations,
            sublattice_termination_mode).

        PARAMETERS:
            value: one of:
                - None -> `default` for every sublattice ID in
                  self.sublattice_dims.
                - a single scalar (str/int/float, not a dict/list/tuple)
                  -> broadcast to every sublattice ID.
                - dict {sublattice_id: value} -> used as-is (an ID present
                  in the dict but not in self.sublattice_dims raises; an ID
                  in self.sublattice_dims but missing from the dict is left
                  for the caller to detect, since not every parameter has a
                  safe default to fall back to -- see __init__'s explicit
                  post-hoc check for nn_x/nn_y/device_length/device_width).
                - list/tuple [value, value, ...] -> DEPRECATED positional
                  form, mapped onto sorted(self.sublattice_dims) in order,
                  with a printed deprecation warning.
            default: the value to use for a sublattice ID when `value` is
                None. May be None itself (see PARAMETERS above) if this
                particular parameter has no safe default.
            name (str): this parameter's name, used only in error/warning
                messages.

        RETURNS:
            Dict[Any, Any]: {sublattice_id: value}, covering every ID
            `value` specified (which may be fewer than every ID in
            self.sublattice_dims, for the dict input case -- see above).

        LIMITATIONS:
            Raises ValueError if a dict's keys are not a subset of
            self.sublattice_dims' keys, or if a list/tuple's length does
            not match the number of sublattices. Raises TypeError if
            `value` is not one of the accepted shapes.
        """
        if value is None:  # no value given at all: use the default for every sublattice
            return {sid: default for sid in self.sublattice_dims}
        if isinstance(value, dict):  # explicit per-sublattice dict: validate its keys, then use as-is
            unknown = set(value.keys()) - set(self.sublattice_dims.keys())
            if unknown:  # catch typos/mismatches early instead of silently ignoring them
                raise ValueError(f"{name} references unknown sublattice ID(s) {sorted(unknown)} not present in sublattice_dims")
            return dict(value)
        if isinstance(value, (list, tuple)):  # deprecated positional form
            sorted_ids = sorted(self.sublattice_dims.keys())
            if len(value) != len(sorted_ids):  # the list must cover exactly every sublattice
                raise ValueError(f"{name} given as a list of length {len(value)} but there are {len(sorted_ids)} sublattices; pass a dict {{sublattice_id: value}} instead to avoid ambiguity")
            print(f"[WARNING] DeviceArray._normalize_sublattice_param: '{name}' was given as a list; this positional "
                  f"form is deprecated. Mapping it onto sublattice IDs {sorted_ids} in that order. Prefer the "
                  f"explicit dict form, e.g. {name}={{{', '.join(f'{sid}: {v}' for sid, v in zip(sorted_ids, value))}}}.")
            return {sid: v for sid, v in zip(sorted_ids, value)}
        if isinstance(value, (str, int, float)):  # a single scalar: broadcast to every sublattice
            return {sid: value for sid in self.sublattice_dims}
        raise TypeError(f"{name} must be a dict, list, tuple, scalar, or None, got {type(value)}")

    # --------------------------
    # ---- Normalize one sublattice's termination spec into per-edge integer counts ----
    @staticmethod
    def _normalize_termination(term, sub_id):
        """
        GOAL:
            Normalize one sublattice's termination specification (however
            it was given) into a plain (pos_x, neg_x, pos_y, neg_y) tuple
            of integer extra-row/column counts.

        HOW IT TALKS TO OTHER FILES:
            Called by update_array and _compute_target_boundaries (this
            class), once per sublattice, to resolve that sublattice's own
            termination entry from self.sublattice_termination.

        PARAMETERS:
            term (dict): one sublattice's termination spec, in either
                accepted form:
                - NEW (per-edge counts): {"pos_x": 2, "neg_x": 0, "pos_y":
                  1, "neg_y": 0} -- add N extra rows/columns of devices in
                  that direction, in this sublattice's own pre-rotation
                  local lattice frame.
                - OLD (booleans, deprecated): {"x": True, "y": True,
                  "neg_x": False, "neg_y": False} -- "x"/"y" mean the
                  positive direction, exactly as in the new form; True
                  means "add exactly 1". Both forms may be mixed; any key
                  not present defaults to 0.
            sub_id: this sublattice's ID, used only in error/warning
                messages.

        RETURNS:
            Tuple[int, int, int, int]: (pos_x, neg_x, pos_y, neg_y).

        LIMITATIONS:
            Raises TypeError if a value under a recognized key is neither
            an int nor a bool.
        """
        if not term:  # no termination requested for this sublattice at all
            return 0, 0, 0, 0
        old_keys_present = any(k in term for k in ("x", "y"))  # "x"/"y" only ever existed in the old boolean scheme; pos_x/pos_y are the new names for the same positive-direction edges

        def _count(new_key, old_key):
            if new_key in term:  # prefer the new per-edge-count key if present
                val = term[new_key]
            elif old_key in term:  # fall back to the old boolean key
                val = term[old_key]
            else:  # neither key given: no extra devices in this direction
                val = 0
            if isinstance(val, bool):  # old scheme: True/False -> 1/0
                return int(val)
            if isinstance(val, int):
                return val
            raise TypeError(f"sublattice_termination[{sub_id}][{new_key!r}/{old_key!r}] must be an int (count) or bool, got {type(val)}")

        if old_keys_present:
            print(f"[WARNING] DeviceArray._normalize_termination: sublattice_termination[{sub_id}] uses the "
                  f"deprecated boolean keys 'x'/'y'. Prefer the explicit per-edge count form: "
                  f"{{'pos_x': N, 'neg_x': N, 'pos_y': N, 'neg_y': N}}.")
        pos_x = _count("pos_x", "x")  # extra devices, positive x direction
        neg_x = _count("neg_x", "neg_x")  # extra devices, negative x direction
        pos_y = _count("pos_y", "y")  # extra devices, positive y direction
        neg_y = _count("neg_y", "neg_y")  # extra devices, negative y direction
        return pos_x, neg_x, pos_y, neg_y


    # --------------------------
    # ---- Rounded rotated lattice basis vectors (integer nm) ----
    def _rotated_basis(self, angle_deg, nn_x_m, nn_y_m):
        """
        GOAL:
            Compute the local x/y lattice step vectors for a sublattice
            with spacing (nn_x_m, nn_y_m), rotated by angle_deg, with each
            component ROUNDED TO THE NEAREST INTEGER NANOMETER -- the
            single source of rotated lattice geometry every device
            position in this class is ultimately built from.

        HOW IT TALKS TO OTHER FILES:
            Called by update_array, _compute_target_boundaries, and
            _apply_boundary_fill (all this class) -- every place a
            sublattice's device positions are generated. Not called
            outside this class.

        PARAMETERS:
            angle_deg (float): rotation angle, degrees.
            nn_x_m, nn_y_m (float): this sublattice's own spacing, in
                meters (passed explicitly rather than read from self,
                since nn_x/nn_y are per-sublattice -- the caller supplies
                whichever sublattice's spacing applies here).

        RETURNS:
            Tuple[np.ndarray, np.ndarray]: (u, v), each a 2-element array
            in meters -- the rotated local x-step and y-step lattice
            vectors, respectively.

        LIMITATIONS:
            Rotating nn_x/nn_y by an arbitrary continuous angle generally
            lands on irrational-looking components (for example,
            140*cos(45 degrees) = 98.99494937 nm). Every device position in
            the array is built as
            mesh_center/anchor + (integer step count) * u
                               + (integer step count) * v,
            so rounding u and v ONCE, here, and reusing the exact same
            rounded vectors everywhere positions get generated is what
            keeps every resulting center-to-center spacing an exact whole
            nanometer, self-consistently, rather than rounding each final
            coordinate independently (which would not guarantee uniform
            spacing across the array). For angle_deg equal to 0, 90, 180,
            or 270, this rounding has no effect anyway, since cos/sin are
            already exactly 0 or 1 at those angles.
        """
        nn_x_nm = nn_x_m * 1e9  # meters -> nm, for the rounding step below
        nn_y_nm = nn_y_m * 1e9
        theta = np.radians(angle_deg)
        u_nm = np.round(np.array([np.cos(theta), np.sin(theta)]) * nn_x_nm)  # rotated x-step vector, rounded to the nearest integer nm
        v_nm = np.round(np.array([-np.sin(theta), np.cos(theta)]) * nn_y_nm)  # rotated y-step vector, rounded to the nearest integer nm
        return u_nm * 1e-9, v_nm * 1e-9  # nm -> m

    # --------------------------
    # ---- A device's own half-extent along the global x/y axes, given its orientation ----
    @staticmethod
    def _device_half_extents(angle_deg, length_m, width_m):
        """
        GOAL:
            Compute how far a device's own footprint extends from its
            center along the GLOBAL x and y axes, given its orientation --
            modeled as an ellipse with semi-axes (length/2, width/2)
            rotated by angle_deg (the same shape used elsewhere in this
            class, for example plot_layout and check_for_overlaps).

        HOW IT TALKS TO OTHER FILES:
            Called by _compute_boundaries and _apply_boundary_fill (this
            class) wherever a device's own rotated footprint (rather than
            a flat, orientation-independent margin) is needed to compute
            array boundaries.

        PARAMETERS:
            angle_deg: device orientation, degrees. May be a scalar or an
                array; the computation broadcasts elementwise.
            length_m, width_m: device length/width, in meters. May be
                scalars or arrays the same length as angle_deg (device
                size is per-sublattice, so callers pass whichever
                value(s) apply to each device/point being measured).

        RETURNS:
            Tuple[float or np.ndarray, float or np.ndarray]: (half_x,
            half_y), in the same units as length_m/width_m.

        LIMITATIONS:
            None beyond the ellipse-shape assumption stated above, which
            is shared with every other footprint calculation in this
            class.
        """
        theta = np.radians(angle_deg)
        a, b = length_m / 2.0, width_m / 2.0  # ellipse semi-axes
        half_x = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)  # this device's own half-extent along the global x axis, at this orientation
        half_y = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)  # this device's own half-extent along the global y axis, at this orientation
        return half_x, half_y

    # --------------------------
    # ---- Compute the array's bounding box, using each device's own rotated footprint ----
    def _compute_boundaries(self, centers, angles, length_m, width_m):
        """
        GOAL:
            Compute the array's physical bounding box using each device's
            OWN rotated footprint (its length/width and orientation
            angle), rather than a single fixed device_width/2 margin
            applied identically in x and y for every device regardless of
            orientation -- a device oriented with its long axis along y
            sticks out further in y (and less in x) than one oriented
            along x, so a uniform margin would either clip the former or
            over-pad the latter.

        HOW IT TALKS TO OTHER FILES:
            Called by update_array and _compute_target_boundaries (this
            class), each time the array's self.boundaries needs to be
            (re)computed.

        PARAMETERS:
            centers (np.ndarray): Nx2 array of device centers.
            angles (np.ndarray): N-length array of device orientations,
                degrees -- must be in the same order/length as `centers`.
            length_m, width_m: device length/width, in meters -- scalars
                or N-length arrays (device size is per-sublattice, so
                callers pass whichever value(s) apply to each device).

        RETURNS:
            Dict[str, float]: {"xmin", "xmax", "ymin", "ymax"}.
        """
        half_x, half_y = self._device_half_extents(np.asarray(angles), length_m, width_m)  # every device's own half-extent, vectorized over the whole array at once
        xs, ys = centers[:, 0], centers[:, 1]
        return {
            "xmin": float(np.min(xs - half_x)),
            "xmax": float(np.max(xs + half_x)),
            "ymin": float(np.min(ys - half_y)),
            "ymax": float(np.max(ys + half_y)),
        }

    # --------------------------
    # ---- Look up a sublattice's own device_length/device_width, with a safe fallback ----
    def _length_nm_for(self, sub_id):
        """
        GOAL:
            Look up device_length (nm) for a given sublattice ID, falling
            back to an arbitrary existing sublattice's value if `sub_id`
            is no longer present in self.device_length (for example, stale
            device_data left over after a geometry change that removed
            that sublattice).

        HOW IT TALKS TO OTHER FILES:
            Called throughout this class wherever a device's own default
            length is needed and that device's explicit device_data entry
            might not have one recorded.

        PARAMETERS:
            sub_id: the sublattice ID to look up.

        RETURNS:
            float: device length, in nm.
        """
        return self.device_length.get(int(sub_id), next(iter(self.device_length.values()))) * 1e9

    def _width_nm_for(self, sub_id):
        """
        GOAL:
            Same as _length_nm_for, for device_width instead of
            device_length.

        HOW IT TALKS TO OTHER FILES:
            Same callers as _length_nm_for.

        PARAMETERS:
            sub_id: the sublattice ID to look up.

        RETURNS:
            float: device width, in nm.
        """
        return self.device_width.get(int(sub_id), next(iter(self.device_width.values()))) * 1e9


    # --------------------------
    # ---- Set magnetization directions for devices in this array ----
    def set_magnetization_pattern(
        self,  # DeviceArray instance
        mode="identical",  # 'identical' | 'opposite' | 'row_opposite' | 'col_opposite' | 'random'
        base_angle=0,  # base magnetization angle, degrees (0 = +x, 90 = +y, 180 = -x, 270 = -y)
        sublattice=None,  # optional sublattice filter -- only devices in this sublattice are updated
        magnetic_sublattice=None,  # set to "stripe" for a spin-ice geometry whose magnetic sublattice differs from its geometric sublattice
        parity_term_fix=1,  # 0 or 1, shifts which lattice sites count as "even"/"odd" for the checkerboard-style modes
        random_seed=None,  # optional random seed, for reproducible output when mode="random"
        verbose=False,  # print the number of devices updated when True
    ):
        """
        GOAL:
            Compute and apply a magnetization pattern across this array's
            devices, writing the result into self.device_data (and keeping
            self.device_array/self.device_m_angles in sync).

        HOW IT TALKS TO OTHER FILES:
            Thin wrapper around the module-level compute_magnetization_
            angles function (this file) -- this method's own job is only
            to: (1) supply the instance's current geometry as inputs,
            (2) write the results back into self.device_data, and
            (3) refresh self.device_m_angles so every other part of this
            class (get_device_array, print_summary, plot_layout,
            check_for_overlaps, etc.) sees the update automatically. To
            prototype a new pattern without touching a live DeviceArray,
            call compute_magnetization_angles(...) directly instead.

        PARAMETERS:
            See compute_magnetization_angles' own documentation for what
            each mode does; every parameter here has the identical meaning
            and is forwarded to that function unchanged, aside from
            device_data/device_centers/translations/nn_x/nn_y, which are
            supplied from this instance's own current state.
            verbose (bool): prints the number of devices whose
                magnetization angle was set when True. Forwarded to
                compute_magnetization_angles, so its own debug message
                appears too.

        RETURNS:
            None. Updates self.device_data, self.device_array (column 4,
            the m_angle column), and self.device_m_angles in place.

        LIMITATIONS:
            This method does NOT call update_array() -- it only updates
            magnetization angles, not positions or geometry angles, so it
            does not trigger the center-of-mass recentering described in
            this class's module-level PURPOSE/LIMITATIONS sections. Device
            indices and positions are unaffected by a call to this method.
        """
        angles = compute_magnetization_angles(
            device_data=self.device_data,  # this array's current per-device data
            device_centers=self.device_centers,  # this array's current (centered) device positions -- unused for the com_x/com_y recovery below now that center_of_mass is supplied directly (kept for the function's other standalone-call fallback path)
            translations=self.translations,  # this array's per-sublattice translation offsets
            nn_x=self.nn_x,  # this array's per-sublattice x spacing
            nn_y=self.nn_y,  # this array's per-sublattice y spacing
            center_of_mass=getattr(self, "_raw_center_of_mass", None),  # this array's REAL pre-centering offset, persisted by update_array -- see compute_magnetization_angles' own `center_of_mass` parameter documentation for why this must be supplied (rather than letting it fall back to re-deriving from self.device_centers, which is already centered)
            mode=mode,
            base_angle=base_angle,
            sublattice=sublattice,
            magnetic_sublattice=magnetic_sublattice,
            parity_term_fix=parity_term_fix,
            random_seed=random_seed,
            verbose=verbose,
        )
        for idx, m_angle in angles.items():  # write each computed angle back into device_data and device_array
            self.device_data[idx]["m_angle"] = m_angle  # keep device_data in sync
            self.device_array[idx, 4] = m_angle  # keep device_array's m_angle column (column 4) in sync -- without this line, get_device_array() would keep returning the PRE-pattern value until an unrelated geometry update happened to call _refresh_data
        self.device_m_angles = np.array([self.device_data[i]["m_angle"] for i in range(len(self.device_array))])  # keep device_m_angles in sync with device_data, same as _refresh_data does elsewhere
        if verbose:
            print(f"[INFO] DeviceArray.set_magnetization_pattern: updated {len(angles)} device(s), mode='{mode}'.")

    # --------------------------
    # ---- Set the magnetization angle for one specific device ----
    def set_device_magnetization(self, index, angle, verbose=False):
        """
        GOAL:
            Set the magnetization angle for exactly one device, by index.

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly;
            provided as a convenience for hand-adjusting one device's
            magnetization without going through set_magnetization_pattern.

        PARAMETERS:
            index (int): device index (as seen in device_data /
                get_device_array()).
            angle (float): magnetization angle, degrees -- wrapped into
                [0, 360) before being stored.
            verbose (bool): prints the updated angle when True.

        RETURNS:
            None. Updates self.device_data[index]["m_angle"] in place.

        LIMITATIONS:
            Updates ONLY self.device_data, not self.device_array's m_angle
            column or self.device_m_angles -- unlike
            set_magnetization_pattern, which keeps all three in sync. A
            caller relying on self.device_array or self.device_m_angles
            reflecting this change should call self._refresh_data()
            afterward, or read the angle back from self.device_data
            instead.

        RAISES:
            IndexError if `index` is not a valid device index.
        """
        if index not in self.device_data:  # the requested device must actually exist
            raise IndexError(f"Invalid device index: {index}")
        self.device_data[index]["m_angle"] = angle % 360  # wrap into [0, 360) before storing
        if verbose:
            print(f"[DEBUG] DeviceArray.set_device_magnetization: device {index} m_angle set to {angle % 360}.")

    # --------------------------
    # ---- Update geometry parameters and regenerate the device array ----
    def update_parameters(self, verbose=False, **kwargs):
        """
        GOAL:
            Update any combination of this array's setup parameters (the
            same ones __init__ accepts) and regenerate the entire array
            via update_array().

        HOW IT TALKS TO OTHER FILES:
            Called by mumax3_parent_superposition.py wherever an existing
            DeviceArray's geometry needs to change (for example, between
            worked examples, or when sweeping a parameter). Calls
            _normalize_sublattice_param (this class) for the parameters
            that need it, and update_array (this class) at the end.

        PARAMETERS:
            verbose (bool): forwarded to update_array and the methods it
                calls, so their own progress messages appear when True.
            **kwargs: any subset of __init__'s own parameters (device_length,
                device_width, nn_x, nn_y, sublattice_dims, rotation_angles,
                translations, device_rotations, sublattice_termination,
                sublattice_termination_mode), or any other attribute name
                to set directly. A value of None for any key is skipped
                entirely (treated as "no change requested for this
                parameter"), rather than being applied as an actual None
                value.

        RETURNS:
            None. Regenerates every DeviceArray attribute via
            update_array().

        LIMITATIONS:
            See this class's module-level PURPOSE/LIMITATIONS sections:
            regenerating the array via update_array() re-centers every
            device on the array's new overall center of mass, so device
            indices and positions from before this call should not be
            assumed to still apply afterward -- this is true for ANY
            parameter change made through this method, not only ones that
            seem geometry-related at first glance.

            If 'sublattice_dims' is included in kwargs, it is applied
            FIRST, regardless of kwarg order, so any rotation_angles/
            device_rotations/translations/sublattice_termination_mode also
            passed in the same call are validated against the NEW
            sublattice set rather than the old one, and any sublattice
            newly introduced by this change gets sensible defaults
            (translation [0, 0], rotation 0, termination mode
            "fixed_row_col") before the rest of kwargs is applied.

            After all of kwargs is applied, every sublattice in the
            (possibly just-updated) sublattice_dims is verified to still
            have a value in nn_x, nn_y, device_length, and device_width --
            these four have no safe default (unlike an angle defaulting to
            0), so a ValueError is raised if any sublattice is left
            without one, whether from a plain sublattice_dims expansion
            that forgot to also supply these, or a partial dict update to
            one of these four that silently dropped an existing
            sublattice.
        """
        if 'sublattice_dims' in kwargs and kwargs['sublattice_dims'] is not None:  # apply sublattice_dims first, regardless of kwarg order -- see LIMITATIONS
            kwargs = dict(kwargs)  # avoid mutating the caller's own dict
            val = kwargs.pop('sublattice_dims')
            self.sublattice_dims = val
            self.sublattices = len(val)
            for sid in val:  # give any newly introduced sublattice sensible defaults before the rest of kwargs is applied
                self.translations.setdefault(sid, [0.0, 0.0])
                self.rotation_angles.setdefault(sid, 0)
                self.device_rotations.setdefault(sid, 0)
                self.sublattice_termination_mode.setdefault(sid, "fixed_row_col")

        for key, val in kwargs.items():  # apply every remaining requested parameter change
            if val is None:  # None means "no change requested for this parameter" -- skip it entirely
                continue
            if key in ('device_length', 'device_width', 'nn_x', 'nn_y'):  # per-sublattice dicts, converted from nm to m -- a scalar broadcasts, a dict REPLACES the whole mapping (so a partial dict must still cover every sublattice to keep)
                normalized = self._normalize_sublattice_param(val, default=None, name=key)
                setattr(self, key, {k: v * 1e-9 for k, v in normalized.items()})
            elif key == 'translations':  # applied before centering, so converted from nm to m to match device_centers' own units
                self.translations = {k: [v[0] * 1e-9, v[1] * 1e-9] for k, v in val.items()}
            elif key in ('rotation_angles', 'device_rotations'):  # same dict/list normalization used at __init__ time
                setattr(self, key, self._normalize_sublattice_param(val, default=0, name=key))
            elif key == 'sublattice_termination_mode':  # same scalar/dict convention as rotation_angles/device_rotations, plus a valid-mode check
                normalized = self._normalize_sublattice_param(val, default="fixed_row_col", name=key)
                invalid_modes = {m for m in normalized.values() if m not in ("fixed_row_col", "rotated_boundary_fill")}
                if invalid_modes:
                    raise ValueError(f"sublattice_termination_mode must be 'fixed_row_col' or 'rotated_boundary_fill', got {invalid_modes}")
                self.sublattice_termination_mode = normalized
            else:  # any other attribute: set directly, unchanged
                setattr(self, key, val)

        for _d, _name in [(self.nn_x, "nn_x"), (self.nn_y, "nn_y"), (self.device_length, "device_length"), (self.device_width, "device_width")]:  # verify every sublattice still has a value for each of these four required parameters
            _missing = set(self.sublattice_dims.keys()) - set(_d.keys())
            if _missing:
                raise ValueError(f"{_name} is missing required sublattice ID(s) {sorted(_missing)} after this update; provide a value for every sublattice in sublattice_dims (or a single scalar to apply to all).")
        self.update_array(verbose=verbose)  # regenerate the array to reflect every applied change

    # --------------------------
    # ---- Vectorized per-device (length, width) lookup ----
    def _per_device_lengths_widths(self, device_ids):
        """
        GOAL:
            Build (length_m, width_m) arrays, one entry per device, each
            looked up from that device's OWN sublattice's device_length/
            device_width.

        HOW IT TALKS TO OTHER FILES:
            Called by update_array (this class) wherever a per-device
            length/width array is needed for a boundary computation
            covering multiple sublattices at once.

        PARAMETERS:
            device_ids (Sequence): one sublattice ID per device, in the
                same order as the corresponding device_centers rows.

        RETURNS:
            Tuple[np.ndarray, np.ndarray]: (length_arr, width_arr), each
            the same length as `device_ids`, in meters.
        """
        length_arr = np.array([self.device_length[int(sid)] for sid in device_ids])  # this device's own sublattice's length
        width_arr = np.array([self.device_width[int(sid)] for sid in device_ids])  # this device's own sublattice's width
        return length_arr, width_arr


    # --------------------------
    # ---- Generate (or regenerate) every device's position and angle from current parameters ----
    def update_array(self, verbose=False):
        """
        GOAL:
            Build (or completely rebuild) every device's center position,
            geometry angle, and initial magnetization angle from this
            instance's current per-sublattice setup parameters -- the
            single method every device position in this class ultimately
            comes from. Called once at the end of __init__, and again
            every time update_parameters() is called.

        HOW IT TALKS TO OTHER FILES:
            Called by __init__ and update_parameters (this class). Calls
            _normalize_termination, _rotated_basis, _per_device_lengths_
            widths, _compute_boundaries, and, when any sublattice uses
            sublattice_termination_mode="rotated_boundary_fill",
            _apply_boundary_fill (all this class). Calls _refresh_data
            (this class) at the end to populate device_data and the other
            derived per-device attributes.

        PARAMETERS:
            verbose (bool): prints a per-sublattice summary (device count,
                rotation) and the final array's overall device count and
                boundaries when True.

        RETURNS:
            None. Populates (overwriting any previous values):
            self.device_centers, self.device_angles, self.device_ids,
            self.device_centers_by_sublattice, self.sublattice_dims_xy,
            self.boundaries, self.effective_m, self.effective_n, and
            (via _refresh_data) self.device_array/self.device_data/
            self.device_m_angles.

        LIMITATIONS:
            See this class's module-level PURPOSE/LIMITATIONS sections for
            the full explanation of why re-running this method (directly,
            or via update_parameters()) shifts every device's final
            position, including devices in sublattices that were not
            themselves changed -- summarized here: every sublattice's
            devices are generated in that sublattice's own local frame,
            rotated about that frame's own center, translated, and only
            THEN does the combined array get re-centered on (0, 0) by its
            new overall center of mass, which depends on every
            sublattice's placement at once.
        """
        self.device_centers = []  # accumulated across every sublattice below, as a list of per-sublattice arrays, before being stacked into one array
        self.device_angles = []  # accumulated across every sublattice below (device orientation, degrees)
        self.device_ids = []  # accumulated across every sublattice below (sublattice ID, repeated once per device in that sublattice)
        self.device_centers_by_sublattice = {}  # same centers as device_centers, but kept separately per sublattice ID (useful for sublattice-specific operations, for example set_magnetization_pattern's sublattice filter)
        self.sublattice_dims_xy = {}  # reset the actual-built (num_x, num_y) metadata for this regeneration

        for sub_id in self.sublattice_dims.keys():  # build one sublattice's devices at a time; arbitrary/non-contiguous sublattice IDs are supported (for example, kagome's 3 directionally-unique sublattices)
            num_x, num_y = self.sublattice_dims[sub_id]  # this sublattice's SETUP size (see sublattice_dims_xy for the actual built size after termination)
            nn_x_sub, nn_y_sub = self.nn_x[sub_id], self.nn_y[sub_id]  # this sublattice's own spacing -- may differ between sublattices (for example, kagome)
            term = self.sublattice_termination.get(sub_id, {})  # this sublattice's termination spec, if any
            x_extra, x_neg, y_extra, y_neg = self._normalize_termination(term, sub_id)  # per-edge extra-device counts: (pos_x, neg_x, pos_y, neg_y), in this sublattice's own pre-rotation local lattice frame

            # ---- Build this sublattice's local (pre-rotation) mesh, including any
            # termination extension. ----
            x_range = np.linspace(
                nn_x_sub / 2 - x_neg * nn_x_sub,  # start: centers the first device at nn_x/2, shifted earlier by nn_x per negative-x termination row
                nn_x_sub * (num_x - 0.5 + x_extra),  # end: centers the last device at nn_x * (num_x - 0.5), extended by nn_x per positive-x termination row
                num_x + x_extra + x_neg  # total device count in x: the setup count plus both termination extensions
            )
            y_range = np.linspace(
                nn_y_sub / 2 - y_neg * nn_y_sub,  # start: same convention as x_range, for y
                nn_y_sub * (num_y - 0.5 + y_extra),  # end: same convention as x_range, for y
                num_y + y_extra + y_neg  # total device count in y
            )
            if len(x_range) == 0 or len(y_range) == 0:  # this sublattice has zero devices in one direction (for example, num_x=0 with no termination): nothing to generate
                continue
            self.sublattice_dims_xy[sub_id] = (num_x + x_extra + x_neg, num_y + y_extra + y_neg)  # actual built size along this sublattice's own local x/y lattice axes, post-termination (overwritten below for "rotated_boundary_fill" sublattices, with the real filled/clipped count)
            mesh = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)  # every (x, y) combination in this sublattice's local mesh, flattened to one row per device
            angle = self.rotation_angles.get(sub_id, 0)  # this sublattice's own LATTICE rotation angle, degrees

            # ---- Rotate the mesh about its own center, using the rounded integer-nm
            # lattice basis (see _rotated_basis) rather than a continuous rotation
            # matrix, so every resulting position lands on the same integer-nm grid
            # with exact, self-consistent spacing. Each mesh point is re-expressed as
            # mesh_center + (integer step count) * rotated_basis_vector -- this
            # changes WHERE each point ends up, not the ORDER of points (still exactly
            # meshgrid's own row order: column-wise, ascending y within each ascending
            # x column). ----
            mesh_center = np.mean(mesh, axis=0)  # this sublattice's own local mesh center -- the pivot rotation happens about
            u, v = self._rotated_basis(angle, nn_x_sub, nn_y_sub)  # rounded-to-integer-nm rotated lattice basis vectors for this sublattice's angle and spacing
            steps_x = (mesh[:, 0] - mesh_center[0]) / nn_x_sub  # each point's exact step count from mesh_center, along this sublattice's own (pre-rotation) x axis
            steps_y = (mesh[:, 1] - mesh_center[1]) / nn_y_sub  # ...and along its own y axis
            rotated = mesh_center + steps_x[:, None] * u + steps_y[:, None] * v  # rotate about mesh_center using the rounded basis (angle=0 reduces exactly to the original mesh, since u=(nn_x,0), v=(0,nn_y) then)
            translated = rotated + np.array(self.translations.get(sub_id, [0, 0]))  # apply this sublattice's own translation, after rotation and before the whole-array centering step below

            self.device_centers.append(translated)  # add this sublattice's devices to the overall (not-yet-combined) list
            self.device_centers_by_sublattice[sub_id] = translated  # also keep this sublattice's devices separately, keyed by ID
            self.device_angles.extend([self.device_rotations.get(sub_id, 0)] * len(rotated))  # this sublattice's own DEVICE (island) orientation, repeated once per device in it -- independent of the lattice rotation `angle` used above
            self.device_ids.extend([sub_id] * len(rotated))  # this sublattice's ID, repeated once per device in it

        self.device_centers = np.vstack(self.device_centers)  # combine every sublattice's devices into one array, one row per device
        self.device_angles = np.array(self.device_angles)  # same combination, for device orientations

        # ---- Center the WHOLE array (every sublattice combined) on (0, 0) by its own
        # center of mass -- see this class's module-level PURPOSE/LIMITATIONS sections
        # for why this step is what makes changing any one sublattice's rotation/
        # translation/size shift EVERY device's final position, not just that
        # sublattice's own. ----
        center_of_mass = np.mean(self.device_centers, axis=0)  # this array's overall center of mass, across every sublattice combined
        self._raw_center_of_mass = center_of_mass.copy()  # persist the RAW (pre-centering) offset -- set_magnetization_pattern passes this into compute_magnetization_angles so _lattice_coords can actually undo this centering step (see compute_magnetization_angles' own `center_of_mass` parameter documentation for why re-deriving this from the already-centered self.device_centers instead, as a prior revision did, silently recovers ~(0, 0) rather than the real offset)
        self.device_centers -= center_of_mass  # re-center the combined array on (0, 0)
        self.device_centers_by_sublattice = {sid: arr - center_of_mass for sid, arr in self.device_centers_by_sublattice.items()}  # apply the same re-centering to the per-sublattice copies

        # ---- Final integer-nm snap: every point was already built from mesh_center/
        # anchor + (integer step count) * (integer-nm rotated basis), so every
        # RELATIVE spacing between any two devices is already an exact whole
        # nanometer. The center-of-mass subtraction just applied one uniform
        # (generally non-integer) shift to every point, so every point now shares the
        # same fractional-nm remainder; rounding each to the nearest nm here removes
        # that shared remainder without disturbing the already-exact relative
        # spacing. ----
        self.device_centers = np.round(self.device_centers * 1e9) * 1e-9
        self.device_centers_by_sublattice = {sid: np.round(arr * 1e9) * 1e-9 for sid, arr in self.device_centers_by_sublattice.items()}

        # ---- Boundaries, using each device's own rotated footprint (see
        # _compute_boundaries) rather than a flat margin. ----
        _length_arr, _width_arr = self._per_device_lengths_widths(self.device_ids)
        self.boundaries = self._compute_boundaries(self.device_centers, self.device_angles, _length_arr, _width_arr)
        self.effective_m = self.boundaries["xmax"] - self.boundaries["xmin"]  # array's overall physical width
        self.effective_n = self.boundaries["ymax"] - self.boundaries["ymin"]  # array's overall physical height

        # ---- Any sublattice using sublattice_termination_mode="rotated_boundary_
        # fill" gets its points REGENERATED here (see _apply_boundary_fill): instead
        # of the fixed pre-rotation row/col block built above, that sublattice's own
        # rotated lattice directions are walked outward from its center and only
        # sites landing inside the just-computed self.boundaries box are kept.
        # Sublattices left on the default "fixed_row_col" mode are untouched. ----
        if any(mode == "rotated_boundary_fill" for mode in self.sublattice_termination_mode.values()):
            self._apply_boundary_fill(verbose=verbose)  # mutates self.device_centers_by_sublattice for the affected sublattice IDs, and updates self.sublattice_dims_xy for them
            # Rebuild the flattened per-device arrays from device_centers_by_sublattice,
            # since boundary-fill may have changed how many devices some sublattices contain.
            self.device_centers = np.vstack([self.device_centers_by_sublattice[sid] for sid in self.sublattice_dims if sid in self.device_centers_by_sublattice])
            self.device_ids = np.concatenate([np.full(len(self.device_centers_by_sublattice[sid]), sid) for sid in self.sublattice_dims if sid in self.device_centers_by_sublattice])
            self.device_angles = np.concatenate([np.full(len(self.device_centers_by_sublattice[sid]), self.device_rotations.get(sid, 0)) for sid in self.sublattice_dims if sid in self.device_centers_by_sublattice])
            # Recompute boundaries/effective extents from the final point set -- this MUST
            # be redone (not reused from above): after filling, the x/y extremes are
            # generally reached by DIFFERENT sublattices at DIFFERENT edges, each with its
            # own device_rotations angle, so the earlier boundaries (computed before filling)
            # would let some devices poke past the boundary while over-padding others.
            _length_arr, _width_arr = self._per_device_lengths_widths(self.device_ids)
            self.boundaries = self._compute_boundaries(self.device_centers, self.device_angles, _length_arr, _width_arr)
            self.effective_m = self.boundaries["xmax"] - self.boundaries["xmin"]
            self.effective_n = self.boundaries["ymax"] - self.boundaries["ymin"]

        device_array = [
            [float(x), float(y), int(sub_id), float(angle), float(m_angle)]  # one row per device: [x, y, sublattice ID, geometry angle, magnetization angle] -- magnetization angle is initialized equal to the geometry angle here, and can be independently changed later via set_magnetization_pattern/set_device_magnetization
            for ((x, y), sub_id, angle, m_angle) in zip(self.device_centers, self.device_ids, self.device_angles, self.device_angles)  # m_angle initialized from `angle` (the same source, twice)
        ]
        self._refresh_data(device_array=device_array, verbose=verbose)  # populate device_data and the other derived per-device attributes from the array just built
        if verbose:
            print(f"[INFO] DeviceArray.update_array: built {len(device_array)} device(s) across {len(self.sublattice_dims)} sublattice(s); "
                  f"boundaries=({self.boundaries['xmin']*1e9:.1f}, {self.boundaries['xmax']*1e9:.1f}) nm x, "
                  f"({self.boundaries['ymin']*1e9:.1f}, {self.boundaries['ymax']*1e9:.1f}) nm y.")


    # --------------------------
    # ---- Compute the fill target box: the footprint the given sublattices would have if unrotated ----
    def _compute_target_boundaries(self, zero_rotation_ids, verbose=False):
        """
        GOAL:
            Rebuild each sublattice's (pre-fill) mesh/rotate/translate
            steps exactly as update_array() does, EXCEPT that any
            sublattice ID in `zero_rotation_ids` is generated with its
            lattice rotation angle (and device orientation angle) forced
            to 0, and return the resulting boundaries -- the TARGET BOX
            _apply_boundary_fill clips its generated points to.

        HOW IT TALKS TO OTHER FILES:
            Called by _apply_boundary_fill (this class), once per
            update_array() call that has at least one
            sublattice_termination_mode="rotated_boundary_fill" sublattice.
            Calls _normalize_termination, _rotated_basis, and
            _compute_boundaries (all this class).

        PARAMETERS:
            zero_rotation_ids (Set): sublattice IDs to generate with
                rotation forced to 0 for this computation only (this
                instance's own self.rotation_angles/self.device_rotations
                are not modified). Sublattices not in this set keep their
                real configured rotation.
            verbose (bool): prints the resulting target box when True.

        RETURNS:
            Dict[str, float]: {"xmin", "xmax", "ymin", "ymax"} -- the
            boundaries of the hypothetical point set described above.

        LIMITATIONS:
            This gives "rotated_boundary_fill" a target box equal to
            sublattice_dims' TRUE unrotated footprint (for example, a real
            6x7 in nn_x/nn_y units), rather than the diamond-inflated box
            the actual rotated points would span -- this is what makes
            rotation_angles=45 fill out to the same overall size as
            rotation_angles=0 would, for the same sublattice_dims, instead
            of a bigger box that grows with the rotation angle. Rotating a
            block about its own local center does not change that block's
            mean position, so the global center of mass computed here is
            mathematically identical to the one update_array() already
            uses to center self.device_centers -- meaning this target box
            lines up correctly with the anchors _apply_boundary_fill uses,
            without needing any extra frame-alignment bookkeeping.
        """
        ref_points, ref_angles, ref_lengths, ref_widths = [], [], [], []  # accumulated across every sublattice below
        for sub_id in self.sublattice_dims.keys():  # rebuild every sublattice's (hypothetical) mesh, one at a time
            num_x, num_y = self.sublattice_dims[sub_id]  # this sublattice's setup size
            nn_x_sub, nn_y_sub = self.nn_x[sub_id], self.nn_y[sub_id]  # this sublattice's own spacing
            term = self.sublattice_termination.get(sub_id, {})  # this sublattice's termination spec, if any
            x_extra, x_neg, y_extra, y_neg = self._normalize_termination(term, sub_id)  # per-edge extra-device counts
            x_range = np.linspace(nn_x_sub / 2 - x_neg * nn_x_sub, nn_x_sub * (num_x - 0.5 + x_extra), num_x + x_extra + x_neg)  # same construction as update_array's own x_range
            y_range = np.linspace(nn_y_sub / 2 - y_neg * nn_y_sub, nn_y_sub * (num_y - 0.5 + y_extra), num_y + y_extra + y_neg)  # same construction as update_array's own y_range
            if len(x_range) == 0 or len(y_range) == 0:  # this sublattice has zero devices in one direction: nothing to include in the target box
                continue
            mesh = np.array(np.meshgrid(x_range, y_range)).T.reshape(-1, 2)  # this sublattice's local mesh
            angle = 0 if sub_id in zero_rotation_ids else self.rotation_angles.get(sub_id, 0)  # force rotation to 0 for a fill sublattice; otherwise use its real configured angle
            mesh_center = np.mean(mesh, axis=0)  # this mesh's own center, the rotation pivot
            u, v = self._rotated_basis(angle, nn_x_sub, nn_y_sub)  # same rounded-integer-nm basis update_array() itself uses, so this reference box lines up with how points are actually generated
            steps_x = (mesh[:, 0] - mesh_center[0]) / nn_x_sub  # each point's step count from mesh_center, x
            steps_y = (mesh[:, 1] - mesh_center[1]) / nn_y_sub  # ...and y
            rotated = mesh_center + steps_x[:, None] * u + steps_y[:, None] * v  # rotated mesh points
            translated = rotated + np.array(self.translations.get(sub_id, [0, 0]))  # this sublattice's translation applied
            ref_points.append(translated)
            # Device orientation for footprint purposes mirrors the same zero-override
            # used for position above: if this sublattice's LATTICE rotation is being
            # treated as 0 for target-box purposes, its device/island orientation is
            # treated as 0 too (what a genuinely unrotated array's footprint would be);
            # otherwise its real configured device_rotations angle is used.
            footprint_angle = 0 if sub_id in zero_rotation_ids else self.device_rotations.get(sub_id, 0)
            ref_angles.append(np.full(len(translated), footprint_angle))
            ref_lengths.append(np.full(len(translated), self.device_length[sub_id]))
            ref_widths.append(np.full(len(translated), self.device_width[sub_id]))
        all_points = np.vstack(ref_points)  # every hypothetical device position, across all sublattices
        all_angles = np.concatenate(ref_angles)  # matching device orientations
        all_lengths = np.concatenate(ref_lengths)  # matching device lengths
        all_widths = np.concatenate(ref_widths)  # matching device widths
        com = np.mean(all_points, axis=0)  # this hypothetical point set's own center of mass
        all_points = all_points - com  # center it, matching update_array's own centering step
        result = self._compute_boundaries(all_points, all_angles, all_lengths, all_widths)  # shape-aware target box (see this method's LIMITATIONS)
        if verbose:
            print(f"[DEBUG] DeviceArray._compute_target_boundaries: target box = "
                  f"x[{result['xmin']*1e9:.1f}, {result['xmax']*1e9:.1f}] nm, "
                  f"y[{result['ymin']*1e9:.1f}, {result['ymax']*1e9:.1f}] nm.")
        return result

    # --------------------------
    # ---- Rotated boundary fill: regenerate a rotated sublattice's points by walking its own lattice and clipping to a target box ----
    def _apply_boundary_fill(self, verbose=False):
        """
        GOAL:
            For each sublattice whose sublattice_termination_mode is
            "rotated_boundary_fill", replace its points with a fresh set
            generated by walking that sublattice's own rotated lattice
            basis vectors outward from its center and keeping every site
            that falls within a TARGET BOX (see _compute_target_
            boundaries) -- so a rotated square (pinwheel) sublattice fills
            out to, and gets clipped at, a well-defined square footprint,
            instead of the diamond-shaped gaps/overhangs a fixed
            rectangular block produces once rotated.

        HOW IT TALKS TO OTHER FILES:
            Called by update_array (this class), only when at least one
            sublattice uses sublattice_termination_mode=
            "rotated_boundary_fill". Calls _compute_target_boundaries and
            _rotated_basis/_device_half_extents (all this class). Requires
            self.boundaries to already exist from a first "fixed_row_col"
            pass earlier in the same update_array() call.

        PARAMETERS:
            verbose (bool): prints the number of sites kept for each
                affected sublattice when True.

        RETURNS:
            None. Mutates self.device_centers_by_sublattice for every
            affected sublattice ID, and updates self.sublattice_dims_xy
            for them (measured from the actual filled/clipped lattice,
            since a plain row/col count no longer applies once fill/clip
            has been used).

        LIMITATIONS:
            This assumes the target footprint is well-approximated by the
            rectangular target box computed by _compute_target_boundaries.
            That is exactly right for square/pinwheel-style arrays. For a
            hexagonal footprint (for example, kagome), the rectangular box
            is only an outer bound -- expect this to overfill the corners,
            which is why single-device removal (see remove_devices()) is
            the recommended next step there, rather than extending this
            method to arbitrary polygon footprints.
        """
        fill_ids = {sid for sid, mode in self.sublattice_termination_mode.items() if mode == "rotated_boundary_fill"}  # every sublattice ID using this fill mode
        target = self._compute_target_boundaries(fill_ids, verbose=verbose)  # the target box every fill sublattice's points get clipped to
        eps = min(min(self.nn_x.values()), min(self.nn_y.values())) * 1e-6  # small floating-point tolerance so sites essentially exactly on the boundary are not dropped due to rounding -- smallest spacing across ALL sublattices, so it stays a safely tiny fraction of a step everywhere

        for sub_id in fill_ids:  # regenerate one fill sublattice's points at a time
            existing = self.device_centers_by_sublattice.get(sub_id)  # this sublattice's points from the earlier "fixed_row_col" pass, used only as an anchor below
            if existing is None or len(existing) == 0:  # nothing to fill from (for example, sublattice_dims gave it zero devices)
                continue
            nn_x_sub, nn_y_sub = self.nn_x[sub_id], self.nn_y[sub_id]  # this sublattice's own spacing
            angle = self.rotation_angles.get(sub_id, 0)  # this sublattice's real (not forced-to-0) lattice rotation angle
            u, v = self._rotated_basis(angle, nn_x_sub, nn_y_sub)  # same rounded-integer-nm basis used everywhere else; the anchor below is already integer-nm, so every generated candidate stays exactly on the integer-nm grid without needing further rounding

            # Clip margin uses THIS sublattice's own device_rotations (island
            # orientation) angle AND its own device_length/device_width, not a flat
            # margin: every candidate device in this sublattice shares the same
            # orientation and size, so its footprint half-extent along x and y is
            # fixed for the whole sublattice, and generally different from another
            # sublattice's if oriented/sized differently.
            half_x, half_y = self._device_half_extents(self.device_rotations.get(sub_id, 0), self.device_length[sub_id], self.device_width[sub_id])
            lo_x, hi_x = target["xmin"] + half_x, target["xmax"] - half_x  # clip bounds, x -- inset by this sublattice's own half-extent so devices near the edge do not poke past the target box
            lo_y, hi_y = target["ymin"] + half_y, target["ymax"] - half_y  # clip bounds, y

            # Anchor at an ACTUAL existing lattice site, not the centroid. In a
            # standard interleaved multi-sublattice array (for example, square ASI),
            # sublattices are deliberately translated so their centroids coincide --
            # the distinguishing offset is a sub-nn-cell phase shift the mean does not
            # preserve. Any real point in this sublattice's own lattice already
            # encodes that phase correctly, so stepping integer multiples of u/v from
            # it reproduces (and extends) exactly the right lattice.
            anchor = existing[0]  # this sublattice's own phase-correct anchor point
            box_diag = np.hypot(hi_x - lo_x, hi_y - lo_y)  # target box diagonal, used to size the walk range below
            n_u = int(box_diag / nn_x_sub) + 2  # generous walk range along u -- guaranteed to reach past the box edges; extra points simply get clipped below
            n_v = int(box_diag / nn_y_sub) + 2  # generous walk range along v
            i_idx, j_idx = np.meshgrid(np.arange(-n_u, n_u + 1), np.arange(-n_v, n_v + 1), indexing="ij")  # every (i, j) integer lattice-step combination in the walk range
            candidates = anchor + i_idx.ravel()[:, None] * u + j_idx.ravel()[:, None] * v  # every (i, j) lattice site as a physical (x, y) point

            keep = (  # which candidate sites actually fall inside the (half-extent-inset) target box
                (candidates[:, 0] >= lo_x - eps) & (candidates[:, 0] <= hi_x + eps) &
                (candidates[:, 1] >= lo_y - eps) & (candidates[:, 1] <= hi_y + eps)
            )
            filled = candidates[keep]  # the final, kept device positions for this sublattice
            kept_i, kept_j = i_idx.ravel()[keep], j_idx.ravel()[keep]  # which (i, j) lattice indices survived, for the built-size report below
            self.device_centers_by_sublattice[sub_id] = filled
            # Report the actual built size along this sublattice's own two lattice
            # directions (number of distinct i, j values kept), since a plain row/col
            # count no longer applies once fill/clip has been used.
            self.sublattice_dims_xy[sub_id] = (len(np.unique(kept_i)) if len(kept_i) else 0, len(np.unique(kept_j)) if len(kept_j) else 0)
            if verbose:
                print(f"[DEBUG] DeviceArray._apply_boundary_fill: sublattice {sub_id} filled to {len(filled)} device(s).")


    # --------------------------
    # ---- Rebuild device_data (and the derived per-device arrays) from device_array ----
    def _refresh_data(self, device_array=None, device_data=None, verbose=False):
        """
        GOAL:
            Rebuild self.device_data (and self.device_centers,
            self.device_ids, self.device_angles, self.device_m_angles)
            from self.device_array -- the single place those derived
            per-device attributes are kept in sync with the underlying
            array whenever it changes.

        HOW IT TALKS TO OTHER FILES:
            Called by update_array, update_from_array, update_from_
            extended_array, set_device_property, and remove_devices (all
            this class), any time self.device_array changes and the
            derived attributes need to be brought back in sync with it.

        PARAMETERS:
            device_array: the array to rebuild from. If not given, falls
                back to self.device_array (raising AttributeError if
                neither is available -- see LIMITATIONS).
            device_data: existing per-device data to preserve values from
                when the new `device_array` does not carry every column
                (specifically device_length_nm/device_width_nm, columns 5
                and 6, which are optional). If not given, falls back to
                self.device_data if it exists, or an empty dict otherwise.
            verbose (bool): prints the number of devices refreshed when
                True.

        RETURNS:
            None. Sets self.device_array, self.device_data,
            self.device_centers, self.device_ids, self.device_angles, and
            self.device_m_angles.

        LIMITATIONS:
            Raises AttributeError if `device_array` is not given and
            self.device_array does not already exist (this happens only if
            _refresh_data is somehow called before update_array has ever
            run once, which does not occur through this class's own normal
            usage).
        """
        if device_array is None:  # no array given explicitly: fall back to the instance's own current array
            device_array = getattr(self, "device_array", None)
            if device_array is None:
                raise AttributeError("device_array must be provided for initial refresh")
        if device_data is None:  # no prior device_data given explicitly: fall back to the instance's own current data, or an empty dict on a true first call
            device_data = getattr(self, "device_data", {})
        self.device_array = np.array(device_array)  # store as a numpy array: one row per device, columns [x, y, sublattice ID, geometry angle, magnetization angle, (optional) device_length_nm, (optional) device_width_nm]
        self.device_data = {}  # rebuilt fresh below, one entry per device

        for idx, row in enumerate(self.device_array):  # populate device_data from each row of device_array, falling back to prior values (or this device's own sublattice default) for the optional length/width columns
            prev_data = device_data.get(idx, {})  # this device's own PRIOR entry, if any -- used only as a fallback for length/width below
            sub_id = int(row[2])  # this device's sublattice ID -- column 2 (see this class's module-level LIMITATIONS regarding the column-order bug this convention exists to keep consistent with)
            self.device_data[idx] = {
                "x": float(row[0]),  # device center, x
                "y": float(row[1]),  # device center, y
                "sublattice": sub_id,  # sublattice ID
                "angle": float(row[3]),  # geometry (rotation) angle -- column 3
                "m_angle": float(row[4]),  # magnetization angle
                "device_length_nm": row[5] if len(row) >= 6 else prev_data.get("device_length_nm", self._length_nm_for(sub_id)),  # optional column 5; falls back to the prior value, then this device's own sublattice default
                "device_width_nm": row[6] if len(row) >= 7 else prev_data.get("device_width_nm", self._width_nm_for(sub_id)),  # optional column 6; same fallback chain
            }
        self.device_centers = self.device_array[:, 0:2]  # x/y columns, as their own array
        self.device_ids = np.array([self.device_data[i]["sublattice"] for i in range(len(self.device_array))])  # sublattice ID per device, read back from the just-rebuilt device_data (not directly from device_array), so it reflects any fallback/default applied above
        self.device_angles = np.array([self.device_data[i]["angle"] for i in range(len(self.device_array))])  # geometry angle per device, same read-back convention
        self.device_m_angles = np.array([self.device_data[i]["m_angle"] for i in range(len(self.device_array))])  # magnetization angle per device, same read-back convention
        if verbose:
            print(f"[DEBUG] DeviceArray._refresh_data: refreshed {len(self.device_array)} device(s).")

    # --------------------------
    # ---- Replace the device array from a plain [x, y, sublattice, angle, m_angle] array, with overlap checking ----
    def update_from_array(self, array, device_data=None, verbose=False):
        """
        GOAL:
            Replace this instance's entire device_array/device_data with a
            caller-supplied array, validating that the result has no
            device-to-device overlaps before committing the change (and
            reverting if it does).

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly;
            provided for a caller that wants to hand-construct or
            externally modify a full device array and apply it as a unit
            (as opposed to update_parameters(), which regenerates the
            array from setup parameters, or set_device_property(), which
            changes one device at a time). Calls _refresh_data and
            check_for_overlaps (both this class).

        PARAMETERS:
            array (array-like): rows of [x, y, sublattice, angle, m_angle]
                (5 columns), in the SAME units this class uses internally
                (meters, degrees) -- unlike most of this class's public
                parameters, which take nm.
            device_data (Dict, optional): explicit per-device data to use
                instead of deriving it from `array` and this instance's
                prior device_data -- see LIMITATIONS for the two different
                code paths this takes depending on whether this argument
                and self.device_data are available.
            verbose (bool): prints the outcome (applied / reverted due to
                overlaps) when True (this method also always prints that
                outcome regardless -- see LIMITATIONS).

        RETURNS:
            None. Updates self.device_array/self.device_data (and the
            attributes _refresh_data derives from them) if the new
            geometry has no overlaps; otherwise leaves them unchanged.

        LIMITATIONS:
            If `device_data` is not given: this instance's own prior
            device_data is reused if it already exists (the ordinary
            case). If it does NOT already exist (a rare, essentially
            first-time-setup scenario that does not occur through this
            class's own normal usage, since __init__ always populates
            device_data first), device_data is instead derived directly
            from `array` itself, expecting a 7-column extended layout
            ([x, y, sublattice, angle, m_angle, device_length_nm,
            device_width_nm]) rather than the plain 5-column layout the
            main `array` parameter above describes -- see this class's
            module-level LIMITATIONS section for a related column-order
            bug that was fixed in this fallback branch specifically.
            Requires the `shapely` package (via check_for_overlaps) to
            actually validate the result; see that method's own
            documentation.
        """
        import copy  # deep-copying device_data to preserve existing values across this update
        array = np.array(array)  # ensure a plain numpy array
        if device_data is None:  # no explicit device_data given
            if hasattr(self, "device_data"):  # the ordinary case: reuse this instance's existing device_data
                device_data = copy.deepcopy(self.device_data)
            else:  # rare fallback: derive device_data directly from `array`, expecting an extended 7-column layout -- see this method's own LIMITATIONS
                device_data = {}
                for i, row in enumerate(array):  # one entry per row of the extended array
                    sub_id = int(row[2])  # sublattice ID -- column 2, matching this class's canonical column order (see this class's module-level LIMITATIONS regarding the column-order bug fixed here)
                    device_data[i] = {
                        "device_length_nm": row[5] if len(row) >= 6 else self._length_nm_for(sub_id),  # optional column 5; falls back to this device's own sublattice default
                        "device_width_nm": row[6] if len(row) >= 7 else self._width_nm_for(sub_id),  # optional column 6; same fallback
                        "angle": float(row[3]),  # geometry angle -- column 3, matching this class's canonical column order
                        "sublattice": sub_id,
                        "m_angle": float(row[4]),  # magnetization angle
                        "x": float(row[0]),  # device center, x
                        "y": float(row[1]),  # device center, y
                    }
        else:  # explicit device_data given: normalize it, falling back to this row's own prior value (or sublattice default) for any missing field
            new_device_data = {}
            for i, row in device_data.items():  # one entry per device
                sub_id = int(row["sublattice"]) if "sublattice" in row else int(self.device_ids[i])  # this device's sublattice ID, or its existing one if not given
                new_device_data[i] = {
                    "device_length_nm": row["device_length_nm"] if "device_length_nm" in row else self._length_nm_for(sub_id),  # falls back to this device's own sublattice default
                    "device_width_nm": row["device_width_nm"] if "device_width_nm" in row else self._width_nm_for(sub_id),  # same fallback
                    "angle": row["angle"] if "angle" in row else self.device_angles[i],  # falls back to this device's existing geometry angle
                    "m_angle": row["m_angle"] if "m_angle" in row else self.device_m_angles[i],  # falls back to this device's existing magnetization angle
                    "sublattice": sub_id,
                    "x": float(row["x"]) if "x" in row else self.device_centers[i, 0],  # falls back to this device's existing x position
                    "y": float(row["y"]) if "y" in row else self.device_centers[i, 1],  # falls back to this device's existing y position
                }
            device_data = new_device_data

        original_array = self.device_array.copy()  # preserved in case the proposed change needs to be reverted below
        original_data = copy.deepcopy(self.device_data)  # preserved in case the proposed change needs to be reverted below

        self.device_array = array  # apply the proposed change
        self.device_data = device_data
        self._refresh_data(verbose=verbose)  # bring the derived per-device attributes back in sync with the just-applied change
        overlaps = self.check_for_overlaps()  # validate the proposed geometry (requires shapely -- see this method's LIMITATIONS)

        if overlaps:  # the proposed change is invalid: revert it
            print("[WARNING] DeviceArray.update_from_array: overlaps detected. Change rejected.")
            self.device_array = original_array
            self.device_data = original_data
            self._refresh_data(verbose=verbose)
        else:
            print("[INFO] DeviceArray.update_from_array: applied successfully. No overlaps.")

    # --------------------------
    # ---- Replace the device array from an extended [x, y, sublattice, angle, m_angle, length_nm, width_nm] array ----
    def update_from_extended_array(self, extended_array, verbose=False):
        """
        GOAL:
            Same as update_from_array, but the input already includes each
            device's own length/width (in nm) as two extra columns,
            avoiding the need for a separate device_data argument.

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly. Calls
            update_from_array (this class) at the end, once the extended
            input has been split into the plain array + device_data shape
            that method expects.

        PARAMETERS:
            extended_array (array-like): rows of [x, y, sublattice, angle,
                m_angle, device_length_nm, device_width_nm] (7 columns), in
                the same units this class uses internally for x/y/angle
                (meters, degrees), with length/width in nm.
            verbose (bool): forwarded to update_from_array, so its own
                outcome message/behavior applies.

        RETURNS:
            None. See update_from_array.

        LIMITATIONS:
            Raises a ValueError (via Python's own tuple-unpacking) if any
            row does not have exactly 7 values.
        """
        array = []  # the plain 5-column array update_from_array's own `array` parameter expects
        device_data = {}  # the corresponding device_data dict, carrying the extra length/width columns
        for idx, row in enumerate(extended_array):  # split each row into its core (5-column) and extended (length/width) parts
            x, y, sublattice, angle, m_angle, length_nm, width_nm = row  # unpack all 7 columns
            array.append([x, y, int(sublattice), angle, m_angle])  # core columns only
            device_data[idx] = {
                "x": x,
                "y": y,
                "sublattice": int(sublattice),
                "angle": angle,
                "m_angle": m_angle,
                "device_length_nm": length_nm,
                "device_width_nm": width_nm,
            }
        self.update_from_array(array, device_data, verbose=verbose)  # apply via update_from_array's own overlap-checked commit/revert logic


    # --------------------------
    # ---- Update one property of one device by index, with overlap checking ----
    def set_device_property(self, index, key, value, verbose=False):
        """
        GOAL:
            Update exactly one property of exactly one device (position,
            angle, sublattice, or size), validating that the result has no
            device-to-device overlaps before committing the change (and
            reverting if it does).

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly;
            provided for hand-adjusting individual devices (see this file's
            own bottom-of-file worked example). Calls _refresh_data and
            check_for_overlaps (both this class).

        PARAMETERS:
            index (int): device index (as seen in device_data /
                get_device_array()).
            key (str): which property to update -- one of "x", "y",
                "angle", "sublattice" (core attributes, stored in
                device_array), or "length", "width" (extended attributes,
                stored only in device_data).
            value: the new value for `key`. Units match this class's
                internal convention for the given key: meters for "x"/"y",
                degrees for "angle", nm for "length"/"width".
            verbose (bool): prints the outcome (applied / reverted due to
                overlaps) when True (this method also always prints that
                outcome regardless -- see LIMITATIONS).

        RETURNS:
            None. Updates the device's property if the resulting geometry
            has no overlaps; otherwise leaves it unchanged.

        LIMITATIONS:
            See this class's module-level LIMITATIONS section: earlier
            revisions of this method mapped "angle" and "sublattice" to
            the wrong columns of device_array (swapped relative to every
            other part of this class's own convention), so calling
            set_device_property(index, "angle", value) actually overwrote
            that device's sublattice ID, and vice versa. Fixed in this
            revision -- col_map below now matches device_array's real
            column layout (column 2 = sublattice, column 3 = angle, the
            same order update_array/_refresh_data/get_device_array all
            use).

            Requires the `shapely` package (via check_for_overlaps) to
            actually validate the result; see that method's own
            documentation. Raises ValueError if `key` is not one of the
            six supported values, or IndexError if `index` is not a valid
            device index and `key` is "length"/"width" (the core-attribute
            branch does not itself validate `index` before writing to
            device_array, relying on numpy's own IndexError for an
            out-of-range index).
        """
        col_map = {"x": 0, "y": 1, "sublattice": 2, "angle": 3}  # matches device_array's real column layout (see this method's own LIMITATIONS regarding the bug this corrects)
        if key not in ["x", "y", "angle", "sublattice", "length", "width"]:  # only these six properties are supported
            raise ValueError("Key must be one of: 'x', 'y', 'angle', 'sublattice', 'length', 'width'")
        import copy  # deep-copying device_data to preserve existing values across this update
        new_device_array = np.copy(self.device_array)  # preserved copy, modified with the proposed change below
        new_device_data = copy.deepcopy(self.device_data)  # preserved copy, modified with the proposed change below

        if key in ["length", "width"]:  # extended attribute: stored only in device_data
            field = "device_length_nm" if key == "length" else "device_width_nm"  # this key's corresponding device_data field name
            if index not in new_device_data:  # the requested device must actually exist
                raise IndexError(f"Device index {index} not found in device_data.")
            new_device_data[index][field] = value
        else:  # core attribute: stored in device_array, with device_data kept in sync
            new_device_array[index, col_map[key]] = value  # write into the correct column (see col_map, and this method's own LIMITATIONS)
            new_device_data[index]["angle"] = new_device_array[index, 3]  # keep device_data's angle in sync with device_array's own angle column (3)
            new_device_data[index]["sublattice"] = int(new_device_array[index, 2])  # keep device_data's sublattice in sync with device_array's own sublattice column (2)
            new_device_data[index]["x"] = new_device_array[index, 0]  # keep device_data's x in sync
            new_device_data[index]["y"] = new_device_array[index, 1]  # keep device_data's y in sync

        # Temporarily apply the change to validate it via check_for_overlaps, since
        # that method reads from self.device_array/self.device_data (via
        # self.device_centers/self.device_angles, refreshed below) rather than
        # accepting a proposed geometry as an argument.
        original_array = self.device_array  # preserved in case the proposed change needs to be reverted below
        original_data = self.device_data  # preserved in case the proposed change needs to be reverted below
        self.device_array = new_device_array
        self.device_data = new_device_data
        self._refresh_data(verbose=verbose)  # bring the derived per-device attributes back in sync with the just-applied change
        overlaps = self.check_for_overlaps()  # validate the proposed geometry (requires shapely -- see this method's LIMITATIONS)
        if overlaps:  # the proposed change is invalid: revert it
            print(f"[WARNING] DeviceArray.set_device_property: overlaps detected after attempting to set '{key}' at index {index}. Change rejected.")
            self.device_array = original_array
            self.device_data = original_data
            self._refresh_data(verbose=verbose)
        else:
            print(f"[INFO] DeviceArray.set_device_property: no overlaps. '{key}' at index {index} updated.")

    # --------------------------
    # ---- Remove specific devices by index, and re-index the remaining devices from 0 ----
    def remove_devices(self, indices, verbose=False):
        """
        GOAL:
            Remove one or more devices by their current index, and re-index
            the remaining devices starting from 0. This is the recommended
            way to hand-carve an irregular footprint (for example, trimming
            a kagome/hexagonal array down from its rectangular
            sublattice_dims footprint) rather than extending
            sublattice_termination_mode="rotated_boundary_fill" to
            arbitrary polygon shapes.

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly;
            provided as the recommended manual-trimming tool for
            non-rectangular footprints (see this class's module-level
            LIMITATIONS section regarding "rotated_boundary_fill"'s own
            rectangular-box limitation). Calls _refresh_data (this class).

        PARAMETERS:
            indices (int or Iterable[int]): device index/indices to remove
                (as seen in device_data / get_device_array()).
            verbose (bool): prints the number of devices removed and
                remaining when True (this method also always prints a
                one-line removal summary regardless -- see LIMITATIONS).

        RETURNS:
            None. Updates self.device_array/self.device_data (re-indexed
            from 0) and the attributes _refresh_data derives from them.

        LIMITATIONS:
            No overlap check is performed here (unlike set_device_property
            and update_from_array) -- removing a device can only reduce the
            footprint, never introduce a new overlap, so this check would
            be redundant. Raises IndexError if any requested index is not
            a valid, currently-existing device index. The one-line "removed
            N device(s)" summary is printed unconditionally (not gated by
            `verbose`), since it reports the direct result of the
            operation the caller just requested.
        """
        if isinstance(indices, int):  # a single index was given: normalize to a collection
            indices = [indices]
        indices = set(indices)  # de-duplicate, and allow fast membership tests below
        unknown = indices - set(self.device_data.keys())  # any requested index that does not actually exist
        if unknown:
            raise IndexError(f"Invalid device index/indices: {sorted(unknown)}")
        keep = [i for i in range(len(self.device_array)) if i not in indices]  # every index NOT being removed, in original order
        new_array = self.device_array[keep]  # the surviving rows, still in original relative order
        new_data = {new_i: self.device_data[old_i] for new_i, old_i in enumerate(keep)}  # re-key device_data (and its device_length_nm/device_width_nm overrides) onto the new 0-based indices
        self.device_array = new_array
        self.device_data = new_data
        self._refresh_data(verbose=verbose)
        print(f"[INFO] DeviceArray.remove_devices: removed {len(indices)} device(s); {len(self.device_array)} remain.")

    # --------------------------
    # ---- Check for device-to-device overlaps, using each device's actual elliptical footprint ----
    def check_for_overlaps(self, verbose=False):
        """
        GOAL:
            Check every pair of devices in the current array for physical
            overlap, modeling each device as an ellipse (semi-axes
            length/2, width/2) at its own position and orientation.

        HOW IT TALKS TO OTHER FILES:
            Called by update_from_array, set_device_property (both this
            class) as the validation step before committing a proposed
            geometry change.

        PARAMETERS:
            verbose (bool): prints the number of overlapping pairs found
                when True.

        RETURNS:
            List[Tuple[int, int]]: (i, j) index pairs (i < j) whose
            elliptical footprints intersect. Empty if no overlaps.

        LIMITATIONS:
            Requires the `shapely` package, imported inside this method
            (not at module level) so the rest of this class remains usable
            in an environment without shapely installed -- at the cost of
            this method specifically raising ModuleNotFoundError at call
            time, rather than the whole module failing to import, if
            shapely is missing. This check has not been extensively
            validated against every geometry this class can produce; treat
            a "no overlaps" result as a useful sanity check rather than an
            absolute guarantee for unusual configurations.
        """
        from shapely.geometry import Point  # geometric point/shape construction
        from shapely.affinity import scale, rotate, translate  # shape transforms, for building each device's own ellipse
        rectangles = []  # one shapely shape per device, accumulated below (named for historical reasons -- these are ellipses, not rectangles)
        for idx, ((x, y), angle_deg) in enumerate(zip(self.device_centers, self.device_angles)):  # build one device's shape at a time
            dev_info = self.device_data.get(idx, {})  # this device's own data, if any
            sub_id = dev_info.get("sublattice", int(self.device_ids[idx]))  # this device's sublattice ID, for its own per-sublattice default length/width if not explicitly stored
            length = dev_info.get("device_length_nm", self._length_nm_for(sub_id)) * 1e-9  # this device's length, nm -> m
            width = dev_info.get("device_width_nm", self._width_nm_for(sub_id)) * 1e-9  # this device's width, nm -> m
            circle = Point(0, 0).buffer(1.0, resolution=64)  # unit circle, high resolution for a smooth ellipse once scaled
            ellipse = scale(circle, length / 2, width / 2)  # scale into this device's own ellipse (semi-axes length/2, width/2)
            rotated = rotate(ellipse, angle_deg, origin=(0, 0), use_radians=False)  # rotate to this device's own orientation
            translated = translate(rotated, xoff=x, yoff=y)  # move to this device's own position
            rectangles.append(translated)
        overlaps = []  # (i, j) index pairs whose shapes intersect
        for i in range(len(rectangles)):  # check every pair once (i < j), skipping self-comparison
            for j in range(i + 1, len(rectangles)):
                if rectangles[i].intersects(rectangles[j]):
                    overlaps.append((i, j))
        if verbose:
            print(f"[DEBUG] DeviceArray.check_for_overlaps: found {len(overlaps)} overlapping pair(s) among {len(rectangles)} device(s).")
        return overlaps


    # --------------------------
    # ---- Get the raw device centers array ----
    def get_device_centers(self):
        """
        GOAL:
            Return this array's device centers.

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly (most
            callers read self.device_centers or get_device_array()
            instead); provided as a simple named accessor.

        PARAMETERS:
            None.

        RETURNS:
            np.ndarray: Nx2 array of device centers, in meters. NOTE: this
            returns self.device_centers directly, not a copy -- mutating
            the returned array mutates this instance's own state.
        """
        return self.device_centers

    # --------------------------
    # ---- Nearest-neighbor / next-nearest-neighbor center-to-center displacement vectors ----
    def get_neighbor_vectors(self, reference_index=None, n_shells=2, verbose=False):
        """
        GOAL:
            Compute center-to-center displacement vectors (in nm) from a
            reference device to its nearest neighbors, next-nearest
            neighbors, and so on, grouped into `n_shells` shells by
            increasing distance -- useful for superposition/
            lattice-alignment work that needs to know exactly how far apart
            neighboring devices sit.

        HOW IT TALKS TO OTHER FILES:
            Called by print_summary (this class) to include neighbor
            vectors in its summary output. Not otherwise called within
            this file, but intended for use by superposition_operations.py
            or mumax3_parent_superposition.py wherever neighbor spacing is
            needed.

        PARAMETERS:
            reference_index (int, optional): which device to measure from.
                Defaults to the device closest to the array's own (centered)
                origin -- a representative, fully-surrounded interior site.
            n_shells (int): how many distinct distances to report (default
                2: nearest-neighbor and next-nearest-neighbor).
            verbose (bool): prints the resolved reference device and shell
                count when True.

        RETURNS:
            List[Dict]: one entry per shell, ordered by increasing
            distance: [{"distance_nm": 99, "vectors_nm": [[99, 0], [-99,
            0], [0, 99], [0, -99]]}, ...].

        LIMITATIONS:
            Since device positions are built from an integer-nm rotated
            lattice basis (see _rotated_basis) and rounded to the nearest
            nm on centering (see update_array's final integer-nm snap),
            these vectors come out as exact whole nanometers regardless of
            rotation_angles -- for example, a 45-degree-rotated 140 nm
            square ASI reports (+/-99, 0) and (0, +/-99) rather than
            (+/-98.99494937, 0), which is what makes this directly usable
            for superposition/lattice-alignment work without further
            rounding.
        """
        centers_nm = self.device_centers * 1e9  # every device center, in nm
        if reference_index is None:  # no explicit reference given: use the device closest to the array's own centered origin
            reference_index = int(np.argmin(np.linalg.norm(centers_nm, axis=1)))
        ref = centers_nm[reference_index]  # the reference device's own position
        deltas = centers_nm - ref  # every device's displacement from the reference device
        dists_nm = np.round(np.linalg.norm(deltas, axis=1)).astype(int)  # every device's distance from the reference device (already integer-nm, so this rounding is exact up to ordinary sqrt precision)
        order = np.argsort(dists_nm)  # process devices in increasing-distance order
        shells, shell_order = {}, []  # shells: {distance: [vectors]}; shell_order: distances, in the order first encountered (i.e. increasing)
        for idx in order:  # walk every device, nearest to farthest
            d = int(dists_nm[idx])
            if d == 0:  # this is the reference device itself
                continue
            if d not in shells:  # first device found at this distance: start a new shell
                if len(shell_order) >= n_shells:  # already have enough shells: stop entirely
                    break
                shells[d] = []
                shell_order.append(d)
            shells[d].append([int(round(deltas[idx][0])), int(round(deltas[idx][1]))])  # this device's own displacement vector, added to its shell
        if verbose:
            print(f"[DEBUG] DeviceArray.get_neighbor_vectors: reference_index={reference_index}, found {len(shell_order)} shell(s).")
        return [{"distance_nm": d, "vectors_nm": shells[d]} for d in shell_order]

    # --------------------------
    # ---- Get the device array, optionally with per-device length/width columns ----
    def get_device_array(self, include_size=False):
        """
        GOAL:
            Return this array's device data as a plain numpy array, either
            the core 5-column layout or an extended 7-column layout that
            also includes each device's own length/width.

        HOW IT TALKS TO OTHER FILES:
            Called throughout mumax3_parent_superposition.py wherever the
            DeviceArray's geometry needs to be read into the superposition
            pipeline (device centers, angles, magnetization angles).

        PARAMETERS:
            include_size (bool): False (default) returns the core 5-column
                layout ([x, y, sublattice, angle, m_angle]). True returns
                an extended 7-column layout, appending each device's own
                device_length_nm and device_width_nm.

        RETURNS:
            np.ndarray: Nx5 or Nx7 array (depending on `include_size`), a
            COPY of the underlying data (safe to mutate without affecting
            this instance).
        """
        if not include_size:  # core layout: a plain copy of device_array is already exactly this shape
            return self.device_array.copy()
        extended_array = []  # accumulated across every device below
        for idx, row in enumerate(self.device_array):  # append this device's own length/width to its core row
            sub_id = int(row[2])  # this device's sublattice ID
            length_nm = self.device_data[idx].get("device_length_nm", self._length_nm_for(sub_id))  # this device's own length, falling back to its sublattice default
            width_nm = self.device_data[idx].get("device_width_nm", self._width_nm_for(sub_id))  # this device's own width, falling back to its sublattice default
            extended_array.append(list(row) + [length_nm, width_nm])
        return np.array(extended_array)

    # --------------------------
    # ---- Print a summary of the array's geometry and attributes ----
    def print_summary(self, count=5):
        """
        GOAL:
            Print a human-readable summary of this array's configuration
            (per-sublattice device counts, a handful of example device
            positions/angles) and return the same information as a
            structured dict.

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly;
            intended as an interactive/debugging convenience (see this
            file's own bottom-of-file worked example). Calls
            get_neighbor_vectors (this class) to include neighbor vectors
            in the returned dict.

        PARAMETERS:
            count (int): how many example devices to print per sublattice.

        RETURNS:
            Dict: {"sublattice_counts", "sublattice_dims_xy",
            "device_centers_by_sublattice", "device_angles",
            "magnetization_angles", "array_center", "device_centers",
            "boundaries", "neighbor_vectors"} -- the same information
            printed above, in a form usable programmatically.
        """
        print("Sublattice Counts:")
        for sid, dims in self.sublattice_dims.items():  # one line per sublattice, noting if termination changed its built size from the setup request
            built_dims = self.sublattice_dims_xy.get(sid, dims)  # the ACTUAL built (num_x, num_y) after termination, vs. the originally-requested sublattice_dims
            if tuple(built_dims) != tuple(dims):
                print(f"  Sublattice {sid}: requested {dims[0]} \u00d7 {dims[1]}, built {built_dims[0]} \u00d7 {built_dims[1]} (after termination)")
            else:
                print(f"  Sublattice {sid}: {dims[0]} \u00d7 {dims[1]}")
            centers_here = self.device_centers_by_sublattice.get(sid, [])  # this sublattice's own centers
            device_rotation_here = self.device_rotations.get(sid, 0)  # this sublattice's own device (island) orientation
            for idx in range(min(count, len(centers_here))):  # print up to `count` example devices for this sublattice (guarding against requesting more than actually exist)
                print(f"Sublattice {sid} --> Device {idx+1}: Center = ({centers_here[idx][0]*1e9:.2f} nm, {centers_here[idx][1]*1e9:.2f} nm), Angle = {device_rotation_here:.2f}\u00b0")
        return {
            "sublattice_counts": {sid: dims for sid, dims in self.sublattice_dims.items()},  # the originally-requested (num_x, num_y) per sublattice
            "sublattice_dims_xy": dict(self.sublattice_dims_xy),  # the ACTUAL built (num_x, num_y) per sublattice after termination
            "device_centers_by_sublattice": dict(self.device_centers_by_sublattice),  # device centers, grouped by sublattice ID
            "device_angles": self.device_angles,  # geometry angle, per device
            "magnetization_angles": [self.device_data[i]["m_angle"] for i in range(len(self.device_array))],  # magnetization angle, per device
            "array_center": (self.effective_m * 1e9, self.effective_n * 1e9),  # the array's overall physical width/height, nm (NOTE: despite the "center" name, this is the array's SIZE, not a position -- kept as-is for compatibility with existing callers of this dict)
            "device_centers": self.device_centers * 1e9,  # device centers, nm
            "boundaries": self.boundaries,  # array bounding box, meters
            "neighbor_vectors": self.get_neighbor_vectors()  # nearest/next-nearest-neighbor displacement vectors, nm -- see get_neighbor_vectors's own documentation
        }

    # --------------------------
    # ---- Plot the array's layout, each device as an ellipse ----
    def plot_layout(self):
        """
        GOAL:
            Render a plot of this array's layout: every device drawn as an
            ellipse at its own position/orientation/size, with the array's
            boundary lines overlaid, and display it interactively.

        HOW IT TALKS TO OTHER FILES:
            Not called by any other file in this pipeline directly (the
            equivalent, file-saving version of this plot is
            utilities_superposition.save_device_array_plot); intended for
            interactive/notebook use (see this file's own bottom-of-file
            worked example).

        PARAMETERS:
            None.

        RETURNS:
            None. Calls plt.show(), which blocks until the plot window is
            closed in most interactive backends.

        LIMITATIONS:
            Uses plt.show() directly rather than returning the Figure/Axes
            objects, so this cannot be composed into a larger figure or
            saved without either modifying this method or reconstructing
            the plot separately (see utilities_superposition.
            save_device_array_plot for a save-to-file equivalent).
        """
        plt.figure(figsize=(8, 6))
        for idx, (center, angle) in enumerate(zip(self.device_centers, self.device_angles)):  # one ellipse per device
            ellipse = patches.Ellipse(
                xy=center,  # device center (already in meters, this class's internal unit)
                width=self.device_data[idx]["device_length_nm"] * 1e-9,  # nm -> m
                height=self.device_data[idx]["device_width_nm"] * 1e-9,  # nm -> m
                angle=angle,  # device orientation, degrees
                linewidth=1,
                edgecolor='blue',
                facecolor='blue'
            )
            plt.gca().add_patch(ellipse)
        plt.title("Device Array Layout")
        plt.xlabel("X Position (nm)")
        plt.ylabel("Y Position (nm)")
        plt.axvline(x=self.boundaries["xmin"], color='r', linestyle='--')  # array boundary, left edge
        plt.axvline(x=self.boundaries["xmax"], color='r', linestyle='--')  # array boundary, right edge
        plt.axhline(y=self.boundaries["ymin"], color='r', linestyle='--')  # array boundary, bottom edge
        plt.axhline(y=self.boundaries["ymax"], color='r', linestyle='--')  # array boundary, top edge
        plt.gca().set_aspect('equal')
        plt.show()


# --------------------------
# ---- Worked example: build, inspect, update, and hand-adjust a small array ----
if __name__ == "__main__":
    array = DeviceArray(
        device_length=60, device_width=60,
        nn_x=120, nn_y=120,
        sublattice_dims={0: (6, 7), 1: (7, 6)},
        rotation_angles={0: 0, 1: 0},
        translations={0: [0, 0], 1: [-60, 60]},
        device_rotations={0: 0, 1: 0}
    )

    array.print_summary()
    array.plot_layout()

    # Update bulk parameters -- see this file's module-level PURPOSE/LIMITATIONS
    # sections regarding how this shifts EVERY device's position, not just the
    # sublattices whose parameters actually changed.
    array.update_parameters(device_length=90, nn_x=180, nn_y=180, device_rotations={0: 0, 1: 90}, translations={0: [0, 0], 1: [-90, 90]})
    array.print_summary()
    array.plot_layout()

    # Rotate a few individual devices by index (requires shapely for the overlap
    # check set_device_property performs -- see that method's own documentation).
    array.set_device_property(8, "angle", 25)
    array.set_device_property(12, "angle", 45)
    array.set_device_property(25, "angle", -90)
    array.set_device_property(0, "angle", 76)
    array.plot_layout()
