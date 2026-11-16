r"""
File Name: masking.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v6

============================================================
PURPOSE
============================================================
Provides mask generation and mask-application primitives for mumax3
vector-field regions. Contains six public functions:
    - elliptical_mask: generates a boolean elliptical mask.
    - rotate_mask / translate_mask: geometric transforms applied to a
      boolean mask.
    - region_crosses_edge: reports whether a region's requested crop
      window would extend past its source array's edge, WITHOUT cropping
      anything -- a cheap pre-check any caller can use before deciding
      whether/how to proceed (see this function's own documentation for
      how the two pipelines in this codebase use it differently: one drops
      the region outright, the other only warns).
    - crop_with_nan_padding: crops a window out of an array, NaN-padding
      (never fabricating or silently shrinking) any part of that window
      that falls outside the array's own bounds.
    - apply_mask: applies a mask ("none", "ellipse", or "file") to every
      region in a regions dictionary, according to each region's own
      RegionSpec. Now uses crop_with_nan_padding for EVERY mask mode
      whenever matrix_components is available, not only "file" +
      translation=True (see this function's own LIMITATIONS section for
      the one case that still falls back to extract_regions' own
      un-padded crop).
    - process_external_mask: converts a mask CSV to NPZ format, or
      validates an already-NPZ mask file path.

This file contains all masking logic used by this pipeline. It does not
crop regions from a full frame (region_extraction.py performs cropping) and
does not orchestrate frame loading (frame_pipeline.py performs that);
apply_mask calls region_extraction.extract_region directly only for the
"file" mask mode with translation enabled, where the underlying data must
be re-cropped once the mask has been aligned to a region's center.

============================================================
LIMITATIONS
============================================================
- apply_mask already honors each RegionSpec's own mask_mode/mask_file
  independently, region by region, so a region_specs list mixing different
  mask modes across regions is supported by this function today. However,
  nothing upstream currently constructs such a mixed list:
  region_extraction.build_region_specs and the parent scripts still assume
  one mask mode applies to an entire call. Supporting user-facing mixed
  mask modes requires wiring that construction, not changes here.
- rotate_mask and translate_mask operate on a single boolean mask and do
  not batch across multiple masks in one call. apply_mask caches rotated
  masks per (mask_file, angle) pair within a single call so repeated
  regions at the same angle are not re-rotated, but this cache does not
  persist across separate calls to apply_mask.
- apply_mask's edge-safe re-crop (crop_with_nan_padding, against
  matrix_components) covers "none", "ellipse", and "file"+translation=True
  -- every mask mode this codebase's parent scripts actually construct.
  The one mode NOT covered is mask_mode="file" with translation=False (a
  region whose mask template is already aligned to the frame's own
  coordinate system, needing no shift) -- that path still uses whatever
  region_extraction.extract_regions already produced upstream, un-padded.
  This is unchanged from before this file's edge-safety pass; no parent
  script in this codebase currently sets translation=False.
"""

from typing import Tuple, List, Dict, Any, Optional  # type hints used throughout this file's signatures
import numpy as np  # array operations and boolean mask handling
from scipy.ndimage import affine_transform as _affine_transform, shift as _nd_shift  # underlying rotation/translation implementations

from region_extraction import extract_region, RegionSpec  # extract_region: re-crops source data for the "file" + translation path; RegionSpec: the per-region spec type this file reads from


# --------------------------
# ---- Elliptical mask generation ----
def elliptical_mask(
        region_shape: Tuple[int, int],  # (height, width) of the region this mask will be applied to
        size: Tuple[int, int],  # (a, b) ellipse semi-axes, in cells
        verbose: bool = False,  # print a debug summary of the generated mask when True
) -> np.ndarray:
    """
    GOAL:
        Generate a boolean mask whose True cells lie inside an axis-aligned
        ellipse, centered within the region.

    HOW IT TALKS TO OTHER FILES:
        Called by apply_mask (this file) when a region's mask_mode is
        "ellipse". Not called from outside this file.

    PARAMETERS:
        region_shape (Tuple[int, int]): shape of the region (height, width),
            in grid cells.
        size (Tuple[int, int]): ellipse semi-axes (a, b), where a is the
            x-radius (columns) and b is the y-radius (rows), in grid cells.
        verbose (bool): prints the resulting mask's cell count when True.

    RETURNS:
        np.ndarray: boolean array of shape region_shape; True where a cell
        lies inside the ellipse.

    LIMITATIONS:
        The ellipse is centered on cell index (width // 2, height // 2) --
        the same convention extract_region uses to place a requested center
        in a cropped array -- not on the array's continuous-domain midpoint
        ((width - 1) / 2, (height - 1) / 2). For odd sizes these are
        identical; for even sizes, centering on the continuous midpoint
        would place the ellipse half a cell off from the actual
        device-center cell.
    """
    height, width = region_shape  # unpack the region's cell dimensions
    a, b = size  # unpack the ellipse's semi-axes
    x_c = width // 2  # ellipse center column, matching extract_region's cell-center convention
    y_c = height // 2  # ellipse center row, matching extract_region's cell-center convention
    y_indices, x_indices = np.ogrid[0:height, 0:width]  # open grid of row/column indices, broadcastable against the ellipse formula below
    norm_x = (x_indices - x_c) / a  # each column's distance from center, normalized by the x semi-axis
    norm_y = (y_indices - y_c) / b  # each row's distance from center, normalized by the y semi-axis
    mask = norm_x ** 2 + norm_y ** 2 <= 1  # standard ellipse membership test
    if verbose:
        print(f"[DEBUG] elliptical_mask: generated {mask.sum()} True cell(s) out of {mask.size} for shape {region_shape}.")
    return mask


# --------------------------
# ---- Rotate a boolean mask about a fixed center ----
def rotate_mask(
        mask: np.ndarray,  # boolean mask to rotate
        angle: float,  # rotation angle, degrees, standard math/CCW-positive convention
        center: Optional[Tuple[float, float]] = None,  # (row, col) pivot; defaults to the mask's own N//2 center
        verbose: bool = False,  # print a debug summary of the rotation when True
) -> np.ndarray:
    """
    GOAL:
        Rotate a boolean mask about a fixed pivot, preserving the original
        array shape.

    HOW IT TALKS TO OTHER FILES:
        Called by apply_mask (this file) whenever a region's mask_mode is
        "file", using the angle carried on that region's RegionSpec.

    PARAMETERS:
        mask (np.ndarray): boolean array to rotate.
        angle (float): rotation angle in degrees. Follows the same
            convention as every other angle in this pipeline: standard
            math/CCW-positive, 0 degrees = +x, matching geometry.py's
            device angles and superposition_operations.rotate_field_spatial.
        center (Tuple[float, float], optional): (row, col) cell index to
            rotate about. Defaults to (rows // 2, cols // 2), matching the
            cell-center convention used everywhere else in this pipeline
            (extract_region's placement, RegionSpec.center, geometry.py
            device centers) rather than scipy's own continuous-domain
            midpoint, which is offset by half a cell for even-sized arrays.
            Pass an explicit center only for a non-default pivot.
        verbose (bool): prints the mask's cell count before and after
            rotation when True (a large change can indicate an angle or
            pivot issue worth checking).

    RETURNS:
        np.ndarray: boolean array, same shape as `mask`, rotated by `angle`
        about `center`.

    LIMITATIONS:
        Rotation is performed with bilinear interpolation on the mask
        treated as a continuous 0-1 coverage field, then thresholded back
        to boolean at 50% coverage. This is more accurate than
        nearest-neighbor resampling (which independently re-aliases an
        already-rasterized boundary), but the result is still a boolean
        mask on the same cell grid, so a perfectly smooth boundary is not
        achievable at small feature sizes; some staircase pattern along the
        rotated edge is an inherent limit of the cell size, not an artifact
        of this function.
    """
    rows, cols = mask.shape  # mask dimensions, needed for the default pivot below
    if center is None:  # no explicit pivot given
        center = (rows // 2, cols // 2)  # default to the mask's own cell-center, matching this pipeline's convention elsewhere
    theta = np.deg2rad(angle)  # convert the requested angle to radians for use with cos/sin
    row_c, col_c = center  # unpack the pivot into its row/column components
    cos_t, sin_t = np.cos(theta), np.sin(theta)  # precompute the rotation matrix's trigonometric terms
    rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])  # standard 2D rotation matrix, applied in (row, col) space to match this pipeline's CCW-positive convention (see function LIMITATIONS)
    c = np.array([row_c, col_c], dtype=float)  # pivot as a coordinate vector
    offset = c - rot_matrix @ c  # affine_transform offset so the pivot maps to itself
    rotated = _affine_transform(  # perform the rotation via scipy's affine transform
        mask.astype(float),  # cast to float so bilinear interpolation can blend coverage between 0 and 1
        rot_matrix,  # the rotation matrix computed above
        offset=offset,  # keeps the pivot cell fixed in place
        order=1,  # bilinear interpolation; see function LIMITATIONS for why this is used instead of nearest-neighbor
        mode='constant',  # cells rotated in from outside the array are filled with a constant value
        cval=0.0  # that constant fill value: 0.0, meaning "outside the mask"
    )
    result = rotated >= 0.5  # threshold the blended coverage back to a clean boolean mask at 50% coverage
    if verbose:
        print(f"[DEBUG] rotate_mask: {mask.sum()} True cell(s) before rotation, {result.sum()} after rotating {angle} degrees.")
    return result


# --------------------------
# ---- Translate a boolean mask by a fixed pixel offset ----
def translate_mask(
        mask: np.ndarray,  # boolean mask to translate
        shift: Tuple[int, int],  # (shift_y, shift_x) displacement, in cells
        verbose: bool = False,  # print a debug summary of the translation when True
) -> np.ndarray:
    """
    GOAL:
        Shift a boolean mask by a fixed number of cells, preserving the
        original array shape (cells shifted in from outside the array are
        filled as False).

    HOW IT TALKS TO OTHER FILES:
        Called by apply_mask (this file), after rotate_mask, whenever a
        region's mask_mode is "file" and translation is enabled on that
        region's RegionSpec.

    PARAMETERS:
        mask (np.ndarray): boolean array to translate.
        shift (Tuple[int, int]): (shift_y, shift_x) displacement in cells.
        verbose (bool): prints the mask's cell count before and after the
            shift when True.

    RETURNS:
        np.ndarray: boolean array, same shape as `mask`, shifted by `shift`.

    LIMITATIONS:
        Uses nearest-neighbor resampling, which is exact for integer-cell
        shifts. A shift with a fractional cell component will be rounded
        rather than sub-cell-interpolated.
    """
    shift_row_col = (shift[1], shift[0])  # convert (shift_y, shift_x) into (shift_row, shift_col), the order scipy's shift function expects
    translated = _nd_shift(  # perform the translation via scipy's shift function
        mask.astype(float),  # cast to float for consistency with the underlying transform
        shift=shift_row_col,  # displacement computed above
        order=0,  # nearest-neighbor: exact for integer-cell shifts (see function LIMITATIONS)
        mode='constant',  # cells shifted in from outside the array are filled with a constant value
        cval=0.0  # that constant fill value: 0.0, meaning "outside the mask"
    )
    result = translated.astype(bool)  # cast back to a boolean mask
    if verbose:
        print(f"[DEBUG] translate_mask: {mask.sum()} True cell(s) before shift, {result.sum()} after shifting by {shift}.")
    return result


def _zero_vectors_to_nan(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    """
    GOAL:
        Convert any cell where all three vector components are exactly zero
        into NaN, in place, across all three arrays, so that cells with no
        real field/magnetization data do not skew downstream statistics or
        color scales (a true (0, 0, 0) vector is treated the same as "no
        data" throughout this pipeline).

    HOW IT TALKS TO OTHER FILES:
        Called only from apply_mask, immediately after masking, in this
        file. Not called from outside this file.

    PARAMETERS:
        x, y, z (np.ndarray): the three vector component arrays, modified
            in place. Must share the same shape.

    RETURNS:
        None. x, y, and z are modified in place.
    """
    zero_vectors = (x == 0) & (y == 0) & (z == 0)  # cells where every component is exactly zero
    x[zero_vectors] = np.nan  # mark those cells as missing data in x
    y[zero_vectors] = np.nan  # mark those cells as missing data in y
    z[zero_vectors] = np.nan  # mark those cells as missing data in z


# --------------------------
# ---- Shared edge-clipping detection + NaN-padded crop (usable by any caller) ----
def _resolve_grid_center_size(center, size, units, cell_size_x, cell_size_y):
    """
    GOAL:
        Convert a RegionSpec-style (center, size) pair into plain grid-cell
        integers, regardless of whether it was given in 'grid' or 'nm'
        units -- the same conversion region_extraction.extract_region
        performs internally, factored out here so region_crosses_edge and
        crop_with_nan_padding's own callers can both use it without
        duplicating the unit-conversion arithmetic.

    HOW IT TALKS TO OTHER FILES:
        Called by region_crosses_edge (this file). Not called from outside
        this file -- external callers should call region_crosses_edge or
        crop_with_nan_padding directly instead.

    PARAMETERS:
        center, size: as in region_extraction.extract_region.
        units, cell_size_x, cell_size_y: as in region_extraction.
            extract_region.

    RETURNS:
        Tuple[int, int, int, int]: (cx, cy, width_cells, height_cells), all
        in grid cells.
    """
    if units == 'nm':
        if cell_size_x is None or cell_size_y is None:
            raise ValueError("cell_size_x and cell_size_y must be provided when units='nm'")
        cx_nm, cy_nm = center
        width_nm, height_nm = size
        return int(cx_nm / cell_size_x), int(cy_nm / cell_size_y), int(width_nm / cell_size_x), int(height_nm / cell_size_y)
    return center[0], center[1], size[0], size[1]


def region_crosses_edge(
    source_shape: Tuple[int, int],  # (rows, cols) of the full-frame array this region would be cropped from -- e.g. matrix_components[0].shape
    center: Tuple[float, float],  # this region's own center, in `units`
    size: Tuple[float, float],  # this region's own (width, height), in `units`
    units: str = 'grid',  # 'grid' or 'nm', matching region_extraction.extract_region's own convention
    cell_size_x: Optional[float] = None,  # required if units='nm'
    cell_size_y: Optional[float] = None,  # required if units='nm'
    margin_cells: int = 0,  # tolerate up to this many cells of overhang on each side before reporting a crossing -- 0 (default) flags ANY overhang, even a single line of cells
    verbose: bool = False,  # print the resolved crop bounds and overhang amounts when True
) -> Tuple[bool, Dict[str, int]]:
    """
    GOAL:
        Report whether a region's own requested crop window (center +/-
        half-size) would extend past its source array's edge -- WITHOUT
        performing any crop itself, so this is cheap enough to call as a
        pre-check before compositing/extracting anything (see, for example,
        propagation_engine.py's own candidate-discovery step, which calls
        this once per candidate before ever recomposing a canvas for it).

    HOW IT TALKS TO OTHER FILES:
        Pure geometry/bookkeeping; no other file dependencies beyond
        _resolve_grid_center_size (this file). Used by BOTH pipelines in
        this codebase wherever a caller wants to know about edge-clipping
        risk before or instead of relying on apply_mask's own automatic
        NaN-padding (see crop_with_nan_padding, below, and this file's own
        module-level PURPOSE section) -- for example, propagation_engine.py
        uses this to decide whether to drop a candidate outright, while
        mumax3_parent_superposition.py's STEP 6 uses it only to print a
        warning (the underlying NaN-padding already keeps the resulting
        stats correct either way; this function is about SURFACING the
        situation to a human, not preventing bad data).

    PARAMETERS:
        source_shape (Tuple[int, int]): (rows, cols) of the full-frame
            array this region would be cropped from.
        center, size, units, cell_size_x, cell_size_y: identical meaning to
            region_extraction.extract_region's own parameters of the same
            names.
        margin_cells (int): tolerate this many cells of overhang on each
            side before reporting a crossing. 0 (default) means even a
            single line of cells overhanging the edge is reported.
        verbose (bool): prints the resolved crop bounds and per-side
            overhang amounts when True.

    RETURNS:
        Tuple[bool, Dict[str, int]]:
            crosses_edge (bool): True if the requested window extends past
                source_shape's bounds by more than margin_cells on any
                side.
            overhang (Dict[str, int]): {"top", "bottom", "left", "right"}
                -- how many cells past the edge on each side (0 if that
                side does not overhang), regardless of margin_cells (so a
                caller can inspect the exact amount even when it chooses to
                tolerate it).
    """
    rows, cols = source_shape
    cx, cy, width_cells, height_cells = _resolve_grid_center_size(center, size, units, cell_size_x, cell_size_y)
    half_width, half_height = width_cells // 2, height_cells // 2

    r0, r1 = cy - half_height, cy - half_height + height_cells
    c0, c1 = cx - half_width, cx - half_width + width_cells

    overhang = {
        "top": max(0, -r0),
        "bottom": max(0, r1 - rows),
        "left": max(0, -c0),
        "right": max(0, c1 - cols),
    }
    crosses_edge = any(v > margin_cells for v in overhang.values())

    if verbose:
        print(f"[DEBUG] region_crosses_edge: requested rows[{r0}:{r1}] cols[{c0}:{c1}] against source_shape={source_shape}; "
              f"overhang={overhang}; crosses_edge={crosses_edge} (margin_cells={margin_cells}).")

    return crosses_edge, overhang


def crop_with_nan_padding(arr: np.ndarray, r0: int, r1: int, c0: int, c1: int, verbose: bool = False) -> np.ndarray:
    """
    GOAL:
        Crop [r0:r1, c0:c1] out of `arr`, NaN-padding any part of the
        requested window that falls outside `arr`'s own bounds, so the
        returned array is always exactly (r1 - r0, c1 - c0) regardless of
        whether the requested crop extends past `arr`'s edge. A plain
        slice with a negative start would otherwise silently wrap around
        to the array's opposite edge rather than erroring or padding, and
        region_extraction.extract_region's own 'crop' mode silently
        SHRINKS the returned window instead of padding it (see that
        function's own LIMITATIONS section) -- this function is the
        edge-safe alternative wherever a caller needs a guaranteed exact
        size without either failure mode.

    HOW IT TALKS TO OTHER FILES:
        Called by apply_mask (this file), for every mask mode, whenever
        matrix_components is available -- promoted from a private helper
        local to apply_mask into this top-level, importable function so
        any other caller in this codebase (either pipeline) can get the
        same edge-safe behavior without duplicating this logic. NaN-padding
        (rather than, say, zero-padding) is deliberate: a masked-region
        mean computed with np.nanmean over the result correctly excludes
        the padded cells entirely, rather than biasing the mean toward
        zero -- see this file's module-level PURPOSE section.

    PARAMETERS:
        arr (np.ndarray): the full source array to crop from.
        r0, r1, c0, c1 (int): requested crop bounds (row start/end, column
            start/end) -- may extend past arr's own bounds in either
            direction.
        verbose (bool): prints the source/output shapes and how much (if
            any) padding was needed when True.

    RETURNS:
        np.ndarray: shape exactly (r1 - r0, c1 - c0), dtype float, with NaN
        in every cell whose position falls outside arr's own bounds.
    """
    src = np.asarray(arr, dtype=float)  # ensure a float array so NaN padding is possible
    h, w = src.shape  # source array's actual bounds
    out = np.full((r1 - r0, c1 - c0), np.nan, dtype=float)  # output array, pre-filled with NaN
    sr0, sr1 = max(0, r0), min(h, r1)  # row bounds clamped to the source array
    sc0, sc1 = max(0, c0), min(w, c1)  # column bounds clamped to the source array
    if sr0 < sr1 and sc0 < sc1:  # only copy if the clamped region is non-empty
        out[sr0 - r0:sr1 - r0, sc0 - c0:sc1 - c0] = src[sr0:sr1, sc0:sc1]  # copy the valid portion into the correct position in the output
    if verbose:
        n_nan = int(np.isnan(out).sum())
        print(f"[DEBUG] crop_with_nan_padding: requested rows[{r0}:{r1}] cols[{c0}:{c1}] from source shape {src.shape}; "
              f"output shape {out.shape}, {n_nan} padded (NaN) cell(s).")
    return out


# --------------------------
# ---- Apply mask (none / ellipse / file) to every region in a regions dictionary ----
def apply_mask(
    regions_dict: Dict[str, Dict[str, Any]],  # output of extract_regions, keyed by region label
    region_specs: List["RegionSpec"],  # one spec per region in regions_dict, matched by label
    matrix_components: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,  # full-frame source data; required only for mask_mode='file' with translation enabled
    units: str = 'grid',  # units RegionSpec.center/size are expressed in
    cell_size_x: Optional[float] = None,  # nm per cell, x direction; required only alongside matrix_components
    cell_size_y: Optional[float] = None,  # nm per cell, y direction; required only alongside matrix_components
    verbose: bool = False,  # print a per-region status line when True
) -> Dict[str, Dict[str, Any]]:
    """
    GOAL:
        Apply each region's own mask ("none", "ellipse", or "file", per
        that region's RegionSpec) to the x, y, z components of that region,
        producing masked_x/masked_y/masked_z arrays with NaN in every cell
        the mask excludes.

    HOW IT TALKS TO OTHER FILES:
        - Reads RegionSpec objects (region_extraction.py) to determine each
          region's mask_mode/mask_file/angle/translation.
        - Calls region_extraction.extract_region directly (not
          extract_regions) for the "file" + translation case, to re-crop
          the full-frame source data once the mask has been aligned to a
          region's center.
        - Calls rotate_mask, translate_mask, and elliptical_mask (all in
          this file) to construct each region's actual mask array.
        - Called by frame_pipeline.process_single_frame, once per frame, on
          the output of region_extraction.extract_regions.

    PARAMETERS:
        regions_dict (Dict): output of extract_regions, keyed by each
            spec's resolved_label(), with 'x', 'y', 'z' arrays per region.
        region_specs (List[RegionSpec]): one spec per region in
            regions_dict, matched by spec.resolved_label() against
            regions_dict's keys.
        matrix_components, units, cell_size_x, cell_size_y: the full-frame
            source data. Required for mask_mode='file' with translation=
            True (so the source data can be re-cropped once the mask is
            aligned to that region's center); when given for ANY mask mode
            (including 'none'/'ellipse'), also used to guarantee an exact-
            size, NaN-padded crop for regions that cross the source
            frame's edge, rather than trusting extract_regions' own
            un-padded crop (see region_crosses_edge/crop_with_nan_padding,
            this file, and this file's own module-level LIMITATIONS
            section for the one case not covered by this).
        verbose (bool): prints a summary line (mask modes in use) and, for
            mask_mode='file' regions, a per-region status line, when True.

    RETURNS:
        Dict: regions_dict, updated in place with masked_x, masked_y,
        masked_z fields added to every region.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding mixed
        mask modes across regions in a single call.
    """
    specs_by_label = {spec.resolved_label(): spec for spec in region_specs}  # lookup from region label to its RegionSpec
    mask_data_cache: Dict[str, np.ndarray] = {}  # loaded external mask npz files, keyed by mask_file path, to avoid re-loading the same file for multiple regions
    rotated_mask_cache: Dict[Tuple[str, float], np.ndarray] = {}  # rotated masks, keyed by (mask_file, angle), to avoid re-rotating the same mask/angle pair for multiple regions

    if verbose:
        print(f"[INFO] apply_mask: applying masks to {len(regions_dict)} region(s); "
              f"mode(s) in use: {sorted({s.mask_mode for s in region_specs})}.")

    for region_key, region_data in regions_dict.items():  # process each region independently
        if region_key not in specs_by_label:  # every region must have a matching spec
            raise KeyError(f"No RegionSpec found for region '{region_key}'; "
                            f"region_specs labels must match extract_regions' output.")
        spec = specs_by_label[region_key]  # this region's spec
        mask_mode = spec.mask_mode  # which masking strategy this region uses
        if mask_mode not in ("none", "ellipse", "file"):  # guard against an invalid mode reaching the branches below
            raise ValueError(f"Invalid mask_mode '{mask_mode}' for region '{region_key}'. "
                              "Choose from 'none', 'ellipse', or 'file'.")

        if isinstance(region_data['x'], list) and hasattr(region_data['x'][0], 'shape'):  # multi-frame region data is stored as a list of arrays
            region_shape = region_data['x'][0].shape  # use the first frame's shape as the region shape
        else:
            region_shape = region_data['x'].shape  # single-frame region data is stored as one array directly

        region_data_x, region_data_y, region_data_z = region_data['x'], region_data['y'], region_data['z']  # this region's raw (unmasked) component arrays -- see immediately below for when these get replaced with an edge-safe re-crop

        # ---- Edge-safe re-crop for "none"/"ellipse" modes. extract_regions' own 'crop' mode
        # (which already ran, upstream, to produce region_data['x']/['y']/['z'] above) silently
        # SHRINKS a region that crosses the source frame's edge instead of padding it (see
        # region_extraction.extract_region's own LIMITATIONS section) -- previously, only
        # mask_mode='file' + translation=True got a guaranteed exact-size, NaN-padded re-crop
        # (below); "none"/"ellipse" regions near an edge would just silently come back smaller
        # than requested, with no warning, feeding a smaller-than-intended sample into any stats
        # computed downstream. Re-crop here from matrix_components directly, the same edge-safe
        # way mask_mode='file' already does, whenever matrix_components is available. When it is
        # NOT available (a caller building regions_dict directly, without going through frame_
        # pipeline.process_single_frame, which always supplies it), this falls back to the
        # original (possibly edge-shrunk) region_data as-is. ----
        if mask_mode in ("none", "ellipse") and matrix_components is not None:
            cx, cy, width_cells, height_cells = _resolve_grid_center_size(spec.center, spec.size, units, cell_size_x, cell_size_y)
            half_width, half_height = width_cells // 2, height_cells // 2
            r0, r1 = cy - half_height, cy - half_height + height_cells
            c0, c1 = cx - half_width, cx - half_width + width_cells
            region_data_x = crop_with_nan_padding(matrix_components[0], r0, r1, c0, c1, verbose=verbose)
            region_data_y = crop_with_nan_padding(matrix_components[1], r0, r1, c0, c1, verbose=verbose)
            region_data_z = crop_with_nan_padding(matrix_components[2], r0, r1, c0, c1, verbose=verbose)
            region_shape = region_data_x.shape

        # ---- External mask from file ----
        if mask_mode == "file":
            if spec.mask_file is None:  # a mask file path is required for this mode
                raise ValueError(f"Region '{region_key}': mask_file must be set on the RegionSpec when mask_mode='file'.")
            if spec.mask_file not in mask_data_cache:  # load this mask file once and reuse it for every region that references it
                mask_data_cache[spec.mask_file] = np.load(spec.mask_file, allow_pickle=True)
            mask_data = mask_data_cache[spec.mask_file]  # the loaded mask npz contents

            mask_x_data = mask_data['x']  # the mask template is stored under key 'x'
            base_mask = mask_x_data if mask_x_data.dtype == np.bool_ else (mask_x_data != 0)  # nonzero cells count as "inside the mask" if not already boolean

            cache_key = (spec.mask_file, spec.angle)  # rotated masks are cached per (file, angle) pair
            if cache_key not in rotated_mask_cache:  # rotate once per unique (file, angle) pair, not once per region
                rotated_mask_cache[cache_key] = rotate_mask(base_mask, spec.angle, verbose=verbose)
            mask = rotated_mask_cache[cache_key]  # this region's rotated mask template

            if spec.translation:  # this region's mask must be moved to align with its own center
                if matrix_components is None:  # the full-frame source data is required to re-crop after translation
                    raise ValueError(f"Region '{region_key}': matrix_components must be provided "
                                      "when mask_mode='file' and translation=True.")

                region_data_updated = extract_region(  # re-crop the full-frame source data using the full grid, since spec.center is an absolute grid position
                    matrix_components=matrix_components,  # the full-frame source data
                    center=spec.center,  # this region's absolute grid center
                    size=spec.size,  # this region's intended size
                    units=units,  # units spec.center/size are expressed in
                    cell_size_x=cell_size_x,  # nm per cell, x direction
                    cell_size_y=cell_size_y,  # nm per cell, y direction
                    process_mode='mask'  # returns the full grid unchanged; cropping to the final region size happens later below
                )
                region_data_updated_x = region_data_updated['x']  # full-grid x component, to be cropped down further below
                region_data_updated_y = region_data_updated['y']  # full-grid y component, to be cropped down further below
                region_data_updated_z = region_data_updated['z']  # full-grid z component, to be cropped down further below

                mask_numeric = mask.astype(np.uint8)  # numeric copy of the rotated mask, needed since extract_region operates on numeric arrays
                mask_dict = extract_region(  # re-crop the mask itself to the same full-grid shape before translating it into position
                    matrix_components=(mask_numeric, mask_numeric, mask_numeric),  # the same mask repeated in all three slots, since extract_region expects a 3-component tuple
                    center=spec.center,  # this region's absolute grid center
                    size=spec.size,  # this region's intended size
                    units=units,  # units spec.center/size are expressed in
                    cell_size_x=cell_size_x,  # nm per cell, x direction
                    cell_size_y=cell_size_y,  # nm per cell, y direction
                    process_mode='mask'  # returns the full grid unchanged
                )
                mask = mask_dict['x'].astype(np.bool_)  # the mask, back to boolean, at full-grid shape

                # translate_mask requires a DISPLACEMENT (how far to move the mask template's own
                # center), not an absolute position, so spec.center is converted into an offset from
                # the mask's own center (mask.shape // 2, the same convention rotate_mask pivots on).
                canvas_center_x, canvas_center_y = mask.shape[1] // 2, mask.shape[0] // 2  # the mask template's own center, in its current full-grid shape
                relative_shift = (spec.center[0] - canvas_center_x, spec.center[1] - canvas_center_y)  # how far to move the mask so its center lands on this region's actual center
                mask = translate_mask(mask, relative_shift, verbose=verbose)  # move the mask into position

                masked_x = np.where(mask, region_data_updated_x, np.nan)  # keep x where the mask is True, NaN elsewhere
                masked_y = np.where(mask, region_data_updated_y, np.nan)  # keep y where the mask is True, NaN elsewhere
                masked_z = np.where(mask, region_data_updated_z, np.nan)  # keep z where the mask is True, NaN elsewhere
                _zero_vectors_to_nan(masked_x, masked_y, masked_z)  # also treat true (0,0,0) vectors as missing data

                # The crop below is inflated beyond spec.size when spec.angle != 0, so that a
                # rotated region's corners are not clipped; the amount of inflation is the rotated
                # bounding box of a spec.size rectangle at spec.angle.
                center_x, center_y = spec.center  # this region's absolute grid center
                size_x, size_y = spec.size  # this region's intended (unrotated) size
                if spec.angle != 0:  # inflate the crop size to avoid clipping a rotated shape's corners
                    angle_rad = np.deg2rad(spec.angle)  # convert to radians for the trigonometric calculation below
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)  # precompute the rotated bounding-box terms
                    size_x, size_y = (
                        int(abs(size_x * cos_a) + abs(size_y * sin_a)),  # rotated bounding-box width
                        int(abs(size_x * sin_a) + abs(size_y * cos_a)),  # rotated bounding-box height
                    )
                row_start, row_end = center_y - size_y // 2, center_y - size_y // 2 + size_y  # row bounds of the crop, centered on this region
                col_start, col_end = center_x - size_x // 2, center_x - size_x // 2 + size_x  # column bounds of the crop, centered on this region

                region_data_updated['x'] = crop_with_nan_padding(region_data_updated_x, row_start, row_end, col_start, col_end)  # final-size, NaN-padded x
                region_data_updated['y'] = crop_with_nan_padding(region_data_updated_y, row_start, row_end, col_start, col_end)  # final-size, NaN-padded y
                region_data_updated['z'] = crop_with_nan_padding(region_data_updated_z, row_start, row_end, col_start, col_end)  # final-size, NaN-padded z
                region_data_updated['masked_x'] = crop_with_nan_padding(masked_x, row_start, row_end, col_start, col_end)  # final-size, NaN-padded masked x
                region_data_updated['masked_y'] = crop_with_nan_padding(masked_y, row_start, row_end, col_start, col_end)  # final-size, NaN-padded masked y
                region_data_updated['masked_z'] = crop_with_nan_padding(masked_z, row_start, row_end, col_start, col_end)  # final-size, NaN-padded masked z

                regions_dict[region_key] = region_data_updated  # store the fully processed region
                if verbose:
                    print(f"[INFO] apply_mask: applied external mask to region '{region_key}' (translated); "
                          f"mask shape {mask.shape}, center {spec.center}, angle {spec.angle}.")
                continue  # region_data_updated is already stored; skip the passthrough store below

        # ---- Built-in "none" / "ellipse" modes ----
        elif mask_mode == "none":
            mask = np.ones(region_shape, dtype=bool)  # every cell is included

        else:  # mask_mode == "ellipse"
            a, b = region_shape[1] // 2, region_shape[0] // 2  # ellipse semi-axes sized to fill the region
            mask = elliptical_mask(region_shape, (a, b), verbose=verbose)  # build the elliptical mask

        masked_x = np.where(mask, region_data_x, np.nan)  # keep x where the mask is True, NaN elsewhere
        masked_y = np.where(mask, region_data_y, np.nan)  # keep y where the mask is True, NaN elsewhere
        masked_z = np.where(mask, region_data_z, np.nan)  # keep z where the mask is True, NaN elsewhere
        _zero_vectors_to_nan(masked_x, masked_y, masked_z)  # also treat true (0,0,0) vectors as missing data

        region_data['masked_x'] = masked_x  # store the masked x component on this region
        region_data['masked_y'] = masked_y  # store the masked y component on this region
        region_data['masked_z'] = masked_z  # store the masked z component on this region

    return regions_dict  # every region now has masked_x/masked_y/masked_z fields in addition to its original x/y/z


# --------------------------
# ---- Process external mask file (CSV -> NPZ, or validate NPZ path) ----
def process_external_mask(
    mask_file: str = "mask.csv",  # path to the mask file, either .csv or .npz
    mask_prefix: str = "mask",  # filename prefix used to locate/derive the mask's frame index
    verbose: bool = False,  # print status messages when True
) -> str:
    """
    GOAL:
        Ensure a mask file is available in .npz format, converting it from
        .csv if necessary, and return the .npz path to hand to
        apply_mask/frame_pipeline.process_single_frame.

    HOW IT TALKS TO OTHER FILES:
        Calls mumax3_csv_reader.process_csv_file to perform the actual
        CSV-to-NPZ conversion when needed. Not called by any other file in
        this pipeline directly; intended for use when setting up a
        RegionSpec's mask_file ahead of time.

    PARAMETERS:
        mask_file (str): path to the mask file, ending in either .csv or
            .npz.
        mask_prefix (str): filename prefix expected in the mask filename,
            used to determine the mask's frame index for the CSV-to-NPZ
            conversion. If the prefix is not found in the filename, the
            frame index defaults to 0 and a warning is printed.
        verbose (bool): prints status messages describing which path was
            taken (already NPZ, converting from CSV, etc.) when True.

    RETURNS:
        str: path to the .npz mask file.

    LIMITATIONS:
        Only .csv and .npz mask file extensions are supported; any other
        extension raises a ValueError.
    """
    from mumax3_csv_reader import process_csv_file  # imported here rather than at module top since it is only needed on the .csv conversion path
    import os  # path manipulation

    if verbose:
        print("[INFO] process_external_mask: resolving external mask file...")
    if mask_file is None:  # a mask file path is required
        raise ValueError("mask_file must be provided when mask_mode='file'.")

    if mask_file.lower().endswith('.npz'):  # already in the required format
        if verbose:
            print("[INFO] process_external_mask: mask file is already in NPZ format.")
        return mask_file

    if not mask_file.lower().endswith('.csv'):  # only .csv and .npz are supported
        raise ValueError("mask_file must be either a .csv or .npz file.")

    if verbose:
        print("[INFO] process_external_mask: mask file is a CSV; converting to NPZ format.")
    mask_filename = os.path.splitext(os.path.basename(mask_file))[0]  # mask filename without its extension
    mask_dir = os.path.dirname(mask_file)  # directory containing the mask CSV
    mask_dir_npz = os.path.join(mask_dir, "npz")  # destination directory for the converted NPZ

    if mask_prefix and mask_prefix in mask_filename:  # the filename follows the expected "{prefix}{index}" pattern
        mask_idx_str = mask_filename.split(mask_prefix)[-1].lstrip('_')  # extract the index portion of the filename
        try:
            mask_index = int(mask_idx_str)  # parse the extracted index as an integer
        except ValueError:
            raise ValueError(f"Could not extract valid index from mask filename: {mask_filename}")
    else:
        print(f"[WARNING] process_external_mask: mask filename does not contain expected prefix "
              f"'{mask_prefix}': {mask_filename}. Defaulting to index 0.")
        mask_index = 0  # fall back to index 0 when the expected prefix is not present

    process_csv_file((mask_index, mask_file, mask_prefix, mask_dir_npz), float_dtype=np.float64)  # perform the actual CSV-to-NPZ conversion
    mask_file_npz = os.path.join(mask_dir_npz, f"{mask_prefix}{mask_index:06d}.npz")  # path the conversion wrote its output to
    if verbose:
        print(f"[INFO] process_external_mask: mask CSV converted to NPZ format at '{mask_file_npz}'.")
    return mask_file_npz
