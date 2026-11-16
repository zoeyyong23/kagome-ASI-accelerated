r"""
File Name: region_extraction.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v4

============================================================
PURPOSE
============================================================
Provides region-cropping and region-structuring functions for mumax3
vector-field frames. These functions take a full-frame (x, y, z) triple and
produce per-region dicts, either cropped to a window (extract_region /
extract_regions) or passed through unchanged (structure_regions_full). No
masking decisions happen in this file -- see masking.py for that.

This file also defines RegionSpec, the single descriptor used everywhere in
this pipeline to describe one region: its center, size, and its own
mask_mode/mask_file/angle/translation. Whether a region came from a
manually specified center list (plain analysis) or from geometry.py's
DeviceArray (superposition device centers/angles), it is converted into a
List[RegionSpec] before reaching extract_regions, masking.apply_mask, or
frame_pipeline.process_frames_parallel -- none of those functions need to
know which case they are in.

============================================================
CENTER CONVENTION
============================================================
A requested center (cx, cy) is placed at local array index
(width_cells // 2, height_cells // 2) within a cropped region. This matches
the convention used everywhere else in this pipeline for "center"
(top-level scripts' sim_size_x / cell_size_x / 2, geometry.py device
centers) and matches mumax3's own cell semantics, where cell i occupies
physical space [i, i + 1) * cell_size, making cell index N // 2
unambiguously "the center cell" for an N-cell axis.

============================================================
LIMITATIONS
============================================================
- structure_regions_full is not currently called by any other file in this
  pipeline. Its behavior (pass a full frame through unchanged, keyed by
  center) is available through extract_regions instead, using a RegionSpec
  whose size equals the full domain and mask_mode="none". It remains in
  this file for reference but should not be assumed to be part of the
  active pipeline.
- resolve_dynamic_center only resolves a fixed (cx, cy) position today. It
  accepts a wider range of input shapes (callables, lists, dicts) intended
  for a future moving center (for example, propagation-front tracking), but
  raises NotImplementedError for all of those for now, since
  frame_pipeline.process_frames_parallel applies one fixed region_specs
  list to every frame in a run and has no mechanism yet for a region that
  changes position between frames.
"""

from dataclasses import dataclass, field  # for the RegionSpec descriptor below
from typing import Tuple, List, Dict, Union, Optional  # type hints used throughout this file's signatures
import numpy as np  # array operations used when cropping/stacking region data


# --------------------------
# ---- Unified per-region descriptor ----
@dataclass
class RegionSpec:
    """
    GOAL:
        Bundle everything needed to extract and mask ONE region from ONE
        frame into a single object, instead of threading several
        separately indexed parallel lists (centers[i], sizes[i], angles[i],
        ...) that must be kept in sync by position.

    HOW IT TALKS TO OTHER FILES:
        Consumed by extract_regions/extract_region (this file),
        masking.apply_mask, and frame_pipeline.process_single_frame /
        process_frames_parallel. All of those functions accept a plain
        List[RegionSpec] and do not need to know whether the specs came
        from a manually specified center list or from geometry.py's
        DeviceArray.

    FIELDS:
        center (Tuple[int, int]): (cx, cy), the region's center in grid
            cells.
        size (Tuple[int, int]): (width, height), the region's size in grid
            cells.
        mask_mode (str): 'none', 'ellipse', or 'file'.
        mask_file (str, optional): required when mask_mode == 'file'.
        angle (float): region rotation in degrees, used for 'file' masks.
        translation (bool): whether to re-align/re-crop a 'file' mask to
            `center`.
        label (str, optional): defaults to f"center_{cx}_{cy}" via
            resolved_label() if not given explicitly.

    LIMITATIONS:
        mask_mode/mask_file/angle/translation are per-region fields, but
        today the constructors in this file (build_region_specs) and both
        parent scripts still typically apply one shared mask_mode to every
        region built in a single call. masking.apply_mask already honors
        each RegionSpec's own mask_mode independently, so a region_specs
        list mixing modes across regions works if constructed by hand; see
        this file's module-level PURPOSE section and masking.py's
        LIMITATIONS for the remaining wiring needed to build one from
        user-facing parameters.
    """
    center: Tuple[int, int]  # (cx, cy) in grid units
    size: Tuple[int, int]  # (width, height) in grid units
    mask_mode: str = "none"  # 'none' | 'ellipse' | 'file'
    mask_file: Optional[str] = None  # required if mask_mode == 'file'
    angle: float = 0.0  # region rotation, degrees (used for 'file' masks)
    translation: bool = False  # re-align/re-crop mask to `center` (used for 'file' masks)
    label: Optional[str] = None  # defaults to f"center_{cx}_{cy}" if not given

    def resolved_label(self) -> str:
        """
        GOAL:
            Return this region's label: the explicit `label` if one was
            given, otherwise f"center_{cx}_{cy}".

        HOW IT TALKS TO OTHER FILES:
            Called throughout this pipeline anywhere a region needs to be
            looked up by name (extract_regions' output keys, masking.
            apply_mask, plotting/stats output paths).

        PARAMETERS:
            None (uses this instance's own center/label fields).

        RETURNS:
            str: this region's resolved label.
        """
        return self.label if self.label is not None else f"center_{self.center[0]}_{self.center[1]}"  # explicit label wins; otherwise derive one from center


# ---- Human-readable display name for each mask_mode, used for output folder naming
# (mumax3_csv_plot.py / mumax3_csv_stats.py's mask-type subfolder) and for the
# auto-disambiguation in build_region_specs below. Extend this dictionary, rather
# than adding scattered string literals elsewhere, when a new mask_mode is added.
MASK_TYPE_DISPLAY_NAMES = {
    "none": "No Mask",  # display name for mask_mode="none"
    "ellipse": "Elliptical Mask",  # display name for mask_mode="ellipse"
    "file": "File Mask",  # display name for mask_mode="file"
}


def mask_type_display_name(mask_mode: str, verbose: bool = False) -> str:
    """
    GOAL:
        Return a human-readable, folder-name-safe label for a given
        mask_mode, for use in output paths and auto-disambiguated region
        labels.

    HOW IT TALKS TO OTHER FILES:
        Called by build_region_specs and build_region_meta (this file), and
        by mumax3_csv_plot.py / mumax3_csv_stats.py when constructing
        output folder names.

    PARAMETERS:
        mask_mode (str): one of 'none', 'ellipse', 'file', or any other
            string (falls back to a title-cased version of the string
            itself).
        verbose (bool): prints the resolved display name when True.

    RETURNS:
        str: the display name for `mask_mode`.
    """
    result = MASK_TYPE_DISPLAY_NAMES.get(mask_mode, str(mask_mode).title())  # look up the known display name, or title-case the raw string as a fallback
    if verbose:
        print(f"[DEBUG] mask_type_display_name: '{mask_mode}' -> '{result}'.")
    return result


def build_region_specs(
    centers: List[Tuple[int, int]],  # one (cx, cy) per region, in grid cells
    sizes: List[Tuple[int, int]],  # one (width, height) per region, in grid cells
    mask_mode: Union[str, List[str]] = "none",  # single mode for every region, or a per-region list
    mask_file: Union[Optional[str], List[Optional[str]]] = None,  # single file for every region, or a per-region list
    angles: Optional[List[float]] = None,  # per-region rotation angles, degrees; defaults to all zero
    translation: bool = False,  # applied to every region built by this call
    labels: Optional[List[str]] = None,  # explicit per-region labels; auto-generated if not given
    verbose: bool = False,  # print a summary of the constructed specs when True
) -> List[RegionSpec]:
    """
    GOAL:
        Build a List[RegionSpec] from parallel lists -- the common case for
        a manually specified center list (plain analysis) or
        DeviceArray.device_centers/.device_angles (superposition).
        mask_mode/mask_file may be given once (applied to every region) or
        as a per-region list.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts (mumax3_parent_spatial_analysis.py,
        mumax3_parent_superposition.py) to build the region_specs list
        handed to frame_pipeline.process_frames_parallel. Produces
        RegionSpec objects (this file).

    PARAMETERS:
        centers, sizes: parallel lists, one entry per region.
        mask_mode, mask_file: a single value applied to every region, or a
            list with one entry per region.
        angles: per-region rotation angle in degrees; defaults to zero for
            every region if not given.
        translation: applied identically to every region built by this
            call.
        labels: explicit per-region labels. If not given, each region's
            label defaults to f"center_{cx}_{cy}", with automatic
            disambiguation applied where needed (see LIMITATIONS).
        verbose (bool): prints the number of regions built and flags any
            labels that were automatically disambiguated, when True.

    RETURNS:
        List[RegionSpec]: one spec per region, each with a guaranteed
        unique resolved_label().

    LIMITATIONS:
        If `labels` is not given, two regions that share the same center
        but use different mask_mode values would otherwise resolve to the
        identical default label (f"center_{cx}_{cy}"), and since that label
        is the dictionary key used at every later stage of this pipeline,
        only the last such region's data would survive. This function
        detects that collision and appends each colliding region's mask
        type to its label (for example, "center_1080_1150_File_Mask",
        "..._No_Mask") so every region keeps distinct data. Regions that do
        not collide keep the plain "center_{cx}_{cy}" label unchanged, so
        this does not affect existing cached data or label lookups that
        assume the plain form. Explicit labels passed in `labels` are never
        modified, even if they collide with each other.
    """
    n = len(centers)  # number of regions being built
    if len(sizes) != n:  # centers and sizes must describe the same regions
        raise ValueError("centers and sizes must be the same length.")
    mask_modes = mask_mode if isinstance(mask_mode, list) else [mask_mode] * n  # expand a single mask_mode into one per region
    mask_files = mask_file if isinstance(mask_file, list) else [mask_file] * n  # expand a single mask_file into one per region
    angles = angles if angles is not None else [0.0] * n  # default every region's angle to zero if not given
    explicit_labels = labels if labels is not None else [None] * n  # default every region's label to "not given" if not provided
    if not (len(mask_modes) == len(mask_files) == len(angles) == len(explicit_labels) == n):  # every per-region list must describe the same number of regions
        raise ValueError("centers, sizes, mask_mode, mask_file, angles, and labels must all be the same length.")

    # Resolve default labels first (matching what RegionSpec.resolved_label() would
    # produce on its own), so collisions among the auto-generated labels can be detected.
    default_labels = [
        explicit_labels[i] if explicit_labels[i] is not None else f"center_{centers[i][0]}_{centers[i][1]}"  # explicit label wins; otherwise derive one from this region's center
        for i in range(n)
    ]
    from collections import Counter  # used to detect which default labels repeat
    label_counts = Counter(default_labels)  # how many regions share each default label
    resolved_labels = list(default_labels)  # working copy that gets adjusted below where needed
    disambiguated_count = 0  # tracked only for the verbose summary at the end
    for i in range(n):
        if explicit_labels[i] is not None:  # a user-supplied label is never auto-modified
            continue
        if label_counts[default_labels[i]] > 1:  # this region's default label collides with another region's
            resolved_labels[i] = f"{default_labels[i]}_{mask_type_display_name(mask_modes[i]).replace(' ', '_')}"  # disambiguate using this region's own mask type
            disambiguated_count += 1

    # In the unlikely case that the same mask_mode also repeats at the same center (for
    # example, two identical file-mask regions with no explicit labels), the disambiguation
    # above still collides -- append each region's own position in the list as a final
    # fallback so every RegionSpec ends up with a guaranteed unique label.
    final_counts = Counter(resolved_labels)  # recount after the first round of disambiguation
    seen = Counter()  # tracks how many times each still-colliding label has been seen so far
    for i in range(n):
        if final_counts[resolved_labels[i]] > 1:  # still colliding after mask-type disambiguation
            seen[resolved_labels[i]] += 1  # this is the Nth region with this label
            resolved_labels[i] = f"{resolved_labels[i]}_{seen[resolved_labels[i]]}"  # append that count to make it unique

    if verbose:
        print(f"[INFO] build_region_specs: built {n} region(s); {disambiguated_count} label(s) auto-disambiguated.")

    return [
        RegionSpec(
            center=centers[i], size=sizes[i],  # this region's position and size
            mask_mode=mask_modes[i], mask_file=mask_files[i],  # this region's masking configuration
            angle=angles[i], translation=translation, label=resolved_labels[i],  # this region's rotation, translation flag, and final label
        )
        for i in range(n)
    ]


def resolve_dynamic_center(center_spec, t: int = 0, verbose: bool = False) -> Tuple[int, int]:
    """
    GOAL:
        Resolve a zoom-mode center specification (see
        build_canvas_zoom_centers_sizes) to a single (cx, cy) absolute grid
        position.

    HOW IT TALKS TO OTHER FILES:
        Called by build_canvas_zoom_centers_sizes (this file), which is in
        turn called from mumax3_parent_superposition.py's canvas-level
        analysis step.

    PARAMETERS:
        center_spec: a plain (cx, cy) tuple today. Also accepts a callable
            center_spec(t) -> (cx, cy), a list of per-frame (cx, cy)
            tuples, or a dict {frame_index -> (cx, cy)}, intended for a
            future moving center (see this file's module-level
            LIMITATIONS), but all three of those currently raise
            NotImplementedError.
        t (int): frame index; unused today since only a fixed center is
            supported, but accepted so the calling code does not need to
            change once a moving center is implemented.
        verbose (bool): prints the resolved (cx, cy) position when True.

    RETURNS:
        Tuple[int, int]: (cx, cy), rounded to the nearest integer grid cell.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section: only a fixed
        center is currently supported.
    """
    if callable(center_spec):  # a moving/per-frame center function was passed
        raise NotImplementedError(
            "Callable (moving/per-frame) zoom centers aren't supported yet -- "
            "frame_pipeline.py still crops one fixed box per region for the whole "
            "time series. Pass a plain (cx, cy) tuple for now."
        )
    if isinstance(center_spec, dict):  # a per-frame dict of centers was passed
        raise NotImplementedError(
            "Per-frame dict zoom centers aren't supported yet -- see "
            "resolve_dynamic_center's docstring."
        )
    if isinstance(center_spec, list):  # a per-frame list of centers was passed
        raise NotImplementedError(
            "Per-frame list zoom centers aren't supported yet -- see "
            "resolve_dynamic_center's docstring."
        )
    cx, cy = center_spec  # a plain fixed (cx, cy) tuple
    result = (int(round(cx)), int(round(cy)))  # round to the nearest integer grid cell
    if verbose:
        print(f"[DEBUG] resolve_dynamic_center: resolved to {result}.")
    return result


def build_canvas_split_centers_sizes(
    canvas_size: Tuple[int, int],  # (width, height) of the full canvas, in grid cells
    canvas_center: Tuple[int, int],  # (cx, cy) absolute grid position of the canvas center
    split: Optional[int] = None,  # None/1 = single tile covering the whole canvas; N (>1) = an NxN grid of tiles
    label_prefix: str = "canvas_split",  # prefix used to build each tile's region label
    verbose: bool = False,  # print the number of tiles built when True
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[str]]:
    """
    GOAL:
        Build (centers, sizes, labels) for tiling a full canvas into an NxN
        grid of equal regions -- for example, a canvas split into quarters
        or eighths for a zoomed-in view of each quadrant.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_parent_superposition.py's canvas-level analysis
        step. Its output is passed directly into
        SpatialAnalysisConfig(centers=..., sizes=..., region_labels=...)
        alongside mask_mode="none" (a canvas tile is not one device, so
        there is no per-device mask to apply to it).

    PARAMETERS:
        canvas_size (Tuple[int, int]): (width, height) of the full canvas,
            in grid cells.
        canvas_center (Tuple[int, int]): (cx, cy) absolute grid position of
            the canvas center.
        split (int, optional): None or 1 tiles the canvas as a single
            region covering the whole area. N (greater than 1) tiles the
            canvas into an NxN grid, each tile sized
            (canvas_width // N, canvas_height // N).
        label_prefix (str): prefix for each tile's region label; tiles are
            labeled "{label_prefix}_1_{N}_row{r}_col{c}" (row 0 is the
            bottom row, matching this pipeline's plotting convention
            elsewhere).
        verbose (bool): prints the number of tiles built when True.

    RETURNS:
        Tuple[List, List, List]: (centers, sizes, labels), three parallel
        lists in grid cells, ready to hand directly to build_region_specs
        or a SpatialAnalysisConfig.

    LIMITATIONS:
        Every tile is the same size; this function does not support
        non-uniform tiling.
    """
    cw, ch = canvas_size  # full canvas width/height, in cells
    cx0, cy0 = canvas_center  # full canvas center, absolute grid position
    if split is None or split <= 1:  # no splitting requested: one region covering the whole canvas
        if verbose:
            print("[INFO] build_canvas_split_centers_sizes: split is None/1; returning a single full-canvas region.")
        return [(cx0, cy0)], [(cw, ch)], [f"{label_prefix}_full"]

    tile_w = cw // split  # width of each tile
    tile_h = ch // split  # height of each tile
    if tile_w < 1 or tile_h < 1:  # the requested split is too fine for this canvas size
        raise ValueError(f"canvas_size {canvas_size} is too small to split into 1/{split} tiles.")

    origin_x = cx0 - cw // 2  # left edge of the canvas, absolute grid position
    origin_y = cy0 - ch // 2  # bottom edge of the canvas, absolute grid position

    centers, sizes, labels = [], [], []  # accumulated across every tile below
    for row in range(split):  # walk each row of the tile grid
        for col in range(split):  # walk each column of the tile grid
            tile_cx = origin_x + col * tile_w + tile_w // 2  # this tile's center, x
            tile_cy = origin_y + row * tile_h + tile_h // 2  # this tile's center, y
            centers.append((tile_cx, tile_cy))  # record this tile's center
            sizes.append((tile_w, tile_h))  # record this tile's size
            labels.append(f"{label_prefix}_1_{split}_row{row}_col{col}")  # record this tile's label
    if verbose:
        print(f"[INFO] build_canvas_split_centers_sizes: built {len(centers)} tile(s) for split={split}.")
    return centers, sizes, labels


def build_canvas_zoom_centers_sizes(
    zoom_targets: Dict[str, Tuple[int, int]],  # {name -> center_spec}, see resolve_dynamic_center
    size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]],  # shared (width, height), or a per-target dict
    label_prefix: str = "canvas_zoom",  # prefix used to build each target's region label
    verbose: bool = False,  # print the number of zoom targets built when True
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[str]]:
    """
    GOAL:
        Build (centers, sizes, labels) for one or more named zoom regions
        -- for example, a fixed-size window around one device of interest,
        as opposed to build_canvas_split_centers_sizes' fixed grid tiling.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_parent_superposition.py's canvas-level analysis
        step, alongside (or instead of) build_canvas_split_centers_sizes.
        Calls resolve_dynamic_center (this file) to resolve each target's
        center.

    PARAMETERS:
        zoom_targets (Dict[str, Tuple[int, int]]): {name -> center_spec}.
            `name` becomes part of the region label (for example,
            "device_8" becomes "canvas_zoom_device_8"). `center_spec` is
            anything resolve_dynamic_center accepts.
        size: (width, height) in grid cells, shared by every target, or a
            {name -> (width, height)} dict for per-target sizes.
        label_prefix (str): prefix for each target's region label.
        verbose (bool): prints the number of zoom targets built when True.

    RETURNS:
        Tuple[List, List, List]: (centers, sizes, labels), same shape and
        usage as build_canvas_split_centers_sizes.
    """
    centers, sizes, labels = [], [], []  # accumulated across every zoom target below
    for name, center_spec in zoom_targets.items():  # walk each named zoom target
        cx, cy = resolve_dynamic_center(center_spec, verbose=verbose)  # resolve this target's center
        this_size = tuple(size[name]) if isinstance(size, dict) else tuple(size)  # this target's own size, or the shared size
        centers.append((cx, cy))  # record this target's center
        sizes.append(this_size)  # record this target's size
        labels.append(f"{label_prefix}_{name}")  # record this target's label
    if verbose:
        print(f"[INFO] build_canvas_zoom_centers_sizes: built {len(centers)} zoom target(s).")
    return centers, sizes, labels


def build_region_meta(region_specs: List["RegionSpec"], verbose: bool = False) -> Dict[str, Dict[str, str]]:
    """
    GOAL:
        Build the {label -> {"center_name", "mask_type", "center_cells"}}
        lookup used to lay out output files as
        .../<center or region name>/<mask type>/... and to place each
        region's plot in a shared coordinate frame when requested.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts immediately after building
        region_specs. Consumed by mumax3_csv_plot.py's run_spatial_plots,
        plot_region_stats_over_time, and compute_region_extent.

    PARAMETERS:
        region_specs (List[RegionSpec]): the regions to build metadata for.
        verbose (bool): prints the number of entries built when True.

    RETURNS:
        Dict[str, Dict[str, str]]: keyed by each region's resolved_label().
        "center_name" is always the plain f"center_{cx}_{cy}" form,
        regardless of any auto-disambiguation suffix build_region_specs may
        have added to the region's actual label -- this keeps the display
        name clean even when multiple mask modes share one center, since
        disambiguation is instead handled by the separate mask-type folder.
        "center_cells" carries the region's absolute grid position, used by
        mumax3_csv_plot.compute_region_extent to place a region's plot in a
        shared coordinate frame (see SpatialAnalysisConfig.plot_origin_cells)
        instead of always starting its own local extent at (0, 0).
    """
    result = {
        spec.resolved_label(): {
            "center_name": f"center_{spec.center[0]}_{spec.center[1]}",  # clean display name, independent of any disambiguation suffix
            "mask_type": mask_type_display_name(spec.mask_mode),  # human-readable mask type, for output folder naming
            "center_cells": spec.center,  # absolute grid position, for shared-coordinate-frame plotting
        }
        for spec in region_specs
    }
    if verbose:
        print(f"[DEBUG] build_region_meta: built metadata for {len(result)} region(s).")
    return result


# --------------------------
# ---- Extract region from vector field matrix ----
def extract_region(
    matrix_components: Tuple[np.ndarray, np.ndarray, np.ndarray],  # (x, y, z) full-frame component arrays
    center: Tuple[int, int],  # (cx, cy), in grid indices or nm depending on `units`
    size: Tuple[int, int],  # (width, height), in grid indices or nm depending on `units`
    units: str = 'grid',  # 'grid' or 'nm'
    cell_size_x: Optional[float] = None,  # nm per cell, x direction; required if units='nm'
    cell_size_y: Optional[float] = None,  # nm per cell, y direction; required if units='nm'
    process_mode: str = 'crop',  # 'crop' to window around `center`, or 'mask' to return the full grid unchanged
    verbose: bool = False,  # print the resulting crop's bounds when True
) -> Dict[str, Union[np.ndarray, Tuple, Dict]]:
    """
    GOAL:
        Extract a rectangular subregion from each component (x, y, z) of a
        vector field matrix, centered on `center`.

    HOW IT TALKS TO OTHER FILES:
        Called by extract_regions (this file) once per RegionSpec, and
        directly by masking.apply_mask (with process_mode='mask') for the
        "file" mask mode with translation enabled.

    PARAMETERS:
        matrix_components (Tuple[np.ndarray, np.ndarray, np.ndarray]): the
            full-frame (x, y, z) component arrays.
        center (Tuple[int, int]): center of the region to extract, in grid
            indices or nm depending on `units`.
        size (Tuple[int, int]): (width, height) of the region to extract,
            in grid indices or nm depending on `units`.
        units (str): 'grid' (default) or 'nm'.
        cell_size_x, cell_size_y (float, optional): grid cell size in nm;
            required if units='nm'.
        process_mode (str): 'crop' windows the array around `center` to
            exactly `size`. 'mask' returns the full matrix unchanged
            (used when a mask will be aligned/translated against the whole
            grid before a final crop happens elsewhere).
        verbose (bool): prints the resolved crop bounds when True.

    RETURNS:
        Dict: 'x'/'y'/'z' cropped components, 'grid_center', 'grid_bounds',
        and (if cell sizes are given) 'physical_bounds_nm'.

    LIMITATIONS:
        For process_mode='crop', the returned window is clipped to the
        source array's actual bounds rather than padded -- a region
        requested near the edge of the source frame will come back smaller
        than `size`. Callers that need a guaranteed exact size regardless
        of edge proximity (for example, a rotated 'file' mask region) apply
        their own NaN-padded crop afterward -- see masking.apply_mask.
    """
    if units not in ('grid', 'nm'):  # only two unit systems are supported
        raise ValueError("Units must be either 'grid' or 'nm'")

    if units == 'nm':  # convert physical units into grid cells before cropping
        if cell_size_x is None or cell_size_y is None:  # cell sizes are required to perform that conversion
            raise ValueError("cell_size_x and cell_size_y must be provided when units='nm'")
        cx_nm, cy_nm = center  # center, in nm
        width_nm, height_nm = size  # size, in nm
        cx = int(cx_nm / cell_size_x)  # center, converted to grid cells, x
        cy = int(cy_nm / cell_size_y)  # center, converted to grid cells, y
        width_cells = int(width_nm / cell_size_x)  # width, converted to grid cells
        height_cells = int(height_nm / cell_size_y)  # height, converted to grid cells
    else:
        cx, cy = center  # center is already in grid cells
        width_cells, height_cells = size  # size is already in grid cells

    half_width = width_cells // 2  # half-width, used to place `cx` at the correct local index (see module CENTER CONVENTION)
    half_height = height_cells // 2  # half-height, used to place `cy` at the correct local index
    Bx, By, Bz = matrix_components  # unpack the full-frame component arrays
    rows, cols = Bx.shape  # full-frame dimensions, used to clip the crop below

    if process_mode == 'crop':
        # r_end/c_end are derived from r_start/c_start plus the exact requested size
        # (rather than from cy/cx + half), so the window is always exactly
        # (height_cells, width_cells) even for odd sizes.
        r_start = max(0, cy - half_height)  # top edge of the crop, clipped to the array
        r_end = min(rows, r_start + height_cells)  # bottom edge of the crop, clipped to the array
        c_start = max(0, cx - half_width)  # left edge of the crop, clipped to the array
        c_end = min(cols, c_start + width_cells)  # right edge of the crop, clipped to the array
    elif process_mode == 'mask':
        r_start, r_end = 0, rows  # 'mask' mode returns the full grid unchanged
        c_start, c_end = 0, cols
    else:
        raise ValueError("process_mode must be 'crop' or 'mask'")

    cropped_data = {
        'x': Bx[r_start:r_end, c_start:c_end],  # cropped x component
        'y': By[r_start:r_end, c_start:c_end],  # cropped y component
        'z': Bz[r_start:r_end, c_start:c_end],  # cropped z component
        'grid_center': (cx, cy),  # the requested center, in grid cells
        'grid_bounds': (r_start, r_end, c_start, c_end)  # the actual crop bounds used
    }
    if cell_size_x is not None and cell_size_y is not None:  # physical bounds can only be reported if cell sizes are known
        x_start_nm = c_start * cell_size_x  # left edge of the crop, in nm
        x_end_nm = c_end * cell_size_x  # right edge of the crop, in nm
        y_start_nm = r_start * cell_size_y  # bottom edge of the crop, in nm
        y_end_nm = r_end * cell_size_y  # top edge of the crop, in nm
        cropped_data['physical_bounds_nm'] = ((x_start_nm, x_end_nm), (y_start_nm, y_end_nm))  # record the physical bounds alongside the grid bounds
    if verbose:
        print(f"[DEBUG] extract_region: process_mode='{process_mode}', bounds rows[{r_start}:{r_end}] cols[{c_start}:{c_end}].")
    return cropped_data


# --------------------------
# ---- Extract multiple regions from vector field matrix ----
def extract_regions(
    matrix_components: Tuple[np.ndarray, np.ndarray, np.ndarray],  # (x, y, z) full-frame component arrays
    region_specs: List[RegionSpec],  # one spec per region to extract
    units: str = 'grid',  # 'grid' or 'nm'
    cell_size_x: Optional[float] = None,  # nm per cell, x direction; required if units='nm'
    cell_size_y: Optional[float] = None,  # nm per cell, y direction; required if units='nm'
    verbose: bool = False,  # print the number of regions extracted when True
) -> Dict[str, Dict[str, Union[np.ndarray, Tuple]]]:
    """
    GOAL:
        Extract multiple subregions from a vector field, one per
        RegionSpec, using extract_region.

    HOW IT TALKS TO OTHER FILES:
        Called by frame_pipeline.process_single_frame, once per frame, with
        the region_specs that frame's parent script built. Calls
        extract_region (this file) once per spec.

    PARAMETERS:
        matrix_components (Tuple[np.ndarray, np.ndarray, np.ndarray]): the
            full-frame (x, y, z) component arrays.
        region_specs (List[RegionSpec]): one spec per region to extract.
        units, cell_size_x, cell_size_y: same meaning as extract_region.
        verbose (bool): prints the number of regions extracted when True.
            Forwarded to extract_region, so per-region debug messages
            appear too if desired.

    RETURNS:
        Dict[str, Dict]: keyed by each spec's resolved_label(), mapping to
        that region's cropped data (see extract_region's return value).
    """
    output = {}  # accumulated across every region spec below
    for spec in region_specs:  # extract one region per spec
        cropped = extract_region(
            matrix_components=matrix_components,  # the full-frame source data
            center=spec.center,  # this region's center
            size=spec.size,  # this region's size
            units=units,  # units center/size are expressed in
            cell_size_x=cell_size_x,  # nm per cell, x direction
            cell_size_y=cell_size_y,  # nm per cell, y direction
            process_mode='crop',  # window around center, unlike masking.py's 'mask'-mode calls
            verbose=verbose,  # forwarded so per-region debug messages appear if desired
        )
        output[spec.resolved_label()] = cropped  # store this region's cropped data under its resolved label
    if verbose:
        print(f"[INFO] extract_regions: extracted {len(output)} region(s).")
    return output


# --------------------------
# ---- Structure full-frame data by center, without cropping (currently unused) ----
def structure_regions_full(
    matrix_components: Tuple[np.ndarray, np.ndarray, np.ndarray],  # (x, y, z) full-frame component arrays
    centers: List[Tuple[int, int]],  # one (cx, cy) per output entry
    sizes: List[Tuple[int, int]],  # accepted for interface consistency with extract_regions; not used for cropping
    units: str = 'grid',  # recorded on the output, not used to crop
    cell_size_x: Optional[float] = None,  # recorded on the output, not used to crop
    cell_size_y: Optional[float] = None,  # recorded on the output, not used to crop
    verbose: bool = False,  # print the number of entries built when True
) -> Dict[str, Dict[str, Union[np.ndarray, Tuple]]]:
    """
    GOAL:
        Structure full vector field data keyed by center, without cropping
        -- intended for the whole-simulation-domain case, where "the
        region" is the entire file.

    HOW IT TALKS TO OTHER FILES:
        Not currently called by any other file in this pipeline; see this
        file's module-level LIMITATIONS section. Retained for reference.

    PARAMETERS:
        matrix_components (Tuple[np.ndarray, np.ndarray, np.ndarray]): the
            full-frame (x, y, z) component arrays.
        centers (List[Tuple[int, int]]): one entry per output key; each
            entry receives the same, uncropped matrix_components.
        sizes (List[Tuple[int, int]]): accepted only to keep this
            function's signature consistent with extract_regions; not used.
        units, cell_size_x, cell_size_y: recorded on each output entry, not
            used to perform any cropping.
        verbose (bool): prints the number of entries built when True.

    RETURNS:
        Dict[str, Dict]: keyed by f"center_{cx}_{cy}" for each entry in
        `centers`, each mapping to the same unmodified matrix_components.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section: this function is
        not part of the active pipeline.
    """
    if len(centers) != len(sizes):  # kept for interface consistency with extract_regions, even though sizes is unused here
        raise ValueError("Length of centers and sizes must match.")

    output = {}  # accumulated across every center below
    for center in centers:  # build one output entry per center
        label = f"center_{center[0]}_{center[1]}"  # this entry's label
        output[label] = {
            'x': matrix_components[0],  # full, uncropped x component
            'y': matrix_components[1],  # full, uncropped y component
            'z': matrix_components[2],  # full, uncropped z component
            'center': center,  # the center this entry is keyed by
            'units': units,  # recorded for reference
            'cell_size_x': cell_size_x,  # recorded for reference
            'cell_size_y': cell_size_y  # recorded for reference
        }
    if verbose:
        print(f"[INFO] structure_regions_full: built {len(output)} entr(y/ies).")
    return output


# --------------------------
# ---- Reorganize regions by label ----
def reorganize_regions(
    region_frames: List[Union[
        Dict[str, Dict[str, np.ndarray]],
        Tuple[int, Dict[str, Dict[str, np.ndarray]]]
    ]],  # one entry per frame: either a region dict directly, or a (frame_index, region dict) pair
    verbose: bool = False,  # print the number of labels/frames reorganized when True
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    GOAL:
        Reorganize region data collected across multiple frames from "one
        dict per frame" into "one dict per region label", stacking each
        component into a single array along the frame axis.

    HOW IT TALKS TO OTHER FILES:
        Called by frame_pipeline.process_frames_parallel, once per chunk of
        processed frames, to produce the shape every plotting/stats
        function downstream expects.

    PARAMETERS:
        region_frames: a list of either a per-frame region dict (as
            returned by extract_regions after masking.apply_mask has added
            masked_x/masked_y/masked_z), or a (frame_index, region_dict)
            tuple.
        verbose (bool): prints the number of region labels and frames
            reorganized when True.

    RETURNS:
        Dict[str, Dict[str, np.ndarray]]: keyed by region label, each value
        holding arrays of shape (frames, height, width) for
        masked_x/masked_y/masked_z/x/y/z.
    """
    if not region_frames:  # nothing to reorganize
        return {}

    first_frame = region_frames[0]  # used only to discover which region labels exist
    regions_dict = first_frame[1] if isinstance(first_frame, (tuple, list)) else first_frame  # unwrap a (frame_index, region_dict) pair if present
    labels = regions_dict.keys()  # every region label present in this batch of frames

    regions: Dict[str, Dict[str, List[np.ndarray]]] = {
        label: {"masked_x": [], "masked_y": [], "masked_z": [], "x": [], "y": [], "z": []}  # one growing list per component, per label
        for label in labels
    }
    for item in region_frames:  # walk every frame in this batch
        frame_regions = item[1] if isinstance(item, (tuple, list)) else item  # unwrap a (frame_index, region_dict) pair if present
        for label, data in frame_regions.items():  # walk every region present in this frame
            regions[label]["masked_x"].append(data.get("masked_x", np.array([])))  # append this frame's masked x (or an empty array if missing)
            regions[label]["masked_y"].append(data.get("masked_y", np.array([])))  # append this frame's masked y
            regions[label]["masked_z"].append(data.get("masked_z", np.array([])))  # append this frame's masked z
            regions[label]["x"].append(data.get("x", np.array([])))  # append this frame's raw x
            regions[label]["y"].append(data.get("y", np.array([])))  # append this frame's raw y
            regions[label]["z"].append(data.get("z", np.array([])))  # append this frame's raw z

    for label in labels:  # convert each label's growing lists into stacked arrays
        for key in ["masked_x", "masked_y", "masked_z", "x", "y", "z"]:
            regions[label][key] = np.array(regions[label][key])  # stack this component's per-frame arrays along a new leading (frame) axis
    if verbose:
        print(f"[INFO] reorganize_regions: reorganized {len(region_frames)} frame(s) into {len(labels)} region label(s).")
    return regions
