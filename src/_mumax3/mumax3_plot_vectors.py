r"""
File Name: mumax3_plot_vectors.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v2

============================================================
PURPOSE
============================================================
Provides reusable, publication-oriented vector/device plotting primitives,
plus frame-sequence stitching into GIF/video. This module is intended to be
used AS SUPPORT by domain-specific scripts, not to replace them -- it owns
only the generic, physics-independent parts:
    - draw_device_segment / draw_device_array_background: draw a device (or
      a whole array of devices) as a line segment along its easy axis.
    - draw_direction_arrow: draw a single directional arrow at a point.
    - quiver_field: draw a subsampled per-cell quiver field over a region.
    - compute_averaged_quiver_grid / averaged_quiver_field: draw one arrow
      per NxM-cell averaging window, mask/edge-aware -- see
      compute_averaged_quiver_grid's own documentation for the edge-case
      rules this follows.
    - new_publication_figure / save_publication_figure: consistent figure
      styling and multi-format (PDF + PNG) export.
    - frames_to_gif / frames_to_video: stitch a sequence of PNG frames into
      an animation.

Every function here takes plain arrays, paths, or region dicts -- nothing
in this file imports DeviceArray, RegionSpec, or any other pipeline-specific
type, so it has no opinion about which pipeline (plain spatial analysis,
superposition, or an external domain-specific script) produced the data it
is asked to plot.

============================================================
LIMITATIONS
============================================================
- Of the functions in this file, only compute_averaged_quiver_grid is
  currently called elsewhere in this pipeline (by mumax3_csv_plot.py's
  "quiver" plot type, which reimplements its own rendering rather than
  calling averaged_quiver_field below). Every other function in this file
  (draw_device_segment, draw_device_array_background, draw_direction_arrow,
  quiver_field, averaged_quiver_field, new_publication_figure,
  save_publication_figure, frames_to_gif, frames_to_video) is not currently
  called by any other file in this pipeline -- they exist as a support
  library for external, domain-specific scripts (for example, a
  propagation-analysis script) that are not part of this pipeline itself.
"""

from pathlib import Path  # path handling for figure/frame output
from typing import Dict, List, Optional, Sequence, Tuple, Union  # type hints used throughout this file's signatures

import numpy as np  # array operations
import matplotlib.pyplot as plt  # figure/axes creation


# =============================================================================
# DEVICE / DIRECTION PRIMITIVES
# =============================================================================

def draw_device_segment(ax, x: float, y: float, angle_deg: float, length_nm: float,
                         color: str = "0.75", linewidth: float = 5.0,
                         alpha: float = 1.0, zorder: int = 1, verbose: bool = False) -> None:
    """
    GOAL:
        Draw one device as a thick line segment along its long/easy axis,
        centered at (x, y).

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline. Called by
        draw_device_array_background (this file), once per device.

    PARAMETERS:
        ax: matplotlib axes to draw on.
        x, y (float): the device's center, in nm.
        angle_deg (float): the device's orientation, in degrees.
        length_nm (float): the device's length, in nm.
        color, linewidth, alpha, zorder: standard matplotlib line styling.
        verbose (bool): prints the segment's endpoints when True.

    RETURNS:
        None. Draws directly onto `ax`.
    """
    theta = np.deg2rad(angle_deg)  # device orientation, in radians
    dx = 0.5 * length_nm * np.cos(theta)  # half-length offset, x
    dy = 0.5 * length_nm * np.sin(theta)  # half-length offset, y
    if verbose:
        print(f"[DEBUG] draw_device_segment: center=({x}, {y}), endpoints=({x-dx}, {y-dy}) to ({x+dx}, {y+dy}).")
    ax.plot(
        [x - dx, x + dx], [y - dy, y + dy],  # the two endpoints of the segment
        color=color, linewidth=linewidth, alpha=alpha,
        solid_capstyle="round", zorder=zorder,
    )


def draw_device_array_background(ax, centers_nm: Sequence[Tuple[float, float]],
                                  angles_deg: Sequence[float], length_nm: float,
                                  color: str = "0.86", linewidth: float = 4.0,
                                  alpha: float = 0.9, zorder: int = 1, verbose: bool = False) -> None:
    """
    GOAL:
        Draw a whole device array as faint background context -- for
        example, the full ASI array behind a highlighted subset of devices.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline. Calls
        draw_device_segment (this file), once per device.

    PARAMETERS:
        ax: matplotlib axes to draw on.
        centers_nm (Sequence[Tuple[float, float]]): one (x, y) center per
            device, in nm.
        angles_deg (Sequence[float]): one orientation per device, in
            degrees, matched by position to `centers_nm`.
        length_nm (float): device length, in nm, shared by every device.
        color, linewidth, alpha, zorder: standard matplotlib line styling.
        verbose (bool): prints the number of devices drawn when True.
            Forwarded to draw_device_segment, so per-device debug messages
            appear too.

    RETURNS:
        None. Draws directly onto `ax`.
    """
    for (x, y), angle in zip(centers_nm, angles_deg):  # one segment per device
        draw_device_segment(ax, float(x), float(y), float(angle), length_nm,
                             color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, verbose=verbose)
    if verbose:
        print(f"[DEBUG] draw_device_array_background: drew {len(centers_nm)} device(s).")


def draw_direction_arrow(ax, x: float, y: float, angle_deg: float, length_nm: float,
                          color: str = "black", width: float = 0.007,
                          headwidth: float = 4.2, headlength: float = 5.8,
                          headaxislength: float = 5.0, zorder: int = 8, verbose: bool = False):
    """
    GOAL:
        Draw a single directional arrow (for example, one device's
        magnetization, or one region's averaged local field) at a point.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline.

    PARAMETERS:
        ax: matplotlib axes to draw on.
        x, y (float): the arrow's origin, in nm.
        angle_deg (float): the arrow's direction, in degrees.
        length_nm (float): the arrow's length, in nm.
        color, width, headwidth, headlength, headaxislength, zorder:
            standard matplotlib quiver styling.
        verbose (bool): prints the arrow's origin and direction when True.

    RETURNS:
        The matplotlib Quiver artist (for example, for use in a legend
        entry).
    """
    theta = np.deg2rad(angle_deg)  # arrow direction, in radians
    if verbose:
        print(f"[DEBUG] draw_direction_arrow: origin=({x}, {y}), angle={angle_deg} degrees, length={length_nm}.")
    return ax.quiver(
        [x], [y], [length_nm * np.cos(theta)], [length_nm * np.sin(theta)],  # single arrow: origin and displacement
        angles="xy", scale_units="xy", scale=1, color=color, width=width,
        headwidth=headwidth, headlength=headlength, headaxislength=headaxislength,
        zorder=zorder,
    )


# =============================================================================
# PER-CELL QUIVER FIELD
# =============================================================================

def _region_component(region: Dict, comp: str, prefer_masked: bool = True, verbose: bool = False) -> np.ndarray:
    """
    GOAL:
        Pull a 2D component array off a region dict, preferring the masked
        version (masked_<comp>) and falling back to the raw version
        (<comp>) when a masked version is not present (a mask_mode='none'
        region only ever has the unmasked keys).

    HOW IT TALKS TO OTHER FILES:
        Called by quiver_field and averaged_quiver_field (this file), once
        per vector component, to extract the u/v arrays to plot.

    PARAMETERS:
        region (Dict): a region dict as returned by
            region_extraction.extract_regions / masking.apply_mask.
        comp (str): which component to extract, for example "x" or "y".
        prefer_masked (bool): if True, use masked_<comp> when present;
            otherwise always use the raw <comp>.
        verbose (bool): prints which key was used and the array's shape
            when True.

    RETURNS:
        np.ndarray: the requested 2D component array. If the underlying
        array has a leading frame axis of size 1, that axis is squeezed
        off.

    LIMITATIONS:
        Raises KeyError if neither masked_<comp> nor <comp> is present in
        `region`.
    """
    masked_key = f"masked_{comp}"  # the preferred, mask-aware key
    if prefer_masked and masked_key in region:  # masked version is available and preferred
        arr = np.asarray(region[masked_key], dtype=float)
        used_key = masked_key
    elif comp in region:  # fall back to the raw, unmasked version
        arr = np.asarray(region[comp], dtype=float)
        used_key = comp
    else:  # neither key is present
        raise KeyError(f"Region has neither '{masked_key}' nor '{comp}'. Keys: {list(region.keys())}")
    result = arr[0] if arr.ndim == 3 else arr  # squeeze a leading frame axis of size 1, if present
    if verbose:
        print(f"[DEBUG] _region_component: used key '{used_key}', shape {result.shape}.")
    return result


def quiver_field(ax, region: Dict, cell_size_x: float = 1.0, cell_size_y: float = 1.0,
                  step: int = 8, field_scaling: float = 1.0, color: str = "black",
                  cmap: Optional[str] = None, width: float = 0.0025,
                  scale: Optional[float] = None, origin_xy: Tuple[float, float] = (0.0, 0.0),
                  prefer_masked: bool = True, zorder: int = 5, verbose: bool = False):
    """
    GOAL:
        Draw a subsampled per-cell quiver field over a masked (or
        unmasked) region, in physical nm coordinates -- showing the actual
        field structure across a region, rather than a single averaged
        arrow per region.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline. Calls _region_component
        (this file) to extract the u/v arrays to plot.

    PARAMETERS:
        ax: matplotlib axes to draw on.
        region (Dict): a region dict as returned by
            region_extraction.extract_regions / masking.apply_mask
            (masked_x/masked_y or x/y, 2D or (1, height, width)).
        cell_size_x, cell_size_y (float): nm per grid cell, for physical
            placement.
        step (int): subsample stride, in cells -- one arrow is drawn every
            `step` cells in each direction, since a dense per-cell quiver
            is usually unreadable.
        field_scaling (float): multiplier applied to the field values (for
            example, Tesla to Oersted) before plotting.
        color, cmap: pass a fixed `color`, or a `cmap` name to color arrows
            by local magnitude instead (cmap takes precedence if both are
            given).
        scale: matplotlib quiver `scale` (None lets matplotlib choose
            automatically).
        origin_xy (Tuple[float, float]): (x0, y0) nm offset of this
            region's own (0, 0) cell within a larger canvas -- set this
            when overlaying multiple regions' quivers on one axes (for
            example, every device in a composed canvas) so each one lands
            in the correct place.
        prefer_masked (bool): use masked_x/masked_y when present (the usual
            case).
        zorder: matplotlib draw-order.
        verbose (bool): prints the number of arrows drawn when True.

    RETURNS:
        The matplotlib Quiver artist.

    LIMITATIONS:
        matplotlib's quiver does not reliably skip NaN entries on its own,
        so NaN cells are filtered explicitly before plotting rather than
        relying on matplotlib to handle them.
    """
    u = _region_component(region, "x", prefer_masked, verbose=verbose) * field_scaling  # x component, scaled
    v = _region_component(region, "y", prefer_masked, verbose=verbose) * field_scaling  # y component, scaled
    h, w = u.shape  # region dimensions, for building the subsampled grid below

    rows = np.arange(0, h, step)  # subsampled row indices
    cols = np.arange(0, w, step)  # subsampled column indices
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")  # every combination of subsampled row/column
    x_nm = origin_xy[0] + col_grid * cell_size_x  # physical x position of each subsampled arrow
    y_nm = origin_xy[1] + row_grid * cell_size_y  # physical y position of each subsampled arrow
    uu = u[row_grid, col_grid]  # x component at each subsampled cell
    vv = v[row_grid, col_grid]  # y component at each subsampled cell

    finite = np.isfinite(uu) & np.isfinite(vv)  # cells with real (non-NaN) data (see function LIMITATIONS)
    x_nm, y_nm, uu, vv = x_nm[finite], y_nm[finite], uu[finite], vv[finite]  # drop NaN cells before plotting
    if verbose:
        print(f"[DEBUG] quiver_field: drawing {finite.sum()} arrow(s) (step={step}).")

    kwargs = dict(angles="xy", scale_units="xy", scale=scale, width=width, zorder=zorder)  # shared quiver styling
    if cmap is not None:  # color arrows by local magnitude
        magnitude = np.hypot(uu, vv)
        return ax.quiver(x_nm, y_nm, uu, vv, magnitude, cmap=cmap, **kwargs)
    return ax.quiver(x_nm, y_nm, uu, vv, color=color, **kwargs)  # fixed-color arrows


# =============================================================================
# AVERAGED QUIVER FIELD -- one arrow per NxM-cell window, instead of one arrow
# per raw cell (quiver_field above) or one arrow per whole region
# (draw_direction_arrow above). Sits between those two.
# =============================================================================

def compute_averaged_quiver_grid(
    u: np.ndarray, v: np.ndarray, window: Tuple[int, int], verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    GOAL:
        Tile a 2D (height, width) vector field into non-overlapping
        `window` = (rows, cols) cell blocks, and average each block's
        valid (non-NaN) cells into one vector -- the building block for "N
        arrows instead of every raw cell", aware of each region's actual
        mask shape and edges (see LIMITATIONS for the edge-handling rules
        this follows, which are the reason this function exists rather
        than simply reshaping into a regular (height // rows,
        width // cols) grid).

    HOW IT TALKS TO OTHER FILES:
        Called by averaged_quiver_field (this file), and directly by
        mumax3_csv_plot.py's "quiver" plot type -- see this file's
        module-level LIMITATIONS section.

    PARAMETERS:
        u, v (np.ndarray): 2D (height, width) field components, NaN where
            masked out.
        window (Tuple[int, int]): (rows, cols) cell-count of each
            averaging block (need not be square -- for example, (2, 8) is
            valid).
        verbose (bool): prints the number of averaging windows produced
            when True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (row_centers, col_centers, avg_u, avg_v) -- four 1D float arrays,
        one entry per window that contained at least one valid cell.
        row_centers/col_centers are fractional CELL indices (the
        geometric center of each window, in this array's own (row, col)
        grid); convert to physical nm the same way quiver_field does
        (origin_xy + col * cell_size_x / row * cell_size_y).

    LIMITATIONS:
        Edge handling:
        - Tiling does not start at cell (0, 0) of the array; it starts at
          the bounding box of wherever the region's VALID (non-NaN) data
          actually is, determined fresh from `u`/`v` themselves on every
          call (not assumed from the region's nominal rectangle or mask
          mode), since the real edge depends on the mask shape, rotation,
          and which device/region this is. The bounding box's starting
          corner (the lowest row index with any valid cell, and the
          lowest column index with any valid cell) is allowed to be an
          empty cell itself -- for example, an ellipse's or a rotated
          device's bounding-box corner is never inside the shape.
        - A window at the far edge of the valid-data bounding box that
          only partially overlaps real data (because it is clipped by the
          array's edge) still produces one arrow, averaged over whichever
          cells in that window are valid. A window with zero valid cells
          (entirely outside the mask) produces no arrow at all.
        - For a region with no NaN gaps at all (for example, a
          rectangular mask_mode='none' region over a field that is
          nonzero and non-NaN everywhere, such as B_demag), the
          bounding-box detection above naturally resolves to the full
          array (every cell counts as valid), which is equivalent to
          using the edge of that rectangular region directly. There is no
          code path here that infers where a physical device sits inside
          a field with no NaN gaps to find it by -- a shaped mask
          (ellipse or file) is required on that region if windows aligned
          to the device's actual footprint (rather than its full
          rectangle) are wanted.
    """
    win_rows, win_cols = window  # unpack the requested averaging window size
    if win_rows < 1 or win_cols < 1:  # a window must cover at least one cell in each dimension
        raise ValueError(f"window must be >= 1 cell in each dimension, got {window}.")

    u = np.asarray(u, dtype=float)  # ensure a plain float array
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:  # u and v must describe the same grid
        raise ValueError(f"u and v must have the same shape, got {u.shape} vs {v.shape}.")
    h, w = u.shape  # grid dimensions, used to clip windows at the array's edge

    valid = np.isfinite(u) & np.isfinite(v)  # cells with real (non-NaN) data in both components
    if not valid.any():  # no valid data anywhere in this array
        empty = np.array([], dtype=float)
        if verbose:
            print("[DEBUG] compute_averaged_quiver_grid: no valid data; returning empty arrays.")
        return empty, empty, empty, empty

    rows_with_valid = np.flatnonzero(valid.any(axis=1))  # every row index that has at least one valid cell
    cols_with_valid = np.flatnonzero(valid.any(axis=0))  # every column index that has at least one valid cell
    row_start, row_end = int(rows_with_valid[0]), int(rows_with_valid[-1]) + 1  # bounding box of valid data, rows
    col_start, col_end = int(cols_with_valid[0]), int(cols_with_valid[-1]) + 1  # bounding box of valid data, columns

    row_centers, col_centers, avg_u, avg_v = [], [], [], []  # accumulated across every window below
    with np.errstate(invalid="ignore"):
        for r0 in range(row_start, row_end, win_rows):  # step through the valid-data bounding box, one window-height at a time
            r1 = min(r0 + win_rows, h)  # clipped to the array's actual edge, not to row_end (see docstring)
            for c0 in range(col_start, col_end, win_cols):  # step through the valid-data bounding box, one window-width at a time
                c1 = min(c0 + win_cols, w)  # clipped to the array's actual edge
                u_win = u[r0:r1, c0:c1]  # this window's x component
                v_win = v[r0:r1, c0:c1]  # this window's y component
                if not (np.isfinite(u_win) & np.isfinite(v_win)).any():  # this window has no usable data at all
                    continue  # no arrow for this window (not a zero-vector arrow)
                row_centers.append((r0 + r1) / 2.0)  # this window's geometric center, row
                col_centers.append((c0 + c1) / 2.0)  # this window's geometric center, column
                avg_u.append(np.nanmean(u_win))  # this window's average x component, ignoring NaN cells
                avg_v.append(np.nanmean(v_win))  # this window's average y component, ignoring NaN cells

    if verbose:
        print(f"[DEBUG] compute_averaged_quiver_grid: produced {len(row_centers)} window(s) from window size {window}.")
    return (np.array(row_centers), np.array(col_centers), np.array(avg_u), np.array(avg_v))


def averaged_quiver_field(ax, region: Dict, window: Tuple[int, int],
                           cell_size_x: float = 1.0, cell_size_y: float = 1.0,
                           field_scaling: float = 1.0, color: str = "black",
                           cmap: Optional[str] = None, width: float = 0.004,
                           scale: Optional[float] = None, origin_xy: Tuple[float, float] = (0.0, 0.0),
                           prefer_masked: bool = True, zorder: int = 6, verbose: bool = False):
    """
    GOAL:
        Rendering wrapper around compute_averaged_quiver_grid -- same
        argument shape and physical-placement convention as quiver_field
        above (a drop-in sibling: swap `step=` for `window=` to switch
        from per-cell subsampling to per-window averaging), but each arrow
        represents the mean field over an NxM-cell block instead of a
        single raw cell.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline (mumax3_csv_plot.py's
        "quiver" plot type calls compute_averaged_quiver_grid directly
        rather than through this wrapper). Calls _region_component and
        compute_averaged_quiver_grid (both this file).

    PARAMETERS:
        window (Tuple[int, int]): (rows, cols) cell-count to average per
            arrow, for example (2, 2) or (2, 8) -- see
            compute_averaged_quiver_grid.
        (all other parameters: identical meaning to quiver_field.)
        verbose (bool): prints whether an artist was produced when True.
            Forwarded to _region_component and compute_averaged_quiver_grid,
            so their own debug messages appear too.

    RETURNS:
        The matplotlib Quiver artist, or None if the region had no valid
        data at all (for example, an empty or fully masked-out region for
        this frame).
    """
    u = _region_component(region, "x", prefer_masked, verbose=verbose) * field_scaling  # x component, scaled
    v = _region_component(region, "y", prefer_masked, verbose=verbose) * field_scaling  # y component, scaled

    row_c, col_c, uu, vv = compute_averaged_quiver_grid(u, v, window, verbose=verbose)  # one averaged vector per window
    if row_c.size == 0:  # no windows had any valid data
        if verbose:
            print("[DEBUG] averaged_quiver_field: no valid windows; returning None.")
        return None

    x_nm = origin_xy[0] + col_c * cell_size_x  # physical x position of each arrow
    y_nm = origin_xy[1] + row_c * cell_size_y  # physical y position of each arrow

    kwargs = dict(angles="xy", scale_units="xy", scale=scale, width=width, zorder=zorder)  # shared quiver styling
    if cmap is not None:  # color arrows by local magnitude
        magnitude = np.hypot(uu, vv)
        return ax.quiver(x_nm, y_nm, uu, vv, magnitude, cmap=cmap, **kwargs)
    return ax.quiver(x_nm, y_nm, uu, vv, color=color, **kwargs)  # fixed-color arrows


# =============================================================================
# PUBLICATION FIGURE SCAFFOLDING
# =============================================================================

def new_publication_figure(figsize: Tuple[float, float] = (8.5, 8.5),
                            xlabel: str = "x (nm)", ylabel: str = "y (nm)",
                            title: Optional[str] = None, equal_aspect: bool = True,
                            verbose: bool = False):
    """
    GOAL:
        Provide a consistent starting point for every figure this module
        produces: clean spines, nm-labeled axes, and equal aspect ratio
        for physical layouts.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline.

    PARAMETERS:
        figsize (Tuple[float, float]): figure size, in inches.
        xlabel, ylabel (str): axis labels.
        title (str, optional): figure title, if any.
        equal_aspect (bool): whether to force equal x/y scaling.
        verbose (bool): prints the figure size when True.

    RETURNS:
        Tuple[Figure, Axes]: the new (fig, ax) pair.
    """
    fig, ax = plt.subplots(figsize=figsize)  # new figure and axes
    ax.set_xlabel(xlabel)  # x axis label
    ax.set_ylabel(ylabel)  # y axis label
    if title:  # an optional title was given
        ax.set_title(title)
    if equal_aspect:  # physical layouts should not be visually stretched
        ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)  # cleaner look: hide the top spine
    ax.spines["right"].set_visible(False)  # cleaner look: hide the right spine
    if verbose:
        print(f"[DEBUG] new_publication_figure: created figure, figsize={figsize}.")
    return fig, ax


def save_publication_figure(fig, output_path: Union[str, Path],
                             formats: Sequence[str] = ("pdf", "png"),
                             dpi: int = 300, close: bool = True, verbose: bool = False) -> List[Path]:
    """
    GOAL:
        Save one figure in multiple formats (PDF for publication or vector
        editing, PNG for quick viewing or slides, by default) with
        consistent tight-bbox behavior.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline.

    PARAMETERS:
        fig: the matplotlib Figure to save.
        output_path (str or Path): output path; may or may not include an
            extension -- any extension given is stripped and replaced with
            each requested format's own extension.
        formats (Sequence[str]): file formats to save, for example
            ("pdf", "png").
        dpi (int): resolution for raster formats.
        close (bool): whether to close the figure after saving.
        verbose (bool): prints each saved path when True.

    RETURNS:
        List[Path]: the paths actually written, one per requested format.
    """
    output_path = Path(output_path)  # ensure a Path object
    stem = output_path.with_suffix("")  # strip any extension given in output_path
    stem.parent.mkdir(parents=True, exist_ok=True)  # ensure the destination directory exists
    written = []  # every path actually written, accumulated below
    for fmt in formats:  # save once per requested format
        path = stem.with_suffix(f".{fmt.lstrip('.')}")  # this format's output path
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")  # write this format's file
        written.append(path)
        if verbose:
            print(f"[DEBUG] save_publication_figure: saved '{path}'.")
    if close:  # free the figure's memory once every format has been saved
        plt.close(fig)
    return written


# =============================================================================
# FRAME SEQUENCE -> GIF / VIDEO
# =============================================================================

def frames_to_gif(frame_paths: List[Path], output_path: Union[str, Path],
                   duration_ms: int = 800, verbose: bool = False) -> Tuple[bool, str]:
    """
    GOAL:
        Stitch a sequence of PNG frames into a looping GIF.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline.

    PARAMETERS:
        frame_paths (List[Path]): ordered list of PNG frame paths to
            stitch together.
        output_path (str or Path): destination path for the GIF.
        duration_ms (int): duration of each frame, in milliseconds.
        verbose (bool): prints the number of frames stitched when True.

    RETURNS:
        Tuple[bool, str]: (success, message) -- True and the saved path on
        success; False and a description of why on failure (fewer than two
        frames given, or Pillow not installed).

    LIMITATIONS:
        Requires the Pillow library; returns a failure tuple rather than
        raising if it is not installed.
    """
    if len(frame_paths) < 2:  # a GIF needs at least two frames to be meaningful
        return False, "not enough frames for GIF"
    try:
        from PIL import Image  # GIF encoding
    except Exception:
        return False, "Pillow is not installed"

    output_path = Path(output_path)  # ensure a Path object
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]  # load and palette-convert every frame
    output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the destination directory exists
    images[0].save(
        output_path, save_all=True, append_images=images[1:],  # stitch every frame into one looping GIF
        duration=duration_ms, loop=0,
    )
    if verbose:
        print(f"[DEBUG] frames_to_gif: stitched {len(frame_paths)} frame(s) into '{output_path}'.")
    return True, f"saved {output_path}"


def frames_to_video(frame_paths: List[Path], output_path: Union[str, Path],
                     fps: float = 5.0, verbose: bool = False) -> Tuple[bool, str]:
    """
    GOAL:
        Stitch a sequence of PNG frames into an MP4 video (via imageio and
        ffmpeg) -- for longer sequences (for example, hundreds of
        superposition regenerations), an MP4 is far smaller and smoother to
        scrub through than a GIF.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline.

    PARAMETERS:
        frame_paths (List[Path]): ordered list of PNG frame paths to
            stitch together.
        output_path (str or Path): destination path for the video.
        fps (float): playback frame rate.
        verbose (bool): prints the number of frames stitched when True.

    RETURNS:
        Tuple[bool, str]: (success, message) -- True and the saved path on
        success; False and a description of why on failure (fewer than two
        frames given, imageio not installed, or video encoding failed,
        typically because ffmpeg is not installed).

    LIMITATIONS:
        Requires the imageio library, and typically ffmpeg on the system
        path for MP4 encoding; returns a failure tuple rather than raising
        if either is unavailable.
    """
    if len(frame_paths) < 2:  # a video needs at least two frames to be meaningful
        return False, "not enough frames for video"
    try:
        import imageio.v2 as imageio  # video encoding, newer imageio API
    except Exception:
        try:
            import imageio  # video encoding, older imageio API
        except Exception:
            return False, "imageio is not installed"

    output_path = Path(output_path)  # ensure a Path object
    output_path.parent.mkdir(parents=True, exist_ok=True)  # ensure the destination directory exists
    try:
        with imageio.get_writer(str(output_path), fps=fps, macro_block_size=None) as writer:  # open the video file for writing
            for p in frame_paths:  # write every frame, in order
                writer.append_data(imageio.imread(p))
    except Exception as e:  # encoding failed, typically because ffmpeg is missing
        return False, f"video encoding failed ({type(e).__name__}: {e}); is ffmpeg installed?"
    if verbose:
        print(f"[DEBUG] frames_to_video: stitched {len(frame_paths)} frame(s) into '{output_path}' at {fps} fps.")
    return True, f"saved {output_path}"
