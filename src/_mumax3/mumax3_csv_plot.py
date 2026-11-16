r"""
File Name: mumax3_csv_plot.py
Author: Connor Sullivan, August Cox
Date: 2026.08.05
Revision: v8

============================================================
PURPOSE
============================================================
Renders scalar-field color maps, hue/gradient RGBA maps, and averaged
quiver plots from mumax3 vector field data, per region and per timestep,
across every color-scale variant (global/local/custom) a caller requests.
This is the plotting engine mumax3_parent_utilities.run_spatial_plots
dispatches to; it does not decide WHICH plot types or scales to render
(mumax3_plot_calls.py's plot_configs decide that) -- it only knows how to
render one region/timestep/plot-type combination once told to.

Contains, in the order they appear in this file:
    - resolve_suptitle: resolve a per-frame figure suptitle.
    - resolve_quiver_window: resolve the per-region averaging window for a
      quiver plot.
    - compute_region_extent: compute the physical-nm imshow/quiver extent
      for one region's plot.
    - save_plotted_data_csv: write a companion CSV of the exact plotted
      array next to a PNG.
    - plot_matrix_map_for_timestep / plot_matrix_map_for_region_per_timestep:
      the core rendering functions -- render every region (or one region,
      for the parallel-dispatch case) at one timestep, for one plot
      type/scale.
    - plot_all_regions_over_time_parallel_helper: dispatches one timestep's
      regions across a process pool, each calling
      plot_matrix_map_for_region_per_timestep.
    - plot_all_regions_over_time: the top-level entry point -- renders
      every timestep for one plot type/scale, either by dispatching
      timesteps across a process pool (processor_mode="timesteps") or by
      dispatching regions within each timestep
      (processor_mode="regions", via plot_all_regions_over_time_parallel_
      helper).
    - sanitize_label: make a region label filesystem-safe.
    - plot_region_stats_over_time: plot each weighted-circular-statistics
      metric over time, per region.

============================================================
LIMITATIONS
============================================================
- plot_matrix_map_for_timestep and plot_matrix_map_for_region_per_timestep
  are near-duplicates of each other by design: the first renders every
  region for one timestep in a single process (used when
  processor_mode="timesteps" dispatches whole timesteps across a process
  pool instead); the second renders exactly one region, meant to be
  dispatched itself (used when processor_mode="regions" dispatches
  individual regions across a process pool instead, via
  plot_all_regions_over_time_parallel_helper). Their four rendering
  branches (plot_type "hue", "gradient", "quiver", and the default
  scalar-map case) are kept logically identical between the two functions;
  a fix to one branch's rendering logic should be mirrored in the other
  function's matching branch.
- Revision v7 of this file (see the parent revision) additionally
  contained six functions with no callers anywhere in this pipeline
  (plot_multi_scalar_maps_for_timestep, plot_multi_scalar_maps_over_time_
  parallel, plot_full_field_for_timestep, plot_full_field_for_all_timesteps,
  plot_combined_full_field_for_timestep, plot_combined_full_field_for_
  all_timesteps), removed in this revision. Some of that plotting
  functionality (multiple scalar maps side by side per region; full,
  uncropped-field plotting) may be relevant to future design work -- see
  v7 of this file if any of it needs to be recovered or adapted.
"""

import functools  # binding fixed keyword arguments into a scalar_func, and detecting a scalar_func that is already a functools.partial
import inspect  # inspecting a scalar_func's real parameter names, to filter which keyword arguments it actually accepts
from typing import Callable, List, Dict, Tuple, Optional, Union, Any  # type hints used throughout this file's signatures
import numpy as np  # array operations


# --------------------------
# ---- Resolve a per-frame figure suptitle ----
def resolve_suptitle(suptitles, t: int, verbose: bool = False) -> Optional[str]:
    """
    GOAL:
        Resolve a single frame's figure suptitle from any of the shapes a
        caller might provide: a fixed callable, a list (one entry per
        frame), or a dict (keyed by frame index).

    HOW IT TALKS TO OTHER FILES:
        Called by plot_all_regions_over_time (this file), once per
        timestep, to resolve that timestep's suptitle before dispatching
        to plot_matrix_map_for_timestep / plot_all_regions_over_time_
        parallel_helper.

    PARAMETERS:
        suptitles: a callable suptitles(t) -> str, a list of per-frame
            titles, a dict keyed by frame index, or None.
        t (int): the frame index to resolve a title for.
        verbose (bool): prints the resolved title when True.

    RETURNS:
        Optional[str]: the resolved title for frame `t`, or None if
        `suptitles` is None, is a list too short to cover `t`, or is a
        dict with no entry for `t`.
    """
    if callable(suptitles):  # a function was given: call it directly with this frame's index
        result = suptitles(t)
    elif isinstance(suptitles, list) and t < len(suptitles):  # a list was given, and it covers this frame index
        result = suptitles[t]
    elif isinstance(suptitles, dict):  # a dict was given: look up this frame index (may return None if absent)
        result = suptitles.get(t)
    else:  # None was given, or a list too short to cover this frame
        result = None
    if verbose:
        print(f"[DEBUG] resolve_suptitle: t={t} -> {result!r}.")
    return result


# --------------------------
# ---- Resolve the quiver averaging window for a given region ----
def resolve_quiver_window(quiver_window, label: str, region_meta: Optional[Dict[str, Dict[str, str]]] = None,
                           verbose: bool = False) -> Tuple[int, int]:
    """
    GOAL:
        Resolve the (rows, cols) averaging window (see mumax3_plot_vectors.
        compute_averaged_quiver_grid) to use for one region's quiver plot.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_matrix_map_for_timestep and plot_matrix_map_for_
        region_per_timestep (this file), once per region, whenever
        plot_type="quiver".

    PARAMETERS:
        quiver_window: either a plain (rows, cols) tuple, used for every
            region (the common case, when every device's mask is
            identical), or a dict for when it is not identical across
            regions or mask types. When a dict, it is looked up, in order,
            by: the region's own label, then its mask_type (from
            region_meta, for example "File Mask"), then its center_name,
            then a "default" key.
        label (str): this region's label.
        region_meta: Optional[Dict[label -> {"center_name", "mask_type",
            ...}]], from region_extraction.build_region_meta -- used only
            when `quiver_window` is a dict, to look up this region's
            mask_type/center_name.
        verbose (bool): prints the resolved window when True.

    RETURNS:
        Tuple[int, int]: the (rows, cols) averaging window for this region.
        Falls back to (2, 2) if `quiver_window` is a dict with no matching
        key, or if `quiver_window` is None.
    """
    if isinstance(quiver_window, dict):  # per-region/per-mask-type window sizes were given
        meta = (region_meta or {}).get(label, {})  # this region's metadata, if any
        for key in (label, meta.get("mask_type"), meta.get("center_name"), "default"):  # check each lookup key, most specific first
            if key is not None and key in quiver_window:
                result = tuple(quiver_window[key])
                if verbose:
                    print(f"[DEBUG] resolve_quiver_window: label='{label}' matched key '{key}' -> {result}.")
                return result
        if verbose:
            print(f"[DEBUG] resolve_quiver_window: label='{label}' matched no key; falling back to (2, 2).")
        return (2, 2)  # no matching key found in the dict
    result = tuple(quiver_window) if quiver_window is not None else (2, 2)  # a plain tuple was given, or nothing was given at all
    if verbose:
        print(f"[DEBUG] resolve_quiver_window: label='{label}' -> {result}.")
    return result


# --------------------------
# ---- Compute the imshow/quiver extent for one region's plot ----
def compute_region_extent(label: str, width: int, height: int, cell_size_x: float, cell_size_y: float,
                           region_meta: Optional[Dict[str, Dict[str, str]]] = None,
                           plot_origin_cells: Optional[Tuple[float, float]] = None,
                           verbose: bool = False) -> List[float]:
    """
    GOAL:
        Compute the physical-nm `extent` passed to imshow/quiver for one
        region's plot -- either each region's own local box starting at
        (0, 0), or, when requested, every region placed in one shared
        coordinate frame.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_matrix_map_for_timestep and plot_matrix_map_for_
        region_per_timestep (this file) for every rendering branch (hue,
        gradient, quiver, and the default scalar-map case) whenever no
        explicit `extent` was already provided.

    PARAMETERS:
        label (str): this region's label, used to look up its metadata in
            `region_meta` when `plot_origin_cells` is given.
        width, height (int): this region's array dimensions, in cells.
        cell_size_x, cell_size_y (float): nm per grid cell.
        region_meta: Optional[Dict[label -> {"center_cells", ...}]], from
            region_extraction.build_region_meta -- only consulted when
            `plot_origin_cells` is given.
        plot_origin_cells: Optional[Tuple[float, float]], an absolute grid
            position (in cells) to map to (0, 0) on this region's plot
            axes, instead of the region's own local box starting at
            (0, 0). For example, passing the canvas center here makes
            every canvas-level plot (the full canvas, each quadrant, each
            per-device zoom) share DeviceArray's own canvas-centered
            convention, instead of each region independently starting its
            own box at (0, 0).
        verbose (bool): prints the resolved extent when True.

    RETURNS:
        List[float]: [xmin, xmax, ymin, ymax], in nm.

    LIMITATIONS:
        Falls back to the default local-box behavior if `region_meta` is
        not given, or if it has no "center_cells" entry for `label` -- this
        is purely additive, so callers that do not need shared-frame
        placement (for example, per-device regions built without this
        metadata) continue to work exactly as before.
    """
    if plot_origin_cells is not None:  # a shared coordinate frame was requested
        meta = (region_meta or {}).get(label, {})  # this region's metadata, if any
        center_cells = meta.get("center_cells")  # this region's absolute grid center, if known
        if center_cells is not None:  # metadata is available: place this region in the shared frame
            rel_cx = (center_cells[0] - plot_origin_cells[0]) * cell_size_x  # this region's center, relative to the shared origin, x (nm)
            rel_cy = (center_cells[1] - plot_origin_cells[1]) * cell_size_y  # this region's center, relative to the shared origin, y (nm)
            result = [
                rel_cx - width / 2.0 * cell_size_x, rel_cx + width / 2.0 * cell_size_x,
                rel_cy - height / 2.0 * cell_size_y, rel_cy + height / 2.0 * cell_size_y,
            ]
            if verbose:
                print(f"[DEBUG] compute_region_extent: label='{label}', shared-frame extent={result}.")
            return result
    result = [0, width * cell_size_x, 0, height * cell_size_y]  # default: this region's own local box, starting at (0, 0)
    if verbose:
        print(f"[DEBUG] compute_region_extent: label='{label}', local extent={result}.")
    return result


# --------------------------
# ---- Write a companion CSV of the exact plotted array, next to a PNG ----
def save_plotted_data_csv(csv_path: str, data, verbose: bool = False) -> None:
    """
    GOAL:
        Write the exact 2D array (or, for plot_type="quiver", the (u, v)
        pair of 2D arrays) that was just plotted to a companion CSV next to
        the PNG, in the same stacked mumax3-style format used throughout
        this codebase (each component is `height` rows of `width`
        comma-separated values, followed by a blank line -- see
        mumax3_csv_reader.py's own documentation). This lets the exported
        CSV round-trip through mumax3_csv_reader.process_csv_file /
        remove_empty_lists exactly like a real mumax3 CSV would, and
        otherwise be opened directly for standalone re-analysis or
        re-plotting (see plot_single_field.py).

    HOW IT TALKS TO OTHER FILES:
        Called by plot_matrix_map_for_timestep and plot_matrix_map_for_
        region_per_timestep (this file), once per region/timestep, when
        `save_plot_data_csv=True` was requested.

    PARAMETERS:
        csv_path (str): destination path for the CSV.
        data: the value returned by this region/timestep's scalar_func --
            either a single 2D array (the default scalar-map case), or a
            tuple whose elements may include 2D arrays alongside other
            values (see LIMITATIONS).
        verbose (bool): prints the number of 2D arrays written when True.

    RETURNS:
        None. Writes a CSV file to `csv_path`, or writes nothing at all if
        `data` contains no genuinely 2D arrays.

    LIMITATIONS:
        `data` is not always a plain 2D array or a tuple of 2D arrays.
        plot_type="hue" returns (angle, magnitude, rgba), and
        plot_type="gradient" returns (angle, magnitude, rgba, mean_angle,
        delta) -- `rgba` is a 3D (height, width, 4) rendered image, not
        per-cell plotted data, and mean_angle/delta are scalars or short
        auxiliary values, not per-cell arrays at all. Only elements that
        are genuinely 2D arrays (ndim == 2) -- the actual per-cell scalar
        fields (angle, magnitude, u, v, ...) -- are written; everything
        else (the rendered RGBA image, colorbar-range scalars, etc.) is
        silently skipped, since "the data plotted" for CSV round-tripping
        purposes means the underlying field values, not the rendering.
        NaN cells (masked-out pixels) are written as the literal text
        "nan", which Python's float() parses back correctly (so
        process_csv_file's float_dtype(cell) round-trips them too) --
        masked-out numeric data is genuinely absent, not zero, and writing
        0.0 there would misrepresent the mask footprint.
    """
    blocks = data if isinstance(data, tuple) else (data,)  # normalize to a tuple of candidate blocks
    arrays_2d = [np.asarray(block) for block in blocks]  # convert every candidate block to an array
    arrays_2d = [arr for arr in arrays_2d if arr.ndim == 2]  # keep only genuinely 2D arrays (see LIMITATIONS)
    if not arrays_2d:  # nothing 2D to export (for example, a plot_type whose data is all metadata)
        if verbose:
            print(f"[DEBUG] save_plotted_data_csv: no 2D array(s) found in data; nothing written for '{csv_path}'.")
        return
    with open(csv_path, "w") as f:
        for arr in arrays_2d:  # write each 2D array as its own stacked block
            for row in arr:
                f.write(",".join(str(float(v)) for v in row) + "\n")
            f.write("\n")  # blank-line separator between blocks, matching mumax3's own CSV format
    if verbose:
        print(f"[DEBUG] save_plotted_data_csv: wrote {len(arrays_2d)} array(s) to '{csv_path}'.")


# --------------------------
# ---- Render every region for one timestep, one plot type/scale (single process) ----
def plot_matrix_map_for_timestep(
    t: int,  # frame index to render
    regions: Dict[str, Dict[str, Any]],  # region data, keyed by label, as returned by region_extraction.reorganize_regions
    output_dir: str,  # root directory this plot type/scale's frames are written under
    cols: int,  # accepted for interface consistency; not used by this function's own layout (each region gets its own figure)
    rows: int,  # accepted for interface consistency; not used by this function's own layout
    scalar_func: Callable[[Dict[str, Any], int], np.ndarray],  # computes the value(s) to plot for one region/timestep
    vmin: Optional[float] = None,  # global color-scale minimum (ignored when use_local_scaling=True)
    vmax: Optional[float] = None,  # global color-scale maximum (ignored when use_local_scaling=True)
    cmap: str = 'viridis',  # matplotlib colormap for the default scalar-map rendering branch
    titles: Optional[Union[Dict[str, str], list]] = None,  # per-region plot titles
    suptitle: Optional[str] = None,  # figure-level title for this timestep
    xlabel: str = "X (nm)",  # x-axis label
    ylabel: str = "Y (nm)",  # y-axis label
    dpi: int = 300,  # resolution for saved PNGs
    cbar_label: str = "Value",  # colorbar label for the default scalar-map rendering branch
    prefix: str = "matrix_map",  # filename prefix; this timestep's frame folder is "{prefix}_{t:06d}"
    use_local_scaling: bool = False,  # if True, scale each region's own color range to its own min/max instead of vmin/vmax
    cell_size_x: float = 1,  # nm per grid cell, x direction
    cell_size_y: float = 1,  # nm per grid cell, y direction
    field_scale: Optional[float] = None,  # multiplier forwarded into scalar_func (for example, Tesla to Oersted)
    plot_type: Optional[str] = None,  # None for a regular scalar map, "hue"/"gradient" for an RGBA image, "quiver" for averaged arrows
    extent: Optional[Tuple[float, float, float, float]] = None,  # explicit axis extent in nm; computed automatically per region if not given
    region_meta: Optional[Dict[str, Dict[str, str]]] = None,  # {label -> {"center_name","mask_type","center_cells"}}, from region_extraction.build_region_meta
    quiver_window: Optional[Union[Tuple[int, int], Dict[str, Tuple[int, int]]]] = None,  # averaging window for plot_type="quiver" -- see resolve_quiver_window
    plot_origin_cells: Optional[Tuple[float, float]] = None,  # shared-coordinate-frame origin -- see compute_region_extent
    save_plot_data_csv: bool = False,  # also write a companion .csv of the exact plotted array next to each PNG
    verbose: bool = False,  # print per-region status messages when True
) -> None:
    """
    GOAL:
        Render every region in `regions` at timestep `t`, for one plot
        type and scale, each region as its own figure. This is the
        implementation used when processor_mode="timesteps" dispatches
        whole timesteps across a process pool (see plot_all_regions_over_
        time) -- one call to this function does every region for one
        timestep, inside a single process.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_all_regions_over_time (this file) when
        processor_mode="timesteps". Calls compute_region_extent,
        resolve_quiver_window, save_plotted_data_csv, and sanitize_label
        (all this file). For plot_type="gradient", calls
        mumax3_pipeline_utilities.build_deviation_cmap.

    PARAMETERS:
        See the inline comments on each parameter above; the four
        rendering branches this function supports are described in this
        file's module-level LIMITATIONS section.
        verbose (bool): prints a debug status line per region (color
            scaling and array shape) when True. Off by default, since this
            function is called once per region per timestep and would
            otherwise produce excessive console output on a large run.

    RETURNS:
        None. Writes one PNG (and, if requested, one companion CSV) per
        region under "<output_dir>/<prefix>_<t:06d>/<center_name>/
        <mask_type>/<label>.png".

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding this
        function's relationship to plot_matrix_map_for_region_per_timestep.
    """
    import os  # output path construction
    import matplotlib.pyplot as plt  # figure rendering

    if extent is None:  # no explicit extent given: recompute it fresh for each region below, since region sizes may differ
        extent_refresh = True

    def call_scalar_func(func, region_data, t, label, field_scale):
        # Calls `func` (this timestep/plot's scalar_func) with only the keyword
        # arguments it actually accepts -- scalar_func implementations across this
        # pipeline do not all share the same parameter names (see mumax3_pipeline_
        # utilities.py's matrix_* functions), so every candidate keyword argument is
        # offered and only the ones the target function's real signature accepts are
        # kept. If `func` is already a functools.partial (built in mumax3_plot_calls.py
        # to bind a fixed argument such as `comp`), its already-bound keywords are
        # merged in first and the underlying function is inspected instead of the
        # partial wrapper itself.
        base_kwargs = {
            "region_data": region_data,
            "t": t,
            "label": label,
            "field_scaling": field_scale
        }
        if isinstance(func, functools.partial):  # unwrap a partial to inspect its underlying function's real signature
            bound = func.keywords or {}
            base_kwargs = {**bound, **base_kwargs}
            func = func.func
        sig = inspect.signature(func)  # the underlying function's real parameter names
        valid_keys = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in valid_keys}  # keep only arguments this function actually accepts
        return func(**filtered_kwargs)

    frame_dir = os.path.join(output_dir, f"{prefix}_{t:06d}")  # this timestep's own output folder
    os.makedirs(frame_dir, exist_ok=True)

    for i, (label, region_data) in enumerate(regions.items()):  # render each region as its own figure

        fig, ax = plt.subplots(figsize=(6, 6))  # one figure per region

        data = call_scalar_func(scalar_func, region_data, t, label, field_scale)  # compute this region's value(s) to plot

        # ---- Hue rendering: scalar_func returned (angle, magnitude, rgba) -- angle
        # drives hue, magnitude drives value/brightness (see mumax3_pipeline_
        # utilities.matrix_hue_hsv_rgba, which builds `rgba`). ----
        if plot_type == "hue" and isinstance(data, tuple):
            import matplotlib.cm as cm  # colorbar rendering
            import matplotlib.colors as mcolors  # colorbar normalization
            angle, magnitude, rgba = data  # unpack the angle/magnitude arrays and the already-rendered RGBA image
            if use_local_scaling:  # scale the magnitude colorbar to this frame's own range
                mag_min = np.nanmin(magnitude)
                mag_max = np.nanmax(magnitude)
            else:  # scale the magnitude colorbar to the caller-provided global range
                mag_min = vmin
                mag_max = vmax
            height, width = angle.shape  # this region's array dimensions, for computing its extent below
            if extent is None:
                extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)

            im = ax.imshow(rgba, origin='lower', extent=extent)  # render the already-built RGBA image directly
            if titles:
                if isinstance(titles, dict):
                    ax.set_title(titles.get(label, label), fontsize=8)
                elif isinstance(titles, list) and i < len(titles):
                    ax.set_title(titles[i], fontsize=8)
                else:
                    ax.set_title(label, fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            # ---- Angle (hue) colorbar, always 0-360 degrees regardless of scaling ----
            angle_norm = mcolors.Normalize(vmin=0, vmax=360)
            angle_cmap = cm.hsv
            sm_angle = cm.ScalarMappable(norm=angle_norm, cmap=angle_cmap)
            sm_angle.set_array([])
            cbar_hue = fig.colorbar(sm_angle, ax=ax, fraction=0.046, pad=0.05)
            cbar_hue.set_label("Angle (deg.)")
            # ---- Magnitude (value/brightness) colorbar, grayscale to avoid confusion with the hue colorbar ----
            mag_norm = mcolors.Normalize(vmin=mag_min, vmax=mag_max)
            mag_cmap = cm.gray
            sm_mag = cm.ScalarMappable(norm=mag_norm, cmap=mag_cmap)
            sm_mag.set_array([])
            cbar_mag = fig.colorbar(sm_mag, ax=ax, fraction=0.046, pad=0.13, orientation='horizontal')
            cbar_mag.set_label("Magnitude (Oe)")

        # ---- Gradient rendering: scalar_func returned (angle, magnitude, rgba,
        # mean_angle, delta) -- hue is fixed at the region's mean angle, value/
        # saturation encode each cell's deviation from that mean (see mumax3_
        # pipeline_utilities.matrix_gradient_hsv_rgba, which builds `rgba`). ----
        elif plot_type == "gradient" and isinstance(data, tuple):
            angle, magnitude, rgba, mean_angle, delta = data  # unpack the angle/magnitude arrays, the RGBA image, the region's mean angle, and its deviation range
            height, width = angle.shape  # this region's array dimensions, for computing its extent below
            if extent is None:
                extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)

            im = ax.imshow(rgba, origin='lower', extent=extent)  # render the already-built RGBA image directly
            if titles:
                if isinstance(titles, dict):
                    ax.set_title(titles.get(label, label), fontsize=8)
                elif isinstance(titles, list) and i < len(titles):
                    ax.set_title(titles[i], fontsize=8)
                else:
                    ax.set_title(label, fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # ---- Deviation colorbar -- not tied to the image itself (the image's
            # colors come directly from `rgba`), but serves as a reference guide for
            # interpreting it. ----
            if use_local_scaling:  # scale the colorbar to this frame's own deviation range
                dmin_vmin = np.nanmin(delta[0])
                dmax_vmax = np.nanmax(delta[1])
            elif vmin is not None and vmax is not None:  # scale the colorbar to the caller-provided global range
                dmin_vmin = vmin
                dmax_vmax = vmax
            else:  # no scaling information available at all: fall back to a fixed +/-180 degree range
                dmin_vmin = -180
                dmax_vmax = 180
            if verbose:
                print(f"[DEBUG] plot_matrix_map_for_timestep: label='{label}', t={t}, "
                      f"deviation colorbar range=[{dmin_vmin}, {dmax_vmax}], mean_angle={mean_angle}.")
            from mumax3_pipeline_utilities import build_deviation_cmap  # builds a colormap matching matrix_gradient_hsv_rgba's own color scheme
            cmap, norm = build_deviation_cmap(mean_angle, dmin_vmin, dmax_vmax)

            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Angular Deviation (\u00b0)")

        # ---- Averaged quiver rendering: scalar_func returned (u, v) (see mumax3_
        # pipeline_utilities.matrix_uv); draws arrows via mumax3_plot_vectors.
        # compute_averaged_quiver_grid instead of a color map. ----
        elif plot_type == "quiver" and isinstance(data, tuple):
            from mumax3_plot_vectors import compute_averaged_quiver_grid  # the mask/edge-aware windowed-averaging implementation
            u, v = data  # unpack the two in-plane vector components
            height, width = u.shape  # this region's array dimensions, for computing its extent below
            if extent is None:
                extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)
            window = resolve_quiver_window(quiver_window, label, region_meta, verbose=verbose)  # this region's averaging window size
            row_c, col_c, uu, vv = compute_averaged_quiver_grid(u, v, window)  # one averaged arrow per window
            if row_c.size:  # at least one window had valid data
                x_nm = col_c * cell_size_x  # physical x position of each arrow
                y_nm = row_c * cell_size_y  # physical y position of each arrow
                ax.quiver(x_nm, y_nm, uu, vv, angles="xy", scale_units="xy", color="black", width=0.004)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_aspect("equal", adjustable="box")
            if titles:
                if isinstance(titles, dict):
                    ax.set_title(titles.get(label, label), fontsize=8)
                elif isinstance(titles, list) and i < len(titles):
                    ax.set_title(titles[i], fontsize=8)
                else:
                    ax.set_title(label, fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # ---- Default rendering: scalar_func returned a plain 2D array, plotted as a
        # standard color map. ----
        else:
            local_vmin = np.nanmin(data) if use_local_scaling else vmin  # this region's color-scale minimum
            local_vmax = np.nanmax(data) if use_local_scaling else vmax  # this region's color-scale maximum
            height, width = data.shape  # this region's array dimensions, for computing its extent below
            if verbose:
                print(f"[DEBUG] plot_matrix_map_for_timestep: label='{label}', t={t}, "
                      f"color range=[{local_vmin}, {local_vmax}], data shape={data.shape}.")
            if extent is None:
                extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)

            im = ax.imshow(
                data,  # the 2D array to plot
                cmap=cmap,  # color map
                origin='lower',  # row 0 at the bottom, matching this pipeline's convention elsewhere
                vmin=local_vmin,  # color-scale minimum
                vmax=local_vmax,  # color-scale maximum
                extent=extent  # physical axis bounds, in nm
            )

            if titles:
                if isinstance(titles, dict):  # look up this region's title by label
                    ax.set_title(titles.get(label, label), fontsize=8)
                elif isinstance(titles, list) and i < len(titles):  # look up this region's title by position
                    ax.set_title(titles[i], fontsize=8)
                else:  # no matching title found: fall back to the region's own label
                    ax.set_title(label, fontsize=8)
            else:  # no titles provided at all: use the region's own label
                ax.set_title(label, fontsize=8)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cbar_label)

        if suptitle:  # an overall figure title was provided for this timestep
            plt.suptitle(suptitle, fontsize=12)

        plt.tight_layout()

        # ---- Output layout: "<frame_dir>/<center or region name>/<mask type>/
        # <label>.png" -- so regions that share a center but differ in mask_mode (for
        # example, mask_mode=["file", "none", "ellipse"] all at the same center) land
        # in separate folders instead of overwriting each other's plot. See
        # region_extraction.build_region_meta for where center_name/mask_type come
        # from. ----
        meta = (region_meta or {}).get(label, {})  # this region's metadata, if any
        center_name = meta.get("center_name", sanitize_label(label))  # this region's display folder name
        mask_type = meta.get("mask_type", "No Mask")  # this region's mask-type folder name
        region_dir = os.path.join(frame_dir, sanitize_label(center_name), sanitize_label(mask_type))
        os.makedirs(region_dir, exist_ok=True)

        region_filename = f"{sanitize_label(label)}.png"  # this region's own PNG filename

        plt.savefig(
            os.path.join(region_dir, region_filename),
            dpi=dpi
        )

        if save_plot_data_csv:  # also export the exact plotted data as a companion CSV
            csv_path = os.path.join(region_dir, f"{sanitize_label(label)}.csv")
            save_plotted_data_csv(csv_path, data, verbose=verbose)

        plt.close(fig)  # free this figure's memory before moving to the next region
        if extent_refresh:  # extent was computed fresh for this region: reset it so the next region computes its own
            extent = None


# --------------------------
# ---- Render exactly one region for one timestep, one plot type/scale (for parallel dispatch) ----
def plot_matrix_map_for_region_per_timestep(
    t: int,  # frame index to render
    region_idx: int,  # this region's position/index among the regions being plotted (used for list-based titles)
    region_data: Dict[str, Any],  # this one region's data
    label: str,  # this region's label
    output_dir: str,  # root directory this plot type/scale's frames are written under
    cols: int,  # accepted for interface consistency; not used by this function's own layout
    rows: int,  # accepted for interface consistency; not used by this function's own layout
    scalar_func: Callable[[Dict[str, Any], int], np.ndarray],  # computes the value(s) to plot for this region/timestep
    vmin: Optional[float] = None,  # global color-scale minimum (ignored when use_local_scaling=True)
    vmax: Optional[float] = None,  # global color-scale maximum (ignored when use_local_scaling=True)
    cmap: str = 'viridis',  # matplotlib colormap for the default scalar-map rendering branch
    titles: Optional[Union[Dict[str, str], list]] = None,  # per-region plot titles
    suptitle: Optional[str] = None,  # figure-level title for this timestep
    xlabel: str = "X (nm)",  # x-axis label
    ylabel: str = "Y (nm)",  # y-axis label
    dpi: int = 300,  # resolution for saved PNGs
    cbar_label: str = "Value",  # colorbar label for the default scalar-map rendering branch
    prefix: str = "matrix_map",  # filename prefix; this timestep's frame folder is "{prefix}_{t:06d}"
    use_local_scaling: bool = False,  # if True, scale this region's own color range to its own min/max instead of vmin/vmax
    cell_size_x: float = 1,  # nm per grid cell, x direction
    cell_size_y: float = 1,  # nm per grid cell, y direction
    field_scale: Optional[float] = None,  # multiplier forwarded into scalar_func (for example, Tesla to Oersted)
    plot_type: Optional[str] = None,  # None for a regular scalar map, "hue"/"gradient" for an RGBA image, "quiver" for averaged arrows
    extent: Optional[Tuple[float, float, float, float]] = None,  # explicit axis extent in nm; computed automatically if not given
    region_meta: Optional[Dict[str, Dict[str, str]]] = None,  # {label -> {"center_name","mask_type","center_cells"}}, see plot_matrix_map_for_timestep
    quiver_window: Optional[Union[Tuple[int, int], Dict[str, Tuple[int, int]]]] = None,  # see plot_matrix_map_for_timestep
    plot_origin_cells: Optional[Tuple[float, float]] = None,  # see plot_matrix_map_for_timestep / compute_region_extent
    save_plot_data_csv: bool = False,  # also write a companion .csv of the exact plotted array next to the PNG
    verbose: bool = False,  # print a status message when True
) -> None:
    """
    GOAL:
        Render exactly one region at timestep `t`, for one plot type and
        scale. This is the implementation used when processor_mode=
        "regions" dispatches individual regions across a process pool (see
        plot_all_regions_over_time_parallel_helper) -- one call to this
        function renders one region for one timestep, meant to be run in
        its own worker process.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_all_regions_over_time_parallel_helper (this file),
        typically once per region per timestep, dispatched across a
        process pool. Calls compute_region_extent, resolve_quiver_window,
        save_plotted_data_csv, and sanitize_label (all this file). For
        plot_type="gradient", calls mumax3_pipeline_utilities.
        build_deviation_cmap.

    PARAMETERS:
        See the inline comments on each parameter above; the four
        rendering branches this function supports are described in this
        file's module-level LIMITATIONS section, which also explains this
        function's relationship to plot_matrix_map_for_timestep (its
        every-region-at-once twin).
        verbose (bool): prints a debug status line for this region (color
            scaling and array shape) when True.

    RETURNS:
        None. Writes one PNG (and, if requested, one companion CSV) for
        this region under "<output_dir>/<prefix>_<t:06d>/<center_name>/
        <mask_type>/<label>.png".

    LIMITATIONS:
        See this file's module-level LIMITATIONS section.
    """
    import os  # output path construction
    import matplotlib.pyplot as plt  # figure rendering

    if extent is None:  # no explicit extent given: it will be computed fresh below
        extent_refresh = True

    def call_scalar_func(func, region_data, t, label, field_scale):
        # See plot_matrix_map_for_timestep's matching inner function for the full
        # explanation of why this signature-filtering step exists.
        base_kwargs = {
            "region_data": region_data,
            "t": t,
            "label": label,
            "field_scaling": field_scale
        }
        if isinstance(func, functools.partial):  # unwrap a partial to inspect its underlying function's real signature
            bound = func.keywords or {}
            base_kwargs = {**bound, **base_kwargs}
            func = func.func
        sig = inspect.signature(func)  # the underlying function's real parameter names
        valid_keys = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in base_kwargs.items() if k in valid_keys}  # keep only arguments this function actually accepts
        return func(**filtered_kwargs)

    frame_dir = os.path.join(output_dir, f"{prefix}_{t:06d}")  # this timestep's own output folder
    os.makedirs(frame_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))  # this region's own figure

    data = call_scalar_func(scalar_func, region_data, t, label, field_scale)  # compute this region's value(s) to plot

    # ---- Hue rendering -- see plot_matrix_map_for_timestep's matching branch. ----
    if plot_type == "hue" and isinstance(data, tuple):
        import matplotlib.cm as cm  # colorbar rendering
        import matplotlib.colors as mcolors  # colorbar normalization
        angle, magnitude, rgba = data  # unpack the angle/magnitude arrays and the already-rendered RGBA image
        if use_local_scaling:  # scale the magnitude colorbar to this frame's own range
            mag_min = np.nanmin(magnitude)
            mag_max = np.nanmax(magnitude)
        else:  # scale the magnitude colorbar to the caller-provided global range
            mag_min = vmin
            mag_max = vmax
        height, width = angle.shape  # this region's array dimensions, for computing its extent below
        if extent is None:
            extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)

        im = ax.imshow(rgba, origin='lower', extent=extent)  # render the already-built RGBA image directly
        if titles:
            if isinstance(titles, dict):
                ax.set_title(titles.get(label, label), fontsize=8)
            elif isinstance(titles, list) and region_idx < len(titles):
                ax.set_title(titles[region_idx], fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
        else:
            ax.set_title(label, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        angle_norm = mcolors.Normalize(vmin=0, vmax=360)  # angle (hue) colorbar, always 0-360 degrees
        angle_cmap = cm.hsv
        sm_angle = cm.ScalarMappable(norm=angle_norm, cmap=angle_cmap)
        sm_angle.set_array([])
        cbar_hue = fig.colorbar(sm_angle, ax=ax, fraction=0.046, pad=0.05)
        cbar_hue.set_label("Angle (deg.)")
        mag_norm = mcolors.Normalize(vmin=mag_min, vmax=mag_max)  # magnitude (value/brightness) colorbar, grayscale
        mag_cmap = cm.gray
        sm_mag = cm.ScalarMappable(norm=mag_norm, cmap=mag_cmap)
        sm_mag.set_array([])
        cbar_mag = fig.colorbar(sm_mag, ax=ax, fraction=0.046, pad=0.13, orientation='horizontal')
        cbar_mag.set_label("Magnitude (Oe)")

    # ---- Gradient rendering -- see plot_matrix_map_for_timestep's matching branch. ----
    elif plot_type == "gradient" and isinstance(data, tuple):
        angle, magnitude, rgba, mean_angle, delta = data  # unpack the angle/magnitude arrays, the RGBA image, the region's mean angle, and its deviation range
        height, width = angle.shape  # this region's array dimensions, for computing its extent below
        if extent is None:
            extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)

        im = ax.imshow(rgba, origin='lower', extent=extent)  # render the already-built RGBA image directly
        if titles:
            if isinstance(titles, dict):
                ax.set_title(titles.get(label, label), fontsize=8)
            elif isinstance(titles, list) and region_idx < len(titles):
                ax.set_title(titles[region_idx], fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
        else:
            ax.set_title(label, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if use_local_scaling:  # scale the colorbar to this frame's own deviation range
            dmin_vmin = np.nanmin(delta[0])
            dmax_vmax = np.nanmax(delta[1])
        elif vmin is not None and vmax is not None:  # scale the colorbar to the caller-provided global range
            dmin_vmin = vmin
            dmax_vmax = vmax
        else:  # no scaling information available at all: fall back to a fixed +/-180 degree range
            dmin_vmin = -180
            dmax_vmax = 180
        if verbose:
            print(f"[DEBUG] plot_matrix_map_for_region_per_timestep: label='{label}', t={t}, "
                  f"deviation colorbar range=[{dmin_vmin}, {dmax_vmax}], mean_angle={mean_angle}.")
        from mumax3_pipeline_utilities import build_deviation_cmap  # builds a colormap matching matrix_gradient_hsv_rgba's own color scheme
        cmap, norm = build_deviation_cmap(mean_angle, dmin_vmin, dmax_vmax)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Angular Deviation (\u00b0)")

    # ---- Averaged quiver rendering -- see plot_matrix_map_for_timestep's matching branch. ----
    elif plot_type == "quiver" and isinstance(data, tuple):
        from mumax3_plot_vectors import compute_averaged_quiver_grid  # the mask/edge-aware windowed-averaging implementation
        u, v = data  # unpack the two in-plane vector components
        height, width = u.shape  # this region's array dimensions, for computing its extent below
        if extent is None:
            extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)
        window = resolve_quiver_window(quiver_window, label, region_meta, verbose=verbose)  # this region's averaging window size
        row_c, col_c, uu, vv = compute_averaged_quiver_grid(u, v, window)  # one averaged arrow per window
        if row_c.size:  # at least one window had valid data
            x_nm = col_c * cell_size_x  # physical x position of each arrow
            y_nm = row_c * cell_size_y  # physical y position of each arrow
            ax.quiver(x_nm, y_nm, uu, vv, angles="xy", scale_units="xy", color="black", width=0.004)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        if titles:
            if isinstance(titles, dict):
                ax.set_title(titles.get(label, label), fontsize=8)
            elif isinstance(titles, list) and region_idx < len(titles):
                ax.set_title(titles[region_idx], fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
        else:
            ax.set_title(label, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # ---- Default rendering -- see plot_matrix_map_for_timestep's matching branch. ----
    else:
        local_vmin = np.nanmin(data) if use_local_scaling else vmin  # this region's color-scale minimum
        local_vmax = np.nanmax(data) if use_local_scaling else vmax  # this region's color-scale maximum
        height, width = data.shape  # this region's array dimensions, for computing its extent below
        if verbose:
            print(f"[DEBUG] plot_matrix_map_for_region_per_timestep: label='{label}', t={t}, "
                  f"color range=[{local_vmin}, {local_vmax}], data shape={data.shape}.")
        if extent is None:
            extent = compute_region_extent(label, width, height, cell_size_x, cell_size_y, region_meta, plot_origin_cells, verbose=verbose)

        im = ax.imshow(
            data,  # the 2D array to plot
            cmap=cmap,  # color map
            origin='lower',  # row 0 at the bottom, matching this pipeline's convention elsewhere
            vmin=local_vmin,  # color-scale minimum
            vmax=local_vmax,  # color-scale maximum
            extent=extent  # physical axis bounds, in nm
        )

        if titles:
            if isinstance(titles, dict):  # look up this region's title by label
                ax.set_title(titles.get(label, label), fontsize=8)
            elif isinstance(titles, list) and region_idx < len(titles):  # look up this region's title by position
                ax.set_title(titles[region_idx], fontsize=8)
            else:  # no matching title found: fall back to the region's own label
                ax.set_title(label, fontsize=8)
        else:  # no titles provided at all: use the region's own label
            ax.set_title(label, fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)

    if suptitle:  # an overall figure title was provided for this timestep
        plt.suptitle(suptitle, fontsize=12)

    plt.tight_layout()

    # ---- Same output layout as plot_matrix_map_for_timestep -- see that function for
    # the full rationale. ----
    meta = (region_meta or {}).get(label, {})  # this region's metadata, if any
    center_name = meta.get("center_name", sanitize_label(label))  # this region's display folder name
    mask_type = meta.get("mask_type", "No Mask")  # this region's mask-type folder name
    region_dir = os.path.join(frame_dir, sanitize_label(center_name), sanitize_label(mask_type))
    os.makedirs(region_dir, exist_ok=True)

    region_filename = f"{sanitize_label(label)}.png"  # this region's own PNG filename

    plt.savefig(
        os.path.join(region_dir, region_filename),
        dpi=dpi
    )

    if save_plot_data_csv:  # also export the exact plotted data as a companion CSV
        csv_path = os.path.join(region_dir, f"{sanitize_label(label)}.csv")
        save_plotted_data_csv(csv_path, data, verbose=verbose)

    plt.close(fig)  # free this figure's memory
    if extent_refresh:  # reset so a subsequent call (in this same worker process) recomputes its own extent
        extent = None


# --------------------------
# ---- Dispatch one timestep's regions across a process pool (processor_mode="regions") ----
def plot_all_regions_over_time_parallel_helper(
    t: int,  # frame index to render
    regions: Dict[str, Dict[str, Any]],  # region data, keyed by label, as returned by region_extraction.reorganize_regions
    output_dir: str,  # root directory this plot type/scale's frames are written under
    cols: int,  # forwarded to plot_matrix_map_for_region_per_timestep
    rows: int,  # forwarded to plot_matrix_map_for_region_per_timestep
    scalar_func: Callable[[Dict[str, Any], int], np.ndarray],  # computes the value(s) to plot for one region/timestep
    vmin: Optional[float] = None,  # global color-scale minimum
    vmax: Optional[float] = None,  # global color-scale maximum
    cmap: str = 'viridis',  # matplotlib colormap
    titles: Optional[Union[Dict[str, str], list]] = None,  # per-region plot titles
    suptitle: Optional[str] = None,  # figure-level title for this timestep
    xlabel: str = "X (nm)",  # x-axis label
    ylabel: str = "Y (nm)",  # y-axis label
    cbar_label: str = "Value",  # colorbar label
    prefix: str = "matrix_map",  # filename prefix
    use_local_scaling: bool = False,  # if True, scale each region's own color range independently
    cell_size_x: float = 1,  # nm per grid cell, x direction
    cell_size_y: float = 1,  # nm per grid cell, y direction
    field_scale: Optional[float] = None,  # multiplier forwarded into scalar_func
    plot_type: Optional[str] = None,  # None, "hue", "gradient", or "quiver"
    extent: Optional[Tuple[float, float, float, float]] = None,  # explicit axis extent in nm
    max_workers: int = 8,  # number of worker processes to render this timestep's regions with
    region_meta: Optional[Dict[str, Dict[str, str]]] = None,  # see plot_matrix_map_for_timestep
    quiver_window: Optional[Union[Tuple[int, int], Dict[str, Tuple[int, int]]]] = None,  # see plot_matrix_map_for_timestep
    plot_origin_cells: Optional[Tuple[float, float]] = None,  # see plot_matrix_map_for_timestep
    save_plot_data_csv: bool = False,  # also write a companion .csv of the exact plotted array next to each PNG
    verbose: bool = False,  # forwarded to every worker, so per-region debug messages appear when True
    force_serial: bool = False,  # if True, ALWAYS render every region directly in this process, never opening a ProcessPoolExecutor -- see PARAMETERS below
) -> None:
    """
    GOAL:
        Render every region at timestep `t` by dispatching each region to
        its own call of plot_matrix_map_for_region_per_timestep, across a
        pool of worker processes.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_all_regions_over_time (this file) once per timestep,
        when processor_mode="regions". Dispatches
        plot_matrix_map_for_region_per_timestep (this file), once per
        region, across a ProcessPoolExecutor.

    PARAMETERS:
        See plot_matrix_map_for_region_per_timestep, whose parameters this
        function forwards directly, plus:
        max_workers (int): number of worker processes to render this
            timestep's regions with. Capped to len(regions) below -- never
            spawns more workers than there are regions to hand them.
        force_serial (bool): if True, always render every region directly
            in this process, regardless of region count. For callers that
            are THEMSELVES already running inside a worker process
            dispatched by an outer pool (for example, propagation_engine.
            _run_canvas_job, one job among several running concurrently
            across propagation_engine's own per-step canvas_pool) --
            opening a SECOND, nested pool in here would oversubscribe CPUs
            (outer_workers x inner_workers processes total, all competing
            for the same cores) rather than actually speed anything up.
            See SpatialAnalysisConfig.force_serial's own documentation for
            how this reaches here from a caller several layers up.

    RETURNS:
        None. Every region at this timestep is rendered to disk before this
        function returns (each worker's result is awaited via
        `future.result()`, which also re-raises any exception a worker
        encountered).

    LIMITATIONS:
        When force_serial or len(regions) <= 1, no ProcessPoolExecutor is
        created at all -- every region is rendered directly, in this
        process (see the TRACKED CHANGE comment below). A worker pool's
        startup cost is fixed and paid on every call regardless of how
        much work is inside it; for a single-region call (for example,
        propagation's own whole-array "canvas_full_image" pass, or any
        per-device pass restricted to just a source + one candidate) that
        cost was previously paid to render exactly one plot.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed  # parallel processing across regions

    # ---- TRACKED CHANGE (perf): see this function's own force_serial/
    # max_workers documentation above. Mirrors the same skip-the-pool-for-
    # trivial-task-count pattern already applied to frame_pipeline.
    # process_frames_parallel and superposition_operations.run_
    # superposition_multiprocessing -- this is the plotting-side instance
    # of the same fix, previously missing here (this function opened a
    # ProcessPoolExecutor(max_workers=8) UNCONDITIONALLY, even for a single
    # region, and that worker count was never connected to any config
    # field -- see SpatialAnalysisConfig.plot_max_workers' own
    # documentation for how a caller now actually controls it). ----
    if force_serial or len(regions) <= 1:
        if verbose:
            reason = "force_serial=True" if force_serial else f"only {len(regions)} region(s)"
            print(f"[DEBUG] plot_all_regions_over_time_parallel_helper: {reason} -- skipping ProcessPoolExecutor, rendering in-process.")
        for region_idx, (label, region_data) in enumerate(regions.items()):
            plot_matrix_map_for_region_per_timestep(
                t=t, region_idx=region_idx, region_data=region_data, label=label,
                output_dir=output_dir, cols=cols, rows=rows, scalar_func=scalar_func,
                vmin=vmin, vmax=vmax, cmap=cmap, titles=titles, suptitle=suptitle,
                xlabel=xlabel, ylabel=ylabel, cbar_label=cbar_label, prefix=prefix,
                use_local_scaling=use_local_scaling, cell_size_x=cell_size_x, cell_size_y=cell_size_y,
                field_scale=field_scale, plot_type=plot_type, extent=extent, region_meta=region_meta,
                quiver_window=quiver_window, plot_origin_cells=plot_origin_cells,
                save_plot_data_csv=save_plot_data_csv, verbose=verbose,
            )
        return

    with ProcessPoolExecutor(max_workers=min(max_workers, len(regions))) as executor:  # never spawn more workers than there are regions to hand them
        futures = [
            executor.submit(
                plot_matrix_map_for_region_per_timestep,
                t=t,  # frame index
                region_idx=region_idx,  # this region's position, for list-based titles
                region_data=region_data,  # this region's data
                label=label,  # this region's label
                output_dir=output_dir,  # output directory
                cols=cols,
                rows=rows,
                scalar_func=scalar_func,  # value(s)-to-plot function
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                titles=titles,
                suptitle=suptitle,
                xlabel=xlabel,
                ylabel=ylabel,
                cbar_label=cbar_label,
                prefix=prefix,
                use_local_scaling=use_local_scaling,
                cell_size_x=cell_size_x,
                cell_size_y=cell_size_y,
                field_scale=field_scale,
                plot_type=plot_type,
                extent=extent,
                region_meta=region_meta,
                quiver_window=quiver_window,
                plot_origin_cells=plot_origin_cells,
                save_plot_data_csv=save_plot_data_csv,
                verbose=verbose,
            )
            for region_idx, (label, region_data) in enumerate(regions.items())  # one worker submission per region
        ]
        for future in as_completed(futures):  # wait for every region to finish, re-raising any worker exception
            future.result()


# --------------------------
# ---- Top-level entry point: render every timestep for one plot type/scale ----
def plot_all_regions_over_time(
    regions: Dict[str, Dict[str, np.ndarray]],  # region data, keyed by label, as returned by region_extraction.reorganize_regions
    output_dir: str,  # root directory this plot type/scale's frames are written under
    scalar_func: Callable[[Dict[str, Any], int], np.ndarray],  # computes the value(s) to plot for one region/timestep
    max_workers: int = 8,  # number of parallel workers (processes or, depending on processor_mode, per-timestep region workers)
    titles: Optional[Union[Dict[str, str], list]] = None,  # per-region plot titles
    suptitles: Optional[Union[list, dict, Callable[[int], str]]] = None,  # per-frame figure titles -- see resolve_suptitle
    cmap: str = 'viridis',  # matplotlib colormap
    vmin: Optional[float] = None,  # global color-scale minimum
    vmax: Optional[float] = None,  # global color-scale maximum
    xlabel: str = "X (nm)",  # x-axis label
    ylabel: str = "Y (nm)",  # y-axis label
    cbar_label: str = "Value",  # colorbar label
    prefix: str = "matrix_map",  # output filename prefix
    local_scale: bool = False,  # if True, scale each region's own color range independently
    max_frames: Optional[int] = None,  # cap the number of timesteps rendered; None renders every timestep
    cell_size_x: float = 1,  # nm per grid cell, x direction
    cell_size_y: float = 1,  # nm per grid cell, y direction
    field_scale: Optional[float] = None,  # multiplier forwarded into scalar_func
    plot_type: Optional[str] = None,  # None, "hue", "gradient", or "quiver"
    extent: Optional[Tuple[float, float, float, float]] = None,  # explicit axis extent in nm
    processor_mode: str = "auto",  # "timesteps": dispatch whole timesteps across a process pool; "regions"/"auto": dispatch regions within each timestep -- see LIMITATIONS
    region_meta: Optional[Dict[str, Dict[str, str]]] = None,  # see plot_matrix_map_for_timestep
    quiver_window: Optional[Union[Tuple[int, int], Dict[str, Tuple[int, int]]]] = None,  # see plot_matrix_map_for_timestep
    plot_origin_cells: Optional[Tuple[float, float]] = None,  # see plot_matrix_map_for_timestep
    save_plot_data_csv: bool = False,  # also write a companion .csv of the exact plotted array next to each PNG
    verbose: bool = False,  # print progress messages when True
    force_serial: bool = False,  # if True, never open a ProcessPoolExecutor at any level of this function -- forwarded to plot_matrix_map_for_timestep's dispatch below and to plot_all_regions_over_time_parallel_helper; see that function's own documentation for why a caller running inside an outer pool needs this
) -> None:
    """
    GOAL:
        Render every timestep of `regions`, for one plot type and scale,
        using one of two parallelization strategies (see `processor_mode`).
        This is the top-level plotting entry point
        mumax3_parent_utilities.run_spatial_plots calls, once per plot
        type/scale combination.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_parent_utilities.run_spatial_plots, once per plot
        type/scale. Calls resolve_suptitle (this file) once per timestep,
        then either plot_matrix_map_for_timestep (processor_mode=
        "timesteps", dispatched across a process pool directly) or
        plot_all_regions_over_time_parallel_helper (processor_mode=
        "regions"/"auto", called once per timestep, which itself dispatches
        regions across a process pool).

    PARAMETERS:
        See the inline comments on each parameter above.
        processor_mode (str): "timesteps" dispatches whole timesteps
            across a process pool, each worker rendering every region for
            one timestep (plot_matrix_map_for_timestep). "regions" (and
            "auto", its alias -- see LIMITATIONS) processes timesteps one
            at a time, dispatching that timestep's regions across a
            process pool each time (plot_all_regions_over_time_parallel_
            helper). Any other value raises ValueError.
        verbose (bool): prints the frame count and worker count at the
            start when True. Forwarded to plot_matrix_map_for_timestep /
            plot_all_regions_over_time_parallel_helper, so their own
            per-region debug messages appear too.
        force_serial (bool): if True, never open a ProcessPoolExecutor,
            in EITHER processor_mode, regardless of timestep/region count
            -- see plot_all_regions_over_time_parallel_helper's own
            documentation for why a caller already running inside an
            outer pool needs this (propagation_engine.py's canvas_pool is
            the current such caller -- see SpatialAnalysisConfig.
            force_serial).

    RETURNS:
        None. Every timestep (up to `max_frames`, if given) is rendered to
        disk before this function returns.

    LIMITATIONS:
        processor_mode="auto" is currently a plain alias for "regions", not
        an actual choice made based on the data (for example, region count
        or frame count) -- despite the name, no automatic strategy
        selection happens. This alias exists because "auto" was previously
        the default value with no matching branch in this function's
        dispatch logic at all, meaning any caller relying on the default
        (rather than explicitly passing "regions") silently rendered
        nothing, with no error. Treating "auto" as "regions" preserves
        every existing caller's behavior (all current callers already pass
        processor_mode="regions" explicitly) while making the default
        value usable instead of silently producing no output.
    """
    import os  # output directory construction
    from concurrent.futures import ProcessPoolExecutor, as_completed  # parallel processing across timesteps
    os.makedirs(output_dir, exist_ok=True)  # ensure the output directory exists

    sample_label = next(iter(regions))  # an arbitrary region, used only to determine the timestep count
    num_timesteps = len(regions[sample_label]['x'])  # number of timesteps, assumed the same for every region
    if max_frames is not None:  # cap the number of timesteps rendered
        num_timesteps = min(num_timesteps, max_frames)
    num_regions = len(regions)  # number of regions, used only to size cols/rows below
    cols = int(np.ceil(np.sqrt(num_regions)))  # a roughly square subplot grid layout (forwarded for interface consistency; the rendering functions this calls place each region in its own figure)
    rows = int(np.ceil(num_regions / cols))
    if verbose:
        print(f"[INFO] plot_all_regions_over_time: rendering {num_timesteps} frame(s) using {max_workers} worker(s), processor_mode='{processor_mode}'.")

    if processor_mode not in ("timesteps", "regions", "auto"):  # guard against a genuinely invalid value, rather than silently rendering nothing (see LIMITATIONS)
        raise ValueError(f"processor_mode must be 'timesteps', 'regions', or 'auto', got '{processor_mode}'.")

    # ---- Dispatch mode: whole timesteps across a process pool ----
    if processor_mode == "timesteps":
        if force_serial or num_timesteps <= 1:  # see plot_all_regions_over_time_parallel_helper's own TRACKED CHANGE comment -- same fast path, one level up (timesteps instead of regions)
            if verbose:
                reason = "force_serial=True" if force_serial else f"only {num_timesteps} timestep(s)"
                print(f"[DEBUG] plot_all_regions_over_time: {reason} -- skipping ProcessPoolExecutor, rendering in-process.")
            for t in range(num_timesteps):
                plot_matrix_map_for_timestep(
                    t=t, regions=regions, output_dir=output_dir, cols=cols, rows=rows, scalar_func=scalar_func,
                    vmin=vmin, vmax=vmax, cmap=cmap, titles=titles,
                    suptitle=resolve_suptitle(suptitles=suptitles, t=t, verbose=verbose),
                    xlabel=xlabel, ylabel=ylabel, cbar_label=cbar_label, prefix=prefix,
                    use_local_scaling=local_scale, cell_size_x=cell_size_x, cell_size_y=cell_size_y,
                    field_scale=field_scale, plot_type=plot_type, extent=extent if extent is not None else None,
                    region_meta=region_meta, quiver_window=quiver_window, plot_origin_cells=plot_origin_cells,
                    save_plot_data_csv=save_plot_data_csv, verbose=verbose,
                )
            return
        with ProcessPoolExecutor(max_workers=min(max_workers, num_timesteps)) as executor:  # never spawn more workers than there are timesteps to hand them
            futures = [
                executor.submit(
                    plot_matrix_map_for_timestep,  # renders every region for one timestep, in one worker process
                    t=t,  # frame index
                    regions=regions,  # every region's data
                    output_dir=output_dir,  # output directory
                    cols=cols,
                    rows=rows,
                    scalar_func=scalar_func,  # value(s)-to-plot function
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    titles=titles,
                    suptitle=resolve_suptitle(suptitles=suptitles, t=t, verbose=verbose),  # this timestep's own figure title
                    xlabel=xlabel,
                    ylabel=ylabel,
                    cbar_label=cbar_label,
                    prefix=prefix,
                    use_local_scaling=local_scale,
                    cell_size_x=cell_size_x,
                    cell_size_y=cell_size_y,
                    field_scale=field_scale,
                    plot_type=plot_type,
                    extent=extent if extent is not None else None,
                    region_meta=region_meta,
                    quiver_window=quiver_window,
                    plot_origin_cells=plot_origin_cells,
                    save_plot_data_csv=save_plot_data_csv,
                    verbose=verbose,
                )
                for t in range(num_timesteps)  # one worker submission per timestep
            ]
            for future in as_completed(futures):  # wait for every timestep to finish, re-raising any worker exception
                future.result()
    # ---- Dispatch mode: regions within each timestep across a process pool ("auto" is an alias -- see LIMITATIONS) ----
    elif processor_mode in ("regions", "auto"):
        for t in range(num_timesteps):  # process one timestep at a time, dispatching that timestep's regions each time
            plot_all_regions_over_time_parallel_helper(
                t=t,  # frame index
                regions=regions,  # every region's data
                output_dir=output_dir,  # output directory
                cols=cols,
                rows=rows,
                scalar_func=scalar_func,  # value(s)-to-plot function
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                titles=titles,
                suptitle=resolve_suptitle(suptitles=suptitles, t=t, verbose=verbose),  # this timestep's own figure title
                xlabel=xlabel,
                ylabel=ylabel,
                cbar_label=cbar_label,
                prefix=prefix,
                use_local_scaling=local_scale,
                cell_size_x=cell_size_x,
                cell_size_y=cell_size_y,
                field_scale=field_scale,
                plot_type=plot_type,
                extent=extent if extent is not None else None,
                max_workers=max_workers,
                region_meta=region_meta,
                quiver_window=quiver_window,
                plot_origin_cells=plot_origin_cells,
                save_plot_data_csv=save_plot_data_csv,
                verbose=verbose,
                force_serial=force_serial,
            )


# --------------------------
# ---- Make a region label filesystem-safe ----
def sanitize_label(label: str, verbose: bool = False) -> str:
    """
    GOAL:
        Replace characters that are unsafe or awkward in filenames/folder
        names (spaces, colons, forward slashes) with filesystem-friendly
        equivalents.

    HOW IT TALKS TO OTHER FILES:
        Called throughout this file, and by mumax3_csv_stats.
        save_region_stats_csv, anywhere a region label or metadata string
        becomes part of an output path.

    PARAMETERS:
        label (str): the string to sanitize.
        verbose (bool): prints the input and output strings when True.

    RETURNS:
        str: `label` with spaces replaced by underscores, and colons and
        forward slashes removed/replaced.
    """
    result = label.replace(" ", "_").replace(":", "").replace("/", "_")  # spaces -> underscores, colons removed, forward slashes -> underscores
    if verbose:
        print(f"[DEBUG] sanitize_label: '{label}' -> '{result}'.")
    return result


# --------------------------
# ---- Plot each weighted-circular-statistics metric over time, per region ----
def plot_region_stats_over_time(
    stats: Dict[str, Dict[str, List[float]]],  # nested stats per region, from mumax3_csv_stats.compute_stats_for_all_timesteps_parallel
    output_dir: str,  # root folder to save plots under
    time_step_ns: float = 0.1,  # nanoseconds per timestep, for the x-axis
    dpi: int = 300,  # resolution for saved PNGs
    figsize: Tuple[int, int] = (6, 4),  # size of each metric's figure
    xlim: Tuple[float, float] = None,  # optional fixed x-axis limits
    region_meta: Optional[Dict[str, Dict[str, str]]] = None,  # {label -> {"center_name","mask_type"}}, see plot_matrix_map_for_timestep
    verbose: bool = False,  # print a per-metric status line when True
) -> None:
    """
    GOAL:
        Plot each statistical metric (mean angle, mean magnitude, angular
        concentration/variance, and their derived deviation metrics) over
        time, as its own figure, for every region.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts' STEP 4, once per dataset, with the
        output of mumax3_csv_stats.compute_stats_for_all_timesteps_parallel.
        Calls sanitize_label (this file).

    PARAMETERS:
        stats (Dict): nested statistics per region, from
            mumax3_csv_stats.compute_stats_for_all_timesteps_parallel.
        output_dir (str): root folder to save plots under.
        time_step_ns (float): nanoseconds per timestep, for the x-axis.
        dpi (int): resolution for saved PNGs.
        figsize (Tuple[int, int]): size of each metric's own figure.
        xlim (Tuple[float, float], optional): fixed x-axis limits, applied
            to every metric's figure.
        region_meta: Optional[Dict[label -> {"center_name", "mask_type"}]],
            from region_extraction.build_region_meta. When given, each
            region's plots go under "<output_dir>/<center_name>/
            <mask_type>/" instead of just "<output_dir>/<region_label>/",
            so regions sharing a center but differing in mask_mode land in
            separate folders (the same treatment applied to the spatial
            maps in plot_matrix_map_for_timestep).
        verbose (bool): prints a status line per metric plotted when True
            (the final "plots saved to" completion message is always shown
            -- see this function's LIMITATIONS).

    RETURNS:
        None. Writes one PNG per (region, metric) pair under
        "<output_dir>/<center_name>/<mask_type>/<metric_name>.png" (or
        "<output_dir>/<region_label>/<metric_name>.png" if `region_meta`
        is not given).

    LIMITATIONS:
        The y-axis auto-scaling for angle-related metrics (deviation:
        0-180 degrees; concentration/variance: 0-1; wrapped angles:
        0-360 degrees; unwrapped angles: no fixed limit) is matched by
        checking substrings in each metric's own name ("angle",
        "unwrapped", "deviation", "concentration", "variance") -- adding a
        new angle-related metric with a name that does not contain one of
        these substrings will not get automatic y-axis scaling and will
        fall through to matplotlib's own automatic scaling instead. The
        final "plots saved to" message is printed unconditionally (not
        gated by `verbose`) since it reports where output was written,
        which is useful to know even when running quietly otherwise.
    """
    import os  # path construction
    import matplotlib.pyplot as plt  # figure rendering
    os.makedirs(output_dir, exist_ok=True)  # ensure the output directory exists

    for region_label, metrics in stats.items():  # one output folder per region
        meta = (region_meta or {}).get(region_label, {})  # this region's metadata, if any
        center_name = meta.get("center_name", region_label)  # this region's display folder name
        mask_type = meta.get("mask_type")  # this region's mask-type folder name, if known
        if mask_type:  # metadata is available: nest under center_name/mask_type
            region_folder = os.path.join(output_dir, sanitize_label(center_name), sanitize_label(mask_type))
        else:  # no metadata available: fall back to a flat, region_label-only folder
            region_folder = os.path.join(output_dir, sanitize_label(region_label))
        os.makedirs(region_folder, exist_ok=True)

        num_timesteps = len(next(iter(metrics.values())))  # number of timesteps, read from any one metric
        time_array = [i * time_step_ns for i in range(num_timesteps)]  # physical time (ns) at each timestep index

        for metric_name, values in metrics.items():  # one figure per metric
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(time_array, values, label=metric_name, color='tab:blue')
            ax.set_title(f"{metric_name.replace('_', ' ').title()} Over Time", fontsize=10)
            ax.set_xlabel("Time (ns)")
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            if xlim:  # a fixed x-axis range was requested
                ax.set_xlim(xlim)
            # ---- Auto y-axis range for angle-related metrics, matched by substring in
            # the metric's own name (see this function's LIMITATIONS). ----
            if "angle" in metric_name.lower():
                if "unwrapped" in metric_name.lower():  # unwrapped angles can exceed 360 degrees; no fixed limit
                    pass
                elif "deviation" in metric_name.lower():  # deviation metrics range from 0 to 180 degrees
                    ax.set_ylim(0, 180)
                elif "concentration" in metric_name.lower():  # concentration ranges from 0 to 1
                    ax.set_ylim(0, 1)
                elif "variance" in metric_name.lower():  # variance ranges from 0 to 1
                    ax.set_ylim(0, 1)
                else:  # mean/wrapped angles range from 0 to 360 degrees
                    ax.set_ylim(0, 360)
            plt.tight_layout()
            plot_path = os.path.join(region_folder, f"{metric_name}.png")
            plt.savefig(plot_path, dpi=dpi)
            plt.close()
            if verbose:
                print(f"[DEBUG] plot_region_stats_over_time: saved '{plot_path}'.")
    print(f"[INFO] plot_region_stats_over_time: region stats plots saved to: {output_dir}")

