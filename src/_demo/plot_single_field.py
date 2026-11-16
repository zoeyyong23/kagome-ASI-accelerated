r"""
File Name: plot_single_field.py
Author: Connor Sullivan
Date: 2026.08.06
Revision: v2

============================================================
PURPOSE
============================================================
Standalone script: load and plot ONE mumax3-style field file (m_full,
B_demag, B_exch, or anything else exported in the same stacked x/y/z CSV
or .npz format -- see mumax3_csv_reader.py's own documentation) with no
dependency on the rest of this pipeline (region_specs, masking,
plot_configs, and so on) -- just this file, numpy, and matplotlib. Meant
as a quick, hackable entry point for one-off exploration of a single
frame, separate from the full spatial-analysis/superposition machinery.
Contains:
    - load_field: loads x, y, z from a CSV or .npz field file.
    - run_custom_analysis: a documented no-op stub, called right after
      loading -- the intended place to add ad hoc analysis.
    - plot_single_field: renders one of magnitude/angle/component/quiver
      for a loaded field.
    - _averaged_quiver_grid: a minimal, dependency-free reimplementation
      of mumax3_plot_vectors.compute_averaged_quiver_grid, used only by
      plot_single_field's quiver mode.

Usage (as a script):
    python plot_single_field.py path/to/B_exch000003.csv --plot_type magnitude
    python plot_single_field.py path/to/m_full000000.csv --plot_type quiver --cell_size 1.0

Usage (as a library, for example from a notebook):
    from plot_single_field import load_field, plot_single_field
    x, y, z = load_field("m_full000000.csv")
    plot_single_field(x, y, z, plot_type="angle", cell_size_x=1.0, cell_size_y=1.0)

Extend run_custom_analysis below -- called right after loading, before
plotting -- with whatever ad hoc analysis is needed; it is a documented
no-op stub on purpose, so this file stays a clean starting point rather
than accumulating pipeline-specific assumptions.

============================================================
LIMITATIONS
============================================================
- _averaged_quiver_grid is a deliberate, small, standalone duplication of
  mumax3_plot_vectors.compute_averaged_quiver_grid, kept here so this
  script has zero dependency on the rest of this pipeline. When working
  inside the full pipeline, use the real one instead; a fix to one is not
  automatically reflected in the other.
"""

import os  # path construction
import numpy as np  # array operations


def load_field(path, verbose=False):
    """
    GOAL:
        Load x, y, z 2D arrays from either a mumax3-style stacked CSV (see
        mumax3_csv_reader.py's own documentation -- blank-line-separated
        x/y/z blocks; also tolerates no blank lines) or a .npz with 'x'/
        'y'/'z' keys (for example, from this pipeline's own extracted-
        region or composed-canvas caches).

    HOW IT TALKS TO OTHER FILES:
        Called by this file's own __main__ block, and intended for direct
        use when this file is imported as a library.

    PARAMETERS:
        path (str): path to a CSV or .npz field file.
        verbose (bool): prints the loaded array shape when True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (x, y, z), each a 2D
        array.

    LIMITATIONS:
        Raises ValueError if a CSV file's row count is not evenly
        divisible by 3 (does not look like a valid stacked x/y/z field
        file).
    """
    if path.lower().endswith(".npz"):  # npz input: load directly by key
        d = np.load(path)
        if verbose:
            print(f"[DEBUG] load_field: loaded '{path}' (npz), shape {d['x'].shape}.")
        return d["x"], d["y"], d["z"]

    with open(path) as f:
        raw_rows = [line.strip() for line in f if line.strip() != ""]  # drop blank block separators
    rows = [list(map(float, r.split(","))) for r in raw_rows]  # every remaining row, parsed to floats
    n = len(rows)
    if n % 3 != 0:  # the row count must split evenly into three equal-height x/y/z blocks
        raise ValueError(
            f"{path}: {n} data rows is not divisible by 3 -- does not look like a valid "
            "stacked x/y/z field file."
        )
    h = n // 3  # height of each of the three blocks
    arr = np.array(rows)
    if verbose:
        print(f"[DEBUG] load_field: loaded '{path}' (csv), shape ({h}, {len(rows[0])}).")
    return arr[0:h], arr[h:2 * h], arr[2 * h:3 * h]  # the three stacked blocks: x, y, z


def run_custom_analysis(x, y, z, cell_size_x=1.0, cell_size_y=1.0, verbose=False):
    """
    GOAL:
        ---- ANALYSIS PORT ----
        Called once, right after loading, before plotting. No-op by
        default -- this is where to add quick checks/metrics/exports for a
        single field file without touching load_field/plot_single_field.

    HOW IT TALKS TO OTHER FILES:
        Called by this file's own __main__ block, immediately after
        load_field.

    PARAMETERS:
        x, y, z: the loaded 2D field component arrays (raw, unscaled --
            apply a field_scaling here if physical units are needed, same
            convention as the rest of this pipeline: for example, *1e4 for
            a B-field in T -> Oe).
        cell_size_x, cell_size_y: nm per cell, as passed to
            plot_single_field.
        verbose (bool): available for use by whatever custom analysis is
            added here; unused by the no-op stub itself.

    RETURNS:
        Whatever is useful -- unused by plot_single_field, purely for
        direct use when calling this module as a library.
    """
    pass


def plot_single_field(x, y, z, plot_type="magnitude", cell_size_x=1.0, cell_size_y=1.0,
                       field_scaling=1.0, cmap="viridis", quiver_window=None,
                       center_at_origin=False, output_path=None, title=None, dpi=200,
                       verbose=False):
    """
    GOAL:
        Render one plot of a loaded field: a scalar color map (magnitude,
        angle, or one raw component) or an averaged quiver plot.

    HOW IT TALKS TO OTHER FILES:
        Called by this file's own __main__ block. Calls
        _averaged_quiver_grid (this file) when plot_type="quiver".

    PARAMETERS:
        x, y, z: 2D field component arrays (from load_field).
        plot_type (str): "magnitude" | "angle" | "mx" | "my" | "mz" |
            "quiver". "quiver" draws one arrow per `quiver_window` cell
            block (see mumax3_plot_vectors.compute_averaged_quiver_grid in
            the main pipeline -- reimplemented minimally and dependency-
            free in _averaged_quiver_grid below) if quiver_window is
            given, else one arrow per raw cell (only sane for small
            arrays -- there is no subsampling here; this script is
            intentionally simple).
        cell_size_x, cell_size_y (float): nm per cell.
        field_scaling (float): multiplies x/y/z before computing/plotting
            (for example, 1e4 for a B-field in T -> Oe, matching the rest
            of this pipeline; 1.0 for m_full).
        cmap (str): matplotlib colormap, used for every plot_type except
            "angle" (always "hsv") and "quiver" (no colormap).
        quiver_window (Tuple[int, int], optional): (rows, cols) averaging
            window, only used if plot_type="quiver". None uses one arrow
            per raw cell.
        center_at_origin (bool): if True, plot axes are centered on the
            array's own middle (x and y both span roughly
            [-W/2*cell_size, +W/2*cell_size]) instead of starting at
            (0, 0) -- matching the rest of this pipeline's
            plot_origin_cells convention, for one array in isolation
            (there is no "canvas" here to share an origin with).
        output_path (str, optional): if given, saves the figure there;
            otherwise shows it interactively (plt.show()).
        title (str, optional): plot title (defaults to `plot_type`).
        dpi (int): resolution for a saved figure.
        verbose (bool): prints the resolved extent and plot type when
            True.

    RETURNS:
        The matplotlib Figure.

    LIMITATIONS:
        Raises KeyError if `plot_type` is not one of the six supported
        values (via the internal scalar-computation dict lookup).
    """
    import matplotlib  # backend selection, before importing pyplot
    if output_path is not None:  # saving to a file: use a non-interactive backend
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float) * field_scaling  # x component, scaled
    y = np.asarray(y, dtype=float) * field_scaling  # y component, scaled
    z = np.asarray(z, dtype=float) * field_scaling  # z component, scaled
    h, w = x.shape

    if center_at_origin:  # array's own middle maps to (0, 0)
        extent = [-w / 2.0 * cell_size_x, w / 2.0 * cell_size_x,
                  -h / 2.0 * cell_size_y, h / 2.0 * cell_size_y]
    else:  # default: local box starting at (0, 0)
        extent = [0, w * cell_size_x, 0, h * cell_size_y]
    if verbose:
        print(f"[DEBUG] plot_single_field: plot_type='{plot_type}', extent={extent}.")

    fig, ax = plt.subplots(figsize=(6, 6))

    if plot_type == "quiver":  # averaged quiver arrows, not a color map
        row_c, col_c, uu, vv = _averaged_quiver_grid(x, y, quiver_window or (1, 1), verbose=verbose)
        if row_c.size:  # at least one window had valid data
            x0, y0 = extent[0], extent[2]  # extent's own origin, so arrow positions land correctly regardless of center_at_origin
            ax.quiver(x0 + col_c * cell_size_x, y0 + row_c * cell_size_y, uu, vv,
                       angles="xy", scale_units="xy", color="black", width=0.004)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal", adjustable="box")
        im = None
    else:  # a scalar color map
        data = {
            "magnitude": lambda: np.hypot(x, y),
            "angle": lambda: np.degrees(np.arctan2(y, x)) % 360,
            "mx": lambda: x, "my": lambda: y, "mz": lambda: z,
        }[plot_type]()  # compute the requested scalar field
        vmin, vmax = (0, 360) if plot_type == "angle" else (None, None)  # fixed 0-360 range for angle; auto-scale otherwise
        cmap_use = "hsv" if plot_type == "angle" else cmap  # cyclic colormap for angle; the requested one otherwise
        im = ax.imshow(data, origin="lower", extent=extent, cmap=cmap_use, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_title(title or plot_type)
    plt.tight_layout()

    if output_path is not None:  # save to a file
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
        plt.close(fig)
        if verbose:
            print(f"[DEBUG] plot_single_field: saved '{output_path}'.")
    else:  # no output path given: show interactively
        plt.show()
    return fig


def _averaged_quiver_grid(u, v, window, verbose=False):
    """
    GOAL:
        Minimal, dependency-free reimplementation of mumax3_plot_vectors.
        compute_averaged_quiver_grid (see that function for the full
        edge-case rationale) -- kept here so this script has zero
        dependency on the rest of this pipeline.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_single_field (this file), when plot_type="quiver".
        See this file's module-level LIMITATIONS section regarding this
        function's relationship to the real implementation elsewhere in
        this codebase.

    PARAMETERS:
        u, v (np.ndarray): 2D (height, width) field components, NaN where
            masked out.
        window (Tuple[int, int]): (rows, cols) cell-count of each
            averaging block.
        verbose (bool): prints the number of averaging windows produced
            when True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (row_centers, col_centers, avg_u, avg_v) -- one entry per window
        that contained at least one valid cell.
    """
    win_rows, win_cols = window  # unpack the requested averaging window size
    u = np.asarray(u, dtype=float)  # ensure a plain float array
    v = np.asarray(v, dtype=float)
    h, w = u.shape
    valid = np.isfinite(u) & np.isfinite(v)  # cells with real (non-NaN) data
    if not valid.any():  # no valid data anywhere in this array
        empty = np.array([], dtype=float)
        if verbose:
            print("[DEBUG] _averaged_quiver_grid: no valid data; returning empty arrays.")
        return empty, empty, empty, empty
    rows_with_valid = np.flatnonzero(valid.any(axis=1))  # every row index with at least one valid cell
    cols_with_valid = np.flatnonzero(valid.any(axis=0))  # every column index with at least one valid cell
    row_start, row_end = int(rows_with_valid[0]), int(rows_with_valid[-1]) + 1  # bounding box of valid data, rows
    col_start, col_end = int(cols_with_valid[0]), int(cols_with_valid[-1]) + 1  # bounding box of valid data, columns
    row_centers, col_centers, avg_u, avg_v = [], [], [], []  # accumulated across every window below
    for r0 in range(row_start, row_end, win_rows):  # step through the valid-data bounding box, one window-height at a time
        r1 = min(r0 + win_rows, h)  # clipped to the array's actual edge
        for c0 in range(col_start, col_end, win_cols):  # step through the bounding box, one window-width at a time
            c1 = min(c0 + win_cols, w)  # clipped to the array's actual edge
            u_win, v_win = u[r0:r1, c0:c1], v[r0:r1, c0:c1]  # this window's components
            if not (np.isfinite(u_win) & np.isfinite(v_win)).any():  # this window has no usable data at all
                continue  # no arrow for this window
            row_centers.append((r0 + r1) / 2.0)  # this window's geometric center, row
            col_centers.append((c0 + c1) / 2.0)  # this window's geometric center, column
            avg_u.append(np.nanmean(u_win))  # this window's average x component
            avg_v.append(np.nanmean(v_win))  # this window's average y component
    if verbose:
        print(f"[DEBUG] _averaged_quiver_grid: produced {len(row_centers)} window(s) from window size {window}.")
    return (np.array(row_centers), np.array(col_centers), np.array(avg_u), np.array(avg_v))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="Path to a mumax3-style CSV or .npz field file")
    parser.add_argument("--plot_type", default="magnitude",
                         choices=["magnitude", "angle", "mx", "my", "mz", "quiver"])
    parser.add_argument("--cell_size", type=float, default=1.0, help="nm per cell (assumes square cells)")
    parser.add_argument("--field_scaling", type=float, default=1.0)
    parser.add_argument("--quiver_window", type=int, nargs=2, default=None, metavar=("ROWS", "COLS"))
    parser.add_argument("--center_at_origin", action="store_true")
    parser.add_argument("--output", default=None, help="Save path (shows interactively if omitted)")
    parser.add_argument("--verbose", action="store_true", help="Print progress/debug messages")
    args = parser.parse_args()

    fx, fy, fz = load_field(args.path, verbose=args.verbose)
    run_custom_analysis(fx, fy, fz, cell_size_x=args.cell_size, cell_size_y=args.cell_size, verbose=args.verbose)
    plot_single_field(
        fx, fy, fz, plot_type=args.plot_type,
        cell_size_x=args.cell_size, cell_size_y=args.cell_size,
        field_scaling=args.field_scaling,
        quiver_window=tuple(args.quiver_window) if args.quiver_window else None,
        center_at_origin=args.center_at_origin,
        output_path=args.output,
        title=f"{os.path.basename(args.path)} -- {args.plot_type}",
        verbose=args.verbose,
    )
