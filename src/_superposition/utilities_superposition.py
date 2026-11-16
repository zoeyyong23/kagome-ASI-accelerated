r"""
File Name: utilities_superposition.py
Author: Connor Sullivan
Date: 2026.07.27
Revision: v1

============================================================
PURPOSE
============================================================
Saves DeviceArray layout plots, magnetization plots, and device data
tables, and exports a composed canvas to the .npz frame format the rest of
this pipeline reads. Contains:
    - save_device_array_plot: saves a plot of the DeviceArray's geometry
      (device positions/orientations, drawn as ellipses).
    - save_magnetization_plot: saves a quiver plot of the DeviceArray's
      magnetization directions.
    - save_device_array_magnetization: thin wrapper around
      save_magnetization_plot.
    - save_device_array_table: saves the DeviceArray's per-device data as a
      table, in any of several formats (CSV, XLSX, TXT, PDF).
    - save_device_array: the combined entry point -- saves a table and/or
      plots in one call, optionally combining a table page and both plot
      pages into a single multi-page PDF.
    - export_canvas_as_npz_frames: exports a composed canvas (as returned
      by superposition_operations.run_superposition_multiprocessing) to the
      "{prefix}{frame_index:06d}.npz" file format frame_pipeline.py and
      mumax3_csv_reader.py read.

============================================================
LIMITATIONS
============================================================
- Revision v1 of this file (see the parent revision) additionally
  contained eleven functions with no callers anywhere in this pipeline,
  removed in this revision:
    - handle_remove_readonly, move_and_reset_directory, get_npz_end_index:
      each an independent duplicate of a same-named, same-purpose function
      in mumax3_pipeline_utilities.py (the copies actually used throughout
      this pipeline). Notably, THIS file's now-removed get_npz_end_index
      used a different filename pattern than the active one (expecting an
      underscore before the frame index, "{prefix}_{index}.npz", instead
      of this pipeline's actual convention, "{prefix}{index:06d}.npz") --
      it would not have matched real files from this pipeline even if it
      had been called.
    - wipe_file, wipe_and_reset_directory: unused directory/file wiping
      utilities, superseded by mumax3_pipeline_utilities.move_and_reset_
      directory (which is safer -- it moves content aside before deleting,
      rather than deleting in place, and handles read-only files).
    - save_geometry_to_temp, load_geometry_from_file: unused simple
      save/load of a device array's raw geometry array to/from .npz.
    - get_components, parse_key: unused helpers for reformatting field
      data dict keys/structure.
    - save_canvas_struct, load_canvas_struct: an unused alternative
      canvas-persistence format (by region/base field/variant, rather than
      export_canvas_as_npz_frames' frame-indexed convention).
  See v1 of this file if any of this needs to be recovered or adapted.
- save_device_array's combined-PDF code path (when both a table and a plot
  format include "pdf") duplicates the table-building logic from
  save_device_array_table and the plot-building logic from
  save_device_array_plot/save_magnetization_plot inline, rather than
  calling those functions directly -- this is necessary because a combined
  PDF needs each page added to the SAME PdfPages object in one open/close
  cycle, which those individual functions do not expose a way to do. A fix
  to one copy of this logic (for example, the table formatting, or the
  boundary-line styling) should be mirrored in the other.
"""

import os  # file path operations
import matplotlib.pyplot as plt  # figure rendering
import matplotlib.patches as patches  # device ellipse rendering
import pandas as pd  # table construction and export


# --------------------------
# ---- Save a plot of the DeviceArray's geometry (device positions/orientations) ----
def save_device_array_plot(device_array_obj, save_path_base, exts=None, verbose=False):
    """
    GOAL:
        Save a plot of a DeviceArray's layout: every device drawn as an
        ellipse at its own position and orientation, with the array's
        boundary lines overlaid.

    HOW IT TALKS TO OTHER FILES:
        Called by save_device_array (this file), and by
        mumax3_parent_superposition.py directly wherever a standalone
        geometry plot is wanted. Reads device_array_obj.device_centers,
        .device_angles, .device_data, and .boundaries -- see geometry.py's
        DeviceArray for what those carry.

    PARAMETERS:
        device_array_obj: a geometry.DeviceArray instance.
        save_path_base (str): output path without extension, for example
            "output/plot".
        exts (List[str], optional): file extensions to save, for example
            ["pdf", "png", "jpeg"]. Defaults to ["pdf"] if not given.
        verbose (bool): prints each saved file's path when True (this
            function always prints a save confirmation regardless -- see
            LIMITATIONS).

    RETURNS:
        None. Saves one file per extension in `exts`.

    LIMITATIONS:
        device_array_obj.device_centers is expected in meters (converted
        to nm for plotting via `* 1e9`), matching geometry.py's own SI-unit
        convention; device_data's own device_length_nm/device_width_nm
        entries, by contrast, are already in nm. The per-file "Saved plot
        as ..." message is printed unconditionally (not gated by
        `verbose`), since it reports where output was written.
    """
    exts = exts if exts is not None else ['pdf']  # default to a single PDF if no extensions given
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (center, angle) in enumerate(zip(device_array_obj.device_centers, device_array_obj.device_angles)):  # one ellipse per device
        ellipse = patches.Ellipse(
            xy=center * 1e9,  # meters -> nm, for plotting
            width=device_array_obj.device_data[idx]["device_length_nm"],  # already in nm
            height=device_array_obj.device_data[idx]["device_width_nm"],  # already in nm
            angle=angle,  # device orientation, degrees
            linewidth=1,
            edgecolor='blue',
            facecolor='blue',
            alpha=0.6
        )
        ax.add_patch(ellipse)

    ax.set_title("Device Array Layout")
    ax.set_xlabel("X Position (nm)")
    ax.set_ylabel("Y Position (nm)")

    ax.axvline(x=device_array_obj.boundaries["xmin"] * 1e9, color='r', linestyle='--')  # array boundary, left edge
    ax.axvline(x=device_array_obj.boundaries["xmax"] * 1e9, color='r', linestyle='--')  # array boundary, right edge
    ax.axhline(y=device_array_obj.boundaries["ymin"] * 1e9, color='r', linestyle='--')  # array boundary, bottom edge
    ax.axhline(y=device_array_obj.boundaries["ymax"] * 1e9, color='r', linestyle='--')  # array boundary, top edge

    ax.set_aspect('equal')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path_base), exist_ok=True)  # ensure the destination directory exists

    for ext in exts:  # save once per requested extension
        full_path = f"{save_path_base}.{ext.lower()}"
        fig.savefig(full_path)
        print(f"[INFO] save_device_array_plot: saved plot as '{full_path}'.")

    plt.close(fig)  # free this figure's memory


# --------------------------
# ---- Save a quiver plot of the DeviceArray's magnetization directions ----
def save_magnetization_plot(device_array_obj, save_path_base, exts=None, arrow_scale=0.8, verbose=False):
    """
    GOAL:
        Save a quiver plot of a DeviceArray's magnetization directions --
        one arrow per device, positioned at that device's center, colored
        by its magnetization angle.

    HOW IT TALKS TO OTHER FILES:
        Called by save_device_array_magnetization (this file), which in
        turn is called by save_device_array (this file). Reads
        device_array_obj.device_centers, .device_data, .device_length,
        .device_width, and .boundaries -- see geometry.py's DeviceArray.

    PARAMETERS:
        device_array_obj: a geometry.DeviceArray instance.
        save_path_base (str): output path without extension.
        exts (List[str], optional): file extensions to save. Defaults to
            ["pdf"] if not given.
        arrow_scale (float): fraction of the largest device dimension used
            to size the arrows -- smaller values shrink the arrows
            relative to device size.
        verbose (bool): prints each saved file's path when True (this
            function always prints a save confirmation regardless -- see
            save_device_array_plot's matching note).

    RETURNS:
        None. Saves one file per extension in `exts`.
    """
    import numpy as np  # angle-to-vector conversion

    exts = exts if exts is not None else ['pdf']  # default to a single PDF if no extensions given
    fig, ax = plt.subplots(figsize=(8, 6))

    x = device_array_obj.device_centers[:, 0] * 1e9  # every device's center, x, meters -> nm
    y = device_array_obj.device_centers[:, 1] * 1e9  # every device's center, y, meters -> nm

    m_angles = np.array([
        device_array_obj.device_data[i]["m_angle"]  # this device's magnetization angle
        for i in range(len(device_array_obj.device_data))
    ])

    theta = np.radians(m_angles)  # magnetization angles, converted to radians for the vector components below
    U = np.cos(theta)  # arrow x component (unit length; actual on-screen length set by `scale` below)
    V = np.sin(theta)  # arrow y component

    scale = (max(max(device_array_obj.device_length.values()), max(device_array_obj.device_width.values())) * 1e9 * arrow_scale)  # arrow length, sized relative to the largest device dimension

    q = ax.quiver(
        x, y, U, V,
        m_angles,  # color each arrow by its own magnetization angle
        angles='xy', scale_units='xy', scale=1 / scale, pivot='middle', cmap='hsv'
    )
    q.set_clim(0, 360)  # fixed color range: 0-360 degrees
    cbar = fig.colorbar(q, ax=ax, orientation='vertical', label='Magnetization Angle (\u00b0)')

    ax.set_title("Magnetization Field")
    ax.set_xlabel("X Position (nm)")
    ax.set_ylabel("Y Position (nm)")

    ax.axvline(x=device_array_obj.boundaries["xmin"] * 1e9, color='r', linestyle='--')  # array boundary, left edge
    ax.axvline(x=device_array_obj.boundaries["xmax"] * 1e9, color='r', linestyle='--')  # array boundary, right edge
    ax.axhline(y=device_array_obj.boundaries["ymin"] * 1e9, color='r', linestyle='--')  # array boundary, bottom edge
    ax.axhline(y=device_array_obj.boundaries["ymax"] * 1e9, color='r', linestyle='--')  # array boundary, top edge

    ax.set_aspect('equal')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path_base), exist_ok=True)  # ensure the destination directory exists

    for ext in exts:  # save once per requested extension
        full_path = f"{save_path_base}.{ext.lower()}"
        fig.savefig(full_path)
        print(f"[INFO] save_magnetization_plot: saved magnetization plot as '{full_path}'.")

    plt.close(fig)  # free this figure's memory


# --------------------------
# ---- Thin wrapper around save_magnetization_plot ----
def save_device_array_magnetization(device_array_obj, save_path_base, exts=None, verbose=False):
    """
    GOAL:
        Thin wrapper around save_magnetization_plot, kept as its own named
        function so save_device_array's plot-saving calls read
        symmetrically (one call for geometry, one call for magnetization).

    HOW IT TALKS TO OTHER FILES:
        Called by save_device_array (this file). Calls
        save_magnetization_plot (this file) directly, forwarding every
        argument unchanged.

    PARAMETERS:
        Same as save_magnetization_plot.

    RETURNS:
        None. See save_magnetization_plot.
    """
    save_magnetization_plot(device_array_obj, save_path_base, exts, verbose=verbose)


# --------------------------
# ---- Save the DeviceArray's per-device data as a table (CSV/XLSX/TXT/PDF) ----
def save_device_array_table(device_array_obj, save_path_base, exts=None, verbose=False):
    """
    GOAL:
        Save a DeviceArray's per-device data (position, sublattice,
        geometry/magnetization angle, size) as a table, in any combination
        of CSV, XLSX, TXT, or PDF formats.

    HOW IT TALKS TO OTHER FILES:
        Called by save_device_array (this file). Reads
        device_array_obj.device_data -- see geometry.py's DeviceArray.

    PARAMETERS:
        device_array_obj: a geometry.DeviceArray instance.
        save_path_base (str): output path without extension, for example
            "output/table".
        exts (List[str], optional): file extensions to save, for example
            ["pdf", "csv", "xlsx", "txt"]. Defaults to ["pdf"] if not
            given. Any extension other than these four is skipped with a
            warning.
        verbose (bool): prints each saved file's path when True (this
            function always prints a save/skip confirmation regardless --
            see save_device_array_plot's matching note).

    RETURNS:
        None. Saves one file per extension in `exts` (except unsupported
        extensions, which are skipped).

    LIMITATIONS:
        The "xlsx" format appends a new, timestamp-named sheet to the
        target file each call (creating the file if it does not already
        exist), rather than overwriting the whole file -- calling this
        repeatedly with the same xlsx path accumulates one sheet per call
        instead of replacing the previous one.
    """
    exts = exts if exts is not None else ['pdf']  # default to a single PDF if no extensions given
    device_data = device_array_obj.device_data  # this array's per-device data dict
    df = pd.DataFrame.from_dict(device_data, orient='index')  # one row per device

    df_formatted = pd.DataFrame({  # human-readable column names and units
        "Index": df.index,
        "x (nm)": df['x'] * 1e9,  # meters -> nm
        "y (nm)": df['y'] * 1e9,  # meters -> nm
        "Sublattice": df['sublattice'],
        "Geometry Angle (\u00b0)": df['angle'],
        "Magnetization Angle (\u00b0)": df['m_angle'],
        "Length (nm)": df['device_length_nm'],  # already in nm
        "Width (nm)": df['device_width_nm']  # already in nm
    })

    os.makedirs(os.path.dirname(save_path_base), exist_ok=True)  # ensure the destination directory exists

    for ext in exts:  # save once per requested extension
        ext = ext.lower()
        full_path = f"{save_path_base}.{ext}"

        if ext == 'csv':
            df_formatted.to_csv(full_path, index=False)
            print(f"[INFO] save_device_array_table: saved table as '{full_path}'.")

        elif ext == 'xlsx':
            with pd.ExcelWriter(full_path, mode='a' if os.path.exists(full_path) else 'w') as writer:  # append a new sheet if the file already exists (see LIMITATIONS)
                import time  # unique-enough sheet name for this call
                sheet_name = f"Sheet_{int(time.time())}"
                df_formatted.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"[INFO] save_device_array_table: saved table as '{full_path}' (sheet: {sheet_name}).")

        elif ext == 'txt':
            df_formatted.to_csv(full_path, sep='\t', index=False)  # tab-separated, same content as the CSV
            print(f"[INFO] save_device_array_table: saved table as '{full_path}'.")

        elif ext == 'pdf':
            fig_height = max(2, len(df_formatted) * 0.25)  # scale figure height to the number of devices
            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.axis('off')
            table = ax.table(cellText=[df_formatted.columns.to_list()] + df_formatted.round(3).astype(str).values.tolist(),
                             loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            plt.tight_layout()
            fig.savefig(full_path)
            plt.close(fig)
            print(f"[INFO] save_device_array_table: saved table as '{full_path}'.")

        else:  # not one of the four supported extensions
            print(f"[WARNING] save_device_array_table: unsupported extension '{ext}', skipping.")


# --------------------------
# ---- Combined entry point: save a table and/or plots, optionally combined into one PDF ----
def save_device_array(device_array_obj, base_save_path, save_table_exts=None, save_plot_exts=None, verbose=False):
    """
    GOAL:
        Save a DeviceArray's table and/or plots (geometry and
        magnetization) in one call. When both a table format and a plot
        format include "pdf", produces one combined multi-page PDF (table
        page, geometry page, magnetization page) instead of three separate
        single-purpose PDFs.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_parent_superposition.py, once per superposition
        run, to save the DeviceArray's own layout documentation. Calls
        save_device_array_table, save_device_array_plot, and
        save_device_array_magnetization (all this file) for the
        individual (non-combined) output formats.

    PARAMETERS:
        device_array_obj: a geometry.DeviceArray instance.
        base_save_path (str): base output path (without extension); table
            outputs are saved as "{base}_table.{ext}", plot outputs as
            "{base}_plot.{ext}" (geometry) and "{base}_magnetization.{ext}"
            (magnetization), and the combined PDF (if produced) as
            "{base}.pdf".
        save_table_exts (List[str], optional): table formats to save, for
            example ["csv", "pdf"]. None saves no table.
        save_plot_exts (List[str], optional): plot formats to save, for
            example ["png", "pdf"]. None saves no plots.
        verbose (bool): forwarded to save_device_array_table/
            save_device_array_plot/save_device_array_magnetization, so
            their own save-confirmation messages appear (those functions
            print unconditionally regardless -- see their own
            documentation).

    RETURNS:
        None. Saves whichever combination of table/plot/combined-PDF
        outputs were requested.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding the
        combined-PDF code path's duplication of table/plot-building logic
        from save_device_array_table/save_device_array_plot/
        save_magnetization_plot.
    """
    from matplotlib.backends.backend_pdf import PdfPages  # multi-page PDF support, for the combined-PDF case
    import numpy as np  # angle-to-vector conversion, for the combined-PDF magnetization page

    save_table_exts = [ext.lower().lstrip(".") for ext in (save_table_exts or [])]  # normalize extensions (lowercase, no leading dot)
    save_plot_exts = [ext.lower().lstrip(".") for ext in (save_plot_exts or [])]

    os.makedirs(os.path.dirname(base_save_path), exist_ok=True)  # ensure the destination directory exists
    base_path, _ = os.path.splitext(base_save_path)  # strip any extension accidentally included in base_save_path

    wants_table_pdf = "pdf" in save_table_exts  # a table PDF was requested
    wants_plot_pdf = "pdf" in save_plot_exts  # a plot PDF was requested
    wants_combined_pdf = wants_table_pdf and wants_plot_pdf  # both were requested: combine them into one multi-page PDF instead of two separate ones

    if wants_combined_pdf:
        combined_path = f"{base_path}.pdf"
        with PdfPages(combined_path) as pdf:
            if verbose:
                print(f"[INFO] save_device_array: saving combined PDF to '{combined_path}'.")

            # ---- Page 1: table (duplicated from save_device_array_table's PDF
            # branch -- see this file's module-level LIMITATIONS) ----
            fig_height = max(2, len(device_array_obj.device_data) * 0.25)  # scale figure height to the number of devices
            fig, ax = plt.subplots(figsize=(10, fig_height))
            ax.axis('off')

            headers = ["Index", "x (nm)", "y (nm)", "sublattice", "Angle (\u00b0)", "Magnetization Angle (\u00b0)", "Length (nm)", "Width (nm)"]
            data_rows = [
                [
                    idx,
                    {k: f"{v * 1e9:.2f}" for k, v in data["x"].items()} if isinstance(data["x"], dict) else f"{data['x']*1e9:.2f}",  # supports either a single value or a per-something dict for x
                    {k: f"{v * 1e9:.2f}" for k, v in data["y"].items()} if isinstance(data["y"], dict) else f"{data['y']*1e9:.2f}",  # supports either a single value or a per-something dict for y
                    f"{data['angle']:.1f}",
                    f"{data['m_angle']:.1f}",
                    f"{data['sublattice']}",
                    f"{data['device_length_nm']:.1f}",
                    f"{data['device_width_nm']:.1f}",
                ]
                for idx, data in device_array_obj.device_data.items()
            ]
            table_data = [headers] + data_rows

            table = ax.table(cellText=table_data, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ---- Page 2: geometry plot (duplicated from save_device_array_plot --
            # see this file's module-level LIMITATIONS) ----
            fig, ax = plt.subplots(figsize=(8, 6))
            for idx, (center, angle) in enumerate(zip(device_array_obj.device_centers, device_array_obj.device_angles)):  # one ellipse per device
                ellipse = patches.Ellipse(
                    xy=center * 1e9,  # meters -> nm
                    width=device_array_obj.device_data[idx]["device_length_nm"],
                    height=device_array_obj.device_data[idx]["device_width_nm"],
                    angle=angle,
                    linewidth=1,
                    edgecolor='blue',
                    facecolor='blue',
                    alpha=0.6
                )
                ax.add_patch(ellipse)
            ax.set_title("Device Array Layout")
            ax.set_xlabel("X Position (nm)")
            ax.set_ylabel("Y Position (nm)")
            ax.axvline(x=device_array_obj.boundaries["xmin"] * 1e9, color='r', linestyle='--')
            ax.axvline(x=device_array_obj.boundaries["xmax"] * 1e9, color='r', linestyle='--')
            ax.axhline(y=device_array_obj.boundaries["ymin"] * 1e9, color='r', linestyle='--')
            ax.axhline(y=device_array_obj.boundaries["ymax"] * 1e9, color='r', linestyle='--')
            ax.set_aspect('equal')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # ---- Page 3: magnetization plot (duplicated from save_magnetization_plot
            # -- see this file's module-level LIMITATIONS) ----
            fig, ax = plt.subplots(figsize=(8, 6))
            x = device_array_obj.device_centers[:, 0] * 1e9  # every device's center, x, meters -> nm
            y = device_array_obj.device_centers[:, 1] * 1e9  # every device's center, y, meters -> nm
            m_angles = np.array([
                device_array_obj.device_data[i]["m_angle"]
                for i in range(len(device_array_obj.device_data))
            ])
            theta = np.radians(m_angles)  # magnetization angles, in radians
            U = np.cos(theta)  # arrow x component
            V = np.sin(theta)  # arrow y component
            scale = (max(max(device_array_obj.device_length.values()), max(device_array_obj.device_width.values())) * 1e9 * 0.8)  # arrow length, sized relative to the largest device dimension

            q = ax.quiver(
                x, y, U, V,
                m_angles,  # color each arrow by its own magnetization angle
                angles='xy', scale_units='xy', scale=1 / scale, pivot='middle', cmap='hsv'
            )
            q.set_clim(0, 360)  # fixed color range: 0-360 degrees
            cbar = fig.colorbar(q, ax=ax, orientation='vertical', label='Magnetization Angle (\u00b0)')
            ax.set_title("Magnetization Field")
            ax.set_xlabel("X Position (nm)")
            ax.set_ylabel("Y Position (nm)")
            ax.axvline(x=device_array_obj.boundaries["xmin"] * 1e9, color='r', linestyle='--')
            ax.axvline(x=device_array_obj.boundaries["xmax"] * 1e9, color='r', linestyle='--')
            ax.axhline(y=device_array_obj.boundaries["ymin"] * 1e9, color='r', linestyle='--')
            ax.axhline(y=device_array_obj.boundaries["ymax"] * 1e9, color='r', linestyle='--')
            ax.set_aspect('equal')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    # ---- Individual (non-combined) table outputs ----
    for ext in save_table_exts:  # save each requested table format on its own
        save_device_array_table(device_array_obj, save_path_base=base_path + "_table", exts=[ext], verbose=verbose)

    # ---- Individual (non-combined) plot outputs ----
    for ext in save_plot_exts:  # save each requested plot format on its own (both geometry and magnetization)
        save_device_array_plot(device_array_obj, save_path_base=base_path + "_plot", exts=[ext], verbose=verbose)
        save_device_array_magnetization(device_array_obj, save_path_base=base_path + "_magnetization", exts=[ext], verbose=verbose)


# --------------------------
# ---- Export a composed canvas to the pipeline's standard .npz frame format ----
def export_canvas_as_npz_frames(canvas_struct, output_dir, output_filename=None, frame_index=None, verbose=False):
    """
    GOAL:
        Export a composed canvas (as returned by superposition_operations.
        run_superposition_multiprocessing) to the
        "{prefix}{frame_index:06d}.npz" file format frame_pipeline.py and
        mumax3_csv_reader.py expect to read region data from -- the bridge
        between "a composed canvas in memory" and "a folder of .npz frames
        the rest of this pipeline can analyze like any other mumax3
        output."

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_parent_superposition.py, once per superposition
        run, on the result of superposition_operations.
        run_superposition_multiprocessing. Writes files in the same
        "{prefix}_npz_cache/{prefix}{index:06d}.npz" layout
        mumax3_csv_reader.save_to_npz produces, so downstream code
        (frame_pipeline.process_frames_parallel,
        mumax3_pipeline_utilities.get_npz_end_index) can read them
        identically regardless of whether they came from a real mumax3 CSV
        or from this composition step.

    PARAMETERS:
        canvas_struct (Dict): nested {region_key: {base_key: {variant_key:
            {"x": ..., "y": ...}}}} -- the composed canvas data, in the
            same shape run_superposition_multiprocessing returns.
        output_dir (str): base directory to export into; each base_key
            gets its own "{base_key}_npz_cache" subdirectory.
        output_filename (str, optional): explicit output filename,
            overriding the usual "{base_key}{frame_idx:06d}.npz" naming.
            Rarely needed -- see LIMITATIONS for why this does not compose
            well with multiple variants.
        frame_index (int, optional): explicit frame index to use for every
            variant exported in this call, overriding each variant's own
            key-derived index (see LIMITATIONS).
        verbose (bool): prints each exported file's path when True (this
            function always prints an export confirmation regardless --
            see save_device_array_plot's matching note in this file).

    RETURNS:
        None. Writes one .npz file per (base_key, variant_key) combination
        found in `canvas_struct`.

    LIMITATIONS:
        A composed canvas has no z component by construction (the
        superposition operations in this pipeline are entirely in-plane);
        a zero-filled z array is written alongside x/y purely because
        downstream code (frame_pipeline.py, mumax3_csv_reader.py)
        universally expects all three components to be present.

        The output frame index for each variant is resolved as: use
        `frame_index` if explicitly given (applied to every variant in
        this call alike); otherwise, use `variant_key` itself if it is a
        plain digit string (the usual case -- superposition regenerations
        are keyed 0, 1, 2, ...); otherwise, default to frame index 0. This
        means a `canvas_struct` with multiple non-numeric variant keys, and
        no explicit `frame_index` given, would have every one of those
        variants collide on frame index 0 and silently overwrite each
        other's output file -- this function does not detect or warn about
        that collision. Passing an explicit `output_filename` has the same
        collision risk across multiple variants, and additionally is
        applied identically across every base_key's own subdirectory in a
        single call, so it should generally be left as None (the default)
        unless exporting exactly one (base_key, variant_key) combination
        per call.
    """
    import numpy as np  # zero-filled z-component array

    for region_key, bases in canvas_struct.items():  # walk every region
        for base_key, variants in bases.items():  # walk every base field within this region

            cache_dir = os.path.join(output_dir, f"{base_key}_npz_cache")  # this base field's own cache subdirectory, matching mumax3_csv_reader.save_to_npz's own layout convention
            os.makedirs(cache_dir, exist_ok=True)

            for variant_key, canvas in variants.items():  # walk every variant of this base field

                x = canvas["x"]  # this variant's composed x component
                y = canvas["y"]  # this variant's composed y component
                z = np.zeros_like(x)  # dummy z component -- see function LIMITATIONS

                if frame_index is not None:  # an explicit frame index was given for this whole call
                    frame_idx = frame_index
                else:  # derive the frame index from this variant's own key (see function LIMITATIONS)
                    frame_idx = int(variant_key) if variant_key.isdigit() else 0
                if output_filename is None:  # the usual case: derive the filename from base_key and frame index
                    filepath = os.path.join(cache_dir, f"{base_key}{frame_idx:06d}.npz")
                else:  # an explicit filename override was given (see function LIMITATIONS regarding its collision risk)
                    filepath = os.path.join(cache_dir, output_filename)

                np.savez_compressed(filepath, x=x, y=y, z=z)

                print(f"[INFO] export_canvas_as_npz_frames: exported '{filepath}'.")


def export_canvas_as_csv_frames(canvas_struct, output_dir, output_filename=None, frame_index=None, verbose=False):
    """
    GOAL:
        Export a composed canvas to a human-viewable CSV, in mumax3's own
        stacked x/y/z format (see mumax3_csv_reader.py's own module-level
        documentation) -- a plain-text companion to export_canvas_as_
        npz_frames' own .npz cache, so a composed canvas can be opened and
        inspected directly (spreadsheet, text editor, mumax3_csv_reader.
        process_csv_file) without needing to load the .npz.

    HOW IT TALKS TO OTHER FILES:
        Called alongside export_canvas_as_npz_frames -- same callers
        (mumax3_parent_superposition.py's STEP 4, propagation_engine.
        compose_and_export_canvas), gated behind its own config toggle
        (SuperpositionConfig.export_canvas_csv / PropagationConfig.
        export_canvas_csv) since it roughly doubles this step's own output
        file count and is not needed for the pipeline to function --
        purely a "let me look at this" convenience. Writes the same
        stacked-block convention mumax3_csv_plot.save_plotted_data_csv
        uses elsewhere in this codebase, so both round-trip through
        mumax3_csv_reader.process_csv_file identically to a real mumax3
        CSV.

    PARAMETERS:
        Identical to export_canvas_as_npz_frames -- see that function's own
        documentation for canvas_struct's shape and every parameter's
        meaning; this function walks the exact same (region_key, base_key,
        variant_key) structure and resolves the same output frame index,
        just writing "{base_key}{frame_idx:06d}.csv" under a
        "{base_key}_csv_export" subdirectory instead of a .npz cache.

    RETURNS:
        None. Writes one .csv file per (base_key, variant_key) combination
        found in `canvas_struct`, each with three blank-line-separated
        blocks (x, then y, then z -- z is the same zero-filled dummy
        export_canvas_as_npz_frames writes, for the same reason: this
        pipeline's superposition is entirely in-plane, but downstream mumax3
        CSV consumers universally expect all three components).

    LIMITATIONS:
        See export_canvas_as_npz_frames' own LIMITATIONS section regarding
        frame-index resolution and output_filename's collision risk --
        identical here.
    """
    import numpy as np  # zero-filled z-component array

    for region_key, bases in canvas_struct.items():
        for base_key, variants in bases.items():

            export_dir = os.path.join(output_dir, f"{base_key}_csv_export")
            os.makedirs(export_dir, exist_ok=True)

            for variant_key, canvas in variants.items():

                x = np.asarray(canvas["x"])
                y = np.asarray(canvas["y"])
                z = np.zeros_like(x)  # dummy z component -- see function LIMITATIONS / export_canvas_as_npz_frames' own matching note

                if frame_index is not None:
                    frame_idx = frame_index
                else:
                    frame_idx = int(variant_key) if variant_key.isdigit() else 0
                if output_filename is None:
                    filepath = os.path.join(export_dir, f"{base_key}{frame_idx:06d}.csv")
                else:
                    filepath = os.path.join(export_dir, output_filename)

                with open(filepath, "w") as f:
                    for component in (x, y, z):  # x block, then y block, then z block -- mumax3's own stacked convention
                        for row in component:
                            f.write(",".join(str(float(v)) for v in row) + "\n")
                        f.write("\n")  # blank-line separator between components

                if verbose:
                    print(f"[DEBUG] export_canvas_as_csv_frames: exported '{filepath}'.")
