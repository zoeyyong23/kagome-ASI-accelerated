r"""
File Name: generate_demo_seed_data.py
Author: Connor Sullivan
Date: 2026.08.06
Revision: v3

============================================================
PURPOSE
============================================================
Writes a synthetic single-device "seed" CSV pair (m_full and B_demag,
frame 0 only) for demo_superposition.py, so that demo can run standalone
without a real mumax3 simulation on hand. Contains:
    - generate_demo_seed_data: the public entry point -- builds an
      elliptical device's m_full pattern, and its real dipole-summed
      B_demag stray field via magnetostatics.Magnetostatics, and writes
      both to mumax3-style CSVs.
    - _supersampled_ellipse_mask: anti-aliased boolean ellipse generation,
      used for the device footprint.
    - _write_mumax3_style_csv: writes the stacked, blank-line-separated CSV
      format mumax3-convert itself produces.

============================================================
LIMITATIONS
============================================================
- This is still not physically self-consistent test data: there is no
  self-demagnetization, no neighbor interactions during m_full generation,
  a single isolated device, and a small domain. It is realistic enough to
  look like real mumax3 output and exercise the whole pipeline honestly
  (correct CSV block format, a genuine dipole-shaped stray field), but is
  not a substitute for an actual simulation.
- The B_demag field is a point-dipole-sum approximation (see
  magnetostatics.Magnetostatics), which diverges near its own source
  points -- expect an artificially spiky/high-magnitude region directly
  over the device's own footprint in the generated B_demag data, most
  visible in a magnitude or gradient plot of that region. The far-field
  shape away from the device is realistic; the near-field over the device
  itself is not (real mumax3 B_demag data, computed via a proper
  convolution/FFT-based demag kernel, does not have this artifact).
"""

import os  # path construction
import numpy as np  # array operations

from magnetostatics import Magnetostatics  # real analytic dipole-summed stray field computation -- see this file's module-level LIMITATIONS regarding its near-field behavior


def _write_mumax3_style_csv(path, x, y, z, verbose=False):
    """
    GOAL:
        Write x, y, z (2D arrays, same shape) in mumax3-convert's real
        stacked CSV format: `height` rows of x, a blank line, `height`
        rows of y, a blank line, `height` rows of z.

    HOW IT TALKS TO OTHER FILES:
        Called by generate_demo_seed_data (this file), twice per call (once
        for m_full, once for B_demag). The blank-line block separator this
        writes is exactly what mumax3_csv_reader.remove_empty_lists strips
        back out on the reading side, so this format round-trips through
        this codebase's real CSV-reading path, not a simplified stand-in
        for it.

    PARAMETERS:
        path (str): destination CSV path.
        x, y, z (np.ndarray): 2D arrays, all the same shape.
        verbose (bool): prints the saved path when True.

    RETURNS:
        None. Writes the CSV file to `path`.
    """
    with open(path, "w") as f:
        for block in (x, y, z):  # write each component as its own stacked block
            for row in block:
                f.write(",".join(f"{v:.8f}" for v in row) + "\n")
            f.write("\n")  # mumax3-style block separator
    if verbose:
        print(f"[DEBUG] _write_mumax3_style_csv: wrote '{path}'.")


def _supersampled_ellipse_mask(H, W, cx, cy, a, b, supersample=8, verbose=False):
    """
    GOAL:
        Generate an anti-aliased boolean ellipse mask: each cell is
        included only if at least half its area (sampled on a
        supersample x supersample sub-grid) falls inside the ellipse,
        instead of testing a single point (the cell's own index).

    HOW IT TALKS TO OTHER FILES:
        Called by generate_demo_seed_data (this file) to build the seed
        device's footprint.

    PARAMETERS:
        H, W (int): mask array dimensions.
        cx, cy (float): ellipse center, in cells.
        a, b (float): ellipse semi-axes, in cells.
        supersample (int): sub-grid resolution per cell for the coverage
            test -- higher is more accurate and slower.
        verbose (bool): prints the resulting mask's cell count when True.

    RETURNS:
        np.ndarray: boolean array of shape (H, W).

    LIMITATIONS:
        This does not eliminate the staircase pattern along the boundary
        itself -- that is an inherent limit of the cell size: a 1nm
        boolean grid cannot draw a true smooth curve at a device scale of
        tens of nanometers, only a more or less faithful staircase
        approximation of one. What this fixes, relative to a plain
        single-point-per-cell test, is the staircase being unnecessarily
        irregular on top of that unavoidable limit, not the staircase
        itself.
    """
    coverage = np.zeros((H, W))  # fraction of each cell's area inside the ellipse, accumulated below
    offsets = (np.arange(supersample) + 0.5) / supersample  # sub-cell sample offsets, centered within each sub-cell
    yy_base, xx_base = np.mgrid[0:H, 0:W]
    for oy in offsets:  # walk every sub-cell row offset
        for ox in offsets:  # walk every sub-cell column offset
            inside = ((xx_base + ox - cx) / a) ** 2 + ((yy_base + oy - cy) / b) ** 2 <= 1.0  # this sub-sample point's ellipse membership test
            coverage += inside
    coverage /= (supersample * supersample)  # normalize to a 0-1 coverage fraction
    result = coverage >= 0.5  # a cell counts as "inside" if at least half its area is
    if verbose:
        print(f"[DEBUG] _supersampled_ellipse_mask: generated {result.sum()} True cell(s) out of {result.size}.")
    return result


def generate_demo_seed_data(
    data_directory,  # where to write m_full{idx:06d}.csv / B_demag{idx:06d}.csv
    sim_size_cells,  # (W, H) -- full seed-domain grid size, in cells
    device_size_cells,  # (device_w, device_h) -- the ELLIPTICAL device's (length, width) in cells, centered in the domain
    cell_size_x=1.0,  # nm per cell, x direction -- needed to convert into the physical (meter) units Magnetostatics works in
    cell_size_y=1.0,  # nm per cell, y direction
    device_angle_deg=0.0,  # the seed device's magnetization angle (0 = +x), mumax3 convention -- matches Magnetostatics' own convention
    Ms=800e3,  # A/m -- passed straight through to Magnetostatics
    thickness_nm=2.5,  # nm -- passed straight through to Magnetostatics
    magnetostatics_resolution=8,  # subdipole grid resolution per axis for the B_demag calculation -- higher is more accurate and slower
    frame_index=0,  # which frame index to write (a demo only needs one -- SuperpositionConfig.seed_frame_index picks it back out)
    verbose=False,  # print status messages when True
):
    """
    GOAL:
        Build one synthetic seed device (an elliptical footprint with a
        uniform magnetization angle) and write its m_full pattern and its
        real dipole-summed B_demag stray field to a matching pair of
        mumax3-style CSVs.

    HOW IT TALKS TO OTHER FILES:
        Called by demo_superposition.py to produce its input data before
        calling mumax3_parent_superposition.run_pipeline. Calls
        _supersampled_ellipse_mask and _write_mumax3_style_csv (this
        file), and magnetostatics.Magnetostatics.compute_field_at_points
        for the B_demag calculation.

    PARAMETERS:
        See the inline comments on each parameter above.

    RETURNS:
        Dict[str, str]: {"m_full": path, "B_demag": path} to the two
        written CSVs.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section.
    """
    os.makedirs(data_directory, exist_ok=True)
    W, H = sim_size_cells
    dw, dh = device_size_cells  # device length (x), width (y), in cells

    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0

    # ---- m_full: an elliptical device, unit magnetization along
    # device_angle_deg inside the ellipse, zero outside. Supersampled/
    # anti-aliased (see _supersampled_ellipse_mask) rather than a single-
    # point-per-cell test -- smoother, more consistent boundary at this
    # cell size (still a boolean mask in the end -- see that function's own
    # documentation for what this does and does not fix). ----
    a, b = dw / 2.0, dh / 2.0  # semi-axes, in cells
    ellipse = _supersampled_ellipse_mask(H, W, cx, cy, a, b, supersample=8, verbose=verbose)
    theta = np.radians(device_angle_deg)  # magnetization angle, in radians
    m_x = np.where(ellipse, np.cos(theta), 0.0)  # unit magnetization, x component
    m_y = np.where(ellipse, np.sin(theta), 0.0)  # unit magnetization, y component
    m_z = np.zeros((H, W))
    m_full_path = os.path.join(data_directory, f"m_full{frame_index:06d}.csv")
    _write_mumax3_style_csv(m_full_path, m_x, m_y, m_z, verbose=verbose)

    # ---- B_demag: the real analytic dipole-summed stray field from that
    # same elliptical device, via Magnetostatics -- see this file's
    # module-level LIMITATIONS section regarding its near-field behavior. ----
    device_length_m = dw * cell_size_x * 1e-9  # cells -> nm -> m
    device_width_m = dh * cell_size_y * 1e-9
    thickness_m = thickness_nm * 1e-9
    center_nm = np.array([[cx * cell_size_x, cy * cell_size_y]])  # device center, nm

    ms = Magnetostatics(
        centers=center_nm,
        angles_deg=[device_angle_deg],
        Ms=Ms,
        device_length=device_length_m,
        device_width=device_width_m,
        thickness=thickness_m,
        resolution=magnetostatics_resolution,
    )

    x_nm = (xx + 0.5) * cell_size_x  # cell-center physical positions, nm, x
    y_nm = (yy + 0.5) * cell_size_y  # cell-center physical positions, nm, y
    probe_points_m = np.stack([x_nm.ravel(), y_nm.ravel()], axis=-1) * 1e-9  # every cell-center position, m, flattened for compute_field_at_points
    if verbose:
        print(f"[DEBUG] generate_demo_seed_data: computing B_demag at {probe_points_m.shape[0]} probe point(s)...")
    b_field = ms.compute_field_at_points(probe_points_m)  # (N, 3), tesla

    b_x = b_field[:, 0].reshape(H, W)  # B_demag, x component
    b_y = b_field[:, 1].reshape(H, W)  # B_demag, y component
    b_z = b_field[:, 2].reshape(H, W)  # B_demag, z component
    b_demag_path = os.path.join(data_directory, f"B_demag{frame_index:06d}.csv")
    _write_mumax3_style_csv(b_demag_path, b_x, b_y, b_z, verbose=verbose)

    if verbose:
        print(f"[INFO] generate_demo_seed_data: wrote seed data to '{data_directory}'.")
    return {
        "m_full": m_full_path,
        "B_demag": b_demag_path,
    }
