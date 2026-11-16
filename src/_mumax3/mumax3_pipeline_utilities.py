r"""
File Name: mumax3_pipeline_utilities.py
Author: Connor Sullivan, Zoey Yong
Date: 2026.08.05
Revision: v3

============================================================
PURPOSE
============================================================
Provides general-purpose utility functions used across this pipeline,
falling into two groups:

Directory/file utilities:
    - handle_remove_readonly: error handler for removing read-only files.
    - move_and_reset_directory: safely reset a directory that may contain
      locked or read-only files.
    - get_npz_end_index: find the highest numeric frame index among .npz
      files in a directory.

Scalar-field and color-mapping utilities, used by the plotting layer
(mumax3_csv_plot.py, mumax3_plot_calls.py) to turn raw region data into the
2D arrays or RGBA images those functions render:
    - matrix_angle, matrix_magnitude, matrix_component, matrix_uv: derive a
      scalar or vector field from region data for one timestep.
    - full_field_magnitude: derive a magnitude field from a full-domain
      (not region-cropped) data dict.
    - compute_global_scalar_limits: find the min/max of a scalar field
      across every region and timestep, for consistent color scaling.
    - matrix_hue_hsv_rgba, matrix_gradient_hsv_rgba, deviation_to_rgb,
      build_deviation_cmap: construct HSV/RGBA color-mapped images and
      colormaps for the hue and gradient plot types.
    - edge_coord: compute a point on an ellipse's boundary at a given angle.

============================================================
LIMITATIONS
============================================================
- handle_remove_readonly, move_and_reset_directory, and get_npz_end_index
  are also independently defined in
  src/_superposition/utilities_superposition.py. The two implementations
  are not shared or imported from one another; keep this in mind if fixing
  a bug in one copy, since the other will not receive the fix
  automatically.
- full_field_magnitude, edge_coord, and deviation_to_rgb are not currently
  called by any other file in this pipeline. They are retained for
  reference (deviation_to_rgb in particular is the color-mapping logic
  build_deviation_cmap samples from, so the two are best read together even
  though only build_deviation_cmap is called externally).
- move_and_reset_directory moves a directory to a temporary location before
  deleting it, rather than deleting in place. This is deliberate (it avoids
  partial-deletion states if a delete is interrupted) but is slow when the
  temporary location is on a different drive than the original directory,
  and has been observed to behave unreliably with cloud-synced folders
  (for example, a Google Drive folder that is not fully synced locally),
  where files may appear present remotely but be inaccessible locally.
"""

from typing import Callable, Dict, Tuple, Any, Optional  # type hints used throughout this file's signatures
import numpy as np  # array operations used throughout this file


# --------------------------
# ---- Remove read-only files handler ----
def handle_remove_readonly(
        func: Callable[[str], None],  # the function that raised the exception (for example, os.remove or os.rmdir)
        path: str,  # path to the file or directory that could not be removed
        exc: Tuple[Any, Any, Any],  # exception info, as returned by sys.exc_info()
        verbose: bool = False,  # print a debug message when a retry is attempted
) -> None:
    """
    GOAL:
        Error handler for removing read-only files on Windows, intended to
        be passed to shutil.rmtree() as its `onerror` handler. Changes a
        file's permission to writable and retries the removal if the
        original failure was a permission error; re-raises any other kind
        of exception.

    HOW IT TALKS TO OTHER FILES:
        Called by move_and_reset_directory (this file) as the `onerror`
        handler for shutil.rmtree.

    PARAMETERS:
        func (Callable): the function that raised the exception.
        path (str): the path to the file or directory that failed to be
            removed.
        exc (tuple): the exception info returned by sys.exc_info().
        verbose (bool): prints a debug message describing the retry when
            True.

    RETURNS:
        None. Either retries the removal successfully, or re-raises the
        original exception.

    LIMITATIONS:
        Only handles permission-denied errors (errno.EACCES) raised by
        os.rmdir, os.remove, or os.unlink; any other exception is
        re-raised unchanged.
    """
    import os  # file operations
    import stat  # file permission constants
    import errno  # error number constants
    exc_type, exc_value, _ = exc  # unpack the exception info tuple
    if func in (os.rmdir, os.remove, os.unlink) and isinstance(exc_value, OSError) and exc_value.errno == errno.EACCES:  # only handle permission-denied errors from these specific functions
        if verbose:
            print(f"[DEBUG] handle_remove_readonly: permission denied on '{path}'; changing permissions and retrying.")
        os.chmod(path, stat.S_IWRITE)  # change the file's permission to writable
        func(path)  # retry the original operation
    else:
        raise exc_value  # any other kind of error is not handled here; re-raise it


# --------------------------
# ---- Move and reset directory ----
def move_and_reset_directory(original_dir: str, temp_base_dir: Optional[str] = None, verbose: bool = False) -> None:
    """
    GOAL:
        Safely reset a directory that may contain locked or read-only
        files, by moving its contents to a temporary location, deleting
        the moved copy, and recreating an empty directory at the original
        path.

    HOW IT TALKS TO OTHER FILES:
        Called throughout this pipeline anywhere an output directory needs
        to be guaranteed empty before writing fresh results into it (for
        example, mumax3_parent_utilities.py before regenerating npz
        caches, mumax3_csv_stats.py before writing a fresh set of per-frame
        stats CSVs). Calls handle_remove_readonly (this file) as the error
        handler for the underlying deletion.

    PARAMETERS:
        original_dir (str): path to the directory to reset. If it does not
            already exist, it is simply created and this function returns.
        temp_base_dir (str, optional): base directory to place the
            temporary folder in. Defaults to the operating system's
            temporary directory.
        verbose (bool): prints status messages describing each step
            (moving, deleting, recreating) when True.

    RETURNS:
        None.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding
        cross-drive and cloud-sync performance/reliability. May require
        administrator permissions depending on system and antivirus
        settings, and has primarily been exercised on Windows; behavior on
        other operating systems has not been verified.
    """
    import os  # file path operations
    import shutil  # moving and deleting directories
    import tempfile  # locating the system temporary directory
    import uuid  # generating a unique temporary directory name
    import time  # brief delay between retries
    original_dir = os.path.abspath(original_dir)  # normalize to an absolute path

    if not os.path.exists(original_dir):  # nothing to reset if the directory does not exist yet
        if verbose:
            print(f"[INFO] move_and_reset_directory: original directory does not exist: '{original_dir}'.")
        os.makedirs(original_dir)  # create it fresh
        if verbose:
            print(f"[INFO] move_and_reset_directory: created directory: '{original_dir}'.")
        return  # nothing further to do

    if temp_base_dir is None:  # no explicit temporary location given
        temp_base_dir = tempfile.gettempdir()  # fall back to the system temporary directory
    temp_target_dir = os.path.join(temp_base_dir, f"temp_delete_folder_{uuid.uuid4().hex}")  # unique temporary directory name, to avoid colliding with any other process
    if verbose:
        print(f"[INFO] move_and_reset_directory: moving '{original_dir}' to temporary location '{temp_target_dir}'...")
    for attempt in range(3):  # retry the move up to three times in case of a transient permission error
        try:
            shutil.move(original_dir, temp_target_dir)  # attempt the move
            break  # succeeded; stop retrying
        except PermissionError as e:
            if attempt < 2:  # not the last attempt yet
                print(f"[WARNING] move_and_reset_directory: permission denied (attempt {attempt + 1}), retrying in 1s...")
                time.sleep(1)  # brief delay before retrying
            else:  # final attempt also failed
                print(f"[ERROR] move_and_reset_directory: failed to move directory after retries: {e}")
                raise  # re-raise so the caller is aware the reset failed

    if verbose:
        print(f"[INFO] move_and_reset_directory: deleting moved directory at '{temp_target_dir}'...")
    shutil.rmtree(temp_target_dir, onerror=lambda func, path, exc: handle_remove_readonly(func, path, exc, verbose=verbose))  # delete the moved copy, retrying read-only files via handle_remove_readonly
    if verbose:
        print(f"[INFO] move_and_reset_directory: deleted directory: '{temp_target_dir}'.")
    os.makedirs(original_dir)  # recreate an empty directory at the original path
    if verbose:
        print(f"[INFO] move_and_reset_directory: recreated empty directory at original location: '{original_dir}'.")


# --------------------------
# ---- Get highest npz index ----
def get_npz_end_index(npz_dir: str, prefix: str = "B_demag", verbose: bool = False) -> int:
    """
    GOAL:
        Find the highest numeric frame index among .npz files matching a
        given prefix in a directory. Expected filename format:
        "{prefix}{index}.npz" (for example, "B_demag000042.npz" has index
        42).

    HOW IT TALKS TO OTHER FILES:
        Called by frame_pipeline.process_frames_parallel to determine how
        many frames exist before processing them.

    PARAMETERS:
        npz_dir (str): path to the directory containing .npz files.
        prefix (str): filename prefix to search for.
        verbose (bool): prints the number of matching files found when
            True.

    RETURNS:
        int: the highest index found among matching files.

    LIMITATIONS:
        Raises FileNotFoundError if no matching files exist, rather than
        returning a sentinel value such as -1 or None; callers should
        expect and handle that exception if an empty directory is a valid
        state for their use case.
    """
    import os  # file path operations
    import re  # pattern matching against filenames
    if not os.path.isdir(npz_dir):  # the directory itself must exist
        raise ValueError(f"Directory does not exist: {npz_dir}")
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.npz$")  # strict match: prefix followed by digits, then .npz
    indices = []  # every matching index found, collected below
    for fname in os.listdir(npz_dir):  # scan every file in the directory
        match = pattern.match(fname)  # check whether this filename matches the expected pattern
        if match:
            indices.append(int(match.group(1)))  # record the index extracted from this filename
    if not indices:  # no matching files were found at all
        raise FileNotFoundError(f"No files matching pattern '{prefix}*.npz' found in '{npz_dir}'.")
    if verbose:
        print(f"[DEBUG] get_npz_end_index: found {len(indices)} matching file(s); highest index is {max(indices)}.")
    return max(indices)  # the highest index found


# --------------------------
# ---- Compute the in-plane field angle for a single frame ----
def matrix_angle(region_data: Dict[str, np.ndarray], t: int, field_scaling: float = 10000.0, verbose: bool = False) -> np.ndarray:
    """
    GOAL:
        Compute the angle (in degrees, [0, 360)) of the in-plane field at
        timestep t, from a region's masked x/y components.

    HOW IT TALKS TO OTHER FILES:
        Used as a scalar_func by mumax3_plot_calls.py's plot configs, and
        called directly by matrix_hue_hsv_rgba/matrix_gradient_hsv_rgba
        (this file).

    PARAMETERS:
        region_data (dict): region data containing 'masked_x' and
            'masked_y', each shaped (frames, height, width).
        t (int): timestep index.
        field_scaling (float): multiplier applied to both components before
            computing the angle (for example, Tesla to Oersted). Does not
            change the resulting angle unless the scaling differs between
            components, but is applied for consistency with the other
            matrix_* functions in this file. Defaults to 10000.0 if None is
            passed.
        verbose (bool): prints the resulting angle array's min/max when
            True.

    RETURNS:
        np.ndarray: 2D array of angles in degrees, in the range [0, 360).
    """
    if field_scaling is None:  # guard against an explicit None being passed through from a caller
        field_scaling = 10000.0  # default scaling, Tesla to Oersted
    bx = region_data['masked_x'][t] * field_scaling  # x component at this timestep, scaled
    by = region_data['masked_y'][t] * field_scaling  # y component at this timestep, scaled

    angle_matrix = np.arctan2(by, bx) * (180.0 / np.pi)  # angle in degrees, range (-180, 180]
    angle_matrix[angle_matrix < 0] += 360.0  # shift negative angles into [0, 360)
    if verbose:
        print(f"[DEBUG] matrix_angle: t={t}, angle range [{np.nanmin(angle_matrix):.1f}, {np.nanmax(angle_matrix):.1f}] degrees.")
    return angle_matrix


# --------------------------
# ---- Compute the in-plane field magnitude for a single frame ----
def matrix_magnitude(region_data: Dict[str, np.ndarray], t: int, field_scaling: float = 10000.0, verbose: bool = False) -> np.ndarray:
    """
    GOAL:
        Compute the magnitude of the in-plane field at timestep t, from a
        region's masked x/y components.

    HOW IT TALKS TO OTHER FILES:
        Used as a scalar_func by mumax3_plot_calls.py's plot configs, and
        called directly by matrix_hue_hsv_rgba/matrix_gradient_hsv_rgba
        (this file).

    PARAMETERS:
        region_data (dict): region data containing 'masked_x' and
            'masked_y', each shaped (frames, height, width).
        t (int): timestep index.
        field_scaling (float): multiplier applied to both components before
            computing the magnitude (for example, Tesla to Oersted).
            Defaults to 10000.0 if None is passed.
        verbose (bool): prints the resulting magnitude array's min/max when
            True.

    RETURNS:
        np.ndarray: 2D array of field magnitudes.
    """
    if field_scaling is None:  # guard against an explicit None being passed through from a caller
        field_scaling = 10000.0  # default scaling, Tesla to Oersted
    bx = region_data['masked_x'][t] * field_scaling  # x component at this timestep, scaled
    by = region_data['masked_y'][t] * field_scaling  # y component at this timestep, scaled
    result = np.sqrt(bx**2 + by**2)  # magnitude of the in-plane vector
    if verbose:
        print(f"[DEBUG] matrix_magnitude: t={t}, magnitude range [{np.nanmin(result):.4g}, {np.nanmax(result):.4g}].")
    return result


# --------------------------
# ---- Compute magnitude from a full (non-region-cropped) data dict ----
def full_field_magnitude(full_data: dict, field_scaling: float = 1.0, verbose: bool = False) -> np.ndarray:
    """
    GOAL:
        Compute the in-plane field magnitude for a single timestep from a
        full-domain data dict (as opposed to matrix_magnitude, which reads
        from a region's masked_x/masked_y).

    HOW IT TALKS TO OTHER FILES:
        Not currently called by any other file in this pipeline; see this
        file's module-level LIMITATIONS section. Retained for reference.

    PARAMETERS:
        full_data (dict): dict with keys 'Bx' and 'By', each a 2D array.
        field_scaling (float): multiplier applied to both components before
            computing the magnitude (for example, Tesla to Oersted).
        verbose (bool): prints the resulting magnitude array's min/max when
            True.

    RETURNS:
        np.ndarray: 2D array (height, width) of field magnitudes.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section: this function is
        not part of the active pipeline.
    """
    bx = full_data['Bx'] * field_scaling  # x component, scaled
    by = full_data['By'] * field_scaling  # y component, scaled
    result = np.sqrt(bx**2 + by**2)  # magnitude of the in-plane vector
    if verbose:
        print(f"[DEBUG] full_field_magnitude: magnitude range [{np.nanmin(result):.4g}, {np.nanmax(result):.4g}].")
    return result


# --------------------------
# ---- Compute global scalar limits across all regions and timesteps ----
def compute_global_scalar_limits(
    regions: Dict[str, Dict[str, np.ndarray]],  # region data, for example from reorganize_regions
    scalar_func: Callable[[Dict[str, np.ndarray], int], np.ndarray],  # function that takes (region_data, t) and returns a 2D array
    verbose: bool = False,  # print per-timestep progress when True
) -> Tuple[float, float]:
    """
    GOAL:
        Compute the global minimum and maximum of a scalar field (as
        produced by `scalar_func`) across every region and every timestep,
        so plots can share one consistent color scale instead of each
        frame/region choosing its own.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_parent_spatial_analysis.py before plotting, once
        per scalar field being plotted (magnitude, angle, each component),
        to establish that field's global color scale. `scalar_func` is
        typically one of this file's own matrix_* functions.

    PARAMETERS:
        regions (dict): region data, keyed by region label, for example as
            returned by region_extraction.reorganize_regions.
        scalar_func (Callable): function taking (region_data, t) and
            returning a 2D array; called once per region, per timestep.
        verbose (bool): prints the timestep/region count at the start, and
            a running note of progress, when True.

    RETURNS:
        Tuple[float, float]: (global_min, global_max) across every region
        and timestep.

    LIMITATIONS:
        Every region must have the same number of timesteps; the timestep
        count is read from the first region encountered and assumed to
        apply to all of them.
    """
    scalar_values = []  # flattened scalar arrays from every region/timestep, concatenated at the end

    sample_region = next(iter(regions.values()))  # an arbitrary region, used only to determine the timestep count
    num_timesteps = len(sample_region.get('x', []))  # number of timesteps, assumed the same for every region
    if verbose:
        print(f"[INFO] compute_global_scalar_limits: computing over {num_timesteps} timestep(s) and {len(regions)} region(s)...")

    for t in range(num_timesteps):  # walk every timestep
        for region_data in regions.values():  # walk every region at this timestep
            scalar_matrix = scalar_func(region_data, t)  # compute this region's scalar field at this timestep
            if scalar_matrix is not None:
                scalar_values.append(scalar_matrix.flatten())  # collect this region/timestep's values for the global min/max below

    if not scalar_values:  # no data was collected at all
        raise ValueError("No scalar data found to compute limits.")

    all_values = np.concatenate(scalar_values)  # every collected value, from every region and timestep, in one array
    result = (np.nanmin(all_values), np.nanmax(all_values))  # global min/max, ignoring NaN (masked-out) cells
    if verbose:
        print(f"[INFO] compute_global_scalar_limits: global range [{result[0]:.4g}, {result[1]:.4g}].")
    return result


# --------------------------
# ---- Extract a scaled field component for a single frame ----
def matrix_component(
    region_data: Dict[str, np.ndarray],  # region data containing the requested component, shaped (frames, height, width)
    t: int,  # timestep index
    field_scaling: float = 10000.0,  # multiplier applied before returning (for example, Tesla to Oersted)
    comp: str = 'masked_x',  # which component to extract, e.g. 'masked_x', 'masked_y', or 'masked_z'
    verbose: bool = False,  # print the resulting array's min/max when True
) -> np.ndarray:
    """
    GOAL:
        Extract and scale a single field component from region data, for
        one timestep.

    HOW IT TALKS TO OTHER FILES:
        Used as a scalar_func (via functools.partial, binding `comp`) by
        mumax3_plot_calls.py's per-component plot configs, and called
        directly by matrix_uv (this file).

    PARAMETERS:
        region_data (dict): region data with keys such as 'masked_x',
            'masked_y', or 'masked_z'.
        t (int): timestep index to extract.
        field_scaling (float): multiplier applied before returning (for
            example, Tesla to Oersted). Defaults to 10000.0 if None is
            passed.
        comp (str): which key of `region_data` to extract.
        verbose (bool): prints the resulting array's min/max when True.

    RETURNS:
        np.ndarray: 2D array of the requested component's values at
        timestep t, scaled by `field_scaling`.

    LIMITATIONS:
        Raises KeyError if `comp` is not present in `region_data`, rather
        than returning None or a zero-filled array.
    """
    if field_scaling is None:  # guard against an explicit None being passed through from a caller
        field_scaling = 10000.0  # default scaling, Tesla to Oersted
    if comp not in region_data:  # the requested component must actually exist in this region's data
        raise KeyError(f"[ERROR] Component '{comp}' not found in region_data. "
                       f"Available keys: {list(region_data.keys())}")
    result = region_data[comp][t] * field_scaling  # this component at this timestep, scaled
    if verbose:
        print(f"[DEBUG] matrix_component: comp='{comp}', t={t}, range [{np.nanmin(result):.4g}, {np.nanmax(result):.4g}].")
    return result


# --------------------------
# ---- Extract both in-plane components together, for quiver plotting ----
def matrix_uv(region_data, t, label=None, field_scaling=10000.0, vmin=None, vmax=None,
              use_local_scaling=False, use_custom_scaling=None, comp_x='masked_x', comp_y='masked_y',
              verbose=False):
    """
    GOAL:
        Return the (u, v) vector-component pair for one timestep, for the
        "quiver" plot type in mumax3_csv_plot.py -- quiver rendering needs
        both in-plane components together, not a single derived scalar.

    HOW IT TALKS TO OTHER FILES:
        Used as a scalar_func by mumax3_plot_calls.py's quiver plot config.
        Calls matrix_component (this file) twice, once per component.

    PARAMETERS:
        region_data (dict): region data containing `comp_x`/`comp_y`.
        t (int): timestep index.
        label, vmin, vmax, use_local_scaling, use_custom_scaling: accepted
            but not used by this function -- present only so this function
            can be called with the same signature used for every other
            scalar_func in mumax3_csv_plot.py's plotting call sites, even
            though a quiver plot has no single color scale to apply them
            to.
        field_scaling (float): multiplier applied to both components (for
            example, Tesla to Oersted).
        comp_x, comp_y (str): which two components of `region_data` to
            extract as u and v.
        verbose (bool): forwarded to matrix_component, so per-component
            debug messages appear when True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray]: (u, v), two 2D arrays, NaN where
        masked out, in the same physical units `field_scaling` converts
        everything else to.
    """
    u = matrix_component(region_data, t, comp=comp_x, field_scaling=field_scaling, verbose=verbose)  # first in-plane component
    v = matrix_component(region_data, t, comp=comp_y, field_scaling=field_scaling, verbose=verbose)  # second in-plane component
    return u, v


# --------------------------
# ---- Compute a point on an ellipse's boundary at a given angle (currently unused) ----
def edge_coord(r_a, r_b, theta, center_x, center_y, verbose=False):
    """
    GOAL:
        Compute the (x, y) coordinate of the point on an ellipse's
        boundary at angle `theta`, given the ellipse's semi-axes and
        center.

    HOW IT TALKS TO OTHER FILES:
        Not currently called by any other file in this pipeline; see this
        file's module-level LIMITATIONS section. Retained for reference.

    PARAMETERS:
        r_a (float): semi-axis in the x direction.
        r_b (float): semi-axis in the y direction.
        theta (float): angle, in radians.
        center_x, center_y (float): the ellipse's center coordinates.
        verbose (bool): prints the resulting coordinate when True.

    RETURNS:
        Tuple[float, float]: (edge_x, edge_y), the boundary point at angle
        `theta`.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section: this function is
        not part of the active pipeline.
    """
    theta_rad = theta  # angle is already expected in radians (parameter name kept as `theta` for the caller's convenience)
    edge_x = center_x + r_a * np.cos(theta_rad)  # x coordinate of the boundary point
    edge_y = center_y - r_b * np.sin(theta_rad)  # y coordinate of the boundary point
    if verbose:
        print(f"[DEBUG] edge_coord: theta={theta}, result=({edge_x}, {edge_y}).")
    return edge_x, edge_y


# --------------------------
# ---- Build an HSV/RGBA hue map (angle as hue, magnitude as value) ----
def matrix_hue_hsv_rgba(region_data, t, field_scaling=10000.0, vmin=None, vmax=None, use_local_scaling=False, verbose=False):
    """
    GOAL:
        Build an RGBA image where each cell's hue encodes the in-plane
        field angle and each cell's value (brightness) encodes the field
        magnitude, for the "hue" plot type.

    HOW IT TALKS TO OTHER FILES:
        Used as a scalar_func by mumax3_plot_calls.py's hue plot configs.
        Calls matrix_angle and matrix_magnitude (this file).

    PARAMETERS:
        region_data (dict): region data containing 'masked_x'/'masked_y'.
        t (int): timestep index.
        field_scaling (float): passed through to matrix_angle/
            matrix_magnitude.
        vmin, vmax (float, optional): magnitude range used for the value
            channel when use_local_scaling is False. Required in that case.
        use_local_scaling (bool): if True, uses this frame's own magnitude
            min/max instead of `vmin`/`vmax`.
        verbose (bool): prints the resolved magnitude scaling range when
            True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (angle, magnitude, rgba)
        -- the underlying angle and magnitude arrays (for any caller that
        wants the raw values, for example a companion CSV export), and the
        rendered RGBA image, shape (height, width, 4).

    LIMITATIONS:
        Raises ValueError if use_local_scaling is False and vmin/vmax are
        not both provided.
    """
    import matplotlib.colors as mcolors  # HSV/RGB conversion
    angle = matrix_angle(region_data, t, field_scaling)  # in-plane field angle, degrees
    magnitude = matrix_magnitude(region_data, t, field_scaling)  # in-plane field magnitude
    valid_mask = np.isfinite(angle) & np.isfinite(magnitude)  # cells with real (non-NaN) data

    h = angle / 360.0  # hue channel: angle normalized to [0, 1]

    if use_local_scaling:  # scale this frame's own magnitude range
        mag_min = np.nanmin(magnitude)
        mag_max = np.nanmax(magnitude)
    else:  # use the caller-provided global scaling
        if vmin is None or vmax is None:
            raise ValueError("vmin/vmax must be precomputed and provided for hue plotting")
        mag_min, mag_max = vmin, vmax
    denom = (mag_max - mag_min) if (mag_max is not None and mag_min is not None) else 1.0  # avoid dividing by an unset range
    if denom == 0:  # avoid dividing by zero when the range is degenerate
        denom = 1.0
    v = (magnitude - mag_min) / denom  # value channel: magnitude normalized to [0, 1]
    v = np.clip(v, 0, 1)  # clamp to a valid value range
    if verbose:
        print(f"[DEBUG] matrix_hue_hsv_rgba: t={t}, magnitude scaling range [{mag_min:.4g}, {mag_max:.4g}].")

    s = np.ones_like(h)  # saturation channel: full saturation everywhere

    # hsv_to_rgb casts h * 6.0 to an integer internally; NaN cells (from masked-out
    # pixels in angle/magnitude) reaching this point trigger an "invalid value
    # encountered in cast" warning on every masked frame. The final image is
    # unaffected either way (invalid cells are zeroed out below via valid_mask), so
    # NaN is replaced with 0 here purely to avoid that warning.
    h = np.nan_to_num(h, nan=0.0)  # replace NaN hue with 0 before HSV conversion
    v = np.nan_to_num(v, nan=0.0)  # replace NaN value with 0 before HSV conversion
    hsv = np.stack((h, s, v), axis=-1)  # combine into an (height, width, 3) HSV image
    rgb = mcolors.hsv_to_rgb(hsv)  # convert to RGB
    rgb[~valid_mask] = 0  # zero out invalid (masked-out) cells
    alpha = valid_mask.astype(float)  # alpha channel: fully opaque where valid, fully transparent elsewhere
    rgba = np.dstack((rgb, alpha))  # combine RGB and alpha into the final RGBA image
    return angle, magnitude, rgba


# --------------------------
# ---- Build an HSV/RGBA gradient map (deviation from the region's mean angle) ----
def matrix_gradient_hsv_rgba(region_data, t, label, field_scaling=10000.0, use_local_scaling=False,
                              use_custom_scaling=None, verbose=False):
    """
    GOAL:
        Build an RGBA image where each cell's hue is fixed at the region's
        mean field angle and each cell's saturation/value encode how far
        that cell's own angle deviates from the mean, for the "gradient"
        plot type -- useful for visualizing spin canting within a region.

    HOW IT TALKS TO OTHER FILES:
        Used as a scalar_func by mumax3_plot_calls.py's gradient plot
        configs. Calls mumax3_csv_stats.
        compute_weighted_circular_stats_for_frame_new to obtain the
        region's mean angle for this timestep, and matrix_angle/
        matrix_magnitude (this file).

    PARAMETERS:
        region_data (dict): region data containing 'masked_x'/'masked_y'.
        t (int): timestep index.
        label (str): region label, passed through to the stats computation
            for identification purposes.
        field_scaling (float): passed through to matrix_angle/
            matrix_magnitude/the stats computation.
        use_local_scaling (bool): if True and use_custom_scaling is not
            active, scales the deviation color range to this frame's own
            maximum absolute deviation instead of the fixed +/-180 degree
            range.
        use_custom_scaling: either None/False (no custom scaling), or a
            [enabled, dmin, dmax] list; when enabled, the deviation color
            range is taken from the larger of abs(dmin)/abs(dmax) instead
            of use_local_scaling or the fixed default.
        verbose (bool): prints the resolved deviation scaling range when
            True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray, np.ndarray, float, List[float]]:
        (angle, magnitude, rgba, mean_angle, [dmin, dmax]) -- the
        underlying angle/magnitude arrays, the rendered RGBA image, the
        region's mean angle for this timestep, and the deviation range
        used for scaling (useful for building a matching colorbar via
        build_deviation_cmap).
    """
    import matplotlib.colors as mcolors  # HSV/RGB conversion
    from mumax3_csv_stats import compute_weighted_circular_stats_for_frame_new  # region's mean angle for this timestep

    use_custom_scaling = use_custom_scaling if use_custom_scaling is not None else [False, None, None]  # default: custom scaling disabled

    _, stats = compute_weighted_circular_stats_for_frame_new(  # compute this region's weighted circular stats for this one timestep
        t, region_data, field_scaling=field_scaling
    )
    mean_angle = stats["mean_angle"]  # this region's mean field angle at this timestep

    angle = matrix_angle(region_data, t, field_scaling)  # in-plane field angle, degrees
    magnitude = matrix_magnitude(region_data, t, field_scaling)  # in-plane field magnitude

    valid_mask = np.isfinite(angle) & np.isfinite(magnitude)  # cells with real (non-NaN) data

    delta = angle - mean_angle  # each cell's raw deviation from the region's mean angle
    delta = (delta + 180) % 360 - 180  # wrap into (-180, 180], the shortest angular distance

    delta_valid = delta[valid_mask]  # deviations at valid cells only, used for scaling below
    v = np.zeros_like(delta)  # value channel, computed below

    if use_custom_scaling[0]:  # explicit custom deviation range provided
        dmin, dmax = use_custom_scaling[1], use_custom_scaling[2]  # caller-provided deviation bounds
        dmax_abs = max(abs(dmin), abs(dmax))  # symmetric scaling range from the larger-magnitude bound
    elif use_local_scaling:  # scale to this frame's own maximum deviation
        dmax_abs = np.max(np.abs(delta_valid))
    else:  # fixed default range
        dmax_abs = 180.0
    if verbose:
        print(f"[DEBUG] matrix_gradient_hsv_rgba: t={t}, label='{label}', mean_angle={mean_angle:.1f}, dmax_abs={dmax_abs:.1f}.")
    dev = delta / dmax_abs  # deviation normalized to [-1, 1] (approximately; can exceed slightly if dmax_abs was capped by custom/global scaling)

    # Value channel: -1 -> 0 (black), 0 -> 0.5 (mid), +1 -> 1 (white).
    v = 1 - np.abs(dev)
    v = np.where(dev >= 0, v + np.abs(dev), v)
    v = np.clip(v, 0, 1)
    v[~valid_mask] = 0  # invalid cells are rendered black
    v = v ** 0.7  # contrast boost

    dmin = -dmax_abs  # lower bound of the deviation range used
    dmax = dmax_abs  # upper bound of the deviation range used

    h = np.full_like(v, mean_angle / 360.0)  # hue channel: fixed at the region's mean angle for every cell
    s = 1 - np.abs(dev)  # saturation channel: reduced for larger deviations, to enhance contrast
    s = np.clip(s, 0, 1)  # clamp to a valid saturation range
    s[~valid_mask] = 0  # invalid cells are rendered grayscale (zero saturation)

    hsv = np.stack((h, s, v), axis=-1)  # combine into an (height, width, 3) HSV image
    rgb = mcolors.hsv_to_rgb(hsv)  # convert to RGB

    alpha = valid_mask.astype(float)  # alpha channel: fully opaque where valid, fully transparent elsewhere
    rgba = np.dstack((rgb, alpha))  # combine RGB and alpha into the final RGBA image

    return angle, magnitude, rgba, mean_angle, [dmin, dmax]


# --------------------------
# ---- Convert deviation values directly to RGB, without the alpha channel ----
def deviation_to_rgb(delta, mean_angle, dmax_abs, verbose=False):
    """
    GOAL:
        Convert an array of angular deviation values directly to RGB
        colors, using the same hue-fixed-at-mean-angle, value-from-deviation
        scheme as matrix_gradient_hsv_rgba, but without needing full region
        data or an alpha channel -- used to sample a colormap's colors at
        arbitrary deviation values.

    HOW IT TALKS TO OTHER FILES:
        Called by build_deviation_cmap (this file) to sample colors across
        a deviation range when constructing a matplotlib colormap.

    PARAMETERS:
        delta: array of deviation values (same units as
            matrix_gradient_hsv_rgba's `delta`, i.e. degrees).
        mean_angle (float): the fixed hue angle, in degrees.
        dmax_abs (float): the deviation magnitude that maps to full
            saturation/black-or-white value.
        verbose (bool): prints the number of values converted when True.

    RETURNS:
        np.ndarray: array of RGB colors, shape (*delta.shape, 3).
    """
    import matplotlib.colors as mcolors  # HSV/RGB conversion

    delta = np.asarray(delta)  # ensure an array, in case a list was passed

    dev = delta / dmax_abs  # deviation normalized by the given scale
    v = 1 - np.abs(dev)  # value channel, same scheme as matrix_gradient_hsv_rgba
    v = np.where(dev >= 0, v + np.abs(dev), v)
    v = np.clip(v, 0, 1)

    h = np.full_like(v, mean_angle / 360.0)  # hue channel: fixed at the given mean angle
    s = 1 - np.abs(dev)  # saturation channel: reduced for larger deviations

    hsv = np.stack((h, s, v), axis=-1)  # combine into an HSV array
    rgb = mcolors.hsv_to_rgb(hsv)  # convert to RGB
    if verbose:
        print(f"[DEBUG] deviation_to_rgb: converted {delta.size} value(s).")
    return rgb


# --------------------------
# ---- Build a matplotlib colormap/norm pair matching the gradient plot's coloring ----
def build_deviation_cmap(mean_angle, dmin, dmax, n=256, verbose=False):
    """
    GOAL:
        Build a matplotlib ListedColormap and Normalize pair that matches
        matrix_gradient_hsv_rgba's coloring scheme, for use as a colorbar
        alongside a gradient plot.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_csv_plot.py wherever a gradient plot's colorbar is
        drawn, using the same mean_angle/dmin/dmax that
        matrix_gradient_hsv_rgba produced for that plot. Calls
        deviation_to_rgb (this file) to sample the colormap's colors.

    PARAMETERS:
        mean_angle (float): the fixed hue angle, in degrees, matching the
            plot this colorbar accompanies.
        dmin, dmax (float): the deviation range the colorbar should span.
        n (int): number of discrete color samples in the colormap.
        verbose (bool): prints the resolved deviation range when True.

    RETURNS:
        Tuple[matplotlib.colors.ListedColormap, matplotlib.colors.Normalize]:
        ready to pass to matplotlib's colorbar machinery.
    """
    import matplotlib.colors as mcolors  # colormap/norm construction
    dmax_abs = max(abs(dmin), abs(dmax))  # symmetric scaling range from the larger-magnitude bound

    values = np.linspace(dmin, dmax, n)  # evenly spaced sample points across the deviation range
    colors = deviation_to_rgb(values, mean_angle, dmax_abs, verbose=verbose)  # colors at each sample point

    cmap = mcolors.ListedColormap(colors)  # discrete colormap built from the sampled colors
    norm = mcolors.Normalize(vmin=dmin, vmax=dmax)  # maps data values in [dmin, dmax] onto the colormap
    if verbose:
        print(f"[DEBUG] build_deviation_cmap: built a {n}-color map over deviation range [{dmin:.1f}, {dmax:.1f}].")
    return cmap, norm
