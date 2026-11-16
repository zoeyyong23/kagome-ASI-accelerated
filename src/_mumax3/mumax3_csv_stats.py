r"""
File Name: mumax3_csv_stats.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v2

============================================================
PURPOSE
============================================================
Computes weighted circular statistics (mean field angle, mean field
magnitude, angular concentration/variance) from mumax3 vector field data,
and saves those statistics to CSV. Contains:
    - elliptical_mask: generates a boolean elliptical mask (see this file's
      LIMITATIONS section -- a separate implementation from masking.py's
      function of the same name).
    - compute_weighted_circular_stats_for_frame /
      compute_weighted_circular_stats_for_frame_new: compute weighted
      circular statistics for one timestep, across a dict of regions or for
      a single region respectively.
    - _compute_region_stats_vectorized /
      compute_stats_for_all_timesteps_parallel: the vectorized,
      all-timesteps-at-once implementation actually used by both parent
      scripts.
    - full_field_angle_worker: samples field magnitude along a radial path
      from a device edge (see this file's LIMITATIONS section).
    - save_region_stats_csv: writes computed statistics to CSV, per-frame
      and/or across all frames.
    - masked_region_mean_xy / hrev_from_field / field_angle_deg_from_xy /
      acute_angle_to_easy_axis / signed_angle_difference_deg /
      load_coercivity_table / lookup_nearest_coercivity: propagation
      switching-physics calculations -- see this file's own LIMITATIONS
      section for how these relate to the weighted-circular-statistics
      functions above (a genuinely different quantity, not a duplicate).
      Originally their own file (propagation_stats.py, formerly mumax3_
      stats.py); merged in here since none of their names collided with
      anything already in this file, at Connor's request, so every
      "calculation" (as opposed to bookkeeping/orchestration) function in
      this codebase lives in one place. src/_propagation/propagation_
      utilities.py (formerly utilities_propagation.py) holds the
      propagation engine's remaining, non-calculation bookkeeping helpers.

============================================================
LIMITATIONS
============================================================
- This file's elliptical_mask centers the ellipse on
  ((width - 1) / 2, (height - 1) / 2) -- the array's continuous-domain
  midpoint. masking.py's elliptical_mask (used for actual region masking
  throughout the rest of this pipeline) instead centers on
  (width // 2, height // 2) -- the same cell-index convention used
  everywhere else for a region's "center" cell. These two conventions agree
  for odd-sized arrays and disagree by half a cell for even-sized arrays.
  The optional `ellipse_size` restriction in this file's stats functions
  uses THIS file's convention, not masking.py's -- so an `ellipse_size`
  restriction is not guaranteed to be centered identically to an
  even-sized region's own mask_mode='ellipse' mask. This has not been
  unified; treat the two elliptical_mask functions as independent.
- compute_weighted_circular_stats_for_frame (without "_new") is not
  currently called by any other file in this pipeline.
  compute_stats_for_all_timesteps_parallel, the function both parent
  scripts actually use, is built on the separate, vectorized
  _compute_region_stats_vectorized instead. compute_weighted_circular_stats_
  for_frame is retained for reference and because
  compute_weighted_circular_stats_for_frame_new (which IS actively called,
  by mumax3_pipeline_utilities.matrix_gradient_hsv_rgba) mirrors its logic
  for a single region.
- full_field_angle_worker is not currently called by any other file in this
  pipeline, and has no automated verification alongside it in this pass --
  treat it as unverified reference code, not a tested part of the active
  pipeline.
- This module processes data entirely in grid coordinates (cell indices),
  not physical coordinates (nm), consistent with mumax3_csv_reader.py and
  region_extraction.py -- EXCEPT masked_region_mean_xy/hrev_from_field and
  the other propagation switching-physics functions below, which operate
  on plain floats/physical field values (Oe) and per-region masked arrays,
  not grid-coordinate geometry -- they were written for the propagation
  engine specifically and do not share this file's other functions' grid-
  coordinate framing.
- masked_region_mean_xy computes the plain Cartesian (component-wise)
  nanmean of a region's masked_x/masked_y -- NOT a circular mean. This is a
  DIFFERENT quantity from what compute_weighted_circular_stats_for_frame/
  _new above compute (those describe how UNIFORM the field direction is
  across a region; masked_region_mean_xy describes the region's own net
  vector). The propagation switching decision (Hrev = dot(<H>, m_hat) in
  hrev_from_field, below) needs the plain vector average specifically, so
  this distinction is intentional, not an oversight -- both are legitimate,
  different quantities to report side by side for the same region.
"""

from pathlib import Path  # coercivity CSV path handling (load_coercivity_table)
from typing import Dict, List, Optional, Tuple  # type hints used throughout this file's signatures
import csv  # coercivity CSV parsing (load_coercivity_table)
import numpy as np  # array operations used throughout this file


# --------------------------
# ---- Elliptical mask generation (see this file's module-level LIMITATIONS) ----
def elliptical_mask(region_shape: Tuple[int, int], size: Tuple[int, int], verbose: bool = False) -> np.ndarray:
    """
    GOAL:
        Generate a boolean mask whose True cells lie inside an
        axis-aligned ellipse, centered within the region.

    HOW IT TALKS TO OTHER FILES:
        Called by compute_weighted_circular_stats_for_frame,
        compute_weighted_circular_stats_for_frame_new, and
        _compute_region_stats_vectorized (all this file) when an optional
        `ellipse_size` restriction is requested. Not called from outside
        this file.

    PARAMETERS:
        region_shape (Tuple[int, int]): shape of the region (height,
            width), in grid cells.
        size (Tuple[int, int]): ellipse semi-axes (a, b), where a is the
            x-radius (columns) and b is the y-radius (rows), in grid cells.
        verbose (bool): prints the resulting mask's cell count when True.

    RETURNS:
        np.ndarray: boolean array of shape region_shape; True where a cell
        lies inside the ellipse.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section: this function
        centers the ellipse differently than masking.py's elliptical_mask
        (this one uses the continuous-domain midpoint
        ((width - 1) / 2, (height - 1) / 2); masking.py uses the cell-index
        convention width // 2, height // 2). The two disagree by half a
        cell for even-sized regions.
    """
    height, width = region_shape  # unpack the region's cell dimensions
    a, b = size  # unpack the ellipse's semi-axes
    x_c = (width - 1) / 2  # ellipse center column -- continuous-domain midpoint (see LIMITATIONS)
    y_c = (height - 1) / 2  # ellipse center row -- continuous-domain midpoint (see LIMITATIONS)
    y_indices, x_indices = np.ogrid[0:height, 0:width]  # open grid of row/column indices, broadcastable against the ellipse formula below
    norm_x = (x_indices - x_c) / a  # each column's distance from center, normalized by the x semi-axis
    norm_y = (y_indices - y_c) / b  # each row's distance from center, normalized by the y semi-axis
    mask = norm_x**2 + norm_y**2 <= 1  # standard ellipse membership test
    if verbose:
        print(f"[DEBUG] elliptical_mask: generated {mask.sum()} True cell(s) out of {mask.size} for shape {region_shape}.")
    return mask


# --------------------------
# ---- Compute weighted circular statistics for a single frame, across every region ----
def compute_weighted_circular_stats_for_frame(
    t: int,  # timestep index
    regions: Dict[str, Dict[str, np.ndarray]],  # region data, keyed by region label
    field_scaling: float = 10000.0,  # multiplier applied to both components (for example, Tesla to Oersted)
    ellipse_size: Tuple[int, int] = None,  # optional EXTRA elliptical restriction, layered on top of each region's own mask
    verbose: bool = False,  # print a per-region summary when True
) -> Dict[str, Dict[str, float]]:
    """
    GOAL:
        Compute weighted circular statistics (mean field angle, mean field
        magnitude, angular concentration/variance) at one timestep, for
        every region in `regions`.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline. Reads each region's
        `masked_x`/`masked_y` arrays -- the same arrays masking.apply_mask
        produces -- so statistics are computed over the same cells a
        region's spatial plots show, honoring that region's own
        mask_mode ('none' gives a full rectangular region, 'ellipse' an
        elliptical one, 'file' the exact mask-file shape).

    PARAMETERS:
        t (int): timestep index to compute statistics for.
        regions (Dict[str, Dict[str, np.ndarray]]): region data, keyed by
            region label, each containing 'masked_x'/'masked_y' arrays
            shaped (frames, height, width).
        field_scaling (float): multiplier applied to both components before
            computing statistics (for example, Tesla to Oersted).
        ellipse_size (Tuple[int, int], optional): an ADDITIONAL elliptical
            restriction applied on top of whatever masked_x/masked_y
            already is -- for example, to look at just the inner portion of
            an already file-masked device. This is not a substitute for
            mask_mode='ellipse'/'file'; leave it None (the default) to use
            exactly each region's own mask as-is. See this file's
            module-level LIMITATIONS section regarding this restriction's
            centering convention.
        verbose (bool): prints each region's computed statistics when True.

    RETURNS:
        Tuple[int, Dict[str, Dict[str, float]]]: the same `t` passed in,
        and a dict keyed by region label, each holding
        {"field_magnitude", "mean_magnitude", "mean_angle",
        "angle_concentration", "angle_variance"} for this one timestep. A
        region with no valid (non-NaN) cells after masking gets NaN for
        every metric.
    """
    results = {}  # accumulated across every region below

    for label, region_data in regions.items():  # compute statistics independently for each region
        bx = region_data['masked_x'][t] * field_scaling  # this region's x component at this timestep, scaled
        by = region_data['masked_y'][t] * field_scaling  # this region's y component at this timestep, scaled

        if ellipse_size is not None:  # an additional elliptical restriction was requested
            mask = elliptical_mask(bx.shape, ellipse_size)  # build the extra restriction mask
            bx = np.where(mask, bx, np.nan)  # apply it on top of the region's own existing mask
            by = np.where(mask, by, np.nan)

        mean_bx = np.nanmean(bx)  # mean x component, ignoring masked-out cells
        mean_by = np.nanmean(by)  # mean y component, ignoring masked-out cells

        magnitude_matrix = np.sqrt(bx**2 + by**2)  # per-cell field magnitude

        field_mag = round(np.sqrt(mean_bx**2 + mean_by**2), 5)  # magnitude of the MEAN vector (distinct from the mean of per-cell magnitudes below)

        valid = ~np.isnan(magnitude_matrix)  # cells with real (non-NaN) data
        angle_matrix = np.degrees(np.arctan2(by, bx))  # per-cell field angle, degrees
        angle_matrix[angle_matrix < 0] += 360.0  # shift negative angles into [0, 360)
        angle_matrix = angle_matrix[valid]  # keep only valid cells for the statistics below
        magnitude_matrix = magnitude_matrix[valid]  # keep only valid cells for the statistics below

        if len(magnitude_matrix) == 0:  # this region had no valid cells at all at this timestep
            results[label] = {
                "field_magnitude": np.nan,
                "mean_magnitude": np.nan,
                "mean_angle": np.nan,
                "angle_concentration": np.nan,
                "angle_variance": np.nan
            }
            continue

        if np.all(magnitude_matrix == magnitude_matrix[0]):  # every valid cell has identical magnitude (including the all-zero case)
            weights = np.ones_like(magnitude_matrix)  # fall back to equal weighting
        else:
            weights = magnitude_matrix / np.nansum(magnitude_matrix)  # weight each cell by its share of the total magnitude

        sum_weights = np.sum(weights)  # total weight, used to normalize the weighted sums below
        mean_magnitude = np.nanmean(magnitude_matrix)  # mean of the PER-CELL magnitudes (distinct from field_mag above)

        if sum_weights == 0:  # degenerate case: no usable weight at all
            mean_angle = np.nan
            concentration = np.nan
            variance = np.nan
        else:
            sin_sum = np.sum(np.sin(np.radians(angle_matrix)) * weights)  # weighted sum of sin(angle)
            cos_sum = np.sum(np.cos(np.radians(angle_matrix)) * weights)  # weighted sum of cos(angle)
            weighted_mean_angle = np.degrees(np.arctan2(sin_sum, cos_sum))  # circular weighted mean angle
            if weighted_mean_angle < 0:  # shift negative angles into [0, 360)
                weighted_mean_angle += 360.0
            concentration = np.sqrt(sin_sum**2 + cos_sum**2) / sum_weights  # circular concentration (1 = perfectly aligned, 0 = uniformly spread)
            variance = 1.0 - concentration  # circular variance, the complement of concentration
            mean_angle = weighted_mean_angle

        results[label] = {
            "field_magnitude": field_mag,
            "mean_magnitude": mean_magnitude,
            "mean_angle": mean_angle,
            "angle_concentration": concentration,
            "angle_variance": variance
        }
        if verbose:
            print(f"[DEBUG] compute_weighted_circular_stats_for_frame: t={t}, label='{label}', "
                  f"mean_angle={mean_angle}, mean_magnitude={mean_magnitude}.")

    return t, results


# --------------------------
# ---- Compute weighted circular statistics for a single frame, for a single region ----
def compute_weighted_circular_stats_for_frame_new(
    t: int,  # timestep index
    region_data: Dict[str, np.ndarray],  # a single region's data (not a dict of regions)
    field_scaling: float = 10000.0,  # multiplier applied to both components (for example, Tesla to Oersted)
    ellipse_size: Tuple[int, int] = None,  # optional EXTRA elliptical restriction, layered on top of the region's own mask
    verbose: bool = False,  # print the computed statistics when True
) -> Dict[str, float]:
    """
    GOAL:
        Compute weighted circular statistics at one timestep, for a SINGLE
        region -- the same computation as
        compute_weighted_circular_stats_for_frame, but taking one region's
        data directly instead of a dict of many regions, and returning a
        flat dict instead of one nested by region label.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_pipeline_utilities.matrix_gradient_hsv_rgba, with
        the SAME region_data that function's matrix_angle/matrix_magnitude
        calls also read -- both read masked_x/masked_y, so the mean angle
        computed here is guaranteed to agree with the angle/magnitude
        arrays it is compared against for that plot's circular deviation.

    PARAMETERS:
        t (int): timestep index to compute statistics for.
        region_data (Dict[str, np.ndarray]): a single region's data,
            containing 'masked_x'/'masked_y' arrays shaped
            (frames, height, width).
        field_scaling (float): multiplier applied to both components before
            computing statistics.
        ellipse_size (Tuple[int, int], optional): same meaning as in
            compute_weighted_circular_stats_for_frame -- see that
            function's docstring and this file's module-level LIMITATIONS
            section.
        verbose (bool): prints the computed statistics when True.

    RETURNS:
        Tuple[int, Dict[str, float]]: the same `t` passed in, and a flat
        dict with {"field_magnitude", "mean_magnitude", "mean_angle",
        "angle_concentration", "angle_variance"} for this one timestep and
        region. Every metric is NaN if the region has no valid (non-NaN)
        cells at this timestep.
    """
    bx = region_data['masked_x'][t] * field_scaling  # this region's x component at this timestep, scaled
    by = region_data['masked_y'][t] * field_scaling  # this region's y component at this timestep, scaled

    if ellipse_size is not None:  # an additional elliptical restriction was requested
        mask = elliptical_mask(bx.shape, ellipse_size)  # build the extra restriction mask
        bx = np.where(mask, bx, np.nan)  # apply it on top of the region's own existing mask
        by = np.where(mask, by, np.nan)

    mean_bx = np.nanmean(bx)  # mean x component, ignoring masked-out cells
    mean_by = np.nanmean(by)  # mean y component, ignoring masked-out cells

    magnitude_matrix = np.sqrt(bx**2 + by**2)  # per-cell field magnitude

    field_mag = round(np.sqrt(mean_bx**2 + mean_by**2), 5)  # magnitude of the MEAN vector (distinct from the mean of per-cell magnitudes below)

    valid = ~np.isnan(magnitude_matrix)  # cells with real (non-NaN) data
    angle_matrix = np.degrees(np.arctan2(by, bx))  # per-cell field angle, degrees
    angle_matrix[angle_matrix < 0] += 360.0  # shift negative angles into [0, 360)
    angle_matrix = angle_matrix[valid]  # keep only valid cells for the statistics below
    magnitude_matrix = magnitude_matrix[valid]  # keep only valid cells for the statistics below

    if len(magnitude_matrix) == 0:  # this region had no valid cells at all at this timestep
        results = {
            "field_magnitude": np.nan,
            "mean_magnitude": np.nan,
            "mean_angle": np.nan,
            "angle_concentration": np.nan,
            "angle_variance": np.nan
        }
        return t, results

    if np.all(magnitude_matrix == magnitude_matrix[0]):  # every valid cell has identical magnitude (including the all-zero case)
        weights = np.ones_like(magnitude_matrix)  # fall back to equal weighting
    else:
        weights = magnitude_matrix / np.nansum(magnitude_matrix)  # weight each cell by its share of the total magnitude

    sum_weights = np.sum(weights)  # total weight, used to normalize the weighted sums below
    mean_magnitude = np.nanmean(magnitude_matrix)  # mean of the PER-CELL magnitudes (distinct from field_mag above)

    if sum_weights == 0:  # degenerate case: no usable weight at all
        mean_angle = np.nan
        concentration = np.nan
        variance = np.nan
    else:
        sin_sum = np.sum(np.sin(np.radians(angle_matrix)) * weights)  # weighted sum of sin(angle)
        cos_sum = np.sum(np.cos(np.radians(angle_matrix)) * weights)  # weighted sum of cos(angle)
        weighted_mean_angle = np.degrees(np.arctan2(sin_sum, cos_sum))  # circular weighted mean angle
        if weighted_mean_angle < 0:  # shift negative angles into [0, 360)
            weighted_mean_angle += 360.0
        concentration = np.sqrt(sin_sum**2 + cos_sum**2) / sum_weights  # circular concentration (1 = perfectly aligned, 0 = uniformly spread)
        variance = 1.0 - concentration  # circular variance, the complement of concentration
        mean_angle = weighted_mean_angle

    results = {
        "field_magnitude": field_mag,
        "mean_magnitude": mean_magnitude,
        "mean_angle": mean_angle,
        "angle_concentration": concentration,
        "angle_variance": variance
    }
    if verbose:
        print(f"[DEBUG] compute_weighted_circular_stats_for_frame_new: t={t}, "
              f"mean_angle={mean_angle}, mean_magnitude={mean_magnitude}.")
    return t, results


# --------------------------
# ---- Per-region vectorized worker for compute_stats_for_all_timesteps_parallel ----
def _compute_region_stats_vectorized(label, region_data, field_scaling=10000.0, ellipse_size=None, verbose=False):
    """
    GOAL:
        Compute weighted circular statistics for ONE region, across ALL
        timesteps at once, using vectorized numpy operations over the full
        (frames, height, width) stacked arrays -- instead of looping over
        timesteps in Python and calling
        compute_weighted_circular_stats_for_frame once per frame.

    HOW IT TALKS TO OTHER FILES:
        Called by compute_stats_for_all_timesteps_parallel (this file),
        either directly (in-process) or dispatched to a separate process
        per region (see that function's `max_workers` parameter).

    PARAMETERS:
        label: this region's label, passed through unchanged so the caller
            can match results back to regions when running many of these in
            parallel.
        region_data (Dict[str, np.ndarray]): this region's data, containing
            'masked_x'/'masked_y' arrays shaped (frames, height, width).
        field_scaling (float): multiplier applied to both components before
            computing statistics.
        ellipse_size (Tuple[int, int], optional): same meaning as in
            compute_weighted_circular_stats_for_frame -- see this file's
            module-level LIMITATIONS section regarding this restriction's
            centering convention. Applied as the same 2D mask, broadcast
            identically over every frame.
        verbose (bool): prints this region's computed statistic ranges
            across all timesteps when True.

    RETURNS:
        Tuple[label, Dict[str, List[float]]]: `label` unchanged, and a dict
        with {"field_magnitude", "mean_magnitude", "mean_angle",
        "angle_concentration", "angle_variance"}, each a list of one value
        per timestep -- the same shape as aggregating
        compute_weighted_circular_stats_for_frame's per-region entry across
        every frame, but computed in one vectorized pass instead of a
        Python loop.

    LIMITATIONS:
        Numerically equivalent to a per-frame loop using
        compute_weighted_circular_stats_for_frame's same weighting logic,
        including its edge cases (an all-NaN frame; a frame where every
        valid cell has identical magnitude, including all exactly zero).
        Weights are magnitude-normalized per frame, except when a frame's
        valid magnitudes sum to exactly zero, in which case equal
        (weights=1) weighting is used instead to avoid a 0/0 division --
        for any uniform NONZERO magnitude, weights=1 and
        weights=magnitude/sum(magnitude) give identical results (both are
        just a positive overall scale factor on the weights, which cancels
        in every ratio computed below), so only the true zero-sum case
        needs this special handling. Each pixel's sin(angle)/cos(angle) is
        computed directly from bx/by (sin(atan2(y, x)) == y / hypot(x, y),
        cos(...) == x / hypot(x, y)) rather than via
        degrees(arctan2(...)) -> radians(...) -> sin/cos, which is
        mathematically identical but avoids building a full-resolution
        angle array. np.arctan2(0, 0) is defined as 0 by convention,
        equivalent to cos_theta=1, sin_theta=0 at an exactly-zero pixel --
        handled explicitly here since bx/magnitude, by/magnitude would
        otherwise be 0/0 at those pixels.
    """
    import warnings  # suppressing expected, harmless warnings from fully-masked-out frames

    bx = region_data['masked_x'] * field_scaling  # (T, H, W) -- x component, scaled, every timestep at once
    by = region_data['masked_y'] * field_scaling  # (T, H, W) -- y component, scaled, every timestep at once

    if ellipse_size is not None:  # an additional elliptical restriction was requested
        mask2d = elliptical_mask(bx.shape[1:], ellipse_size)  # build the extra restriction mask, once, for the (height, width) shape shared by every frame
        bx = np.where(mask2d, bx, np.nan)  # apply it on top of the region's own existing mask, broadcasting across every frame
        by = np.where(mask2d, by, np.nan)

    axes = (1, 2)  # reduce over (height, width), keep the frame axis
    with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
        # Frames that are fully masked out (no valid pixels) legitimately trigger
        # "Mean of empty slice" from nanmean -- expected and handled below via
        # `no_data`/`valid_count`, so this is silenced rather than left to print.
        warnings.filterwarnings('ignore', message='Mean of empty slice')

        mean_bx = np.nanmean(bx, axis=axes)  # mean x component per frame, ignoring masked-out cells
        mean_by = np.nanmean(by, axis=axes)  # mean y component per frame, ignoring masked-out cells
        field_mag = np.round(np.sqrt(mean_bx**2 + mean_by**2), 5)  # (T,) -- magnitude of the MEAN vector per frame

        magnitude = np.sqrt(bx**2 + by**2)  # (T, H, W) -- per-cell field magnitude, every frame at once
        valid = ~np.isnan(magnitude)  # cells with real (non-NaN) data, every frame
        valid_count = valid.sum(axis=axes)  # (T,) -- number of valid cells per frame
        mean_magnitude = np.nanmean(magnitude, axis=axes)  # (T,) -- mean of the PER-CELL magnitudes per frame

        magnitude_sum = np.nansum(magnitude, axis=axes)  # (T,) -- total magnitude per frame, used for weight normalization
        zero_sum = magnitude_sum == 0  # frames where every valid cell has exactly zero magnitude (see function LIMITATIONS)
        safe_sum = np.where(zero_sum, 1.0, magnitude_sum)  # avoid a literal 0/0 division for zero-sum frames (result discarded for those frames anyway)
        weights = np.where(
            zero_sum[:, None, None],
            np.where(valid, 1.0, np.nan),  # zero-sum frames: fall back to equal weighting
            magnitude / safe_sum[:, None, None],  # normal frames: weight each cell by its share of the total magnitude
        )
        sum_weights = np.nansum(weights, axis=axes)  # (T,) -- total weight per frame, used to normalize the weighted sums below

        zero_mag = magnitude == 0  # cells with exactly zero magnitude (see function LIMITATIONS regarding arctan2(0,0))
        safe_magnitude = np.where(zero_mag, 1.0, magnitude)  # avoid a literal 0/0 division at exactly-zero cells
        cos_theta = np.where(zero_mag, 1.0, bx / safe_magnitude)  # cos(angle) at every cell, every frame
        sin_theta = np.where(zero_mag, 0.0, by / safe_magnitude)  # sin(angle) at every cell, every frame
        sin_sum = np.nansum(sin_theta * weights, axis=axes)  # (T,) -- weighted sum of sin(angle) per frame
        cos_sum = np.nansum(cos_theta * weights, axis=axes)  # (T,) -- weighted sum of cos(angle) per frame

        weighted_mean_angle = np.degrees(np.arctan2(sin_sum, cos_sum))  # (T,) -- circular weighted mean angle per frame
        weighted_mean_angle = np.where(weighted_mean_angle < 0, weighted_mean_angle + 360.0, weighted_mean_angle)  # shift negative angles into [0, 360)

        safe_sum_weights = np.where(sum_weights == 0, np.nan, sum_weights)  # avoid a literal 0/0 division for a degenerate zero-weight frame
        concentration = np.sqrt(sin_sum**2 + cos_sum**2) / safe_sum_weights  # (T,) -- circular concentration per frame
        variance = 1.0 - concentration  # (T,) -- circular variance per frame

    no_data = valid_count == 0  # frames with no valid cells at all after masking
    field_mag = np.where(no_data, np.nan, field_mag)  # force every metric to NaN for those frames
    mean_magnitude = np.where(no_data, np.nan, mean_magnitude)
    weighted_mean_angle = np.where(no_data, np.nan, weighted_mean_angle)
    concentration = np.where(no_data, np.nan, concentration)
    variance = np.where(no_data, np.nan, variance)

    if verbose:
        print(f"[DEBUG] _compute_region_stats_vectorized: label='{label}', "
              f"mean_angle range [{np.nanmin(weighted_mean_angle):.1f}, {np.nanmax(weighted_mean_angle):.1f}] degrees.")

    return label, {
        "field_magnitude": field_mag.tolist(),  # one value per timestep
        "mean_magnitude": mean_magnitude.tolist(),  # one value per timestep
        "mean_angle": weighted_mean_angle.tolist(),  # one value per timestep
        "angle_concentration": concentration.tolist(),  # one value per timestep
        "angle_variance": variance.tolist(),  # one value per timestep
    }


# --------------------------
# ---- Compute weighted circular statistics for every region, across every timestep ----
def compute_stats_for_all_timesteps_parallel(
    regions: Dict[str, Dict[str, np.ndarray]],  # region data, keyed by region label
    field_scaling: float = 10000.0,  # multiplier applied to both components (for example, Tesla to Oersted)
    ellipse_size: Tuple[int, int] = None,  # optional EXTRA elliptical restriction, layered on top of each region's own mask
    max_workers: int = 1,  # >1 splits regions across separate processes; see this function's LIMITATIONS
    verbose: bool = False,  # print per-region progress when True
) -> Dict[str, Dict[str, List[float]]]:
    """
    GOAL:
        Compute weighted circular statistics for every region in `regions`,
        across every timestep, using a vectorized per-region computation
        (see _compute_region_stats_vectorized) rather than looping over
        timesteps in Python. This is the function both parent scripts
        actually call for statistics.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts' STEP 4. Calls
        _compute_region_stats_vectorized (this file), once per region,
        either directly or dispatched across a process pool depending on
        `max_workers`.

    PARAMETERS:
        regions (Dict[str, Dict[str, np.ndarray]]): region data, keyed by
            region label, as returned by
            region_extraction.reorganize_regions.
        field_scaling (float): multiplier applied to both components before
            computing statistics.
        ellipse_size (Tuple[int, int], optional): same meaning as in
            compute_weighted_circular_stats_for_frame -- see this file's
            module-level LIMITATIONS section.
        max_workers (int): 1 (default) computes every region's statistics
            sequentially, in-process. Values greater than 1 split regions
            across separate processes instead -- see this function's
            LIMITATIONS for when that helps versus hurts.
        verbose (bool): prints progress when True. Forwarded to
            _compute_region_stats_vectorized, so per-region messages appear
            too.

    RETURNS:
        Dict[str, Dict[str, List[float]]]: keyed by region label, each
        holding {"field_magnitude", "mean_magnitude", "mean_angle",
        "mean_angle_unwrapped", "angle_concentration", "angle_variance",
        "angle_deviation_from_mean", "angle_deviation_from_start"}, each a
        list of one value per timestep.

    LIMITATIONS:
        Two independent performance levers, which help in different
        regimes and neither is a universal win on its own:

        1. Vectorization (always applied): each region's full
           (frames, height, width) stack is reduced with a handful of numpy
           calls covering every timestep at once, instead of one Python
           loop iteration per timestep. This is a clear win when there are
           many timesteps over a modest region size (for example, a single
           small device region swept over hundreds or thousands of
           regenerations). For a few very large regions it is roughly a
           wash: the total data volume read is the same either way, so once
           each numpy call is already working on enough data to amortize
           its own overhead, there is nothing further to save -- this
           computation is memory-bandwidth-bound, not
           Python-overhead-bound, once regions get large.

        2. Cross-region multiprocessing (opt-in via max_workers > 1):
           regions are fully independent, so for an array with many device
           regions, splitting them across processes is a genuine additional
           lever on top of vectorization, bounded by core count rather than
           by array size. Off by default (max_workers=1) since spinning up
           a process pool has its own fixed overhead that is not worth
           paying for a handful of regions, and can be slower than the
           sequential path for a small region count.
    """
    region_labels = list(regions.keys())  # every region label to compute statistics for
    if verbose:
        print(f"[INFO] compute_stats_for_all_timesteps_parallel: computing statistics for {len(region_labels)} region(s), max_workers={max_workers}.")

    if max_workers > 1 and len(region_labels) > 1:  # cross-region multiprocessing requested, and there is more than one region to split across processes
        from concurrent.futures import ProcessPoolExecutor  # parallel processing across regions
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_compute_region_stats_vectorized, label, regions[label], field_scaling, ellipse_size, verbose)  # one region's statistics per worker process
                for label in region_labels
            ]
            aggregated_results = dict(f.result() for f in futures)  # collect every region's results into one dict, keyed by label
    else:  # sequential, in-process computation
        aggregated_results = dict(
            _compute_region_stats_vectorized(label, regions[label], field_scaling, ellipse_size, verbose)  # one region's statistics per call
            for label in region_labels
        )

    for label, stats in aggregated_results.items():  # add unwrapped-angle and deviation metrics for every region, after the main statistics are computed
        mean_angles = np.array(stats["mean_angle"])  # this region's mean angle at every timestep
        unwrapped = np.degrees(np.unwrap(np.radians(mean_angles)))  # angle sequence with 360-degree wraparound jumps removed, for smoother plotting
        stats["mean_angle_unwrapped"] = unwrapped.tolist()

        mean_angle_wrapped = np.nanmean(mean_angles)  # this region's own average angle across every timestep

        stats["angle_deviation_from_mean"] = [
            min(abs(angle - mean_angle_wrapped), 360 - abs(angle - mean_angle_wrapped))  # shortest angular distance from this region's own overall mean
            if not np.isnan(angle) else np.nan
            for angle in mean_angles
        ]

        if not np.isnan(mean_angles[0]):  # the first timestep has a valid angle to measure deviation from
            first_angle = mean_angles[0]  # this region's angle at the first timestep
            stats["angle_deviation_from_start"] = [
                min(abs(angle - first_angle), 360 - abs(angle - first_angle))  # shortest angular distance from the first timestep's angle
                if not np.isnan(angle) else np.nan
                for angle in mean_angles
            ]
        else:  # the first timestep itself has no valid angle; deviation from it is undefined for every timestep
            stats["angle_deviation_from_start"] = [np.nan] * len(mean_angles)

    if verbose:
        print(f"[INFO] compute_stats_for_all_timesteps_parallel: finished computing statistics for {len(region_labels)} region(s).")
    return aggregated_results


# --------------------------
# ---- Sample field magnitude along a radial path from a device edge (see LIMITATIONS) ----
def full_field_angle_worker(
    frame_index,  # which frame to load
    npz_directory,  # directory containing the .npz frame files
    prefix,  # filename prefix for the frame files
    theta,  # sampling direction, degrees
    cellsize,  # nm per grid cell (assumed square cells)
    start_length,  # distance past the device edge to start sampling, nm
    max_length,  # maximum distance past the device edge to sample, nm
    r_a,  # device semi-axis in the sampling direction's x component, nm
    r_b,  # device semi-axis in the sampling direction's y component, nm
    center_x,  # device center, x, nm
    center_y,  # device center, y, nm
    output_dir,  # directory to save the sampled path data to
    field_scaling=1.0,  # multiplier applied to the loaded field before sampling
    verbose=False,  # print the number of sample points collected when True
):
    """
    GOAL:
        Sample field magnitude at evenly spaced points along a straight
        radial path, starting just past an elliptical device's edge (at
        angle `theta` from the device center) and continuing outward to
        `max_length`, using bilinear interpolation between grid cells.
        Saves the sampled path (radial distance and field magnitude at each
        point) plus a heatmap marking the sampled cells, to a .npz file.

    HOW IT TALKS TO OTHER FILES:
        See this file's module-level LIMITATIONS section: not currently
        called by any other file in this pipeline.

    PARAMETERS:
        frame_index (int): which frame to load, formatted into the input
            filename as "{prefix}{frame_index:06d}.npz".
        npz_directory (str): directory containing the .npz frame files.
        prefix (str): filename prefix for the frame files.
        theta (float): sampling direction, in degrees, measured the same
            way as every other angle in this pipeline (standard math/
            CCW-positive, 0 degrees = +x).
        cellsize (float): nm per grid cell (assumed square: the same value
            is used for both x and y).
        start_length (float): distance past the device's edge (along the
            sampling direction) to start sampling, in nm.
        max_length (float): maximum distance past the device's edge to
            sample, in nm.
        r_a, r_b (float): the device's semi-axes, in nm, used to compute
            where the device's edge lies along the sampling direction.
        center_x, center_y (float): the device's center, in nm.
        output_dir (str): directory the sampled path .npz file is saved to.
        field_scaling (float): multiplier applied to the loaded field
            components before computing magnitude.
        verbose (bool): prints the number of sample points collected when
            True.

    RETURNS:
        None. Saves a .npz file named
        "angle_{theta:03d}_data{frame_index:06d}.npz" under `output_dir`,
        containing 'heatmap' (a 2D array marking sampled cell locations),
        'r' (radial distance from the device edge at each sample point),
        and 'B' (interpolated field magnitude at each sample point).

    LIMITATIONS:
        See this file's module-level LIMITATIONS section: this function has
        no automated verification alongside it in this documentation pass
        -- treat it as unverified reference code. Sampling stops as soon as
        a sample point would fall outside the loaded frame's grid, even if
        `max_length` has not yet been reached.
    """
    import os  # file path operations
    npz_path = os.path.join(npz_directory, f"{prefix}{frame_index:06d}.npz")  # path to this frame's data
    data = np.load(npz_path)  # load the frame

    Bx = data["x"] * field_scaling  # x component, scaled
    By = data["y"] * field_scaling  # y component, scaled

    H, W = Bx.shape  # grid dimensions, used to bound the sampling loop below

    theta_rad = np.deg2rad(theta)  # sampling direction, in radians
    ux = np.cos(theta_rad)  # unit vector x component, along the sampling direction
    uy = np.sin(theta_rad)  # unit vector y component, along the sampling direction

    Bmag = np.sqrt(Bx**2 + By**2)  # per-cell field magnitude

    heatmap = np.zeros_like(Bmag)  # marks which cells were actually sampled, for visualization/debugging
    r_edge = 1.0 / np.sqrt((ux**2) / (r_a**2) + (uy**2) / (r_b**2))  # distance from the device center to its own edge, along the sampling direction

    r_start = r_edge + start_length  # where sampling begins: just past the device edge, plus the requested offset
    r_end = r_edge + max_length  # where sampling would stop, if the grid didn't run out first

    r_list = []  # radial distance (from the device edge) at each collected sample point
    B_list = []  # interpolated field magnitude at each collected sample point

    for r in np.arange(r_start, r_end, cellsize):  # step outward along the sampling direction, one cell size at a time
        x = center_x + r * ux  # this sample point's physical x position
        y = center_y + r * uy  # this sample point's physical y position

        gx = x / cellsize  # this sample point's position, in grid cells, x
        gy = y / cellsize  # this sample point's position, in grid cells, y

        ix = int(np.floor(gx))  # grid column just below this sample point
        iy = int(np.floor(gy))  # grid row just below this sample point

        dx = gx - ix  # fractional offset within the cell, x
        dy = gy - iy  # fractional offset within the cell, y

        if 0 <= ix < W - 1 and 0 <= iy < H - 1:  # this sample point (and its interpolation neighbors) fall within the grid
            B00 = Bmag[iy, ix]  # magnitude at the bottom-left neighboring cell
            B10 = Bmag[iy, ix + 1]  # magnitude at the bottom-right neighboring cell
            B01 = Bmag[iy + 1, ix]  # magnitude at the top-left neighboring cell
            B11 = Bmag[iy + 1, ix + 1]  # magnitude at the top-right neighboring cell

            B_val = (  # bilinear interpolation between the four neighboring cells
                (1 - dx) * (1 - dy) * B00 +
                dx * (1 - dy) * B10 +
                (1 - dx) * dy * B01 +
                dx * dy * B11
            )

            r_list.append(r - r_edge)  # record this sample point's distance PAST the device edge (not from the device center)
            B_list.append(B_val)  # record this sample point's interpolated magnitude

            heatmap[iy, ix] = 1.0  # mark this cell's approximate location as sampled
        else:  # this sample point fell outside the grid; stop sampling further out
            break

    output_file = os.path.join(output_dir, f"angle_{int(theta):03d}_data{frame_index:06d}.npz")  # output path for this angle/frame's sampled path
    np.savez(
        output_file,
        heatmap=heatmap,  # which cells were sampled
        r=r_list,  # radial distance from the device edge at each sample point
        B=B_list  # interpolated field magnitude at each sample point
    )
    if verbose:
        print(f"[DEBUG] full_field_angle_worker: collected {len(r_list)} sample point(s), saved to '{output_file}'.")


# --------------------------
# ---- Save region stats to CSV (per-frame and/or all-frames-all-regions) ----
def save_region_stats_csv(
    stats,  # nested stats per region, from compute_stats_for_all_timesteps_parallel
    output_dir,  # root folder for this dataset's stats CSVs
    time_step_ns=0.1,  # nanoseconds per timestep, for a time_ns column alongside the raw frame index
    save_per_frame=True,  # whether to write one CSV per timestep
    save_all_frames=True,  # whether to write one CSV covering every (region, timestep) pair
    region_meta=None,  # optional {label -> {"center_name","mask_type"}}, for extra columns
    verbose=False,  # print status messages when True
):
    """
    GOAL:
        Save compute_stats_for_all_timesteps_parallel's output to CSV, in
        up to two forms kept as separate files so a fresh call never
        silently mixes with a previous one:
            - save_per_frame=True: one CSV per timestep, under
              "<output_dir>/per_frame/frame_{t:06d}.csv", each holding
              every region's metrics at that single timestep (one row per
              region). The per_frame directory is reset at the start of
              each call, so stale per-frame CSVs from a previous run do not
              linger alongside a shorter or differently shaped new run.
            - save_all_frames=True: one CSV at
              "<output_dir>/all_regions_all_frames.csv" holding every
              region's metrics across every timestep, long format (one row
              per region/timestep pair). Overwritten each call.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts' STEP 4, once per dataset, with that
        dataset's own output_dir (so m_full and B_demag statistics never
        share a CSV). Calls
        mumax3_pipeline_utilities.move_and_reset_directory to clear the
        per_frame directory before writing into it.

    PARAMETERS:
        stats (Dict): nested statistics per region, from
            compute_stats_for_all_timesteps_parallel (Dict[region_label ->
            Dict[metric_name -> List[float] over time]]).
        output_dir (str): root folder for this dataset's stats CSVs.
        time_step_ns (float): nanoseconds per timestep, used to add a
            time_ns column alongside the raw frame index (matches
            mumax3_csv_plot.plot_region_stats_over_time's x-axis
            convention).
        save_per_frame (bool): whether to write the per-timestep,
            all-regions CSVs.
        save_all_frames (bool): whether to write the single
            all-timesteps, all-regions CSV.
        region_meta: Optional[Dict[region_label -> {"center_name",
            "mask_type"}]], from region_extraction.build_region_meta. These
            CSVs are already long-format (one row per region x frame), so
            unlike the per-region PNG plots, folding mask type in here just
            means adding center_name/mask_type columns rather than new
            subfolders -- a region_label like
            "center_1080_1150_File_Mask" (auto-disambiguated by
            build_region_specs when multiple mask modes share a center)
            stays a single unambiguous row, and these columns make it easy
            to filter/group by mask type in the CSV itself. Omit
            region_meta to get the previous columns unchanged (just
            region_label/frame/time_ns/metrics).
        verbose (bool): prints additional status detail when True (the
            save-completion messages themselves are always shown -- see
            this function's LIMITATIONS).

    RETURNS:
        None. Writes CSV file(s) to disk under `output_dir`.

    LIMITATIONS:
        Assumes every region in `stats` has the same set of metric names
        and the same number of timesteps -- both are read from the first
        region encountered and assumed to apply to all of them. The final
        "saved N file(s)" status messages are printed unconditionally
        (not gated by `verbose`) since they report where output was
        written, which is useful to know even when running quietly
        otherwise.
    """
    import os  # path construction
    import pandas as pd  # CSV writing
    from mumax3_pipeline_utilities import move_and_reset_directory  # clearing the per_frame directory before writing into it

    if not stats:  # nothing to save
        print("[WARNING] save_region_stats_csv: no stats to save (empty stats dict).")
        return

    region_labels = list(stats.keys())  # every region to include in the output
    metric_names = list(next(iter(stats.values())).keys())  # every metric name, read from the first region (see LIMITATIONS)
    num_timesteps = len(next(iter(stats.values()))[metric_names[0]])  # number of timesteps, read from the first region/metric (see LIMITATIONS)
    time_array = [t * time_step_ns for t in range(num_timesteps)]  # physical time (ns) at each timestep index

    os.makedirs(output_dir, exist_ok=True)  # ensure the output directory exists
    if verbose:
        print(f"[INFO] save_region_stats_csv: {len(region_labels)} region(s), {num_timesteps} timestep(s), {len(metric_names)} metric(s).")

    if save_per_frame:  # write one CSV per timestep
        per_frame_dir = os.path.join(output_dir, "per_frame")  # subfolder for the per-timestep CSVs
        move_and_reset_directory(per_frame_dir, verbose=verbose)  # start with a clean directory, so stale files from a previous run do not linger
        for t in range(num_timesteps):  # one CSV per timestep
            rows = []  # accumulated across every region at this timestep
            for label in region_labels:
                row = {"region_label": label, "frame": t, "time_ns": time_array[t]}  # base identifying columns
                if region_meta and label in region_meta:  # extra columns, if region metadata was provided
                    row["center_name"] = region_meta[label].get("center_name")
                    row["mask_type"] = region_meta[label].get("mask_type")
                for metric in metric_names:  # one column per statistic
                    row[metric] = stats[label][metric][t]
                rows.append(row)
            df = pd.DataFrame(rows)  # one row per region, for this timestep
            df.to_csv(os.path.join(per_frame_dir, f"frame_{t:06d}.csv"), index=False)  # write this timestep's CSV
        print(f"[INFO] save_region_stats_csv: saved {num_timesteps} per-frame region-stats CSV(s) to: {per_frame_dir}")

    if save_all_frames:  # write one CSV covering every (region, timestep) pair
        rows = []  # accumulated across every region and timestep
        for label in region_labels:
            for t in range(num_timesteps):
                row = {"region_label": label, "frame": t, "time_ns": time_array[t]}  # base identifying columns
                if region_meta and label in region_meta:  # extra columns, if region metadata was provided
                    row["center_name"] = region_meta[label].get("center_name")
                    row["mask_type"] = region_meta[label].get("mask_type")
                for metric in metric_names:  # one column per statistic
                    row[metric] = stats[label][metric][t]
                rows.append(row)
        df = pd.DataFrame(rows)  # one row per (region, timestep) pair
        all_frames_path = os.path.join(output_dir, "all_regions_all_frames.csv")  # single combined output file
        df.to_csv(all_frames_path, index=False)  # write the combined CSV
        print(f"[INFO] save_region_stats_csv: saved all-region/all-frame stats CSV to: {all_frames_path}")


# ============================================================
# ---- PROPAGATION SWITCHING-PHYSICS CALCULATIONS ----
# Merged in from propagation_stats.py (formerly mumax3_stats.py) -- see this
# file's own module-level PURPOSE/LIMITATIONS sections above for why.
# ============================================================

# --------------------------
# ---- Masked-region field extraction (plain Cartesian mean, not circular) ----
def masked_region_mean_xy(
    region: Dict,  # one region's dict, as returned by mumax3_parent_utilities.compute_regions/region_processor -- must contain "masked_x"/"masked_y", each indexed [frame_index] -> 2D array
    frame_index: int = 0,  # which frame's masked_x/masked_y to average -- almost always 0 for a single-regeneration composed canvas (see mumax3_parent_superposition.py's num_regenerations)
    field_scaling: float = 1.0,  # multiplier applied to both components after averaging (for example, B_TO_OE = 1e4 to convert Tesla to Oe)
    verbose: bool = False,  # print the resulting mean when True
) -> Tuple[float, float]:
    """
    GOAL:
        Reduce one region's masked field to a single (mean_x, mean_y) pair
        -- the plain Cartesian, component-wise nanmean over every
        non-masked cell in that region's footprint. This is the quantity
        the propagation switching criterion needs (Hrev = dot(<H>, m_hat)
        in mumax3_csv_stats.hrev_from_field, below), NOT a circular mean.

    HOW IT TALKS TO OTHER FILES:
        Reads the masked_x/masked_y arrays masking.apply_mask produces
        (via mumax3_parent_utilities.compute_regions/region_processor),
        the same arrays mumax3_csv_stats.compute_weighted_circular_stats_
        for_frame reads for its own (different) circular statistics. Called
        by src/_propagation's per-candidate field-extraction step.

    PARAMETERS:
        region (Dict): one region's data, keyed at minimum by "masked_x"
            and "masked_y", each an array indexed [frame_index] giving a
            2D (height, width) array for that frame, with NaN in every
            masked-out cell (region_extraction/masking.apply_mask's own
            convention).
        frame_index (int): which frame to average.
        field_scaling (float): multiplier applied to both mean components
            after averaging -- for example, B_demag comes out of mumax3 in
            Tesla; pass 1e4 to convert to Oe before returning.
        verbose (bool): prints the resulting (mean_x, mean_y) when True.

    RETURNS:
        Tuple[float, float]: (mean_x, mean_y), both already multiplied by
        field_scaling. NaN in either component if every cell in this
        region's mask was itself NaN (an empty/degenerate mask).

    RAISES:
        KeyError if `region` does not contain both "masked_x" and
        "masked_y".

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding this
        function's expected region-dict shape.
    """
    if "masked_x" not in region or "masked_y" not in region:  # this region was not masked (mask_mode="none") or was built by an incompatible caller
        raise KeyError(
            f"masked_region_mean_xy: region is missing 'masked_x'/'masked_y'. "
            f"Available keys: {list(region.keys())}. Was this region built with "
            f"mask_mode='file' (or 'ellipse')?"
        )
    hx = np.asarray(region["masked_x"][frame_index], dtype=float)  # this frame's masked x-component, 2D (height, width), NaN outside the mask
    hy = np.asarray(region["masked_y"][frame_index], dtype=float)
    mean_x = float(np.nanmean(hx) * field_scaling)  # average over every non-masked cell, then apply the caller's unit conversion
    mean_y = float(np.nanmean(hy) * field_scaling)
    if verbose:
        print(f"[DEBUG] masked_region_mean_xy: frame={frame_index}, mean=({mean_x:.4f}, {mean_y:.4f}), field_scaling={field_scaling}.")
    return mean_x, mean_y


# --------------------------
# ---- Physical easy-axis field component ----
def hrev_from_field(Hx_Oe: float, Hy_Oe: float, m_angle_deg: float) -> float:
    """
    GOAL:
        Compute Hrev = dot(H_demag, m_hat), the local effective field's
        component ALONG an island's current magnetization direction.

    HOW IT TALKS TO OTHER FILES:
        Pure-math helper; called by src/_propagation's switching-decision
        step with (Hx_Oe, Hy_Oe) from masked_region_mean_xy (this file) and
        m_angle_deg from geometry.DeviceArray's own device_m_angles/
        device_data.

    PARAMETERS:
        Hx_Oe, Hy_Oe (float): local effective field components, in Oe (or
            any consistent field unit -- the result is in the same unit).
        m_angle_deg (float): the island's current magnetization angle,
            degrees (0 = +x, 90 = +y).

    RETURNS:
        float: Hrev, in the same unit as Hx_Oe/Hy_Oe. Positive Hrev means
        the local field points along the island's current magnetization
        (stabilizing); a flip is possible only when Hrev < 0 (destabilizing)
        AND |Hrev| exceeds the island's coercivity (see
        lookup_nearest_coercivity, below).
    """
    theta = np.deg2rad(m_angle_deg)  # magnetization direction, radians
    m_hat = np.array([np.cos(theta), np.sin(theta)])  # unit vector along the current magnetization
    H = np.array([Hx_Oe, Hy_Oe])
    return float(np.dot(H, m_hat))


# --------------------------
# ---- Local field direction ----
def field_angle_deg_from_xy(Hx_Oe: float, Hy_Oe: float) -> float:
    """
    GOAL:
        Return the direction of the local effective field, in degrees.

    HOW IT TALKS TO OTHER FILES:
        Pure-math helper; called by src/_propagation alongside hrev_from_
        field, for logging/plotting the field direction (independent of
        which island's magnetization it is being compared against).

    PARAMETERS:
        Hx_Oe, Hy_Oe (float): local effective field components.

    RETURNS:
        float: field direction in degrees, 0-360 (0 = +x, 90 = +y).
        np.nan if the in-plane field is essentially zero (both components
        within np.isclose of zero), since a zero vector has no well-defined
        direction.
    """
    if np.isclose(Hx_Oe, 0.0) and np.isclose(Hy_Oe, 0.0):  # zero-length field: direction is undefined
        return np.nan
    return float((np.rad2deg(np.arctan2(Hy_Oe, Hx_Oe)) + 360.0) % 360.0)


# --------------------------
# ---- Acute field/easy-axis angle (for angle-dependent coercivity lookup) ----
def acute_angle_to_easy_axis(field_angle_deg: float, easy_axis_deg: float) -> float:
    """
    GOAL:
        Return the UNSIGNED acute angle (0-90 deg) between the local field
        direction and an island's easy axis.

    HOW IT TALKS TO OTHER FILES:
        Pure-math helper; the result is the lookup key for lookup_nearest_
        coercivity, below, and for any angle-dependent-Hc CSV table (see
        load_coercivity_table's own documentation for the CSV's expected
        angle range).

    PARAMETERS:
        field_angle_deg (float): local field direction, degrees (0-360, or
            np.nan if undefined -- see field_angle_deg_from_xy).
        easy_axis_deg (float): the island's geometry (easy-axis)
            orientation, degrees. An island's easy axis is UNDIRECTED --
            0 deg and 180 deg describe the same physical axis for a
            horizontal island -- so this folds the raw difference into
            0-90 deg rather than treating 0 and 180 as different axes.

    RETURNS:
        float: acute angle, 0-90 deg. np.nan if field_angle_deg is np.nan.
    """
    if not np.isfinite(field_angle_deg):  # undefined field direction: no well-defined angle to the easy axis either
        return np.nan
    diff = (field_angle_deg - easy_axis_deg + 90.0) % 180.0 - 90.0  # fold into [-90, 90) using the easy axis's own 180-deg periodicity
    return float(abs(diff))


# --------------------------
# ---- Signed field/easy-axis angle (diagnostic only, not used for Hc lookup) ----
def signed_angle_difference_deg(angle_deg: float, reference_deg: float) -> float:
    """
    GOAL:
        Return the SIGNED angle difference (angle_deg - reference_deg),
        folded into [-180, 180) deg -- useful for reporting the raw field
        direction relative to an island's easy axis BEFORE folding into
        the acute 0-90 deg form acute_angle_to_easy_axis (above) uses for
        the actual Hc lookup.

    HOW IT TALKS TO OTHER FILES:
        Pure-math helper; called by src/_propagation purely for CSV/plot
        diagnostics (delta_theta_deg column), not for any switching
        decision itself.

    PARAMETERS:
        angle_deg (float): the angle being compared, degrees (or np.nan).
        reference_deg (float): the reference angle it is compared against,
            degrees.

    RETURNS:
        float: signed difference in [-180, 180) deg. np.nan if angle_deg is
        np.nan.
    """
    if not np.isfinite(angle_deg):
        return np.nan
    return float((angle_deg - reference_deg + 180.0) % 360.0 - 180.0)


# --------------------------
# ---- Angle-dependent coercivity table ----
def load_coercivity_table(csv_path: "Path", verbose: bool = False) -> Optional[List[Tuple[float, float]]]:
    """
    GOAL:
        Load an angle-dependent coercivity table from a two-column CSV
        (angle_deg, Hc_Oe), covering 0-90 deg only.

    HOW IT TALKS TO OTHER FILES:
        Called once by src/_propagation at run start; the returned table is
        passed into lookup_nearest_coercivity (this file) on every
        candidate evaluation thereafter.

    PARAMETERS:
        csv_path (Path): path to the coercivity CSV. Expected columns:
                angle_deg,Hc_Oe
                0,70.0
                5,72.0
                ...
                90,100.0
            angle_deg must be 0-90 deg -- the caller is responsible for
            folding a raw field/easy-axis angle into that range first (see
            acute_angle_to_easy_axis, above) before ever looking a value
            up in this table.
        verbose (bool): prints the number of rows loaded when True.

    RETURNS:
        Optional[List[Tuple[float, float]]]: a list of (angle_deg, Hc_Oe)
        pairs, sorted by angle_deg ascending. None if csv_path does not
        exist -- the caller should fall back to a constant default
        coercivity in that case (see lookup_nearest_coercivity's own
        `default_coercivity_oe` parameter).

    RAISES:
        ValueError if the CSV exists but is missing a required column, has
        an angle_deg outside 0-90 deg, has a negative Hc_Oe, or is empty.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():  # missing CSV: caller falls back to a constant default coercivity
        if verbose:
            print(f"[WARNING] load_coercivity_table: CSV not found: {csv_path}. Caller should fall back to a default Hc.")
        return None

    table: List[Tuple[float, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"angle_deg", "Hc_Oe"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):  # required columns must be present
            raise ValueError(f"Coercivity CSV must contain columns {sorted(required)}. Found: {reader.fieldnames}")
        for row_number, row in enumerate(reader, start=2):  # start=2: row 1 is the header
            if not row or not str(row.get("angle_deg", "")).strip():  # skip blank rows
                continue
            angle = float(row["angle_deg"])
            hc = float(row["Hc_Oe"])
            if angle < 0.0 or angle > 90.0:  # this table is keyed by the ACUTE (folded) angle only -- see acute_angle_to_easy_axis
                raise ValueError(
                    f"Coercivity angle {angle} deg on CSV row {row_number} is outside 0-90 deg. "
                    "Use the folded easy-axis angle, not a 0-180 deg magnetization-field angle."
                )
            if hc < 0.0:
                raise ValueError(f"Coercivity Hc must be nonnegative; row {row_number} has Hc={hc}")
            table.append((angle, hc))

    if not table:  # CSV existed but had no usable rows
        raise ValueError(f"Coercivity table is empty: {csv_path}")
    table.sort(key=lambda x: x[0])
    if verbose:
        print(f"[INFO] load_coercivity_table: loaded {len(table)} row(s) from {csv_path}.")
    return table


def lookup_nearest_coercivity(
    theta_hc_deg: float,  # acute field/easy-axis angle, 0-90 deg (see acute_angle_to_easy_axis) -- or np.nan
    table: Optional[List[Tuple[float, float]]],  # from load_coercivity_table, or None
    default_coercivity_oe: float,  # fallback value -- caller-supplied, not a module-level constant (see this file's module-level LIMITATIONS section)
    verbose: bool = False,  # print the resolved Hc when True
) -> Tuple[float, float, float, str]:
    """
    GOAL:
        Look up the nearest-angle coercivity for a given acute field/
        easy-axis angle, falling back to a constant default when no table
        is available or the angle itself is undefined.

    HOW IT TALKS TO OTHER FILES:
        Called by src/_propagation once per candidate evaluation, with
        theta_hc_deg from acute_angle_to_easy_axis (this file) and `table`
        from load_coercivity_table (this file).

    PARAMETERS:
        theta_hc_deg (float): acute field/easy-axis angle, 0-90 deg, or
            np.nan.
        table (Optional[List[Tuple[float, float]]]): sorted (angle_deg,
            Hc_Oe) pairs, or None.
        default_coercivity_oe (float): value returned when `table` is None
            or theta_hc_deg is not finite.
        verbose (bool): prints the resolved (Hc_Oe, angle_used, source)
            when True.

    RETURNS:
        Tuple[float, float, float, str]:
            Hc_Oe (float): the resolved coercivity.
            Hc_angle_used_deg (float): the table angle actually used
                (np.nan if the default fallback was used).
            Hc_angle_error_deg (float): |Hc_angle_used_deg - theta_hc_deg|
                (np.nan if the default fallback was used).
            Hc_source (str): "default" or "csv_nearest".
    """
    if table is None or not np.isfinite(theta_hc_deg):  # no table, or an undefined lookup angle: fall back to the constant default
        if verbose:
            print(f"[DEBUG] lookup_nearest_coercivity: using default Hc={default_coercivity_oe:.4f} Oe (table={'missing' if table is None else 'present'}, theta_hc_deg={theta_hc_deg}).")
        return default_coercivity_oe, np.nan, np.nan, "default"

    angles = np.array([a for a, _ in table], dtype=float)
    hcs = np.array([h for _, h in table], dtype=float)
    idx = int(np.argmin(np.abs(angles - theta_hc_deg)))  # nearest-angle lookup (no interpolation)
    angle_used = float(angles[idx])
    hc_used = float(hcs[idx])
    angle_error = float(abs(angle_used - theta_hc_deg))
    if verbose:
        print(f"[DEBUG] lookup_nearest_coercivity: theta_hc_deg={theta_hc_deg:.2f}, matched table angle={angle_used:.1f} (error={angle_error:.2f} deg), Hc={hc_used:.4f} Oe.")
    return hc_used, angle_used, angle_error, "csv_nearest"
