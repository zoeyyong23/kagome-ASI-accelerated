r"""
File Name: propagation_plot.py
Author: Connor Sullivan (original neighbor-field-comparison and flip-path
        visualizations, in the two uploaded standalone propagation
        scripts); ported nearly as-is, re-plumbed onto the current engine's
        live call points/CSV schema, and generalized to per-device size
        (geometry-general) by Claude at Connor's request -- see this
        file's own LIMITATIONS section for exactly what changed.
Date: 2026.08.10
Revision: v1

============================================================
PURPOSE
============================================================
Two visualizations for verifying propagation_engine.py's own behavior
visually, both ported nearly as-is from the uploaded scripts' own
plotting code:

1. plot_neighbor_field_comparison: one figure per (step, source device) --
   the source and its candidate neighbors, a blue quiver arrow showing the
   direction of H_demag each candidate would feel with ITS OWN
   contribution excluded, a black arrow showing each device's current
   magnetization direction, and switch/no-switch coloring (gold = the
   strongest accepted candidate, green = eligible but not the strongest,
   red = not eligible). Called LIVE from inside propagation_engine.
   run_propagation's own stepping loop, once per source per step, using
   that source's already-computed `results` list -- see this file's own
   LIMITATIONS section for why this one is called inline rather than
   after the fact.

2. plot_flip_path / generate_flip_path_visualizations: the whole-array
   propagation-front map -- every generation of flipped devices colored by
   how many synchronous steps after the initial injection(s) they were
   accepted, thin dotted arrows for every candidate checked (green =
   eligible, red = not), thick black arrows for actually-accepted
   source -> candidate flips (multiple arrows converging on one device if
   two paths collapsed onto it). Produces a static PNG, one PNG per
   generation (for an animation), and a GIF if Pillow is available. Called
   ONCE, after propagation_engine.run_propagation finishes, since it needs
   the complete flip history.

============================================================
WHAT CHANGED FROM THE UPLOADED SCRIPTS (mechanical, not algorithmic)
============================================================
- Per-device LENGTH/WIDTH now come from the live DeviceArray's own
  device_data (device_data[idx]["device_length_nm"]/["device_width_nm"],
  always populated -- see geometry.DeviceArray._refresh_data), not a flat
  DEVICE_LENGTH_NM/DEVICE_WIDTH_NM module constant. This is the one change
  that actually matters for geometry generality: a kagome array can (and
  the worked kagome examples in mumax3_parent_superposition.py do) use the
  same size for every device, but this file no longer assumes that.
- "nn_id" (a square-ASI compass label like "NN_UL") is "slot_label" in the
  comparison CSV now (propagation_neighbors.find_candidate_neighbors' own
  geometry-general label form) -- displayed as-is, unchanged rendering
  logic otherwise.
- The uploaded visualize_flip_path_multi_injection.py script's own
  standalone-reload machinery (parsing flip_history.npy/.txt, re-importing
  a sibling script's build_square_asi() by file path to recover geometry,
  a fallback_square_asi_geometry() reconstruction) is NOT ported. This
  file's own generate_flip_path_visualizations takes the live DeviceArray
  and propagation_engine.run_propagation's own in-memory result dict
  directly -- there is no history to "reload" here, since it is called
  once, right after the run that produced it, from the same process (see
  mumax3_parent_propagation.py). A future standalone re-plot-from-disk
  entry point would only need comparison_csv/step_summary_csv (both
  already written to disk) plus a way to rebuild the array from a saved
  PropagationConfig -- not attempted here since Connor asked specifically
  to verify THIS run's engine behavior, not to build a general later-reload
  tool.

============================================================
LIMITATIONS
============================================================
- plot_neighbor_field_comparison is called INLINE, from inside
  propagation_engine.run_propagation's own loop, using the array's LIVE
  state at the moment each source's candidates were evaluated (before that
  step's accepted flips are applied) -- not reconstructed after the fact
  from saved CSVs. A device's magnetization angle changes over the course
  of a run, so a faithful post-hoc reconstruction would need to replay the
  flip history up to the exact moment in question; calling this inline
  instead, at the exact point the physics were computed, sidesteps that
  entirely and is far simpler. See propagation_engine.py's own call site
  for where this happens.
- generate_flip_path_visualizations, by contrast, only needs GEOMETRY
  (device_centers/device_angles/device sizes), which never changes during
  a propagation run (only magnetization angle does) -- so it is safe to
  call once, after the fact, from the final array.
"""

import os  # path construction
from typing import Dict, Iterable, List, Optional, Tuple, Union  # type hints used throughout this file's signatures

import numpy as np  # array operations used throughout this file
import matplotlib.pyplot as plt  # all plotting in this file

from propagation_utilities import get_device_mesh  # fresh live-array reads (see that file)


# --------------------------
# ---- Shared low-level helper: draw one island as a thick line segment along its easy axis ----
def _draw_island_segment(ax, x_nm: float, y_nm: float, angle_deg: float, length_nm: float,
                          color: str = "0.82", linewidth: float = 5.0, alpha: float = 1.0, zorder: int = 1) -> None:
    """
    GOAL:
        Draw one island as a thick line segment centered at (x_nm, y_nm),
        oriented along angle_deg, length_nm long -- the shared rendering
        primitive both visualizations in this file use for every island.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_neighbor_field_comparison and plot_flip_path (both
        this file). Pure matplotlib; no other file dependencies.

    PARAMETERS:
        ax: matplotlib Axes to draw on.
        x_nm, y_nm (float): island center, nm.
        angle_deg (float): island orientation, degrees.
        length_nm (float): THIS island's own length, nm -- per-device (see
            this file's module-level "WHAT CHANGED" section), not a flat
            constant.
        color, linewidth, alpha, zorder: plain matplotlib styling.

    RETURNS:
        None.
    """
    theta = np.deg2rad(angle_deg)
    dx = 0.5 * length_nm * np.cos(theta)
    dy = 0.5 * length_nm * np.sin(theta)
    ax.plot([x_nm - dx, x_nm + dx], [y_nm - dy, y_nm + dy], color=color, linewidth=linewidth,
            alpha=alpha, solid_capstyle="round", zorder=zorder)


def _device_length_nm(array, idx: int) -> float:
    """
    GOAL:
        Return one device's own length, nm -- always populated on
        device_data (see geometry.DeviceArray._refresh_data), so this is a
        plain lookup, kept as its own tiny helper only so every call site
        in this file reads the same way.

    PARAMETERS:
        array (geometry.DeviceArray): the live array.
        idx (int): device index.

    RETURNS:
        float: device_data[idx]["device_length_nm"].
    """
    return float(array.device_data[idx]["device_length_nm"])


def _unit_vector(dx: float, dy: float, fallback: Tuple[float, float] = (0.7071, -0.7071)) -> Tuple[float, float]:
    """
    GOAL:
        Normalize (dx, dy) to a unit vector; if the input is (numerically)
        the zero vector -- e.g. a candidate that happens to sit exactly at
        the local cluster center -- return `fallback` instead of dividing
        by zero.

    PARAMETERS:
        dx, dy (float): vector components.
        fallback (Tuple[float, float]): returned when hypot(dx, dy) is
            ~0.

    RETURNS:
        Tuple[float, float]: (ux, uy), a unit vector.
    """
    norm = float(np.hypot(dx, dy))
    if norm < 1e-9:
        return fallback
    return dx / norm, dy / norm


def _local_lattice_spacing_nm(points_xy: np.ndarray, default_nm: float = 90.0) -> float:
    """
    GOAL:
        Estimate a representative device-to-device spacing, nm, from a
        small set of points (typically one step's source + its
        candidates). Label push-out distance is scaled off THIS number
        rather than a fixed nm constant, so placement stays correct
        whether the array is square, pinwheel (45deg rotated), or kagome,
        and whether the lattice pitch itself is large or small -- the same
        reasoning this file already applies to per-device LENGTH/WIDTH
        (see this file's module-level "WHAT CHANGED" section); label
        placement was the one place that had NOT been made geometry-
        general yet.

    PARAMETERS:
        points_xy (np.ndarray): Mx2 array of (x, y) nm positions.
        default_nm (float): returned when fewer than 2 points are given
            (nothing to measure a spacing from -- e.g. a source with zero
            candidates).

    RETURNS:
        float: median nearest-neighbor distance among points_xy, nm.
    """
    if len(points_xy) < 2:
        return default_nm
    diffs = points_xy[:, None, :] - points_xy[None, :, :]
    dist = np.hypot(diffs[..., 0], diffs[..., 1])
    np.fill_diagonal(dist, np.inf)
    nearest = np.min(dist, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    return float(np.median(nearest)) if len(nearest) else default_nm


def _rotate_unit_vector(ux: float, uy: float, angle_deg: float) -> Tuple[float, float]:
    """
    GOAL:
        Rotate a unit vector by angle_deg (CCW-positive -- this file's
        pipeline-wide rotation convention, same as everywhere else in this
        codebase; see superposition_operations.rotate_field_spatial's own
        sign-convention note for the reasoning).

    PARAMETERS:
        ux, uy (float): unit vector components.
        angle_deg (float): rotation angle, degrees, CCW-positive.

    RETURNS:
        Tuple[float, float]: rotated (ux, uy), still unit length.
    """
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    return ux * c - uy * s, ux * s + uy * c


def _push_label_clear_of_devices(
    anchor_xy: Tuple[float, float],
    direction_xy: Tuple[float, float],
    base_radius_nm: float,
    obstacle_centers: np.ndarray,
    obstacle_radii_nm: np.ndarray,
    label_half_width_nm: float,
    label_half_height_nm: float,
    step_nm: float = 18.0,
    max_rings: int = 8,
    angle_offsets_deg: Optional[List[float]] = None,
) -> Tuple[float, float]:
    """
    GOAL:
        Find a label position, near anchor_xy and roughly along
        direction_xy, whose (approximate, axis-aligned) bounding box
        doesn't overlap any obstacle device's own footprint circle.

        This is a FAN search, not a straight-line walk: at each radius
        ring (base_radius_nm + k*step_nm, k=0..max_rings), it tries
        direction_xy first, then a handful of small angular deviations
        from it, before growing the radius further. A straight-line walk
        (the earlier version of this function) very often runs parallel
        to an entire ROW of neighboring devices in a packed lattice --
        pinwheel and kagome both have long straight rows in several
        directions -- so growing the radius alone just keeps walking
        along that same blocked row, and the caller ends up needing a
        very large max distance to guarantee escape. Trying nearby
        angles instead usually clears in one or two rings. That keeps
        the typical label displacement small, which is what actually
        controls how far plot_neighbor_field_comparison has to zoom out
        to keep every label in frame (see that function's own
        xs_for_lim/ys_for_lim, which are built from this function's
        return values) -- so a small, reliable displacement here is what
        keeps that plot zoomed in and the on-plot arrows a sensible size,
        not a separate fix on the caller's side.

    HOW IT TALKS TO OTHER FILES:
        Called by plot_neighbor_field_comparison (this file) once per
        label placed (every candidate label and the source label).

    PARAMETERS:
        anchor_xy (Tuple[float, float]): the device position this label
            belongs to, nm -- radius is measured from here, not from an
            already-offset starting point.
        direction_xy (Tuple[float, float]): unit vector for the
            PREFERRED direction (radius ring 0, angle offset 0 is tried
            first) -- see plot_neighbor_field_comparison for how this is
            chosen per label (radially outward from the local
            source+candidates cluster).
        base_radius_nm (float): starting radius, nm, before any ring
            growth.
        obstacle_centers (np.ndarray): Kx2, every device center to avoid,
            nm -- the full array, so a label can't land on a device that
            isn't even one of this step's candidates.
        obstacle_radii_nm (np.ndarray): K, each device's own exclusion
            radius, nm (device_length_nm / 2 plus a small buffer, widened
            further for devices that also have quiver arrows drawn out of
            them -- see call site).
        label_half_width_nm, label_half_height_nm (float): approximate
            half-extent of the label's own text box, nm -- used as an
            axis-aligned box around the anchor point for the overlap
            test. This is intentionally approximate (an exact
            pixel-perfect text extent would need a live renderer at call
            time, before the figure's final layout exists) -- close
            enough to stop a label landing ON a device without needing a
            draw-measure-redraw cycle.
        step_nm (float): radius grown per ring.
        max_rings (int): cap on how many rings this will try (bounded so
            a pathological case -- e.g. every direction at every radius
            blocked -- can't push a label arbitrarily far off the plot).
        angle_offsets_deg (Optional[List[float]]): angular deviations
            from direction_xy to try at each ring, in try-order (closest
            to the preferred direction first). Defaults to a +/-54deg
            fan in 18deg steps -- wide enough to sidestep a blocked row
            without ending up pointed back toward the cluster center.

    RETURNS:
        Tuple[float, float]: (x, y), the final label position, nm. If no
        clear spot is found within the search budget, returns the
        farthest ring tried along the original (unrotated) direction --
        still an improvement over the un-nudged anchor.
    """
    if angle_offsets_deg is None:
        angle_offsets_deg = [0.0, 18.0, -18.0, 36.0, -36.0, 54.0, -54.0, 72.0, -72.0]
    ax_, ay_ = anchor_xy
    ux0, uy0 = direction_xy
    if len(obstacle_centers) == 0:
        return ax_ + ux0 * base_radius_nm, ay_ + uy0 * base_radius_nm
    for ring in range(max_rings + 1):
        radius = base_radius_nm + ring * step_nm
        for angle_offset in angle_offsets_deg:
            ux, uy = _rotate_unit_vector(ux0, uy0, angle_offset) if angle_offset else (ux0, uy0)
            x, y = ax_ + ux * radius, ay_ + uy * radius
            dx = obstacle_centers[:, 0] - x
            dy = obstacle_centers[:, 1] - y
            clear = (np.abs(dx) > (label_half_width_nm + obstacle_radii_nm)) | \
                    (np.abs(dy) > (label_half_height_nm + obstacle_radii_nm))
            if np.all(clear):
                return x, y
    radius = base_radius_nm + max_rings * step_nm
    return ax_ + ux0 * radius, ay_ + uy0 * radius


def _widest_angular_gap_direction(vectors_xy: np.ndarray, fallback: Tuple[float, float] = (0.7071, -0.7071)) -> Tuple[float, float]:
    """
    GOAL:
        Given vectors from a common point out to several obstacles
        (here: source -> each of its candidates), return the unit
        direction that bisects the WIDEST angular gap between them --
        i.e. the most open direction around that point.

        Used for the source label's push direction. An earlier version
        used "opposite the mean direction toward the candidates," which
        works fine when candidates cluster on one side of the source but
        degenerates toward the zero vector when candidates roughly
        surround the source (common once a source has many neighbors,
        e.g. kagome or a dense pinwheel neighborhood) -- the mean of
        several roughly-opposing unit vectors is small, so the direction
        is dominated by whichever fallback is picked, which has no reason
        to actually be clear. The widest-gap approach instead reasons
        about where the candidates ARE (their angular positions) rather
        than an average that can cancel out, so it degrades gracefully
        for any number of candidates in any arrangement.

    PARAMETERS:
        vectors_xy (np.ndarray): Mx2 array of (dx, dy) vectors from the
            source to each candidate, nm.
        fallback (Tuple[float, float]): returned when there are no
            candidates to bisect around.

    RETURNS:
        Tuple[float, float]: unit vector pointing into the widest open
        angular gap.
    """
    if len(vectors_xy) == 0:
        return fallback
    angles_deg = np.degrees(np.arctan2(vectors_xy[:, 1], vectors_xy[:, 0])) % 360.0
    angles_sorted = np.sort(angles_deg)
    n = len(angles_sorted)
    best_gap, best_mid = -1.0, 0.0
    for i in range(n):
        a0 = angles_sorted[i]
        a1 = angles_sorted[(i + 1) % n] + (360.0 if i == n - 1 else 0.0)
        gap = a1 - a0
        if gap > best_gap:
            best_gap, best_mid = gap, (a0 + a1) / 2.0
    theta = np.deg2rad(best_mid)
    return float(np.cos(theta)), float(np.sin(theta))


# --------------------------
# ---- Visualization 1: per-step, per-source neighbor-field comparison ----
def plot_neighbor_field_comparison(
    array,  # geometry.DeviceArray -- LIVE state at the moment `results` was computed (see this file's module-level LIMITATIONS section)
    results: List[Dict],  # this source's own candidate rows for this step (same shape as propagation_engine's comparison CSV rows)
    source_idx: int,
    step: int,
    output_path: str,
    best_idx: Optional[Union[int, Iterable[int]]] = None,  # the accepted candidate's device index, if any (None = no eligible candidate this source/step) -- accepts a single int OR an iterable of ints (config.flip_mode="all_eligible"/branch_first_step can accept more than one candidate for the same source in the same step; every index given here is drawn gold, not just the single strongest)
    show_full_array_background: bool = True,
    arrow_length_nm: Optional[float] = None,  # fixed on-plot length for the blue H_demag-direction arrow, nm; None (default) auto-scales to this step's local lattice spacing (see PASS 1 below) so arrows look a sensible size whether the array's pitch is 30nm or 500nm, instead of one fixed value that only looks right at one particular spacing. Pass an explicit number to override.
    mag_arrow_length_nm: Optional[float] = None,  # same auto-scaling as arrow_length_nm, for the black magnetization-direction arrow
    margin_nm: float = 60.0,  # small breathing-room padding beyond each label's own real bounding box (which is now computed directly, see PASS 1 below) -- reduced from an earlier 220.0 now that the box itself, not a flat guess, drives the plot extent
    label_device_buffer_nm: float = 12.0,  # extra clearance added on top of each device's own half-length when testing whether a label's box overlaps that device
    dpi: int = 250,
    verbose: bool = False,
) -> None:
    """
    GOAL:
        Save one figure showing this source device, its candidate
        neighbors this step, each candidate's H_demag direction/magnitude
        (excluding that candidate's own contribution) as a blue arrow,
        each device's current magnetization as a black arrow, and switch/
        no-switch/accepted coloring.

    HOW IT TALKS TO OTHER FILES:
        Called by propagation_engine.run_propagation, once per active
        source per step, immediately after that source's `results` list is
        fully computed (see propagation_engine.py's own call site for
        why this must happen before that step's accepted flips are
        applied). Reads array.device_data (geometry.py) for per-device
        length; calls propagation_utilities.get_device_mesh for the live
        mesh.

    PARAMETERS:
        See the inline comments on each parameter above. `results` rows
        are expected to have the same keys propagation_engine.py's
        comparison-CSV rows do (candidate_idx, slot_label,
        magnetization_angle_deg, Hdemag_x_Oe, Hdemag_y_Oe, Hdemag_mag_Oe,
        H_reverse_Oe, H_switch_drive_Oe, Hc_Oe, field_angle_deg, switch).

    RETURNS:
        None. Writes output_path (parent directories created if needed).
    """
    if not results:
        if verbose:
            print(f"[DEBUG] plot_neighbor_field_comparison: no results for source={source_idx}, step={step}; skipping.")
        return

    centers_nm, geom_angles_deg, mag_angles_deg = get_device_mesh(array)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if best_idx is None:  # normalize best_idx to a set, whether given as a single int or an iterable -- see this parameter's own documentation
        accepted_indices = set()
    elif isinstance(best_idx, int):
        accepted_indices = {best_idx}
    else:
        accepted_indices = {int(i) for i in best_idx}

    sx, sy = float(centers_nm[source_idx][0]), float(centers_nm[source_idx][1])

    # ---- PASS 1: figure out where every label actually goes before drawing anything ----
    # This has to happen before ax.set_xlim/set_ylim (below) because the plot limits need to
    # include the pushed-out label positions, not just the devices, or a label can land off-plot.
    # See this file's module-level "WHAT CHANGED" note and the docstrings on _local_lattice_spacing_nm
    # / _push_label_clear_of_devices for why this replaces the old fixed (+34, +34) nm offset.
    #
    # NOTE on box size vs. render: labels are drawn CENTERED (ha="center", va="center") on the
    # final anchor point below, so the half_width/half_height used here for the clearance test
    # match the box actually drawn -- pixel_footprint of the rendered text.
    # An earlier version anchored a CORNER at the pushed point instead (ha/va set from push
    # direction), which let the real box extend up to 2x further outward than the clearance test
    # accounted for -- that's what was cutting into the arrows and, in tightly spaced arrays,
    # into the plot edge. Centering is what actually keeps drawn box == tested box.
    candidate_positions = np.array([[float(centers_nm[int(row["candidate_idx"])][0]),
                                      float(centers_nm[int(row["candidate_idx"])][1])] for row in results])
    candidate_idxs = [int(row["candidate_idx"]) for row in results]
    cluster_points = np.vstack([[sx, sy], candidate_positions])
    cluster_center = cluster_points.mean(axis=0)
    local_spacing_nm = _local_lattice_spacing_nm(cluster_points)

    # Auto-scale arrow lengths to THIS step's actual lattice spacing when not given explicitly --
    # a fixed nm arrow only looks right at one particular spacing; clipped to a sane visible range
    # so a pathologically tiny or huge spacing can't make an arrow disappear or dominate the plot.
    if arrow_length_nm is None:
        arrow_length_nm = float(np.clip(0.8 * local_spacing_nm, 20.0, 140.0))
    if mag_arrow_length_nm is None:
        mag_arrow_length_nm = float(np.clip(0.8 * local_spacing_nm, 15.0, 100.0))

    label_push_base_nm = 0.42 * local_spacing_nm  # how far out a label starts before the clearance search (pulled in from an earlier, too-generous 0.6x)
    label_half_width_nm = max(45.0, 0.30 * local_spacing_nm)  # candidate labels are a 9-line block -- taller than wide
    label_half_height_nm = max(65.0, 0.48 * local_spacing_nm)
    source_label_half_width_nm, source_label_half_height_nm = 55.0, 20.0  # "source N" is a single short line

    # Step size scales with local_spacing_nm (not a fixed nm constant) so ring growth stays
    # proportionate whether the array is tightly or loosely packed; max_rings is a small, fixed
    # search budget because _push_label_clear_of_devices now fans out over several angles per ring
    # (see that function's own docstring) instead of relying on radius growth alone to escape a
    # blocked row -- that fan is what keeps typical displacement small, which is what keeps this
    # plot's zoom tight.
    push_step_nm = max(8.0, 0.10 * local_spacing_nm)
    push_max_rings = 11


    obstacle_centers = np.asarray(centers_nm, dtype=float)
    obstacle_radii_nm = np.array([_device_length_nm(array, idx) / 2.0 + label_device_buffer_nm
                                   for idx in range(len(centers_nm))])
    # The source device and every candidate device also have quiver arrows drawn out from their own
    # centers (up to arrow_length_nm / mag_arrow_length_nm long) -- widen just THEIR exclusion radius
    # so a label can never land inside its own device's arrow, in addition to not landing on a device.
    arrow_reach_nm = max(arrow_length_nm, mag_arrow_length_nm)
    own_arrow_devices = set(candidate_idxs) | {source_idx}
    for idx in own_arrow_devices:
        obstacle_radii_nm[idx] = max(obstacle_radii_nm[idx], _device_length_nm(array, idx) / 2.0 + arrow_reach_nm + label_device_buffer_nm)

    # Source label: push into the widest open angular gap between the candidates, as seen from the
    # source -- not "opposite the mean direction toward them," which cancels toward zero (and falls
    # back to an arbitrary direction) once candidates roughly surround the source, as they often do
    # once a source has many neighbors. See _widest_angular_gap_direction's own docstring.
    vecs_to_candidates = candidate_positions - np.array([sx, sy])
    source_dir = _widest_angular_gap_direction(vecs_to_candidates)
    source_label_x, source_label_y = _push_label_clear_of_devices(
        (sx, sy), source_dir, label_push_base_nm, obstacle_centers, obstacle_radii_nm,
        source_label_half_width_nm, source_label_half_height_nm, step_nm=push_step_nm, max_rings=push_max_rings)

    # Candidate labels: push radially outward from the local source+candidates cluster center, so
    # each label moves away from the other devices in ITS OWN local neighborhood rather than along
    # one direction shared by every candidate -- this is what actually generalizes across rotation.
    candidate_label_positions: List[Tuple[float, float]] = []
    for cx, cy in candidate_positions:
        direction = _unit_vector(float(cx) - cluster_center[0], float(cy) - cluster_center[1])
        lx, ly = _push_label_clear_of_devices((float(cx), float(cy)), direction, label_push_base_nm,
                                               obstacle_centers, obstacle_radii_nm,
                                               label_half_width_nm, label_half_height_nm,
                                               step_nm=push_step_nm, max_rings=push_max_rings)
        candidate_label_positions.append((lx, ly))

    # ---- PASS 2: draw everything, now that every label's final position is known ----
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    if show_full_array_background:
        for idx in range(len(centers_nm)):
            _draw_island_segment(ax, float(centers_nm[idx][0]), float(centers_nm[idx][1]), float(geom_angles_deg[idx]),
                                  _device_length_nm(array, idx), color="0.86", linewidth=4.0, alpha=0.9, zorder=1)

    _draw_island_segment(ax, sx, sy, float(geom_angles_deg[source_idx]),
                          _device_length_nm(array, source_idx), color="orange", linewidth=8.0, alpha=1.0, zorder=4)
    ax.scatter([sx], [sy], s=120, color="orange", edgecolor="black", zorder=5)
    ax.plot([sx, source_label_x], [sy, source_label_y], linewidth=0.6, color="0.6", alpha=0.7, zorder=5)
    ax.text(source_label_x, source_label_y, f"source {source_idx}", ha="center", va="center",
            fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.85), zorder=6)

    src_theta = np.deg2rad(float(mag_angles_deg[source_idx]))
    ax.quiver([sx], [sy], [mag_arrow_length_nm * np.cos(src_theta)], [mag_arrow_length_nm * np.sin(src_theta)],
              angles="xy", scale_units="xy", scale=1, color="black", width=0.007,
              headwidth=4.2, headlength=5.8, headaxislength=5.0, zorder=8)

    # Bounding-box corners (not just anchor points) go into the plot-limit calculation below, so a
    # label's actual drawn extent -- not just its center point -- is guaranteed to stay on-plot.
    xs_for_lim = [sx, source_label_x - source_label_half_width_nm, source_label_x + source_label_half_width_nm]
    ys_for_lim = [sy, source_label_y - source_label_half_height_nm, source_label_y + source_label_half_height_nm]
    for row, (cx, cy), (lx, ly) in zip(results, candidate_positions, candidate_label_positions):
        cand_idx = int(row["candidate_idx"])
        slot_label = str(row.get("slot_label", ""))
        cx, cy = float(cx), float(cy)
        gx = float(geom_angles_deg[cand_idx])
        mdeg = float(row.get("magnetization_angle_deg", mag_angles_deg[cand_idx]))
        hx, hy = float(row["Hdemag_x_Oe"]), float(row["Hdemag_y_Oe"])
        hmag = float(row["Hdemag_mag_Oe"])
        hrev = float(row["H_reverse_Oe"])
        hdrive = float(row.get("H_switch_drive_Oe", -hrev))
        hc = float(row["Hc_Oe"])
        field_angle_deg = float(row["field_angle_deg"]) if str(row.get("field_angle_deg", "")).strip() not in ("", "nan", "None") else np.nan
        switch = str(row["switch"]).strip().upper() == "YES"

        xs_for_lim.extend([cx, lx - label_half_width_nm, lx + label_half_width_nm])
        ys_for_lim.extend([cy, ly - label_half_height_nm, ly + label_half_height_nm])

        if cand_idx in accepted_indices:
            island_color = "darkorchid"
        elif switch:
            island_color = "green"
        else:
            island_color = "tomato"

        _draw_island_segment(ax, cx, cy, gx, _device_length_nm(array, cand_idx),
                              color=island_color, linewidth=7.0, alpha=1.0, zorder=6)
        ax.scatter([cx], [cy], s=95, color=island_color, edgecolor="black", zorder=7)

        mtheta = np.deg2rad(mdeg)
        ax.quiver([cx], [cy], [mag_arrow_length_nm * np.cos(mtheta)], [mag_arrow_length_nm * np.sin(mtheta)],
                  angles="xy", scale_units="xy", scale=1, color="black", width=0.007,
                  headwidth=4.2, headlength=5.8, headaxislength=5.0, zorder=8)

        if np.isfinite(hmag) and hmag > 0:
            ux, uy = (hx / hmag) * arrow_length_nm, (hy / hmag) * arrow_length_nm
            ax.quiver([cx], [cy], [ux], [uy], angles="xy", scale_units="xy", scale=1, color="tab:blue",
                      width=0.007, headwidth=4.2, headlength=5.8, headaxislength=5.0, zorder=8)

        blue_angle_str = f"{field_angle_deg:.1f}\u00b0" if np.isfinite(field_angle_deg) else "nan"
        label_lines = [
            # f"{slot_label} idx {cand_idx}" if slot_label else f"idx {cand_idx}",
            f"idx {cand_idx}",
            # f"|H|={hmag:.1f} Oe", f"H ang={blue_angle_str}", f"m ang={mdeg:.1f}\u00b0", f"EA={gx:.1f}\u00b0",
            f"|H|={hmag:.1f} Oe", f"H ang={blue_angle_str}", f"m ang={mdeg:.1f}\u00b0",
            # f"Hrev={hrev:.1f}", f"drive={hdrive:.1f}", f"Hc={hc:.1f}", f"{'YES' if switch else 'NO'}",
            f"Hrev={hrev:.1f}", f"Hc={hc:.1f}", f"{'YES' if switch else 'NO'}",
        ]
        ax.plot([cx, lx], [cy, ly], linewidth=0.6, color="0.6", alpha=0.7, zorder=8)
        ax.text(lx, ly, "\n".join(label_lines), ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.92), zorder=9)

    for row in results:
        cand_idx = int(row["candidate_idx"])
        cx, cy = centers_nm[cand_idx]
        ax.plot([sx, cx], [sy, cy], linestyle="--", linewidth=1.0, color="0.55", alpha=0.8, zorder=2)

    edge_margin_nm = margin_nm  # small breathing-room margin now that box corners (above) already set the real extent
    xmin, xmax = min(xs_for_lim) - edge_margin_nm, max(xs_for_lim) + edge_margin_nm
    ymin, ymax = min(ys_for_lim) - edge_margin_nm, max(ys_for_lim) + edge_margin_nm
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x position (nm)")
    ax.set_ylabel("y position (nm)")

    title_suffix = "no accepted flip" if not accepted_indices else f"accepted candidate(s) = {sorted(accepted_indices)}"
    ax.set_title(f"Nearest-neighbor field comparison | step {step} | source island {source_idx}\n"
                 f"Blue arrow = H_demag direction excluding the candidate itself; black arrow = magnetization; "
                 f"text = |H_demag|, blue-arrow angle, Hrev, switch drive | {title_suffix}")

    # Color-code legend lives OUTSIDE the data axes (figure coordinates, below the plot) rather than
    # inset at a fixed data-axes corner -- an inset legend sits at a fixed screen position regardless
    # of where labels get pushed, so it could still collide with a label in a tightly packed cluster.
    # Living outside the axes entirely removes that failure mode structurally instead of needing yet
    # another collision check.
    fig.subplots_adjust(bottom=0.20)
    fig.text(0.5, 0.03,
          "orange = source   |   purple = strongest accepted candidate   |   green = switch=YES   |   red = switch=NO\n"
          "blue arrow = field direction   |   black arrow = magnetization direction",
              ha="center", va="bottom", fontsize=8,
              bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.96))

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    if verbose:
        print(f"[DEBUG] plot_neighbor_field_comparison: saved {output_path}")


# --------------------------
# ---- Visualization 2: whole-array flip-path map / animation / GIF ----
def _stepwise_up_to(stepwise_history: List[List[int]], max_generation: Optional[int]) -> List[List[int]]:
    return stepwise_history if max_generation is None else stepwise_history[:max_generation + 1]


def _edges_up_to(accepted_edges: List[Dict], max_generation: Optional[int]) -> List[Dict]:
    if max_generation is None:
        return accepted_edges
    max_step = max_generation - 1
    return [] if max_step < 0 else [e for e in accepted_edges if int(e["step"]) <= max_step]


def _candidate_rows_up_to(comparison_rows: List[Dict], max_generation: Optional[int]) -> List[Dict]:
    if max_generation is None:
        return comparison_rows
    max_step = max_generation - 1
    if max_step < 0:
        return []
    return [r for r in comparison_rows if int(r["step"]) <= max_step]


def load_accepted_edges(comparison_rows: List[Dict]) -> List[Dict]:
    """
    GOAL:
        Extract accepted source -> candidate propagation edges from a
        propagation run's comparison rows.

    HOW IT TALKS TO OTHER FILES:
        Called by generate_flip_path_visualizations (this file), reading
        rows in the exact shape propagation_engine.run_propagation writes
        to its own comparison_csv.

    PARAMETERS:
        comparison_rows (List[Dict]): every row from propagation_engine's
            comparison_csv (as dicts -- for example, via csv.DictReader).

    RETURNS:
        List[Dict]: one entry per accepted edge: {"step", "source_idx",
        "candidate_idx", "source_origins", "collapsed_origins",
        "collapsed_with_sources"}.
    """
    edges: List[Dict] = []
    for r in comparison_rows:
        if str(r.get("accepted_unique_flip", "")).strip().upper() != "YES":
            continue
        edges.append({
            "step": int(float(r["step"])), "source_idx": int(float(r["source_idx"])), "candidate_idx": int(float(r["candidate_idx"])),
            "source_origins": str(r.get("source_origins", "")), "collapsed_origins": str(r.get("collapsed_origins", "")),
            "collapsed_with_sources": str(r.get("collapsed_with_sources", "")),
        })
    return edges


def plot_flip_path(
    centers_nm: np.ndarray,  # Nx2, every device's own center, nm (geometry -- constant across a run)
    geom_angles_deg: np.ndarray,  # N, every device's own geometry angle, degrees (geometry -- constant across a run)
    device_length_nm_per_device: np.ndarray,  # N, every device's own length, nm (see this file's module-level "WHAT CHANGED" section)
    initial_mag_angles_deg: np.ndarray, # initial mag arrows
    stepwise_history: List[List[int]],  # [[initial injection(s)], [flips accepted step 0], [flips accepted step 1], ...]
    accepted_edges: List[Dict],  # from load_accepted_edges
    comparison_rows: List[Dict],  # every row from the comparison CSV (candidate arrows, faint)
    output_path: str,
    title: str = "ASI multi-injection flip paths",
    show_candidates: bool = True,
    show_accepted_arrows: bool = True,
    max_generation_to_show: Optional[int] = None,  # None = every generation; 0 = initial injections only; N = cumulative through generation N (used by generate_flip_path_visualizations to build animation frames)
    dpi: int = 250,
) -> None:
    """
    GOAL:
        Draw the ASI array layout plus cumulative flip-propagation paths,
        colored by generation, with faint candidate arrows and thick
        accepted-flip arrows.

    HOW IT TALKS TO OTHER FILES:
        Called by generate_flip_path_visualizations (this file), once for
        the static full-history plot and once per generation for the
        animation frames.

    PARAMETERS:
        See the inline comments on each parameter above.

    RETURNS:
        None. Writes output_path (parent directories created if needed).
    """
    history_to_plot = _stepwise_up_to(stepwise_history, max_generation_to_show)
    accepted_edges_to_plot = _edges_up_to(accepted_edges, max_generation_to_show)
    candidate_rows_to_plot = _candidate_rows_up_to(comparison_rows, max_generation_to_show)

    # ---- Reconstruct THIS frame's magnetization state ----
    # initial_mag_angles_deg is the TRUE ground state, captured before any
    # injection or propagation flip (see propagation_engine.py's own capture
    # point). Build this frame's displayed state by applying a 180deg flip,
    # starting from that ground state, once per occurrence of a device in
    # history_to_plot (cumulative through this frame) -- naturally handles
    # reflips too, since a reflipped device just appears twice and nets out
    # to its original orientation.
    display_mag_angles_deg = np.array(initial_mag_angles_deg, dtype=float).copy()
    for generation in history_to_plot:
        for idx in generation:
            display_mag_angles_deg[int(idx)] = (display_mag_angles_deg[int(idx)] + 180.0) % 360.0

    fig, ax = plt.subplots(figsize=(9, 9))


    for idx in range(len(centers_nm)):
        _draw_island_segment(ax, float(centers_nm[idx][0]), float(centers_nm[idx][1]), float(geom_angles_deg[idx]),
                              float(device_length_nm_per_device[idx]), color="0.82",
                              linewidth=max(3.0, float(device_length_nm_per_device[idx]) / 14), alpha=0.8, zorder=1)
    # magnetization arrows
    mag_arrow_len = float(np.median(device_length_nm_per_device)) * 0.6
    mtheta = np.deg2rad(display_mag_angles_deg)
    ax.quiver(centers_nm[:, 0], centers_nm[:, 1],
            mag_arrow_len * np.cos(mtheta), mag_arrow_len * np.sin(mtheta),
            angles="xy", scale_units="xy", scale=1, color="black", width=0.004,
            headwidth=4, headlength=5, headaxislength=4.5, alpha=0.75, zorder=10)
    
    if show_candidates and candidate_rows_to_plot:
        for r in candidate_rows_to_plot:
            src, cand = int(float(r.get("source_idx", -1))), int(float(r.get("candidate_idx", -1)))
            if src < 0 or cand < 0 or src >= len(centers_nm) or cand >= len(centers_nm):
                continue
            x0, y0 = centers_nm[src]
            x1, y1 = centers_nm[cand]
            switch = str(r.get("switch", "")).strip().upper() == "YES"
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", linestyle=":", linewidth=1.0,
                                        color="tab:green" if switch else "tab:red", alpha=0.22, shrinkA=12, shrinkB=12),
                        zorder=2)

    if history_to_plot:
        cmap = plt.get_cmap("cool")
        denom = max(1, len(history_to_plot) - 1)
        for generation_idx, generation in enumerate(history_to_plot):
            for local_idx, idx in enumerate(generation):
                idx = int(idx)
                if idx < 0 or idx >= len(centers_nm):
                    continue
                x, y = centers_nm[idx]
                color = cmap(generation_idx / denom)
                _draw_island_segment(ax, float(x), float(y), float(geom_angles_deg[idx]), float(device_length_nm_per_device[idx]),
                                      color=color, linewidth=max(6.0, float(device_length_nm_per_device[idx]) / 8), alpha=1.0, zorder=5)
                ax.scatter([x], [y], s=170, color=color, edgecolor="black", zorder=6)
                # if generation_idx == 0:
                #     label = f"I{local_idx + 1}" if len(generation) > 1 else "I"
                # else:
                #     label = f"{generation_idx}.{local_idx + 1}" if len(generation) > 1 else str(generation_idx)
                # ax.text(x, y, label, ha="center", va="center", fontsize=7.5, fontweight="bold", color="white", zorder=7)

    if show_accepted_arrows:
        for e in accepted_edges_to_plot:
            src, cand = int(e["source_idx"]), int(e["candidate_idx"])
            if src >= len(centers_nm) or cand >= len(centers_nm):
                continue
            x0, y0 = centers_nm[src]
            x1, y1 = centers_nm[cand]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", linewidth=2.6, color="black", alpha=0.92, shrinkA=16, shrinkB=16),
                        zorder=8)

    margin = float(np.max(device_length_nm_per_device)) * 2.3 if len(device_length_nm_per_device) else 160.0
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(np.nanmin(centers_nm[:, 0]) - margin, np.nanmax(centers_nm[:, 0]) + margin)
    ax.set_ylim(np.nanmin(centers_nm[:, 1]) - margin, np.nanmax(centers_nm[:, 1]) + margin)
    ax.set_xlabel("x position (nm)")
    ax.set_ylabel("y position (nm)")

    flat_shown = [x for gen in history_to_plot for x in gen]
    if max_generation_to_show is None:
        subtitle = f"{len(flat_shown)} flipped island(s) in {len(history_to_plot)} synchronous generation(s)"
    elif max_generation_to_show == 0:
        subtitle = "Initial injection generation"
    else:
        subtitle = f"Cumulative through propagation generation {max_generation_to_show}"
    ax.set_title(f"{title}\n{subtitle}")

    ax.plot([], [], color="black", linewidth=2.6, label="Accepted source \u2192 flip")
    if show_candidates and comparison_rows:
        ax.plot([], [], color="tab:green", linestyle=":", linewidth=1.1, label="Candidate above Hc")
        ax.plot([], [], color="tab:red", linestyle=":", linewidth=1.1, label="Candidate below Hc")
    ax.plot([], [], marker="o", color="none", markerfacecolor="gray", markeredgecolor="black", label="Flipped generation")
    ax.legend(loc="upper right", frameon=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _try_make_gif(frame_paths: List[str], gif_path: str, duration_ms: int = 800) -> Tuple[bool, str]:
    """
    GOAL:
        Assemble the per-generation frame PNGs into an animated GIF, if
        Pillow is available and there are at least 2 frames.

    PARAMETERS:
        frame_paths (List[str]): frame PNGs, in generation order.
        gif_path (str): output GIF path.
        duration_ms (int): per-frame display duration.

    RETURNS:
        Tuple[bool, str]: (True, "saved <path>") on success; (False,
        reason) otherwise (fewer than 2 frames, or Pillow not installed).
    """
    if len(frame_paths) < 2:
        return False, "not enough frames for GIF"
    try:
        from PIL import Image
    except Exception:
        return False, "Pillow is not installed"
    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
    os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return True, f"saved {gif_path}"


def generate_flip_path_visualizations(
    array,  # geometry.DeviceArray -- final state (geometry only is read; see this file's module-level LIMITATIONS section)
    result: Dict,  # propagation_engine.run_propagation's own return dict
    output_dir: str,
    show_candidates: bool = True,
    show_accepted_arrows: bool = True,
    make_gif: bool = True,
    verbose: bool = False,
) -> Dict[str, str]:
    """
    GOAL:
        Produce the complete flip-path visualization set for one
        propagation run: a static cumulative PNG, one PNG per generation,
        and (if Pillow is available) an animated GIF.

    HOW IT TALKS TO OTHER FILES:
        Called once by mumax3_parent_propagation.run_propagation after
        propagation_engine.run_propagation returns (or directly by a
        caller holding that same result dict). Reads result["comparison_
        csv"] via csv.DictReader, and result["flip_history_by_step"]/
        result["array"] directly. Calls load_accepted_edges and
        plot_flip_path (both this file).

    PARAMETERS:
        array (geometry.DeviceArray): the array (final state) -- only its
            geometry (device_centers/device_angles/per-device length) is
            read; magnetization state is irrelevant here (see this file's
            module-level LIMITATIONS section).
        result (Dict): propagation_engine.run_propagation's own return
            value.
        output_dir (str): base directory for this visualization set --
            typically alongside the run's other propagation output.
        show_candidates, show_accepted_arrows: forwarded to plot_flip_path.
        make_gif (bool): attempt to assemble the per-generation frames into
            a GIF when True.
        verbose (bool): prints each saved file's path when True.

    RETURNS:
        Dict[str, str]: {"static": <path>, "frames_dir": <path>, "gif":
        <path or "">, "gif_status": <str>}.
    """
    import csv

    
    centers_nm, geom_angles_deg, _final_mag_angles_deg = get_device_mesh(array)  # magnetization comes from result instead
    device_length_nm_per_device = np.array([_device_length_nm(array, i) for i in range(len(centers_nm))])
    initial_mag_angles_deg = result["initial_mag_angles_deg"]
    
    with open(result["comparison_csv"], newline="", encoding="utf-8") as f:
        comparison_rows = list(csv.DictReader(f))
    accepted_edges = load_accepted_edges(comparison_rows)
    stepwise_history = result["flip_history_by_step"]

    static_path = os.path.join(output_dir, "flip_path_static.png")
    frame_dir = os.path.join(output_dir, "flip_path_frames")
    gif_path = os.path.join(output_dir, "flip_path.gif")

    plot_flip_path(centers_nm, geom_angles_deg, device_length_nm_per_device, initial_mag_angles_deg, stepwise_history, accepted_edges,
                    comparison_rows, static_path, show_candidates=show_candidates, show_accepted_arrows=show_accepted_arrows)
    if verbose:
        print(f"[INFO] generate_flip_path_visualizations: saved {static_path}")

    frame_paths: List[str] = []
    for generation in range(len(stepwise_history)):
        frame_path = os.path.join(frame_dir, f"frame_{generation:03d}.png")
        plot_flip_path(centers_nm, geom_angles_deg, device_length_nm_per_device, initial_mag_angles_deg, stepwise_history, accepted_edges,
                        comparison_rows, frame_path, show_candidates=show_candidates, show_accepted_arrows=show_accepted_arrows,
                        max_generation_to_show=generation)
        frame_paths.append(frame_path)
    if verbose:
        print(f"[INFO] generate_flip_path_visualizations: saved {len(frame_paths)} frame(s) under {frame_dir}")

    gif_status = "GIF skipped (make_gif=False)"
    gif_out = ""
    if make_gif:
        ok, gif_status = _try_make_gif(frame_paths, gif_path)
        gif_out = gif_path if ok else ""
        if verbose:
            print(f"[INFO] generate_flip_path_visualizations: {gif_status}")

    return {"static": static_path, "frames_dir": frame_dir, "gif": gif_out, "gif_status": gif_status}
