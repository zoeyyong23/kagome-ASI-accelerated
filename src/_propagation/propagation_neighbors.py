r"""
File Name: propagation_neighbors.py
Author: Connor Sullivan, Starr Feng
Date: 2026.08.10
Revision: v1

============================================================
PURPOSE
============================================================
Answers one question for the propagation engine: "given a source device
index, which OTHER devices should be checked as flip candidates this
step?" -- for ANY ASI geometry DeviceArray can build (square, pinwheel,
kagome, or anything else), not only square.

============================================================
WHY THIS FILE EXISTS (replacing fixed_relative_neighbor_offsets_nm)
============================================================
The uploaded propagation script's candidate rule (fixed_relative_neighbor_
offsets_nm) hardcoded eight square-ASI-specific offsets: four diagonal
"vertex-neighbor" slots at (+/-nn_x/2, +/-nn_y/2), and four same-sublattice
slots at (+/-nn_x, 0)/(0, +/-nn_y). That is correct ONLY for a square
lattice with nn_x==nn_y and two perpendicular sublattices -- it does not
describe kagome's honeycomb-vertex-sharing neighbor structure at all (a
kagome island's real nearest neighbors are not at any of those eight
offsets), and it silently assumes a specific spacing/sublattice-count
relationship pinwheel/kagome do not share.

geometry.DeviceArray.get_neighbor_vectors already solves this generally:
called with a specific reference_index, it measures every OTHER device's
actual distance from that reference device and groups them into shells by
increasing distance -- a purely geometric measurement with no assumption
about how many sublattices exist or what angle they sit at. This file is a
thin wrapper around that existing method: find_candidate_neighbors below
calls get_neighbor_vectors(reference_index=source_idx, n_shells=...) and
resolves each returned vector back to the device index actually sitting
there (see _device_at_offset), which naturally omits any slot that does
not exist in the finite array (an edge/corner device's missing shell
positions simply are not present in get_neighbor_vectors' own output,
since it only reports vectors to devices that actually exist) -- the same
"omit missing slots, do not borrow a farther-away device to force a fixed
candidate count" behavior the original square-only rule aimed for, now
derived from the array's actual measured geometry instead of a hardcoded
square formula.

============================================================
LIMITATIONS
============================================================
- get_neighbor_vectors groups strictly by EXACT rounded-nm distance (see
  that method's own documentation) -- for kagome/pinwheel geometries where
  the true nearest-neighbor shell may itself contain two DIFFERENT exact
  distances that are physically "the same shell" in some other sense (for
  example, a slightly non-uniform kagome layout), those would be reported
  as two separate shells here rather than merged into one. This has not
  come up in the worked geometry examples in mumax3_parent_superposition.py
  (every one of them produces exact-integer-nm, cleanly separated shell
  distances -- see get_neighbor_vectors' own LIMITATIONS section) but is
  worth knowing if a future geometry's true near-neighbor shell is
  distance-degenerate only up to floating-point noise rather than exactly.
- Candidate shells are measured from EACH source device's own position
  (find_candidate_neighbors passes reference_index=source_idx every call),
  not from one shared "canonical interior device" template -- this is
  intentionally correct for edge/corner devices (their own true measured
  neighbor shells naturally have fewer members) but means this file
  performs the get_neighbor_vectors measurement freshly for every source
  device checked, rather than computing one offset template once and
  reusing it. This is not currently a performance concern at the array
  sizes this pipeline's worked geometry examples use (get_neighbor_vectors
  itself is a single vectorized numpy pass over all device centers).
"""

from typing import Dict, Iterable, List, Optional, Tuple  # type hints used throughout this file's signatures

import numpy as np  # array operations used throughout this file


# --------------------------
# ---- Resolve one displacement vector back to a device index ----
def _device_at_offset(
    centers_nm: np.ndarray,  # Nx2 array, every device's own center, nm
    source_center_nm: np.ndarray,  # (x, y), the source device's own center, nm
    offset_nm: Tuple[float, float],  # (dx, dy), nm, from get_neighbor_vectors
    excluded_indices: Iterable[int],  # device indices to never return (already-flipped/self)
    tolerance_nm: float,  # maximum center-to-target distance still counted as a match
) -> Optional[int]:
    """
    GOAL:
        Find the device index whose center sits at
        source_center_nm + offset_nm, within tolerance_nm.

    HOW IT TALKS TO OTHER FILES:
        Called by find_candidate_neighbors (this file), once per vector
        returned by geometry.DeviceArray.get_neighbor_vectors.

    PARAMETERS:
        centers_nm, source_center_nm, offset_nm, excluded_indices,
        tolerance_nm: see the inline comments above.

    RETURNS:
        Optional[int]: the matching device index, or None if no device
        sits at that target position within tolerance (should not normally
        happen when offset_nm itself came from get_neighbor_vectors called
        on the SAME array, since that vector was measured from an actual
        device's actual position -- this guard exists mainly for the
        excluded_indices case, where the nearest match is masked out and no
        other device happens to be close enough).

    LIMITATIONS:
        Returns the single closest match, not every device within
        tolerance -- if tolerance_nm is set larger than half the array's
        own minimum spacing, two distinct devices could tie; this pipeline
        does not currently need that case (tolerance_nm defaults to a tiny
        sub-nm value; see find_candidate_neighbors).
    """
    target = source_center_nm + np.asarray(offset_nm, dtype=float)
    d = np.linalg.norm(centers_nm - target, axis=1)
    for idx in excluded_indices:
        if 0 <= int(idx) < len(d):
            d[int(idx)] = np.inf
    best_idx = int(np.argmin(d))
    if np.isfinite(d[best_idx]) and d[best_idx] <= tolerance_nm:
        return best_idx
    return None


# --------------------------
# ---- Main entry point: geometry-general candidate discovery ----
def find_candidate_neighbors(
    array,  # geometry.DeviceArray instance -- ANY geometry it can build (square, pinwheel, kagome, ...)
    source_index: int,  # device index to find candidates for
    n_shells: int = 2,  # how many distance shells to check, outward from source_index -- see geometry.DeviceArray.get_neighbor_vectors's own n_shells parameter
    exclude: Optional[Iterable[int]] = None,  # device indices to never return as candidates (for example, already-flipped devices when ALLOW_REFLIP is False)
    tolerance_nm: float = 1e-6,  # maximum center-to-target distance still counted as a match when resolving a shell vector back to a device index
    verbose: bool = True,  # print the resolved candidate list and shell structure when True
) -> List[Tuple[str, int]]:
    """
    GOAL:
        Return this source device's flip candidates for the current
        propagation step, for ANY geometry `array` was built with -- the
        geometry-general replacement for the uploaded propagation script's
        square-only fixed_relative_neighbor_offsets_nm/fixed_relative_
        neighbor_slot_report. See this file's module-level PURPOSE section
        for why get_neighbor_vectors is the right general-geometry
        primitive for this.

    HOW IT TALKS TO OTHER FILES:
        Calls array.get_neighbor_vectors (geometry.py) to measure this
        source device's own real neighbor shells, then _device_at_offset
        (this file) to resolve each shell vector back to a device index.
        Called once per active source, per propagation step, by
        src/_propagation's propagation engine.

    PARAMETERS:
        array (geometry.DeviceArray): the live array to search.
        source_index (int): device index to find candidates for.
        n_shells (int): how many distance shells (nearest, next-nearest,
            ...) to include, per get_neighbor_vectors' own n_shells
            parameter. For a square/pinwheel array (2 perpendicular
            sublattices), n_shells=2 reproduces the original script's
            "8 candidates" behavior (4 diagonal nearest + 4 same-axis
            next-nearest) as an emergent result of the actual measured
            geometry, rather than a hardcoded formula. A kagome array's
            shells will generally contain a different number of vectors
            per shell, and possibly a different useful n_shells -- this is
            the intended behavior (see this file's module-level PURPOSE
            section), not a bug to fix by forcing a specific count.
        exclude (Optional[Iterable[int]]): device indices to never return
            (for example, the source device itself -- added automatically
            below -- and, when the propagation engine's ALLOW_REFLIP is
            False, every already-flipped device).
        tolerance_nm (float): forwarded to _device_at_offset.
        verbose (bool): prints the resolved candidate list (and any shell
            vectors that failed to resolve to a device, which should only
            happen for excluded indices -- see _device_at_offset's own
            LIMITATIONS) when True.

    RETURNS:
        List[Tuple[str, int]]: (slot_label, device_index) pairs, ordered by
        increasing shell distance, then by get_neighbor_vectors' own
        within-shell order. slot_label is a geometry-general, human-
        readable string of the form "shell{K}_v{n}_dx{dx}_dy{dy}" (K = shell
        index starting at 1, n = position within that shell, dx/dy = the
        integer-nm displacement) -- deliberately NOT a compass label like
        the original script's "NN_UL"/"NNN_U" (those are square-ASI-
        specific directional names that do not generalize to kagome's
        non-perpendicular sublattice directions).

    LIMITATIONS:
        See this file's module-level LIMITATIONS section.
    """
    exclude_set = set(int(i) for i in exclude) if exclude else set()
    exclude_set.add(int(source_index))  # a device is never its own candidate

    centers_nm = array.device_centers * 1e9  # every device's center, nm (array.device_centers is meters -- see geometry.DeviceArray's own unit convention)
    source_center_nm = centers_nm[int(source_index)]

    shells = array.get_neighbor_vectors(reference_index=int(source_index), n_shells=n_shells, verbose=verbose)  # [{"distance_nm":..., "vectors_nm":[[dx,dy],...]}, ...], purely geometric -- see geometry.py

    found: List[Tuple[str, int]] = []
    unresolved: List[str] = []

    for shell_rank, shell in enumerate(shells, start=1):  # shell_rank: 1 = nearest shell, 2 = next-nearest, ...
        for vec_rank, (dx, dy) in enumerate(shell["vectors_nm"], start=1):
            slot_label = f"shell{shell_rank}_v{vec_rank}_dx{dx}_dy{dy}"
            cand_idx = _device_at_offset(
                centers_nm=centers_nm, source_center_nm=source_center_nm,
                offset_nm=(dx, dy), excluded_indices=exclude_set, tolerance_nm=tolerance_nm,
            )
            if cand_idx is None:  # target slot outside the finite array, or excluded (already flipped) -- omit, do not substitute a farther device
                unresolved.append(f"{slot_label}@({source_center_nm[0] + dx:.0f},{source_center_nm[1] + dy:.0f})")
            else:
                found.append((slot_label, cand_idx))

    if verbose:
        print(f"[DEBUG] find_candidate_neighbors: source={source_index}, n_shells={n_shells}, "
              f"resolved {len(found)} candidate(s), {len(unresolved)} slot(s) missing/excluded.")
        if unresolved:
            print(f"[DEBUG] find_candidate_neighbors: missing/excluded slots = {unresolved}")

    return found


def candidate_slot_label(source_center_nm: np.ndarray, candidate_center_nm: np.ndarray) -> str:
    """
    GOAL:
        Build a compact, geometry-general human-readable label for a
        (source, candidate) pair, for CSV/plot display -- the replacement
        for the original square-only candidate_neighbor_key (which
        produced compass labels like "NN_UL" that assume perpendicular
        square-ASI sublattice directions).

    HOW IT TALKS TO OTHER FILES:
        Called by src/_propagation wherever a human-readable relative
        label is wanted for logging/plotting, as an alternative to (or
        alongside) the slot_label already returned by
        find_candidate_neighbors above.

    PARAMETERS:
        source_center_nm, candidate_center_nm: (x, y) nm.

    RETURNS:
        str: "dx{dx}_dy{dy}", the integer-nm displacement from source to
        candidate -- geometry-general by construction, since it is just the
        measured displacement rather than a named compass direction.
    """
    dx = int(round(float(candidate_center_nm[0] - source_center_nm[0])))
    dy = int(round(float(candidate_center_nm[1] - source_center_nm[1])))
    return f"dx{dx}_dy{dy}"
