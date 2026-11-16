r"""
File Name: propagation_utilities.py
Author: Starr Feng
Date: 2026.08.10
Revision: v1

============================================================
PURPOSE
============================================================
Small, non-physics bookkeeping helpers the propagation engine needs but
that are not themselves part of the switching physics (see mumax3_csv_stats.py
for that) or the geometry-general candidate search (see
propagation_neighbors.py): reading a DeviceArray's current state, flipping
one device, and tracking multi-injection propagation-front "origin sets"
(which initial injection(s) a given active source's propagation path
traces back to). Starr's original utility functions for the propagation engine.

============================================================
LIMITATIONS
============================================================
- get_device_mesh below REPLACES the original script's _current_mesh
  helper (which called DeviceArray.print_summary() under a stdout
  redirect purely to obtain print_summary()'s freshly-rebuilt
  magnetization-angle list -- see that method's own "magnetization_angles"
  return key, which reads self.device_data fresh rather than the
  potentially-stale self.device_m_angles attribute). get_device_mesh reads
  the same self.device_data-backed source directly, without going through
  print_summary or any stdout redirection, since DeviceArray's own
  attributes (device_centers, device_angles) are already directly
  readable, and only the magnetization angle needs the device_data-backed
  read (see that function's own documentation for why).
"""

from typing import Dict, Iterable, List, Optional, Tuple  # type hints used throughout this file's signatures

import numpy as np  # array construction used throughout this file


# --------------------------
# ---- Fresh read of a DeviceArray's current per-device state ----
def get_device_mesh(array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GOAL:
        Return this array's current device centers (nm), geometry angles
        (degrees), and magnetization angles (degrees), always freshly
        read -- see this file's module-level LIMITATIONS section for why
        magnetization angle specifically needs a device_data-backed read
        rather than array.device_m_angles directly.

    HOW IT TALKS TO OTHER FILES:
        Reads geometry.DeviceArray's own device_centers/device_angles/
        device_data attributes directly (no method call into geometry.py
        beyond plain attribute access). Called by the propagation engine
        every time it needs an up-to-date snapshot -- in particular, right
        after any set_device_magnetization call (flip_device, below, or the
        engine's own candidate-exclusion nan trick), since
        set_device_magnetization intentionally does NOT keep
        array.device_m_angles in sync (see that method's own LIMITATIONS
        section in geometry.py).

    PARAMETERS:
        array (geometry.DeviceArray): the live array to read.

    RETURNS:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            centers_nm: Nx2, device centers in nm.
            geom_angles_deg: N, geometry (easy-axis) angle per device.
            mag_angles_deg: N, CURRENT magnetization angle per device,
                read from array.device_data (not array.device_m_angles),
                so it reflects any set_device_magnetization call made since
                the array was last fully rebuilt.
    """
    n = len(array.device_centers)
    centers_nm = array.device_centers * 1e9  # meters -> nm
    geom_angles_deg = np.asarray(array.device_angles, dtype=float)
    mag_angles_deg = np.array([array.device_data[i]["m_angle"] for i in range(n)], dtype=float)  # device_data-backed, not device_m_angles -- see this function's own documentation
    return centers_nm, geom_angles_deg, mag_angles_deg


# --------------------------
# ---- Flip one device's magnetization by 180 degrees ----
def flip_device(array, index: int, verbose: bool = False) -> Tuple[float, float]:
    """
    GOAL:
        Flip one device's magnetization by 180 degrees -- the propagation
        engine's actual "this island switched" action.

    HOW IT TALKS TO OTHER FILES:
        Calls array.set_device_magnetization (geometry.py). Called by the
        propagation engine once per accepted unique flip, per step, and
        once per injection at step 0.

    PARAMETERS:
        array (geometry.DeviceArray): the live array to modify.
        index (int): device index to flip.
        verbose (bool): prints the old/new angle when True.

    RETURNS:
        Tuple[float, float]: (old_angle_deg, new_angle_deg).
    """
    _, _, mag_angles_deg = get_device_mesh(array)
    old_angle = float(mag_angles_deg[index])
    new_angle = (old_angle + 180.0) % 360.0
    array.set_device_magnetization(index=index, angle=new_angle, verbose=verbose)
    if verbose:
        print(f"[DEBUG] flip_device: device {index} m_angle {old_angle:.1f} -> {new_angle:.1f} deg.")
    return old_angle, new_angle


# --------------------------
# ---- Choose a default injection device ----
def find_center_device(centers_nm: np.ndarray) -> int:
    """
    GOAL:
        Choose the device closest to the array's own (centered) origin
        (0, 0) -- the default initial-injection device when the caller does
        not specify one explicitly.

    HOW IT TALKS TO OTHER FILES:
        Called by resolve_injection_indices (this file) when neither an
        explicit injection index nor index list is given.

    PARAMETERS:
        centers_nm (np.ndarray): Nx2, every device's center, nm.

    RETURNS:
        int: the device index closest to (0, 0).
    """
    return int(np.argmin(np.sum(centers_nm ** 2, axis=1)))


# --------------------------
# ---- Resolve one or more initial injection device indices ----
def resolve_injection_indices(
    centers_nm: np.ndarray,  # Nx2, every device's center, nm
    injection_indices: Optional[List[int]] = None,  # explicit multi-injection list; takes priority if given
    injection_index: Optional[int] = None,  # backward-compatible single-injection index; used only if injection_indices is None
) -> List[int]:
    """
    GOAL:
        Resolve one or more initial injection device indices, deduplicated
        while preserving order.

    HOW IT TALKS TO OTHER FILES:
        Called once by the propagation engine at run start. Calls
        find_center_device (this file) as the ultimate fallback.

    PARAMETERS:
        centers_nm (np.ndarray): Nx2, every device's center, nm.
        injection_indices (Optional[List[int]]): explicit list of injection
            device indices. If given (not None), used as-is (after
            validation/deduplication) and injection_index is ignored.
        injection_index (Optional[int]): single injection device index,
            used only if injection_indices is None. If also None, the
            device closest to the array's own origin is used (see
            find_center_device).

    RETURNS:
        List[int]: resolved, deduplicated (order-preserving) injection
        device indices -- always at least one entry.

    RAISES:
        ValueError if any resolved index is outside 0..len(centers_nm)-1,
        or if no indices were resolved at all (should not normally happen,
        since the find_center_device fallback always resolves exactly one).
    """
    if injection_indices is not None:
        raw_indices = [int(i) for i in injection_indices]
    elif injection_index is None:
        raw_indices = [find_center_device(centers_nm)]
    else:
        raw_indices = [int(injection_index)]

    resolved: List[int] = []
    seen = set()
    for idx in raw_indices:
        if idx < 0 or idx >= len(centers_nm):
            raise ValueError(f"Injection index {idx} is outside valid range 0..{len(centers_nm) - 1}")
        if idx not in seen:
            resolved.append(idx)
            seen.add(idx)

    if not resolved:  # should not normally happen -- guards against an explicitly empty injection_indices list
        raise ValueError("No injection indices were resolved.")
    return resolved


# --------------------------
# ---- Multi-injection propagation-front "origin set" bookkeeping ----
def origin_label_for_injection(idx: int) -> str:
    """
    GOAL:
        Build a stable, human-readable label for one initial injection
        device, used as the "origin" every propagation path downstream of
        it is tagged with.

    HOW IT TALKS TO OTHER FILES:
        Called by the propagation engine once per injection at step 0, to
        seed each injection device's own initial origin-set
        ({device_index: {origin_label}}).

    PARAMETERS:
        idx (int): injection device index.

    RETURNS:
        str: f"inj{idx:03d}".
    """
    return f"inj{int(idx):03d}"


def format_origin_set(origins: Iterable[str]) -> str:
    """
    GOAL:
        Format a set of origin labels into one sorted, "+"-joined string,
        for CSV/log display.

    HOW IT TALKS TO OTHER FILES:
        Called by the propagation engine anywhere an origin set needs to be
        written to a CSV column or printed.

    PARAMETERS:
        origins (Iterable[str]): origin labels (see origin_label_for_
            injection).

    RETURNS:
        str: sorted labels joined by "+", for example "inj000+inj007".
    """
    return "+".join(sorted(str(x) for x in origins))


def merge_origin_sets(origin_sets: Iterable[Iterable[str]]) -> set:
    """
    GOAL:
        Merge several origin sets into one -- used when two or more active
        propagation paths independently propose the same next device
        (a "collapse" event), so the resulting merged path is tagged with
        every contributing injection's origin.

    HOW IT TALKS TO OTHER FILES:
        Called by the propagation engine once per collapsed target device,
        per step.

    PARAMETERS:
        origin_sets (Iterable[Iterable[str]]): one origin set per
            proposing source device.

    RETURNS:
        set: the union of every given origin set.
    """
    merged: set = set()
    for origin_set in origin_sets:
        merged.update(str(x) for x in origin_set)
    return merged
