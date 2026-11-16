r"""
File Name: superposition_operations.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v5

============================================================
PURPOSE
============================================================
Provides the field-rotation and canvas-composition logic behind ASI
superposition: taking one seed device's field, rotating it (both spatially
and vectorially) to match each device's own geometry and magnetization
angle, and adding every device's rotated field into the correct place on a
shared canvas. Contains:
    - rotate_field_spatial: rotates a field's SPATIAL footprint (where the
      device shape sits in the array) to match a device's geometry angle.
    - rotate_field_vectors: rotates a field's VECTOR components (the
      direction each cell's field points) to match a device's
      magnetization angle. Distinct from rotate_field_spatial -- a device
      can have its shape rotated one way and its magnetization pointed a
      different way.
    - compose_fields: places every device's spatially- and
      vectorially-rotated field onto one shared canvas, via superposition
      (summing overlapping contributions).
    - init_pool / compose_fields_task / run_superposition_multiprocessing:
      run compose_fields across many devices/fields/variants at once, in
      parallel, using a multiprocessing worker-initializer pattern (see
      init_pool's own documentation for why this differs from simply
      passing the device data as a normal function argument).

============================================================
LIMITATIONS
============================================================
- compose_fields' spatial rotation cache (`spatial_cache`) is keyed by
  (field_name, geometry_angle); its vector rotation cache (`vector_cache`)
  is keyed by magnetization_angle alone. Both caches are local to a single
  compose_fields call and do not persist across separate calls (each
  worker process, in the multiprocessing path below, gets its own fresh
  caches).
"""

import numpy as np  # array operations used throughout this file


# --------------------------
# ---- Rotate a field's spatial footprint to match a device's geometry angle ----
def rotate_field_spatial(mx, my, angle_deg, verbose=False):
    """
    GOAL:
        Rotate the SPATIAL footprint of a field (where the device shape
        sits within the array) by `angle_deg`, so it matches a device's
        geometry angle -- distinct from rotating the field's VECTOR
        components (see rotate_field_vectors below); a device's shape and
        its magnetization direction can be rotated independently.

    HOW IT TALKS TO OTHER FILES:
        Called by compose_fields (this file), once per unique (field_name,
        geometry_angle) combination across every device being placed on a
        canvas (see compose_fields' spatial_cache).

    PARAMETERS:
        mx, my (np.ndarray): the field's x/y components, before spatial
            rotation.
        angle_deg (float): rotation angle in degrees. Follows the same
            convention as every other angle in this pipeline: standard
            math/CCW-positive, 0 degrees = +x, matching geometry.py's
            device angles (which feed matplotlib.patches.Ellipse(angle=...),
            itself CCW-positive) and rotate_field_vectors below (which
            uses the standard CCW rotation matrix).
        verbose (bool): prints the rotation angle and array shape when
            True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray]: (mx_r, my_r), the spatially rotated
        x/y components, same shape as the input.

    LIMITATIONS:
        scipy.ndimage.rotate is clockwise-positive in this canvas's own
        (row index increases with y, column index increases with x)
        indexing -- verified empirically: rotate(arr, +90) moves a point
        from +x to -y, a clockwise turn, not the +y a CCW +90 would give.
        The angle passed to scipy's rotate is therefore negated here, to
        keep this function's positive-angle direction consistent with the
        rest of the codebase (see PARAMETERS above). Do not remove this
        negation without re-verifying scipy's convention against this
        canvas's row/col-to-y/x mapping.
    """
    from scipy.ndimage import rotate  # underlying spatial rotation implementation
    mx_r = rotate(mx, -angle_deg, reshape=False, order=1)  # negated angle -- see function LIMITATIONS; order=1 is linear interpolation
    my_r = rotate(my, -angle_deg, reshape=False, order=1)  # negated angle -- see function LIMITATIONS
    if verbose:
        print(f"[DEBUG] rotate_field_spatial: rotated by {angle_deg} degrees, shape {mx_r.shape}.")
    return mx_r, my_r


# --------------------------
# ---- Rotate a field's vector components to match a device's magnetization angle ----
def rotate_field_vectors(mx, my, angle_deg, verbose=False):
    """
    GOAL:
        Rotate the VECTOR components of a field (the direction each cell's
        field points) by `angle_deg`, so it matches a device's
        magnetization angle -- a standard in-plane 2D rotation of the
        (mx, my) components at every cell, independent of the field's
        spatial footprint (see rotate_field_spatial above).

    HOW IT TALKS TO OTHER FILES:
        Called by compose_fields (this file), once per unique
        magnetization_angle across every device being placed on a canvas
        (see compose_fields' vector_cache).

    PARAMETERS:
        mx, my (np.ndarray): the field's x/y components, after spatial
            rotation (see rotate_field_spatial), before vector rotation.
        angle_deg (float): rotation angle in degrees, same CCW-positive
            convention as rotate_field_spatial.
        verbose (bool): prints the rotation angle when True.

    RETURNS:
        Tuple[np.ndarray, np.ndarray]: (mx_final, my_final), the
        vectorially rotated x/y components, same shape as the input. A
        field's z component (if any) is unaffected by this in-plane
        rotation and is not touched by this function at all.

    LIMITATIONS:
        None beyond the standard 2D rotation this function performs.
    """
    theta = np.deg2rad(angle_deg)  # rotation angle, in radians, for use with cos/sin
    mx_final = (mx * np.cos(theta) - my * np.sin(theta))  # standard 2D rotation, x component
    my_final = (mx * np.sin(theta) + my * np.cos(theta))  # standard 2D rotation, y component
    if verbose:
        print(f"[DEBUG] rotate_field_vectors: rotated by {angle_deg} degrees.")
    return mx_final, my_final


# --------------------------
# ---- Place every device's rotated field onto one shared canvas via superposition ----
def compose_fields(
    device_centers,  # one (cx, cy) per device, in nm
    geometry_angles,  # one spatial (shape) rotation angle per device, degrees -- see rotate_field_spatial
    magnetization_angles,  # one vector (magnetization) rotation angle per device, degrees -- see rotate_field_vectors; NaN skips that device entirely
    base_field,  # dict: {"x": ..., "y": ...} -- the single seed device's field, before any rotation
    field_name,  # name of the field being composed (for example, "m_full"), used only for cache keys and status messages
    cell_size_x,  # nm per grid cell, x direction
    cell_size_y,  # nm per grid cell, y direction
    margin_nm=0,  # extra canvas margin beyond the outermost device centers, in nm
    verbose=False,  # print per-device placement status when True
):
    """
    GOAL:
        Build one shared canvas containing every device's contribution to
        `field_name`, by spatially rotating (rotate_field_spatial) and
        vectorially rotating (rotate_field_vectors) the seed field for
        each device's own geometry/magnetization angle, then adding
        (superposing) each device's rotated field into its correct
        position on the canvas.

    HOW IT TALKS TO OTHER FILES:
        Called by compose_fields_task (this file), once per (region,
        base field, variant) combination, typically from within a worker
        process dispatched by run_superposition_multiprocessing. Calls
        rotate_field_spatial and rotate_field_vectors (this file).

    PARAMETERS:
        See the inline comments on each parameter above.
        verbose (bool): prints a status line at the start and end of
            composition, and one line per device placed (or skipped) in
            between, when True. Off by default, since this prints once per
            device and an array with many devices would otherwise produce
            excessive console output.

    RETURNS:
        Dict: {"x": canvas_x, "y": canvas_y, "extent": (xmin, xmax, ymin,
        ymax)} -- the composed canvas, cropped back down to the original
        seed simulation's own footprint (centered on (0, 0), the
        DeviceArray convention), and the physical extent that crop
        corresponds to.

    LIMITATIONS:
        A device whose magnetization_angle is NaN is skipped entirely --
        this can be intentional (for example, to exclude one device's
        contribution from a stray-field sweep) rather than a data error,
        so it is reported at the WARNING level rather than treated as a
        failure. A device whose field does not overlap the canvas at all
        after placement (fully clipped away) is also skipped, with its own
        status message.
    """
    if verbose:
        print(f"[INFO] compose_fields: composing field '{field_name}' onto canvas with superposition...")

    mx0 = base_field["x"]  # the seed device's own x component, before any rotation
    my0 = base_field["y"]  # the seed device's own y component, before any rotation

    ny, nx = mx0.shape  # seed field's own grid dimensions
    sim_size_x = nx * cell_size_x  # seed field's own physical width, nm
    sim_size_y = ny * cell_size_y  # seed field's own physical height, nm

    centers_x = np.array([c[0] for c in device_centers])  # every device's center, x
    centers_y = np.array([c[1] for c in device_centers])  # every device's center, y

    # ---- Canvas bounds: large enough to contain every device's own full seed-field
    # footprint (not just its center point), plus the requested margin. ----
    xmin = int(centers_x.min() - sim_size_x / 2 - margin_nm)  # canvas left edge, nm
    xmax = int(centers_x.max() + sim_size_x / 2 + margin_nm)  # canvas right edge, nm
    ymin = int(centers_y.min() - sim_size_y / 2 - margin_nm)  # canvas bottom edge, nm
    ymax = int(centers_y.max() + sim_size_y / 2 + margin_nm)  # canvas top edge, nm

    canvas_nx = int(np.ceil((xmax - xmin) / cell_size_x))  # canvas width, cells
    canvas_ny = int(np.ceil((ymax - ymin) / cell_size_y))  # canvas height, cells

    canvas_x = np.zeros((canvas_ny, canvas_nx), dtype=float)  # the shared canvas, x component, accumulated below
    canvas_y = np.zeros((canvas_ny, canvas_nx), dtype=float)  # the shared canvas, y component, accumulated below

    spatial_cache = {}  # rotated (spatial) fields, keyed by (field_name, geometry_angle), to avoid redundant rotations across devices sharing an angle
    vector_cache = {}  # rotated (vector) unit rotation factors, keyed by magnetization_angle, to avoid redundant trig calls across devices sharing an angle
    for i, (center, g_angle, m_angle) in enumerate(zip(device_centers, geometry_angles, magnetization_angles)):  # place one device at a time
        if np.isnan(m_angle):  # this device is intentionally excluded from the superposition
            print(f"[WARNING] compose_fields: skipping device {i} due to NaN magnetization angle "
                  "(can be intentional, e.g. to exclude a device from a stray-field sweep).")
            continue
        cx, cy = center  # this device's center, nm

        # ---- Spatial rotation (cached per unique geometry angle) ----
        key = (field_name, g_angle)  # cache key for this (field, geometry angle) combination
        if key not in spatial_cache:  # first device seen at this geometry angle: compute and cache its rotated field
            spatial_cache[key] = rotate_field_spatial(mx0, my0, g_angle, verbose=verbose)
        mx_s, my_s = spatial_cache[key]  # this device's spatially rotated field

        # ---- Vector rotation (cached per unique magnetization angle) ----
        if m_angle not in vector_cache:  # first device seen at this magnetization angle: compute and cache its rotation factors
            theta = np.deg2rad(m_angle)  # magnetization angle, in radians
            vector_cache[m_angle] = (np.cos(theta), np.sin(theta))
        c, s = vector_cache[m_angle]  # this device's cached rotation factors
        mx_v = mx_s * c - my_s * s  # this device's spatially- and vectorially-rotated field, x component
        my_v = mx_s * s + my_s * c  # this device's spatially- and vectorially-rotated field, y component

        # ---- Placement: where this device's field lands on the canvas ----
        y0 = cy - sim_size_y / 2  # this device's field's bottom edge, nm
        x0 = cx - sim_size_x / 2  # this device's field's left edge, nm

        ix0 = int(round((x0 - xmin) / cell_size_x))  # this device's field's left edge, canvas cell index
        iy0 = int(round((y0 - ymin) / cell_size_y))  # this device's field's bottom edge, canvas cell index

        iy1 = iy0 + ny  # this device's field's top edge, canvas cell index
        ix1 = ix0 + nx  # this device's field's right edge, canvas cell index

        # ---- Clip to the canvas's actual bounds (a device near the canvas edge may
        # have part of its own seed-field footprint fall outside it). ----
        cx0 = max(ix0, 0)  # clipped left edge, canvas cell index
        cy0 = max(iy0, 0)  # clipped bottom edge, canvas cell index
        cx1 = min(ix1, canvas_nx)  # clipped right edge, canvas cell index
        cy1 = min(iy1, canvas_ny)  # clipped top edge, canvas cell index

        if cx1 <= cx0 or cy1 <= cy0:  # this device's field does not overlap the canvas at all after clipping
            print(f"[WARNING] compose_fields: skipping device {i} -- field does not overlap the canvas after clipping.")
            continue
        fx0 = cx0 - ix0  # left edge of the region to take from this device's OWN field array (offset by how much was clipped)
        fy0 = cy0 - iy0  # bottom edge of the region to take from this device's OWN field array
        fx1 = fx0 + (cx1 - cx0)  # right edge of the region to take from this device's OWN field array
        fy1 = fy0 + (cy1 - cy0)  # top edge of the region to take from this device's OWN field array
        canvas_slice_x = canvas_x[cy0:cy1, cx0:cx1]  # the canvas region this device's field will be added into, x
        canvas_slice_y = canvas_y[cy0:cy1, cx0:cx1]  # the canvas region this device's field will be added into, y

        field_slice_x = mx_v[fy0:fy1, fx0:fx1]  # this device's own rotated field, x, cropped to the overlapping region
        field_slice_y = my_v[fy0:fy1, fx0:fx1]  # this device's own rotated field, y, cropped to the overlapping region

        # ---- Superposition: add this device's contribution into the canvas, only
        # where the field itself has a finite (non-NaN) value, so a device's own
        # NaN cells (for example, outside its own device shape) do not corrupt
        # whatever another device may have already placed at that same canvas cell. ----
        np.add(canvas_slice_x, field_slice_x, out=canvas_slice_x, where=np.isfinite(field_slice_x))
        np.add(canvas_slice_y, field_slice_y, out=canvas_slice_y, where=np.isfinite(field_slice_y))

        if verbose:
            print(f"[DEBUG] compose_fields: placed device {i} | field='{field_name}' | "
                  f"geom={g_angle} deg | mag={m_angle} deg.")
    extent = (xmin, xmax, ymin, ymax)  # the full (pre-crop) canvas's physical bounds, nm

    # ---- Crop the canvas back down to the original seed simulation's own footprint,
    # centered on (0, 0) -- the DeviceArray convention -- discarding the extra margin
    # around the outermost devices that was needed only to avoid clipping their fields
    # during placement above. ----
    crop_xmin = int(np.ceil((0 - sim_size_x / 2 - margin_nm - xmin) / cell_size_x))  # left crop edge, canvas cell index
    crop_xmax = int(np.ceil((0 + sim_size_x / 2 + margin_nm - xmin) / cell_size_x))  # right crop edge, canvas cell index
    crop_ymin = int(np.ceil((0 - sim_size_y / 2 - margin_nm - ymin) / cell_size_y))  # bottom crop edge, canvas cell index
    crop_ymax = int(np.ceil((0 + sim_size_y / 2 + margin_nm - ymin) / cell_size_y))  # top crop edge, canvas cell index
    canvas_x = canvas_x[crop_ymin:crop_ymax, crop_xmin:crop_xmax]  # cropped canvas, x component
    canvas_y = canvas_y[crop_ymin:crop_ymax, crop_xmin:crop_xmax]  # cropped canvas, y component
    if verbose:
        print(f"[INFO] compose_fields: composed field '{field_name}' onto canvas with superposition. "
              f"Final canvas shape: {canvas_x.shape}.")

    return {
        "x": canvas_x,  # the composed, cropped canvas, x component
        "y": canvas_y,  # the composed, cropped canvas, y component
        "extent": extent  # the full (pre-crop) canvas's physical bounds, nm
    }


# --------------------------
# ---- Multiprocessing worker initializer: share device geometry across worker processes ----
def init_pool(device_centers, geometry_angles, magnetization_angles, verbose=False):
    """
    GOAL:
        Store the device array's centers/angles as module-level globals
        inside each worker process, once, when that worker process starts
        -- so compose_fields_task (below) can read them without needing to
        re-send this same (potentially large) device geometry data with
        every individual task.

    HOW IT TALKS TO OTHER FILES:
        Passed as the `initializer`/`initargs` to ProcessPoolExecutor in
        run_superposition_multiprocessing (this file); called once per
        worker process, automatically, by the multiprocessing machinery
        itself -- not called directly by any other function.

    PARAMETERS:
        device_centers, geometry_angles, magnetization_angles: the same
            device array data compose_fields expects, sent once per worker
            process instead of once per task.
        verbose (bool): prints a status message when True.

    RETURNS:
        None. Sets GLOBAL_DEVICE_CENTERS, GLOBAL_GEOM_ANGLES, and
        GLOBAL_MAG_ANGLES as module-level globals within whichever process
        this function runs in.

    LIMITATIONS:
        Relies on Python multiprocessing's initializer pattern: each
        worker process gets its OWN separate copy of these globals (there
        is no shared memory or synchronization between processes here),
        set once when that process starts and read by every task that
        process subsequently handles. This is done specifically to avoid
        re-pickling and re-sending the full device geometry with every
        single task in compose_fields_task's `task` tuple, which would be
        wasteful for a device array with many devices/fields/variants.
    """
    if verbose:
        print("[DEBUG] init_pool: initializing multiprocessing pool with device centers and angles...")
    global GLOBAL_DEVICE_CENTERS  # shared, per-worker-process copy of the device centers
    global GLOBAL_GEOM_ANGLES  # shared, per-worker-process copy of the geometry angles
    global GLOBAL_MAG_ANGLES  # shared, per-worker-process copy of the magnetization angles

    GLOBAL_DEVICE_CENTERS = device_centers
    GLOBAL_GEOM_ANGLES = geometry_angles
    GLOBAL_MAG_ANGLES = magnetization_angles


# --------------------------
# ---- Multiprocessing-safe worker: compose one (region, field, variant) combination ----
def compose_fields_task(task, verbose=False):
    """
    GOAL:
        Unpack one task tuple and call compose_fields with it, using the
        device geometry already stored in this worker process's globals
        (see init_pool) rather than data sent with the task itself.

    HOW IT TALKS TO OTHER FILES:
        Called by run_superposition_multiprocessing (this file), once per
        task, dispatched across a ProcessPoolExecutor. Calls compose_fields
        (this file), reading GLOBAL_DEVICE_CENTERS/GLOBAL_GEOM_ANGLES/
        GLOBAL_MAG_ANGLES (set by init_pool) rather than accepting them as
        its own parameters.

    PARAMETERS:
        task (Tuple): (region_key, base_key, variant_key, base_field,
            cell_size_x, cell_size_y) -- identifies which (region, base
            field, variant) combination this task composes, plus that
            combination's own seed field and cell size.
        verbose (bool): prints a status message when this worker starts,
            and forwards to compose_fields for its own status messages,
            when True.

    RETURNS:
        Tuple: (region_key, base_key, variant_key, canvas_i) -- the same
        three key values from `task`, plus compose_fields' result dict, so
        the caller can match this result back to which combination it
        belongs to once results return (potentially out of submission
        order) from the process pool.
    """
    if verbose:
        print("[DEBUG] compose_fields_task: worker started.")
    (region_key, base_key, variant_key, base_field, cell_size_x, cell_size_y) = task  # unpack this task's identifying keys and data

    canvas_i = compose_fields(
        device_centers=GLOBAL_DEVICE_CENTERS,  # set once per worker process by init_pool
        geometry_angles=GLOBAL_GEOM_ANGLES,  # set once per worker process by init_pool
        magnetization_angles=GLOBAL_MAG_ANGLES,  # set once per worker process by init_pool
        base_field=base_field,  # this task's own seed field
        field_name=base_key + "_" + variant_key,  # composite name, for cache keys and status messages
        cell_size_x=cell_size_x,
        cell_size_y=cell_size_y,
        margin_nm=0,
        verbose=verbose,
    )
    return region_key, base_key, variant_key, canvas_i


# --------------------------
# ---- Compose every (region, field, variant) combination across a process pool ----
def run_superposition_multiprocessing(
    data_struct_all,  # nested {region_key: {base_key: {variant_key: base_field}}} structure -- every seed field to compose
    device_centers,  # one (cx, cy) per device, in nm -- shared across every task
    geometry_angles,  # one spatial rotation angle per device, degrees -- shared across every task
    magnetization_angles,  # one vector rotation angle per device, degrees -- shared across every task
    cell_size_x,  # nm per grid cell, x direction
    cell_size_y,  # nm per grid cell, y direction
    verbose=False,  # print progress messages when True
    force_serial=False,  # if True, ALWAYS compose in-process, regardless of task count -- see PARAMETERS below
):
    """
    GOAL:
        Compose every (region, base field, variant) combination in
        `data_struct_all` onto its own canvas, in parallel across a pool
        of worker processes.

    HOW IT TALKS TO OTHER FILES:
        Called by mumax3_parent_superposition.py's run_pipeline, once per
        superposition run, with every seed field that needs composing.
        Dispatches compose_fields_task (this file) across a
        ProcessPoolExecutor, initialized once per worker via init_pool
        (this file).

    PARAMETERS:
        data_struct_all (Dict): nested {region_key: {base_key: {variant_key:
            base_field}}} -- every seed field to compose, however many
            regions/fields/variants exist.
        device_centers, geometry_angles, magnetization_angles: the device
            array's geometry, shared across every task (sent once per
            worker process via init_pool, not once per task).
        cell_size_x, cell_size_y (float): nm per grid cell.
        verbose (bool): prints a completion summary when True. Forwarded
            to compose_fields_task/compose_fields, so their own progress
            messages appear too.
        force_serial (bool): if True, always compose every task directly in
            this process, never opening a ProcessPoolExecutor, regardless of
            task count. For callers that are THEMSELVES already running
            inside a worker process dispatched by an outer pool (for
            example, propagation_engine._run_canvas_job, one job among
            several running concurrently across propagation_engine's own
            per-step canvas_pool) -- opening a SECOND, nested pool in here
            would oversubscribe CPUs (outer_workers x inner_workers
            processes total) and re-pay a pool's fixed startup cost once per
            OUTER job, exactly the overhead this function's own task-count
            fast path (see LIMITATIONS) exists to avoid. False (default)
            preserves the task-count-based decision described in
            LIMITATIONS, appropriate for a call made directly from a single-
            threaded caller (for example, mumax3_parent_superposition.py's
            own STEP 4, or setup_seed_and_geometry's one-time seed
            extraction).

    RETURNS:
        Dict[region_key, Dict[base_key, Dict[variant_key, Dict]]]: the
        composed canvas for every (region, base field, variant)
        combination, in the same nested shape as `data_struct_all`, each
        leaf holding compose_fields' own return value ({"x", "y",
        "extent"}).

    LIMITATIONS:
        Results are accumulated into a nested `collections.defaultdict`
        while being built (convenient for constructing this nested
        structure without pre-declaring every key), but converted to
        plain, ordinary nested dicts before being returned. A
        defaultdict's auto-vivifying behavior is a real hazard for a
        RETURNED structure specifically: any wrong or mistyped key looked
        up downstream at the region_key or base_key level would not raise
        where the mistake actually is -- it would silently create an
        empty dict instead, and the code would only fail one or two
        levels deeper (for example, on ['x']) with a KeyError that looks
        like it is pointing at the wrong thing entirely. Converting to
        plain dicts here means a lookup with a wrong key anywhere in
        canvas_struct[region][field][variant] raises a normal, immediate
        KeyError naming the actual bad key.
        When len(tasks) <= max_pool_worthwhile_tasks (currently 1), no
        ProcessPoolExecutor is created at all -- every task is composed
        directly in this process (see the TRACKED CHANGE comment below).
        A worker pool's startup cost (process spawn, plus pickling/sending
        device_centers/geometry_angles/magnetization_angles to every
        worker via init_pool) is fixed and paid on every call; for the
        1-3-task case (typically one task per confirmed dataset, e.g.
        m_full/B_demag), that cost dwarfs the actual compose_fields work
        and was previously paid on every single propagation step's
        baseline AND every distinct excluded-candidate canvas (see
        propagation_engine.compose_and_export_canvas). When the pool IS
        used, max_workers is capped to len(tasks) so idle workers are
        never spawned only to sit unused.
    """
    import os  # os.cpu_count(), to cap worker count at the actual task count below
    from concurrent.futures import ProcessPoolExecutor  # parallel processing across tasks
    from collections import defaultdict  # convenient nested-structure accumulation, converted to plain dicts before returning (see LIMITATIONS)
    tasks = []  # every (region, base field, variant) combination to compose, flattened into one list
    for region_key in data_struct_all:  # walk every region
        for base_key in data_struct_all[region_key]:  # walk every base field within this region
            for variant_key in data_struct_all[region_key][base_key]:  # walk every variant of this base field
                tasks.append((
                    region_key,
                    base_key,
                    variant_key,
                    data_struct_all[region_key][base_key][variant_key],  # this combination's own seed field
                    cell_size_x,
                    cell_size_y
                ))

    # ---- TRACKED CHANGE (perf): a ProcessPoolExecutor's startup cost (spawning
    # worker processes, and -- via init_pool's initargs -- pickling/sending
    # device_centers/geometry_angles/magnetization_angles to every one of
    # them) is fixed and paid on every call, regardless of how many tasks
    # exist. With ProcessPoolExecutor's own default worker count (os.cpu_count()
    # when max_workers is not passed, as here), that overhead was previously
    # paid to run as few as 1-3 tasks (one per confirmed dataset, e.g. m_full/
    # B_demag -- see setup_seed_and_geometry's data_struct_all, which has
    # exactly one region), which is the case on EVERY propagation step (once
    # for the baseline canvas, once more per distinct excluded candidate --
    # see propagation_engine.compose_and_export_canvas). Below max_pool_
    # worthwhile_tasks, skip the pool and compose every task directly, in
    # this process. Larger task lists (many regions/fields/variants -- the
    # normal static-superposition case) are unaffected and still use the
    # pool below, capped to len(tasks) workers so idle workers are never
    # spawned just to sit unused. ----
    max_pool_worthwhile_tasks = 1  # a pool is not worth spinning up to compose this many tasks or fewer; tune only if profiling shows the crossover point is different on your machine
    if force_serial or len(tasks) <= max_pool_worthwhile_tasks:
        if verbose:
            reason = "force_serial=True" if force_serial else f"only {len(tasks)} task(s)"
            print(f"[DEBUG] run_superposition_multiprocessing: {reason} -- skipping ProcessPoolExecutor, composing in-process.")
        init_pool(device_centers, geometry_angles, magnetization_angles, verbose=verbose)  # set this (single, current) process's own globals directly, same globals compose_fields_task reads in a worker
        results = [compose_fields_task(task, verbose=verbose) for task in tasks]
        if verbose:
            print(f"[INFO] run_superposition_multiprocessing: completed in-process superposition for {len(results)} task(s).")
    else:
        with ProcessPoolExecutor(
            max_workers=min(len(tasks), (os.cpu_count() or 1)),  # never spawn more workers than there are tasks to hand them
            initializer=init_pool,  # runs once per worker process, storing the device geometry as that process's own globals
            initargs=(device_centers, geometry_angles, magnetization_angles)
        ) as pool:
            results = list(pool.map(compose_fields_task, tasks))  # compose every task, across the worker pool
            if verbose:
                print(f"[INFO] run_superposition_multiprocessing: completed multiprocessing superposition for {len(results)} task(s).")

    canvas_struct = defaultdict(lambda: defaultdict(dict))  # accumulates results into a nested structure -- see function LIMITATIONS regarding the plain-dict conversion below
    for region_key, base_key, variant_key, canvas_i in results:  # place each task's result into the nested structure
        canvas_struct[region_key][base_key][variant_key] = {
            "x": canvas_i["x"],
            "y": canvas_i["y"],
            "extent": canvas_i["extent"]
        }
    return {region_key: {base_key: dict(variant_dict) for base_key, variant_dict in base_dict.items()}  # convert every level to a plain dict (see function LIMITATIONS)
            for region_key, base_dict in canvas_struct.items()}
