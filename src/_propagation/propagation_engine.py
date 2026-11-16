r"""
File Name: propagation_engine.py
Author: Starr Feng, Connor Sullivan
Date: 2026.08.10
Revision: v1

============================================================
PURPOSE
============================================================
Runs multi-injection synchronous nearest-neighbor propagation across an ASI
array of ANY geometry DeviceArray can build (square, pinwheel, kagome):
starting from one or more initially-flipped "injection" devices, each
active source device proposes at most one flip candidate per step (its
strongest eligible neighbor, from propagation_neighbors.find_candidate_
neighbors), duplicate proposals for the same target device collapse into
one accepted flip carrying the union of every contributing injection's
"origin" label, and the accepted flips become the next step's active
sources. This is the SAME algorithm as the uploaded propagation script --
see this file's own LIMITATIONS section for what was intentionally left
unchanged.

============================================================
WHAT CHANGED FROM STARR'S PROPAGATION SCRIPT
============================================================
1. GEOMETRY-GENERAL CANDIDATE SEARCH. The uploaded script's candidate rule
   was a hardcoded square-ASI offset table (see propagation_neighbors.py's
   own module-level PURPOSE section for the full explanation). This file
   uses propagation_neighbors.find_candidate_neighbors instead, which works
   for any geometry.DeviceArray-buildable geometry.

2. B_demag IS RECOMPUTED THROUGH THE REAL SUPERPOSITION + SPATIAL-ANALYSIS
   PIPELINE EVERY STEP, using RESTRICTED region lists, not full-array
   plots. Per step:
     a. ONE "baseline" canvas: run_superposition_multiprocessing over the
        CURRENT magnetization pattern (no exclusion), exported as this
        step's array-level .npz.
     b. From that baseline canvas: ONE restricted-region analysis pass
        (this step's active sources + every one of their candidates, each
        as its own file-masked device region -- NOT every device in the
        array) via mumax3_parent_spatial_analysis.run_pipeline, AND one
        whole-array pass (mask_mode='none', the full canvas as one
        region) -- both real calls into the existing plotting/stats
        pipeline, landing in this step's own output folder.
     c. For every DISTINCT candidate device proposed by ANY active source
        this step (deduplicated -- a candidate proposed by two sources is
        only excluded/recomposed ONCE): a second "excluded" canvas, with
        that candidate's own magnetization angle set to NaN before
        recomposing (see geometry.DeviceArray.set_device_magnetization),
        then the SAME two analysis passes (SAME restricted region list,
        including the excluded candidate's own region -- it is analyzed
        under the field that excludes its own self-contribution, which is
        exactly the quantity the switching decision needs; and the SAME
        whole-array pass, for visualization of the effect of removing one
        island) against this excluded canvas.
   The candidate's masked B_demag mean (from step (c)'s restricted-region
   pass, via mumax3_csv_stats.masked_region_mean_xy) feeds the actual switching
   decision. Switching criterion can be chosen between |H| and Hrev based on if angle dependent
   coercivity table is used or not.

3. FLIP_MODE / BRANCH_FIRST_STEP / candidate-edge filtering, ported from a
   parallel team member's cascade script (zy_spatialmask_pipeline.py) at
   Connor's request, after comparing it against this file: that script
   predates this codebase's current RegionSpec/SpatialAnalysisConfig
   architecture (dict-based compute_regions calls, a flat single-spacing
   nearest-neighbor search) and most of it does not apply here for the
   same reason the original uploaded propagation script's own region/
   neighbor plumbing didn't (see propagation_neighbors.py's own PURPOSE
   section) -- but three things in it were genuinely new relative to this
   engine and are ported in:
     - PropagationConfig.flip_mode ("max", the original default -- only
       the single strongest eligible candidate per source flips each
       step -- or "all_eligible" -- every eligible candidate for a given
       source flips in the same step, evaluated synchronously against
       that step's own pre-flip field state) and PropagationConfig.
       branch_first_step (step 0 always behaves like "all_eligible",
       regardless of flip_mode, letting the initial injection branch out
       before later steps settle into whatever flip_mode was chosen).
       Originally FLIP_MODE/BRANCH_FIRST_STEP.
     - PropagationConfig.candidate_edge_margin_cells: a candidate is
       dropped (with a warning) if its own mask footprint would cross the
       seed canvas's edge, via the shared masking.region_crosses_edge
       utility -- not because masking.apply_mask corrupts the resulting
       data (it NaN-pads an edge-crossing region correctly; see masking.
       py's own module-level PURPOSE section), but because the resulting
       masked mean is still averaged over LESS than that candidate's true
       full footprint, silently, with nothing flagging that this
       happened -- and can degenerate to all-NaN (silently read as
       switch=False) if most or all of the footprint is off-canvas. The
       parallel script did this by filtering candidates against the
       composed canvas's own reported extent; this engine does the
       equivalent bounds check directly against each candidate's own
       footprint and the seed canvas size, in run_propagation's own
       candidate-discovery step, before any canvas is even composed. The
       underlying detection (region_crosses_edge) is shared with the
       static superposition/spatial-analysis pipeline -- see masking.py's
       own module-level documentation, and mumax3_parent_superposition.
       py's STEP 6, which uses the same function as a warning rather than
       a drop.
     - The comparison CSV's H_angle_vs_mag_deg column: field direction
       relative to a candidate's OWN current magnetization (as opposed
       to delta_theta_deg, already present, which is relative to its
       easy axis) -- a diagnostic column only, not used in the switching
       decision itself. Originally computed inline in that script's own
       live-CSV writer.
   Not ported, and why: that script's own region-extraction path (dict-
   style compute_regions(input_args)) and neighbor search (get_nearest_
   neighbors, a flat single-spacing distance cutoff) are both superseded
   here by RegionSpec/build_region_specs and propagation_neighbors.
   find_candidate_neighbors respectively, for the same reasons detailed
   in propagation_neighbors.py's own module-level PURPOSE section (most
   directly, that script's neighbor search assumes one lattice spacing for
   the whole array and cannot describe kagome's own irregular near-
   neighbor shells). Its hardcoded angle-dependent coercivity table
   (HC_ANGLES/HC_VALUES) is already superseded by this engine's own CSV-
   driven mumax3_csv_stats.load_coercivity_table/lookup_nearest_
   coercivity, which is the same mechanism generalized to load from a
   file instead of a hardcoded array. Its redundant cascade_results.npz
   (a second copy of the same tabular data already in cascade_results.csv,
   written via np.savez on named 1D arrays) was not ported -- this
   engine's own comparison_csv/step_summary_csv already cover that same
   data without a second, harder-to-version copy of it (see this
   codebase's own prior .npy/.npz audit). Its plotting script's own
   candidate-direction labels ("UR", "BL", ...) are square-ASI-specific
   compass directions, the same generality problem propagation_neighbors.
   py's own slot_label already replaced with a measured-displacement
   label; not reintroduced here. Its extensive inline DEBUG print
   blocks (inspecting canvas_struct's own internal shape/keys) reflect
   that script's own bootstrapping process against an earlier canvas API
   and were not carried over.

============================================================
WHAT DID NOT CHANGE (kept close to the original script, at Connor's request)
============================================================
- The synchronous multi-injection stepping algorithm itself: active_sources
  as {device_index: {origin_label, ...}}, one proposal per active source
  per step (its strongest eligible candidate), duplicate-target collapse
  merging origin sets, MAX_STEPS/ALLOW_REFLIP semantics -- unchanged from
  the uploaded script's own main() loop.
- The switching physics itself (mumax3_csv_stats.hrev_from_field/field_angle_
  deg_from_xy/acute_angle_to_easy_axis/lookup_nearest_coercivity) --
  unchanged, only relocated (see that file's own LIMITATIONS section).
- The comparison/step-summary CSV schema is the same shape as the uploaded
  script's, with "nn_id" (a square-ASI compass label like "NN_UL") renamed
  to "slot_label" (propagation_neighbors' geometry-general
  "shell{K}_v{n}_dx{dx}_dy{dy}" form -- see that file's own documentation).

============================================================
LIMITATIONS
============================================================
- Both visualizations from the uploaded scripts are now wired in via
  propagation_plot.py -- plot_neighbor_field_comparison called inline,
  once per active source per step (config.save_neighbor_field_figures),
  and generate_flip_path_visualizations called once at the end
  (config.save_flip_path_visualizations). See that file's own module-level
  documentation for what changed in the port.
- Every step's restricted-region analysis pass reuses SpatialAnalysisConfig
  fields taken from PropagationConfig (see mumax3_parent_propagation.py) --
  compute_stats, plot_spatial_maps, etc. -- so a propagation run's per-step
  output volume (plots + stats CSVs, now per STEP rather than once for a
  single static composition) scales with both step count and however many
  of those toggles are left on. Consider plot_spatial_maps=False/
  compute_stats=False for a long exploratory run, and only turning them on
  for a final run over a shorter step range (see PropagationConfig.
  plot_spatial_limiter, forwarded through unchanged).
- Excluded-canvas composition still runs run_superposition_multiprocessing
  over the FULL array every time (not just near the candidate), because
  B_demag is long-range -- excluding one device's contribution changes the
  field everywhere, not just locally, so there is no shortcut to a
  restricted-array recomposition. Only the ANALYSIS pass afterward is
  restricted to a small region list; the composition itself is not.
- CANDIDATE-LEVEL PARALLELIZATION (perf): baseline + every distinct
  excluded-candidate canvas, per step, are independent given that step's
  starting array state, and are dispatched across canvas_pool (a
  ProcessPoolExecutor persisted for the whole run, not recreated per step
  or per job) rather than composed one at a time -- see the module-level
  block just above _init_canvas_pool_worker/_run_canvas_job, and
  PropagationConfig.parallelize_step_candidates/canvas_job_max_workers.
  The live `array` object is NEVER mutated to build an excluded canvas
  (unlike compose_and_export_canvas's own exclude_index parameter, kept
  as a single-caller-at-a-time building block for custom scripts) -- every
  job instead carries its own private copy of that step's mag_angles_deg,
  so concurrently running jobs cannot race on, or otherwise disturb, one
  another's inputs.
"""

import os  # path construction
from collections import defaultdict  # per-step bookkeeping (active sources, proposals, collapses)
from contextlib import ExitStack  # manages the two per-run CSV files + the optional canvas_pool as one set of context managers, without an extra indentation level around the whole step loop (see run_propagation's own pool-setup comment)
from typing import Dict, List, Optional, Tuple  # type hints used throughout this file's signatures

import numpy as np  # array operations used throughout this file

from propagation_utilities import (  # this codebase's own _propagation utilities (see that file)
    get_device_mesh, flip_device, resolve_injection_indices,
    origin_label_for_injection, format_origin_set, merge_origin_sets,
)
from propagation_neighbors import find_candidate_neighbors  # geometry-general candidate search (see that file)
from mumax3_csv_stats import (  # switching physics + masked-region field extraction (see that file)
    masked_region_mean_xy, hrev_from_field, field_angle_deg_from_xy,
    acute_angle_to_easy_axis, signed_angle_difference_deg,
    load_coercivity_table, lookup_nearest_coercivity,
)
from propagation_plot import plot_neighbor_field_comparison, generate_flip_path_visualizations  # visualizations (see that file) -- called inline (per source/step) and once at the end, respectively


# --------------------------
# ---- CANDIDATE-LEVEL PARALLELIZATION (perf) ----
# Every propagation step's baseline canvas and every distinct excluded-
# candidate canvas (see run_propagation's own STEP (b)/(c)) are INDEPENDENT
# given that step's starting array state: each one only reads the CURRENT
# device mesh (never mutates the live `array`) and writes to its own,
# distinct canvas_dir/output_dir. That independence is what makes them
# safe to run concurrently, and is deliberately enforced by the design
# below, not just assumed:
#   - _init_canvas_pool_worker stores the seed field (data_struct_all) and
#     the STATIC part of the device mesh (centers_nm, geom_angles_deg --
#     device positions/geometry angles never change across a propagation
#     run, only magnetization does) as this process's own globals, once.
#   - Each job carries its OWN PRIVATE COPY of mag_angles_deg (with that
#     job's own excluded candidate, if any, set to NaN in that copy only --
#     see run_propagation's own job-construction loop). The shared `array`
#     object is never mutated for exclusion at all, in either the parallel
#     or the serial fallback path -- there is nothing left for two
#     concurrently running jobs to race on.
#   - When dispatched across canvas_pool (a ProcessPoolExecutor), each job
#     additionally runs in its own OS process with its own copy of every
#     global above, so even a hypothetical in-process mistake in one job
#     could not corrupt another job's results.
# _run_canvas_job is the single implementation both the parallel
# (canvas_pool.map) and serial (plain Python loop) dispatch paths in
# run_propagation call -- see PropagationConfig.parallelize_step_candidates
# -- so the two paths cannot silently compute different things.
_CANVAS_POOL_DATA_STRUCT_ALL = None  # this worker process's own copy of the seed field -- set once by _init_canvas_pool_worker, read by every job this process handles
_CANVAS_POOL_CENTERS_NM = None  # this worker process's own copy of the (static) device centers, nm
_CANVAS_POOL_GEOM_ANGLES_DEG = None  # this worker process's own copy of the (static) device geometry angles, degrees
_CANVAS_POOL_CELL_SIZE_X = None  # nm per grid cell, x direction
_CANVAS_POOL_CELL_SIZE_Y = None  # nm per grid cell, y direction


def _init_canvas_pool_worker(data_struct_all, centers_nm, geom_angles_deg, cell_size_x, cell_size_y, verbose=False):
    """
    GOAL:
        Store the seed field and the STATIC part of the device mesh
        (centers/geometry angles -- never magnetization, which changes
        every step) as this process's own module-level globals, once --
        so _run_canvas_job (below) can read them without re-sending this
        same (potentially large, for data_struct_all) data with every
        individual job.

    HOW IT TALKS TO OTHER FILES:
        Passed as canvas_pool's own `initializer`/`initargs` in
        run_propagation (this file, once per regeneration, when
        config.parallelize_step_candidates is True) -- called once per
        worker process, automatically, by the multiprocessing machinery.
        ALSO called directly (not via a pool) once in the main process by
        run_propagation, so the serial fallback path (parallelize_step_
        candidates=False, or a step with only a baseline job) can call
        _run_canvas_job directly using the SAME globals a worker process
        would have. Mirrors superposition_operations.init_pool's own
        pattern, one level up (seed field + mesh, not just mesh).

    PARAMETERS:
        data_struct_all (Dict): the seed device's own field -- see
            setup_seed_and_geometry's own RETURNS documentation. Identical,
            read-only, for every step and every job within a run.
        centers_nm, geom_angles_deg (np.ndarray): the array's STATIC device
            centers/geometry angles (see get_device_mesh) -- read once,
            before the step loop begins, since neither changes as flips
            accumulate (only magnetization angle does).
        cell_size_x, cell_size_y (float): nm per grid cell.
        verbose (bool): prints a status message when True.

    RETURNS:
        None. Sets _CANVAS_POOL_DATA_STRUCT_ALL/_CANVAS_POOL_CENTERS_NM/
        _CANVAS_POOL_GEOM_ANGLES_DEG/_CANVAS_POOL_CELL_SIZE_X/_CANVAS_POOL_
        CELL_SIZE_Y as module-level globals within whichever process this
        function runs in.
    """
    global _CANVAS_POOL_DATA_STRUCT_ALL, _CANVAS_POOL_CENTERS_NM, _CANVAS_POOL_GEOM_ANGLES_DEG, _CANVAS_POOL_CELL_SIZE_X, _CANVAS_POOL_CELL_SIZE_Y
    _CANVAS_POOL_DATA_STRUCT_ALL = data_struct_all
    _CANVAS_POOL_CENTERS_NM = centers_nm
    _CANVAS_POOL_GEOM_ANGLES_DEG = geom_angles_deg
    _CANVAS_POOL_CELL_SIZE_X = cell_size_x
    _CANVAS_POOL_CELL_SIZE_Y = cell_size_y
    if verbose:
        print("[DEBUG] _init_canvas_pool_worker: initialized with seed field + static device mesh.")


def _run_canvas_job(job: Dict) -> Dict:
    """
    GOAL:
        Compose ONE canvas (baseline, or one candidate's own excluded
        canvas -- see job["job_type"]) and run its restricted-region
        analysis pass, using ONLY this job's own private inputs (never the
        live `array`, never any other job's data) -- the single unit of
        work run_propagation dispatches, either across canvas_pool
        (parallel) or directly in a plain Python loop (serial fallback);
        see this file's own CANDIDATE-LEVEL PARALLELIZATION section above
        for the isolation guarantees this relies on.

    HOW IT TALKS TO OTHER FILES:
        Calls superposition_operations.run_superposition_multiprocessing
        (force_serial=True -- see that parameter's own documentation for
        why: this function may itself already be running inside one of
        canvas_pool's worker processes, and must never open a SECOND,
        nested pool in here), utilities_superposition.export_canvas_as_
        npz_frames/export_canvas_as_csv_frames, analyze_canvas (this
        file), and mumax3_csv_stats.masked_region_mean_xy (candidate jobs
        only).

    PARAMETERS:
        job (Dict): one entry from run_propagation's own `jobs` list for
            this step -- see that construction for every key's exact
            meaning: job_type ("baseline"|"candidate"), cand_idx
            (Optional[int]), own_label (Optional[str], candidate jobs
            only), mag_angles_deg (this job's OWN private copy -- the
            excluded candidate's angle, if any, already set to NaN),
            canvas_dir, output_dir, extracted_regions_subdir_prefix,
            confirmed_datasets, restricted_centers/sizes/angles/labels,
            canvas_size, canvas_center, extra_zoom_targets, config,
            verbose, force_serial_analysis (bool -- forwarded to analyze_
            canvas's own force_serial parameter; see run_propagation's own
            job-dispatch logic for how this is set).

    RETURNS:
        Dict: {"job_type", "cand_idx"} always; candidate jobs additionally
        carry {"Hx_Oe", "Hy_Oe"} -- the candidate's own masked B_demag mean
        field (from analyze_canvas's device-region pass), exactly what
        run_propagation's own candidate_field_cache needs. Only these few
        floats cross back out of a worker process -- the (potentially
        large) device_regions/whole_array_regions structures analyze_canvas
        returns are read here and discarded, not returned, to keep
        canvas_pool's own IPC small.
    """
    config = job["config"]
    verbose = job["verbose"]

    from superposition_operations import run_superposition_multiprocessing
    from utilities_superposition import export_canvas_as_npz_frames, export_canvas_as_csv_frames

    os.makedirs(job["canvas_dir"], exist_ok=True)
    canvas_struct = run_superposition_multiprocessing(
        data_struct_all=_CANVAS_POOL_DATA_STRUCT_ALL,
        device_centers=[tuple(c) for c in _CANVAS_POOL_CENTERS_NM],
        geometry_angles=list(_CANVAS_POOL_GEOM_ANGLES_DEG),
        magnetization_angles=list(job["mag_angles_deg"]),
        cell_size_x=_CANVAS_POOL_CELL_SIZE_X, cell_size_y=_CANVAS_POOL_CELL_SIZE_Y,
        verbose=verbose,
        force_serial=True,  # never open a nested pool -- see this function's own HOW IT TALKS TO OTHER FILES section
    )
    export_canvas_as_npz_frames(canvas_struct, job["canvas_dir"], frame_index=0, verbose=verbose)
    if config.export_canvas_csv:
        export_canvas_as_csv_frames(canvas_struct, job["canvas_dir"], frame_index=0, verbose=verbose)

    device_regions, _whole_array_regions = analyze_canvas(
        config, canvas_dir=job["canvas_dir"], confirmed_datasets=job["confirmed_datasets"],
        device_centers=job["restricted_centers"], device_sizes=job["restricted_sizes"],
        device_angles=job["restricted_angles"], device_labels=job["restricted_labels"],
        canvas_size=job["canvas_size"], canvas_center=job["canvas_center"],
        output_dir=job["output_dir"], extracted_regions_subdir_prefix=job["extracted_regions_subdir_prefix"],
        extra_zoom_targets=job["extra_zoom_targets"], verbose=verbose,
        force_serial=job["force_serial_analysis"],  # see this file's own CANDIDATE-LEVEL PARALLELIZATION section -- True whenever this job was actually dispatched across canvas_pool, so plotting/stats never open a SECOND, nested pool inside this (already parallel) worker
    )

    result = {"job_type": job["job_type"], "cand_idx": job["cand_idx"]}
    if job["job_type"] == "candidate":
        cand_region = device_regions["B_demag"][job["own_label"]]
        Hx_Oe, Hy_Oe = masked_region_mean_xy(cand_region, field_scaling=config.b_to_oe, verbose=verbose)
        result["Hx_Oe"] = Hx_Oe
        result["Hy_Oe"] = Hy_Oe
    return result


# --------------------------
# ---- STEP A: seed field + geometry setup (mirrors mumax3_parent_superposition.run_pipeline's own STEP 1-3) ----
def setup_seed_and_geometry(config, regeneration_index: int = 0, verbose: bool = False):
    """
    GOAL:
        Extract the seed device's own field (m_full/B_demag, unrotated) and
        build the ASI geometry array -- the same two things mumax3_parent_
        superposition.run_pipeline's STEP 1-3 do, factored out here so the
        propagation engine can reuse them WITHOUT also running that
        function's own STEP 4-7 (a single static composition), since this
        engine performs its own repeated composition instead (see this
        file's module-level PURPOSE section).

    HOW IT TALKS TO OTHER FILES:
        Calls mumax3_parent_utilities.convert_npz_per_dataset, region_
        extraction.build_region_specs, mumax3_parent_utilities.
        region_processor/compute_regions (seed extraction), and geometry.
        DeviceArray (array construction) -- the identical calls mumax3_
        parent_superposition.run_pipeline's own STEP 1-3 make, so a
        PropagationConfig's geometry fields behave identically to the same
        fields on SuperpositionConfig.

    PARAMETERS:
        config (PropagationConfig): see mumax3_parent_propagation.py.
        regeneration_index (int): which regeneration this is (0-based) --
            only affects the random seed used for magnetization_mode=
            "random" (seed = config.random_seed_start + regeneration_index,
            matching SuperpositionConfig's own num_regenerations/
            random_seed_start convention). 0 for a single-regeneration run
            (config.num_regenerations == 1, the default).
        verbose (bool): forwarded into every call below.

    RETURNS:
        Tuple[DeviceArray, Dict, List[str], Tuple[int,int], Tuple[int,int]]:
            array: the built geometry.DeviceArray.
            data_struct_all: {seed_label: {dataset: {"masked": {"x":...,
                "y":...}}}} -- the seed device's own unrotated field, in
                the exact shape superposition_operations.run_superposition_
                multiprocessing expects.
            confirmed_datasets: dataset names actually available (subset of
                config.fields).
            seed_size: (width, height) grid cells -- the seed/composed-
                canvas domain size.
            canvas_center: (cx, cy) grid cells -- the composed canvas's own
                center, used to convert DeviceArray's canvas-relative
                device centers into absolute grid positions (see
                mumax3_parent_superposition.py's STEP 5 for the identical
                conversion).
    """
    from mumax3_parent_utilities import convert_npz_per_dataset, region_processor, compute_regions
    from region_extraction import build_region_specs
    from geometry import DeviceArray

    confirmed_datasets, _confirmed_metadata = convert_npz_per_dataset(
        config.data_directory, config.npz_directory, set(config.fields),
        config.use_cached_npz, verbose=verbose,
    )

    seed_center = (
        int(round((config.sim_size_x / config.cell_size_x) / 2)),
        int(round((config.sim_size_y / config.cell_size_y) / 2)),
    )
    seed_size = (
        int(round(config.sim_size_x / config.cell_size_x)),
        int(round(config.sim_size_y / config.cell_size_y)),
    )
    seed_specs = build_region_specs(centers=[seed_center], sizes=[seed_size], mask_mode="none", verbose=verbose)
    seed_label = seed_specs[0].resolved_label()

    npz_dir_processed = os.path.join(config.npz_directory, "extracted_regions")
    os.makedirs(npz_dir_processed, exist_ok=True)

    base_field: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset in confirmed_datasets:
        seed_regions = region_processor(
            output_path=os.path.join(npz_dir_processed, f"extracted_regions_{dataset}_seed.npz"),
            compute_fn=compute_regions(
                npz_directory=config.npz_directory, cell_size_x=config.cell_size_x, cell_size_y=config.cell_size_y,
                region_specs=seed_specs, prefix=dataset, max_workers=config.max_workers, verbose=verbose,
            ),
            use_cached=config.use_cached_npz,
            description=f"{dataset} seed region",
            verbose=verbose,
        )
        region = seed_regions[seed_label]
        base_field[dataset] = {
            "x": region["masked_x"][config.seed_frame_index],
            "y": region["masked_y"][config.seed_frame_index],
        }

    array = DeviceArray(
        device_length=config.device_length_nm, device_width=config.device_width_nm,
        nn_x=config.nn_x_nm, nn_y=config.nn_y_nm,
        sublattice_dims=config.sublattice_dims,
        device_rotations=config.device_rotations,
        rotation_angles=config.rotation_angles,
        translations=config.translations,
        sublattice_termination=config.sublattice_termination,
        sublattice_termination_mode=config.sublattice_termination_mode,
        verbose=verbose,
    )
    if config.indices_to_remove:
        array.remove_devices(indices=config.indices_to_remove, verbose=verbose)

    for j in range(len(config.sublattice_dims)):  # initial magnetization pattern, one sublattice at a time -- see SuperpositionConfig.magnetization_mode/magnetization_base_angle
        array.set_magnetization_pattern(
            mode=config.magnetization_mode, sublattice=j,
            base_angle=config.magnetization_base_angle.get(j) if config.magnetization_base_angle is not None else None,
            random_seed=(config.random_seed_start + regeneration_index) if config.magnetization_mode == "random" else None,
            verbose=verbose,
        )
        #array.set_device_magnetization(index=24, angle=45, verbose=verbose)
        
    # Seed here means template
    # MANUALLY EDIT HERE
    # ---- Opposite to ground state ----
    misc_indices = [25, 19, 20, 8, 9, 13, 11, 22]
    bottom_row_indices = [15, 26, 17, 28]
    second_row_indices = [0, 2, 4]
    # extra_flipped_indices = [0, 6, 10, 21, 1, 25, 7, 2]
    # all_indices = bottom_row_indices + second_row_indices + misc_indices
    # ---- Opposite to Spin Ice 1 ----
    all_indices = [21, 23, 14, 1, 3, 25, 16, 27, 18, 29, 6, 7, 8, 9,10, 11, 13, 4, 15, 26]
    # -------MISC??-------
    # all_indices = [21, 14, 25, 27, 19, 6, 8, 9, 20, 11, 0, 17]
    # all_indices = [2, 17, 26, 11, 22]
    # ---- Opposite to Spin Ice 2 ---
    # all_indices = [23, 1, 3, 16, 18, 29, 19, 7, 10, 20, 13, 0, 4, 15, 26, 17, 28]
    for idx in all_indices:
        current_angle = array.device_m_angles[idx]
        new_angle = current_angle + 180
        array.set_device_magnetization(idx, (new_angle) % 360)
        print(f"idx={idx}: {current_angle} -> {new_angle}")

 

    from utilities_superposition import save_device_array  # geometry save (below) -- same call mumax3_parent_superposition.py's own STEP 3 makes
    output_dir_geom = os.path.join(config.output_dir, "ASI_geometry")
    os.makedirs(output_dir_geom, exist_ok=True)
    save_device_array(
        device_array_obj=array, base_save_path=os.path.join(output_dir_geom, f"ASI_{config.magnetization_mode}"),
        save_table_exts=["csv", "pdf"], save_plot_exts=["png", "pdf"], verbose=verbose,
    )  # writes this run's INITIAL geometry/magnetization pattern (before any propagation flip) -- see this function's own RETURNS documentation

    data_struct_all = {seed_label: {dataset: {"masked": base_field[dataset]} for dataset in confirmed_datasets}}
    canvas_center = (seed_size[0] // 2, seed_size[1] // 2)
    return array, data_struct_all, confirmed_datasets, seed_size, canvas_center


# --------------------------
# ---- STEP B: compose (optionally excluding one device) + export one canvas ----
def compose_and_export_canvas(config, array, data_struct_all, exclude_index: Optional[int], canvas_dir: str, verbose: bool = False) -> None:
    """
    GOAL:
        Run the full-array superposition ONCE and export it as a composed-
        canvas .npz set -- optionally with one device's own contribution
        excluded first (by temporarily setting its magnetization angle to
        NaN; see geometry.DeviceArray.set_device_magnetization), then
        restored afterward regardless of success or failure.

    HOW IT TALKS TO OTHER FILES:
        Calls superposition_operations.run_superposition_multiprocessing
        and utilities_superposition.export_canvas_as_npz_frames -- the same
        two calls mumax3_parent_superposition.run_pipeline's own STEP 4
        makes for a single static composition.

    PARAMETERS:
        config (PropagationConfig): see mumax3_parent_propagation.py.
        array (geometry.DeviceArray): the live array (current magnetization
            pattern reflects every flip applied so far).
        data_struct_all (Dict): the seed device's own field, from setup_
            seed_and_geometry.
        exclude_index (Optional[int]): device index to exclude (NaN its
            magnetization angle) before composing, or None for the plain
            "baseline" composition.
        canvas_dir (str): output directory for this composition's exported
            .npz frame (frame_index=0 always -- each call gets its own
            directory, so there is never more than one frame per
            canvas_dir).
        verbose (bool): forwarded into both calls below.

    RETURNS:
        None. Writes canvas_dir/<dataset>_npz_cache/<dataset>000000.npz for
        every dataset in data_struct_all.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding why the
        FULL array (not a restricted subset) is always recomposed here.
        run_propagation's own step loop no longer calls this function --
        see this file's own CANDIDATE-LEVEL PARALLELIZATION section and
        _run_canvas_job, which run several (baseline + one per distinct
        candidate) canvas compositions concurrently, each against its own
        PRIVATE mag_angles_deg array copy, rather than mutating `array`
        in place and restoring it afterward (this function's own approach,
        below) -- a mutate-then-restore pattern is safe for one caller at a
        time but is NOT safe to call concurrently from multiple threads/
        processes against the SAME shared `array` object, which is exactly
        why _run_canvas_job never does it. This function is kept, and still
        used by setup_seed_and_geometry's own callers/custom scripts (see
        mumax3_parent_propagation.py's own "BUILDING A CUSTOM PROPAGATION
        SCRIPT" section) that want the single-composition building block
        directly, one call at a time -- do not call this function from
        more than one thread/process against the same `array` at once.
    """
    from superposition_operations import run_superposition_multiprocessing
    from utilities_superposition import export_canvas_as_npz_frames, export_canvas_as_csv_frames

    os.makedirs(canvas_dir, exist_ok=True)
    original_angle = None
    if exclude_index is not None:
        _, _, mag_angles_now = get_device_mesh(array)
        original_angle = float(mag_angles_now[exclude_index])
        array.set_device_magnetization(index=exclude_index, angle=np.nan, verbose=verbose)

    try:
        centers_nm, geom_angles_deg, mag_angles_deg = get_device_mesh(array)  # fresh read -- reflects the exclusion above, if any
        canvas_struct = run_superposition_multiprocessing(
            data_struct_all=data_struct_all,
            device_centers=[tuple(c) for c in centers_nm],
            geometry_angles=list(geom_angles_deg),
            magnetization_angles=list(mag_angles_deg),
            cell_size_x=config.cell_size_x, cell_size_y=config.cell_size_y,
            verbose=verbose,
        )
        export_canvas_as_npz_frames(canvas_struct, canvas_dir, frame_index=0, verbose=verbose)
        if config.export_canvas_csv:  # optional human-viewable companion CSV -- see utilities_superposition.export_canvas_as_csv_frames' own documentation
            export_canvas_as_csv_frames(canvas_struct, canvas_dir, frame_index=0, verbose=verbose)
    finally:
        if exclude_index is not None:  # always restore, even if composition raised
            array.set_device_magnetization(index=exclude_index, angle=original_angle, verbose=verbose)


# --------------------------
# ---- STEP C: restricted-region + whole-array analysis passes over one composed canvas ----
def analyze_canvas(
    config, canvas_dir: str, confirmed_datasets: List[str],
    device_centers: List[Tuple[int, int]], device_sizes: List[Tuple[int, int]],
    device_angles: List[float], device_labels: List[str],
    canvas_size: Tuple[int, int], canvas_center: Tuple[int, int],
    output_dir: str, extracted_regions_subdir_prefix: str,
    extra_zoom_targets: Optional[Dict[str, Tuple[int, int]]] = None, verbose: bool = False,
    force_serial: bool = False,
) -> Tuple[Dict, Optional[Dict]]:
    """
    GOAL:
        Run the SAME two analysis passes mumax3_parent_superposition.
        run_pipeline's own STEP 6 (per-device, mask_mode='file') and
        STEP 7 (whole canvas, mask_mode='none') perform, against ONE
        already-composed canvas, restricted to a caller-supplied device
        list rather than every device in the array -- see this file's
        module-level PURPOSE section for why (only the active sources +
        their candidates need analyzing each step, not the whole array).

    HOW IT TALKS TO OTHER FILES:
        Calls mumax3_parent_spatial_analysis.SpatialAnalysisConfig/
        run_pipeline twice (device-region pass, whole-canvas pass) -- the
        exact same function mumax3_parent_superposition.py's own STEP 6/7
        call, so per-device masking/plotting/stats behave identically to
        the main superposition pipeline.

    PARAMETERS:
        config (PropagationConfig): supplies mask_file, plotting/stats
            toggles (plot_spatial_maps, compute_stats, etc.), all forwarded
            straight through to both SpatialAnalysisConfig calls below, and
            every canvas_* field (canvas_split, canvas_zoom_targets,
            canvas_zoom_size, canvas_zoom_all_devices, canvas_include_full,
            canvas_mask_mode, canvas_mask_file, canvas_compute_stats,
            canvas_zoom_propagation_front, canvas_zoom_candidates) --
            mirrors SuperpositionConfig's own STEP 7 construction exactly
            (same region_extraction.build_canvas_split_centers_sizes/
            build_canvas_zoom_centers_sizes calls), PLUS two propagation-
            specific auto zoom targets on top (see extra_zoom_targets,
            below) -- this pass runs EVERY step (baseline and every
            per-candidate excluded canvas), not once for a single static
            composition, so it is the visual record of how the field
            actually looks at the propagation front as the run proceeds.
        canvas_dir (str): the already-composed canvas's own directory (see
            compose_and_export_canvas).
        confirmed_datasets (List[str]): dataset names to analyze.
        device_centers, device_sizes, device_angles, device_labels: the
            RESTRICTED per-device region list for the first pass (this
            step's active sources + every one of their candidates -- see
            propagation_engine.run_propagation's own construction of this
            list). Every one of these four lists must be the same length.
        canvas_size, canvas_center: the whole canvas's own size/center.
        extra_zoom_targets (Dict[str, Tuple[int, int]]): propagation-
            specific zoom targets to merge in on top of config.
            canvas_zoom_targets -- built by run_propagation each step as
            {"propagation_front": <centroid of this step's active
            source(s)>, "candidate_<idx>": <that candidate's own center>,
            ...} (gated by config.canvas_zoom_propagation_front/
            canvas_zoom_candidates respectively). Explicit config.
            canvas_zoom_targets entries win over these on a name collision,
            matching SuperpositionConfig's own "explicit names win over
            automatic ones" merge rule (see STEP 7's own canvas_zoom_
            all_devices handling).
        output_dir (str): base output directory for THIS canvas's own
            plots/stats (this step's baseline or one candidate's own
            excluded-canvas folder -- see run_propagation).
        extracted_regions_subdir_prefix (str): distinguishes this call's
            region-extraction cache from any other call sharing the same
            canvas_dir (there are none today, since every canvas_dir here
            is single-use, but kept for consistency with SpatialAnalysis
            Config.extracted_regions_subdir's own collision-avoidance
            rationale).
        verbose (bool): forwarded into both SpatialAnalysisConfig calls.
        force_serial (bool): forwarded as SpatialAnalysisConfig.force_
            serial to BOTH calls below -- set True when this analyze_canvas
            call is itself running inside a worker process dispatched by
            an outer pool (propagation_engine.py's own canvas_pool -- see
            _run_canvas_job and PropagationConfig.parallelize_step_
            candidates), so plotting/stats never open a SECOND, nested
            pool in here. False (the default) preserves normal plotting/
            stats parallelism (config.plot_max_workers/stats_max_workers)
            for a caller with no outer parallel layer of its own.

    RETURNS:
        Tuple[Dict, Optional[Dict]]: (device_regions, whole_array_regions),
        each shaped Dict[dataset -> Dict[region_label -> {masked_x,
        masked_y, ...}]] -- see mumax3_parent_spatial_analysis.run_
        pipeline's own RETURNS documentation. device_regions is keyed by
        device_labels; whole_array_regions is keyed by whatever combination
        of "canvas_full_image"/split-tile/zoom-target labels config.canvas_*
        resolves to (see region_extraction.build_canvas_split_centers_sizes/
        build_canvas_zoom_centers_sizes), or None if config.plot_canvas_maps
        is False (the whole-array pass is skipped entirely in that case).
    """
    from mumax3_parent_spatial_analysis import SpatialAnalysisConfig, run_pipeline as run_spatial_analysis_pipeline

    device_config = SpatialAnalysisConfig(
        data_directory=canvas_dir, npz_directory=canvas_dir,
        output_dir=os.path.join(output_dir, "devices"),
        cell_size_x=config.cell_size_x, cell_size_y=config.cell_size_y,
        fields=tuple(confirmed_datasets),
        centers=device_centers, sizes=device_sizes,
        region_labels=device_labels, mask_angles=device_angles,
        mask_mode="file", mask_file=config.mask_file, mask_translation=True,
        extracted_regions_subdir=f"{extracted_regions_subdir_prefix}_devices",
        skip_npz_conversion=True,
        use_cached_npz=config.use_cached_npz, max_workers=config.max_workers,
        plot_spatial_maps=config.plot_spatial_maps and config.plot_device_maps, plot_spatial_limiter=config.plot_spatial_limiter,
        plot_max_workers=config.plot_max_workers, force_serial=force_serial,
        Ms=config.Ms, suptitle=f"{config.suptitle} -- propagation devices",
        compute_stats=config.compute_stats, plot_region_stats=config.plot_region_stats,
        stats_ellipse_size=config.stats_ellipse_size, stats_time_step_ns=config.stats_time_step_ns,
        stats_max_workers=config.stats_max_workers,
        save_stats_csv=config.save_stats_csv, save_stats_csv_per_frame=config.save_stats_csv_per_frame,
        save_stats_csv_all_frames=config.save_stats_csv_all_frames,
        plot_spatial_quiver=config.plot_spatial_quiver and config.plot_device_maps, quiver_window=config.quiver_window,
        save_plot_data_csv=config.save_plot_data_csv and config.plot_device_maps,
        verbose=verbose,
    )
    device_regions_result = run_spatial_analysis_pipeline(device_config)
    device_regions = device_regions_result[0] if config.compute_stats else device_regions_result

    if not config.plot_canvas_maps:  # whole-array pass skipped entirely -- see PropagationConfig.plot_canvas_maps
        return device_regions, None

    from region_extraction import build_canvas_split_centers_sizes, build_canvas_zoom_centers_sizes

    # build_canvas_split_centers_sizes returns a single FULL-canvas region when split is None/1,
    # which would silently add a 3000x3000 plot/CSV to every canvas even with canvas_include_full=False.
    # Only call it when the caller actually asked for a split; the full-canvas fallback below
    # keeps the old behavior ONLY when nothing else (split, zoom target, include_full) was requested.
    if config.canvas_split is not None:
        split_centers, split_sizes, split_labels = build_canvas_split_centers_sizes(
            canvas_size, canvas_center, split=config.canvas_split, verbose=verbose,
        )
        canvas_centers, canvas_sizes, canvas_labels = split_centers, split_sizes, split_labels
    else:
        canvas_centers, canvas_sizes, canvas_labels = [], [], []

    zoom_targets = dict(extra_zoom_targets or {})  # propagation-front/candidate auto targets first
    if config.canvas_zoom_all_devices:
        auto_targets = {f"device_{i}": device_centers[i] for i in range(len(device_centers))}  # NOTE: uses the RESTRICTED device_centers list, not every device in the array -- see PropagationConfig.canvas_zoom_all_devices's own documentation
        auto_targets.update(zoom_targets)
        zoom_targets = auto_targets
    if config.canvas_zoom_targets:  # explicit user targets win over every automatic one above
        zoom_targets.update(config.canvas_zoom_targets)

    if zoom_targets:
        zoom_centers, zoom_sizes, zoom_labels = build_canvas_zoom_centers_sizes(
            zoom_targets, size=config.canvas_zoom_size, verbose=verbose,
        )
        canvas_centers = canvas_centers + zoom_centers
        canvas_sizes = canvas_sizes + zoom_sizes
        canvas_labels = canvas_labels + zoom_labels

    if config.canvas_include_full or not canvas_centers:  # include_full requested, or nothing else requested at all (old behavior: whole canvas)
        canvas_centers = canvas_centers + [canvas_center]
        canvas_sizes = canvas_sizes + [canvas_size]
        canvas_labels = canvas_labels + ["canvas_full_image"]

    whole_array_config = SpatialAnalysisConfig(
        data_directory=canvas_dir, npz_directory=canvas_dir,
        output_dir=os.path.join(output_dir, "whole_array"),
        cell_size_x=config.cell_size_x, cell_size_y=config.cell_size_y,
        fields=tuple(confirmed_datasets),
        centers=canvas_centers, sizes=canvas_sizes, region_labels=canvas_labels,
        plot_origin_cells=canvas_center if config.canvas_center_plots_at_origin else None,
        mask_mode=config.canvas_mask_mode, mask_file=config.canvas_mask_file,
        extracted_regions_subdir=f"{extracted_regions_subdir_prefix}_whole_array",
        skip_npz_conversion=True,
        use_cached_npz=config.use_cached_npz, max_workers=config.max_workers,
        plot_spatial_maps=config.plot_spatial_maps, plot_spatial_limiter=config.plot_spatial_limiter,
        plot_max_workers=config.plot_max_workers, force_serial=force_serial,
        Ms=config.Ms, suptitle=f"{config.suptitle} -- propagation whole array",
        compute_stats=config.canvas_compute_stats,
        plot_spatial_quiver=config.plot_spatial_quiver, quiver_window=config.quiver_window,
        save_plot_data_csv=config.save_plot_data_csv,
        verbose=verbose,
    )
    whole_array_result = run_spatial_analysis_pipeline(whole_array_config)
    whole_array_regions = whole_array_result[0] if config.canvas_compute_stats else whole_array_result

    return device_regions, whole_array_regions


# --------------------------
# ---- Main entry point: the full multi-injection synchronous propagation run ----
def _run_single_propagation(config, regeneration_index: int = 0, verbose: Optional[bool] = None) -> Dict:
    """
    GOAL:
        Run ONE complete multi-injection synchronous propagation (see this
        file's module-level PURPOSE section), writing per-step comparison/
        summary CSVs and (via analyze_canvas) real plots/stats for every
        step, and return the full flip history. This is the single-
        regeneration body run_propagation (below) calls once (config.
        num_regenerations == 1, the default) or config.num_regenerations
        times (greater than 1 -- a fresh array + initial pattern each time,
        for example to sweep magnetization_mode="random" disorder
        realizations -- see PropagationConfig.num_regenerations).

    HOW IT TALKS TO OTHER FILES:
        Calls setup_seed_and_geometry (this file, once), then loops calling
        propagation_neighbors.find_candidate_neighbors, builds one job per
        step (baseline + one per distinct candidate), and dispatches every
        job to _run_canvas_job (this file -- which itself calls
        superposition_operations.run_superposition_multiprocessing,
        utilities_superposition.export_canvas_as_npz_frames/export_canvas_
        as_csv_frames, analyze_canvas, and mumax3_csv_stats.masked_region_
        mean_xy) -- either across canvas_pool (config.parallelize_step_
        candidates, the default) or one at a time in this process. Also
        calls mumax3_csv_stats' hrev_from_field/field_angle_deg_from_xy/
        acute_angle_to_easy_axis/lookup_nearest_coercivity, and
        propagation_utilities' flip_device/origin-set helpers.

    PARAMETERS:
        config (PropagationConfig): see mumax3_parent_propagation.py.
        regeneration_index (int): which regeneration this is (0-based) --
            forwarded to setup_seed_and_geometry (only affects the random
            seed for magnetization_mode="random") and used to give this
            regeneration's own output its own subfolder when config.
            num_regenerations > 1.
        verbose (Optional[bool]): overrides config.verbose when given
            explicitly; otherwise config.verbose is used.

    RETURNS:
        Dict with keys:
            "array": the final geometry.DeviceArray (reflects every
                accepted flip).
            "flat_flip_history": List[int], every flipped device index in
                the order accepted (initial injections first).
            "flip_history_by_step": List[List[int]], the same history
                grouped by synchronous step (index 0 = initial injections).
            "comparison_csv": path to the per-candidate comparison CSV.
            "step_summary_csv": path to the per-step summary CSV.
            "flip_path_visualizations" (only if config.save_flip_path_
                visualizations): {"static", "frames_dir", "gif",
                "gif_status"} -- see propagation_plot.generate_flip_path_
                visualizations' own RETURNS documentation.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section.
    """
    import csv
    from masking import region_crosses_edge  # shared edge-detection utility (see masking.py's own documentation) -- also used by mumax3_parent_superposition.py's STEP 6, as a warning rather than a drop (see that call site)

    v = config.verbose if verbose is None else verbose
    base_output_root = config.step_output_dir or os.path.join(config.output_dir, "propagation")
    output_root = base_output_root if config.num_regenerations <= 1 else os.path.join(base_output_root, f"regen_{regeneration_index:03d}")
    os.makedirs(output_root, exist_ok=True)

    # save the initial magnetization angles
    array, data_struct_all, confirmed_datasets, seed_size, canvas_center = setup_seed_and_geometry(config, regeneration_index=regeneration_index, verbose=v)
    centers_nm, geom_angles_deg, mag_angles_deg = get_device_mesh(array)
    initial_mag_angles_deg = np.array(mag_angles_deg, dtype=float).copy()  # true ground state, before injections or any propagation flip
    n_devices = len(centers_nm)
    if v:
        print(f"[INFO] run_propagation: array built with {n_devices} device(s).")

    # centers_composed: DeviceArray's own canvas-relative centers, converted to ABSOLUTE
    # grid positions -- identical conversion to mumax3_parent_superposition.py's STEP 5.
    centers_composed = [
        (canvas_center[0] + int(round(cx / config.cell_size_x)), canvas_center[1] + int(round(cy / config.cell_size_y)))
        for cx, cy in centers_nm
    ]

    print(f"device_12 absolute grid center: {centers_composed[12]}")
    
    device_size_cells = (int(round(config.device_mask_size_nm[0] / config.cell_size_x)),
                          int(round(config.device_mask_size_nm[1] / config.cell_size_y)))

    coercivity_table = load_coercivity_table(config.coercivity_csv_path, verbose=v) if config.coercivity_csv_path else None

    injection_indices = resolve_injection_indices(centers_nm, injection_indices=config.injection_indices, injection_index=config.injection_index)
    for idx in injection_indices:
        flip_device(array, idx, verbose=v)
    active_sources: Dict[int, set] = {int(i): {origin_label_for_injection(i)} for i in injection_indices}
    flipped_set = set(int(i) for i in injection_indices)
    flat_flip_history: List[int] = list(injection_indices)
    flip_history_by_step: List[List[int]] = [list(injection_indices)]

    comparison_csv = os.path.join(output_root, "propagation_comparisons.csv")
    step_summary_csv = os.path.join(output_root, "propagation_step_summary.csv")
    comparison_fields = [
        "step", "active_source_count", "source_idx", "source_origins",
        "candidate_idx", "slot_label", "magnetization_angle_deg", "geom_angle_EA_deg",
        "Hdemag_x_Oe", "Hdemag_y_Oe", "Hdemag_mag_Oe", "H_reverse_Oe", "H_switch_drive_Oe",
        "field_angle_deg", "delta_theta_deg", "H_angle_vs_mag_deg", "theta_Hc_deg",
        "Hc_angle_used_deg", "Hc_angle_error_deg", "Hc_source", "Hc_Oe", "margin_Oe", "switch",
        "proposed_by_source", "accepted_unique_flip", "collapsed_with_sources", "collapsed_origins",
    ]
    step_summary_fields = [
        "step", "active_sources", "active_source_count", "num_unique_flips",
        "unique_flips", "flip_origins", "num_collapsed_flips", "collapsed_flip_details",
    ]

    # ---- set up the per-step canvas job pool (baseline + one per distinct
    # excluded candidate, see this file's own CANDIDATE-LEVEL PARALLELIZATION
    # section above and PropagationConfig.parallelize_step_candidates) --
    # ONE pool for the whole run (not recreated per step or per job), so the
    # seed field (data_struct_all, potentially large) is only pickled to each
    # worker ONCE via canvas_pool's own initializer, not once per job. Also
    # initialize the SAME globals in this (main) process directly, so the
    # serial fallback path (parallelize_step_candidates=False, or a step
    # with only the baseline job) can call _run_canvas_job without a pool at
    # all, using the identical code path. centers_nm/geom_angles_deg here are
    # the STATIC mesh read once above (before the step loop) -- see
    # _init_canvas_pool_worker's own documentation for why device positions/
    # geometry angles are safe to set once, unlike magnetization angle.
    # canvas_pool is entered into `stack` below (an ExitStack alongside the
    # two CSV files) rather than a separate try/finally, so it is shut down
    # on every exit path -- normal completion, an early `break`, or an
    # exception -- without adding another indentation level around the
    # entire step loop. ----
    canvas_pool = None
    _init_canvas_pool_worker(data_struct_all, centers_nm, geom_angles_deg, config.cell_size_x, config.cell_size_y, verbose=v)
    if config.parallelize_step_candidates:
        from concurrent.futures import ProcessPoolExecutor
        pool_workers = config.canvas_job_max_workers if config.canvas_job_max_workers else (os.cpu_count() or 1)
        canvas_pool = ProcessPoolExecutor(
            max_workers=pool_workers,
            initializer=_init_canvas_pool_worker,
            initargs=(data_struct_all, centers_nm, geom_angles_deg, config.cell_size_x, config.cell_size_y, v),
        )
        if v:
            print(f"[INFO] run_propagation: canvas_pool started with {pool_workers} worker process(es) for candidate-level step parallelism.")

    with ExitStack() as stack:
        f = stack.enter_context(open(comparison_csv, "w", newline="", encoding="utf-8"))
        sf = stack.enter_context(open(step_summary_csv, "w", newline="", encoding="utf-8"))
        if canvas_pool is not None:
            stack.enter_context(canvas_pool)  # ProcessPoolExecutor is itself a context manager -- shutdown(wait=True) runs automatically when this ExitStack unwinds, on any exit path
        writer = csv.DictWriter(f, fieldnames=comparison_fields)
        summary_writer = csv.DictWriter(sf, fieldnames=step_summary_fields)
        writer.writeheader()
        summary_writer.writeheader()

        step = 0
        while active_sources and step < config.max_steps:
            step_dir = os.path.join(output_root, f"step_{step:04d}")
            centers_nm, geom_angles_deg, mag_angles_deg = get_device_mesh(array)  # fresh read for this step

            # ---- candidate discovery, geometry-general, one call per active source ----
            candidates_by_source: Dict[int, List[Tuple[str, int]]] = {}
            needed_indices = set(active_sources.keys())
            for source_idx in active_sources.keys():
                exclude = flipped_set if not config.allow_reflip else set()
                found = find_candidate_neighbors(
                    array, source_idx, n_shells=config.neighbor_shells,
                    exclude=exclude, tolerance_nm=config.neighbor_tolerance_nm, verbose=v,
                )

                # ---- drop any candidate whose OWN mask footprint would cross the seed
                # canvas's edge -- see masking.region_crosses_edge's own documentation for
                # the underlying mechanism (masking.apply_mask NaN-pads an edge-crossing
                # region rather than corrupting it, but the resulting masked mean is still
                # averaged over LESS than the candidate's true full footprint, silently,
                # which is not a data-integrity bug but is not what this engine's switching
                # decision wants either). See PropagationConfig.candidate_edge_margin_cells'
                # own documentation -- ported from the parallel team's cascade script (which
                # filtered candidates against canvas_struct's own extent) at Connor's
                # request. The detection itself (region_crosses_edge) is shared with the
                # static superposition/spatial-analysis pipeline (mumax3_parent_
                # superposition.py's STEP 6 uses the same function, but only to WARN rather
                # than drop -- see that call site's own comment for why the two pipelines
                # make different choices here). ----
                kept, dropped = [], []
                for slot_label, cand_idx in found:
                    crosses, overhang = region_crosses_edge(
                        source_shape=(seed_size[1], seed_size[0]),  # (rows, cols) -- seed_size itself is (width, height) = (cols, rows)
                        center=centers_composed[cand_idx], size=device_size_cells,
                        units='grid', margin_cells=config.candidate_edge_margin_cells, verbose=v,
                    )
                    if crosses:
                        dropped.append((slot_label, cand_idx, overhang))
                    else:
                        kept.append((slot_label, cand_idx))
                if dropped and v:
                    print(f"[WARNING] run_propagation: step {step}: source {source_idx}: dropped {len(dropped)} "
                          f"candidate(s) whose mask footprint would cross the canvas edge: {dropped}")

                candidates_by_source[source_idx] = kept
                needed_indices.update(cand_idx for _label, cand_idx in kept)

            if not any(candidates_by_source.values()):  # no active source has ANY candidate left -- propagation stops
                summary_writer.writerow({"step": step, "active_sources": ";".join(str(x) for x in sorted(active_sources)),
                                          "active_source_count": len(active_sources), "num_unique_flips": 0,
                                          "unique_flips": "", "flip_origins": "", "num_collapsed_flips": 0, "collapsed_flip_details": ""})
                sf.flush()
                if v:
                    print(f"[INFO] run_propagation: step {step}: no active source has any candidate; stopping.")
                break

            restricted_indices = sorted(needed_indices)
            restricted_centers = [centers_composed[i] for i in restricted_indices]
            restricted_sizes = [device_size_cells] * len(restricted_indices)
            restricted_angles = [float(geom_angles_deg[i]) for i in restricted_indices]
            restricted_labels = [f"device_{i}" for i in restricted_indices]
            label_by_index = dict(zip(restricted_indices, restricted_labels))

            # ---- auto canvas-level zoom targets for this step: the propagation front
            # (centroid of this step's own active source(s)) and every distinct
            # candidate device -- see PropagationConfig.canvas_zoom_propagation_front/
            # canvas_zoom_candidates and analyze_canvas's own documentation. ----
            distinct_candidates = sorted({cand_idx for found in candidates_by_source.values() for _label, cand_idx in found})
            zoom_targets_this_step: Dict[str, Tuple[int, int]] = {}
            if config.canvas_zoom_propagation_front:
                front_indices = sorted(active_sources.keys())
                front_x = int(round(sum(centers_composed[i][0] for i in front_indices) / len(front_indices)))
                front_y = int(round(sum(centers_composed[i][1] for i in front_indices) / len(front_indices)))
                zoom_targets_this_step["propagation_front"] = (front_x, front_y)
            if config.canvas_zoom_candidates:
                for cand_idx in distinct_candidates:
                    zoom_targets_this_step[f"candidate_{cand_idx}"] = centers_composed[cand_idx]

            # ---- (b)+(c) baseline pass (no exclusion) + one excluded pass per
            # DISTINCT candidate proposed this step -- each is an INDEPENDENT job
            # given this step's starting array state (see this file's own
            # CANDIDATE-LEVEL PARALLELIZATION section, near the top, for exactly
            # why: every job carries its own private mag_angles_deg copy, the
            # live `array` is never mutated for exclusion, and every job writes
            # to its own distinct canvas_dir/output_dir). Built here, in this
            # single-threaded main process, then either dispatched across
            # canvas_pool or run one at a time -- both paths call the identical
            # _run_canvas_job, so they cannot silently diverge in what they
            # compute (see PropagationConfig.parallelize_step_candidates).
            #
            # dispatch_parallel is computed ONCE, here, and used for BOTH every
            # job's own "force_serial_analysis" flag AND the actual dispatch
            # decision below -- a single source of truth, so a job's plotting/
            # stats behavior can never disagree with whether it actually ran
            # inside canvas_pool or not (see analyze_canvas's own force_serial
            # parameter and SpatialAnalysisConfig.force_serial's own
            # documentation for why that distinction matters: plotting/stats
            # opening their OWN pool INSIDE a job that is already one of several
            # running concurrently across canvas_pool would oversubscribe CPUs
            # rather than help). ----
            dispatch_parallel = canvas_pool is not None and (1 + len(distinct_candidates)) > 1

            baseline_dir = os.path.join(step_dir, "baseline")
            jobs: List[Dict] = [{
                "job_type": "baseline", "cand_idx": None, "own_label": None,
                "mag_angles_deg": mag_angles_deg,  # this step's un-excluded mesh -- read-only; every candidate job below gets its OWN separate copy, so nothing here is shared-and-mutated
                "canvas_dir": os.path.join(baseline_dir, "canvas"), "output_dir": baseline_dir,
                "extracted_regions_subdir_prefix": "baseline",
                "confirmed_datasets": confirmed_datasets,
                "restricted_centers": restricted_centers, "restricted_sizes": restricted_sizes,
                "restricted_angles": restricted_angles, "restricted_labels": restricted_labels,
                "canvas_size": seed_size, "canvas_center": canvas_center,
                "extra_zoom_targets": zoom_targets_this_step, "config": config, "verbose": v,
                "force_serial_analysis": dispatch_parallel,
            }]
            for cand_idx in distinct_candidates:
                excl_dir = os.path.join(step_dir, f"excluded_{cand_idx:04d}")
                mag_angles_excl = mag_angles_deg.copy()  # THIS job's own private copy -- never shared with, or mutated by, any other job (baseline's or another candidate's)
                mag_angles_excl[cand_idx] = np.nan
                jobs.append({
                    "job_type": "candidate", "cand_idx": int(cand_idx), "own_label": label_by_index[cand_idx],
                    "mag_angles_deg": mag_angles_excl,
                    "canvas_dir": os.path.join(excl_dir, "canvas"), "output_dir": excl_dir,
                    "extracted_regions_subdir_prefix": f"excl{cand_idx}",
                    "confirmed_datasets": confirmed_datasets,
                    "restricted_centers": restricted_centers, "restricted_sizes": restricted_sizes,
                    "restricted_angles": restricted_angles, "restricted_labels": restricted_labels,
                    "canvas_size": seed_size, "canvas_center": canvas_center,
                    "extra_zoom_targets": zoom_targets_this_step, "config": config, "verbose": v,
                    "force_serial_analysis": dispatch_parallel,
                })

            if dispatch_parallel:  # candidate-level parallelism: dispatch every job (baseline + every distinct candidate) across canvas_pool at once
                job_results = list(canvas_pool.map(_run_canvas_job, jobs))
            else:  # parallelize_step_candidates=False, or only the baseline job exists this step (should not normally happen -- distinct_candidates is non-empty by the time this line runs, see the "no active source has any candidate" break above) -- same _run_canvas_job, called directly, one at a time
                job_results = [_run_canvas_job(job) for job in jobs]

            candidate_field_cache: Dict[int, Tuple[float, float]] = {}
            for res in job_results:
                if res["job_type"] == "candidate":
                    candidate_field_cache[res["cand_idx"]] = (res["Hx_Oe"], res["Hy_Oe"])

            # ---- switching decision for every (source, candidate) pair, using the cached excluded field ----
            all_rows_this_step: List[Dict] = []
            proposed_rows_by_source: Dict[int, List[Dict]] = {}
            proposed_rows_by_candidate: Dict[int, List[Dict]] = defaultdict(list)

            for source_idx, source_origins in active_sources.items():
                results: List[Dict] = []
                for slot_label, cand_idx in candidates_by_source[source_idx]:
                    Hx_Oe, Hy_Oe = candidate_field_cache[cand_idx]
                    m_angle = float(mag_angles_deg[cand_idx])
                    geom_angle = float(geom_angles_deg[cand_idx])

                    # Switching criterions: choose between Hrev or Hmag
                    H_reverse = hrev_from_field(Hx_Oe, Hy_Oe, m_angle)
                    Hmag = float(np.sqrt(Hx_Oe ** 2 + Hy_Oe ** 2))
                    H_switch_drive = Hmag if config.switch_field_metric == "Hmag" else -H_reverse
                    field_angle_deg = field_angle_deg_from_xy(Hx_Oe, Hy_Oe)
                    delta_theta_deg = signed_angle_difference_deg(field_angle_deg, geom_angle)
                    H_angle_vs_mag_deg = signed_angle_difference_deg(field_angle_deg, m_angle)  # field direction relative to the candidate's OWN current magnetization (not its easy axis, unlike delta_theta_deg above) -- diagnostic only, ported from the parallel team's cascade script at Connor's request; not used in the switching decision itself
                    theta_Hc_deg = acute_angle_to_easy_axis(field_angle_deg, geom_angle)
                    Hc_Oe, Hc_angle_used, Hc_angle_err, Hc_source = lookup_nearest_coercivity(
                        theta_Hc_deg, coercivity_table, config.default_coercivity_oe, verbose=v,
                    )
                    margin = H_switch_drive - Hc_Oe
                    switch = (H_reverse < 0.0) and (H_switch_drive > Hc_Oe)

                    row = {
                        "step": step, "active_source_count": len(active_sources),
                        "source_idx": source_idx, "source_origins": format_origin_set(source_origins),
                        "candidate_idx": cand_idx, "slot_label": slot_label,
                        "magnetization_angle_deg": m_angle, "geom_angle_EA_deg": geom_angle,
                        "Hdemag_x_Oe": Hx_Oe, "Hdemag_y_Oe": Hy_Oe, "Hdemag_mag_Oe": Hmag,
                        "H_reverse_Oe": H_reverse, "H_switch_drive_Oe": H_switch_drive,
                        "field_angle_deg": field_angle_deg, "delta_theta_deg": delta_theta_deg,
                        "H_angle_vs_mag_deg": H_angle_vs_mag_deg,
                        "theta_Hc_deg": theta_Hc_deg, "Hc_angle_used_deg": Hc_angle_used,
                        "Hc_angle_error_deg": Hc_angle_err, "Hc_source": Hc_source, "Hc_Oe": Hc_Oe,
                        "margin_Oe": margin, "switch": "YES" if switch else "NO",
                        "proposed_by_source": "NO", "accepted_unique_flip": "NO",
                        "collapsed_with_sources": "", "collapsed_origins": "",
                    }
                    results.append(row)
                    all_rows_this_step.append(row)

                eligible = [r for r in results if r["switch"] == "YES"]
                eligible.sort(key=lambda r: r["H_switch_drive_Oe"], reverse=True)

                # ---- how many of this source's eligible candidates flip this step -- see
                # PropagationConfig.flip_mode/branch_first_step's own documentation. ----
                if not eligible:
                    chosen = []
                elif (config.branch_first_step and step == 0) or config.flip_mode == "all_eligible":
                    chosen = eligible  # every eligible candidate flips, evaluated synchronously against this step's own pre-flip field state
                else:
                    chosen = [eligible[0]]  # flip_mode == "max" (default): only the single strongest eligible candidate

                proposed_rows_by_source[int(source_idx)] = chosen
                for r in chosen:
                    r["proposed_by_source"] = "YES"
                    proposed_rows_by_candidate[int(r["candidate_idx"])].append(r)

                if config.save_neighbor_field_figures:  # LIVE call, using the array's CURRENT state -- see propagation_plot.py's own module-level LIMITATIONS section for why this must happen here, before this step's accepted flips are applied below
                    fig_path = os.path.join(step_dir, config.neighbor_field_fig_dirname, f"source_{int(source_idx):04d}_neighbor_field_comparison.png")
                    plot_neighbor_field_comparison(
                        array=array, results=results, source_idx=int(source_idx), step=step, output_path=fig_path,
                        best_idx=[int(r["candidate_idx"]) for r in chosen] if chosen else None,
                        show_full_array_background=config.neighbor_fig_show_full_array_background,
                        arrow_length_nm=config.neighbor_fig_arrow_length_nm, mag_arrow_length_nm=config.neighbor_fig_mag_arrow_length_nm,
                        margin_nm=config.neighbor_fig_margin_nm, verbose=v,
                    )

            # ---- collapse duplicate proposals onto one accepted flip per target ----
            unique_flip_indices: List[int] = []
            next_active_sources: Dict[int, set] = {}
            collapsed_details: List[str] = []
            for target_idx, proposed_rows in sorted(proposed_rows_by_candidate.items()):
                proposed_source_indices = [int(r["source_idx"]) for r in proposed_rows]
                merged_origins = merge_origin_sets(active_sources[int(r["source_idx"])] for r in proposed_rows)
                unique_flip_indices.append(int(target_idx))
                next_active_sources[int(target_idx)] = merged_origins
                collapsed_sources_str = ";".join(str(x) for x in proposed_source_indices)
                collapsed_origins_str = format_origin_set(merged_origins)
                for r in proposed_rows:
                    r["accepted_unique_flip"] = "YES"
                    r["collapsed_with_sources"] = collapsed_sources_str
                    r["collapsed_origins"] = collapsed_origins_str
                if len(proposed_rows) > 1:
                    collapsed_details.append(f"target {target_idx} from sources {collapsed_sources_str} origins {collapsed_origins_str}")

            for row in all_rows_this_step:
                writer.writerow(row)
            f.flush()

            if not unique_flip_indices:
                summary_writer.writerow({"step": step, "active_sources": ";".join(str(x) for x in sorted(active_sources)),
                                          "active_source_count": len(active_sources), "num_unique_flips": 0,
                                          "unique_flips": "", "flip_origins": "", "num_collapsed_flips": 0, "collapsed_flip_details": ""})
                sf.flush()
                if v:
                    print(f"[INFO] run_propagation: step {step}: no active source proposed an eligible flip; stopping.")
                break

            for flip_idx in unique_flip_indices:
                flip_device(array, flip_idx, verbose=v)
                flipped_set.add(int(flip_idx))
            flat_flip_history.extend(unique_flip_indices)
            flip_history_by_step.append(list(unique_flip_indices))

            summary_writer.writerow({
                "step": step, "active_sources": ";".join(str(x) for x in sorted(active_sources)),
                "active_source_count": len(active_sources), "num_unique_flips": len(unique_flip_indices),
                "unique_flips": ";".join(str(x) for x in unique_flip_indices),
                "flip_origins": ";".join(f"{idx}:{format_origin_set(next_active_sources[int(idx)])}" for idx in unique_flip_indices),
                "num_collapsed_flips": len(collapsed_details), "collapsed_flip_details": " | ".join(collapsed_details),
            })
            sf.flush()

            if v:
                print(f"[INFO] run_propagation: step {step}: accepted flips = {unique_flip_indices}")
            active_sources = next_active_sources
            step += 1

    result = {
        "array": array,
        "flat_flip_history": flat_flip_history,
        "flip_history_by_step": flip_history_by_step,
        "comparison_csv": comparison_csv,
        "step_summary_csv": step_summary_csv,
        "initial_mag_angles_deg": initial_mag_angles_deg,
    }

    if config.save_flip_path_visualizations:  # ONE call, after the fact -- only geometry is read, which is stable across the whole run (see propagation_plot.py's own module-level LIMITATIONS section)
        flip_path_output_dir = os.path.join(output_root, "flip_path_visualizations")
        flip_path_paths = generate_flip_path_visualizations(
            array=array, result=result, output_dir=flip_path_output_dir,
            show_candidates=config.flip_path_show_candidates, show_accepted_arrows=config.flip_path_show_accepted_arrows,
            make_gif=config.flip_path_make_gif, verbose=v,
        )
        result["flip_path_visualizations"] = flip_path_paths
        if v:
            print(f"[INFO] run_propagation: flip-path visualizations -> {flip_path_paths}")

    return result


# --------------------------
# ---- Public entry point: single run, or config.num_regenerations independent runs ----
def run_propagation(config, verbose: Optional[bool] = None):
    """
    GOAL:
        Run propagation once (config.num_regenerations == 1, the default),
        or config.num_regenerations independent times, each with a freshly
        built array + initial magnetization pattern (see setup_seed_and_
        geometry's own regeneration_index parameter) -- the propagation
        analog of SuperpositionConfig.num_regenerations, most useful
        combined with magnetization_mode="random" to sweep disorder
        realizations. This is the function mumax3_parent_propagation.
        run_propagation forwards to directly.

    HOW IT TALKS TO OTHER FILES:
        Calls _run_single_propagation (this file) once per regeneration.

    PARAMETERS:
        config (PropagationConfig): see mumax3_parent_propagation.py.
        verbose (Optional[bool]): overrides config.verbose when given
            explicitly; otherwise config.verbose is used.

    RETURNS:
        If config.num_regenerations <= 1 (the default): a single Dict --
        see _run_single_propagation's own RETURNS documentation.
        If config.num_regenerations > 1: List[Dict], one entry per
        regeneration, in order -- each shaped identically, and each under
        its own output_root/regen_<i> subfolder (see _run_single_
        propagation).
    """
    v = config.verbose if verbose is None else verbose
    if config.num_regenerations <= 1:
        return _run_single_propagation(config, regeneration_index=0, verbose=v)

    results = []
    for i in range(config.num_regenerations):
        if v:
            print(f"[INFO] run_propagation: regeneration {i + 1}/{config.num_regenerations}...")
        results.append(_run_single_propagation(config, regeneration_index=i, verbose=v))
    return results