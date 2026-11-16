r"""
File Name: mumax3_parent_propagation.py
Author: Connor Sullivan, Starr Feng, Zoey Yong
Date: 2026.08.17
Revision: v2

============================================================
PURPOSE
============================================================
Parent script for multi-injection synchronous nearest-neighbor propagation
across an ASI array of ANY geometry (square, pinwheel, kagome -- see
src/_propagation/propagation_neighbors.py). PropagationConfig subclasses
mumax3_parent_superposition.SuperpositionConfig directly, so every geometry/
seed/masking field documented on that class (device_length_nm, nn_x_nm,
sublattice_dims, device_rotations, translations, mask_file, ...) behaves
identically here -- this file only adds the propagation-specific fields
below. The actual per-step engine lives in src/_propagation/
propagation_engine.py; see that file's own module-level documentation for
exactly what happens each step.

============================================================
BUILDING A CUSTOM PROPAGATION SCRIPT ON TOP OF THIS ONE
============================================================
Same pattern as mumax3_parent_superposition.py: copy this file, keep
PropagationConfig (or add fields to it), and either call run_propagation
as-is or call src/_propagation/propagation_engine.py's individual pieces
(setup_seed_and_geometry, compose_and_export_canvas, analyze_canvas)
directly in a different order/combination. The same 11 worked geometry
examples at the bottom of mumax3_parent_superposition.py apply here
unchanged -- PropagationConfig's geometry fields are the exact same fields.

============================================================
LIMITATIONS
============================================================
- See src/_propagation/propagation_engine.py's own module-level
  LIMITATIONS section for what this run_pipeline-style function does NOT
  yet do (flip-path/neighbor-field visualization -- pending
  propagation_plot.py) and the per-step output-volume caveat.
"""

import os  # path construction
import sys  # sys.path manipulation, for add_script_folders
from dataclasses import dataclass
from typing import List, Optional
import time
import numpy as np
import pandas as pd

# start timer 
start_time = time.time()



# --------------------------
# ---- Add this codebase's own folders to sys.path (edit ACTIVE_SCRIPT_DIRS for your machine) ----
def add_script_folders(script_paths, levels_up=2, verbose=False):
    """
    GOAL:
        Identical implementation/rationale to mumax3_parent_superposition.
        add_script_folders -- kept as its own copy here rather than a
        shared import, since this parent script needs to work standalone
        before any of this codebase's own modules are on sys.path yet (see
        that function's own documentation for the full rationale, which
        applies unchanged here).
    """
    for original_path in script_paths:
        path = os.path.abspath(original_path)
        drive = os.path.splitdrive(path)[0]
        if drive and not os.path.exists(drive):
            if verbose:
                print(f"[DEBUG] add_script_folders: drive not found, skipping: {drive} for path {path}")
            continue
        for _ in range(levels_up + 1):
            if os.path.exists(path) and path not in sys.path:
                sys.path.append(path)
                if verbose:
                    print(f"[DEBUG] add_script_folders: added '{path}'.")
            path = os.path.dirname(path)


ACTIVE_SCRIPT_DIRS = [
    r"C:\Users\abcdz\OneDrive\Documents\ASI Code Parallel\demo_spatial_superposition_propagation_updated_PERF\demo_spatial_superposition_propagation_updated\src\_mumax3",
    r"C:\Users\abcdz\OneDrive\Documents\ASI Code Parallel\demo_spatial_superposition_propagation_updated_PERF\demo_spatial_superposition_propagation_updated\src\_superposition",
    r"C:\Users\abcdz\OneDrive\Documents\ASI Code Parallel\demo_spatial_superposition_propagation_updated_PERF\demo_spatial_superposition_propagation_updated\src\_propagation",
]


# --------------------------
# ---- Everything a user edits lives here ----
@dataclass
class PropagationConfig:
    """
    GOAL:
        Hold every setting run_propagation needs. This does NOT subclass
        SuperpositionConfig directly (dataclass field-ordering makes mixing
        this file's own required-looking fields awkward against that
        class's own defaults) -- instead, construct a SuperpositionConfig-
        shaped set of geometry/seed fields identically by NAME, mirrored
        field-for-field from that class, so copying a working
        SuperpositionConfig call and changing the class name to
        PropagationConfig (and adding the propagation-only fields below)
        just works. See SuperpositionConfig's own field-by-field
        documentation in mumax3_parent_superposition.py for every field
        through sublattice/masking -- not re-documented line by line here
        to avoid the two copies silently drifting apart; only the fields
        NEW to this class (propagation-specific) are documented below.
    """
    # ==== Mirrored from SuperpositionConfig -- see that class's own docs ====
    data_directory: str
    output_dir: Optional[str] = None
    npz_directory: Optional[str] = None
    cell_size_x: float = 1.0
    cell_size_y: float = 1.0
    sim_size_x: float = 3000.0
    sim_size_y: float = 3000.0
    seed_frame_index: int = 0
    fields: tuple = ("m_full", "B_demag")
    device_length_nm: float = 70
    device_width_nm: float = 56
    nn_x_nm: float = 140
    nn_y_nm: float = 140
    sublattice_dims: dict = None
    rotation_angles: Optional[List[float]] = None
    device_rotations: Optional[List[float]] = None
    translations: Optional[dict] = None
    magnetization_mode: str = "identical"
    magnetization_base_angle: Optional[dict] = None
    sublattice_termination_mode: Optional[str] = "fixed_row_col"
    sublattice_termination: Optional[dict] = None
    indices_to_remove: Optional[List[int]] = None
    mask_file: str = ""
    device_mask_size_nm: Optional[tuple] = None
    use_cached_npz: bool = False
    max_workers: int = 2
    plot_spatial_maps: bool = False
    plot_spatial_limiter: Optional[int] = None
    plot_max_workers: int = 8  # see SpatialAnalysisConfig.plot_max_workers -- forwarded straight through to both the per-device pass and the whole-array pass, every canvas (baseline + every excluded candidate), every step. Only takes effect when this call is NOT running inside canvas_pool (see analyze_canvas's own force_serial parameter, and PropagationConfig.parallelize_step_candidates) -- when it IS, plotting is forced serial regardless of this value, to avoid a nested pool inside an already-parallel worker.
    Ms: float = 800e3
    suptitle: str = "ASI Propagation"
    compute_stats: bool = False
    plot_region_stats: bool = True
    stats_ellipse_size: Optional[tuple] = None
    stats_time_step_ns: float = 0.1
    stats_max_workers: int = 1
    save_stats_csv: bool = False
    save_stats_csv_per_frame: bool = False
    save_stats_csv_all_frames: bool = False
    plot_spatial_quiver: bool = False
    quiver_window: tuple = (2, 2)
    save_plot_data_csv: bool = False
    plot_device_maps: bool = False  # gates plots/CSVs for the restricted per-device pass (devices/ folders) -- see analyze_canvas. The per-device pass still RUNS either way (the switching decision needs its extracted fields); this only controls whether it also writes color maps/quiver plots/plot CSVs. The whole-array pass (plot_canvas_maps) is unaffected.
    canvas_center_plots_at_origin: bool = False
    num_regenerations: int = 1  # how many independent propagation RUNS to perform, each with a freshly-built array + initial magnetization pattern -- the propagation analog of SuperpositionConfig.num_regenerations (which re-randomizes and re-exports a single static composition N times; here, each "regeneration" instead means "run the whole synchronous propagation loop to completion once more"). 1 (default) = a single run, returning the plain result dict propagation_engine.run_propagation always has; greater than 1 = a list of that many result dicts, one per regeneration, each under its own step_output_dir/regen_<i> subfolder -- see mumax3_parent_propagation.run_propagation's own RETURNS documentation. Only meaningful in combination with magnetization_mode="random" (otherwise every regeneration builds an identical initial pattern and just repeats the same run).
    random_seed_start: int = 0  # seed for regeneration 0; regeneration i uses random_seed_start + i (only meaningful for magnetization_mode="random") -- identical role to SuperpositionConfig.random_seed_start
    export_canvas_csv: bool = False  # ALSO export every composed canvas (baseline AND every per-candidate excluded canvas, every step) as a human-viewable CSV -- see SuperpositionConfig.export_canvas_csv's own identical documentation. Off by default -- a propagation run already recomposes many canvases (baseline + one per distinct candidate, every step), so this option multiplies output file count fast; consider a short max_steps/neighbor_shells run before turning this on.

    # ==== Canvas-level (whole-array) analysis pass -- mirrors SuperpositionConfig's
    # own STEP 7 fields exactly (same names, same meaning), applied here to EVERY
    # step's baseline canvas AND every step's per-candidate excluded canvas, not just
    # once for a single static composition. See propagation_engine.analyze_canvas's
    # own documentation for exactly how these combine with the propagation-specific
    # auto-zoom targets (propagation front + every candidate this step -- see
    # canvas_zoom_propagation_front/canvas_zoom_candidates below) each step. ====
    plot_canvas_maps: bool = False  # if False, the whole-array analysis pass is skipped entirely each step (only the restricted per-device NN+source pass still runs) -- see analyze_canvas
    canvas_split: Optional[int] = None  # tile the canvas into an NxN grid -- see region_extraction.build_canvas_split_centers_sizes
    canvas_zoom_targets: Optional[dict] = None  # {name: (cx, cy)} EXTRA hand-picked absolute-grid zoom targets, merged on top of the automatic propagation-front/candidate targets below (explicit names win over automatic ones, matching SuperpositionConfig's own merge rule)
    canvas_zoom_size: tuple = (200, 200)  # (width, height) grid cells around each zoom target
    canvas_zoom_all_devices: bool = False  # auto-populate a zoom target for EVERY device in the array, every step -- expensive (one more region per device, every baseline AND every excluded canvas), off by default; canvas_zoom_propagation_front/canvas_zoom_candidates below usually give the visual context actually needed for propagation verification without this
    canvas_include_full: bool = False  # ALSO include one region covering the whole canvas, regardless of split/zoom above
    canvas_mask_mode: str = "none"  # 'none' | 'ellipse' | 'file' -- applied to every canvas-level region (split tiles, zoom targets, full image) above
    canvas_mask_file: Optional[str] = None  # required if canvas_mask_mode='file'
    canvas_compute_stats: bool = False  # weighted circular stats over canvas-level regions (many devices at once), independent from compute_stats above (which is per-device, in the restricted NN+source pass)
    canvas_zoom_propagation_front: bool = False  # auto-add ONE zoom target, "propagation_front", centered on the centroid of this step's own active source(s) -- see analyze_canvas's own documentation
    canvas_zoom_candidates: bool = False  # auto-add one zoom target per DISTINCT candidate device checked this step ("candidate_<idx>") -- see analyze_canvas's own documentation
    verbose: bool = False  # master switch: forwarded into every propagation_engine step below (identical role to SuperpositionConfig.verbose)

    # ==== Propagation-specific (NEW to this class) ====
    injection_indices: Optional[List[int]] = None  # explicit multi-injection device index list; takes priority over injection_index
    injection_index: Optional[int] = None  # single-injection device index; used only if injection_indices is None (None = device closest to array origin)
    neighbor_shells: int = 2  # how many distance shells (nearest, next-nearest, ...) propagation_neighbors.find_candidate_neighbors checks -- geometry-general; see that file's own documentation for what this means per geometry (2 reproduces the original square-ASI script's "8 candidates" as an emergent result, not a hardcoded count)
    neighbor_tolerance_nm: float = 1e-6  # position-match tolerance when resolving a neighbor-shell vector back to a device index
    allow_reflip: bool = False  # if False, an already-flipped device is never proposed as a candidate again (exclude=flipped_set every step); if True, candidates are drawn from every device except the source itself
    max_steps: int = 50  # hard cap on synchronous propagation steps
    flip_mode: str = "max"  # 'max' (only the single strongest eligible candidate per active source flips each step -- the original behavior) | 'all_eligible' (every candidate exceeding coercivity for a given source flips in the same step, evaluated synchronously against that step's own pre-flip field state -- not a serial re-check after each individual flip). Ported from the parallel team's cascade script (originally FLIP_MODE) at Connor's request.
    branch_first_step: bool = False  # if True, step 0 (the initial injection's own first proposal round) always flips every eligible candidate, regardless of flip_mode -- lets the initial injection branch out before flip_mode='max' settles into single-strongest-candidate stepping for later steps. Ignored if flip_mode is already 'all_eligible' (every step already behaves this way). Ported from the parallel team's cascade script (originally BRANCH_FIRST_STEP) at Connor's request.
    candidate_edge_margin_cells: int = 0  # a candidate whose OWN mask footprint (device_mask_size_nm, centered at that candidate's own position) would cross the seed canvas's edge by more than this many grid cells is dropped from consideration this step, with a warning -- see run_propagation's own candidate-discovery step and masking.region_crosses_edge (the shared detection utility this check is built on -- also used, as a warning rather than a drop, by mumax3_parent_superposition.py's STEP 6). masking.apply_mask already NaN-pads an edge-crossing region rather than corrupting it with invalid data, so this is not a data-integrity guard -- it is a guard against silently averaging a candidate's switching field over LESS than its true full footprint (or, in the extreme, over zero valid cells) with nothing flagging that this happened. Ported from the parallel team's cascade script (which filtered candidates against canvas_struct's own extent) at Connor's request. 0 (default) = a candidate is dropped as soon as ANY part of its own footprint would cross the edge; increase to tolerate a small overhang.
    coercivity_csv_path: Optional[str] = None  # path to a 2-column (angle_deg, Hc_Oe) CSV, 0-90 deg -- see mumax3_csv_stats.load_coercivity_table. None = use default_coercivity_oe for every candidate.
    default_coercivity_oe: float = 10  # fallback Hc when coercivity_csv_path is None or a candidate's field angle is undefined
    switch_field_metric: str = "Hmag"  # 'Hmag' = switch iff the field opposes m (Hrev < 0) and |H| > Hc(theta) | 'Hrev' = switch iff -Hrev > Hc(theta) (field component along the easy axis)
    b_to_oe: float = 1e4  # B_demag comes out of mumax3 in Tesla; multiplier applied when extracting each candidate's masked mean field (see mumax3_csv_stats.masked_region_mean_xy)
    step_output_dir: Optional[str] = None  # base directory for every step's own output; default: output_dir/propagation
    parallelize_step_candidates: bool = True  # if True (default), every step's baseline canvas + every distinct excluded-candidate canvas are dispatched across a persistent per-run ProcessPoolExecutor (propagation_engine.py's own canvas_pool) instead of composed one at a time -- see that file's own CANDIDATE-LEVEL PARALLELIZATION section for the isolation guarantees this relies on (every job gets its own private magnetization-angle copy; the live array is never mutated for exclusion). Set False to fall back to the original fully-serial behavior (same _run_canvas_job either way, so results are identical) -- useful for debugging, or on a machine where process-pool startup itself is the bottleneck (for example, very few candidates per step, or a very constrained CPU count).
    canvas_job_max_workers: Optional[int] = None  # worker process count for canvas_pool, when parallelize_step_candidates is True. None (default) = os.cpu_count(). A step's own job count (1 baseline + however many distinct candidates that step has) varies step to step, so this is set ONCE, for the whole pool's lifetime, rather than per step -- excess workers beyond a given step's own job count simply sit idle for that step, not spawned/torn down.

    # ==== Visualization -- see src/_propagation/propagation_plot.py ====
    save_neighbor_field_figures: bool = True  # one figure per active source per step (plot_neighbor_field_comparison) -- called LIVE, inside the stepping loop
    neighbor_field_fig_dirname: str = "neighbor_field_visualizations"  # subfolder name, under each step's own output dir
    neighbor_fig_show_full_array_background: bool = True  # faintly draw every device in the array as context, not just the source/candidates
    neighbor_fig_arrow_length_nm: float = 55.0  # fixed on-plot length for the blue H_demag-direction arrow
    neighbor_fig_mag_arrow_length_nm: float = 38.0  # fixed on-plot length for the black magnetization-direction arrow
    neighbor_fig_margin_nm: float = 220.0  # padding around the source + candidates when setting plot limits
    save_flip_path_visualizations: bool = True  # static + per-generation frames + GIF (generate_flip_path_visualizations) -- called ONCE, after the run completes
    flip_path_show_candidates: bool = False  # faint dotted arrows for every candidate checked (green = eligible, red = not)
    flip_path_show_accepted_arrows: bool = False  # thick black arrows for actually-accepted source -> candidate flips
    flip_path_make_gif: bool = True  # assemble the per-generation frames into an animated GIF, if Pillow is available (see propagation_plot._try_make_gif)

    def __post_init__(self):
        """
        GOAL:
            Resolve default paths and validate required fields -- the same
            checks SuperpositionConfig.__post_init__ performs, since this
            class mirrors that one's geometry/masking fields by name (see
            this class's own GOAL section) rather than by inheritance.

        RAISES:
            ValueError if mask_file is not set, or sublattice_dims is not
            given.
        """
        if self.output_dir is None:
            self.output_dir = os.path.join(self.data_directory, "mumax3_analysis_output")
        if self.npz_directory is None:
            self.npz_directory = os.path.join(self.data_directory, "temp", "npz")
        if not self.mask_file:
            raise ValueError("config.mask_file must be set -- the per-device mask template (a single-device seed .npz).")
        if not self.sublattice_dims:
            raise ValueError("config.sublattice_dims must be set -- see geometry.DeviceArray's own documentation.")
        if self.flip_mode not in ("max", "all_eligible"):
            raise ValueError(f"config.flip_mode must be 'max' or 'all_eligible'; got '{self.flip_mode}'.")
        if self.device_mask_size_nm is None:
            if isinstance(self.device_length_nm, dict) or isinstance(self.device_width_nm, dict):
                raise ValueError(
                    "config.device_mask_size_nm must be set explicitly when device_length_nm/"
                    "device_width_nm are per-sublattice dicts -- see SuperpositionConfig's own "
                    "identical requirement in mumax3_parent_superposition.py."
                )
            self.device_mask_size_nm = (self.device_length_nm, self.device_width_nm)


# --------------------------
# ---- The pipeline itself ----
def run_propagation(config: PropagationConfig, verbose: Optional[bool] = None):
    """
    GOAL:
        Run multi-injection synchronous nearest-neighbor propagation. Thin
        wrapper around src/_propagation/propagation_engine.run_propagation
        -- see that function's own documentation for exactly what each step
        does.

    PARAMETERS:
        config (PropagationConfig): every setting for this run.
        verbose (Optional[bool]): overrides config.verbose (inherited via
            the mirrored-field convention above -- add verbose: bool =
            False to PropagationConfig if not already present) when given.

    RETURNS:
        Dict (config.num_regenerations <= 1, the default) or List[Dict]
        (config.num_regenerations > 1) -- see propagation_engine.
        run_propagation's own RETURNS documentation.
    """
    from propagation_engine import run_propagation as _run_propagation
    return _run_propagation(config, verbose=verbose)


if __name__ == "__main__":
    add_script_folders(ACTIVE_SCRIPT_DIRS, levels_up=0, verbose=True)
    data_directory = r"C:\Users\abcdz\OneDrive\Documents\ASI Code Parallel\demo_spatial_superposition_propagation_updated_PERF\demo_spatial_superposition_propagation_updated"
    mask_file_path = os.path.join(data_directory, "temp", "npz", "m_full_npz_cache", "m_full000000.npz")  # NOTE: this is a FILE path, not a directory -- named to match config.mask_file below
    cc_spacing = 80
    config = PropagationConfig(
        data_directory=data_directory,
        cell_size_x=1, cell_size_y=1,
        sim_size_x=3000, sim_size_y=3000,
        fields=("m_full", "B_demag"),
        sublattice_dims={0:(4,2), 1:(5,1), 2:(4,2), 3:(4,2), 4:(4,2), 5:(4,2)},  
        device_length_nm={0: 70, 1: 70, 2: 70, 3: 70, 4: 70, 5: 70},
        device_width_nm={0: 56, 1: 56, 2: 56, 3: 56, 4: 56, 5: 56},
        nn_x_nm={0: 2 * cc_spacing, 1:2 * cc_spacing, 2:2 * cc_spacing, 3:2 * cc_spacing, 4:2 * cc_spacing, 5:2 * cc_spacing},
        nn_y_nm={0: int(2 * cc_spacing * (np.sqrt(3))), 1:int(2 * cc_spacing * (np.sqrt(3))), 2:int(2 * cc_spacing * (np.sqrt(3))), 3:int(2 * cc_spacing * (np.sqrt(3))), 4:int(2 * cc_spacing * (np.sqrt(3))), 5:int(2 * cc_spacing * (np.sqrt(3)))},
        translations={1:[-cc_spacing, cc_spacing * np.sqrt(3)], 2:[-cc_spacing * 1/2, cc_spacing * np.sqrt(3)/2],  3:[cc_spacing * 1/2, -cc_spacing * np.sqrt(3)/2], 4:[cc_spacing * 1/2, cc_spacing * np.sqrt(3)/2], 5:[-cc_spacing * 1/2, -cc_spacing * np.sqrt(3)/2]},
        device_rotations={0:90, 1:90, 2:120, 3:120, 4:60, 5:60},
        sublattice_termination={0:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
                                1:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
                                2:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
                                3:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
                                4:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
                                5:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0}},
        indices_to_remove=[11, 20, 27, 28],
        magnetization_mode="opposite",

        magnetization_base_angle={0: 90, 1: 90, 2: 120, 3: 120, 4: 60, 5: 60},
        device_mask_size_nm=(70, 56),
        mask_file=mask_file_path,
        Ms=800e3,
        # injection_indices=[1, 2, 16, 17, 22, 25, 26],
        # injection_indices=[0, 6, 10, 21],
        # injection_indices=[6, 9],
        injection_indices=[23],
        # injection_indices=[21, 12,23, 14, 19, 9],
        # injection_indices=[0, 6, 10, 21, 12, 23],
        # injection_indices=[23, 14, 5, 29, 18, 3],
        # injection_indices=[12, 23, 27, 18, 11, 22, 26, 17], #try
        # injection_indices=[2,3,7, 8],
        # 27, 11 leave unflipped
        # injection_indices= [21, 12, 3, 16, 1, 7, 25, 6, 10, 20, 0, 15, 26, 2], #made middle chiral
        # injection_indices= [21, 12, 3, 16, 1, 7, 25, 6, 10, 20, 0, 15, 26, 2, 18, 22, 7, 8],
        # injection_indices=[18, 7, 22],
        # injection_index=[25,6,10],
        neighbor_shells=4,
        allow_reflip=True,
        max_steps=2,
        coercivity_csv_path=os.path.join(data_directory, "coercivity_table.csv"), # run with coercivity table
        # coercivity_csv_path=None, # run with default coercivity
        default_coercivity_oe=10,
        switch_field_metric="Hmag",
        compute_stats=False,
        plot_spatial_quiver=True,
        quiver_window=(5, 5),
        plot_spatial_maps=False,
        # canvas maps only
        plot_canvas_maps=True,
        canvas_include_full=False,
        canvas_split=None,
        canvas_zoom_targets={"device_12": (1460, 1708)},
        canvas_zoom_propagation_front=False,
        canvas_zoom_candidates=False,
        canvas_compute_stats=False,
        canvas_mask_mode="none",
        export_canvas_csv=False,
        save_plot_data_csv=True,
        use_cached_npz=False,
        canvas_job_max_workers=1,
        parallelize_step_candidates=False,
    )
    result = run_propagation(config, verbose=True)
    print(f"flat flip history: {result['flat_flip_history']}")
    print(f"comparisons CSV: {result['comparison_csv']}")
    print(f"step summary CSV: {result['step_summary_csv']}")
    # Stop the timer
    end_time = time.time()
    # Calculate and display elapsed time
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.2f} seconds")