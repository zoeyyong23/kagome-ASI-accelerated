r"""
File Name: mumax3_parent_superposition.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v4

============================================================
PURPOSE
============================================================
Parent script for superposing a single mumax3 seed CSV (one device's
m_full and/or B_demag) across an ASI array geometry, exported as a
composed canvas, then analyzed at each device's own region -- reusing
mumax3_parent_spatial_analysis.run_pipeline for that analysis step (same
config dataclass + run_pipeline design as that script, so the design is
not redone twice -- see that script's own module-level documentation).

Supports regenerating the composed canvas multiple times (for example,
re-randomizing the magnetization pattern) and exporting each regeneration
as a successive frame, so "regenerate the superposition 1000 times and
visualize all the data spatially" is just num_regenerations=1000 below --
the same downstream analysis code runs either way.

============================================================
BUILDING A CUSTOM PARENT SCRIPT ON TOP OF THIS ONE
============================================================
This file is a reasonable starting point for a custom superposition
script: copy it, keep SuperpositionConfig (or add fields to it), and
either call run_pipeline as-is or call the individual pieces it calls
(geometry.DeviceArray, superposition_operations.
run_superposition_multiprocessing, utilities_superposition.
export_canvas_as_npz_frames, mumax3_parent_spatial_analysis.run_pipeline)
directly, in a different order or combination. The 11 worked geometry
examples commented out at the bottom of this file (square, triangular,
kagome, hexagonal, pinwheel, and their rotated/shaped variants) are a
practical reference for DeviceArray's own parameter combinations -- see
geometry.py's own module-level documentation for what each parameter does
and what changing it does to an already-built array. See the API.txt at
the root of this package for the full function-level reference across
every file, and the src/_mumax3/README.txt and src/_superposition/
README.txt for what each individual file provides.

============================================================
LIMITATIONS
============================================================
- One shared mask size (device_mask_size_nm) is used for every device's
  region, regardless of which sublattice it belongs to -- matching the
  plain spatial-analysis pipeline's single-mask_mode-for-all-regions
  design (see SpatialAnalysisConfig.mask_mode's own documentation), just
  not yet per-sublattice-size-aware. This is not a limitation in practice
  when every sublattice shares one device size (true of every worked
  example at the bottom of this file); mixing device sizes across
  sublattices would need this to become per-region, the way
  centers_composed/geometry_angles already are.
"""

import os  # path construction
import sys  # sys.path manipulation, for add_script_folders
import numpy as np  # used in a few of the worked geometry examples at the bottom of this file (np.sqrt)
from dataclasses import dataclass, field  # SuperpositionConfig
from typing import List, Tuple, Optional, Dict, Union  # type hints used throughout this file's signatures


# --------------------------
# ---- Add this codebase's own folders to sys.path (edit ACTIVE_SCRIPT_DIRS for your machine) ----
def add_script_folders(script_paths, levels_up=2, verbose=False):
    """
    GOAL:
        Add each path in `script_paths`, and that path's parent
        directories up to `levels_up` levels, to sys.path -- so this
        codebase's own modules (src/_mumax3, src/_superposition) can be
        imported by plain module name regardless of where this script is
        actually run from.

    HOW IT TALKS TO OTHER FILES:
        Called once, at the bottom of this file's own __main__ block, with
        ACTIVE_SCRIPT_DIRS (below). Identical implementation to mumax3_
        parent_spatial_analysis.add_script_folders -- kept as its own copy
        in each parent script rather than a shared import, since each
        parent script needs to work standalone before any of this
        codebase's own modules are on sys.path yet.

    PARAMETERS:
        script_paths (List[str]): one or more starting paths to add (along
            with their parent directories).
        levels_up (int): how many parent directories above each path in
            `script_paths` to also add.
        verbose (bool): prints each path added, and each path skipped due
            to a missing drive, when True.

    RETURNS:
        None. Mutates sys.path in place.

    LIMITATIONS:
        A path whose drive does not exist (relevant on Windows) is skipped
        entirely for that starting path, rather than raising -- see
        mumax3_parent_spatial_analysis.add_script_folders' own
        documentation for the full rationale.
    """
    for original_path in script_paths:  # process one starting path at a time
        path = os.path.abspath(original_path)
        drive = os.path.splitdrive(path)[0]  # this path's drive letter, if any (Windows only; empty string elsewhere)
        if drive and not os.path.exists(drive):  # this path's drive is not present on this machine: skip it entirely
            if verbose:
                print(f"[DEBUG] add_script_folders: drive not found, skipping: {drive} for path {path}")
            continue
        for _ in range(levels_up + 1):  # add this path, then walk upward through its parents
            if os.path.exists(path) and path not in sys.path:
                sys.path.append(path)
                if verbose:
                    print(f"[DEBUG] add_script_folders: added '{path}'.")
            path = os.path.dirname(path)


ACTIVE_SCRIPT_DIRS = [
    r"E:\current_superposition_code\src\_mumax3",  # edit these two for your machine -- see add_script_folders
    r"E:\current_superposition_code\src\_superposition",
]

SUPPORTED_FIELDS = ("m_full", "B_demag")  # this parent script only supports these field types


# --------------------------
# ---- Everything a user edits lives here ----
@dataclass
class SuperpositionConfig:
    """
    GOAL:
        Hold every setting run_pipeline needs -- the single object a
        caller constructs and passes to run_pipeline to perform a
        superposition run. See the inline comment on each field below for
        what it controls; fields are grouped by the pipeline step they
        affect.

    See run_pipeline's own documentation for what each step actually does
    with these settings.
    """
    # ---- Paths: directory containing the single-device SEED .csv file(s) ----
    data_directory: str
    output_dir: Optional[str] = None  # default: data_directory/mumax3_analysis_output
    npz_directory: Optional[str] = None  # default: data_directory/temp/npz

    # ---- Simulation grid of the seed file (nm) ----
    cell_size_x: float = 1.0  # nm per grid cell, x direction
    cell_size_y: float = 1.0  # nm per grid cell, y direction
    sim_size_x: float = 3000.0  # nm -- size of the seed simulation domain, x
    sim_size_y: float = 3000.0  # nm -- size of the seed simulation domain, y
    seed_frame_index: int = 0  # which frame of the seed csv(s) to use as the source device

    # ---- Which fields to compose -- subset of SUPPORTED_FIELDS ----
    fields: Tuple[str, ...] = SUPPORTED_FIELDS

    # ---- ASI array geometry -- passed straight to geometry.DeviceArray (see that
    # class's own module-level documentation for full parameter meaning).
    # sublattice_dims can have any number of entries, for example
    # {0: (6,7), 1: (6,7), 2: (6,7)} for a 3-sublattice kagome-style layout, not
    # only the 2-sublattice square/pinwheel case. ----
    device_length_nm: float = 70  # device length, nm -- scalar or per-sublattice dict
    device_width_nm: float = 56  # device width, nm -- scalar or per-sublattice dict
    nn_x_nm: float = 140  # lattice spacing, x, nm -- scalar or per-sublattice dict
    nn_y_nm: float = 140  # lattice spacing, y, nm -- scalar or per-sublattice dict
    sublattice_dims: Dict[int, Tuple[int, int]] = field(default_factory=lambda: {0: (6, 7)})  # {sublattice_id: (num_x, num_y)}
    rotation_angles: Optional[List[float]] = None  # per-sublattice LATTICE rotation, degrees
    device_rotations: Optional[List[float]] = None  # per-sublattice DEVICE (island) rotation, degrees
    translations: Optional[Dict[int, Tuple[float, float]]] = None  # per-sublattice offset, nm
    magnetization_mode: str = "identical"  # 'identical' | 'opposite' | 'row_opposite' | 'col_opposite' | 'random'
    magnetization_base_angle: Optional[List[float]] = None  # per-sublattice base magnetization angle, degrees
    sublattice_termination_mode: Optional[str] = "fixed_row_col"  # 'fixed_row_col' | 'rotated_boundary_fill' -- see geometry.DeviceArray
    sublattice_termination: Optional[Dict[int, Dict[str, int]]] = None  # per-sublattice edge-extension spec -- see geometry.DeviceArray
    indices_to_remove: Optional[List[int]] = None  # device indices to remove after the array is built (see geometry.DeviceArray.remove_devices)

    # ---- Regenerations: rebuild the magnetization pattern and re-export this many
    # times as successive frames. 1 = a single static composition. ----
    num_regenerations: int = 1
    random_seed_start: int = 0  # seed for regeneration 0; regeneration i uses random_seed_start + i (only meaningful for magnetization_mode="random")
    export_canvas_csv: bool = False  # ALSO export each regeneration's composed canvas as a human-viewable CSV (see utilities_superposition.export_canvas_as_csv_frames), alongside the .npz cache STEP 4 always writes. Off by default -- roughly doubles this step's own output file count; purely a "let me look at this" convenience, not needed for the pipeline itself.

    # ---- Per-device masking (post-composition): a single-device seed .npz used as
    # the mask template, rotated to each device's own geometry angle and translated
    # to that device's own position. ----
    mask_file: str = ""  # required -- path to the single-device mask template (.npz)
    device_mask_size_nm: Optional[Tuple[float, float]] = None  # default: (device_length_nm, device_width_nm) -- required if those are per-sublattice dicts (see this file's module-level LIMITATIONS section)

    # ---- Execution / plotting -- passed straight through to
    # mumax3_parent_spatial_analysis.run_pipeline for the analysis step (STEP 6) ----
    use_cached_npz: bool = False  # reuse a valid existing .npz cache instead of reconverting/recomposing
    max_workers: int = 2  # parallel worker processes for region/mask extraction
    plot_spatial_maps: bool = False  # color-map plots (magnitude, angle, components, hue, gradient), per device
    plot_spatial_limiter: Optional[int] = None  # cap the number of regenerations plotted, for a quick test run
    plot_max_workers: int = 8  # see SpatialAnalysisConfig.plot_max_workers -- forwarded straight through to both the per-device pass (STEP 6) and the canvas-level pass (STEP 7)
    Ms: float = 800e3  # A/m -- used for m_full's plot color-scale limits
    suptitle: str = "Superposition of ASI Array"  # figure-level title for every spatial plot

    # ---- Weighted circular statistics at each device's region on the COMPOSED
    # canvas -- passed straight through to mumax3_parent_spatial_analysis.
    # run_pipeline exactly like the plotting options above. "Frame" here means
    # "regeneration" (see num_regenerations): with mask_mode='file' (set
    # automatically in STEP 6 below), region_specs already covers every device, so
    # compute_stats=True gives every device x every regeneration in one pass, the
    # same way it would give every device x every mumax3 timestep in the plain
    # spatial-analysis pipeline. ----
    compute_stats: bool = False
    plot_region_stats: bool = False
    stats_ellipse_size: Optional[Tuple[float, float]] = None
    stats_time_step_ns: float = 0.1
    stats_max_workers: int = 1  # see SpatialAnalysisConfig.stats_max_workers -- forwarded straight through
    save_stats_csv: bool = False
    save_stats_csv_per_frame: bool = False
    save_stats_csv_all_frames: bool = False

    # ---- Averaged quiver plots at each device's region -- see
    # SpatialAnalysisConfig.plot_spatial_quiver/quiver_window for full semantics,
    # forwarded straight through. ----
    plot_spatial_quiver: bool = False
    quiver_window: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = (2, 2)
    save_plot_data_csv: bool = False  # see SpatialAnalysisConfig.save_plot_data_csv --
    # forwarded straight through to BOTH the per-device pass (STEP 6) and the canvas
    # pass (STEP 7) below.

    # ---- Canvas-level plots: the SAME m_full/B_demag maps as the per-device
    # analysis above, but of the FULL composed canvas (or a zoomed-in piece of it)
    # instead of one region per device. This is a SEPARATE analysis pass (its own
    # call to mumax3_parent_spatial_analysis.run_pipeline, pointed at the same
    # canvas_dir but with canvas-tile/zoom regions instead of centers_composed),
    # landing under output_dir/canvas/... so it never collides with the per-device
    # output above. ----
    plot_canvas_maps: bool = False
    canvas_split: Optional[int] = None  # "split mode": None/1 = no split (whole
    # canvas, one tile); N (greater than 1) = tile the canvas into an NxN grid, each
    # tile sized (canvas_width/N, canvas_height/N) -- for example, split=4 on a
    # 500x500nm canvas gives sixteen 125x125nm tiles. See region_extraction.
    # build_canvas_split_centers_sizes.
    canvas_zoom_targets: Optional[Dict[str, Tuple[int, int]]] = None  # "zoom mode":
    # {name -> (cx, cy)} absolute-grid center(s) to zoom into instead of/alongside
    # split mode -- for example, {"device_8": (1234, 987)} to zoom around one device
    # for a propagation-rule study. See region_extraction.
    # build_canvas_zoom_centers_sizes and resolve_dynamic_center's own documentation
    # for the moving-center case that function does not support yet. Both split and
    # zoom can be used together in one call -- each target/tile becomes its own
    # region, all analyzed in the same pass.
    canvas_zoom_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = (200, 200)
    # (width, height) in grid cells around each canvas_zoom_targets entry -- a
    # shared tuple, or a {name -> size} dict for per-target sizes.
    canvas_zoom_all_devices: bool = False  # auto-populate canvas_zoom_targets with
    # EVERY device from the DeviceArray (named "device_0", "device_1", ... in the
    # same order as device_centers_nm/centers_composed below), each sized
    # canvas_zoom_size -- instead of hand-listing absolute grid centers, which
    # cannot be known until the array is actually built (STEP 3/5 below). Any names
    # ALSO given explicitly in canvas_zoom_targets are merged in on top (so extra
    # named targets, for example a hand-picked propagation-front point, can be added
    # alongside "every device").
    canvas_include_full: bool = False  # ALSO include one region covering the whole
    # canvas (labeled "canvas_full_image"), regardless of canvas_split/canvas_zoom_*
    # above -- so "give me the whole canvas" and "also split it into quadrants" (or
    # "also zoom on every device") are not mutually exclusive; every one of them is
    # produced in one pass. Set False to get only the split tiles / zoom targets.
    canvas_mask_mode: str = "none"  # a canvas tile/zoom is usually not one device --
    # 'none' (plain rectangular crop) is the sane default; set to 'file'/'ellipse' to
    # analyze canvas regions with their own aperiodic mask, or for an elliptical crop
    # of a zoom target.
    canvas_mask_file: Optional[str] = None
    canvas_compute_stats: bool = False  # independent from compute_stats above --
    # stats over a canvas tile/zoom region (many devices at once) mean something
    # different from stats over one device's own region; off by default.
    canvas_center_plots_at_origin: bool = False  # DeviceArray already reports every
    # device's position relative to the ARRAY CENTER at (0, 0); when True (default),
    # every canvas-level plot (full canvas, quadrants, per-device zooms) shares that
    # same convention -- the canvas center maps to (0, 0) on every one of their axes,
    # instead of each region independently starting its own local extent at (0, 0).
    # Set False to keep the old absolute-grid-position axes. Purely a plot-axis/
    # labeling choice -- does not touch the underlying data at all. Only affects THIS
    # canvas pass (STEP 7) -- the per-device regions in STEP 6 keep their own local
    # per-region axes either way, since "device N's own local frame" and
    # "DeviceArray's array-wide frame" are different, equally reasonable things to
    # want depending on what is being examined.

    verbose: bool = False  # master switch: forwarded to every step below and every
    # function they call, so setting this to True turns on progress/debug printing
    # across the entire pipeline in one place, without editing individual function
    # calls. Every print statement in this pipeline that is not a WARNING or ERROR is
    # gated behind verbose somewhere along this chain.

    def __post_init__(self):
        """
        GOAL:
            Resolve default paths and validate this config's fields once,
            immediately after construction.

        HOW IT TALKS TO OTHER FILES:
            Called automatically by Python's dataclass machinery right
            after __init__.

        PARAMETERS:
            None (operates on self).

        RETURNS:
            None.

        RAISES:
            ValueError if any field in `fields` is not in SUPPORTED_FIELDS,
            or if mask_file is not set.
        """
        if self.output_dir is None:  # no explicit output directory given: default under data_directory
            self.output_dir = os.path.join(self.data_directory, "mumax3_analysis_output")
        if self.npz_directory is None:  # no explicit npz cache directory given: default under data_directory
            self.npz_directory = os.path.join(self.data_directory, "temp", "npz")
        bad_fields = set(self.fields) - set(SUPPORTED_FIELDS)  # any requested field this parent script does not support
        if bad_fields:
            raise ValueError(f"Unsupported field(s) {bad_fields}; this parent script only "
                              f"supports {SUPPORTED_FIELDS}.")
        if not self.mask_file:  # the per-device mask template is required
            raise ValueError("config.mask_file must be set -- the per-device mask template "
                              "(a single-device seed .npz).")


# --------------------------
# ---- The pipeline itself ----
def run_pipeline(config: SuperpositionConfig):
    """
    GOAL:
        Run seed extraction, ASI geometry construction, field composition
        (repeated config.num_regenerations times), and per-device
        region/mask analysis (reusing mumax3_parent_spatial_analysis.
        run_pipeline for that last step), plus optional canvas-level
        analysis of the composed canvas as a whole.

    HOW IT TALKS TO OTHER FILES:
        Calls mumax3_parent_utilities.convert_npz_per_dataset (STEP 1),
        region_extraction.build_region_specs and mumax3_parent_utilities.
        region_processor/compute_regions (STEP 2), geometry.DeviceArray and
        utilities_superposition.save_device_array (STEP 3),
        superposition_operations.run_superposition_multiprocessing and
        utilities_superposition.export_canvas_as_npz_frames (STEP 4),
        region_extraction.build_region_specs again (STEP 5), and
        mumax3_parent_spatial_analysis.run_pipeline (STEPs 6 and 7 -- the
        per-device and canvas-level analysis passes, respectively).

    PARAMETERS:
        config (SuperpositionConfig): every setting for this run -- see
            that class's own field-by-field documentation.

    RETURNS:
        If config.plot_canvas_maps is False: whatever mumax3_parent_
        spatial_analysis.run_pipeline returns for the per-device analysis
        (STEP 6) alone -- see that function's own documentation for the
        exact shape, keeping in mind that "frame" there means
        "regeneration" here (0..num_regenerations-1), and region_label is
        keyed by each DEVICE's own resolved RegionSpec label (from
        centers_composed), not a user-specified center.

        If config.plot_canvas_maps is True (the default): {"devices":
        device_result, "canvas": canvas_result}, where device_result is
        the STEP 6 return value described above, and canvas_result is the
        equivalent return value from STEP 7's canvas-level analysis pass
        (region_label there is keyed by canvas tile/zoom label instead of
        device).

    LIMITATIONS:
        See this file's module-level LIMITATIONS section.
    """
    verbose = config.verbose  # master verbose toggle, forwarded into every step below
    os.makedirs(config.npz_directory, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    from mumax3_parent_utilities import convert_npz_per_dataset, region_processor, compute_regions
    from region_extraction import build_region_specs
    from geometry import DeviceArray
    from utilities_superposition import export_canvas_as_npz_frames, export_canvas_as_csv_frames, save_device_array
    from superposition_operations import run_superposition_multiprocessing
    from mumax3_parent_spatial_analysis import SpatialAnalysisConfig, run_pipeline as run_spatial_analysis_pipeline

    # ---- STEP 1: NPZ-convert the seed csv(s) ----
    confirmed_datasets, _confirmed_metadata = convert_npz_per_dataset(
        config.data_directory, config.npz_directory, set(config.fields),
        config.use_cached_npz, verbose=verbose,
    )

    # ---- STEP 2: extract the seed region -- the WHOLE seed file, at its own center
    # (to build the superposition, the region must be the entire mumax3 csv file,
    # centered at sim_size/2). Same build_region_specs/compute_regions call as the
    # plain spatial-analysis pipeline -- just one region, size = full domain,
    # mask_mode='none'. ----
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

    base_field: Dict[str, Dict[str, "np.ndarray"]] = {}  # {dataset -> {"x": ..., "y": ...}} -- the seed device's own field, one entry per requested dataset
    for dataset in confirmed_datasets:  # extract this dataset's seed region
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
            "x": region["masked_x"][config.seed_frame_index],  # the seed frame's x component, unrotated -- rotate_field_spatial/rotate_field_vectors handle rotation per device later, in superposition_operations.compose_fields
            "y": region["masked_y"][config.seed_frame_index],
        }

    # ---- STEP 3: ASI geometry (any number of sublattices; see geometry.DeviceArray) ----
    array = DeviceArray(
        device_length=config.device_length_nm, device_width=config.device_width_nm,  # required
        nn_x=config.nn_x_nm, nn_y=config.nn_y_nm,  # required
        sublattice_dims=config.sublattice_dims,  # required
        device_rotations=config.device_rotations,
        rotation_angles=config.rotation_angles,
        translations=config.translations,
        sublattice_termination=config.sublattice_termination,
        sublattice_termination_mode=config.sublattice_termination_mode,
        verbose=verbose,
    )

    if config.indices_to_remove:  # optionally hand-trim specific devices after the array is built (see geometry.DeviceArray.remove_devices)
        array.remove_devices(indices=config.indices_to_remove, verbose=verbose)

    output_dir_geom = os.path.join(config.output_dir, "ASI_geometry")
    os.makedirs(output_dir_geom, exist_ok=True)
    save_device_array(device_array_obj=array, base_save_path=os.path.join(output_dir_geom, f"ASI_{config.magnetization_mode}"),
                       save_table_exts=["csv", "pdf"], save_plot_exts=["png", "pdf"], verbose=verbose)
    device_centers_nm = array.get_device_centers() * 1e9  # meters -> nm
    geometry_angles = list(array.device_angles)  # each device's own geometry (shape) orientation, degrees

    data_struct_all = {
        seed_label: {dataset: {"masked": base_field[dataset]} for dataset in confirmed_datasets}
    }
    if verbose:
        print(f"[DEBUG] run_pipeline: seed data structure built for dataset(s) {list(confirmed_datasets)}.")

    # ---- STEP 4: compose + export, num_regenerations times ----
    canvas_dir = os.path.join(config.npz_directory, "composed_canvas")
    os.makedirs(canvas_dir, exist_ok=True)

    for i in range(config.num_regenerations):  # rebuild the magnetization pattern and re-export, once per regeneration
        for j in range(len(config.sublattice_dims)):  # set this regeneration's magnetization pattern, one sublattice at a time
            if verbose:
                print(f"[INFO] run_pipeline: regeneration {i + 1}/{config.num_regenerations}, "
                      f"sublattice {j + 1}/{len(config.sublattice_dims)}: setting magnetization pattern...")
            array.set_magnetization_pattern(
                mode=config.magnetization_mode,
                sublattice=j,  # per sublattice
                base_angle=config.magnetization_base_angle.get(j) if config.magnetization_base_angle is not None else None,
                random_seed=(config.random_seed_start + i) if config.magnetization_mode == "random" else None,
                verbose=verbose,
            )
        magnetization_angles = list(array.device_m_angles)  # each device's own magnetization angle for this regeneration, degrees

        canvas_struct = run_superposition_multiprocessing(
            data_struct_all=data_struct_all,
            device_centers=[tuple(c) for c in device_centers_nm],
            geometry_angles=geometry_angles,
            magnetization_angles=magnetization_angles,
            cell_size_x=config.cell_size_x, cell_size_y=config.cell_size_y,
            verbose=verbose,
        )
        export_canvas_as_npz_frames(canvas_struct, canvas_dir, frame_index=i, verbose=verbose)
        if config.export_canvas_csv:  # optional human-viewable companion CSV -- see export_canvas_as_csv_frames' own documentation
            export_canvas_as_csv_frames(canvas_struct, canvas_dir, frame_index=i, verbose=verbose)
        if verbose:
            print(f"[INFO] run_pipeline: exported regeneration {i + 1}/{config.num_regenerations}.")

    # ---- STEP 5: per-device region_specs for the composed canvas ----
    # DeviceArray hands back device_centers COM-centered at 0 (canvas-relative);
    # RegionSpec.center is defined as an ABSOLUTE grid position everywhere else in
    # this pipeline. Convert here, once, rather than inside masking.apply_mask.
    canvas_center_x, canvas_center_y = seed_size[0] // 2, seed_size[1] // 2
    centers_composed = []  # accumulated across every device below
    for cx_nm, cy_nm in device_centers_nm:  # convert each device's canvas-relative center into an absolute grid position
        rel_x = int(round(cx_nm / config.cell_size_x))
        rel_y = int(round(cy_nm / config.cell_size_y))
        centers_composed.append((canvas_center_x + rel_x, canvas_center_y + rel_y))

    # device_mask_size_nm is REQUIRED here if device_length_nm/device_width_nm are
    # per-sublattice dicts rather than a single shared scalar -- there is no
    # sensible way to collapse a dict into one mask size applied identically to
    # every device below, so this fails loudly instead of silently building a
    # garbage region size.
    if config.device_mask_size_nm is not None:  # an explicit shared mask size was given
        mask_size_nm = config.device_mask_size_nm
    elif isinstance(config.device_length_nm, dict) or isinstance(config.device_width_nm, dict):  # per-sublattice sizes with no explicit fallback: fail loudly (see this file's module-level LIMITATIONS section)
        raise ValueError(
            "config.device_mask_size_nm must be set explicitly when device_length_nm/"
            "device_width_nm are per-sublattice dicts -- there's no single shared "
            "(length, width) to fall back to."
        )
    else:  # device_length_nm/device_width_nm are plain scalars: use them directly
        mask_size_nm = (config.device_length_nm, config.device_width_nm)

    sizes_composed = [
        (int(round(mask_size_nm[0] / config.cell_size_x)), int(round(mask_size_nm[1] / config.cell_size_y)))
    ] * len(centers_composed)  # see this file's module-level LIMITATIONS section regarding this one-shared-size-for-every-device design

    # ---- Warn (do not drop) about any device whose own mask footprint crosses the
    # seed canvas's edge. masking.apply_mask already NaN-pads an edge-crossing region
    # rather than corrupting it (see that file's own module-level PURPOSE section for
    # the mechanism), so this device's own stats/plots are NOT biased by invalid data
    # -- but they DO reflect less of that device's true footprint than an interior
    # device's stats do, silently, unless flagged here. This is a WARNING, not a drop
    # (unlike propagation_engine.py's own candidate_edge_margin_cells, which drops an
    # edge-crossing candidate outright): a static per-device analysis run typically
    # still wants that device's own (partial) stats/plots, just with the caveat made
    # visible, rather than silently excluding it from the output entirely the way a
    # propagation switching decision needs to. Uses the same shared masking.
    # region_crosses_edge utility propagation_engine.py's own guard is built on -- see
    # that function's own documentation. ----
    from masking import region_crosses_edge
    edge_crossing_devices = []
    for i, (center, size) in enumerate(zip(centers_composed, sizes_composed)):
        crosses, overhang = region_crosses_edge(source_shape=(seed_size[1], seed_size[0]), center=center, size=size, units='grid')
        if crosses:
            edge_crossing_devices.append((i, overhang))
    if edge_crossing_devices:
        print(f"[WARNING] run_pipeline: {len(edge_crossing_devices)} device(s) have a mask footprint crossing the "
              f"seed canvas's edge -- their masked stats/plots are averaged over LESS than their true full footprint "
              f"(NaN-padded, not biased by invalid data, but incomplete). Consider a larger sim_size_x/sim_size_y "
              f"if this is unintentional. Device index -> overhang (cells): "
              f"{ {i: o for i, o in edge_crossing_devices[:10]} }{' ...' if len(edge_crossing_devices) > 10 else ''}")

    # ---- STEP 6: reuse the plain spatial-analysis pipeline for the actual
    # analysis, pointed at the composed canvas instead of raw CSV output.
    # skip_npz_conversion=True because canvas_dir already holds ready npz frames
    # (from export_canvas_as_npz_frames), not CSVs -- see that flag's own
    # documentation in mumax3_parent_spatial_analysis.py.
    #
    # compute_stats=True (via config.compute_stats) runs STEP 4 of THAT pipeline
    # over centers_composed -- every device's own region -- so this gives weighted
    # circular stats per device, per regeneration ("frame" there = regeneration
    # here), for m_full and B_demag in their own separate output folders/CSVs
    # exactly as in the plain spatial-analysis pipeline. Those stats are computed
    # over each device's masked_x/masked_y -- the exact mask_file-shaped footprint
    # set up above, not the full composed canvas -- so a device's stats reflect just
    # that device, not the whole array. ----
    analysis_config = SpatialAnalysisConfig(
        data_directory=canvas_dir,  # unused (skip_npz_conversion=True), but required
        npz_directory=canvas_dir,  # compute_regions looks for {prefix}_npz_cache here, matching export_canvas_as_npz_frames' own naming
        output_dir=config.output_dir,
        cell_size_x=config.cell_size_x, cell_size_y=config.cell_size_y,
        fields=tuple(confirmed_datasets),
        centers=centers_composed,
        sizes=sizes_composed,
        mask_mode="file",
        mask_file=config.mask_file,
        mask_angles=geometry_angles,  # each device's OWN angle, not one shared angle
        mask_translation=True,
        use_cached_npz=config.use_cached_npz,
        max_workers=config.max_workers,
        plot_spatial_maps=config.plot_spatial_maps,
        plot_spatial_limiter=config.plot_spatial_limiter,
        plot_max_workers=config.plot_max_workers,
        Ms=config.Ms,
        suptitle=config.suptitle,
        skip_npz_conversion=True,
        # ---- stats, forwarded straight through from SuperpositionConfig ----
        compute_stats=config.compute_stats,
        plot_region_stats=config.plot_region_stats,
        stats_ellipse_size=config.stats_ellipse_size,
        stats_time_step_ns=config.stats_time_step_ns,
        stats_max_workers=config.stats_max_workers,
        save_stats_csv=config.save_stats_csv,
        save_stats_csv_per_frame=config.save_stats_csv_per_frame,
        save_stats_csv_all_frames=config.save_stats_csv_all_frames,
        plot_spatial_quiver=config.plot_spatial_quiver,
        quiver_window=config.quiver_window,
        save_plot_data_csv=config.save_plot_data_csv,
        verbose=verbose,
    )
    device_result = run_spatial_analysis_pipeline(analysis_config)

    # ---- STEP 7: canvas-level plots (opt-in via config.plot_canvas_maps) -- a
    # SEPARATE run_spatial_analysis_pipeline call, same canvas_dir, but with
    # canvas-tile/zoom regions (region_extraction.build_canvas_split_centers_sizes /
    # build_canvas_zoom_centers_sizes) instead of centers_composed -- so this reuses
    # every bit of the per-device pipeline above (folder hierarchy, quiver, stats)
    # for free, landing under output_dir/canvas/... so it never collides with the
    # per-device output. Split and zoom tiles/targets can both be requested in one
    # call; if neither is set, plot_canvas_maps=True alone gives one region covering
    # the whole canvas (canvas_split=None's default behavior). ----
    if not config.plot_canvas_maps:
        return device_result

    from region_extraction import build_canvas_split_centers_sizes, build_canvas_zoom_centers_sizes

    canvas_size = seed_size  # the composed canvas is exactly the seed domain's grid size (see STEP 4/5)
    canvas_center = (canvas_center_x, canvas_center_y)

    split_centers, split_sizes, split_labels = build_canvas_split_centers_sizes(
        canvas_size, canvas_center, split=config.canvas_split, verbose=verbose,
    )
    canvas_centers, canvas_sizes, canvas_labels = split_centers, split_sizes, split_labels

    # ---- Auto-populate zoom targets from every device in the array. centers_composed
    # is already in ABSOLUTE grid coordinates (see STEP 5's canvas_center_x + rel_x
    # conversion), the same convention build_canvas_zoom_centers_sizes expects, so
    # device i's zoom target is just centers_composed[i] -- no separate lookup
    # needed. Explicit canvas_zoom_targets (if any) are merged in on top, so a
    # hand-picked name is not silently dropped by turning this on. ----
    zoom_targets = dict(config.canvas_zoom_targets or {})
    if config.canvas_zoom_all_devices:
        auto_targets = {f"device_{i}": centers_composed[i] for i in range(len(centers_composed))}
        auto_targets.update(zoom_targets)  # explicit names win over the auto ones
        zoom_targets = auto_targets

    if zoom_targets:  # at least one zoom target was requested (explicit or auto-populated)
        zoom_centers, zoom_sizes, zoom_labels = build_canvas_zoom_centers_sizes(
            zoom_targets, size=config.canvas_zoom_size, verbose=verbose,
        )
        canvas_centers = canvas_centers + zoom_centers
        canvas_sizes = canvas_sizes + zoom_sizes
        canvas_labels = canvas_labels + zoom_labels

    # ---- Also include one region covering the whole canvas outright, independent
    # of split/zoom above -- added directly rather than through canvas_zoom_targets
    # so it is not affected by canvas_zoom_size. ----
    if config.canvas_include_full:
        canvas_centers = canvas_centers + [canvas_center]
        canvas_sizes = canvas_sizes + [canvas_size]
        canvas_labels = canvas_labels + ["canvas_full_image"]

    canvas_analysis_config = SpatialAnalysisConfig(
        data_directory=canvas_dir,
        npz_directory=canvas_dir,
        extracted_regions_subdir="extracted_regions_canvas",  # keeps this pass's
        # region-extraction cache from colliding with STEP 6's above, which points
        # at the same canvas_dir (both need the same underlying composed-canvas
        # frames) but extracts a completely different set of regions -- see
        # SpatialAnalysisConfig.extracted_regions_subdir's own documentation.
        output_dir=os.path.join(config.output_dir, "canvas"),
        cell_size_x=config.cell_size_x, cell_size_y=config.cell_size_y,
        fields=tuple(confirmed_datasets),
        centers=canvas_centers,
        sizes=canvas_sizes,
        region_labels=canvas_labels,
        plot_origin_cells=canvas_center if config.canvas_center_plots_at_origin else None,
        mask_mode=config.canvas_mask_mode,
        mask_file=config.canvas_mask_file,
        use_cached_npz=config.use_cached_npz,
        max_workers=config.max_workers,
        plot_spatial_maps=config.plot_spatial_maps,
        plot_spatial_limiter=config.plot_spatial_limiter,
        plot_max_workers=config.plot_max_workers,
        Ms=config.Ms,
        suptitle=f"{config.suptitle} -- canvas",
        skip_npz_conversion=True,
        compute_stats=config.canvas_compute_stats,
        plot_region_stats=config.plot_region_stats,
        stats_time_step_ns=config.stats_time_step_ns,
        stats_max_workers=config.stats_max_workers,
        save_stats_csv=config.save_stats_csv,
        save_stats_csv_per_frame=config.save_stats_csv_per_frame,
        save_stats_csv_all_frames=config.save_stats_csv_all_frames,
        plot_spatial_quiver=config.plot_spatial_quiver,
        quiver_window=config.quiver_window,
        save_plot_data_csv=config.save_plot_data_csv,
        verbose=verbose,
    )
    canvas_result = run_spatial_analysis_pipeline(canvas_analysis_config)

    return {"devices": device_result, "canvas": canvas_result}


if __name__ == "__main__":
    add_script_folders(ACTIVE_SCRIPT_DIRS, levels_up=0, verbose=True)
    data_directory = r"E:\04_Angle_Dependence_MTJs\test_data"
    mask_file_path = os.path.join(data_directory, "temp", "npz", "m_full_npz_cache", "m_full000000.npz")  # NOTE: this is a FILE path, not a directory -- named to match config.mask_file below

    config = SuperpositionConfig(
        data_directory=data_directory,
        cell_size_x=1, cell_size_y=1,
        sim_size_x=3000, sim_size_y=3000,
        fields=("m_full", "B_demag"),

        sublattice_dims={0: (6, 7), 1: (7, 6)},
        device_length_nm={0: 70, 1: 70},
        device_width_nm={0: 56, 1: 56},
        nn_x_nm={0: 140, 1: 140},
        nn_y_nm={0: 140, 1: 140},
        translations={1: [-70, 70]},
        device_rotations={0: 45, 1: 135},

        magnetization_mode="identical",
        magnetization_base_angle={0: 45, 1: 135},
        device_mask_size_nm=(70, 56),  # mask size

        num_regenerations=1,
        mask_file=mask_file_path,
        Ms=800e3,
        # Per-device weighted circular stats on the composed canvas, on top of the
        # spatial maps above. Each device gets its own region (via centers_composed/
        # geometry_angles in STEP 5/6), so this is "stats at every device center"
        # across the whole array, for m_full and B_demag in their own separate
        # output files.
        compute_stats=False,
        stats_time_step_ns=0.1,
        verbose=False,  # turn on progress/debug printing across the whole pipeline
    )
    result = run_pipeline(config)
    if config.compute_stats:
        regions, stats_results = result
    else:
        regions = result

# ============================================================
# WORKED GEOMETRY EXAMPLES -- see geometry.py's own module-level documentation for
# what each parameter does. Uncomment one example's fields into the SuperpositionConfig
# call above (in place of the square-ASI example already there) to try it.
# ============================================================

# ---- XY Square (Example 1) ----
# sublattice_dims={0: (5, 5)},
# device_length_nm={0: 60},
# device_width_nm={0: 60},
# nn_x_nm={0: 120},
# nn_y_nm={0: 120},

# ---- XY Triangular (Example 2) ----
# sublattice_dims={0: (6, 3), 1: (6, 3)},
# device_length_nm={0: 60, 1: 60},
# device_width_nm={0: 60, 1: 60},
# nn_x_nm={0: 120, 1: 120},
# nn_y_nm={0: int(120 * (np.sqrt(3))), 1: int(120 * (np.sqrt(3)))},
# translations={1: [-60, int(np.sqrt(3)/2 * 120)]},
# sublattice_termination={0:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0}, 1:{'pos_x': 0, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0}},

# ---- XY Kagome (Example 3) ----
# sublattice_dims={0: (6, 3), 1: (2, 1), 2: (3, 2)},
# device_length_nm={0: 60, 1: 60, 2: 60},
# device_width_nm={0: 60, 1: 60, 2: 60},
# nn_x_nm={0: 120, 1: 240, 2: 240},
# nn_y_nm={0: int(120 * (np.sqrt(3))), 1: int(3 * 120 * (np.sqrt(3))), 2: int(np.sqrt(3) * 120 * 2)},
# translations={1: [120, -int(np.sqrt(3)/2 * 120)], 2: [0, -int(np.sqrt(3) * 120)]},

# ---- XY Hexagonal (Example 4) ----
# sublattice_dims={0: (2, 3), 1: (2, 3), 2: (2, 3), 3: (2, 3)},
# device_length_nm={0: 60, 1: 60, 2: 60, 3: 60},
# device_width_nm={0: 60, 1: 60, 2: 60, 3: 60},
# nn_x_nm={0: 360, 1: 360, 2: 360, 3: 360},
# nn_y_nm={0: 240, 1: 240, 2: 240, 3: 240},
# translations={1: [120, 0], 2: [180, 120], 3: [-60, 120]},

# ---- XY Square (Example 5) ----
# sublattice_dims={0: (6, 7), 1: (7, 6)},
# device_length_nm={0: 60, 1: 60},
# device_width_nm={0: 60, 1: 60},
# nn_x_nm={0: 120, 1: 120},
# nn_y_nm={0: 120, 1: 120},
# translations={1: [-60, 60]},

# ---- Traditional Square (Example 6) ----
# sublattice_dims={0: (6, 7), 1: (7, 6)},
# device_length_nm={0: 70, 1: 56},
# device_width_nm={0: 70, 1: 56},
# nn_x_nm={0: 140, 1: 140},
# nn_y_nm={0: 140, 1: 140},
# translations={1: [-70, 70]},
# device_rotations={0:0, 1:90},

# ---- Rotated Square (Example 7) ----
# sublattice_dims={0: (6, 7), 1: (7, 6)},
# device_length_nm={0: 70, 1: 70},
# device_width_nm={0: 56, 1: 56},
# nn_x_nm={0: 140, 1: 140},
# nn_y_nm={0: 140, 1: 140},
# translations={1: [-70, 70]},
# device_rotations={0:45, 1:135},
# rotation_angles={0:45, 1:45},
# sublattice_termination_mode="rotated_boundary_fill", # "rotated_boundary_fill", "fixed_row_col"


# ---- Traditional Pinwheel (Example 8)
# sublattice_dims={0: (6, 7), 1: (7, 6)},
# device_length_nm={0: 70, 1: 70},
# device_width_nm={0: 56, 1: 56},
# nn_x_nm={0: 140, 1: 140},
# nn_y_nm={0: 140, 1: 140},
# translations={1: [-70, 70]},
# device_rotations={0:45, 1:135},

# ---- Rotated Pinwheel (Example 9) ----
# device_length_nm={0: 70, 1: 70},
# device_width_nm={0: 56, 1: 56},
# nn_x_nm={0: 140, 1: 140},
# nn_y_nm={0: 140, 1: 140},
# sublattice_dims={0: (6, 7), 1: (7, 6)},
# translations={1: [-70, 70]},
# device_rotations={0:0, 1:90},
# rotation_angles={0:45, 1:45},
# sublattice_termination_mode="rotated_boundary_fill" # (Default  Termination: "fixed_row_col"),

# ---- Kagome (Example 10) ----
# sublattice_dims={0: (3, 1), 1: (4, 1), 2:(3, 1), 3:(3, 2), 4:(3, 1), 5:(3, 2)},
# device_length_nm={0: 70, 1: 70, 2: 70, 3: 70, 4: 70, 5: 70},
# device_width_nm={0: 56, 1: 56, 2: 56, 3: 56, 4: 56, 5: 56},
# nn_x_nm={0: 280, 1:280, 2:280, 3:280, 4: 280, 5: 280},
# nn_y_nm={0: int(2 * 140 * (np.sqrt(3))), 1:int(2 * 140 * (np.sqrt(3))), 2:int(2 * 140 * (np.sqrt(3))), 3:int(2 * 140 * (np.sqrt(3))), 4:int(2 * 140 * (np.sqrt(3))), 5:int(2 * 140 * (np.sqrt(3)))},
# translations={1:[-140, 140 * np.sqrt(3)], 2:[-140 * 1/2, 140 * np.sqrt(3)/2],  3:[140 * 1/2, -140 * np.sqrt(3)/2], 4:[140 * 1/2, 140 * np.sqrt(3)/2], 5:[-140 * 1/2, -140 * np.sqrt(3)/2]},
# device_rotations={0:90, 1:90, 2:120, 3:120, 4:60, 5:60},

# ---- Kagome Shaped (Example 11) ----
# sublattice_dims={0: (3, 1), 1: (4, 1), 2:(3, 1), 3:(3, 2), 4:(3, 1), 5:(3, 2)},
# device_length_nm={0: 70, 1: 70, 2: 70, 3: 70, 4: 70, 5: 70},
# device_width_nm={0: 56, 1: 56, 2: 56, 3: 56, 4: 56, 5: 56},
# nn_x_nm={0: 280, 1:280, 2:280, 3:280, 4: 280, 5: 280},
# nn_y_nm={0: int(2 * 140 * (np.sqrt(3))), 1:int(2 * 140 * (np.sqrt(3))), 2:int(2 * 140 * (np.sqrt(3))), 3:int(2 * 140 * (np.sqrt(3))), 4:int(2 * 140 * (np.sqrt(3))), 5:int(2 * 140 * (np.sqrt(3)))},
# translations={1:[-140, 140 * np.sqrt(3)], 2:[-140 * 1/2, 140 * np.sqrt(3)/2],  3:[140 * 1/2, -140 * np.sqrt(3)/2], 4:[140 * 1/2, 140 * np.sqrt(3)/2], 5:[-140 * 1/2, -140 * np.sqrt(3)/2]},
# device_rotations={0:90, 1:90, 2:120, 3:120, 4:60, 5:60},
# sublattice_termination={0:{'pos_x': 0, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
#                         1:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
#                         2:{'pos_x': 0, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
#                         3:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
#                         4:{'pos_x': -1, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0},
#                         5:{'pos_x': 0, 'pos_y': 0, 'neg_x': 0, 'neg_y': 0}},
# indices_to_remove=[15, 20],
# magnetization_base_angle={0: 90, 1: 90, 2:150, 3:150, 4:30, 5:30},
