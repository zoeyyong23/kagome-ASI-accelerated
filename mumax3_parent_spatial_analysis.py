r"""
File Name: mumax3_parent_spatial_analysis.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v5

============================================================
PURPOSE
============================================================
Parent script for spatial analysis of a directory of mumax3 CSV output
files (m_full, B_demag, and/or B_exch), at any number of region centers,
with a single mask mode (none / ellipse / file) applied to all of them.
The same pipeline call covers every scale of this task without any change
in shape: 1 region across 2 files with no mask; 10 regions across 800
files with an ellipse mask; and so on.

Structured as one config dataclass (SpatialAnalysisConfig) holding every
user-editable setting, and one function (run_pipeline) that performs the
actual work end to end: NPZ conversion, region/mask extraction, spatial
plotting, averaged quiver plots, and weighted circular statistics -- each
step independently toggleable. To get oriented in this file, read
SpatialAnalysisConfig first (that is everything that can be changed), then
read run_pipeline top to bottom (that is what actually happens, in order).

============================================================
WHY A CONFIG DATACLASS + run_pipeline(config), NOT A FLAT SCRIPT
============================================================
A flat script (edit module-level constants, run top to bottom) reads
slightly more simply start to end. But a flat script cannot be called from
another script (for example, a sweep driver that runs this pipeline once
per parameter set, or a notebook), cannot easily be re-run on just part of
its own work interactively, and this codebase's region_specs/RegionSpec
unification already pushes toward "pass a structured object into a
function" as the natural shape. The concrete payoff: mumax3_parent_
superposition.py's post-composition analysis step calls this exact same
run_pipeline (see that script's own STEP 6/7) to analyze a composed
canvas -- without this structure, that reuse would mean copying this
file's body into the other one.

============================================================
BUILDING A CUSTOM PARENT SCRIPT ON TOP OF THIS ONE
============================================================
This file is a reasonable starting point for a custom analysis script:
copy it, keep SpatialAnalysisConfig (or add fields to it), and either call
run_pipeline as-is or call the individual pieces it calls
(mumax3_parent_utilities.convert_npz_per_dataset/region_processor/
compute_regions/run_spatial_plots, mumax3_csv_stats.
compute_stats_for_all_timesteps_parallel) directly, in a different order or
combination. See the API.txt at the root of this package for the full
function-level reference across every file, and the src/_mumax3/README.txt
and src/_superposition/README.txt for what each individual file provides.

============================================================
LIMITATIONS
============================================================
- This parent script only supports the field types listed in
  SUPPORTED_FIELDS (m_full, B_demag, B_exch). Extending it to another
  mumax3 output requires adding that name to SUPPORTED_FIELDS and to
  mumax3_parent_utilities.validate_field_processors' own DATASETS dict.
- Masking (mask_mode, mask_file, mask_angles, mask_translation) is applied
  identically to every region in a single run_pipeline call -- there is no
  per-region mask_mode override in this config. masking.apply_mask itself
  already supports different mask modes across different regions in one
  call (see that function's own documentation), so building a mixed-mode
  region_specs list by hand and calling the lower-level pieces of
  run_pipeline directly is possible today; SpatialAnalysisConfig simply
  does not expose that construction as a single field yet.
"""

import os  # path construction
import sys  # sys.path manipulation, for add_script_folders
from dataclasses import dataclass, field  # SpatialAnalysisConfig
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
        ACTIVE_SCRIPT_DIRS (below). Not called by any other file.

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
        A path whose drive does not exist (relevant on Windows, where a
        path like "E:\\..." may reference a drive letter not present on
        the current machine) is skipped entirely for that starting path,
        rather than raising -- this is what makes it safe to leave
        ACTIVE_SCRIPT_DIRS pointing at a specific development machine's
        paths without this script failing to even import on a different
        machine; it simply will not find those modules there.
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
    r"E:\current_superposition_code\src\_mumax3",  # edit this for your machine -- see add_script_folders
]

SUPPORTED_FIELDS = ("m_full", "B_demag", "B_exch")  # this parent script only supports these field types -- see this file's module-level LIMITATIONS section


# --------------------------
# ---- Everything a user edits lives here ----
@dataclass
class SpatialAnalysisConfig:
    """
    GOAL:
        Hold every setting run_pipeline needs -- the single object a
        caller constructs and passes to run_pipeline to perform a spatial
        analysis run. See the inline comment on each field below for what
        it controls; fields are grouped by the pipeline step they affect.

    See run_pipeline's own documentation for what each step actually does
    with these settings.
    """
    # ---- Paths ----
    data_directory: str  # directory containing the mumax3 .csv output files
    output_dir: Optional[str] = None  # default: data_directory/mumax3_analysis_output
    npz_directory: Optional[str] = None  # default: data_directory/temp/npz

    # ---- Simulation grid (nm) ----
    cell_size_x: float = 1.0  # nm per grid cell, x direction
    cell_size_y: float = 1.0  # nm per grid cell, y direction

    # ---- Which fields to analyze -- subset of SUPPORTED_FIELDS ----
    fields: Tuple[str, ...] = SUPPORTED_FIELDS

    # ---- Region centers/sizes, in grid units (cells) -- `center` is the absolute
    # grid index of the region's center cell, the same convention used everywhere
    # else in this codebase. ----
    centers: List[Tuple[int, int]] = field(default_factory=list)  # one (cx, cy) per region
    sizes: List[Tuple[int, int]] = field(default_factory=list)  # one (width, height) per region, matched by position to `centers`
    region_titles: Optional[Dict[str, str]] = None  # optional {label: title}; auto-generated from each region's center if None
    region_labels: Optional[List[str]] = None  # optional explicit label per (center, size) pair, forwarded to
    # region_extraction.build_region_specs(labels=...) -- one entry per `centers`. None
    # (default): auto-generated "center_cx_cy" labels. Mainly useful for canvas
    # split/zoom regions (region_extraction.build_canvas_split_centers_sizes /
    # build_canvas_zoom_centers_sizes already return meaningful labels like
    # "canvas_split_1_4_row0_col2" or "canvas_zoom_device_8"), which should be passed
    # straight through here instead of getting a meaningless "center_x_y" label.
    plot_origin_cells: Optional[Tuple[float, float]] = None  # absolute grid position
    # (cells) to map to (0, 0) on every region's plot axes, instead of each region's
    # own local extent starting at (0, 0) (the default, unaffected by this field being
    # None). Set this to, for example, the canvas center to make every region's plot
    # share DeviceArray's own canvas-centered coordinate convention -- see
    # mumax3_csv_plot.compute_region_extent for the full explanation, and
    # mumax3_parent_superposition.py's STEP 7 for where the canvas pass sets this
    # automatically. Requires region_meta's "center_cells" entry (from
    # region_extraction.build_region_meta) to have any effect -- silently falls back
    # to the old per-region-local behavior otherwise.
    save_plot_data_csv: bool = True  # also write a companion .csv of the exact
    # plotted array next to every PNG (both STEP 3's color maps and STEP 3.5's quiver
    # plots) -- see mumax3_csv_plot.save_plotted_data_csv. Same mumax3-style stacked
    # format used throughout this codebase, so it round-trips through
    # mumax3_csv_reader.process_csv_file if it is ever loaded back that way, or can
    # be opened directly for standalone re-analysis (see plot_single_field.py). Off
    # by default since it roughly doubles the output file count.

    # ---- Masking (applied identically to every region above -- see this file's
    # module-level LIMITATIONS section regarding mixed per-region modes) ----
    mask_mode: str = "none"  # 'none' | 'ellipse' | 'file'
    mask_file: Optional[str] = None  # required if mask_mode == 'file'
    mask_angles: Optional[List[float]] = None  # per-region rotation for mask_mode='file';
    # one entry per region in `centers`, or None for all-zero. A single shared angle for
    # every region is just [angle] * len(centers) -- per-region angles (for example, each
    # ASI device's own geometry angle) are exactly why this is not a single scalar.
    mask_translation: bool = False  # re-align/re-crop the mask to each region's own center

    # ---- Execution ----
    use_cached_npz: bool = False  # reuse a valid existing .npz cache instead of reconverting from CSV
    max_workers: int = 2  # parallel worker processes for region/mask extraction (STEP 2)
    plot_spatial_maps: bool = True  # STEP 3: color-map plots (magnitude, angle, components, hue, gradient)
    plot_spatial_limiter: Optional[int] = 2  # cap the number of frames plotted, for a quick test run
    plot_max_workers: int = 8  # parallel worker processes for STEP 3/3.5 plotting (color maps + quiver) -- independent of max_workers above (STEP 2's region/mask extraction) and of stats_max_workers below (STEP 4). Applied per timestep-or-region call (see mumax3_csv_plot.plot_all_regions_over_time's own processor_mode) -- automatically capped to the actual timestep/region count for that call, and automatically skipped (no pool at all) below 2 timesteps/regions or when force_serial=True, so this default is safe to leave even for small runs; lower it only if plotting itself is oversubscribing a memory- or core-constrained machine.
    force_serial: bool = False  # master override: when True, STEP 3/3.5 (plotting) and STEP 4 (stats) NEVER open their own ProcessPoolExecutor for this call, regardless of plot_max_workers/stats_max_workers -- set this when THIS SpatialAnalysisConfig/run_pipeline call is itself already running inside an outer worker pool (for example, propagation_engine.py's analyze_canvas, called from inside canvas_pool -- see PropagationConfig.parallelize_step_candidates). Opening a second, nested pool inside an already-parallel worker oversubscribes CPUs rather than speeding anything up. Does NOT affect STEP 2 (region/mask extraction), which already skips its own pool automatically whenever there is only one frame to process (the case for every propagation canvas) -- see frame_pipeline.process_frames_parallel's own TRACKED CHANGE comment.
    Ms: float = 800e3  # A/m -- used for m_full's plot color-scale limits
    suptitle: str = "Spatial Analysis"  # figure-level title for every spatial plot

    # ---- Weighted circular statistics (STEP 4) -- off by default so a caller that
    # does not set compute_stats keeps getting exactly the plain `regions` return
    # value, unchanged. ----
    compute_stats: bool = False
    plot_region_stats: bool = False  # only meaningful if compute_stats=True
    stats_ellipse_size: Optional[Tuple[float, float]] = None  # an OPTIONAL EXTRA
    # restriction, applied on top of each region's own masked_x/masked_y (which
    # already reflects that region's own mask_mode -- 'none'/'ellipse'/'file', the
    # same data STEP 3's plots use). For a case like "only the inner portion of an
    # already file-masked device" -- not a way to recreate mask_mode here, and still
    # ONE shared (a, b) applied identically to every region, unlike mask_mode=
    # 'ellipse' in STEP 2/3, which auto-derives per-region semi-axes from each
    # region's own `sizes` entry. Leave None (default) to just use each region's own
    # real mask.
    stats_time_step_ns: float = 0.1  # x-axis spacing for plot_region_stats_over_time / the CSV time_ns column
    stats_max_workers: int = 1  # greater than 1 splits regions across separate
    # processes for STEP 4 (see mumax3_csv_stats.compute_stats_for_all_timesteps_
    # parallel's own documentation for the two independent performance levers this
    # controls) -- independent of max_workers above (which is for region/mask
    # extraction, STEP 2). Default 1 (sequential, still fully vectorized per region)
    # since the multiprocessing win is hardware- and region-count-dependent: for a
    # handful of large regions, the per-process array pickling overhead can outweigh
    # the parallelism, while it mainly pays off with many modest-sized regions (for
    # example, a full ASI device array) on a machine with several free cores.
    # Benchmark both ways for your own data before leaving this above 1.

    # ---- CSV export of the same stats, via mumax3_csv_stats.save_region_stats_csv.
    # Kept as its own toggle since the numbers may be wanted without the plots, or
    # vice versa. ----
    save_stats_csv: bool = False
    save_stats_csv_per_frame: bool = False  # one CSV per timestep, every region in that frame
    save_stats_csv_all_frames: bool = False  # one CSV with every region x every timestep

    # ---- Averaged quiver plots (STEP 3.5) -- ONE arrow per NxM-cell window instead
    # of a color map; independent toggle from plot_spatial_maps above, so quiver
    # plots alone can be regenerated by setting plot_spatial_quiver=True with
    # compute_stats=False and plot_spatial_maps=False. Applies to every region in
    # `regions` -- every entry in config.centers/sizes, the SAME regions STEP 3's
    # color maps cover, whether those are individual device regions or canvas
    # split/zoom tiles -- there is no separate "which regions get quiver" filter;
    # configure a second run_pipeline call with a narrower `centers` list for a
    # quiver-only subset. ----
    plot_spatial_quiver: bool = False
    quiver_window: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = (2, 2)
    # (rows, cols) cell-count averaging window -- for example (2, 2) or (2, 8), need
    # not be square. A plain tuple applies to every region. A dict is looked up by
    # region label, then mask_type ("File Mask"/"Elliptical Mask"/"No Mask"), then
    # center_name, then a "default" key -- see mumax3_csv_plot.resolve_quiver_window.
    # The window's actual placement is recomputed from each region/frame's real
    # non-NaN footprint every call (not assumed from mask_mode/size) -- see
    # mumax3_plot_vectors.compute_averaged_quiver_grid for the edge-detection rules
    # (bottom-left bounding-box start, a partially-covered window still gets one
    # arrow, a region with no NaN gaps at all -- for example, B_demag with
    # mask_mode='none' -- naturally falls back to tiling the full rectangle).

    skip_npz_conversion: bool = False  # True when npz_directory already holds
    # ready-to-use .npz frames that did NOT come from CSV conversion (for example,
    # mumax3_parent_superposition.py's composed-canvas export) -- STEP 1 would
    # otherwise try to glob for .csv files that do not exist there and reset the
    # directory, destroying the already-exported frames.

    extracted_regions_subdir: str = "extracted_regions"  # the per-region/per-dataset
    # extraction cache (STEP 2 below) lives at
    # npz_directory/<this>/extracted_regions_<dataset>.npz. Only matters if two
    # different run_pipeline calls are pointed at the SAME npz_directory with
    # DIFFERENT `centers`/`sizes` (for example, mumax3_parent_superposition.py's
    # STEP 6 -- per-device regions -- and STEP 7 -- canvas split/zoom/full regions --
    # both read from the same composed-canvas npz_directory, but extract completely
    # different regions from it): give the second call its own subdir (for example,
    # "extracted_regions_canvas") so the two caches do not collide -- with
    # use_cached_npz=True, a collision means whichever call runs second silently
    # loads the first call's cached region data instead of computing its own. The
    # plain single-call case (the vast majority of usage) is unaffected by this
    # default.

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
            if `centers` is empty, if `sizes` is not the same length as
            `centers`, or if mask_mode='file' but mask_file is not set.
        """
        if self.output_dir is None:  # no explicit output directory given: default under data_directory
            self.output_dir = os.path.join(self.data_directory, "mumax3_analysis_output")
        if self.npz_directory is None:  # no explicit npz cache directory given: default under data_directory
            self.npz_directory = os.path.join(self.data_directory, "temp", "npz")
        bad_fields = set(self.fields) - set(SUPPORTED_FIELDS)  # any requested field this parent script does not support
        if bad_fields:
            raise ValueError(f"Unsupported field(s) {bad_fields}; this parent script only "
                              f"supports {SUPPORTED_FIELDS}.")
        if not self.centers:  # at least one region must be requested
            raise ValueError("config.centers must have at least one (cx, cy).")
        if len(self.sizes) != len(self.centers):  # centers and sizes must describe the same regions
            raise ValueError("config.sizes must be the same length as config.centers.")
        if self.mask_mode == "file" and not self.mask_file:  # a file mask requires a file path
            raise ValueError("config.mask_file must be set when mask_mode='file'.")


# --------------------------
# ---- The pipeline itself ----
def run_pipeline(config: SpatialAnalysisConfig):
    """
    GOAL:
        Run NPZ conversion, unified region/mask extraction, spatial
        plotting, averaged quiver plots, and (optionally) weighted
        circular statistics, for every field in config.fields, using the
        same region_specs for all of them. This is the single entry point
        both this script's own __main__ block and mumax3_parent_
        superposition.py's post-composition analysis step call.

    HOW IT TALKS TO OTHER FILES:
        Calls mumax3_parent_utilities.convert_npz_per_dataset (STEP 1),
        region_extraction.build_region_specs/build_region_meta and
        mumax3_parent_utilities.region_processor/compute_regions (STEP 2),
        mumax3_plot_calls.build_plot_configs_mfull/build_plot_configs_
        Bdemag and mumax3_parent_utilities.run_spatial_plots (STEP 3),
        mumax3_plot_calls.build_plot_configs_quiver and mumax3_parent_
        utilities.run_spatial_plots again (STEP 3.5), and mumax3_csv_stats.
        compute_stats_for_all_timesteps_parallel/save_region_stats_csv
        plus mumax3_csv_plot.plot_region_stats_over_time (STEP 4).

    PARAMETERS:
        config (SpatialAnalysisConfig): every setting for this run -- see
            that class's own field-by-field documentation.

    RETURNS:
        - If config.compute_stats is False (default): Dict[field_name ->
          Dict[region_label -> {masked_x/y/z, x/y/z}]].
        - If config.compute_stats is True: (regions, stats_results), where
          stats_results is Dict[field_name -> Dict[region_label ->
          Dict[metric_name -> List[float] over time]]] -- field_name/
          region_label keyed the same way as `regions`, with each region's
          value being the output of compute_stats_for_all_timesteps_
          parallel.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section.
    """
    verbose = config.verbose  # master verbose toggle, forwarded into every step below
    os.makedirs(config.npz_directory, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)

    from mumax3_parent_utilities import convert_npz_per_dataset, region_processor, compute_regions, run_spatial_plots, validate_field_processors
    from mumax3_pipeline_utilities import matrix_magnitude, matrix_component, compute_global_scalar_limits
    from mumax3_plot_calls import build_plot_configs_Bdemag, build_plot_configs_mfull
    from region_extraction import build_region_specs, build_region_meta

    # ---- STEP 1: NPZ conversion (only for the requested, supported fields) ----
    if config.skip_npz_conversion:  # npz_directory already holds ready-to-use frames -- see SpatialAnalysisConfig.skip_npz_conversion
        confirmed_datasets, confirmed_metadata = validate_field_processors(set(config.fields), verbose=verbose)
    else:
        confirmed_datasets, confirmed_metadata = convert_npz_per_dataset(
            config.data_directory, config.npz_directory, set(config.fields),
            config.use_cached_npz, verbose=verbose,
        )

    # ---- STEP 2: build ONE region_specs list, reused unchanged for every dataset ----
    # This is the unification point: the exact same region_specs, and the exact same
    # compute_regions()/process_frames_parallel() call below, are what mumax3_parent_
    # superposition.py's post-composition step also uses -- just pointed at a
    # different npz_directory/prefix (the composed canvas instead of raw CSV output).
    region_specs = build_region_specs(
        centers=config.centers,
        sizes=config.sizes,
        mask_mode=config.mask_mode,
        mask_file=config.mask_file,
        angles=config.mask_angles if config.mask_angles is not None else [0.0] * len(config.centers),
        translation=config.mask_translation,
        labels=config.region_labels,
        verbose=verbose,
    )
    # {label -> {"center_name","mask_type","center_cells"}} -- drives the output
    # folder layout below (STEP 3 spatial plots and STEP 4 stats), so regions sharing
    # a center but using different mask_mode values land in separate
    # <center>/<mask type>/ folders instead of overwriting each other.
    region_meta = build_region_meta(region_specs, verbose=verbose)

    npz_dir_processed = os.path.join(config.npz_directory, config.extracted_regions_subdir)
    os.makedirs(npz_dir_processed, exist_ok=True)

    regions: Dict[str, Dict] = {}  # accumulated across every dataset below
    for dataset in confirmed_datasets:  # extract (and mask) every region, for one dataset at a time
        description = confirmed_metadata.get(dataset, {}).get("description", f"{dataset} regions")
        regions[dataset] = region_processor(
            output_path=os.path.join(npz_dir_processed, f"extracted_regions_{dataset}.npz"),
            compute_fn=compute_regions(
                npz_directory=config.npz_directory,
                cell_size_x=config.cell_size_x,
                cell_size_y=config.cell_size_y,
                region_specs=region_specs,
                prefix=dataset,
                max_workers=config.max_workers,
                verbose=verbose,
            ),
            use_cached=config.use_cached_npz,
            description=description,
            verbose=verbose,
        )

    # ---- STEP 3: plotting (skippable independently of STEP 4 below) ----
    if config.plot_spatial_maps:
        region_titles = config.region_titles or {  # auto-generate a title per region if none was given
            spec.resolved_label(): (
                f"Region {i + 1}: Center ({spec.center[0] * config.cell_size_x:.0f} nm, "
                f"{spec.center[1] * config.cell_size_y:.0f} nm)"
            )
            for i, spec in enumerate(region_specs)
        }
        common_args = dict(
            titles=region_titles,
            suptitles=config.suptitle,
            max_frames=config.plot_spatial_limiter,
            cell_size_x=config.cell_size_x,
            cell_size_y=config.cell_size_y,
            processor_mode="regions",
            max_workers=config.plot_max_workers,  # was previously left unset here, silently falling back to plot_all_regions_over_time's own hardcoded default (8) regardless of this config -- see SpatialAnalysisConfig.plot_max_workers' own documentation
            force_serial=config.force_serial,  # see SpatialAnalysisConfig.force_serial's own documentation
        )

        for dataset, region_data in regions.items():  # one plot_configs set per dataset (m_full has its own component/scale conventions from B_-style fields)
            if dataset == "m_full":
                if verbose:
                    print(f"[INFO] run_pipeline: plotting m_full...")
                plot_configs = build_plot_configs_mfull(-config.Ms, config.Ms, verbose=verbose)
            else:  # B_demag (or any other B_-style field routed through the same magnitude/x/y/z convention)
                if verbose:
                    print(f"[INFO] run_pipeline: plotting {dataset}...")
                vmin_mag, vmax_mag = compute_global_scalar_limits(region_data, lambda r, t: matrix_magnitude(r, t, field_scaling=10000))
                vmin_x, vmax_x = compute_global_scalar_limits(region_data, lambda r, t: matrix_component(r, t, comp='masked_x', field_scaling=10000))
                vmin_y, vmax_y = compute_global_scalar_limits(region_data, lambda r, t: matrix_component(r, t, comp='masked_y', field_scaling=10000))
                vmin_z, vmax_z = compute_global_scalar_limits(region_data, lambda r, t: matrix_component(r, t, comp='masked_z', field_scaling=10000))
                plot_configs = build_plot_configs_Bdemag(vmin_mag, vmax_mag, vmin_x, vmax_x, vmin_y, vmax_y, vmin_z, vmax_z, verbose=verbose)
            # Output layout: data_type/plot_type/scale/<frame>/<center>/<mask type>/plot.png
            run_spatial_plots(regions=region_data, base_output_dir=os.path.join(config.output_dir, dataset),
                               plot_configs=plot_configs, common_args=common_args, region_meta=region_meta,
                               plot_origin_cells=config.plot_origin_cells, save_plot_data_csv=config.save_plot_data_csv,
                               verbose=verbose)
        if verbose:
            print("[INFO] run_pipeline: spatial analysis complete.")

    # ---- STEP 3.5: averaged quiver plots (opt-in; see SpatialAnalysisConfig.
    # plot_spatial_quiver) -- independent of STEP 3's color maps and STEP 4's stats
    # below: can be run alone (plot_spatial_maps=False, compute_stats=False,
    # plot_spatial_quiver=True) to only redo quiver plots, still respecting STEP 2's
    # use_cached_npz-gated region extraction above (quiver reads the SAME
    # already-extracted `regions[dataset]` dict STEP 3/4 use -- no separate
    # extraction pass). Applies to every dataset in config.fields (m_full, B_demag,
    # ...) via the same matrix_uv scalar_func regardless of dataset. ----
    if config.plot_spatial_quiver:
        from mumax3_plot_calls import build_plot_configs_quiver

        for dataset, region_data in regions.items():  # one quiver pass per dataset
            field_scaling = confirmed_metadata.get(dataset, {}).get("scale", 1.0)
            if verbose:
                print(f"[INFO] run_pipeline: plotting {dataset} quiver (window={config.quiver_window})...")
            quiver_configs = build_plot_configs_quiver(dataset, field_scale=field_scaling, verbose=verbose)
            quiver_common_args = dict(
                titles=config.region_titles or {
                    spec.resolved_label(): spec.resolved_label() for spec in region_specs
                },
                suptitles=config.suptitle,
                max_frames=config.plot_spatial_limiter,
                cell_size_x=config.cell_size_x,
                cell_size_y=config.cell_size_y,
                processor_mode="regions",
                quiver_window=config.quiver_window,
                max_workers=config.plot_max_workers,  # see common_args' own identical addition above (STEP 3) -- same config field, same reason
                force_serial=config.force_serial,
            )
            run_spatial_plots(regions=region_data, base_output_dir=os.path.join(config.output_dir, dataset),
                               plot_configs=quiver_configs, common_args=quiver_common_args, region_meta=region_meta,
                               plot_origin_cells=config.plot_origin_cells, save_plot_data_csv=config.save_plot_data_csv,
                               verbose=verbose)
        if verbose:
            print("[INFO] run_pipeline: quiver plotting complete.")

    # ---- STEP 4: weighted circular statistics (opt-in; see SpatialAnalysisConfig.
    # compute_stats). Uses compute_stats_for_all_timesteps_parallel, which reads each
    # region's masked_x/masked_y (mask-mode-aware, the same arrays STEP 3's plots
    # use) -- see that function's own documentation in mumax3_csv_stats.py. The same
    # `regions[dataset]` dict STEP 3 already built is handed in directly;
    # field_scaling comes from the same DATASETS metadata (`confirmed_metadata`)
    # STEP 1 already resolved, so m_full and B_demag/B_exch/etc. each get their own
    # correct scale automatically. Plots (plot_region_stats_over_time) and CSVs
    # (save_region_stats_csv) are both written under `output_dir/<dataset>/stats/...`
    # -- so m_full and B_demag never share a file -- and the CSVs come in two
    # separate forms: one CSV per frame (every region in that frame) and one CSV
    # total (every region x every frame). save_region_stats_csv resets the per-frame
    # directory each call, so results from one run_pipeline call never mix with a
    # previous call's leftovers. stats_ellipse_size is an OPTIONAL extra restriction
    # layered on top of each region's own mask, not a substitute for it -- leave it
    # None to just use the region's own mask. ----
    stats_results: Dict[str, Dict] = {}
    if config.compute_stats:
        from mumax3_csv_stats import compute_stats_for_all_timesteps_parallel, save_region_stats_csv
        from mumax3_csv_plot import plot_region_stats_over_time

        effective_stats_max_workers = 1 if config.force_serial else config.stats_max_workers  # see SpatialAnalysisConfig.force_serial's own documentation -- compute_stats_for_all_timesteps_parallel already skips its own pool below 2 regions, but a caller running inside an outer pool (force_serial=True) must not open one even with >1 region
        if config.force_serial and config.stats_max_workers > 1 and verbose:
            print(f"[WARNING] run_pipeline: force_serial=True -- ignoring stats_max_workers={config.stats_max_workers} for this call (using 1) to avoid a nested pool inside an outer worker.")
        for dataset, region_data in regions.items():  # one stats pass per dataset
            field_scaling = confirmed_metadata.get(dataset, {}).get("scale", 1.0)
            if verbose:
                print(f"[INFO] run_pipeline: computing circular stats for {dataset} (field_scaling={field_scaling})...")
            stats_results[dataset] = compute_stats_for_all_timesteps_parallel(
                region_data,
                field_scaling=field_scaling,
                ellipse_size=config.stats_ellipse_size,
                max_workers=effective_stats_max_workers,
                verbose=verbose,
            )
            stats_out_dir = os.path.join(config.output_dir, dataset, "stats")  # dataset-specific subdir -> each dataset's plots and CSVs land in their own folder, never mixed together

            if config.plot_region_stats:
                plot_region_stats_over_time(
                    stats_results[dataset],
                    output_dir=stats_out_dir,
                    time_step_ns=config.stats_time_step_ns,
                    region_meta=region_meta,
                    verbose=verbose,
                )

            if config.save_stats_csv:
                save_region_stats_csv(
                    stats_results[dataset],
                    output_dir=os.path.join(stats_out_dir, "csv"),
                    time_step_ns=config.stats_time_step_ns,
                    save_per_frame=config.save_stats_csv_per_frame,
                    save_all_frames=config.save_stats_csv_all_frames,
                    region_meta=region_meta,
                    verbose=verbose,
                )
        if verbose:
            print("[INFO] run_pipeline: circular statistics complete.")
        return regions, stats_results

    return regions


if __name__ == "__main__":
    add_script_folders(ACTIVE_SCRIPT_DIRS, levels_up=0, verbose=True)

    config = SpatialAnalysisConfig(
        data_directory=r"E:\04_Angle_Dependence_MTJs\test_data",
        cell_size_x=1, cell_size_y=1,
        fields=("m_full", "B_demag"),
        # Example: 1 region at the simulation center. Add more (cx, cy)/(w, h) pairs
        # for a multi-region sweep -- the same run_pipeline call handles any number.
        centers=[(1500, 1500)],
        sizes=[(3000, 3000)],
        mask_mode=["file"],
        mask_file=r"E:\04_Angle_Dependence_MTJs\test_data\temp\npz\m_full_npz_cache\m_full000000.npz",
        Ms=800e3,
        # Compute + plot weighted circular stats for each region/dataset, on top of
        # the spatial maps above. Stats use each region's own masked_x/masked_y
        # (mask-mode-aware -- see compute_stats_for_all_timesteps_parallel's
        # documentation), so they match what the STEP 3 plots show for that region.
        compute_stats=False,
        stats_time_step_ns=0.1,
        verbose=False,  # turn on progress/debug printing across the whole pipeline
    )
    result = run_pipeline(config)
    if config.compute_stats:
        regions, stats_results = result
    else:
        regions = result
