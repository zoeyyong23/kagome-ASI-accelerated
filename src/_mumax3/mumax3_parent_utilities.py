r"""
File Name: mumax3_parent_utilities.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v7

============================================================
PURPOSE
============================================================
Provides the utility functions both parent scripts
(mumax3_parent_spatial_analysis.py and mumax3_parent_superposition.py) use
to convert CSV files to cached .npz data, extract/mask regions with
caching, and dispatch plotting across every configured plot type and scale.
Specifically:
    - csv_npz_pairs_valid / npz_convert / convert_npz_per_dataset: convert
      mumax3 CSV output to .npz, per dataset, reusing a valid cache when one
      exists.
    - validate_field_processors: validates a list of requested dataset
      names (for example, "m_full", "B_demag") against the set this
      pipeline knows how to scale/describe.
    - region_processor: generic caching wrapper for any computation whose
      result can be saved to and reloaded from a single .npz file.
    - compute_regions: builds the specific compute function region_processor
      caches for the region-extraction-and-masking step, via
      frame_pipeline.process_frames_parallel.
    - run_spatial_plots: dispatches every configured plot type/scale
      combination to mumax3_csv_plot.plot_all_regions_over_time, splitting
      each plot config's arguments between the ones the plotting function
      itself needs and the ones its scalar_func needs.

============================================================
LIMITATIONS
============================================================
- run_spatial_plots' PLOT_FUNC_KEYS set (inside that function) determines
  which keys in a plot config's combined arguments are passed to
  plot_all_regions_over_time itself, versus which are passed to the
  scalar_func that function calls per region/frame. Adding a new argument
  to either mumax3_csv_plot.plot_all_regions_over_time's signature or to a
  scalar_func's signature that happens to share a name with an existing key
  in the other group will misroute that argument; new plotting-function
  parameters should be added to PLOT_FUNC_KEYS explicitly.
"""

from functools import partial  # used to bind scalar_func's fixed keyword arguments in run_spatial_plots


# --------------------------
# ---- Verify that every CSV file has a valid, up-to-date .npz counterpart ----
def csv_npz_pairs_valid(csv_dir, npz_dir, prefix, verbose=False):
    """
    GOAL:
        Check whether every CSV file matching `prefix` in `csv_dir` has a
        corresponding .npz file in `npz_dir` that is newer than the CSV --
        i.e. whether the .npz cache is fully up to date.

    HOW IT TALKS TO OTHER FILES:
        Called by npz_convert (this file) to decide whether a fresh CSV
        conversion is needed.

    PARAMETERS:
        csv_dir (str): directory containing the CSV files.
        npz_dir (str): directory expected to contain the corresponding .npz
            files.
        prefix (str): filename prefix to match, e.g. "B_demag".
        verbose (bool): prints the validation result and, if invalid, which
            check failed, when True.

    RETURNS:
        bool: True if every matching CSV file has an up-to-date .npz
        counterpart; False otherwise, including when no CSV files match at
        all (nothing to validate against).
    """
    import os  # file existence and modification-time checks
    import glob  # finding CSV files matching the given prefix
    csv_files = sorted(glob.glob(os.path.join(csv_dir, f"{prefix}*.csv")))  # every CSV file matching this prefix
    if not csv_files:  # nothing to validate a cache against
        if verbose:
            print(f"[DEBUG] csv_npz_pairs_valid: no CSV files found matching prefix '{prefix}' in '{csv_dir}'.")
        return False
    for csv_path in csv_files:  # check every CSV file has a valid, up-to-date .npz counterpart
        base = os.path.splitext(os.path.basename(csv_path))[0]  # CSV filename without its extension
        npz_path = os.path.join(npz_dir, base + ".npz")  # the .npz file this CSV should correspond to
        if not os.path.exists(npz_path):  # the expected .npz file does not exist at all
            if verbose:
                print(f"[DEBUG] csv_npz_pairs_valid: missing '{npz_path}' for '{csv_path}'.")
            return False
        if os.path.getmtime(npz_path) < os.path.getmtime(csv_path):  # the .npz file is older than its source CSV
            if verbose:
                print(f"[DEBUG] csv_npz_pairs_valid: '{npz_path}' is older than '{csv_path}'.")
            return False
    if verbose:
        print(f"[DEBUG] csv_npz_pairs_valid: all {len(csv_files)} CSV/npz pair(s) valid for prefix '{prefix}'.")
    return True


# --------------------------
# ---- Ensure the npz cache for one dataset is valid, regenerating it if necessary ----
def npz_convert(csv_dir, npz_dir, prefix, use_cached, verbose=False):
    """
    GOAL:
        Ensure a valid .npz cache exists for one dataset (for example,
        "B_demag"), converting from CSV if the cache is missing, invalid,
        or caching is disabled.

    HOW IT TALKS TO OTHER FILES:
        Called by convert_npz_per_dataset (this file), once per requested
        dataset. Calls csv_npz_pairs_valid (this file) to check cache
        validity, mumax3_pipeline_utilities.move_and_reset_directory to
        clear a stale cache directory, and
        mumax3_csv_reader.process_all_csv_files to perform the actual
        conversion.

    PARAMETERS:
        csv_dir (str): directory containing the CSV files for this dataset.
        npz_dir (str): base directory the .npz cache is organized under; a
            "{prefix}_npz_cache" subdirectory is created within it.
        prefix (str): filename prefix for this dataset, e.g. "B_demag".
        use_cached (bool): if True and a valid cache already exists, reuse
            it without reconversion. If False, always reconvert.
        verbose (bool): prints which path was taken (using cache,
            regenerating due to invalid cache, or regenerating because
            caching is disabled) when True. Also forwarded to
            process_all_csv_files, so per-file conversion messages appear
            too.

    RETURNS:
        None. The .npz cache directory for this dataset is left in a valid
        state on disk.
    """
    import os  # path construction
    from mumax3_pipeline_utilities import move_and_reset_directory  # clearing a stale cache directory before regenerating it
    from mumax3_csv_reader import process_all_csv_files  # performs the actual CSV-to-npz conversion
    npz_dir_gen = os.path.join(npz_dir, f"{prefix}_npz_cache")  # this dataset's own cache subdirectory, keyed by prefix so multiple datasets never collide
    valid_cache = csv_npz_pairs_valid(csv_dir, npz_dir_gen, prefix, verbose=verbose)  # whether the existing cache (if any) is fully up to date
    if not use_cached or not valid_cache:  # caching disabled, or the existing cache is stale/missing
        if verbose:
            if use_cached:
                print(f"[INFO] npz_convert: cache invalid for '{prefix}'; regenerating NPZ files...")
            else:
                print(f"[INFO] npz_convert: caching disabled; generating NPZ files for '{prefix}'...")
        move_and_reset_directory(npz_dir_gen, verbose=verbose)  # clear out any stale .npz files before regenerating
        process_all_csv_files(csv_dir, prefix, npz_dir_gen, verbose=verbose)  # convert every matching CSV file to .npz
    else:  # the existing cache is valid and caching is enabled
        if verbose:
            print(f"[INFO] npz_convert: using valid cached NPZ files for '{prefix}'.")


# --------------------------
# ---- Known mumax3 spatial datasets and their scaling/description metadata ----
def validate_field_processors(required_datasets, verbose=False):
    """
    GOAL:
        Validate a list of requested dataset names against the set this
        pipeline knows how to scale and describe, returning only the
        recognized ones along with their metadata.

    HOW IT TALKS TO OTHER FILES:
        Called by convert_npz_per_dataset (this file) and by both parent
        scripts wherever a dataset's scaling factor is needed for plotting
        or stats.

    PARAMETERS:
        required_datasets (List[str]): dataset names to validate, for
            example ["m_full", "B_demag"].
        verbose (bool): prints the number of valid datasets found when
            True.

    RETURNS:
        Tuple[List[str], Dict[str, Dict]]: (valid_datasets, dataset_metadata)
        -- the subset of `required_datasets` that are recognized, and a
        dict mapping each of those to its {"scale", "description"} metadata.
        Unrecognized dataset names are skipped with a printed warning
        (always shown, not gated by `verbose` -- see this function's
        LIMITATIONS).

    LIMITATIONS:
        The set of known datasets (m_full, m, B_anis, B_custom, B_demag,
        B_eff, B_exch, B_ext, B_mel, B_therm) is fixed in this function's
        body; adding support for a new mumax3 output requires editing this
        function directly. An unrecognized dataset name produces a
        [WARNING] print (not an exception), so a typo in a requested
        dataset name is silently dropped rather than raising -- check the
        returned `valid_datasets` list length against the input if this
        matters for your use case.
    """
    DATASETS = {
        "m_full": {"scale": 1.0, "description": "m_full regions"},
        "m": {"scale": 1.0, "description": "m regions"},
        "B_anis": {"scale": 1e4, "description": "B_anis regions"},
        "B_custom": {"scale": 1e4, "description": "B_custom regions"},
        "B_demag": {"scale": 1e4, "description": "B_demag regions"},
        "B_eff": {"scale": 1e4, "description": "B_eff regions"},
        "B_exch": {"scale": 1e4, "description": "B_exch regions"},
        "B_ext": {"scale": 1e4, "description": "B_ext regions"},
        "B_mel": {"scale": 1e4, "description": "B_mel regions"},
        "B_therm": {"scale": 1e4, "description": "B_therm regions"},
    }  # every dataset this pipeline knows how to scale/describe, and the metadata for each

    valid_datasets = []  # subset of required_datasets that were recognized
    dataset_metadata = {}  # metadata for each recognized dataset
    for dataset in required_datasets:  # validate each requested dataset name
        if dataset in DATASETS:  # this dataset is known
            valid_datasets.append(dataset)
            dataset_metadata[dataset] = DATASETS[dataset]
        else:  # this dataset is not recognized
            print(f"[WARNING] validate_field_processors: dataset '{dataset}' is not recognized. Skipping.")
    if verbose:
        print(f"[DEBUG] validate_field_processors: {len(valid_datasets)} of {len(required_datasets)} requested dataset(s) recognized.")
    return valid_datasets, dataset_metadata


# --------------------------
# ---- Ensure the npz cache is valid for every requested dataset ----
def convert_npz_per_dataset(csv_dir, npz_dir, required_datasets, use_cached, verbose=False):
    """
    GOAL:
        Ensure a valid .npz cache exists for every requested, recognized
        dataset, converting from CSV as needed. This is the entry point
        both parent scripts' STEP 1 calls.

    HOW IT TALKS TO OTHER FILES:
        Calls validate_field_processors (this file) to filter and describe
        the requested datasets, then npz_convert (this file) once per valid
        dataset.

    PARAMETERS:
        csv_dir (str): directory containing the CSV files for every
            requested dataset (all datasets are expected in the same
            directory, distinguished by filename prefix).
        npz_dir (str): base directory the .npz cache is organized under.
        required_datasets (List[str]): dataset names requested by the
            caller.
        use_cached (bool): forwarded to npz_convert for every dataset.
        verbose (bool): prints a completion summary when True. Forwarded to
            validate_field_processors and npz_convert, so their own status
            messages appear too.

    RETURNS:
        Tuple[List[str], Dict[str, Dict]]: (datasets_to_process,
        dataset_metadata), the same values validate_field_processors
        returned -- the recognized dataset names and their metadata, after
        every one of them has had its .npz cache ensured valid.
    """
    datasets_to_process, dataset_metadata = validate_field_processors(required_datasets, verbose=verbose)  # filter to only recognized datasets, with their scaling/description metadata

    for dataset in datasets_to_process:  # ensure a valid .npz cache for each recognized dataset
        npz_convert(
            csv_dir,  # directory containing this dataset's CSV files
            npz_dir,  # base .npz cache directory
            dataset,  # this dataset's filename prefix
            use_cached,  # whether to reuse an existing valid cache
            verbose=verbose,
        )
    if verbose:
        print(f"[INFO] convert_npz_per_dataset: NPZ conversion completed for dataset(s): {datasets_to_process}.")
    return datasets_to_process, dataset_metadata


# --------------------------
# ---- Generic caching wrapper: reuse a saved .npz result, or compute and save a new one ----
def region_processor(output_path, compute_fn, use_cached, description="step", verbose=False):
    """
    GOAL:
        Generic caching wrapper for any computation whose result can be
        saved to and reloaded from a single .npz file: if a cached result
        already exists at `output_path` and `use_cached` is True, load and
        return it; otherwise call `compute_fn`, save its result, and return
        it.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts around compute_regions' output
        (this file) -- the region-extraction-and-masking step -- so a
        second run with the same inputs can skip re-extracting regions
        entirely.

    PARAMETERS:
        output_path (str): path to the cached .npz file -- both where a
            valid cache is looked for, and where a freshly computed result
            is saved.
        compute_fn (Callable[[], dict]): a zero-argument function returning
            a dict-like result to cache. See compute_regions for how this
            pipeline builds one.
        use_cached (bool): if True and `output_path` already exists, load
            and return it without calling `compute_fn`.
        description (str): human-readable description of this step, used
            only in status messages.
        verbose (bool): prints which path was taken (loading cache vs.
            computing fresh) when True.

    RETURNS:
        dict: either the loaded cached result, or the freshly computed
        (and now-cached) result. Every value in a freshly computed result
        is stored under a string key, converting non-string keys as
        needed for .npz compatibility.

    LIMITATIONS:
        Cached results are loaded with allow_pickle=True and each array
        unwrapped via `.item()`, which assumes every top-level value in the
        cached result is a 0-dimensional object array (as produced by
        `np.savez_compressed` on a dict of non-array Python objects, which
        is what compute_regions' result typically is). A compute_fn whose
        result values are plain numeric arrays rather than pickled Python
        objects would not round-trip correctly through this caching
        mechanism as written.
    """
    import os  # cache file existence check
    import numpy as np  # saving/loading the cached .npz result
    if use_cached and os.path.exists(output_path):  # a cached result exists and reuse is allowed
        if verbose:
            print(f"[INFO] region_processor: loading cached {description}.")
        loaded = np.load(output_path, allow_pickle=True)  # open the cached .npz file
        return {k: loaded[k].item() for k in loaded.files}  # unwrap each cached 0-dimensional object array back into its original Python object
    if verbose:
        print(f"[INFO] region_processor: computing {description}...")
    result = compute_fn()  # perform the actual computation
    result = {str(k): v for k, v in result.items()}  # ensure every key is a string, for .npz compatibility
    np.savez_compressed(output_path, **result)  # cache the result for future calls
    if verbose:
        print(f"[DEBUG] region_processor: cached {description} to '{output_path}'.")
    return result


# --------------------------
# ---- Build the region-extraction-and-masking compute function, for use with region_processor ----
def compute_regions(
    npz_directory,  # base directory containing this dataset's .npz cache
    cell_size_x,  # nm per cell, x direction
    cell_size_y,  # nm per cell, y direction
    region_specs,  # List[RegionSpec] -- see region_extraction.RegionSpec / build_region_specs
    prefix,  # this dataset's filename prefix, e.g. "B_demag"
    max_workers=2,  # parallel worker processes for frame_pipeline.process_frames_parallel
    reorganize=True,  # whether to group the result by region label (see process_frames_parallel)
    verbose=False,  # forwarded into the returned function, so per-frame progress appears when it is called
):
    """
    GOAL:
        Build the zero-argument compute function region_processor's caching
        expects, for the region-extraction-and-masking step. This is the
        single entry point both parent scripts use for that step: plain
        multi-region/multi-file analysis and post-superposition
        device-region analysis both call this exact function, differing
        only in which `npz_directory`, `prefix`, and `region_specs` they
        pass in.

    HOW IT TALKS TO OTHER FILES:
        Calls frame_pipeline.process_frames_parallel (inside the returned
        function) to perform the actual extraction/masking work. The
        function this returns is intended to be passed as region_processor's
        `compute_fn` argument.

    PARAMETERS:
        npz_directory, cell_size_x, cell_size_y, region_specs, prefix,
        max_workers, reorganize: forwarded directly to
        frame_pipeline.process_frames_parallel when the returned function is
        called -- see that function's own documentation.
        verbose (bool): forwarded to process_frames_parallel when the
            returned function is called, so per-frame/per-chunk progress
            messages appear if desired.

    RETURNS:
        Callable[[], dict]: a zero-argument function that, when called,
        runs process_frames_parallel with the arguments bound above and
        returns its result.
    """
    import os  # path construction
    from frame_pipeline import process_frames_parallel  # performs the actual extraction/masking work

    npz_directory_prefix = os.path.join(npz_directory, f"{prefix}_npz_cache")  # this dataset's specific .npz cache subdirectory

    def fn():
        return process_frames_parallel(
            npz_directory=npz_directory_prefix,  # this dataset's .npz cache directory
            cell_size_x=cell_size_x,  # nm per cell, x direction
            cell_size_y=cell_size_y,  # nm per cell, y direction
            region_specs=region_specs,  # which regions to extract/mask from every frame
            max_workers=max_workers,  # parallel worker processes
            prefix=prefix,  # this dataset's filename prefix
            reorganize=reorganize,  # whether to group the result by region label
            verbose=verbose,  # per-frame/per-chunk progress messages
        )
    return fn


# --------------------------
# ---- Dispatch every configured plot type/scale combination for a set of regions ----
def run_spatial_plots(regions, base_output_dir, plot_configs, common_args, region_meta=None,
                       plot_origin_cells=None, save_plot_data_csv=False, verbose=False):
    """
    GOAL:
        Render every plot type and scale combination described in
        `plot_configs`, for a given set of regions, by dispatching to
        mumax3_csv_plot.plot_all_regions_over_time once per (plot type,
        scale) pair.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts, once per dataset, after
        compute_regions/region_processor has produced that dataset's region
        data. Calls mumax3_pipeline_utilities.move_and_reset_directory to
        clear each plot type/scale's output directory before rendering into
        it, and mumax3_csv_plot.plot_all_regions_over_time to perform the
        actual rendering.

    PARAMETERS:
        regions (Dict): region data for one dataset, as returned by
            compute_regions/region_processor (grouped by region label).
        base_output_dir (str): root directory this dataset's plots are
            written under; each plot config's own name/scale becomes a
            subdirectory beneath it.
        plot_configs (List[Dict]): one entry per plot type (for example,
            "magnitude", "angle", "quiver"), each with its own "scales"
            list (for example, "global"/"local"/"custom") -- see
            mumax3_plot_calls.py for how these are built.
        common_args (Dict): arguments shared across every plot config in
            this call (for example, titles, cell_size_x/y, max_frames).
            Spread directly into plot_all_regions_over_time via **common_
            args below -- this is also how a caller controls plotting-
            level parallelism: include "max_workers" and/or "force_serial"
            here to reach plot_all_regions_over_time's own parameters of
            the same name (see SpatialAnalysisConfig.plot_max_workers/
            force_serial, which mumax3_parent_spatial_analysis.run_
            pipeline sets into common_args for exactly this purpose).
        region_meta: Optional[Dict[label -> {"center_name", "mask_type",
            "center_cells"}]], from region_extraction.build_region_meta.
            When given, each region's plot lands under
            .../<prefix>_<scale>_<t>/<center_name>/<mask_type>/<label>.png
            instead of directly under .../<prefix>_<scale>_<t>/<label>.png
            -- this keeps regions that share a center but differ in
            mask_mode from overwriting each other's output.
        plot_origin_cells: Optional[Tuple[float, float]], an absolute grid
            position (in cells) to map to (0, 0) on every region's plot
            axes, instead of each region's own local extent starting at
            (0, 0) -- see mumax3_csv_plot.compute_region_extent.
        save_plot_data_csv (bool): if True, also writes a companion .csv of
            the exact plotted array next to every PNG -- see
            mumax3_csv_plot.save_plotted_data_csv.
        verbose (bool): prints which plot type/scale is currently being
            rendered when True. Forwarded to plot_all_regions_over_time, so
            its own progress messages appear too.

    RETURNS:
        None. Every configured plot type/scale combination is rendered to
        disk under `base_output_dir`.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding
        PLOT_FUNC_KEYS and adding new plotting-function parameters.
    """
    import os  # output directory construction
    from mumax3_pipeline_utilities import move_and_reset_directory  # clearing each plot type/scale's output directory before rendering
    from mumax3_csv_plot import plot_all_regions_over_time  # performs the actual rendering

    for cfg in plot_configs:  # one iteration per plot type (magnitude, angle, quiver, ...)
        name = cfg["name"]  # this plot type's name, used for its output subdirectory

        for scale_mode in cfg["scales"]:  # one iteration per scale variant (global, local, custom, ...) of this plot type
            scale_name = scale_mode["name"]  # this scale variant's name, used for its output subdirectory

            out_dir = os.path.join(base_output_dir, name, scale_name)  # this plot type/scale's own output directory
            os.makedirs(out_dir, exist_ok=True)  # ensure it exists

            runner_args = cfg.get("runner_args", {})  # options controlling how this plot type is run (currently just reset_dir)
            plot_args = cfg.get("plot_args", {})  # arguments shared by every scale variant of this plot type

            if runner_args.get("reset_dir", True):  # by default, start each plot type/scale's output directory empty
                move_and_reset_directory(out_dir, verbose=verbose)

            if verbose:
                print(f"[INFO] run_spatial_plots: plotting '{name}' ({scale_name})...")

            combined_args = {
                **plot_args,  # this plot type's shared arguments
                **scale_mode.get("plot_args", {})  # this scale variant's own arguments, overriding shared ones where they overlap
            }

            # Split combined_args between what plot_all_regions_over_time itself accepts and
            # what its scalar_func accepts -- see this file's module-level LIMITATIONS section
            # for what adding a new plotting-function parameter requires.
            PLOT_FUNC_KEYS = {
                "field_scale", "plot_type", "max_workers", "titles", "suptitles", "cmap",
                "vmin", "vmax", "xlabel", "ylabel", "cbar_label", "prefix", "local_scale",
                "max_frames", "cell_size_x", "cell_size_y", "extent", "processor_mode",
                "quiver_window",  # (rows, cols) averaging window for plot_type="quiver" -- see mumax3_csv_plot.resolve_quiver_window
            }

            plot_func_args = {}  # arguments destined for plot_all_regions_over_time itself
            scalar_func_args = {}  # arguments destined for the scalar_func plot_all_regions_over_time calls per region/frame

            for k, v in combined_args.items():  # route each combined argument to the function that actually accepts it
                if k in PLOT_FUNC_KEYS:
                    plot_func_args[k] = v
                else:
                    scalar_func_args[k] = v

            wrapped_scalar_func = partial(
                scale_mode["scalar_func"],  # the underlying scalar_func for this plot type (for example, matrix_angle)
                vmin=cfg["vmin"],  # this plot type's global color-scale minimum
                vmax=cfg["vmax"],  # this plot type's global color-scale maximum
                use_local_scaling=scale_mode.get("local_scale", False),  # whether this scale variant uses per-frame scaling instead
                use_custom_scaling=scale_mode.get("custom_scale", [False, None, None]),  # optional explicit override range
            )

            plot_all_regions_over_time(
                regions=regions,  # region data to plot
                output_dir=out_dir,  # this plot type/scale's output directory
                scalar_func=wrapped_scalar_func,  # pre-bound scalar_func for this plot type/scale
                cmap=cfg.get("cmap"),  # color map for this plot type
                vmin=cfg["vmin"],  # global color-scale minimum
                vmax=cfg["vmax"],  # global color-scale maximum
                local_scale=scale_mode["local_scale"],  # whether this scale variant uses per-frame scaling
                prefix=f"{cfg['prefix']}_{scale_name}",  # filename prefix combining this plot type and scale
                region_meta=region_meta,  # folder-layout metadata (center/mask-type subfolders)
                plot_origin_cells=plot_origin_cells,  # shared-coordinate-frame origin, if requested
                save_plot_data_csv=save_plot_data_csv,  # whether to also export a companion CSV per plot
                verbose=verbose,  # per-frame/per-region progress messages
                **common_args,  # arguments shared across every plot config in this run_spatial_plots call
                **plot_func_args,  # this plot type/scale's own plotting-function arguments
            )
