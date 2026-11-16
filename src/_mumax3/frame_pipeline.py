r"""
File Name: frame_pipeline.py
Author: Connor Sullivan
Date: 2026.07.15
Revision: v3

============================================================
PURPOSE
============================================================
Coordinates converting a folder of .npz field frames into extracted, masked
region data. This file does not crop regions itself (region_extraction.py
performs cropping) and does not determine masking itself (masking.py
performs masking); its role is to load each frame and call those two files
in the correct order, across as many frames as exist in a directory, in
parallel.

Both parent scripts (mumax3_parent_spatial_analysis.py and
mumax3_parent_superposition.py) call this file to convert raw .npz frames
into region data; neither parent script crops or masks data directly.

============================================================
UNIFIED CALLING INTERFACE
============================================================
process_single_frame and process_frames_parallel accept only a plain
List[RegionSpec] (see region_extraction.RegionSpec), together with an
npz_directory and a filename prefix. This single interface covers every
case this pipeline supports:
    - 10 region centers across 800 files, all ellipse-masked
    - 1 region center across 2 files, with no mask
    - N device centers (from geometry.DeviceArray) across however many
      superposition-canvas regenerations exist, each rotated and translated
      per device

Constructing the appropriate region_specs (from a manual center list, or
from DeviceArray.device_centers/device_angles) is the responsibility of the
calling code, not this file. See region_extraction.build_region_specs for
the manual-list case.

============================================================
LIMITATIONS
============================================================
- region_specs is applied identically to every frame processed in a single
  call. This file does not support a region that moves or changes shape
  between frames (for example, a moving center for propagation-front
  tracking). Supporting that would require changes here, not only in
  region_extraction.py or masking.py.
- chunk_size (inside process_frames_parallel) is fixed at 5 frames per
  batch. This bounds memory use on large frame counts but is not currently
  exposed as a parameter. If this needs to be tuned for a specific machine's
  memory or CPU balance, it is a local variable near the top of that
  function's loop.
- If a single frame's .npz file is missing or corrupted,
  process_single_frame raises an exception inside its worker process, which
  surfaces as a concurrent.futures exception when future.result() is
  called. There is no per-frame retry or skip-and-continue behavior.
- process_frames_parallel skips ProcessPoolExecutor entirely (running
  process_single_frame directly, in-process) whenever end_index <=
  max_pool_worthwhile_frames (currently 1) -- see that function's own
  TRACKED CHANGE comment. A worker pool's startup cost is fixed and paid on
  every call regardless of how much work is inside it, so it is a net loss
  for the single-frame case, which is the ONLY case propagation_engine.py's
  per-step canvas region-extraction calls ever hit (compose_and_export_
  canvas always writes exactly one frame per canvas_dir). This does not
  change this function's return shape or the multi-frame path below in any
  way.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np  # used for stacking loaded field components into arrays

from region_extraction import RegionSpec, extract_regions, reorganize_regions  # cropping logic lives here
from masking import apply_mask  # masking logic lives here


# --------------------------
# ---- Process a single .npz frame: load it, crop every region, mask every region ----
def process_single_frame(
    frame_index: int,
    npz_directory: str,
    cell_size_x: float,
    cell_size_y: float,
    region_specs: List[RegionSpec],
    prefix: str = "B_demag",
    verbose: bool = False,
) -> Tuple[int, Dict[str, Dict[str, Any]]]:
    """
    GOAL:
        Load ONE frame from disk, crop every region in `region_specs` out of
        it, apply each region's own mask, and hand back the result. This is
        the unit of work that process_frames_parallel below runs many copies
        of at once (one call per frame, across worker processes).

    HOW IT TALKS TO OTHER FILES:
        - mumax3_csv_reader.load_npz_frame: loads the raw x/y/z arrays for
          this one frame from disk.
        - region_extraction.extract_regions: crops every region in
          `region_specs` out of the loaded frame (rectangular crop only --
          no masking decision happens here).
        - masking.apply_mask: given the SAME `regions` dict extract_regions
          just built, decides (per region_specs' own mask_mode/mask_file/
          angle/translation) which cells in each region actually count, and
          fills the rest with NaN. This mutates `regions` in place -- that's
          why this function does not need to capture apply_mask's return
          value separately before returning `regions`.

    PARAMETERS:
        frame_index (int): which frame number to load, e.g. 3 loads
            "{prefix}000003.npz".
        npz_directory (str): folder containing the .npz files -- works
            whether they came from raw CSV conversion or from a
            superposition canvas export; this function does not check.
        cell_size_x, cell_size_y (float): nm per grid cell, needed to convert
            each region's physical size/position into cell counts.
        region_specs (List[RegionSpec]): one spec per region to extract from
            this frame -- see region_extraction.RegionSpec for what a spec
            carries (center, size, mask_mode, mask_file, angle, translation).
        prefix (str): filename prefix, e.g. "B_demag" for "B_demag000003.npz".
        verbose (bool): if True, prints a one-line status showing which frame
            was loaded and how many regions were extracted from it. Off by
            default since process_frames_parallel calls this once per frame
            and the per-frame message would otherwise produce excessive
            console output on a
            large run -- see that function's own verbose handling.

    RETURNS:
        Tuple[int, Dict]: the same frame_index passed in (so the caller can
        match results back up after they come back out of order from a
        worker pool), and the regions dict -- {region_label: {"x", "y", "z",
        "masked_x", "masked_y", "masked_z", ...}}.

    LIMITATIONS:
        Loads and processes exactly one frame -- there is no batching inside
        this function itself; process_frames_parallel is what runs many of
        these at once.
    """
    from mumax3_csv_reader import load_npz_frame  # imported here (not at module top) so this file does not incur the import cost when only process_frames_parallel's multiprocessing path is used

    Bx_frame, By_frame, Bz_frame = load_npz_frame(  # load this one frame's raw x/y/z arrays from disk
        directory=npz_directory,  # folder holding the .npz files
        frame_index=frame_index,  # which frame number to load
        keys=('x', 'y', 'z'),  # which arrays to pull out of the .npz file
        filename_pattern=f"{prefix}{frame_index:06d}.npz"  # exact filename this frame is expected at
    )
    matrix_components = (np.array(Bx_frame), np.array(By_frame), np.array(Bz_frame))  # bundle the three loaded components together for extract_regions/apply_mask, which both expect this (x, y, z) tuple shape

    regions = extract_regions(  # crop every region in region_specs out of this one frame
        matrix_components=matrix_components,  # the just-loaded (x, y, z) frame data
        region_specs=region_specs,  # which regions to crop, and where
        units='grid',  # region_specs' center/size are in grid cells, not nm
        cell_size_x=cell_size_x,  # nm per cell, x direction
        cell_size_y=cell_size_y  # nm per cell, y direction
    )

    apply_mask(  # decide which cells in each cropped region actually count, per its own mask_mode
        regions,  # mutated in place: adds "masked_x"/"masked_y"/"masked_z" to every region
        region_specs=region_specs,  # carries each region's mask_mode/mask_file/angle/translation
        matrix_components=matrix_components,  # needed again here for mask_mode='file' + translation, which re-crops from the full frame
        units='grid',  # same units convention as extract_regions above
        cell_size_x=cell_size_x,  # nm per cell, x direction
        cell_size_y=cell_size_y,  # nm per cell, y direction
    )

    if verbose:
        print(f"[DEBUG] Loaded frame {frame_index} from '{npz_directory}' and extracted {len(region_specs)} region(s).")

    return frame_index, regions  # hand back both so a caller running many of these in parallel can match results to frame numbers


# --------------------------
# ---- Process every frame in a folder, in parallel, in memory-safe chunks ----
def process_frames_parallel(
    npz_directory: str,
    cell_size_x: float,
    cell_size_y: float,
    region_specs: List[RegionSpec],
    max_workers: Optional[int] = 2,
    prefix: str = "B_demag",
    reorganize: bool = True,
    verbose: bool = False,
) -> Union[List[Dict[str, Dict[str, Any]]], Dict[str, Dict[str, np.ndarray]]]:
    """
    GOAL:
        Run process_single_frame over EVERY frame found in `npz_directory`,
        using multiple worker processes at once, without loading every
        frame's full data into memory simultaneously (frames are processed
        in small chunks, and each chunk's raw per-frame results are
        discarded as soon as that chunk has been folded into the running
        totals). This is the function both parent scripts actually call --
        process_single_frame above serves as an internal building block.

    HOW IT TALKS TO OTHER FILES:
        - mumax3_pipeline_utilities.get_npz_end_index: figures out how many
          frames actually exist in `npz_directory` for this `prefix`, so
          this function knows how far to count.
        - process_single_frame (this file): each frame is handed to a
          separate worker process running this function.
        - region_extraction.reorganize_regions: when reorganize=True,
          reshapes the results from "one dict per frame" into "one dict per
          region label, with a stacked (frames, height, width) array for
          each component" -- the shape every downstream plotting/stats
          function in this pipeline expects.

    PARAMETERS:
        npz_directory (str): folder containing the .npz files to process.
        cell_size_x, cell_size_y (float): nm per grid cell.
        region_specs (List[RegionSpec]): one spec per region, applied
            IDENTICALLY to every frame (see this file's module-level
            LIMITATIONS section for the extent of that support).
        max_workers (int, optional): how many frames to process at once, in
            separate processes. None lets Python pick a default based on CPU
            count.
        prefix (str): filename prefix, e.g. "B_demag".
        reorganize (bool): True (default) returns data grouped by region
            label with frames stacked into one array per component -- what
            every plotting/stats function downstream expects. False returns
            the raw "one dict per frame index" shape instead, which is
            rarely useful outside debugging this function itself.
        verbose (bool): if True, prints the frame-count summary at the start
            plus per-chunk progress (which frames are being processed). Also
            forwarded to every process_single_frame call, so per-frame debug
            messages appear too if you want that much detail.

    RETURNS:
        - If reorganize=True: Dict[region_label, Dict[component, ndarray]],
          each array shaped (total_frames, height, width).
        - If reorganize=False: Dict[frame_index, Dict[region_label, ...]],
          one entry per frame, in the raw per-frame shape.

    LIMITATIONS:
        - See this file's module-level LIMITATIONS section for the
          fixed-region-per-frame and hardcoded-chunk-size notes -- both
          apply here directly, since this is where chunk_size actually lives.
        - No partial-failure handling: if any single frame raises inside its
          worker process, the whole call raises once that chunk's
          `future.result()` is reached, and any prior chunks' results are
          lost (this function does not return partial results on error).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed  # for running frames across multiple processes at once
    from mumax3_pipeline_utilities import get_npz_end_index  # figures out how many frames exist on disk
    import os  # for creating npz_directory if it does not already exist
    import gc  # for explicitly freeing memory between chunks on large runs

    os.makedirs(npz_directory, exist_ok=True)  # make sure the target directory exists (no-op if it already does)
    end_index = get_npz_end_index(npz_directory, prefix=prefix) + 1  # highest frame number found, +1 to turn it into a count
    if verbose:
        print(f"[INFO] Processing {end_index} frame(s) across {len(region_specs)} region(s).")
    final_results: Dict[int, Dict[str, Dict[str, Any]]] = {}  # accumulates results across every chunk as the loop below runs

    # ---- TRACKED CHANGE (perf): a ProcessPoolExecutor has a real, fixed startup
    # cost (spawning worker processes -- on Windows this re-imports the whole
    # module tree per worker, since Windows has no fork()) that is paid EVERY
    # time this function is called, regardless of how much work is actually
    # inside it. That cost is wasted whenever there is only one frame to
    # process -- for example, propagation_engine.py's per-step canvas exports
    # always write exactly one frame per canvas_dir (see compose_and_export_
    # canvas), so every propagation step was previously paying full pool-
    # startup overhead here to run a single process_single_frame call that
    # gains nothing from being out-of-process. Below max_pool_worthwhile_
    # frames, skip the pool and call process_single_frame directly, in this
    # process. Multi-frame callers (the normal spatial-analysis/superposition
    # case, many .npz frames in one directory) are unaffected -- they still
    # take the ProcessPoolExecutor path below. ----
    max_pool_worthwhile_frames = 1  # a pool is not worth spinning up to process this many frames or fewer; tune only if profiling shows the crossover point is different on your machine
    if end_index <= max_pool_worthwhile_frames:
        if verbose:
            print(f"[DEBUG] process_frames_parallel: only {end_index} frame(s) -- skipping ProcessPoolExecutor, running in-process.")
        frame_dict: Dict[int, Dict[str, Dict[str, Any]]] = {}
        for i in range(end_index):
            frame_index, regions = process_single_frame(i, npz_directory, cell_size_x, cell_size_y, region_specs, prefix, verbose)
            frame_dict[frame_index] = regions
        if reorganize:
            final_results = reorganize_regions(list(frame_dict.items()))  # already fully stacked per label/component -- NOT the list-per-chunk shape the multi-frame path below builds, so return immediately rather than falling into that path's own list-of-chunks concatenation step
        else:
            final_results = frame_dict
        if verbose:
            print(f"[DEBUG] Finished processing all {end_index} frame(s).")
        return final_results  # either {region_label: {component: full_array}} or {frame_index: {region_label: ...}}, depending on `reorganize`
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:  # worker pool that stays open for the whole run
            chunk_size = 5  # how many frames to hold in memory at once -- see module LIMITATIONS for why this is not currently exposed as a parameter
            for chunk_start in range(0, end_index, chunk_size):  # walk through all frames, chunk_size at a time
                chunk_end = min(chunk_start + chunk_size, end_index)  # last frame index in this chunk (clipped to the true end)
                if verbose:
                    print(f"[DEBUG] Processing frames {chunk_start} to {chunk_end - 1}...")
                results: Dict[int, Dict[str, Dict[str, Any]]] = {}  # holds this chunk's results only, discarded once folded into final_results
                futures = [
                    executor.submit(  # hand one frame's worth of work to the process pool
                        process_single_frame,  # the function each worker process runs
                        i,  # which frame this worker should load
                        npz_directory,  # where to find it
                        cell_size_x,  # nm per cell, x direction
                        cell_size_y,  # nm per cell, y direction
                        region_specs,  # which regions to extract from it
                        prefix,  # filename prefix
                        verbose,  # forwarded so per-frame debug prints work even inside worker processes
                    )
                    for i in range(chunk_start, chunk_end)  # one submission per frame in this chunk
                ]
                for future in as_completed(futures):  # collect results as each worker finishes (not necessarily in frame order)
                    frame_index, regions = future.result()  # unpack this worker's (frame_index, regions) return value
                    results[frame_index] = regions  # store it under its own frame index so order can be restored below
                frame_dict = {i: results[i] for i in sorted(results.keys())}  # put this chunk's results back into frame order
                if reorganize:
                    # Reorganize this chunk (label -> component -> ndarray(chunk_frames, h, w)),
                    # then append each chunk's arrays into final_results so every chunk's arrays
                    # can be concatenated along the frame axis once the whole run is done.
                    chunk_regions = reorganize_regions(list(frame_dict.items()))  # regroup this chunk from "per frame" to "per region label"
                    for label, components in chunk_regions.items():  # walk every region label produced by this chunk
                        if label not in final_results:  # first time seeing this label: start a new list per component
                            final_results[label] = {comp: [arr] for comp, arr in components.items()}
                        else:  # label already seen in an earlier chunk: append this chunk's array onto the running list
                            for comp, arr in components.items():
                                final_results[label][comp].append(arr)
                    del chunk_regions  # free this chunk's intermediate reorganized data now that it's been folded in
                else:
                    for frame_idx, frame_regions in frame_dict.items():  # reorganize=False: keep the raw per-frame shape as-is
                        final_results[frame_idx] = frame_regions
                del results  # free this chunk's raw per-frame results now that they're no longer needed
                gc.collect()  # explicitly reclaim memory before starting the next chunk, since large frame counts can otherwise build up unfreed memory across chunks

    if reorganize:
        for label, components in final_results.items():  # final pass: turn each region's list-of-per-chunk-arrays into one concatenated array
            for comp, arr_list in components.items():
                final_results[label][comp] = (
                    np.concatenate(arr_list, axis=0) if len(arr_list) > 1 else arr_list[0]  # skip concatenation entirely if there was only one chunk (nothing to join)
                )
    if verbose:
        print(f"[DEBUG] Finished processing all {end_index} frame(s).")
    return final_results  # either {region_label: {component: full_array}} or {frame_index: {region_label: ...}}, depending on `reorganize`
