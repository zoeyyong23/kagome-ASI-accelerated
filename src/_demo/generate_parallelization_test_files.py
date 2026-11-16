r"""
File Name: generate_parallelization_test_files.py
Author: Connor Sullivan
Date: 2026.08.06
Revision: v2

============================================================
PURPOSE
============================================================
Produces a handful of standalone m_full/B_demag CSV files -- one per
"regeneration" (a re-randomized magnetization pattern across the same ASI
array, via SuperpositionConfig.num_regenerations with magnetization_mode=
"random") -- for testing parallelization (max_workers, stats_max_workers)
against real multi-frame data, and as ready-made input for demo_spatial.py,
which needs multiple m_full files representing a full ASI array with
different spin configurations.

Each output file is a full composed-canvas snapshot (the whole small ASI
array demo_superposition.py builds, not a single device) -- the same
mumax3-style stacked CSV format used everywhere else in this pipeline (see
generate_demo_seed_data.py's own documentation), so it drops straight into
mumax3_parent_spatial_analysis.py's normal CSV-conversion STEP 1 like any
other mumax3 output would.

This deliberately runs the superposition step only (geometry + composition
+ export), with every downstream plotting/stats step turned off -- this
file exists to produce the files fast, not to demonstrate the whole
pipeline again (see demo_superposition.py for that).
"""

import os  # path construction
import shutil  # clearing prior output before a fresh run
import sys  # sys.path manipulation, so this codebase's own modules import cleanly
import numpy as np  # loading composed-canvas .npz frames

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "_mumax3"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "_superposition"))

from generate_demo_seed_data import generate_demo_seed_data, _write_mumax3_style_csv  # produces this script's own seed device, and the mumax3-style CSV writer this script reuses for its own output


def main(num_regenerations: int = 6, verbose: bool = True):
    """
    GOAL:
        Generate (if not already present) a single synthetic seed device,
        run mumax3_parent_superposition.run_pipeline with magnetization_
        mode="random" for `num_regenerations` regenerations with every
        downstream plotting/stats step disabled, then convert each
        resulting composed-canvas .npz frame into a standalone mumax3-style
        CSV file.

    HOW IT TALKS TO OTHER FILES:
        Calls generate_demo_seed_data.generate_demo_seed_data to produce
        this script's seed device if not already present. Calls
        mumax3_parent_superposition.run_pipeline to perform the actual
        composition. Calls generate_demo_seed_data._write_mumax3_style_csv
        to write each resulting CSV.

    PARAMETERS:
        num_regenerations (int): how many randomized regenerations (and
            therefore output files) to produce.
        verbose (bool): the single toggle for this entire run -- forwarded
            into SuperpositionConfig.verbose, which itself cascades into
            every step of mumax3_parent_superposition.run_pipeline and
            every function that pipeline calls. True by default, so a
            first run shows what is actually happening; set False for a
            quiet run.

    RETURNS:
        List[str]: paths to every CSV file written (num_regenerations x
        number of fields in config.fields).
    """
    demo_root = os.path.join(os.path.dirname(__file__), "demo_output")
    data_directory = os.path.join(demo_root, "seed_data")
    out_dir = os.path.join(demo_root, "parallelization_test_files")
    shutil.rmtree(out_dir, ignore_errors=True)  # start this script's own output directory empty on every run
    os.makedirs(out_dir, exist_ok=True)

    cell_size_nm = 1.0
    sim_size_nm = 300.0
    device_length_nm, device_width_nm = 70.0, 56.0

    # Reuse the same seed device generate_demo_seed_data.py already knows how to
    # build (elliptical device + real Magnetostatics-computed B_demag) -- only
    # regenerated if not already present from a prior demo_superposition.py run.
    seed_paths = {
        "m_full": os.path.join(data_directory, "m_full000000.csv"),
        "B_demag": os.path.join(data_directory, "B_demag000000.csv"),
    }
    if not all(os.path.exists(p) for p in seed_paths.values()):  # at least one seed file is missing: (re)generate it
        generate_demo_seed_data(
            data_directory=data_directory,
            sim_size_cells=(int(sim_size_nm / cell_size_nm), int(sim_size_nm / cell_size_nm)),
            device_size_cells=(int(device_length_nm / cell_size_nm), int(device_width_nm / cell_size_nm)),
            cell_size_x=cell_size_nm, cell_size_y=cell_size_nm,
            device_angle_deg=0.0,
            frame_index=0,
            verbose=verbose,
        )

    from mumax3_parent_superposition import SuperpositionConfig, run_pipeline

    npz_directory = os.path.join(data_directory, "temp", "npz")
    mask_file = os.path.join(npz_directory, "m_full_npz_cache", "m_full000000.npz")

    config = SuperpositionConfig(
        data_directory=data_directory,
        output_dir=os.path.join(demo_root, "parallelization_run"),  # scratch; not the deliverable -- see the CSV conversion step below for that
        npz_directory=npz_directory,
        cell_size_x=cell_size_nm, cell_size_y=cell_size_nm,
        sim_size_x=sim_size_nm, sim_size_y=sim_size_nm,
        fields=("m_full", "B_demag"),
        device_length_nm=device_length_nm, device_width_nm=device_width_nm,
        nn_x_nm=140.0, nn_y_nm=140.0,
        sublattice_dims={0: (1, 2), 1: (2, 1)},  # the same small 4-device array demo_superposition.py builds
        device_rotations={0: 45, 1: 135},
        translations={1: [-70, 70]},
        magnetization_mode="random",  # a fresh random pattern every regeneration, same array/geometry each time
        magnetization_base_angle={0: 45, 1: 135},  # required for mode="random" --
        # each device is randomly flipped between base_angle and base_angle+180
        num_regenerations=num_regenerations,
        random_seed_start=0,
        mask_file=mask_file,
        device_mask_size_nm=(device_length_nm, device_width_nm),
        use_cached_npz=True,
        max_workers=2,
        # ---- Everything downstream of "compose + export" turned off -- this script
        # only wants the composed-canvas frames themselves, produced fast. ----
        plot_spatial_maps=False,
        plot_spatial_quiver=False,
        plot_canvas_maps=False,
        compute_stats=False,
        verbose=verbose,  # the master toggle for this whole run -- see this function's own PARAMETERS documentation
    )
    run_pipeline(config)

    # ---- Convert each regeneration's composed-canvas npz frame to a standalone CSV ----
    canvas_dir = os.path.join(npz_directory, "composed_canvas")
    written = []  # accumulated across every dataset/regeneration below
    for dataset in config.fields:  # one output file per (dataset, regeneration) combination
        cache_dir = os.path.join(canvas_dir, f"{dataset}_npz_cache")
        for i in range(num_regenerations):
            npz_path = os.path.join(cache_dir, f"{dataset}{i:06d}.npz")
            d = np.load(npz_path)
            csv_path = os.path.join(out_dir, f"{dataset}{i:06d}.csv")
            _write_mumax3_style_csv(csv_path, d["x"], d["y"], d["z"], verbose=verbose)
            written.append(csv_path)

    print(f"\nWrote {len(written)} file(s) ({num_regenerations} regeneration(s) x {len(config.fields)} dataset(s)) to:")
    print(f"  {out_dir}")
    return written


if __name__ == "__main__":
    main(verbose=True)
