Files:
	- mumax3_parent_spatial_analysis.py
		- add_script_folders
		- SpatialAnalysisConfig
		- run_pipeline
	- mumax3_parent_superposition.py
		- add_script_folders
		- SuperpositionConfig
		- run_pipeline
	- demo_spatial.py
		- main
	- demo_superposition.py
		- main


Description of each File:

- File Name: mumax3_parent_spatial_analysis.py
	- Overall Description:
		Parent script for spatial analysis of a directory of mumax3 CSV output
		files (m_full, B_demag, and/or B_exch), at any number of region centers,
		with a single mask mode (none / ellipse / file) applied to all of them.
		The same pipeline call covers every scale of this task without any
		change in shape: 1 region across 2 files with no mask; 10 regions
		across 800 files with an ellipse mask; and so on.

		Structured as one config dataclass (SpatialAnalysisConfig) holding
		every user-editable setting, and one function (run_pipeline) that
		performs the actual work end to end: NPZ conversion, region/mask
		extraction, spatial plotting, averaged quiver plots, and weighted
		circular statistics -- each step independently toggleable.

		This is also the analysis engine mumax3_parent_superposition.py reuses
		for its own per-device and canvas-level analysis passes (see that
		file's own description below) -- read this file's own module-level
		documentation for the full rationale of that reuse.
	- Functions:
		- add_script_folders(script_paths, levels_up=2, verbose=False)
			- Function Description:
				Adds each path in `script_paths`, and that path's parent
				directories up to `levels_up` levels, to sys.path -- so this
				codebase's own modules (src/_mumax3, src/_superposition) can be
				imported by plain module name regardless of where this script
				is actually run from. A path whose drive does not exist
				(Windows) is skipped, not raised, so this stays safe to leave
				pointed at a specific development machine's paths.
			- Parameter Identification and Description:
				- script_paths (List[str]): starting paths to add.
				- levels_up (int): how many parent directories above each path
				  to also add.
				- verbose (bool): print each path added/skipped.
		- run_pipeline(config)
			- Function Description:
				Runs NPZ conversion, unified region/mask extraction, spatial
				plotting, averaged quiver plots, and (optionally) weighted
				circular statistics, for every field in config.fields, using
				the same region_specs for all of them.
			- Parameter Identification and Description:
				- config (SpatialAnalysisConfig): every setting for this run.
			- Returns:
				Dict[field -> region data] if config.compute_stats is False;
				(regions, stats_results) if True.
	- Important Parameters (SpatialAnalysisConfig -- see the class's own
	  field-by-field comments in the source for the complete list; grouped
	  here by pipeline step):
		- Parameter Name: data_directory / output_dir / npz_directory
			- Description: input CSV directory, output root, and npz cache
			  location (the latter two default under data_directory).
			- Usage: output_dir and npz_directory can be pointed elsewhere to
			  keep a shared npz cache across multiple analysis runs.
		- Parameter Name: centers / sizes / region_labels
			- Description: one region per (center, size) pair, in grid cells;
			  region_labels optionally overrides the default "center_cx_cy"
			  naming.
			- Usage: any number of regions in one call -- 1 region and 100
			  regions cost the same code path.
		- Parameter Name: mask_mode / mask_file / mask_angles / mask_translation
			- Description: one shared mask configuration applied to every
			  region in the call.
			- Usage: mask_angles is a per-region list (for example, each
			  device's own geometry angle); a single shared angle is just
			  [angle] * len(centers).
		- Parameter Name: compute_stats / plot_spatial_maps / plot_spatial_quiver
			- Description: the three major output types, each independently
			  toggleable.
			- Usage: turn off plot_spatial_maps and compute_stats to only
			  regenerate quiver plots, for example.
		- Parameter Name: verbose
			- Description: the master switch -- forwarded to every step and
			  every function they call.
			- Usage: set True for a first run to see what is actually
			  happening; False for a quiet run.

- File Name: mumax3_parent_superposition.py
	- Overall Description:
		Parent script for superposing a single mumax3 seed CSV (one device's
		m_full and/or B_demag) across an ASI array geometry, exported as a
		composed canvas, then analyzed at each device's own region -- reusing
		mumax3_parent_spatial_analysis.run_pipeline for that analysis step.

		Supports regenerating the composed canvas multiple times (for
		example, re-randomizing the magnetization pattern) and exporting each
		regeneration as a successive frame, so "regenerate the superposition
		1000 times and visualize all the data spatially" is just
		num_regenerations=1000 -- the same downstream analysis code runs
		either way.

		The 11 worked geometry examples commented out at the bottom of this
		file (square, triangular, kagome, hexagonal, pinwheel, and their
		rotated/shaped variants) are a practical reference for
		geometry.DeviceArray's own parameter combinations.
	- Functions:
		- add_script_folders(script_paths, levels_up=2, verbose=False)
			- Function Description:
				Identical implementation to mumax3_parent_spatial_analysis.
				add_script_folders -- kept as its own copy in each parent
				script rather than a shared import, since each parent script
				needs to work standalone before any of this codebase's own
				modules are on sys.path yet.
			- Parameter Identification and Description: same as above.
		- run_pipeline(config)
			- Function Description:
				Runs seed extraction, ASI geometry construction, field
				composition (repeated config.num_regenerations times), and
				per-device region/mask analysis (STEP 6, via mumax3_parent_
				spatial_analysis.run_pipeline), plus optional canvas-level
				analysis of the composed canvas as a whole (STEP 7 -- full
				image, quadrant split, and/or per-device zoom, any
				combination in one pass).
			- Parameter Identification and Description:
				- config (SuperpositionConfig): every setting for this run.
			- Returns:
				The STEP 6 (per-device) result alone if
				config.plot_canvas_maps is False; {"devices": ...,
				"canvas": ...} if True.
	- Important Parameters (SuperpositionConfig -- grouped by pipeline step):
		- Parameter Name: sublattice_dims / device_length_nm / device_width_nm /
		  nn_x_nm / nn_y_nm / rotation_angles / device_rotations / translations
			- Description: passed straight to geometry.DeviceArray to build
			  the ASI array -- any number of sublattices, each independently
			  sized/spaced/rotated/translated.
			- Usage: see geometry.py's own module-level documentation for
			  what changing any of these does to an already-built array
			  (device indices/positions are not stable across a change).
		- Parameter Name: magnetization_mode / magnetization_base_angle /
		  num_regenerations / random_seed_start
			- Description: how each device's magnetization angle is set, and
			  how many times to rebuild-and-re-export it as a successive
			  frame.
			- Usage: magnetization_mode="random" with num_regenerations=1000
			  is the "regenerate many times" use case this script is built
			  around.
		- Parameter Name: mask_file / device_mask_size_nm
			- Description: the per-device mask template (a single-device
			  seed .npz), automatically rotated to each device's own
			  geometry angle and translated to its own position.
			- Usage: device_mask_size_nm must be set explicitly when
			  device_length_nm/device_width_nm are per-sublattice dicts
			  rather than one shared scalar.
		- Parameter Name: plot_canvas_maps / canvas_split / canvas_zoom_targets /
		  canvas_zoom_all_devices / canvas_include_full
			- Description: canvas-level analysis of the composed canvas as a
			  whole -- full image, quadrant split, and/or per-device zoom
			  regions, any combination in one pass.
			- Usage: canvas_zoom_all_devices=True auto-populates one zoom
			  region per device without hand-listing centers.
		- Parameter Name: verbose
			- Description: the master switch -- forwarded through this
			  script's own steps AND into every SpatialAnalysisConfig this
			  script builds internally (STEP 6/7), so one True/False here
			  controls progress printing across the entire superposition +
			  analysis run.

- File Name: demo_spatial.py
	- Overall Description:
		Self-contained demonstration of mumax3_parent_spatial_analysis.py on
		synthetic test data: multiple m_full/B_demag files, each a full small
		ASI array (not one device) with a different random magnetization
		configuration. Generates its own input data via
		generate_parallelization_test_files.py (see src/_demo/README.txt) if
		not already present, then analyzes one region covering the whole
		canvas plus one region per individual device, across every
		regenerated frame. Every SpatialAnalysisConfig field is set
		explicitly and commented, so the full menu of what is configurable
		is visible in one file.
	- Functions:
		- main(verbose=True)
			- Function Description:
				Generates (if needed) the synthetic multi-frame test data,
				builds a fully-populated SpatialAnalysisConfig, and calls
				run_pipeline.
			- Parameter Identification and Description:
				- verbose (bool): the single toggle for this entire demo run.

- File Name: demo_superposition.py
	- Overall Description:
		Self-contained demonstration of mumax3_parent_superposition.py's full
		feature set, on synthetic test data (via generate_demo_seed_data.py --
		see src/_demo/README.txt) instead of a real mumax3 simulation.
		Demonstrates per-device analysis with the geometry pulled automatically
		from the DeviceArray, per-device mask rotation/translation, averaged
		quiver plots, all three canvas-level plotting modes at once (full
		image, quadrant split, per-device zoom), and weighted circular
		statistics both per-device and per-canvas-region -- all in one run.
		Every SuperpositionConfig field is set explicitly and commented.
	- Functions:
		- main(verbose=True)
			- Function Description:
				Generates the synthetic seed device, builds a fully-populated
				SuperpositionConfig demonstrating every major feature, and
				calls run_pipeline.
			- Parameter Identification and Description:
				- verbose (bool): the single toggle for this entire demo run.


File Crossover:
	- mumax3_parent_superposition.py's run_pipeline calls mumax3_parent_
	  spatial_analysis.run_pipeline twice per run (once for STEP 6, per-device
	  analysis; once for STEP 7, canvas-level analysis, if enabled) -- this is
	  the single most important cross-file relationship in this package. Both
	  calls build their own SpatialAnalysisConfig internally and forward the
	  relevant SuperpositionConfig fields into it; neither parent script
	  duplicates the other's analysis logic.
	- Both parent scripts' run_pipeline functions live in src/_mumax3 and
	  src/_superposition for their actual implementation (region_extraction.py,
	  masking.py, frame_pipeline.py, mumax3_csv_plot.py, mumax3_csv_stats.py,
	  geometry.py, superposition_operations.py, utilities_superposition.py) --
	  see src/_mumax3/README.txt and src/_superposition/README.txt for what
	  each of those files provides. Neither parent script implements region
	  cropping, masking, plotting, or geometry generation itself.
	- Both demo scripts import from src/_demo (generate_demo_seed_data.py /
	  generate_parallelization_test_files.py) to produce their own synthetic
	  input data -- see src/_demo/README.txt. These are separated from the
	  main package specifically so the package (this README's four files plus
	  src/_mumax3 and src/_superposition) can be shipped and used on its own,
	  without needing the demo/test-data-generation machinery.
	- Both parent scripts' verbose field is the single master toggle for an
	  entire run: SpatialAnalysisConfig.verbose and SuperpositionConfig.
	  verbose each cascade into every function their own run_pipeline calls,
	  including (for the superposition parent script) the SpatialAnalysisConfig
	  instances it builds internally for STEP 6/7.
	- See API.txt at the root of this package for the full function-level
	  reference across every file in this package, and src/_demo/README.txt
	  for the four support scripts (generate_demo_seed_data.py,
	  generate_parallelization_test_files.py, plot_single_field.py,
	  magnetostatics.py) not covered in this README.


Parent Scripts:
	- Necessary Parameters in a Parent Script to Function:
		- mumax3_parent_spatial_analysis.SpatialAnalysisConfig requires:
		  data_directory, and at least one entry in both centers and sizes.
		  mask_file is additionally required if mask_mode='file'. Every other
		  field has a working default.
		- mumax3_parent_superposition.SuperpositionConfig requires:
		  data_directory, sublattice_dims (defaults to a small 6x7 single-
		  sublattice array if not overridden), device_length_nm/device_width_nm/
		  nn_x_nm/nn_y_nm (all have working defaults matching that sublattice),
		  and mask_file (no default -- always required, since there is no
		  sensible default mask template). Every other field has a working
		  default.
		- Building a custom parent script on top of either one: copy the file,
		  keep the config dataclass (or add fields to it), and either call
		  run_pipeline as-is or call the individual pieces it calls directly,
		  in a different order or combination -- see each file's own module-
		  level "BUILDING A CUSTOM PARENT SCRIPT ON TOP OF THIS ONE" section
		  for specifics, and API.txt for the full function-level reference.
