Files:
	- geometry.py
		- compute_magnetization_angles
		- DeviceArray (class)
			- __init__
			- _normalize_sublattice_param
			- _normalize_termination
			- _rotated_basis
			- _device_half_extents
			- _compute_boundaries
			- _length_nm_for / _width_nm_for
			- set_magnetization_pattern
			- set_device_magnetization
			- update_parameters
			- _per_device_lengths_widths
			- update_array
			- _compute_target_boundaries
			- _apply_boundary_fill
			- _refresh_data
			- update_from_array
			- update_from_extended_array
			- set_device_property
			- remove_devices
			- check_for_overlaps
			- get_device_centers
			- get_neighbor_vectors
			- get_device_array
			- print_summary
			- plot_layout
	- superposition_operations.py
		- rotate_field_spatial
		- rotate_field_vectors
		- compose_fields
		- init_pool
		- compose_fields_task
		- run_superposition_multiprocessing
	- utilities_superposition.py
		- save_device_array_plot
		- save_magnetization_plot
		- save_device_array_magnetization
		- save_device_array_table
		- save_device_array
		- export_canvas_as_npz_frames


Description of each File:

- File Name: geometry.py
	- Overall Description:
		Defines DeviceArray: the geometry generator for a periodic
		artificial spin ice (ASI) device array, and the class the rest of
		this pipeline's superposition machinery is built around. Given a
		per-sublattice device size, spacing, rotation, and translation,
		DeviceArray builds every device's (x, y) center, geometry angle, and
		magnetization angle, re-buildable on demand whenever any of those
		parameters change.

		THIS IS THE MOST USER-FACING CLASS IN THIS PACKAGE. DeviceArray can
		build ANY periodic array: any number of sublattices, each with its
		own device size, lattice spacing, lattice rotation, device (island)
		rotation, and translation offset, combined into one array and
		automatically centered on (0, 0). This covers square ASI, pinwheel
		ASI (square ASI with the lattice rotated 45 degrees), kagome ASI
		(three directionally-unique sublattices), and any other periodic
		geometry expressible as "N sublattices, each a rotated/translated
		rectangular grid."

		The tradeoff for that generality: the array is defined ENTIRELY by
		its per-sublattice setup parameters, regenerated from scratch every
		time ANY of them changes -- there is no notion of "nudge this one
		sublattice's rotation angle slightly and keep everything else
		exactly where it was." Three consequences of this are worth
		understanding before building an array:
			1. Changing a sublattice's geometry angle, device rotation, or
			   translation moves that sublattice's devices RELATIVE TO THE
			   ARRAY'S ORIGIN, not in place. Every device position is built
			   from that sublattice's own local lattice mesh, rotated about
			   that mesh's OWN center, translated, and only then does the
			   WHOLE array (every sublattice combined) get re-centered on
			   (0, 0) by its new overall center of mass. Changing one
			   sublattice's rotation or translation shifts where that
			   sublattice's devices land relative to every other
			   sublattice, which shifts the array's overall center of mass,
			   which shifts EVERY device's final position -- including
			   devices in sublattices that were not touched -- once the
			   whole array is re-centered.
			2. Changing the array SIZE (sublattice_dims, or a termination
			   setting that changes how many devices a sublattice actually
			   gets built with) changes the same center-of-mass-recentering
			   chain, for the same reason: a different device count changes
			   each sublattice's own local mesh center, which changes that
			   sublattice's placement, which changes the overall center of
			   mass every device gets re-centered against. Growing or
			   shrinking the array is NOT equivalent to adding or removing
			   devices at the edges of an otherwise-unchanged layout -- the
			   entire array shifts.
			3. Because of (1) and (2), device indices and absolute (x, y)
			   positions are only valid for ONE specific, fully-built
			   configuration. Do not cache a device's index or position and
			   assume it still refers to the same physical device after any
			   call that regenerates the array (update_parameters();
			   set_magnetization_pattern() is the one exception -- it does
			   NOT call update_array(), so it does not trigger this).
			   Re-read device_centers/device_data/get_device_array() fresh
			   after any regenerating change.
	- Functions:
		- compute_magnetization_angles(device_data, device_centers,
		  translations, nn_x, nn_y, mode="identical", base_angle=0,
		  sublattice=None, magnetic_sublattice=None, parity_term_fix=1,
		  random_seed=None, verbose=False)
			- Function Description: module-level, standalone version of the
			  same magnetization-pattern computation DeviceArray.
			  set_magnetization_pattern uses internally -- usable without a
			  DeviceArray instance. Modes: 'identical', 'opposite',
			  'row_opposite', 'col_opposite', 'random'.
		- DeviceArray.__init__(device_length, device_width, nn_x, nn_y,
		  sublattice_dims, rotation_angles=None, translations=None,
		  device_rotations=None, sublattice_termination=None,
		  sublattice_termination_mode=None, verbose=False)
			- Function Description: builds a new array from per-sublattice
			  setup parameters, via update_array().
			- Parameter Identification and Description: device_length/
			  device_width/nn_x/nn_y (nm, scalar or per-sublattice dict, no
			  default -- required); sublattice_dims (Dict[id, (num_x,
			  num_y)], required); rotation_angles (per-sublattice LATTICE
			  rotation); device_rotations (per-sublattice DEVICE/island
			  orientation, independent of rotation_angles); translations
			  (per-sublattice offset, nm); sublattice_termination (per-edge
			  extension spec); sublattice_termination_mode ('fixed_row_col'
			  or 'rotated_boundary_fill').
		- set_magnetization_pattern(mode="identical", base_angle=0,
		  sublattice=None, magnetic_sublattice=None, parity_term_fix=1,
		  random_seed=None, verbose=False)
			- Function Description: computes and applies a magnetization
			  pattern, keeping device_data/device_array/device_m_angles in
			  sync. Does NOT call update_array() -- positions/indices are
			  unaffected.
		- set_device_magnetization(index, angle, verbose=False)
			- Function Description: sets one device's magnetization angle by
			  index. Updates device_data only (not device_array's own
			  m_angle column) -- call _refresh_data() afterward if
			  device_array/device_m_angles need to reflect it too.
		- update_parameters(verbose=False, **kwargs)
			- Function Description: updates any combination of setup
			  parameters and regenerates the entire array via
			  update_array(). sublattice_dims (if present) is applied first,
			  regardless of kwarg order, before anything else in the same
			  call is validated against it.
		- update_array(verbose=False)
			- Function Description: builds (or rebuilds) every device's
			  position/angle from current setup parameters -- the single
			  method every device position in this class ultimately comes
			  from. Called by __init__ and update_parameters.
		- update_from_array(array, device_data=None, verbose=False)
			- Function Description: replaces the entire device_array/
			  device_data from a caller-supplied array, validating no
			  overlaps before committing (reverts if any are found).
		- update_from_extended_array(extended_array, verbose=False)
			- Function Description: same as update_from_array, but the
			  input already includes each device's own length/width as two
			  extra columns.
		- set_device_property(index, key, value, verbose=False)
			- Function Description: updates one property ('x','y','angle',
			  'sublattice','length','width') of one device by index,
			  validating no overlaps before committing.
		- remove_devices(indices, verbose=False)
			- Function Description: removes one or more devices by index and
			  re-indexes the remaining devices from 0. The recommended way
			  to hand-carve an irregular footprint (for example, trimming a
			  kagome/hexagonal array) rather than extending "rotated_
			  boundary_fill" to arbitrary polygon shapes.
		- check_for_overlaps(verbose=False)
			- Function Description: checks every device pair for physical
			  (elliptical footprint) overlap. Requires the shapely package.
		- get_device_centers()
			- Function Description: returns device_centers directly (not a
			  copy).
		- get_neighbor_vectors(reference_index=None, n_shells=2,
		  verbose=False)
			- Function Description: center-to-center displacement vectors
			  (nm) from a reference device to its nearest/next-nearest
			  neighbors, grouped into shells by distance. Vectors come out
			  as exact whole nanometers regardless of rotation_angles.
		- get_device_array(include_size=False)
			- Function Description: returns the device array as a plain
			  numpy copy -- 5-column [x,y,sublattice,angle,m_angle] core
			  layout, or 7-column with length/width appended if
			  include_size=True. Called throughout mumax3_parent_
			  superposition.py.
		- print_summary(count=5)
			- Function Description: prints a human-readable summary and
			  returns the same information as a structured dict.
		- plot_layout()
			- Function Description: interactive plot of the array layout
			  (blocks on plt.show()). See utilities_superposition.
			  save_device_array_plot for a save-to-file equivalent.
	- Important Parameters
		- Parameter Name: sublattice_dims / rotation_angles /
		  device_rotations / translations
			- Description: the per-sublattice geometry parameters every
			  device position is ultimately built from.
			- Usage: see this file's own extensive "WHAT THIS CLASS CAN AND
			  CANNOT DO" documentation above (and in the source's module
			  docstring) before changing any of these on an already-built
			  array.

- File Name: superposition_operations.py
	- Overall Description:
		The field-rotation and canvas-composition logic behind ASI
		superposition: taking one seed device's field, rotating it (both
		spatially and vectorially) to match each device's own geometry and
		magnetization angle, and adding every device's rotated field into
		the correct place on a shared canvas.
	- Functions:
		- rotate_field_spatial(mx, my, angle_deg, verbose=False)
			- Function Description: rotates a field's SPATIAL footprint
			  (where the device shape sits) to match a device's geometry
			  angle. Distinct from rotate_field_vectors -- shape and
			  magnetization can rotate independently.
		- rotate_field_vectors(mx, my, angle_deg, verbose=False)
			- Function Description: rotates a field's VECTOR components (the
			  direction each cell's field points) to match a device's
			  magnetization angle -- a standard in-plane 2D rotation.
		- compose_fields(device_centers, geometry_angles,
		  magnetization_angles, base_field, field_name, cell_size_x,
		  cell_size_y, margin_nm=0, verbose=False)
			- Function Description: builds one shared canvas containing
			  every device's contribution, by spatially and vectorially
			  rotating the seed field per device and adding (superposing)
			  each device's rotated field into its correct position. A
			  device with a NaN magnetization_angle is skipped (can be
			  intentional, for example to exclude one device from a
			  stray-field sweep).
		- init_pool(device_centers, geometry_angles, magnetization_angles,
		  verbose=False)
			- Function Description: multiprocessing worker initializer --
			  stores the device geometry as module-level globals inside each
			  worker process, once, instead of re-sending it with every
			  task.
		- compose_fields_task(task, verbose=False)
			- Function Description: multiprocessing-safe worker -- unpacks
			  one task tuple and calls compose_fields, reading the device
			  geometry from init_pool's globals.
		- run_superposition_multiprocessing(data_struct_all, device_centers,
		  geometry_angles, magnetization_angles, cell_size_x, cell_size_y,
		  verbose=False)
			- Function Description: composes every (region, base field,
			  variant) combination across a process pool. Called by
			  mumax3_parent_superposition.py's run_pipeline.
	- Important Parameters
		- Parameter Name: field_name (compose_fields)
			- Description: used only for cache keys and status messages --
			  not a physical parameter.
			- Usage: compose_fields' spatial/vector rotation caches are
			  local to a single call and do not persist across separate
			  calls (each worker process gets its own fresh caches).

- File Name: utilities_superposition.py
	- Overall Description:
		Saves DeviceArray layout plots, magnetization plots, and device data
		tables, and exports a composed canvas to the .npz frame format the
		rest of this pipeline reads.
	- Functions:
		- save_device_array_plot(device_array_obj, save_path_base,
		  exts=None, verbose=False)
			- Function Description: saves a plot of the array's geometry
			  (device positions/orientations as ellipses), plus boundary
			  lines.
		- save_magnetization_plot(device_array_obj, save_path_base,
		  exts=None, arrow_scale=0.8, verbose=False)
			- Function Description: saves a quiver plot of the array's
			  magnetization directions, one arrow per device, colored by
			  angle.
		- save_device_array_magnetization(device_array_obj, save_path_base,
		  exts=None, verbose=False)
			- Function Description: thin wrapper around
			  save_magnetization_plot.
		- save_device_array_table(device_array_obj, save_path_base,
		  exts=None, verbose=False)
			- Function Description: saves per-device data as a table, in any
			  combination of csv/xlsx/txt/pdf.
		- save_device_array(device_array_obj, base_save_path,
		  save_table_exts=None, save_plot_exts=None, verbose=False)
			- Function Description: the combined entry point -- saves a
			  table and/or plots in one call; when both a table format and a
			  plot format include "pdf", combines a table page and both
			  plot pages into one multi-page PDF instead of three separate
			  files. Called by mumax3_parent_superposition.py's run_pipeline.
		- export_canvas_as_npz_frames(canvas_struct, output_dir,
		  output_filename=None, frame_index=None, verbose=False)
			- Function Description: exports a composed canvas (as returned
			  by superposition_operations.run_superposition_multiprocessing)
			  to the "{prefix}{frame_index:06d}.npz" format the rest of this
			  pipeline reads -- the bridge between "a composed canvas in
			  memory" and "a folder of frames analyzable like any other
			  mumax3 output." A composed canvas has no z component by
			  construction (superposition here is entirely in-plane); a
			  zero-filled z array is written alongside x/y purely because
			  downstream code expects all three components present.
	- Important Parameters
		- Parameter Name: canvas_struct (export_canvas_as_npz_frames)
			- Description: nested {region_key: {base_key: {variant_key:
			  {"x":..., "y":...}}}}, matching run_superposition_
			  multiprocessing's own return shape.
			- Usage: the output frame index for each variant defaults to
			  variant_key itself if it is a plain digit string (the usual
			  regeneration-indexed case); a canvas_struct with multiple
			  non-numeric variant keys and no explicit frame_index would
			  collide on frame index 0.


File Crossover:
	- geometry.DeviceArray is the geometry source both superposition_
	  operations.py and utilities_superposition.py consume:
	  superposition_operations.run_superposition_multiprocessing takes
	  device_centers/geometry_angles/magnetization_angles read off a
	  DeviceArray (via .get_device_centers()/.device_angles/.
	  device_m_angles), and utilities_superposition.save_device_array takes
	  a DeviceArray instance directly.
	- mumax3_parent_superposition.py's run_pipeline is the only caller of
	  both superposition_operations.run_superposition_multiprocessing and
	  utilities_superposition.export_canvas_as_npz_frames/save_device_array
	  -- neither of those two files calls the other directly.
	- The .npz frame format export_canvas_as_npz_frames writes matches
	  mumax3_csv_reader.save_to_npz's own layout convention exactly, so
	  downstream code (frame_pipeline.process_frames_parallel, mumax3_
	  pipeline_utilities.get_npz_end_index, in src/_mumax3) reads a composed
	  canvas identically to real mumax3 CSV output, with no special-casing.
	- Column convention: geometry.DeviceArray's device_array rows are
	  [x, y, sublattice, angle, m_angle] (column 2 = sublattice, column 3 =
	  angle) throughout this file -- this convention was the source of a
	  real, now-fixed bug (set_device_property had columns 2/3 swapped for
	  the "angle"/"sublattice" keys specifically) and is worth keeping in
	  mind if extending DeviceArray or writing code that reads its raw
	  array directly rather than through device_data/get_device_array().


Parent Scripts:
	- Necessary Parameters in a Parent Script to Function:
		A parent script built directly on this directory (rather than
		through mumax3_parent_superposition.py) needs: a geometry.
		DeviceArray instance (device_length/device_width/nn_x/nn_y/
		sublattice_dims at minimum), a single-device seed field dict
		({"x": ..., "y": ...}, for example from mumax3_csv_reader.
		load_npz_frame), and the grid's cell_size_x/cell_size_y. From
		there, superposition_operations.run_superposition_multiprocessing
		composes the canvas, and utilities_superposition.export_canvas_
		as_npz_frames writes it out in a form the src/_mumax3 pipeline can
		analyze like any other mumax3 output -- see API.txt at the root of
		this package for the full function-level reference.
