r"""
File Name: magnetostatics.py
Author: Connor Sullivan
Date: 2025.08.09
Revision: 0.0.6

============================================================
PURPOSE
============================================================
Analytic point-dipole model for the magnetostatic (stray) field generated
by an array of elliptical magnetic elements. Each device is discretized
into a grid of sub-dipoles inside its own elliptical footprint; every
sub-dipole's field contribution is summed analytically via the standard
dipole field formula. This is a standalone physics utility with no
dependency on the rest of this codebase's pipeline (region_specs,
DeviceArray, RegionSpec, and so on) -- it operates purely on plain center/
angle arrays and returns plain field arrays.

Contains the Magnetostatics class:
    - __init__ / generate_device_grid: build every device's sub-dipole
      grid (positions and moments) once, at construction time.
    - compute_field_at_points: the core field calculation -- sum every
      sub-dipole's contribution at a set of query points.
    - compute_field_over_device_volume: average the field over a device-
      shaped probe volume (for example, to evaluate the field acting on
      one device due to every other device).
    - compute_local_fields_for_all_devices: convenience wrapper that runs
      compute_field_over_device_volume once per device in a
      geometry.DeviceArray-shaped array, excluding each device's own
      contribution to its own local field.

generate_demo_seed_data.py uses this class directly (constructing a
Magnetostatics instance for one synthetic seed device and calling
compute_field_at_points across a grid) to produce a physically realistic
B_demag test field without a real mumax3 simulation -- see that file's own
documentation.

============================================================
LIMITATIONS
============================================================
- This is a point-dipole-sum approximation, not a self-consistent
  micromagnetic solve. The field it produces diverges near its own source
  points (the sub-dipoles), so evaluating the field very close to (or
  inside) a device's own footprint produces artificially large, spiky
  values -- physically expected of this class of model, not a bug. The
  Ms_taper edge-tapering in generate_device_grid reduces some of this
  right at a device's own boundary, but does not eliminate the underlying
  near-field divergence at any individual sub-dipole's exact location.
- compute_local_fields_for_all_devices expects `device_array_obj` to
  expose a get_device_array() method returning rows in this codebase's own
  [x, y, sublattice, angle, m_angle] column convention (the same
  convention geometry.DeviceArray uses -- see that class's own module-
  level documentation) -- this is the one place in this file that assumes
  anything about the caller's data shape; every other method in this class
  takes plain arrays with no assumed column order.
- compute_local_fields_for_all_devices was affected by three real bugs
  prior to this revision, all fixed here (see the parent revision if any
  of these fixes need to be reviewed against the original behavior):
    1. It read the angle and sublattice columns from a device_array row in
       the opposite order from this codebase's own convention (angle from
       column 2, sublattice from column 3, instead of the correct column
       2 = sublattice, column 3 = angle) -- the same class of bug found
       and fixed in geometry.py's set_device_property.
    2. It called generate_points_inside_ellipse, a function that was only
       ever defined inside this file's own __main__ block -- calling this
       function from another file raised NameError. That function is now
       defined at module level (see generate_points_inside_ellipse below),
       available regardless of how this file is imported or run.
    3. It unpacked `Bx, By, Bz = B_local`, treating the six-key dict
       compute_field_over_device_volume returns as if it were a
       3-element (Bx, By, Bz) tuple -- this raised ValueError on any call.
       Fixed by computing the genuine averaged field vector from
       B_local["B_vectors"] instead.
  This function is not currently called by any other file in this
  pipeline, so none of these three bugs have been exercised through this
  codebase's own usage; treat this function as freshly fixed but not yet
  exercised against real data.
"""

import numpy as np  # array operations used throughout this file
import matplotlib.pyplot as plt  # used by the worked example at the bottom of this file
import matplotlib as mpl  # used by the worked example at the bottom of this file
import pandas as pd  # used by compute_local_fields_for_all_devices' returned DataFrame


class Magnetostatics:
    # --------------------------
    # ---- Build every device's sub-dipole grid at construction time ----
    def __init__(self, centers, angles_deg, Ms=800e3, device_length=60e-9, device_width=60e-9,
                 thickness=2.5e-9, resolution=5, verbose=False):
        """
        GOAL:
            Construct a Magnetostatics instance: store the array's device
            geometry, and precompute every device's sub-dipole grid
            (positions and moments) once, up front, so repeated field
            queries (compute_field_at_points) do not need to regenerate
            that grid on every call.

        HOW IT TALKS TO OTHER FILES:
            Called by generate_demo_seed_data.py to build a single synthetic
            seed device's field source. Calls generate_device_grid (this
            class), once per device, during construction.

        PARAMETERS:
            centers (array-like): Nx2 array of device centers, in nm.
            angles_deg (array-like): N-length array of device orientations,
                degrees -- 0 = +x, 90 = +y, matching mumax3's own
                convention.
            Ms (float): saturation magnetization, A/m, shared by every
                device.
            device_length, device_width (float): device size, in meters
                (note: unlike `centers`, which is in nm -- see LIMITATIONS),
                shared by every device.
            thickness (float): device thickness, meters, shared by every
                device.
            resolution (int): sub-dipole grid resolution per axis -- higher
                is more accurate and slower. Every device shares this same
                resolution.
            verbose (bool): prints the number of devices and total
                sub-dipole count when True.

        RETURNS:
            None. Populates self.centers, self.angles_rad, self.Ms,
            self.device_length, self.device_width, self.resolution,
            self.thickness, self.mu0, self.subvolume, self.submoment, and
            self.device_grids.

        LIMITATIONS:
            `centers` is expected in nm and converted to meters internally
            (`* 1e-9`); `device_length`/`device_width`/`thickness` are
            expected ALREADY in meters, not nm -- this asymmetry matches
            how this class has always been called (see the worked example
            at the bottom of this file) but is easy to get backwards when
            calling this class fresh; double-check units at each call site.
        """
        self.centers = np.array(centers) * 1e-9  # nm -> m
        self.angles_rad = np.radians(angles_deg)  # degrees -> radians
        self.Ms = Ms  # saturation magnetization, A/m
        self.device_length = device_length  # meters (see this function's own LIMITATIONS regarding units)
        self.device_width = device_width  # meters
        self.resolution = resolution  # sub-dipole grid resolution per axis
        self.thickness = thickness  # meters
        self.mu0 = 4 * np.pi * 1e-7  # vacuum permeability, SI units

        self.subvolume = (device_length / resolution) * (device_width / resolution) * thickness  # volume represented by one sub-dipole
        self.submoment = Ms * self.subvolume  # magnetic moment represented by one sub-dipole, at full (untapered) Ms

        # Precompute every device's sub-dipole grid once, up front, so
        # compute_field_at_points does not regenerate it on every call.
        self.device_grids = [
            self.generate_device_grid(center, angle_rad, verbose=verbose)
            for center, angle_rad in zip(self.centers, self.angles_rad)
        ]
        if verbose:
            total_dipoles = sum(len(positions) for positions, _ in self.device_grids)
            print(f"[INFO] Magnetostatics.__init__: built {len(self.device_grids)} device grid(s), "
                  f"{total_dipoles} sub-dipole(s) total.")

    # --------------------------
    # ---- Build one device's sub-dipole grid (positions and tapered moments) ----
    def generate_device_grid(self, center, angle_rad, verbose=False):
        """
        GOAL:
            Generate the sub-dipole positions and moments for one
            elliptical device: a resolution x resolution rectangular grid,
            masked down to the points that actually fall inside the
            ellipse, with each sub-dipole's moment magnitude tapered
            (tanh-based falloff) toward the device's own edge to reduce
            numerical artifacts in the resulting field maps, then rotated
            and translated into the device's actual position/orientation.

        HOW IT TALKS TO OTHER FILES:
            Called by __init__ (this class), once per device, and by
            compute_field_over_device_volume (this class) when no explicit
            probe_points are given (using this same grid as the default
            probe volume).

        PARAMETERS:
            center (np.ndarray): this device's center, meters, (x, y).
            angle_rad (float): this device's orientation, radians.
            verbose (bool): prints the number of sub-dipoles kept after
                the ellipse mask when True.

        RETURNS:
            Tuple[np.ndarray, np.ndarray]: (positions_3d, moments_3d), each
            shape (N_dipoles, 3) -- this device's sub-dipole positions
            (meters) and moment vectors (in the device's own in-plane
            direction, magnitude tapered by position within the ellipse).

        LIMITATIONS:
            The edge taper uses a fixed beta=5.0 -- not exposed as a
            parameter. Increasing beta sharpens the falloff (closer to a
            hard cutoff at the ellipse boundary); decreasing it spreads the
            taper further into the device's interior.
        """
        x = np.linspace(-self.device_length / 2, self.device_length / 2, self.resolution)  # local grid, x, before ellipse masking
        y = np.linspace(-self.device_width / 2, self.device_width / 2, self.resolution)  # local grid, y, before ellipse masking
        grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)  # every (x, y) combination, flattened to one row per candidate sub-dipole

        # ---- Keep only sub-dipoles that fall inside the ellipse ----
        a = self.device_length / 2  # ellipse semi-major axis
        b = self.device_width / 2  # ellipse semi-minor axis
        inside_ellipse_mask = (grid[:, 0] / a) ** 2 + (grid[:, 1] / b) ** 2 <= 1.0  # standard ellipse membership test
        grid = grid[inside_ellipse_mask]  # keep only the sub-dipoles inside the ellipse

        # ---- Taper each sub-dipole's moment magnitude toward the device's own edge
        # (reduces numerical artifacts -- spurious peaks in magnitude/angle maps --
        # right at the boundary where the discretized grid otherwise cuts off sharply). ----
        rx = grid[:, 0] / a  # normalized x position within the ellipse
        ry = grid[:, 1] / b  # normalized y position within the ellipse
        r_ellipse = np.sqrt(rx**2 + ry**2)  # normalized radial position: 0 at center, 1 at edge
        beta = 5.0  # taper sharpness (see this function's own LIMITATIONS)
        Ms_taper = np.tanh(beta * (1 - r_ellipse))  # smooth falloff from ~1 at center to ~0 at the edge
        local_Ms = self.Ms * Ms_taper  # this sub-dipole's own tapered saturation magnetization
        local_moment = local_Ms * self.subvolume  # this sub-dipole's own tapered moment magnitude, shape (N_dipoles,)

        # ---- Rotate and translate the grid into this device's actual position/orientation ----
        rot_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])  # standard 2D rotation matrix
        positions_2d = grid @ rot_matrix.T + center  # rotate about the origin, then shift to this device's actual center

        positions_3d = np.hstack([positions_2d, np.zeros((len(positions_2d), 1))])  # expand to 3D, z=0 (this class is purely in-plane)

        # ---- Every sub-dipole in one device shares the same in-plane orientation
        # (the device's own magnetization angle); only the magnitude varies (via
        # the taper above). ----
        mx, my = np.cos(angle_rad), np.sin(angle_rad)  # this device's own in-plane direction
        moments_2d = np.stack([local_moment * mx, local_moment * my], axis=1)  # each sub-dipole's own moment vector, x/y
        moments_3d = np.hstack([moments_2d, np.zeros((len(moments_2d), 1))])  # expand to 3D, z=0

        if verbose:
            print(f"[DEBUG] Magnetostatics.generate_device_grid: kept {len(positions_3d)} sub-dipole(s) "
                  f"after ellipse mask (of {self.resolution ** 2} candidates).")
        return positions_3d, moments_3d

    # --------------------------
    # ---- Core field calculation: sum every sub-dipole's contribution at a set of points ----
    def compute_field_at_points(self, points, verbose=False):
        """
        GOAL:
            Compute the total magnetostatic stray field at a set of query
            points, by summing the analytic dipole field contribution from
            every sub-dipole in every device.

        HOW IT TALKS TO OTHER FILES:
            Called by generate_demo_seed_data.py directly, and by
            compute_field_over_device_volume (this class).

        PARAMETERS:
            points (array-like): Nx2 or Nx3 array of query points, in
                meters. A 2D input is expanded to 3D with z=0.
            verbose (bool): prints the number of points and total
                sub-dipole count when True.

        RETURNS:
            np.ndarray: (N, 3) field vectors, in tesla, at each query
            point.

        LIMITATIONS:
            See this file's module-level LIMITATIONS section regarding
            near-field divergence: a query point that lands extremely
            close to (within about 1 nm of) an actual sub-dipole position
            has that specific sub-dipole's own contribution excluded (via
            the `valid` mask below) rather than producing a numerically
            infinite value, but points close to -- without being exactly
            at -- a sub-dipole position can still produce very large field
            magnitudes, which is expected of this model, not a numerical
            error to fix.
        """
        points = np.atleast_2d(points)  # ensure a 2D array even if a single point was given
        if points.shape[1] == 2:  # a 2D (x, y) input: expand to 3D with z=0
            points = np.hstack([points, np.zeros((len(points), 1))])

        B_total = np.zeros_like(points)  # accumulated field, one row per query point

        n_dipoles = 0  # tracked only for the verbose summary below
        for sub_positions, sub_moments in self.device_grids:  # walk every device's own sub-dipole grid
            for rj, m_j in zip(sub_positions, sub_moments):  # walk every sub-dipole in this device
                n_dipoles += 1
                r_ij = points - rj  # displacement from this sub-dipole to every query point, shape (N_points, 3)
                r_mag = np.linalg.norm(r_ij, axis=1)  # distance from this sub-dipole to every query point
                valid = r_mag > 1e-9  # exclude query points essentially exactly at this sub-dipole's own position (see this function's own LIMITATIONS)

                r_hat = np.zeros_like(r_ij)  # unit displacement vector, only computed where valid
                r_hat[valid] = r_ij[valid] / r_mag[valid][:, np.newaxis]

                dot = np.sum(r_hat * m_j, axis=1)  # dot product of the unit displacement and this sub-dipole's moment, at every query point

                B = np.zeros_like(r_ij)  # this sub-dipole's own field contribution, only computed where valid
                B[valid] = (self.mu0 / (4 * np.pi)) * (
                    (3 * r_hat[valid] * dot[valid][:, np.newaxis] - m_j) / r_mag[valid][:, np.newaxis]**3
                )  # standard analytic dipole field formula

                B_total += B  # accumulate this sub-dipole's contribution into the running total
        if verbose:
            print(f"[DEBUG] Magnetostatics.compute_field_at_points: summed {n_dipoles} sub-dipole(s) "
                  f"at {points.shape[0]} query point(s).")
        return B_total


    # --------------------------
    # ---- Average the field over a device-shaped probe volume ----
    def compute_field_over_device_volume(self, position, angle_deg, probe_points=None, verbose=False):
        """
        GOAL:
            Compute the detailed spatial field over the volume of a
            hypothetical probe device at `position` with orientation
            `angle_deg`, and summarize it (average strength, average
            angle) -- typically used to evaluate the field acting ON one
            device due to every OTHER device (by constructing a separate
            Magnetostatics instance whose source devices exclude the probe
            device itself -- see compute_local_fields_for_all_devices for
            exactly that usage).

        HOW IT TALKS TO OTHER FILES:
            Called by compute_local_fields_for_all_devices (this class).
            Calls generate_device_grid (this class) when no explicit
            `probe_points` are given, and compute_field_at_points (this
            class) to evaluate the field at the resolved probe positions.

        PARAMETERS:
            position (np.ndarray): the probe device's center, meters.
            angle_deg (float): the probe device's orientation, degrees.
            probe_points (np.ndarray, optional): explicit Nx2 probe
                positions, meters, relative to (0, 0) -- rotated and
                translated to `position`/`angle_deg` before use. If not
                given, defaults to this instance's own device grid shape
                (generate_device_grid), evaluated at `position`/`angle_deg`.
            verbose (bool): prints the resolved average field strength and
                angle when True.

        RETURNS:
            Dict: {"probe_positions": the 3D coordinates actually sampled,
            "B_vectors": the full (Bx, By, Bz) field vector at each sampled
            point, "B_magnitudes_T": in-plane field magnitude at each
            point, "B_avg_strength_T": the mean of those magnitudes,
            "B_angles_deg": in-plane field angle at each point, degrees,
            "B_avg_angle_deg": the arithmetic mean of those angles, degrees
            (see LIMITATIONS)}.

        LIMITATIONS:
            B_avg_angle_deg is computed as a plain arithmetic mean of
            per-point angles in degrees, not a circular (weighted-vector)
            mean -- this can give a misleading result for a probe volume
            whose sampled angles span across the 0/360 degree wraparound
            boundary (for example, angles clustered around 359 and 1
            degrees average to 180 degrees, not the intuitively correct
            ~0 degrees). This differs from mumax3_csv_stats.py's own
            circular-mean statistics elsewhere in this codebase, which do
            handle wraparound correctly; this class predates that code and
            has not been reconciled with it.
        """
        angle_rad = np.radians(angle_deg)  # probe device orientation, radians

        if probe_points is None:  # no explicit probe points given: use this instance's own device grid shape
            sample_positions, _ = self.generate_device_grid(position, angle_rad, verbose=verbose)
        else:  # explicit probe points given: rotate and translate them into position
            rot_matrix = np.array([
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)]
            ])  # standard 2D rotation matrix
            relative_points = probe_points - position[:2]  # probe points, relative to the probe device's own center
            rotated_points = relative_points @ rot_matrix.T  # rotated into the probe device's orientation
            sample_positions_2d = rotated_points + position[:2]  # shifted back to the probe device's actual position
            sample_positions = np.hstack([sample_positions_2d, np.zeros((len(sample_positions_2d), 1))])  # expand to 3D, z=0

        fields = self.compute_field_at_points(sample_positions, verbose=verbose)  # field vector at every sampled position, shape (N_points, 3)

        B_magnitudes_T = np.linalg.norm(fields[:, :2], axis=0)  # in-plane field magnitude at each sampled point
        B_avg_strength_T = np.mean(B_magnitudes_T)  # mean in-plane field magnitude across the probe volume

        B_angles = np.arctan2(fields[:, 1], fields[:, 0])  # in-plane field angle at each sampled point, radians
        B_angles_deg = np.degrees(B_angles) % 360  # converted to degrees, wrapped into [0, 360)
        angle_avg_deg = np.mean(B_angles_deg)  # plain arithmetic mean (see this function's own LIMITATIONS regarding wraparound)

        if verbose:
            print(f"[DEBUG] Magnetostatics.compute_field_over_device_volume: "
                  f"avg strength={B_avg_strength_T:.4g} T, avg angle={angle_avg_deg:.1f} deg.")
        return {
            "probe_positions": sample_positions,  # 3D coordinates of sampled points
            "B_vectors": fields,  # full (Bx, By, Bz) field vectors at those points
            "B_magnitudes_T": B_magnitudes_T,  # field magnitudes (in-plane)
            "B_avg_strength_T": B_avg_strength_T,  # average field strength (scalar magnitude)
            "B_angles_deg": B_angles_deg,  # angles of field vectors (in-plane)
            "B_avg_angle_deg": angle_avg_deg  # average field angle (direction of mean vector; see this function's own LIMITATIONS)
        }


# --------------------------
# ---- Generate a fine probe-point grid inside an ellipse (module-level: see LIMITATIONS below) ----
def generate_points_inside_ellipse(center_nm, device_length_nm, device_width_nm, fine_res_nm, verbose=False):
    """
    GOAL:
        Generate a fine-resolution 2D point grid within an elliptical
        region, for use as high-resolution probe points (finer than the
        source device's own sub-dipole `resolution`).

    HOW IT TALKS TO OTHER FILES:
        Called by compute_local_fields_for_all_devices (this file), and by
        this file's own worked example at the bottom.

    PARAMETERS:
        center_nm (np.ndarray): ellipse center, nm, (x, y).
        device_length_nm, device_width_nm (float): ellipse full length/
            width (not semi-axes), nm.
        fine_res_nm (float): grid spacing, nm.
        verbose (bool): prints the number of points generated when True.

    RETURNS:
        np.ndarray: (N, 2) array of points inside the ellipse, nm.

    LIMITATIONS:
        This function used to be defined only inside this file's own
        __main__ block, which meant compute_local_fields_for_all_devices
        (a method of the Magnetostatics class, defined at module scope)
        referenced a name that did not exist unless this file was run
        directly as a script -- calling compute_local_fields_for_all_
        devices from another file raised NameError. Moved to module level,
        alongside the class, so it is always available to callers of that
        method regardless of how this file is imported or run.
    """
    a = device_length_nm / 2  # semi-major axis
    b = device_width_nm / 2  # semi-minor axis

    x = np.arange(center_nm[0] - a, center_nm[0] + a + fine_res_nm, fine_res_nm)  # candidate grid, x, spanning the ellipse's bounding box
    y = np.arange(center_nm[1] - b, center_nm[1] + b + fine_res_nm, fine_res_nm)  # candidate grid, y
    X, Y = np.meshgrid(x, y)

    X_rel = X - center_nm[0]  # candidate points, relative to the ellipse center
    Y_rel = Y - center_nm[1]
    mask = (X_rel / a)**2 + (Y_rel / b)**2 <= 1  # standard ellipse membership test

    inside_points = np.vstack([X[mask], Y[mask]]).T  # only the points inside the ellipse, shape (N, 2)
    if verbose:
        print(f"[DEBUG] generate_points_inside_ellipse: generated {len(inside_points)} point(s).")
    return inside_points


# --------------------------
# ---- Local field at every device in a DeviceArray, excluding each device's own contribution ----
def compute_local_fields_for_all_devices(device_array_obj, device_length_nm=60, device_width_nm=60,
                                          fine_res_nm=2, resolution=20, verbose=False):
    """
    GOAL:
        For every device in a geometry.DeviceArray, compute the local
        stray field acting on it due to every OTHER device in the array
        (its own contribution to its own local field is excluded, by
        constructing a fresh Magnetostatics instance per device whose
        source list omits the probe device itself).

    HOW IT TALKS TO OTHER FILES:
        Not currently called by any other file in this pipeline (see this
        file's module-level LIMITATIONS section regarding the device_
        array_obj shape this function assumes). Calls generate_points_
        inside_ellipse (this file), and constructs/uses a Magnetostatics
        instance (this file) per device.

    PARAMETERS:
        device_array_obj: an object exposing get_device_array() -> Nx5
            array in [x, y, sublattice, angle, m_angle] column order (the
            same convention geometry.DeviceArray uses) -- see this file's
            module-level LIMITATIONS section.
        device_length_nm, device_width_nm (float): probe device size, nm,
            shared by every device (used to build each device's own probe
            grid via generate_points_inside_ellipse).
        fine_res_nm (float): probe grid spacing, nm -- typically finer
            than the source devices' own sub-dipole `resolution`, for a
            more accurate local-field average.
        resolution (int): sub-dipole grid resolution per axis for the
            SOURCE devices (every device except the one currently being
            probed).
        verbose (bool): shows a progress bar (via tqdm) across devices
            when True; the tqdm dependency is only imported here, not at
            module level, specifically so constructing a plain
            Magnetostatics instance elsewhere in this codebase does not
            require tqdm to be installed.

    RETURNS:
        pandas.DataFrame: one row per device, with columns
        {"device_index", "x_nm", "y_nm", "angle_deg", "sublattice", "Bx_T",
        "By_T", "Bz_T", "B_magnitude_T", "angle_deg_from_field"}.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section regarding the
        [x, y, sublattice, angle, m_angle] column order this function
        assumes from device_array_obj.get_device_array() -- earlier
        revisions of this function read the angle and sublattice columns
        in the opposite order (angle from column 2, sublattice from column
        3), the same class of column-order bug found and fixed in
        geometry.py's set_device_property; fixed here to match the
        convention used everywhere else in this codebase (column 2 =
        sublattice, column 3 = angle).
    """
    from tqdm import tqdm  # progress bar -- imported here, not at module level, so constructing a Magnetostatics instance elsewhere does not require tqdm to be installed
    device_array = device_array_obj.get_device_array()
    n_devices = device_array.shape[0]

    results = []  # accumulated across every device below
    for i in tqdm(range(n_devices), desc="Computing local fields", disable=not verbose):  # one probe device at a time
        # ---- Separate the probe device from its neighbors (the field sources) ----
        probe_device = device_array[i]
        probe_center_nm = probe_device[:2]  # this device's own center, nm
        probe_angle_deg = probe_device[3]  # this device's own geometry angle -- column 3 (see this function's own LIMITATIONS regarding the column-order fix)
        sublattice_id = int(probe_device[2])  # this device's own sublattice ID -- column 2 (see this function's own LIMITATIONS)

        source_devices = np.delete(device_array, i, axis=0)  # every OTHER device -- the field sources for this probe device
        source_centers = source_devices[:, :2]  # source devices' own centers, nm
        source_angles = source_devices[:, 3]  # source devices' own geometry angles -- column 3, same fix as above

        ms = Magnetostatics(centers=source_centers, angles_deg=source_angles, resolution=resolution)  # source field, excluding the probe device itself

        probe_points_nm = generate_points_inside_ellipse(  # fine probe grid inside the probe device's own footprint
            center_nm=probe_center_nm,
            device_length_nm=device_length_nm,
            device_width_nm=device_width_nm,
            fine_res_nm=fine_res_nm,
        )
        probe_points_m = probe_points_nm * 1e-9  # nm -> m
        probe_center_m = probe_center_nm * 1e-9  # nm -> m

        B_local = ms.compute_field_over_device_volume(  # average field over the probe device's own footprint, due to every other device
            position=probe_center_m,
            angle_deg=probe_angle_deg,
            probe_points=probe_points_m
        )

        # NOTE (bug fix -- see this function's own LIMITATIONS): the previous
        # revision read `Bx, By, Bz = B_local`, unpacking the six-key dict
        # compute_field_over_device_volume returns as if it were a 3-element
        # (Bx, By, Bz) tuple -- this raised ValueError on any call, since a dict
        # unpacks by iterating its keys, and there are six of them, not three.
        # Fixed by computing the genuine average field VECTOR (not just the
        # average magnitude/angle already in B_local) directly from B_local's
        # own "B_vectors" array.
        Bx, By, Bz = np.mean(B_local["B_vectors"], axis=0)  # averaged field vector across the probe device's own footprint
        B_mag = B_local["B_avg_strength_T"]  # average in-plane field magnitude (see compute_field_over_device_volume's own return value)
        angle_deg = B_local["B_avg_angle_deg"]  # average in-plane field angle, degrees (see that function's own LIMITATIONS regarding circular-mean handling)

        results.append({
            "device_index": i,
            "x_nm": probe_center_nm[0],
            "y_nm": probe_center_nm[1],
            "angle_deg": probe_angle_deg,
            "sublattice": sublattice_id,
            "Bx_T": Bx,
            "By_T": By,
            "Bz_T": Bz,
            "B_magnitude_T": B_mag,
            "angle_deg_from_field": angle_deg
        })

    if verbose:
        print(f"[INFO] compute_local_fields_for_all_devices: computed local fields for {n_devices} device(s).")
    return pd.DataFrame(results)


# --------------------------
# ---- Worked example: NN/NNN dipole configuration sweep and angle-map plotting ----
if __name__ == "__main__":
    # Guarded so that `import magnetostatics` is a clean, side-effect-free library
    # import (no live plt.show() call) -- running `python magnetostatics.py`
    # directly still executes this worked example exactly as before.
    centers = np.array([
        [0, 0],  # X NN (SW)
        [120, 0],  # X NN (SE)
        [0, 120],  # X NN (NW)
        [120, 120],  # X NN (NE)
        [60, 180],  # Y NNN (N)
        [60, -60],  # Y NNN (S)
        [180, 60],  # Y NNN (E)
        [-60, 60],  # Y NNN (W)
    ])

    # Practice angles:
    angles_input_deg = [0, 180, 180, 0, 90, 270]  # A1 90

    # ---- NN/NNN(NS) configurations (uncomment one to try it) ----
    # angles_deg = [0, 180, 180, 0, 90, 270] # A1 90
    # angles_deg = [0, 180, 0, 0, 90, 90]  # B1 74
    # angles_deg = [0, 0, 180, 0, 270, 90]  # C1 69
    # angles_deg = [0, 0, 180, 0, 270, 270] # D1 56
    # angles_deg = [0, 0, 0, 0, 90, 90]  # E1 27.5

    # ---- NN/NNN(NS)/NNN(EW) configurations (uncomment one to try it) ----
    # angles_input_deg = [0, 180, 180, 0, 90, 270, 270, 270] # A1-1 90, 74 Oe (chiral)
    # angles_input_deg = [0, 180, 180, 0, 90, 270, 90, 270] # A1-2 89.94 (single defect)
    # angles_input_deg = [0, 180, 180, 0, 90, 270, 270, 90] # A1-3 90.06 (single defect, inverted)
    # angles_input_deg = [0, 180, 180, 0, 90, 270, 90, 90] # A1-4 90 (double defect)

    device_properties = {
        "device_length": 60e-9,
        "device_width": 60e-9,
        "thickness": 2.5e-9
    }

    ms = Magnetostatics(centers, angles_input_deg, resolution=20)

    field_at_center = ms.compute_field_at_points(np.array([[60, 60]]) * 1e-9)
    print("Field at center (T):", field_at_center[0])
    print("Angle at center (deg):", np.degrees(np.arctan2(field_at_center[0][1], field_at_center[0][0])))

    position_nm = np.array([90, 60])
    position_m = position_nm * 1e-9
    center_angle = 0  # in-plane angle of the device

    center_nm = np.array([60, 60])
    device_length_nm = 60  # 2a
    device_width_nm = 60  # 2b
    cell_size = 2  # resolution

    # ---- Full area map: compute field at every probe point in a grid ----
    map_size = (np.arange(-100, 220 + cell_size, cell_size), np.arange(-100, 220 + cell_size, cell_size))
    X_nm, Y_nm = np.meshgrid(*map_size, indexing='ij')  # shape (N_x, N_y)
    x_range_nm = map_size[0]
    y_range_nm = map_size[1]
    probe_points = np.stack([X_nm, Y_nm], axis=-1).reshape(-1, 2) * 1e-9  # nm -> m
    B_full = ms.compute_field_at_points(probe_points)  # shape (N_points, 3)
    Bx = B_full[:, 0].reshape(X_nm.shape)
    By = B_full[:, 1].reshape(Y_nm.shape)

    # ---- Zoomed-in region: compute field at probe points inside one device's own ellipse ----
    probe_points_center = generate_points_inside_ellipse(center_nm, device_length_nm, device_width_nm, fine_res_nm=cell_size)
    probe_points_center = probe_points_center * 1e-9  # nm -> m
    B_center = ms.compute_field_over_device_volume(
        position=position_m,
        angle_deg=center_angle,
        probe_points=probe_points_center
    )
    print("Average field strength (T):", B_center["B_avg_strength_T"] * 10000)  # tesla -> Oe
    print("Average field angle (deg):", B_center["B_avg_angle_deg"])

    # Compute angle of the field vector at each grid point (in degrees)
    angles_deg = np.degrees(np.arctan2(By, Bx))
    for idx in range(len(angles_deg.flat)):
        if angles_deg.flat[idx] < 0:
            angles_deg.flat[idx] += 360

    # ---- Ellipse mask, for the zoomed-in subplots below ----
    center_nm = np.array([60, 60])
    a = 30  # semi-major axis
    b = 30  # semi-minor axis

    X_rel = X_nm - center_nm[0]
    Y_rel = Y_nm - center_nm[1]
    ellipse_mask = (X_rel / a)**2 + (Y_rel / b)**2 <= 1.0
    final_mask = ellipse_mask
    masked_angles = np.where(final_mask, angles_deg, np.nan)
    center_x = (x_range_nm[0] + x_range_nm[-1]) / 2  # center of full map x coordinates
    center_y = (y_range_nm[0] + y_range_nm[-1]) / 2  # center of full map y coordinates
    half_width = 30  # half of 60 nm (zoomed-in region)

    # ---- Plot: full map, zoomed map (global scale), zoomed map (local scale) ----
    with mpl.rc_context(rc={'axes.labelsize': 20, 'xtick.labelsize': 15, 'ytick.labelsize': 15}):
        fig, axs = plt.subplots(1, 3, figsize=(24, 8), constrained_layout=True)

        # Left: full angle map
        im0 = axs[0].imshow(
            angles_deg,
            origin='lower',
            extent=[x_range_nm[0], x_range_nm[-1], y_range_nm[0], y_range_nm[-1]],
            cmap='hsv',
            aspect='equal',
            vmin=0,
            vmax=360  # force full hue cycle
        )
        axs[0].set_title('Full Angle Map')
        axs[0].set_xlabel('x (nm)')
        axs[0].set_ylabel('y (nm)')
        axs[0].set_title('E1 Configuration Dipole Map - Mismatched meshes')
        cbar_im0 = fig.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.04, pad=0.04)
        cbar_im0.set_label('Angle (degrees)')

        # Center: zoomed-in to a 60x60nm region, global color scale
        im1 = axs[1].imshow(
            masked_angles,
            origin='lower',
            extent=[x_range_nm[0], x_range_nm[-1], y_range_nm[0], y_range_nm[-1]],
            cmap='hsv',
            aspect='equal',
            vmin=0,
            vmax=360  # force full hue cycle
        )
        axs[1].set_xlim(center_x - half_width, center_x + half_width)
        axs[1].set_ylim(center_y - half_width, center_y + half_width)
        axs[1].set_xlabel('x (nm)')
        axs[1].set_ylabel('y (nm)')
        axs[1].set_title('Global Angle Map (60x60 nm)')
        cbar_im1 = fig.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.04, pad=0.04)
        cbar_im1.set_label('Angle (degrees)')

        # Right: zoomed-in to a 60x60nm region, local color scale (gradient of angles)
        local_min = np.nanmin(masked_angles)  # min angle in the zoomed region
        local_max = np.nanmax(masked_angles)  # max angle in the zoomed region
        im2 = axs[2].imshow(
            masked_angles,
            origin='lower',
            extent=[x_range_nm[0], x_range_nm[-1], y_range_nm[0], y_range_nm[-1]],
            cmap='twilight_shifted',
            aspect='equal',
            vmin=local_min,
            vmax=local_max  # local hue cycle
        )
        axs[2].set_xlim(center_x - half_width, center_x + half_width)
        axs[2].set_ylim(center_y - half_width, center_y + half_width)
        axs[2].set_title('Gradient Angle Map (60x60 nm)')
        axs[2].set_xlabel('x (nm)')
        axs[2].set_ylabel('y (nm)')
        cbar_im2 = fig.colorbar(im2, ax=axs[2], orientation='vertical', fraction=0.04, pad=0.04)
        cbar_im2.set_label('Angle (degrees)')

        plt.show()
