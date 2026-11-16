r"""
File Name: mumax3_plot_calls.py
Author: Connor Sullivan
Date: 2026.08.05
Revision: v2

============================================================
PURPOSE
============================================================
Builds the `plot_configs` data structures mumax3_parent_utilities.
run_spatial_plots consumes to render every color-map and quiver plot type
for a given dataset. Each plot_configs entry describes one plot type (for
example, "magnitude", "angle", "quiver"), including its color map, color
scale bounds, which scalar_func (from mumax3_pipeline_utilities.py)
computes the values to plot, and which scale variants (global/local/custom)
to render. Contains:
    - build_plot_configs_quiver: the averaged-quiver plot type, usable for
      any dataset.
    - build_plot_configs_Bdemag: every plot type for the B_demag dataset
      (angle, magnitude, Bx/By/Bz components, gradient, hue).
    - build_plot_configs_mfull: every plot type for the m_full dataset
      (magnitude, angle, mx/my components; see this file's LIMITATIONS
      section regarding additional disabled configs in this function).

This file does not perform any plotting itself, and does not read or write
any files -- it only constructs and returns the configuration dictionaries
mumax3_parent_utilities.run_spatial_plots and mumax3_csv_plot.py's
plotting functions act on.

============================================================
LIMITATIONS
============================================================
- build_plot_configs_mfull contains a block of commented-out plot config
  entries (mz component maps, gradient maps, hue maps) following the active
  ones. These are intentionally disabled, not stale leftovers -- flagged
  here as something worth confirming is still the desired state, since it
  was not obvious from the surrounding code alone which of "temporarily
  disabled" or "no longer wanted" was intended. Uncomment the relevant
  block (and add a matching `vmin_mfull`/`vmax_mfull`-style bound if the
  hue config's vmin_mfull/vmax_mfull references do not already match this
  function's own parameters) to re-enable.
- These functions perform no validation of their own beyond what Python
  itself enforces -- for example, an incorrect vmin/vmax pair (vmin greater
  than vmax) would be passed through unchanged into the resulting
  plot_configs and would only surface as a visual problem once plotting is
  attempted.
"""

from functools import partial  # binding a fixed `comp` argument into matrix_component for each field-component plot config
from mumax3_pipeline_utilities import (
    matrix_angle, matrix_magnitude, matrix_component, matrix_gradient_hsv_rgba,
    matrix_hue_hsv_rgba, matrix_uv, compute_global_scalar_limits,
)  # the scalar_func implementations every plot config below references


def build_plot_configs_quiver(dataset: str, field_scale: float = 1.0, verbose: bool = False):
    """
    GOAL:
        Build the plot_configs entry for the averaged-quiver plot type --
        one arrow per averaging window (see
        mumax3_plot_vectors.compute_averaged_quiver_grid) instead of a
        color map. Works for any dataset (m_full, B_demag, ...) via the
        same matrix_uv scalar_func.

    HOW IT TALKS TO OTHER FILES:
        Called by both parent scripts wherever quiver plots are requested.
        The returned plot_configs list is passed to
        mumax3_parent_utilities.run_spatial_plots, which dispatches to
        mumax3_csv_plot.py's plotting functions -- the "plot_type": "quiver"
        entry inside `plot_args` below is what tells that plotting code to
        render arrows instead of a color map.

    PARAMETERS:
        dataset (str): dataset name (for example, "m_full" or "B_demag"),
            used only for this plot type's own name/folder naming -- not
            passed to matrix_uv itself.
        field_scale (float): multiplier applied to the field values before
            plotting (for example, 1.0 for m_full, 10000 for B_demag/
            B_exch/etc, matching build_plot_configs_Bdemag/_mfull below).
        verbose (bool): prints the built config's name when True.

    RETURNS:
        List[Dict]: a one-entry plot_configs list for the quiver plot type.

    LIMITATIONS:
        Unlike the color-map configs in build_plot_configs_Bdemag/_mfull,
        quiver has no vmin/vmax color scale to sweep global/local/custom
        variants over, so there is exactly one "scales" entry (named "raw",
        so the output folder hierarchy stays data_type/plot_type/scale/...
        consistent with every other plot type). The actual per-region
        averaging window size is not set here -- it comes from
        `quiver_window` in the caller's SpatialAnalysisConfig, forwarded
        through as a separate argument by run_spatial_plots itself.
    """
    result = [{
        "name": f"{dataset}_quiver_maps",  # this plot type's name, used for its output subfolder
        "prefix": f"{dataset}_quiver",  # filename prefix for this plot type's output
        "cmap": None,  # quiver arrows have no color map
        "vmin": None,  # quiver arrows have no color scale
        "vmax": None,
        "scales": [
            {"name": "raw", "local_scale": False, "scalar_func": matrix_uv},  # the single scale variant for quiver -- see function LIMITATIONS
        ],
        "plot_args": {
            "field_scale": field_scale,  # multiplier applied to the field values before plotting
            "plot_type": "quiver",  # tells the plotting layer to render arrows, not a color map
        },
        "runner_args": {"reset_dir": True},  # start this plot type's output directory empty on each run
    }]
    if verbose:
        print(f"[DEBUG] build_plot_configs_quiver: built config for dataset '{dataset}'.")
    return result


def build_plot_configs_Bdemag(vmin_Bdemag, vmax_Bdemag, vmin_Bx, vmax_Bx, vmin_By, vmax_By, vmin_Bz, vmax_Bz, verbose=False):
    """
    GOAL:
        Build the plot_configs list covering every plot type for the
        B_demag dataset: angle, magnitude, Bx/By/Bz components, gradient
        (circular deviation from the region's mean angle), and hue
        (angle-as-hue, magnitude-as-value).

    HOW IT TALKS TO OTHER FILES:
        Called by whichever parent script is analyzing a B_demag dataset,
        typically after computing each metric's global min/max via
        mumax3_pipeline_utilities.compute_global_scalar_limits. The
        returned plot_configs list is passed to
        mumax3_parent_utilities.run_spatial_plots.

    PARAMETERS:
        vmin_Bdemag, vmax_Bdemag (float): global color-scale bounds for the
            magnitude map.
        vmin_Bx, vmax_Bx, vmin_By, vmax_By, vmin_Bz, vmax_Bz (float): global
            color-scale bounds for each field component's map.
        verbose (bool): prints the number of plot types built when True.

    RETURNS:
        List[Dict]: one entry per plot type (angle, magnitude, Bx, By, Bz,
        gradient, hue), each with its own "global"/"local" (and, for
        gradient, "custom") scale variants.
    """
    result = [
        # Angle maps: hsv colormap, fixed 0-360 degree scale (an angle's meaningful
        # range does not change frame to frame, so "global" and "local" scaling are
        # identical here -- both variants are kept for interface consistency with
        # every other plot type, which does have a real global/local distinction).
        {
            "name": "B_demag_angle_maps",  # this plot type's name, used for its output subfolder
            "prefix": "Bdemag_angle",  # filename prefix for this plot type's output
            "cmap": "hsv",  # cyclic color map, appropriate for an angle
            "vmin": 0,  # fixed lower bound: 0 degrees
            "vmax": 360,  # fixed upper bound: 360 degrees
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": matrix_angle},  # shared color scale across every frame
                {"name": "local", "local_scale": True, "scalar_func": matrix_angle},  # each frame scaled to its own min/max
            ],
            "runner_args": {"reset_dir": True}  # start this plot type's output directory empty on each run
        },
        # Magnitude maps: viridis colormap, scaled to Oe (field_scale=10000).
        {
            "name": "B_demag_magnitude_maps",
            "prefix": "Bdemag_magnitude",
            "cmap": "viridis",
            "vmin": vmin_Bdemag,  # global color-scale lower bound, provided by the caller
            "vmax": vmax_Bdemag,  # global color-scale upper bound, provided by the caller
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": matrix_magnitude},
                {"name": "local", "local_scale": True, "scalar_func": matrix_magnitude},
            ],
            "plot_args": {"field_scale": 10000},  # Tesla -> Oersted
            "runner_args": {"reset_dir": True}
        },
        # Bx component maps: seismic colormap (diverging, so positive/negative are
        # visually distinct), scaled to Oe.
        {
            "name": "B_demag_Bx_maps",
            "prefix": "Bdemag_Bx",
            "cmap": "seismic",
            "vmin": vmin_Bx,
            "vmax": vmax_Bx,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": partial(matrix_component, comp='masked_x')},  # matrix_component with its `comp` argument pre-bound to the x component
                {"name": "local", "local_scale": True, "scalar_func": partial(matrix_component, comp='masked_x')},
            ],
            "plot_args": {"field_scale": 10000},
            "runner_args": {"reset_dir": True}
        },
        # By component maps: same treatment as Bx above, for the y component.
        {
            "name": "B_demag_By_maps",
            "prefix": "Bdemag_By",
            "cmap": "seismic",
            "vmin": vmin_By,
            "vmax": vmax_By,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": partial(matrix_component, comp='masked_y')},
                {"name": "local", "local_scale": True, "scalar_func": partial(matrix_component, comp='masked_y')},
            ],
            "plot_args": {"field_scale": 10000},
            "runner_args": {"reset_dir": True}
        },
        # Bz component maps: same treatment as Bx/By above, for the z component.
        {
            "name": "Bz_maps",
            "prefix": "Bdemag_Bz",
            "cmap": "seismic",
            "vmin": vmin_Bz,
            "vmax": vmax_Bz,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": partial(matrix_component, comp='masked_z')},
                {"name": "local", "local_scale": True, "scalar_func": partial(matrix_component, comp='masked_z')},
            ],
            "plot_args": {"field_scale": 10000},
            "runner_args": {"reset_dir": True}
        },
        # Gradient maps: circular deviation from the region's own mean angle at each
        # frame, rendered as an RGBA image (see mumax3_pipeline_utilities.
        # matrix_gradient_hsv_rgba). Includes a "custom" scale variant, fixed to
        # +/-50 degrees, to focus on a specific deviation range regardless of the
        # frame's actual spread.
        {
            "name": "B_demag_gradient_maps",
            "prefix": "Bdemag_gradient",
            "cmap": "hsv",
            "vmin": 0,
            "vmax": 360,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": matrix_gradient_hsv_rgba},
                {"name": "local", "local_scale": True, "scalar_func": matrix_gradient_hsv_rgba},
                {
                    "name": "custom",
                    "local_scale": True,
                    "custom_scale": [True, -50, 50],  # fixed deviation range: -50 to +50 degrees
                    "scalar_func": matrix_gradient_hsv_rgba,
                },
            ],
            "plot_args": {
                "field_scale": 10000,
                "plot_type": "gradient"  # tells the plotting layer to render the RGBA image this scalar_func produces
            },
            "runner_args": {"reset_dir": True}
        },
        # Hue maps: angle-as-hue, magnitude-as-value, rendered as an RGBA image (see
        # mumax3_pipeline_utilities.matrix_hue_hsv_rgba).
        {
            "name": "B_demag_hue_maps",
            "prefix": "Bdemag_hue",
            "cmap": "hsv",
            "vmin": vmin_Bdemag,
            "vmax": vmax_Bdemag,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": matrix_hue_hsv_rgba},
                {"name": "local", "local_scale": True, "scalar_func": matrix_hue_hsv_rgba},
            ],
            "plot_args": {
                "field_scale": 10000,
                "plot_type": "hue"  # tells the plotting layer to render the RGBA image this scalar_func produces
            },
            "runner_args": {"reset_dir": True}
        },
    ]
    if verbose:
        print(f"[DEBUG] build_plot_configs_Bdemag: built {len(result)} plot type config(s).")
    return result


def build_plot_configs_mfull(vmin_mfull, vmax_mfull, verbose=False):
    """
    GOAL:
        Build the plot_configs list covering every ACTIVE plot type for the
        m_full dataset: magnitude, angle, mx, my. See this function's
        LIMITATIONS regarding additional plot types present in this
        function's source but currently disabled.

    HOW IT TALKS TO OTHER FILES:
        Called by whichever parent script is analyzing an m_full dataset,
        typically after computing each metric's global min/max via
        mumax3_pipeline_utilities.compute_global_scalar_limits. The
        returned plot_configs list is passed to
        mumax3_parent_utilities.run_spatial_plots.

    PARAMETERS:
        vmin_mfull, vmax_mfull (float): global color-scale bounds for the
            magnitude map.
        verbose (bool): prints the number of active plot types built when
            True.

    RETURNS:
        List[Dict]: one entry per ACTIVE plot type (magnitude, angle, mx,
        my), each with its own "global"/"local" scale variants.

    LIMITATIONS:
        See this file's module-level LIMITATIONS section: this function's
        source also contains a commented-out block covering mz component
        maps, gradient maps, and hue maps for m_full, intentionally
        disabled rather than removed. Confirm whether that block should
        stay disabled, be re-enabled, or be removed outright.
    """
    result = [
        # Magnitude maps: viridis colormap. field_scale=1 since m_full is already
        # dimensionless (a unit magnetization vector), unlike B_demag's Tesla ->
        # Oersted conversion.
        {
            "name": "m_full_magnitude_maps",
            "prefix": "m_full_magnitude",
            "cmap": "viridis",
            "vmin": vmin_mfull,  # global color-scale lower bound, provided by the caller
            "vmax": vmax_mfull,  # global color-scale upper bound, provided by the caller
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": matrix_magnitude},
                {"name": "local", "local_scale": True, "scalar_func": matrix_magnitude},
            ],
            "plot_args": {"field_scale": 1},  # m_full is already dimensionless
            "runner_args": {"reset_dir": True}
        },
        # Angle maps: hsv colormap, fixed 0-360 degree scale (see the equivalent
        # B_demag config in build_plot_configs_Bdemag for why "global"/"local" are
        # identical here).
        {
            "name": "m_full_angle_maps",
            "prefix": "m_full_angle",
            "cmap": "hsv",
            "vmin": 0,
            "vmax": 360,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": matrix_angle},
                {"name": "local", "local_scale": True, "scalar_func": matrix_angle},
            ],
            "plot_args": {"field_scale": 1},
            "runner_args": {"reset_dir": True}
        },
        # mx component maps: seismic colormap, fixed -1 to +1 scale (a unit
        # magnetization component's natural range).
        {
            "name": "m_full_mx_maps",
            "prefix": "m_full_mx",
            "cmap": "seismic",
            "vmin": -1,
            "vmax": 1,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": partial(matrix_component, comp='masked_x')},
                {"name": "local", "local_scale": True, "scalar_func": partial(matrix_component, comp='masked_x')},
            ],
            "plot_args": {"field_scale": 1},
            "runner_args": {"reset_dir": True}
        },
        # my component maps: same treatment as mx above, for the y component.
        {
            "name": "m_full_my_maps",
            "prefix": "m_full_my",
            "cmap": "seismic",
            "vmin": -1,
            "vmax": 1,
            "scales": [
                {"name": "global", "local_scale": False, "scalar_func": partial(matrix_component, comp='masked_y')},
                {"name": "local", "local_scale": True, "scalar_func": partial(matrix_component, comp='masked_y')},
            ],
            "plot_args": {"field_scale": 1},
            "runner_args": {"reset_dir": True}
        },
        # ---- The following plot types are intentionally disabled, not stale --
        # see this function's LIMITATIONS section. Uncomment to re-enable.
        #
        # mz component maps: same treatment as mx/my above, for the z component.
        # {
        #     "name": "m_full_mz_maps",
        #     "prefix": "m_full_mz",
        #     "cmap": "seismic",
        #     "vmin": -1,
        #     "vmax": 1,
        #     "scales": [
        #         {
        #             "name": "global",
        #             "local_scale": False,
        #             "scalar_func": partial(matrix_component, comp='masked_z'),
        #         },
        #         {
        #             "name": "local",
        #             "local_scale": True,
        #             "scalar_func": partial(matrix_component, comp='masked_z'),
        #         },
        #     ],
        #     "plot_args": {
        #         "field_scale": 1
        #     },
        #     "runner_args": {
        #         "reset_dir": True
        #     }
        # },
        # Gradient maps: same treatment as B_demag's gradient config -- see
        # build_plot_configs_Bdemag.
        # {
        #     "name": "m_full_gradient_maps",
        #     "prefix": "m_full_gradient",
        #     "cmap": "hsv",
        #     "vmin": 0,
        #     "vmax": 360,
        #     "scales": [
        #         {
        #             "name": "global",
        #             "local_scale": False,
        #             "scalar_func": matrix_gradient_hsv_rgba,
        #         },
        #         {
        #             "name": "local",
        #             "local_scale": True,
        #             "scalar_func": matrix_gradient_hsv_rgba,
        #         },
        #         {
        #             "name": "custom",
        #             "local_scale": True,
        #             "custom_scale": [True, -50, 50], # custom scaling for gradient maps to focus on specific range of angles (e.g., -50 to 50 degrees)
        #             "scalar_func": matrix_gradient_hsv_rgba,
        #         },
        #     ],
        #     "plot_args": {
        #         "field_scale": 1,
        #         "plot_type": "gradient"
        #     },
        #     "runner_args": {"reset_dir": True}
        # },
        # Hue maps: same treatment as B_demag's hue config -- see
        # build_plot_configs_Bdemag.
        # {
        #     "name": "m_full_hue_maps",
        #     "prefix": "m_full_hue",
        #     "cmap": "hsv",
        #     "vmin": vmin_mfull,
        #     "vmax": vmax_mfull,
        #     "scales": [
        #         {
        #             "name": "global",
        #             "local_scale": False,
        #             "scalar_func": matrix_hue_hsv_rgba,
        #         },
        #         {
        #             "name": "local",
        #             "local_scale": True,
        #             "scalar_func": matrix_hue_hsv_rgba,
        #         },
        #     ],
        #     "plot_args": {
        #         "field_scale": 1,
        #         "plot_type": "hue"
        #     },
        #     "runner_args": {"reset_dir": True}
        # },
    ]
    if verbose:
        print(f"[DEBUG] build_plot_configs_mfull: built {len(result)} active plot type config(s) (see function LIMITATIONS regarding disabled ones).")
    return result
