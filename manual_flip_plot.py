r"""
File Name: manual_flip_plot.py

PURPOSE
    Standalone plotting of a HAND-WRITTEN propagation: you list which devices
    flip at each time step, and this draws one frame per step, a summary
    figure, a flip-log CSV, and a GIF -- on the same geometry as your
    propagation runs. No simulation, no CSVs, no engine needed.

HOW TO USE
    1. Put this file in the project root (next to mumax3_parent_propagation.py).
    2. Edit the "USER INPUT" section: geometry (copied from your config),
       the starting magnetization, and STEPS.
    3. python manual_flip_plot.py

STEPS FORMAT
    STEPS[0] is generation 0 (the injections). STEPS[g] is what flips at step g.
    Each entry is either
        12            -> device 12 flips
        (7, 12)       -> device 12 flips, caused by source 7 (draws an arrow 7 -> 12)
    Listing a device in more than one step flips it again (180 deg each time).

SOURCES OF TRUTH
    Geometry comes from geometry.DeviceArray (built from the same parameters as
    PropagationConfig). Starting magnetization = the pattern from
    magnetization_mode/base angles, then INITIAL_FLIP_INDICES flipped 180 deg
    (the same idea as the hand-edit block in propagation_engine.setup_seed_and_geometry).
"""

import os
import sys
import csv

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# USER INPUT
# =====================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_DIRS = [  # folders containing geometry.py and propagation_utilities.py
    os.path.join(HERE, "src", "_superposition"),
    os.path.join(HERE, "src", "_propagation"),
    os.path.join(HERE, "src", "_mumax3"),
]

OUTPUT_DIR = os.path.join(HERE, "manual_flip_output")

# ---- geometry: copy from your PropagationConfig ----
cc = 80
GEOM = dict(
    device_length=70, device_width=56,
    nn_x=2 * cc, nn_y=int(2 * cc * np.sqrt(3)),
    sublattice_dims={0: (4, 2), 1: (5, 1), 2: (4, 2), 3: (4, 2), 4: (4, 2), 5: (4, 2)},
    device_rotations={0: 90, 1: 90, 2: 120, 3: 120, 4: 60, 5: 60},
    rotation_angles=None,
    translations={1: [-cc, cc * np.sqrt(3)], 2: [-cc / 2, cc * np.sqrt(3) / 2],
                  3: [cc / 2, -cc * np.sqrt(3) / 2], 4: [cc / 2, cc * np.sqrt(3) / 2],
                  5: [-cc / 2, -cc * np.sqrt(3) / 2]},
    sublattice_termination={j: {"pos_x": -1, "pos_y": 0, "neg_x": 0, "neg_y": 0} for j in range(6)},
    sublattice_termination_mode="fixed_row_col",
)
INDICES_TO_REMOVE = [11, 20, 27, 28]

# ---- starting magnetization ----
MAG_MODE = "opposite"  # 'identical' | 'opposite' | 'random'
MAG_BASE_ANGLE = {0: 90, 1: 90, 2: 120, 3: 120, 4: 60, 5: 60}
RANDOM_SEED = None
INITIAL_FLIP_INDICES = []  # devices flipped 180 deg from that pattern BEFORE step 0 (e.g. your hand-edit list)

# ---- the propagation you want to draw ----
STEPS = [
    [0, 6, 10, 21],            # generation 0: injections
    [1, 2, 7, 25],    # generation 1  ((source, target) draws an arrow; bare int just flips)
    [1, 2, 16, 17, 22, 25, 26],                   # generation 2
    [1, 7, 11, 16, 21, 25, 27],
]

# ---- plot options ----
SHOW_INDEX_LABELS = True
SHOW_MAG_ARROWS = True
SHOW_SOURCE_ARROWS = True      # arrows for (source, target) entries
MAKE_GIF = True
GIF_FRAME_MS = 900
DPI = 200
COLORMAP = "cool"
# =====================================================================


def build_array():
    """Build the geometry + starting magnetization. Returns (centers_nm, geom_angles_deg, lengths_nm, mag_angles_deg)."""
    for d in SRC_DIRS:
        if os.path.isdir(d) and d not in sys.path:
            sys.path.append(d)
    from geometry import DeviceArray
    from propagation_utilities import get_device_mesh

    array = DeviceArray(verbose=False, **GEOM)
    if INDICES_TO_REMOVE:
        array.remove_devices(indices=INDICES_TO_REMOVE, verbose=False)
    for j in range(len(GEOM["sublattice_dims"])):
        array.set_magnetization_pattern(
            mode=MAG_MODE, sublattice=j,
            base_angle=MAG_BASE_ANGLE.get(j) if MAG_BASE_ANGLE is not None else None,
            random_seed=RANDOM_SEED if MAG_MODE == "random" else None, verbose=False)
    for idx in INITIAL_FLIP_INDICES:
        array.set_device_magnetization(idx, (array.device_m_angles[idx] + 180.0) % 360.0)

    centers, geom_angles, mag_angles = get_device_mesh(array)
    lengths = np.array([float(array.device_data[i]["device_length_nm"]) for i in range(len(centers))])
    return (np.asarray(centers, float), np.asarray(geom_angles, float), lengths, np.asarray(mag_angles, float))


def parse_steps(steps, n_devices):
    """Normalize STEPS into [(targets, edges)] per generation and validate indices."""
    parsed = []
    for g, step in enumerate(steps):
        targets, edges = [], []
        for item in step:
            if isinstance(item, (tuple, list)):
                src, tgt = int(item[0]), int(item[1])
                edges.append((src, tgt))
                for k in (src, tgt):
                    if not 0 <= k < n_devices:
                        raise ValueError(f"STEPS[{g}]: index {k} out of range 0..{n_devices - 1}")
            else:
                tgt = int(item)
                if not 0 <= tgt < n_devices:
                    raise ValueError(f"STEPS[{g}]: index {tgt} out of range 0..{n_devices - 1}")
            targets.append(tgt)
        dupes = {t for t in targets if targets.count(t) > 1}
        if dupes:
            print(f"[WARNING] STEPS[{g}]: device(s) {sorted(dupes)} listed more than once in the same step; "
                  f"they will be flipped once.")
            targets = list(dict.fromkeys(targets))
        parsed.append((targets, edges))
    return parsed


def draw_island(ax, x, y, angle_deg, length, color="0.82", lw=5.0, alpha=1.0, zorder=1):
    t = np.deg2rad(angle_deg)
    dx, dy = 0.5 * length * np.cos(t), 0.5 * length * np.sin(t)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=lw, alpha=alpha,
            solid_capstyle="round", zorder=zorder)


def render(centers, geom_angles, lengths, mag0, steps, out_dir=OUTPUT_DIR):
    parsed = parse_steps(steps, len(centers))
    n = len(centers)
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cmap = plt.get_cmap(COLORMAP)
    denom = max(1, len(parsed) - 1)
    margin = float(np.max(lengths)) * 2.3
    xlim = (np.nanmin(centers[:, 0]) - margin, np.nanmax(centers[:, 0]) + margin)
    ylim = (np.nanmin(centers[:, 1]) - margin, np.nanmax(centers[:, 1]) + margin)
    arrow_len = float(np.median(lengths)) * 0.6

    def base(ax):
        for i in range(n):
            draw_island(ax, centers[i, 0], centers[i, 1], geom_angles[i], lengths[i],
                        lw=max(3.0, lengths[i] / 14), alpha=0.8)
            if SHOW_INDEX_LABELS:
                ax.text(centers[i, 0] + 6, centers[i, 1] - 14, str(i), fontsize=6.5, color="0.35", zorder=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x position (nm)")
        ax.set_ylabel("y position (nm)")

    mag = mag0.copy()
    flip_gens = {}
    frame_paths = []
    for g, (targets, edges) in enumerate(parsed):
        for t in targets:
            mag[t] = (mag[t] + 180.0) % 360.0
            flip_gens.setdefault(t, []).append(g)

        fig, ax = plt.subplots(figsize=(9, 9))
        base(ax)
        for pg in range(g):  # earlier flips, faded
            for t in parsed[pg][0]:
                draw_island(ax, centers[t, 0], centers[t, 1], geom_angles[t], lengths[t],
                            color=cmap(pg / denom), lw=6.0, alpha=0.35, zorder=3)
        for t in targets:  # this step's flips, full color
            draw_island(ax, centers[t, 0], centers[t, 1], geom_angles[t], lengths[t],
                        color=cmap(g / denom), lw=9.0, zorder=5)
            ax.scatter([centers[t, 0]], [centers[t, 1]], s=170, color=cmap(g / denom),
                       edgecolor="black", zorder=6)
        if SHOW_SOURCE_ARROWS:
            for src, tgt in edges:
                ax.scatter([centers[src, 0]], [centers[src, 1]], s=420, facecolors="none",
                           edgecolors="orange", linewidths=2.5, zorder=7)
                ax.annotate("", xy=tuple(centers[tgt]), xytext=tuple(centers[src]), zorder=8,
                            arrowprops=dict(arrowstyle="->", linewidth=2.4, color="black",
                                            alpha=0.9, shrinkA=16, shrinkB=16))
        if SHOW_MAG_ARROWS:
            th = np.deg2rad(mag)
            ax.quiver(centers[:, 0], centers[:, 1], arrow_len * np.cos(th), arrow_len * np.sin(th),
                      angles="xy", scale_units="xy", scale=1, color="black", width=0.004,
                      headwidth=4, headlength=5, headaxislength=4.5, alpha=0.75, zorder=10)
        label = "injections" if g == 0 else f"step {g}"
        ax.set_title(f"Generation {g} ({label}) | flipped: {sorted(targets)}\n"
                     f"{len(flip_gens)} device(s) flipped so far")
        fig.tight_layout()
        path = os.path.join(frames_dir, f"gen_{g:03d}.png")
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
        frame_paths.append(path)

    # summary: generation each device flipped in
    fig, ax = plt.subplots(figsize=(9, 9))
    base(ax)
    for t, gens in flip_gens.items():
        draw_island(ax, centers[t, 0], centers[t, 1], geom_angles[t], lengths[t],
                    color=cmap(gens[0] / denom), lw=8.0, zorder=5)
        ax.text(centers[t, 0], centers[t, 1], "/".join(map(str, gens)), ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="circle,pad=0.2", facecolor="white", edgecolor="black", alpha=0.9))
    ax.set_title("Generation in which each device flipped (0 = injection)")
    fig.tight_layout()
    summary_path = os.path.join(out_dir, "summary.png")
    fig.savefig(summary_path, dpi=DPI)
    plt.close(fig)

    # flip log
    log_path = os.path.join(out_dir, "flip_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generation", "device", "source"])
        for g, (targets, edges) in enumerate(parsed):
            src_of = {t: s for s, t in edges}
            for t in targets:
                w.writerow([g, t, src_of.get(t, "injection" if g == 0 else "")])

    gif_path = ""
    if MAKE_GIF and len(frame_paths) >= 2:
        try:
            from PIL import Image
            imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in frame_paths]
            gif_path = os.path.join(out_dir, "propagation.gif")
            imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=GIF_FRAME_MS, loop=0)
        except ImportError:
            print("[INFO] Pillow not installed; skipping GIF.")

    print(f"frames: {frames_dir}\nsummary: {summary_path}\nlog: {log_path}\ngif: {gif_path or '(none)'}")
    return frame_paths


if __name__ == "__main__":
    centers, geom_angles, lengths, mag0 = build_array()
    print(f"geometry built: {len(centers)} devices")
    render(centers, geom_angles, lengths, mag0, STEPS)