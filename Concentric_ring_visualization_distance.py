# -*- coding: utf-8 -*-
"""
Created on Fri May 30 11:27:46 2025

@author: jec822
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_activity_by_distance(
    signals,                      # 2D array: (n_cells, n_timepoints)
    distances,                    # 1D array: (n_cells,)
    distance_bins=None,          # Array of distance bin edges (optional)
    time_bins=None,              # Array of time bin edges (optional)
    ring_width=50,               # Used only if distance_bins is None
    time_bin_size=1,             # Used only if time_bins is None
    colormap='viridis',          # Colormap for the heatmap
    vmin=None,                   # Color scale min
    vmax=None,                   # Color scale max
    figsize=(6, 10),             # NEW: Figure size
    normalize=False,             # NEW: Normalize averaged traces to max=1
    return_data=False            # If True, return binned data instead of plotting
):
    # Validate input
    n_cells, n_timepoints = signals.shape
    assert len(distances) == n_cells, "Mismatch between number of cells and distances"

    # Handle distance bins
    if distance_bins is None:
        max_distance = np.nanmax(distances)
        distance_bins = np.arange(0, max_distance + ring_width, ring_width)
    else:
        distance_bins = np.asarray(distance_bins)

    # Handle time bins
    if time_bins is None:
        time_bins = np.arange(0, n_timepoints + time_bin_size, time_bin_size)
    else:
        time_bins = np.asarray(time_bins)

    n_distance_bins = len(distance_bins) - 1
    n_time_bins = len(time_bins) - 1
    binned_activity = np.full((n_distance_bins, n_time_bins), np.nan)

    for i in range(n_distance_bins):
        in_bin = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
        if not np.any(in_bin):
            continue
        cell_signals = signals[in_bin]

        for j in range(n_time_bins):
            start, end = time_bins[j], time_bins[j + 1]
            if end > n_timepoints:
                continue
            time_slice = cell_signals[:, start:end]
            avg_trace = np.nanmean(time_slice, axis=1)  # average within each cell
            mean_val = np.nanmean(avg_trace)            # then across cells
            binned_activity[i, j] = mean_val

    # Normalize each distance-bin trace (i.e., each row)
    if normalize:
        row_max = np.nanmax(binned_activity, axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            binned_activity = binned_activity / row_max

    if return_data:
        return binned_activity, distance_bins, time_bins

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        binned_activity,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        xticklabels=[f"{int(time_bins[i])}-{int(time_bins[i+1])}" for i in range(n_time_bins)],
        yticklabels=[f"{int(distance_bins[i])}-{int(distance_bins[i+1])}" for i in range(n_distance_bins)],
        cbar_kws={'label': 'Mean Activity (normalized)' if normalize else 'Mean Activity'}
    )
    plt.xlabel("Time (bins)")
    plt.ylabel("Distance from stimulation site (μm)")
    plt.title("Average Activity by Distance and Time")
    plt.tight_layout()
    plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

def animate_activity_by_distance(
    signals,                      # 2D array: (n_cells, n_timepoints)
    distances,                    # 1D array: (n_cells,)
    distance_bins=None,          # Array of distance bin edges (optional)
    time_bins=None,              # Array of time bin edges (optional)
    ring_width=50,               # Used only if distance_bins is None
    time_bin_size=10,            # Time step between frames
    colormap='viridis',          # Colormap for the heatmap
    vmin=None,                   # Color scale min
    vmax=None,                   # Color scale max
    figsize=(6, 6),              # Figure size
    normalize=False,             # Normalize spatial profiles
    interval=500,                # Interval between frames in ms
    save_path=None,             # Optional path to save animation (e.g., 'animation.mp4')
    time_labels=None               
):
    n_cells, n_timepoints = signals.shape
    assert len(distances) == n_cells, "Mismatch between number of cells and distances"

    # Handle distance bins
    if distance_bins is None:
        max_distance = np.nanmax(distances)
        distance_bins = np.arange(0, max_distance + ring_width, ring_width)
    else:
        distance_bins = np.asarray(distance_bins)

    n_distance_bins = len(distance_bins) - 1

    # Prepare output array: [n_distance_bins x n_timepoints]
    distance_profiles = np.full((n_distance_bins, n_timepoints), np.nan)

    # Compute the average signal per distance bin over all time
    for i in range(n_distance_bins):
        in_bin = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
        if not np.any(in_bin):
            continue
        cell_signals = signals[in_bin]
        distance_profiles[i, :] = np.nanmean(cell_signals, axis=0)

    # Normalize if requested
    if normalize:
        row_max = np.nanmax(distance_profiles, axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            distance_profiles = distance_profiles / row_max

    # Time binning setup for frames
    if time_bins is None:
        time_bins = np.arange(0, n_timepoints + 1, time_bin_size).astype(int)
    else:
        time_bins = np.asarray(time_bins).astype(int)

    n_frames = len(time_bins) - 1

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    heat = ax.imshow(
        np.zeros((n_distance_bins, 1)),
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        origin='lower'
    )
    cbar = plt.colorbar(heat, ax=ax, label='Activity (normalized)' if normalize else 'Activity')
    ax.set_xlabel("Time bin")
    ax.set_ylabel("Distance bin (μm)")
    yticklabels = [f"{int(distance_bins[i])}-{int(distance_bins[i + 1])}" for i in range(n_distance_bins)]
    ax.set_yticks(np.arange(n_distance_bins))
    ax.set_yticklabels(yticklabels)

    title = ax.set_title("")

    def update(frame_idx):
        start, end = time_bins[frame_idx], time_bins[frame_idx + 1]
        time_slice = distance_profiles[:, start:end]
        frame_data = np.nanmean(time_slice, axis=1, keepdims=True)  # Shape: (n_distance_bins, 1)
        heat.set_data(frame_data)
        
        # Use time_labels if provided, otherwise use frame index
        if time_labels is not None and len(time_labels) == len(time_bins):
            title.set_text(f"Time: {time_labels[frame_idx]}–{time_labels[frame_idx + 1]}")
        else:
            title.set_text(f"Frame: {start}–{end}")
        
        return heat, title

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        ani.save(save_path, dpi=150)
    else:
        plt.show()

    return ani  # Return animation object
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.animation import FuncAnimation

def animate_activity_concentric(
    signals,
    distances,
    distance_bins=None,
    time_bins=None,
    ring_width=50,
    time_bin_size=10,
    colormap='viridis',
    vmin=None,
    vmax=None,
    figsize=(6, 6),
    normalize=False,
    interval=200,
    save_path=None,
    show_labels=False,
    show_dividers=False,
    time_labels=None
):
    n_cells, n_timepoints = signals.shape
    assert len(distances) == n_cells, "Mismatch between number of cells and distances"

    # Handle distance bins
    if distance_bins is None:
        max_distance = np.nanmax(distances)
        distance_bins = np.arange(0, max_distance + ring_width, ring_width).astype(int)
    else:
        distance_bins = np.asarray(distance_bins).astype(int)

    n_distance_bins = len(distance_bins) - 1

    # Compute binned activity (rows = rings, cols = time)
    binned_activity = np.full((n_distance_bins, n_timepoints), np.nan)
    for i in range(n_distance_bins):
        in_bin = (distances >= distance_bins[i]) & (distances < distance_bins[i + 1])
        if not np.any(in_bin):
            continue
        cell_signals = signals[in_bin]
        binned_activity[i, :] = np.nanmean(cell_signals, axis=0)

    # Normalize each row (distance bin) if requested
    if normalize:
        row_max = np.nanmax(binned_activity, axis=1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            binned_activity = binned_activity / row_max

    # Handle time bins for animation frames
    if time_bins is None:
        time_bins = np.arange(0, n_timepoints + 1, time_bin_size).astype(int)
    else:
        time_bins = np.asarray(time_bins).astype(int)

    n_frames = len(time_bins) - 1

    # Prepare the figure
    fig, ax = plt.subplots(figsize=figsize)
    max_r = distance_bins[-1]
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_aspect('equal')
    ax.axis('off')

    # Pre-define Wedge patches
    rings = []
    for i in range(n_distance_bins):
        wedge = Wedge(
            center=(0, 0),
            r=distance_bins[i + 1],
            theta1=0,
            theta2=360,
            width=distance_bins[i + 1] - distance_bins[i],
            color='black'
        )
        ax.add_patch(wedge)
        rings.append(wedge)

    # Optional white dividers
    if show_dividers:
        for r in distance_bins[1:-1]:  # Skip inner and outermost
            circle = Circle((0, 0), r, color='white', fill=False, linewidth=1.0)
            ax.add_patch(circle)

    # Optional white text labels at bottom of rings
    label_texts = []
    if show_labels:
        for i in range(n_distance_bins):
            r = distance_bins[i + 1]  # upper limit
            text = ax.text(
                0, -r + 5, f"{r}",  # 5 units inward for readability
                color='white', ha='center', va='top', fontsize=6
            )
            label_texts.append(text)

    # Colormap and colorbar
    cmap = plt.get_cmap(colormap)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activity (normalized)' if normalize else 'Activity')

    title = ax.set_title("")

    def update(frame_idx):
        start, end = time_bins[frame_idx], time_bins[frame_idx + 1]
        time_slice = binned_activity[:, start:end]
        frame_vals = np.nanmean(time_slice, axis=1)

        for ring, val in zip(rings, frame_vals):
            ring.set_color(cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1)) if not np.isnan(val) else 'gray')

        # Use time_labels if provided, otherwise use frame index
        if time_labels is not None and len(time_labels) == len(time_bins):
            title.set_text(f"Time: {time_labels[frame_idx]}–{time_labels[frame_idx + 1]}")
        else:
            title.set_text(f"Frame: {start}–{end}")

        return rings

    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)

    if save_path:
        ani.save(save_path, dpi=150)
    else:
        plt.show()

    return ani


# %%
# signals: numpy array of shape (19000, 1000)
# distances: numpy array of shape (19000,)

distance_bins=np.linspace(25, 250, 10).astype(int)
time_bins=np.linspace(100, 350, 21).astype(int)

signals=stim_cells_results_V1_avg['trig_dff_avgs']
distances=full_resp_stats['V1_stats']['dist_from_stim_um']

visualize_activity_by_distance(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=40, time_bin_size=20,vmin=0,vmax=0.7)

signals=stim_cells_results_M2_avg['trig_dff_avgs']
distances=full_resp_stats['M2_stats']['dist_from_stim_um']

visualize_activity_by_distance(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=40, time_bin_size=20,vmin=0,vmax=0.7)


# %%
# signals: numpy array of shape (19000, 1000)
# distances: numpy array of shape (19000,)

distance_bins=np.linspace(25, 250, 10).astype(int)
time_bins=np.linspace(100, 350, 25).astype(int)

signals=stim_cells_results_V1_avg_sig['trig_dff_avgs']
distances=full_resp_stats_sig['V1_stats']['dist_from_stim_um']
mask=full_resp_stats_sig['V1_stats']['is_sig']
distances = distances[mask == 1]

visualize_activity_by_distance(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=50, time_bin_size=20,vmin=0,vmax=1,normalize=True)

signals=stim_cells_results_M2_avg_sig['trig_dff_avgs']
distances=full_resp_stats_sig['M2_stats']['dist_from_stim_um']
mask=full_resp_stats_sig['M2_stats']['is_sig']
distances = distances[mask == 1]

visualize_activity_by_distance(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=50, time_bin_size=20,vmin=0,vmax=1,normalize=True)

# %%
distance_bins=np.linspace(25, 250, 10).astype(int)
time_bins=np.linspace(100, 350, 25).astype(int)
time_labels = np.round(time_bins * 0.033, 1)

signals=stim_cells_results_V1_avg_sig['trig_dff_avgs']
distances=full_resp_stats_sig['V1_stats']['dist_from_stim_um']
mask=full_resp_stats_sig['V1_stats']['is_sig']
distances = distances[mask == 1]

animate_activity_concentric(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=50, time_bin_size=20,vmin=1,vmax=2,normalize=False,interval=300,show_labels=True,show_dividers=True,time_labels=time_labels,save_path=None)
# %%
signals=stim_cells_results_M2_avg_sig['trig_dff_avgs']
distances=full_resp_stats_sig['M2_stats']['dist_from_stim_um']
mask=full_resp_stats_sig['M2_stats']['is_sig']
distances = distances[mask == 1]

animate_activity_concentric(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=50, time_bin_size=20,vmin=1,vmax=2,normalize=False,interval=300,show_labels=True,show_dividers=True,time_labels=time_labels,save_path=None)
# %%
distance_bins=np.linspace(25, 250, 10).astype(int)
time_bins=np.linspace(100, 350, 25).astype(int)
time_labels = np.round(time_bins * 0.033, 1)

signals=stim_cells_results_V1_avg_sig['trig_dff_avgs']
distances=full_resp_stats_sig['V1_stats']['dist_from_stim_um']
mask=full_resp_stats_sig['V1_stats']['is_sig']
distances = distances[mask == 1]

animate_activity_by_distance(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=50, time_bin_size=20,vmin=1,vmax=2,normalize=False,interval=300,time_labels=time_labels,save_path=None)

signals=stim_cells_results_M2_avg_sig['trig_dff_avgs']
distances=full_resp_stats_sig['M2_stats']['dist_from_stim_um']
mask=full_resp_stats_sig['M2_stats']['is_sig']
distances = distances[mask == 1]

animate_activity_by_distance(signals, distances,distance_bins=distance_bins,time_bins=time_bins, ring_width=50, time_bin_size=20,vmin=1,vmax=2,normalize=False,interval=300,time_labels=time_labels,save_path=None)
