# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:56:56 2025

@author: jec822
"""

# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Illustrator-compatible settings
mpl.rcParams['pdf.fonttype'] = 42  # Embed fonts as TrueType
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'  # Editable standard font

# Extract trial data
trial_projs_basel = v1_pca_results['single_expt_results'][0]['trial_projs_basel'][4]
trial_projs_resp = v1_pca_results['single_expt_results'][0]['trial_projs_resp'][4]

# Stack and average across trials
a_stack = np.stack(trial_projs_basel, axis=0)
b_stack = np.stack(trial_projs_resp, axis=0)
a_mean = np.mean(a_stack, axis=0)
b_mean = np.mean(b_stack, axis=0)

# Transparency progression
t_basel = np.linspace(0.2, 1.0, a_mean.shape[0])
t_resp = np.linspace(0.2, 1.0, b_mean.shape[0])

# Plotting
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Baseline scatter
for i in range(a_mean.shape[0]):
    ax.scatter(a_mean[i, 0], a_mean[i, 1], a_mean[i, 2],
               color='blue', alpha=t_basel[i], s=35)

# Response scatter
for i in range(b_mean.shape[0]):
    ax.scatter(b_mean[i, 0], b_mean[i, 1], b_mean[i, 2],
               color='red', alpha=t_resp[i], s=35)

# Dashed stim transition
ax.plot([a_mean[-1, 0], b_mean[0, 0]],
        [a_mean[-1, 1], b_mean[0, 1]],
        [a_mean[-1, 2], b_mean[0, 2]],
        color='gray', linestyle='--', linewidth=2)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Baseline',
           markerfacecolor='blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Post-stim',
           markerfacecolor='red', markersize=8),
    Line2D([0], [0], linestyle='--', color='gray', label='Stim period')
]
ax.legend(handles=legend_elements, fontsize=12)

# Axis labels and title
ax.set_xlabel('PC1', fontsize=14)
ax.set_ylabel('PC2', fontsize=14)
ax.set_zlabel('PC3', fontsize=14)
ax.set_title('Average PCA Trajectories', fontsize=16)

# Export as vector graphic
plt.savefig("pca_trajectory_illustrator_ready.pdf", bbox_inches='tight')
plt.show()
        
# %%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Illustrator-compatible settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# Output folder
output_dir = "pca_trajectory_plots_V1"
os.makedirs(output_dir, exist_ok=True)

# Iterate through all experiments (i) and projection indices (j)
for i, expt in enumerate(v1_pca_results['single_expt_results']):
    num_indices = len(expt['trial_projs_basel'])  # assumes both 'basel' and 'resp' have same j length
    for j in range(num_indices):
        trial_projs_basel = expt['trial_projs_basel'][j]
        trial_projs_resp = expt['trial_projs_resp'][j]

        if not trial_projs_basel or not trial_projs_resp:
            continue  # skip if data missing

        try:
            a_stack = np.stack(trial_projs_basel, axis=0)
            b_stack = np.stack(trial_projs_resp, axis=0)
        except ValueError:
            continue  # skip if shape mismatch

        a_mean = np.mean(a_stack, axis=0)
        b_mean = np.mean(b_stack, axis=0)

        t_basel = np.linspace(0.2, 1.0, a_mean.shape[0])
        t_resp = np.linspace(0.2, 1.0, b_mean.shape[0])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for k in range(a_mean.shape[0]):
            ax.scatter(a_mean[k, 0], a_mean[k, 1], a_mean[k, 2],
                       color='blue', alpha=t_basel[k], s=35)
        for k in range(b_mean.shape[0]):
            ax.scatter(b_mean[k, 0], b_mean[k, 1], b_mean[k, 2],
                       color='red', alpha=t_resp[k], s=35)

        ax.plot([a_mean[-1, 0], b_mean[0, 0]],
                [a_mean[-1, 1], b_mean[0, 1]],
                [a_mean[-1, 2], b_mean[0, 2]],
                color='gray', linestyle='--', linewidth=2)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Baseline',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Post-stim',
                   markerfacecolor='red', markersize=8),
            Line2D([0], [0], linestyle='--', color='gray', label='Stim period')
        ]
        ax.legend(handles=legend_elements, fontsize=12)

        ax.set_xlabel('PC1', fontsize=14)
        ax.set_ylabel('PC2', fontsize=14)
        ax.set_zlabel('PC3', fontsize=14)
        ax.set_title(f'PCA Trajectory (Expt {i}, Index {j})', fontsize=16)

        filename = f"pca_traj_expt{i}_index{j}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()
# %%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Illustrator-compatible settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# Output folder
output_dir = "pca_trajectory_plots_m2"
os.makedirs(output_dir, exist_ok=True)

# Iterate through all experiments (i) and projection indices (j)
for i, expt in enumerate(m2_pca_results['single_expt_results']):
    num_indices = len(expt['trial_projs_basel'])  # assumes both 'basel' and 'resp' have same j length
    for j in range(num_indices):
        trial_projs_basel = expt['trial_projs_basel'][j]
        trial_projs_resp = expt['trial_projs_resp'][j]

        if not trial_projs_basel or not trial_projs_resp:
            continue  # skip if data missing

        try:
            a_stack = np.stack(trial_projs_basel, axis=0)
            b_stack = np.stack(trial_projs_resp, axis=0)
        except ValueError:
            continue  # skip if shape mismatch

        a_mean = np.mean(a_stack, axis=0)
        b_mean = np.mean(b_stack, axis=0)

        t_basel = np.linspace(0.2, 1.0, a_mean.shape[0])
        t_resp = np.linspace(0.2, 1.0, b_mean.shape[0])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for k in range(a_mean.shape[0]):
            ax.scatter(a_mean[k, 0], a_mean[k, 1], a_mean[k, 2],
                       color='blue', alpha=t_basel[k], s=35)
        for k in range(b_mean.shape[0]):
            ax.scatter(b_mean[k, 0], b_mean[k, 1], b_mean[k, 2],
                       color='red', alpha=t_resp[k], s=35)

        ax.plot([a_mean[-1, 0], b_mean[0, 0]],
                [a_mean[-1, 1], b_mean[0, 1]],
                [a_mean[-1, 2], b_mean[0, 2]],
                color='gray', linestyle='--', linewidth=2)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Baseline',
                   markerfacecolor='blue', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Post-stim',
                   markerfacecolor='red', markersize=8),
            Line2D([0], [0], linestyle='--', color='gray', label='Stim period')
        ]
        ax.legend(handles=legend_elements, fontsize=12)

        ax.set_xlabel('PC1', fontsize=14)
        ax.set_ylabel('PC2', fontsize=14)
        ax.set_zlabel('PC3', fontsize=14)
        ax.set_title(f'PCA Trajectory (Expt {i}, Index {j})', fontsize=16)

        filename = f"pca_traj_expt{i}_index{j}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()
# %%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Illustrator-compatible settings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# Settings
output_dir = "m2_pca_trial_trajectory_png"
plot_average = True  # Toggle for average overlay
os.makedirs(output_dir, exist_ok=True)

for i, expt in enumerate(m2_pca_results['single_expt_results']):
    num_indices = len(expt['trial_projs_basel'])
    for j in range(num_indices):
        trial_projs_basel = expt['trial_projs_basel'][j]
        trial_projs_resp = expt['trial_projs_resp'][j]

        if not trial_projs_basel or not trial_projs_resp:
            continue

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        num_trials = len(trial_projs_basel)
        cmap = plt.get_cmap('tab20')  # or 'nipy_spectral', 'hsv', etc.

        # Plot baseline and response trials with different colors
        for idx, (basel_trial, resp_trial) in enumerate(zip(trial_projs_basel, trial_projs_resp)):
            color = cmap(idx % cmap.N)
            if basel_trial.shape[1] >= 3:
                ax.plot(basel_trial[:, 0], basel_trial[:, 1], basel_trial[:, 2],
                        color=color, alpha=0.8, linewidth=1.5)
            if resp_trial.shape[1] >= 3:
                ax.plot(resp_trial[:, 0], resp_trial[:, 1], resp_trial[:, 2],
                        color=color, alpha=0.8, linewidth=1.5, linestyle='--')

        # Optional average overlay
        if plot_average:
            try:
                a_stack = np.stack(trial_projs_basel, axis=0)
                b_stack = np.stack(trial_projs_resp, axis=0)
                a_mean = np.mean(a_stack, axis=0)
                b_mean = np.mean(b_stack, axis=0)

                ax.plot(a_mean[:, 0], a_mean[:, 1], a_mean[:, 2],
                        color='black', linewidth=3, label='Mean Baseline')
                ax.plot(b_mean[:, 0], b_mean[:, 1], b_mean[:, 2],
                        color='gray', linewidth=3, label='Mean Response')

                ax.plot([a_mean[-1, 0], b_mean[0, 0]],
                        [a_mean[-1, 1], b_mean[0, 1]],
                        [a_mean[-1, 2], b_mean[0, 2]],
                        color='black', linestyle='--', linewidth=2)
            except Exception as e:
                print(f"Error computing mean for expt {i}, index {j}: {e}")

        # Axis labels and title
        ax.set_xlabel('PC1', fontsize=14)
        ax.set_ylabel('PC2', fontsize=14)
        ax.set_zlabel('PC3', fontsize=14)
        ax.set_title(f'Trial Trajectories (Expt {i}, Index {j})', fontsize=16)

        # Save figure
        filename = f"pca_trial_lines_expt{i}_index{j}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()





# %%
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.signal import medfilt

# Illustrator-compatible settings
mpl.rcParams['pdf.fonttype'] = 42 # To make text editable
mpl.rcParams['ps.fonttype']  = 42 # To make text editable
mpl.rcParams['svg.fonttype'] = 'none' # To make text editable

def smooth_trajectory_median(traj, kernel_size=5):
    return np.stack([medfilt(traj[:, i], kernel_size=kernel_size) for i in range(traj.shape[1])], axis=1)

def plot_single_trial_trajectory(
    m2_pca_results, i, j, output_dir="m2_pca_trial_trajectory_svg",
    plot_average=True, trial_indices=None, stride=1, skip=1,
    plot_style='solid', line_width=1.5, stim_period=False,
    smoothing_method='median', marker_same_color_as_line=False,
    baseline_as_centroid=False, join=False
):
    os.makedirs(output_dir, exist_ok=True)

    expt = m2_pca_results['single_expt_results'][i]
    trial_projs_basel = expt['trial_projs_basel'][j]
    trial_projs_resp = expt['trial_projs_resp'][j]

    if not trial_projs_basel or not trial_projs_resp:
        print(f"Empty data at expt {i}, index {j}")
        return

    if trial_indices is not None:
        trial_projs_basel = [trial_projs_basel[k] for k in trial_indices]
        trial_projs_resp = [trial_projs_resp[k] for k in trial_indices]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab20')

    for idx, (basel_trial, resp_trial) in enumerate(zip(trial_projs_basel, trial_projs_resp)):
        color = cmap(idx % cmap.N)

        if smoothing_method == 'median':
            basel_trial = smooth_trajectory_median(basel_trial, kernel_size=stride)
            resp_trial = smooth_trajectory_median(resp_trial, kernel_size=stride)

        basel_trial = basel_trial[::skip]
        resp_trial = resp_trial[::skip]

        linestyle = '--' if plot_style == 'dashed' else '-'

        if baseline_as_centroid:
            centroid = np.mean(basel_trial, axis=0)
            combined = np.vstack([centroid[np.newaxis, :], resp_trial])
            ax.plot(combined[:, 0], combined[:, 1], combined[:, 2],
                    color=color, alpha=0.8, linewidth=line_width, linestyle=linestyle)
        elif plot_style == 'scatter':
            ax.scatter(basel_trial[:, 0], basel_trial[:, 1], basel_trial[:, 2], color=color, alpha=0.8)
            ax.scatter(resp_trial[:, 0], resp_trial[:, 1], resp_trial[:, 2], color=color, alpha=0.8, marker='x')
        else:
            ax.plot(basel_trial[:, 0], basel_trial[:, 1], basel_trial[:, 2],
                    color=color, alpha=0.8, linewidth=line_width, linestyle=linestyle)
            ax.plot(resp_trial[:, 0], resp_trial[:, 1], resp_trial[:, 2],
                    color=color, alpha=0.8, linewidth=line_width, linestyle=linestyle)

        marker_color = color if marker_same_color_as_line else '#444444'
        if baseline_as_centroid:
            ax.scatter(centroid[0], centroid[1], centroid[2],
                       color=marker_color, s=100, marker='^', alpha=0.9, edgecolor='black')
        else:
            ax.scatter(basel_trial[0, 0], basel_trial[0, 1], basel_trial[0, 2],
                       color=marker_color, s=30, marker='^', alpha=0.7)
            ax.scatter(basel_trial[-1, 0], basel_trial[-1, 1], basel_trial[-1, 2],
                       color=marker_color, s=30, marker='o', alpha=0.7)

        ax.scatter(resp_trial[-1, 0], resp_trial[-1, 1], resp_trial[-1, 2],
                   color=marker_color, s=30, marker='s', alpha=0.7)

        if join and not baseline_as_centroid:
            ax.plot([basel_trial[-1, 0], resp_trial[0, 0]],
                    [basel_trial[-1, 1], resp_trial[0, 1]],
                    [basel_trial[-1, 2], resp_trial[0, 2]],
                    color=color, linestyle=linestyle, linewidth=line_width)

    if plot_average:
        try:
            a_stack = np.stack(trial_projs_basel, axis=0)
            b_stack = np.stack(trial_projs_resp, axis=0)
            a_mean = np.mean(a_stack, axis=0)
            b_mean = np.mean(b_stack, axis=0)

            if smoothing_method == 'median':
                a_mean = smooth_trajectory_median(a_mean, kernel_size=stride)
                b_mean = smooth_trajectory_median(b_mean, kernel_size=stride)

            a_mean = a_mean[::skip]
            b_mean = b_mean[::skip]

            linestyle = '--' if plot_style == 'dashed' else '-'
            ax.plot(a_mean[:, 0], a_mean[:, 1], a_mean[:, 2],
                    color='black', linewidth=3, label='Mean Baseline', linestyle=linestyle)
            ax.plot(b_mean[:, 0], b_mean[:, 1], b_mean[:, 2],
                    color='gray', linewidth=3, label='Mean Response', linestyle=linestyle)

            ax.plot([a_mean[-1, 0], b_mean[0, 0]],
                    [a_mean[-1, 1], b_mean[0, 1]],
                    [a_mean[-1, 2], b_mean[0, 2]],
                    color='black', linestyle='--', linewidth=2)
        except Exception as e:
            print(f"Error computing mean for expt {i}, index {j}: {e}")

    if stim_period:
        try:
            stim_start, stim_end = 0, min(len(b_mean) - 1, 1)
            ax.plot([b_mean[stim_start, 0], b_mean[stim_end, 0]],
                    [b_mean[stim_start, 1], b_mean[stim_end, 1]],
                    [b_mean[stim_start, 2], b_mean[stim_end, 2]],
                    color='black', linestyle='--', linewidth=1.5, label='Stim. period')
        except Exception as e:
            print(f"Error plotting stim period for expt {i}, index {j}: {e}")

    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_zlabel('PC3', fontsize=14)
    ax.set_title(f'Trial Trajectories (Expt {i}, Index {j})', fontsize=16)

    legend_elements = [
        Line2D([0], [0], marker='^', color='#444444', label='Trial start', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='o', color='#444444', label='Stim start', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='s', color='#444444', label='Trial end', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='*', color='#444444', label='Baseline centroid', markersize=8, linestyle='None'),
        Line2D([0], [0], color='black', label='Stim. period', linestyle='--', linewidth=1)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

    filename = f"pca_trial_lines_expt{i}_index{j}.svg"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', format='svg')


# %% 
#  i=15,j=7, trial_indices=[4,3,1]
plot_single_trial_trajectory(
    m2_pca_results,
    i=15,
    j=7,
    output_dir="pca_trajectory_plots_svg",
    plot_average=False,
    trial_indices=[4,3,1],
    stride=15,
    plot_style='solid',  # options: 'scatter', 'dashed', 'solid'
    line_width=2.0,
    skip=3,
    stim_period=None,
    smoothing_method='median', # enable median filter smoothing
    join=True,         # Optional connection line
    baseline_as_centroid=True,
    marker_same_color_as_line=True  # Optional matching marker color
)

# %%
plot_single_trial_trajectory(
    v1_pca_results,
    i=18,
    j=4,
    output_dir="pca_trajectory_plots_svg",
    plot_average=False,
    trial_indices=[1,2,0],
    stride=9,
    plot_style='solid',  # options: 'scatter', 'dashed', 'solid'
    line_width=2.0,
    skip=1,
    stim_period=None,
    smoothing_method='median', # enable median filter smoothing
    join=True,         # Optional connection line
    baseline_as_centroid=True
    # marker_same_color_as_line=True  # Optional matching marker color
)
