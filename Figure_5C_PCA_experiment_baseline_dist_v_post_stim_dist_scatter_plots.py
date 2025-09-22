# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:33:56 2025

@author: jec822

"""

def plot_scatter_with_stats(x, y, title='', save_path=None, equal_xy=False, fit_line=True):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr, linregress

    # Calculate Pearson correlation
    r, p = pearsonr(x, y)

    # Create square plot
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x, y, color='black', s=20, edgecolor='white', label='Data')

    # Fit line if requested
    if fit_line:
        slope, intercept, _, _, _ = linregress(x, y)
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, color='red', linestyle='--', label='Fit')

    # Labels and title
    ax.set_xlabel('Baseline Distance')
    ax.set_ylabel('Response Distance')
    ax.set_title(title)
    # ax.axis('equal')

    # Equal XY axis scaling if requested
    if equal_xy:
        min_val = min(np.min(x), np.min(y))
        max_val = max(np.max(x), np.max(y))
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)

    # Display r and p
    stats_text = f"r = {r:.2f}, p = {p:.3f}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', ha='left')

    # Clean plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if fit_line:
        ax.legend(frameon=False)

    # Save if path is given
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
    # plt.close(fig)
# %%
# Create output folder
output_folder = "pca_scatter_plots"
os.makedirs(output_folder, exist_ok=True)
# %%
# Iterate through experiments
for i, result in enumerate(m2_pca_results['single_expt_results']):
    x = np.array(result['basel_dists'])
    y = np.array(result['resp_dists'])

    # Title and filename
    title = f'Experiment {i}'
    filename = f'experiment_{i}_scatter.svg'
    save_path = os.path.join(output_folder, filename)

    # Plot and save
    plot_scatter_with_stats(x, y, title=title, save_path=save_path)
# %%
# Iterate through experiments
for i,result in enumerate(points):
    x = np.array(result[0])
    y = np.array(result[1])

    # Title and filename
    title = f'Experiment {i}'
    filename = f'experiment_{i}_scatter.svg'
    save_path = os.path.join(output_folder, filename)

    # Plot and save
    plot_scatter_with_stats(x, y, title=title, save_path=save_path)

# %% Figure 5 C


# plot_combined_scatter_by_significance(
#     m2_pca_results,
#     save_path='combined_scatter.svg',
#     equal_xy=False,
#     fit_line=True
# )

PCA_functions.plot_combined_scatter_by_significance_with_fitlines(
    m2_pca_results,
    # save_path=os.path.join(output_folder, 'm2_combined_scatter.svg'),
    equal_xy=False,
    fit_line=True,
    xlim=(0,250),
    ylim=(0,600),
    p_values='by_stim_type'
)

PCA_functions.plot_combined_scatter_by_significance_with_fitlines(
    v1_pca_results,
    # save_path=os.path.join(output_folder, 'm2_combined_scatter.svg'),
    equal_xy=False,
    fit_line=True,
    xlim=(0,250),
    ylim=(0,600),
    p_values='by_stim_type'
)


# PCA_functions.plot_combined_scatter_by_significance_with_fitlines(
#     m2_pca_results_pseudo,
#     # save_path=os.path.join(output_folder, 'm2_combined_scatter.svg'),
#     equal_xy=False,
#     fit_line=True,
#     xlim=(0,250),
#     ylim=(0,600)
# )

# PCA_functions.plot_combined_scatter_by_significance_with_fitlines(
#     v1_pca_results_pseudo,
#     # save_path=os.path.join(output_folder, 'm2_combined_scatter.svg'),
#     equal_xy=False,
#     fit_line=True,
#     xlim=(0,250),
#     ylim=(0,600)
# )
# %%

PCA_functions.plot_combined_scatter_by_significance_with_fitlines(
    m2_pca_results_scramble,
    # save_path=os.path.join(output_folder, 'm2_combined_scatter.svg'),
    equal_xy=False,
    fit_line=True,
    xlim=(0,250),
    ylim=(0,600)
)

PCA_functions.plot_combined_scatter_by_significance_with_fitlines(
    v1_pca_results_scramble,
    # save_path=os.path.join(output_folder, 'm2_combined_scatter.svg'),
    equal_xy=False,
    fit_line=True,
    xlim=(0,250),
    ylim=(0,600)
)

# plot_combined_scatter_by_significance(
#     points,
#     save_path='combined_scatter.svg',
#     equal_xy=False,
#     fit_line=True
# )