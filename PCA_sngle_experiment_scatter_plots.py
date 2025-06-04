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

    
# %%
def plot_combined_scatter_by_significance_with_fitlines(
    v1_pca_results, save_path=None, equal_xy=False, fit_line=True,
    xlim=None, ylim=None
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from scipy.stats import pearsonr, linregress

    # Illustrator-compatible settings
    mpl.rcParams['pdf.fonttype'] = 42  # To make text editable
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    all_x = []
    all_y = []
    all_significant = []

    significant_fit_lines = []

    # Gather and process each experiment
    for result in v1_pca_results['single_expt_results']:
        x = np.array(result['basel_dists'])
        y = np.array(result['resp_dists'])

        if len(x) != len(y) or len(x) == 0:
            continue

        r, p = pearsonr(x, y)
        is_significant = p < 0.05

        all_x.extend(x)
        all_y.extend(y)
        all_significant.extend([is_significant] * len(x))

        if is_significant:
            slope, intercept, *_ = linregress(x, y)
            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = slope * x_fit + intercept
            significant_fit_lines.append((x_fit, y_fit))

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_significant = np.array(all_significant)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot points
    ax.scatter(all_x[~all_significant], all_y[~all_significant],
               color='lightgray', edgecolor='white', label='p ≥ 0.05', s=20)
    ax.scatter(all_x[all_significant], all_y[all_significant],
               color='crimson', edgecolor='white', label='p < 0.05', s=20)

    # Plot individual fit lines
    for x_fit, y_fit in significant_fit_lines:
        ax.plot(x_fit, y_fit, color='steelblue', linestyle='-', alpha=0.4, label='_nolegend_')

    # Overall regression line (optional)
    if fit_line and len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_line = np.linspace(np.min(all_x), np.max(all_x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='black', linestyle='--', label='Overall Fit')

        stats_text = f"r = {r_val:.2f}, p = {p_val:.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', ha='left')

    # Labels and formatting
    ax.set_xlabel('Baseline Distance')
    ax.set_ylabel('Response Distance')
    ax.set_title('Combined Scatter with Significant Fits')

    # Axis limits logic
    if equal_xy:
        min_val = min(np.min(all_x), np.min(all_y))
        max_val = max(np.max(all_x), np.max(all_y))
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
    else:
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    # Save SVG with editable text
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

# %%
def get_scaled_fonts(fig_width_inch, base_width=6.0, base_fontsize=12):
    """
    Returns a dictionary of font sizes scaled to the figure width.
    
    Parameters:
    - fig_width_inch: Width of the figure in inches
    - base_width: Reference width (in inches) that uses base_fontsize (default: 6.0 inches)
    - base_fontsize: Reference font size for base_width (default: 12 pt)

    Returns:
    - dict with font sizes for title, labels, ticks, legend, and text
    """
    scale = fig_width_inch / base_width
    return {
        'title': base_fontsize * 1.2 * scale,
        'label': base_fontsize * scale,
        'tick': base_fontsize * 0.9 * scale,
        'legend': base_fontsize * 0.9 * scale,
        'text': base_fontsize * scale
    }


# %%
# plot_combined_scatter_by_significance(
#     m2_pca_results,
#     save_path='combined_scatter.svg',
#     equal_xy=False,
#     fit_line=True
# )

plot_combined_scatter_by_significance_with_fitlines(
    m2_pca_results,
    save_path=os.path.join(output_folder, 'm2_combined_scatter.svg'),
    equal_xy=False,
    fit_line=True,
    xlim=(0,250),
    ylim=(0,1000)
)

# plot_combined_scatter_by_significance(
#     points,
#     save_path='combined_scatter.svg',
#     equal_xy=False,
#     fit_line=True
# )