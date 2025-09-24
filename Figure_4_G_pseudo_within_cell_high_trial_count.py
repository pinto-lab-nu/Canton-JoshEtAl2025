# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:42:45 2025

@author: jec822
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def plot_comparison_boxplots_from_2d_dfs(
    df1,
    df2,
    label1="Group 1",
    label2="Group 2",
    color1="#1f77b4",
    color2="#ff7f0e",
    ylabel="Correlation (r)",
    title="",
    show_significance=True,
    alpha_thresholds=[0.05, 0.01, 0.001],
    figsize=(10, 6)
):
    # Melt to long format for seaborn
    df1_long = df1.melt(var_name='Sample Size', value_name='Value')
    df1_long['Group'] = label1

    df2_long = df2.melt(var_name='Sample Size', value_name='Value')
    df2_long['Group'] = label2

    df_combined = pd.concat([df1_long, df2_long], ignore_index=True)

    plt.figure(figsize=figsize)
    sns.boxplot(data=df_combined, x='Sample Size', y='Value', hue='Group',
                palette={label1: color1, label2: color2})

    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title="")
    
    # Significance markers
    if show_significance:
        sample_sizes = df1.columns.intersection(df2.columns)
        y_max = df_combined['Value'].max()
        y_min = df_combined['Value'].min()
        y_range = y_max - y_min
        offset = y_range * 0.05

        for i, sample_size in enumerate(sample_sizes):
            vals1 = df1[sample_size].dropna()
            vals2 = df2[sample_size].dropna()

            if len(vals1) > 5 and len(vals2) > 5:
                p = mannwhitneyu(vals1, vals2, alternative='two-sided').pvalue
                if p < alpha_thresholds[2]:
                    marker = '***'
                elif p < alpha_thresholds[1]:
                    marker = '**'
                elif p < alpha_thresholds[0]:
                    marker = '*'
                else:
                    continue

                plt.text(i, y_max + offset, marker, ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

def plot_violin_comparison(
    df1,
    df2,
    group_labels=("Real", "Pseudo"),
    ylabel="R value",
    xlabel="Group",
    title="Comparison of Distributions",
    alpha_thresholds=[0.05, 0.01, 0.001],
    marker_y_offset=0.02,
    figsize=(4, 5),
    colors=("skyblue", "lightcoral"),
    significance_marker=True,
    show=True
):
    """
    Plots violin plots comparing the distribution of each column between df1 and df2.
    """
    # Melt the data
    df1_long = df1.melt(var_name="group", value_name="value")
    df1_long["source"] = group_labels[0]

    df2_long = df2.melt(var_name="group", value_name="value")
    df2_long["source"] = group_labels[1]

    full_df = pd.concat([df1_long, df2_long], ignore_index=True)

    plt.figure(figsize=figsize)
    ax = sns.violinplot(
        x="group", y="value", hue="source", data=full_df,
        palette=colors, cut=0, inner="box", linewidth=1,width=1,gap=0
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine()

    # Add significance markers
    if significance_marker:
        unique_groups = df1.columns
        y_max = full_df["value"].max()
        height_offset = y_max * marker_y_offset

        for i, group in enumerate(unique_groups):
            data1 = df1[group].dropna()
            data2 = df2[group].dropna()

            if len(data1) > 5 and len(data2) > 5:
                p = mannwhitneyu(data1, data2, alternative='two-sided').pvalue

                if p < alpha_thresholds[2]:
                    marker = "***"
                elif p < alpha_thresholds[1]:
                    marker = "**"
                elif p < alpha_thresholds[0]:
                    marker = "*"
                else:
                    continue

                x_pos = i  # same x as group label
                y_pos = max(data1.max(), data2.max()) + height_offset
                ax.text(x_pos, y_pos, marker, ha="center", va="bottom", fontsize=12)

    plt.legend(title="Source", loc="upper right")
    plt.tight_layout()
    if show:
        plt.show()

# %%

trial_data=result_M2_high_trial_count_non_stimd_filt

pseudo_dfs_iteration_M2=calculate_single_trial_features.generate_pseudo_trials_select_cells(
    roi_ids = np.array(trial_data['roi_id_extended_dataset']),
    stim_ids = np.array(trial_data['stim_id']),
    roi_keys_all = np.array(trial_data['roi_keys']),
    trials = trial_data['trials_arrays'],
    n_iters=1,
    baseline_window=(1000, 10000),
    trial_crop_idx=None,
    baseline_start_idx=0,
    max_offset=5000,
    zscore_to_baseline=True,
    baseline_window_for_z=80,
    repeat_count=1 #only one stim type
)
# %%

pseudo_results_multi_iter_M2 = {}

# Loop over all iterations in your pseudo dataframe dictionary
for iter_label, pseudo_trials_dffs_iter in pseudo_dfs_iteration_M2.items():

    # Run your processing function
    result_M2_pseudo= calculate_single_trial_features.process_and_filter_response_matrix_from_df(
        entire_df=pseudo_trials_dffs_iter,
        kernel_size=7,
        peak_thresh=2.96,
        group_threshold=2,
        time_variance='peak_time', # or False to use COM
        subsample=False,
        min_width_val=2
    )
    
    # Store the result DataFrame in the dictionary
    pseudo_results_multi_iter_M2[iter_label] =  result_M2_pseudo
    
    
# %% Features to filter responsive cells

# ---- Common Setup ----
features_to_track = [
    'peak_time_std', 'com_std', 'auc_avg', 'peak_array_mean_trial', 'peak_amp_avg',
    'peak_time_array_mean_trial', 'com_array_mean_trial', 'response_proportion',
    'com_calc_abs_diff_mean_trial', 'peak_time_calc_abs_diff_mean_trial'
]

# ---- Define filters ----
filter_steps = [

    ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0])
    ]
# %%

pseudo_results_filt_multi_iter_M2 = {}

# Loop over all iterations in your pseudo dataframe dictionary
for iter_label, pseudo_trials_dffs_iter in pseudo_results_multi_iter_M2.items():

    # Run your processing function
    result_M2_pseudo_filt, summaries_M2_pseudo = calculate_single_trial_features.filter_and_summarize(pseudo_trials_dffs_iter, pseudo_trials_dffs_iter, filter_steps, features_to_track, label="pseudo")

    
    # Store the result DataFrame in the dictionary
    pseudo_results_filt_multi_iter_M2[iter_label] =  result_M2_pseudo_filt


# %% Figure 4G

xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
    peak_or_com_series=result_M2_high_trial_count_non_stimd_filt['peak_time_by_trial'],
    metric='peak',
    n_iter=10
    # random_seed=42
)

analyzeEvoked2P.plot_trial_xval(area='M2', 
                                                     params=opto_params, 
                                                     expt_type='standard', 
                                                     resp_type='dff', 
                                                     signif_only=True, 
                                                     which_neurons='non_stimd', 
                                                     # axis_handle=row3_axs[0],
                                                     axis_handle=None,
                                                     # fig_handle=fig,
                                                     trial_data=None,
                                                     xval_results=xval_results)
       
       
# %%

xval_summary = []
iterations_xval = 1000

for i in range(iterations_xval):
    xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
        peak_or_com_series=result_M2_high_trial_count_non_stimd_filt['peak_time_by_trial'],
        metric='peak',
        n_iter=5,  # single iteration per call
        random_seed=None  # or set = i for reproducibility
    )

    # Extract r and r² values
    xval_summary.append({
        'iteration': i,
        'correlation_r': xval_results['correlation_r'],
        'correlation_r2': xval_results['correlation_r2']
    })

# Convert to DataFrame
xval_df_high_trial = pd.DataFrame(xval_summary)
# %%

xval_summary = []
iterations_xval = 1000

for i in range(iterations_xval):
    xval_results = calculate_cross_corr.crossval_peak_timing_shuffle(
        peak_or_com_series=result_M2_pseudo_filt['peak_time_by_trial'],
        metric='peak',
        n_iter=5,  # single iteration per call
        random_seed=None  # or set = i for reproducibility
    )

    # Extract r and r² values
    xval_summary.append({
        'iteration': i,
        'correlation_r': xval_results['correlation_r'],
        'correlation_r2': xval_results['correlation_r2']
    })

# Convert to DataFrame
xval_df = pd.DataFrame(xval_summary)

# %%

analysis_plotting_functions.plot_ecdf_comparison(xval_df['correlation_r'],xval_df_high_trial['correlation_r'],
                     label1='"MOs high trial count pseudo_data', label2="MOs high trial count data",
                     line_color1='k',line_color2='k',
                     xlabel="r values",
                     ylabel="proportion of iterations",
                     title='',
                     stat_test='auto',
                     figsize=[4,5],
                     # xticks_start=0, xticks_end=25, xticks_step=5,
                     # yticks_start=0, yticks_end=1, yticks_step=0.5,
                     # show_normality_pvals=True,
                     )

# %%
n_iter_list=[5]
# n_iter_list=[1, 5, 10, 50]

xval_nested_dfs_high_trial = calculate_cross_corr.run_xval_across_iterations(
    series_data=result_M2_high_trial_count_non_stimd_filt['peak_time_by_trial'],
    n_iter_list=n_iter_list,
    iter_total=1000,
    metric='peak',
    base_seed=42,
    verbose=True
)
# 

xval_nested_dfs = calculate_cross_corr.run_xval_across_iterations(
    series_data=result_M2_pseudo_filt['peak_time_by_trial'],
    n_iter_list=n_iter_list,
    iter_total=1000,
    metric='peak',
    base_seed=42,
    verbose=True
)

# %%

r_matrix_df_real = pd.DataFrame({n_iter: df['correlation_r'].values for n_iter, df in xval_nested_dfs_high_trial.items()})

r_matrix_df_pseudo = pd.DataFrame({n_iter: df['correlation_r'].values for n_iter, df in xval_nested_dfs.items()})


distances = list(r_matrix_df_real.columns)
distances_scaled = [val / max(distances) for val in distances]


# %% Figure 4H

analysis_plotting_functions.plot_ecdf_comparison(r_matrix_df_real[5],r_matrix_df_pseudo[5],
                     label1='"MOs high trial count pseudo_data', label2="MOs high trial count data",
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="r values",
                     ylabel="proportion of iterations",
                     title='',
                     stat_test='auto',
                     figsize=[3,4],
                     # xticks_start=0, xticks_end=25, xticks_step=5,
                     # yticks_start=0, yticks_end=1, yticks_step=0.5,
                     show_normality_pvals=True,
                     )

