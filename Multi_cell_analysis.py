# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 22:47:34 2025

@author: jec822
"""

# %%
def reorder_and_rename_stim_columns(df, custom_order, label_map):
    """
    Reorders and renames stim columns in a DataFrame based on a custom order and label map.

    Parameters:
    - df: pandas DataFrame with stim_id columns.
    - custom_order: List of stim_ids specifying the desired column order.
    - label_map: Dict mapping stim_id to desired column label.

    Returns:
    - A new DataFrame with reordered and renamed columns.
    """
    existing_cols = [col for col in custom_order if col in df.columns]
    df_reordered = df[existing_cols]
    df_renamed = df_reordered.rename(columns=label_map)
    return df_renamed

# %%
import pandas as pd
import numpy as np

def get_sorted_peak_time_2d(df, value_col='peak_time_avg', group_col='stim_id'):
    """
    Create a 2D DataFrame where each column corresponds to a group (e.g., stim_id),
    containing the sorted values (e.g., peak_time_avg), padded with NaNs.

    Parameters:
    - df: Input DataFrame containing the data
    - value_col: Column name with the values to sort (default: 'peak_time_avg')
    - group_col: Column name to group by (default: 'stim_id')

    Returns:
    - A 2D pandas DataFrame with sorted values per group, padded with NaNs
    """
    grouped = df.groupby(group_col)[value_col].apply(lambda x: x.sort_values().reset_index(drop=True))
    max_len = grouped.groupby(level=0).size().max()

    result_df = pd.DataFrame({
        group: values.values.tolist() + [np.nan] * (max_len - len(values))
        for group, values in grouped.groupby(level=0)
    })

    result_df.columns.name = group_col
    return result_df

# %%
import pandas as pd

def calculate_proportion_above_threshold(df_cells, df_summary, feature_col, threshold, direction='>'):
    """
    Calculate the proportion of responsive cells above/below a threshold per group.

    Parameters:
    - df_cells: DataFrame with individual cell data.
    - df_summary: DataFrame with 'cells_per_roi' and group columns.
    - feature_col: str, name of the feature column to threshold.
    - threshold: numeric, value to threshold against.
    - direction: str, either '>' or '<' to filter cells.

    Returns:
    - DataFrame with proportion of cells per group that pass the threshold.
    """
    group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

    # Filter cells
    if direction == '>':
        df_filtered = df_cells[df_cells[feature_col] > threshold]
    elif direction == '<':
        df_filtered = df_cells[df_cells[feature_col] < threshold]
    else:
        raise ValueError("`direction` must be '>' or '<'.")

    # Count passing cells per group
    passing_counts = df_filtered.groupby(group_cols).size().reset_index(name='passing_cells')

    # Merge with summary to get total cells per ROI
    merged = pd.merge(df_summary, passing_counts, on=group_cols, how='left')

    # Fill NaN (no passing cells) with 0
    merged['passing_cells'] = merged['passing_cells'].fillna(0)

    # Calculate proportion
    merged['proportion_passing'] = merged['passing_cells'] / merged['cells_per_roi']

    return merged[[*group_cols, 'proportion_passing']]


# %%


custom_order = [5, 1, 2, 3, 4]
stim_label_map = {5: '1 stim', 1: '3 stim', 2: '5 stim', 3: '10 stim', 4: '20 stim'}

group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']
feature_column = 'peak_time_avg'
# feature_column = 'peak_time_array_mean_trial'

threshold=1


V1_multi_early_proportion = calculate_proportion_above_threshold(
    df_cells=result_V1_multi_cell_non_stimd_filt,
    df_summary=summary_stats_V1_multi_cell_non_stimd,
    feature_col=feature_column,
    threshold=threshold,
    direction='<'
)

V1_multi_early_proportion_2d = V1_multi_early_proportion.pivot(
    index=['subject_fullname', 'session_date', 'scan_number'],
    columns='stim_id',
    values='proportion_passing'
)


V1_multi_early_proportion_2d = reorder_and_rename_stim_columns(
    V1_multi_early_proportion_2d,
    custom_order,
    stim_label_map
)




M2_multi_early_proportion = calculate_proportion_above_threshold(
    df_cells=result_M2_multi_cell_non_stimd_filt,
    df_summary=summary_stats_M2_multi_cell_non_stimd,
    feature_col=feature_column,
    threshold=threshold,
    direction='<'
)


M2_multi_early_proportion_2d = M2_multi_early_proportion.pivot(
    index=['subject_fullname', 'session_date', 'scan_number'],
    columns='stim_id',
    values='proportion_passing'
)


# Only keep those stim_ids that actually exist in the DataFrame
existing_cols = [col for col in custom_order if col in M2_multi_early_proportion_2d.columns]
M2_multi_early_proportion_2d = M2_multi_early_proportion_2d[existing_cols]
M2_multi_early_proportion_2d = M2_multi_early_proportion_2d.rename(columns=stim_label_map)



M2_multi_early_proportion_2d =M2_multi_early_proportion_2d.replace(0, 1e-3)
V1_multi_early_proportion_2d =V1_multi_early_proportion_2d.replace(0, 1e-3)



# M2_multi_early_proportion_2d = M2_multi_early_proportion_2d[M2_multi_early_proportion_2d > 0]
# V1_multi_early_proportion_2d = V1_multi_early_proportion_2d[V1_multi_early_proportion_2d > 0]

# prop_M2 = prop_M2[prop_M2 > 0]

# plot_violin_comparison(V1_multi_early_proportion_2d,M2_multi_early_proportion_2d)

# plot_comparison_boxplots_from_2d_dfs(V1_multi_early_proportion_2d,M2_multi_early_proportion_2d)

analysis_plotting_functions.plot_single_point_sem_with_anova(V1_multi_early_proportion_2d,M2_multi_early_proportion_2d,
                    group_labels=("VISp", "MOs"),
                    colors=(params['general_params']['V1_cl'], params['general_params']['M2_cl']),
                    ylabel="Prop. of resp. cells",
                    xlabel="Number of stim. sites",
                    title="",
                    show_scatter=True,
                    figsize=(6, 3),
                    show=True,
                    # ylim=(-0.01,.08),
                    show_regression=True,
                    log_y=True
)
# %%

V1_multi_late_proportion = calculate_proportion_above_threshold(
    df_cells=result_V1_multi_cell_non_stimd_filt,
    df_summary=summary_stats_V1_multi_cell_non_stimd,
    feature_col=feature_column,
    threshold=threshold,
    direction='>'
)


V1_multi_late_proportion_2d = V1_multi_late_proportion.pivot(
    index=['subject_fullname', 'session_date', 'scan_number'],
    columns='stim_id',
    values='proportion_passing'
)


# Only keep those stim_ids that actually exist in the DataFrame
existing_cols = [col for col in custom_order if col in V1_multi_late_proportion_2d.columns]
V1_multi_late_proportion_2d = V1_multi_late_proportion_2d[existing_cols]
V1_multi_late_proportion_2d = V1_multi_late_proportion_2d.rename(columns=stim_label_map)



M2_multi_late_proportion = calculate_proportion_above_threshold(
    df_cells=result_M2_multi_cell_non_stimd_filt,
    df_summary=summary_stats_M2_multi_cell_non_stimd,
    feature_col=feature_column,
    threshold=threshold,
    direction='>'
)


M2_multi_late_proportion_2d = M2_multi_late_proportion.pivot(
    index=['subject_fullname', 'session_date', 'scan_number'],
    columns='stim_id',
    values='proportion_passing'
)


# Only keep those stim_ids that actually exist in the DataFrame
existing_cols = [col for col in custom_order if col in M2_multi_late_proportion_2d.columns]
M2_multi_late_proportion_2d = M2_multi_late_proportion_2d[existing_cols]
M2_multi_late_proportion_2d = M2_multi_late_proportion_2d.rename(columns=stim_label_map)


M2_multi_late_proportion_2d =M2_multi_late_proportion_2d.replace(0, 1e-3)
V1_multi_late_proportion_2d =V1_multi_late_proportion_2d.replace(0, 1e-3)

# M2_multi_late_proportion_2d = M2_multi_late_proportion_2d[M2_multi_late_proportion_2d > 0]
# V1_multi_late_proportion_2d = V1_multi_late_proportion_2d[V1_multi_late_proportion_2d > 0]

analysis_plotting_functions.plot_single_point_sem_with_anova(V1_multi_late_proportion_2d,M2_multi_late_proportion_2d,
                    group_labels=("VISp", "MOs"),
                    colors=(params['general_params']['V1_cl'], params['general_params']['M2_cl']),
                    ylabel="Prop. of resp. cells",
                    xlabel="Number of stim. sites",
                    title="",
                    show_scatter=True,
                    figsize=(6, 3),
                    show=True,
                    # ylim=(-0.01,.08),
                    show_regression=True,
                    log_y=True
)
# %% Peak times are averaged across stim type first


a = summary_stats_V1_multi_cell_non_stimd.pivot(
    index=['subject_fullname', 'session_date', 'scan_number'],
    columns='stim_id',
    values='peak_time_avg'
    # values='peak_time_array_mean_t'
)


a=reorder_and_rename_stim_columns(
    a,
    custom_order,
    stim_label_map
)


a2 = summary_stats_M2_multi_cell_non_stimd.pivot(
    index=['subject_fullname', 'session_date', 'scan_number'],
    columns='stim_id',
    values='peak_time_avg'
)


a2=reorder_and_rename_stim_columns(
    a2,
    custom_order,
    stim_label_map
)

analysis_plotting_functions.plot_single_point_sem_with_anova(a,a2,
                    group_labels=("VISp", "MOs"),
                    colors=(params['general_params']['V1_cl'], params['general_params']['M2_cl']),
                    ylabel="Peak time (s)",
                    xlabel="Number of stim. sites",
                    title="",
                    show_scatter=True,
                    figsize=(6, 3),
                    show=True,
                    # ylim=(-0.01,.08),
                    show_regression=True,
                    # log_y=True
)
# %%
import pandas as pd
import numpy as np



peak_time_avg_2d_v1_early=get_sorted_peak_time_2d(result_V1_multi_cell_non_stimd_filt)

peak_time_avg_2d_v1_early=reorder_and_rename_stim_columns(
    peak_time_avg_2d_v1_early,
    custom_order,
    stim_label_map
)


peak_time_avg_2d_m2_early=get_sorted_peak_time_2d(result_M2_multi_cell_non_stimd_filt)

peak_time_avg_2d_m2_early=reorder_and_rename_stim_columns(
    peak_time_avg_2d_m2_early,
    custom_order,
    stim_label_map
)


analysis_plotting_functions.plot_single_point_sem_with_anova(peak_time_avg_2d_v1_early,peak_time_avg_2d_m2_early,
                    group_labels=("VISp", "MOs"),
                    colors=(params['general_params']['V1_cl'], params['general_params']['M2_cl']),
                    ylabel="Peak time (s)",
                    xlabel="Number of stim. sites",
                    title="",
                    show_scatter=True,
                    figsize=(6, 3),
                    show=True,
                    # ylim=(-0.01,.08),
                    show_regression=True,
                    # log_y=True
)
# %%

# multi_cell_pseudo_trials_M2=calculate_single_trial_features.generate_pseudo_trials(
#     single_trial_data_multi_cell_M2,
#     n_iters=1,
#     baseline_window=(1000, 10000),
#     trial_crop_idx=None,
#     baseline_start_idx=0,
#     max_offset=7000,
#     zscore_to_baseline=True,
#     baseline_window_for_z=80
# )

# %%
# joblib.dump(pseudo_dfs_iteration_M2, 'pseudo_dfs_iteration_M2_20_it.joblib')


# # %%
# import joblib
# single_trial_data_standard_non_stimd_M2_all_cells = joblib.load('single_trial_data_standard_non_stimd_M2_all_cells.joblib')

# single_trial_data_standard_non_stimd_V1_all_cells = joblib.load('single_trial_data_standard_non_stimd_V1_all_cells.joblib')

# pseudo_dfs_iteration_V1=joblib.load('pseudo_dfs_iteration_V1_20_it.joblib')
# pseudo_dfs_iteration_M2=joblib.load('pseudo_dfs_iteration_M2_20_it.joblib')

# # %%

# pseudo_trials_dffs_iter_V1=pseudo_dfs_iteration_V1['iter_0']
# pseudo_trials_list_V1 = [row.values for _, row in pseudo_trials_dffs_iter_V1.iterrows()]

# pseudo_trials_dffs_iter_M2=multi_cell_pseudo_trials_M2['iter_0']
# pseudo_trials_list_M2_multi_cell = [row.values for _, row in pseudo_trials_dffs_iter_M2.iterrows()]


# pseudo_trial_data=pseudo_trials_list_M2_multi_cell

# single_trial_data=single_trial_data_multi_cell_M2

# filtered_array_M2_pseudo,a,result_M2_pseudo_high= calculate_single_trial_features.process_and_filter_response_matrix(
#     dff_trials=pseudo_trial_data,
#     roi_ids=single_trial_data['roi_ids'],
#     stim_ids=single_trial_data['stim_ids'],
#     roi_keys=single_trial_data['roi_keys'],
#     kernel_size=7,
#     peak_thresh=2.94,
#     group_threshold=2,
#     time_variance='peak_time', # or False to use COM
#     subsample=False,
#     max_trials_per_group=20
# )
# %%

# single_trial_data=single_trial_data_multi_cell_M2

# filtered_array_M2,a,result_M2_high = calculate_single_trial_features.process_and_filter_response_matrix(
#     dff_trials=single_trial_data['trig_dff_trials'],
#     roi_ids=single_trial_data['roi_ids'],
#     stim_ids=single_trial_data['stim_ids'],
#     roi_keys=single_trial_data['roi_keys'],
#     time_axes=single_trial_data['time_axis_sec'],
#     kernel_size=7,
#     peak_thresh=2.94,
#     group_threshold=2,
#     time_variance='peak_time',
#     subsample=False,
#     scramble_stim_ids=False,
#     max_trials_per_group=20,
#     min_width_val=2
# )


# %%
# ---- Common Setup ----
features_to_track = [
    'peak_time_std', 'com_std', 'auc_avg', 'peak_array_mean_trial', 'peak_amp_avg',
    'peak_time_array_mean_trial', 'com_array_mean_trial', 'response_proportion',
    'com_calc_abs_diff_mean_trial', 'peak_time_calc_abs_diff_mean_trial'
]

# ---- Define filters ----
filter_steps = [
    # ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    ('response_proportion >= 0.6', lambda df: df[df['response_proportion'] >= 0.6]),
    # ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 5]),
    # ('peak_array_mean_trial > 0', lambda df: df[df['peak_array_mean_trial'] > 2]),
    ('peak_time_std <= 0.5', lambda df: df[df['peak_time_std'] <= 0.75]),
    # ('peak_time_avg <= 0.5', lambda df: df[df['peak_time_avg'] <= 2.0]),
    # ('peak_time_calc_abs_diff_mean_trial <= 0.4', lambda df: df[df['peak_time_calc_abs_diff_mean_trial'] <= 0.2]),
    # ('peak_amp_avg >= 2', lambda df: df[df['peak_amp_avg'] >= 2]),
    # ('roi_occurrence_all > 0', lambda df: df[df['roi_id_extended_dataset'].map(df['roi_id_extended_dataset'].value_counts()) > 0]),
    ('stim_id', lambda df: df[df['stim_id'] == 2])
]
# %%
# ---- Process real ----
# result_M2_high['is_shuffled'] = False
result_M2_high_filt, summaries_M2 = calculate_single_trial_features.filter_and_summarize(result_M2_multi_cell_non_stimd, result_M2_multi_cell_non_stimd, filter_steps, features_to_track, label="real")


# result_M2_high['is_shuffled'] = False
result_V1_high_filt, summaries_V1 = calculate_single_trial_features.filter_and_summarize(result_V1_multi_cell_non_stimd, result_V1_multi_cell_non_stimd, filter_steps, features_to_track, label="real")


analysis_plotting_functions.prepare_and_plot_heatmap(result_M2_high_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=0,
    vmax=1,
    # start_index=0,
    # sampling_interval=0.032958316,
    exclude_window=None,
    cbar_shrink=0.2,  # shrink colorbar height to 60%
    invert_y=True,
    cmap='bone',
    norm_type='minmax',
    figsize=(6,6),
    xlim=[-3,10])


analysis_plotting_functions.prepare_and_plot_heatmap(result_V1_high_filt,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=0,
    vmax=1,
    # start_index=0,
    # sampling_interval=0.032958316,
    exclude_window=None,
    cbar_shrink=0.2,  # shrink colorbar height to 60%
    invert_y=True,
    cmap='bone',
    norm_type='minmax',
    figsize=(6,6),
    xlim=[-3,10])


