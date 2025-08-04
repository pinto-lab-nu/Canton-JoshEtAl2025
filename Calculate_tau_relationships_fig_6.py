# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 18:05:53 2025

@author: jec822
"""

# Proportion of a respnse by tau
#  from the neurons that respons in a given stim are the y short etc tau

result_V1= result_V1_standard_non_stimd_filt[result_V1_standard_non_stimd_filt['is_good_tau'] >= 1]
result_M2= result_M2_standard_non_stimd_filt[result_M2_standard_non_stimd_filt['is_good_tau'] >= 1]

bins=np.arange(0,10,1)
 
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

dist_col = 'peak_time_avg'
bin_column='tau'

xtick_labels = [0, .5, 1, 1.5, 2., 2.50, 3.00] 

V1_dist_df, V1_values_T, V1_labels, V1_bin_centers = analyzeEvoked2P.calculate_condition_proportion_by_bin(
    # result_V1,
    result_V1_standard_non_stimd_filt,
    group_cols,
    bin_column,
    dist_col,
    bins
)

M2_dist_df, M2_values_T, M2_labels, M2_bin_centers = analyzeEvoked2P.calculate_condition_proportion_by_bin(
    # result_M2,
    result_M2_standard_non_stimd_filt,
    group_cols,
    bin_column,
    dist_col,
    bins
)

analysis_plotting_functions.plot_mean_sem_with_anova(
    V1_values_T, M2_values_T, V1_bin_centers,  # use centers for x-axis
    label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],
    line_color1= params['general_params']['V1_cl'], 
    line_color2= params['general_params']['M2_cl'],
    xlabel='Dist. from stim. cell (µm)', ylabel='Proportion of Cells',
    title='',
    figsize=[4, 4],
    custom_xticks=xtick_labels,
    custom_xtick_labels=xtick_labels
)

# %%

result_V1= result_V1_standard_non_stimd_filt[result_V1_standard_non_stimd_filt['is_good_tau'] >= 1]
result_M2= result_M2_standard_non_stimd_filt[result_M2_standard_non_stimd_filt['is_good_tau'] >= 1]

# bins=[0, 50, 100, 150, 200, 250, 300, np.inf]

bins=np.arange(0,5,2)
 
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']
dist_col = 'tau'

xtick_labels = [0, .5, 1, 1.5, 2., 2.50, 3.00] 

V1_dist_df, V1_values_T, V1_labels, V1_bin_centers = analyzeEvoked2P.calculate_responsive_proportion_by_distance_bin_by_all_cells(
    result_V1_standard_non_stimd_filt,
    result_V1_standard_non_stimd,
    group_cols,
    dist_col,
    bins
)

M2_dist_df, M2_values_T, M2_labels, M2_bin_centers = analyzeEvoked2P.calculate_responsive_proportion_by_distance_bin_by_all_cells(
    result_M2_standard_non_stimd_filt,
    result_M2_standard_non_stimd,
    group_cols,
    dist_col,
    bins
)

analysis_plotting_functions.plot_mean_sem_with_anova(
    V1_values_T, M2_values_T, V1_bin_centers,  # use centers for x-axis
    label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],
    line_color1= params['general_params']['V1_cl'], 
    line_color2= params['general_params']['M2_cl'],
    xlabel='Dist. from stim. cell (µm)', ylabel='Proportion of Cells',
    title='',
    figsize=[4, 4],
    custom_xticks=xtick_labels,
    custom_xtick_labels=xtick_labels
)
# %%

# Proportion of a respnse by tau
#  from the neurons that respons in a given stim are the y short etc tau

# bins=[0, 50, 100, 150, 200, 250, 300, np.inf]

# bins=params['dist_bins_resp_prob']

result_V1= result_V1_standard_non_stimd_filt[result_V1_standard_non_stimd_filt['is_good_tau'] >= 1]
result_M2= result_M2_standard_non_stimd_filt[result_M2_standard_non_stimd_filt['is_good_tau'] >= 1]

bins=np.arange(0,3.5,1)
 
# bin_col = 'tau'
bin_col = 'peak_time_avg'

# value_col = 'peak_time_avg'
value_col='tau'

xtick_labels = [0, .5, 1, 1.5, 2., 2.50, 3.00] 

V1_dist_df,bin_labels, V1_bin_centers = bin_column_values(
    result_V1,
    value_col,
    bin_col,
    bins=bins
)

M2_dist_df,bin_labels, bin_centers = bin_column_values(
    result_M2,
    value_col,
    bin_col,
    bins=bins
)

analysis_plotting_functions.plot_mean_sem_with_anova(
    V1_dist_df, M2_dist_df, V1_bin_centers,  # use centers for x-axis
    label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],
    line_color1= params['general_params']['V1_cl'], 
    line_color2= params['general_params']['M2_cl'],
    xlabel='Dist. from stim. cell (µm)', ylabel='Proportion of Cells',
    title='',
    figsize=[4, 4],
    custom_xticks=xtick_labels,
    custom_xtick_labels=xtick_labels
)