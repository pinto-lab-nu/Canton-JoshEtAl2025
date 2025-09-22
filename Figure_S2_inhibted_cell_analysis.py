# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:36:18 2025

@author: jec822
"""
import datetime


target_dates = [datetime.date(2024, 12, 17),
                datetime.date(2024, 12, 18),
                datetime.date(2024, 12, 19),
                # datetime.date(2025, 1, 15)
                ]

result_V1_short_stim_non_stimd_inh_filt = result_V1_short_stim_non_stimd_inh_filt[~result_V1_short_stim_non_stimd_inh_filt["session_date"].isin(target_dates)]


summary_stats_V1_short_stim_non_stimd_inh = summary_stats_V1_short_stim_non_stimd_inh[~summary_stats_V1_short_stim_non_stimd_inh["session_date"].isin(target_dates)]

# %%

# Set variables and exclude 0s
prop_V1 = summary_stats_V1_standard_non_stimd_inh['responsive_proportion']
# prop_V1 = summary_stats_M2_short_stim_non_stimd_exc['responsive_proportion']

prop_M2 = summary_stats_M2_standard_non_stimd_inh['responsive_proportion']

prop_V1 = prop_V1[prop_V1 > 0]
prop_M2 = prop_M2[prop_M2 > 0]

# Plot ECDF comparison
analysis_plotting_functions.plot_ecdf_comparison(
    prop_V1, prop_M2,
    label1=params['general_params']['V1_lbl'],
    label2=params['general_params']['M2_lbl'],
    title='',
    line_color1=params['general_params']['V1_cl'],
    line_color2=params['general_params']['M2_cl'],
    xlabel="Prop. of inhibited neurons",
    ylabel="Prop. of stimulated sites",
    # xticks_start=0, xticks_end=0.15, xticks_step=0.05,
    yticks_start=0, yticks_end=1, yticks_step=0.5,
    stat_test='auto',
    xlim=[0,.1],
    figsize=[3, 4],
    show_normality_pvals=True
)

# %%

# Set variables and exclude 0s
prop_V1 = summary_stats_V1_short_stim_non_stimd_inh['responsive_proportion']
# prop_V1 = summary_stats_M2_short_stim_non_stimd_exc['responsive_proportion']

prop_M2 = summary_stats_M2_short_stim_non_stimd_inh['responsive_proportion']

prop_V1 = prop_V1[prop_V1 > 0]
prop_M2 = prop_M2[prop_M2 > 0]

# Plot ECDF comparison
analysis_plotting_functions.plot_ecdf_comparison(
    prop_V1, prop_M2,
    label1=params['general_params']['V1_lbl'],
    label2=params['general_params']['M2_lbl'],
    title='',
    line_color1=params['general_params']['V1_cl'],
    line_color2=params['general_params']['M2_cl'],
    xlabel="Prop. of  inhibited neurons",
    ylabel="Prop. of stimulated sites",
    # xticks_start=0, xticks_end=0.10, xticks_step=0.05,
    yticks_start=0, yticks_end=1, yticks_step=0.5,
    xlim=[0,.10],
    stat_test='auto',
    figsize=[3, 4],
    show_normality_pvals=True
)
# %%

analysis_plotting_functions.prepare_and_plot_heatmap(result_M2_standard_non_stimd_inh_filt,trace_column='averaged_traces_all',
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

analysis_plotting_functions.prepare_and_plot_heatmap(result_V1_standard_non_stimd_inh_filt,trace_column='averaged_traces_all',
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

# %%

analysis_plotting_functions.prepare_and_plot_heatmap(result_M2_short_stim_non_stimd_inh_filt,trace_column='averaged_traces_all',
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

analysis_plotting_functions.prepare_and_plot_heatmap(result_V1_short_stim_non_stimd_inh_filt,trace_column='averaged_traces_all',
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