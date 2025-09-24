# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 17:51:19 2025

@author: jec822
"""

# %% Figure 4A

time_array,m2_avgs_array_2d=analyzeEvoked2P.align_averaged_traces_from_lists(result_M2_standard_non_stimd_exc_filt, trace_col='averaged_traces_all', time_col='time_axes')

time_array,v1_avgs_array_2d=analyzeEvoked2P.align_averaged_traces_from_lists(result_V1_standard_non_stimd_exc_filt, trace_col='averaged_traces_all', time_col='time_axes')


analysis_plotting_functions.plot_mean_trace_multiple_dataframe_input(v1_avgs_array_2d,m2_avgs_array_2d,
                                                                
                                                                     norm_mean=True,
                                                                     norm_type_row='minmax',
                                                                     norm_rows=True,
                                                                     y_limits=[-.2, 1.2],
                                                                     x_limits=[-3, 10],
                                                                     legend_names=[params['general_params']['V1_lbl'], 
                                                                                   params['general_params']['M2_lbl']],
                                                                     line_colors=[params['general_params']['V1_cl'],
                                                                                   params['general_params']['M2_cl']],
                                                                     fill_alpha=0.3,
                                                                     xticks=np.linspace(-2,10,7),
                                                                     yticks=np.linspace(0,1,6),
                                                                     xlabel="Time (s)",
                                                                     ylabel="dF/F",
                                                                     time_array=time_array,
                                                                     exclude_windows=[(-0.15,.25)]
                                                                     )



# %% Figure 4E  
 # peaktime for each trial of responsive cells

peak_v1 = result_V1_standard_non_stimd_exc_filt['peak_time_avg'].explode().astype(float).to_numpy()
peak_m2 = result_M2_standard_non_stimd_exc_filt['peak_time_avg'].explode().astype(float).to_numpy()

# threshold = 1
# count_below_threshold = np.sum(peak_m2 < threshold)
# percentage_below_threshold = (count_below_threshold / len(peak_m2)) * 100

analysis_plotting_functions.plot_ecdf_comparison(peak_v1,peak_m2,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak time (s)",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=8, xticks_step=2,
                     yticks_start=0, yticks_end=1, yticks_step=0.2,
                     xlim=[0,8],
                     stat_test='auto',
                     figsize=[4,4],
                     show_normality_pvals=True)


# %% Figure 4F

analysis_plotting_functions.prepare_and_plot_heatmap(result_M2_standard_non_stimd_exc_filt,trace_column='averaged_traces_all',
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

analysis_plotting_functions.prepare_and_plot_heatmap(result_V1_standard_non_stimd_exc_filt,trace_column='averaged_traces_all',
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

