# -*- coding: utf-8 -*-
"""
Created on Thu May 29 12:29:41 2025

@author: jec822
"""

# %% Stim'd cells



# %%

peak_array_v1=result_V1_standard_stimd['peak_array_mean_trial']
peak_array_m2=result_M2_standard_stimd['peak_array_mean_trial']


analysis_plotting_functions.plot_ecdf_comparison(peak_array_v1,peak_array_m2,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='Z-scores of directly stimulated cells',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak z-score",
                     ylabel="Directly stimulated cells",
                     stat_test='auto',
                     figsize=[3,4],
                     xticks_start=0, xticks_end=25, xticks_step=5,
                     yticks_start=0, yticks_end=1, yticks_step=0.5,
                     show_normality_pvals=True,
                     )
# %%
proportions_v1=peak_array_v1=result_V1_standard_stimd['response_proportion']
proportions_m2=peak_array_v1=result_M2_standard_stimd['response_proportion']


analysis_plotting_functions.plot_ecdf_comparison((1-proportions_v1),(1-proportions_m2),
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='Succesful trials in directly stimulated cells',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Proportion of non-responsive trials",
                     ylabel="Directly stimulated cells",
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True,
                     )

# %% peak time for each trial of stimd cells

peak_v1 = result_V1_standard_stimd['peak_time_avg'].explode().astype(float).to_numpy()
peak_m2 = result_M2_standard_stimd['peak_time_avg'].explode().astype(float).to_numpy()


analysis_plotting_functions.plot_ecdf_comparison(peak_v1,peak_m2,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak time (s)",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=2, xticks_step=1,
                     yticks_start=0, yticks_end=1, yticks_step=0.5,
                     xlim=[0,2],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True,
                     )


# %% width for each trial of responsive cells

fhwm_v1 = result_V1_standard_stimd['fwhm_by_trial'].explode().astype(float).to_numpy()
fhwm_m2 = result_M2_standard_stimd['fwhm_by_trial'].explode().astype(float).to_numpy()


analysis_plotting_functions.plot_ecdf_comparison(fhwm_v1*sampling_interval,fhwm_m2*sampling_interval,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak width (s)",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=3, xticks_step=1,
                     xlim=[0,3],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)


# %%

analysis_plotting_functions.plot_ecdf_comparison(summary_stats_V1_standard_stimd['responsive_proportion'],summary_stats_M2_standard_stimd['responsive_proportion'],
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Prop. of responding neurons",
                     ylabel="Prop. of stimulated neuron",
                     xticks_start=0, xticks_end=0.15, xticks_step=0.05,
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)


# %% Non stimd cells



# %% width for each trial of responsive cells

fhwm_v1 = result_V1_standard_non_stimd_filt['fwhm_by_trial'].explode().astype(float).to_numpy()
fhwm_m2 = result_M2_standard_non_stimd_filt['fwhm_by_trial'].explode().astype(float).to_numpy()


analysis_plotting_functions.plot_ecdf_comparison(fhwm_v1*sampling_interval,fhwm_m2*sampling_interval,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak width (s)",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=3, xticks_step=1,
                     yticks_start=0, yticks_end=1, yticks_step=0.5,
                     xlim=[0,3],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)


# %% peak for each trial of responsive cells

peak_v1 = result_V1_standard_non_stimd_filt['peak_amp_avg'].explode().astype(float).to_numpy()
peak_m2 = result_M2_standard_non_stimd_filt['peak_amp_avg'].explode().astype(float).to_numpy()


analysis_plotting_functions.plot_ecdf_comparison(peak_v1,peak_m2,
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Peak width (s)",
                     ylabel="Directly stimulated cells",
                     xticks_start=0, xticks_end=15, xticks_step=5,
                     yticks_start=0, yticks_end=1, yticks_step=0.5,
                     xlim=[0,15],
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)
# %%

analysis_plotting_functions.plot_ecdf_comparison(peak_array_v1,peak_array_m2)

# %% this is reponse proportion each cell

proportions_v1=result_V1_standard_non_stimd_filt['response_proportion']
proportions_m2=result_M2_standard_non_stimd_filt['response_proportion']


analysis_plotting_functions.plot_ecdf_comparison((1-proportions_v1),(1-proportions_m2),
                     label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='Proportion of non-responses in non-stimd cells',
                     line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                     xlabel="Proportion of non-responsive trials by cell",
                     ylabel="Non-stimulated cells")



# %% avg trial sig non stimd cells all


peak_array_mean_V1=result_V1_standard_non_stimd_filt['max_or_min_dff']
peak_array_mean_M2=result_M2_standard_non_stimd_filt['max_or_min_dff']

analysis_plotting_functions.plot_ecdf_comparison(peak_array_mean_V1,peak_array_mean_M2)



# %%

# Set variables and exclude 0s
prop_V1 = summary_stats_V1_standard_non_stimd['responsive_proportion']
# prop_V1 = summary_stats_M2_short_stim_non_stimd['responsive_proportion']

prop_M2 = summary_stats_M2_standard_non_stimd['responsive_proportion']

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
    xlabel="Prop. of responding neurons",
    ylabel="Prop. of stimulated neuron",
    xticks_start=0, xticks_end=0.15, xticks_step=0.05,
    yticks_start=0, yticks_end=1, yticks_step=0.5,
    stat_test='auto',
    figsize=[3, 4],
    show_normality_pvals=True
)

# %%

# Set variables and exclude 0s
prop_V1 = summary_stats_V1_standard_non_stimd['responsive_proportion']
# prop_V1 = summary_stats_M2_short_stim_non_stimd['responsive_proportion']

prop_M2 = summary_stats_M2_standard_non_stimd['responsive_proportion']

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
    xlabel="Prop. of responding neurons",
    ylabel="Prop. of stimulated neuron",
    xticks_start=0, xticks_end=0.15, xticks_step=0.05,
    yticks_start=0, yticks_end=1, yticks_step=0.5,
    stat_test='auto',
    figsize=[3, 4],
    show_normality_pvals=True
)
# %% Distance binning proportion of cells out of count of responses


# bins=[0, 50, 100, 150, 200, 250, 300, np.inf]

bins=params['dist_bins_resp_prob']
 
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']
dist_col = 'min_dist_from_stim_um'

# xtick_labels = [0, 50, 100, 150, 200, 250, 300] 
xtick_labels = np.linspace(50,400,8).astype('int')

V1_dist_df, V1_values_T, V1_labels, V1_bin_centers = analyzeEvoked2P.calculate_proportion_by_distance_bin(
    result_V1_standard_non_stimd_filt,
    group_cols,
    dist_col,
    bins
)

M2_dist_df, M2_values_T, M2_labels, M2_bin_centers = analyzeEvoked2P.calculate_proportion_by_distance_bin(
    result_M2_standard_non_stimd_filt,
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
    custom_xtick_labels=xtick_labels,
    xlim=[25,400]
)

# %% Distance binning proportion of cells out proportion of responder by number of cells in bin


# bins=[0, 50, 100, 150, 200, 250, 300, np.inf]

bins=params['dist_bins_resp_prob']
 
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']
dist_col = 'min_dist_from_stim_um'

# xtick_labels = [0, 50, 100, 150, 200, 250, 300] 
xtick_labels = np.linspace(50,400,8).astype('int')


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
    custom_xtick_labels=xtick_labels,
    xlim=[25,425]
)
# %% peak time binning proportion of cells


# bins=[0, 50, 100, 150, 200, 250, 300, np.inf]

# bins=params['dist_bins_resp_prob']

bins=np.arange(0,10,1)
 
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']
dist_col = 'peak_time_avg'

xtick_labels = [0, .5, 1, 1.5, 2., 2.50, 3.00] 

V1_dist_df, V1_values_T, V1_labels, V1_bin_centers = analyzeEvoked2P.calculate_proportion_by_distance_bin(
    result_V1_standard_non_stimd_filt,
    group_cols,
    dist_col,
    bins
)

M2_dist_df, M2_values_T, M2_labels, M2_bin_centers = analyzeEvoked2P.calculate_proportion_by_distance_bin(
    result_M2_standard_non_stimd_filt,
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

