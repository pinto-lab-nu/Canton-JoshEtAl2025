# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 22:59:20 2025

@author: jec822
"""
import analyzeEvoked2P
import Canton_Josh_et_al_2025_analysis_plotting_functions as analysis_plotting_functions
# %%

data = get_single_trial_data_from_individual_session_stim(opto_params,fig3_params['sess_eg_m2'], fig3_params['stim_eg_m2'],area='M2')

data_V1 = get_single_trial_data_from_individual_session_stim(opto_params,fig3_params['sess_eg_v1'], fig3_params['stim_eg_v1'],area='V1')

# %%
data_stim = get_single_trial_data_from_individual_session_stim(opto_params,fig3_params['sess_eg_v1'], fig3_params['stim_eg_v1'],area='V1',which_neurons='stimd')

# %%

plt.plot(data['trig_dff_trials'][100])

# %%

list_of_arrays = [np.expand_dims(x, axis=0) for x in data['trig_dff_trials']]

# %%

analysis_plotting_functions.plot_nonoverlapping_traces(
    list_of_arrays,
    trace_indices=[0,5,10,15,20,25,30,35,40,45],
    sampling_rate=30,
    apply_medfilt=True,
    medfilt_kernel=1,
    offset=10,
    color=opto_params['general_params']['M2_cl'],
    alpha=1.0,
    cutoff=None,
    save_path=None,
    y_scale_bar=True,
    scale_bar_length=1.0,
    scale_bar_label='z-score',
    zscore_traces=False
)


# %% Stimulated cells
list_of_arrays = [np.expand_dims(x, axis=0) for x in data_stim['trig_dff_trials']]

analysis_plotting_functions.plot_nonoverlapping_traces(
    list_of_arrays,
    trace_indices=[0,1,2,3,4],
    sampling_rate=30,
    apply_medfilt=True,
    medfilt_kernel=1,
    offset=10,
    color=opto_params['general_params']['M2_cl'],
    alpha=1.0,
    cutoff=None,
    save_path=None,
    y_scale_bar=True,
    scale_bar_length=1.0,
    scale_bar_label='z-score',
    zscore_traces=False
)


# %%

list_of_arrays_V1 = [np.expand_dims(x, axis=0) for x in data_V1['trig_dff_trials']]

# %%
analysis_plotting_functions.plot_nonoverlapping_traces(
    list_of_arrays_V1,
    trace_indices=[3,5,10,15,20,26,30,35,40,45],
    sampling_rate=30,
    apply_medfilt=True,
    medfilt_kernel=1,
    offset=10,
    color=opto_params['general_params']['V1_cl'],
    alpha=1.0,
    cutoff=None,
    save_path=None,
    y_scale_bar=True,
    scale_bar_length=1.0,
    scale_bar_label='z-score',
    zscore_traces=False
)