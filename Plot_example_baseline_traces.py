# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:22:20 2025

@author: jec822
"""

# %% Plot different  baseline traces


# sess_keys_m2=get_keys_for_expt_types('M2', params=opto_params, expt_type='standard')
# sess_keys_v1=get_keys_for_expt_types('V1', params=opto_params, expt_type='standard')

non_stimd_roi_keys_sig=m2_avgs['roi_keys']

# tau_entries = (spont_timescales.TwopTau & non_stimd_roi_keys_sig).fetch('tau', 'KEY')
min_tau = 0
max_tau = 5.0

tau_values, tau_keys = (spont_timescales.TwopTau & non_stimd_roi_keys_sig).fetch('tau', 'KEY')
filtered_keys = [key for tau, key in zip(tau_values, tau_keys) if min_tau <= tau <= max_tau]
filtered_taus_and_keys = [(tau, key) for tau, key in zip(tau_values, tau_keys) if min_tau <= tau <= max_tau]

is_good_tau = np.array((spont_timescales.TwopTauInclusion & non_stimd_roi_keys_sig).fetch('is_good_tau_roi'))

keys_with_taus_m2=filtered_keys

# %%
non_stimd_roi_keys_sig=v1_avgs['roi_keys']

min_tau = 0
max_tau = 5.0

tau_values, tau_keys = (spont_timescales.TwopTau & non_stimd_roi_keys_sig).fetch('tau', 'KEY')
filtered_keys = [key for tau, key in zip(tau_values, tau_keys) if min_tau <= tau <= max_tau]
filtered_taus_and_keys = [(tau, key) for tau, key in zip(tau_values, tau_keys) if min_tau <= tau <= max_tau]

is_good_tau = np.array((spont_timescales.TwopTauInclusion & non_stimd_roi_keys_sig).fetch('is_good_tau_roi'))

keys_with_taus_v1=filtered_keys

# %%
# For M2
dff_m2 = (VM['twophoton'].Dff2P & keys_with_taus_m2).fetch('dff')
snr_m2, snr_keys_m2 = (VM['twophoton'].Snr2P & keys_with_taus_m2).fetch('snr', 'KEY')

# For V1
dff_V1 = (VM['twophoton'].Dff2P & keys_with_taus_v1).fetch('dff')
snr_V1, snr_keys_V1 = (VM['twophoton'].Snr2P & keys_with_taus_v1).fetch('snr', 'KEY')

# %%
def filter_dff_by_snr(dff, snr, snr_keys, min_snr=2.0, max_snr=10.0):
    """Returns dff traces and keys with SNR within the given range."""
    filtered_dff = []
    filtered_keys = []
    for trace, s, key in zip(dff, snr, snr_keys):
        if min_snr <= s <= max_snr:
            filtered_dff.append(trace)
            filtered_keys.append(key)
    return filtered_dff, filtered_keys

# %%

filtered_dff_m2, filtered_keys_m2 = filter_dff_by_snr(dff_m2, snr_m2, snr_keys_m2, min_snr=3, max_snr=8)
filtered_dff_V1, filtered_keys_V1 = filter_dff_by_snr(dff_V1, snr_V1, snr_keys_V1, min_snr=3, max_snr=8)


# %%

analysis_plotting_functions.plot_nonoverlapping_traces(
    filtered_dff_m2,
    trace_indices=[0, 1, 2, 3, 4],
    sampling_rate=30,
    apply_medfilt=True,
    medfilt_kernel=15,
    offset=5,
    color=params['general_params']['M2_cl'],
    alpha=1,
    save_path='dff_plot_m2',
    cutoff=(0,10000),
    zscore_traces=False,
    scale_bar_label='dF/F'
)

analysis_plotting_functions.plot_nonoverlapping_traces(
    filtered_dff_V1,
    trace_indices=[4, 1, 2, 3, 0],
    sampling_rate=30,
    apply_medfilt=True,
    medfilt_kernel=15,
    offset=5,
    color=params['general_params']['V1_cl'],
    alpha=1,
    save_path='dff_plot_v1',
    cutoff=(0,10000),
    zscore_traces=False,
    scale_bar_label='dF/F'
)



# %%

params = opto_params
dff_type = 'residuals_dff'

# Extract IDs
tau_param_set_id = params['general_params']['tau_param_set_id']
corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
incl_set_id = params['general_params']['twop_inclusion_param_set_id']

# Make a copy of the key and add all needed parameters
key_with_param = dict(filtered_keys_m2[0])
key_with_param['tau_param_set_id'] = tau_param_set_id
key_with_param['corr_param_set_id'] = corr_param_set_id
key_with_param['inclusion_param_set_id'] = incl_set_id


# Now use the key to fetch data
acorr_vec, t_vec, mono_params, dual_params, r2_s, bic_s, r2_d, bic_d = analyzeSpont2P.extract_fit_data_for_key(key_with_param)


# %%

plot_autocorrelation_and_fit(
    acorr_vector=acorr_vec,
    time_vector=t_vec,
    mono_fit_params=mono_params,
    dual_fit_params=dual_params,
    r2_mono=r2_s,
    bic_mono=bic_s,
    r2_dual=r2_d,
    bic_dual=bic_d,
    cell_id=filtered_keys_m2[10].get('roi_id', 0),  # or adjust based on your key structure
    save_dir="fit_plots",
    title_prefix="overlay_fit",
    color_raw=params['general_params']['M2_cl'],
    color_mono="orange",
    color_dual="blue",
    figsize=(2,2),
    custom_filename="M2_fit_plot",
    text_size=8,
    num_ticks=3,
    tick_bounds=((0, 30), (0, 1)),
    xlim=(-1,30),
    ylim=(-0.1,1)
)

# %%

params = opto_params
dff_type = 'residuals_dff'

# Extract IDs
tau_param_set_id = params['general_params']['tau_param_set_id']
corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
incl_set_id = params['general_params']['twop_inclusion_param_set_id']

# Make a copy of the key and add all needed parameters
key_with_param = dict(filtered_keys_V1[4])
key_with_param['tau_param_set_id'] = tau_param_set_id
key_with_param['corr_param_set_id'] = corr_param_set_id
key_with_param['inclusion_param_set_id'] = incl_set_id

# Now use the key to fetch data
acorr_vec, t_vec, mono_params, dual_params, r2_s, bic_s, r2_d, bic_d = analyzeSpont2P.extract_fit_data_for_key(key_with_param)


# %%

plot_autocorrelation_and_fit(
    acorr_vector=acorr_vec,
    time_vector=t_vec,
    mono_fit_params=mono_params,
    dual_fit_params=dual_params,
    r2_mono=r2_s,
    bic_mono=bic_s,
    r2_dual=r2_d,
    bic_dual=bic_d,
    cell_id=filtered_keys_V1[0].get('cell_id', 0),  # or adjust based on your key structure
    save_dir="fit_plots",
    title_prefix="overlay_fit",
    color_raw=params['general_params']['V1_cl'],
    color_mono="orange",
    color_dual="blue",
    figsize=(2,2),
    custom_filename="V1_fit_plot",
    text_size=8,
    num_ticks=3,
    tick_bounds=((0, 30), (0, 1)),
    xlim=(-1,30),
    ylim=(-0.1,1)
)
