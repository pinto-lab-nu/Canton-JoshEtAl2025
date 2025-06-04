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

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.stats import zscore
import os

def plot_nonoverlapping_traces(
    data,
    trace_indices=None,
    sampling_rate=30,
    apply_medfilt=False,
    medfilt_kernel=5,
    offset=1.0,
    color='black',
    alpha=0.6,
    cutoff=None,
    save_path=None,
    y_scale_bar=True,
    scale_bar_length=1.0,
    scale_bar_label='z-score',
    zscore_traces=True
):
    """
    Plots selected traces from a nested list of dF/F data in a non-overlapping vertical stack.

    Parameters:
    - data: list of list of arrays, indexed as data[i][0][:]
    - trace_indices: list of integers, specifies which i-th traces to plot
    - sampling_rate: float, samples per second to convert x-axis to time
    - apply_medfilt: bool, whether to apply median filtering
    - medfilt_kernel: int, kernel size for median filter
    - offset: float, vertical offset between traces
    - color: str, color of the traces
    - alpha: float, transparency of the traces
    - cutoff: tuple (start, stop), range to slice each trace like [start:stop]
    - save_path: str or None, if given, save figure as SVG to this path
    - y_scale_bar: bool, whether to add a vertical scale bar
    - scale_bar_length: float, height of the scale bar in y-units
    - scale_bar_label: str, label for the scale bar
    - zscore_traces: bool, whether to z-score each trace individually
    """

    if trace_indices is None:
        trace_indices = list(range(min(5, len(data))))  # Default to first 5

    plt.figure(figsize=(8, 6))
    for i, idx in enumerate(trace_indices):
        trace = data[idx][0]
        if cutoff is not None:
            trace = trace[cutoff[0]:cutoff[1]]
        if apply_medfilt:
            trace = medfilt(trace, kernel_size=medfilt_kernel)
        if zscore_traces:
            trace = zscore(trace, nan_policy='omit')

        time = np.arange(len(trace)) / sampling_rate
        plt.plot(time, trace + i * offset, color=color, alpha=alpha)

    # Add scale bar
    if y_scale_bar:
        x0 = time[0] - 0.05 * (time[-1] - time[0])  # slightly left of start
        y0 = offset * 0.25  # low on first trace
        plt.plot([x0, x0], [y0, y0 + scale_bar_length], color='black', lw=2, clip_on=False)
        plt.text(x0, y0 + scale_bar_length / 2, scale_bar_label, va='center', ha='right', fontsize=10)

    # Illustrator-friendly setup
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Trace Offset', fontsize=12)
    plt.yticks([])
    plt.grid(False)
    plt.box(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        if not save_path.endswith('.svg'):
            save_path += '.svg'
        plt.savefig(save_path, format='svg', transparent=True)
        print(f"SVG saved to: {os.path.abspath(save_path)}")

    plt.show()


# %%

plot_nonoverlapping_traces(
    filtered_dff_m2,
    trace_indices=[0, 1, 2, 3, 4],
    sampling_rate=30,
    apply_medfilt=True,
    medfilt_kernel=15,
    offset=10,
    color=params['general_params']['M2_cl'],
    alpha=0.5,
    save_path='dff_plot_m2',
    cutoff=(0,10000)
)

plot_nonoverlapping_traces(
    filtered_dff_V1,
    trace_indices=[0, 1, 2, 3, 4],
    sampling_rate=30,
    apply_medfilt=True,
    medfilt_kernel=15,
    offset=10,
    color=params['general_params']['V1_cl'],
    alpha=0.5,
    save_path='dff_plot_v1',
    cutoff=(0,10000)
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
acorr_vec, t_vec, mono_params, dual_params, r2_s, bic_s, r2_d, bic_d = extract_fit_data_for_key(key_with_param)


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
    cell_id=keys[0].get('cell_id', 0),  # or adjust based on your key structure
    save_dir="fit_plots",
    title_prefix="overlay_fit",
    color_raw=params['general_params']['M2_cl'],
    color_mono="orange",
    color_dual="blue",
    figsize=(6,6),
    custom_filename="M2_fit_plot"
)

# %%

params = opto_params
dff_type = 'residuals_dff'

# Extract IDs
tau_param_set_id = params['general_params']['tau_param_set_id']
corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
incl_set_id = params['general_params']['twop_inclusion_param_set_id']

# Make a copy of the key and add all needed parameters
key_with_param = dict(filtered_keys_V1[0])
key_with_param['tau_param_set_id'] = tau_param_set_id
key_with_param['corr_param_set_id'] = corr_param_set_id
key_with_param['inclusion_param_set_id'] = incl_set_id

# Now use the key to fetch data
acorr_vec, t_vec, mono_params, dual_params, r2_s, bic_s, r2_d, bic_d = extract_fit_data_for_key(key_with_param)


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
    cell_id=keys[0].get('cell_id', 0),  # or adjust based on your key structure
    save_dir="fit_plots",
    title_prefix="overlay_fit",
    color_raw=params['general_params']['V1_cl'],
    color_mono="orange",
    color_dual="blue",
    figsize=(6,6),
    custom_filename="V1_fit_plot"
)
