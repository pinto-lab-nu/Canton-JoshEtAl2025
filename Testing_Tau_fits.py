# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:00:00 2025

@author: jec822
"""
import analyzeSpont2P
import spont_timescales

# %%
area = "V1"
params = opto_params
dff_type = "residuals_dff"

keys_V1 = get_twoptau_keys(area, params, dff_type)
keys_M2 = get_twoptau_keys(area='M2', params=opto_params, dff_type="residuals_dff")

# %%
acorr_M2,lags_V1,k_m2  = (spont_timescales.TwopAutocorr & keys).fetch('autocorr_vals','lags_sec','KEY')

acorr_V1,lags_V1,k  = (spont_timescales.TwopAutocorr & keys).fetch('autocorr_vals','lags_sec','KEY')


# %%
# fetch taus and associated keys, then filter for tau > 0.1
tau_entries = (spont_timescales.TwopTau & keys).fetch('tau', 'KEY')

aicc_double = (spont_timescales.TwopTau & keys).fetch('aicc_double', 'KEY')
r2_fit_double = (spont_timescales.TwopTau & keys).fetch('r2_fit_double', 'KEY')

r_fit_single = (spont_timescales.TwopTau & keys).fetch('r2_fit_single', 'KEY')

is_good = np.array((spont_timescales.TwopTauInclusion & keys).fetch('is_good_tau_roi'))

bic_single = (spont_timescales.TwopTau & keys).fetch('bic_single', 'KEY')
bic_double = (spont_timescales.TwopTau & keys).fetch('bic_double', 'KEY')

tau_single=(spont_timescales.TwopTau & keys).fetch('single_fit_params', 'KEY')


tau_double=(spont_timescales.TwopTau & keys).fetch('double_fit_params', 'KEY')


taus_all, keys_all = tau_entries
mask = taus_all > 0.0
taus = taus_all[mask]
keys = [keys_all[i] for i in np.where(mask)[0]]

# figure out how many of those are somas
seg_keys = VM['twophoton'].Segmentation2P & keys
is_soma = np.array((VM['twophoton'].Roi2P & seg_keys).fetch('is_soma'))
total_soma = np.sum(is_soma)

   # end_time = time.time()
   # print("     done after {: 1.1f} min".format((end_time - start_time) / 60))

   # return taus, keys, total_soma
# %%

bic_d =bic_double[0]
bic_s=bic_single[0]

a=bic_d<bic_s
# double=

mask = r2_fit_double[0] > 0.9
double= r2_fit_double[0][mask]

mask = r_fit_single[0] > 0.9
single= r_fit_single[0][mask]

mask = (r2_fit_double[0] > 0.9) & (r_fit_single[0]<r2_fit_double[0])

mask2 = (r_fit_single[0] > 0.9) & (r_fit_single[0]>r2_fit_double[0])


# %%
r2_s = r_fit_single[0]
r2_d = r2_fit_double[0]
# r2 = max(r2_s, r2_d)

b=(r2_s < r2_d) & (r2_d> 0.9)

a=(r2_s > r2_d) & (r2_s> 0.9)

c= (r2_s> 0.9)

# %%
a=load_tau_data(area='M2', params=opto_params, dff_type='residuals_dff')
# %%
b=load_tau_data(area='V1', params=opto_params, dff_type='residuals_dff')

# %% This function loads tau values from 2025 dj pipeline from a given area

def load_tau_data(area, params, dff_type):
    """
    Fetch and process tau-related data for a given brain area and data type.

    Parameters:
    - area: str, area identifier (e.g., "M2", "V1")
    - params: dict, containing 'general_params' and necessary ID mappings
    - dff_type: str, e.g. 'residuals_dff'

    Returns:
    - A dictionary of all relevant variables
    """
    import numpy as np

    # Extract parameters
    general = params['general_params']
    mice = general[f'{area}_mice']
    corr_param_set_id = general[f'corr_param_id_{dff_type}']
    tau_param_set_id = general['tau_param_set_id']
    incl_set_id = general['twop_inclusion_param_set_id']

    # Collect all keys
    keys = []
    for mouse in mice:
        pk = {
            'subject_fullname': mouse,
            'corr_param_set_id': corr_param_set_id,
            'tau_param_set_id': tau_param_set_id,
            'twop_inclusion_param_set_id': incl_set_id
        }
        keys.extend((spont_timescales.TwopTauInclusion & pk).fetch('KEY'))

    # Fetch all tau-related data
    taus_all, keys_all = (spont_timescales.TwopTau & keys).fetch('tau', 'KEY')
    mask_valid_tau = taus_all > 0.0
    taus = taus_all[mask_valid_tau]
    keys = [keys_all[i] for i in np.where(mask_valid_tau)[0]]

    # Associated metrics
    tau_entries = (spont_timescales.TwopTau & keys).fetch('tau', 'KEY')

    aicc_double = (spont_timescales.TwopTau & keys).fetch('aicc_double', 'KEY')
    r2_fit_double = (spont_timescales.TwopTau & keys).fetch('r2_fit_double', 'KEY')

    r_fit_single = (spont_timescales.TwopTau & keys).fetch('r2_fit_single', 'KEY')

    is_good = np.array((spont_timescales.TwopTauInclusion & keys).fetch('is_good_tau_roi'))

    bic_single = (spont_timescales.TwopTau & keys).fetch('bic_single', 'KEY')
    bic_double = (spont_timescales.TwopTau & keys).fetch('bic_double', 'KEY')

    tau_single=(spont_timescales.TwopTau & keys).fetch('single_fit_params', 'KEY')


    tau_double=(spont_timescales.TwopTau & keys).fetch('double_fit_params', 'KEY')

    # Autocorrelation
    acorr, lags = (spont_timescales.TwopAutocorr & keys).fetch('autocorr_vals', 'lags_sec')

    # Quality filter
    is_good = np.array((spont_timescales.TwopTauInclusion & keys).fetch('is_good_tau_roi'))

    # Soma identification
    seg_keys = VM['twophoton'].Segmentation2P & keys
    is_soma = np.array((VM['twophoton'].Roi2P & seg_keys).fetch('is_soma'))
    total_soma = np.sum(is_soma)

    # Fit comparisons
    bic_prefers_double = bic_double[0] < bic_single[0]

    r2_s = r_fit_single[0]
    r2_d = r2_fit_double[0]

    is_better_double = (r2_d > 0.9) & (r2_d > r2_s)
    is_better_single = (r2_s > 0.9) & (r2_s > r2_d)
    both_good = (r2_s > 0.9)

    result = {
        'taus': taus,
        'keys': keys,
        'total_soma': total_soma,
        'aicc_double': aicc_double,
        'r2_fit_double': r2_fit_double,
        'r2_fit_single': r_fit_single,
        'bic_single': bic_single,
        'bic_double': bic_double,
        'tau_single': tau_single,
        'tau_double': tau_double,
        'acorr': acorr,
        'lags': lags,
        'is_good': is_good,
        'is_soma': is_soma,
        'bic_prefers_double': bic_prefers_double,
        'is_better_double': is_better_double,
        'is_better_single': is_better_single,
        'both_good': both_good,
    }

    return result

# %%
import numpy as np
import matplotlib.pyplot as plt
import os

cell = 5055
t = np.linspace(0, 29, 850)

# Create save directory
save_dir = "fit_plots"
os.makedirs(save_dir, exist_ok=True)

def save_plot(fig, title_prefix, tau_info):
    tau_str = "_".join([f"tau{i}={tau:.2f}" for i, tau in enumerate(tau_info)])
    filename = f"{title_prefix}_cell{cell}_{tau_str}.png"
    fig.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    print(f"Saved: {filename}")

# Dual fit
fit_params = tau_double[0][cell]
A0 = fit_params['A_fit_0']
tau0 = fit_params['tau_fit_0']
A1 = fit_params['A_fit_1']
tau1 = fit_params['tau_fit_1']
offset_d = fit_params['offset_fit_dual']
fit_curve_dual = A0 * np.exp(-t / tau0) + A1 * np.exp(-t / tau1) + offset_d

# Mono fit
fit_params_mono = tau_single[0][cell]
A = fit_params_mono['A_fit_mono']
tau = fit_params_mono['tau_fit_mono']
offset_m = fit_params_mono['offset_fit_mono']
fit_curve_mono = A * np.exp(-t / tau) + offset_m

# Raw data
raw_curve = acorr[cell][0:len(t)]

# Plot all on one figure
fig, ax = plt.subplots(figsize=(12, 6))
fig.canvas.manager.full_screen_toggle()

# Plot curves
ax.plot(t, raw_curve, label='Original Data', color='gray', alpha=0.6)
ax.plot(t, fit_curve_mono, label='Mono-exponential Fit', color='orange')
ax.plot(t, fit_curve_dual, label='Dual-exponential Fit', color='blue')

# Labels
ax.set_title(f'Exponential Fits vs Data (cell {cell})')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()

# Add annotation boxes
ax.text(
    0.7 * max(t),
    max(fit_curve_dual),
    f"Dual Fit\nτ₀ = {tau0:.2f}, τ₁ = {tau1:.2f}\nR² = {r2_d[cell]:.3f}, BIC = {bic_d[cell]:.2f}",
    fontsize=9,
    bbox=dict(facecolor='lightblue', alpha=0.7)
)

ax.text(
    0.7 * max(t),
    0.75 * max(fit_curve_dual),
    f"Mono Fit\nτ = {tau:.2f}\nR² = {r2_s[cell]:.3f}, BIC = {bic_s[cell]:.2f}",
    fontsize=9,
    bbox=dict(facecolor='navajowhite', alpha=0.7)
)

# Save and show
save_plot(fig, "overlay_fit", [tau0, tau1])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# %%
acorr_vec, t_vec, mono_params, dual_params, r2_s, bic_s, r2_d, bic_d = analyzeSpont2P.extract_fit_data_for_key(keys[0])

acorr_vec, t_vec, mono_params, dual_params, r2_s, bic_s, r2_d, bic_d = analyzeSpont2P.extract_fit_data_for_key(a)

# %%

analysis_plotting_functions.plot_autocorrelation_and_fit(
    acorr_vector=acorr_vec,
    time_vector=t_vec,
    mono_fit_params=mono_params,
    dual_fit_params=dual_params,
    r2_mono=r2_s,
    bic_mono=bic_s,
    r2_dual=r2_d,
    bic_dual=bic_d,
    # cell_id=keys[0].get('cell_id', 0),
    cell_id=5,# or adjust based on your key structure
    save_dir="fit_plots",
    title_prefix="overlay_fit",
    color_raw="gray",
    color_mono="orange",
    color_dual="blue"
)

# %%
import numpy as np
import matplotlib.pyplot as plt
import os

# Time axis
t = np.linspace(0, 29, 850)

# Save directory
save_dir = "fit_plots"
os.makedirs(save_dir, exist_ok=True)

def save_plot(fig, title_prefix, tau_info, cell):
    tau_str = "_".join([f"tau{i}={tau:.2f}" for i, tau in enumerate(tau_info)])
    filename = f"{title_prefix}_cell{cell}_{tau_str}.png"
    fig.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    print(f"Saved: {filename}")

# Iterate over all available cells
num_cells = len(tau_double[0])

for cell in range(num_cells):
    try:
        # --- Dual fit ---
        fit_params = tau_double[0][cell]
        A0 = fit_params['A_fit_0']
        tau0 = fit_params['tau_fit_0']
        A1 = fit_params['A_fit_1']
        tau1 = fit_params['tau_fit_1']
        offset_d = fit_params['offset_fit_dual']
        fit_curve_dual = A0 * np.exp(-t / tau0) + A1 * np.exp(-t / tau1) + offset_d

        # --- Mono fit ---
        fit_params_mono = tau_single[0][cell]
        A = fit_params_mono['A_fit_mono']
        tau = fit_params_mono['tau_fit_mono']
        offset_m = fit_params_mono['offset_fit_mono']
        fit_curve_mono = A * np.exp(-t / tau) + offset_m

        # --- Raw data ---
        raw_curve = acorr[cell][0:len(t)]

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.canvas.manager.full_screen_toggle()

        ax.plot(t, raw_curve, label='Original Data', color='gray', alpha=0.6)
        ax.plot(t, fit_curve_mono, label='Mono-exponential Fit', color='orange')
        ax.plot(t, fit_curve_dual, label='Dual-exponential Fit', color='blue')

        ax.set_title(f'Exponential Fits vs Data (cell {cell})')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()

        # Annotate dual fit
        ax.text(
            0.7 * max(t),
            max(fit_curve_dual),
            f"Dual Fit\nτ₀ = {tau0:.2f}, τ₁ = {tau1:.2f}\nR² = {r2_d[cell]:.3f}, BIC = {bic_d[cell]:.2f}",
            fontsize=9,
            bbox=dict(facecolor='lightblue', alpha=0.7)
        )

        # Annotate mono fit
        ax.text(
            0.7 * max(t),
            0.75 * max(fit_curve_dual),
            f"Mono Fit\nτ = {tau:.2f}\nR² = {r2_s[cell]:.3f}, BIC = {bic_s[cell]:.2f}",
            fontsize=9,
            bbox=dict(facecolor='navajowhite', alpha=0.7)
        )

        # Save and close
        save_plot(fig, "overlay_fit", [tau0, tau1], cell)
        plt.close(fig)

    except Exception as e:
        print(f"Skipping cell {cell} due to error: {e}")

# %% this is where I had previous recalculating tau code, move to own 
# The code afte this point was using those calculated taus for stim plots


# %%

non_stimd_roi_keys_sig=m2_avgs['roi_keys']


keys_to_remove = ['ts_param_set_id', 'corr_param_set_id', 'glm_param_set_id']

# Create a new list with those keys removed
longer_list_cleaned = [
    {k: v for k, v in d.items() if k not in keys_to_remove}
    for d in subset
    ]

# %%
def find_matching_indices(list1, list2):
    # Initialize two lists to store the indices
    indices_list1 = []
    indices_list2 = []
    
    # Iterate through both lists
    for i, dict1 in enumerate(list1):
        for j, dict2 in enumerate(list2):
            # Compare the dictionaries
            if dict1 == dict2:
                indices_list1.append(i)
                indices_list2.append(j)
    
    return indices_list1, indices_list2


# %%
indices_list1, indices_list2 = find_matching_indices(longer_list_cleaned, non_stimd_roi_keys_sig)
print(f"Matching indices in list1: {indices_list1}")
print(f"Matching indices in list2: {indices_list2}")
# %%

tau=np.array(subset_df.tau2[indices_list1]/30)



peaktimes=np.array(m2_avgs['peak_times_sec'])
peak_ts=peaktimes[indices_list2]

is_good_r2=np.array([value < 300 for value in subset_df.tau2[indices_list1]])

# %%
 # response properties by tau (need to implement peak width)
 bins     = opto_params['tau_bins']
 bins = np.arange(0, 3, 0.5)
 num_bins = np.size(bins) - 1
 bin_centers = (bins[:-1] + bins[1:]) / 2  # For plotting on x-axis

 num_bins = np.size(bins)-1
 peakt_by_tau_avg  = np.zeros(num_bins)
 peakt_by_tau_sem  = np.zeros(num_bins)
 peakt_by_tau_expt = [None]*num_bins
  # peakm_by_tau_avg  = np.zeros(num_bins)
 # peakm_by_tau_sem  = np.zeros(num_bins)
 # peakm_by_tau_expt = [None]*num_bins
 
 
 # sess_ids    = analyzeSpont2P.sess_ids_from_tau_keys(m2_avgs['roi_keys']) 
 # stim_ids    = deepcopy(m2_avgs['stim_ids'])
 
 for iBin in range(num_bins):
     # idx     = np.logical_and(is_good_tau==1,np.logical_and(tau>bins[iBin], tau<=bins[iBin+1]))
     # idx     = np.logical_and(is_good_tau==1,np.logical_and(is_stimd==0,np.logical_and(is_sig==1,idx)))
     
     
     idx     = np.logical_and(tau>bins[iBin], tau<=bins[iBin+1])
     # idx     = np.logical_and(is_good_tau==1,idx)
     # idx     = np.logical_and(is_good_acorr==True,idx)
     # idx     = np.logical_and(is_good_SNR==True,idx)
     # idx     = np.logical_and(is_good_SNR==True,idx)



     # idx     = np.logical_and(is_sig==1,idx)
     # idx     = np.logical_and(is_stimd==0,idx)
     
     
     sem_den = np.sqrt(np.sum(idx==1)-1)
     peakt_by_tau_avg[iBin] = np.mean(peak_ts[idx])
     peakt_by_tau_sem[iBin] = (np.std(peak_ts[idx],ddof=1))/sem_den
     # peakm_by_tau_avg[iBin] = np.mean(peak_mag[idx])
     # peakm_by_tau_sem[iBin] = (np.std(peak_mag[idx],ddof=1))/sem_den
     # sess = np.unique(sess_ids[idx])
     # # peaks = list()
     # # # mags  = list()
     # for s in sess:
     #      idx_sess = np.logical_and(sess_ids==s,idx)
     #      peaks.append(peak_ts[idx_sess])
     # #     # mags.append(peak_mag[idx_sess])
     # peakt_by_tau_expt[iBin] = peaks
     # peakm_by_tau_expt[iBin] = mags
 
# Plotting with bin centers and SEM
plt.figure()
plt.errorbar(bin_centers, peakt_by_tau_avg, yerr=peakt_by_tau_sem, fmt='-o', capsize=4)
plt.xlabel('Tau bin center')
plt.ylabel('Peak Time (mean ± SEM)')
plt.title('Peak Time by Tau Bin')
plt.grid(True)

# plt.figure()
# plt.errorbar(bin_centers, peakm_by_tau_avg, yerr=peakm_by_tau_sem, fmt='-o', capsize=4)
# plt.xlabel('Tau bin center')
# plt.ylabel('Peak Magnitude (mean ± SEM)')
# plt.title('Peak Magnitude by Tau Bin')
# plt.grid(True)
# %%

# Define bins based on peak_ts instead of tau
bins = np.arange(0, 8, 1)
num_bins = np.size(bins) - 1
bin_centers = (bins[:-1] + bins[1:]) / 2  # For plotting on x-axis

tau_by_peakt_avg  = np.zeros(num_bins)
tau_by_peakt_sem  = np.zeros(num_bins)
tau_by_peakt_expt = [None] * num_bins

for iBin in range(num_bins):
    # Select indices where peak_ts is within the current bin
    idx = np.logical_and(peak_ts > bins[iBin], peak_ts <= bins[iBin + 1])
    idx     = np.logical_and(is_good_r2==True,idx)

    # Avoid empty bins
    if np.sum(idx) > 1:
        sem_den = np.sqrt(np.sum(idx) - 1)
        tau_by_peakt_avg[iBin] = np.mean(tau[idx])
        tau_by_peakt_sem[iBin] = (np.std(tau[idx], ddof=1)) / sem_den
    else:
        tau_by_peakt_avg[iBin] = np.nan
        tau_by_peakt_sem[iBin] = np.nan

# Plotting
plt.figure()
plt.errorbar(bin_centers, tau_by_peakt_avg, yerr=tau_by_peakt_sem, fmt='-o', capsize=4)
plt.xlabel('Peak Time bin center')
plt.ylabel('Tau (mean ± SEM)')
plt.title('Tau by Peak Time Bin')
plt.grid(True)
