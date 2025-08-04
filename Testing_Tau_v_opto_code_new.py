# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:37:33 2025

@author: jec822
"""


# %% Completely separate way of getting the data based on calculations from figure 4

non_stimd_roi_keys_sig=m2_avgs['roi_keys']

tau_entries = (spont_timescales.TwopTau & non_stimd_roi_keys_sig).fetch('tau', 'KEY')

is_good_tau = np.array((spont_timescales.TwopTauInclusion & non_stimd_roi_keys_sig).fetch('is_good_tau_roi'))

keys_with_taus=tau_entries[1]

# acorr,lags  = (spont_timescales.TwopAutocorr & non_stimd_roi_keys_sig).fetch('autocorr_vals','lags_sec')
acorr,lags  = (spont_timescales.TwopAutocorr & keys_with_taus).fetch('autocorr_vals','lags_sec')

snr=(VM['twophoton'].Snr2P & keys_with_taus).fetch('snr','KEY')

event_rate=(VM['twophoton'].Snr2P & keys_with_taus).fetch('events_per_min')

r2_fit_double = (spont_timescales.TwopTau & keys_with_taus).fetch('r2_fit_double', 'KEY')


# %%
a=(VM['twophoton'].Dff2P & keys_with_taus).fetch('dff')
# %%

keys_to_remove = ['ts_param_set_id', 'corr_param_set_id', 'glm_param_set_id']

# Create a new list with those keys removed
shorter_list_cleaned = [
    {k: v for k, v in d.items() if k not in keys_to_remove}
    for d in keys_with_taus
    ]

match_indices = []

for short_dict in shorter_list_cleaned:
    try:
        # Find the first index in longer_list where the dict matches
        index = next(i for i, long_dict in enumerate(non_stimd_roi_keys_sig) if long_dict == short_dict)
        match_indices.append(index)
    except StopIteration:
        # No match found
        match_indices.append(None)  # or skip with `continue`
# %%

peaktimes=np.array(m2_avgs['peak_times_sec'])
peak_ts=peaktimes[match_indices]


#  This needs fixing
peak_magnitude=np.array(m2_avgs['max_or_min_vals'])
peak_mag=peak_magnitude[match_indices]


index_1_values = [subarray[1] for subarray in acorr]
is_good_acorr=np.array([value > 0.1 for value in index_1_values])

is_good_SNR=np.array([value > 5 for value in snr[0]])

is_good_event_rate=np.array([value > 5 for value in event_rate])


is_good_r2=np.array([value > 0.8 for value in r2_fit_double[0]])

tau=tau_entries[0]


# %%
#  # response properties by tau (need to implement peak width)
#  bins     = opto_params['tau_bins']
#  bins=np.arange(0,2.5,.25)

#  num_bins = np.size(bins)-1
#  peakt_by_tau_avg  = np.zeros(num_bins)
#  peakt_by_tau_sem  = np.zeros(num_bins)
#  peakt_by_tau_expt = [None]*num_bins
#  peakm_by_tau_avg  = np.zeros(num_bins)
#  peakm_by_tau_sem  = np.zeros(num_bins)
#  peakm_by_tau_expt = [None]*num_bins
 
 
#  sess_ids    = analyzeSpont2P.sess_ids_from_tau_keys(keys_with_taus) 
#  # stim_ids    = deepcopy(m2_avgs['stim_ids'])
 
#  for iBin in range(num_bins):
#      # idx     = np.logical_and(is_good_tau==1,np.logical_and(tau>bins[iBin], tau<=bins[iBin+1]))
#      # idx     = np.logical_and(is_good_tau==1,np.logical_and(is_stimd==0,np.logical_and(is_sig==1,idx)))
     
     
#      idx     = np.logical_and(tau>bins[iBin], tau<=bins[iBin+1])
#      # idx     = np.logical_and(is_good_tau==1,idx)
#      idx     = np.logical_and(is_good_acorr==True,idx)
#      idx     = np.logical_and(is_good_SNR==True,idx)
#      idx     = np.logical_and(is_good_r2==True,idx)



#      # idx     = np.logical_and(is_sig==1,idx)
#      # idx     = np.logical_and(is_stimd==0,idx)
     
     
#      sem_den = np.sqrt(np.sum(idx==1)-1)
#      peakt_by_tau_avg[iBin] = np.mean(peak_ts[idx])
#      peakt_by_tau_sem[iBin] = (np.std(peak_ts[idx],ddof=1))/sem_den
#      peakm_by_tau_avg[iBin] = np.mean(peak_mag[idx])
#      peakm_by_tau_sem[iBin] = (np.std(peak_mag[idx],ddof=1))/sem_den
#      sess = np.unique(sess_ids[idx])
#      peaks = list()
#      mags  = list()
#      for s in sess:
#            idx_sess = np.logical_and(sess_ids==s,idx)
#            peaks.append(peak_ts[idx_sess])
#            mags.append(peak_mag[idx_sess])
#      peakt_by_tau_expt[iBin] = peaks
#      peakm_by_tau_expt[iBin] = mags
 
# plt.figure()
# plt.plot(peakt_by_tau_avg)

# plt.figure()
# plt.plot(peakm_by_tau_avg)

# %%

import numpy as np
import matplotlib.pyplot as plt

bins = np.arange(0, 2.5, 0.25)
num_bins = np.size(bins) - 1
bin_centers = (bins[:-1] + bins[1:]) / 2  # For plotting on x-axis

peakt_by_tau_avg  = np.zeros(num_bins)
peakt_by_tau_sem  = np.zeros(num_bins)
peakt_by_tau_expt = [None]*num_bins
peakm_by_tau_avg  = np.zeros(num_bins)
peakm_by_tau_sem  = np.zeros(num_bins)
peakm_by_tau_expt = [None]*num_bins

sess_ids = analyzeSpont2P.sess_ids_from_tau_keys(keys_with_taus) 

for iBin in range(num_bins):
    idx = np.logical_and(tau > bins[iBin], tau <= bins[iBin+1])
    # idx = np.logical_and(is_good_acorr == True, idx)
    # idx = np.logical_and(is_good_SNR == True, idx)
    idx = np.logical_and(is_good_event_rate == True, idx)

    idx = np.logical_and(is_good_r2 == True, idx)

    sem_den = np.sqrt(np.sum(idx == 1) - 1)
    peakt_by_tau_avg[iBin] = np.mean(peak_ts[idx])
    peakt_by_tau_sem[iBin] = (np.std(peak_ts[idx], ddof=1)) / sem_den
    peakm_by_tau_avg[iBin] = np.mean(peak_mag[idx])
    peakm_by_tau_sem[iBin] = (np.std(peak_mag[idx], ddof=1)) / sem_den

    sess = np.unique(sess_ids[idx])
    peaks = list()
    mags  = list()
    for s in sess:
        idx_sess = np.logical_and(sess_ids == s, idx)
        peaks.append(peak_ts[idx_sess])
        mags.append(peak_mag[idx_sess])
    peakt_by_tau_expt[iBin] = peaks
    peakm_by_tau_expt[iBin] = mags

# Plotting with bin centers and SEM
plt.figure()
plt.errorbar(bin_centers, peakt_by_tau_avg, yerr=peakt_by_tau_sem, fmt='-o', capsize=4)
plt.xlabel('Tau bin center')
plt.ylabel('Peak Time (mean ± SEM)')
plt.title('Peak Time by Tau Bin')
plt.grid(True)

plt.figure()
plt.errorbar(bin_centers, peakm_by_tau_avg, yerr=peakm_by_tau_sem, fmt='-o', capsize=4)
plt.xlabel('Tau bin center')
plt.ylabel('Peak Magnitude (mean ± SEM)')
plt.title('Peak Magnitude by Tau Bin')
plt.grid(True)


# %%


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import numpy as np

# Example data (replace with your actual data)
# snr, event_rate, index_1_values: 1D arrays of same length
# labels: list of names for each point (e.g., cell IDs or variable names)
# snr = ...
# event_rate = ...
# index_1_values = ...


labels = ['snr','event_rate', 'acorr']

X = np.column_stack((snr, event_rate, index_1_values))

# Perform clustering (change n_clusters as appropriate)
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_ids = kmeans.fit_predict(X)

# Choose a colormap
colors = plt.cm.get_cmap('hot', n_clusters)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color-coded clusters
sc = ax.scatter(snr, event_rate, index_1_values, c=cluster_ids, cmap=colors, marker='o', s=50, edgecolor='k')

# Add labels at each point
for i, label in enumerate(labels):
    ax.text(snr[i], event_rate[i], index_1_values[i], label, fontsize=8)

# Axes labels
ax.set_xlabel('SNR')
ax.set_ylabel('Event Rate')
ax.set_zlabel('Index 1 Value')

# Optional rotation
ax.view_init(elev=30, azim=135)  # Adjust elevation and azimuth here

# Color bar for clusters
cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
cbar.set_label('Cluster ID')

plt.title('3D Scatter Plot with Labels and Clustering')
plt.tight_layout()
plt.show()

# %%

import numpy as np

def calculate_peak_to_noise_snr(data, signal_range=(0, 10000), baseline_range=(0, 1000)):
    """
    Calculate the peak-to-noise ratio for each signal in a nested array.
    
    Parameters:
    - data: list or array-like of nested arrays, accessed via data[i][0]
    - signal_range: tuple, range of samples to consider as the full signal (default: 0 to 10000)
    - baseline_range: tuple, range of samples to consider as baseline noise (default: 0 to 1000)

    Returns:
    - snr_list: list of SNR values (peak / std of baseline)
    """
    snr_list = []

    for i in range(len(data)):
        try:
            signal_full = np.array(data[i][0][signal_range[0]:signal_range[1]])
            baseline = signal_full[baseline_range[0]:baseline_range[1]]

            if len(signal_full) == 0 or len(baseline) == 0:
                snr = np.nan
            elif np.all(np.isnan(signal_full)) or np.std(baseline) == 0:
                snr = np.nan
            else:
                peak = np.nanmax(signal_full)
                noise_std = np.nanstd(baseline)
                snr = peak / noise_std
        except Exception as e:
            snr = np.nan  # If indexing or structure fails

        snr_list.append(snr)

    return snr_list

# %%

import numpy as np

def calculate_autocorrelations(data, signal_range=(0, 10000), max_lags=1000):
    """
    Calculate autocorrelations up to max_lags for each signal in the nested array.
    
    Parameters:
    - data: list or array-like of nested arrays, accessed via data[i][0]
    - signal_range: tuple, range of samples to extract (default: 0 to 10000)
    - max_lags: int, number of lags for autocorrelation (default: 1000)

    Returns:
    - acorr_list: list of 1D autocorrelation arrays (length: max_lags + 1)
    """
    acorr_list = []

    for i in range(len(data)):
        try:
            signal = np.array(data[i][0][signal_range[0]:signal_range[1]])
            signal = signal - np.nanmean(signal)  # remove mean
            if np.all(np.isnan(signal)) or len(signal) < max_lags:
                acorr = np.full(max_lags + 1, np.nan)
            else:
                acorr_full = np.correlate(signal, signal, mode='full')
                mid = len(acorr_full) // 2
                acorr = acorr_full[mid:mid + max_lags + 1]
                acorr = acorr / acorr[0] if acorr[0] != 0 else np.full_like(acorr, np.nan)
        except Exception:
            acorr = np.full(max_lags + 1, np.nan)

        acorr_list.append(acorr)

    return acorr_list

# %%
snr_values = calculate_peak_to_noise_snr(a)
autocorrelations = calculate_autocorrelations(a, signal_range=(0, 10000), max_lags=1000)