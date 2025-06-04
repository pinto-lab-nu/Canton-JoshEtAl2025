# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 11:00:00 2025

@author: jec822
"""

# def get_all_tau(area, params=params, dff_type='residuals_dff'):

   area="V1"
   params=opto_params
   dff_type='residuals_dff'
  
   # get primary keys for query
   mice = params['general_params']['{}_mice'.format(area)]
   corr_param_set_id = params['general_params']['corr_param_id_{}'.format(dff_type)]
   tau_param_set_id = params['general_params']['tau_param_set_id']
   incl_set_id = params['general_params']['twop_inclusion_param_set_id']

   # get relevant keys, filtering for inclusion for speed
   keys = list()
   for mouse in mice:
       primary_key = {
           'subject_fullname': mouse,
           'corr_param_set_id': corr_param_set_id,
           'tau_param_set_id': tau_param_set_id,
           'twop_inclusion_param_set_id': incl_set_id
       }
       # keys.append((spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=1').fetch('KEY'))
       # keys.append((spont_timescales.TwopTauInclusion & primary_key & 'is_good_tau_roi=0').fetch('KEY'))
       keys.append((spont_timescales.TwopTauInclusion & primary_key).fetch('KEY'))

   keys = [entry for sublist in keys for entry in sublist]  # flatten

   # fetch taus and associated keys, then filter for tau > 0.1
   tau_entries = (spont_timescales.TwopTau & keys).fetch('tau', 'KEY')
   
   aicc_double = (spont_timescales.TwopTau & keys).fetch('aicc_double', 'KEY')
   r2_fit_double = (spont_timescales.TwopTau & keys).fetch('r2_fit_double', 'KEY')
   acorr,lags  = (spont_timescales.TwopAutocorr & keys).fetch('autocorr_vals','lags_sec')
   
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

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
cell=5055
# Time axis
t = np.linspace(0, 29, 850)

fit_params=tau_double[0][cell]

# Extract parameters
A0 = fit_params['A_fit_0']
tau0 = fit_params['tau_fit_0']
A1 = fit_params['A_fit_1']
tau1 = fit_params['tau_fit_1']
offset = fit_params['offset_fit_dual']

# Compute the fitted curve
fit_curve = A0 * np.exp(-t / tau0) + A1 * np.exp(-t / tau1) + offset

# Plot
plt.plot(t, fit_curve, label='Dual exponential fit')
plt.xlabel('Time')
plt.ylabel('Fitted Value')
plt.legend()
plt.show()


# Time axis
# t = np.linspace(0, 10, 1000)
t = np.linspace(0, 29, 850)

fit_params_mono=tau_single[0][cell]

# Extract parameters
A = fit_params_mono['A_fit_mono']
tau = fit_params_mono['tau_fit_mono']
offset = fit_params_mono['offset_fit_mono']

# Compute mono-exponential fit
fit_curve = A * np.exp(-t / tau) + offset

# Plot
plt.plot(t, fit_curve, label='Mono-exponential fit', color='orange')
plt.xlabel('Time')
plt.ylabel('Fitted Value')
plt.legend()
plt.show()
# 
plt.plot(t,acorr[cell][0:len(t)])


# %%
def fit_exponentials_nested(acorr, exclude_low_acorr=False, acorr_threshold=0.1):
    results = []
    filtered_acorr = []  # to collect successfully analyzed traces
    valid_indices = []   # to track their original indices

    for i, y_full in enumerate(acorr):
        y_full = np.array(y_full)
        x_full = np.arange(len(y_full))

        # Skip if invalid: empty, NaNs, all zeros, or constant
        if len(y_full) == 0 or np.all(np.isnan(y_full)) or np.all(y_full == 0) or np.all(y_full == y_full[0]):
            continue

        # Apply low autocorrelation exclusion if enabled
        if exclude_low_acorr and y_full[1] < acorr_threshold:
            continue  # Skip this trace if acorr[1] is below the threshold

        # Track this trace as valid
        filtered_acorr.append(y_full)
        valid_indices.append(i)

        for trim in [False, True]:
            x = x_full[2:] if trim else x_full
            y = y_full[2:] if trim else y_full

            # skip if resulting trimmed array is too short
            if len(y) < 5:
                continue

            entry = {'index': i, 'exclude_first_two': trim}

            # --- Single exponential fit ---
            try:
                popt_s, _ = curve_fit(single_exp, x, y, p0=(1, 1, 0), maxfev=10000)
                y_pred_s = single_exp(x, *popt_s)
                r2_s = compute_r2(y, y_pred_s)
                bic_s = compute_bic(y, y_pred_s, num_params=3)
                entry_single = {
                    **entry,
                    'type': 'single_exp',
                    'params': dict(zip(['a', 'tau', 'c'], popt_s)),
                    'r2': r2_s,
                    'bic': bic_s
                }
            except RuntimeError:
                entry_single = {
                    **entry,
                    'type': 'single_exp',
                    'params': {'a': np.nan, 'tau': np.nan, 'c': np.nan},
                    'r2': np.nan,
                    'bic': np.nan
                }

            results.append(entry_single)

            # --- Double exponential fit ---
            try:
                popt_d, _ = curve_fit(double_exp, x, y, p0=(1, 1, 0.5, 5, 0), maxfev=10000)
                y_pred_d = double_exp(x, *popt_d)
                r2_d = compute_r2(y, y_pred_d)
                bic_d = compute_bic(y, y_pred_d, num_params=5)
                entry_double = {
                    **entry,
                    'type': 'double_exp',
                    'params': dict(zip(['a1', 'tau1', 'a2', 'tau2', 'c'], popt_d)),
                    'r2': r2_d,
                    'bic': bic_d
                }
            except RuntimeError:
                entry_double = {
                    **entry,
                    'type': 'double_exp',
                    'params': {'a1': np.nan, 'tau1': np.nan, 'a2': np.nan, 'tau2': np.nan, 'c': np.nan},
                    'r2': np.nan,
                    'bic': np.nan
                }

            results.append(entry_double)

    return results, filtered_acorr, valid_indices


# %%
exclude_low_acorr = True  # Toggle exclusion on or off
acorr_threshold = 0.1     # Set the threshold for acorr[1]

# Fit the model with exclusion based on acorr[1]
fit_results, filtered_acorr, valid_indices = fit_exponentials_nested(acorr, exclude_low_acorr, acorr_threshold)


# %%
# Create empty lists for each permutation
single_inc = []
single_exc = []
double_inc = []
double_exc = []

for result in fit_results:
    idx = result['index']
    exclude = result['exclude_first_two']
    r2 = result['r2']
    bic = result['bic']
    
    if result['type'] == 'single_exp':
        tau = result['params']['tau']
        entry = {'index': idx, 'tau': tau, 'r2': r2, 'bic': bic}
        if exclude:
            single_exc.append(entry)
        else:
            single_inc.append(entry)

    elif result['type'] == 'double_exp':
        tau1 = result['params']['tau1']
        tau2 = result['params']['tau2']
        entry = {'index': idx, 'tau1': tau1, 'tau2': tau2, 'r2': r2, 'bic': bic}
        if exclude:
            double_exc.append(entry)
        else:
            double_inc.append(entry)

# Convert to DataFrames (optional)
df_single_inc = pd.DataFrame(single_inc)
df_single_exc = pd.DataFrame(single_exc)
df_double_inc = pd.DataFrame(double_inc)
df_double_exc = pd.DataFrame(double_exc)

# Display a preview
print("Single Exp (include first 2):")
print(df_single_inc.head())

print("\nSingle Exp (exclude first 2):")
print(df_single_exc.head())

print("\nDouble Exp (include first 2):")
print(df_double_inc.head())

print("\nDouble Exp (exclude first 2):")
print(df_double_exc.head())
# %%
# bins=np.arange(5,120,6)
bins=np.arange(0.5,4,.1)


plt.figure()
plt.hist(df_single_exc.tau/30,bins)


plt.figure()

plt.hist(df_single_inc.tau/30,bins)
plt.figure()

plt.hist(df_double_exc.tau2/30,bins)
plt.figure()

plt.hist(df_double_inc.tau2/30,bins)

plt.figure()

plt.hist(taus_all,bins)

# %%


r2_indices = [i for i, val in enumerate(df_double_inc.r2) if val > 0.9]

overlap = [x for x in r2_indices if x in valid_indices]

subset = [keys[i] for i in valid_indices]



# you can place a r2 cutoff here

# %%

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

non_stimd_roi_keys_sig=v1_avgs['roi_keys']

# tau_entries = (spont_timescales.TwopTau & non_stimd_roi_keys_sig).fetch('tau', 'KEY')

# is_good_tau = np.array((spont_timescales.TwopTauInclusion & non_stimd_roi_keys_sig).fetch('is_good_tau_roi'))

# keys_with_taus=tau_entries[1]


keys_to_remove = ['ts_param_set_id', 'corr_param_set_id', 'glm_param_set_id']

# Create a new list with those keys removed
longer_list_cleaned = [
    {k: v for k, v in d.items() if k not in keys_to_remove}
    for d in subset
    ]

overlap_indices = [i for i, d in enumerate(longer_list_cleaned) if d in non_stimd_roi_keys_sig]

# peak_ts     = deepcopy(opto_data['max_or_min_times_sec'])

# peaktimes=np.array(m2_avgs['peak_times_sec'])

# ae=peaktimes[match_indices]

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



indices_list1, indices_list2 = find_matching_indices(longer_list_cleaned, non_stimd_roi_keys_sig)
print(f"Matching indices in list1: {indices_list1}")
print(f"Matching indices in list2: {indices_list2}")
# %%
a=np.array(df_single_exc.tau[indices_list1]/30)


peaktimes=np.array(m2_avgs['peak_times_sec'])
b=peaktimes[indices_list2]