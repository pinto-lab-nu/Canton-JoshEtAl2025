# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 20:27:06 2025

@author: jec822
"""

from scipy.optimize import curve_fit
import pandas as pd

import analyzeSpont2P
import spont_timescales

# %%
area = "V1"
params = opto_params
dff_type = "residuals_dff"

keys_V1 = analyzeSpont2P.get_twop_tau_keys(area, params, dff_type)
keys_M2 = analyzeSpont2P.get_twop_tau_keys(area='M2', params=opto_params, dff_type="residuals_dff")


# %%
acorr_M2,lags_M2,k_M2  = (spont_timescales.TwopAutocorr & keys_M2).fetch('autocorr_vals','lags_sec','KEY')
acorr_V1,lags_V1,k_V1  = (spont_timescales.TwopAutocorr & keys_V1).fetch('autocorr_vals','lags_sec','KEY')


# %%
def single_exp(x, a, tau, c):
    return a * np.exp(-x / tau) + c

def double_exp(x, a1, tau1, a2, tau2, c):
    return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c

# Define model functions
def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def compute_bic(y_true, y_pred, num_params):
    n = len(y_true)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    if residual_sum_of_squares == 0:
        return -np.inf  # perfect fit
    return n * np.log(residual_sum_of_squares / n) + num_params * np.log(n)

def fit_exponentials_nested(acorr, keys, exclude_low_acorr=False, acorr_threshold=0.1, max_lag=None):
    results = []
    filtered_acorr = []
    valid_indices = []

    for i, y_full in enumerate(acorr):
        y_full = np.array(y_full)
        x_full = np.arange(len(y_full))

        # Skip if invalid: empty, NaNs, all zeros, or constant
        if len(y_full) == 0 or np.all(np.isnan(y_full)) or np.all(y_full == 0) or np.all(y_full == y_full[0]):
            continue

        acorr1_val = y_full[1] - np.nanmin(y_full) if len(y_full) > 1 else np.nan

        if exclude_low_acorr and acorr1_val < acorr_threshold:
            continue

        filtered_acorr.append(y_full)
        valid_indices.append(i)

        for trim in [False, True]:
            x = x_full[2:] if trim else x_full
            y = y_full[2:] if trim else y_full

            if max_lag is not None:
                x = x[:max_lag]
                y = y[:max_lag]

            if len(y) < 5:
                continue

            entry = {
                'index': i,
                'exclude_first_two': trim,
                'acorr1': acorr1_val
            }

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

    subset = [keys[i] for i in valid_indices]

    return results, filtered_acorr, valid_indices, subset

# %%


# %%

exclude_low_acorr = True  # Toggle exclusion on or off
acorr_threshold = 0.0    # Set the threshold for acorr[1]

# Fit the model with exclusion based on acorr[1]
fit_results_M2, filtered_acorr, valid_indices,subset_M2 = fit_exponentials_nested(acorr_M2,k_M2, exclude_low_acorr, 
                                                                                  acorr_threshold,max_lag=300)

# %%

fit_results_V1, filtered_acorr_V1, valid_indices_V1,subset_v1 = fit_exponentials_nested(acorr_V1,k_V1, exclude_low_acorr,
                                                                                        acorr_threshold,max_lag=300)

# %%

def classify_fit_results(fit_results, subset, snr_threshold=1.5, event_rate_threshold=0.5):

    # Fetch SNR and event rate arrays
    snr_array = (VM['twophoton'].Snr2P & subset).fetch('snr')
    event_rate_array = (VM['twophoton'].Snr2P & subset).fetch('events_per_min')

    # Check matching lengths
    assert len(snr_array) == len(event_rate_array), "Mismatched SNR and event rate lengths"

    # Apply thresholds
    valid_indices = (snr_array > snr_threshold) & (event_rate_array > event_rate_threshold)

    # Containers
    single_inc, single_exc = [], []
    double_inc, double_exc = [], []

    for result in fit_results:
        idx = result['index']
        exclude = result['exclude_first_two']
        r2 = result['r2']
        bic = result['bic']
        acorr1= result['acorr1']
        
        if result['type'] == 'single_exp':
            tau = result['params']['tau']
            entry = {'index': idx, 'tau': tau, 'r2': r2, 'bic': bic, 'acorr_1_index': acorr1}
            if exclude:
                single_exc.append(entry)
            else:
                single_inc.append(entry)

        elif result['type'] == 'double_exp':
            tau1 = result['params']['tau1']
            tau2 = result['params']['tau2']
            entry = {'index': idx, 'tau1': tau1, 'tau2': tau2, 'r2': r2, 'bic': bic,'acorr_1_index': acorr1}
            if exclude:
                double_exc.append(entry)
            else:
                double_inc.append(entry)

    
    # Convert lists to DataFrames
    df_single_inc = pd.DataFrame(single_inc)
    df_single_exc = pd.DataFrame(single_exc)
    df_double_inc = pd.DataFrame(double_inc)
    df_double_exc = pd.DataFrame(double_exc)
    
    # Add snr and event_rate based on 'index' column
    for df in [df_single_inc, df_single_exc, df_double_inc, df_double_exc]:
       if not df.empty:
           df['snr'] = df.index.map(lambda i: snr_array[i] if i < len(snr_array) else pd.NA)
           df['event_rate'] = df.index.map(lambda i: event_rate_array[i] if i < len(event_rate_array) else pd.NA)

    # Compare average R² across categories
    avg_r2 = {
        'single_inc': df_single_inc['r2'].mean() if not df_single_inc.empty else float('-inf'),
        'single_exc': df_single_exc['r2'].mean() if not df_single_exc.empty else float('-inf'),
        'double_inc': df_double_inc['r2'].mean() if not df_double_inc.empty else float('-inf'),
        'double_exc': df_double_exc['r2'].mean() if not df_double_exc.empty else float('-inf')
    }
    best_fit_type = max(avg_r2, key=avg_r2.get)

    return df_single_inc, df_single_exc, df_double_inc, df_double_exc, best_fit_type

# %%

df_si, df_se_M2, df_di_M2, df_de_M2, best_type = classify_fit_results(
    fit_results=fit_results_M2,
    subset=subset_M2,
    snr_threshold=-20,
    event_rate_threshold=-20
)

print(f"The best fitting model type based on average R² is: {best_type}")


# %%

df_si, df_se_V1, df_di_V1, df_de_V1, best_type = classify_fit_results(
    fit_results=fit_results_V1,
    subset=subset_v1,
    snr_threshold=-20,
    event_rate_threshold=-20
)

print(f"The best fitting model type based on average R² is: {best_type}")

# %%


df_se_V1 = df_se_V1[df_se_V1["tau"] <= 200]
df_se_M2 = df_se_M2[df_se_M2["tau"] <= 200]

df_se_V1 = df_se_V1[df_se_V1["tau"] >= 2]
df_se_M2 = df_se_M2[df_se_M2["tau"] >= 2]

df_se_V1 = df_se_V1[df_se_V1["r2"] >= 0.8]
df_se_M2 = df_se_M2[df_se_M2["r2"] >= 0.8]

# df_se_V1 = df_se_V1[df_se_V1["event_rate"] >= 0.1]
# df_se_M2 = df_se_M2[df_se_M2["event_rate"] >= 0.1]

# df_se_V1 = df_se_V1[df_se_V1["snr"] >= 1]
# df_se_M2 = df_se_M2[df_se_M2["snr"] >= 1]

df_se_V1 = df_se_V1[df_se_V1["acorr_1_index"] >= .2]
df_se_M2 = df_se_M2[df_se_M2["acorr_1_index"] >= .2]
# %%

plot_ecdf_comparison(df_se_M2.tau/30,df_se_V1.tau/30,log_x=True)

# %%



df_di_V1 = df_di_V1[df_di_V1["tau2"] <= 100]
df_di_M2 = df_di_M2[df_di_M2["tau2"] <= 100]

df_di_V1 = df_di_V1[df_di_V1["tau2"] >= 2]
df_di_M2 = df_di_M2[df_di_M2["tau2"] >= 2]

df_di_V1 = df_di_V1[df_di_V1["r2"] >= 0.8]
df_di_M2 = df_di_M2[df_di_M2["r2"] >= 0.8]

# df_de_V1 = df_de_V1[df_de_V1["event_rate"] >= 0.1]
# df_de_M2 = df_de_M2[df_de_M2["event_rate"] >= 0.1]

# df_de_V1 = df_de_V1[df_de_V1["snr"] >= 1]
# df_de_M2 = df_de_M2[df_de_M2["snr"] >= 1]

df_di_V1 = df_di_V1[df_di_V1["acorr_1_index"] >= .2]
df_di_M2 = df_di_M2[df_di_M2["acorr_1_index"] >= .2]
# %%

plot_ecdf_comparison(df_di_M2.tau2/30,df_di_V1.tau2/30,log_x=True)
# %%

# plot_ecdf_comparison(df_de_M2.event_rate,df_se_V1.event_rate,log_x=True)
