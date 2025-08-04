# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:00:35 2025

@author: jec822
"""

# %%

def bootstrap_population_average(data, sample_size=None, n_samples=1000):
    N, T = data.shape  # N = total neurons, T = timepoints
    
    if sample_size is None:
        sample_size = N  # default: use all neurons

    bootstrap_averages = []

    for _ in range(n_samples):
        sample_indices = np.random.choice(N, sample_size, replace=True)
        sample = data[sample_indices]
        bootstrap_avg = sample.mean(axis=0)
        bootstrap_averages.append(bootstrap_avg)

    return np.array(bootstrap_averages)  # Shape: (n_samples, T)

# %%

def plot_acorr_with_fit_dual(
    acorr_trace_1d_1,
    fit_results_1,
    index_in_df_1,
    acorr_trace_1d_2=None,
    fit_results_2=None,
    index_in_df_2=None,
    fit_type="best",  # 'single_exp', 'double_exp', or 'best'
    sampling_interval=None,
    xlim=None,
    label_1="Dataset 1",
    label_2="Dataset 2",
    color_1="black",
    color_2="gray",
    fig_size=(8, 4)
):
    import matplotlib.pyplot as plt
    import numpy as np

    def get_fit_and_label(fit_results, index, x, sampling_interval, tau_unit):
        fits = [fit for fit in fit_results if fit['index'] == index]
        if not fits:
            return None, None, "No fit"

        if fit_type == "best":
            fit = sorted(fits, key=lambda f: (-f['r2'], f['type'] == 'double_exp'))[0]
        else:
            selected = [f for f in fits if f['type'] == fit_type]
            if not selected:
                return None, None, f"No {fit_type} fit"
            fit = selected[0]

        if fit['type'] == 'single_exp':
            p = fit['params']
            tau = p['tau'] * sampling_interval if sampling_interval else p['tau']
            y_fit = single_exp(x, p['a'], tau, p['c'])
            label = f"Single exp (τ = {tau:.2f} {tau_unit})"
        else:
            p = fit['params']
            tau1 = p['tau1'] * sampling_interval if sampling_interval else p['tau1']
            tau2 = p['tau2'] * sampling_interval if sampling_interval else p['tau2']
            y_fit = double_exp(x, p['a1'], tau1, p['a2'], tau2, p['c'])
            label = f"Double exp (τ₁ = {tau1:.2f}, τ₂ = {tau2:.2f} {tau_unit})"

        return y_fit, label, fit['type']

    # --- Prepare time axis ---
    x_frames_1 = np.arange(len(acorr_trace_1d_1))
    x_frames_2 = np.arange(len(acorr_trace_1d_2)) if acorr_trace_1d_2 is not None else None

    if sampling_interval:
        x_1 = x_frames_1 * sampling_interval
        x_2 = x_frames_2 * sampling_interval if x_frames_2 is not None else None
        tau_unit = "s"
        xlim = tuple(np.array(xlim) * sampling_interval) if xlim else None
    else:
        x_1 = x_frames_1
        x_2 = x_frames_2
        tau_unit = "frames"

    # --- Begin plot ---
    plt.figure(figsize=fig_size)

    # Dataset 1
    plt.plot(x_1, acorr_trace_1d_1, label=label_1, color=color_1, lw=2)
    fit_y1, fit_label1, _ = get_fit_and_label(fit_results_1, index_in_df_1, x_1, sampling_interval, tau_unit)
    if fit_y1 is not None:
        plt.plot(x_1, fit_y1, linestyle="--", color=color_1, alpha=0.6, label=fit_label1)

    # Dataset 2 (optional)
    if acorr_trace_1d_2 is not None and fit_results_2 is not None and index_in_df_2 is not None:
        plt.plot(x_2, acorr_trace_1d_2, label=label_2, color=color_2, lw=2)
        fit_y2, fit_label2, _ = get_fit_and_label(fit_results_2, index_in_df_2, x_2, sampling_interval, tau_unit)
        if fit_y2 is not None:
            plt.plot(x_2, fit_y2, linestyle="--", color=color_2, alpha=0.6, label=fit_label2)

    plt.xlabel(f"Lag ({tau_unit})")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation with Exponential Fit(s)")
    if xlim:
        plt.xlim(xlim)
    plt.legend()
    plt.tight_layout()
    plt.show()


# %%
# sample_size=50
# n_samples=50
start_idx=100
signal_range=(0, 288)
max_lag=280
fit_max_lag=250

# %%

# Using direct from dj
# bootstrap_avgs_m2=bootstrap_population_average(m2_avgs['trig_dff_avgs'], sample_size=10,n_samples=1)


# Using newer pipeline from dataframes

# Find the minimum length among all arrays
min_len = min(len(arr) for arr in result_M2_standard_non_stimd_filt['averaged_traces_all'])
# Truncate all arrays to that length
m2_avgs_array_2d = np.vstack([arr[:min_len] for arr in result_M2_standard_non_stimd_filt['averaged_traces_all']])
bootstrap_avgs_m2=bootstrap_population_average(m2_avgs_array_2d, sample_size=10,n_samples=1)


bootstrap_avgs_m22 = pd.DataFrame(bootstrap_avgs_m2.tolist())

bootstrap_df_short_m2 = bootstrap_avgs_m22.iloc[:, start_idx:]

# bootstrap_df_short_m2 = (bootstrap_df_short_m2.subtract(bootstrap_df_short_m2.min(axis=1), axis=0)
#                               .divide(bootstrap_df_short_m2.max(axis=1) - bootstrap_df_short_m2.min(axis=1), axis=0))


bootstrap_mean_trace_m2 = pd.DataFrame(bootstrap_df_short_m2.mean(axis=0, skipna=True)).T
# Normalize to [0, 1]
# min_val = bootstrap_mean_trace_m2.min(axis=1).values[:, None]
# max_val = bootstrap_mean_trace_m2.max(axis=1).values[:, None]
# bootstrap_mean_trace_m2 = (bootstrap_mean_trace_m2 - min_val) / (max_val - min_val)

# bootstrap_mean_trace_m2 -= bootstrap_mean_trace_m2.min(axis=1).values[:, None]

acorr_df_m2 = calculate_autocorrelations_df(bootstrap_mean_trace_m2, signal_range=signal_range, max_lags=max_lag)
fit_results_m2, _, _ = fit_exponentials_from_df(
    acorr_df_m2, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=fit_max_lag,bounded_fit=True
)

# fit_results_m2 = fit_results_m2[2:]
# %%
# --- V1 ---
# bootstrap_avgs_v1=pd.DataFrame(bootstrap_population_average(v1_avgs['trig_dff_avgs'], sample_size=10,n_samples=1))
# bootstrap_avgs_v1=pd.DataFrame(v1_avgs['trig_dff_avgs'])

# Using newer pipeline from dataframes

# Find the minimum length among all arrays
min_len = min(len(arr) for arr in result_V1_standard_non_stimd_filt['averaged_traces_all'])
# Truncate all arrays to that length
v1_avgs_array_2d = np.vstack([arr[:min_len] for arr in result_V1_standard_non_stimd_filt['averaged_traces_all']])
bootstrap_avgs_v1=bootstrap_population_average(v1_avgs_array_2d, sample_size=10,n_samples=1)


bootstrap_avgs_v11 = pd.DataFrame(bootstrap_avgs_v1.tolist())
bootstrap_df_short_v1 = bootstrap_avgs_v11.iloc[:, start_idx:]

# bootstrap_df_short_v1= (bootstrap_df_short_v1.subtract(bootstrap_df_short_v1.min(axis=1), axis=0)
                              # .divide(bootstrap_df_short_v1.max(axis=1) - bootstrap_df_short_v1.min(axis=1), axis=0))

bootstrap_mean_trace_v1 = pd.DataFrame(bootstrap_df_short_v1.mean(axis=0, skipna=True)).T
# bootstrap_mean_trace_v1 -= bootstrap_mean_trace_v1.min(axis=1).values[:, None]


acorr_df_v1 = calculate_autocorrelations_df(bootstrap_mean_trace_v1, signal_range=signal_range, max_lags=max_lag)

fit_results_v1, _, _ = fit_exponentials_from_df(
    acorr_df_v1, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=fit_max_lag,bounded_fit=True
)


# fit_results_v1 = fit_results_v1[2:]
# 
# %%

plot_acorr_with_fit_dual(
    acorr_trace_1d_1=acorr_df_m2.iloc[0],
    fit_results_1=fit_results_m2,
    index_in_df_1=acorr_df_m2.index[0],
    acorr_trace_1d_2=acorr_df_v1.iloc[0],
    fit_results_2=fit_results_v1,
    index_in_df_2=acorr_df_v1.index[0],
    sampling_interval=sampling_interval,
    label_1="M2",
    label_2="V1",
    color_1=params['general_params']['M2_cl'],
    color_2=params['general_params']['V1_cl'],
    fig_size=(4, 4),
    fit_type="single_exp"
)

# %%
# bootstrap_avgs_v1=bootstrap_population_average(v1_avgs_array_2d, sample_size=100,n_samples=100)

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bootstrap_acorr_analysis(
    data,
    sample_size=50,
    n_samples=50,
    start_idx=100,
    signal_range=(0, 288),
    max_lag=280,
    fit_max_lag=250,
    tau_bounds=(2, 200),
    r2_threshold=0.9,
    acorr_1_threshold=0.2,
    plot_title="Mean Autocorrelation",
    label="",
):
    # Step 1: Bootstrap
    bootstrap_data = bootstrap_population_average(data, sample_size=sample_size, n_samples=n_samples)
    
    # Step 2: Slice time range and convert to DataFrame
    bootstrap_df = pd.DataFrame(bootstrap_data.tolist())
    bootstrap_df_short = bootstrap_df.iloc[:, start_idx:]
    
    # bootstrap_df_short = pd.DataFrame(bootstrap_df_short.mean(axis=0, skipna=True)).T

    
    # Step 3: Autocorrelation
    acorr_df = analyzeSpont2P.calculate_autocorrelations_df(bootstrap_df_short, signal_range=signal_range, max_lags=max_lag)
    
    # Step 4: Plot mean autocorrelation
    # mean_acorr = acorr_df.mean(axis=0, skipna=True)
    # plt.figure()
    # plt.plot(mean_acorr)
    # plt.xlabel('Lag')
    # plt.ylabel('Mean autocorrelation')
    # plt.title(f'{plot_title} - {label}')
    
    # Step 5: Fit exponentials
    fit_results, good_acorr, valid_rows = analyzeSpont2P.fit_exponentials_from_df(
        acorr_df, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=fit_max_lag,
        bounded_fit=True,tau_bounds=(0,300)
    )
    
    # Step 6: Classify results
    df_si, df_se, df_di, df_de, best_type = analyzeSpont2P.classify_fit_results_simple(fit_results)
    
    # Step 7: Filter results
    df_se = df_se[
        (df_se["tau"] >= tau_bounds[0]) &
        (df_se["tau"] <= tau_bounds[1]) &
        (df_se["r2"] >= r2_threshold) &
        (df_se["acorr_1_index"] >= acorr_1_threshold)
    ]
    # Step 7: Filter results
    df_di = df_di[
        (df_di["tau2"] >= tau_bounds[0]) &
        (df_di["tau2"] <= tau_bounds[1]) &
        (df_di["r2"] >= r2_threshold) &
        (df_di["acorr_1_index"] >= acorr_1_threshold)
    ]
    
    return df_se, best_type

# %%

df_se_m2, best_type_m2 = bootstrap_acorr_analysis(
    data=m2_avgs_array_2d, 
    sample_size=10, 
    n_samples=50,
    label="M2",
    # bounded_fit=True
    )

df_se_v1, best_type_v1 = bootstrap_acorr_analysis(
    data=v1_avgs_array_2d, 
    sample_size=10, 
    n_samples=50,
    label="V1",
    # bounded_fit=True
    )

# Compare ECDFs
analysis_plotting_functions.plot_ecdf_comparison(df_se_v1.tau*sampling_interval,df_se_m2.tau*sampling_interval,
                      label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                      line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                      xlabel="Population Taus",
                      ylabel="Cum. prop. bootstraped expts",
                      xticks_start=0, xticks_end=4, xticks_step=2,
                      xlim=[0,4],
                      stat_test='auto',
                      figsize=[3,4],
                      show_normality_pvals=True)

# %%

from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd

def compare_tau_distributions_vs_sample_size(
    v1_data,
    m2_data,
    sample_sizes,
    n_samples=100,
    start_idx=100,
    signal_range=(0, 288),
    max_lag=280,
    fit_max_lag=250,
    tau_bounds=(2, 200),
    r2_threshold=0.8,
    acorr_1_threshold=0.2,
):
    results = []

    for sample_size in sample_sizes:
        print(f"Analyzing sample size: {sample_size}")

        # Run bootstrap acorr and fit for both V1 and M2
        df_se_v1, _ = bootstrap_acorr_analysis(
            v1_data, sample_size=sample_size, n_samples=n_samples,
            start_idx=start_idx, signal_range=signal_range, max_lag=max_lag,
            fit_max_lag=fit_max_lag, tau_bounds=tau_bounds,
            r2_threshold=r2_threshold, acorr_1_threshold=acorr_1_threshold,
            label="V1"
        )
        df_se_m2, _ = bootstrap_acorr_analysis(
            m2_data, sample_size=sample_size, n_samples=n_samples,
            start_idx=start_idx, signal_range=signal_range, max_lag=max_lag,
            fit_max_lag=fit_max_lag, tau_bounds=tau_bounds,
            r2_threshold=r2_threshold, acorr_1_threshold=acorr_1_threshold,
            label="M2"
        )

        taus_v1 = df_se_v1["tau"].values
        taus_m2 = df_se_m2["tau"].values

        # Only proceed if both groups have enough data
        if len(taus_v1) > 5 and len(taus_m2) > 5:
            median_diff = np.median(taus_m2) - np.median(taus_v1)
            stat, pval = mannwhitneyu(taus_m2, taus_v1, alternative='two-sided')
        else:
            median_diff, pval = np.nan, np.nan

        results.append({
            "sample_size": sample_size,
            "median_tau_v1": np.median(taus_v1) if len(taus_v1) > 0 else np.nan,
            "median_tau_m2": np.median(taus_m2) if len(taus_m2) > 0 else np.nan,
            "median_diff": median_diff,
            "p_value": pval,
            "n_v1": len(taus_v1),
            "n_m2": len(taus_m2)
        })

    return pd.DataFrame(results)
# %%
# sample_sizes = [1,2,3,4,5, 10, 25, 50, 100, 200]
# results_df = compare_tau_distributions_vs_sample_size(
#     v1_data=v1_avgs['trig_dff_avgs'],
#     m2_data=m2_avgs['trig_dff_avgs'],
#     sample_sizes=sample_sizes,
#     n_samples=170
# )
# %%
from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd
from itertools import product

def cliffs_delta(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx = len(x)
    ny = len(y)
    greater = sum(xi > yj for xi, yj in product(x, y))
    less = sum(xi < yj for xi, yj in product(x, y))
    return (greater - less) / (nx * ny)

def compare_tau_distributions_vs_sample_size_repeat(
    v1_data,
    m2_data,
    sample_sizes,
    n_samples=100,
    n_repeats=10,
    start_idx=100,
    signal_range=(0, 288),
    max_lag=280,
    fit_max_lag=250,
    tau_bounds=(2, 200),
    r2_threshold=0.8,
    acorr_1_threshold=0.2,
):
    all_results = []

    for sample_size in sample_sizes:
        print(f"\nSample size: {sample_size}")
        for repeat_id in range(n_repeats):
            print(f"  Repeat {repeat_id+1}/{n_repeats}")

            # Run bootstrap acorr and fit for both V1 and M2
            df_se_v1, _ = bootstrap_acorr_analysis(
                v1_data, sample_size=sample_size, n_samples=n_samples,
                start_idx=start_idx, signal_range=signal_range, max_lag=max_lag,
                fit_max_lag=fit_max_lag, tau_bounds=tau_bounds,
                r2_threshold=r2_threshold, acorr_1_threshold=acorr_1_threshold,
                label="V1"
            )
            df_se_m2, _ = bootstrap_acorr_analysis(
                m2_data, sample_size=sample_size, n_samples=n_samples,
                start_idx=start_idx, signal_range=signal_range, max_lag=max_lag,
                fit_max_lag=fit_max_lag, tau_bounds=tau_bounds,
                r2_threshold=r2_threshold, acorr_1_threshold=acorr_1_threshold,
                label="M2"
            )

            taus_v1 = df_se_v1["tau"].values
            taus_m2 = df_se_m2["tau"].values

            # Compute metrics only if enough data
            if len(taus_v1) > 5 and len(taus_m2) > 5:
                median_tau_v1 = np.median(taus_v1)
                median_tau_m2 = np.median(taus_m2)
                median_diff = median_tau_m2 - median_tau_v1
                pval = mannwhitneyu(taus_m2, taus_v1, alternative='two-sided').pvalue
                d_cliff = cliffs_delta(taus_m2, taus_v1)
            else:
                median_tau_v1 = np.nan
                median_tau_m2 = np.nan
                median_diff = np.nan
                pval = np.nan
                d_cliff = np.nan

            all_results.append({
                "sample_size": sample_size,
                "repeat_id": repeat_id,
                "median_tau_v1": median_tau_v1,
                "median_tau_m2": median_tau_m2,
                "median_diff": median_diff,
                "p_value": pval,
                "cliffs_delta": d_cliff,
                "n_v1": len(taus_v1),
                "n_m2": len(taus_m2)
            })

    return pd.DataFrame(all_results)

# %%

# sample_sizes = [1,2,3,4,5, 10, 25, 50, 100, 200]

sample_sizes = [1,5, 10, 25, 50, 100]

results_df_se = compare_tau_distributions_vs_sample_size_repeat(
    v1_data=v1_avgs_array_2d,
    m2_data=m2_avgs_array_2d,
    sample_sizes=sample_sizes,
    n_samples=100,
    n_repeats=50
)
# %%

results_df_di = compare_tau_distributions_vs_sample_size_repeat(
    v1_data=v1_avgs_array_2d,
    m2_data=m2_avgs_array_2d,
    sample_sizes=sample_sizes,
    n_samples=100,
    n_repeats=50
)

# %%
def plot_pvalue_vs_sample_size(
    results_df,
    significance_level=0.05,
    fig_size=(8, 5),
    title="–log10(p-value) vs. Sample Size",
    xlabel="Sample Size",
    ylabel="-log10(p-value)",
    color="gray"
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    df = results_df.copy()
    df = df[df["p_value"].notna()]
    df["neg_log10_p"] = -np.log10(df["p_value"])

    plt.figure(figsize=fig_size)
    ax = sns.violinplot(
        x="sample_size",
        y="neg_log10_p",
        data=df,
        inner="point",
        scale="width",
        linewidth=1,
        color=color,
        alpha=0.5
    )

    # Add horizontal threshold line (optional)
    if significance_level:
        threshold = -np.log10(significance_level)
        plt.axhline(threshold, color='red', linestyle='--', label=f'p = {significance_level}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine(top=True, right=True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
def plot_effects_vs_sample_size(results_df):
    df = results_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    # Median difference
    sns.boxplot(x="sample_size", y="median_diff", data=df, ax=axes[0])
    axes[0].set_title("Median Tau Difference (M2 - V1)")
    axes[0].set_xlabel("Sample Size")
    axes[0].set_ylabel("Tau Difference")

    # Cliff's delta
    sns.boxplot(x="sample_size", y="cliffs_delta", data=df, ax=axes[1])
    axes[1].set_title("Cliff's Delta")
    axes[1].set_xlabel("Sample Size")
    axes[1].set_ylabel("Effect Size (d)")

    plt.tight_layout()
    plt.show()
    
# %%
def plot_median_taus_vs_sample_size(
    results_df,
    fig_size=(8, 5),
    ylabel="Median Tau",
    xlabel="Sample Size",
    title="Median Tau vs. Sample Size",
    colors={"V1": "#1f77b4", "M2": "#ff7f0e"},
    significance_marker=True,
    alpha_thresholds=[0.05, 0.01, 0.001],
    marker_y_offset=0.02,
    sampling_interval=None  # in seconds, e.g., 1/30 if 30 Hz
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import mannwhitneyu

    df = results_df.copy()
    df = df[["sample_size", "repeat_id", "median_tau_v1", "median_tau_m2"]].dropna()

    if sampling_interval is not None:
        df["median_tau_v1"] *= sampling_interval
        df["median_tau_m2"] *= sampling_interval
        if ylabel == "Median Tau":  # auto-update only if not customized
            ylabel = "Median Tau (s)"

    df_melted = df.melt(
        id_vars=["sample_size", "repeat_id"],
        value_vars=["median_tau_v1", "median_tau_m2"],
        var_name="region",
        value_name="median_tau"
    )

    df_melted["region"] = df_melted["region"].replace({
        "median_tau_v1": "V1",
        "median_tau_m2": "M2"
    })

    plt.figure(figsize=fig_size)
    ax = sns.boxplot(
        x="sample_size",
        y="median_tau",
        hue="region",
        data=df_melted,
        palette=colors
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    sns.despine(top=True, right=True)
    plt.legend(title="Region")

    # Significance markers
    if significance_marker:
        y_max = df_melted["median_tau"].max()
        height_offset = y_max * marker_y_offset

        for i, sample_size in enumerate(sorted(df["sample_size"].unique())):
            taus_v1 = df_melted[
                (df_melted["sample_size"] == sample_size) &
                (df_melted["region"] == "V1")
            ]["median_tau"].values

            taus_m2 = df_melted[
                (df_melted["sample_size"] == sample_size) &
                (df_melted["region"] == "M2")
            ]["median_tau"].values

            if len(taus_v1) > 5 and len(taus_m2) > 5:
                p = mannwhitneyu(taus_m2, taus_v1, alternative='two-sided').pvalue

                if p < alpha_thresholds[2]:
                    marker = "***"
                elif p < alpha_thresholds[1]:
                    marker = "**"
                elif p < alpha_thresholds[0]:
                    marker = "*"
                else:
                    continue

                y_pos = max(taus_v1.max(), taus_m2.max())
                ax.text(i, y_pos + height_offset, marker,
                        ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.show()


# %%
plot_pvalue_vs_sample_size(results_df_se, significance_level=0.05,fig_size=(5, 3),
                           xlabel="Number of sampled neurons")
# %%

plot_median_taus_vs_sample_size(
    results_df_se,
    fig_size=(5, 3),
    ylabel="Tau (seconds)",
    xlabel="Number of sampled neurons",
    title="V1 vs M2: Median Tau vs. Sample Size",
    colors={"V1": params['general_params']['V1_cl'], "M2": params['general_params']['M2_cl']},
    sampling_interval=sampling_interval
)