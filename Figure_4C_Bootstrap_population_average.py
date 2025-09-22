# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:00:35 2025

@author: jec822
"""


def bootstrap_population_average(data, sample_size=None, n_samples=1000):
    N, T = data.shape
    
    if sample_size is None:
        sample_size = N  # default: use all neurons

    # Pre-generate all bootstrap indices: shape (n_samples, sample_size)
    idx = np.random.randint(0, N, size=(n_samples, sample_size))

    # Gather all samples in one shot: shape (n_samples, sample_size, T)
    sampled = data[idx]

    # Average across sample dimension → (n_samples, T)
    bootstrap_averages = sampled.mean(axis=1)

    return bootstrap_averages

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

    def single_exp(x, a, tau, c):
        return a * np.exp(-x / tau) + c

    def double_exp(x, a1, tau1, a2, tau2, c):
        return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c

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

        r2 = fit['r2']

        if fit['type'] == 'single_exp':
            p = fit['params']
            tau = p['tau'] * sampling_interval if sampling_interval else p['tau']
            y_fit = single_exp(x, p['a'], tau, p['c'])
            label = f"Single exp (τ = {tau:.2f} {tau_unit}, R² = {r2:.4f})"
        else:
            p = fit['params']
            tau1 = p['tau1'] * sampling_interval if sampling_interval else p['tau1']
            tau2 = p['tau2'] * sampling_interval if sampling_interval else p['tau2']
            y_fit = double_exp(x, p['a1'], tau1, p['a2'], tau2, p['c'])
            label = f"Double exp (τ₁ = {tau1:.2f}, τ₂ = {tau2:.2f} {tau_unit}, R² = {r2:.4f})"

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
def plot_pvalue_vs_sample_size(
    results_df,
    significance_level=0.05,
    fig_size=(8, 5),
    title="–log10(p-value) vs. Sample Size",
    xlabel="Sample Size",
    ylabel="-log10(p-value)",
    color="gray",
    p_stat="p_value_perm_test"
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    df = results_df.copy()
    df = df[df[p_stat].notna()]
    df["neg_log10_p"] = -np.log10(df[p_stat])

    plt.figure(figsize=fig_size)
    
    # Violin for distribution
    sns.violinplot(
        x="sample_size",
        y="neg_log10_p",
        data=df,
        inner=None,  # don't plot points inside violin
        scale="width",
        linewidth=1,
        color=color,
        alpha=0.5
    )

    # Stripplot for individual points with jitter
    sns.stripplot(
        x="sample_size",
        y="neg_log10_p",
        data=df,
        color="gray",
        size=3,
        jitter=0,   # horizontal jitter to separate points
        alpha=0.5
    )

    # Threshold line
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
    import pandas as pd
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

    # ---- NEW: compute means and SEMs ----
    summary_table = (
        df_melted.groupby(["sample_size", "region"])["median_tau"]
        .agg(["mean", "std"])
        .reset_index()
    )
    print("\n=== Means and SEMs by Sample Size and Region ===")
    print(summary_table.to_string(index=False))

    # ---- plotting (unchanged) ----
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


import time

def bootstrap_acorr_analysis(
    data,
    sample_size=50,
    n_samples=50,
    start_idx=100,
    signal_range=(0, 288),
    max_lag=280,
    fit_max_lag=250,
    tau_bounds=(2, 400),
    r2_threshold=0.9,
    acorr_1_threshold=0.2,
    plot_title="Mean Autocorrelation",
    label="",
):
    # Step 1: Vectorized bootstrap
    # t0 = time.time()
    bootstrap_data = bootstrap_population_average(data, sample_size=sample_size, n_samples=n_samples)
    # t1 = time.time()
    # print(f"Bootstrap time: {t1 - t0:.3f} s")

    # Step 2: Slice time range (NumPy)
    # t0 = time.time()
    bootstrap_short = bootstrap_data[:, start_idx:]
    # t1 = time.time()
    # print(f"Slicing time: {t1 - t0:.3f} s")

    # Step 3: Autocorrelation
    # t0 = time.time()
    acorr_df = analyzeSpont2P.calculate_autocorrelations_df(
        bootstrap_short, signal_range=signal_range, max_lags=max_lag
    )
    # t1 = time.time()
    # print(f"Autocorrelation time: {t1 - t0:.3f} s")

    # Step 4: Fit exponentials
    # t0 = time.time()
    fit_results, _, _ = analyzeSpont2P.fit_exponentials_from_df(
        acorr_df,
        exclude_low_acorr=True,
        acorr_threshold=0.1,
        max_lag=fit_max_lag,
        bounded_fit=True,
        tau_bounds=tau_bounds
    )
    # t1 = time.time()
    # print(f"Exponential fit time: {t1 - t0:.3f} s")

    # Step 5: Classify
    # t0 = time.time()
    df_si, df_se, df_di, df_de, best_type = analyzeSpont2P.classify_fit_results_simple(fit_results)
    # t1 = time.time()
    # print(f"Classification time: {t1 - t0:.3f} s")

    # Step 6: Filter
    # t0 = time.time()
    df_se = df_se[
        df_se["tau"].between(*tau_bounds) &
        (df_se["r2"] >= r2_threshold) &
        (df_se["acorr_1_index"] >= acorr_1_threshold)
    ]
    df_di = df_di[
        df_di["tau2"].between(*tau_bounds) &
        (df_di["r2"] >= r2_threshold) &
        (df_di["acorr_1_index"] >= acorr_1_threshold)
    ]
    # t1 = time.time()
    # print(f"Filtering time: {t1 - t0:.3f} s")

    return df_se, best_type


# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bootstrap_trace_decay_analysis(
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
    plot_title="",
    label="",
):
    # Step 1: Bootstrap
    bootstrap_data = bootstrap_population_average(data, sample_size=sample_size, n_samples=n_samples)
    
    # Step 2: Slice time range and convert to DataFrame
    bootstrap_df = pd.DataFrame(bootstrap_data.tolist())
    
    bootstrap_df = analysis_plotting_functions.normalize_rows(bootstrap_df,'minmax',min_window=(0,80),max_window=(100,300))

    bootstrap_df_short = bootstrap_df.iloc[:, start_idx:]
    
    # bootstrap_df_short = pd.DataFrame(bootstrap_df_short.mean(axis=0, skipna=True)).T

    
    # Step 3: Autocorrelation
    # acorr_df = analyzeSpont2P.calculate_autocorrelations_df(bootstrap_df_short, signal_range=signal_range, max_lags=max_lag)
    
    # Step 4: Plot mean autocorrelation
    # mean_acorr = acorr_df.mean(axis=0, skipna=True)
    # plt.figure()
    # plt.plot(mean_acorr)
    # plt.xlabel('Lag')
    # plt.ylabel('Mean autocorrelation')
    # plt.title(f'{plot_title} - {label}')
    
    # Step 5: Fit exponentials
    fit_results, good_acorr, valid_rows = analyzeSpont2P.fit_exponentials_from_df(
        bootstrap_df_short, exclude_low_acorr=True, acorr_threshold=acorr_1_threshold, max_lag=fit_max_lag,
        bounded_fit=True,tau_bounds=tau_bounds
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

def proportion_crossing_median(coef_df, ref_df, threshold=0.05):
    """
    For each feature (column), compute the fraction of coef_df values 
    above or below the median of the same column in ref_df.
    
    Args:
        coef_df: DataFrame with coefficient values.
        ref_df: DataFrame with same columns as coef_df, 
                used to compute reference medians.
        threshold: float, cutoff for deciding if distribution crosses median.
    
    Returns:
        DataFrame with per-feature summary.
    """
    results = []
    ref_medians = ref_df.median(axis=0, skipna=True)  # median per column in ref_df
    
    for feature in coef_df.columns:
        coefs = coef_df[feature].dropna().values
        if len(coefs) == 0:
            continue  # skip empty columns

        ref_value = ref_medians[feature]
        n = len(coefs)

        frac_above = (coefs >= ref_value).sum() / n
        frac_below = (coefs < ref_value).sum() / n
        frac_crossing = min(frac_above, frac_below) * 2

        results.append({
            "feature": feature,
            "median_coef": pd.Series(coefs).median(),
            "ref_median": ref_value,
            "frac_above_ref": frac_above,
            "frac_below_ref": frac_below,
            "crosses_ref": frac_crossing > threshold
        })
        
    return pd.DataFrame(results)

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
    tau_bounds=(2, 400),
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

import numpy as np
from scipy.stats import mannwhitneyu
from itertools import product

def permutation_test_prop_crossing(taus_v1, taus_m2, n_permutations=10000):
    """Permutation test for proportion of taus_m2 above median of taus_v1."""
    combined = np.concatenate([taus_v1, taus_m2])
    n_v1 = len(taus_v1)
    observed = np.mean(taus_m2 > np.median(taus_v1))
    perm_stats = []

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_v1 = combined[:n_v1]
        perm_m2 = combined[n_v1:]
        perm_stats.append(np.mean(perm_m2 > np.median(perm_v1)))

    perm_stats = np.array(perm_stats)
    # +1 correction avoids exact zeros
    pval = (np.sum(perm_stats >= observed) + 1) / (n_permutations + 1)
    return observed, pval

import numpy as np
from scipy.stats import mannwhitneyu

def permutation_test_difference(taus_v1, taus_m2, n_permutations=10000):
    """
    Permutation test comparing two datasets directly using the median difference.
    Returns observed median difference and permutation p-value.
    """
    combined = np.concatenate([taus_v1, taus_m2])
    n_v1 = len(taus_v1)
    observed_diff = np.median(taus_m2) - np.median(taus_v1)
    perm_diffs = []

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_v1 = combined[:n_v1]
        perm_m2 = combined[n_v1:]
        perm_diffs.append(np.median(perm_m2) - np.median(perm_v1))

    perm_diffs = np.array(perm_diffs)
    # Two-tailed p-value
    pval = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)
    return observed_diff, pval


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
    tau_bounds=(2,400),
    r2_threshold=0.8,
    acorr_1_threshold=0.2,
    acorr_or_trace='trace',
    n_permutations=10000,
    sampling_interval=sampling_interval
):
    all_results = []

    for sample_size in sample_sizes:
        print(f"\nSample size: {sample_size}")
        for repeat_id in range(n_repeats):
            print(f"  Repeat {repeat_id+1}/{n_repeats}")

            if acorr_or_trace=='trace':
                df_se_v1, _ = bootstrap_trace_decay_analysis(
                    v1_data, sample_size=sample_size, n_samples=n_samples,
                    start_idx=start_idx, signal_range=signal_range, max_lag=max_lag,
                    fit_max_lag=fit_max_lag, tau_bounds=tau_bounds,
                    r2_threshold=r2_threshold, acorr_1_threshold=acorr_1_threshold,
                    label="V1"
                )
                df_se_m2, _ = bootstrap_trace_decay_analysis(
                    m2_data, sample_size=sample_size, n_samples=n_samples,
                    start_idx=start_idx, signal_range=signal_range, max_lag=max_lag,
                    fit_max_lag=fit_max_lag, tau_bounds=tau_bounds,
                    r2_threshold=r2_threshold, acorr_1_threshold=acorr_1_threshold,
                    label="M2"
                )
            else:
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

            taus_v1 = df_se_v1["tau"].values*sampling_interval
            taus_m2 = df_se_m2["tau"].values*sampling_interval

            if len(taus_v1) > 5 and len(taus_m2) > 5:
                median_tau_v1 = np.median(taus_v1)
                median_tau_m2 = np.median(taus_m2)
                median_diff = median_tau_m2 - median_tau_v1
                pval_mw = mannwhitneyu(taus_m2, taus_v1, alternative='two-sided').pvalue
                d_cliff = cliffs_delta(taus_m2, taus_v1)

                # Proportion crossing medians
                prop_m2_above_v1_median = np.mean(taus_m2 > median_tau_v1)
                prop_m2_below_v1_median = np.mean(taus_m2 < median_tau_v1)
                prop_v1_above_m2_median = np.mean(taus_v1 > median_tau_m2)
                prop_v1_below_m2_median = np.mean(taus_v1 < median_tau_m2)

                # Permutation test for prop_m2_above_v1_median
                observed_prop_perm, pval_perm = permutation_test_prop_crossing(
                    taus_v1, taus_m2, n_permutations=n_permutations
                )

                # Permutation test for direct difference in medians
                observed_diff_perm, pval_diff_perm = permutation_test_difference(
                    taus_v1, taus_m2, n_permutations=n_permutations
                )

            else:
                median_tau_v1 = np.nan
                median_tau_m2 = np.nan
                median_diff = np.nan
                pval_mw = np.nan
                d_cliff = np.nan
                prop_m2_above_v1_median = np.nan
                prop_m2_below_v1_median = np.nan
                prop_v1_above_m2_median = np.nan
                prop_v1_below_m2_median = np.nan
                observed_prop_perm = np.nan
                pval_perm = np.nan

            all_results.append({
                "sample_size": sample_size,
                "repeat_id": repeat_id,
                "median_tau_v1": median_tau_v1,
                "median_tau_m2": median_tau_m2,
                "median_diff": median_diff,
                "p_value_mannwhitney": pval_mw,
                "cliffs_delta": d_cliff,
                "n_v1": len(taus_v1),
                "n_m2": len(taus_m2),
                "taus_v1_distribution": taus_v1,
                "taus_m2_distribution": taus_m2,
                "prop_m2_above_v1_median": prop_m2_above_v1_median,
                "prop_m2_below_v1_median": prop_m2_below_v1_median,
                "prop_v1_above_m2_median": prop_v1_above_m2_median,
                "prop_v1_below_m2_median": prop_v1_below_m2_median,
                "observed_prop_perm_test": observed_prop_perm,
                "p_value_perm_test": pval_perm,
                "observed_median_diff_perm_test": observed_diff_perm,
                "p_value_median_diff_perm_test": pval_diff_perm,
            })

    return pd.DataFrame(all_results)

# %%
def proportion_crossing_median(coef_df, ref_df, threshold=0.05):
    """
    For each feature (column), compute the fraction of coef_df values 
    above or below the median of the same column in ref_df.
    
    Args:
        coef_df: DataFrame with coefficient values.
        ref_df: DataFrame with same columns as coef_df, 
                used to compute reference medians.
        threshold: float, cutoff for deciding if distribution crosses median.
    
    Returns:
        DataFrame with per-feature summary.
    """
    results = []
    ref_medians = ref_df.median(axis=0, skipna=True)  # median per column in ref_df
    
    for feature in coef_df.columns:
        coefs = coef_df[feature].dropna().values
        if len(coefs) == 0:
            continue  # skip empty columns

        ref_value = ref_medians[feature]
        n = len(coefs)

        frac_above = (coefs >= ref_value).sum() / n
        frac_below = (coefs < ref_value).sum() / n
        frac_crossing = min(frac_above, frac_below) * 2

        results.append({
            "feature": feature,
            "median_coef": pd.Series(coefs).median(),
            "ref_median": ref_value,
            "frac_above_ref": frac_above,
            "frac_below_ref": frac_below,
            "crosses_ref": frac_crossing > threshold
        })
        
    return pd.DataFrame(results)



# %%
import numpy as np
import pandas as pd

def summarize_prop_crossing_per_row(all_results, n_permutations=10000):
    """
    Compute median proportion crossing for each row (repeat) individually,
    with permutation p-values to avoid zero inflation.
    
    Returns a DataFrame of the same number of rows as all_results.
    """
    summary_rows = []

    for idx, row in all_results.iterrows():
        taus_v1 = np.array(row['taus_v1_distribution'])
        taus_m2 = np.array(row['taus_m2_distribution'])

        # Observed proportions for this row
        prop_m2_above_v1 = np.mean(taus_m2 > np.median(taus_v1))
        prop_v1_above_m2 = np.mean(taus_v1 > np.median(taus_m2))

        # Permutation test for M2 > V1
        combined = np.concatenate([taus_v1, taus_m2])
        n_v1 = len(taus_v1)
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_v1 = combined[:n_v1]
            perm_m2 = combined[n_v1:]
            perm_stats.append(np.mean(perm_m2 > np.median(perm_v1)))
        perm_stats = np.array(perm_stats)
        pval_m2 = (np.sum(perm_stats >= prop_m2_above_v1) + 1) / (len(perm_stats) + 1)

        # Permutation test for V1 > M2
        perm_stats_v1 = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_v1 = combined[:n_v1]
            perm_m2 = combined[n_v1:]
            perm_stats_v1.append(np.mean(perm_v1 > np.median(perm_m2)))
        perm_stats_v1 = np.array(perm_stats_v1)
        pval_v1 = (np.sum(perm_stats_v1 >= prop_v1_above_m2) + 1) / (len(perm_stats_v1) + 1)

        # Save row-wise results
        summary_rows.append({
            'sample_size': row['sample_size'],
            'prop_m2_above_v1': prop_m2_above_v1,
            'prop_v1_above_m2': prop_v1_above_m2,
            'pval_m2_above_v1': pval_m2,
            'pval_v1_above_m2': pval_v1
        })

    return pd.DataFrame(summary_rows)

# %%

import numpy as np
import pandas as pd

def summarize_median_diff_directional(all_results, alpha=0.05):
    """
    Summarize median differences between M2 and V1 per sample size using
    per-repeat medians, with a directional permutation-style p-value
    resistant to bootstrap inflation.
    
    Returns one row per sample size with:
        - observed median difference
        - 95% CI of differences across repeats
        - directional consistency p-value
    """
    summary_list = []
    ci_bounds = [100*alpha/2, 100*(1-alpha/2)]

    for sample_size, group_df in all_results.groupby('sample_size'):
        # Per-repeat medians
        medians_v1 = group_df['median_tau_v1'].values
        medians_m2 = group_df['median_tau_m2'].values

        # Median difference across repeats
        median_diff_per_repeat = medians_m2 - medians_v1
        observed_diff = np.median(median_diff_per_repeat)

        # 95% CI across repeats
        ci_lower, ci_upper = np.percentile(median_diff_per_repeat, ci_bounds)


        # Directional “permutation” p-value
        #this couts the fraction of repeats that have opposite direction
        # If almost all repeats agree in sign → very small p-value.
        # If about half disagree → p-value ~ 0.5.,If all disagree → p-value ~ 1.
        
        n_opposite = np.sum(np.sign(median_diff_per_repeat) != np.sign(observed_diff))
        p_value_directional = (n_opposite + 1) / (len(median_diff_per_repeat) + 1)  # +1 for stability

        summary_list.append({
            'sample_size': sample_size,
            'median_diff': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value_directional': p_value_directional
        })

    return pd.DataFrame(summary_list)


# %%
import numpy as np
import pandas as pd

def summarize_median_props(all_results, alpha=0.05):
    """
    Summarize median differences between M2 and V1 per sample size using
    per-repeat medians, with:
        - median difference
        - 95% CI
        - proportion of repeats where M2 > V1
        - proportion of repeats where V1 > M2
        - empirical directional p-value with +1 correction

    all_results: DataFrame with columns 'sample_size', 'median_tau_m2', 'median_tau_v1'
    """
    summary_list = []
    ci_bounds = [100*alpha/2, 100*(1-alpha/2)]

    for sample_size, group_df in all_results.groupby('sample_size'):
        m2_medians = group_df['median_tau_m2'].values
        v1_medians = group_df['median_tau_v1'].values

        # per-repeat differences
        diffs = m2_medians - v1_medians
        median_diff = np.median(diffs)

        # 95% CI
        ci_lower, ci_upper = np.percentile(diffs, ci_bounds)

        # proportions
        prop_m2_above = np.mean(diffs > 0)
        prop_v1_above = np.mean(diffs < 0)

        # empirical p-value (directional)
        n_repeats = len(diffs)
        n_opposite = np.sum(diffs <= 0) if median_diff > 0 else np.sum(diffs >= 0)
        p_empirical = (n_opposite + 1) / (n_repeats + 1)

        is_significant=alpha>p_empirical
        # friendly note for reporting
        if n_opposite == 0:
            p_note = f"p < {1/n_repeats:.3g} (corrected {p_empirical:.4f})"
        else:
            p_note = f"p = {p_empirical:.4f}"

        summary_list.append({
            'sample_size': sample_size,
            'median_diff': median_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'prop_m2_above': prop_m2_above,
            'prop_v1_above': prop_v1_above,
            'empirical_p': p_empirical,
            'p_note': p_note,
            'is_significant':is_significant
            
        })

    return pd.DataFrame(summary_list)

# %%

import matplotlib.pyplot as plt

def plot_median_diff_summary(summary_df, title="Median τ Difference (M2 - V1)"):
    """
    Plot median difference summary with CI shading across sample sizes.
    Expects output of summarize_median_diff_with_ci_p.
    """
    x = summary_df['sample_size']
    y = summary_df['median_diff']
    lower = summary_df['ci_lower']
    upper = summary_df['ci_upper']

    plt.figure(figsize=(7,5))
    plt.plot(x, y, marker='o', color='black', label="Mean median diff")
    plt.fill_between(x, lower, upper, alpha=0.3, color='gray', label="95% CI")
    plt.axhline(0, color='red', linestyle='--', label="Null (0 difference)")

    # Add significance stars above points
    for xi, yi, sig in zip(x, y, summary_df['is_significant']):
        if sig:
            plt.text(xi, yi + (upper.max()-lower.min())*0.05, "*", 
                     ha='center', va='bottom', fontsize=12, color='red')

    plt.xlabel("Sample size")
    plt.ylabel("Median(M2) - Median(V1)")
    plt.title(title)
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

# Find the minimum length among all arrays
min_len = min(len(arr) for arr in result_M2_standard_non_stimd_exc_filt['averaged_traces_all'])
# Truncate all arrays to that length
m2_avgs_array_2d = np.vstack([arr[:min_len] for arr in result_M2_standard_non_stimd_exc_filt['averaged_traces_all']])
bootstrap_avgs_m2=bootstrap_population_average(m2_avgs_array_2d, sample_size=25,n_samples=10)


bootstrap_avgs_m22 = pd.DataFrame(bootstrap_avgs_m2.tolist())

bootstrap_df_short_m2 = bootstrap_avgs_m22.iloc[:, start_idx:]


bootstrap_mean_trace_m2 = pd.DataFrame(bootstrap_df_short_m2.mean(axis=0, skipna=True)).T


acorr_df_m2 = analyzeSpont2P.calculate_autocorrelations_df(bootstrap_mean_trace_m2, signal_range=signal_range, max_lags=max_lag)
fit_results_m2, _, _ = analyzeSpont2P.fit_exponentials_from_df(
    acorr_df_m2, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=fit_max_lag,bounded_fit=True,tau_bounds=(1,300)
)


# Find the minimum length among all arrays
min_len = min(len(arr) for arr in result_V1_standard_non_stimd_exc_filt['averaged_traces_all'])
# Truncate all arrays to that length
v1_avgs_array_2d = np.vstack([arr[:min_len] for arr in result_V1_standard_non_stimd_exc_filt['averaged_traces_all']])
bootstrap_avgs_v1=bootstrap_population_average(v1_avgs_array_2d, sample_size=25,n_samples=10)


bootstrap_avgs_v11 = pd.DataFrame(bootstrap_avgs_v1.tolist())
bootstrap_df_short_v1 = bootstrap_avgs_v11.iloc[:, start_idx:]



bootstrap_mean_trace_v1 = pd.DataFrame(bootstrap_df_short_v1.mean(axis=0, skipna=True)).T


acorr_df_v1 = analyzeSpont2P.calculate_autocorrelations_df(bootstrap_mean_trace_v1, signal_range=signal_range, max_lags=max_lag)

fit_results_v1, _, _ = analyzeSpont2P.fit_exponentials_from_df(
    acorr_df_v1, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=fit_max_lag,bounded_fit=True,tau_bounds=(1,300)
)


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

fit_results_m2, _, _ = analyzeSpont2P.fit_exponentials_from_df(
    bootstrap_df_short_m2, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=fit_max_lag,bounded_fit=True
)

fit_results_v1, _, _ = analyzeSpont2P.fit_exponentials_from_df(
    bootstrap_df_short_v1, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=fit_max_lag,bounded_fit=True
)


plot_acorr_with_fit_dual(
    acorr_trace_1d_1=bootstrap_df_short_m2.iloc[0],
    fit_results_1=fit_results_m2,
    index_in_df_1=acorr_df_m2.index[0],
    acorr_trace_1d_2=bootstrap_df_short_v1.iloc[0],
    fit_results_2=fit_results_v1,
    index_in_df_2=acorr_df_v1.index[0],
    sampling_interval=sampling_interval,
    label_1="M2",
    label_2="V1",
    color_1=params['general_params']['M2_cl'],
    color_2=params['general_params']['V1_cl'],
    fig_size=(4, 4),
    fit_type="double_exp"
)


# %% Trace decay

df_se_m2, best_type_m2 = bootstrap_trace_decay_analysis(
    data=m2_avgs_array_2d, 
    sample_size=100, 
    n_samples=100,
    label="M2",
    # bounded_fit=True,
    acorr_1_threshold=0.0,
    r2_threshold=0.0,
    tau_bounds=(0,30000)
    )

df_se_v1, best_type_v1 = bootstrap_trace_decay_analysis(
    data=v1_avgs_array_2d, 
    sample_size=100, 
    n_samples=100,
    label="V1",
    # bounded_fit=True,
    acorr_1_threshold=0.0,
    r2_threshold=0.0,
    tau_bounds=(0,30000)
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
# %% acorr analysis

df_se_m2, best_type_m2 = bootstrap_acorr_analysis(
    data=m2_avgs_array_2d, 
    sample_size=25, 
    n_samples=1000,
    label="M2",
    # bounded_fit=True,
    tau_bounds=(0,3000)
    )

df_se_v1, best_type_v1 = bootstrap_acorr_analysis(
    data=v1_avgs_array_2d, 
    sample_size=25, 
    n_samples=1000,
    label="V1",
    tau_bounds=(0,3000)
    # bounded_fit=True,
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

dist_crosses_V1 = proportion_crossing_median(pd.DataFrame(df_se_v1.tau),pd.DataFrame(df_se_m2.tau))

dist_crosses_M2 = proportion_crossing_median(pd.DataFrame(df_se_m2.tau),pd.DataFrame(df_se_v1.tau))

# %% Loop data for figure 4C


sample_sizes = [1,5, 10, 25, 50, 100]
# sample_sizes = [1]


results_df_se = compare_tau_distributions_vs_sample_size_repeat(
    v1_data=v1_avgs_array_2d,
    m2_data=m2_avgs_array_2d,
    sample_sizes=sample_sizes,
    n_samples=100,
    n_repeats=1000,
    acorr_or_trace='acorr',
    r2_threshold=0.8,
    tau_bounds=(0,3000)
)

# # %%
# joblib.dump(results_df_se, 'pop_bootstrap_acorr_taus_1000_repeats_1-100_sample_sizes.joblib')


# # %%

# results_df_se=joblib.load('pop_bootstrap_acorr_taus_1000_repeats_1-100_sample_sizes.joblib')
# %%

results_df_di = compare_tau_distributions_vs_sample_size_repeat(
    v1_data=v1_avgs_array_2d,
    m2_data=m2_avgs_array_2d,
    sample_sizes=sample_sizes,
    n_samples=100,
    n_repeats=100,
    acorr_or_trace='acorr',
    r2_threshold=0.8,
    tau_bounds=(0,3000)
)

# %%

a=summarize_prop_crossing_per_row(results_df_se,n_permutations=100)


median_tau_analysis_CI_p=summarize_median_props(results_df_se)

plot_median_diff_summary(median_tau_analysis_CI_p)


# %% Figure 4C
plot_pvalue_vs_sample_size(results_df_se, significance_level=0.05,fig_size=(5, 3),
                           xlabel="Number of sampled neurons")

plot_pvalue_vs_sample_size(a, significance_level=0.05,fig_size=(5, 3),
                            xlabel="Number of sampled neurons",
                            p_stat='pval_m2_above_v1')
# %%

plot_median_taus_vs_sample_size(
    results_df_se,
    fig_size=(5, 3),
    ylabel="Tau (seconds)",
    xlabel="Number of sampled neurons",
    title="V1 vs M2: Median Tau vs. Sample Size",
    colors={"V1": params['general_params']['V1_cl'], "M2": params['general_params']['M2_cl']},
    sampling_interval=1
)