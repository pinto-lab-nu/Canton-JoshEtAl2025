# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 18:48:01 2025

@author: jec822
"""
# df_combined = pd.concat([result_M2_pseudo_filt,result_M2_filt], ignore_index=True)


# df_combined = pd.concat([result_M2_pseudo,result_M2], ignore_index=True)
df_combined = pd.concat([result_M2], ignore_index=True)

# df_combined = pd.concat([result_V1], ignore_index=True)



# %%
import numpy as np
import pandas as pd

# Define sweep ranges for each feature
threshold_ranges = {
    'peak_amp_avg':     np.linspace(0, 20, 20),
    'peak_time_avg':     np.linspace(0, 10, 50),
    'auc_avg':          np.linspace(0, 1000, 50),
    'response_proportion': np.linspace(0, 1, 5),
    'peak_array_mean_trial': np.linspace(0, 100, 20),
    'com_std':          np.linspace(0, 10, 50),
    'com_avg':          np.linspace(0, 10, 50),
    'peak_time_std':    np.linspace(0, 10, 50),
}

results = []

# Loop through each feature
for feature, thresholds in threshold_ranges.items():
    best_ratio = 0
    best_thresh = None
    best_mode = None
    best_n_real = 0
    best_n_shuff = 0

    for t in thresholds:
        # Try threshold as a minimum
        df_min = df_combined[df_combined[feature] >= t]
        n_real_min = (df_min['is_shuffled'] == False).sum()
        n_shuff_min = (df_min['is_shuffled'] == True).sum()
        if n_real_min > 0 and n_shuff_min > 0:
            ratio_min = n_real_min / n_shuff_min
            if ratio_min > best_ratio:
                best_ratio = ratio_min
                best_thresh = t
                best_mode = '>='
                best_n_real = n_real_min
                best_n_shuff = n_shuff_min

        # Try threshold as a maximum
        df_max = df_combined[df_combined[feature] <= t]
        n_real_max = (df_max['is_shuffled'] == False).sum()
        n_shuff_max = (df_max['is_shuffled'] == True).sum()
        if n_real_max > 0 and n_shuff_max > 0:
            ratio_max = n_real_max / n_shuff_max
            if ratio_max > best_ratio:
                best_ratio = ratio_max
                best_thresh = t
                best_mode = '<='
                best_n_real = n_real_max
                best_n_shuff = n_shuff_max

    results.append({
        'feature': feature,
        'best_thresh': best_thresh,
        'direction': best_mode,
        'real_count': best_n_real,
        'shuff_count': best_n_shuff,
        'real/shuff ratio': round(best_ratio, 2)
    })

# Format as a DataFrame
best_cutoffs_df = pd.DataFrame(results).sort_values(by='real/shuff ratio', ascending=False)
print(best_cutoffs_df)

# %%
import itertools
# Define features and threshold ranges
feature_thresholds = {
    'peak_amp_avg': np.linspace(0, 8, 8),  # fewer steps to keep it fast
    'auc_avg': np.linspace(20, 80, 3),
    'response_proportion': np.linspace(0.2, 0.8, 3),
    'peak_array_mean_trial': np.linspace(30, 70, 3),
    'com_std': np.linspace(20, 1, 3),
    'peak_time_std': np.linspace(0, 1, 3),
}


threshold_ranges = {
    'peak_amp_avg':     np.linspace(0, 20, 20),
    'peak_time_avg':     np.linspace(0, 10, 50),
    'auc_avg':          np.linspace(0, 1000, 50),
    'response_proportion': np.linspace(0, 1, 5),
    'peak_array_mean_trial': np.linspace(0, 15, 20),
    'com_std':          np.linspace(0, 2, 50),
    'com_avg':          np.linspace(0, 10, 50),
    'peak_time_std':    np.linspace(0, 1, 50),
}

# Define all combinations of (feature, threshold, mode)
threshold_options = []
for feature, thresholds in feature_thresholds.items():
    for t in thresholds:
        threshold_options.append((feature, t, '>='))
        threshold_options.append((feature, t, '<='))

# Try all combinations of thresholds across features (1 per feature)
feature_names = list(feature_thresholds.keys())
threshold_per_feature = []
for f in feature_names:
    steps = feature_thresholds[f]
    per_feature_options = [(f, t, '>=') for t in steps] + [(f, t, '<=') for t in steps]
    threshold_per_feature.append(per_feature_options)

# Brute-force combinations
best_combo = None
best_ratio = 0
best_counts = (0, 0)

for combo in itertools.product(*threshold_per_feature):
    df_filtered = df_combined.copy()
    for feature, thresh, mode in combo:
        if mode == '>=':
            df_filtered = df_filtered[df_filtered[feature] >= thresh]
        else:
            df_filtered = df_filtered[df_filtered[feature] <= thresh]
        if df_filtered.empty:
            break
    else:
        n_real = (df_filtered['is_shuffled'] == False).sum()
        n_shuff = (df_filtered['is_shuffled'] == True).sum()
        if n_shuff > 0 and n_real / n_shuff > best_ratio:
            best_ratio = n_real / n_shuff
            best_combo = combo
            best_counts = (n_real, n_shuff)

best_combo, best_counts, best_ratio

# %%
# plt.figure()
# plt.plot(np.mean(filtered_traces_avg_m2_array,axis=0))
# plt.plot(np.mean(filtered_traces_avg_v1_array,axis=0))

plt.figure()
plt.plot(np.mean(filtered_array_m2,axis=0))
plt.plot(np.mean(filtered_array_v1,axis=0))

# %%

analysis_plotting_functions.plot_heatmap(pd.DataFrame(filtered_array_v1),vmin=-2, vmax=2, start_index=0, sampling_interval=0.032958316, exclude_window=None)

# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def run_pca_on_response_features(
    df,
    feature_cols=None,
    color_by='response_proportion',
    n_components=2,
    plot=True,
    biplot=False,
    return_data=False,
    xlim=None,
    ylim=None,
    cmap='viridis',
    fit_pca=True,
    use_existing_pca=None
):
    """
    Perform PCA on selected features of a DataFrame and plot the result with optional biplot.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with per-ROI or per-stimulus features.
    feature_cols : list of str or None
        Columns to include in PCA. If None, a default list is used.
    color_by : str
        Column name to use for coloring points in the plot.
    n_components : int
        Number of principal components to compute (ignored if using existing PCA).
    plot : bool
        If True, display a scatter plot of the first two PCs.
    biplot : bool
        If True, draw arrows showing feature contributions (PCA loadings).
    return_data : bool
        If True, return PCA results and cleaned DataFrame.
    xlim, ylim : tuple or None
        Axis limits.
    cmap : str
        Colormap for point coloring.
    fit_pca : bool
        Whether to fit a new PCA (True) or use an existing one (False).
    use_existing_pca : PCA object or None
        Optional pretrained PCA model for projection only.

    Returns
    -------
    If return_data is True:
        X_pca : np.ndarray
            PCA-transformed feature matrix (n_samples x n_components)
        df_clean : pd.DataFrame
            Subset of df used in PCA (NaNs removed)
        pca : PCA object
            Fitted PCA model (or the reused one)
        explained_variance : np.ndarray
            Variance explained by each PC
        loadings_df : pd.DataFrame
            PCA loadings (contribution of each feature to each PC)
    """

    if feature_cols is None:
        feature_cols = [
            'peak_amp_avg', 'peak_amp_std',
            'com_avg', 'com_std',
            'auc_avg', 'auc_std',
            'peak_time_avg', 'peak_time_std',
            'fwhm_avg', 'fwhm_std',
            'response_proportion',
            'peak_array_mean_trial',
            # 'event_rate'
        ]

    # Drop rows with NaNs in the selected features
    features_df = df[feature_cols]
    features_df_clean = features_df.dropna()
    df_clean = df.loc[features_df_clean.index]

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df_clean)

    # PCA Fit or Transform
    if fit_pca:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
    else:
        if use_existing_pca is None:
            raise ValueError("fit_pca=False requires a valid use_existing_pca model.")
        pca = use_existing_pca
        X_pca = pca.transform(X_scaled)

    # Extract variance and loadings if possible
    explained_variance = getattr(pca, 'explained_variance_ratio_', np.full(n_components, np.nan))
    if hasattr(pca, 'components_'):
        loadings = pca.components_
        loadings_df = pd.DataFrame(
            loadings.T,
            index=feature_cols,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)]
        )
    else:
        loadings_df = pd.DataFrame()

    # Plot
    if plot and pca.n_components_ >= 2:
        color_values = df_clean[color_by] if color_by in df_clean.columns else None
        plt.figure(figsize=(9, 7))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                              c=color_values, cmap=cmap,
                              edgecolor='k')
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% var)')
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% var)')
        plt.title('PCA of Response Features')

        if color_values is not None:
            plt.colorbar(scatter, label=color_by)

        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)

        if biplot and hasattr(pca, 'components_'):
            arrow_scale = 2.5
            for i, feature in enumerate(feature_cols):
                x_vec, y_vec = loadings[0, i], loadings[1, i]
                plt.arrow(0, 0, x_vec * arrow_scale, y_vec * arrow_scale,
                          color='red', alpha=0.6, head_width=0.05, length_includes_head=True)
                plt.text(x_vec * arrow_scale * 4,
                         y_vec * arrow_scale * 4,
                         feature,
                         color='k', fontsize=12)

        plt.tight_layout()
        plt.show()

    if return_data:
        return X_pca, df_clean, pca, explained_variance, loadings_df

# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plots

def run_pca_3d_response_features(
    df,
    feature_cols=None,
    color_by='response_proportion',
    plot=True,
    biplot=True,
    return_data=False,
    cmap='viridis',
    arrow_scale=3.0
):
    """
    Perform 3D PCA on selected features and plot with optional biplot.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with per-ROI or per-stimulus features.
    feature_cols : list of str or None
        Columns to include in PCA. If None, a default list is used.
    color_by : str
        Column name to use for coloring points.
    plot : bool
        If True, display a 3D scatter plot.
    biplot : bool
        If True, overlay PCA loadings as arrows in 3D.
    return_data : bool
        If True, return PCA outputs and cleaned data.
    cmap : str
        Matplotlib colormap for points.
    arrow_scale : float
        Scaling factor for feature arrows.

    Returns
    -------
    If return_data is True:
        X_pca : np.ndarray
        df_clean : pd.DataFrame
        pca : PCA object
        explained_variance : np.ndarray
        loadings_df : pd.DataFrame
    """
    if feature_cols is None:
        feature_cols = [
            'peak_amp_avg', 'peak_amp_std',
            'com_avg', 'com_std',
            'auc_avg', 'auc_std',
            'peak_time_avg', 'peak_time_std',
            'fwhm_avg', 'fwhm_std',
            'response_proportion',
            'peak_array_mean_trial',
            'is_shuffled'
        ]

    features_df = df[feature_cols].dropna()
    # features_df = df[feature_cols]

    df_clean = df.loc[features_df.index]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    # PCA (3 components)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Variance and loadings
    explained_variance = pca.explained_variance_ratio_
    loadings = pca.components_
    loadings_df = pd.DataFrame(
        loadings.T,
        index=feature_cols,
        columns=[f'PC{i+1}' for i in range(3)]
    )

    # Plot
    if plot:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Point coloring
        color_values = df_clean[color_by] if color_by in df_clean.columns else None
        scatter = ax.scatter(
            X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
            c=color_values,
            cmap=cmap,
            edgecolor='k',
            s=50
        )

        ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)')
        ax.set_title("3D PCA of Response Features")

        if color_values is not None:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1, label=color_by)

        # Feature arrows
        if biplot:
            for i, feature in enumerate(feature_cols):
                x, y, z = loadings[0, i], loadings[1, i], loadings[2, i]
                ax.quiver(
                    0, 0, 0,
                    x * arrow_scale, y * arrow_scale, z * arrow_scale,
                    color='red', linewidth=1, arrow_length_ratio=0.1
                )
                ax.text(
                    x * arrow_scale * 1.1,
                    y * arrow_scale * 1.1,
                    z * arrow_scale * 1.1,
                    feature,
                    color='red',
                    fontsize=9
                )

        plt.tight_layout()
        plt.show()

    if return_data:
        return X_pca, df_clean, pca, explained_variance, loadings_df

# %%

X_pca, df_used, pca_model,explained_variance, loadings_df= run_pca_on_response_features(
    results_v1,
    color_by='peak_time_std',
    return_data=True,
    xlim=(-5, 15),
    ylim=(-5, 8),
    biplot=True
)

X_pca, df_used, pca_model,explained_variance_m2, loadings_df_m2 = run_pca_on_response_features(
    result_m2,
    color_by='peak_time_std',
    return_data=True,
    xlim=(-5, 15),
    ylim=(-5, 8),
    biplot=True
)

# %%
X_pca, df_used, pca_model,explained_variance, loadings_df= run_pca_3d_response_features(
    results_v1,
    color_by='responsive_ratio',
    return_data=True,
    # xlim=(-5, 15),
    # ylim=(-5, 8),
    biplot=True
)

X_pca, df_used, pca_model,explained_variance_m2, loadings_df_m2 = run_pca_3d_response_features(
    result_m2,
    color_by='responsive_ratio',
    return_data=True,
    # xlim=(-5, 15),
    # ylim=(-5, 8),
    biplot=True
)

# %%
from scipy.spatial.distance import mahalanobis, cdist

# Get PCA-transformed data
X_pca, df_clean, pca, var, loadings = run_pca_on_response_features(
    df_combined,color_by='is_shuffled', return_data=True,n_components=8
)


# Separate real and shuffled
real_idx = df_clean['is_shuffled'] == False
shuff_idx = df_clean['is_shuffled'] == False
real_data = X_pca[real_idx]
shuff_data = X_pca[shuff_idx]

# Compute centroid and covariance of control
shuff_mean = np.mean(shuff_data, axis=0)
shuff_cov = np.cov(shuff_data.T)
shuff_cov_inv = np.linalg.inv(shuff_cov)

# Mahalanobis distance of real data to control centroid
dists = [mahalanobis(x, shuff_mean, shuff_cov_inv) for x in real_data]

# Threshold based on distance percentile of control data
control_dists = [mahalanobis(x, shuff_mean, shuff_cov_inv) for x in shuff_data]
threshold = np.percentile(control_dists, 90)  # 95% of control

# Find real responses that are outside control space
non_overlapping_idx = real_idx[real_idx].index[dists > threshold]
non_overlapping_df = df_clean.loc[non_overlapping_idx]

# Find real responses that are outside control space
overlapping_idx = real_idx[real_idx].index[dists < threshold]
overlapping_df = df_clean.loc[overlapping_idx]
# %%
# STEP 1: Fit PCA only on real (non-shuffled) data
X_pca_real, df_real_clean, pca, var, loadings = run_pca_on_response_features(
    result_M2_pseudo,
    return_data=True,
    n_components=8
)

# STEP 2: Extract the full combined dataset (real + pseudo)
# This should already contain the 'is_shuffled' column
X_features_all, df_combined_clean, _, _, _ = run_pca_on_response_features(
    df_combined,
    return_data=True,
    n_components=8,
    fit_pca=False,           # You need to allow `run_pca_on_response_features` to skip fitting
    use_existing_pca=pca     # Project using the PCA fit from real data
)

# STEP 3: Separate real and shuffled in projected space
real_idx = df_combined_clean['is_shuffled'] == False
shuff_idx = df_combined_clean['is_shuffled'] == True
real_data = X_features_all[real_idx]
shuff_data = X_features_all[shuff_idx]

# STEP 4: Compute Mahalanobis distances
from scipy.spatial.distance import mahalanobis

shuff_mean = np.mean(shuff_data, axis=0)
shuff_cov = np.cov(shuff_data.T)
shuff_cov_inv = np.linalg.inv(shuff_cov)

# Distance of each real point from shuffled centroid
real_dists = [mahalanobis(x, shuff_mean, shuff_cov_inv) for x in real_data]
control_dists = [mahalanobis(x, shuff_mean, shuff_cov_inv) for x in shuff_data]

# STEP 5: Threshold by percentile of shuffled distribution
threshold = np.percentile(control_dists, 90)  # or 95

# STEP 6: Find "responsive" real cells beyond the null boundary
responsive_idx = df_combined_clean[real_idx].index[np.array(real_dists) > threshold]
non_responsive_idx = df_combined_clean[real_idx].index[np.array(real_dists) <= threshold]

responsive_df = df_combined_clean.loc[responsive_idx]
nonresponsive_df = df_combined_clean.loc[non_responsive_idx]


# %%
# non_overlapping_df_sorted = responsive_df.sort_values(by='peak_time_array_mean_trial')


non_overlapping_df_sorted = non_overlapping_df.sort_values(by='peak_time_avg')
# overlapping_df_sorted = overlapping_df.sort_values(by='com_avg')

a=non_overlapping_df_sorted['averaged_traces_all']
# a=overlapping_df_sorted['averaged_traces_all']
import numpy as np

# Find the max length
max_len = a.apply(len).max()

# Pad each array to max_len with NaNs
a_padded = a.apply(lambda x: np.pad(x, (0, max_len - len(x)), mode='constant', constant_values=np.nan))

# Convert to 2D array
a_2d = np.vstack(a_padded.values)



analysis_plotting_functions.plot_heatmap(pd.DataFrame(a_2d),vmin=-2, vmax=4, start_index=0, sampling_interval=0.032958316, exclude_window=None)
# %%

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
feature_cols = [
    'peak_amp_avg', 'peak_amp_std',
    'com_avg', 'com_std',
    'auc_avg', 'auc_std',
    'peak_time_avg', 'peak_time_std',
    'fwhm_avg', 'fwhm_std',
    'response_proportion',
    'peak_array_mean_trial',
    'event_rate',
    'snr',
    # 'is_shuffled'
    # 'roi_occurrence_all'
]
# Select and scale features
X = df_combined[feature_cols].dropna()
y = df_combined.loc[X.index, 'is_shuffled'].astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit logistic regression
clf = LogisticRegression(penalty='l2')
clf.fit(X_scaled, y)

# Inspect weights
feature_importance = pd.Series(clf.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print(feature_importance)
# %%

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_scaled, y)

# Feature weights (linear discriminant coefficients)
lda_weights = pd.Series(lda.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print(lda_weights)
# %%
from sklearn.feature_selection import mutual_info_classif, f_classif

mi_scores = mutual_info_classif(X_scaled, y)
f_scores, _ = f_classif(X_scaled, y)

mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
f_series = pd.Series(f_scores, index=X.columns).sort_values(ascending=False)

print("Mutual Information:\n", mi_series)
print("F-scores:\n", f_series)
