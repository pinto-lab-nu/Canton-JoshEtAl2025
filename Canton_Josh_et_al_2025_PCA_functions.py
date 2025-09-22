# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 10:15:46 2025

@author: jec822
"""
import matplotlib as mpl
# %%

import numpy as np

def compute_3d_transition_distances(v1_pca_results):
    """
    Computes 3D distances between the last point of the baseline and 
    the first point of the response for each trial, across all experiments 
    and indices in the PCA projection results.
    
    Returns:
        all_distance_matrices: list of list of 1D numpy arrays
            Structure: [experiment][index][trial] = distance
    """
    all_distance_matrices = []

    for expt_idx, expt in enumerate(v1_pca_results['single_expt_results']):
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']
        
        num_indices = len(basel_all)
        distance_matrix = []

        for idx in range(num_indices):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]

            trial_distances = []
            for a, b in zip(trial_projs_basel, trial_projs_resp):
                if a.shape[1] >= 3 and b.shape[1] >= 3:
                    last_baseline = a[-1, :3]
                    first_response = b[0, :3]
                    dist = np.linalg.norm(last_baseline - first_response)
                    trial_distances.append(dist)
                else:
                    trial_distances.append(np.nan)
            
            distance_matrix.append(np.array(trial_distances))

        all_distance_matrices.append(distance_matrix)
    
    return all_distance_matrices

# %%
import numpy as np

def compute_3d_transition_distances(v1_pca_results, num_pcs=3):
    """
    Computes normalized distances between the last point of the baseline and 
    the first point of the response for each trial using the specified number 
    of principal components.

    The distance is normalized by the pooled standard deviation of both the 
    baseline and response data for each trial.

    Args:
        v1_pca_results: dict containing trial PCA projections.
        num_pcs: int, number of principal components to use (default is 3).

    Returns:
        all_distance_matrices: list of list of 1D numpy arrays
            Structure: [experiment][index][trial] = normalized distance
    """
    all_distance_matrices = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        num_indices = len(basel_all)
        distance_matrix = []

        for idx in range(num_indices):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]

            trial_distances = []
            for a, b in zip(trial_projs_basel, trial_projs_resp):
                if a.shape[1] >= num_pcs and b.shape[1] >= num_pcs:
                    # Extract last baseline and first response point in selected PCs
                    last_baseline = a[-1, :num_pcs]
                    first_response = b[0, :num_pcs]

                    # Compute Euclidean distance
                    dist = np.linalg.norm(last_baseline - first_response)

                    # Compute pooled standard deviation for normalization
                    pooled = np.vstack((a[:, :num_pcs], b[:, :num_pcs]))
                    pooled_std = np.std(pooled, axis=0, ddof=1)
                    pooled_std_norm = np.linalg.norm(pooled_std)

                    # Normalize distance
                    if pooled_std_norm > 0:
                        norm_dist = dist / pooled_std_norm
                    else:
                        norm_dist = np.nan  # Avoid divide-by-zero

                    trial_distances.append(norm_dist)
                else:
                    trial_distances.append(np.nan)

            distance_matrix.append(np.array(trial_distances))
        all_distance_matrices.append(distance_matrix)

    return all_distance_matrices



# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_distance_histogram(all_distance_matrices, bins=30, color='blue', alpha=0.7):
    """
    Flattens a nested list of distance matrices and plots a histogram.
    
    Parameters:
        all_distance_matrices (list): Output from compute_3d_transition_distances
        bins (int): Number of bins in histogram
        color (str): Color of the bars
        alpha (float): Transparency of the bars
    
    Returns:
        flattened_distances (np.ndarray): 1D array of all distances (NaNs removed)
    """
    # Flatten into a single 1D list of distances
    flattened_distances = []
    for expt in all_distance_matrices:
        for index_distances in expt:
            flattened_distances.extend(index_distances)
    
    # Convert to numpy array and remove NaNs
    flattened_distances = np.array(flattened_distances)
    flattened_distances = flattened_distances[~np.isnan(flattened_distances)]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flattened_distances, bins=bins, color=color, edgecolor='black', alpha=alpha)
    plt.xlabel('Distance', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of 3D Transition Distances', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return flattened_distances
# %%


import numpy as np

def compute_max_poststim_distance_from_baseline_3d(v1_pca_results):
    """
    For each trial, computes the maximum 3D distance of any post-stim point
    from the centroid of the baseline period.
    
    Returns:
        all_max_distance_matrices: list of list of 1D numpy arrays
            Structure: [experiment][index][trial] = max distance
    """
    all_max_distance_matrices = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        num_indices = len(basel_all)
        max_distance_matrix = []

        for idx in range(num_indices):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]

            trial_max_distances = []
            for a, b in zip(trial_projs_basel, trial_projs_resp):
                if a.shape[1] >= 3 and b.shape[1] >= 3:
                    baseline_centroid = np.mean(a[:, :3], axis=0)
                    dists = np.linalg.norm(b[:, :3] - baseline_centroid, axis=1)
                    max_dist = np.max(dists)
                    trial_max_distances.append(max_dist)
                else:
                    trial_max_distances.append(np.nan)

            max_distance_matrix.append(np.array(trial_max_distances))

        all_max_distance_matrices.append(max_distance_matrix)

    return all_max_distance_matrices
# %%

import numpy as np

def compute_index_of_max_poststim_distance_from_baseline_3d(v1_pca_results):
    """
    For each trial, computes the index (time point) at which the maximum 3D distance 
    from the centroid of the baseline period occurs in the response period.
    
    Returns:
        all_max_index_matrices: list of list of 1D numpy arrays
            Structure: [experiment][index][trial] = index of max distance
    """
    all_max_index_matrices = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        num_indices = len(basel_all)
        max_index_matrix = []

        for idx in range(num_indices):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]

            trial_max_indices = []
            for a, b in zip(trial_projs_basel, trial_projs_resp):
                if a.shape[1] >= 3 and b.shape[1] >= 3:
                    baseline_centroid = np.mean(a[:, :3], axis=0)
                    dists = np.linalg.norm(b[:, :3] - baseline_centroid, axis=1)
                    max_idx = int(np.argmax(dists))
                    trial_max_indices.append(max_idx)
                else:
                    trial_max_indices.append(np.nan)

            max_index_matrix.append(np.array(trial_max_indices))

        all_max_index_matrices.append(max_index_matrix)

    return all_max_index_matrices

# %%
import numpy as np

def compute_index_of_max_poststim_distance_from_baseline(
    v1_pca_results, num_pcs=3
):
    """
    For each trial, computes:
        - the index (time point) at which the maximum distance from the baseline centroid occurs
        - the SEM of the distances across time within that trial

    Args:
        v1_pca_results: dict containing trial projections
        num_pcs: number of principal components to use

    Returns:
        all_max_index_matrices: [experiment][index][trial] = max index
        all_sem_max_distance_matrices: [experiment][index][trial] = SEM of max distance (scalar per trial)
    """
    all_max_index_matrices = []
    all_sem_max_distance_matrices = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        num_indices = len(basel_all)
        max_index_matrix = []
        sem_max_dist_matrix = []

        for idx in range(num_indices):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]

            trial_max_indices = []
            trial_max_dists = []

            for a, b in zip(trial_projs_basel, trial_projs_resp):
                if a.shape[1] >= num_pcs and b.shape[1] >= num_pcs:
                    baseline_centroid = np.mean(a[:, :num_pcs], axis=0)
                    dists = np.linalg.norm(b[:, :num_pcs] - baseline_centroid, axis=1)

                    max_idx = int(np.argmax(dists))
                    max_dist = dists[max_idx]

                    trial_max_indices.append(max_idx)
                    trial_max_dists.append(max_dist)
                else:
                    trial_max_indices.append(np.nan)
                    trial_max_dists.append(np.nan)

            max_index_matrix.append(np.array(trial_max_indices))
            sem_value = np.nanstd(trial_max_dists, ddof=1) / np.sqrt(np.sum(~np.isnan(trial_max_dists)))
            sem_max_dist_matrix.append(sem_value)  # scalar per index

        all_max_index_matrices.append(max_index_matrix)
        all_sem_max_distance_matrices.append(sem_max_dist_matrix)

    return all_max_index_matrices, all_sem_max_distance_matrices

# %%
import numpy as np

def compute_sem_from_max_index_matrices(all_max_index_matrices):
    """
    Computes the SEM of trial max indices from a nested list of arrays.

    Args:
        all_max_index_matrices: list of list of 1D numpy arrays
            Structure: [experiment][index][trial] = index of max distance

    Returns:
        all_sem_matrices: list of list of SEM values
            Structure: [experiment][index] = SEM of trial max indices
    """
    all_sem_matrices = []

    for exp_data in all_max_index_matrices:
        exp_sem = []
        for trial_indices in exp_data:
            valid = ~np.isnan(trial_indices)
            if np.sum(valid) > 1:
                sem = np.std(trial_indices[valid], ddof=1) / np.sqrt(np.sum(valid))
            else:
                sem = np.nan
            exp_sem.append(sem)
        all_sem_matrices.append(exp_sem)

    return all_sem_matrices


# %%
import numpy as np

def compute_normalized_max_deviation(v1_pca_results, num_pcs=3):
    """
    Computes normalized max deviation of response from baseline centroid
    across all experiments and indices.
    
    Normalization is based on standard deviation of baseline trajectory.
    
    Returns:
        normalized_devs_all: list of list of np.arrays [experiment][index][trial]
    """
    normalized_devs_all = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']
        
        num_indices = len(basel_all)
        expt_norm_devs = []

        for idx in range(num_indices):
            trial_basel = basel_all[idx]
            trial_resp = resp_all[idx]

            trial_devs = []
            for a, b in zip(trial_basel, trial_resp):
                if a.shape[1] >= num_pcs and b.shape[1] >= num_pcs:
                    # Use first num_pcs
                    baseline = a[:, :num_pcs]
                    response = b[:, :num_pcs]

                    # Centroid of baseline
                    baseline_mean = baseline.mean(axis=0)

                    # Max deviation
                    dists = np.linalg.norm(response - baseline_mean, axis=1)
                    max_dev = np.max(dists)

                    # Baseline variability
                    baseline_flat = baseline.flatten()
                    baseline_std = np.std(baseline_flat)

                    if baseline_std > 0:
                        normalized = max_dev / baseline_std
                    else:
                        normalized = np.nan
                    
                    trial_devs.append(normalized)
                else:
                    trial_devs.append(np.nan)

            expt_norm_devs.append(np.array(trial_devs))

        normalized_devs_all.append(expt_norm_devs)

    return normalized_devs_all

# %%

def compute_path_length_and_curvature(v1_pca_results, num_pcs=3):
    """
    Computes path length and curvature (mean directional change) for baseline and response trajectories.

    Returns:
        metrics_all: list of list of dicts [experiment][index][trial] -> dict with path length and curvature
    """
    metrics_all = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        num_indices = len(basel_all)
        expt_metrics = []

        for idx in range(num_indices):
            trial_basel = basel_all[idx]
            trial_resp = resp_all[idx]

            trial_metrics = []
            for a, b in zip(trial_basel, trial_resp):
                if a.shape[1] >= num_pcs and b.shape[1] >= num_pcs:
                    baseline = a[:, :num_pcs]
                    response = b[:, :num_pcs]

                    metrics = {}

                    # Path length
                    metrics['baseline_path_length'] = np.sum(np.linalg.norm(np.diff(baseline, axis=0), axis=1))
                    metrics['response_path_length'] = np.sum(np.linalg.norm(np.diff(response, axis=0), axis=1))

                    # Curvature (mean angle between segments)
                    def mean_angle(traj):
                        diffs = np.diff(traj, axis=0)
                        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
                        diffs_unit = np.divide(diffs, norms, out=np.zeros_like(diffs), where=norms!=0)
                        angles = []
                        for i in range(1, len(diffs_unit)):
                            cos_sim = np.clip(np.dot(diffs_unit[i], diffs_unit[i-1]), -1.0, 1.0)
                            angle = np.arccos(cos_sim)
                            angles.append(angle)
                        return np.mean(angles) if angles else np.nan

                    metrics['baseline_curvature'] = mean_angle(baseline)
                    metrics['response_curvature'] = mean_angle(response)

                    trial_metrics.append(metrics)
                else:
                    trial_metrics.append({
                        'baseline_path_length': np.nan,
                        'response_path_length': np.nan,
                        'baseline_curvature': np.nan,
                        'response_curvature': np.nan
                    })

            expt_metrics.append(trial_metrics)
        metrics_all.append(expt_metrics)

    return metrics_all
# %%

def extract_metric_from_metrics_all(metrics_all, key):
    """
    Extracts and flattens a specific metric from the nested metrics_all structure.

    Args:
        metrics_all (list): Nested list of dicts from compute_path_length_and_curvature.
        key (str): Metric key to extract, e.g., 'response_path_length'.

    Returns:
        flat_array (np.ndarray): Flattened array of the specified metric.
    """
    values = [
        trial[key]
        for expt in metrics_all
        for index in expt
        for trial in index
        if key in trial and trial[key] is not None
    ]
    return np.array(values)

# %%
import numpy as np
from scipy.spatial import distance

def compute_centroid_distances(v1_pca_results, num_pcs=3, distance_type='euclidean'):
    """
    Computes the centroid positions of baseline and post-stim periods and their distance
    (Euclidean or Mahalanobis) for each trial, across all experiments and PCA indices.

    Args:
        v1_pca_results: dict containing 'single_expt_results'
        num_pcs: number of principal components to use (default=3)
        distance_type: 'euclidean' or 'mahalanobis'

    Returns:
        all_baseline_centroids: list of list of arrays [experiment][index][trial, num_pcs]
        all_poststim_centroids: list of list of arrays [experiment][index][trial, num_pcs]
        all_centroid_distances: list of list of arrays [experiment][index][trial] = distance
    """
    all_baseline_centroids = []
    all_poststim_centroids = []
    all_centroid_distances = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        expt_baseline_centroids = []
        expt_poststim_centroids = []
        expt_distances = []

        for idx in range(len(basel_all)):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]

            baseline_centroids = []
            poststim_centroids = []
            centroid_dists = []

            # compute covariance matrix across ALL baseline+poststim points for Mahalanobis
            if distance_type == 'mahalanobis':
                combined = np.vstack([np.vstack(trial_projs_basel), np.vstack(trial_projs_resp)])
                cov = np.cov(combined[:, :num_pcs].T)
                try:
                    inv_cov = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    inv_cov = np.linalg.pinv(cov)  # fallback if not invertible

            for a, b in zip(trial_projs_basel, trial_projs_resp):
                if a.shape[1] >= num_pcs and b.shape[1] >= num_pcs:
                    baseline_centroid = np.mean(a[:, :num_pcs], axis=0)
                    poststim_centroid = np.mean(b[:, :num_pcs], axis=0)

                    if distance_type == 'euclidean':
                        dist = np.linalg.norm(poststim_centroid - baseline_centroid)
                    elif distance_type == 'mahalanobis':
                        dist = distance.mahalanobis(poststim_centroid,
                                                    baseline_centroid,
                                                    inv_cov)
                    else:
                        raise ValueError("Unsupported distance_type")

                    baseline_centroids.append(baseline_centroid)
                    poststim_centroids.append(poststim_centroid)
                    centroid_dists.append(dist)
                else:
                    baseline_centroids.append(np.full(num_pcs, np.nan))
                    poststim_centroids.append(np.full(num_pcs, np.nan))
                    centroid_dists.append(np.nan)

            expt_baseline_centroids.append(np.vstack(baseline_centroids))
            expt_poststim_centroids.append(np.vstack(poststim_centroids))
            expt_distances.append(np.array(centroid_dists))

        all_baseline_centroids.append(expt_baseline_centroids)
        all_poststim_centroids.append(expt_poststim_centroids)
        all_centroid_distances.append(expt_distances)

    return all_baseline_centroids, all_poststim_centroids, all_centroid_distances

# %%

from scipy.spatial.distance import pdist, squareform
import numpy as np

def compute_pairwise_centroid_distances(centroid_data):
    """
    Computes pairwise distances between trial centroids within each experiment and index.

    Args:
        centroid_data: list of list of arrays [experiment][index][trial, num_pcs]

    Returns:
        all_pairwise_matrices: list of list of 2D arrays [experiment][index][trial x trial]
            Each array is a square matrix of pairwise distances between trials
    """
    all_pairwise_matrices = []

    for expt_centroids in centroid_data:
        expt_pairwise = []

        for centroids in expt_centroids:
            if centroids.shape[0] < 2 or np.isnan(centroids).all():
                expt_pairwise.append(np.full((centroids.shape[0], centroids.shape[0]), np.nan))
                continue

            # Compute pairwise Euclidean distances
            dists = squareform(pdist(centroids, metric='euclidean'))
            expt_pairwise.append(dists)

        all_pairwise_matrices.append(expt_pairwise)

    return all_pairwise_matrices
# %%
from scipy.spatial.distance import pdist, squareform
import numpy as np

def compute_flat_upper_centroid_distances(centroid_data):
    """
    Computes pairwise distances between trial centroids within each experiment and index,
    keeping only the upper triangle (excluding diagonal), flattened to 1D.

    Args:
        centroid_data: list of list of arrays [experiment][index][trial, num_pcs]

    Returns:
        all_upper_flat_dists: list of list of 1D arrays [experiment][index] = pairwise distances
    """
    all_upper_flat_dists = []

    for expt_centroids in centroid_data:
        expt_flat_dists = []

        for centroids in expt_centroids:
            # Check for valid 2D shape and at least 2 trials
            if (
                not isinstance(centroids, np.ndarray) or
                centroids.ndim != 2 or
                centroids.shape[0] < 2 or
                np.isnan(centroids).all()
            ):
                expt_flat_dists.append(np.array([]))
                continue

            # Remove rows with all NaNs (invalid trials)
            centroids_clean = centroids[~np.isnan(centroids).all(axis=1)]

            # Need at least 2 valid rows to compute distances
            if centroids_clean.shape[0] < 2:
                expt_flat_dists.append(np.array([]))
                continue

            # Compute pairwise Euclidean distances
            dists = squareform(pdist(centroids_clean, metric='euclidean'))

            # Get upper triangle (excluding diagonal), flatten
            upper_flat = dists[np.triu_indices_from(dists, k=1)]
            expt_flat_dists.append(upper_flat)

        all_upper_flat_dists.append(expt_flat_dists)

    return all_upper_flat_dists



# %%
import numpy as np
from scipy.stats import sem

def summarize_pairwise_distances(pairwise_distance_data):
    """
    Summarizes pairwise centroid distances with mean, std, and SEM for each index in each experiment.

    Args:
        pairwise_distance_data: list of list of 2D arrays [experiment][index][trial x trial]

    Returns:
        summary_stats: dict with keys 'mean', 'std', 'sem', each mapping to a list of lists [experiment][index]
    """
    means = []
    stds = []
    sems = []

    for expt_data in pairwise_distance_data:
        expt_means = []
        expt_stds = []
        expt_sems = []

        for dist_matrix in expt_data:
            # Extract upper triangle without diagonal to avoid double counting and self-comparison
            if dist_matrix.shape[0] < 2:
                expt_means.append(np.nan)
                expt_stds.append(np.nan)
                expt_sems.append(np.nan)
                continue
            if dist_matrix.ndim==2:
                upper_tri_vals = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            else:
                upper_tri_vals = dist_matrix
                
            expt_means.append(np.nanmean(upper_tri_vals))
            expt_stds.append(np.nanstd(upper_tri_vals))
            expt_sems.append(sem(upper_tri_vals, nan_policy='omit'))

        means.append(expt_means)
        stds.append(expt_stds)
        sems.append(expt_sems)

    return {
        'mean': means,
        'std': stds,
        'sem': sems
    }
# %%
import numpy as np

def compute_pointwise_distances(v1_pca_results, num_pcs=3, distance_type='euclidean', reduction='sum'):
    """
    Computes point-by-point distances between baseline and post-stim period projections
    for each trial, across all experiments and PCA indices.

    Args:
        v1_pca_results: dict containing 'single_expt_results'
        num_pcs: number of principal components to use (default=3)
        distance_type: 'euclidean' or 'manhattan'
        reduction: 'sum' or 'mean' across time points

    Returns:
        all_pointwise_distances: list of list of arrays [experiment][index][trial] = distance
    """
    all_pointwise_distances = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        expt_distances = []

        for idx in range(len(basel_all)):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]

            trial_dists = []

            for a, b in zip(trial_projs_basel, trial_projs_resp):
                if a.shape[1] >= num_pcs and b.shape[1] >= num_pcs:
                    # Match length of a and b by truncating to shorter one
                    min_len = min(a.shape[0], b.shape[0])
                    a_cut = a[:min_len, :num_pcs]
                    b_cut = b[:min_len, :num_pcs]

                    if distance_type == 'euclidean':
                        dists = np.linalg.norm(a_cut - b_cut, axis=1)
                    elif distance_type == 'manhattan':
                        dists = np.sum(np.abs(a_cut - b_cut), axis=1)
                    else:
                        raise ValueError("Unsupported distance_type. Use 'euclidean' or 'manhattan'.")

                    if reduction == 'sum':
                        dist = np.sum(dists)
                    elif reduction == 'mean':
                        dist = np.mean(dists)
                    else:
                        raise ValueError("Unsupported reduction. Use 'sum' or 'mean'.")

                    trial_dists.append(dist)
                else:
                    trial_dists.append(np.nan)

            expt_distances.append(np.array(trial_dists))

        all_pointwise_distances.append(expt_distances)

    return all_pointwise_distances

# %%
import numpy as np

def compute_pairwise_pointwise_distances(v1_pca_results, num_pcs=3, distance_type='euclidean', reduction='sum'):
    """
    Computes pairwise point-by-point distances between all trials, separately for
    baseline and post-stim projections, across all experiments and PCA indices.

    Args:
        v1_pca_results: dict containing 'single_expt_results'
        num_pcs: number of principal components to use (default=3)
        distance_type: 'euclidean' or 'manhattan'
        reduction: 'sum' or 'mean'

    Returns:
        all_baseline_pairwise: list of list of 2D arrays [experiment][index][trial x trial]
        all_poststim_pairwise: list of list of 2D arrays [experiment][index][trial x trial]
    """
    all_baseline_pairwise = []
    all_poststim_pairwise = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        expt_baseline_matrices = []
        expt_poststim_matrices = []

        for idx in range(len(basel_all)):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]
            n_trials = len(trial_projs_basel)

            baseline_matrix = np.full((n_trials, n_trials), np.nan)
            poststim_matrix = np.full((n_trials, n_trials), np.nan)

            for i in range(n_trials):
                a_basel = trial_projs_basel[i]
                a_resp = trial_projs_resp[i]

                for j in range(n_trials):
                    b_basel = trial_projs_basel[j]
                    b_resp = trial_projs_resp[j]

                    # Only compute if both trials have valid shape
                    if (
                        a_basel.shape[1] >= num_pcs and b_basel.shape[1] >= num_pcs and
                        a_resp.shape[1] >= num_pcs and b_resp.shape[1] >= num_pcs
                    ):
                        # --- Baseline ---
                        min_len_basel = min(a_basel.shape[0], b_basel.shape[0])
                        a_b = a_basel[:min_len_basel, :num_pcs]
                        b_b = b_basel[:min_len_basel, :num_pcs]

                        if distance_type == 'euclidean':
                            dists_b = np.linalg.norm(a_b - b_b, axis=1)
                        elif distance_type == 'manhattan':
                            dists_b = np.sum(np.abs(a_b - b_b), axis=1)
                        else:
                            raise ValueError("Unsupported distance_type")

                        baseline_matrix[i, j] = np.sum(dists_b) if reduction == 'sum' else np.mean(dists_b)

                        # --- Post-stim ---
                        min_len_resp = min(a_resp.shape[0], b_resp.shape[0])
                        a_r = a_resp[:min_len_resp, :num_pcs]
                        b_r = b_resp[:min_len_resp, :num_pcs]

                        if distance_type == 'euclidean':
                            dists_r = np.linalg.norm(a_r - b_r, axis=1)
                        elif distance_type == 'manhattan':
                            dists_r = np.sum(np.abs(a_r - b_r), axis=1)

                        poststim_matrix[i, j] = np.sum(dists_r) if reduction == 'sum' else np.mean(dists_r)

            expt_baseline_matrices.append(baseline_matrix)
            expt_poststim_matrices.append(poststim_matrix)
            
            # breakpoint()

        all_baseline_pairwise.append(expt_baseline_matrices)
        all_poststim_pairwise.append(expt_poststim_matrices)

    return all_baseline_pairwise, all_poststim_pairwise

# %%

import numpy as np

def compute_pairwise_trial_distances_static(v1_pca_results, num_pcs=3, mode='static'):
    """
    Computes pairwise trial distances using a static projection (no time dimension).

    Args:
        v1_pca_results: dict containing 'single_expt_results'
        num_pcs: number of principal components to use (default=3)
        mode: must be 'static' for this implementation

    Returns:
        all_baseline_pairwise: list of list of 2D arrays [experiment][stim][trial x trial]
        all_poststim_pairwise: list of list of 2D arrays [experiment][stim][trial x trial]
    """
    if mode != 'static':
        raise ValueError("Only 'static' mode is supported by this function.")

    all_baseline_pairwise = []
    all_poststim_pairwise = []
    
    # dist = np.linalg.norm(trial_projs_basel[iStim][iTrial1][:num_pcs]-trial_projs_basel[iStim][iTrial2][:num_pcs])
    # basel_dist.append(dist)
    
    # dist = np.linalg.norm(trial_projs_resp[iStim][iTrial1][:num_pcs]-trial_projs_resp[iStim][iTrial2][:num_pcs])
    # resp_dist.append(dist)
    
    
    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        expt_baseline_matrices = []
        expt_poststim_matrices = []

        for trial_projs_basel, trial_projs_resp in zip(basel_all, resp_all):
            n_trials = len(trial_projs_basel)

            baseline_matrix = np.full((n_trials, n_trials), np.nan)
            poststim_matrix = np.full((n_trials, n_trials), np.nan)

            for i in range(n_trials):
                a_basel = trial_projs_basel[i]
                a_resp = trial_projs_resp[i]

                for j in range(n_trials):
                    b_basel = trial_projs_basel[j]
                    b_resp = trial_projs_resp[j]

                    # Check valid input shape
                    if (
                        a_basel.shape[1] >= num_pcs and b_basel.shape[1] >= num_pcs and
                        a_resp.shape[1] >= num_pcs and b_resp.shape[1] >= num_pcs
                    ):
                        # Take mean projection across time
                        a_b = np.mean(a_basel[:, :num_pcs], axis=0)
                        b_b = np.mean(b_basel[:, :num_pcs], axis=0)
                        # baseline_matrix[i, j] = np.linalg.norm(a_b - b_b)
                        baseline_matrix[i, j] = np.linalg.norm(a_basel[:,:num_pcs]-b_basel[:,:num_pcs])

                        a_r = np.mean(a_resp[:, :num_pcs], axis=0)
                        b_r = np.mean(b_resp[:, :num_pcs], axis=0)
                        # poststim_matrix[i, j] = np.linalg.norm(a_r - b_r)
                        poststim_matrix[i, j] = np.linalg.norm(a_resp[:,:num_pcs]-b_resp[:,:num_pcs])


            expt_baseline_matrices.append(baseline_matrix)
            expt_poststim_matrices.append(poststim_matrix)

        all_baseline_pairwise.append(expt_baseline_matrices)
        all_poststim_pairwise.append(expt_poststim_matrices)
        
        # BREAKPOINT HERE
        # breakpoint()  # This will drop you into an interactive debugger

    return all_baseline_pairwise, all_poststim_pairwise

# %%

import numpy as np

def extract_upper_triangle_nested(all_distance_matrices):
    """
    Extracts the upper triangle (excluding diagonal) from each trial x trial matrix,
    returning a nested structure of 1D arrays [experiment][index].

    Args:
        all_distance_matrices: list of list of 2D arrays (n_trials x n_trials)

    Returns:
        all_upper_dists: list of list of 1D arrays [experiment][index]
    """
    all_upper_dists = []

    for expt_matrices in all_distance_matrices:
        expt_upper = []

        for dist_matrix in expt_matrices:
            if dist_matrix.ndim != 2 or np.isnan(dist_matrix).all():
                expt_upper.append(np.array([np.nan]))
                continue

            # Get upper triangle indices (excluding diagonal)
            triu_indices = np.triu_indices(dist_matrix.shape[0], k=1)
            upper_values = dist_matrix[triu_indices]
            expt_upper.append(upper_values)

        all_upper_dists.append(expt_upper)

    return all_upper_dists
# %%
from scipy.stats import pearsonr
import numpy as np

def correlate_baseline_poststim_dists(all_baseline_dists_upper, all_poststim_dists_upper):
    """
    Computes Pearson correlation (r, p) between baseline and post-stim distance vectors
    for each [experiment][index] entry.

    Args:
        all_baseline_dists_upper: list of list of 1D arrays [experiment][index]
        all_poststim_dists_upper: list of list of 1D arrays [experiment][index]

    Returns:
        correlations: list of list of tuples (r, p) [experiment][index]
    """
    correlations = []

    for base_expt, post_expt in zip(all_baseline_dists_upper, all_poststim_dists_upper):
        expt_results = []

        for base_vec, post_vec in zip(base_expt, post_expt):
            if len(base_vec) == 0 or len(post_vec) == 0 or np.isnan(base_vec).all() or np.isnan(post_vec).all():
                expt_results.append((np.nan, np.nan))
                continue

            # Remove any NaN pairs
            valid_mask = ~np.isnan(base_vec) & ~np.isnan(post_vec)
            base_clean = base_vec[valid_mask]
            post_clean = post_vec[valid_mask]

            if len(base_clean) < 2:
                expt_results.append((np.nan, np.nan))
                continue

            r, p = pearsonr(base_clean, post_clean)
            expt_results.append((r, p))

        correlations.append(expt_results)

    return correlations

# %%
from scipy.stats import pearsonr
import numpy as np

def correlate_per_experiment_flat(all_baseline_dists_upper, all_poststim_dists_upper):
    """
    Concatenates all baseline and post-stim distance vectors across indices (stim types),
    and computes one Pearson correlation per experiment.

    Args:
        all_baseline_dists_upper: list of list of 1D arrays [experiment][index]
        all_poststim_dists_upper: list of list of 1D arrays [experiment][index]

    Returns:
        correlations: list of tuples (r, p) for each experiment
        points_per_experiment: list of tuples (x_vals, y_vals) used for correlation per experiment
    """
    correlations = []
    
    points_per_experiment = []

    for base_expt, post_expt in zip(all_baseline_dists_upper, all_poststim_dists_upper):
        base_all = []
        post_all = []

        for base_vec, post_vec in zip(base_expt, post_expt):
            if len(base_vec) == 0 or len(post_vec) == 0:
                continue

            valid_mask = ~np.isnan(base_vec) & ~np.isnan(post_vec)
            base_clean = base_vec[valid_mask]
            post_clean = post_vec[valid_mask]

            if len(base_clean) >= 2:
                base_all.append(base_clean)
                post_all.append(post_clean)

        if len(base_all) == 0:
            correlations.append((np.nan, np.nan))
            points_per_experiment.append((np.array([]), np.array([])))
            continue

        base_concat = np.concatenate(base_all)
        post_concat = np.concatenate(post_all)

        valid_mask = ~np.isnan(base_concat) & ~np.isnan(post_concat)
        base_final = base_concat[valid_mask]
        post_final = post_concat[valid_mask]

        if len(base_final) < 2:
            correlations.append((np.nan, np.nan))
        else:
            r, p = pearsonr(base_final, post_final)
            correlations.append((r, p))
        
        
        points_per_experiment.append((base_final, post_final))
        
    return correlations, points_per_experiment
# %%
import numpy as np

def compute_pairwise_pointwise_vectors(v1_pca_results, num_pcs=3, distance_type='euclidean'):
    """
    Computes point-by-point vector distances between all trial pairs for both baseline
    and post-stim periods, across experiments and PCA indices.

    Args:
        v1_pca_results: dict containing 'single_expt_results'
        num_pcs: number of principal components to use
        distance_type: 'euclidean' or 'manhattan'

    Returns:
        all_baseline_vectors: list of list of 3D arrays [experiment][index][trial x trial x time]
        all_poststim_vectors: list of list of 3D arrays [experiment][index][trial x trial x time]
    """
    all_baseline_vectors = []
    all_poststim_vectors = []

    for expt in v1_pca_results['single_expt_results']:
        basel_all = expt['trial_projs_basel']
        resp_all = expt['trial_projs_resp']

        expt_baseline_vecs = []
        expt_poststim_vecs = []

        for idx in range(len(basel_all)):
            trial_projs_basel = basel_all[idx]
            trial_projs_resp = resp_all[idx]
            n_trials = len(trial_projs_basel)

            # Determine minimum length across all trials for alignment
            min_len_basel = min([a.shape[0] for a in trial_projs_basel if a.shape[1] >= num_pcs], default=0)
            min_len_resp = min([a.shape[0] for a in trial_projs_resp if a.shape[1] >= num_pcs], default=0)

            if min_len_basel == 0 or min_len_resp == 0:
                # If no valid data, fill with NaNs
                expt_baseline_vecs.append(np.full((n_trials, n_trials, 1), np.nan))
                expt_poststim_vecs.append(np.full((n_trials, n_trials, 1), np.nan))
                continue

            baseline_vecs = np.full((n_trials, n_trials, min_len_basel), np.nan)
            poststim_vecs = np.full((n_trials, n_trials, min_len_resp), np.nan)

            for i in range(n_trials):
                for j in range(n_trials):
                    # Baseline
                    a_b = trial_projs_basel[i]
                    b_b = trial_projs_basel[j]

                    if a_b.shape[1] >= num_pcs and b_b.shape[1] >= num_pcs:
                        A = a_b[:min_len_basel, :num_pcs]
                        B = b_b[:min_len_basel, :num_pcs]

                        if distance_type == 'euclidean':
                            vec = np.linalg.norm(A - B, axis=1)
                        elif distance_type == 'manhattan':
                            vec = np.sum(np.abs(A - B), axis=1)
                        else:
                            raise ValueError("Unsupported distance_type")

                        baseline_vecs[i, j, :] = vec

                    # Post-stim
                    a_r = trial_projs_resp[i]
                    b_r = trial_projs_resp[j]

                    if a_r.shape[1] >= num_pcs and b_r.shape[1] >= num_pcs:
                        A = a_r[:min_len_resp, :num_pcs]
                        B = b_r[:min_len_resp, :num_pcs]

                        if distance_type == 'euclidean':
                            vec = np.linalg.norm(A - B, axis=1)
                        elif distance_type == 'manhattan':
                            vec = np.sum(np.abs(A - B), axis=1)

                        poststim_vecs[i, j, :] = vec

            expt_baseline_vecs.append(baseline_vecs)
            expt_poststim_vecs.append(poststim_vecs)

        all_baseline_vectors.append(expt_baseline_vecs)
        all_poststim_vectors.append(expt_poststim_vecs)

    return all_baseline_vectors, all_poststim_vectors



# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def histogram_colored_by_pvalue(covariance, p_values, bins=10, cmap='viridis', center_bins_around_zero=False):
    """
    Plot a histogram of covariance values, color-coded by the average p-value in each bin.

    Parameters:
    - covariance: array-like, covariance values
    - p_values: array-like, p-values associated with each covariance
    - bins: int or sequence, optional number of bins or bin edges (default=10)
    - cmap: str, colormap name (default='viridis')
    - center_bins_around_zero: bool, if True, centers bins around 0 and uses symmetric x-limits
    """
    covariance = np.asarray(covariance)
    p_values = np.asarray(p_values)

    if center_bins_around_zero:
        max_val = np.max(np.abs(covariance))
        if isinstance(bins, int):
            # Ensure even number of bins
            if bins % 2 != 0:
                bins += 1
            bin_edges = np.linspace(-max_val, max_val, bins + 1)
        else:
            bin_edges = np.asarray(bins)
    else:
        bin_edges = np.histogram_bin_edges(covariance, bins=bins)

    hist, _ = np.histogram(covariance, bins=bin_edges)
    bin_indices = np.digitize(covariance, bin_edges, right=True)

    avg_p_values = []
    for i in range(1, len(bin_edges)):
        p_in_bin = p_values[bin_indices == i]
        if len(p_in_bin) > 0:
            avg_p_values.append(np.mean(p_in_bin))
        else:
            avg_p_values.append(np.nan)

    # Normalize p-values to use as color intensities
    norm = plt.Normalize(np.nanmin(avg_p_values), np.nanmax(avg_p_values))
    colors = cm.get_cmap(cmap)(norm(avg_p_values))

    # Plot histogram bars manually
    fig, ax = plt.subplots()
    for i in range(len(hist)):
        center = (bin_edges[i] + bin_edges[i+1]) / 2
        width = bin_edges[i+1] - bin_edges[i]
        ax.bar(center, hist[i], width=width, color=colors[i], edgecolor='black', align='center')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Average p-value')

    ax.set_xlabel('r value')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of r value\nColored by Average p-value per Bin')

    if center_bins_around_zero:
        ax.set_xlim(bin_edges[0], bin_edges[-1])

    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def histogram_color_by_significant_p(covariance, p_values, bins=10, cmap='viridis', center_bins_around_zero=False, p_threshold=0.05):
    """
    Plot a histogram of covariance values, color-coded by the number of p-values below a given threshold in each bin.

    Parameters:
    - covariance: array-like, covariance values
    - p_values: array-like, p-values associated with each covariance
    - bins: int or sequence, optional number of bins or bin edges (default=10)
    - cmap: str, colormap name (default='viridis')
    - center_bins_around_zero: bool, if True, centers bins around 0 and uses symmetric x-limits
    - p_threshold: float, threshold for significance (default=0.05)
    """
    covariance = np.asarray(covariance)
    p_values = np.asarray(p_values)

    if center_bins_around_zero:
        max_val = np.max(np.abs(covariance))
        if isinstance(bins, int):
            if bins % 2 != 0:
                bins += 1
            bin_edges = np.linspace(-max_val, max_val, bins + 1)
        else:
            bin_edges = np.asarray(bins)
    else:
        bin_edges = np.histogram_bin_edges(covariance, bins=bins)

    hist, _ = np.histogram(covariance, bins=bin_edges)
    bin_indices = np.digitize(covariance, bin_edges, right=True)

    sig_p_counts = []
    for i in range(1, len(bin_edges)):
        p_in_bin = p_values[bin_indices == i]
        count_sig = np.sum(p_in_bin < p_threshold)
        sig_p_counts.append(count_sig)

    # Normalize for colormap
    norm = plt.Normalize(vmin=0, vmax=max(sig_p_counts) if max(sig_p_counts) > 0 else 1)
    colors = cm.get_cmap(cmap)(norm(sig_p_counts))

    fig, ax = plt.subplots()
    for i in range(len(hist)):
        center = (bin_edges[i] + bin_edges[i+1]) / 2
        width = bin_edges[i+1] - bin_edges[i]
        ax.bar(center, hist[i], width=width, color=colors[i], edgecolor='black', align='center')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'Count of p < {p_threshold}')

    ax.set_xlabel('r value')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of r value\nColored by Significant p-value Count')

    if center_bins_around_zero:
        ax.set_xlim(bin_edges[0], bin_edges[-1])

    plt.tight_layout()
    plt.show()
# %%

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def histogram_color_by_significant_proportion(covariance, p_values, bins=10, cmap='viridis',
                                               center_bins_around_zero=False, p_threshold=0.05):
    """
    Plot a histogram of covariance values, color-coded by the proportion of p-values < p_threshold in each bin.

    Parameters:
    - covariance: array-like, covariance values
    - p_values: array-like, associated p-values
    - bins: int or sequence, number of bins or bin edges
    - cmap: str, matplotlib colormap name (default='viridis')
    - center_bins_around_zero: bool, if True, bins are symmetric around 0
    - p_threshold: float, threshold for significance (default=0.05)
    """
    covariance = np.asarray(covariance)
    p_values = np.asarray(p_values)
    
    import matplotlib as mpl

    
    # Illustrator-compatible settings
    mpl.rcParams['pdf.fonttype'] = 42 # To make text editable
    mpl.rcParams['ps.fonttype']  = 42 # To make text editable
    mpl.rcParams['svg.fonttype'] = 'none' # To make text editable


    if center_bins_around_zero:
        max_val = np.max(np.abs(covariance))
        if isinstance(bins, int):
            if bins % 2 != 0:
                bins += 1
            bin_edges = np.linspace(-max_val, max_val, bins + 1)
        else:
            bin_edges = np.asarray(bins)
    else:
        bin_edges = np.histogram_bin_edges(covariance, bins=bins)

    hist, _ = np.histogram(covariance, bins=bin_edges)
    bin_indices = np.digitize(covariance, bin_edges, right=True)

    proportions = []
    for i in range(1, len(bin_edges)):
        p_in_bin = p_values[bin_indices == i]
        if len(p_in_bin) > 0:
            prop_sig = np.sum(p_in_bin < p_threshold) / len(p_in_bin)
        else:
            prop_sig = np.nan
        proportions.append(prop_sig)

    # Normalize for colormap
    norm = plt.Normalize(0, 1)
    colors = cm.get_cmap(cmap)(norm(proportions))

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(hist)):
        center = (bin_edges[i] + bin_edges[i+1]) / 2
        width = bin_edges[i+1] - bin_edges[i]
        color = colors[i] if not np.isnan(proportions[i]) else 'gray'
        ax.bar(center, hist[i], width=width, color=color, edgecolor='black', align='center')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'Proportion of p < {p_threshold}')

    ax.set_xlabel('r value')
    ax.set_ylabel('Count')
    ax.set_title(f'Histogram of r value\nColored by Proportion of Significant p-values (<{p_threshold})')

    if center_bins_around_zero:
        ax.set_xlim(bin_edges[0], bin_edges[-1])

    plt.tight_layout()
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def overlay_significant_histogram(r_values, p_values, bins=10, center_bins_around_zero=False,
                                   p_threshold=0.05, alpha=1.0, bar_gap=0.0, median_label="median",
                                   xlim=None, ylim=None, bar_width=None):
    """
    Overlayed histogram of r-values:
    - Black bars: all values.
    - White bars: values with p < threshold.
    - Arrow and label for the median r-value above the bars.

    Parameters:
    - r_values: array-like, r-values (e.g., correlations)
    - p_values: array-like, associated p-values
    - bins: int or array-like, number of bins or bin edges
    - center_bins_around_zero: bool, center bins symmetrically around 0
    - p_threshold: float, significance threshold (default=0.05)
    - alpha: float, bar transparency (default=1.0)
    - bar_gap: float, proportion of bin width to leave as spacing between bars
    - median_label: str, text label for the median arrow
    - xlim: tuple (xmin, xmax), limits for x-axis (optional)
    - ylim: tuple (ymin, ymax), limits for y-axis (optional)
    - bar_width: float, fixed bar width (optional). If None, auto-computed from bins.
    """
    r_values = np.asarray(r_values)
    p_values = np.asarray(p_values)

    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype']  = 42
    plt.rcParams['svg.fonttype'] = 'none'

    # Define bins
    if center_bins_around_zero:
        max_val = np.max(np.abs(r_values))
        if isinstance(bins, int):
            if bins % 2 != 0:
                bins += 1
            bin_edges = np.linspace(-max_val, max_val, bins + 1)
        else:
            bin_edges = np.asarray(bins)
    else:
        bin_edges = np.histogram_bin_edges(r_values, bins=bins)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_widths = np.diff(bin_edges)

    if bar_width is None:
        bar_widths = bin_widths * (1 - bar_gap)
    else:
        bar_widths = np.full_like(bin_centers, bar_width, dtype=float)

    # Histogram counts
    full_hist, _ = np.histogram(r_values, bins=bin_edges)
    sig_hist, _ = np.histogram(r_values[p_values < p_threshold], bins=bin_edges)

    # Plot setup
    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Plot all bars
    ax.bar(bin_centers, full_hist, width=bar_widths, color='black', alpha=alpha,
           align='center', edgecolor='black', label='All', zorder=1)
    ax.bar(bin_centers, sig_hist, width=bar_widths, color='white', alpha=alpha,
           align='center', edgecolor='black', label=f'p < {p_threshold}', zorder=2)

    # Median arrow and label
    median_val = np.median(r_values)
    ymax = np.max(full_hist)
    arrow_y_bottom = ymax + ymax * 0.03
    arrow_y_top = ymax + ymax * 0.15
    text_y = arrow_y_top + ymax * 0.05

    ax.annotate('', xy=(median_val, arrow_y_bottom), xytext=(median_val, arrow_y_top),
                arrowprops=dict(facecolor='black', width=1.5, headwidth=6), zorder=10)
    ax.text(median_val, text_y, f"{median_label} = {median_val:.2f}",
            ha='center', va='bottom', fontsize=9, color='black', zorder=10)

    # Final formatting
    ax.set_xlabel('r-value')
    ax.set_ylabel('Count')
    ax.set_title(f'Overlayed Histogram of r-values\n(White = p < {p_threshold})')
    ax.legend()

    # Set axis limits
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(top=text_y + ymax * 0.1)

    if xlim is not None:
        ax.set_xlim(xlim)
    elif center_bins_around_zero:
        ax.set_xlim(bin_edges[0], bin_edges[-1])

    plt.tight_layout()
    plt.show()


# %%
import numpy as np

def extract_cum_var_explained_matrix(v1_pca_results, n_pcs):
    return np.vstack([
        expt['pca_results']['cum_var_explained'][:n_pcs]
        for expt in v1_pca_results['single_expt_results']
    ])

# %%
import matplotlib.pyplot as plt
import numpy as np

def plot_two_cum_var_explained_with_sem(cum_var_matrix_1, cum_var_matrix_2, label1='Group 1', label2='Group 2',title='ECDF cumulative variance explained by PCs'):
    """
    Plots mean ± SEM of cumulative variance explained for two groups.

    Args:
        cum_var_matrix_1 (np.ndarray): 2D array for first group (experiments x PCs)
        cum_var_matrix_2 (np.ndarray): 2D array for second group (experiments x PCs)
        label1 (str): Label for first group
        label2 (str): Label for second group
    """
    def compute_mean_sem(matrix):
        mean = np.nanmean(matrix, axis=0)
        sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(matrix), axis=0))
        return mean, sem

    mean1, sem1 = compute_mean_sem(cum_var_matrix_1)
    mean2, sem2 = compute_mean_sem(cum_var_matrix_2)

    pcs = np.arange(1, cum_var_matrix_1.shape[1] + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(pcs, mean1, label=f'{label1} Mean', color='blue')
    plt.fill_between(pcs, mean1 - sem1, mean1 + sem1, color='blue', alpha=0.3)

    plt.plot(pcs, mean2, label=f'{label2} Mean', color='red')
    plt.fill_between(pcs, mean2 - sem2, mean2 + sem2, color='red', alpha=0.3)

    plt.xlabel('Principal Component (PC)')
    plt.ylabel('Cumulative Variance Explained')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%

import numpy as np

def expand_trial_pvals(dist_stims, stim_trial_p):
    """
    Expand stim_trial_p to match dist_stims.
    Each contiguous block of identical stim IDs in dist_stims 
    gets assigned the next p-value from stim_trial_p.
    """
    dist_stims = np.asarray(dist_stims)
    stim_trial_p = np.asarray(stim_trial_p)

    expanded = []
    trial_idx = 0

    # Iterate over runs of consecutive stim IDs
    start = 0
    for i in range(1, len(dist_stims) + 1):
        if i == len(dist_stims) or dist_stims[i] != dist_stims[i - 1]:
            block_len = i - start
            if trial_idx >= len(stim_trial_p):
                raise ValueError("Not enough stim_trial_p values for dist_stims blocks")
            expanded.extend([stim_trial_p[trial_idx]] * block_len)
            trial_idx += 1
            start = i

    if trial_idx != len(stim_trial_p):
        raise ValueError("Unused stim_trial_p values remain after expansion")

    return np.array(expanded)

# %%

# %%

def plot_combined_scatter_by_significance_with_fitlines(
    pca_results, save_path=None, equal_xy=False, fit_line=True,
    xlim=None, ylim=None, p_values='by_correlation'
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from scipy.stats import pearsonr, linregress

    # Illustrator-compatible settings
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    all_x = []
    all_y = []
    all_significant = []

    significant_fit_lines = []

    if p_values == 'by_stim_type':
        # New behavior: expand trial-level p-values
        dist_stims = pca_results['dist_stims']
        stim_trial_p = pca_results['stim_trial_p']
        expanded_p = expand_trial_pvals(dist_stims, stim_trial_p)
        all_significant = expanded_p < 0.05

    # Gather and process each experiment
    for result in pca_results['single_expt_results']:
        x = np.array(result['basel_dists'])
        y = np.array(result['resp_dists'])

        if len(x) != len(y) or len(x) == 0:
            continue

        if p_values == "by_correlation":
            # Old behavior: correlation-based significance
            r, p = pearsonr(x, y)
            is_significant = [p < 0.05] * len(x)

            if p < 0.05:
                slope, intercept, *_ = linregress(x, y)
                x_fit = np.linspace(np.min(x), np.max(x), 100)
                y_fit = slope * x_fit + intercept
                significant_fit_lines.append((x_fit, y_fit))
            
            all_significant.extend(is_significant)
  
        

        all_x.extend(x)
        all_y.extend(y)
        # all_significant.extend(is_significant)

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_significant = np.array(all_significant)
    # all_significant = np.asarray(all_significant, dtype=bool)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot points
    ax.scatter(all_x[~all_significant], all_y[~all_significant],
               color='lightgray', edgecolor='white', label='p ≥ 0.05', s=40)
    ax.scatter(all_x[all_significant], all_y[all_significant],
               color='k', edgecolor='white', label='p < 0.05', s=40)

    # Plot individual fit lines (only relevant for correlation-based mode)
    for x_fit, y_fit in significant_fit_lines:
        ax.plot(x_fit, y_fit, color='steelblue', linestyle='-', alpha=0.9,
                label='_nolegend_', linewidth=2)

    # Overall regression line
    if fit_line and len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_min, x_max = ax.get_xlim()
        x_line = np.linspace(x_min, x_max + 100, 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='black', linestyle='--', label='Overall Fit')
    
        # Format p-value safely
        if p_val == 0.0:
            p_str = "p < 1e-308"   # underflow in float64
        elif p_val >= 0.001:
            p_str = f"p = {p_val:.3f}"
        else:
            p_str = f"p = {p_val:.2e}"
    
        stats_text = f"r = {r_val:.2f}, {p_str}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', ha='left')


    # Labels and formatting
    ax.set_xlabel('Baseline Distance')
    ax.set_ylabel('Response Distance')
    ax.set_title('Combined Scatter with Significant Fits')

    if equal_xy:
        min_val = min(np.min(all_x), np.min(all_y))
        max_val = max(np.max(all_x), np.max(all_y))
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
    else:
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')


# %%
def get_scaled_fonts(fig_width_inch, base_width=6.0, base_fontsize=12):
    """
    Returns a dictionary of font sizes scaled to the figure width.
    
    Parameters:
    - fig_width_inch: Width of the figure in inches
    - base_width: Reference width (in inches) that uses base_fontsize (default: 6.0 inches)
    - base_fontsize: Reference font size for base_width (default: 12 pt)

    Returns:
    - dict with font sizes for title, labels, ticks, legend, and text
    """
    scale = fig_width_inch / base_width
    return {
        'title': base_fontsize * 1.2 * scale,
        'label': base_fontsize * scale,
        'tick': base_fontsize * 0.9 * scale,
        'legend': base_fontsize * 0.9 * scale,
        'text': base_fontsize * scale
    }

# %%
def find_trial_pairs_by_threshold(
    all_baseline_dists,
    all_poststim_dists,
    baseline_thresh=0.9,
    poststim_thresh=0.9,
    baseline_mode='above',
    poststim_mode='above',
    all_pointwise_dists=None,
    pointwise_thresh=0.2,
    pointwise_mode='above'
):
    """
    Identifies trial pairs meeting baseline and poststim distribution thresholds,
    with optional filtering based on trial-level baseline→poststim distances.

    Args:
        all_baseline_dists: list of list of 2D arrays [experiment][index]
        all_poststim_dists: same structure
        baseline_thresh, poststim_thresh: quantiles (0–1)
        baseline_mode, poststim_mode: 'above' or 'below'
        all_pointwise_dists: optional list of list of 1D arrays [experiment][index][trial]
        pointwise_thresh: float threshold for per-trial filter
        pointwise_mode: 'above' or 'below'

    Returns:
        selected_trial_pairs: list of list of lists [experiment][index] = list of (i, j) pairs
    """
    selected_trial_pairs = []

    for exp_idx, (expt_baseline, expt_poststim) in enumerate(zip(all_baseline_dists, all_poststim_dists)):
        expt_selected = []

        for idx, (base_mat, post_mat) in enumerate(zip(expt_baseline, expt_poststim)):
            if base_mat.ndim != 2 or np.isnan(base_mat).all():
                expt_selected.append([])
                continue

            n_trials = base_mat.shape[0]
            triu_i, triu_j = np.triu_indices(n_trials, k=1)
            base_vals = base_mat[triu_i, triu_j]
            post_vals = post_mat[triu_i, triu_j]

            # Thresholds
            base_cut = np.nanquantile(base_vals, baseline_thresh)
            post_cut = np.nanquantile(post_vals, poststim_thresh)

            # Threshold masks
            base_mask = base_vals > base_cut if baseline_mode == 'above' else base_vals < base_cut
            post_mask = post_vals > post_cut if poststim_mode == 'above' else post_vals < post_cut

            combined_mask = base_mask & post_mask

            # Optionally apply pointwise distance filter
            if all_pointwise_dists is not None:
                try:
                    trial_dists = all_pointwise_dists[exp_idx][idx]
                except (IndexError, TypeError):
                    trial_dists = np.full(n_trials, np.nan)

                if pointwise_mode == 'above':
                    valid_trials = trial_dists > pointwise_thresh
                elif pointwise_mode == 'below':
                    valid_trials = trial_dists < pointwise_thresh
                else:
                    raise ValueError("pointwise_mode must be 'above' or 'below'")

                # Make a mask where both trials in the pair must pass pointwise threshold
                pointwise_mask = np.array([
                    valid_trials[i] and valid_trials[j]
                    for i, j in zip(triu_i, triu_j)
                ])
                combined_mask = combined_mask & pointwise_mask

            selected_pairs = list(zip(triu_i[combined_mask], triu_j[combined_mask]))
            expt_selected.append(selected_pairs)

        selected_trial_pairs.append(expt_selected)

    return selected_trial_pairs
# %%

def find_trial_pairs_by_absolute_threshold(
    all_baseline_dists,
    all_poststim_dists,
    baseline_thresh=0.9,
    poststim_thresh=0.9,
    baseline_mode='above',
    poststim_mode='above',
    all_pointwise_dists=None,
    pointwise_thresh=0.2,
    pointwise_mode='above'
):
    """
    Identifies trial pairs meeting baseline and poststim absolute distance thresholds,
    with optional filtering based on trial-level baseline→poststim distances.

    Args:
        all_baseline_dists: list of list of 2D arrays [experiment][index]
        all_poststim_dists: same structure
        baseline_thresh, poststim_thresh: absolute distance thresholds
        baseline_mode, poststim_mode: 'above' or 'below'
        all_pointwise_dists: optional list of list of 1D arrays [experiment][index][trial]
        pointwise_thresh: float threshold for per-trial filter
        pointwise_mode: 'above' or 'below'

    Returns:
        selected_trial_pairs: list of list of lists [experiment][index] = list of (i, j) pairs
    """
    selected_trial_pairs = []

    for exp_idx, (expt_baseline, expt_poststim) in enumerate(zip(all_baseline_dists, all_poststim_dists)):
        expt_selected = []

        for idx, (base_mat, post_mat) in enumerate(zip(expt_baseline, expt_poststim)):
            if base_mat.ndim != 2 or np.isnan(base_mat).all():
                expt_selected.append([])
                continue

            n_trials = base_mat.shape[0]
            triu_i, triu_j = np.triu_indices(n_trials, k=1)
            base_vals = base_mat[triu_i, triu_j]
            post_vals = post_mat[triu_i, triu_j]

            # Absolute threshold masks
            base_mask = base_vals > baseline_thresh if baseline_mode == 'above' else base_vals < baseline_thresh
            post_mask = post_vals > poststim_thresh if poststim_mode == 'above' else post_vals < poststim_thresh

            combined_mask = base_mask & post_mask

            # Optionally apply pointwise distance filter
            if all_pointwise_dists is not None:
                try:
                    trial_dists = all_pointwise_dists[exp_idx][idx]
                except (IndexError, TypeError):
                    trial_dists = np.full(n_trials, np.nan)

                if pointwise_mode == 'above':
                    valid_trials = trial_dists > pointwise_thresh
                elif pointwise_mode == 'below':
                    valid_trials = trial_dists < pointwise_thresh
                else:
                    raise ValueError("pointwise_mode must be 'above' or 'below'")

                # Both trials in the pair must pass the pointwise filter
                pointwise_mask = np.array([
                    valid_trials[i] and valid_trials[j]
                    for i, j in zip(triu_i, triu_j)
                ])
                combined_mask = combined_mask & pointwise_mask

            selected_pairs = list(zip(triu_i[combined_mask], triu_j[combined_mask]))
            expt_selected.append(selected_pairs)

        selected_trial_pairs.append(expt_selected)

    return selected_trial_pairs


# %%
def find_similar_different_triplets(trial_pairs_similar, trial_pairs_different):
    """
    For each experiment and index, find trial triplets (i, j, k) where:
    - (i, j) are similar
    - (i, k) and (j, k) are different

    Args:
        trial_pairs_similar: list of list of (i, j) pairs
        trial_pairs_different: same structure

    Returns:
        triplet_sets: list of list of lists [experiment][index] = list of (i, j, k) triplets
    """
    triplet_sets = []

    for exp_sim_pairs, exp_diff_pairs in zip(trial_pairs_similar, trial_pairs_different):
        exp_triplets = []

        for sim_pairs, diff_pairs in zip(exp_sim_pairs, exp_diff_pairs):
            sim_set = set(tuple(sorted(p)) for p in sim_pairs)
            diff_set = set(tuple(sorted(p)) for p in diff_pairs)

            triplets = []

            for i, j in sim_set:
                # Look for k such that (i, k) and (j, k) are both in diff_set
                candidate_ks = set()
                for a, b in diff_set:
                    if a == i or b == i:
                        k = b if a == i else a
                        candidate_ks.add(k)
                    if a == j or b == j:
                        k = b if a == j else a
                        candidate_ks.add(k)

                for k in candidate_ks:
                    # Ensure both (i, k) and (j, k) are in diff_set
                    if tuple(sorted((i, k))) in diff_set and tuple(sorted((j, k))) in diff_set:
                        triplets.append((i, j, k))

            exp_triplets.append(triplets)
        triplet_sets.append(exp_triplets)

    return triplet_sets

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import matplotlib.cm as cm

def plot_all_experiments_one_plot(all_baseline_dists_upper, all_poststim_dists_upper):
    """
    Plots all experiments' baseline vs poststim distances on one scatter plot
    with different colors for each dataset, and shows Pearson r values in the legend.
    """
    n_experiments = len(all_baseline_dists_upper)
    colors = cm.get_cmap("tab10", n_experiments)  # distinct colors
    
    plt.figure(figsize=(7, 6))
    
    legend_entries = []
    for i, (baseline, poststim) in enumerate(zip(all_baseline_dists_upper, all_poststim_dists_upper)):
        # Remove NaNs
        mask = ~np.isnan(baseline) & ~np.isnan(poststim)
        baseline_clean = np.array(baseline)[mask]
        poststim_clean = np.array(poststim)[mask]

        # Pearson correlation
        if len(baseline_clean) > 1:
            r_val, p_val = pearsonr(baseline_clean, poststim_clean)
            label = f"Exp {i+1}: R={r_val:.2f}, p={p_val:.3g}"
        else:
            label = f"Exp {i+1}: R=nan, p=nan"

        # Scatter
        plt.scatter(baseline_clean, poststim_clean, color=colors(i), alpha=0.7, label=label)

    # Formatting
    plt.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Baseline Distances")
    plt.ylabel("Poststim Distances")
    plt.title("Baseline vs Poststim Distances (All Experiments)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()
    
# %%
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import numpy as np
# from scipy.stats import pearsonr, linregress

# def plot_baseline_poststim_with_significance(
#     all_baseline_dists_upper,
#     all_poststim_dists_upper,
#     single_color=True,
#     sig_color='crimson',
#     nonsig_color='lightgray',
#     fit_line=True,
#     p_thresh=0.05,
#     equal_xy=False,
#     xlim=None,
#     ylim=None,
#     save_path=None
# ):
#     """
#     Plots baseline vs poststim distances for all datasets in a single plot.
#     Optionally colors significant datasets differently and adds fit lines.

#     Args:
#         all_baseline_dists_upper (list of arrays): Baseline distances per experiment
#         all_poststim_dists_upper (list of arrays): Poststim distances per experiment
#         single_color (bool): If True, all points start as nonsig_color and only significant
#                              datasets are recolored.
#         sig_color (str): Color for significant datasets
#         nonsig_color (str): Color for nonsignificant datasets
#         fit_line (bool): If True, add overall fit line
#         p_thresh (float): p-value threshold for significance
#         equal_xy (bool): If True, force equal x/y axis limits
#         xlim, ylim (tuple): Manual axis limits
#         save_path (str): If provided, saves the plot (SVG with editable text)
#     """
#     # Illustrator-compatible text
#     mpl.rcParams['pdf.fonttype'] = 42
#     mpl.rcParams['ps.fonttype'] = 42
#     mpl.rcParams['svg.fonttype'] = 'none'

#     all_x, all_y, all_significant = [], [], []
#     significant_fit_lines = []

#     # Loop over each experiment
#     for baseline, poststim in zip(all_baseline_dists_upper, all_poststim_dists_upper):
#         x = np.array(baseline)
#         y = np.array(poststim)

#         # Remove NaNs
#         mask = ~np.isnan(x) & ~np.isnan(y)
#         x = x[mask]
#         y = y[mask]

#         if len(x) != len(y) or len(x) == 0:
#             continue

#         # Pearson correlation
#         r, p = pearsonr(x, y)
#         is_significant = p < p_thresh

#         all_x.extend(x)
#         all_y.extend(y)
#         all_significant.extend([is_significant] * len(x))

#         # Store fit line for significant datasets
#         if is_significant:
#             slope, intercept, *_ = linregress(x, y)
#             x_fit = np.linspace(np.min(x), np.max(x), 100)
#             y_fit = slope * x_fit + intercept
#             significant_fit_lines.append((x_fit, y_fit))

#     all_x = np.array(all_x)
#     all_y = np.array(all_y)
#     all_significant = np.array(all_significant)

#     fig, ax = plt.subplots(figsize=(5, 5))

#     # Scatter points
#     if single_color:
#         ax.scatter(all_x[~all_significant], all_y[~all_significant],
#                    color=nonsig_color, edgecolor='white', label=f'p ≥ {p_thresh}', s=40)
#         ax.scatter(all_x[all_significant], all_y[all_significant],
#                    color=sig_color, edgecolor='white', label=f'p < {p_thresh}', s=40)
#     else:
#         # Could assign a different color to each dataset here if desired
#         pass

#     # Plot individual fit lines for significant datasets
#     for x_fit, y_fit in significant_fit_lines:
#         ax.plot(x_fit, y_fit, color='steelblue', linestyle='-', alpha=0.9,
#                 label='_nolegend_', linewidth=2)

#     # Overall regression line
#     if fit_line and len(all_x) > 1:
#         slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
#         x_line = np.linspace(np.min(all_x), np.max(all_x), 100)
#         y_line = slope * x_line + intercept
#         ax.plot(x_line, y_line, color='black', linestyle='--', label='Overall Fit')

#         stats_text = f"r = {r_val:.2f}, p = {p_val:.3f}"
#         ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
#                 fontsize=10, verticalalignment='top', ha='left')

#     # Labels and formatting
#     ax.set_xlabel('Baseline Distance')
#     ax.set_ylabel('Poststim Distance')
#     ax.set_title('Baseline vs Poststim with Significant Fits')
#     ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)

#     # Axis limits
#     if equal_xy:
#         min_val = min(np.min(all_x), np.min(all_y))
#         max_val = max(np.max(all_x), np.max(all_y))
#         ax.set_xlim(min_val, max_val)
#         ax.set_ylim(min_val, max_val)
#     else:
#         if xlim is not None:
#             ax.set_xlim(xlim)
#         if ylim is not None:
#             ax.set_ylim(ylim)

#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.legend(fontsize=8, frameon=False)

#     # Save if requested
#     if save_path:
#         plt.savefig(save_path, format='svg', bbox_inches='tight')

#     plt.show()
# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress

def plot_baseline_poststim_with_significance(
    all_baseline_dists_upper,
    all_poststim_dists_upper,
    single_color=True,
    sig_color='crimson',
    nonsig_color='lightgray',
    fit_line=True,
    p_thresh=0.05,
    equal_xy=False,
    xlim=None,
    ylim=None,
    save_path=None
):
    """
    Plots baseline vs poststim distances for all datasets in a single plot.
    Optionally colors significant datasets differently and adds fit lines.

    Returns:
        results_df (pd.DataFrame): DataFrame with r and p values per dataset
    """
    # Illustrator-compatible text
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'

    all_x, all_y, all_significant = [], [], []
    significant_fit_lines = []
    results = []

    # Loop over each experiment
    for i, (baseline, poststim) in enumerate(zip(all_baseline_dists_upper, all_poststim_dists_upper)):
        x = np.array(baseline)
        y = np.array(poststim)

        # Remove NaNs
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        if len(x) != len(y) or len(x) == 0:
            continue

        # Pearson correlation
        r, p = pearsonr(x, y)
        is_significant = p < p_thresh

        results.append({"dataset": i, "r": r, "p": p, "significant": is_significant})

        all_x.extend(x)
        all_y.extend(y)
        all_significant.extend([is_significant] * len(x))

        # Store fit line for significant datasets
        if is_significant:
            slope, intercept, *_ = linregress(x, y)
            x_fit = np.linspace(np.min(x), np.max(x), 100)
            y_fit = slope * x_fit + intercept
            significant_fit_lines.append((x_fit, y_fit))

    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_significant = np.array(all_significant)

    results_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(5, 5))

    # Scatter points
    if single_color:
        ax.scatter(all_x[~all_significant], all_y[~all_significant],
                   color=nonsig_color, edgecolor='white', label=f'p ≥ {p_thresh}', s=40)
        ax.scatter(all_x[all_significant], all_y[all_significant],
                   color=sig_color, edgecolor='white', label=f'p < {p_thresh}', s=40)
    else:
        # Could assign a different color to each dataset here if desired
        pass

    # Plot individual fit lines for significant datasets
    for x_fit, y_fit in significant_fit_lines:
        ax.plot(x_fit, y_fit, color='steelblue', linestyle='-', alpha=0.9,
                label='_nolegend_', linewidth=2)

    # Overall regression line
    if fit_line and len(all_x) > 1:
        slope, intercept, r_val, p_val, _ = linregress(all_x, all_y)
        x_line = np.linspace(np.min(all_x), np.max(all_x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color='black', linestyle='--', label='Overall Fit')

        stats_text = f"r = {r_val:.2f}, p = {p_val:.3f}"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', ha='left')

    # Labels and formatting
    ax.set_xlabel('Baseline Distance')
    ax.set_ylabel('Poststim Distance')
    ax.set_title('Baseline vs Poststim with Significant Fits')
    ax.axline((0, 0), slope=1, color='gray', linestyle='--', linewidth=1)

    # Axis limits
    if equal_xy:
        min_val = min(np.min(all_x), np.min(all_y))
        max_val = max(np.max(all_x), np.max(all_y))
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
    else:
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    # Save if requested
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

    plt.show()

    return results_df

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_cum_var_explained(
    trial_pca_results,
    normalize_rows=False,
    figsize=(8,5),
    pc_colors=None,
    xlim=None,
    ylim=None,
    yticks=None
):
    """
    Plots cumulative variance explained per PC across experiments.

    Parameters
    ----------
    trial_pca_results : dict
        Output of batch_trial_pca_faster containing 'cum_var_expl_per_expt'.
    normalize_rows : bool, optional
        If True, each row is normalized so that the total variance = 1.
    figsize : tuple, optional
        Figure size.
    pc_colors : list or None
        Optional color list for each PC. If None, uses default matplotlib colors.
    xlim : tuple or None
        Limits for x-axis.
    ylim : tuple or None
        Limits for y-axis.
    yticks : list or None
        Custom tick locations for y-axis.
    """
    cum_var = np.array(trial_pca_results['cum_var_expl_per_expt'])

    # --- NEW: drop any row that contains NaN anywhere ---
    cum_var = cum_var[~np.isnan(cum_var).any(axis=1)]

    if normalize_rows and cum_var.size > 0:
        cum_var = cum_var / cum_var[:, -1][:, np.newaxis]  # normalize each row so last PC = 1

    if cum_var.size == 0:
        raise ValueError("No valid rows left after excluding NaN-containing rows.")

    num_expts, num_pcs = cum_var.shape

    # Average and SEM across experiments
    mean_cum_var = np.mean(cum_var, axis=0)
    sem_cum_var  = np.std(cum_var, axis=0, ddof=1) / np.sqrt(num_expts)

    # --- NEW: print means and SEMs ---
    print("\n=== Cumulative Variance Explained ===")
    for i, (mean, sem) in enumerate(zip(mean_cum_var, sem_cum_var), start=1):
        print(f"PC{i}: mean = {mean:.4f}, SEM = {sem:.4f}")

    # Colors for bars
    if pc_colors is None:
        pc_colors = plt.cm.viridis(np.linspace(0,1,num_pcs))

    # Plot cumulative variance as bars with error bars
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(np.arange(1, num_pcs+1), mean_cum_var, yerr=sem_cum_var,
           color=pc_colors, edgecolor='k', capsize=5)
    
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Variance Explained" + (" (fraction)" if normalize_rows else ""))
    ax.set_title("Cumulative Variance Explained Across Experiments")
    ax.set_xticks(np.arange(1, num_pcs+1))

    # Apply optional limits and ticks
    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(0.5, num_pcs+0.5)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, 1.05 if normalize_rows else 1.2*mean_cum_var.max())

    if yticks is not None:
        ax.set_yticks(yticks)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
