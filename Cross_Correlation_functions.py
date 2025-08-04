# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 10:49:48 2024

@author: jec822
"""

import sys
sys.path.append('./lib')

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import os

import npy_reader

import pandas as pd
import matplotlib.pyplot as plt
import pylab
import seaborn as sns

from scipy import stats
import copy

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d

from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D

# import Neuropil_subtract
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
from scipy.optimize import curve_fit

from scipy.signal import find_peaks
import statsmodels.api as sm 
import scipy as scipy
from scipy.signal import medfilt

# %

# %%
from scipy.stats import pearsonr
from scipy.signal import correlate2d


def calculate_cross_correlation(matrix1, matrix2):
    cross_corr_results = []
    for _, row1 in matrix1.iterrows():
        cross_corr_row = []
        for _, row2 in matrix2.iterrows():
            # Calculate the Pearson correlation coefficient
            corr_coefficient, _ = pearsonr(row1, row2)
            cross_corr_row.append(corr_coefficient)
        cross_corr_results.append(cross_corr_row)
    return pd.DataFrame(cross_corr_results, index=matrix1.index, columns=matrix2.index)



# %%

def scramble_dataframe(dataframe, n_times=100):
    # Initialize an empty DataFrame to accumulate shuffled values
    shuffled_df = pd.DataFrame(index=range(dataframe.shape[0]), columns=range(dataframe.shape[1]))
    
    for _ in range(n_times):
        # Shuffle the values of the original DataFrame
        shuffled_values = np.apply_along_axis(np.random.permutation, 1, dataframe.values)
        # Add shuffled values to the accumulator DataFrame
        shuffled_df = shuffled_df.add(pd.DataFrame(shuffled_values), fill_value=0)

    # Compute the average of each element across all scrambles
    average_df = shuffled_df / n_times
    return average_df


def scramble_dataframe_rows(dataframe, n_times=100):
    # Initialize an empty DataFrame to accumulate shuffled values
    shuffled_df = pd.DataFrame(data=np.zeros_like(dataframe), columns=dataframe.columns)

    for _ in range(n_times):
        # Shuffle the rows of the original DataFrame
        shuffled_values = np.apply_along_axis(np.random.permutation, 0, dataframe.values)
        # Add shuffled values to the accumulator DataFrame
        shuffled_df = shuffled_df.add(pd.DataFrame(shuffled_values), fill_value=0)

    # Compute the average of each element across all scrambles
    average_df = shuffled_df / n_times
    return average_df

# %%

def set_upper_diagonal_to_nan(dataframe):
    # Create a mask to identify upper triangle
    mask = np.triu(np.ones_like(dataframe, dtype=bool))

    # Set upper triangle values to NaN
    dataframe[mask] = np.nan
    
def set_diagonal_to_nan(dataframe):
    np.fill_diagonal(dataframe.values, np.nan)
    

# %%
# import numpy as np
# import pandas as pd
# from scipy.signal import correlate, correlation_lags

# def normalize_cross_correlation(signal1, signal2):
#     """
#     Compute the normalized cross-correlation of two signals.
    
#     Parameters:
#     signal1, signal2: 1D numpy arrays
#         The signals to cross-correlate.

#     Returns:
#     cross_corr: 1D numpy array
#         The normalized cross-correlation of the two signals.
#     """
#     cross_corr = correlate(signal1, signal2, mode='full', method='auto')
#     norm_factor = np.sqrt(np.dot(signal1, signal1) * np.dot(signal2, signal2))
#     return cross_corr / norm_factor


# %%

def cross_correlation_from_list(array_list, indices=None):
    """
    Calculate normalized cross-correlation for each combination of arrays in a list,
    calculate the center of mass for each cross-correlation result, and save these 
    results in DataFrames.

    Parameters:
    array_list: list of 2D numpy arrays or pandas DataFrames
        Each array contains rows of activity data for neurons.

    Returns:
    tuple:
        - cross_corr_df: DataFrame
            A DataFrame where each cell contains another DataFrame representing the cross-correlation 
            results for the corresponding pair of input arrays.
        - center_of_mass_df: DataFrame
            A DataFrame where each cell contains the center of mass for each row of the cross-correlation 
            results for the corresponding pair of input arrays.
        - abs_diff_lags_df: DataFrame
            A DataFrame where each cell contains the absolute difference between the center of mass lags 
            and the equivalent of 0 lags for the corresponding pair of input arrays.
        - median_abs_diff_lags_df: DataFrame
            A DataFrame where each cell contains the median of the absolute differences in lags for the 
            corresponding pair of input arrays.
    """
    num_arrays = len(array_list)
    cross_corr_df = pd.DataFrame(index=range(num_arrays), columns=range(num_arrays))
    center_of_mass_df = pd.DataFrame(index=range(num_arrays), columns=range(num_arrays))
    abs_diff_lags_df = pd.DataFrame(index=range(num_arrays), columns=range(num_arrays))
    median_abs_diff_lags_df = pd.DataFrame(index=range(num_arrays), columns=range(num_arrays))

    for i in range(num_arrays):
        for j in range(i, num_arrays):
            # Convert arrays to numpy if they are pandas DataFrames
            array1 = np.array(array_list[i])
            array2 = np.array(array_list[j])
            
           
            array1 = array1[:, 301:600]
            array2 = array2[:, 301:600]
            
            # Get the number of rows (neurons) in each array
            num_neurons = array1.shape[0]
            
            # Initialize lists to store results for this pair of arrays
            pairwise_corrs = []
            center_of_masses = []
            abs_diff_lags = []

            # Iterate over each row (neuron)
            for k in range(num_neurons):
                # Calculate normalized cross-correlation for the kth row
                cross_corr = normalize_cross_correlation(array1[k], array2[k])
                pairwise_corrs.append(cross_corr)
                
                # Calculate the center of mass for the cross-correlation result
                center_of_mass = center_of_mass_1d_min_corrected(cross_corr)
                center_of_masses.append(center_of_mass)
                
                # Calculate the absolute difference between the center of mass lag and 0 lag
                zero_lag_index = len(cross_corr) // 2
                abs_diff_lag = np.abs(center_of_mass - zero_lag_index)
                abs_diff_lags.append(abs_diff_lag)
            
            # Store the cross-correlation results as a DataFrame
            cross_corr_df.iloc[i, j] = pd.DataFrame(pairwise_corrs)
            cross_corr_df.iloc[j, i] = cross_corr_df.iloc[i, j]  # Symmetric results
            
            # Store the center of mass results as a DataFrame
            center_of_mass_df.iloc[i, j] = pd.DataFrame(center_of_masses)
            center_of_mass_df.iloc[j, i] = center_of_mass_df.iloc[i, j]  # Symmetric results
            
            # Store the absolute difference between lags as a DataFrame
            abs_diff_lags_df.iloc[i, j] = pd.DataFrame(abs_diff_lags)
            abs_diff_lags_df.iloc[j, i] = abs_diff_lags_df.iloc[i, j]  # Symmetric results
            
            # Store the median of the absolute differences as a scalar
            median_abs_diff_lags_df.iloc[i, j] = np.median(abs_diff_lags)
            median_abs_diff_lags_df.iloc[j, i] = median_abs_diff_lags_df.iloc[i, j]  # Symmetric results

    return cross_corr_df, center_of_mass_df, abs_diff_lags_df, median_abs_diff_lags_df

# %%
# def normalize_cross_correlation(array1, array2):
#     """
#     Calculate the normalized cross-correlation of two arrays.
    
#     Parameters:
#     array1, array2: 1D numpy arrays
#         Arrays containing the activity for a neuron.
    
#     Returns:
#     1D numpy array
#         The normalized cross-correlation of the two input arrays.
#     """
#     return np.correlate(array1, array2, mode='full') / (np.std(array1) * np.std(array2) * len(array1))

def normalize_cross_correlation(signal1, signal2, eps=1e-10):
    """
    Compute the normalized cross-correlation of two signals.

    Parameters:
    signal1, signal2: 1D numpy arrays
        The signals to cross-correlate.

    Returns:
    cross_corr: 1D numpy array
        The normalized cross-correlation of the two signals.
    """
    cross_corr = correlate(signal1, signal2, mode='full', method='auto')
    norm1 = np.dot(signal1, signal1)
    norm2 = np.dot(signal2, signal2)
    norm_factor = np.sqrt(norm1 * norm2) + eps  # prevent div by 0
    return cross_corr / norm_factor



def correlation_lags(len1, len2, mode='full'):
    """
    Compute the lags for cross-correlation based on array lengths.
    
    Parameters:
    len1, len2: int
        Lengths of the two input arrays.
    mode: str
        Mode for correlation, can be 'full', 'valid', or 'same'.
    
    Returns:
    1D numpy array
        An array of lags corresponding to the cross-correlation results.
    """
    if mode == 'full':
        return np.arange(-len2 + 1, len1)
    elif mode == 'valid':
        return np.arange(0, len1 - len2 + 1)
    elif mode == 'same':
        return np.arange(-len2 // 2, len1 - len2 // 2)

# Example usage:
# array_list = [array1, array2, array3]
# result = cross_correlation_from_list(array_list)

# %%

def sort_cross_correlation_by_peak(cross_corr_df):
    """
    Sort the cross-correlation DataFrame by the peak value of each row.
    
    Parameters:
    cross_corr_df: DataFrame
        A DataFrame where each row contains cross-correlation values.

    Returns:
    DataFrame
        The input DataFrame sorted by the peak cross-correlation value of each row.
    """
    # Calculate the peak value for each row
    peak_values = cross_corr_df.max(axis=1)
    
    # Add the peak values as a new column
    cross_corr_df['peak'] = peak_values
    
    # Sort the DataFrame by the peak values
    sorted_cross_corr_df = cross_corr_df.sort_values(by='peak', ascending=False)
    
    # Drop the peak column before returning
    sorted_cross_corr_df = sorted_cross_corr_df.drop(columns=['peak'])
    
    return sorted_cross_corr_df


# %%
def cross_correlation_with_mean_minus_trial(trial_matrix_result, row_index):
    """
    Calculate normalized cross-correlation of each trial with the mean of the specified row.
    
    Parameters:
    mean_matrix: 2D numpy array or pandas DataFrame
        Matrix where each row contains the mean activity of many trials.
    trial_matrix: 2D numpy array or pandas DataFrame
        Matrix where each row contains the activity of a single trial for the same cell.
    row_index: int
        The index of the row in mean_matrix to use as the mean signal.
    
    Returns:
    DataFrame
        A DataFrame where each row contains the normalized cross-correlation values 
        for each trial in trial_matrix with the mean signal from mean_matrix.
    """
    # Ensure inputs are numpy arrays
    # mean_matrix = np.array(mean_matrix)
    
    
    trial_matrix = np.array(trial_matrix_result.get('Stim_cell_z_score_trials').iloc[row_index][0])
    
    # Extract the mean signal for the specified row
    # mean_signal = mean_matrix[row_index]
    
    # Get the number of trials in trial_matrix
    num_trials = trial_matrix.shape[0]
    
    # Initialize a list to store cross-correlation results for each trial
    cross_corr_list = []
    
    # Iterate over each trial
    for i in range(num_trials):
        
        # mean_signal = trial_matrix[i]
        matrix_excluding_row = np.delete(trial_matrix, i, axis=0)
    
        # Calculate the mean of the remaining rows
        mean_signal = np.mean(matrix_excluding_row, axis=0)
        # Calculate normalized cross-correlation for the ith trial with the mean signal
        cross_corr = normalize_cross_correlation(mean_signal, trial_matrix[i])
        cross_corr_list.append(cross_corr)
    
    # Determine the length of the cross-correlation results
    cross_corr_length = len(cross_corr_list[0])
    
    # Create a DataFrame from the cross-correlation list
    cross_corr_df = pd.DataFrame(cross_corr_list, columns=correlation_lags(mean_signal.shape[0], trial_matrix.shape[1], mode='full'))
    
    return cross_corr_df



# %%
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

def cross_correlation_with_mean_random(trial_dffs_list, start_index, end_index,number_of_trials=1,  combination_number=1000,kernel_size=1):
    """
    Calculate normalized cross-correlation between the mean signals of two groups of trials
    by iterating through a limited number of combinations of 'number_of_trials' trials out of total trials.
    
    Parameters:
    trial_matrix_result: dict
        Dictionary containing 'Stim_cell_z_score_trials' as a 3D numpy array or pandas DataFrame.
    row_index: int
        The index of the row in Stim_cell_z_score_trials to use for trial-by-trial cross-correlation.
    number_of_trials: int
        Number of trials to include in the first group. The second group will have total_trials - number_of_trials.
    start_index: int
        The start index for slicing the trials in the matrix.
    end_index: int
        The end index for slicing the trials in the matrix.
    combination_number: int
        The maximum number of combinations to be evaluated.
    
    Returns:
    tuple of DataFrames
        A DataFrame where each row contains the normalized cross-correlation values 
        for each combination of trial groups.
        A DataFrame where each row contains the non-normalized cross-correlation values.
        A DataFrame where each row contains the pair labels corresponding to the cross-correlation results.
        A DataFrame where each row contains the lag corresponding to the maximum cross-correlation value.
    """
    # Ensure inputs are numpy arrays
    trial_matrix = np.array([trial[start_index:end_index] for trial in trial_dffs_list])
    # Apply median filter to the trace starting at index 100
    trial_matrix = np.array([medfilt(trial, kernel_size=kernel_size) for trial in trial_matrix])
    # trial_matrix = medfilt(trial_matrix, kernel_size=kernel_size)
    
    # Get the number of trials in trial_matrix
    num_trials = trial_matrix.shape[0]
    
    if number_of_trials <= 0 or number_of_trials >= num_trials:
        raise ValueError("Number of trials must be positive and less than the total number of trials.")
    
    # Generate all possible combinations of 'number_of_trials' trials out of total trials
    trial_combinations = list(combinations(range(num_trials), number_of_trials))
    
    # Limit the number of combinations to the specified combination_number
    if combination_number < len(trial_combinations):
        trial_combinations = trial_combinations[:combination_number]
    
    # Initialize lists to store cross-correlation results for each combination of trial groups
    cross_corr_list = []
    non_normalized_cross_corr_list = []
    pair_labels = []
    lags_list = []

    # Iterate over each combination of 'number_of_trials' trials
    for combo in trial_combinations:
        group1_indices = list(combo)
        group2_indices = list(set(range(num_trials)) - set(group1_indices))
        
        # Calculate the mean of the two groups
        mean_signal1 = np.mean(trial_matrix[group1_indices], axis=0)
        mean_signal2 = np.mean(trial_matrix[group2_indices], axis=0)
        
        # Calculate normalized cross-correlation between the two mean signals
        cross_corr = normalize_cross_correlation(mean_signal1, mean_signal2)
        cross_corr_list.append(cross_corr)
        
        # Calculate non-normalized cross-correlation between the two mean signals
        non_normalized_cross_corr = correlate(mean_signal1, mean_signal2, mode='full')
        non_normalized_cross_corr_list.append(non_normalized_cross_corr)
        
        # Calculate the lag corresponding to the maximum cross-correlation value
        lags = correlation_lags(mean_signal1.size, mean_signal2.size, mode='full')
        lag = lags[np.argmax(non_normalized_cross_corr)]
        lags_list.append(lag)
        
        pair_labels.append((group1_indices, group2_indices))

    # Determine the length of the cross-correlation results
    cross_corr_length = len(cross_corr_list[0])
    
    # Create DataFrames from the cross-correlation lists
    cross_corr_df = pd.DataFrame(cross_corr_list, columns=correlation_lags(mean_signal1.size, mean_signal2.size, mode='full'))
    non_normalized_cross_corr_df = pd.DataFrame(non_normalized_cross_corr_list, columns=correlation_lags(mean_signal1.size, mean_signal2.size, mode='full'))
    
    # Create a DataFrame for the pair labels
    pair_labels_df = pd.DataFrame(pair_labels, columns=['Group_1_Indices', 'Group_2_Indices'])
    
    # Create a DataFrame for the lags
    lags_df = pd.DataFrame(lags_list, columns=['Lag'])

    return cross_corr_df, non_normalized_cross_corr_df, pair_labels_df, lags_df
# %%


def cross_correlation_trial_by_trial(trial_dffs_list, start_index, end_index):
    """
    Calculate normalized and non-normalized cross-correlation for each pair of trials in the specified row,
    excluding cross-correlation of the same trial to itself and repeated cross-correlation between pairs of trials.
    
    Parameters:
    trial_matrix_result: dict
        Dictionary containing 'Stim_cell_z_score_trials' as a 3D numpy array or pandas DataFrame.
    row_index: int
        The index of the row in Stim_cell_z_score_trials to use for trial-by-trial cross-correlation.
    
    Returns:
    tuple of DataFrames
        A DataFrame where each row contains the normalized cross-correlation values 
        for each pair of trials in trial_matrix, excluding the same trial to itself.
        A DataFrame where each row contains the non-normalized cross-correlation values.
        A DataFrame where each row contains the pair labels corresponding to the cross-correlation results.
        A DataFrame where each row contains the lags corresponding to the cross-correlation results.
    """
    # Ensure inputs are numpy arrays
    trial_matrix = np.array([trial[start_index:end_index] for trial in trial_dffs_list])
    
    # Get the number of trials in trial_matrix
    num_trials = trial_matrix.shape[0]
    
    # Initialize lists to store cross-correlation results for each pair of trials
    cross_corr_list = []
    non_normalized_cross_corr_list = []
    pair_labels = []
    lags_list = []

    # Iterate over each pair of trials
    for i in range(num_trials):
        for j in range(i + 1, num_trials):
            # Calculate normalized cross-correlation for the ith trial with the jth trial
            normalized_cross_corr = normalize_cross_correlation(trial_matrix[i], trial_matrix[j])
            cross_corr_list.append(normalized_cross_corr)
            
            # Calculate non-normalized cross-correlation for the ith trial with the jth trial
            non_normalized_cross_corr = correlate(trial_matrix[i], trial_matrix[j], mode='full')
            non_normalized_cross_corr_list.append(non_normalized_cross_corr)
            
            # Calculate the lags for the cross-correlation
            lags = correlation_lags(trial_matrix[i].size, trial_matrix[j].size, mode='full')
            lag=lags[np.argmax(non_normalized_cross_corr)]
            lags_list.append(lag)
            
            pair_labels.append((i, j))

    # Determine the length of the cross-correlation results
    cross_corr_length = len(cross_corr_list[0])
    
    # Create DataFrames from the cross-correlation lists
    cross_corr_df = pd.DataFrame(cross_corr_list)
    non_normalized_cross_corr_df = pd.DataFrame(non_normalized_cross_corr_list)
    
    # Create a DataFrame for the pair labels
    pair_labels_df = pd.DataFrame(pair_labels, columns=['Trial_1', 'Trial_2'])
    
    # Create a DataFrame for the lags
    lags_df = pd.DataFrame(lags_list)
    
    return cross_corr_df, non_normalized_cross_corr_df, pair_labels_df, lags_df


# %%
def compute_cross_correlation_summary(
    df,
    trace_column='trials_arrays',
    roi_column='roi_id_extended_dataset',
    stim_column='stim_id_extended_dataset',
    stimulus_idx=(100, 380),
    kernel_size=9,
    min_trials=2,
    number_of_trials=2
):
    """
    Compute mean cross-correlation for each row in a DataFrame.

    Parameters:
        df : pd.DataFrame
            DataFrame where each row contains trials (list of dF/F arrays).
        trace_column : str
            Column name containing the list of trial traces.
        roi_column : str
            Column name for ROI identifier.
        stim_column : str
            Column name for stimulus identifier.
        stimulus_idx : tuple
            Start and end indices for the cross-correlation window.
        kernel_size : int
            Kernel size for smoothing in cross-correlation.
        min_trials : int
            Minimum number of trials required to compute cross-correlation.

    Returns:
        pd.DataFrame
            DataFrame with one row per ROI-stim combination and columns:
            ['roi', 'stim', 'mean_cross_corr', 'lags']
    """
    results = []

    for idx, row in df.iterrows():
        roi = row[roi_column]
        stim = row[stim_column]
        dffs = row[trace_column]

        # Skip if too few trials
        if len(dffs) < min_trials:
            continue

        # Truncate to minimum length
        min_len = min(len(trace) for trace in dffs)
        dffs_trunc = [trace[:min_len] for trace in dffs]

        # Compute cross-correlation
        cross_corr_df, _, _, lags_df = cross_correlation_with_mean_random(
            dffs_trunc,
            start_index=stimulus_idx[0],
            end_index=stimulus_idx[1],
            number_of_trials=number_of_trials,
            kernel_size=kernel_size
        )

        # Remove rows where max correlation is 0 (flat signal)
        cross_corr_df = cross_corr_df[cross_corr_df.max(axis=1) > 0]
        
        peaks=cross_corr_df.max(axis=1)
        peaks_df = pd.DataFrame({'Peaks': list(peaks)})
        
        mean_corr = cross_corr_df.mean()

        results.append({
            'roi': roi,
            'stim': stim,
            'mean_cross_corr': mean_corr,
            'lags': lags_df,
            'peaks':peaks_df
        })

    return pd.DataFrame(results)
# %%
import numpy as np
from copy import deepcopy
from scipy.stats import pearsonr

def crossval_peak_timing_shuffle(
    peak_or_com_series,
    sampling_interval=0.032958316,
    metric='peak',  # or 'com'
    n_iter=1000,
    min_valid_frac=0.5,
    random_seed=None
):
    """
    Performs cross-validated shuffling of peak/com times for each entry in a Series.

    Parameters:
    - peak_or_com_series : pd.Series of 1D arrays or lists of trial-level peak/com times (in frames).
    - sampling_interval : float, frame to seconds conversion.
    - metric : 'peak' or 'com', for label purposes only.
    - n_iter : int, number of shuffle iterations.
    - min_valid_frac : float, minimum fraction of valid (non-nan) trials to include.
    - random_seed : int or None, for reproducibility.

    Returns:
    - dict with timing medians for both halves across all entries, SEM, correlation, and parameters.
    """
    rng = np.random.default_rng(seed=random_seed)

    sem_overall = []
    median_half1 = []
    median_half2 = []

    for trial_times in peak_or_com_series:
        trial_times = np.array(trial_times, dtype=float) 

        valid_mask = ~np.isnan(trial_times)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) / len(trial_times) < min_valid_frac:
            continue

        timing_set1 = np.zeros(n_iter)
        timing_set2 = np.zeros(n_iter)

        for i in range(n_iter):
            shuff_idx = deepcopy(valid_indices)
            rng.shuffle(shuff_idx)
            n_half = len(shuff_idx) // 2
            half1 = shuff_idx[:n_half]
            half2 = shuff_idx[n_half:]

            timing_set1[i] = np.nanmedian(trial_times[half1])
            timing_set2[i] = np.nanmedian(trial_times[half2])

            if i == 0:
                sem = np.nanstd(trial_times[valid_indices]) / np.sqrt(len(valid_indices) - 1)
                sem_overall.append(sem)

        median_half1.append(np.nanmedian(timing_set1))
        median_half2.append(np.nanmedian(timing_set2))

    median_half1 = np.array(median_half1)
    median_half2 = np.array(median_half2)
    sem_overall = np.array(sem_overall)

    # Compute correlation and R² between half1 and half2 medians
    if len(median_half1) > 1:
        r, _ = pearsonr(median_half1, median_half2)
        r2 = r ** 2
    else:
        r, r2 = np.nan, np.nan  # Not enough data for correlation

    return {
        'timing_metric': metric,
        'trial_sem': sem_overall,
        'median_trialset1': median_half1,
        'median_trialset2': median_half2,
        'correlation_r': r,
        'correlation_r2': r2,
        'response_type': 'dff',
        'experiment_type': 'standard',
        'analysis_params': {
            'sampling_interval': sampling_interval,
            'xval_num_iter': n_iter,
            'xval_timing_metric': metric,
            'random_seed': random_seed,
        }
    }


# %%
import pandas as pd

def run_xval_across_iterations(
    series_data,
    n_iter_list=[1, 5, 10, 50],
    iter_total=1000,
    metric='peak',
    base_seed=42,
    verbose=False
):
    """
    Runs crossval_peak_timing_shuffle over a range of n_iter values.

    Parameters:
    - series_data: pd.Series of arrays/lists (e.g., peak times by trial).
    - n_iter_list: list of n_iter values to test.
    - iter_total: number of repeat experiments for each n_iter value.
    - metric: 'peak' or 'com'.
    - base_seed: int base seed for reproducibility.
    - verbose: print progress messages.

    Returns:
    - dict: keys are n_iter values, values are DataFrames with r, r² for each iteration.
    """
    xval_results_by_iter = {}

    for n_iter in n_iter_list:
        if verbose:
            print(f"\nRunning {iter_total} iterations for n_iter = {n_iter}")

        xval_summary = []

        for j in range(iter_total):
            xval_results = crossval_peak_timing_shuffle(
                peak_or_com_series=series_data,
                metric=metric,
                n_iter=n_iter,
                random_seed=base_seed + j  # ensure different seed each outer loop
            )

            xval_summary.append({
                'iteration': j,
                'correlation_r': xval_results['correlation_r'],
                'correlation_r2': xval_results['correlation_r2'],
                'n_iter': n_iter  # for easy merging later
            })

        xval_df = pd.DataFrame(xval_summary)
        xval_results_by_iter[n_iter] = xval_df

    return xval_results_by_iter

