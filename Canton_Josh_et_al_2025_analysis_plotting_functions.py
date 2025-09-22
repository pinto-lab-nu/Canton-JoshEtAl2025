# -*- coding: utf-8 -*-
"""
Created on Sun May 26 17:25:34 2024

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
import analyzeEvoked2P



    
    # %%

def plot_heatmap(data, vmin=None, vmax=None, start_index=6.9, sampling_interval=0.032958316, exclude_window=None):
    """
    Plot a heatmap with the option to set minimum and maximum values for the color scale,
    and add a white bar overlay for a specified exclusion window.

    Parameters:
    data: 2D array-like
        The data to be plotted as a heatmap.
    vmin: float, optional
        The minimum value for the color scale.
    vmax: float, optional
        The maximum value for the color scale.
    start_index: float, optional
        The start time in seconds for plotting the heatmap.
    sampling_interval: float, optional
        The interval between samples in seconds.
    exclude_window: tuple of two floats, optional
        The start and end time of the window to exclude, adding a white bar overlay.
    """
    
    # Convert start_index to the corresponding index in the data array
    start_index = int(start_index / sampling_interval)
    
    # Convert data to a NumPy array and slice it from the start_index onward
    data = data.to_numpy()[:, start_index:]
    
    # Get the number of samples (columns) in the data
    data_samples = data.shape[1]
    
    # Create a time vector corresponding to the data points
    time = np.linspace(sampling_interval, sampling_interval * data_samples, data_samples)
    
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

    # Plot heatmap, using 'time' for the x-axis ticks
    im = plt.imshow(data, cmap='coolwarm', interpolation='nearest', aspect='auto', 
                    vmin=vmin, vmax=vmax, extent=[time[0], time[-1], 0, data.shape[0]])
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('z-Score')
    
    # Add exclusion window as a white bar overlay
    if exclude_window:
        exclude_start, exclude_end = exclude_window
        exclude_start_idx = int(exclude_start / sampling_interval)
        exclude_end_idx = int(exclude_end / sampling_interval)
        
        # Ensure indices are within bounds
        exclude_start_idx = max(exclude_start_idx - start_index, 0)
        exclude_end_idx = min(exclude_end_idx - start_index, data.shape[1])
        
        plt.axvspan(time[exclude_start_idx], time[exclude_end_idx], color='white', alpha=1.0)

    # Set axis labels
    plt.xlabel('Time (s)')
    plt.ylabel('Trials/Cells')  # Update this as per your data structure, e.g., Trials or Cells

    # Ensure the plot fits well within the figure
    plt.tight_layout()  
    plt.show()

    

# %%


def plot_mean_trace_multiple(result1, result2=None, result3=None, result4=None, result5=None, 
                    y_limits=[-8, 8], legend_names=None, normalize=False, exclude_windows=None):
    
    sampling_interval = 0.032958316
    data_samples = len(np.nanmean(result1.get("normalizedDataIndividual"), axis=1))
    time = np.linspace(sampling_interval, sampling_interval * data_samples, data_samples)

    # Generate HSV color scheme for up to 5 results
    colors = plt.cm.jet(np.linspace(0, 1, 5))

    # Default legend names if none provided
    if legend_names is None:
        legend_names = ["Result 1", "Result 2", "Result 3", "Result 4", "Result 5"]

    # Function to normalize a trace and its SEM to the peak value of the trace
    def normalize_trace_and_sem(trace, sem):
        max_val = np.nanmax(trace)
        if max_val != 0:
            return trace / max_val, sem / max_val
        return trace, sem

    # Function to exclude windows of data
    def exclude_trace_windows(trace, sem, time, exclude_windows):
        if exclude_windows:
            for window in exclude_windows:
                start, end = window
                exclude_mask = (time >= start) & (time <= end)
                trace[exclude_mask] = np.nan
                sem[exclude_mask] = np.nan
        return trace, sem

    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)

    # Plot result1
    y1 = result1.get("conditional_zscores_no_nan_mean_across_cells")
    y1_sem = result1.get("average_trace_sem")
    if normalize:
        y1, y1_sem = normalize_trace_and_sem(y1, y1_sem)
    if exclude_windows:
        y1, y1_sem = exclude_trace_windows(y1, y1_sem, time, exclude_windows)
    plt.plot(time, y1, label=legend_names[0], color=colors[0])
    plt.fill_between(time, (y1 + y1_sem).flatten(), (y1 - y1_sem).flatten(), facecolor=colors[0], alpha=0.75)

    # Plot result2 if provided
    if result2:
        y2 = result2.get("conditional_zscores_no_nan_mean_across_cells")
        y2_sem = result2.get("average_trace_sem")
        if normalize:
            y2, y2_sem = normalize_trace_and_sem(y2, y2_sem)
        if exclude_windows:
            y2, y2_sem = exclude_trace_windows(y2, y2_sem, time, exclude_windows)
        plt.plot(time, y2, label=legend_names[1], color=colors[1])
        plt.fill_between(time, (y2 + y2_sem).flatten(), (y2 - y2_sem).flatten(), facecolor=colors[1], alpha=0.75)

    # Plot result3 if provided
    if result3:
        y3 = result3.get("conditional_zscores_no_nan_mean_across_cells")
        y3_sem = result3.get("average_trace_sem")
        if normalize:
            y3, y3_sem = normalize_trace_and_sem(y3, y3_sem)
        if exclude_windows:
            y3, y3_sem = exclude_trace_windows(y3, y3_sem, time, exclude_windows)
        plt.plot(time, y3, label=legend_names[2], color=colors[2])
        plt.fill_between(time, (y3 + y3_sem).flatten(), (y3 - y3_sem).flatten(), facecolor=colors[2], alpha=0.75)

    # Plot result4 if provided
    if result4:
        y4 = result4.get("conditional_zscores_no_nan_mean_across_cells")
        y4_sem = result4.get("average_trace_sem")
        if normalize:
            y4, y4_sem = normalize_trace_and_sem(y4, y4_sem)
        if exclude_windows:
            y4, y4_sem = exclude_trace_windows(y4, y4_sem, time, exclude_windows)
        plt.plot(time, y4, label=legend_names[3], color=colors[3])
        plt.fill_between(time, (y4 + y4_sem).flatten(), (y4 - y4_sem).flatten(), facecolor=colors[3], alpha=0.75)

    # Plot result5 if provided
    if result5:
        y5 = result5.get("conditional_zscores_no_nan_mean_across_cells")
        y5_sem = result5.get("average_trace_sem")
        if normalize:
            y5, y5_sem = normalize_trace_and_sem(y5, y5_sem)
        if exclude_windows:
            y5, y5_sem = exclude_trace_windows(y5, y5_sem, time, exclude_windows)
        plt.plot(time, y5, label=legend_names[4], color=colors[4])
        plt.fill_between(time, (y5 + y5_sem).flatten(), (y5 - y5_sem).flatten(), facecolor=colors[4], alpha=0.75)

    plt.ylim(y_limits[0], y_limits[1])
    plt.legend()
    plt.show()
# %%

def normalize_rows(df, norm_type='minmax', min_window=None, max_window=None, peak_window=None):
    """
    Normalize each row of the DataFrame using the specified normalization type.

    Parameters:
        df (pd.DataFrame): Input data (each row is a trace).
        norm_type (str): 'minmax' or 'peak'.
        min_window (tuple): (start, end) for min (used only if norm_type == 'minmax').
        max_window (tuple): (start, end) for max (used only if norm_type == 'minmax').
        peak_window (tuple): (start, end) for peak (used only if norm_type == 'peak').

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    df_norm = df.copy()
    n_cols = df.shape[1]

    for i in range(df.shape[0]):
        row = df.iloc[i, :]

        if norm_type == 'minmax':
            # Get min from specified window
            if min_window is not None:
                min_start, min_end = min_window
                min_start = max(0, min_start)
                min_end = min(n_cols, min_end)
                min_val = np.nanmean(row.iloc[min_start:min_end])
            else:
                min_val = np.nanmin(row)

            # Get max from specified windowa
            if max_window is not None:
                max_start, max_end = max_window
                max_start = max(0, max_start)
                max_end = min(n_cols, max_end)
                max_val = np.nanmax(row.iloc[max_start:max_end])
            else:
                max_val = np.nanmax(row)

            if max_val > min_val:
                df_norm.iloc[i, :] = (row - min_val) / (max_val - min_val)

        elif norm_type == 'peak':
            if peak_window is not None:
                peak_start, peak_end = peak_window
                peak_start = max(0, peak_start)
                peak_end = min(n_cols, peak_end)
                peak_val = np.nanmax(np.abs(row.iloc[peak_start:peak_end]))
            else:
                peak_val = np.nanmax(np.abs(row))

            if peak_val > 0:
                df_norm.iloc[i, :] = row / peak_val

        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")

    return df_norm

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_mean_trace_multiple_dataframe_input(
    result1, result2=None, result3=None, result4=None, result5=None, 
    y_limits=[-8, 8],x_limits=[-3, 10], legend_names=None, norm_mean=False,
    norm_type_row=None, norm_rows=True, time_array=None, exclude_windows=None,
    line_colors=None, fill_alpha=0.75,
    xticks=None, yticks=None, xlabel=None, ylabel=None
):
    sampling_interval = 0.032958316
    data_samples = result1.shape[1]

    # Time vector logic
    if time_array is not None:
        if isinstance(time_array, pd.DataFrame):
            time = time_array.squeeze().to_numpy()
        else:
            time = np.array(time_array)
    else:
        time = np.linspace(sampling_interval, sampling_interval * data_samples, data_samples)
        
    # Default colors if not provided
    default_colors = plt.cm.jet(np.linspace(0, 1, 5))
    if line_colors is None:
        line_colors = default_colors
    else:
        while len(line_colors) < 5:
            line_colors.append('gray')

    # Default legend names if not provided
    if legend_names is None:
        legend_names = ["Result 1", "Result 2", "Result 3", "Result 4", "Result 5"]

    # Normalize trace and SEM to peak
    def normalize_trace_and_sem(trace, sem):
        max_val = np.nanmax(trace)
        if max_val != 0:
            return trace / max_val, sem / max_val
        return trace, sem

    # Exclude time windows
    def exclude_trace_windows(trace, sem, time, exclude_windows):
        if exclude_windows:
            for start, end in exclude_windows:
                mask = (time >= start) & (time <= end)
                trace[mask] = np.nan
                sem[mask] = np.nan
        return trace, sem

    # Plotting function
    def process_and_plot(df, label, color):
        if norm_rows:
            if norm_type_row == 'minmax':
                df = normalize_rows(df, norm_type_row, min_window=(0, 80), max_window=(100, 300))
            else:
                df = normalize_rows(df, norm_type_row, peak_window=(100, 300))

        y = df.mean(axis=0).to_numpy()
        y_sem = (df.std(axis=0) / np.sqrt(df.shape[0])).to_numpy()

        if norm_mean:
            y, y_sem = normalize_trace_and_sem(y, y_sem)

        if exclude_windows:
            y, y_sem = exclude_trace_windows(y, y_sem, time, exclude_windows)

        plt.plot(time, y, label=label, color=color)
        plt.fill_between(time, (y + y_sem).flatten(), (y - y_sem).flatten(), color=color, alpha=fill_alpha)

    # Begin plotting
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)

    for idx, (result, name) in enumerate(zip(
        [result1, result2, result3, result4, result5],
        legend_names
    )):
        if result is not None:
            process_and_plot(result, name, line_colors[idx])

    plt.ylim(y_limits)
    plt.xlim(x_limits)

    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.legend()
    plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def general_scatter_plot(x_data, y_data, a_data=None, z_data=None, 
                         xlim=(None, None), ylim=(None, None), 
                         vmin=None, vmax=None, size_scale=100, 
                         label_points=False, cmap='Blues',
                         xlabel='X-axis', ylabel='Y-axis', title='Scatter Plot',
                         auto_limits=False, plot_regression_line=False,
                         multiple=False, dot_size=120, alpha=1.0):
    """
    Create a scatter plot with optional color and size scaling, and the option to plot a regression line.
    Returns a DataFrame containing R² values, r values, and p-values for the regression.

    Parameters:
    - x_data: array-like
        Data for the x-axis.
    - y_data: array-like
        Data for the y-axis.
    - a_data: array-like, optional
        Data for color mapping. Default is None.
    - z_data: array-like, optional
        Data for size scaling. Default is None.
    - xlim: tuple, optional
        Limits for the x-axis (min, max). Default is (None, None).
    - ylim: tuple, optional
        Limits for the y-axis (min, max). Default is (None, None).
    - vmin: float, optional
        Minimum value for color mapping. Default is None.
    - vmax: float, optional
        Maximum value for color mapping. Default is None.
    - size_scale: float, optional
        Scaling factor for point sizes. Default is 100.
    - dot_size: float, optional
        Default size for the scatter points. Default is 120.
    - alpha: float, optional
        Transparency value for the points (0.0 to 1.0). Default is 1.0 (fully opaque).
    - label_points: bool, optional
        Whether to label points with their index. Default is False.
    - cmap: str, optional
        Colormap for color mapping. Default is 'Blues'.
    - xlabel: str, optional
        Label for the x-axis. Default is 'X-axis'.
    - ylabel: str, optional
        Label for the y-axis. Default is 'Y-axis'.
    - title: str, optional
        Title of the plot. Default is 'Scatter Plot'.
    - auto_limits: bool, optional
        Whether to automatically set xlim and ylim based on data. Default is False.
    - plot_regression_line: bool, optional
        Whether to plot the regression line. Default is False.
    - multiple: bool, optional
        If True, do not create a new figure and plot on the existing figure. Default is False.

    Returns:
    - DataFrame with R² values, r values, and p-values.
    """

    if not multiple:
        plt.figure(figsize=(8, 8))

    # Set sizes based on z_data if provided, otherwise use default dot_size
    sizes = z_data * size_scale if z_data is not None else dot_size

    # Create the scatter plot with color mapping and alpha
    scatter = plt.scatter(
        x_data, 
        y_data, 
        s=sizes, 
        c=a_data, 
        cmap=cmap, 
        edgecolor='k', 
        vmin=vmin, 
        vmax=vmax,
        alpha=alpha  # Set transparency
    )

    # Add colorbar if a_data is provided
    if a_data is not None:
        cbar = plt.colorbar(scatter)
        cbar.set_label('Color scale')  # Adjust the label based on context

    # Annotate each point with its index if label_points is True
    if label_points:
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            plt.text(x, y, str(i), fontsize=9, ha='right')

    # Initialize lists to store r, R² values, and p-values
    r_values = []
    r_squared_values = []
    p_values = []

    # Perform linear regression and plot regression line if specified
    if plot_regression_line:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
        regression_line = slope * np.array(x_data) + intercept
        plt.plot(x_data, regression_line, color='gray', linestyle='--')
        
        # Add r, R², and p-value text to the plot
        text_x = 0.05 * (plt.xlim()[1] - plt.xlim()[0]) + plt.xlim()[0]
        text_y = 0.95 * (plt.ylim()[1] - plt.ylim()[0]) + plt.ylim()[0]
        plt.text(text_x, text_y, f'r = {r_value:.2g}', fontsize=9, ha='left', va='top')
        plt.text(text_x, text_y - 0.05 * (plt.ylim()[1] - plt.ylim()[0]), f'R² = {r_value**2:.2g}', fontsize=9, ha='left', va='top')
        plt.text(text_x, text_y - 0.10 * (plt.ylim()[1] - plt.ylim()[0]), f'p-value = {p_value:.2g}', fontsize=9, ha='left', va='top')

        # Store the r, R², and p-value in the lists
        r_values.append(r_value)
        r_squared_values.append(r_value**2)
        p_values.append(p_value)

    # Set axis limits
    if auto_limits:
        plt.xlim(auto=True)
        plt.ylim(auto=True)
    else:
        plt.xlim(xlim)
        plt.ylim(ylim)

    # Labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Show plot
    plt.show()

    # Create a DataFrame for r, R² values, and p-values
    result_df = pd.DataFrame({'r': r_values, 'R²': r_squared_values, 'p-value': p_values})
    return result_df
# %% Plot mean with SEM

import numpy as np
import matplotlib.pyplot as plt

def plot_curve_with_sem(data, labels=None, color=None, alpha=0.3):
    """
    Plots a curve with a shaded area representing the standard error of the mean (SEM).
    
    Parameters:
    data : list of pd.DataFrame or np.ndarray
        List of datasets, where each dataset is a DataFrame or 2D array (rows are observations, columns are data points).
    labels : list of str, optional
        List of labels for each dataset. Defaults to None.
    color : list of str, optional
        List of colors for each dataset. Defaults to None.
    alpha : float, optional
        Transparency level for the shaded area. Defaults to 0.3.
    """
    if not isinstance(data, list):
        data = [data]

    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(data))]

    if color is None:
        color = [None] * len(data)
    
    plt.figure()

    for i, dataset in enumerate(data):
        if isinstance(dataset, pd.DataFrame) or isinstance(dataset, np.ndarray):
            mean_values = np.mean(dataset, axis=0)
            sem_values = np.std(dataset, axis=0) / np.sqrt(dataset.shape[0])

            x = np.arange(mean_values.shape[0])

            plt.plot(x, mean_values, label=labels[i], color=color[i])
            plt.fill_between(x, mean_values - sem_values, mean_values + sem_values, color=color[i], alpha=alpha)

    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.title('Curve with SEM Shaded Area')
    plt.legend()
    plt.show()


    

# %%
from sklearn.linear_model import LinearRegression

def plot_dist_responses_sorted(result1, result2, limit=1.96, sort_order=None, labels=None):
    """
    Plot a scatter plot with SEM error bars for stim_responsive data in result1 and result2, with optional labels and regression lines.
    
    Each column of stim_responsive is treated as a separate dataset. Data from result1 and result2 are
    plotted next to each other for comparison. Optionally, the data can be sorted by a given numeric order.

    Parameters:
    - result1: dict
        Contains 'stim_responsive' data (array or DataFrame) for group 1.
    - result2: dict
        Contains 'stim_responsive' data (array or DataFrame) for group 2.
    - limit: float, optional
        Threshold value used for title description (default is 1.96).
    - sort_order: array-like, optional
        A 1D array of numeric values used to sort the columns of stim_responsive data. The length
        of this array must match the number of columns in stim_responsive.
    - labels: list, optional
        A list of labels for each dataset. If provided, should have 2 elements representing the labels for
        result1 and result2 respectively.
    """
    
    # Extract stim_responsive data from both result1 and result2
    stim_responsive1 = result1.get("stim_responsive")
    stim_responsive2 = result2.get("stim_responsive")

    # Ensure both results have the same number of columns for proper comparison
    assert stim_responsive1.shape[1] == stim_responsive2.shape[1], "Both results must have the same number of columns."
    
    num_columns = stim_responsive1.shape[1]
    
    if sort_order is not None:
        # Ensure that sort_order is the same length as the number of columns
        assert len(sort_order) == num_columns, "sort_order must have the same length as the number of columns in stim_responsive."
        
        # Get the sorted indices based on sort_order
        sorted_indices = np.argsort(sort_order)
        
        # Sort stim_responsive1, stim_responsive2, and sort_order using the sorted indices
        stim_responsive1 = stim_responsive1[:, sorted_indices]
        stim_responsive2 = stim_responsive2[:, sorted_indices]
        sort_order = np.array(sort_order)[sorted_indices]

    # Calculate mean and SEM for each column (dataset) in result1 and result2
    means1 = np.nanmean(stim_responsive1, axis=0)
    sems1 = stats.sem(stim_responsive1, axis=0, nan_policy='omit')
    
    means2 = np.nanmean(stim_responsive2, axis=0)
    sems2 = stats.sem(stim_responsive2, axis=0, nan_policy='omit')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Use sort_order as x values
    x = np.array(sort_order).reshape(-1, 1)  # Reshape x for sklearn

    # Scatter plot for result1
    ax.errorbar(x.flatten() - 0.175, means1, yerr=sems1, fmt='o', 
                label=labels[0] if labels is not None else 'Result 1', 
                capsize=5, color='black', markersize=8)

    # Scatter plot for result2
    ax.errorbar(x.flatten() + 0.175, means2, yerr=sems2, fmt='o', 
                label=labels[1] if labels is not None else 'Result 2', 
                capsize=5, color='red', markersize=8)

    # Fit linear regression to the means of result1
    reg1 = LinearRegression().fit(x, means1)
    predicted_means1 = reg1.predict(x)
    ax.plot(x.flatten(), predicted_means1, 'k--', label=f'Regression {labels[0] if labels is not None else "Result 1"}')

    # Fit linear regression to the means of result2
    reg2 = LinearRegression().fit(x, means2)
    predicted_means2 = reg2.predict(x)
    ax.plot(x.flatten(), predicted_means2, 'r--', label=f'Regression {labels[1] if labels is not None else "Result 2"}')

    # Customize plot
    ax.set_xlabel('Sort Order Values')
    ax.set_ylabel('Proportion of Responsive Cells')
    ax.set_title(f'Average Proportion of Cells Responding to Stim (Z-score > {limit})')

    # Set x-ticks to be the sort_order values
    ax.set_xticks(x.flatten())
    ax.set_xticklabels([f'{val:.2f}' for val in sort_order])
    
    ax.legend()
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_box_aspect(1)

    # Add text for regression stats (slope, intercept, and R-squared)
    slope1, intercept1 = reg1.coef_[0], reg1.intercept_
    r_squared1 = reg1.score(x, means1)
    
    slope2, intercept2 = reg2.coef_[0], reg2.intercept_
    r_squared2 = reg2.score(x, means2)
    
    plt.text(0.1, 0.75, f'{labels[0] if labels is not None else "Result 1"}: R² = {r_squared1:.2f}', fontsize=10, transform=ax.transAxes)
    plt.text(0.1, 0.7, f'{labels[1] if labels is not None else "Result 2"}:  R² = {r_squared2:.2f}', fontsize=10, transform=ax.transAxes)
    plt.text(0.3, 0.75, f'{labels[0] if labels is not None else "Result 1"}:  slope = {slope1:.4f}', fontsize=10, transform=ax.transAxes)
    plt.text(0.3, 0.7, f'{labels[1] if labels is not None else "Result 2"}:  slope = {slope2:.4f}', fontsize=10, transform=ax.transAxes)
    
    
    # Add text for median and KS test result
    median1 = np.nanmedian(stim_responsive1)
    median2 = np.nanmedian(stim_responsive2)
    a = stats.ks_2samp(stim_responsive1.flatten(), stim_responsive2.flatten())
    
    # plt.text(0.1, 0.9, f'Median {labels[0] if labels is not None else "Result 1"}: {median1:.2f}', fontsize=10, transform=ax.transAxes)
    # plt.text(0.1, 0.85, f'Median {labels[1] if labels is not None else "Result 2"}: {median2:.2f}', fontsize=10, transform=ax.transAxes)
    # plt.text(0.1, 0.8, f'KS Test p-value: {a[1]:.4f}', fontsize=10, transform=ax.transAxes)
    
    # Show plot
    plt.tight_layout()
    plt.show()

    return a






# %%

def plot_mean_sem(mean_df, sem_df):
    """
    Plot the mean and SEM for each column in the dataframes.
    
    Parameters:
    mean_df (pd.DataFrame): DataFrame containing the mean values.
    sem_df (pd.DataFrame): DataFrame containing the SEM values.
    """
    
    # Ensure the dataframes contain float64 data
    mean_df = mean_df.astype(np.float64)
    sem_df = sem_df.astype(np.float64)
    
    # Number of columns to plot
    num_plots = mean_df.shape[1]
    
    # Determine the number of rows and columns for the subplots
    num_rows = 3
    num_cols = (num_plots + num_rows - 1) // num_rows  # Ceiling division to ensure at least 3 rows
    
    # Create the subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 5 * num_rows), sharey=True)
    
    # Flatten axes array if it is 2D for easier iteration
    axes = axes.flatten()
    
    # Plot each column
    for i, col in enumerate(mean_df.columns):
        ax = axes[i]
        
        # Mean and SEM values for the current column
        mean_values = mean_df[col]
        sem_values = sem_df[col]
        
        # Plot mean values
        ax.plot(mean_values.index, mean_values, label='Mean')
        
        # Fill between mean ± SEM
        ax.fill_between(mean_values.index, mean_values - sem_values, mean_values + sem_values, facecolor="orange",alpha=0.5)
        
        # Set title and labels
        ax.set_title(col)
        ax.set_xlabel('Time')
        ax.set_ylabel('Response')
        ax.legend()
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# %%
def plot_mean_trace_single(
    df,
    distance_from_origin,
    distances,
    dist_btwn_spot,
    y_limits=[-8, 8],
    normalize=False,
    exclude_windows=None,
    x_scale=1,
    y_scale=1,
    color_floor=0.2
):
    """
    Plots the nan mean and SEM of traces in each column of a DataFrame.

    Args:
        df (pd.DataFrame): A 2D DataFrame where each cell contains a 1D array (trace).
        distance_from_origin (list of tuples): List of (x, y) shifts for each column. 
                                               Each tuple shifts the trace on the x and y axes respectively.
        distances (list): Distances from origin for each trace.
        dist_btwn_spot (float): Distance between steps in microns.
        y_limits (list, optional): Y-axis limits. Defaults to [-8, 8].
        normalize (bool, optional): Whether to normalize each trace and its SEM by the maximum value of the average trace in the first column. Defaults to False.
        exclude_windows (list of tuples, optional): List of time windows to exclude [(start, end), ...]. Defaults to None.
        x_scale (float, optional): Scaling factor for the x-axis. Defaults to 1.
        y_scale (float, optional): Scaling factor for the y-axis. Defaults to 1.
        color_floor (float, optional): Minimum intensity for color scaling to prevent dimming. Defaults to 0.2.
    """
    # Remove cells with NaN values
    df = df.dropna(how='any')

    # Ensure all cells are converted to float64 arrays
    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float64))

    sampling_interval = 0.032958316
    
    # Extract the number of samples based on the first trace
    example_trace = df.iloc[0, 0]
    data_samples = len(example_trace) if example_trace is not None else 0
    time = np.linspace(sampling_interval, sampling_interval * data_samples, data_samples)

    # Calculate the average trace in the first column and its peak value
    first_column_traces = np.array(df.iloc[:, 0].tolist())  # Convert to a 2D array
    avg_first_column_trace = np.nanmean(first_column_traces, axis=0)  # Average trace in the first column
    first_column_peak = np.nanmax(avg_first_column_trace)  # Peak value of the average trace in the first column

    # Function to normalize a trace and its SEM by the peak of the average trace in the first column
    def normalize_trace_and_sem(trace, sem):
        peak_value = first_column_peak
        return trace / peak_value, sem / peak_value

    # Function to exclude windows of data
    def exclude_trace_windows(trace, sem, time, exclude_windows):
        if exclude_windows:
            for window in exclude_windows:
                start, end = window
                exclude_mask = (time >= start) & (time <= end)
                trace[exclude_mask] = np.nan
                sem[exclude_mask] = np.nan
        return trace, sem

    # Prepare the plot
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)

    # Define the color map and calculate distances
    colormap = plt.cm.Blues  # Example: single-color colormap
    max_distance = max(np.sqrt((np.array([x for x, y in distance_from_origin]))**2 +
                               (np.array([y for x, y in distance_from_origin]))**2))  # Max scaled distance

    for idx, column in enumerate(df.columns):
        # Calculate the nanmean and SEM for the current column
        traces = np.array(df[column].tolist())  # Convert to a 2D array
        mean_trace = np.nanmean(traces, axis=0)
        sem_trace = np.nanstd(traces, axis=0) / np.sqrt(np.sum(~np.isnan(traces), axis=0))

        if normalize:
            mean_trace, sem_trace = normalize_trace_and_sem(mean_trace, sem_trace)
        if exclude_windows:
            mean_trace, sem_trace = exclude_trace_windows(mean_trace, sem_trace, time, exclude_windows)

        # Calculate actual distance in microns
        shift = distances[idx]
        micron_distance = shift * dist_btwn_spot

        # Apply distance_from_origin shifts
        x_shift, y_shift = distance_from_origin[idx]

        # Calculate color intensity based on scaled distance from origin
        scaled_distance = np.sqrt((x_shift / x_scale) ** 2 + (y_shift / y_scale) ** 2)
        color_intensity = max(color_floor, scaled_distance / max_distance)  # Ensure intensity does not drop below color_floor
        color = colormap(1 - color_intensity)  # Invert so closer distances are darker

        # Plot the mean trace with SEM shading
        ax.plot(time + x_shift, mean_trace + y_shift, color=color)
        ax.fill_between(time + x_shift, mean_trace + sem_trace + y_shift, mean_trace - sem_trace + y_shift, 
                        color=color, alpha=0.3)

        # Annotate the plot with the distance in microns
        ax.text(time[-1] + x_shift + 1, mean_trace[-1] + y_shift - 1, f'{micron_distance:.1f} µm', 
                color=color, fontsize=9, ha='center')

    ax.set_ylim(y_limits)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean Trace')
    plt.show()

    plt.rcParams['pdf.fonttype'] = 42  # To make text editable
    plt.rcParams['ps.fonttype'] = 42  # To make text editable
    plt.rcParams['svg.fonttype'] = 'none'  # To make text editable


# # Example usage
# n_rows = 10
# n_columns = 3
# trace_length = 100

# data = {
#     f'Column_{i+1}': [np.random.randn(trace_length) for _ in range(n_rows)] 
#     for i in range(n_columns)
# }

# df = pd.DataFrame(data)
# distance_from_origin = [(0, 0), (1, 1), (2, 2)]
# dist_btwn_spot = 50  # Example: 50 microns per step

# plot_mean_trace_single(
#     df=df,
#     distance_from_origin=distance_from_origin,
#     dist_btwn_spot=dist_btwn_spot,
#     y_limits=[-5, 5],
#     normalize=False,
#     x_scale=1,
#     y_scale=1
# )
#
# %%
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.interpolate import interp1d

def plot_peak_bar_single(df, distances, dist_btwn_spot, normalize=False, exclude_windows=None, analysis_range=None, bin_by_distance=False):
    """
    Calculates peak values, decay rates, and width at half-maximum (FWHM) for each trace in a DataFrame 
    and returns them as independent 2D DataFrames. Optionally bins the output DataFrames by distances.

    Args:
        df (pd.DataFrame): A 2D DataFrame where each cell contains a 1D array (trace).
        distances (list): List of distances representing each group.
        dist_btwn_spot (float): Distance between steps in microns.
        normalize (bool, optional): Whether to normalize peaks by the maximum peak value. Defaults to False.
        exclude_windows (list of tuples, optional): List of time windows to exclude [(start, end), ...]. Defaults to None.
        analysis_range (tuple, optional): Tuple (start_index, end_index) specifying the range of the array to analyze. 
                                          Defaults to None (analyze the entire array).
        bin_by_distance (bool, optional): If True, bins the results by distances, with padding for unequal group sizes. Defaults to False.

    Returns:
        peaks_df (pd.DataFrame): DataFrame with peak values for each trace.
        decay_rates_df (pd.DataFrame): DataFrame with decay rates for each trace.
        fwhms_df (pd.DataFrame): DataFrame with FWHMs for each trace.
    """
    # Ensure all cells are converted to float64 arrays
    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float64))

    # Function to exclude windows of data
    def exclude_trace_windows(trace, time, exclude_windows):
        if exclude_windows:
            for window in exclude_windows:
                start, end = window
                exclude_mask = (time >= start) & (time <= end)
                trace[exclude_mask] = np.nan
        return trace

    # Function to calculate the peak value
    def calculate_peak(array):
        return np.nanmax(array)

    # Function to calculate the decay rate (exponential decay)
    def calculate_decay_rate(trace, time, peak_idx):
        try:
            decay_region = trace[peak_idx:]  # Trace after the peak
            if len(decay_region) > 1:
                decay_time = time[peak_idx:]
                log_decay = np.log(decay_region / decay_region[0])  # Logarithm of normalized decay
                slope, _ = np.polyfit(decay_time, log_decay, 1)  # Linear fit
                return -slope  # Decay rate
        except Exception:
            return np.nan
        return np.nan

    # Function to calculate width at half-maximum (FWHM)
    def calculate_fwhm(trace, time):
        try:
            if np.all(np.isnan(trace)):
                return np.nan
            
            peak_idx = np.nanargmax(trace)
            peak_val = trace[peak_idx]
            half_max = peak_val / 2.0
    
            # Search left of the peak for the first crossing
            left_cross = np.nan
            for i in range(peak_idx - 1, 0, -1):
                if trace[i] < half_max and trace[i + 1] >= half_max:
                    # Linear interpolation
                    t1, t2 = time[i], time[i + 1]
                    y1, y2 = trace[i], trace[i + 1]
                    left_cross = t1 + (half_max - y1) * (t2 - t1) / (y2 - y1)
                    break
    
            # Search right of the peak for the second crossing
            right_cross = np.nan
            for i in range(peak_idx, len(trace) - 1):
                if trace[i] >= half_max and trace[i + 1] < half_max:
                    # Linear interpolation
                    t1, t2 = time[i], time[i + 1]
                    y1, y2 = trace[i], trace[i + 1]
                    right_cross = t1 + (half_max - y1) * (t2 - t1) / (y2 - y1)
                    break
    
            if np.isnan(left_cross) or np.isnan(right_cross):
                return np.nan
    
            return right_cross - left_cross
        except Exception:
            return np.nan


    # Create empty DataFrames to store results
    peaks_df = pd.DataFrame(index=df.index, columns=df.columns)
    decay_rates_df = pd.DataFrame(index=df.index, columns=df.columns)
    fwhms_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Analyze each trace
    for col in df.columns:
        for idx in df.index:
            trace = df.at[idx, col]

            # Apply median filter BEFORE any processing
            trace = medfilt(trace, kernel_size=31)

            time = np.linspace(0, len(trace), len(trace))  # Generate time array

            # Apply analysis range
            if analysis_range:
                start, end = analysis_range
                trace = trace[start:end]
                time = time[start:end]

            if exclude_windows:
                trace = exclude_trace_windows(trace, time, exclude_windows)

            # Calculate metrics
            peak_value = calculate_peak(trace)
            peak_idx = np.nanargmax(trace)
            decay_rate = calculate_decay_rate(trace, time, peak_idx)
            fwhm = calculate_fwhm(trace, time)

            # Store metrics in corresponding DataFrames
            peaks_df.at[idx, col] = peak_value
            decay_rates_df.at[idx, col] = decay_rate
            fwhms_df.at[idx, col] = fwhm

    # Normalize peaks if required
    if normalize:
        max_peak = peaks_df.max().max()
        peaks_df = peaks_df / max_peak

    return peaks_df, decay_rates_df, fwhms_df


# %%

    
# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def bar_plot_with_scatter(
    df,
    figsize=(10, 6),
    bar_color='skyblue',
    scatter_color='black',
    alpha=0.7,
    x_positions=None,
    x_scale=1.0,
    x_label=None,
    y_label=None,
    normalize_by_column_index=None,
    distances=None  # <- new input
):
    """
    Creates a bar plot with individual scatter points overlaid for a 2D DataFrame.

    Parameters:
    - df (pd.DataFrame): Input 2D DataFrame.
    - figsize (tuple): Size of the figure.
    - bar_color (str): Color for the bars.
    - scatter_color (str): Color for the scatter points' edges.
    - alpha (float): Transparency for scatter points.
    - x_positions (array-like): Custom x positions for the bars and scatter points.
    - x_scale (float): Scaling factor for the x positions and bar width. Default is 1.0.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - normalize_by_column_index (int or None): If set, each row will be normalized by the value in this column.
    - distances (list or array-like): Labels for x-axis ticks (used instead of column names or positions).
    """
    # Optional normalization
    if normalize_by_column_index is not None:
        norm_values = df.iloc[:, normalize_by_column_index]
        df = df.divide(norm_values, axis=0)

    # Use default x positions if none provided
    if x_positions is None:
        x_positions = range(len(df.columns))
    elif len(x_positions) != len(df.columns):
        raise ValueError("Length of x_positions must match the number of columns in df.")
    
    # Apply the scaling factor to x_positions
    x_positions = [x * x_scale for x in x_positions]
    
    # Calculate means and standard errors
    means = df.mean()
    sems = df.sem()
    
    # Set up the figure
    plt.figure(figsize=figsize)
    
    # Create bar plot
    bar_width = 0.8 * x_scale
    bars = plt.bar(x_positions, means, yerr=sems, width=bar_width, color=bar_color, capsize=5, alpha=0.8, label='Mean ± SEM')
    
    # Overlay scatter points
    for i, (col, x) in enumerate(zip(df.columns, x_positions)):
        plt.scatter(
            [x] * len(df[col]),
            df[col],
            edgecolor=scatter_color,
            facecolors='none',
            alpha=alpha,
            zorder=2,
            s=50,
            linewidth=1.0
        )
    
    # Set axis labels
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    
    # X-tick labels: use distances if provided
    # tick_labels = distances if distances is not None else df.columns
    tick_labels = x_positions if x_positions is not None else df.columns
    plt.xticks(x_positions, tick_labels, rotation=45)

    # Final layout
    plt.title('Bar Plot with Scatter Points')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Ensure saved plots are editable
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

# %%
import matplotlib.pyplot as plt
import pandas as pd

def bar_plot_with_lines(
    df,
    figsize=(10, 6),
    bar_color='skyblue',
    line_color='gray',
    line_alpha=0.5,
    x_positions=None,
    x_scale=1.0,
    x_label=None,
    y_label=None,
    normalize_to_column_index=None
):
    """
    Creates a bar plot with mean ± SEM and lines connecting individual data rows across columns.

    Parameters:
    - df (pd.DataFrame): Input 2D DataFrame.
    - figsize (tuple): Size of the figure.
    - bar_color (str): Color for the bars.
    - line_color (str): Color for the connecting lines.
    - line_alpha (float): Transparency for the lines.
    - x_positions (array-like): Custom x positions for the bars. If None, uses default positions.
    - x_scale (float): Scaling factor for the x positions and bar width.
    - x_label (str): Label for the x-axis.
    - y_label (str): Label for the y-axis.
    - normalize_to_column_index (int or None): If specified, normalizes each row to the value at this column index.
    """
    df = df.copy()

    # Optional normalization
    if normalize_to_column_index is not None:
        if normalize_to_column_index < 0 or normalize_to_column_index >= df.shape[1]:
            raise ValueError(f"normalize_to_column_index {normalize_to_column_index} is out of bounds for DataFrame with {df.shape[1]} columns.")
        norm_col = df.columns[normalize_to_column_index]
        df = df.div(df[norm_col], axis=0)

    # X positions
    if x_positions is None:
        x_positions = list(range(len(df.columns)))
    elif len(x_positions) != len(df.columns):
        raise ValueError("Length of x_positions must match the number of columns in df.")
    x_positions = [x * x_scale for x in x_positions]

    # Mean and SEM
    means = df.mean()
    sems = df.sem()

    # Start plot
    plt.figure(figsize=figsize)

    # Plot bars
    bar_width = 0.8 * x_scale
    plt.bar(x_positions, means, yerr=sems, width=bar_width, color=bar_color,
            capsize=5, alpha=0.8, label='Mean ± SEM')

    # Plot individual lines
    for i in range(len(df)):
        plt.plot(x_positions, df.iloc[i], color=line_color, alpha=line_alpha, linewidth=1)

    # Labels
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    # X ticks and final touches
    plt.title('Bar Plot with Individual Lines')
    tick_labels = x_positions if x_positions is not None else df.columns
    plt.xticks(x_positions, tick_labels, rotation=45)
    # plt.xticks(x_positions, df.columns, rotation=45)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # Editable text settings for export
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype']  = 42
    plt.rcParams['svg.fonttype'] = 'none'


# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bar_plot_with_lines_and_median(
    df,
    distances,
    label='Group',
    figsize=(10, 5),
    bar_color='skyblue',
    line_color='gray',
    line_alpha=0.4,
    x_scale=20.0,
    x_label=None,
    y_label=None,
    normalize_to_column_index=None,
    show_lines=True,
    show_bars=True,
    sem_on='mean'  # 'mean' or 'median'
):
    """
    Creates a bar plot with mean/median ± SEM, individual traces, and optional normalization and sorting.

    Parameters:
    - df (pd.DataFrame): Data to plot (subjects x bins).
    - distances (array-like): Corresponding distances for each column.
    - label (str): Label for the group.
    - bar_color (str): Color for bars.
    - line_color (str): Color for individual lines.
    - normalize_to_column_index (int or None): If specified, normalize each row to this column.
    - show_lines (bool): Whether to draw individual subject lines.
    - show_bars (bool): Whether to draw bar summary.
    - sem_on (str): 'mean' or 'median' - determines central tendency and SEM calculation.
    """
    df = df.copy()
    
    if len(distances) != df.shape[1]:
        raise ValueError("Length of distances must match number of columns in df")

    original_columns = df.columns.tolist()

    # Normalize before sorting
    if normalize_to_column_index is not None:
        norm_col = original_columns[normalize_to_column_index]
        df = df.div(df[norm_col], axis=0)

    # Sort by distances
    distance_series = pd.Series(distances, index=original_columns)
    sorted_columns = distance_series.sort_values().index
    sorted_distances = distance_series.sort_values().values

    df = df[sorted_columns]

    # X-axis setup
    x_positions = np.arange(len(sorted_columns)) * x_scale
    bar_width = 0.5 * x_scale

    # Summary stats
    if sem_on == 'mean':
        y = df.mean().values
        sem = df.sem().values
    elif sem_on == 'median':
        y = df.median().values
        sem = np.std(df.values, axis=0, ddof=1) / np.sqrt(df.shape[0])
    else:
        raise ValueError("sem_on must be 'mean' or 'median'")

    # Plot
    plt.figure(figsize=figsize)

    if show_bars:
        plt.bar(x_positions, y, yerr=sem, width=bar_width, color=bar_color,
                capsize=5, alpha=0.8, label=f'{label} {"Mean" if sem_on=="mean" else "Median"} ± SEM')

    if show_lines:
        for i in range(df.shape[0]):
            plt.plot(x_positions, df.iloc[i].values, color=line_color, alpha=line_alpha, linewidth=1)

    if not show_bars:
        plt.errorbar(x_positions, y, yerr=sem, fmt='o-', color=bar_color,
                     label=f'{label} {"Mean" if sem_on=="mean" else "Median"} ± SEM')

    # Add median trace
    plt.plot(x_positions, df.median().values, color='black', linewidth=3, linestyle='--', label='Median')

    # X-ticks and labels
    plt.xticks(x_positions, labels=np.round(sorted_distances * x_scale, 1), rotation=45)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.title('Group Summary Plot')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return sorted_distances


# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def robust_sem(arr):
    """Robust SEM based on median absolute deviation."""
    arr = np.array(arr, dtype=np.float64)
    mad = np.nanmedian(np.abs(arr - np.nanmedian(arr)))
    n = np.sum(~np.isnan(arr))
    return 1.253 * mad / np.sqrt(n) if n > 0 else np.nan

from scipy.stats import mannwhitneyu, ttest_ind
import scipy.stats as stats

def bar_plot_two_dfs_with_lines_and_median(
    df1,
    df2,
    distances,
    label1='Group 1',
    label2='Group 2',
    figsize=(12, 6),
    bar_colors=('skyblue', 'lightcoral'),
    line_colors=('gray', 'darkred'),
    line_alpha=0.4,
    x_scale=20.0,
    x_label=None,
    y_label=None,
    normalize_to_column_index1=None,
    normalize_to_column_index2=None,
    show_lines=True,
    show_bars=True,
    error_on='mean',  # 'mean' or 'median'
    stat_test='mannwhitneyu',  # 'mannwhitneyu', 'ttest', or 'two_way_anova'
    title='Comparison of Two Groups',
    median_line_colors=('black', 'black')
):
    df1 = df1.copy()
    df2 = df2.copy()

    if list(df1.columns) != list(df2.columns):
        raise ValueError("df1 and df2 must have the same column structure")
    if len(distances) != len(df1.columns):
        raise ValueError("Length of distances must match number of columns in df1 and df2")

    original_columns = df1.columns.tolist()

    # Normalize BEFORE sorting
    if normalize_to_column_index1 is not None:
        norm_col1 = original_columns[normalize_to_column_index1]
        df1 = df1.div(df1[norm_col1], axis=0)

    if normalize_to_column_index2 is not None:
        norm_col2 = original_columns[normalize_to_column_index2]
        df2 = df2.div(df2[norm_col2], axis=0)

    # Sort columns by distance
    distance_series = pd.Series(distances, index=original_columns)
    sorted_columns = distance_series.sort_values().index
    sorted_distances = distance_series.sort_values().values

    df1 = df1[sorted_columns]
    df2 = df2[sorted_columns]

    x_positions = np.arange(len(sorted_columns)) * x_scale
    bar_width = 0.35 * x_scale
    offset1 = x_positions - bar_width / 2
    offset2 = x_positions + bar_width / 2

    mean1 = df1.mean()
    mean2 = df2.mean()
    median1 = df1.median()
    median2 = df2.median()

    if error_on == 'mean':
        sem1 = df1.sem()
        sem2 = df2.sem()
        y1 = mean1
        y2 = mean2
    elif error_on == 'median':
        sem1 = df1.apply(robust_sem, axis=0)
        sem2 = df2.apply(robust_sem, axis=0)
        y1 = median1
        y2 = median2
    else:
        raise ValueError("error_on must be 'mean' or 'median'")
        
  
    # Statistical test
    p_values = []
    anova_results = None
    
    if stat_test == 'two_way_anova':
        # Prepare data for two-way ANOVA
        # Factor 1: Group (df1 vs df2)
        # Factor 2: Bin (column/distance)
        
        anova_data = []
        for col in sorted_columns:
            # Add df1 data
            group1_data = pd.to_numeric(df1[col], errors='coerce').dropna()
            for val in group1_data:
                anova_data.append({
                    'value': val,
                    'group': label1,
                    'bin': col,
                    'distance': distance_series[col]
                })
            
            # Add df2 data
            group2_data = pd.to_numeric(df2[col], errors='coerce').dropna()
            for val in group2_data:
                anova_data.append({
                    'value': val,
                    'group': label2,
                    'bin': col,
                    'distance': distance_series[col]
                })
        
        anova_df = pd.DataFrame(anova_data)
        
        # Perform two-way ANOVA
        try:
            # Create dummy variables for categorical factors
            group_dummies = pd.get_dummies(anova_df['group'], prefix='group')
            bin_dummies = pd.get_dummies(anova_df['bin'], prefix='bin')
            
            # Combine all predictors
            X = pd.concat([group_dummies.iloc[:, :-1], bin_dummies.iloc[:, :-1]], axis=1)
            y = anova_df['value']
            
            # Perform ANOVA using scipy
            from scipy.stats import f_oneway
            
            # Main effect of group
            group1_all = anova_df[anova_df['group'] == label1]['value']
            group2_all = anova_df[anova_df['group'] == label2]['value']
            f_group, p_group = f_oneway(group1_all, group2_all)
            
            # Main effect of bin
            bin_groups = [anova_df[anova_df['bin'] == col]['value'] for col in sorted_columns]
            f_bin, p_bin = f_oneway(*bin_groups)
            
            anova_results = {
                'group_effect': {'F': f_group, 'p': p_group},
                'bin_effect': {'F': f_bin, 'p': p_bin},
                'data': anova_df
            }
            
            # For plotting, still do pairwise tests for each bin
            for col in sorted_columns:
                group1 = pd.to_numeric(df1[col], errors='coerce').dropna()
                group2 = pd.to_numeric(df2[col], errors='coerce').dropna()
                _, p = mannwhitneyu(group1, group2, alternative='two-sided')
                p_values.append(p)
                
        except Exception as e:
            print(f"Two-way ANOVA failed: {e}. Falling back to Mann-Whitney U tests.")
            # Fallback to pairwise tests
            for col in sorted_columns:
                group1 = pd.to_numeric(df1[col], errors='coerce').dropna()
                group2 = pd.to_numeric(df2[col], errors='coerce').dropna()
                _, p = mannwhitneyu(group1, group2, alternative='two-sided')
                p_values.append(p)
    
    else:
        # Original pairwise tests
        for col in sorted_columns:
            group1 = pd.to_numeric(df1[col], errors='coerce').dropna()
            group2 = pd.to_numeric(df2[col], errors='coerce').dropna()
            
            if stat_test == 'mannwhitneyu':
                _, p = mannwhitneyu(group1, group2, alternative='two-sided')
            elif stat_test == 'ttest':
                _, p = ttest_ind(group1, group2, equal_var=False)
            else:
                raise ValueError("stat_test must be 'mannwhitneyu', 'ttest', or 'two_way_anova'")
            p_values.append(p)

    # Plot
    plt.figure(figsize=figsize)

    if show_bars:
        plt.bar(offset1, y1, yerr=sem1, width=bar_width, color=bar_colors[0],
                capsize=5, alpha=0.8, label=f'{label1} {error_on.capitalize()} ± SEM')
        plt.bar(offset2, y2, yerr=sem2, width=bar_width, color=bar_colors[1],
                capsize=5, alpha=0.8, label=f'{label2} {error_on.capitalize()} ± SEM')
    else:
        plt.errorbar(offset1, y1, yerr=sem1, fmt='o-', color='black', capsize=5, label=f'{label1} {error_on.capitalize()} ± SEM')
        plt.errorbar(offset2, y2, yerr=sem2, fmt='o--', color='black', capsize=5, label=f'{label2} {error_on.capitalize()} ± SEM')

    if show_lines:
        for i in range(len(df1)):
            plt.plot(offset1, df1.iloc[i].values, color=line_colors[0], alpha=line_alpha, linewidth=1)
        for i in range(len(df2)):
            plt.plot(offset2, df2.iloc[i].values, color=line_colors[1], alpha=line_alpha, linewidth=1)

    # Annotate significance - FIXED VERSION
    for i, (col, p) in enumerate(zip(sorted_columns, p_values)):
        if pd.isna(p):
            continue  # Skip if p-value is NaN or invalid

        # Determine significance stars
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            continue
    
        # Get bar/line height using column names instead of integer indexing
        try:
            y_val1 = float(y1[col]) if not pd.isna(y1[col]) else 0
            y_val2 = float(y2[col]) if not pd.isna(y2[col]) else 0
            sem_val1 = float(sem1[col]) if not pd.isna(sem1[col]) else 0
            sem_val2 = float(sem2[col]) if not pd.isna(sem2[col]) else 0
    
            y_max = max(y_val1 + sem_val1, y_val2 + sem_val2)
            plt.text(x_positions[i], y_max * 1.05, star, ha='center', va='bottom', fontsize=14, color='black')
        except Exception as e:
            print(f"Could not annotate star at column {col} due to: {e}")
            continue
    

    plt.plot(offset1, median1, color=median_line_colors[0], linewidth=3, label=f'{label1} Median')
    plt.plot(offset2, median2, color=median_line_colors[1], linewidth=3, label=f'{label2} Median')

    plt.xticks(x_positions, labels=np.round(sorted_distances * x_scale, 1), rotation=45)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    # Print ANOVA results if performed
    if anova_results:
        print("\n" + "="*50)
        print("TWO-WAY ANOVA RESULTS")
        print("="*50)
        print(f"Main Effect of Group ({label1} vs {label2}):")
        print(f"  F = {anova_results['group_effect']['F']:.4f}")
        print(f"  p = {anova_results['group_effect']['p']:.4f}")
        if anova_results['group_effect']['p'] < 0.05:
            print("  *** SIGNIFICANT ***")
        else:
            print("  Not significant")
            
        print(f"\nMain Effect of Bin (Distance):")
        print(f"  F = {anova_results['bin_effect']['F']:.4f}")
        print(f"  p = {anova_results['bin_effect']['p']:.4f}")
        if anova_results['bin_effect']['p'] < 0.05:
            print("  *** SIGNIFICANT ***")
        else:
            print("  Not significant")
        print("="*50)
    
    plt.show()

    # Font settings for vector graphics
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'

    return sorted_distances, p_values, anova_results

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import mannwhitneyu, ttest_ind

def robust_sem(arr):
    arr = np.array(arr, dtype=np.float64)
    mad = np.nanmedian(np.abs(arr - np.nanmedian(arr)))
    n = np.sum(~np.isnan(arr))
    return 1.253 * mad / np.sqrt(n) if n > 0 else np.nan

def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def bar_plot_two_dfs_with_lines_and_median(
    df1,
    df2,
    distances,
    label1='Group 1',
    label2='Group 2',
    figsize=(12, 6),
    bar_colors=('skyblue', 'lightcoral'),
    line_colors=('gray', 'darkred'),
    line_alpha=0.4,
    x_scale=20.0,
    x_label=None,
    y_label=None,
    normalize_to_column_index1=None,
    normalize_to_column_index2=None,
    show_lines=True,
    show_bars=True,
    error_on='mean',  # 'mean' or 'median'
    stat_test='mannwhitneyu',
    title='Comparison of Two Groups',
    median_line_colors=('black', 'black'),
    fit_gaussian=False,
    offset_off=True
):
    df1 = df1.copy()
    df2 = df2.copy()

    if list(df1.columns) != list(df2.columns):
        raise ValueError("df1 and df2 must have the same column structure")
    if len(distances) != len(df1.columns):
        raise ValueError("Length of distances must match number of columns in df1 and df2")

    original_columns = df1.columns.tolist()

    if normalize_to_column_index1 is not None:
        norm_col1 = original_columns[normalize_to_column_index1]
        df1 = df1.div(df1[norm_col1], axis=0)

    if normalize_to_column_index2 is not None:
        norm_col2 = original_columns[normalize_to_column_index2]
        df2 = df2.div(df2[norm_col2], axis=0)

    distance_series = pd.Series(distances, index=original_columns)
    sorted_columns = distance_series.sort_values().index
    sorted_distances = distance_series.sort_values().values

    df1 = df1[sorted_columns]
    df2 = df2[sorted_columns]

    x_positions = np.arange(len(sorted_columns)) * x_scale
    bar_width = 0.35 * x_scale
    offset1 = x_positions - bar_width / 2
    offset2 = x_positions + bar_width / 2
    
    if offset_off:
        offset1=x_positions
        offset2=x_positions

    mean1 = df1.mean()
    mean2 = df2.mean()
    median1 = df1.median()
    median2 = df2.median()

    if error_on == 'mean':
        sem1 = df1.sem()
        sem2 = df2.sem()
        y1 = mean1
        y2 = mean2
    elif error_on == 'median':
        sem1 = df1.apply(robust_sem, axis=0)
        sem2 = df2.apply(robust_sem, axis=0)
        y1 = median1
        y2 = median2
    else:
        raise ValueError("error_on must be 'mean' or 'median'")
        
    # Statistical tests
    p_values = []
    for col in sorted_columns:
        group1 = pd.to_numeric(df1[col], errors='coerce').dropna()
        group2 = pd.to_numeric(df2[col], errors='coerce').dropna()
        
        if stat_test == 'mannwhitneyu':
            _, p = mannwhitneyu(group1, group2, alternative='two-sided')
        elif stat_test == 'ttest':
            _, p = ttest_ind(group1, group2, equal_var=False)
        else:
            raise ValueError("stat_test must be 'mannwhitneyu' or 'ttest'")
        p_values.append(p)

    # Plotting
    plt.figure(figsize=figsize)

    if show_bars:
        plt.bar(offset1, y1, yerr=sem1, width=bar_width, color=bar_colors[0],
                capsize=5, alpha=0.8, label=f'{label1} {error_on.capitalize()} ± SEM')
        plt.bar(offset2, y2, yerr=sem2, width=bar_width, color=bar_colors[1],
                capsize=5, alpha=0.8, label=f'{label2} {error_on.capitalize()} ± SEM')
    else:
        plt.errorbar(offset1, y1, yerr=sem1, fmt='o-', color='black', capsize=5, label=f'{label1} {error_on.capitalize()} ± SEM')
        plt.errorbar(offset2, y2, yerr=sem2, fmt='o--', color='black', capsize=5, label=f'{label2} {error_on.capitalize()} ± SEM')

    if show_lines:
        for i in range(len(df1)):
            plt.plot(offset1, df1.iloc[i].values, color=line_colors[0], alpha=line_alpha, linewidth=1)
        for i in range(len(df2)):
            plt.plot(offset2, df2.iloc[i].values, color=line_colors[1], alpha=line_alpha, linewidth=1)

    for i, (col, p) in enumerate(zip(sorted_columns, p_values)):
        if pd.isna(p):
            continue
        if p < 0.001:
            star = '***'
        elif p < 0.01:
            star = '**'
        elif p < 0.05:
            star = '*'
        else:
            continue

        try:
            y_val1 = float(y1[col]) if not pd.isna(y1[col]) else 0
            y_val2 = float(y2[col]) if not pd.isna(y2[col]) else 0
            sem_val1 = float(sem1[col]) if not pd.isna(sem1[col]) else 0
            sem_val2 = float(sem2[col]) if not pd.isna(sem2[col]) else 0

            y_max = max(y_val1 + sem_val1, y_val2 + sem_val2)
            plt.text(x_positions[i], y_max * 1.05, star, ha='center', va='bottom', fontsize=14, color='black')
        except Exception as e:
            print(f"Could not annotate star at column {col} due to: {e}")
            continue

    # Median overlays
    plt.plot(offset1, median1, color=median_line_colors[0], linewidth=3, label=f'{label1} Median')
    plt.plot(offset2, median2, color=median_line_colors[1], linewidth=3, label=f'{label2} Median')

    # Gaussian fit
    if fit_gaussian:
        for x_vals, y_vals, label, color in zip(
            [offset1, offset2],
            [y1.values, y2.values],
            [label1, label2],
            ['blue', 'red']
        ):
            try:
                popt, _ = curve_fit(gaussian, x_vals, y_vals, p0=[np.max(y_vals), np.mean(x_vals), np.std(x_vals)])
                
                # Unpack Gaussian parameters
                amp, mean, std = popt
                fwhm = 2.355 * std
                half_max = amp / 2
                
                # Generate fitted curve
                x_gauss = np.linspace(min(x_vals), max(x_vals), 500)
                y_gauss = gaussian(x_gauss, *popt)
                
                # Plot Gaussian fit
                plt.plot(x_gauss, y_gauss, '--', color=color, linewidth=2, label=f'{label} Gaussian Fit')
                
                # Compute x-limits of half-max
                delta = np.sqrt(2 * np.log(2)) * std
                x_left = mean - delta
                x_right = mean + delta
                
                # Plot horizontal line at half-max between the two half-max x-values
                plt.hlines(half_max, x_left, x_right, linestyle='--', color=color, alpha=0.8)
                
                # Add FWHM label centered above the half-max line
                plt.text(mean, half_max * 1.05, f'FWHM = {fwhm:.2f}', fontsize=10,
                color=color, ha='center', va='bottom')
                                
            except RuntimeError:
                print(f"Gaussian fit failed for {label}.")

    plt.xticks(x_positions, labels=np.round(sorted_distances * x_scale, 1), rotation=45)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['svg.fonttype'] = 'none'
    plt.show()

    return sorted_distances, p_values, None

# %%

import matplotlib.pyplot as plt

def plot_histogram(
    data,
    bins=20,
    xlabel="X-axis",
    ylabel="Y-axis",
    title="Histogram",
    color="blue",
    edgecolor="black",
    figsize=(6, 4),
    xlim=None,
    ylim=None,
    xticks=None,
    yticks=None
):
    """
    General function to plot a histogram.

    Parameters
    ----------
    data : array-like
        Input data to plot.
    bins : int or sequence, optional
        Number of bins or bin edges.
    xlabel, ylabel, title : str, optional
        Axis labels and plot title.
    color : str, optional
        Fill color of histogram bars.
    edgecolor : str, optional
        Color of bar edges.
    figsize : tuple, optional
        Figure size (width, height).
    xlim, ylim : tuple, optional
        Axis limits (min, max).
    xticks, yticks : list, optional
        Custom tick locations.
    """

    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

    plt.tight_layout()
    plt.show()


# %%

def create_anova_dataframe(df1, df2, label1, label2, sorted_columns, subject_id_col=None):
    """
    Create a properly formatted dataframe for two-way repeated measures ANOVA.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First dataset
    df2 : pandas.DataFrame
        Second dataset
    label1 : str
        Label for first group
    label2 : str
        Label for second group
    sorted_columns : list
        List of column names to analyze (these become the bins)
    subject_id_col : str, optional
        Column name for subject IDs. If None, will create sequential IDs
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame formatted for ANOVA analysis with columns:
        ['value', 'group', 'bin', 'subject']
    """
    
    # Prepare data for two-way repeated measures ANOVA
    anova_data = []
    
    # Determine subject IDs
    if subject_id_col and subject_id_col in df1.columns and subject_id_col in df2.columns:
        # Use provided subject ID column
        df1_subjects = df1[subject_id_col].tolist()
        df2_subjects = df2[subject_id_col].tolist()
    else:
        # Create sequential subject IDs
        df1_subjects = list(range(len(df1)))
        df2_subjects = list(range(len(df2)))
    
    for col in sorted_columns:
        # Add df1 data
        group1_data = pd.to_numeric(df1[col], errors='coerce').dropna()
        valid_indices = group1_data.index
        
        for idx, val in enumerate(group1_data):
            original_idx = valid_indices[idx]
            subject_id = df1_subjects[original_idx] if original_idx < len(df1_subjects) else f"subj_{original_idx}"
            
            anova_data.append({
                'value': val,
                'group': label1,
                'bin': str(col),
                'subject': f"{label1}_{subject_id}"  # Unique subject ID per group
            })
        
        # Add df2 data
        group2_data = pd.to_numeric(df2[col], errors='coerce').dropna()
        valid_indices = group2_data.index
        
        for idx, val in enumerate(group2_data):
            original_idx = valid_indices[idx]
            subject_id = df2_subjects[original_idx] if original_idx < len(df2_subjects) else f"subj_{original_idx}"
            
            anova_data.append({
                'value': val,
                'group': label2,
                'bin': str(col),
                'subject': f"{label2}_{subject_id}"  # Unique subject ID per group
            })
    
    return pd.DataFrame(anova_data)
# %%


import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def perform_two_way_repeated_anova(df, alpha=0.05, post_hoc=True):
    """
    Perform two-way repeated measures ANOVA on the formatted dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns ['value', 'group', 'bin', 'subject']
        Output from create_anova_dataframe function
    alpha : float
        Significance level (default: 0.05)
    post_hoc : bool
        Whether to perform post-hoc tests (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'anova_table': ANOVA results table
        - 'summary': Summary of significant effects
        - 'post_hoc_group': Post-hoc results for group comparisons (if requested)
        - 'post_hoc_bin': Post-hoc results for bin comparisons (if requested)
        - 'post_hoc_interaction': Post-hoc results for interaction (if requested)
        - 'assumptions': Results of assumption checks
    """
    
    # Ensure proper data types
    df = df.copy()
    df['group'] = df['group'].astype('category')
    df['bin'] = df['bin'].astype('category')
    df['subject'] = df['subject'].astype('category')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    print(f"Data shape: {df.shape}")
    print(f"Groups: {df['group'].unique()}")
    print(f"Bins: {df['bin'].unique()}")
    print(f"Number of subjects: {df['subject'].nunique()}")
    
    # Check for balanced design
    design_check = df.groupby(['group', 'bin']).size().reset_index(name='count')
    print("\nDesign balance check:")
    print(design_check.pivot(index='group', columns='bin', values='count'))
    
    results = {}
    
    # Fit the model with subject as random effect
    try:
        # Using mixed effects model approach
        formula = 'value ~ C(group) * C(bin) + C(subject)'
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        
        results['anova_table'] = anova_table
        
        # Extract p-values for summary
        group_p = anova_table.loc['C(group)', 'PR(>F)']
        bin_p = anova_table.loc['C(bin)', 'PR(>F)']
        interaction_p = anova_table.loc['C(group):C(bin)', 'PR(>F)']
        
        # Create summary
        summary = {
            'group_effect': {
                'F': anova_table.loc['C(group)', 'F'],
                'p_value': group_p,
                'significant': group_p < alpha
            },
            'bin_effect': {
                'F': anova_table.loc['C(bin)', 'F'],
                'p_value': bin_p,
                'significant': bin_p < alpha
            },
            'interaction_effect': {
                'F': anova_table.loc['C(group):C(bin)', 'F'],
                'p_value': interaction_p,
                'significant': interaction_p < alpha
            }
        }
        
        results['summary'] = summary
        
        # Print summary
        print(f"\n{'='*50}")
        print("TWO-WAY REPEATED MEASURES ANOVA RESULTS")
        print(f"{'='*50}")
        print(f"Main effect of Group: F = {summary['group_effect']['F']:.3f}, p = {summary['group_effect']['p_value']:.4f}")
        if summary['group_effect']['significant']:
            print("  *** SIGNIFICANT ***")
        
        print(f"Main effect of Bin: F = {summary['bin_effect']['F']:.3f}, p = {summary['bin_effect']['p_value']:.4f}")
        if summary['bin_effect']['significant']:
            print("  *** SIGNIFICANT ***")
            
        print(f"Group × Bin Interaction: F = {summary['interaction_effect']['F']:.3f}, p = {summary['interaction_effect']['p_value']:.4f}")
        if summary['interaction_effect']['significant']:
            print("  *** SIGNIFICANT ***")
        
    except Exception as e:
        print(f"Error fitting ANOVA model: {e}")
        return None
    
    # Post-hoc tests if requested
    if post_hoc:
        print(f"\n{'='*50}")
        print("POST-HOC TESTS")
        print(f"{'='*50}")
        
        # Group comparisons (if main effect is significant)
        if summary['group_effect']['significant']:
            try:
                tukey_group = pairwise_tukeyhsd(df['value'], df['group'], alpha=alpha)
                results['post_hoc_group'] = tukey_group
                print("\nTukey HSD for Group:")
                print(tukey_group)
            except Exception as e:
                print(f"Error in group post-hoc test: {e}")
        
        # Bin comparisons (if main effect is significant)
        if summary['bin_effect']['significant']:
            try:
                tukey_bin = pairwise_tukeyhsd(df['value'], df['bin'], alpha=alpha)
                results['post_hoc_bin'] = tukey_bin
                print("\nTukey HSD for Bin:")
                print(tukey_bin)
            except Exception as e:
                print(f"Error in bin post-hoc test: {e}")
        
        # Simple effects analysis if interaction is significant
        if summary['interaction_effect']['significant']:
            print("\nSimple Effects Analysis (Group differences within each Bin):")
            simple_effects = []
            
            for bin_val in df['bin'].unique():
                bin_data = df[df['bin'] == bin_val]
                if len(bin_data['group'].unique()) > 1:
                    try:
                        # T-test between groups for this bin
                        group1_data = bin_data[bin_data['group'] == bin_data['group'].unique()[0]]['value']
                        group2_data = bin_data[bin_data['group'] == bin_data['group'].unique()[1]]['value']
                        
                        t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
                        simple_effects.append({
                            'bin': bin_val,
                            't_statistic': t_stat,
                            'p_value': p_val,
                            'significant': p_val < alpha
                        })
                        
                        print(f"  Bin {bin_val}: t = {t_stat:.3f}, p = {p_val:.4f}", end="")
                        if p_val < alpha:
                            print(" ***")
                        else:
                            print()
                            
                    except Exception as e:
                        print(f"  Error analyzing bin {bin_val}: {e}")
            
            results['simple_effects'] = simple_effects
    
    # Basic assumption checks
    print(f"\n{'='*50}")
    print("ASSUMPTION CHECKS")
    print(f"{'='*50}")
    
    assumptions = {}
    
    # Normality check (Shapiro-Wilk on residuals)
    try:
        residuals = model.resid
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        assumptions['normality'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'assumption_met': shapiro_p > alpha
        }
        print(f"Normality (Shapiro-Wilk): W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")
        if shapiro_p > alpha:
            print("  Normality assumption MET")
        else:
            print("  Normality assumption VIOLATED")
    except Exception as e:
        print(f"Error in normality test: {e}")
    
    # Homogeneity of variance (Levene's test)
    try:
        groups_for_levene = [df[df['group'] == group]['value'].values for group in df['group'].unique()]
        levene_stat, levene_p = stats.levene(*groups_for_levene)
        assumptions['homogeneity'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'assumption_met': levene_p > alpha
        }
        print(f"Homogeneity of Variance (Levene): W = {levene_stat:.3f}, p = {levene_p:.4f}")
        if levene_p > alpha:
            print("  Homogeneity assumption MET")
        else:
            print("  Homogeneity assumption VIOLATED")
    except Exception as e:
        print(f"Error in homogeneity test: {e}")
    
    results['assumptions'] = assumptions
    results['model'] = model
    
    return results

# Example usage function
def run_anova_analysis(df1, df2, label1, label2, sorted_columns, subject_id_col=None):
    """
    Complete pipeline: format data and run ANOVA analysis.
    
    Parameters:
    -----------
    df1, df2 : pandas.DataFrame
        Your two datasets
    label1, label2 : str
        Labels for the groups
    sorted_columns : list
        Column names to analyze
    subject_id_col : str, optional
        Subject ID column name
        
    Returns:
    --------
    dict : ANOVA results
    """
    # Format data for ANOVA
    anova_df = create_anova_dataframe(df1, df2, label1, label2, sorted_columns, subject_id_col)
    
    # Run ANOVA
    results = perform_two_way_repeated_anova(anova_df)
    
    return results
#
# %%
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import pandas as pd
import statsmodels.formula.api as smf

def run_mixed_effects_anova(df):
    """
    Perform a two-way repeated measures ANOVA using mixed effects model.

    Assumes:
    - Column 0: dependent variable (value)
    - Column 1: first factor (e.g., group)
    - Column 2: second factor (e.g., bin)
    - Column 4: subject ID
    """
    df = df.copy()
    df.columns = ['value', 'group', 'bin', 'irrelevant', 'subject']

    # Convert to categorical
    df['group'] = df['group'].astype('category')
    df['bin'] = df['bin'].astype('category')
    df['subject'] = df['subject'].astype('category')

    # Fit the mixed effects model
    model = smf.mixedlm("value ~ group * bin", df, groups=df["subject"])
    result = model.fit(reml=False)

    print(result.summary())
    return result





# %%


import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind, shapiro

def plot_ecdf_comparison(distances1, distances2,
                         label1='Group 1', label2='Group 2',
                         line_color1='blue', line_color2='red',
                         line_width=2, figsize=(4, 4),
                         xlabel='Distance', ylabel='ECDF',
                         title='ECDF Comparison of 3D Transition Distances',
                         show_stats=True,
                         stat_test='ks',
                         log_x=False,
                         xlim=None,
                         ylim=None,
                         xticks_start=None, xticks_end=None, xticks_step=None,
                         yticks_start=None, yticks_end=None, yticks_step=None,
                         show_normality_pvals=False):
    """
    Plots ECDFs of two distance arrays for comparison, with optional statistical and normality test annotations.

    Optional tick control:
        xticks_start, xticks_end, xticks_step: control x-axis ticks
        yticks_start, yticks_end, yticks_step: control y-axis ticks

    stat_test:
        'ks', 'mannwhitney', 'ttest', or 'auto' (auto selects test based on normality)
    """
    from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind, shapiro

    # Remove NaNs
    distances1 = distances1[~np.isnan(distances1)]
    distances2 = distances2[~np.isnan(distances2)]

    # Sort values
    x1 = np.sort(distances1)
    y1 = np.arange(1, len(x1)+1) / len(x1)

    x2 = np.sort(distances2)
    y2 = np.arange(1, len(x2)+1) / len(x2)

    # Plot ECDFs
    plt.figure(figsize=figsize)
    plt.plot(x1, y1, label=label1, color=line_color1, lw=line_width)
    plt.plot(x2, y2, label=label2, color=line_color2, lw=line_width)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)

    # Apply log scale if requested
    if log_x:
        plt.xscale('log')

    # Set axis limits if provided
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    # Set custom ticks if specified
    if None not in (xticks_start, xticks_end, xticks_step):
        plt.xticks(np.arange(xticks_start, xticks_end + xticks_step, xticks_step))
    if None not in (yticks_start, yticks_end, yticks_step):
        plt.yticks(np.arange(yticks_start, yticks_end + yticks_step, yticks_step))

    # Show mean and median
    median1, mean1 = np.median(distances1), np.mean(distances1)
    median2, mean2 = np.median(distances2), np.mean(distances2)

    stats_text = (f"{label1}: median={median1:.2g}, mean={mean1:.2g}\n"
                  f"{label2}: median={median2:.2g}, mean={mean2:.2g}")
    plt.text(0.05, 0.95, stats_text, ha='left', va='top',
             fontsize=10, transform=plt.gca().transAxes)

    # Run statistical test
    if show_stats:
        norm1_p = shapiro(distances1)[1]
        norm2_p = shapiro(distances2)[1]

        if stat_test == 'auto':
            if norm1_p > 0.05 and norm2_p > 0.05:
                stat, p_value = ttest_ind(distances1, distances2, equal_var=False)
                stat_label = f"t-test: p = {p_value:.3g}"
            else:
                stat, p_value = mannwhitneyu(distances1, distances2, alternative='two-sided')
                stat_label = f"Mann-Whitney U: p = {p_value:.3g}"
        elif stat_test == 'ks':
            stat, p_value = ks_2samp(distances1, distances2)
            stat_label = f"KS test: p = {p_value:.3g}"
        elif stat_test == 'mannwhitney':
            stat, p_value = mannwhitneyu(distances1, distances2, alternative='two-sided')
            stat_label = f"Mann-Whitney U: p = {p_value:.3g}"
        elif stat_test == 'ttest':
            stat, p_value = ttest_ind(distances1, distances2, equal_var=False)
            stat_label = f"t-test: p = {p_value:.3g}"
        else:
            raise ValueError("stat_test must be 'auto', 'ks', 'mannwhitney', or 'ttest'")

        # Annotate main stat test result
        plt.text(0.95, 0.05, stat_label, ha='right', va='bottom',
                 fontsize=10, transform=plt.gca().transAxes)
        
            # Plot triangle markers at median x-values on x-axis
        arrow_y = -0.02  # small value just below 0
        plt.plot(median1, arrow_y, marker='v', color=line_color1, markersize=10, clip_on=False)
        plt.plot(median2, arrow_y, marker='v', color=line_color2, markersize=10, clip_on=False)
            
        # Optionally show normality test p-values
        if show_normality_pvals:
            normality_text = (f"Shapiro-Wilk:\n"
                              f"{label1} p={norm1_p:.3g}, "
                              f"{label2} p={norm2_p:.3g}")
            plt.text(0.95, 0.95, normality_text, ha='right', va='top',
                     fontsize=10, transform=plt.gca().transAxes)

    # Styling
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import sem
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def plot_mean_sem_with_anova(values1, values2, distances,
                              label1='Group 1', label2='Group 2',
                              line_color1='blue', line_color2='red',
                              line_width=2, figsize=(5, 4),
                              custom_xticks=None, custom_xtick_labels=None,
                              xlabel='Distance from Stim', ylabel='Proportion',
                              title='Proportion by Distance Bin',
                              show_stats=True,
                              xlim=None, ylim=None):
    """
    Plots mean ± SEM of two datasets across distance bins and performs two-way ANOVA.
    
    Parameters
    ----------
    values1, values2 : 2D np.arrays or DataFrames
        Rows = distance bins; columns = samples or repetitions.
    distances : 1D array-like
        Distance bin labels (used for x-axis).
    """
    # Convert to numpy arrays
    values1 = np.array(values1)
    values2 = np.array(values2)
    distances = np.array(distances)

    # Compute means and SEMs
    mean1 = np.nanmean(values1, axis=1)
    sem1 = sem(values1, axis=1, nan_policy='omit')

    mean2 = np.nanmean(values2, axis=1)
    sem2 = sem(values2, axis=1, nan_policy='omit')

    # Plot
    plt.figure(figsize=figsize)
    plt.errorbar(distances, mean1, yerr=sem1, label=label1,
             color=line_color1, lw=line_width, capsize=5, fmt='-o')

    plt.errorbar(distances, mean2, yerr=sem2, label=label2,
             color=line_color2, lw=line_width, capsize=5, fmt='-o')


    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Styling
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Custom x-tick positions and labels
    if custom_xticks is not None and custom_xtick_labels is not None:
        plt.xticks(custom_xticks, custom_xtick_labels)

    plt.tight_layout()

    # Run 2-way ANOVA
    if show_stats:
        # Prepare melted dataframe for statsmodels
        df1 = pd.DataFrame(values1, index=distances)
        df1 = df1.reset_index().melt(id_vars='index', var_name='rep', value_name='value')
        df1['group'] = label1

        df2 = pd.DataFrame(values2, index=distances)
        df2 = df2.reset_index().melt(id_vars='index', var_name='rep', value_name='value')
        df2['group'] = label2

        df_all = pd.concat([df1, df2], ignore_index=True)
        df_all.rename(columns={'index': 'distance'}, inplace=True)

        # ANOVA
        model = ols('value ~ C(group) + C(distance) + C(group):C(distance)', data=df_all).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        # Display ANOVA results
        print("Two-way ANOVA results:")
        print(anova_table)

        # Optional annotation on plot
        pval_group = anova_table.loc['C(group)', 'PR(>F)']
        pval_dist = anova_table.loc['C(distance)', 'PR(>F)']
        pval_interaction = anova_table.loc['C(group):C(distance)', 'PR(>F)']
        stat_text = (f"Group p={pval_group:.3g}\n"
                     f"Distance p={pval_dist:.3g}\n"
                     f"Interaction p={pval_interaction:.3g}")
        plt.text(0.95, 0.05, stat_text, ha='right', va='bottom',
                 transform=plt.gca().transAxes, fontsize=10)

    plt.show()


# %%


def plot_autocorrelation_and_fit(
    acorr_vector,
    time_vector,
    mono_fit_params,
    dual_fit_params,
    r2_mono,
    bic_mono,
    r2_dual,
    bic_dual,
    cell_id=0,
    save_dir="fit_plots",
    title_prefix="overlay_fit",
    color_raw="gray",
    color_mono="orange",
    color_dual="blue",
    figsize=(12, 6),
    custom_filename=None,
    text_size=12,
    num_ticks=5,
    xlim=None,
    ylim=None,
    tick_bounds=None  # expected format: ((x_tick_min, x_tick_max), (y_tick_min, y_tick_max))
):
    """
    Plot and save mono- and dual-exponential fits over autocorrelation data.

    Parameters:
    - acorr_vector: 1D array, raw autocorrelation
    - time_vector: 1D array, same length as acorr_vector
    - mono_fit_params: dict with ['A', 'tau', 'offset']
    - dual_fit_params: dict with ['A0', 'tau0', 'A1', 'tau1', 'offset']
    - r2_mono, bic_mono: float
    - r2_dual, bic_dual: float
    - cell_id: for labeling
    - save_dir: where to save SVG
    - title_prefix: prefix for filename
    - color_raw, color_mono, color_dual: colors for plots
    - figsize: tuple of (width, height)
    - custom_filename: optional fixed filename
    - text_size: font size for labels and annotations
    - num_ticks: number of ticks on each axis
    - xlim, ylim: range to display
    - tick_bounds: tuple ((x_tick_min, x_tick_max), (y_tick_min, y_tick_max)) to set ticks independently
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract fit parameters
    A = mono_fit_params['A']
    tau = mono_fit_params['tau']
    offset_m = mono_fit_params['offset']

    A0 = dual_fit_params['A0']
    tau0 = dual_fit_params['tau0']
    A1 = dual_fit_params['A1']
    tau1 = dual_fit_params['tau1']
    offset_d = dual_fit_params['offset']

    # Generate fit curves
    fit_curve_mono = A * np.exp(-time_vector / tau) + offset_m
    fit_curve_dual = A0 * np.exp(-time_vector / tau0) + A1 * np.exp(-time_vector / tau1) + offset_d

    # Start plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_vector, acorr_vector, label='Original Data', color=color_raw, alpha=0.6)
    ax.plot(time_vector, fit_curve_mono, label='Mono-exponential Fit', color=color_mono)
    ax.plot(time_vector, fit_curve_dual, label='Dual-exponential Fit', color=color_dual)

    ax.set_title(f'Exponential Fits vs Data (cell {cell_id})', fontsize=text_size + 2)
    ax.set_xlabel('Time', fontsize=text_size)
    ax.set_ylabel('Value', fontsize=text_size)
    ax.tick_params(labelsize=text_size - 1)
    ax.legend(fontsize=text_size)

    # Remove upper and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Apply xlim and ylim if provided
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Apply tick bounds independently
    if tick_bounds is not None:
        x_ticks = np.linspace(*tick_bounds[0], num_ticks)
        y_ticks = np.linspace(*tick_bounds[1], num_ticks)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

    # Annotations
    ax.text(
        0.7 * time_vector[-1],
        max(fit_curve_dual),
        f"Dual Fit\nτ₀ = {tau0:.2f}, τ₁ = {tau1:.2f}\nR² = {r2_dual:.3f}, BIC = {bic_dual:.2f}",
        fontsize=text_size - 2,
        bbox=dict(facecolor='lightblue', alpha=0.7)
    )
    ax.text(
        0.7 * time_vector[-1],
        0.75 * max(fit_curve_dual),
        f"Mono Fit\nτ = {tau:.2f}\nR² = {r2_mono:.3f}, BIC = {bic_mono:.2f}",
        fontsize=text_size - 2,
        bbox=dict(facecolor='navajowhite', alpha=0.7)
    )

    # File saving
    if custom_filename:
        filename = custom_filename if custom_filename.endswith('.jpg') else custom_filename + '.jpg'
    else:
        tau_str = f"tau0_{tau0:.2f}_tau1_{tau1:.2f}"
        filename = f"{title_prefix}_cell{cell_id}_{tau_str}.jpg"

    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, bbox_inches='tight', format='jpg', dpi=300)
    print(f"Saved SVG: {save_path}")


# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.stats import zscore

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
    zscore_traces=True,
    title=None,
    peak_times=None  # in seconds
):
    """
    Plots selected traces from a nested list of dF/F data in a non-overlapping vertical stack.
    Optionally annotates each trace with a peak time arrow if provided.

    Parameters:
    - peak_times: list or array (same length as `data`) of peak times in seconds.
                  Arrows will be drawn if value is not NaN.
    """

    if trace_indices is None:
        trace_indices = list(range(min(5, len(data))))  # Default to first 5

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, idx in enumerate(trace_indices):
        trace = data[idx][0]
        if cutoff is not None:
            trace = trace[cutoff[0]:cutoff[1]]
        if apply_medfilt:
            trace = medfilt(trace, kernel_size=medfilt_kernel)
        if zscore_traces:
            trace = zscore(trace, nan_policy='omit')

        time = np.arange(len(trace)) / sampling_rate
        trace_offset = trace + i * offset

        ax.plot(time, trace_offset, color=color, alpha=alpha)

        # ✅ Draw arrow at peak time if provided and valid
        if peak_times is not None and not np.isnan(peak_times[idx]):
            peak_time = peak_times[idx]
            if cutoff is not None:
                start_time = cutoff[0] / sampling_rate
                end_time = cutoff[1] / sampling_rate
                if not (start_time <= peak_time <= end_time):
                    continue  # Skip if outside visible window
            arrow_y = np.max(trace_offset) + offset * 0.1
            ax.annotate(
                '', 
                xy=(peak_time, arrow_y),
                xytext=(peak_time, arrow_y + offset * 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2, mutation_scale=20),
                clip_on=False
            )
            
        if peak_times is not None and not np.isnan(peak_times[idx]):
            peak_time = peak_times[idx]
            if cutoff is not None:
                start_time = cutoff[0] / sampling_rate
                end_time = cutoff[1] / sampling_rate
                if not (start_time <= peak_time <= end_time):
                    continue  # Skip if peak is outside cropped window
        
            # Draw a long dashed vertical line
            ax.axvline(
                x=peak_time,
                color='red',
                linestyle='--',
                linewidth=1.5,
                alpha=0.8
            )

    # Add scale bar
    if y_scale_bar:
        x0 = time[0] - 0.05 * (time[-1] - time[0])
        y0 = offset * 0.25
        ax.plot([x0, x0], [y0, y0 + scale_bar_length], color='black', lw=2, clip_on=False)
        ax.text(x0, y0 + scale_bar_length / 2, scale_bar_label,
                va='center', ha='right', fontsize=10)

    if title:
        ax.set_title(title, fontsize=12)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Trace Offset', fontsize=12)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    current_ylim = ax.get_ylim()
    ax.set_ylim(top=current_ylim[1] + offset * 0.5)

    if save_path:
        if not save_path.endswith('.svg'):
            save_path += '.svg'
        fig.savefig(save_path, format='svg', transparent=True)
        print(f"SVG saved to: {os.path.abspath(save_path)}")

    return fig


# %%


def plot_cross_correlation_means(
    summary_dfs,
    labels=None,
    colors=None,
    alphas=None,
    lags=None,
    figsize=(6, 6),
    title="Mean Cross-Correlation",
    ylabel="Cross-Corr",
    xlabel="Lag (frames)"
):
    """
    Plots the mean ± SEM cross-correlation for up to 4 datasets.
    
    Parameters:
    - summary_dfs: list of DataFrames, each containing a 'mean_cross_corr' column (pd.Series of 1D arrays)
    - labels: list of strings, labels for each trace
    - colors: list of color strings
    - alphas: list of floats (0-1) for transparency
    - lags: optional 1D array or list of lag values to use for x-axis
    - figsize: tuple for figure size
    - title: plot title
    - ylabel: y-axis label
    - xlabel: x-axis label
    """
    plt.figure(figsize=figsize)
    
    n_sets = len(summary_dfs)
    assert n_sets <= 4, "Can only plot up to 4 datasets."

    for i in range(n_sets):
        df = summary_dfs[i]
        data_series = df["mean_cross_corr"]
        
        # Stack into 2D array: (trials × lags)
        array_2d = np.vstack(data_series.values)

        mean_vals = np.mean(array_2d, axis=0)
        sem_vals  = np.std(array_2d, axis=0) / np.sqrt(array_2d.shape[0])
        
        # Set x-axis values (lags)
        if lags is None:
            mid = array_2d.shape[1] // 2
            x_vals = np.arange(-mid, array_2d.shape[1] - mid)
        else:
            x_vals = np.array(lags)
            if len(x_vals) != array_2d.shape[1]:
                raise ValueError("Length of `lags` must match the number of cross-correlation points.")

        label = labels[i] if labels else f"Set {i+1}"
        color = colors[i] if colors else None
        alpha = alphas[i] if alphas else 1.0
        
        plt.plot(x_vals, mean_vals, label=label, color=color, alpha=alpha)
        plt.fill_between(x_vals, mean_vals - sem_vals, mean_vals + sem_vals,
                         color=color, alpha=alpha * 0.3)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    if lags is None or 0 in x_vals:
        plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    # plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()

    
# %%
import numpy as np
import matplotlib.pyplot as plt

def prepare_and_plot_heatmap(
    result_df,
    trace_column='averaged_traces_all',
    time_column='time_axes',
    sort_by='peak_time_array_mean_trial',
    vmin=-2,
    vmax=2,
    cmap='viridis',
    exclude_window=None,  # tuple (start_time, end_time) in seconds
    invert_y=False,
    xticks=None,
    yticks=None,
    xlim=None,
    figsize=(8, 6),
    cbar_shrink=1.0,  # scalar to shrink colorbar height (1.0 = no shrink)
    norm_type=None,
    align=True
):
    """
    Aligns traces using the first time axis in each row, sorts them, and plots a heatmap.

    Parameters:
        result_df (pd.DataFrame): DataFrame with columns for sorting, traces, and time axes.
        trace_column (str): Column name containing the trace arrays.
        time_column (str): Column name containing list of time axes (first one used).
        sort_by (str): Column name to sort the DataFrame by.
        vmin (float): Min value for heatmap color scale.
        vmax (float): Max value for heatmap color scale.
        cmap (str): Matplotlib colormap name.
        exclude_window (tuple or None): Time window (start, end) to exclude from plotting.
        invert_y (bool): If True, y-axis counts go down (top-to-bottom).
        xticks (list or None): Custom x tick locations.
        yticks (list or None): Custom y tick locations.
        xlim (tuple or None): Limits for x-axis (min, max).
        figsize (tuple): Figure size.
        cbar_height_shrink (float): Scalar to shrink colorbar height (default 1.0 = no shrink).
    """
    # Align all traces to a shared time axis
    
    common_time_df, aligned_traces_df = analyzeEvoked2P.align_averaged_traces_from_lists(
        result_df,
        trace_col=trace_column,
        time_col=time_column
    )

    # Reset index before sorting
    result_df_reset = result_df.reset_index(drop=True)

    # Sort result_df and get sorted positional indices
    sorted_df = result_df_reset.sort_values(by=sort_by,ascending=False)
    sorted_positions = sorted_df.index

    # Reorder aligned traces by sorted positions using iloc
    aligned_traces_sorted = aligned_traces_df.iloc[sorted_positions].reset_index(drop=True)
    ascending=False

    # Extract time axis as 1D array
    time_axis = common_time_df['time'].values

    # Exclude window mask (True = keep)
    if exclude_window is not None:
        keep_mask = (time_axis < exclude_window[0]) | (time_axis > exclude_window[1])
    else:
        keep_mask = np.ones_like(time_axis, dtype=bool)

    # Apply exclusion mask to traces and time axis
    time_axis_plot = time_axis[keep_mask]
    # data_plot = aligned_traces_sorted.loc[:, keep_mask]
    data_plot = aligned_traces_sorted.loc[:, keep_mask].to_numpy(copy=True)

    # Normalize each row to its max (optional)
    if norm_type == 'minmax':
        for i in range(data_plot.shape[0]):
            row = data_plot[i, :]
            row_min = np.nanmin(row)
            row_max = np.nanmax(row)
            if row_max > row_min:
                data_plot[i, :] = (row - row_min) / (row_max - row_min)
    elif norm_type == 'absmax':
        for i in range(data_plot.shape[0]):
            max_abs = np.nanmax(np.abs(data_plot[i, :]))
            if max_abs != 0:
                data_plot[i, :] = data_plot[i, :] / max_abs

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)

    extent = [time_axis_plot[0], time_axis_plot[-1], 0, data_plot.shape[0]]

    im = ax.imshow(
        data_plot,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        origin='upper' if invert_y else 'lower'
    )

    # Axis labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trials')

    # xlim
    if xlim is not None:
        ax.set_xlim(xlim)

    # xticks
    if xticks is not None:
        ax.set_xticks(xticks)

    # yticks
    if yticks is not None:
        ax.set_yticks(yticks)

    # Colorbar with height shrink
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    cbar = fig.colorbar(im, cax=cax)
    # cbar = ax.figure.colorbar(im, ax=ax, shrink=cbar_shrink)
    cbar.set_label('Signal')

    # Shrink colorbar height
    if cbar_shrink != 1.0:
        pos = cax.get_position()
        new_height = pos.height * cbar_shrink
        new_y0 = pos.y0 + (pos.height - new_height) / 2
        cax.set_position([pos.x0, new_y0, pos.width, new_height])

    # Invert y axis if requested
    if invert_y:
        ax.invert_yaxis()

    plt.tight_layout()
    plt.show()
# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import sem, linregress
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def plot_single_point_sem_with_anova(
    df1,
    df2,
    group_labels=("Real", "Pseudo"),
    ylabel="Value",
    xlabel="Condition",
    title="Mean ± SEM per Condition",
    colors=("skyblue", "lightcoral"),
    show_scatter=True,
    show_regression=False,
    log_y=False,
    figsize=(6, 5),
    xlim=None,
    ylim=None,
    show=True
):
    """
    Plots mean ± SEM per condition per group, with optional scatter and regression.
    Performs two-way ANOVA and Tukey HSD posthoc tests.
    Annotates significant Tukey results directly on the plot.
    """
    assert df1.columns.equals(df2.columns), "df1 and df2 must have the same columns"

    conditions = df1.columns
    x = np.arange(len(conditions))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot means and SEMs
    for i, (df, label, color) in enumerate(zip([df1, df2], group_labels, colors)):
        means = df.mean()
        errors = df.apply(sem, nan_policy='omit')

        offset = -0.15 if i == 0 else 0.15
        x_offset = x + offset

        ax.errorbar(
            x_offset, means, yerr=errors,
            fmt='o', color=color, capsize=4, markersize=8,
            label=label
        )

        if show_scatter:
            for j, cond in enumerate(conditions):
                yvals = df[cond].dropna()
                jitter = np.random.normal(loc=offset, scale=0.03, size=len(yvals))
                ax.scatter(x[j] + jitter, yvals, alpha=0.6, color=color, s=20)

        if show_regression:
            slope, intercept, r_value, _, _ = linregress(x, means)
            ax.plot(x, intercept + slope * x, linestyle='--', color=color)
            ax.text(0.05, 0.9 - 0.08 * i,
                    f"{label} R={r_value:.2f}",
                    transform=ax.transAxes, fontsize=10, color=color)

    # Aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if log_y:
        ax.set_yscale("log")

    # ===== Two-way ANOVA =====
    df1_long = df1.melt(var_name="condition", value_name="value")
    df1_long["group"] = group_labels[0]

    df2_long = df2.melt(var_name="condition", value_name="value")
    df2_long["group"] = group_labels[1]

    full_df = pd.concat([df1_long, df2_long], ignore_index=True)
    model = ols('value ~ C(group) * C(condition)', data=full_df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\n=== Two-Way ANOVA ===")
    print(anova_table)

    # ===== Posthoc Tukey HSD =====
    print("\n=== Posthoc Tukey HSD (all group-condition combinations) ===")
    full_df["group_condition"] = full_df["group"].astype(str) + "_" + full_df["condition"].astype(str)
    tukey = pairwise_tukeyhsd(endog=full_df["value"], groups=full_df["group_condition"], alpha=0.05)
    print(tukey.summary())

    # ===== Annotate Tukey results (Real vs Pseudo within each condition) =====
    def p_to_asterisks(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    y_max = np.nanmax([df1.max().max(), df2.max().max()])
    y_offset = 0.05 * y_max if not log_y else 0.1 * y_max

    for i, cond in enumerate(conditions):
        group1 = f"{group_labels[0]}_{cond}"
        group2 = f"{group_labels[1]}_{cond}"

        # Look up Tukey result for this pair
        for res in tukey._results_table.data[1:]:
            g1, g2, meandiff, p_adj, lower, upper, reject = res
            # Convert to proper types
            p_adj = float(p_adj)
            reject = str(reject) == "True"

            if (g1 == group1 and g2 == group2) or (g1 == group2 and g2 == group1):
                if reject:
                    y_star = max(df1[cond].max(), df2[cond].max()) + y_offset
                    ax.text(i, y_star, p_to_asterisks(p_adj), ha='center', va='bottom', fontsize=14, color='black')

    # ===== Add ANOVA effect p-values to plot (in lower right) =====
    effects = ['C(group)', 'C(condition)', 'C(group):C(condition)']
    y_text = 0.05
    for i, effect in enumerate(effects):
        pval = anova_table.loc[effect, "PR(>F)"]
        label = effect.replace('C(', '').replace(')', '').replace(':', ' x ')
        ax.text(0.95, y_text + i * 0.05,
                f"{label}: p={pval:.3e}",
                ha='right', va='bottom', transform=ax.transAxes, fontsize=9, color='gray')

    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if show:
        plt.show()

    return anova_table, tukey


