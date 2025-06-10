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

    
    
#%%
def heatmap_of_responses_sorted(result1,result2,max_COM="max"):
    
    conditional_zscores_no_nan_df=result1.get("conditional_zscores_no_nan_df").T
    
    if max_COM=="max":
        moving_avg=result2.get("moving_avg")
        conditional_zscores_no_nan_df_max=conditional_zscores_no_nan_df.iloc[moving_avg.idxmax(axis=1).argsort()]

    if max_COM=="COM":
        center_of_mass_stim_time=result2.get("conditional_zscores_no_nan_df_max_time").to_numpy()
        conditional_zscores_no_nan_df_max=conditional_zscores_no_nan_df.iloc[center_of_mass_stim_time.argsort()]
    
    
    
    
    # conditional_zscores_no_nan_df_max=conditional_zscores_no_nan_df
    
    plt.figure(figsize=(10, 10))
    vmin=-2
    vmax=2
    
    sns.heatmap(conditional_zscores_no_nan_df_max,annot=False,vmin=vmin, vmax=vmax,cmap="coolwarm",cbar_kws={'label': 'Z-Score'})
    # sns.ecdfplot(conditional_zscores_no_nan_df_max_time) 
    

# %%

def scatter_plots_of_stim_responses(result):
    
    peaktimes_mean = result.get("peaktimes_mean")
    peaktimes_std = result.get("peaktimes_std")
    peaks_mean = result.get("peaks_mean")
    peaks_std = result.get("peaks_std")
    peak_width_mean = result.get("peak_width_mean")
    peak_width_std = result.get("peak_width_std")
    peak_count = result.get("peak_count")
    Stim_cell_mean_peak = result.get("Stim_cell_mean_peak")
    
    
    # plt.scatter(peaktimes_mean,peaktimes_std)
    
    # plt.scatter(peaks_mean,peaks_std)
    
    # plt.scatter(peaktimes_mean,peaktimes_std)
    
    # plt.scatter(peak_width_mean,peak_width_std)
    
    
    # plt.scatter(peak_width_mean,peaks_mean)
    
    # plt.scatter(peak_count,peaks_mean)
    
    # plt.scatter(Stim_cell_mean_peak,peaks_mean)
    
    # plt.scatter(peak_count,peaktimes_mean)
    # plt.scatter(Stim_cell_mean_peak,peak_count)
    
    
    # plt.scatter(dist_opto,peak_count)
    
    # plt.scatter(dist_opto,peaktimes_std)
    
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(peaktimes_mean,peaks_mean,peak_count)
    
    print(np.mean(peak_count.to_numpy().flatten()))
    # %%
def plot_tau(result1,result2):
   
    sampling_interval=0.032958316
    data_samples=len(np.nanmean(result1.get("normalizedDataIndividual"),axis=1))
    time=np.linspace(sampling_interval, sampling_interval*data_samples, data_samples)
    tau=result1.get("tau")
    
    
    plt.figure()
    ecdf = sm.distributions.ECDF(tau)
    
    x = np.linspace(min(tau), max(tau),1000)
    y = ecdf(x)
    plt.step(x, y)
    plt.show()
    
    
    # plt.xlim(0,300)
    plt.title('Distribution of taus', fontsize=20)
    plt.xlabel('Tau of imaged cell');
    
    
    tau=result2.get("tau")
    plt.figure()
    ecdf = sm.distributions.ECDF(tau)
    
    x = np.linspace(min(tau), max(tau),1000)
    y = ecdf(x)
    plt.step(x, y)
    plt.show()
    
    # fig, ax = plt.subplots()
    # # ax.step(x, y)
    # # ax.spines[['right', 'top']].set_visible(False)
    plt.xlim(0,300)
    plt.ylim(0,1)
    plt.show()
    # plt.axis('square')
    
    
    
# %%
def plot_mean_trace(result1,result2,y_limits=[-8,8]):
    
    sampling_interval=0.032958316
    data_samples=len(np.nanmean(result1.get("normalizedDataIndividual"),axis=1))
    time=np.linspace(sampling_interval, sampling_interval*data_samples, data_samples)
    
    
    # x = np.linspace(min(tau), max(tau),1000)
    # # x=x*sampling_interval
    # y = ecdf(x)
    # x=x*sampling_interval
    
    fig, ax = plt.subplots()
    # ax.step(x, y)
    # ax.spines[['right', 'top']].set_visible(False)
    # # plt.xlim(0,300)
    # # plt.ylim(0,1)
    # # plt.show()
    # # plt.axis('square')
    
    
    
    # ax.set_ylim(-.5, 1)
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)
    
    # y1=np.nanmean(result1.get("normalizedDataIndividual"),axis=1)
    # y1=result1.get("normalizedData_all")
    # y1_sem=result1.get("normalizedDataSem")
    
    # y2=result2.get("normalizedData_all")
    # y2_sem=result2.get("normalizedDataSem")
    
    # plt.plot(time,y1)
    # plt.plot(time,y2)
    
        
    # plt.fill_between(time, (y1+y1_sem).flatten(), (y1-y1_sem).flatten(), facecolor='black', alpha=0.75)
    # plt.fill_between(time, (y2+y2_sem).flatten(), (y2-y2_sem).flatten(), facecolor='red', alpha=0.75)
    
    # plt.fill_between(time, (y1+y1_sem).to_numpy().flatten(), (y1-y1_sem).to_numpy().flatten(), facecolor='black', alpha=0.75)
    # plt.fill_between(time, (y2+y2_sem).to_numpy().flatten(), (y2-y2_sem).to_numpy().flatten(), facecolor='red', alpha=0.75)

    # plt.ylim(-.4,1.2)
    
    
    # plt.plot(time,mPFC_long_result.get("normalizedData"))
    # plt.plot(time,mPFC_short_result.get("normalizedData"))
    
    # plt.plot(time,V1_long_result.get("normalizedData"))
    # plt.plot(time,V1_short_result.get("normalizedData"))
    
    
    # plt.plot(time,mPFC_late_result.get("normalizedData"))
    # plt.plot(time,mPFC_early_result.get("normalizedData"))
    
    # plt.plot(time,V1_late_result.get("normalizedData"))
    # plt.plot(time,V1_early_result.get("normalizedData"))
    
    # ax.legend(['mPFC_long_tau', 'mPFC_short_tau','V1_long_tau', 'V1_short_tau'])
    
    
    
    y1=result1.get("conditional_zscores_no_nan_mean_across_cells")
    y1_sem=result1.get("average_trace_sem")
    
    y2=result2.get("conditional_zscores_no_nan_mean_across_cells")
    y2_sem=result2.get("average_trace_sem")
    
    plt.plot(time,y1)
    plt.plot(time,y2)
    
    plt.fill_between(time, (y1+y1_sem).flatten(), (y1-y1_sem).flatten(), facecolor='black', alpha=0.75)
    plt.fill_between(time, (y2+y2_sem).flatten(), (y2-y2_sem).flatten(), facecolor='red', alpha=0.75)
    
    
    plt.ylim(y_limits[0],y_limits[1])
# 

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

def plot_mean_trace_multiple_dataframe_input(result1, result2=None, result3=None, result4=None, result5=None, 
                             y_limits=[-8, 8], legend_names=None, normalize=False, exclude_windows=None):
    
    sampling_interval = 0.032958316
    data_samples = result1.shape[1]  # Direct access to DataFrame columns
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
    y1 = result1.mean(axis=0).to_numpy()  # Access columns in DataFrame
    y1_sem = result1.std(axis=0).to_numpy()  # Access columns in DataFrame
    if normalize:
        y1, y1_sem = normalize_trace_and_sem(y1, y1_sem)
    if exclude_windows:
        y1, y1_sem = exclude_trace_windows(y1, y1_sem, time, exclude_windows)
    plt.plot(time, y1, label=legend_names[0], color=colors[0])
    plt.fill_between(time, (y1 + y1_sem).flatten(), (y1 - y1_sem).flatten(), facecolor=colors[0], alpha=0.75)

    # Plot result2 if provided
    if result2 is not None:
        y2 = result2.mean(axis=0).to_numpy()
        y2_sem = result2.std(axis=0).to_numpy()
        if normalize:
            y2, y2_sem = normalize_trace_and_sem(y2, y2_sem)
        if exclude_windows:
            y2, y2_sem = exclude_trace_windows(y2, y2_sem, time, exclude_windows)
        plt.plot(time, y2, label=legend_names[1], color=colors[1])
        plt.fill_between(time, (y2 + y2_sem).flatten(), (y2 - y2_sem).flatten(), facecolor=colors[1], alpha=0.75)

    # Plot result3 if provided
    if result3 is not None:
        y3 = result3.mean(axis=0)
        y3_sem = result3.std(axis=0)
        if normalize:
            y3, y3_sem = normalize_trace_and_sem(y3, y3_sem)
        if exclude_windows:
            y3, y3_sem = exclude_trace_windows(y3, y3_sem, time, exclude_windows)
        plt.plot(time, y3, label=legend_names[2], color=colors[2])
        plt.fill_between(time, (y3 + y3_sem).flatten(), (y3 - y3_sem).flatten(), facecolor=colors[2], alpha=0.75)

    # Plot result4 if provided
    if result4 is not None:
        y4 = result4.mean(axis=0)
        y4_sem = result4.std(axis=0)
        if normalize:
            y4, y4_sem = normalize_trace_and_sem(y4, y4_sem)
        if exclude_windows:
            y4, y4_sem = exclude_trace_windows(y4, y4_sem, time, exclude_windows)
        plt.plot(time, y4, label=legend_names[3], color=colors[3])
        plt.fill_between(time, (y4 + y4_sem).flatten(), (y4 - y4_sem).flatten(), facecolor=colors[3], alpha=0.75)

    # Plot result5 if provided
    if result5 is not None:
        y5 = result5.mean(axis=0)
        y5_sem = result5.std(axis=0)
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

def plot_mean_trace_by_tau(result1,result2,result3,result4):

    sampling_interval=0.032958316
    data_samples=len(np.nanmean(result1.get("normalizedDataIndividual"),axis=1))
    time=np.linspace(sampling_interval, sampling_interval*data_samples, data_samples)
    
    
    # x = np.linspace(min(tau), max(tau),1000)
    # # x=x*sampling_interval
    # y = ecdf(x)
    # x=x*sampling_interval
    
    fig, ax = plt.subplots()
    
    # ax.set_ylim(-.5, 1)
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)
    
    
    
    y1=result1.get("normalizedData_all")
    y1_sem=result1.get("normalizedDataSem")
    
    y2=result2.get("normalizedData_all")
    y2_sem=result2.get("normalizedDataSem")
    
    y3=result3.get("normalizedData_all")
    y3_sem=result3.get("normalizedDataSem")
    
    y4=result4.get("normalizedData_all")
    y4_sem=result4.get("normalizedDataSem")
    
    
    
    # y1=result1.get("conditional_zscores_no_nan_mean_across_cells")
    # y1_sem=result1.get("average_trace_sem")
    
    # y2=result2.get("conditional_zscores_no_nan_mean_across_cells")
    # y2_sem=result2.get("average_trace_sem")
    
    # y3=result3.get("conditional_zscores_no_nan_mean_across_cells")
    # y3_sem=result3.get("average_trace_sem")
    
    # y4=result4.get("conditional_zscores_no_nan_mean_across_cells")
    # y4_sem=result4.get("average_trace_sem")
 
    plt.plot(time,y1)
    plt.plot(time,y2)
    plt.plot(time,y3)
    plt.plot(time,y4)
    
    
    plt.fill_between(time, (y1+y1_sem).flatten(), (y1-y1_sem).flatten(), facecolor='black', alpha=0.75)
    plt.fill_between(time, (y2+y2_sem).flatten(), (y2-y2_sem).flatten(), facecolor='red', alpha=0.75)
    
    plt.fill_between(time, (y3+y3_sem).flatten(), (y3-y3_sem).flatten(), facecolor='black', alpha=0.75)
    plt.fill_between(time, (y4+y4_sem).flatten(), (y4-y4_sem).flatten(), facecolor='red', alpha=0.75)
    
       
       
    
    # plt.plot(time,mPFC_long_long.get("conditional_zscores_no_nan_mean_across_cells"))
    # plt.plot(time,mPFC_long_short.get("conditional_zscores_no_nan_mean_across_cells"))
    
    # plt.plot(time,mPFC_short_long.get("conditional_zscores_no_nan_mean_across_cells"))
    # plt.plot(time,mPFC_short_short.get("conditional_zscores_no_nan_mean_across_cells"))
    
    
    # plt.plot(time,V1_long_long.get("normalizedData"))
    # plt.plot(time,V1_long_short.get("normalizedData"))
    
    # plt.plot(time,V1_short_long.get("normalizedData"))
    # plt.plot(time,V1_short_short.get("normalizedData"))
    
    # plt.plot(time,V1_long_long.get("conditional_zscores_no_nan_mean_across_cells"))
    # plt.plot(time,V1_long_short.get("conditional_zscores_no_nan_mean_across_cells"))
    
    # plt.plot(time,V1_short_long.get("conditional_zscores_no_nan_mean_across_cells"))
    # plt.plot(time,V1_short_short.get("conditional_zscores_no_nan_mean_across_cells"))
    
    ax.legend(['long_long', 'long_short','short_long', 'short_short'])
    
    
    # plt.plot(time,mPFC_result.get("conditional_zscores_no_nan_mean_across_cells"))
    # plt.plot(time,V1_result.get("conditional_zscores_no_nan_mean_across_cells"))

    plt.ylim(-2,2)
  # %%
def plot_cum_dist_response(result1,result2,limit=1.96):  
    
  
    fig, ax = plt.subplots()

    sns.ecdfplot(result1.get("stim_responsive").flatten(), ax = ax)
    sns.ecdfplot(result2.get("stim_responsive").flatten(), ax = ax)
    
    # ax.set_xlim(0, 10)
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)
    
    
    plt.text(.2, .2, 'median= '+ str(np.nanmedian(result1.get("stim_responsive").flatten())), fontsize = 10)
    plt.text(.2, .1, 'median= '+ str(np.nanmedian(result2.get("stim_responsive").flatten())), fontsize = 10)
    
    
    a=(stats.ks_2samp(result1.get("stim_responsive").flatten(),result2.get("stim_responsive").flatten()))
    
    plt.text(.2, .3, 'p= '+ str(a[1]), fontsize = 10)
    
    ax.set_xlabel('Proportion of total cells responding');
    ax.set_ylabel('Cumulative Proportion');
    
    
    ax.set_title('Average_proportion_of_cells_responding_to_stim_above_a_z-score_of: ' + str(limit), fontsize=20)

    return a
 # %%
def plot_cum_dist_succesful_stim(result1,result2,limit=1.96):  
    
    plt.figure(figsize=(25, 25))
       
    # stim_responsive=Stim_cell_mean_peak_above_threshold.count().to_numpy('int')/len(Stim_cell_mean_peak_above_threshold)
    
    temp1=result1.get("peak_count").to_numpy().flatten()
    temp2=result2.get("peak_count").to_numpy().flatten()
    
    sns.ecdfplot(temp1) 
    sns.ecdfplot(temp2) 
    # plt.xlim(0,5)
    
    plt.xlabel('Proportion of total cells responding');
    plt.ylabel('Cumulative Proportion');
    
    
    plt.text(.2, .2, 'median= '+ str(np.nanmedian(temp1)), fontsize = 10)
    plt.text(.2, .1, 'median= '+ str(np.nanmedian(temp2)), fontsize = 10)
    
    
    a=(stats.ks_2samp(temp1,temp2))
    
    plt.text(.2, .3, 'p= '+ str(a[1]), fontsize = 10)
    
    plt.title('Average_proportion_of_cells_responding_to_stim_above_a_z-score_of: ' + str(limit), fontsize=20)



    

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


# %% Plot distance  vs reponse amplitude

def plot_dist_relationships(result1,result2,yvalues):
    
    sampling_interval=0.032958316

    # plt.figure(figsize=(25, 25))
    
    # plt.scatter(mPFC_result.get("dist_opto"),result1.get("Stim_cell_mean_peak").to_numpy().flatten())
    # plt.scatter(V1_result.get("dist_opto"),V1_result.get("Stim_cell_mean_peak").to_numpy().flatten())
    
    # plt.scatter(result1.get("dist_opto"),result1.get("ccc").to_numpy().flatten())
    # plt.scatter(V1_result.get("dist_opto"),V1_result.get("ccc").to_numpy().flatten())
    # def count_excluding_nan(arr):
    #     return np.count_nonzero(~np.isnan(arr))/len(arr)
    
    Stim_cell_peaks_trials_count = result1.get("ccc").count(axis='columns').to_numpy()
    fig, ax = plt.subplots()

    # np.count_nonzero(~np.isnan(arr))/len(arr)
    bins=[10,200,11]
    bins_list=np.linspace(bins[0],bins[1],bins[2])
    bins_list=bins_list+(bins_list[1]-bins_list[0])*.5
    
    if yvalues=="peaks":
        aes=bin_data(bins,result1.get("dist_opto"),result1.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        aes2=bin_data(bins,result2.get("dist_opto"),result2.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
    # aes=bin_data(bins,result1.get("dist_opto").to_numpy()[:,0],Stim_cell_peaks_trials_count,result1.get("dist_opto"))
        # ax.set_ylim([3,8]);
        
        y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
        y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
    
    if yvalues=="number_of_cells":
        # aes=bin_data(bins,result1.get("dist_opto"),result1.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        # aes2=bin_data(bins,result2.get("dist_opto"),result2.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        
        # binned_dist=bin_data(bins,result1.get("dist_opto"),result1.get("dist_opto"),result1.get("dist_opto"))
        # binned_dist2=bin_data(bins,result2.get("dist_opto"),result2.get("dist_opto"),result1.get("dist_opto"))
        
        # y1=pd.DataFrame(aes).count(axis=1)
        # y2=pd.DataFrame(aes2).count(axis=1)
        
        # dist_y1=pd.DataFrame(binned_dist).count(axis=1)
        # dist_y2=pd.DataFrame(binned_dist2).count(axis=1)
        
        
        
        # y1=y1/dist_y1
        # y2=y2/dist_y2
        
        y1=result1.get("responsive_cells_by_dist_bin")
        y2=result2.get("responsive_cells_by_dist_bin")
        
        y_tau_dist_err=pd.DataFrame(y1).sem(axis=1)
        y_tau_dist_err2=pd.DataFrame(y2).sem(axis=1)
        
        y1=np.nanmean(y1,axis=1)
        y2=np.nanmean(y2,axis=1)
        
    # aes=bin_data(bins,result1.get("dist_opto").to_numpy()[:,0],Stim_cell_peaks_trials_count,result1.get("dist_opto"))
    
    if yvalues=="peaktimes":
        
        temp1=(result1.get("peaktimes_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan"))-300)*sampling_interval
        temp2=(result2.get("peaktimes_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan"))-300)*sampling_interval
        
        # temp1=(result1.get("peaktimes_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan"))-300)
        # temp2=(result2.get("peaktimes_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan"))-300)
        
        aes=bin_data(bins,result1.get("dist_opto"),temp1.to_numpy().flatten(),result1.get("dist_opto"))
        aes2=bin_data(bins,result2.get("dist_opto"),temp2.to_numpy().flatten(),result1.get("dist_opto"))
    
        # ax.set_ylim([3,8]);
        # aes=bin_data(bins,result1.get("dist_opto").to_numpy(),result1.get("peaktimes_mean").to_numpy().flatten(),result1.get("dist_opto"))
        # aes2=bin_data(bins,result2.get("dist_opto").to_numpy(),result2.get("peaktimes_mean").to_numpy().flatten(),result1.get("dist_opto"))
        
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
        
        y_tau_dist_err=pd.DataFrame(y1.sem(axis=1))
        y_tau_dist_err=pd.DataFrame(y2.sem(axis=1))
        
    if yvalues=="peak_count":
        temp1=result1.get("peak_count").where(~result1.get("ccc").isnull(),float("nan"))
        temp2=result2.get("peak_count").where(~result2.get("ccc").isnull(),float("nan"))
        
        # aes=bin_data(bins,result1.get("dist_opto").to_numpy(),result1.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
        # aes2=bin_data(bins,result2.get("dist_opto").to_numpy(),result2.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
    

        aes=bin_data(bins,result1.get("dist_opto"),temp1.to_numpy().flatten(),result1.get("dist_opto"))
        aes2=bin_data(bins,result2.get("dist_opto"),temp2.to_numpy().flatten(),result1.get("dist_opto"))
    
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
        
        y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
        y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
  
    # y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
    # y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
   
    ax.errorbar(bins_list[0:-1],y1,y_tau_dist_err)
    ax.errorbar(bins_list[0:-1],y2,y_tau_dist_err2)
    
    ax.plot(bins_list[0:-1],y1)
    ax.plot(bins_list[0:-1],y2)
    
  
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)
    
    # ax.set_xlim(0, 10)
    # ax.set_ylim([3,8]);
    # ax.set_ylabel('Mean z-score');
    
    ax.set_xlabel('Distance from stimulated cell');
    ax.set_ylabel('Mean z-score');
    
    # plt.gca().set_aspect('square')

# %% Plot tau of all cells binned vs etc

def plot_tau_relationships(result1,result2,yvalues):

    sampling_interval=0.032958316

    # plt.figure(figsize=(25, 25))
    
    # plt.scatter(result1.get("dist_opto"),result1.get("Stim_cell_mean_peak").to_numpy().flatten())
    # plt.scatter(result2.get("dist_opto"),result2.get("Stim_cell_mean_peak").to_numpy().flatten())
    
    # plt.scatter(result1.get("dist_opto"),result1.get("ccc").to_numpy().flatten())
    # plt.scatter(result2.get("dist_opto"),result2.get("ccc").to_numpy().flatten())
    # def count_excluding_nan(arr):
    #     return np.count_nonzero(~np.isnan(arr))/len(arr)
    
    Stim_cell_peaks_trials_count = result1.get("ccc").count(axis='columns').to_numpy()
    # 
    # np.count_nonzero(~np.isnan(arr))/len(arr)
    
    bins=[0,60,15]
    bins_list=np.linspace(bins[0],bins[1],bins[2])
    bins_list=bins_list+(bins_list[1]-bins_list[0])*.5
    bins_list=bins_list*sampling_interval
    
    fig, ax = plt.subplots()
    
   
    if yvalues=="peaks":
        aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("ccc").to_numpy().flatten(),result1.get("tau_all_cells"))
        aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("ccc").to_numpy().flatten(),result1.get("tau_all_cells"))
        
        # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("Stim_cell_mean_peak").to_numpy().flatten(),result1.get("tau_all_cells"))
        # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("Stim_cell_mean_peak").to_numpy().flatten(),result1.get("tau_all_cells"))
        
        
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
  
       
    if yvalues=="number_of_cells":
        aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        
        binned_dist=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("tau_all_cells").to_numpy(),result1.get("dist_opto"))
        binned_dist2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("tau_all_cells").to_numpy(),result1.get("dist_opto"))
        
        y1=pd.DataFrame(aes).count(axis=1)
        y2=pd.DataFrame(aes2).count(axis=1)
        
        dist_y1=pd.DataFrame(binned_dist).count(axis=1)
        dist_y2=pd.DataFrame(binned_dist2).count(axis=1)
    
         
        y1=y1/dist_y1
        y2=y2/dist_y2
        
        # y1=dist_y1
        # y2=dist_y2
        
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy()[:,0],Stim_cell_peaks_trials_count,result1.get("tau_all_cells"))
    
    if yvalues=="peaktimes":
        
        temp1=(result1.get("peaktimes_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan"))-300)*sampling_interval
        temp2=(result2.get("peaktimes_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan"))-300)*sampling_interval
        
        aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),temp1.to_numpy().flatten(),result1.get("tau_all_cells"))
        aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),temp2.to_numpy().flatten(),result1.get("tau_all_cells"))
        
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
        
        # ax.set_ylim([3,8]);
        # ax.set_ylabel('Mean z-score');
    
    if yvalues=="peak_width_mean":
        
        temp1=(result1.get("peak_width_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan")))*sampling_interval
        temp2=(result2.get("peak_width_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan")))*sampling_interval
        
        # temp1=(result1.get("peak_width_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan")))
        # temp2=(result2.get("peak_width_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan")))
        
        aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),temp1.to_numpy().flatten(),result1.get("tau_all_cells"))
        aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),temp2.to_numpy().flatten(),result1.get("tau_all_cells"))
        
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
        
        # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("peaktimes_mean").to_numpy().flatten(),result1.get("tau_all_cells"))
        # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("peaktimes_mean").to_numpy().flatten(),result1.get("tau_all_cells"))
        
    
    if yvalues=="peak_count":
        
        temp1=result1.get("peak_count").where(~result1.get("ccc").isnull(),float("nan"))
        temp2=result2.get("peak_count").where(~result2.get("ccc").isnull(),float("nan"))
        
        # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("peak_count").to_numpy().flatten(),result1.get("tau_all_cells"))
        # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("peak_count").to_numpy().flatten(),result1.get("tau_all_cells"))
    

        aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),temp1.to_numpy().flatten(),result1.get("tau_all_cells"))
        aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),temp2.to_numpy().flatten(),result1.get("tau_all_cells"))
    
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
  
    
    y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
    y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
    # y_tau_dist_err=np.nanvar(aes,axis=1)/np.sqrt(len(binned_tau_expanded.iloc[0]))
    
    # plt.bar(bins_list[0:-1],np.nanmean(aes,axis=1),width=0.15)
    # plt.bar(bins_list[0:-1],np.nanmean(aes2,axis=1),width=0.1)
    
    
    # plt.gca().set_aspect('equal')
    
    
    # fig, ax = plt.subplots()
    # ax.step(x, y)
    # ax.spines[['right', 'top']].set_visible(False)
    # # plt.xlim(0,300)
    # # plt.ylim(0,1)
    # # plt.show()
    # # plt.axis('square')
    
   
    ax.errorbar(bins_list[0:-1],y1,y_tau_dist_err)
    ax.errorbar(bins_list[0:-1],y2,y_tau_dist_err2)
    
    
    # ax.set_xlim(0, 10)
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)
    
    
    # ax.set_ylim([3,8]);
    # ax.set_ylabel('Mean z-score');
    
    ax.set_xlabel('Tau all cells');
    # ax.set_ylabel('Mean z-score');
    ax.set_ylabel('Peak_width_mean(second)');

    # ax.set_ylabel('time_to_peak');
    
    
    # ax.set_title('Average_proportion_of_cells_responding_to_stim_above_a_z-score_of: ' + str(limit), fontsize=20)
    
    # plt.ylim(([0,10]))
    
    # %% Plot delta tau of all cells binned vs etc

def plot_delta_tau_relationships(result1,result2,yvalues):

    sampling_interval=0.032958316

    
    Stim_cell_peaks_trials_count = result1.get("ccc").count(axis='columns').to_numpy()
    # 
    # np.count_nonzero(~np.isnan(arr))/len(arr)
    
    bins=[0,300,11]
    bins_list=np.linspace(bins[0],bins[1],bins[2])
    bins_list=bins_list+(bins_list[1]-bins_list[0])*.5
    bins_list=bins_list*sampling_interval
    
    # temp1=(result1.get("peaktimes_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan"))-300)*sampling_interval
    # temp2=(result2.get("peaktimes_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan"))-300)*sampling_interval
    
    delta_tau1=np.abs(result1.get("tau_all_cells")-result1.get("tau_stim_cells"))
    delta_tau2=np.abs(result2.get("tau_all_cells")-result2.get("tau_stim_cells"))
    
    # delta_tau1=(result1.get("tau_all_cells")-result1.get("tau_stim_cells"))
    # delta_tau2=(result2.get("tau_all_cells")-result2.get("tau_stim_cells"))
    
   
    fig, ax = plt.subplots()

    
    if yvalues=="peaks":
        aes=bin_data(bins,delta_tau1.to_numpy(),result1.get("ccc").to_numpy().flatten(),result1.get("tau_all_cells"))
        aes2=bin_data(bins,delta_tau2.to_numpy(),result2.get("ccc").to_numpy().flatten(),result1.get("tau_all_cells"))
        
        # # ax.set_ylim([3,8]);
        # aes=bin_data(bins,delta_tau1.to_numpy(),result1.get("Stim_cell_mean_peak").to_numpy().flatten(),result1.get("dist_opto"))
        # aes2=bin_data(bins,delta_tau2.to_numpy(),result2.get("Stim_cell_mean_peak").to_numpy().flatten(),result1.get("dist_opto"))
    
    # aes=bin_data(bins,delta_tau1.to_numpy()[:,0],Stim_cell_peaks_trials_count,result1.get("tau_all_cells"))
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
        
        y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
        y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
    
    if yvalues=="number_of_cells":
        
        # aes=bin_data(bins,delta_tau1.to_numpy(),result1.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        # aes2=bin_data(bins,delta_tau2.to_numpy(),result2.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
        
        # binned_dist=bin_data(bins,delta_tau1.to_numpy(),delta_tau1.to_numpy(),result1.get("dist_opto"))
        # binned_dist2=bin_data(bins,delta_tau2.to_numpy(),delta_tau2.to_numpy(),result1.get("dist_opto"))
        
        # y1=pd.DataFrame(aes).count(axis=1)
        # y2=pd.DataFrame(aes2).count(axis=1)
        
        # dist_y1=pd.DataFrame(binned_dist).count(axis=1)
        # dist_y2=pd.DataFrame(binned_dist2).count(axis=1)
    
        
    
        # y1=y1/dist_y1
        # y2=y2/dist_y2
        
        # y1=dist_y1
        # y2=dist_y2
        
        y1=result1.get("responsive_cells_by_delta_tau_bin")
        y2=result2.get("responsive_cells_by_delta_tau_bin")
        
        y_tau_dist_err=pd.DataFrame(y1).sem(axis=1)
        y_tau_dist_err2=pd.DataFrame(y2).sem(axis=1)
        
        y1=np.nanmean(y1,axis=1)
        y2=np.nanmean(y2,axis=1)
        
        
    if yvalues=="peaktimes":
        
        temp1=(result1.get("peaktimes_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan"))-300)*sampling_interval
        temp2=(result2.get("peaktimes_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan"))-300)*sampling_interval
        
        aes=bin_data(bins,delta_tau1.to_numpy(),temp1.to_numpy().flatten(),result1.get("tau_all_cells"))
        aes2=bin_data(bins,delta_tau2.to_numpy(),temp2.to_numpy().flatten(),result1.get("tau_all_cells"))
    
        
        # aes=bin_data(bins,delta_tau1.to_numpy(),result1.get("peaktimes_mean").to_numpy().flatten(),result1.get("tau_all_cells"))
        # aes2=bin_data(bins,delta_tau2.to_numpy(),result2.get("peaktimes_mean").to_numpy().flatten(),result1.get("tau_all_cells"))
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
        
        y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
        y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
    
    if yvalues=="peak_count":
        temp1=result1.get("peak_count").where(~result1.get("ccc").isnull(),float("nan"))
        temp2=result2.get("peak_count").where(~result2.get("ccc").isnull(),float("nan"))
        
        # aes=bin_data(bins,delta_tau1.to_numpy(),result1.get("peak_count").to_numpy().flatten(),result1.get("tau_all_cells"))
        # aes2=bin_data(bins,delta_tau2.to_numpy(),result2.get("peak_count").to_numpy().flatten(),result1.get("tau_all_cells"))
    

        aes=bin_data(bins,delta_tau1.to_numpy(),temp1.to_numpy().flatten(),result1.get("tau_all_cells"))
        aes2=bin_data(bins,delta_tau2.to_numpy(),temp2.to_numpy().flatten(),result1.get("tau_all_cells"))
    
        y1=np.nanmean(aes,axis=1)
        y2=np.nanmean(aes2,axis=1)
        
        y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
        y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
    
    
    
   
    ax.errorbar(bins_list[0:-1],y1,y_tau_dist_err)
    ax.errorbar(bins_list[0:-1],y2,y_tau_dist_err2)
    
    # ax.plot(bins_list[0:-1],y1)
    # ax.plot(bins_list[0:-1],y2)
    
  
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)
    
    
    # ax.set_ylim([3,8]);
    # ax.set_ylabel('Mean z-score');
    
    ax.set_xlabel('Abs. Delta Tau (Stim cell-recorded)');
    ax.set_ylabel('Mean z-score');
    # ax.set_ylabel('time_to_peak');
    
    
    # ax.set_title('Average_proportion_of_cells_responding_to_stim_above_a_z-score_of: ' + str(limit), fontsize=20)
    
    # plt.ylim(([0,10]))
# %% Plot distance  vs reponse amplitude

def plot_peaktime_relationships(result1,result2):
    
    # plt.figure(figsize=(25, 25))
    
    # plt.scatter(result1.get("dist_opto"),result1.get("Stim_cell_mean_peak").to_numpy().flatten())
    # plt.scatter(result2.get("dist_opto"),result2.get("Stim_cell_mean_peak").to_numpy().flatten())
    
    # plt.scatter(result1.get("dist_opto"),result1.get("ccc").to_numpy().flatten())
    # plt.scatter(result2.get("dist_opto"),result2.get("ccc").to_numpy().flatten())
    # def count_excluding_nan(arr):
    #     return np.count_nonzero(~np.isnan(arr))/len(arr)
    
    Stim_cell_peaks_trials_count = result1.get("ccc").count(axis='columns').to_numpy()
    # 
    # np.count_nonzero(~np.isnan(arr))/len(arr)
    bins=[0,100,11]
    bins_list=np.linspace(bins[0],bins[1],bins[2])
    bins_list=bins_list+(bins_list[1]-bins_list[0])*.5
    bins_list=bins_list*sampling_interval
    
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
    # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
    
    
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
    # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("ccc").to_numpy().flatten(),result1.get("dist_opto"))
    
    
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy()[:,0],Stim_cell_peaks_trials_count,result1.get("dist_opto"))
    
    
    # temp1=result1.get("peaktimes_mean").where(~result1.get("ccc").isnull(),float("nan"))
    # temp2=result2.get("peaktimes_mean").where(~result2.get("ccc").isnull(),float("nan"))
    
    
    temp1=(result1.get("peaktimes_mean").where(~result1.get("ccc").isnull() & (result1.get("peaktimes_mean")>300) ,float("nan"))-300)
    temp2=(result2.get("peaktimes_mean").where(~result2.get("ccc").isnull() & (result2.get("peaktimes_mean")>300),float("nan"))-300)
    
    # # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),temp1.to_numpy().flatten(),result1.get("dist_opto"))
    # # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),temp2.to_numpy().flatten(),result1.get("dist_opto"))
    
    
    # aes=bin_data(bins,temp1.to_numpy().flatten(),result1.get("tau_all_cells").to_numpy(),result1.get("dist_opto"))
    # aes2=bin_data(bins,temp2.to_numpy().flatten(),result2.get("tau_all_cells").to_numpy(),result1.get("dist_opto"))
    
    
    
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("peaktimes_mean").to_numpy().flatten(),result1.get("dist_opto"))
    # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("peaktimes_mean").to_numpy().flatten(),result1.get("dist_opto"))
    
    aes=bin_data(bins,temp1.to_numpy().flatten(),result1.get("dist_opto"),result1.get("dist_opto"))
    aes2=bin_data(bins,temp2.to_numpy().flatten(),result2.get("dist_opto"),result1.get("dist_opto"))
    
    
    
    # temp1=result1.get("peak_count").where(~result1.get("ccc").isnull(),float("nan"))
    # temp2=result2.get("peak_count").where(~result2.get("ccc").isnull(),float("nan"))
    
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
    # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
    
    
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),result1.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
    # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),result2.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
    
    # aes=bin_data(bins,result1.get("tau_all_cells").to_numpy(),temp1.to_numpy().flatten(),result1.get("dist_opto"))
    # aes2=bin_data(bins,result2.get("tau_all_cells").to_numpy(),temp2.to_numpy().flatten(),result1.get("dist_opto"))
    
    
    
    # aes=bin_data(bins,result1.get("tau_stim_cells").to_numpy(),result1.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
    # aes2=bin_data(bins,result2.get("tau_stim_cells").to_numpy(),result2.get("peak_count").to_numpy().flatten(),result1.get("dist_opto"))
    
    y_tau_dist_err=pd.DataFrame(aes).sem(axis=1)
    y_tau_dist_err2=pd.DataFrame(aes2).sem(axis=1)
    # y_tau_dist_err=np.nanvar(aes,axis=1)/np.sqrt(len(binned_tau_expanded.iloc[0]))
  
    
    
    fig, ax = plt.subplots()
    
    # sns.ax.ecdfplot(result1.get("stim_responsive").flatten()) 
    # sns.ax.ecdfplot(result2.get("stim_responsive").flatten()) 
    ax.errorbar(bins_list[0:-1],np.nanmean(aes,axis=1),y_tau_dist_err)
    ax.errorbar(bins_list[0:-1],np.nanmean(aes2,axis=1),y_tau_dist_err2)
    
    # ax.set_xlim(0, 10)
    ax.set_box_aspect(1)
    ax.spines[['right', 'top']].set_visible(False)
    
    
    # ax.set_ylim([3,8]);
    # ax.set_ylabel('Mean z-score');
    
    # ax.set_xlabel('Abs. Delta Tau (Stim cell-recorded)');
    # ax.set_ylabel('Mean z-score');
    # ax.set_ylabel('time_to_peak');
    
# %%

def boxplots_of_stim_responses(result):

    
    data_1 = result.get("stim_responsive_above_above").flatten()
    data_2 = result.get("stim_responsive_above_below").flatten()
    data_3 = result.get("stim_responsive_below_above").flatten()
    data_4 = result.get("stim_responsive_below_below").flatten()
    data1 = [data_1, data_2, data_3, data_4]




    plt.figure(figsize=(10, 10))
    
    # plt.plot(mPFC_stim_responsive_above_above.flatten())
    
    # plt.scatter(np.ones(len(mPFC_stim_responsive_above_above.flatten())),mPFC_stim_responsive_above_above.flatten())
    # plt.plot(mPFC_stim_responsive_above_below.flatten())
    
    
     
    
    # data_1 = V1_stim_responsive_above_above.flatten()
    # data_2 = V1_stim_responsive_above_below.flatten()
    # data_3 = V1_stim_responsive_below_above.flatten()
    # data_4 = V1_stim_responsive_below_below.flatten()
    # data2 = [data_1, data_2, data_3, data_4]
    
    # data=[data1,data2]
    # data_1 = np.random.normal(100, 10, 200)
    # data_2 = np.random.normal(90, 20, 200)
    # data_3 = np.random.normal(80, 30, 200)
    # data_4 = np.random.normal(70, 40, 200)
    # data = [data_1, data_2, data_3, data_4]
    # fig = plt.figure(figsize =(10, 7))
     
    # # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])
     
    # # Creating plot
    # bp = ax.boxplot(data)
     
    # # show plot
    # plt.show()
    bar_labels = ['long_long', 'long_short','short_long', 'short_short']
    
    
    sns.boxplot(data1).set(xlabel=bar_labels)
    sns.stripplot(data1,color="black") 
    # plt.ylim(0,.3)
    plt.legend(['long_long', 'long_short','short_long', 'short_short'])

    # sns.heatmap(np.nanmean(data1,axis=1),annot=False,vmin=0, vmax=.3,cmap="coolwarm",cbar_kws={'label': 'dF/F'})

# %%

def boxplots_of_stim_responses_time(result1,result2,result3,result4):

    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))    
    
    bar_labels = ['all', '0 to 3.3 seconds','3.3 sec to 6.6', '6.6 to 9.9']

    data_1 = result1.get("stim_responsive_above_above").flatten()
    data_2 = result2.get("stim_responsive_above_above").flatten()
    data_3 = result3.get("stim_responsive_above_above").flatten()
    data_4 = result4.get("stim_responsive_above_above").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    sns.boxplot(data1,ax=ax1).set(xlabel=bar_labels)
    sns.stripplot(data1,ax=ax1,color="black")
    ax1.set_ylim(0, .3)
    ax1.set_title('Long_long')
    
    data_1 = result1.get("stim_responsive_above_below").flatten()
    data_2 = result2.get("stim_responsive_above_below").flatten()
    data_3 = result3.get("stim_responsive_above_below").flatten()
    data_4 = result4.get("stim_responsive_above_below").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    sns.boxplot(data1,ax=ax2).set(xlabel=bar_labels)
    sns.stripplot(data1,ax=ax2,color="black") 
    ax2.set_ylim(0, .3)
    ax2.set_title('Long_short')
    
    
    
    data_1 = result1.get("stim_responsive_below_above").flatten()
    data_2 = result2.get("stim_responsive_below_above").flatten()
    data_3 = result3.get("stim_responsive_below_above").flatten()
    data_4 = result4.get("stim_responsive_below_above").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    sns.boxplot(data1,ax=ax3).set(xlabel=bar_labels)
    sns.stripplot(data1,ax=ax3,color="black") 
    ax3.set_ylim(0, .3)
    ax3.set_title('Short_long')
    
    data_1 = result1.get("stim_responsive_below_below").flatten()
    data_2 = result2.get("stim_responsive_below_below").flatten()
    data_3 = result3.get("stim_responsive_below_below").flatten()
    data_4 = result4.get("stim_responsive_below_below").flatten()
    
    
    data1 = [data_1, data_2, data_3, data_4]

    sns.boxplot(data1,ax=ax4).set(xlabel=bar_labels)
    sns.stripplot(data1,ax=ax4,color="black") 
    ax4.set_ylim(0, .3)
    ax4.set_title('Short_Short')
 
    
    
    # sns.boxplot(data1).set(xlabel=bar_labels)
    # sns.stripplot(data1,color="black") 
    # plt.ylim(0,.3)
    plt.legend(['long_long', 'long_short','short_long', 'short_short'])

    # sns.heatmap(np.nanmean(data1,axis=1),annot=False,vmin=0, vmax=.3,cmap="coolwarm",cbar_kws={'label': 'dF/F'})

# boxplots_of_stim_responses_time(mPFC_result,mPFC_early_result,mPFC_middle_late_result,mPFC_late_late_result)
# %%

def heatmap_of_stim_responses(result):

    
    data_1 = result.get("stim_responsive_above_above").flatten()
    data_2 = result.get("stim_responsive_above_below").flatten()
    data_3 = result.get("stim_responsive_below_above").flatten()
    data_4 = result.get("stim_responsive_below_below").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    # data1=np.reshape(data1,[2,2])

    data1=np.nanmean(data1,axis=1)

    data1=np.reshape(data1,[2,2])
    plt.figure(figsize=(10, 10))
    
    
    # sns.heatmap(data1,vmin=0.01, vmax=0.04,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    sns.heatmap(data1,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
# 
    # plt.ylim(0,.15)

# %%

def average_columns_to_dataframe(df):
    averaged_data = {}
    
    for col in df.columns:
        # Stack the arrays in the column vertically to create a 2D array
        stacked_arrays = np.stack(df[col].values)
        
        # Compute the mean along the vertical axis (axis=0) to get a 1D array
        column_mean = np.mean(stacked_arrays, axis=0)
        
        averaged_data[col] = column_mean
    
    # Create a DataFrame from the averaged data
    averaged_df = pd.DataFrame(averaged_data)
    
    return averaged_df


def sem_columns_to_dataframe(df):
    sem_data = {}
    
    for col in df.columns:
        # Stack the arrays in the column vertically to create a 2D array
        stacked_arrays = np.stack(df[col].values)
        
        stacked_arrays =pd.DataFrame(stacked_arrays)
        # Compute the standard deviation along the vertical axis (axis=0)
        # column_std = np.std(stacked_arrays, axis=0, ddof=1)
        column_sem = stacked_arrays.sem(axis=0,skipna=True)
        
        # Calculate the sample size (number of arrays)
        # sample_size = stacked_arrays.shape[0]
        
        # Compute the SEM
        # column_sem = column_std / np.sqrt(sample_size)
        
        sem_data[col] = column_sem
    
    # Create a DataFrame from the SEM data
    sem_df = pd.DataFrame(sem_data)
    
    return sem_df


# %%

from scipy.signal import medfilt

def analyze_columns_to_dataframes(df, start_index=0, limit=0, kernel_size=15):
    peaks_data = {}
    times_of_peaks_data = {}
    widths_at_half_max_data = {}

    for col in df.columns:
        # Stack the arrays in the column vertically to create a 2D array
        stacked_arrays = np.stack(df[col].values)

        peaks = []
        times_of_peaks = []
        widths_at_half_max = []

        for row in stacked_arrays:
            # Apply median filter to the row
            smoothed_row = medfilt(row, kernel_size=kernel_size)

            # Analyze only the part of the row from the starting index onward
            row_to_analyze = smoothed_row[start_index:]
            peak_value = np.max(row_to_analyze)
            peak_index = np.argmax(row_to_analyze) + start_index
            time_of_peak = peak_index  # Assuming time points correspond to array indices

            if peak_value < limit:
                # If the peak value is below the threshold, set peak time and width to NaN and 0 respectively
                peaks.append(np.nan)
                times_of_peaks.append(np.nan)
                widths_at_half_max.append(0)
            else:
                # Find the width at half maximum
                half_max = peak_value / 2
                # Find indices where the row crosses the half maximum value
                indices_above_half_max = np.where(row_to_analyze >= half_max)[0] + start_index

                if len(indices_above_half_max) > 1:
                    # The width at half maximum is the difference between the last and first indices
                    width_at_half_max = indices_above_half_max[-1] - indices_above_half_max[0]
                else:
                    width_at_half_max = 0  # Set width to 0 if it cannot be determined

                peaks.append(peak_value)
                times_of_peaks.append(time_of_peak)
                widths_at_half_max.append(width_at_half_max)

        peaks_data[col] = peaks
        times_of_peaks_data[col] = times_of_peaks
        widths_at_half_max_data[col] = widths_at_half_max

    # Create DataFrames from the analysis data
    peaks_df = pd.DataFrame(peaks_data)
    times_of_peaks_df = pd.DataFrame(times_of_peaks_data)
    widths_at_half_max_df = pd.DataFrame(widths_at_half_max_data)

    return peaks_df, times_of_peaks_df, widths_at_half_max_df
# Example usage with sample dataframes
# df=sorted_df1
# peaks_df, times_of_peaks_df, widths_at_half_max_df = analyze_columns_to_dataframes(df,251,4)



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
def heatmap_of_stim_responses_time(result,result2,result3,result4):

    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))    
    
    data_1 = result.get("stim_responsive_above_above").flatten()
    data_2 = result.get("stim_responsive_above_below").flatten()
    data_3 = result.get("stim_responsive_below_above").flatten()
    data_4 = result.get("stim_responsive_below_below").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    data1=np.nanmean(data1,axis=1)

    data1=np.reshape(data1,[2,2])
    
    # sns.heatmap(data1,ax=ax1,vmin=0, vmax=0.1,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    sns.heatmap(data1,ax=ax1,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    ax1.set_title('all')
    
    data_1 = result2.get("stim_responsive_above_above").flatten()
    data_2 = result2.get("stim_responsive_above_below").flatten()
    data_3 = result2.get("stim_responsive_below_above").flatten()
    data_4 = result2.get("stim_responsive_below_below").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    data1=np.nanmean(data1,axis=1)

    data1=np.reshape(data1,[2,2])
    
    # sns.heatmap(data1,ax=ax2,vmin=0, vmax=0.1,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    sns.heatmap(data1,ax=ax2,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    ax2.set_title('0 to 3.3 sec')
        
    data_1 = result3.get("stim_responsive_above_above").flatten()
    data_2 = result3.get("stim_responsive_above_below").flatten()
    data_3 = result3.get("stim_responsive_below_above").flatten()
    data_4 = result3.get("stim_responsive_below_below").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    data1=np.nanmean(data1,axis=1)

    data1=np.reshape(data1,[2,2])
    
    # sns.heatmap(data1,ax=ax3,vmin=0, vmax=0.1,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    sns.heatmap(data1,ax=ax3,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    ax3.set_title('3.3 to 6.6 sec')
    
    data_1 = result4.get("stim_responsive_above_above").flatten()
    data_2 = result4.get("stim_responsive_above_below").flatten()
    data_3 = result4.get("stim_responsive_below_above").flatten()
    data_4 = result4.get("stim_responsive_below_below").flatten()
    
    data1 = [data_1, data_2, data_3, data_4]

    data1=np.nanmean(data1,axis=1)

    data1=np.reshape(data1,[2,2])
    
    # sns.heatmap(data1,ax=ax4,vmin=0, vmax=0.1,cmap="bone", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])
    sns.heatmap(data1,ax=ax4,cmap="jet", square=True,linewidths=1, linecolor='black',clip_on=False,yticklabels=["Stim_long", "Stim_short"],xticklabels=["response_long", "response_short"])

    ax4.set_title('6.6 to 9.9 sec')

# %%
def nested_heatmap_of_stim_responses(result):

    
    data_1 = result.get("stim_responsive_above_above")
    data_2 = result.get("stim_responsive_above_below")
    data_3 = result.get("stim_responsive_below_above")
    data_4 = result.get("stim_responsive_below_below")
    
    # data1 = [data_1, data_2, data_3, data_4]
    # data1 = [data_1, data_3]
    # data1 = np.concatenate(data_1, data_2)
    data1 = np.concatenate((data_1, data_2),axis=0)
    data2 = np.concatenate((data_3, data_4),axis=0)
    
    data1=np.concatenate((data1, data2),axis=1)
    # data1=np.reshape(data1,[2,2])

    # data1=np.nanmean(data1,axis=1)

    # data1=np.reshape(data1,[2,2])
    plt.figure(figsize=(10, 10))
    
    # plt.plot(mPFC_stim_responsive_above_above.flatten())
    
    # plt.scatter(np.ones(len(mPFC_stim_responsive_above_above.flatten())),mPFC_stim_responsive_above_above.flatten())
    # plt.plot(mPFC_stim_responsive_above_below.flatten())
    
    
     
    
    # data_1 = V1_stim_responsive_above_above.flatten()
    # data_2 = V1_stim_responsive_above_below.flatten()
    # data_3 = V1_stim_responsive_below_above.flatten()
    # data_4 = V1_stim_responsive_below_below.flatten()
    # data2 = [data_1, data_2, data_3, data_4]
    
    # data=[data1,data2]
    # data_1 = np.random.normal(100, 10, 200)
    # data_2 = np.random.normal(90, 20, 200)
    # data_3 = np.random.normal(80, 30, 200)
    # data_4 = np.random.normal(70, 40, 200)
    # data = [data_1, data_2, data_3, data_4]
    # fig = plt.figure(figsize =(10, 7))
     
    # # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])
     
    # # Creating plot
    # bp = ax.boxplot(data)
     
    # # show plot
    # plt.show()
    
    # sns.heatmap(data1,square=True)
    sns.heatmap(data1)

    # plt.ylim(0,.3)
    
    


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
            trace = medfilt(trace, kernel_size=5)

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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_and_plot_metrics(df, sampling_interval=0.032958316):
    """
    Analyzes and plots the peak, peak width (FWHM), and decay rate for each trace in the DataFrame.

    Args:
        df (pd.DataFrame): A 2D DataFrame where each cell contains a 1D array (trace).
        sampling_interval (float): Time between samples, in seconds.
    """
    df = df.dropna(how='any')

    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float64))

    time = np.linspace(0, sampling_interval * (len(df.iloc[0, 0]) - 1), len(df.iloc[0, 0]))

    peaks, widths, decay_rates = [], [], []

    for column in df.columns:
        for trace in df[column]:
            # Peak value
            peak = np.nanmax(trace)
            peaks.append(peak)

            # Width at half maximum (FWHM)
            half_max = peak / 2
            indices_above_half = np.where(trace >= half_max)[0]
            if len(indices_above_half) > 1:
                fwhm = (indices_above_half[-1] - indices_above_half[0]) * sampling_interval
            else:
                fwhm = np.nan
            widths.append(fwhm)

            # Decay rate (time to 37% of peak)
            decay_threshold = 0.37 * peak
            decay_idx = np.where(trace <= decay_threshold)[0]
            if len(decay_idx) > 0:
                decay_time = decay_idx[0] * sampling_interval
            else:
                decay_time = np.nan
            decay_rates.append(decay_time)

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    metric_labels = ["Peak", "Width (FWHM)", "Decay Rate"]
    metric_data = [peaks, widths, decay_rates]

    for i, (ax, metric, label) in enumerate(zip(axs[:3], metric_data, metric_labels)):
        ax.boxplot(metric, vert=True, patch_artist=True)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True)

    # Overlay traces
    axs[3].set_title("Traces")
    for column in df.columns:
        for trace in df[column]:
            axs[3].plot(time, trace, alpha=0.6)
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Amplitude")
    axs[3].grid(True)

    plt.show()
    
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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_and_plot_metrics(df, distances, sampling_interval=0.032958316):
    """
    Analyzes and plots the peak, peak width (FWHM), and decay rate for each trace in the DataFrame.

    Args:
        df (pd.DataFrame): A 2D DataFrame where each cell contains a 1D array (trace).
        distances (list or np.ndarray): Array of float distances for binning.
        sampling_interval (float): Time between samples, in seconds.

    Returns:
        pd.DataFrame: A DataFrame containing the binned metrics (peaks, widths, decay rates).
    """
    df = df.dropna(how='any')

    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.array(x, dtype=np.float64))

    time = np.linspace(0, sampling_interval * (len(df.iloc[0, 0]) - 1), len(df.iloc[0, 0]))

    peaks, widths, decay_rates, bin_labels = [], [], [], []

    # Convert distances to integers
    distances = np.array(distances, dtype=int)

    for idx, column in enumerate(df.columns):
        for trace in df[column]:
            # Peak value
            peak = np.nanmax(trace)
            peaks.append(peak)

            # Width at half maximum (FWHM)
            half_max = peak / 2
            indices_above_half = np.where(trace >= half_max)[0]
            if len(indices_above_half) > 1:
                fwhm = (indices_above_half[-1] - indices_above_half[0]) * sampling_interval
            else:
                fwhm = np.nan
            widths.append(fwhm)

            # Decay rate (time to 37% of peak)
            decay_threshold = 0.37 * peak
            decay_idx = np.where(trace <= decay_threshold)[0]
            if len(decay_idx) > 0:
                decay_time = decay_idx[0] * sampling_interval
            else:
                decay_time = np.nan
            decay_rates.append(decay_time)

            # Assign bin label based on the corresponding distance
            bin_labels.append(distances[idx])

    # Create a DataFrame for binned metrics
    binned_data = pd.DataFrame({
        "Bin": bin_labels,
        "Peak": peaks,
        "Width (FWHM)": widths,
        "Decay Rate": decay_rates,
    })

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    metric_labels = ["Peak", "Width (FWHM)", "Decay Rate"]
    metric_data = [peaks, widths, decay_rates]

    for i, (ax, metric, label) in enumerate(zip(axs[:3], metric_data, metric_labels)):
        ax.boxplot(metric, vert=True, patch_artist=True)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(True)

    # Overlay traces
    axs[3].set_title("Traces")
    for column in df.columns:
        for trace in df[column]:
            axs[3].plot(time, trace, alpha=0.6)
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Amplitude")
    axs[3].grid(True)

    plt.show()

    return binned_data
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import ks_2samp, mannwhitneyu

# Ensure font is editable in SVG (for Illustrator)
rcParams['svg.fonttype'] = 'none'

def plot_ecdf_comparison(distances1, distances2,
                         label1='Group 1', label2='Group 2',
                         line_color1='blue', line_color2='red',
                         line_width=2, figsize=(4, 4),
                         xlabel='Distance', ylabel='ECDF',
                         title='ECDF Comparison of 3D Transition Distances',
                         show_stats=True,
                         stat_test='ks'):
    """
    Plots ECDFs of two distance arrays for comparison, with optional statistical test annotation.

    Parameters:
        distances1, distances2 (np.ndarray): 1D arrays of distances
        label1, label2 (str): Labels for the two groups
        line_color1, line_color2 (str): Colors for the ECDF lines
        line_width (float): Width of the ECDF lines
        figsize (tuple): Size of the figure in inches (width, height)
        xlabel, ylabel (str): Axis labels
        title (str): Plot title
        show_stats (bool): Whether to compute and display statistical test results
        stat_test (str): Statistical test to run ('ks' or 'mannwhitney')
    """
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

    # Run statistical test
    if show_stats:
        if stat_test == 'ks':
            stat, p_value = ks_2samp(distances1, distances2)
            stat_label = f"KS test: p = {p_value:.3g}"
        elif stat_test == 'mannwhitney':
            stat, p_value = mannwhitneyu(distances1, distances2, alternative='two-sided')
            stat_label = f"Mann-Whitney U: p = {p_value:.3g}"
        else:
            raise ValueError("stat_test must be 'ks' or 'mannwhitney'")

        # Annotate p-value
        plt.text(0.95, 0.05, stat_label, ha='right', va='bottom', fontsize=10, transform=plt.gca().transAxes)

    # Styling
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()


# %%

# import numpy as np
# import pandas as pd

# # Create a sample DataFrame with synthetic traces
# num_traces = 5
# num_timepoints = 100
# columns = ['Trace Group 1', 'Trace Group 2']

# data = {
#     'Trace Group 1': [np.sin(np.linspace(0, 2 * np.pi, num_timepoints)) * np.random.uniform(0.8, 1.2) for _ in range(num_traces)],
#     'Trace Group 2': [np.cos(np.linspace(0, 2 * np.pi, num_timepoints)) * np.random.uniform(0.8, 1.2) for _ in range(num_traces)],
# }

# df = pd.DataFrame(data)

# # Example distances array (floats)
# distances = np.random.uniform(0, 10, len(df) * len(df.columns))

# # Call the function
# results = analyze_and_plot_metrics(df, distances)

# # Access binned results
# print("Binned Peaks:")
# print(results['binned_peaks'])

# print("\nBinned Widths (FWHM):")
# print(results['binned_widths'])

# print("\nBinned Decay Rates:")
# print(results['binned_decay_rates'])