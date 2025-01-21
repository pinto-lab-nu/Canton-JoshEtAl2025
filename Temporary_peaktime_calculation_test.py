# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 10:42:05 2025

@author: jec822
"""
# %%
import pickle
file_path='C:\\Users\\Jec822\\Documents\\GitHub\\Canton-JoshEtAl2025\\temp_trial_dff.txt'
with open(file_path, 'rb') as f:
    
    data_dict={
        
    'aa': [pickle.load(f)],
    
    }
    
aa=data_dict['aa'][0]

# %%
list_of_arrays = aa["trig_dff_trials"]
array1=aa["stim_ids"]
array2=aa["roi_ids"]

# %% One way of doing this

# # Cantor pairing function
# def cantor_pairing(x, y):
#     return (x + y) * (x + y + 1) // 2 + y

# # Generate unique combination numbers
# unique_combinations = cantor_pairing(array1, array2)

# # %% simpler conceptually
# # Generate unique combination numbers using hashing
# unique_combinations = np.array([hash((x, y)) for x, y in zip(array1, array2)])

# print("Array1:", array1)
# print("Array2:", array2)
# print("Unique combination numbers:", unique_combinations)

# %% Simplest but avoids negative number that may be an issue in next step, concerned it might make repeats but dont think it will because roi id unique

# Create unique combination numbers using string concatenation
unique_combinations = np.array([int(f"{x}{y}") for x, y in zip(array1, array2)])


# %%

import pandas as pd
import numpy as np

# # Example data
# list_of_arrays = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9]), np.array([10, 11, 12])]
# unique_combinations = ['A', 'A', 'B', 'B']

# Group arrays by unique_combinations
grouped = {}
for combination, array in zip(unique_combinations, list_of_arrays):
    if combination not in grouped:
        grouped[combination] = []
    grouped[combination].append(array)

# Create a DataFrame where each cell stores a 2D array
data = {
    key: [np.vstack(value)]  # Wrap each 2D array in a list to store it as a single DataFrame cell
    for key, value in grouped.items()
}

df = pd.DataFrame.from_dict(data, orient='index', columns=['Trials'])

# Display the result
print("Grouped DataFrame:")
print(df)


# %%
from scipy.signal import find_peaks

Stim_cell_mean=pd.DataFrame(index=range(0,len(df)),columns=['mean_trace'])
Stim_cell_mean_peak=pd.DataFrame(index=range(0,len(df)),columns=['mean_trace'])
Stim_cell_peaks_trials=pd.DataFrame(index=range(0,len(df)),columns=['Peaks'])
Stim_cell_peaktimes_trials=pd.DataFrame(index=range(0,len(df)),columns=["PeakTimes"])
Stim_cell_peaks_width_trials=pd.DataFrame(index=range(0,len(df)),columns=['Widths'])

for j in range(0,len(df)):

    # yyy=Stim_cell_z_score_trials[trial_stim_types[i]][j]
    yyy=df.iloc[j][0]
    yyy=yyy[:,0:380]
    width_trace=np.full([len(yyy),1],float('nan'))
    peak_trace=np.full([len(yyy),1],float('nan'))
    peak_times_trace=np.full([len(yyy)],float('nan'))
    
    
    for k in range(0,len(yyy)):
        x=yyy[k,0:-1]
        x = np.nan_to_num(x, nan=0)
        mov_mean_array=(np.convolve(x, np.ones(10), 'valid') / 10)
        # peaks, properties = find_peaks(mov_mean_array, prominence=1.96,distance=len(mov_mean_array)-1, width=5)
        # peaks, properties = find_peaks(mov_mean_array, prominence=1.96,distance=len(mov_mean_array)/3, width=5)
        # properties["prominences"], properties["widths"]
        
        #OK the problem with Peaks is that the output for peak width is nonsense, the problem with prominences is it cnat accurately find early peaks
        
        # peaks, properties = find_peaks(mov_mean_array, prominence=1.96,distance=len(mov_mean_array)/100, width=0)
        # properties["prominences"], properties["widths"]
        peaks, properties = find_peaks(mov_mean_array, height=1.96,distance=len(mov_mean_array)/100, width=1)
        properties["peak_heights"], properties["widths"]

        # index=np.where(peaks>300)[0][0]
        # peaks=peaks+100
        
        if len(np.where(peaks>99)[0])>0:
            # peak_times_trace[k]=peaks[0]
            index=np.where(peaks>99)[0][0]
            peak_times_trace[k]=peaks[index]
            peak_trace[k] = mov_mean_array[peaks][index]
            width_trace[k]=properties['widths'][index]        
            
    Stim_cell_mean.loc[j, "mean_trace"] = np.mean(yyy,axis=0)
    x=np.mean(yyy,axis=0)
    x = np.nan_to_num(x, nan=0)
    mov_mean_array=(np.convolve(x, np.ones(5), 'valid') / 5)
    peaks, properties = find_peaks(mov_mean_array, height=1.96, width=0)
    if len(np.where(peaks>99)[0])>0:
        index=np.where(peaks>99)[0][0]
        Stim_cell_mean_peak.loc[j, "mean_trace"] = mov_mean_array[peaks][index]
    
    
    Stim_cell_peaks_trials.loc[j, "Peaks"] = peak_trace
    Stim_cell_peaktimes_trials.loc[j, "PeakTimes"] = peak_times_trace
    Stim_cell_peaks_width_trials.loc[j, "Widths"] = width_trace
    
Stim_cell_peaks_trials=Stim_cell_peaks_trials.T.reset_index(drop=True).T
Stim_cell_peaktimes_trials=Stim_cell_peaktimes_trials.T.reset_index(drop=True).T
Stim_cell_peaks_width_trials=Stim_cell_peaks_width_trials.T.reset_index(drop=True).T

# %%

def mean_excluding_nan(arr):
    return np.nanmean(arr)

def std_excluding_nan(arr):
    return np.nanstd(arr)

def count_excluding_nan(arr):
    return np.count_nonzero(~np.isnan(arr))/len(arr)

def proportion_above_threshold(array, threshold):
    return np.sum(array > threshold) / len(array)

# Apply the mean_excluding_nan function element-wise
Stim_cell_peaktimes_trials_mean = Stim_cell_peaktimes_trials.applymap(mean_excluding_nan)
Stim_cell_peaktimes_trials_std = Stim_cell_peaktimes_trials.applymap(std_excluding_nan)

Stim_cell_peaks_trials_mean = Stim_cell_peaks_trials.applymap(mean_excluding_nan)
Stim_cell_peaks_trials_std = Stim_cell_peaks_trials.applymap(std_excluding_nan)

Stim_cell_peaks_trials_count = Stim_cell_peaks_trials.applymap(count_excluding_nan)

Stim_cell_peaktimes_trials_mean=Stim_cell_peaktimes_trials_mean.T.reset_index(drop=True).T
Stim_cell_peaktimes_trials=Stim_cell_peaktimes_trials.T.reset_index(drop=True).T
Stim_cell_peaktimes_abs_diff=abs(Stim_cell_peaktimes_trials-Stim_cell_peaktimes_trials_mean)

threshold=15
Stim_cell_peaktimes_abs_diff_count=Stim_cell_peaktimes_abs_diff.applymap(lambda x: proportion_above_threshold(x, threshold))

# %%

response_proportion=0.5
limit=1.96
peaktime_deviation=45
peaktime_min=99

peaktime_max=300
# ccc = pd.DataFrame(np.where(Stim_cell_peaks_trials_mean < limit, float("nan"), Stim_cell_peaks_trials_mean))

ccc = pd.DataFrame(np.where(Stim_cell_mean_peak < limit, float("nan"), Stim_cell_mean_peak))

ccc = ccc.where(Stim_cell_peaks_trials_count > response_proportion, float("nan"))   # reminder that the arrow signs for where(pd) and np.where are reversed in relation to each other

# ccc = ccc.where((Stim_cell_peaktimes_trials_mean < peaktime_max)&(Stim_cell_peaktimes_trials_mean > peaktime_min), float("nan"))

# ccc = ccc.where(dist_opto > distance_cutoff, float("nan"))



#  use one or the other!!!
# ccc = ccc.where(Stim_cell_peaktimes_trials_std[0] < peaktime_deviation, float("nan"))

ccc = ccc.where(Stim_cell_peaktimes_abs_diff_count[0] < response_proportion, float("nan"))

# 


# ccc = ccc.where(dist_opto > distance_cutoff, float("nan"))
# ccc=pd.DataFrame(ccc)
# %%
Stim_cell_mean=Stim_cell_mean.T.reset_index(drop=True).T

conditional_zscores=Stim_cell_mean.where(~ccc.isnull(),float("nan"))

 
conditional_zscores_no_nan=pd.DataFrame(conditional_zscores.stack().to_numpy()).dropna()

 
conditional_zscores_no_nan_df=conditional_zscores_no_nan.explode(0).to_numpy()


conditional_zscores_no_nan_df=np.reshape(conditional_zscores_no_nan_df, (len(conditional_zscores_no_nan),len(conditional_zscores_no_nan.iloc[0][0]))).astype('float32')

conditional_zscores_no_nan_df=pd.DataFrame(conditional_zscores_no_nan_df)
 # conditional_zscores_no_nan_df=conditional_zscores_no_nan_df[:,301:-1]
 
 
conditional_zscores_no_nan_df_max=conditional_zscores_no_nan_df.iloc[conditional_zscores_no_nan_df.idxmax(axis=1).argsort()]
conditional_zscores_no_nan_df_max_time=conditional_zscores_no_nan_df.idxmax(axis=1).to_numpy()

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

# plot_heatmap(Stim_cell_mean, vmin=-2, vmax=2, start_index=1, sampling_interval=0.032958316, exclude_window=None)
plot_heatmap(conditional_zscores_no_nan_df_max, vmin=-2, vmax=2, start_index=1, sampling_interval=0.032958316, exclude_window=None)