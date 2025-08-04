# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 22:59:20 2025

@author: jec822
"""
import analyzeEvoked2P
import Canton_Josh_et_al_2025_analysis_plotting_functions as analysis_plotting_functions


# %%
trial_data = get_single_trial_data(area='M2', expt_type='standard',
params=params,
resp_type='dff', 
signif_only=True, 
which_neurons='non_stimd', 
relax_timing_criteria=False
)

# %% Using the DJ pipeline

import numpy as np
import os
from copy import deepcopy
from scipy.signal import medfilt
import matplotlib.pyplot as plt

# Set parameters
apply_medfilt = False  # Optional: toggle median filter
medfilt_kernel = 3     # Optional: kernel size
figure_save_path = "figures/responsive_cells_roi_traces_dj_pipelin_V1"
os.makedirs(figure_save_path, exist_ok=True)

unique_rois = list(np.unique(trial_data['roi_ids']))

for roi in unique_rois:
    ridx = trial_data['roi_ids'] == roi

    these_trials = np.array(trial_data['trial_ids'])[ridx]
    these_stims = np.array(trial_data['stim_ids'])[ridx]
    coms = np.array(trial_data['com_sec'])[ridx]
    peaks = np.array(trial_data['peak_or_trough_time_sec'])[ridx]
    t_axes = np.array(trial_data['time_axis_sec'], dtype=object)[ridx]
    roi_keys = list(np.array(trial_data['roi_keys'])[ridx])

    # Extract and truncate all trials
    trials = trial_data['trig_dff_trials']
    axes = trial_data['time_axis_sec']
    min_len = min([len(trial) for trial in trials])
    a = np.array([trial[:min_len] for trial in trials])
    axes_temp = np.array([ax[:min_len] for ax in axes])
    
    trial_dffs_all = list(a)
    t_axes = list(axes_temp)
    unique_stims = np.unique(these_stims)

    for stim in unique_stims:
        sidx = trial_data['stim_ids'] == stim
        tidx = np.where(ridx & sidx)[0]
        tidx_shuff = deepcopy(tidx)
        ntrials = len(tidx)
        
        if ntrials == 0:
            continue
        
        taxis = t_axes[tidx[0]]
        frame_int = np.diff(taxis)[0]
        dffs = [trial_dffs_all[i] for i in tidx]

        if apply_medfilt:
            dffs = [medfilt(trial, kernel_size=medfilt_kernel) for trial in dffs]

        dffs = [np.expand_dims(trial, axis=0) for trial in dffs]

        roi_meta = roi_keys[0]
        subj = roi_meta.get('subject_fullname', 'unknown_subj')
        sess = roi_meta.get('session_date', 'unknown_date')
        roi_id = roi_meta.get('roi_id', 'unknown_roi')
        stim_id = stim if isinstance(stim, str) else str(stim)

        title = f"{subj} | {sess} | ROI {roi_id} | Stim {stim_id}"
        filename_base = f"{subj}_{sess}_roi{roi_id}_stim{stim_id}"
        full_png_path = os.path.join(figure_save_path, f"{filename_base}.png")
        full_svg_path = os.path.join(figure_save_path, f"{filename_base}.svg")

        # Call your plotting function
        fig = analysis_plotting_functions.plot_nonoverlapping_traces(
            dffs,
            sampling_rate=30,
            apply_medfilt=False,  # Already applied above
            medfilt_kernel=9,
            offset=10,
            color=opto_params['general_params']['M2_cl'],
            alpha=1.0,
            cutoff=None,
            save_path=None,
            y_scale_bar=True,
            scale_bar_length=1.0,
            scale_bar_label='z-score',
            zscore_traces=False,
            title=title  # Optional: your plotting function must support this
        )

        if fig is not None:
            fig.savefig(full_png_path, dpi=300)
            fig.savefig(full_svg_path)
            plt.close(fig)

# %% using the new pipeline i created in july 2025

results_df=result_M2_high_trial_count_non_stimd_filt
# results_df=result_M2_filt


# %%
import numpy as np
import os
from copy import deepcopy
from scipy.signal import medfilt
import matplotlib.pyplot as plt

# Set parameters
apply_medfilt = True
medfilt_kernel =7
figure_save_path = "figures/responsive_cells_roi_traces_medfilt_7_high_trial_count_2.96_15_peak_width_offset"
os.makedirs(figure_save_path, exist_ok=True)

for idx, row in results_df.iterrows():
    
    roi = row['roi_keys']
    dffs = row['trials_arrays']
    # peak_times=row['peak_times_by_trial']+(100*0.032958316)
    peak_times = [pt + 100 * 0.032958316 if not np.isnan(pt) else np.nan for pt in row['peak_time_by_trial']]

    # Truncate to minimum length
    min_len = min(len(trace) for trace in dffs)
    dffs = [np.expand_dims(trace[:min_len], axis=0) for trace in dffs]

    # Safely extract metadata
    roi_meta = roi[0]
    subj = roi_meta.get('subject_fullname', f'unknownsubj_{idx}')
    sess = roi_meta.get('session_date', 'unknowndate')
    roi_id = roi_meta.get('roi_id', f'roi{idx}')
    stim_id = row.get('stim_id', f'stim{idx}')  # ← fallback if stim_id missing

    # Build title and filenames
    title = f"{subj} | {sess} | ROI {roi_id} | Stim {stim_id}"
    filename_base = f"{subj}_{sess}_roi{roi_id}_stim{stim_id}_{idx}"
    full_png_path = os.path.join(figure_save_path, f"{filename_base}.png")
    full_svg_path = os.path.join(figure_save_path, f"{filename_base}.svg")
    
    # Plot
    fig = analysis_plotting_functions.plot_nonoverlapping_traces(
        dffs,
        sampling_rate=1/0.032958316,
        apply_medfilt=apply_medfilt,
        medfilt_kernel=medfilt_kernel,
        offset=5,
        color=opto_params['general_params']['M2_cl'],
        alpha=1.0,
        cutoff=None,
        save_path=None,
        y_scale_bar=True,
        scale_bar_length=1.0,
        scale_bar_label='z-score',
        zscore_traces=False,
        title=title,
        peak_times=peak_times
    )

    if fig is not None:
        fig.savefig(full_png_path, dpi=300)
        fig.savefig(full_svg_path)
        plt.close(fig)
