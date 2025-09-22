# Canton-JoshEtAl2025
Scripts and functions used to generate the figures in Canton-Josh et al., "Network dynamics underlying activity-timescale differences between cortical regions"

This repository provides analysis and plotting tools for studying spontaneous and evoked neural activity in two-photon calcium imaging experiments. It integrates with a DataJoint database for retrieving experimental data and includes functions for PCA-based trajectory analyses and visualization of response dynamics.

---

# Analysis Pipeline Documentation

## Analysis Workflow

### Typical Usage Pattern:
1. **Start by running _initial_step_CantonJosh_25_Create_load_dataframes_for_all_subsequent_analysis.py
    this either loads data from datajoint or loads from JOBLIB saved files previously saved
    dj load is slowish but only needs to be done once and will automatically save JOBLIB files

2. **.py files labeled with "Figure_x_xxxx" format are those used in figure creation for the paper
    in several cases there will be futher processing of the data to create pseudo trials etc.
    this can be slow but once _initial_step_CantonJosh_25_Create_load_dataframes_for_all_subsequent_analysis.py
    you run each "Figure_x_xxxx.py" as an individual entity
3. ** below is a list of support code and descriptions. These are called by the "Figure_x_xxxx.py" files






## Core Analysis Modules


### **`_initial_step_CantonJosh_25_Create_load_dataframes_for_all_subsequent_analysis.py`
**Data Loading and Processing Pipeline**
Initial data processing script that creates and loads dataframes for all subsequent analysis:
- **Data Fetching**: Loads single-trial calcium imaging data from database or cache
- **Response Processing**: Processes neural response matrices and identifies responsive cells
- **Quality Filtering**: Applies filtering criteria for response reliability and timing consistency
- **Metadata Integration**: Adds cell properties (SNR, timescales, stimulation distance)
- **Multi-condition Analysis**: Processes data across brain areas (M2, V1) and experiment types
- **Caching System**: Implements efficient data caching for faster reprocessing
- **Summary Statistics**: Generates session-level responsive cell proportions

Key functions:
- `fetch_or_load_single_trial_data()` - Load cached data or fetch from database
- `process_and_summarize_single_trial_data()` - Main processing pipeline
- `add_roi_metadata_individual_fetches()` - Add cell metadata properties
- `summarize_responsive_proportion()` - Calculate responsive cell statistics
- `save_results_bundle()` / `load_results_bundle()` - Data caching utilities


### `analyzeEvoked2P.py`
**Main evoked response analysis pipeline**

Core functionality:
- **Response detection**: Automated detection of calcium transients using peak detection, center-of-mass, and area-under-curve metrics
- **Cross-validation**: Trial-splitting analysis to assess response timing reliability
- **Spatial analysis**: Distance-dependent response probability calculations
- **PCA analysis**: Principal component analysis of trial-to-trial variability
- **FOV visualization**: Field-of-view heatmaps showing response properties
- **Statistical comparisons**: Between-area comparisons (V1 vs M2) with appropriate statistical tests

Key functions:
- `get_avg_trig_responses()`: Extract trial-averaged responses across experiments
- `get_full_resp_stats()`: Comprehensive response statistics including timing and amplitude
- `compare_response_stats()`: Statistical comparison between brain areas
- `xval_trial_data()`: Cross-validation of response timing across trial splits
- `plot_response_grand_average()`: Visualization of population responses

### `Canton_Josh_et_al_2025_analysis_plotting_functions.py`
**Specialized plotting and visualization functions**

Features:
- **Multi-trace plotting**: Functions for overlaying multiple experimental conditions
- **Statistical visualization**: Bar plots with individual data points, error bars, and significance testing
- **Distance-response analysis**: Spatial plotting of response properties vs distance from stimulation
- **Normalization utilities**: Various normalization schemes (min-max, z-score, peak normalization)
- **Custom statistical tests**: Integration of Mann-Whitney U, t-tests, and ANOVA with visualization

Key plotting functions:
- `plot_mean_trace_multiple_dataframe_input()`: Multi-condition trace plotting
- `bar_plot_two_dfs_with_lines_and_median()`: Comparative bar plots with statistical annotation
- `plot_ecdf_comparison()`: Empirical cumulative distribution function comparisons
- `prepare_and_plot_heatmap()`: Sortable heatmap visualization

### `Cross_Correlation_functions.py`
**Cross-correlation and temporal relationship analysis**

Capabilities:
- **Trial-to-trial correlation**: Normalized cross-correlation between individual trials
- **Mean-vs-trial analysis**: Cross-correlation of individual trials against population average
- **Shuffle controls**: Generation of null distributions through data scrambling
- **Lag analysis**: Quantification of temporal offsets in neural responses
- **Reliability metrics**: Assessment of response consistency across trials

Key functions:
- `cross_correlation_trial_by_trial()`: Pairwise trial correlations
- `cross_correlation_with_mean_random()`: Random subsampling cross-correlation
- `crossval_peak_timing_shuffle()`: Cross-validation of peak timing with shuffling controls

### `General_functions_Calculate_responsive_cells_create_pseudo_trial_arrays.py`
**Data processing and pseudo-trial generation utilities**

Core utilities:
- **Response classification**: Automated detection of responsive cells using statistical thresholds
- **Pseudo-trial generation**: Creation of control datasets from baseline periods
- **Feature extraction**: Comprehensive extraction of response features (peak amplitude, timing, width)
- **Quality control**: Filtering based on response reliability and signal quality
- **Data organization**: Grouping and aggregation functions for multi-level analysis

Key functions:
- `process_and_filter_response_matrix()`: End-to-end processing pipeline from raw trials to filtered response matrix
- `generate_pseudo_trials()`: Generate control datasets from spontaneous activity periods  
- `group_peaks_by_roi_and_stim()`: Organize responses by cell and stimulation condition
- `filter_and_summarize()`: Apply sequential filters with logging


### `Canton_Josh_et_al_2025_PCA_functions.py`
**Principal Component Analysis Functions**

Contains functions for analyzing neural population dynamics using PCA:

- **Distance Calculations**: Functions to compute 3D transition distances between baseline and response periods
- **PCA Projections**: Tools for analyzing trial projections in PCA space
- **Trajectory Analysis**: Functions for path length, curvature, and centroid distance calculations
- **Statistical Analysis**: Pairwise distance comparisons and correlation analysis
- **Visualization**: Plotting functions for PCA results including histograms and scatter plots

Key functions:
- `compute_3d_transition_distances()` - Calculates distances between baseline and response periods
- `compute_centroid_distances()` - Analyzes centroid positions and distances
- `compute_pairwise_pointwise_distances()` - Pairwise comparisons between trials
- Various plotting utilities for visualizing PCA results

### `Cross_Correlation_functions.py`
**Cross-Correlation Analysis Tools**

Specialized functions for cross-correlation analysis of neural time series:

- **Cross-Correlation Calculations**: Normalized cross-correlation between signals
- **Trial-by-Trial Analysis**: Functions for analyzing correlations across individual trials
- **Peak Timing Analysis**: Cross-validation of peak timing measurements
- **Shuffling Controls**: Statistical controls using data scrambling
- **Autocorrelation**: Functions for calculating and fitting autocorrelation data

Key functions:
- `calculate_cross_correlation()` - Basic cross-correlation between matrices
- `cross_correlation_with_mean_random()` - Cross-correlation with randomized trial groupings
- `crossval_peak_timing_shuffle()` - Cross-validation of timing measurements
- `normalize_cross_correlation()` - Normalized cross-correlation computation

### `analyzeSpont2P.py`
**Spontaneous Activity Analysis**

Comprehensive module for analyzing spontaneous neural activity from two-photon imaging:

- **Database Integration**: Functions for querying DataJoint databases
- **Timescale Analysis**: Extraction and analysis of neural timescales
- **Spatial Clustering**: Analysis of spatial organization of timescales
- **Exponential Fitting**: Single and double exponential fitting to autocorrelation data
- **Statistical Comparisons**: Cross-area comparisons and statistical testing
- **Visualization**: Plotting functions for timescale distributions and spatial maps

Key functions:
- `get_all_tau()` - Retrieve timescales for a brain area
- `clustering_by_tau()` - Spatial clustering analysis
- `fit_exponentials_from_df()` - Fit exponential models to autocorrelation data
- `plot_area_tau_comp()` - Compare timescales across brain areas
- `extract_fit_data_for_keys_df()` - Extract fitting parameters from database

### `Canton_Josh_et_al_2025_analysis_plotting_functions.py`
**Analysis and Plotting Utilities**

Comprehensive collection of plotting and analysis functions:

- **Trace Visualization**: Functions for plotting mean traces with error bars
- **Heatmaps**: Neural activity heatmap generation with customizable parameters
- **Statistical Plots**: Scatter plots with regression analysis and statistical testing
- **Data Normalization**: Various normalization methods for neural data
- **Comparative Analysis**: Functions for comparing multiple datasets
- **Distance Analysis**: Plotting functions for spatial distance relationships
- **ANOVA Analysis**: Two-way repeated measures ANOVA implementations

Key functions:
- `plot_mean_trace_multiple_dataframe_input()` - Plot multiple traces with error bars
- `prepare_and_plot_heatmap()` - Generate aligned neural activity heatmaps
- `bar_plot_two_dfs_with_lines_and_median()` - Compare two datasets with statistics
- `plot_ecdf_comparison()` - Empirical cumulative distribution comparisons
- `general_scatter_plot()` - Flexible scatter plotting with regression analysis



