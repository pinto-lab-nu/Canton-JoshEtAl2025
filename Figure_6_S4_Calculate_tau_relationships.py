# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 18:05:53 2025

@author: jec822
"""
# %% add the tau stim to
import pandas as pd

def add_tau_column(df_source, df_target, group_cols, tau_col='tau', new_col_name='tau_stim'):
    """
    Merge tau values from df_source into df_target based on shared group columns.

    Parameters:
    - df_source: DataFrame containing the original tau values.
    - df_target: DataFrame to which the tau values will be added.
    - group_cols: List of columns to merge on.
    - tau_col: Name of the tau column in df_source.
    - new_col_name: Name to use for the tau column in df_target.

    Returns:
    - df_target with new column added.
    """
    tau_df = df_source[group_cols + [tau_col]].rename(columns={tau_col: new_col_name})
    return df_target.merge(tau_df, on=group_cols, how='left')
# %%
def add_columns(df_source, df_target, group_cols, col_map):
    """
    Merge multiple tau columns from df_source into df_target based on shared group columns.

    Parameters:
    - df_source: DataFrame containing the original tau values.
    - df_target: DataFrame to which the tau values will be added.
    - group_cols: List of columns to merge on.
    - col_map: Dictionary mapping original column names in df_source to new column names in df_target.

    Returns:
    - df_target with new columns added.
    """
    tau_df = df_source[group_cols + list(col_map.keys())].copy()
    tau_df = tau_df.rename(columns=col_map)
    return df_target.merge(tau_df, on=group_cols, how='left')


# %%
def categorize_and_normalize(df_with_tau, df_rois, group_cols, tau_col='tau', tau_stim_col='tau_stim', roi_col='cells_per_roi'):
    """
    Categorizes each row in df_with_tau into one of four median-based categories based on tau and tau_stim,
    counts the number of rows in each category per group, and normalizes by the number_of_rois from df_rois.

    Returns:
        DataFrame with counts and normalized counts for each category.
    """

    # Step 1: Compute global medians
    tau_median = df_with_tau[tau_col].median()
    tau_stim_median = df_with_tau[tau_stim_col].median()

    # Step 2: Categorize each row
    def categorize(row):
        tau_cat = 'Short' if row[tau_col] <= tau_median else 'Long'
        stim_cat = 'Short' if row[tau_stim_col] < tau_stim_median else 'Long'
        return f"{stim_cat} tau stim & {tau_cat} tau"

    df_with_tau['category'] = df_with_tau.apply(categorize, axis=1)

    # Step 3: Count rows in each category per group
    category_counts = df_with_tau.groupby(group_cols + ['category']).size().reset_index(name='count')

    # Step 4: Merge with number_of_rois
    merged = category_counts.merge(df_rois[group_cols + [roi_col]], on=group_cols, how='left')

    # Step 5: Normalize count by number_of_rois
    merged['normalized_count'] = merged['count'] / merged[roi_col]

    return merged
# %%

def categorize_and_normalize_from_population_medians(
    df_with_tau,
    df_rois,
    df_population,
    group_cols,
    tau_col='tau',
    tau_stim_col='tau_stim',
    roi_col='cells_per_roi'
):
    """
    Categorizes each row in df_with_tau using global medians from df_population,
    counts the number of rows in each category per group, and normalizes by ROI counts.

    Args:
        df_with_tau: DataFrame with tau and tau_stim for a subset of cells.
        df_rois: DataFrame with ROI counts per group.
        df_population: DataFrame of the full population to derive medians from.
        group_cols: List of column names to group by.
        tau_col: Name of column containing tau values.
        tau_stim_col: Name of column containing stim tau values.
        roi_col: Name of column with ROI counts.

    Returns:
        DataFrame with counts and normalized counts per category and group.
    """

    # Step 1: Compute global medians from the full population
    tau_median = df_population[tau_col].median()
    tau_stim_median = df_population[tau_col].median()  # same as tau_median

    # Step 2: Categorize each row in df_with_tau
    def categorize(row):
        tau_cat = 'Short' if row[tau_col] <= tau_median else 'Long'
        stim_cat = 'Short' if row[tau_stim_col] < tau_stim_median else 'Long'
        return f"{stim_cat} tau stim & {tau_cat} tau"

    df_with_tau = df_with_tau.copy()
    df_with_tau['category'] = df_with_tau.apply(categorize, axis=1)

    # Step 3: Count cells per category per group
    category_counts = df_with_tau.groupby(group_cols + ['category']).size().reset_index(name='count')

    # Step 4: Merge with ROI counts
    merged = category_counts.merge(df_rois[group_cols + [roi_col]], on=group_cols, how='left')

    # Step 5: Normalize
    merged['normalized_count'] = merged['count'] / merged[roi_col]

    return merged
# %%

def categorize_and_normalize_from_population_medians_2(
    df_with_tau,
    df_rois,
    df_population,
    group_cols,
    tau_col='tau',
    tau_stim_col='tau_stim',
    roi_col='cells_per_roi',
    denominator='total_in_tau_range'
):
    """
    Categorizes each row in df_with_tau using global medians from df_population,
    counts the number of rows in each category per group, and normalizes by ROI counts.
    Also reports if a zero count is due to no cells in tau range or no responders.
    """

    # Step 1: Compute global medians from the full population
    tau_median = df_population[tau_col].median()
    tau_stim_median = df_population[tau_col].median()  # same as tau_median

    # Step 2: Function to categorize rows
    def categorize(row):
        tau_cat = 'Short' if row[tau_col] <= tau_median else 'Long'
        stim_cat = 'Short' if row[tau_stim_col] <= tau_stim_median else 'Long'
        return f"{stim_cat} tau stim & {tau_cat} tau"

    # Step 3: Categorize both datasets
    df_with_tau = df_with_tau.copy()
    df_population = df_population.copy()

    df_with_tau['category'] = df_with_tau.apply(categorize, axis=1)
    df_population['category'] = df_population.apply(categorize, axis=1)

    # Step 4: Count responsive cells (subset) per group/category
    responsive_counts = (
        df_with_tau.groupby(group_cols + ['category'])
        .size()
        .reset_index(name='responsive_count')
    )

    # Step 5: Count total cells in population per group/category
    population_counts = (
        df_population.groupby(group_cols + ['category'])
        .size()
        .reset_index(name='total_in_tau_range')
    )

    # Step 6: Merge counts together
    merged = population_counts.merge(
        responsive_counts, on=group_cols + ['category'], how='left'
    )
    merged['responsive_count'] = merged['responsive_count'].fillna(0)

    # Step 7: Merge ROI counts
    merged = merged.merge(df_rois[group_cols + [roi_col]], on=group_cols, how='left')

    # Step 8: Normalize responsive counts by ROI counts
    merged['normalized_count'] = merged['responsive_count'] / merged[denominator]

    # Step 9: Reason for zero
    def reason(row):
        if row['responsive_count'] == 0:
            if row['total_in_tau_range'] == 0:
                return 'no cells in tau range'
            else:
                return 'cells in range but no responders'
        return 'has responders'

    merged['zero_reason'] = merged.apply(reason, axis=1)

    return merged

# %%

def categorize_and_normalize_by_population_range(
    df_with_tau,
    df_population,
    group_cols,
    tau_col='tau',
    tau_stim_col='tau_stim'
):
    """
    Categorizes each row in df_with_tau using global medians from df_population,
    counts the number of rows in each category per group, and normalizes by the
    number of population cells in the same tau category and group.

    Args:
        df_with_tau: DataFrame with tau and tau_stim for a subset of cells.
        df_population: DataFrame of the full population to derive medians from and category sizes.
        group_cols: List of column names to group by.
        tau_col: Name of column containing tau values.
        tau_stim_col: Name of column containing stim tau values.

    Returns:
        DataFrame with counts and normalized counts per category and group,
        normalized by the number of cells in the population in that category.
    """

    # Step 1: Compute global medians from the full population
    tau_median = df_population[tau_col].median()
    tau_stim_median = df_population[tau_col].median()  # same as tau_median

    # Step 2: Function to categorize rows
    def categorize(row):
        tau_cat = 'Short' if row[tau_col] <= tau_median else 'Long'
        stim_cat = 'Short' if row[tau_stim_col] < tau_stim_median else 'Long'
        return f"{stim_cat} tau stim & {tau_cat} tau"

    # Step 3: Categorize both datasets
    df_with_tau = df_with_tau.copy()
    df_population = df_population.copy()

    df_with_tau['category'] = df_with_tau.apply(categorize, axis=1)
    df_population['category'] = df_population.apply(categorize, axis=1)

    # Step 4: Count responsive cells (subset) per group/category
    responsive_counts = (
        df_with_tau.groupby(group_cols + ['category'])
        .size()
        .reset_index(name='responsive_count')
    )

    # Step 5: Count total cells in population per group/category
    population_counts = (
        df_population.groupby(group_cols + ['category'])
        .size()
        .reset_index(name='population_count')
    )

    # Step 6: Merge counts together
    merged = population_counts.merge(
        responsive_counts, on=group_cols + ['category'], how='left'
    )
    merged['responsive_count'] = merged['responsive_count'].fillna(0)

    # Step 7: Normalize by population category count
    merged['normalized_count'] = merged['responsive_count'] / merged['population_count']

    return merged

# %%
def categorize_and_normalize_per_experiment(
    df_with_tau,
    df_rois,
    df_population,
    group_cols,
    tau_col='tau',
    tau_stim_col='tau_stim',
    roi_col='cells_per_roi'
):
    """
    Categorizes each row in df_with_tau into one of four categories based on per-experiment median taus
    from df_population (not just the responsive subset). Uses the same tau median for both tau and tau_stim.

    Arguments:
        df_with_tau: DataFrame of responsive cells with tau and tau_stim
        df_rois: DataFrame with group_cols + [roi_col] for normalization
        df_population: DataFrame with the full set of potentially responsive cells
        group_cols: list of columns to group by (e.g., ['experiment_id'])
        tau_col: column name of tau (used for both tau and tau_stim)
        tau_stim_col: column name of tau_stim in df_with_tau
        roi_col: column name indicating total cells per group (for normalization)

    Returns:
        Merged DataFrame with raw and normalized counts per tau category per group.
    """

    # Step 1: Compute per-group median tau from full population
    tau_medians = df_population.groupby(group_cols)[tau_col].median().reset_index()
    tau_medians = tau_medians.rename(columns={tau_col: 'tau_median'})
    # Duplicate to use the same for tau_stim median
    tau_medians['tau_stim_median'] = tau_medians['tau_median']

    # Step 2: Merge medians into df_with_tau
    df = df_with_tau.merge(tau_medians, on=group_cols, how='left')

    # Step 3: Categorize each row
    def categorize(row):
        tau_cat = 'Short' if row[tau_col] <= row['tau_median'] else 'Long'
        stim_cat = 'Short' if row[tau_stim_col] <= row['tau_stim_median'] else 'Long'
        return f"{stim_cat} tau stim & {tau_cat} tau"

    df['category'] = df.apply(categorize, axis=1)

    # Step 4: Count rows in each category per group
    category_counts = df.groupby(group_cols + ['category']).size().reset_index(name='count')

    # Step 5: Merge with ROI data
    merged = category_counts.merge(df_rois[group_cols + [roi_col]], on=group_cols, how='left')

    # Step 6: Normalize by number of ROIs
    merged['normalized_count'] = merged['count'] / merged[roi_col]

    return merged



# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def plot_2x2_category_heatmap_from_pivot(
    plot_df,
    cbar_label='Mean Normalized Count',
    title='2x2 Category Heatmap',
    vmin=None,
    vmax=None,
    mean_or_median='median'
):
    """
    Plot a 2x2 heatmap from a pivoted DataFrame with categories as columns.
    Runs a Chi-squared test on the raw counts (summed across groups).
    
    Args:
        plot_df: DataFrame indexed by groups (can be multiple),
                 columns are categories like:
                 'Short tau stim & Short tau', 'Short tau stim & Long tau',
                 'Long tau stim & Short tau', 'Long tau stim & Long tau'
                 Values should be counts per group.
        cbar_label: Optional label for the colorbar.
        title: Optional plot title.
        vmin, vmax: Optional min and max values for heatmap color scale.
        mean_or_median: 'mean' or 'median' to aggregate values for plotting.
    """
    # Define the category order
    category_order = [
        'Short tau stim & Short tau',   # top-left
        'Short tau stim & Long tau',    # top-right
        'Long tau stim & Short tau',    # bottom-left
        'Long tau stim & Long tau'      # bottom-right
    ]
    
    # Ensure all categories exist; fill missing with zeros
    for cat in category_order:
        if cat not in plot_df.columns:
            plot_df[cat] = 0
    
    # ---------------------------
    # Build raw contingency table
    # ---------------------------
    raw_counts = plot_df[category_order].sum(axis=0).values  # total counts
    contingency_table = np.array([
        [raw_counts[0], raw_counts[1]],
        [raw_counts[2], raw_counts[3]]
    ])
    
    # Run Chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print("Chi-squared test on raw counts:")
    print(f"  Chi2 statistic = {chi2:.4f}")
    print(f"  p-value        = {p:.4e}")
    print(f"  Degrees of freedom = {dof}")
    print("  Expected frequencies:")
    print(expected)
    
    # ---------------------------
    # For plotting: aggregate with median/mean
    # ---------------------------
    if mean_or_median == 'median':
        values = plot_df[category_order].replace(0, np.nan).median(axis=0, skipna=True).values
    else:
        values = plot_df[category_order].mean(axis=0).values
    
    heatmap_data = np.array([
        [values[0], values[1]],
        [values[2], values[3]]
    ])
    
    # Plot heatmap
    plt.imshow(heatmap_data, cmap='gray', interpolation='nearest', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    
    plt.xticks([0,1], ['Short tau', 'Long tau'])
    plt.yticks([0,1], ['Short tau stim', 'Long tau stim'])
    plt.xlabel('tau')
    plt.ylabel('tau stim')
    plt.title(title)
    plt.show()


# %%
def process_roi_keys_and_add_fits(result_df, group_cols, col_map):
    import numpy as np
    from schemas import spont_timescales
    import analyzeSpont2P

    # Extract ROI keys
    keys = result_df['roi_keys'].reset_index(drop=True)

    # Fetch matching keys from spont_timescales
    k_fetched = (spont_timescales.TwopAutocorr & keys).fetch('KEY')

    # Extract fit data
    df_fits = analyzeSpont2P.extract_fit_data_for_keys_df(k_fetched)

    # Add acorr[1] index value
    df_fits['acorr_1_index'] = [
        vec[1] if isinstance(vec, (list, np.ndarray)) and len(vec) > 1 else np.nan
        for vec in df_fits['autocorr_vals']
    ]

    # Merge back into original DataFrame
    return add_columns(df_fits, result_df, group_cols, col_map)
# %%

def filter_fit_results(df,
                       tau_min=None, tau_max=None,
                       acorr_min=None,
                       r2_d_min=None, r2_s_min=None,
                       is_good_tau_min=None):
    """
    Filters a fit results DataFrame based on tau, acorr_1_index, is_good_tau,
    and optional r2 thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing fit results.
    tau_min : float, optional
        Lower bound for tau (inclusive).
    tau_max : float, optional
        Upper bound for tau (inclusive).
    acorr_min : float, optional
        Lower bound for acorr_1_index (inclusive).
    r2_d_min : float, optional
        Minimum acceptable r2_d value.
    r2_s_min : float, optional
        Minimum acceptable r2_s value.
    is_good_tau_min : float, optional
        Minimum acceptable is_good_tau value.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    filtered = df.copy()

    # is_good_tau bound
    if is_good_tau_min is not None and 'is_good_tau' in filtered.columns:
        filtered = filtered[filtered['is_good_tau'] >= is_good_tau_min]

    # Tau bounds
    if tau_min is not None:
        filtered = filtered[filtered['tau'] >= tau_min]
    if tau_max is not None:
        filtered = filtered[filtered['tau'] <= tau_max]

    # acorr_1_index bound
    if acorr_min is not None:
        filtered = filtered[filtered['acorr_1_index'] >= acorr_min]

    # Optional R² cutoffs
    if r2_d_min is not None and 'r2_d' in filtered.columns:
        filtered = filtered[filtered['r2_d'] >= r2_d_min]
    if r2_s_min is not None and 'r2_s' in filtered.columns:
        filtered = filtered[filtered['r2_s'] >= r2_s_min]

    return filtered
# %%

def add_is_significant_column(df_population, df_significant, group_cols):
    """
    Adds a binary column 'is_significant' to df_population.
    'is_significant' is 1 if the same group_cols combination exists in df_significant, else 0.
    
    Args:
        df_population: DataFrame containing the full population.
        df_significant: DataFrame containing significant group combinations.
        group_cols: List of column names used to match rows.
        
    Returns:
        df_population with a new 'is_significant' column.
    """
    df_population = df_population.copy()

    # Create a set of tuples for quick lookup
    sig_groups = set(tuple(row) for row in df_significant[group_cols].drop_duplicates().values)
    
    # Check membership
    df_population['is_significant'] = df_population[group_cols].apply(
        lambda row: 1 if tuple(row) in sig_groups else 0,
        axis=1
    )
    
    return df_population
# %%
def add_cells_per_roi(df_population, df_rois, group_cols, roi_col='cells_per_roi'):
    """
    Adds a 'cells_per_roi' column to df_population by merging values from df_rois.
    
    Args:
        df_population: DataFrame to which the ROI counts will be added.
        df_rois: DataFrame containing ROI counts per group.
        group_cols: List of column names to use for matching rows.
        roi_col: Name of the ROI count column in df_rois.
    
    Returns:
        df_population with an added 'cells_per_roi' column.
    """
    df_population = df_population.copy()
    
    return df_population.merge(
        df_rois[group_cols + [roi_col]],
        on=group_cols,
        how='left'
    )

# %%
import numpy as np
import pandas as pd

def add_tau_medians_and_lengths_with_counts(df, group_cols, tau_col='tau', tau_stim_col='tau_stim'):
    """
    Adds tau/tau_stim medians, length categories, and a count of how many cells
    in each group (group_cols) match the same category.
    NaN values in tau or tau_stim produce NaN in the category columns.
    
    Args:
        df: DataFrame with tau and tau_stim columns.
        group_cols: List of columns to group by for counts.
        tau_col: Name of tau column.
        tau_stim_col: Name of tau_stim column.
    
    Returns:
        Updated DataFrame with new columns:
        - tau_median
        - tau_stim_median
        - tau_length ('Short'/'Long'/NaN)
        - tau_stim_length ('Short'/'Long'/NaN)
        - matching_cells_count (int)
    """
    df = df.copy()
    
    # Calculate medians ignoring NaNs
    tau_median = df[tau_col].median(skipna=True)
    tau_stim_median = df[tau_stim_col].median(skipna=True)
    
    # Add as constant columns
    df['tau_median'] = tau_median
    df['tau_stim_median'] = tau_stim_median
    
    # Categorize relative to median, with NaN handling
    def categorize(x, median):
        if pd.isna(x):
            return np.nan
        return 'Short' if x <= median else 'Long'
    
    df['tau_length'] = df[tau_col].apply(lambda x: categorize(x, tau_median))
    df['tau_stim_length'] = df[tau_stim_col].apply(lambda x: categorize(x, tau_stim_median))
    
    # Count how many cells in the same group have the same category
    df['matching_cells_count'] = (
        df.groupby(group_cols + ['tau_length', 'tau_stim_length'])[tau_col]
        .transform('count')
    )
    
    return df

    
# %%

import numpy as np
import pandas as pd

def add_tau_medians_lengths_counts_and_category(
    df, group_cols, tau_col='tau', tau_stim_col='tau_stim'
):
    """
    Adds tau/tau_stim medians, length categories, matching cell counts, 
    and the combined 'category' column.

    Args:
        df: DataFrame with tau and tau_stim columns.
        group_cols: List of columns to group by for counts.
        tau_col: Name of tau column.
        tau_stim_col: Name of tau_stim column.

    Returns:
        Updated DataFrame with new columns:
        - tau_median
        - tau_stim_median
        - tau_length ('Short'/'Long'/NaN)
        - tau_stim_length ('Short'/'Long'/NaN)
        - matching_cells_count (int)
        - category (string: "<stim length> tau stim & <tau length> tau")
    """
    df = df.copy()
    
    # Calculate medians ignoring NaNs
    tau_median = df[tau_col].median(skipna=True)
    tau_stim_median = df[tau_stim_col].median(skipna=True)
    
    # Add median columns
    df['tau_median'] = tau_median
    df['tau_stim_median'] = tau_stim_median
    
    # Categorize relative to median
    def categorize(x, median):
        if pd.isna(x):
            return np.nan
        return 'Short' if x <= median else 'Long'
    
    df['tau_length'] = df[tau_col].apply(lambda x: categorize(x, tau_median))
    df['tau_stim_length'] = df[tau_stim_col].apply(lambda x: categorize(x, tau_stim_median))
    
    # Count matching cells in same group + category combo
    df['matching_cells_count'] = (
        df.groupby(group_cols + ['tau_length', 'tau_stim_length'])[tau_col]
        .transform('count')
    )

    # Create category label
    df['category'] = df['tau_stim_length'] + ' tau stim & ' + df['tau_length'] + ' tau'

    # Define canonical category order for downstream use
    category_order = [
        'Short tau stim & Short tau',
        'Short tau stim & Long tau',
        'Long tau stim & Short tau',
        'Long tau stim & Long tau'
    ]
    
    # Optional: set category as Categorical for ordering
    df['category'] = pd.Categorical(df['category'], categories=category_order, ordered=True)
    
    return df

# %%

def proportion_significant_by_length(df, group_cols,divide_by='matching_cells_count'):
    """
    Calculates the proportion of significant cells in each (tau_length, tau_stim_length)
    category per group, dividing by matching_cells_count.

    Args:
        df: DataFrame containing at least:
            - group_cols
            - is_significant (0/1)
            - tau_length ('Short'/'Long'/NaN)
            - tau_stim_length ('Short'/'Long'/NaN)
            - matching_cells_count (int)
        group_cols: List of columns to group by.

    Returns:
        DataFrame with proportions for each category per group.
    """
    # Drop rows where tau_length or tau_stim_length is NaN
    df_valid = df.dropna(subset=['tau_length', 'tau_stim_length']).copy()

    # Create category label
    df_valid['category'] = (
        df_valid['tau_stim_length'] + ' tau stim & ' + df_valid['tau_length'] + ' tau'
    )

    # Get total matching_cells_count per group/category
    totals = (
        df_valid.groupby(group_cols + ['category'])[divide_by]
        .sum()
        .reset_index(name='total_cells')
    )

    # Get significant cell count per group/category
    sig_counts = (
        df_valid[df_valid['is_significant'] == 1]
        .groupby(group_cols + ['category'])
        .size()
        .reset_index(name='sig_count')
    )

    # Merge and compute proportions
    merged = totals.merge(sig_counts, on=group_cols + ['category'], how='left')
    merged['sig_count'] = merged['sig_count'].fillna(np.nan)
    merged['proportion'] = merged['sig_count'] / merged['total_cells']

    # Pivot to wide format
    category_order = [
        'Short tau stim & Short tau',
        'Short tau stim & Long tau',
        'Long tau stim & Short tau',
        'Long tau stim & Long tau'
    ]
    prop_pivot = merged.pivot_table(
        index=group_cols,
        columns='category',
        values='proportion',
        fill_value=np.nan
    ).reset_index()

    # Ensure all four categories are present
    for cat in category_order:
        if cat not in prop_pivot.columns:
            prop_pivot[cat] = np.nan

    # Reorder columns
    prop_pivot = prop_pivot[group_cols + category_order]

    return prop_pivot

# %%

def group_traces_by_tau_cutoffs(df, group_cols):
    """
    Groups 'averaged_traces_all' and 'time_axes' entries for significant cells (is_significant == 1)
    based on tau_length and tau_stim_length categories.
    """
    # Filter to significant only and valid tau categories
    df_valid = df[
        (df['is_significant'] == 1) &
        df['tau_length'].notna() &
        df['tau_stim_length'].notna()
    ].copy()

    # Create category label
    df_valid['category'] = (
        df_valid['tau_stim_length'] + ' tau stim & ' + df_valid['tau_length'] + ' tau'
    )

    # Group and flatten without nesting
    grouped = (
        df_valid.groupby(group_cols + ['category'])
        .agg({
            'averaged_traces_all': lambda x: list(x),   # Keep as list of arrays
            'time_axes': lambda x: list(x)              # Keep as list of arrays
        })
        .reset_index()
    )

    # Ensure all four categories exist
    category_order = [
        'Short tau stim & Short tau',
        'Short tau stim & Long tau',
        'Long tau stim & Short tau',
        'Long tau stim & Long tau'
    ]

    # Build all combinations
    all_combos = pd.MultiIndex.from_product(
        [df_valid[group_cols].drop_duplicates().itertuples(index=False, name=None),
         category_order],
        names=['group', 'category']
    ).to_frame(index=False)

    # Split tuple into columns
    for i, col in enumerate(group_cols):
        all_combos[col] = all_combos['group'].apply(lambda g: g[i])
    all_combos = all_combos.drop(columns='group')

    # Merge
    merged = all_combos.merge(grouped, on=group_cols + ['category'], how='left')

    # Replace NaNs with empty lists
    for col in ['averaged_traces_all', 'time_axes']:
        merged[col] = merged[col].apply(lambda x: x if isinstance(x, list) else [])

    return merged


# %%

def group_values_by_tau_cutoffs(df, group_cols,value_col):
    """
    Groups 'averaged_traces_all' entries for significant cells (is_significant == 1)
    based on tau_length and tau_stim_length categories.

    Args:
        df: DataFrame containing at least:
            - group_cols
            - averaged_traces_all (iterable: list/array)
            - tau_length ('Short'/'Long'/NaN)
            - tau_stim_length ('Short'/'Long'/NaN)
            - is_significant (0/1)
        group_cols: List of columns to group by.

    Returns:
        DataFrame with one row per group/category, containing a list of traces.
    """
    # Filter to significant only and valid tau categories
    df_valid = df[
        (df['is_significant'] == 1) &
        df['tau_length'].notna() &
        df['tau_stim_length'].notna()
    ].copy()

    # Create category label
    df_valid['category'] = (
        df_valid['tau_stim_length'] + ' tau stim & ' + df_valid['tau_length'] + ' tau'
    )

    # Group and aggregate traces into a single list
    grouped = (
        df_valid.groupby(group_cols + ['category'])[value_col]
        .apply(list)
        .reset_index()
    )

    # Ensure all four categories exist for each group
    category_order = [
        'Short tau stim & Short tau',
        'Short tau stim & Long tau',
        'Long tau stim & Short tau',
        'Long tau stim & Long tau'
    ]

    # Build all combinations
    all_combos = (
        pd.MultiIndex.from_product(
            [df_valid[group_cols].drop_duplicates().itertuples(index=False, name=None),
             category_order],
            names=['group', 'category']
        )
        .to_frame(index=False)
    )
    # Split tuple into columns
    for i, col in enumerate(group_cols):
        all_combos[col] = all_combos['group'].apply(lambda g: g[i])
    all_combos = all_combos.drop(columns='group')

    # Merge
    merged = all_combos.merge(grouped, on=group_cols + ['category'], how='left')

    # Replace NaNs with empty lists
    merged[value_col] = merged[value_col].apply(
        lambda x: x if isinstance(x, list) else []
    )

    return merged


# %%
import pandas as pd
import numpy as np

def split_traces_to_2d_dfs(df, group_cols):
    """
    Takes the output of group_traces_by_tau_cutoffs and returns
    4 separate DataFrames (one per category), each 2D with traces as rows.

    Skips empty rows and unnests nested arrays/lists.
    """
    category_order = [
        'Short tau stim & Short tau',
        'Short tau stim & Long tau',
        'Long tau stim & Short tau',
        'Long tau stim & Long tau'
    ]

    dfs = {}

    for cat in category_order:
        sub_df = df[df['category'] == cat].copy()

        all_rows = []

        for _, row in sub_df.iterrows():
            traces = row['averaged_traces_all']
            if not traces:  # Skip empty lists
                continue

            # Unnest: flatten if it's a list of lists/arrays
            flat_traces = []
            for t in traces:
                if isinstance(t, (list, np.ndarray)):
                    t_arr = np.array(t)
                    if t_arr.ndim == 1:  # already a single trace
                        flat_traces.append(t_arr)
                    elif t_arr.ndim == 2:  # multiple traces inside
                        flat_traces.extend(t_arr)
                else:
                    flat_traces.append(np.array(t))

            # Append each trace with metadata from group_cols
            for ft in flat_traces:
                all_rows.append(list(row[group_cols]) + list(ft))

        if not all_rows:
            dfs[cat] = pd.DataFrame()  # Empty DataFrame if nothing
            continue

        # Build column names: group_cols + timepoints
        n_timepoints = len(all_rows[0]) - len(group_cols)
        col_names = group_cols + [f"t{i}" for i in range(n_timepoints)]

        dfs[cat] = pd.DataFrame(all_rows, columns=col_names)

    return dfs

# Example usage:
# grouped_df = group_traces_by_tau_cutoffs(df, ['area', 'mouse_id'])
# dfs_by_category = split_traces_to_2d_dfs(grouped_df, ['area', 'mouse_id'])
# dfs_by_category['Short tau stim & Short tau']  # 2D DataFrame of traces


def compare_regularization_with_significance(
    df, feature_cols, target_col,
    alpha_ridge=1.0, lasso_alpha=None,
    lasso_alphas_cv=None, cv=5, include_interactions=False,
    random_state=0, interaction_pairs=None
):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    import statsmodels.api as sm
    from itertools import combinations
    from sklearn.model_selection import cross_val_score

    # --- Drop NaNs ---
    df_filtered = df.dropna(subset=feature_cols + [target_col])
    X = df_filtered[feature_cols].copy()
    y = df_filtered[target_col].copy()

    # --- Standardize features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    if include_interactions:
        interaction_cols = []
        for f1, f2 in combinations(feature_cols, 2):
            if interaction_pairs is not None and (f1, f2) not in interaction_pairs and (f2, f1) not in interaction_pairs:
                continue
            inter_name = f"{f1}_x_{f2}"
            interaction_term = X_df_scaled[f1] * X_df_scaled[f2]
            interaction_term = (interaction_term - interaction_term.mean()) / interaction_term.std()
            X_df_scaled[inter_name] = interaction_term
            interaction_cols.append(inter_name)
        print(f"Added {len(interaction_cols)} interaction terms")

    all_features = X_df_scaled.columns.tolist()
    X_scaled_full = X_df_scaled.values

    # --- Initialize models ---
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=alpha_ridge)
    }

    results = {}
    coef_df = pd.DataFrame(index=all_features)
    pval_df = pd.DataFrame(index=all_features, dtype=float)
    coef_se_df = pd.DataFrame(index=all_features, dtype=float)  # <-- NEW

    # --- Linear regression ---
    lin_model = models['Linear']
    lin_model.fit(X_scaled_full, y)
    y_pred = lin_model.predict(X_scaled_full)
    coef_df['Linear'] = lin_model.coef_
    results['Linear'] = {
        'model': lin_model,
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred)
    }

    # --- Linear p-values with statsmodels ---
    X_const = sm.add_constant(X_scaled_full)
    ols_model = sm.OLS(y, X_const).fit()
    pval_df['Linear'] = pd.Series(ols_model.pvalues[1:], index=all_features)

    # --- Coefficient SEs (from OLS) ---
    coef_se_df['Linear'] = np.sqrt(np.diag(ols_model.cov_params()))[1:]

    # --- SEE for Linear ---
    n, k = X_scaled_full.shape
    rss = np.sum((y - y_pred) ** 2)
    see_linear = np.sqrt(rss / (n - k - 1))

    print(f"OLS Training R²: {ols_model.rsquared:.4f}")
    r2_cv = cross_val_score(LinearRegression(), X_scaled_full, y, cv=5, scoring="r2")
    print("CV R² scores:", r2_cv)
    print("Mean CV R²:", r2_cv.mean())
    print(f"SEE (Linear): {see_linear:.4f}")

    # --- Ridge regression ---
    ridge_model = models['Ridge']
    ridge_model.fit(X_scaled_full, y)
    y_pred = ridge_model.predict(X_scaled_full)
    coef_df['Ridge'] = ridge_model.coef_
    results['Ridge'] = {
        'model': ridge_model,
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred)
    }
    pval_df['Ridge'] = np.nan
    coef_se_df['Ridge'] = np.nan  # Approximate SEs possible, left as NaN

    # --- Lasso regression ---
    if lasso_alpha is not None:
        lasso_model = Lasso(alpha=lasso_alpha, max_iter=5000, random_state=random_state)
        lasso_model.fit(X_scaled_full, y)
        used_alpha = lasso_alpha
    else:
        if lasso_alphas_cv is None:
            lasso_alphas_cv = np.logspace(-5, -0.5, 50)
        lasso_cv = LassoCV(alphas=lasso_alphas_cv, cv=cv, random_state=random_state, max_iter=5000)
        lasso_cv.fit(X_scaled_full, y)
        used_alpha = lasso_cv.alpha_
        lasso_model = Lasso(alpha=used_alpha, max_iter=5000, random_state=random_state)
        lasso_model.fit(X_scaled_full, y)

        mse_mean = lasso_cv.mse_path_.mean(axis=1)
        y_var = np.var(y, ddof=1)
        r2_values = 1 - mse_mean / y_var
        plt.figure(figsize=(6,4))
        plt.semilogx(lasso_cv.alphas_, r2_values, marker='o')
        plt.axvline(used_alpha, color='red', linestyle='--', label='Selected alpha')
        plt.xlabel('Alpha')
        plt.ylabel('R²')
        plt.title('LassoCV R² vs Alpha')
        plt.legend()
        plt.gca().invert_xaxis()
        plt.show()

    y_pred = lasso_model.predict(X_scaled_full)
    r2_final = r2_score(y, y_pred)
    print(f"Final Lasso R² (alpha={used_alpha:.6f}): {r2_final:.4f}")

    coef_df['Lasso'] = lasso_model.coef_
    results['Lasso'] = {
        'model': lasso_model,
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'used_alpha': used_alpha
    }

    nonzero_idx = np.where(lasso_model.coef_ != 0)[0]
    pvals_lasso = pd.Series(np.nan, index=all_features)
    if len(nonzero_idx) > 0:
        X_lasso = X_scaled_full[:, nonzero_idx]
        X_lasso_const = sm.add_constant(X_lasso)
        ols_lasso = sm.OLS(y, X_lasso_const).fit()
        pvals_lasso.iloc[nonzero_idx] = ols_lasso.pvalues[1:]
        coef_se_df.loc[pvals_lasso.index[nonzero_idx], 'Lasso'] = np.sqrt(np.diag(ols_lasso.cov_params()))[1:]
    pval_df['Lasso'] = pvals_lasso

    for name in ['Linear', 'Ridge', 'Lasso']:
        df_plot = pd.DataFrame({
            'Feature': all_features,
            'Coefficient': coef_df[name],
            'p_value': pval_df[name]
        }).sort_values(by='Coefficient', key=abs)

        plt.figure(figsize=(8, 6))
        bars = sns.barplot(data=df_plot, x='Coefficient', y='Feature', color='blue')
        for bar, p in zip(bars.patches, df_plot['p_value']):
            if pd.notna(p) and p < 0.05:
                bar.set_alpha(1.0)
            else:
                bar.set_alpha(0.2)
        plt.axvline(0, color='gray', linestyle='--')
        plt.title(f"{name} Coefficients (Shaded by Significance)")
        plt.tight_layout()
        plt.show()

    return results, coef_df, pval_df, coef_se_df, used_alpha, r2_final, see_linear

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations


def run_logistic_regression(
    df, 
    feature_cols, 
    target_col='is_significant', 
    include_interactions=False, 
    interaction_pairs=None,
    firth=True,
    C=1
):
    """
    Runs logistic regression with optional interaction terms.
    Includes sklearn + statsmodels outputs, coefficient plots, and ROC/PR curves.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list): List of feature column names.
        target_col (str): Binary target column (default 'is_significant').
        include_interactions (bool): Whether to add pairwise interaction terms.
        interaction_pairs (list of tuple): Specific feature pairs to interact (optional).

    Returns:
        model (LogisticRegression): Fitted sklearn logistic regression model.
        summary_df (pd.DataFrame): Coefficients, p-values, std errors from statsmodels.
    """

    # Drop rows with NaNs
    df_filtered = df.dropna(subset=feature_cols + [target_col])

    # Extract X and y
    X_df = df_filtered[feature_cols].copy()
    y = df_filtered[target_col]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X_df.index)

    # --- Interaction terms ---
    interaction_cols = []
    if include_interactions:
        for f1, f2 in combinations(feature_cols, 2):
            # Respect interaction_pairs if provided
            if interaction_pairs is not None and (f1, f2) not in interaction_pairs and (f2, f1) not in interaction_pairs:
                continue

            inter_name = f"{f1}_x_{f2}"
            interaction_term = X_df_scaled[f1] * X_df_scaled[f2]

            # Standardize interaction term
            interaction_term = (interaction_term - interaction_term.mean()) / interaction_term.std()

            X_df_scaled[inter_name] = interaction_term
            interaction_cols.append(inter_name)

        print(f"Added {len(interaction_cols)} interaction terms")

    # Final design matrix
    X_final = X_df_scaled.values
    all_features = list(X_df_scaled.columns)


    # --- sklearn Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000,class_weight="balanced",C=C)
    log_reg.fit(X_final, y)

    

    y_pred = log_reg.predict(X_final)
    y_prob = log_reg.predict_proba(X_final)[:, 1]

    print("\n--- Sklearn Logistic Regression ---")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # --- statsmodels for p-values ---
    X_const = sm.add_constant(X_final)
    sm_model = sm.Logit(y, X_const)
    sm_results = sm_model.fit_regularized(method='l1', L1_wt=0, alpha=1/C)


    summary_df = pd.DataFrame({
        'feature': ['Intercept'] + all_features,
        'coef': sm_results.params,
        'p_value': sm_results.pvalues,
        'std_err': sm_results.bse,
        'z_score': sm_results.tvalues
    })

    print("\n--- Statsmodels Logistic Regression Summary ---")
    print(summary_df.sort_values('p_value'))

    # --- Coefficient Plot ---
    sorted_df = summary_df[summary_df['feature'] != 'Intercept'].sort_values(
        by='coef', key=abs, ascending=True
    )

    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(
        data=sorted_df,
        x='coef',
        y='feature',
        color='blue'
    )

    for p_val, bar in zip(sorted_df['p_value'], barplot.patches):
        if p_val < 0.05:
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.2)

    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Logistic Regression Coefficients (Significance Highlighted)")
    plt.xlabel("Coefficient (log-odds)")
    plt.tight_layout()
    plt.show()

    # --- ROC and Precision-Recall Curves ---
    fpr, tpr, _ = roc_curve(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)

    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # Precision-Recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="purple")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.show()

    return log_reg, summary_df, X_df_scaled

# %%

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from itertools import combinations
from sklearn.utils import resample


def bootstrap_logistic_regression(
    df, 
    feature_cols, 
    target_col='is_significant', 
    include_interactions=False, 
    interaction_pairs=None,
    C=1,
    n_bootstrap=1000,
    random_state=None
):
    """
    Bootstraps logistic regression coefficients with downsampling.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (list): Feature column names.
        target_col (str): Binary target column.
        include_interactions (bool): Whether to include interactions.
        interaction_pairs (list of tuple): Optional list of pairs for interactions.
        C (float): Inverse of regularization strength.
        n_bootstrap (int): Number of bootstrap iterations.
        random_state (int): For reproducibility.
        
    Returns:
        coef_bootstrap (pd.DataFrame): DataFrame of bootstrapped coefficients.
    """
    rng = np.random.default_rng(random_state)
    
    # Drop rows with missing data
    df_filtered = df.dropna(subset=feature_cols + [target_col])

    coef_list = []

    for b in range(n_bootstrap):
        # --- Downsample majority class to match minority ---
        df_majority = df_filtered[df_filtered[target_col] == 0]
        df_minority = df_filtered[df_filtered[target_col] == 1]

        df_majority_down = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=rng.integers(0, 1e9)
        )

        df_balanced = pd.concat([df_majority_down, df_minority])

        # --- Features and target ---
        X_df = df_balanced[feature_cols].copy()
        y = df_balanced[target_col]

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        X_df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X_df.index)

        # --- Interaction terms ---
        if include_interactions:
            for f1, f2 in combinations(feature_cols, 2):
                if interaction_pairs is not None and (f1, f2) not in interaction_pairs and (f2, f1) not in interaction_pairs:
                    continue

                inter_name = f"{f1}_x_{f2}"
                interaction_term = X_df_scaled[f1] * X_df_scaled[f2]
                interaction_term = (interaction_term - interaction_term.mean()) / interaction_term.std()

                X_df_scaled[inter_name] = interaction_term

        X_final = X_df_scaled.values
        all_features = list(X_df_scaled.columns)

        # --- Statsmodels Logistic Regression ---
        X_const = sm.add_constant(X_final)
        sm_model = sm.Logit(y, X_const)
        sm_results = sm_model.fit(disp=False)
        # sm_results = sm_model.fit_regularized(
        #     method='l1', L1_wt=0, alpha=1/C, disp=False
        # )

        # Save coefficients
        coef_dict = dict(zip(['Intercept'] + all_features, sm_results.params))
        coef_list.append(coef_dict)

    # Convert to DataFrame
    coef_bootstrap = pd.DataFrame(coef_list)
    return coef_bootstrap
# %%
import pandas as pd
from scipy.stats import wilcoxon

def test_coefs_nonparametric(coef_df):
    """
    Run Wilcoxon signed-rank test for each feature column in coef_df
    against 0.

    Args:
        coef_df (pd.DataFrame): rows = repetitions, cols = features

    Returns:
        pd.DataFrame: feature, median_coef, p_value
    """
    results = []
    for feature in coef_df.columns:
        coefs = coef_df[feature].dropna().values

        # Wilcoxon test against 0
        try:
            stat, p = wilcoxon(coefs)
        except ValueError:
            # happens if all values identical
            p = float("nan")

        results.append({
            "feature": feature,
            "median_coef": pd.Series(coefs).median(),
            'mean_coef':pd.Series(coefs).mean(),
            "p_value": p
        })

    return pd.DataFrame(results)

from statsmodels.stats.descriptivestats import sign_test

def test_coefs_sign(coef_df):
    results = []
    for feature in coef_df.columns:
        coefs = coef_df[feature].dropna().values
        try:
            stat, p = sign_test(coefs, mu0=0)  # median test
        except Exception:
            p = float("nan")
        results.append({
            "feature": feature,
            "median_coef": pd.Series(coefs).median(),
            "p_value": p
        })
    return pd.DataFrame(results)

def proportion_crossing_value(coef_df, threshold=0.05,value=0):
    results = []
    for feature in coef_df.columns:
        coefs = coef_df[feature].dropna().values
        n = len(coefs)
        frac_positive = (coefs >= value).sum() / n
        frac_negative = (coefs < value).sum() / n
        # frac_zero = (coefs == 0).sum() / n
        frac_crossing = min(frac_positive, frac_negative) * 2  # fraction on minority side *2

        results.append({
            "feature": feature,
            "median_coef": pd.Series(coefs).median(),
            "frac_above_zero": frac_positive,
            "frac_below_zero": frac_negative,
            # "frac_zero": frac_zero,
            "crosses_zero": frac_crossing > threshold
        })
    return pd.DataFrame(results)


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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_grouped_bar_with_error_vertical(
    coef_df1, coef_df2,
    dataset_labels=("Dataset 1", "Dataset 2"),
    colors=("skyblue", "salmon"),
    figsize=(8, 10),
    bar_width=0.35,
    bar_spacing=0.15,
    feature_order=None,
    selected_features=None,
    crosses_df1=None,
    crosses_df2=None,
    xlim=None  # New input for x-axis limits
):
    """
    Make horizontal paired bar plots for two datasets (mean ± std) by feature.

    Args:
        coef_df1, coef_df2 : pd.DataFrame
            Each column is a feature, rows are repetitions of coefficients.
        dataset_labels : tuple
            Labels for the two datasets.
        colors : tuple
            Colors for dataset 1 and dataset 2 bars.
        figsize : tuple
            Figure size (width, height).
        bar_width : float
            Width of each bar.
        bar_spacing : float
            Extra spacing between paired bars for each feature.
        feature_order : list
            Order in which to plot features (top to bottom).
        selected_features : list
            If provided, only these features will be plotted.
        crosses_df1, crosses_df2 : pd.DataFrame or None
            Each should have rows for features and a boolean column 'crosses_zero'.
            If provided, asterisks are drawn for features where crosses_zero == False.
        xlim : tuple or None
            (xmin, xmax) to set x-axis limits.
    """
    # Ensure both datasets have the same features
    common_features = list(set(coef_df1.columns) & set(coef_df2.columns))

    # Subset if selected_features provided
    if selected_features is not None:
        features = [f for f in selected_features if f in common_features]
    else:
        features = common_features

    # Apply order if provided
    if feature_order is not None:
        features = [f for f in feature_order if f in features]

    # Compute mean and std
    means1 = coef_df1[features].mean()
    stds1 = coef_df1[features].std()
    means2 = coef_df2[features].mean()
    stds2 = coef_df2[features].std()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(features))  # feature positions

    bars1 = ax.barh(
        y - bar_width/2 - bar_spacing/2, means1, xerr=stds1,
        height=bar_width, color=colors[0], label=dataset_labels[0],
        edgecolor=colors[0],
        error_kw=dict(ecolor=colors[0], lw=1.2, capsize=4, capthick=1.2)
    )
    bars2 = ax.barh(
        y + bar_width/2 + bar_spacing/2, means2, xerr=stds2,
        height=bar_width, color=colors[1], label=dataset_labels[1],
        edgecolor=colors[1],
        error_kw=dict(ecolor=colors[1], lw=1.2, capsize=4, capthick=1.2)
    )

    # --- Add significance markers (asterisks) ---
    if crosses_df1 is not None:
        for i, feature in enumerate(features):
            row = crosses_df1[crosses_df1["feature"] == feature]
            if not row.empty and row["crosses_zero"].iloc[0] == False:
                bar = bars1[i]  # dataset1 bar
                ax.text(
                    bar.get_width(), bar.get_y() + bar.get_height()/2,
                    "*", va="center", ha="left", fontsize=18, color=colors[0]
                )

    if crosses_df2 is not None:
        for i, feature in enumerate(features):
            row = crosses_df2[crosses_df2["feature"] == feature]
            if not row.empty and row["crosses_zero"].iloc[0] == False:
                bar = bars2[i]  # dataset2 bar
                ax.text(
                    bar.get_width(), bar.get_y() + bar.get_height()/2,
                    "*", va="center", ha="left", fontsize=18, color=colors[1]
                )

    # Add vertical line at 0
    ax.axvline(0, color="black", linestyle="--", linewidth=1)

    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # so features go top → bottom
    ax.set_xlabel("Coefficient (mean ± std)")
    ax.legend()

    # Set x-axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_single_beta_bars(
    coef_df1, coef_df2,
    pval_df1=None, pval_df2=None,
    coef_se_df1=None, coef_se_df2=None,   # <-- NEW
    method='Lasso',
    dataset_labels=("Dataset 1", "Dataset 2"),
    colors=("skyblue", "salmon"),
    figsize=(8, 10),
    bar_width=0.35,
    bar_spacing=0.15,
    feature_order=None,
    selected_features=None,
    alpha_signif=0.05,
    xlim=None
):
    """
    Plot paired vertical bars of beta coefficients for two datasets,
    with optional significance asterisks and error bars.

    Args:
        coef_df1, coef_df2 : pd.DataFrame
            Rows are features, columns are methods (e.g., 'Lasso').
        pval_df1, pval_df2 : pd.DataFrame or None
            Same shape as coef_df*, used to draw asterisks if p < alpha_signif.
        coef_se_df1, coef_se_df2 : pd.DataFrame or None
            Same shape as coef_df*, used to plot error bars for coefficients.
        method : str
            Column to select from coef_df and pval_df.
    """
    # Select method column
    coefs1 = coef_df1[method]
    coefs2 = coef_df2[method]

    # Features from index
    features = coefs1.index.intersection(coefs2.index).tolist()

    # Subset selected_features
    if selected_features is not None:
        features = [f for f in selected_features if f in features]

    # Apply order if provided
    if feature_order is not None:
        features = [f for f in feature_order if f in features]

    coefs1_vals = coefs1.loc[features].values
    coefs2_vals = coefs2.loc[features].values

    # Error bar values
    errs1 = coef_se_df1[method].loc[features].values if coef_se_df1 is not None else None
    errs2 = coef_se_df2[method].loc[features].values if coef_se_df2 is not None else None

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(features))

    bars1 = ax.barh(
        y - bar_width/2 - bar_spacing/2, coefs1_vals,
        xerr=errs1,  # <-- error bars
        ecolor=colors[0], error_kw=dict(capsize=4, lw=2),
        height=bar_width, color=colors[0], label=dataset_labels[0],
        edgecolor=colors[0], alpha=0.8
    )
    bars2 = ax.barh(
        y + bar_width/2 + bar_spacing/2, coefs2_vals,
        xerr=errs2,  # <-- error bars
        ecolor=colors[1], error_kw=dict(capsize=4, lw=2),
        height=bar_width, color=colors[1], label=dataset_labels[1],
        edgecolor=colors[1], alpha=0.8
    )

    # Add significance asterisks if p-value data provided
    if pval_df1 is not None:
        for i, f in enumerate(features):
            pval = pval_df1.loc[f, method]
            if pval < alpha_signif:
                bar = bars1[i]
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, "*",
                        va="center", ha="left", fontsize=14, color=colors[0])

    if pval_df2 is not None:
        for i, f in enumerate(features):
            pval = pval_df2.loc[f, method]
            if pval < alpha_signif:
                bar = bars2[i]
                ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, "*",
                        va="center", ha="left", fontsize=14, color=colors[1])

    # Vertical line at 0
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    
    # Set x-axis limits if provided
    if xlim is not None:
        ax.set_xlim(xlim)
        
    # Formatting
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel(f"{method} Beta Coefficient")
    ax.legend()
    plt.tight_layout()
    plt.show()


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, balanced_accuracy_score
)
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
from imblearn.over_sampling import SMOTE


def run_logistic_regression_smote(
    df, 
    feature_cols, 
    target_col='is_significant', 
    include_interactions=False, 
    interaction_pairs=None,
    use_smote=True,
    random_state=42,
    C=1
):
    """
    Runs logistic regression with optional interaction terms.
    Includes sklearn + statsmodels outputs, coefficient plots, and ROC/PR curves.
    Optionally applies SMOTE to balance classes.
    """

    # Drop rows with NaNs
    df_filtered = df.dropna(subset=feature_cols + [target_col])

    # Extract X and y
    X_df = df_filtered[feature_cols].copy()
    y = df_filtered[target_col]

    # --- Class distribution before SMOTE ---
    print("\n--- Original Class Distribution ---")
    print(Counter(y))
    majority_class_acc = max(Counter(y).values()) / len(y)
    print(f"Baseline (always predict majority class): {majority_class_acc:.3f}")

    # --- Apply SMOTE ---
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_df, y)
        X_df = pd.DataFrame(X_res, columns=feature_cols)
        y = pd.Series(y_res)
        print("\n--- After SMOTE ---")
        print(Counter(y))

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    # --- Interaction terms ---
    interaction_cols = []
    if include_interactions:
        for f1, f2 in combinations(feature_cols, 2):
            if interaction_pairs is not None and (f1, f2) not in interaction_pairs and (f2, f1) not in interaction_pairs:
                continue

            inter_name = f"{f1}_x_{f2}"
            interaction_term = X_df_scaled[f1] * X_df_scaled[f2]
            interaction_term = (interaction_term - interaction_term.mean()) / interaction_term.std()

            X_df_scaled[inter_name] = interaction_term
            interaction_cols.append(inter_name)

        print(f"Added {len(interaction_cols)} interaction terms")

    # Final design matrix
    X_final = X_df_scaled.values
    all_features = list(X_df_scaled.columns)

    # --- sklearn Logistic Regression ---
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced",C=C)
    log_reg.fit(X_final, y)

    y_pred = log_reg.predict(X_final)
    y_prob = log_reg.predict_proba(X_final)[:, 1]

    print("\n--- Sklearn Logistic Regression ---")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Balanced Accuracy:", balanced_accuracy_score(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_prob))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

    # --- statsmodels for p-values ---
    # X_const = sm.add_constant(X_final)
    # sm_model = sm.Logit(y, X_const)
    # sm_results = sm_model.fit(disp=False)
    
    X_const = sm.add_constant(X_final)
    sm_model = sm.Logit(y, X_const)
    sm_results = sm_model.fit_regularized(method='l1', L1_wt=0, alpha=1/C)

    summary_df = pd.DataFrame({
        'feature': ['Intercept'] + all_features,
        'coef': sm_results.params,
        'p_value': sm_results.pvalues,
        'std_err': sm_results.bse,
        'z_score': sm_results.tvalues
    })

    print("\n--- Statsmodels Logistic Regression Summary ---")
    print(summary_df.sort_values('p_value'))

    # --- Coefficient Plot ---
    sorted_df = summary_df[summary_df['feature'] != 'Intercept'].sort_values(
        by='coef', key=abs, ascending=True
    )

    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(
        data=sorted_df,
        x='coef',
        y='feature',
        color='blue'
    )

    for p_val, bar in zip(sorted_df['p_value'], barplot.patches):
        if p_val < 0.05:
            bar.set_alpha(1.0)
        else:
            bar.set_alpha(0.2)

    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Logistic Regression Coefficients (Significance Highlighted)")
    plt.xlabel("Coefficient (log-odds)")
    plt.tight_layout()
    plt.show()

    # --- ROC and Precision-Recall Curves ---
    fpr, tpr, _ = roc_curve(y, y_prob)
    precision, recall, _ = precision_recall_curve(y, y_prob)

    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    # Precision-Recall
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color="purple")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.tight_layout()
    plt.show()

    return log_reg, summary_df, X_df_scaled

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import statsmodels.api as sm
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

def plot_logistic_regression_results(model, summary_df, X_df_scaled, y=None, predictors=None, num_points=100):
    """
    Visualizes logistic regression results:
    1) Coefficient plot (forest plot style)
    2) Predicted probability curves for selected predictors
       - Automatically handles interactions by showing curves at low/medium/high levels of the other variable.
    3) Predicted probability distributions for each class
    4) ROC and Precision-Recall curves
    """

        # ----------------------------
    # 1) Coefficient forest plot
    # ----------------------------
    coef_df = summary_df[summary_df['feature'] != 'Intercept'].copy()
    
    # Ensure numeric type
    coef_df['coef'] = pd.to_numeric(coef_df['coef'])
    coef_df['p_value'] = pd.to_numeric(coef_df['p_value'])
    
    # Sort by absolute value
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df = coef_df.sort_values('abs_coef', ascending=True)
    
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(
        data=coef_df,
        x='coef',
        y='feature',
        color='blue'
    )
    
    # Highlight significance
    for p_val, bar in zip(coef_df['p_value'], barplot.patches):
        bar.set_alpha(1.0 if p_val < 0.05 else 0.2)
    
    plt.axvline(0, color='gray', linestyle='--')
    plt.title("Logistic Regression Coefficients (Significance Highlighted)")
    plt.xlabel("Coefficient (log-odds)")
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 2) Predicted probability distributions
    # ----------------------------
    if y is not None:
        y_prob = model.predict_proba(X_df_scaled.values)[:, 1]

        plt.figure(figsize=(8,5))
        sns.kdeplot(y_prob[y==0], label='Class 0', fill=True, alpha=0.5)
        sns.kdeplot(y_prob[y==1], label='Class 1', fill=True, alpha=0.5)
        plt.axvline(0.5, color='gray', linestyle='--', label='Decision threshold')
        plt.xlabel("Predicted Probability")
        plt.ylabel("Density")
        plt.title("Predicted Probabilities by Class")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ----------------------------
        # 3) ROC and Precision-Recall Curves
        # ----------------------------
        fpr, tpr, _ = roc_curve(y, y_prob)
        precision, recall, _ = precision_recall_curve(y, y_prob)
        ap_score = average_precision_score(y, y_prob)

        plt.figure(figsize=(12,5))

        # ROC
        plt.subplot(1,2,1)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, y_prob):.2f}")
        plt.plot([0,1],[0,1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        # Precision-Recall
        plt.subplot(1,2,2)
        plt.plot(recall, precision, color="purple", label=f"AP = {ap_score:.2f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()

        plt.tight_layout()
        plt.show()

    # ----------------------------
    # 4) Predicted probability curves for main effects
    # ----------------------------
    plt.figure(figsize=(8,6))
    if predictors is None:
        predictors = [f for f in X_df_scaled.columns if "_x_" not in f]

    main_effects = [f for f in predictors if "_x_" not in f]
    interaction_terms = [f for f in X_df_scaled.columns if "_x_" in f]

    for predictor in main_effects:
        values = np.linspace(X_df_scaled[predictor].min(), X_df_scaled[predictor].max(), num_points)
        X_mean = X_df_scaled.mean().to_dict()
        X_pred = pd.DataFrame([X_mean]*num_points)
        X_pred[predictor] = values

        for col in interaction_terms:
            f1, f2 = col.split("_x_")
            X_pred[col] = X_pred[f1] * X_pred[f2]

        y_prob_curve = model.predict_proba(X_pred.values)[:, 1]
        plt.plot(values, y_prob_curve, label=predictor)

    plt.xlabel("Standardized predictor value")
    plt.ylabel("Predicted probability of target=1")
    plt.title("Predicted probability curves (main effects)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----------------------------
    # 5) Interaction effects
    # ----------------------------
    for inter in interaction_terms:
        f1, f2 = inter.split("_x_")
        levels = [X_df_scaled[f2].quantile(q) for q in [0.25,0.5,0.75]]
        labels = [f"f2 {q:.2f}" for q in [0.25,0.5,0.75]]
        tau1_vals = np.linspace(X_df_scaled[f1].min(), X_df_scaled[f1].max(), num_points)

        plt.figure(figsize=(8,6))
        for val, label in zip(levels, labels):
            X_mean = X_df_scaled.mean().to_dict()
            probs = []
            for tau1 in tau1_vals:
                X_mean[f1] = tau1
                X_mean[f2] = val
                X_mean[inter] = tau1 * val
                prob = model.predict_proba(np.array(list(X_mean.values())).reshape(1,-1))[:,1][0]
                probs.append(prob)
            plt.plot(tau1_vals, probs, label=f"{f2}={val:.2f}")
        plt.xlabel(f"{f1} (standardized)")
        plt.ylabel("Predicted probability")
        plt.title(f"Interaction effect: {f1} × {f2}")
        plt.legend()
        plt.tight_layout()
        plt.show()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from itertools import combinations


def run_lasso_logistic_regression(
    df, 
    feature_cols, 
    target_col="is_significant",
    include_interactions=False,
    interaction_pairs=None,
    alpha=None,       # None = no reg, float = fixed alpha, list/array = CV search
    penalty="l1",
    solver="saga",
    max_iter=5000,
    cv=5
):
    """
    Run logistic regression with optional Lasso regularization and interactions.
    Similar inputs to run_logistic_regression.

    Returns:
        model: fitted sklearn logistic regression model
        X_df_scaled: scaled feature dataframe (with interactions if applied)
        chosen_alpha: best/fixed alpha used
        r2: pseudo-R² (McFadden’s)
    """
    # Drop NaNs
    df_filtered = df.dropna(subset=feature_cols + [target_col])

    # Extract predictors + target
    X = df_filtered[feature_cols]
    y = df_filtered[target_col]

    # Scale predictors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

    # Add interaction terms if requested
    if include_interactions:
        interaction_cols = []
        for f1, f2 in combinations(feature_cols, 2):
            if interaction_pairs is not None and (f1, f2) not in interaction_pairs and (f2, f1) not in interaction_pairs:
                continue
            inter_name = f"{f1}_x_{f2}"
            interaction_term = X_df_scaled[f1] * X_df_scaled[f2]
            interaction_term = (interaction_term - interaction_term.mean()) / interaction_term.std()
            X_df_scaled[inter_name] = interaction_term
            interaction_cols.append(inter_name)
        print(f"Added {len(interaction_cols)} interaction terms")

    # Handle alpha input
    if alpha is None:
        # No regularization
        model = LogisticRegression(penalty="none", max_iter=max_iter)
        model.fit(X_df_scaled, y)
        chosen_alpha = None
    elif isinstance(alpha, (list, np.ndarray)):
        # Cross-validation for best alpha
        model = LogisticRegressionCV(
            Cs=alpha,          # grid of C values (inverse of alpha)
            penalty=penalty,
            solver=solver,
            cv=cv,
            max_iter=max_iter,
            scoring="accuracy"
        )
        model.fit(X_df_scaled, y)
        chosen_alpha = 1 / model.C_[0]
    else:
        # Fixed alpha
        model = LogisticRegression(
            C=1/alpha,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter
        )
        model.fit(X_df_scaled, y)
        chosen_alpha = alpha

    # Compute pseudo-R² (McFadden’s)
    ll_null = -np.sum(np.log([np.mean(y), 1-np.mean(y)])[y])
    ll_model = -np.sum(np.log(model.predict_proba(X_df_scaled)[np.arange(len(y)), y]))
    r2 = 1 - ll_model/ll_null

    return model, X_df_scaled, chosen_alpha, r2
# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm

def plot_lasso_logistic_with_inference(model, X_df_scaled, y_filtered, chosen_alpha=None, r2=None):
    """
    Plot predicted probabilities vs true labels and LASSO coefficient bar plot
    with formal significance from statsmodels Logit on LASSO-selected features.
    
    Args:
        model: fitted sklearn LASSO LogisticRegression
        X_df_scaled: scaled predictors (with interactions if any)
        y_filtered: matching target variable
        chosen_alpha: selected alpha
        r2: pseudo-R²
    """
    # --- Predicted probabilities scatter ---
    y_pred_prob = model.predict_proba(X_df_scaled)[:, 1]
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred_prob, y_filtered, alpha=0.6, edgecolor="k")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed outcome")
    plt.title("LASSO Logistic Regression Predictions")
    plt.plot([0,1],[0,1],"r--")
    plt.show()

    # --- Identify selected features ---
    selected_features = X_df_scaled.columns[model.coef_.flatten() != 0]
    if len(selected_features) == 0:
        print("No features selected by LASSO.")
        return
    
    print(f"Selected {len(selected_features)} features by LASSO: {list(selected_features)}")

    # --- Fit unregularized Logit for formal inference ---
    X_selected = X_df_scaled[selected_features]
    X_const = sm.add_constant(X_selected)
    sm_model = sm.Logit(y_filtered, X_const)
    sm_results = sm_model.fit(disp=False)

    # --- Prepare coefficient DataFrame ---
    summary_df = pd.DataFrame({
        "feature": sm_results.params.index,
        "coef": sm_results.params.values,
        "p_value": sm_results.pvalues.values,
        "std_err": sm_results.bse.values,
        "z_score": sm_results.tvalues
    })

    # --- Bar plot with significance highlighting ---
    coef_df_sorted = summary_df[summary_df['feature'] != 'const'].sort_values(
        by='coef', key=abs, ascending=True
    )

    plt.figure(figsize=(8,6))
    colors = ["blue" if p < 0.05 else "lightgray" for p in coef_df_sorted['p_value']]
    sns.barplot(x="coef", y="feature", data=coef_df_sorted, palette=colors)
    plt.axvline(0, color="black", linestyle="--")
    plt.title("LASSO Selected Features with Formal Significance (p < 0.05)")
    plt.xlabel("Coefficient (log-odds)")
    plt.tight_layout()
    plt.show()

    # --- Print summary table ---
    print("\n--- LASSO Selected Features with Formal Inference ---")
    print(summary_df.sort_values("p_value"))
    
    if chosen_alpha is not None:
        print(f"\nChosen alpha (regularization strength): {chosen_alpha:.4f}")
    if r2 is not None:
        print(f"Pseudo-R²: {r2:.4f}")


# %%
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def chi_square_test_report(df, category_col='category', binary_col='is_significant'):
    """
    Run chi-square test of independence, compute effect size (Cramér's V),
    and return APA-style formatted results.
    """
    # Build contingency table
    contingency_table = pd.crosstab(df[category_col], df[binary_col])
    n = contingency_table.sum().sum()
    
    # Run chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Compute Cramér's V
    r, k = contingency_table.shape
    phi2 = chi2 / n
    V = np.sqrt(phi2 / (min(r, k) - 1))
    
    # Standardized residuals
    residuals = (contingency_table - expected) / np.sqrt(expected)
    residuals = pd.DataFrame(residuals, 
                             index=contingency_table.index, 
                             columns=contingency_table.columns)
    
    # APA-style report
    if p < 0.001:
        p_str = "p < .001"
    else:
        p_str = f"p = {p:.3f}".replace("0.", ".")
    
    report = (f"χ²({dof}, N = {n}) = {chi2:.2f}, {p_str}, "
              f"Cramér's V = {V:.2f}")
    
    return {
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "n": n,
        "contingency_table": contingency_table,
        "expected_freq": expected,
        "std_residuals": residuals,
        "cramers_v": V,
        "report": report
    }


# %%

tau_min=0.1
tau_max=30
acorr_min=0.1
is_good_tau_min=0
r2_d_min=0.8
r2_s_min=0.8



# Stimulated
result_M2_stimd = filter_fit_results(result_M2_standard_stimd_exc_filt,
                                     tau_min=tau_min,
                                     tau_max=tau_max,
                                     acorr_min=acorr_min,
                                     is_good_tau_min=is_good_tau_min,
                                     r2_d_min=r2_d_min,
                                     r2_s_min=r2_s_min
                                     )

result_V1_stimd = filter_fit_results(result_V1_standard_stimd_exc_filt,
                                     tau_min=tau_min,
                                     tau_max=tau_max,
                                     acorr_min=acorr_min,
                                     is_good_tau_min=is_good_tau_min,
                                     r2_d_min=r2_d_min,
                                     r2_s_min=r2_s_min
                                     )
# Non-stimulated filtered
result_M2_non_stimd = filter_fit_results(result_M2_standard_non_stimd_exc_filt,
                                         tau_min=tau_min,
                                         tau_max=tau_max,
                                         acorr_min=acorr_min,
                                         is_good_tau_min=is_good_tau_min,
                                         r2_d_min=r2_d_min,
                                         r2_s_min=r2_s_min
                                         )

result_V1_non_stimd = filter_fit_results(result_V1_standard_non_stimd_exc_filt,
                                         tau_min=tau_min,
                                         tau_max=tau_max,
                                         acorr_min=acorr_min,
                                         is_good_tau_min=is_good_tau_min,
                                         r2_d_min=r2_d_min,
                                         r2_s_min=r2_s_min
                                         )


result_M2_non_stimd_all = filter_fit_results(result_M2_standard_non_stimd_exc,
                                             tau_min=tau_min,
                                             tau_max=tau_max,
                                             acorr_min=acorr_min,
                                             is_good_tau_min=is_good_tau_min,
                                             r2_d_min=r2_d_min,
                                             r2_s_min=r2_s_min
                                             )

result_V1_non_stimd_all = filter_fit_results(result_V1_standard_non_stimd_exc,
                                             tau_min=tau_min,
                                             tau_max=tau_max,
                                             acorr_min=acorr_min,
                                             is_good_tau_min=is_good_tau_min,
                                             r2_d_min=r2_d_min,
                                             r2_s_min=r2_s_min
                                             )

# %% This is to look at distributions of taus after our cutoff


analysis_plotting_functions.plot_ecdf_comparison(result_V1_non_stimd_all['tau'],result_M2_non_stimd_all['tau'],
                                                 label1=params['general_params']['V1_lbl'], label2=params['general_params']['M2_lbl'],title='',
                                                 line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                                                 xlabel="Tau (s)",
                                                 ylabel="",
                                                 # xticks_start=0, xticks_end=8, xticks_step=2,
                                                 # yticks_start=0, yticks_end=1, yticks_step=0.2,
                                                 # xlim=[0,8],
                                                 stat_test='auto',
                                                 figsize=[5,5],
                                                 show_normality_pvals=True,
                                                 log_x=True)

# %%

group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

result_M2_non_stimd_with_stim = add_tau_column(result_M2_stimd, result_M2_non_stimd, group_cols)
result_V1_non_stimd_with_stim = add_tau_column(result_V1_stimd, result_V1_non_stimd, group_cols)

result_M2_non_stimd_all = add_tau_column(result_M2_stimd, result_M2_non_stimd_all, group_cols)
result_V1_non_stimd_all = add_tau_column(result_V1_stimd, result_V1_non_stimd_all, group_cols)

# %% ADDs parameters to dataframes for analysis



# --- result_M2_non_stimd_with_stim ---
result_M2_non_stimd_with_stim['tau_abs_diff'] = (result_M2_non_stimd_with_stim['tau'] - result_M2_non_stimd_with_stim['tau_stim']).abs()
result_M2_non_stimd_with_stim['tau_tau_abs_diff_interaction'] = result_M2_non_stimd_with_stim['tau'] * result_M2_non_stimd_with_stim['tau_abs_diff']
result_M2_non_stimd_with_stim['tau_stim_tau_abs_diff_interaction'] = result_M2_non_stimd_with_stim['tau_stim'] * result_M2_non_stimd_with_stim['tau_abs_diff']
result_M2_non_stimd_with_stim['peak_time_avg_log'] = np.log(result_M2_non_stimd_with_stim['peak_time_avg'])


# --- result_V1_non_stimd_with_stim ---
result_V1_non_stimd_with_stim['tau_abs_diff'] = (result_V1_non_stimd_with_stim['tau'] - result_V1_non_stimd_with_stim['tau_stim']).abs()
result_V1_non_stimd_with_stim['tau_tau_abs_diff_interaction'] = result_V1_non_stimd_with_stim['tau'] * result_V1_non_stimd_with_stim['tau_abs_diff']
result_V1_non_stimd_with_stim['tau_stim_tau_abs_diff_interaction'] = result_V1_non_stimd_with_stim['tau_stim'] * result_V1_non_stimd_with_stim['tau_abs_diff']
result_V1_non_stimd_with_stim['peak_time_avg_log'] = np.log(result_V1_non_stimd_with_stim['peak_time_avg'])


# --- result_M2_non_stimd_all ---
result_M2_non_stimd_all['tau_abs_diff'] = (result_M2_non_stimd_all['tau'] - result_M2_non_stimd_all['tau_stim']).abs()
result_M2_non_stimd_all['tau_tau_abs_diff_interaction'] = result_M2_non_stimd_all['tau'] * result_M2_non_stimd_all['tau_abs_diff']
result_M2_non_stimd_all['tau_stim_tau_abs_diff_interaction'] = result_M2_non_stimd_all['tau_stim'] * result_M2_non_stimd_all['tau_abs_diff']
result_M2_non_stimd_all['peak_time_avg_log'] = np.log(result_M2_non_stimd_all['peak_time_avg'])


# --- result_V1_non_stimd_all ---
result_V1_non_stimd_all['tau_abs_diff'] = (result_V1_non_stimd_all['tau'] - result_V1_non_stimd_all['tau_stim']).abs()
result_V1_non_stimd_all['tau_tau_abs_diff_interaction'] = result_V1_non_stimd_all['tau'] * result_V1_non_stimd_all['tau_abs_diff']
result_V1_non_stimd_all['tau_stim_tau_abs_diff_interaction'] = result_V1_non_stimd_all['tau_stim'] * result_V1_non_stimd_all['tau_abs_diff']
result_V1_non_stimd_all['peak_time_avg_log'] = np.log(result_V1_non_stimd_all['peak_time_avg'])




result_M2_non_stimd_with_stim['tau_diff'] = (result_M2_non_stimd_with_stim['tau'] - result_M2_non_stimd_with_stim['tau_stim'])
result_V1_non_stimd_with_stim['tau_diff'] = (result_V1_non_stimd_with_stim['tau'] - result_V1_non_stimd_with_stim['tau_stim'])

result_M2_non_stimd_all['tau_diff']       = (result_M2_non_stimd_all['tau'] - result_M2_non_stimd_all['tau_stim'])
result_V1_non_stimd_all['tau_diff']       = (result_V1_non_stimd_all['tau'] - result_V1_non_stimd_all['tau_stim'])



# %% ADDs parameters to dataframes for analysis


group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id','roi_id']

df_M2 = add_is_significant_column(result_M2_non_stimd_all, result_M2_non_stimd_with_stim, group_cols)

group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

df_M2 = add_cells_per_roi(df_M2,summary_stats_M2_standard_non_stimd_exc, group_cols)


df_M2=add_tau_medians_lengths_counts_and_category(df_M2,group_cols, tau_col='tau', tau_stim_col='tau_stim')


# prop_df_M2 = proportion_significant_by_length(df_M2, group_cols,divide_by='cells_per_roi')
prop_df_M2 = proportion_significant_by_length(df_M2, group_cols,divide_by='matching_cells_count')

value_col='peak_time_avg'
grouped_peak_time_M2=group_values_by_tau_cutoffs(df_M2, group_cols,value_col=value_col)
grouped_peak_time_M2_exploded = grouped_peak_time_M2.explode(value_col, ignore_index=True)

group_cols = ['subject_fullname', 'session_date', 'scan_number','stim_id']

M2_peak_time_by_tau_median_splits = (
    grouped_peak_time_M2_exploded
    .pivot_table(
        index=group_cols,
        columns="category",
        values=value_col,
        fill_value=np.nan
    )
    .reset_index()
)
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

# separates out traces 
grouped_traces_M2=group_traces_by_tau_cutoffs(df_M2, group_cols)
dfs_by_category_M2 = split_traces_to_2d_dfs(grouped_traces_M2,['category'])

# %% ADDs parameters to dataframes for analysis


group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id','roi_id']

df_V1 = add_is_significant_column(result_V1_non_stimd_all, result_V1_non_stimd_with_stim, group_cols)

group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

df_V1 = add_cells_per_roi(df_V1,summary_stats_V1_standard_non_stimd_exc, group_cols)
df_V1=add_tau_medians_lengths_counts_and_category(df_V1,group_cols, tau_col='tau', tau_stim_col='tau_stim')


# prop_df_V1 = proportion_significant_by_length(df_V1, group_cols,divide_by='cells_per_roi')
prop_df_V1 = proportion_significant_by_length(df_V1, group_cols,divide_by='matching_cells_count')


value_col='peak_time_avg'
grouped_peak_time_V1=group_values_by_tau_cutoffs(df_V1, group_cols,value_col=value_col)
grouped_peak_time_V1_exploded = grouped_peak_time_V1.explode(value_col, ignore_index=True)

group_cols = ['subject_fullname', 'session_date', 'scan_number','stim_id']

V1_peak_time_by_tau_median_splits = (
    grouped_peak_time_V1_exploded
    .pivot_table(
        index=group_cols,
        columns="category",
        values=value_col,
        fill_value=np.nan
    )
    .reset_index()
)
group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

# separates out traces 
grouped_traces_V1=group_traces_by_tau_cutoffs(df_V1, group_cols)
dfs_by_category_V1 = split_traces_to_2d_dfs(grouped_traces_V1,['category'])



# %% chi square results based on is_significant column 

results = chi_square_test_report(df_M2)

print("APA-style report:")
print(results["report"])

print("\nStandardized residuals:")
print(results["std_residuals"])

results = chi_square_test_report(df_V1)

print("APA-style report:")
print(results["report"])

print("\nStandardized residuals:")
print(results["std_residuals"])
# %%

# %%  Figure 6D
# plot peak time by tau spilts


plt.figure()
plot_2x2_category_heatmap_from_pivot(M2_peak_time_by_tau_median_splits,
                                     cbar_label='Peak times (s)')

plt.figure()
plot_2x2_category_heatmap_from_pivot(V1_peak_time_by_tau_median_splits,
                                     cbar_label='Peak times (s)')


M2_peak_time_by_tau_median_splits=M2_peak_time_by_tau_median_splits.drop(columns=['subject_fullname', 'session_date','scan_number'])
V1_peak_time_by_tau_median_splits=V1_peak_time_by_tau_median_splits.drop(columns=['subject_fullname', 'session_date','scan_number'])

# %% Figure 6A calculating proportions

group_cols = ['subject_fullname', 'session_date', 'scan_number', 'stim_id']

#

result_df = categorize_and_normalize_from_population_medians_2(result_M2_non_stimd_with_stim,
                                      summary_stats_M2_standard_non_stimd_exc,
                                      result_M2_non_stimd_all,
                                      group_cols,
                                      roi_col='cells_per_roi',
                                      denominator='total_in_tau_range')


result_df_V1 = categorize_and_normalize_from_population_medians_2(result_V1_non_stimd_with_stim,
                                      summary_stats_V1_standard_non_stimd_exc,
                                      result_V1_non_stimd_all,
                                      group_cols,
                                      denominator='total_in_tau_range'
)



group_cols = ['subject_fullname', 'session_date', 'scan_number','stim_id']

# Pivot the result_df: rows = group, columns = category, values = normalized_count
plot_df = result_df.pivot_table(
    index=group_cols, 
    columns='category', 
    values='normalized_count', 
    # fill_value=np.nan
    fill_value=0
)

plot_df_V1 = result_df_V1.pivot_table(
    index=group_cols, 
    columns='category', 
    values='normalized_count', 
    # fill_value=np.nan
    fill_value=0
)

plt.figure()
plot_2x2_category_heatmap_from_pivot(plot_df,mean_or_median='mean',
                                        vmin=0.002,vmax=0.012
                                     )

plt.figure()
plot_2x2_category_heatmap_from_pivot(plot_df_V1,mean_or_median='mean',
                                        vmin=0.002,vmax=0.012
                                     )




# %%  FIGURE 6E

feature_cols = [ 'tau', 'tau_stim', 'tau_abs_diff']
# results, coef_df, pval_df, coef_se_df, used_alpha, r2_final, see_linear

results, coef_df_M2, pval_df_M2,coef_se_df_M2, used_alpha, r2_final, see_linear = compare_regularization_with_significance(result_M2_non_stimd_with_stim,
                                          feature_cols,
                                            'peak_time_avg_log',
                                            # 'tau_abs_diff',
                                           # 'peak_time_array_mean_trial',
                                            # alpha_ridge=1,
                                            lasso_alpha=.005,
                                            # lasso_alphas_cv=np.logspace(-5, -1, 50),
                                           include_interactions=False,
                                            interaction_pairs=[('tau','tau_abs_diff')]
                                           )


results, coef_df_V1, pval_df_V1,coef_se_df_V1, used_alpha, r2_final, see_linear = compare_regularization_with_significance(result_V1_non_stimd_with_stim,
                                          feature_cols,
                                            'peak_time_avg_log',
                                            # 'tau_abs_diff',
                                           # 'peak_time_array_mean_trial',
                                            # alpha_ridge=1,
                                            lasso_alpha=.005,
                                            # lasso_alphas_cv=np.logspace(-5, -1, 50),
                                           include_interactions=False,
                                            interaction_pairs=[('tau','tau_abs_diff')]
                                           )

plot_single_beta_bars(
    coef_df_M2,
    coef_df_V1,
    pval_df1= pval_df_M2, pval_df2=pval_df_V1,
    coef_se_df1=coef_se_df_M2, coef_se_df2=coef_se_df_V1,
    dataset_labels=(params['general_params']['M2_lbl'],params['general_params']['V1_lbl']), 
    colors=(params['general_params']['M2_cl'],params['general_params']['V1_cl']), 
    figsize=(3, 4),
    bar_width=0.35,
    bar_spacing=0.05,
    feature_order=['tau','tau_stim','tau_abs_diff'],
    xlim=(-1,1)
    # selected_features=None,
)
# %% LOGISTIC REGRESSION



# %%Figure 6B

coef_boot_V1 = bootstrap_logistic_regression(
    df=df_V1,
    feature_cols=["tau", "tau_stim", "tau_abs_diff"],
    target_col="is_significant",

    n_bootstrap=1000,
    random_state=42
)


coef_boot_M2 = bootstrap_logistic_regression(
    df=df_M2,
    feature_cols=["tau", "tau_stim", "tau_abs_diff"],
    target_col="is_significant",
    n_bootstrap=1000,
    random_state=42
)

# %% Figure 6B

zero_crosses_V1 = proportion_crossing_value(coef_boot_V1,value=0)

zero_crosses_M2 = proportion_crossing_value(coef_boot_M2,value=0)


dist_crosses_V1 = proportion_crossing_median(coef_boot_V1,coef_boot_M2)

dist_crosses_M2 = proportion_crossing_median(coef_boot_M2,coef_boot_V1)


plot_grouped_bar_with_error_vertical(
    coef_boot_M2, 
    coef_boot_V1,
    dataset_labels=(params['general_params']['M2_lbl'],params['general_params']['V1_lbl']), 
    colors=(params['general_params']['M2_cl'],params['general_params']['V1_cl']), 
    figsize=(3, 4),
    bar_width=0.35,
    bar_spacing=0.05,
    feature_order=['tau','tau_stim','tau_abs_diff','tau_x_tau_stim'],
    # feature_order=['tau','tau_stim','tau_abs_diff','tau_x_tau_abs_diff','tau_stim_x_tau_abs_diff','tau_x_tau_stim'],
    selected_features=None,
    crosses_df1=zero_crosses_M2,
    crosses_df2=zero_crosses_V1,
    xlim=(-1,1)
    )

# %%

analysis_plotting_functions.plot_ecdf_comparison(coef_boot['tau'].to_numpy('float32'),coef_boot['tau_stim'].to_numpy('float32'),
                                                
                      label1='tau', label2='tau_stim',title='',
                      line_color1=params['general_params']['V1_cl'],line_color2=params['general_params']['M2_cl'],
                      xlabel="Peak width (s)",
                      ylabel="Directly stimulated cells",
                      xlim=([-.5,.5]),
                     
                     stat_test='auto',
                     figsize=[3,4],
                     show_normality_pvals=True)

# %%
analysis_plotting_functions.general_scatter_plot(df_filtered['tau'],df_filtered['tau_abs_diff'])


# %% Figure S4


time_array,m2_avgs_array_2d=analyzeEvoked2P.align_averaged_traces_from_lists(result_M2_standard_non_stimd_exc_filt, trace_col='averaged_traces_all', time_col='time_axes')

# # V1
# df_clean = dfs_by_category_V1['Long tau stim & Long tau'].drop(columns=['category']).reset_index(drop=True)
# df_clean_2= dfs_by_category_V1['Short tau stim & Long tau'].drop(columns=['category']).reset_index(drop=True)
# df_clean_3= dfs_by_category_V1['Short tau stim & Short tau'].drop(columns=['category']).reset_index(drop=True)
# df_clean_4= dfs_by_category_V1['Long tau stim & Short tau'].drop(columns=['category']).reset_index(drop=True)

# # # M2
df_clean = dfs_by_category_M2['Long tau stim & Long tau'].drop(columns=['category']).reset_index(drop=True)
df_clean_2= dfs_by_category_M2['Short tau stim & Long tau'].drop(columns=['category']).reset_index(drop=True)
df_clean_3= dfs_by_category_M2['Short tau stim & Short tau'].drop(columns=['category']).reset_index(drop=True)
df_clean_4= dfs_by_category_M2['Long tau stim & Short tau'].drop(columns=['category']).reset_index(drop=True)

# %%

analysis_plotting_functions.plot_mean_trace_multiple_dataframe_input(
                                                                    result1=df_clean,result2=df_clean_2,
                                                                    result3=df_clean_3,result4=df_clean_4,
                                                                     norm_mean=False,
                                                                     norm_type_row='minmax',
                                                                     # norm_rows=True,
                                                                     y_limits=[-.2, 1.0],
                                                                     x_limits=[-3, 10],
                                                                     legend_names=['Long tau stim & Long tau', 
                                                                                   'Short tau stim & Long tau',
                                                                                    'Short tau stim & Short tau',
                                                                                     'Long tau stim & Short tau'],
                                                                     line_colors=["#984EA3",
                                                                                   "#E41A1C",
                                                                                   'k',
                                                                                   "#228B22"
                                                                                   ],
                                                                     fill_alpha=0.3,
                                                                     xticks=np.linspace(-2,10,7),
                                                                     yticks=np.linspace(0,1,6),
                                                                     xlabel="Time (s)",
                                                                     ylabel="dF/F",
                                                                     time_array=time_array,
                                                                     exclude_windows=[(-0.15,.25)]
                                                                     )

# %%  Figure 6C
 # Heat maps by tau 

df=df_M2
# df=df_V1

subset=df[df['is_significant'] == 1]
subset=subset[subset['tau_stim']<subset['tau_median']]
subset=subset[subset['tau']<subset['tau_median']]



analysis_plotting_functions.prepare_and_plot_heatmap(subset,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=0,
    vmax=1,
    # start_index=0,
    # sampling_interval=0.032958316,
    exclude_window=None,
    cbar_shrink=0.2,  # shrink colorbar height to 60%
    invert_y=True,
    cmap='bone',
    norm_type='minmax',
    figsize=(6,6),
    xlim=[-3,10])



subset=df[df['is_significant'] == 1]
subset=subset[subset['tau_stim']<subset['tau_median']]
subset=subset[subset['tau']>subset['tau_median']]


analysis_plotting_functions.prepare_and_plot_heatmap(subset,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=0,
    vmax=1,
    # start_index=0,
    # sampling_interval=0.032958316,
    exclude_window=None,
    cbar_shrink=0.2,  # shrink colorbar height to 60%
    invert_y=True,
    cmap='bone',
    norm_type='minmax',
    figsize=(6,6),
    xlim=[-3,10])


subset=df[df['is_significant'] == 1]
subset=subset[subset['tau_stim']>subset['tau_median']]
subset=subset[subset['tau']<subset['tau_median']]


analysis_plotting_functions.prepare_and_plot_heatmap(subset,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=0,
    vmax=1,
    # start_index=0,
    # sampling_interval=0.032958316,
    exclude_window=None,
    cbar_shrink=0.2,  # shrink colorbar height to 60%
    invert_y=True,
    cmap='bone',
    norm_type='minmax',
    figsize=(6,6),
    xlim=[-3,10])



subset=df[df['is_significant'] == 1]
subset=subset[subset['tau_stim']>subset['tau_median']]
subset=subset[subset['tau']>subset['tau_median']]


analysis_plotting_functions.prepare_and_plot_heatmap(subset,trace_column='averaged_traces_all',
    sort_by='peak_time_avg',
    vmin=0,
    vmax=1,
    # start_index=0,
    # sampling_interval=0.032958316,
    exclude_window=None,
    cbar_shrink=0.2,  # shrink colorbar height to 60%
    invert_y=True,
    cmap='bone',
    norm_type='minmax',
    figsize=(6,6),
    xlim=[-3,10])


