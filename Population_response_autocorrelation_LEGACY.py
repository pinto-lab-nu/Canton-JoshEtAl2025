

# %% Collecting data from the dj pipeline

# v1_avgs = analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff',signif_only=False, which_neurons='non_stimd')
# m2_avgs = analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff',signif_only=False, which_neurons='non_stimd')
# %% Collecting data from the dj pipeline

v1_avgs_sig = analyzeEvoked2P.get_avg_trig_responses('V1', params=opto_params, expt_type='standard', resp_type='dff',signif_only=True, which_neurons='non_stimd')
m2_avgs_sig = analyzeEvoked2P.get_avg_trig_responses('M2', params=opto_params, expt_type='standard', resp_type='dff',signif_only=True, which_neurons='non_stimd')
# %%

# df_nested = analyzeEvoked2P.group_trig_responses_by_expt(m2_avgs_sig['roi_keys'],m2_avgs_sig['stim_ids'],m2_avgs_sig['trig_dff_avgs'])


# Find the minimum length among all arrays


# min_len = min(len(arr) for arr in result_M2_standard_non_stimd_filt['averaged_traces_all'])
# # Truncate all arrays to that length
# m2_avgs_array_2d = np.vstack([arr[:min_len] for arr in result_M2_standard_non_stimd_filt['averaged_traces_all']])


df_nested = analyzeEvoked2P.group_trig_responses_by_expt( result_M2_standard_non_stimd_filt['roi_keys'], result_M2_standard_non_stimd_filt['stim_id'], result_M2_standard_non_stimd_filt['averaged_traces_all'])


df_nested['mean_trig_dff_avg'] = df_nested['trig_dff_avgs'].apply(lambda arr_list: np.nanmean(np.stack(arr_list, axis=0), axis=0))
# %%

# plt.plot(df_nested['mean_trig_dff_avg'][0])'
# plt.plot(df_nested['trig_dff_avgs'][21][1])
# %%

a=bootstrap_avgs_m2

a_df = pd.DataFrame(a.tolist())
a_df_short=a_df.iloc[:, 100:]

acorr=calculate_autocorrelations_df(a_df_short, signal_range=(0, 288), max_lags=280)

# %%
# plt.plot(acorr.iloc[3])

mean_acorr = acorr.mean(axis=0, skipna=True)
plt.plot(mean_acorr)
plt.xlabel('Lag')
plt.ylabel('Mean autocorrelation')
plt.title('Average Autocorrelation Across Signals')

# %%


df_nested = group_trig_responses_by_expt(v1_avgs_sig)
df_nested['mean_trig_dff_avg'] = df_nested['trig_dff_avgs'].apply(lambda arr_list: np.nanmean(np.stack(arr_list, axis=0), axis=0))
# %%

plt.plot(df_nested['mean_trig_dff_avg'][100])
# %%
# a=df_nested['mean_trig_dff_avg']

a=bootstrap_avgs_v1
a_df = pd.DataFrame(a.tolist())
a_df_short=a_df.iloc[:, 100:]

acorr_v1=calculate_autocorrelations_df(a_df_short, signal_range=(0, 288), max_lags=280)

# %%
# plt.plot(acorr_v1.iloc[100])

mean_acorr = acorr_v1.mean(axis=0, skipna=True)
plt.plot(mean_acorr)
plt.xlabel('Lag')
plt.ylabel('Mean autocorrelation')
plt.title('Average Autocorrelation Across Signals')


# %%
fit_results_M2, good_acorr, valid_rows = analyzeSpont2P.fit_exponentials_from_df(acorr, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=250)
# %%
df_si, df_se_M2, df_di_M2, df_de_M2, best_type = classify_fit_results_simple(
    fit_results=fit_results_M2
)

# %%
fit_results_v1, good_acorr, valid_rows = analyzeSpont2P.fit_exponentials_from_df(acorr_v1, exclude_low_acorr=True, acorr_threshold=0.1, max_lag=250)

# %%
df_si, df_se_V1, df_di_V1, df_de_V1, best_type = classify_fit_results_simple(
    fit_results=fit_results_v1
)

print(f"The best fitting model type based on average R² is: {best_type}")

# %%


df_se_V1 = df_se_V1[df_se_V1["tau"] <= 200]
df_se_M2 = df_se_M2[df_se_M2["tau"] <= 200]

df_se_V1 = df_se_V1[df_se_V1["tau"] >= 2]
df_se_M2 = df_se_M2[df_se_M2["tau"] >= 2]

df_se_V1 = df_se_V1[df_se_V1["r2"] >= 0.8]
df_se_M2 = df_se_M2[df_se_M2["r2"] >= 0.8]

# df_se_V1 = df_se_V1[df_se_V1["event_rate"] >= 0.1]
# df_se_M2 = df_se_M2[df_se_M2["event_rate"] >= 0.1]

# df_se_V1 = df_se_V1[df_se_V1["snr"] >= 1]
# df_se_M2 = df_se_M2[df_se_M2["snr"] >= 1]

df_se_V1 = df_se_V1[df_se_V1["acorr_1_index"] >= .2]
df_se_M2 = df_se_M2[df_se_M2["acorr_1_index"] >= .2]
# %%

analysis_plotting_functions.plot_ecdf_comparison(df_se_M2.tau/30,df_se_V1.tau/30,log_x=False)
