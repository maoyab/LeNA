import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, theilslopes, kendalltau

from EE_model import*
from RF_model import*



def open_lnc_data(data_f, filter_lnc=True):
    df_ts_ = pd.read_csv(data_f)
    if filter_lnc:
        df_ts_ = df_ts_[(df_ts_['yrange']>=10) & (df_ts_['count']>=6) 
                        & (df_ts_['MODIS_Lai'].notna()) & (df_ts_['SATPSI'].notna())
                        & (df_ts_['LNC'].notna())]
        df_ts_ = df_ts_[(df_ts_['year']>=1995) & (df_ts_['year']<=2016)]
    else:
        df_ts_ = df_ts_[(df_ts_['yrange']>=10) & (df_ts_['count']>=6) 
                        & (df_ts_['MODIS_Lai'].notna()) & (df_ts_['SATPSI'].notna())]
        df_ts_ = df_ts_[(df_ts_['year']>=1995) & (df_ts_['year']<=2016)]


    df_ts_['TC_AI_gs_LTavg'] = df_ts_['TC_pet_gs_LTavg'] / df_ts_['TC_pre_gs_LTavg']
    df_ts_['TC_WI_gs_LTavg'] = 1 / df_ts_['TC_AI_gs_LTavg'] 
    df_ts_['CO2_LTavg'] = np.mean(df_ts_['CO2'])
    df_ts_['CO2_yi'] = df_ts_['CO2']
    df_ts_['surface_pressure'] = df_ts_['elev_pa']

    cm = []
    for wi, lat in zip(df_ts_['TC_WI_gs_LTavg'], df_ts_['Lat']):
        if lat >=60:
            cm.append(1)
        elif (wi< 0.45) and (lat<=44):
            cm.append(3)
        else:
            cm.append(2)
    df_ts_['climate_zone'] = cm
    # Boreal, q=1'
    # Temperate, q=2
    # Mediterannean, q=3
    # q exponent of water stress function
    return df_ts_


def standardize_to_common_mean(y_var, df_ts_, common_mean = None, print_m=False, metric='mean'):
    var_m = []
    for ix, dfi in df_ts_.iterrows():
        lat = dfi['Lat']
        lon = dfi['Lon']
        sp = dfi['sp']
        if metric == 'mean':
            var_m.append(np.nanmean(df_ts_[(np.round(df_ts_['Lat'], 8)==np.round(lat, 8)) 
                      & (np.round(df_ts_['Lon'], 8)==np.round(lon, 8)) & (df_ts_['sp']==sp)][y_var]))
        elif metric == 'median':
            var_m.append(np.nanmedian(df_ts_[(np.round(df_ts_['Lat'], 8)==np.round(lat, 8)) 
                      & (np.round(df_ts_['Lon'], 8)==np.round(lon, 8)) & (df_ts_['sp']==sp)][y_var]))
    df_ts_['%s_m' % y_var] = var_m
    if common_mean is None:
        common_mean = np.nanmean(df_ts_['%s_m' % y_var])
    if print_m:
        print('common mean:', common_mean)
    df_ts_['%s_stdm' % y_var] = df_ts_[y_var] / df_ts_['%s_m' % y_var] * common_mean
    return df_ts_


def trend_summary_df(y_var, df, min_ny = 5): 
    df_ts_s = df[np.isnan(df[y_var])==0]
    locs = df_ts_s.groupby(['Lat', 'Lon', 'sp']).mean().index
    
    trend_results = []
    for lat, lon, sp in locs:
        dfi = df_ts_s[(np.round(df_ts_s['Lat'], 8)==np.round(lat, 8)) 
                      & (np.round(df_ts_s['Lon'], 8)==np.round(lon, 8)) & (df_ts_s['sp']==sp)]

        if len(dfi.index)>=min_ny:
            dfi = dfi[(np.isnan(dfi[y_var])==0) & np.isnan(dfi['year'])==0]
            tt, b, c, d = theilslopes(dfi[y_var].values, dfi['year'].values, 0.95)
            rho = kendalltau(dfi['year'], dfi[y_var])
            if (rho[0]>0) and (rho[1]<0.05):
                sigi=1
            elif (rho[0]<0) and (rho[1]<0.05):
                sigi = -1
            else:
                sigi = 0
            d = tt/np.nanmean(dfi[y_var].values) * 100 * 22 #  relative change over 1995-2016 period
            dc = tt/np.nanmean(dfi[y_var].values) * 100 * 22 /44 * 50 #  relative to 50ppm Co2 increase
            trend_results.append([lat, lon, sp, tt, sigi, rho[1], d, dc, len(dfi.index)])
            
        else:
            trend_results.append([lat, lon, sp, np.nan, np.nan, np.nan, np.nan, np.nan, len(dfi.index)])
    
    lat, lon, sp, slope, sig, slope_p, delta, deltaC, n = zip(*trend_results)
    trend_results = {'Lat': lat, 'Lon':lon, 'sp': sp, 'n_years': n,
                    'trend %s' % y_var: slope, 
                    'delta %s' % y_var: delta, 
                    'deltaC %s' % y_var: deltaC, 
                    'trend_p %s' % y_var: slope_p}
    return pd.DataFrame(trend_results)


def regional_trend_summary(y_vars, y_vars_n, df):
    period = range(1995, 2017)
    y_units='$mg$ $g^{-1}$ / $year$'
    
    df = df[(df['year']>=period[0]) & (df['year']<=period[-1])][y_vars +[ 'year', 'CO2']].dropna()
    delta_co2 = np.max(df['CO2'])* 10 ** 6 - np.min(df['CO2'])* 10 ** 6
    delta_y = (np.max(df['year']) - np.min(df['year']) + 1)
    print('delta CO2', delta_co2)
    print('delta Year', delta_y)
    summary = []
    for y_var in y_vars:
        print('')
        print(y_var)
        [rho_k, p_k] = kendalltau(df['year'], df[y_var])
        print('kendalltau', rho_k, 'p-',p_k)
        trend, b, a1, a2 = theilslopes(df[y_var], df['year'], 0.95)
        trend_err = trend - a1
        print('theilslope-year', trend * 1000, 'err', trend_err * 1000)
        trendC, b, a1, a2 = theilslopes(df[y_var], df['CO2']*10**6, 0.95)
        trendC_err = trendC - a1
        print('theilslope-CO2', trendC / np.mean(df[y_var])*100 * 50, 'err', trendC_err / np.mean(df[y_var])*100 * 50)
        delta = trend / np.mean(df[y_var]) * 100 * delta_y 
        delta_err = (trend - a1)  / np.mean(df[y_var]) * 100 * delta_y
        print('delta', delta * delta_co2 / 50, 'err', delta_err * delta_co2 / 50)

        y_obs = [np.nanmedian(df[df.year==yi][y_vars[0]]) for yi in period]
        y_model = [np.nanmedian(df[df.year==yi][y_var]) for yi in period]
        [rho_r, p_r] = pearsonr(y_obs, y_model)
        
        [rho_a, p_a] = pearsonr(df[y_vars].dropna()[y_vars[0]], df[y_vars].dropna()[y_var])

        summary.append([trend * 1000, trend_err * 1000, 
                        delta * delta_co2 / 50, delta_err * delta_co2 / 50,
                        rho_k, rho_r, rho_a, p_k, p_r, p_a])
    df_names = ['trend', 'trend_err', 'delta', 'delta_err', 'rho_k', 'rh_r', 'rho_a', 'p_k', 'p_r', 'p_a']
    summary = zip(*summary)
    trend_df = pd.DataFrame({'var': y_vars, 'name': y_vars_n})
    for y, vv in zip(df_names, summary):
        trend_df[y] = vv 
    return trend_df


def add_swb_model(df_ts_, variant):
    df_ts_['rf_alpha'] = df_ts_['ERA5_rf_alpha_%s' % variant]
    df_ts_['rf_lambda'] = df_ts_['ERA5_rf_lambda_%s' % variant]
    df_ts_['pet'] = df_ts_['TC_pet_%s' % variant] # alternative with theoretical Gc*D to account for Co2 etc..
    df_ts_['D'] = df_ts_['TC_vpd_%s' % variant] # Pa
    df_ts_['Fpar'] = df_ts_['MODIS_Fpar_gs']
    df_ts_['LAI'] = df_ts_['MODIS_Lai_gs']

    swb_vv = [cal_swb_0(dd, q=dd['climate_zone']) for ix, dd in df_ts_.iterrows()]
    swb_names = ['mean_sm', 's_stressB', #'d_stressB', 
                's_epsilon',  #'d_epsilon', 
                'pi_F', 'T_ww',
                'tet_part', 'mean_ef', 'mean_tf', 'mean_di', 'rf']
    for nn, vv in zip(swb_names, zip(*swb_vv)):          
        df_ts_['%s_%s_canopy_swb' % (nn, variant)] = vv
    return df_ts_


def add_p_model(df_ts_, trend_result_all, variant, common_mean, sunlit=True, metric='mean'):
    [model_tag, clim_v, co2_v, stressB] = variant
    pm_names = ['g1', 'chi', 'lue', 'iwue',  'gpp', 'vcmax', 'vcmax25', 'gs', 'nue']

    df_ts_['tmp'] = df_ts_['TC_tmean_%s' % clim_v]     # [C]
    df_ts_['vpd'] = df_ts_['TC_vpd_%s' % clim_v]       # [Pa]
    df_ts_['par'] = df_ts_['TC_par_%s' % clim_v]       # [mol/m2/s]
    df_ts_['pa'] = df_ts_['elev_pa']  # [Pa] df_locmi['ERA5_pa_gs_LTavg']
    df_ts_['CO2'] = df_ts_[co2_v]
    df_ts_['stressB'] = stressB
    if sunlit:
        df_ts_['fpar'] = 0.91             # [-] 
    else:
        df_ts_['MODIS_Fpar_gs']

    results = [cal_opt_states(dd['tmp'], dd['vpd'], dd['pa'], dd['CO2'], dd['par'],  
                        fapar=dd['fpar'], soilmstress=dd['stressB'], nu_star=None, g1 = None, 
                        k0=k0_Pmodel, beta=cost_beta_stocker19,
                        ref='Smith2019', print_summary=None) for i, dd in df_ts_.iterrows()]
    for k, r in zip(pm_names, zip(*results)):
        df_ts_['%s_%s_%s' % (k, clim_v, model_tag)] = r

    df_ts_ = standardize_to_common_mean('vcmax25_%s_%s' % (clim_v, model_tag), 
                                        df_ts_, common_mean = common_mean, metric=metric)

    trend_result_model = trend_summary_df('vcmax25_%s_%s_stdm' % (clim_v, model_tag),  df_ts_)
    for k in trend_result_model:
        if k not in ['Lat', 'Lon', 'sp', 'n_years']:
            trend_result_all[k] = trend_result_model[k]

    trend_result_model = trend_summary_df('nue_%s_%s' % (clim_v, model_tag),  df_ts_)
    for k in trend_result_model:
        if k not in ['Lat', 'Lon', 'sp', 'n_years']:
            trend_result_all[k] = trend_result_model[k]

    return trend_result_all, df_ts_


def add_rf_model(df_ts_, trend_result_all, features_list, target_name = 'LNC_stdm', rfi = 1):
    variant = 'gs_yi'
    df_ts_['tmp'] = df_ts_['TC_tmean_%s' % variant]     # [C]
    df_ts_['vpd'] = df_ts_['TC_vpd_%s' % variant]       # [Pa]
    df_ts_['par'] = df_ts_['TC_par_%s' % variant]       # [mol/m2/s]
    df_ts_['stressB'] = df_ts_['s_stressB_%s_canopy_swb' % variant]
    df_ts_['CO2'] = df_ts_['CO2_yi']


    k_results, df_ts_ = run_random_forest_kfold(df_ts_, features_list, target_name, 
                                                    n_splits=5, model_variant='_m%s' %rfi)

    trend_result_model = trend_summary_df('RF_LNC_m%s' % rfi,  df_ts_)
    for k in trend_result_model:
        if k not in ['Lat', 'Lon', 'sp', 'n_years']:
            trend_result_all[k] = trend_result_model[k]

    vi = []
    for ki, k_results_i in enumerate(k_results):
        vi.append([ii[1] for ii in k_results_i[3]])
    vi = list(zip(*vi))
    names = [n[0] for n in k_results[0][3]]
    rf_vi = []
    print('')
    for vii, ni in zip(vi, names):
        rf_vi.append([ni, np.mean(vii), np.std(vii)/5])
        print(ni, np.round(np.mean(vii), 2), np.round(np.std(vii)/5, 3))

    return trend_result_all, df_ts_, k_results, rf_vi


def add_p_model_CMIP(df_ts_, variant, common_mean, sunlit=True, bias_corr_clim=0, source_id=None):
    [model_tag, cmip_label, co2_v, stressB] = variant
    pm_names = ['g1', 'chi', 'lue', 'iwue',  'gpp', 'vcmax', 'vcmax25', 'gs', 'nue']
    
    if bias_corr_clim == 0:
        b_tas = 1
        b_vpd = 1
        b_par = 1
    else:
        b_tas = df_ts_['TC_tmean_gs_LTavg'] / df_ts_['tas_historical_%s_gs_LTavg'% source_id]
        b_vpd = df_ts_['TC_vpd_gs_LTavg'] / df_ts_['vpd_historical_%s_gs_LTavg'% source_id]
        b_par = df_ts_['TC_par_gs_LTavg'] / df_ts_['par_historical_%s_gs_LTavg'% source_id]
        print(b_tas)
    
    df_ts_['tmp'] = df_ts_['tas_%s' % cmip_label] * b_tas       # [C]
    df_ts_['vpd'] = df_ts_['vpd_%s' % cmip_label] * b_vpd       # [Pa]
    df_ts_['par'] = df_ts_['par_%s' % cmip_label] * b_par       # [mol/m2/s]
    df_ts_['pa'] = df_ts_['elev_pa']                    # [Pa]
    df_ts_['stressB'] = stressB
    df_ts_['CO2'] = df_ts_[co2_v]
    if sunlit:
        df_ts_['fpar'] = 0.91             # [-] 
    else:
        df_ts_['MODIS_Fpar_gs']

    results = [cal_opt_states(dd['tmp'], dd['vpd'], dd['pa'], dd['CO2'], dd['par'],  
                        fapar=dd['fpar'], soilmstress=dd['stressB'], nu_star=None, g1 = None, 
                        k0=k0_Pmodel, beta=cost_beta_stocker19,
                        ref='Smith2019', print_summary=None) for i, dd in df_ts_.iterrows()]
    for k, r in zip(pm_names, zip(*results)):
        df_ts_['%s_%s_%s' % (k, cmip_label, model_tag)] = r

    df_ts_ = standardize_to_common_mean('vcmax25_%s_%s' % (cmip_label, model_tag), 
                                        df_ts_, common_mean = common_mean)

    return df_ts_


def fig_1_trend(y_vars, y_vars_n, df, figname= None,
                       colors =['gray', 'tomato'], ylim=None, markers=['s', 'o']):
    fig = plt.figure(figsize=(8.5, 4))


    
    ax1 = plt.subplot2grid((2, 4), (0, 0), colspan=3, rowspan=2)
    ax2 = plt.subplot2grid((2, 4), (0, 3), colspan=1, rowspan=1)
    ax3 = plt.subplot2grid((2, 4), (1, 3), colspan=1, rowspan=1)
    
    period = range(1995, 2017)
    y_units='$μg$ $g^{-1}$ $year^{-1}$ '
    
    df = df[(df['year']>=period[0]) & (df['year']<=period[-1])][y_vars +[ 'year',]].dropna()
    for y_var, name, color, marker in zip(y_vars, y_vars_n, colors, markers):

        if name == 'Random forest':
            model_abb = 'RF'
        else:
            model_abb = 'Th.'

        y_b = [df[df.year==yi][y_var] for yi in period]
        y_bm = [np.nanmedian(yi) for yi in y_b]
        y_l = [np.nanmedian(yi) - np.nanstd(yi) / np.sqrt(np.size(yi)) if len(yi)>2 else np.nan for yi in y_b]
        y_u = [np.nanmedian(yi) + np.nanstd(yi) / np.sqrt(np.size(yi)) if len(yi)>2 else np.nan for yi in y_b]
        
        rho_k = kendalltau(df['year'], df[y_var])
        trend, b, a1, a2 = theilslopes(df[y_var], df['year'], 0.95)
        trend_err = trend - a1
        delta = trend / np.mean(df[y_var]) * 100 * (period[-1] - period[0] + 1) * 44 / 50
        delta_err = (trend - a1)  / np.mean(df[y_var]) * 100 * (period[-1] - period[0] + 1) * 44 / 50
        trend_label = '%s (%d $\mp$ %d %s)' % (name,np.round(trend*1000), np.round(trend_err*1000), y_units)

        ax1.fill_between(period, y_l, y_u, color=color, alpha=0.15)
        ax1.plot(period, y_bm, color=color, marker=marker, linestyle=':', label=trend_label)
        
        ax1.plot(df['year'].values, trend * df['year'].values + b, color=color)

    ax1.set_ylabel('Leaf nitrogen concentration ($mg$ $g^{-1}$)')
    if ylim is None:
        ylim = [np.percentile(df[y_var], 3), np.percentile(df[y_var], 97)]
    ax1.set_ylim(ylim)
    ax1.set_yticks([17, 18, 19])
    ax1.set_xticks(np.linspace(period[0], period[-1], 6))
    ax1.set_xticklabels([np.int32(ti) for ti in np.linspace(period[0], period[-1], 6)])
    ax1.set_xlim([period[0]-0.25, period[-1]+0.25])

    ax1.legend(frameon=False, loc='lower left')
    ax1.text(.95, .07, 'A', ha='left', va='top', transform=ax1.transAxes, fontsize=12)
    

    for y_var, color, name in zip(y_vars, colors, y_vars_n):
        if y_var != y_vars[0]:
            y_obs = [np.nanmedian(df[df.year==yi][y_vars[0]]) for yi in range(1995, 2017)]
            y_model = [np.nanmedian(df[df.year==yi][y_var]) for yi in range(1995, 2017)]
            [rho_r, p] = pearsonr(y_obs, y_model)
            plot_label = r'rho=%-5.3f (p%5.3f)' % (rho_r, p)
            ax2.scatter(y_obs, y_model, label=plot_label, color=color, s=17)
            ax2.set_ylabel('%s regional $LNC$'% model_abb)
            ax2.set_xlabel('Obs. regional $LNC$')
            print(plot_label)
    ax2.set_ylim([ylim[0]+0.5, ylim[1]-0.5])
    ax2.set_xlim([ylim[0]+0.5, ylim[1]-0.5])
    ax2.set_yticks([17, 18, 19])
    ax2.set_xticks([17, 18, 19])
    ax2.plot(ylim, ylim,color='k', linestyle='--', lw=0.75)
    #ax2.legend(frameon=False, loc='upper right')
    ax2.text(.85, .15, 'B', ha='left', va='top', transform=ax2.transAxes, fontsize=12)

    for y_var, color, name in zip(y_vars, colors, y_vars_n):
        if y_var != y_vars[0]:
    
            [rho_a, p] = pearsonr(df[y_vars].dropna()[y_vars[0]], 
                        df[y_vars].dropna()[y_var])
            plot_label = r'rho: %-5.3f (p%-5.3f)' % (rho_a, p)
            ax3.scatter(df[y_vars].dropna()[y_vars[0]], 
                        df[y_vars].dropna()[y_var],
                        color=color, s=5, alpha=0.2,
                       label=plot_label)
            print(plot_label)
    ymin =  ylim[0] - 2 #np.percentile(df[y_vars].dropna()[y_vars[0]], 1)
    ymax = ylim[1] + 2 #np.percentile(df[y_vars].dropna()[y_vars[0]], 99)
    ax3.plot([ymin, ymax], [ymin, ymax], color='k', linestyle='--', lw=0.75)
    ax3.set_xlim([ymin, ymax])
    ax3.set_ylim([ymin, ymax])
    ax3.set_xticks([16, 18, 20])
    ax3.set_yticks([16, 18, 20])
    ax3.set_ylabel('%s plot-level $LNC$' % model_abb)
    ax3.set_xlabel('Obs. plot-level $LNC$')
    ax3.text(.85, .15, 'C', ha='left', va='top', transform=ax3.transAxes, fontsize=12)
    #ax3.legend(frameon=False, loc='upper right')
    
    plt.tight_layout()

    if figname is not None:
         plt.savefig(figname)

    plt.show()


def fig_2_trend_variants(trend_df, rf_vi, y_vars_n, figname=None):

    x = range(1, len(y_vars_n)+1)
    y = trend_df['trend']
    yerr= trend_df['trend_err']
    colors = ['silver', 'OliveDrab', 'DarkGreen', 'DarkGreen', 'DarkGreen', 'DarkGreen']
    alphas = [0.75, 0.75 , 1, 0.75, 0.5, 0.25,  0.75]
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)

    print('Model Variant Trends')
    for tt, xi, yi, yerri, ci, ai in zip(trend_df['name'], x, y, yerr, colors, alphas):
        ax1.bar(xi, yi,  yerr= yerri, color = ci, alpha=ai)
        print(tt, yi)
    ax1.set_xticks(x)
    ax1.set_xticklabels(trend_df['name'], rotation=30, ha='right')
    ax1.set_ylabel('$LNC$ trend ($μg$ $g^{-1}$ $year^{-1}$)')
    ax1.text(.95, .1, 'A', ha='left', va='top', transform=ax1.transAxes, fontsize=12)

    print('')
    print('RF Feature importance')
    labels = ['air\ntemp', 'VPD', 'PAR', 'water\nstress', 'CO$_2$']
    labels_o = ['$PAR$', '$T_a$', '$VPD$', 'water\nstress', 'CO$_2$']
    for (ni, vi, verr), p in zip(rf_vi, [1, 2, 0, 3, 4]): 
        ax2.barh(p, float(vi), yerr=float(verr), color = 'OliveDrab', alpha=0.75)
        print(vi, p, labels_o[p])
    ax2.set_yticks(range(len(labels)))
    ax2.set_yticklabels(labels_o, rotation=0)
    ax2.set_xlabel('Feature importance')
    ax2.text(.90, .1, 'B', ha='left', va='top', transform=ax2.transAxes, fontsize=12)

    plt.tight_layout()
    if figname is not None:
         plt.savefig(figname)
    plt.show()


def fig_3_theory(trend_result_all_f, df_ts_, trend_name, bin_th=5, figname=None):

    fig = plt.figure(figsize=(7.5, 8))

    axa1 = plt.subplot2grid((4, 3), (0, 0), colspan=1, rowspan=1)
    axa2 = plt.subplot2grid((4, 3), (1, 0), colspan=1, rowspan=1)
    axa3 = plt.subplot2grid((4, 3), (2, 0), colspan=1, rowspan=1)
    axa4 = plt.subplot2grid((4, 3), (3, 0), colspan=1, rowspan=1)
    axb1 = plt.subplot2grid((4, 3), (0, 1), colspan=2, rowspan=2)
    axb2 = plt.subplot2grid((4, 3), (2, 1), colspan=2, rowspan=2)

    for ii, ax, var_v, varname, scale, va in [['A', axa1, 'CO2_yi', 'CO$_2$', 10**6, 0.1],
                               ['B', axa2, 'TC_tmean_gs_yi', 'Ta', 1, 0.1],
                               ['C', axa3, 'TC_vpd_gs_yi', 'VPD', 1/1000, 0.8],
                               ['D', axa4, 'TC_par_gs_yi', 'PAR', 10**6, 0.8]]:

        dd = []
        var_l = np.linspace(np.nanmin(df_ts_[var_v])*0.75, np.nanmax(df_ts_[var_v])*1.25, 101)
        for k in ['TC_tmean_gs_yi', 'TC_vpd_gs_yi', 'elev_pa', 'CO2_yi', 'TC_par_gs_yi']:
            if k !=var_v:
                dd.append(np.nanmean(df_ts_[k]))
            else:
                dd.append(var_l)

        
        vcmax = cal_opt_states(dd[0], 
                            dd[1], 
                            dd[2], 
                            dd[3], 
                            dd[4])[6]
   
        ax.plot(var_l*scale, vcmax/np.mean(vcmax), color='k')
        ax.set_xlabel(varname)
        fig.text(0.03, 0.5, 'Optimal relative photosynthetic capacity (–)', 
                            va='center', rotation='vertical', fontsize=12)
        ax.text(va, .2, ii, ha='left', va='top', transform=ax.transAxes, fontsize=12)
    

    color1 = 'dodgerblue'
    color2 = 'navy'
    var_p = [('TC_tmean_gs_LTavg', 'Air temperature', 1, 'E'), 
             ('TC_vpd_gs_LTavg', 'Vapor pressure deficit', 1/10, 'F')]
    nn = 11
    v_ranges = {'TC_tmean_gs_LTavg': np.linspace(np.min(trend_result_all_f['TC_tmean_gs_LTavg']), 
                               np.max(trend_result_all_f['TC_tmean_gs_LTavg']), nn),
                'TC_vpd_gs_LTavg': np.linspace(np.min(trend_result_all_f['TC_vpd_gs_LTavg']), 
                                   np.max(trend_result_all_f['TC_vpd_gs_LTavg']), nn)}

    shade = [5, 95]

    for ax, (vary_v, v_name, v_scale, spi)  in zip([axb1, axb2], (var_p)): 

        
        bw = (v_ranges[vary_v][1] - v_ranges[vary_v][0])/2
        delta = [np.nanmedian(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name]) 
                 if len(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name])>bin_th else np.nan
                 for b in v_ranges[vary_v]]

        delta_75 = [np.percentile(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name], shade[1]) 
                    if len(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name])>bin_th else np.nan
                    for b in v_ranges[vary_v]]
        delta_25 = [np.percentile(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name], shade[0])
                    if len(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name])>bin_th else np.nan
                    for b in v_ranges[vary_v]]

        ax.set_xlabel(v_name)
        fig.text(0.34, 0.5, 'Change in photosynthetic nitrogen demand (% 50 ppm$^{-1}$ CO$_2$)', 
                            va='center', rotation='vertical', fontsize=12, color=color1)
        ax.set_ylim([-5.35, -3.4])
        ax.text(.9, .1, spi, ha='left', va='top', transform=ax.transAxes, fontsize=12)
        ax.set_yticks([-5, -4.5, -4, -3.5])
        
        trend_name2 = 'deltaC nue_gs_LTavg_sunlit_ww'

        ax.plot(v_ranges[vary_v]*v_scale, np.array(delta), color=color1)
        ax.fill_between(v_ranges[vary_v]*v_scale, np.array(delta_25), np.array(delta_75), color=color1, alpha=0.15)

        bw = (v_ranges[vary_v][1] - v_ranges[vary_v][0])/2
        delta = [np.nanmedian(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name2]) 
                 if len(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name2])>bin_th else np.nan
                 for b in v_ranges[vary_v]]

        delta_75 = [np.percentile(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name2].dropna(), shade[1]) 
                    if len(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name2])>bin_th else np.nan
                    for b in v_ranges[vary_v]]
        delta_25 = [np.percentile(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name2].dropna(), shade[0])
                    if len(trend_result_all_f[(trend_result_all_f[vary_v] <=b + bw) 
                                            & (trend_result_all_f[vary_v] >b - bw)]
                           [trend_name2])>bin_th else np.nan
                    for b in v_ranges[vary_v]]

        ax2 = ax.twinx()
        ax2.plot(v_ranges[vary_v]*v_scale, np.array(delta), color=color2)
        ax2.fill_between(v_ranges[vary_v]*v_scale, np.array(delta_25), np.array(delta_75), color=color2, alpha=0.15)

        ax2.set_xlabel(v_name)
        fig.text(0.95, 0.5, 'Change in photosynthetic nitrogen use efficiency (% 50 ppm$^{-1}$ CO$_2$)', 
                            va='center', rotation='vertical', fontsize=12, color=color2)
        ax2.set_yticks([7, 8, 9, 10])
        ax.text(.9, .1, spi, ha='left', va='top', transform=ax.transAxes, fontsize=12)

    plt.subplots_adjust(hspace=0.55, wspace=1.1)
    #plt.tight_layout()
    if figname is not None:
         plt.savefig(figname)
    plt.show()


def fig_4_cmip_trends(y_vars, df_, df_h, df_p, y_label ='', ylim=None, xlim=[1870, 2100], figname=None):
    fig = plt.figure(figsize=(6.5, 4))
    ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
    df_h = df_h[(df_h['year']>=1995) & (df_h['year']<=2016)]
    df_p = df_p[df_p['year']<xlim[1]]
    df_ = df_[df_['year']>xlim[0]]
    #df_ = df_[df_['year']<=2014]
    for y_var, name, color, ls in y_vars:
        if 'historical_CESM2_gs_sunlit' in y_var:
            df = df_h
            scale = -1
            co2_N = 'CO2_historical'
        elif y_var == 'LNC_stdm':
            df = df_
            scale = 2014 - 1995
            co2_N = 'CO2_yi'
        else:
            df = df_p
            scale = 0
            if 'ssp245' in y_var:
                scenario = 'ssp245'
            elif 'ssp370' in y_var:
                scenario = 'ssp370'
            elif 'ssp585' in y_var:
                scenario = 'ssp585'
            co2_N = 'CO2_%s' % scenario

        delta_co2 = (np.max(df[co2_N]) - np.min(df[co2_N])) * 10 ** 6
        delta_y = (np.max(df['year']) - np.min(df['year']) + 1)

        #print(delta_co2)
        if 'median' in y_var:
            y_var_err = y_var.replace("median", "stderr")
            df = df[[y_var, 'year', co2_N, y_var_err]].dropna()
        else:
            df = df[[y_var, 'year', co2_N]].dropna()
        if 'LNC observations' == name:
            period = range(np.min(df['year']), 2016 + 1)
        else:
            period = range(np.min(df['year']), np.max(df['year']) + 1)

        y_b = [df[df.year==yi][y_var] for yi in period]
        y_bm = np.array([np.nanmedian(yi) for yi in y_b])


        #trend, b, a1, a2 = theilslopes(df[y_var], df['year'], 0.95)
        #trend_c, b_c, a1_c, a2_c = theilslopes(df[y_var], df[co2_N]*10**6, 0.95)
        ##trend, b, a1, a2 = theilslopes(y_bm, period, 0.95)
        #trend_err = trend - a1
        #trend_ts = trend * period + b
        #delta = trend_c / np.mean(df[y_var]) * 100 * 50
        #delta_err = (trend - a1)  / np.mean(df[y_var]) * 100 * (period[-1] - period[0] + 1) * 44 / 50

        print('')
        print(y_var, delta_co2, delta_co2 / (np.min(df[co2_N])* 10 ** 6)*100, ls)
        trend, b, a1, a2 = theilslopes(df[y_var], df['year'], 0.95)
        trend_err = trend - a1
        trend_ts = trend * period + b
        print('theilslope-year', trend * 1000, 'err', trend_err * 1000)

        trendC, b, a1, a2 = theilslopes(df[y_var], df[co2_N]*10**6, 0.95)
        trendC_err = trendC - a1
        print('theilslope-CO2', trendC / np.mean(df[y_var])*100 * 50, 'err', trendC_err / np.mean(df[y_var])*100 * 50)

        delta = trend / np.mean(df[y_var]) * 100 * delta_y 
        delta_err = (trend - a1)  / np.mean(df[y_var]) * 100 * delta_y
        print('delta', delta * delta_co2 / 50, 'err', delta_err * delta_co2 / 50)
        #print(y_var, delta, trend_ts[scale], delta_co2, ls)
        if ('Theory' in name) :
            stderr = np.array([np.nanmedian(df[df.year==yi][y_var_err]) for yi in period])
            ax1.plot(period, y_bm/trend_ts[scale], color=color, marker='', linewidth=0.75, alpha=0.5, linestyle=ls)
            y_l = [(yi - stderri)/trend_ts[scale]  for yi, stderri in zip(y_bm, stderr)]
            y_u = [(yi + stderri)/trend_ts[scale] for yi, stderri in zip(y_bm, stderr)]
            ax1.fill_between(period, y_l, y_u, color=color, alpha=0.15)

        elif 'LNC observations' == name:
            y_l = [(np.nanmedian(yi) - np.nanstd(yi) / np.sqrt(np.size(yi)))/trend_ts[scale] if len(yi)>2 else np.nan for yi in y_b]
            y_u = [(np.nanmedian(yi) + np.nanstd(yi) / np.sqrt(np.size(yi)))/trend_ts[scale] if len(yi)>2 else np.nan for yi in y_b]
            ax1.fill_between(period, y_l, y_u, color=color, alpha=0.15)
            ax1.plot(period, y_bm/trend_ts[scale], color=color, alpha=0.15)

        #if ls == '-':
        #    ax1.plot(period, trend_ts/trend_ts[scale], color=color,linestyle=ls, label= name)
        #elif 'Empirical' in name:
        #    ax1.plot(period, trend_ts/trend_ts[scale], color=color,linestyle=ls, label= name)
        #else:
        ax1.plot(period, trend_ts/trend_ts[scale], color=color,linestyle=ls)
        
    ax1.set_ylabel(y_label)
    #ax1.set_xlabel('Year')
    #ax1.legend(frameon=False)
    from matplotlib.lines import Line2D
    legend_elements = [ 
                        Line2D([0], [0], marker= 's', linestyle='', color='k', label='$LNC$ observations', alpha=0.15),
                        Line2D([0], [0], marker= 's', linestyle='', color='limegreen', label='Middle road (SSP245)'),
                        Line2D([0], [0], marker= 's', linestyle='', color='purple', label='Fossil fuel development (SSP585)'),
                        Line2D([0], [0], marker= '', linestyle='', color='k', label=''),
                        Line2D([0], [0], marker= '', linestyle='-', color='k', label='Theory (climate, $CO_2$)'),
                        Line2D([0], [0], marker= '', linestyle='--', color='k', label='Theory (climate)'),
                        Line2D([0], [0], marker= '', linestyle=':', color='k', label='Theory ($CO_2$)'),
                        #Line2D([0], [0], marker= '', linestyle=(0, (1, 5)), color='k', label='Empirical ($CO_2$)'),

                        ]
    ax1.legend(handles=legend_elements, frameon=False, ncol=2, fontsize=7, loc='upper right', bbox_to_anchor=(1, 0.99), borderaxespad=0.)
    
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    plt.tight_layout()
    ax1.text(.02, .98, 'A', ha='left', va='top', transform=ax1.transAxes, fontsize=10)
    if figname is not None:
         plt.savefig(figname)
    plt.show()


if __name__ == "__main__":
    pass
