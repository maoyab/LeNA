import numpy as np
from sswm0 import SM_C_H as SM_C_H_0
from scipy import optimize
from scipy.interpolate import interp1d

k_CtoK = 273.15            # Conversion from °C to K
Rd = 8.314                 # [m3 Pa/K/mol] Dry air gas constant
Cao = 210 * 10 ** (-3)     # [mol/mol]     Atmospheric O2
k_c_molmass = 12.0107     # [g/mol]       Molecular mass of carbon 
k_To = 25.0                # [°C]          Reference temperature
k_Po = 101325.0            # [Pa]          Standard atmosphere (Allen, 1973)

# Bernacchi, 2003
Kc_25C = 404.9 * 10 ** (-6)           # [mol/mol]   Michelis-Menten constants for CO2 inhibition at 25 C
Ko_25C = 278 * 10 ** (-3)             # [mol/mol]   Michelis-Menten constants for O2 inhibition at 25 C
gamma_star_25C = 42.75 * 10 ** (-6)   # [mol/mol]   CO2 compensation point at 25 C C3 plants

bernacchi_dhac = 79430   # [J/mol] Kc Activation energy 
bernacchi_dhao = 36380   # [J/mol] Ko Activation energy
bernacchi_dha = 37830    # [J/mol] gamma* Activation energy

#  Kattge 2007
kattge_knorr_a_ent = 668.39 # [J/mol/K]  Offset of entropy vs. temperature relationship
kattge_knorr_b_ent = -1.07  # [J/mol/K2] Slope of entropy vs. temperature relationship
kattge_knorr_Ha = 71513     # [J/mol]    Activation energy
kattge_knorr_Hd = 200000    # [J/mol]    Deactivation energy

ha_vcmax25 = 65330
ha_jmax25 = 43900

c_smith19 = 0.053     # Smith 2019 cost parameter
theta_smith19 = 0.85  # Smith 2019 curvature response of J and Iabs
cost_beta_stocker19 = 146     # Stocker 2019 cost ration a/b
c_wang17 = 0.41     # unit carbon cost for the maintenance of electron transport capacity (-)
k0_Pmodel = 0.081785   #[-]

# Viscosity of water Huber, 2009
huber_tk_ast = 647.096 #  Reference temperature (Kelvin)
huber_rho_ast = 322.0  # Reference density (kg/m^3)
huber_mu_ast = 1e-06   # Reference pressure (Pa s)
huber_H_i = (1.67752, 2.20462, 0.6366564, -0.241605)                # Values of H_i (Table 2)
huber_H_ij = ((0.520094, 0.0850895, -1.08374, -0.289555, 0.0, 0.0), # alues of H_ij (Table 3)
        (0.222531, 0.999115, 1.88797, 1.26613, 0.0, 0.120573),
        (-0.281378, -0.906851, -0.772479, -0.489837, -0.25704, 0.0),
        (0.161913, 0.257399, 0.0, 0.0, 0.0, 0.0),
        (-0.0325372, 0.0, 0.0, 0.0698452, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.00872102, 0.0),
        (0.0, 0.0, 0.0, -0.00435673, 0.0, -0.000593264))

# Density of water Table 5 of Fisher 1975
fisher_dial_lambda = (1788.316, 21.55053, -0.4695911, 0.003096363, -7.341182e-06)
fisher_dial_Po = (5918.499, 58.05267, -1.1253317, 0.0066123869, -1.4661625e-05)
fisher_dial_Vinf = (0.6980547, -0.0007435626, 3.704258e-05, -6.315724e-07, 
                    9.829576e-09, -1.197269e-10, 1.005461e-12, -5.437898e-15, 1.69946e-17, -2.295063e-20)


def arrh_temp_function(tc, Ha):
    # tc in deg C, T ref = 25 C
    f = np.exp(Ha * (tc - 25) / (298. * Rd * (tc + 273.15)))
    return f

def entropy_factor(tc):
    # Table 3 of Kattge 2007, Pmodel
    deltaS = kattge_knorr_a_ent + kattge_knorr_b_ent * tc
    return deltaS

def peaked_function_Medlyn(tc, Ha):
    # Temperature scaling factor
    # peaked function (eq. 17) in Medlyn et al. 2002
    # tc in deg C, T ref = 25 C
    deltaS = entropy_factor(tc)
    num = 1 + np.exp((298 * deltaS - kattge_knorr_Hd) / (298. * Rd)) 
    denom = 1 + np.exp(((tc + 273.15) * deltaS - kattge_knorr_Hd)/ ((tc + 273.15) * Rd)) 
    return arrh_temp_function(tc, Ha) * num / denom

def cal_gamma_star_temp(tc):
    # tc in deg C
    # Bernacchi 2001 as reported in Medlyn et al. 2002 (eq. 6)
    gamma_star = gamma_star_25C * arrh_temp_function(tc, bernacchi_dha)
    return gamma_star

def cal_MM_Kc_temp(tc):
    # ct in deg C
    # Bernacchi 2001 as reported in Medlyn et al. 2002 (eq. 5)
    Kc = Kc_25C * arrh_temp_function(tc, bernacchi_dhac)
    return Kc

def cal_MM_Ko_temp(tc):
    # ct in deg C
    # Bernacchi 2001 as reported in Medlyn et al. 2002 (eq. 6)
    Ko = Ko_25C * arrh_temp_function(tc, bernacchi_dhao)
    return Ko

def cal_K_Rub(tc):
    # [mol/mol]
    Kc = cal_MM_Kc_temp(tc)
    Ko = cal_MM_Ko_temp(tc)
    a2 = Kc * (1 + Cao / Ko)
    return a2

def cal_ns_star(tc, Pa):
    #[-]
    visc_env = cal_viscosity_h2o(tc, Pa)
    visc_std = cal_viscosity_h2o(k_To, k_Po)
    return visc_env / visc_std

def cal_density_h2o(tc, patm):
    # [kg/m^3]
    # Get powers of tc, including tc^0 = 1 for constant terms
    tc_pow = np.power.outer(tc, np.arange(0, 10))
    # Calculate lambda, (bar cm^3)/g:
    lambda_val = np.sum(np.array(fisher_dial_lambda) * tc_pow[..., :5], axis=-1)
    # Calculate po, bar
    po_val = np.sum(np.array(fisher_dial_Po) * tc_pow[..., :5], axis=-1)
    # Calculate vinf, cm^3/g
    vinf_val = np.sum(np.array(fisher_dial_Vinf) * tc_pow, axis=-1)
    # Convert pressure to bars (1 bar <- 100000 Pa)
    pbar = 1e-5 * patm
    # Calculate the specific volume (cm^3 g^-1):
    spec_vol = vinf_val + lambda_val / (po_val + pbar)
    # Convert to density (g cm^-3) -> 1000 g/kg; 1000000 cm^3/m^3 -> kg/m^3:
    rho = 1e3 / spec_vol
    return rho

def cal_viscosity_h2o(tc, patm):
    # [kg/m^3]

    rho = cal_density_h2o(tc, patm)
    # Calculate dimensionless parameters:
    tbar = (tc + k_CtoK) / huber_tk_ast
    rbar = rho / huber_rho_ast
    # Calculate mu0 (Eq. 11 & Table 2, Huber et al., 2009):
    tbar_pow = np.power.outer(tbar, np.arange(0, 4))
    mu0 = (1e2 * np.sqrt(tbar)) / np.sum(np.array(huber_H_i) / tbar_pow, axis=-1)
    # Calculate mu1 (Eq. 12 & Table 3, Huber et al., 2009):
    h_array = np.array(huber_H_ij)
    ctbar = (1.0 / tbar) - 1.0
    row_j, _ = np.indices(h_array.shape)
    mu1 = h_array * np.power.outer(rbar - 1.0, row_j)
    mu1 = np.power.outer(ctbar, np.arange(0, 6)) * np.sum(mu1, axis=(-2))
    mu1 = np.exp(rbar * mu1.sum(axis=-1))
    # Calculate mu_bar (Eq. 2, Huber et al., 2009), assumes mu2 = 1
    mu_bar = mu0 * mu1
    # Calculate mu (Eq. 1, Huber et al., 2009)
    return mu_bar * huber_mu_ast  # Pa s

def cal_intrinsic_quantum_yield(tc, k0=1 / 8.):
    # [mol/mol]
    kphio = k0 * (0.352 + 0.022 * tc - 0.00034 * tc ** 2) 
    return kphio

def cal_chi_opt(vpd, co2, g1, gamma_star):
    # [-]
    chi = gamma_star / co2 + (1 - gamma_star / co2) * g1 / (g1 + vpd ** 0.5)    
    return chi

def cal_g1(tc, pa, kmm, gamma_star, beta=cost_beta_stocker19, nu_star=None):
    # [Pa**0.5]
    if nu_star is None:
        nu_star = cal_ns_star(tc, pa)         # [-]
    else:
        nu_star = 1
    g1 =  (beta * (kmm + gamma_star) * pa / (1.6 * nu_star))**0.5              
    return g1

def cal_g1_colim(tc, pa, kmm, gamma_star, beta=cost_beta_stocker19, nu_star=None):
    # [Pa**0.5]
    gamma0 = j / (4  / vcmax) * kmm
    g1 =  (beta * (gamma0 + gamma_star) * pa / 1.6 )**0.5   

    '''
    gamma0 = kphio * fapar * ppfd * fJlim_root / (2 * theta_curve)\
            (4  / (kphio * fapar * ppfd * mj * f_v / mc) \
                * kmm

    gamma0 =  fJlim_root / (2 * theta_curve) / \
            (4  * mc / (mj * f_v) \
                * kmm

    gamma0 =  kmm / (4  * (mc/mj)) \
             * (fJlim_root * f_v / (2 * theta_curve))

        gamma0 =  kmm / (4  * (mc/mj)) * lim_tot)

    mc / mj = (ci + 2 * gamma_star) / (ci + kmm)

    '''           
    return g1

def electron_transport_rate(ppfd, j_max, theta_curve=0.85, fapar=0.91, leaf_scattering=0.2):
    # Bonan Book 2019
    j_psii = kphio * fapar * ppfd
    j1 = (j_psii + j_max - ((j_psii + j_max) ** 2 - 4 * theta_curve * j_psii * j_max) ** 0.5) \
            / (2 * theta_curve)
    j2 = (j_psii + j_max + ((j_psii + j_max) ** 2 - 4 * theta_curve * j_psii * j_max) ** 0.5) \
            / (2 * theta_curve)

    '''
    jmax = 4 * f_j * kphio * fapar * ppfd = 4 * f_j * j_psii
    j_psii = kphio * fapar * ppfd
    j1 = (j_psii + (4 * f_j * j_psii) 
        - ((j_psii + (4 * f_j * j_psii)) ** 2
         - 4 * theta_curve * j_psii * (4 * f_j * j_psii)) ** 0.5) \
            / (2 * theta_curve)
    '''

    '''
    ## j_psii + j_max = kphio * fapar * ppfd * (1 + 4 * f_j) 
    j1 = (j_psii * (1 + 4 * f_j) 
        - ((j_psii * (1 + 4 * f_j)) ** 2
         - 4 * theta_curve * j_psii** 2 * 4 * f_j) ** 0.5) \
            / (2 * theta_curve)
    '''

    '''
    j1 = (j_psii * 
        ((1 + 4 * f_j) \
            - ((1 + 4 * f_j) ** 2 - 4 * theta_curve  * 4 * f_j) ** 0.5) \
            / (2 * theta_curve)

    j = (j_psii * fJlim_root  / (2 * theta_curve)
    j = kphio * fapar * ppfd * fJlim_root / (2 * theta_curve)
    fJlim_root1 = ((1 + 4 * f_j) \
            - ((1 + 4 * f_j) ** 2 - 4 * theta_curve  * 4 * f_j) ** 0.5)

    fJlim_root2 = ((1 + 4 * f_j) \
            + ((1 + 4 * f_j) ** 2 - 4 * theta_curve  * 4 * f_j) ** 0.5)
    '''

    return np.min([j1, j2])

def photo_k2(j, vc_max, kmm):
    # Vico et al., 2013
    k2 =  j / 4 * kmm  / (vc_max)

    # k2 =  j / 4 * kmm  / (kphio * fapar * ppfd * mj * f_v / mc )
    
    return k2

def cal_jlim(mj, ref='Smith2019'):
    # [-]
    if ref == 'Smith2019':
        cm = 4 * c_smith19 / mj * (1 - theta_smith19 * 4 * c_smith19 / mj)
        
        #omega = -(1 - 2 * theta_smith19) + ((1 - theta_smith19) * (1 / (cm) - 4 * theta_smith19)) ** 0.5

        cap_p = (((1 / 1.4) - 0.7) ** 2 / (1 - theta_smith19)) + 3.4
        roots = np.polynomial.polynomial.polyroots([-1, cap_p, -(cap_p * theta_smith19)])
        m_star = (4 * c_smith19) / roots[0].real
        v = 1 / (cm * (1 - theta_smith19 * cm)) - 4 * theta_smith19
        omega = np.where(mj < m_star,
                        -(1 - (2 * theta_smith19)) - np.sqrt((1 - theta_smith19) * v),
                        -(1 - (2 * theta_smith19)) + np.sqrt((1 - theta_smith19) * v),)
        omega = omega.item() if np.ndim(omega) == 0 else omega
        
        omega_star = 1 + omega - ((1 + omega)**2 - 4 * theta_smith19 * omega) ** 0.5
        f_v = omega_star / (2 * theta_smith19) 
        #2 instead of 8 because phi0 here is quantum efficiency as photosynthesis 
        #and not electron transport as in Smith 2019 
        f_j = omega                                        

    elif ref == 'Wang2017':
        #vals_defined = np.greater(mj, c_wang17)
        f_v = (1 - (c_wang17 / mj) ** (2.0 / 3.0)) ** 0.5
        f_j = ((mj / c_wang17) ** (2.0 / 3.0) - 1) ** 0.5

    elif ref == 'Prentice2014':
        f_v = 1
        f_j = 1

    return f_v, f_j

def cal_opt_states(tair, vpd, pa, co2, ppfd,  
                    fapar=0.91, soilmstress=1, nu_star=None, g1 = None, 
                    k0=0.081785, beta=cost_beta_stocker19,
                    ref='Smith2019', print_summary=None):
    #co2 = dd['CO2']                # [mol/mol]
    #tair = dd['tmp']               # [C]
    #vpd = dd['vpd']                # [Pa]
    #ppfd = dd['par']               # [mol/m2/s]
    #pa = dd['pa']                  # [Pa]

    gamma_star = cal_gamma_star_temp(tair)  # [mol/mol]
    kmm = cal_K_Rub(tair)                   # [mol/mol]
    kphio = soilmstress * cal_intrinsic_quantum_yield(tair, k0=k0) # [mol/mol]

    if g1 is None:
        g1 = cal_g1(tair, pa, kmm, gamma_star, beta=beta, nu_star=nu_star)             # [Pa**0.5]
    chi = cal_chi_opt(vpd, co2, g1, gamma_star)
    ci = co2 * chi                                                  # mol/mol]
    
    mc = (ci - gamma_star) / (ci + kmm)
    mj = (ci - gamma_star) / (ci + 2 * gamma_star)
    #mj = (co2 - gamma_star) / (co2 + 2 * gamma_star + 3 * gamma_star / g1 * vpd** 0.5) # [mol/mol]
    f_v, f_j = cal_jlim(mj, ref=ref)

    # pmodel
    lue = kphio * mj * f_v * k_c_molmass  # [gC / mol photon] 
    gpp = lue * fapar * ppfd                            # [gC / m2 / s]
    gpp_mol = gpp / k_c_molmass                         # [mol C / m2 / s]
    
    
    vcmax = kphio * fapar * ppfd * mj / mc * f_v                           # [mol/m2/s]
    vcmax25 = vcmax / peaked_function_Medlyn(tair, ha_vcmax25)              # [mol/m2/s]
    #iwue = (co2 - gamma_star) / ( 1.6 * (1 + g1 / vpd ** 0.5))             # [mol/mol]
    iwue = (co2 - ci) / 1.6                                                 # [mol/mol]
    #gs = (1 + g1 / (vpd ** 0.5)) * gpp_mol / (co2 - gamma_star)            # [mol C / m2 / s]
    gs = gpp / k_c_molmass / (co2 - ci)                                     # [mol C / m2 / s]

    nue = gpp_mol / vcmax25 * (3.5 * 8 * 44) / (14 * 552000 * 0.0144) * 1000 # [mg C g N-1]

    results = [g1, chi, lue, iwue,  gpp, vcmax, vcmax25, gs, nue]
    
    if print_summary:
        print('gamma*: %-5.2f [Pa]' % (gamma_star * pa))
        print('kmm: %-5.2f [Pa]' % (kmm * pa))
        print('mc: %-5.2f [-]' % mc)
        print('mj: %-5.2f [-]' % mj)
        print('mj_prime: %-5.2f [-]' % (mj * f_v))
        print('mjoc: %-5.2f [-]' % (mj / mc))
        print('kphio: %-5.4f [mol/mol]' % kphio)
        print('')

        pm_names = ['g1', 'chi', 'lue', 'iwue',  'gpp', 'vcmax', 'vcmax25', 'gs']
        units = [1, 1, 1, 10**6, 10**6, 10**6, 10**6, 10**3]
        unit_names = ['[Pa**0.5]', '[mol/mol]', '[gC / mol photon]', '[umol/mol]', 
                      '[ug C / m2 / s]', '[umol C / m2 / s]', '[umol C / m2 / s]', '[mmol C / m2 / s]']
        for k, r, u, un in zip(pm_names, results, units, unit_names):
            print(k,'%-5.3f\t%s'% (r*u, un))

    return results

def cal_swb_0(dd, q=1):
    #try:
    model_p = [dd['T_GS'], dd['rf_alpha'], dd['rf_lambda'], dd['pet'], dd['Td'],
               dd['LAI'], dd['Fpar'], dd['RAI'], dd['hc'], dd['Zr'],
               dd['SATPSI'], dd['BB'], dd['Ks'], dd['porosity'], dd['s_h'], dd['s_fc'], 
               dd['kxlmax'], dd['Px50'], dd['Pg50']]

    smpdf = SM_C_H_0(model_p, f_wilt=0.05, f_star=0.95, constraints=False, q=q) 
    s_stressB = 1 - smpdf.get_mean_stress(smpdf.p0)
    #d_stressB = 1 - smpdf.mean_dynamic_water_stress(smpdf.p0)

    p_fitted_norm = smpdf.p0
    cdf = np.cumsum(p_fitted_norm)
    f = interp1d(cdf, np.linspace(0, 1, len(p_fitted_norm)))
    random_p = [np.random.uniform(0, 1) for r in range(365)]
    fit_sm = np.array(f(random_p))
    mean_sm = np.mean(fit_sm)

    return [mean_sm, s_stressB, #d_stressB, 
            smpdf.epsilon_static, #smpdf.epsilon_dynamic, 
            smpdf.pi_F, smpdf.T_ww,
            smpdf.tet_part, smpdf.mean_ef, smpdf.mean_tf, 
            smpdf.mean_di, smpdf.rf]

def cal_cost_trend_fit(trend_n, dd):
    def __residual_trend(beta, inputs, trend_n):
        if beta < 20 :
            res = 10**10
        elif beta >500:
            res = 10 **10
        else:
            tc, patm, co2, vpd, ppfd, fapar, year = inputs
            [g1, chi, lue, iwue,  gpp, vcmax, vcmax25, gs] = cal_opt_states(tc, vpd, patm, co2, ppfd,  
                                fapar=fapar, soilmstress=1, nu_star=None, g1 = None, 
                                k0=0.081785, beta=beta, ref='Smith2019', print_summary=None)
            tt, b, c, d = theilslopes(vcmax25, year, 0.95)
            d = tt / np.nanmean(vcmax25) * 100 #...............?
            res = (d - trend_n) ** 2
        return res

    inputs = [dd['tmp'].values, dd['surface_pressure'].values, dd['CO2'].values, dd['vpd'].values, dd['par'].values, dd['Fpar'].values, dd['year'].values]
    beta_fit = optimize.leastsq(__residual_trend, 146., args=(inputs, trend_n))[0][0]

    optchi, lue, iwue, vcmax, vcmax25, gpp, gs = model_p(dd['tmp'].values, dd['surface_pressure'].values, dd['CO2'].values, dd['vpd'].values, dd['par'].values, fapar=dd['Fpar'].values, beta_cost_ratio=beta_fit)
    tt, b, c, d = theilslopes(vcmax25, dd['year'].values, 0.95)
    d = tt / np.nanmean(vcmax25) * 100 

    return beta_fit, d

def empirical_vcmax_Co2sensitivity(co2, co2ref):
    #co2ref=353.9
    return (-0.000257 * co2 * 10 **6 +1.090) / (-0.000257 * co2ref * 10 **6 +1.090)

def beta_co2_log(ye, ya, co2e, co2a):
    return np.log(ye/ya)/ np.log(co2e/co2a)
    
def beta_co2(yt, y0, co2t, co20):
    return ((yt - y0) / y0) / ((co2t - co20) / co20)
