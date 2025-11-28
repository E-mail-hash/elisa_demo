import numpy as np
import numba
import defaultParameters as dp

def timeIntegration(params):
    # simulation parameters
    dt = params["dt"]
    sqrt_dt = np.sqrt(dt)
    duration = params["duration"]
    RNGseed = params["seed"]

    t = np.arange(0, duration, dt) 

    # neuron parameters
    N_cor2tha = 1.2
    N_tha2cor = 0.12
    ## thalamus
    tau = params["tau"]
    Q_max = params["Q_max"]
    C1 = params["C1"]
    theta = params["theta"]
    sigma = params["sigma"]
    g_L = params["g_L"]
    E_L = params["E_L"]
    g_AMPA = params["g_AMPA"]
    g_GABA = params["g_GABA"]
    E_AMPA = params["E_AMPA"]
    E_GABA = params["E_GABA"]
    g_LK = params["g_LK"]
    E_K = params["E_K"]
    g_T_t = params["g_T_t"]
    g_T_r = params["g_T_r"]
    E_Ca = params["E_Ca"]
    g_h = params["g_h"]
    g_inc = params["g_inc"]
    E_h = params["E_h"]
    C_m = params["C_m"]
    alpha_Ca = params["alpha_Ca"]
    Ca_0 = params["Ca_0"]
    tau_Ca = params["tau_Ca"]
    k1 = params["k1"]
    k2 = params["k2"]
    k3 = params["k3"]
    k4 = params["k4"]
    n_P = params["n_P"]
    gamma_e = params["gamma_e"]
    gamma_r = params["gamma_r"]
    d_phi = params["d_phi"]
    N_rt = params["N_rt"]
    N_tr = params["N_tr"]
    N_rr = params["N_rr"]

    ext_current_t = params["ext_current_t"]
    ext_current_r = params["ext_current_r"]

    ## aln
    de = params['de']
    di = params['di']
    ndt_de = int(de/dt)
    ndt_di = int(di/dt)
    ndt_dall = int(13/dt)

    startind = int(np.max([ndt_de, ndt_di, ndt_dall]))

    c_gl     = params['c_gl']       # global coupling strength between areas(unitless)
    Ke_gl    = params['Ke_gl']     # number of incoming E connections (to E population) from each area

    Ke = params['Ke']
    Ki = params['Ki']

    tau_ou = params['tau_ou']
    sigma_ou = params['sigma_ou']
    mue_ext_mean = params['mue_ext_mean']
    mui_ext_mean = params['mui_ext_mean']
    mue_ext = mue_ext_mean
    mui_ext = mui_ext_mean

    sigmae_ext = params['sigmae_ext']
    sigmai_ext = params['sigmai_ext']
    
    tau_se = params['tau_se']
    tau_si = params['tau_si']

    cee = params['cee']
    cei = params['cei']
    cie = params['cie']
    cii = params['cii']

    Jee_max = params['Jee_max']
    Jie_max = params['Jie_max']
    Jei_max = params['Jei_max']
    Jii_max = params['Jii_max']

    cee = cee * tau_se / Jee_max # ms?
    cei = cei * tau_si / abs(Jei_max)
    cie = cie * tau_se / Jie_max
    cii = cii * tau_si / abs(Jii_max)
    c_gl = c_gl * tau_se / Jee_max

    a = params['a']
    b = params['b']
    EA = params['EA']
    tauA = params['tauA']

    C = params['C']
    gl = params['gl']
    taum = C / gl


    # neuron initialization
    ## thalamus
    V_t = np.dot(params['V_t_init'], np.ones(len(t))) 
    V_r = np.dot(params['V_r_init'], np.ones(len(t)))
    Q_t = np.zeros(len(t))
    Q_r = np.zeros(len(t)) 

    # neuron initialization 
    Ca = float(params["Ca_init"])
    h_T_t = float(params["h_T_t_init"])
    h_T_r = float(params["h_T_r_init"])
    m_h1 = float(params["m_h1_init"])
    m_h2 = float(params["m_h2_init"])
    s_et = float(params["s_et_init"])
    s_gt = float(params["s_gt_init"])
    s_er = float(params["s_er_init"])
    s_gr = float(params["s_gr_init"])
    ds_et = float(params["ds_et_init"])
    ds_gt = float(params["ds_gt_init"])
    ds_er = float(params["ds_er_init"])
    ds_gr = float(params["ds_gr_init"])

    np.random.seed(RNGseed)
    noise = np.random.standard_normal((len(t)))

    # aln
    Q_e = np.dot(params['Q_e_init'], np.ones(len(t)))
    Q_i = np.dot(params['Q_i_init'], np.ones(len(t)))

    if RNGseed:
        np.random.seed(RNGseed)
    Q_e[startind:] = np.random.standard_normal(len(range(startind, len(t))))
    Q_i[startind:] = np.random.standard_normal(len(range(startind, len(t))))

    mufe = float(params['mufe_init'])
    mufi = float(params['mufi_init'])
    seem = float(params['seem_init'])
    siem = float(params['siem_init'])
    seim = float(params['seim_init'])
    siim = float(params['siim_init'])
    seev = float(params['seev_init'])
    seiv = float(params['seiv_init'])
    siev = float(params['siev_init'])
    siiv = float(params['siiv_init'])
    IA = float(params['IA_init'])

    precalc_r, precalc_V, precalc_tau_mu = params['precalc_r'], params['precalc_V'], params['precalc_tau_mu']
    dI = params['dI']
    ds = params['ds']
    sigmarange = params['sigmarange']
    Irange = params['Irange']
    (
        t,
        V_t,
        V_r,
        Q_t,
        Q_r,
        Q_e, 
        Q_i
    ) = timeIntegration_njit_elementwise(
        startind,
        t,
        dt,
        sqrt_dt,
        Q_max,
        C1,
        theta,
        sigma,
        g_L,
        E_L,
        g_AMPA,
        g_GABA,
        E_AMPA,
        E_GABA,
        g_LK,
        E_K,
        g_T_t,
        g_T_r,
        E_Ca,
        g_h,
        g_inc,
        E_h,
        C_m,
        tau,
        alpha_Ca,
        Ca_0,
        tau_Ca,
        k1,
        k2,
        k3,
        k4,
        n_P,
        gamma_e,
        gamma_r,
        d_phi,
        noise,
        ext_current_t,
        ext_current_r,
        N_cor2tha,
        N_tha2cor,
        N_rt,
        N_tr,
        N_rr,
        V_t,
        V_r,
        Q_t,
        Q_r,
        Ca,
        h_T_t,
        h_T_r,
        m_h1,
        m_h2,
        s_et,
        s_gt,
        s_er,
        s_gr,
        ds_et,
        ds_gt,
        ds_er,
        ds_gr, # below for aln
        c_gl, Ke_gl,
        tau_ou, sigma_ou,
        mue_ext_mean, mui_ext_mean,
        sigmae_ext, sigmai_ext,
        Ke, Ki,
        tau_se, tau_si,
        cee, cie, cii, cei,
        Jee_max, Jei_max, Jie_max, Jii_max,
        a, b,
        EA, tauA,
        C, taum,
        mufe, mufi,
        IA,
        seem, seim, seev, seiv, siim, siem, siiv, siev,
        precalc_r, precalc_V, precalc_tau_mu,
        dI, ds,
        sigmarange, Irange,
        Q_e, Q_i,
        ndt_de, ndt_di, ndt_dall,
        mue_ext, mui_ext,
    )


    return t, V_t, V_r, Q_t, Q_r, Q_e, Q_i
 
@numba.njit()
def timeIntegration_njit_elementwise(
    startind,
    t,
    dt,
    sqrt_dt,
    Q_max,
    C1,
    theta,
    sigma,
    g_L,
    E_L,
    g_AMPA,
    g_GABA,
    E_AMPA,
    E_GABA,
    g_LK,
    E_K,
    g_T_t,
    g_T_r,
    E_Ca,
    g_h,
    g_inc,
    E_h,
    C_m,
    tau,
    alpha_Ca,
    Ca_0,
    tau_Ca,
    k1,
    k2,
    k3,
    k4,
    n_P,
    gamma_e,
    gamma_r,
    d_phi,
    noise,
    ext_current_t,
    ext_current_r,
    N_cor2tha,
    N_tha2cor,
    N_rt,
    N_tr,
    N_rr,
    V_t,
    V_r,
    Q_t,
    Q_r,
    Ca,
    h_T_t,
    h_T_r,
    m_h1,
    m_h2,
    s_et,
    s_gt,
    s_er,
    s_gr,
    ds_et,
    ds_gt,
    ds_er,
    ds_gr, # below for aln
    c_gl, Ke_gl,
    tau_ou, sigma_ou,
    mue_ext_mean, mui_ext_mean,
    sigmae_ext, sigmai_ext,
    Ke, Ki,
    tau_se, tau_si,
    cee, cie, cii, cei,
    Jee_max, Jei_max, Jie_max, Jii_max,
    a, b,
    EA, tauA,
    C, taum,
    mufe, mufi,
    IA,
    seem, seim, seev, seiv, siim, siem, siiv, siev,
    precalc_r, precalc_V, precalc_tau_mu,
    dI, ds,
    sigmarange, Irange,
    Q_e, Q_i,
    ndt_de, ndt_di, ndt_dall,
    mue_ext, mui_ext,
):
    def _firing_rate(voltage):
        return Q_max / (1.0 + np.exp(-C1 * (voltage - theta) / sigma))

    def _leak_current(voltage):
        return g_L * (voltage - E_L)

    def _potassium_leak_current(voltage):
        return g_LK * (voltage - E_K)

    def _syn_exc_current(voltage, synaptic_rate):
        return g_AMPA * synaptic_rate * (voltage - E_AMPA)

    def _syn_inh_current(voltage, synaptic_rate):
        return g_GABA * synaptic_rate * (voltage - E_GABA)

    for i in range(startind, len(t)):
        # thalamus derivatives
        # leak current
        I_leak_t = _leak_current(V_t[i - 1])
        I_leak_r = _leak_current(V_r[i - 1])

        # synaptic currents
        I_et = _syn_exc_current(V_t[i - 1], s_et)
        I_gt = _syn_inh_current(V_t[i - 1], s_gt)
        I_er = _syn_exc_current(V_r[i - 1], s_er)
        I_gr = _syn_inh_current(V_r[i - 1], s_gr)

        # potassium leak current
        I_LK_t = _potassium_leak_current(V_t[i - 1])
        I_LK_r = _potassium_leak_current(V_r[i - 1])

        # T-type Ca current
        m_inf_T_t = 1.0 / (1.0 + np.exp(-(V_t[i - 1] + 59.0) / 6.2))
        m_inf_T_r = 1.0 / (1.0 + np.exp(-(V_r[i - 1] + 52.0) / 7.4))
        I_T_t = g_T_t * m_inf_T_t * m_inf_T_t * h_T_t * (V_t[i - 1] - E_Ca)
        I_T_r = g_T_r * m_inf_T_r * m_inf_T_r * h_T_r * (V_r[i - 1] - E_Ca)

        # h-type current
        I_h = g_h * (m_h1 + g_inc * m_h2) * (V_t[i - 1] - E_h)

        ### define derivatives
        # membrane potential
        d_V_t = -(I_leak_t + I_et + I_gt + ext_current_t) / tau - (1.0 / C_m) * (I_LK_t + I_T_t + I_h)
        d_V_r = -(I_leak_r + I_er + I_gr + ext_current_r) / tau - (1.0 / C_m) * (I_LK_r + I_T_r)
        # Calcium concentration
        d_Ca = alpha_Ca * I_T_t - (Ca - Ca_0) / tau_Ca
        # channel dynamics
        h_inf_T_t = 1.0 / (1.0 + np.exp((V_t[i - 1] + 81.0) / 4.0))
        h_inf_T_r = 1.0 / (1.0 + np.exp((V_r[i - 1] + 80.0) / 5.0))
        tau_h_T_t = (
            30.8 + (211.4 + np.exp((V_t[i - 1] + 115.2) / 5.0)) / (1.0 + np.exp((V_t[i - 1] + 86.0) / 3.2))
        ) / 3.7371928
        tau_h_T_r = (
            85.0 + 1.0 / (np.exp((V_r[i - 1] + 48.0) / 4.0) + np.exp(-(V_r[i - 1] + 407.0) / 50.0))
        ) / 3.7371928
        d_h_T_t = (h_inf_T_t - h_T_t) / tau_h_T_t
        d_h_T_r = (h_inf_T_r - h_T_r) / tau_h_T_r
        m_inf_h = 1.0 / (1.0 + np.exp((V_t[i - 1] + 75.0) / 5.5))
        tau_m_h = 20.0 + 1000.0 / (np.exp((V_t[i - 1] + 71.5) / 14.2) + np.exp(-(V_t[i - 1] + 89.0) / 11.6))
        # Calcium channel dynamics
        P_h = k1 * Ca ** n_P / (k1 * Ca ** n_P + k2)
        d_m_h1 = (m_inf_h * (1.0 - m_h2) - m_h1) / tau_m_h - k3 * P_h * m_h1 + k4 * m_h2
        d_m_h2 = k3 * P_h * m_h1 - k4 * m_h2
        # synaptic dynamics
        d_s_et = ds_et
        d_s_er = ds_er
        d_s_gt = ds_gt
        d_s_gr = ds_gr
        d_ds_et = 0.0
        d_ds_et = gamma_e ** 2 * (N_cor2tha * Q_e[i-ndt_dall-1] - s_et) - 2 * gamma_e * ds_et
        d_ds_er = gamma_e ** 2 * (N_rt * _firing_rate(V_t[i - 1]) + N_cor2tha * Q_e[i-ndt_dall-1] - s_er) - 2 * gamma_e * ds_er
        # d_ds_er = gamma_e ** 2 * (N_rt * _firing_rate(V_t[0, i - 1]) + N_rp * cortical_rowsum - s_er) - 2 * gamma_e * ds_er
        d_ds_gt = gamma_r ** 2 * (N_tr * _firing_rate(V_r[i - 1]) - s_gt) - 2 * gamma_r * ds_gt
        d_ds_gr = gamma_r ** 2 * (N_rr * _firing_rate(V_r[i - 1]) - s_gr) - 2 * gamma_r * ds_gr

        ### Euler integration
        V_t[i] = V_t[i - 1] + dt * d_V_t
        V_r[i] = V_r[i - 1] + dt * d_V_r
        Q_t[i] = _firing_rate(V_t[i]) * 1e3  # convert kHz to Hz
        Q_r[i] = _firing_rate(V_r[i]) * 1e3  # convert kHz to Hz
        Ca = Ca + dt * d_Ca
        h_T_t = h_T_t + dt * d_h_T_t
        h_T_r = h_T_r + dt * d_h_T_r
        m_h1 = m_h1 + dt * d_m_h1
        m_h2 = m_h2 + dt * d_m_h2
        s_et = s_et + dt * d_s_et
        s_gt = s_gt + dt * d_s_gt
        s_er = s_er + dt * d_s_er
        s_gr = s_gr + dt * d_s_gr
        # noisy variable
        ds_et = ds_et + dt * d_ds_et + gamma_e ** 2 * d_phi * sqrt_dt * noise[i - startind]
        ds_gt = ds_gt + dt * d_ds_gt
        ds_er = ds_er + dt * d_ds_er
        ds_gr = ds_gr + dt * d_ds_gr

        # aln deravatives
        sq_Jee_max = Jee_max ** 2
        sq_Jei_max = Jei_max ** 2
        sq_Jie_max = Jie_max ** 2
        sq_Jii_max = Jii_max ** 2

        noise_exc = Q_e[i]
        noise_inh = Q_i[i]

        mue = Jee_max * seem + Jei_max * seim + mue_ext
        mui = Jie_max * siem + Jii_max * siim + mui_ext

        Q_e_d = Q_e[i-ndt_de-1] # is it necessary to -1
        Q_i_d = Q_i[i-ndt_di-1]

        z1ee = cee*Ke*Q_e_d + c_gl*Ke_gl*N_tha2cor*Q_t[i-ndt_dall-1]
        z1ei = cei*Ki*Q_i_d
        z1ie = cie*Ke*Q_e_d
        z1ii = cii*Ki*Q_i_d
        z2ee = cee**2*Ke*Q_e_d + c_gl*Ke_gl*N_tha2cor*Q_t[i-ndt_dall-1]
        z2ei = cei**2*Ki*Q_i_d
        z2ie = cie**2*Ke*Q_e_d
        z2ii = cii**2*Ki*Q_i_d

        sigmae = np.sqrt(
            2*sq_Jee_max*seev*tau_se*taum / ((1+z1ee)*taum+tau_se) 
            + 2*sq_Jei_max*seiv*tau_si*taum / ((1+z1ei)*taum+tau_si)
            + sigmae_ext**2
            )
        sigmai = np.sqrt(
            2*sq_Jie_max*siev*tau_se*taum / ((1+z1ie)*taum+tau_se) 
            + 2*sq_Jii_max*siiv*tau_si*taum / ((1+z1ii)*taum+tau_si)
            + sigmai_ext**2
            )
        
        ## look up from the table
        xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmae, Irange, dI, mufe-IA/C)
        xid1, yid1 = int(xid1), int(yid1)
        Q_e[i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid)
        Vmean_exc = interpolate_values(precalc_V, xid1, yid1, dxid, dyid)
        tau_exc = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
        
        xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai, Irange, dI, mufi)
        xid1, yid1 = int(xid1), int(yid1)
        Q_i[i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid)
        tau_inh = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
 

        # r.h.s
        mufe_rhs = (mue - mufe)/tau_exc
        mufi_rhs = (mui - mufi)/tau_inh

        IA_rhs = (a*(Vmean_exc-EA) - IA +tauA*b*Q_e[i])/tauA

        seem_rhs = ((1 - seem) * z1ee - seem)/tau_se
        seim_rhs = ((1 - seim) * z1ei - seim)/tau_si
        siem_rhs = ((1 - siem) * z1ie - siem)/tau_se
        siim_rhs = ((1 - siim) * z1ii - siim)/tau_si

        ## ? why z1ee
        seev_rhs = ((1 - seem)**2 * z2ee + (z2ee - 2*tau_se*(z1ee+1))*seev)/tau_se**2
        seiv_rhs = ((1 - seim)**2 * z2ei + (z2ei - 2*tau_si*(z1ei+1))*seiv)/tau_si**2
        siev_rhs = ((1 - siem)**2 * z2ie + (z2ie - 2*tau_se*(z1ie+1))*siev)/tau_se**2
        siiv_rhs = ((1 - siim)**2 * z2ii + (z2ii - 2*tau_si*(z1ii+1))*siiv)/tau_si**2


        mufe = mufe + dt * mufe_rhs
        mufi = mufi + dt * mufi_rhs
        seem = seem + dt * seem_rhs
        seim = seim + dt * seim_rhs
        siem = siem + dt * siem_rhs
        siim = siim + dt * siim_rhs
        
        seev = seev + dt * seev_rhs
        seiv = seiv + dt * seiv_rhs
        siev = siev + dt * siev_rhs
        siiv = siiv + dt * siiv_rhs
       
        # can this part be more simple?
        if seev < 0:
            seev = 0
        if siev < 0:
            siev = 0
        if seiv < 0:
            seiv = 0
        if siiv < 0:
            siiv = 0

        IA = IA + dt * IA_rhs

        mue_ext = mue_ext + (mue_ext_mean - mue_ext)*dt/tau_ou + sigma_ou*sqrt_dt*noise_exc
        mui_ext = mui_ext + (mui_ext_mean - mui_ext)*dt/tau_ou + sigma_ou*sqrt_dt*noise_inh

    return t, V_t, V_r, Q_t, Q_r, Q_e, Q_i

@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64})
def interpolate_values(table, xid1, yid1, dxid, dyid):
    output = (
        table[yid1, xid1] * (1 - dxid) * (1 - dyid)
        + table[yid1, xid1 + 1] * dxid * (1 - dyid)
        + table[yid1 + 1, xid1] * (1 - dxid) * dyid
        + table[yid1 + 1, xid1 + 1] * dxid * dyid
    )
    return output



@numba.njit(locals={"xid1": numba.int64, "yid1": numba.int64, "dxid": numba.float64, "dyid": numba.float64})
def fast_interp2_opt(x, dx, xi, y, dy, yi):
    """
    Returns the values needed for interpolation:
    - bilinear (2D) interpolation within ranges,
    - linear (1D) if "one edge" is crossed,
    - corner value if "two edges" are crossed

    x     ... range of the x value
    xi    ... interpolation value on x-axis
    dx    ... grid width of x ( dx = x[1]-x[0] )
    (same for y)

    return:   xid1    ... index of the lower interpolation value
              dxid    ... distance of xi to the lower interpolation value
              (same for y)
    """

    # within all boundaries
    if xi >= x[0] and xi < x[-1] and yi >= y[0] and yi < y[-1]:
        xid = (xi - x[0]) / dx
        xid1 = np.floor(xid)
        dxid = xid - xid1
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    # outside one boundary
    if yi < y[0]:
        yid1 = 0
        dyid = 0.0
        if xi >= x[0] and xi < x[-1]:
            xid = (xi - x[0]) / dx
            xid1 = np.floor(xid)
            dxid = xid - xid1

        elif xi < x[0]:
            xid1 = 0
            dxid = 0.0
        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return xid1, yid1, dxid, dyid

    if yi >= y[-1]:
        yid1 = -1
        dyid = 0.0
        if xi >= x[0] and xi < x[-1]:
            xid = (xi - x[0]) / dx
            xid1 = np.floor(xid)
            dxid = xid - xid1

        elif xi < x[0]:
            xid1 = 0
            dxid = 0.0

        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return xid1, yid1, dxid, dyid

    if xi < x[0]:
        xid1 = 0
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    if xi >= x[-1]:
        xid1 = -1
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1

    return xid1, yid1, dxid, dyid

