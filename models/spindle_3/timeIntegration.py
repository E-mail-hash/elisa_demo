"""
for spindles and SO(ALN)
"""
import numpy as np
import numba
import defaultParameters as dp

def timeIntegration(params):
    # simulation parameters
    dt = params["dt"]
    dt2 = dt * 1000
    sqrt_dt2 = np.sqrt(dt2)
    duration = params["duration"]
    RNGseed = params["seed"]

    t = np.arange(1, round(duration, 6)/dt+1)

    # neuron parameters
    #startind = 1 


    tau_t = params["tau_t"]
    tau_r = params["tau_r"]

    f_t_max = params["f_t_max"]
    f_r_max = params["f_r_max"]
    f_t_th = params["f_t_th"]
    f_r_th = params["f_r_th"]
    gamma_t = params["gamma_t"]
    gamma_r = params["gamma_r"]
    At = params["At"]
    Ar = params["Ar"]

    tau_u_t = params["tau_u_t"]
    tau_u_r = params["tau_u_r"]
    RT_t = params["RT_t"]
    RT_r = params["RT_r"]
    VT_t = params["VT_t"]
    VT_r = params["VT_r"]
    gT_t = params["gT_t"]
    gT_r = params["gT_r"]
    Lt = params["Lt"]
    Lr = params["Lr"]
    RB_t = params["RB_t"]
    RB_r = params["RB_r"]
    VB_t = params["VB_t"]
    VB_r = params["VB_r"]
    gB_t = params["gB_t"]
    gB_r = params["gB_r"]   

    Nt = params["Nt"]
    Nr = params["Nr"]
    Prr = params["Prr"]
    Ptr = params["Ptr"]
    Prt = params["Prt"]
    Jrr = params["Jrr"]
    Jtr = params["Jtr"]
    Jrt = params["Jrt"]

    # cortex
    tau_e  = params["tau_e"]
    tau_i  = params["tau_i"]
    tau_c  = params["tau_c"]

    Ne     = params["Ne"]
    Ni     = params["Ni"]
    delta_c= params["delta_c"]
    delta_c2=params["delta_c2"]

    ge     = params["ge"]
    gi     = params["gi"]
    Rm     = params["Rm"]
    V_star = params["V_star"]
    c_star = params["c_star"]
    gc     = params["gc"]

    Jee0   = params["Jee0"]
    Je2e0  = params["Je2e0"]
    Jee20  = params["Jee20"]

    Jet    = params["Jet"]
    Je2t   = params["Je2t"]
    Jer    = params["Jer"]
    Je2r   = params["Je2r"]

    Jie    = params["Jie"]
    Jte    = params["Jte"]
    Jii    = params["Jii"]
    Jei    = params["Jei"]
    Je2i   = params["Je2i"]
    Jti    = params["Jti"]

    Ji2e2  = params["Ji2e2"]
    Jte2   = params["Jte2"]
    Ji2i2  = params["Ji2i2"]
    Je2i2  = params["Je2i2"]
    Jei2   = params["Jei2"]
    Jti2   = params["Jti2"]

    Pee    = params["Pee"]
    Pe2e   = params["Pe2e"]
    Pee2   = params["Pee2"]
    Pie    = params["Pie"]
    Pte    = params["Pte"]
    Pii    = params["Pii"]
    Pei    = params["Pei"]
    Pe2i   = params["Pe2i"]
    Pei2   = params["Pei2"]
    Pti    = params["Pti"]
    Pet    = params["Pet"]
    Per    = params["Per"]

    # ALN
    de = params['de']
    di = params['di']
    ndt_de = int(de/dt2)
    ndt_di = int(di/dt2)

    startind = int(np.max([ndt_de, ndt_di]))

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

    sq_Jee_max = Jee_max ** 2
    sq_Jei_max = Jei_max ** 2
    sq_Jie_max = Jie_max ** 2
    sq_Jii_max = Jii_max ** 2

    
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
    '''
    these are all scalars in defaultParameters 
    '''
    V_e = np.dot(params["V_e_init"], np.ones(len(t)))
    V_i = np.dot(params["V_i_init"], np.ones(len(t)))
    Q_e = np.dot(params["Q_e_init"], np.ones(len(t)))
    Q_i = np.dot(params["Q_i_init"], np.ones(len(t)))
    c = np.dot(params["c_init"], np.ones(len(t)))
    V_e2 = np.dot(params["V_e2_init"], np.ones(len(t)))
    V_i2 = np.dot(params["V_i2_init"], np.ones(len(t)))
    c2 = np.dot(params["c2_init"], np.ones(len(t)))
    
    V_t = np.dot(params["V_t_init"], np.ones(len(t)))
    V_r = np.dot(params["V_r_init"], np.ones(len(t)))
    Q_t = np.dot(params["Q_t_init"], np.ones(len(t)))
    Q_r = np.dot(params["Q_r_init"], np.ones(len(t)))
    u_t = float(params["u_t_init"])
    u_r = float(params["u_r_init"])

    # ALN
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

    if RNGseed:
        np.random.seed(RNGseed)
    Q_e[startind:] = np.random.standard_normal(len(range(startind, len(t))))
    Q_i[startind:] = np.random.standard_normal(len(range(startind, len(t))))



    a_chunk = np.zeros(len(t))
    b_chunk = np.zeros(len(t))
    c_chunk = np.zeros(len(t))
    u_chunk = np.zeros(len(t))

    ar_chunk = np.zeros(len(t))
    br_chunk = np.zeros(len(t))
    cr_chunk = np.zeros(len(t))
    dr_chunk = np.zeros(len(t))
    ur_chunk = np.zeros(len(t))

    return timeIntegration_njit_elementwise(  
    startind, t, dt,
    V_e, V_i, Q_e, Q_i, c, V_e2, V_i2, c2,
    tau_e, tau_i, tau_c,
    Ne, Ni, delta_c, delta_c2,
    ge, gi, Rm, V_star, c_star, gc, 
    Jee0, Je2e0, Jee20, 
    Jet, Je2t, Jer, Je2r, 
    Jie, Jte, Jii, Jei, Je2i, Jti, 
    Ji2e2, Jte2, Ji2i2, Je2i2, Jei2, Jti2,
    Pee, Pe2e, Pee2, Pie, Pte, Pii, Pei, Pe2i, Pei2, Pti, Pet, Per,# for cortex
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, tau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt, # below for ALN
    dt2, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv, mue_ext, mui_ext,
    ndt_de, ndt_di, Jee_max, Jei_max, Jie_max, Jii_max,
    cee, Ke, c_gl, Ke_gl, cei, Ki, cie, cii,
    sq_Jee_max, tau_se, taum, sq_Jei_max, tau_si, sigmae_ext,
    sq_Jie_max, sq_Jii_max, sigmai_ext,
    sigmarange, ds, Irange, dI, C,
    precalc_r, precalc_V, precalc_tau_mu,
    a, EA, tauA, b, 
    mue_ext_mean, mui_ext_mean, 
    tau_ou, sigma_ou, sqrt_dt2
    )



# @numba.njit()
def timeIntegration_njit_elementwise(
    startind, t, dt,
    V_e, V_i, Q_e, Q_i, c, V_e2, V_i2, c2,
    tau_e, tau_i, tau_c,
    Ne, Ni, delta_c, delta_c2,
    ge, gi, Rm, V_star, c_star, gc, 
    Jee0, Je2e0, Jee20, 
    Jet, Je2t, Jer, Je2r, 
    Jie, Jte, Jii, Jei, Je2i, Jti, 
    Ji2e2, Jte2, Ji2i2, Je2i2, Jei2, Jti2,
    Pee, Pe2e, Pee2, Pie, Pte, Pii, Pei, Pe2i, Pei2, Pti, Pet, Per,# for cortex
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, tau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt, # below for ALN
    dt2, mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv, mue_ext, mui_ext,
    ndt_de, ndt_di, Jee_max, Jei_max, Jie_max, Jii_max,
    cee, Ke, c_gl, Ke_gl, cei, Ki, cie, cii,
    sq_Jee_max, tau_se, taum, sq_Jei_max, tau_si, sigmae_ext,
    sq_Jie_max, sq_Jii_max, sigmai_ext,
    sigmarange, ds, Irange, dI, C,
    precalc_r, precalc_V, precalc_tau_mu,
    a, EA, tauA, b, 
    mue_ext_mean, mui_ext_mean, 
    tau_ou, sigma_ou, sqrt_dt2
    ):

    def rhs(i, y):
        """
        返回 4 个导数：dVt/dt, dVr/dt, dut/dt, dur/dt
        """
        Vt, Vr, ut, ur = y[0], y[1], y[2], y[3]
        Ve, Vi, c      = y[4], y[5], y[6]
        Ve2, Vi2, c2   = y[7], y[8], y[9]

        mufe, mufi, IA = y[10], y[11], y[12]
        seem, seim, siem, siim = y[13:17]
        seev, seiv, siev, siiv = np.maximum(y[17:21],0.0) #y[7], y[8], y[9], y[10]
        mue_ext, mui_ext = y[21], y[22]



        Qt = RT_t * np.exp(Lt*ut) / (1 + np.exp((VT_t-Vt)/gT_t)) + RB_t * (1-np.exp(Lt*ut)) / (1 + np.exp((VB_t-Vt)/gB_t))
        Qr = RT_r * np.exp(Lr*ur) / (1 + np.exp((VT_r-Vr)/gT_r)) + RB_r * (1-np.exp(Lr*ur)) / (1 + np.exp((VB_r-Vr)/gB_r))

        '''
        def Q_m(V, type):
            if type == 0: # e
                gm = ge
            elif type == 1: # i
                gm = gi 
            return Rm / (1 + np.exp(-(V - V_star) / gm)) # +0.1

        Qe = Q_m(Ve, 0)
        Qe2 = Q_m(Ve2, 0)
        Qi = Q_m(Vi, 1)
        Qi2 = Q_m(Vi2, 1)
        '''
        Qe = Q_e[i-ndt_de-1] # is it necessary to -1
        Qi = Q_i[i-ndt_di-1]
        Qe2 = 0
        Qi2 = 0 



        # 快速计算 u(V)
        def fu(u, m):
            if m == 0:   # t
                return f_t_max / (1 + np.exp((u+f_t_th)/gamma_t))
            else:        # r
                return f_r_max / (1 + np.exp((u+f_r_th)/gamma_r))

        dVt = -Vt/tau_t + fu(ut,0)/At + Nr*Prt*Jrt*Qr + Ne*Pet*Jet*Qe + Ne*Pet*Je2t*Qe2
        dVr = -Vr/tau_r + fu(ur,1)/Ar + Nr*Prr*Jrr*Qr + Nt*Ptr*Jtr*Qt + Ne*Per*Jer*Qe + Ne*Per*Je2r*Qe2

        '''
        def J(c, type):
            if type == "ee":
                J0 = Jee0
            elif type == "e2e":
                J0 = Je2e0
            elif type == "ee2":
                J0 = Jee20
            return J0 / (1 + np.exp((c - c_star) / gc))
       
        dVe = -Ve/tau_e + Ne*Pee*J(c, "ee")*Qe + Ni*Pie*Jie*Qi + Ne*Pe2e*J(c, "e2e")*Qe2 + Nt*Pte*Jte*Qt
        dVi = -Vi/tau_i + Ni*Pii*Jii*Qi + Ne*Pei*Jei*Qe + Ne*Pe2i*Je2i*Qe2 + Nt*Pti*Jti*Qt
        dc = -c/tau_c + delta_c*(Ne*Pee*Qe + Ne*Pe2e*Qe2)

        dVe2 = -Ve2/tau_e + Ne*Pee*J(c2, "ee")*Qe2 + Ni*Pie*Ji2e2*Qi2 + Ne*Pee2*J(c2,"ee2")*Qe + Nt*Pte*Jte2*Qt
        dVi2 = -Vi2/tau_i + Ni*Pii*Ji2i2*Qi2 + Ne*Pei*Je2i2*Qe2 + Ne*Pei2*Jei2*Qe + Nt*Pti*Jti2*Qt
        dc2 = -c2/tau_c + delta_c2*(Ne*Pee*Qe2 + Ne*Pee2*Qe)

        '''
        # b(V) 函数
        def bV(V, m):
            Balance = 0
            if m == 0:  # t
                return -200.0 if V - Balance < -0.1 else 0.0 
            else:       # r
                return -200.0 if V - Balance <= 0.0 else 0.0

        dut = (bV(Vt,0) - ut) / tau_u_t
        dur = (bV(Vr,1) - ur) / tau_u_r

        # ALN
        mue = Jee_max * seem + Jei_max * seim + mue_ext
        mui = Jie_max * siem + Jii_max * siim + mui_ext

        Q_e_d = Q_e[i-ndt_de-1] # is it necessary to -1
        Q_i_d = Q_i[i-ndt_di-1]

        z1ee = cee*Ke*Q_e_d #+ c_gl*Ke_gl*Qt # 100*Qt
        if 0 :#<= 1000:
            print("Q_e_d:", Q_e_d)
            print("z1ee:", z1ee)
        z1ei = cei*Ki*Q_i_d
        z1ie = cie*Ke*Q_e_d
        z1ii = cii*Ki*Q_i_d
        z2ee = cee**2*Ke*Q_e_d #+ c_gl**2*Ke_gl*Qt
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
        #Q_e[i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid)
        Qe = interpolate_values(precalc_r, xid1, yid1, dxid, dyid)
        Vmean_exc = interpolate_values(precalc_V, xid1, yid1, dxid, dyid)
        tau_exc = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
        
        xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai, Irange, dI, mufi)
        xid1, yid1 = int(xid1), int(yid1)
        Qi = interpolate_values(precalc_r, xid1, yid1, dxid, dyid)
        #Q_i[i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid)
        tau_inh = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
 

        # r.h.s
        mufe_rhs = (mue - mufe)/tau_exc
        mufi_rhs = (mui - mufi)/tau_inh


        #IA_rhs = (a*(Vmean_exc-EA) - IA +tauA*b*Qe)/tauA
        IA_rhs = (a*(Vmean_exc-EA) - IA +tauA*b*Q_e[i])/tauA

        seem_rhs = ((1 - seem) * z1ee - seem)/tau_se
        if 0 :#<= 1000:
            print("seem:", seem)
            print("i, seem_rhs:", i, seem_rhs)
        seim_rhs = ((1 - seim) * z1ei - seim)/tau_si
        siem_rhs = ((1 - siem) * z1ie - siem)/tau_se
        siim_rhs = ((1 - siim) * z1ii - siim)/tau_si

        ## ? why z1ee
        seev_rhs = ((1 - seem)**2 * z2ee + (z2ee - 2*tau_se*(z1ee+1))*seev)/tau_se**2
        seiv_rhs = ((1 - seim)**2 * z2ei + (z2ei - 2*tau_si*(z1ei+1))*seiv)/tau_si**2
        siev_rhs = ((1 - siem)**2 * z2ie + (z2ie - 2*tau_se*(z1ie+1))*siev)/tau_se**2
        siiv_rhs = ((1 - siim)**2 * z2ii + (z2ii - 2*tau_si*(z1ii+1))*siiv)/tau_si**2

        mue_ext_rhs = (mue_ext_mean - mue_ext) / tau_ou
        mui_ext_rhs = (mui_ext_mean - mui_ext) / tau_ou
 
        dVe= dVi= dc= dVe2= dVi2= dc2 = 0

        return np.array([dVt, dVr, dut, dur, dVe, dVi, dc, dVe2, dVi2, dc2,
                        mufe_rhs, mufi_rhs, IA_rhs, 
                        seem_rhs, seim_rhs, siem_rhs, siim_rhs, 
                        seev_rhs, seiv_rhs, siev_rhs, siiv_rhs, 
                        mue_ext_rhs, mui_ext_rhs])

    def rk4_step(y, dt, dt2, i):
        scale = np.zeros(len(y))
        scale[0:4] = dt
        scale[10:23] = dt2
        k1 = rhs(i, y)
        k2 = rhs(i, y + 0.5*scale*k1)
        k3 = rhs(i, y + 0.5*scale*k2)
        k4 = rhs(i, y + scale*k3)
        y_next = y + scale*(k1 + 2*k2 + 2*k3 + k4)/6.0 
        return y_next    
    '''
    def Q_m(V, type):
        if type == 0: # e
            gm = ge
        elif type == 1: # i
            gm = gi 
        return Rm / (1 + np.exp(-(V - V_star) / gm)) # +0.1
    '''
    def calc_Q(i):
        Q_e_d = Q_e[i-ndt_de-1] # is it necessary to -1
        Q_i_d = Q_i[i-ndt_di-1]

        z1ee = cee*Ke*Q_e_d #+ c_gl*Ke_gl*Q_t[i-1]
        z1ei = cei*Ki*Q_i_d
        z1ie = cie*Ke*Q_e_d
        z1ii = cii*Ki*Q_i_d

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
 

    for i in range(startind, len(t)):

        noise_exc = Q_e[i]
        noise_inh = Q_i[i]

        Q_t[i-1] = RT_t * np.exp(Lt*u_t) / (1 + np.exp((VT_t-V_t[i-1])/gT_t)) + RB_t * (1-np.exp(Lt*u_t)) / (1 + np.exp((VB_t-V_t[i-1])/gB_t))
        Q_r[i-1] = RT_r * np.exp(Lr*u_r) / (1 + np.exp((VT_r-V_r[i-1])/gT_r)) + RB_r * (1-np.exp(Lr*u_r)) / (1 + np.exp((VB_r-V_r[i-1])/gB_r))


        # RK4 单步
        y = np.array([V_t[i-1], V_r[i-1], u_t, u_r, V_e[i-1], V_i[i-1], c[i-1], V_e2[i-1], V_i2[i-1], c2[i-1],
                      mufe, mufi, IA, 
                      seem, seim, siem, siim, 
                      seev, seiv, siev, siiv,
                      mue_ext, mui_ext])

        y = rk4_step(y, dt, dt2, i)
        V_t[i], V_r[i], u_t, u_r = y[0], y[1], y[2], y[3]
        #V_e[i], V_i[i], c[i]     = y[4], y[5], y[6]
        #V_e2[i], V_i2[i], c2[i]  = y[7], y[8], y[9]
        mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv, mue_ext, mui_ext = y[10:23]

        mue_ext = mue_ext + sigma_ou*sqrt_dt2*noise_exc
        mui_ext = mui_ext + sigma_ou*sqrt_dt2*noise_inh

        calc_Q(i)


        u_chunk[i]  = u_t
        ur_chunk[i] = u_r
   

    return Q_t, Q_r, V_t, V_r, Q_e, Q_i, V_e, V_i, c, V_e2, V_i2, c2, t

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

