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
    #t = np.arange(0, duration, dt)

    # neuron parameters
    ## aln
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
    
    ###? how we manage the startind here?
    '''
    Q_e = np.zeros((1, startind+len(t)))
    Q_e[:, :startind] = params['Q_e_init']
    '''
    Q_e = np.dot(params['Q_e_init'], np.ones(len(t)))
    Q_i = np.dot(params['Q_i_init'], np.ones(len(t)))

    if RNGseed:
        np.random.seed(RNGseed)
    Q_e[startind:] = np.random.standard_normal(len(range(startind, len(t))))
    Q_i[startind:] = np.random.standard_normal(len(range(startind, len(t))))

 
    # it is just a scalar
    #Q_e_d = 0
    #Q_i_d = 0

    rate_thalamus = 0

    def rhs(i, y):

        mufe, mufi, IA = y[0], y[1], y[2]
        seem, seim, siem, siim = np.maximum(y[3:7], 0)
        seev, seiv, siev, siiv = np.maximum(y[7:11], 0)#y[7], y[8], y[9], y[10]
        mue_ext, mui_ext = y[11], y[12]


        mue = Jee_max * seem + Jei_max * seim + mue_ext
        mui = Jie_max * siem + Jii_max * siim + mui_ext

        Q_e_d = Q_e[i-ndt_de-1] # is it necessary to -1
        Q_i_d = Q_i[i-ndt_di-1]

        z1ee = cee*Ke*Q_e_d + c_gl*Ke_gl*rate_thalamus
        if 0 :#<= 1000:
            print("Q_e_d:", Q_e_d)
            print("z1ee:", z1ee)
        z1ei = cei*Ki*Q_i_d
        z1ie = cie*Ke*Q_e_d
        z1ii = cii*Ki*Q_i_d
        z2ee = cee**2*Ke*Q_e_d + c_gl*Ke_gl*rate_thalamus
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


        IA_rhs = (a*(Vmean_exc-EA) - IA +tauA*b*Qe)/tauA
        #IA_rhs = (a*(Vmean_exc-EA) - IA +tauA*b*Q_e[i])/tauA

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
        
        return np.array([mufe_rhs, mufi_rhs, IA_rhs, 
                        seem_rhs, seim_rhs, siem_rhs, siim_rhs, 
                        seev_rhs, seiv_rhs, siev_rhs, siiv_rhs, 
                        mue_ext_rhs, mui_ext_rhs])

    def rk4_step(y, dt, i):
        k1 = rhs(i, y)
        scale = np.zeros(len(k1))
        scale[0:len(k1)] = dt 
        tmp = y + 0.5*scale*k1               # 先正常算
        tmp[7:11] = np.maximum(tmp[7:11], 0.0)  # 只裁 7-10
        k2 = rhs(i, tmp)

        tmp = y + 0.5*dt*k2
        tmp[7:11] = np.maximum(tmp[7:11], 0.0)
        k3 = rhs(i, tmp)

        tmp = y + dt*k3
        tmp[7:11] = np.maximum(tmp[7:11], 0.0)
        k4 = rhs(i, tmp)
        y_next = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0 
        return y_next    
 
    '''
    def rk4_step(y, dt, i):
        k1 = rhs(i, y)
        k2 = rhs(i, y + 0.5*dt*k1)
        k3 = rhs(i, y + 0.5*dt*k2)
        k4 = rhs(i, y + dt*k3)
        y_next = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0 
        return y_next    
    '''
    def calc_Q(i):
        Q_e_d = Q_e[i-ndt_de-1] # is it necessary to -1
        Q_i_d = Q_i[i-ndt_di-1]

        z1ee = cee*Ke*Q_e_d + c_gl*Ke_gl*rate_thalamus
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
 

    # derivatives
    for i in range(startind, len(t)):
        noise_exc = Q_e[i]
        noise_inh = Q_i[i]

        y = np.array([mufe, mufi, IA, 
                      seem, seim, siem, siim, 
                      seev, seiv, siev, siiv,
                      mue_ext, mui_ext])
        y = rk4_step(y, dt2, i)

        mufe, mufi, IA, seem, seim, siem, siim, seev, seiv, siev, siiv, mue_ext, mui_ext = y

        if seev < 0:
            seev = 0
        if siev < 0:
            siev = 0
        if seiv < 0:
            seiv = 0
        if siiv < 0:
            siiv = 0

        # can this part be more simple?
        mue_ext = mue_ext + sigma_ou*sqrt_dt2*noise_exc
        mui_ext = mui_ext + sigma_ou*sqrt_dt2*noise_inh

        calc_Q(i)



    return Q_e, Q_i, t



@numba.njit()
def timeIntegration_njit_elementwise():
    pass

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

