"""
for spindles 
"""
import numpy as np
import numba
import defaultParameters as dp

def timeintegration(params):
    # simulation parameters
    dt = params["dt"]
    duration = params["duration"]

    t = np.arange(1, round(duration, 6)/dt+1)

    # neuron parameters
    startind = 1 

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
    V_e    = params["V_e"]
    V_i    = params["V_i"]
    c      = params["c"]
    V_e2   = params["V_e2"]
    V_i2   = params["V_i2"]
    c2     = params["c2"]

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
    Jte2   = params["Jte2"]
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


    # neuron initialization
    '''
    these are all scalars in defaultParameters 
    '''
    V_e = np.dot(params["V_e_init"], np.ones(len(t)))
    V_i = np.dot(params["V_i_init"], np.ones(len(t)))
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
    V_e, V_i, c, V_e2, V_i2, c2,
    tau_e, tau_i, tau_c,
    Ne, Ni, delta_c, delta_c2,
    ge, gi, Rm, V_star, c_star, gc, 
    Jee0, Je2e0, Jee20, Jet, Je2t, Jer, Je2r, Jie, Jte, Jii, Jei, Je2i, Jti, Jte2, Jei2, Jti2,
    Pee, Pe2e, Pee2, Pie, Pte, Pii, Pei, Pe2i, Pei2, Pti, Pet, Per,# for cortex
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, tau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt)



# @numba.njit()
def timeIntegration_njit_elementwise(
    startind, t, dt,
    V_e, V_i, c, V_e2, V_i2, c2,
    tau_e, tau_i, tau_c,
    Ne, Ni, delta_c, delta_c2,
    ge, gi, Rm, V_star, c_star, gc, 
    Jee0, Je2e0, Jee20, Jet, Je2t, Jer, Je2r, Jie, Jte, Jii, Jei, Je2i, Jti, Jte2, Jei2, Jti2,
    Pee, Pe2e, Pee2, Pie, Pte, Pii, Pei, Pe2i, Pei2, Pti, Pet, Per,# for cortex
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, tau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt):

    def rhs(t, y):
        """
        返回 4 个导数：dVt/dt, dVr/dt, dut/dt, dur/dt
        """
        Vt, Vr, ut, ur = y[0], y[1], y[2], y[3]
        Ve, Vi, c      = y[4], y[5], y[6]
        Ve2, Vi2, c2   = y[7], y[8], y[9]

        Qt = RT_t * np.exp(Lt*ut) / (1 + np.exp((VT_t-Vt)/gT_t)) + RB_t * (1-np.exp(Lt*ut)) / (1 + np.exp((VB_t-Vt)/gB_t))
        Qr = RT_r * np.exp(Lr*ur) / (1 + np.exp((VT_r-Vr)/gT_r)) + RB_r * (1-np.exp(Lr*ur)) / (1 + np.exp((VB_r-Vr)/gB_r))

        def Q_m(V, type):
            if type == 0: # e
                gm = ge
            elif type == 1: # i
                gm = gi 
            return Rm / (1 + np.exp(-(V - V_star) / gm))

        Qe = Q_m(Ve, 0)
        Qe2 = Q_m(Ve2, 0)
        Qi = Q_m(Vi, 1)
        Qi2 = Q_m(Vi2, 1)


        # 快速计算 u(V)
        def fu(u, m):
            if m == 0:   # t
                return f_t_max / (1 + np.exp((u+f_t_th)/gamma_t))
            else:        # r
                return f_r_max / (1 + np.exp((u+f_r_th)/gamma_r))

        dVt = -Vt/tau_t + fu(ut,0)/At + Nr*Prt*Jrt*Qt + Ne*Pet*Jet*Qe + Ne*Pet*Je2t*Qe2
        dVr = -Vr/tau_r + fu(ur,1)/Ar + Nr*Prr*Jrr*Qr + Nt*Ptr*Jtr*Qt + Ne*Per*Jer*Qe + Ne*Per*Je2r*Qe2

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

        dVe2 = -Ve2/tau_e + Ne*Pee*J(c2, "ee")*Qe2 + Ni*Pie*Jie*Qi2 + Ne*Pee2*J(c2,"ee2")*Qe + Nt*Pte*Jte2*Qt
        dVi2 = -Vi2/tau_i + Ni*Pii*Jii*Qi2 + Ne*Pei*Jei*Qe2 + Ne*Pei2*Jei2*Qe + Nt*Pti*Jti2*Qt
        dc2 = -c2/tau_c + delta_c2*(Ne*Pee*Qe2 + Ne*Pee2*Qe)

        # b(V) 函数
        def bV(V, m):
            Balance = 0
            if m == 0:  # t
                return -200.0 if V - Balance < -0.1 else 0.0
            else:       # r
                return -200.0 if V - Balance <= 0.0 else 0.0

        dut = (bV(Vt,0) - ut) / tau_u_t
        dur = (bV(Vr,1) - ur) / tau_u_r

        return np.array([dVt, dVr, dut, dur, dVe, dVi, dc, dVe2, dVi2, dc2])

    def rk4_step(y, dt):
        k1 = rhs(0.0, y)
        k2 = rhs(0.0, y + 0.5*dt*k1)
        k3 = rhs(0.0, y + 0.5*dt*k2)
        k4 = rhs(0.0, y + dt*k3)
        y_next = y + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0 
        return y_next    

    for i in range(startind, len(t)):
        Q_t[i-1] = RT_t * np.exp(Lt*u_t) / (1 + np.exp((VT_t-V_t[i-1])/gT_t)) + RB_t * (1-np.exp(Lt*u_t)) / (1 + np.exp((VB_t-V_t[i-1])/gB_t))

        Q_r[i-1] = RT_r * np.exp(Lr*u_r) / (1 + np.exp((VT_r-V_r[i-1])/gT_r)) + RB_r * (1-np.exp(Lr*u_r)) / (1 + np.exp((VB_r-V_r[i-1])/gB_r))

        # RK4 单步
        y = np.array([V_t[i-1], V_r[i-1], u_t, u_r, V_e[i-1], V_i[i-1], c[i-1], V_e2[i-1], V_i2[i-1], c2[i-1]])
        y = rk4_step(y, dt)
        V_t[i], V_r[i], u_t, u_r = y[0], y[1], y[2], y[3]
        V_e[i], V_i[i], c[i]     = y[4], y[5], y[6]
        V_e2[i], V_i2[i], c2[i]  = y[7], y[8], y[9]
 
        u_chunk[i]  = u_t
        ur_chunk[i] = u_r
   

    return Q_t, Q_r, V_t, V_r, V_e, V_i, c, V_e2, V_i2, c2, t

