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




    # neuron initialization
    '''
    these are all scalars in defaultParameters 
    '''
    V_t = np.dot(params["V_t_init"], np.ones(len(t)))
    V_r = np.dot(params["V_r_init"], np.ones(len(t)))
    Q_t = np.dot(params["Q_t_init"], np.ones(len(t)))
    Q_r = np.dot(params["Q_r_init"], np.ones(len(t)))
    u_t = float(params["u_t_init"])
    u_r = float(params["u_r_init"])

    a_chunk = np.zeros(len(t))
    b_chunk = np.zeros(len(t))
    c_chunk = np.zeros(len(t))

    return timeIntegration_njit_elementwise(  
    startind, t, dt,
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, tau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    a_chunk, b_chunk, c_chunk,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt)



# @numba.njit()
def timeIntegration_njit_elementwise(
    startind, t, dt,
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, rau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    a_chunk, b_chunk, c_chunk,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt):
    def _function_u(u, m):
        if m == "t":
            f_max = f_t_max # 250
            f_th = f_t_th # 50
            gamma = gamma_t # 0.005
        elif m == "r":
            f_max = f_r_max
            f_th = f_r_th
            gamma = gamma_r
        return -f_max / (1 + np.exp((u+f_th)/gamma))
    
    def _function_b(V, m):
        Balance = 0
        if m == "t":
            if V-Balance >= -0.1:# change as HTC
                b = 0
            else:
                b = -200
        elif m == "r":
            if V-Balance > 0:# change as HTC
                b = 0
            else:
                b = -200
        return b

    def _function_Q(V, u, m):
        if m == "t":
            RT = RT_t # 100
            VT = VT_t # 26
            gT = gT_t # 6
            L = Lt  # 0.04
            RB = RB_t # 250
            VB = VB_t # 15
            gB = gB_t # 0.9
        elif m == "r":
            RT = RT_r
            VT = VT_r
            gT = gT_r
            L = Lr 
            RB = RB_r
            VB = VB_r
            gB = gB_r
        return RT * np.exp(L*u) / (1 + np.exp((V-VT)/gT)) + RB * (1-np.exp(L*u)) / (1 + np.exp((V-VB)/gB))
 


    # derivatives
    for i in range(startind, len(t)):

        Q_t[i-1] = _function_Q(V_t[i-1], u_t, "t")
        Q_r[i-1] = _function_Q(V_r[i-1], u_r, "r")
        

        # r.h.s
        a = -V_t[i-1] / tau_t
        b = _function_u(u_t, "t") / At
        c = Nr*Prt*Jrt*Q_r[i-1] / 1000
        V_t_rhs = a + b + c
        a_chunk[i] = a
        b_chunk[i] = b
        c_chunk[i] = c
        # V_t_rhs = -V_t[i-1] / tau_t + _function_u(u_t, "t") / At + Nr*Prt*Jrt*Q_r[i-1] 
        V_r_rhs = -V_r[i-1] / rau_r + _function_u(u_r, "r") / Ar + Nr*Prr*Jrr*Q_r[i-1] / 1000  + Nt*Ptr*Jtr*Q_t[i-1] / 1000

        u_t_rhs = (_function_b(V_t[i-1], "t") - u_t) / tau_u_t
        u_r_rhs = (_function_b(V_r[i-1], "r") - u_r) / tau_u_r
        
        # add r.h.s
        V_t[i] = V_t[i-1] + dt * V_t_rhs
        V_r[i] = V_r[i-1] + dt * V_r_rhs

        u_t = u_t + dt * u_t_rhs
        u_r = u_r + dt * u_r_rhs

   
    

    return Q_t, Q_r, V_t, V_r, t, a_chunk, b_chunk, c_chunk

