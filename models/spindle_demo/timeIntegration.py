"""
for spindles 
"""
import numpy as np
import numba
import defaultparameters as dp

def timeintegration(params):
    # simulation parameters
    # dt = params["dt"]
    # duration = params["duration"]
    # rngseed = params["seed"]

    for par in params.keys():
        exec(f'{par} = params["{par}"]')

    t = np.arange(1, round(duration, 6)/dt+1)

    # neuron parameters
    startind = 1 

    # tau_t = params["tau_t"]
    # tau_r = params["tau_r"]


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

    return timeIntegration_njit_elementwise(  
    startind, t, dt,
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, tau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt,
    )



@numba.njit()
def timeIntegration_njit_elementwise(
    startind, t, dt,
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, rau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt,
):
    def _function_u(u, m):
        if m == "t":
            f_max = f_t_max
            f_th = f_t_th
            gamma = gamma_t
        elif m == "r":
            f_max = f_r_max
            f_th = f_r_th
            gamma = gamma_r
        return -f_max / (1 + np.exp((u+f_th)/gamma))
    
    def _function_b(V, m):
        if m == "t":
            if V >= -0.1:
                b = 0
            else:
                b = -200
        elif m == "r":
            if V > 0:
                b = 0
            else:
                b = -200
        return b

    def _function_Q(V, u, m):
        if m == "t":
            RT = RT_t
            VT = VT_t
            gT = gT_t
            L = Lt 
            RB = RB_t
            VB = VB_t
            gB = gB_t
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
        

        # r.h.s
        V_t_rhs = -V_t[i-1] / tau_t + _function_u(u_t, "t") / At + Nr*Prt*Jrt*Q_r[i-1]
        V_r_rhs = -V_r[i-1] / rau_r + _function_u(u_r, "r") / Ar + Nr*Prr*Jrr*Q_r[i-1] + Nt*Ptr*Jtr*Q_t[i-1]

        u_t_rhs = (_function_b(V_t[i-1], "t") - u_t) / tau_u_t
        u_r_rhs = (_function_b(V_r[i-1], "r") - u_r) / tau_u_r
        
        # add r.h.s
        V_t[i] = V_t[i-1] + dt * V_t_rhs
        V_r[i] = V_r[i-1] + dt * V_r_rhs

        u_t = u_t + dt * u_t_rhs
        u_r = u_r + dt * u_r_rhs

        Q_t[i] = _function_Q(V_t[i], u_t, "t")
        Q_r[i] = _function_Q(V_r[i], u_r, "r")
    
    

    return Q_t, Q_r, V_t, V_r, t

