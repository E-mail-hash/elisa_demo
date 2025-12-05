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
    u_chunk = np.zeros(len(t))

    ar_chunk = np.zeros(len(t))
    br_chunk = np.zeros(len(t))
    cr_chunk = np.zeros(len(t))
    dr_chunk = np.zeros(len(t))
    ur_chunk = np.zeros(len(t))

    return timeIntegration_njit_elementwise(  
    startind, t, dt,
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
    V_t, tau_t, f_t_max, f_t_th, gamma_t, At,
    V_r, tau_r, f_r_max, f_r_th, gamma_r, Ar,
    u_t, tau_u_t,
    u_r, tau_u_r,
    RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
    RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
    Q_t, Q_r,
    a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk,
    Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt):

    def rhs(t, Vt, Vr, ut, ur,
            tau_t, f_t_max, f_t_th, gamma_t, At,
            tau_r, f_r_max, f_r_th, gamma_r, Ar,
            tau_u_t, tau_u_r,
            RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
            RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
            Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt):
        """
        返回 4 个导数：dVt/dt, dVr/dt, dut/dt, dur/dt
        """
        #Qt = RT_t * np.exp(Lt*ut) / (1 + np.exp((Vt-VT_t)/gT_t)) + RB_t * (1-np.exp(Lt*ut)) / (1 + np.exp((Vt-VB_t)/gB_t))
        Qt = RT_t * np.exp(Lt*ut) / (1 + np.exp((VT_t-Vt)/gT_t)) + RB_t * (1-np.exp(Lt*ut)) / (1 + np.exp((VB_t-Vt)/gB_t))
        
        Qr = RT_r * np.exp(Lr*ur) / (1 + np.exp((VT_r-Vr)/gT_r)) + RB_r * (1-np.exp(Lr*ur)) / (1 + np.exp((VB_r-Vr)/gB_r))

        # 快速计算 u(V)
        def fu(u, m):
            if m == 0:   # t
                return f_t_max / (1 + np.exp((u+f_t_th)/gamma_t))
            else:        # r
                return f_r_max / (1 + np.exp((u+f_r_th)/gamma_r))

        dVt = -Vt/tau_t + fu(ut,0)/At + Nr*Prt*Jrt*Qt/1000.0
        dVr = -Vr/tau_r + fu(ur,1)/Ar + Nr*Prr*Jrr*Qr/1000.0 + Nt*Ptr*Jtr*Qt/1000.0

        # b(V) 函数
        def bV(V, m):
            Balance = 0
            if m == 0:  # t
                return -200.0 if V - Balance < -0.1 else 0.0
            else:       # r
                return -200.0 if V - Balance <= 0.0 else 0.0

        dut = (bV(Vt,0) - ut) / tau_u_t
        dur = (bV(Vr,1) - ur) / tau_u_r

        return np.array([dVt, dVr, dut, dur])
    '''
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
    '''

    def rk4_step(Vt, Vr, ut, ur, dt, *args):
        k1 = rhs(0.0, Vt, Vr, ut, ur, *args)
        k2 = rhs(0.0, Vt+0.5*dt*k1[0], Vr+0.5*dt*k1[1],
                ut+0.5*dt*k1[2], ur+0.5*dt*k1[3], *args)
        k3 = rhs(0.0, Vt+0.5*dt*k2[0], Vr+0.5*dt*k2[1],
                ut+0.5*dt*k2[2], ur+0.5*dt*k2[3], *args)
        k4 = rhs(0.0, Vt+dt*k3[0], Vr+dt*k3[1],
                ut+dt*k3[2], ur+dt*k3[3], *args)

        Vt += dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6.0
        Vr += dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6.0
        ut += dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])/6.0
        ur += dt*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])/6.0

        return Vt, Vr, ut, ur    

    # 预先把所有常量参数打包，避免每次都传
    rhs_args = (tau_t, f_t_max, f_t_th, gamma_t, At,
                tau_r, f_r_max, f_r_th, gamma_r, Ar,
                tau_u_t, tau_u_r,
                RT_t, VT_t, gT_t, Lt, RB_t, VB_t, gB_t,
                RT_r, VT_r, gT_r, Lr, RB_r, VB_r, gB_r,
                Nr, Nt, Prr, Ptr, Prt, Jrr, Jtr, Jrt)

    for i in range(startind, len(t)):
        # 记录 chunk（可选）
        Q_t[i-1] = RT_t * np.exp(Lt*u_t) / (1 + np.exp((VT_t-V_t[i-1])/gT_t)) + RB_t * (1-np.exp(Lt*u_t)) / (1 + np.exp((VB_t-V_t[i-1])/gB_t))
        #Q_t[i-1] = RT_t * np.exp(Lt*u_t) / (1 + np.exp((V_t[i-1]-VT_t)/gT_t)) + RB_t * (1-np.exp(Lt*u_t)) / (1 + np.exp((V_t[i-1]-VB_t)/gB_t))

        Q_r[i-1] = RT_r * np.exp(Lr*u_r) / (1 + np.exp((VT_r-V_r[i-1])/gT_r)) + RB_r * (1-np.exp(Lr*u_r)) / (1 + np.exp((VB_r-V_r[i-1])/gB_r))
        #Q_r[i-1] = RT_r * np.exp(Lr*u_r) / (1 + np.exp((V_r[i-1]-VT_r)/gT_r)) + RB_r * (1-np.exp(Lr*u_r)) / (1 + np.exp((V_r[i-1]-VB_r)/gB_r))

        # RK4 单步
        V_t[i], V_r[i], u_t, u_r = rk4_step(
            V_t[i-1], V_r[i-1], u_t, u_r, dt, *rhs_args)

        # 继续记录辅助变量
        u_chunk[i]  = u_t
        ur_chunk[i] = u_r
    '''
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
        ar = -V_r[i-1] / tau_r
        br = _function_u(u_r, "r") / Ar
        cr = Nr*Prr*Jrr*Q_r[i-1] / 1000
        dr = Nt*Ptr*Jtr*Q_t[i-1] / 1000
        V_r_rhs = ar + br + cr + dr
        ar_chunk[i] = ar
        br_chunk[i] = br
        cr_chunk[i] = cr
        dr_chunk[i] = dr
        #V_r_rhs = -V_r[i-1] / tau_r + _function_u(u_r, "r") / Ar + Nr*Prr*Jrr*Q_r[i-1] / 1000  + Nt*Ptr*Jtr*Q_t[i-1] / 1000

        u_t_rhs = (_function_b(V_t[i-1], "t") - u_t) / tau_u_t
        u_r_rhs = (_function_b(V_r[i-1], "r") - u_r) / tau_u_r
        
        # add r.h.s
        V_t[i] = V_t[i-1] + dt * V_t_rhs
        V_r[i] = V_r[i-1] + dt * V_r_rhs

        u_t = u_t + dt * u_t_rhs
        u_r = u_r + dt * u_r_rhs

        u_chunk[i] = u_t
        ur_chunk[i] = u_r

   '''
    

    #return Q_t, Q_r, V_t, V_r, t, a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk
    return Q_t, Q_r, V_t, V_r, t, u_chunk, ur_chunk

