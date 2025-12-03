import numpy as np

def loadDefaultParams(seed=None):
    class struct(object):
        pass 
    params = struct()

    params.dt = 1
    params.duration = 10 * 1000
    params.model = "spindle1"
    
    # model parameters
    params.tau_t = 0.044 * 1000
    params.tau_r = 0.022* 1000

    params.f_t_max = 250
    params.f_r_max = 200

    params.f_t_th = 50
    params.f_r_th = 110

    params.gamma_t = 0.25#0.005 # 0.25 # change as HTC
    params.gamma_r = 0.25#0.005 # 0.25# change as HTC

    params.At = 0.22 * 1000
    params.Ar = 0.11 * 1000

    ###
    params.tau_u_t = 0.22* 1000
    params.tau_u_r = 0.11* 1000
    params.RT_t = 100
    params.RT_r = 180
    params.RB_t = 250
    params.RB_r = 250
    params.VT_t = 26
    params.VT_r = 23
    params.VB_t = 15
    params.VB_r = 15
    params.gT_t = 6
    params.gT_r = 3
    params.gB_t = 0.9
    params.gB_r = 0.6

    params.Lt = 0.04
    params.Lr = 0.04

    params.Nt = 0.5 * 1000 * 2
    params.Nr = 0.5 * 1000

    params.Prr = 0.22
    params.Ptr = 0.01
    params.Prt = 0.04

    params.Jrr =  -0.09 #1.5 # change as HTC
    params.Jtr = 3.42
    params.Jrt = -0.5 #1.6# change as HTC

    # initialization
    params.V_t_init = -67
    params.V_r_init = -67

    params.Q_t_init = 0
    params.Q_r_init = 0

    params.u_t_init = 0
    params.u_r_init = 0




    return params.__dict__
