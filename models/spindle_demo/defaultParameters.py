import numpy as np

def loadDefaultParams(seed=None):
    class struct(object):
        pass 
    params = struct()

    params.dt = 0.0001
    params.duration = 10 
    params.model = "spindle1"
    
    # model parameters
    a = 1.3
    params.tau_t = 0.044   /a
    params.tau_r = 0.022   /a

    params.f_t_max = 250 
    params.f_r_max = 200 # 200

    params.f_t_th = 50#/4
    params.f_r_th = 110 # 110

    b = 0.012 # 0.012
    params.gamma_t = b #0.008 # 0.125 # change as HTC 0.125
    params.gamma_r = b #0.008 #0.125# change as HTC 0.125

    c = 32 #64
    params.At = 0.22 / c #32

    params.Ar = 0.11 /c #32

    ###
    params.tau_u_t = 0.22  /a
    params.tau_u_r = 0.11  /a
    params.RT_t = 100
    params.RT_r = 180
    params.RB_t = 250
    params.RB_r = 250
    params.VT_t = 26#26-20 #26
    params.VT_r = 23#23-20 #23
    params.VB_t = 15#15-20#15
    params.VB_r = 15#15-20 #15
    params.gT_t = 6 
    params.gT_r = 3 
    params.gB_t = 0.9 
    params.gB_r = 0.6

    params.Lt = 0.04
    params.Lr = 0.04

    params.Nt = 0.5 * 1000#
    params.Nr = 0.5 * 1000

    params.Prr = 0.02
    params.Ptr = 0.01
    params.Prt = 0.04
    
    params.Jrr =  -1.5 #-0.09 #1.5 # change as HTC
    params.Jtr =  3.42 * 4#* 4 #3.42 
    params.Jrt =  -1.6 * 6#*4 #1.6# change as HTC

    # initialization
    params.V_t_init = 0
    params.V_r_init = 0

    params.Q_t_init = 0
    params.Q_r_init= 0

    params.u_t_init = 0
    params.u_r_init = 0




    return params.__dict__
