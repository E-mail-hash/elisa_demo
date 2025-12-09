import numpy as np

def loadDefaultParams(seed=None):
    class struct(object):
        pass 
    params = struct()
    '''
    https://orcid.org/0000-0001-8671-6728
    '''

    params.dt = 0.001
    params.duration = 10
    params.model = "spindle1"
    
    # model parameters
    params.tau_t = 0.044  #/ 2
    params.tau_r = 0.022 #/ 2

    params.f_t_max = 250 
    params.f_r_max = 200 # 200

    params.f_t_th = 50
    params.f_r_th = 110 # 110

    params.gamma_t = 0.125 # 0.25 # change as HTC
    params.gamma_r = 0.125 # 0.25# change as HTC

    params.At = 0.22 * 1000  #/ 8

    params.Ar = 0.11 * 1000  #/ 8

    ###
    params.tau_u_t = 0.22 #/2
    params.tau_u_r = 0.11 #/2
    params.RT_t = 100
    params.RT_r = 180
    params.RB_t = 250
    params.RB_r = 250
    params.VT_t = 26 #26
    params.VT_r = 23 #23
    params.VB_t = 15#15
    params.VB_r = 15 #15
    params.gT_t = 6 
    params.gT_r = 3 
    params.gB_t = 0.9 
    params.gB_r = 0.6

    params.Lt = 0.04
    params.Lr = 0.04

    params.Nt = 0.5 * 1000 
    params.Nr = 0.5 * 1000

    params.Prr = 0.02
    params.Ptr = 0.01
    params.Prt = 0.04

    params.Jrr =  -0.09 #-1.5 #-0.09 #1.5 # change as HTC
    params.Jtr =  3.42 #3.42 
    params.Jrt =  -0.5# -1.6 #-0.5 #1.6# change as HTC

    ### cortex
    params.tau_e = 0.02
    params.tau_i = 0.01
    params.tau_c = 0.5

    params.Ne = 0.8*10000
    params.Ni = 0.2*10000
    params.delta_c = 0.005
    params.delta_c2 = 0.0054

    params.ge = 5
    params.gi = 2
    params.Rm = 70
    params.V_star = 30

    params.c_star = 10
    params.gc = 3

    params.Pee  = 0.2
    params.Pe2e = params.Pee2 = 0.01
    params.Pie  =  0.2
    params.Pte  =  0.02
    params.Pii = 0.2
    params.Pei = 0.2
    params.Pe2i = params.Pei2 = 0.01
    params.Pti = 0.02

    params.Pet = 0.01
    params.Per = 0.01

    params.Jee0 = 0.38
    params.Je2e0 = 0.30
    params.Jee20 = 0.67

    params.Jet = 1.98
    params.Je2t = 1.67
    params.Jer = 3.24
    params.Je2r = 3.42

    params.Jie = -1.5
    params.Jte = 0.30
    params.Jii = -0.29
    params.Jei = 0.43
    params.Je2i = 1.8
    params.Jti = 0.22

    params.Jte2 = 0.45
    params.Jei2 = 0.48
    params.Jti2 = 0.18




    # initialization
    params.V_t_init = -1
    params.V_r_init = 1

    params.Q_t_init = 0
    params.Q_r_init = 0

    params.u_t_init = 0
    params.u_r_init = 0

    params.V_e_init = 0
    params.V_i_init = 0
    params.c_init = 0
    params.V_e2_init = 0
    params.V_i2_init = 0
    params.c2_init = 0


    




    return params.__dict__
