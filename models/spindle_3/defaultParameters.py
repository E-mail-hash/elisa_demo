import h5py
import numpy as np

def loadDefaultParams(seed=None):
    lookupTableFileName = 'models/aln_demo/aln-precalc/quantities_cascade.h5'
    class struct(object):
        pass 
    params = struct()

    params.dt = 0.0001 # s
    params.duration = 10
    params.model = "spindle3"

    params.seed = np.int64(0)
    
    # model parameters
    params.tau_t = 0.044  #/ 2
    params.tau_r = 0.022  #/ 2

    params.f_t_max = 250 
    params.f_r_max = 200 # 200

    params.f_t_th = 50
    params.f_r_th = 110 # 110

    params.gamma_t = 0.005 #0.25 # # change as HTC
    params.gamma_r = 0.005 #0.25##  change as HTC

    params.At = 0.22 #* 1000  #/ 8

    params.Ar = 0.11 #* 1000  #/ 8

    ###
    params.tau_u_t = 0.22 #/2
    params.tau_u_r = 0.11 #/2
    params.RT_t = 100
    params.RT_r = 180
    params.RB_t = 250
    params.RB_r = 250
    params.VT_t = 26 #26
    params.VT_r = 23 #23
    params.VB_t = 15 #15
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

    params.Jrr =  -0.09 #1.5 # change as HTC
    params.Jtr =  3.42 #3.42 
    params.Jrt =  -0.5 #1.6# change as HTC

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
    params.Je2e0 = 0.67 #0.30
    params.Jee20 = 0.30 #/ 2 #0.67

    params.Jet = 1.98 #0.525 /2 #1.98
    params.Je2t = 1.67 #0.85 /2 #1.67
    params.Jer = 3.24 #2.35 /2#3.24
    params.Je2r = 3.42 #2.35 /2#3.42

    params.Jie = -1.5
    params.Jte = 0.30
    params.Jii = -0.29
    params.Jei = 0.43
    params.Je2i = 1.8
    params.Jti = 0.22

    params.Ji2e2 = -1.4
    params.Jte2 = 0.45
    params.Ji2i2 = -0.5
    params.Je2i2 = 0.48
    params.Jei2 = 1.62 
    params.Jti2 = 0.18




    # initialization
    params.V_t_init = -1
    params.V_r_init = -1

    params.Q_t_init = 0
    params.Q_r_init = 0

    params.u_t_init = 0
    params.u_r_init = 0

    params.V_e_init = 1
    params.V_i_init = 1
    #params.Q_e_init = 0
    #params.Q_i_init = 0
    
    params.c_init = 0
    params.V_e2_init = 1
    params.V_i2_init = 1
    params.c2_init = 0


    
    # for ALN
    params.de = 1.0 # ms
    params.di = 1.0 # ms

    params.c_gl = 0.4 # mv/ms
    params.Ke_gl = 250.0 

    params.Ke = 800.0
    params.Ki = 200.0
    
    params.tau_ou = 5.0 # ms
    params.sigma_ou = 0.00
    params.mue_ext_mean = 2.8
    params.mui_ext_mean = 2.0

    params.sigmae_ext = 1.5
    params.sigmai_ext = 1.5

    params.tau_se = 2.0
    params.tau_si = 5.0

    params.cee = 0.3 # mV/ms
    params.cei = 0.3
    params.cie = 0.5
    params.cii = 0.5

    params.Jee_max = 2.43 # mV/ms
    params.Jei_max = -3.3
    params.Jie_max = 2.60
    params.Jii_max = -1.64 

    params.a = 0.0
    params.b = 15.0
    params.EA = -80.0
    params.tauA = 1000.0

    params.C = 200.0
    params.gl = 10.0

    hf = h5py.File(lookupTableFileName, 'r')
    params.Irange = hf.get("mu_vals")[()]
    params.sigmarange = hf.get("sigma_vals")[()]
    params.dI = params.Irange[1] - params.Irange[0]
    params.ds = params.sigmarange[1] - params.sigmarange[0]

    params.precalc_r = hf.get("r_ss")[()][()]
    params.precalc_V = hf.get("V_mean_ss")[()]
    params.precalc_tau_mu = hf.get("tau_mu_exp")[()]

    # neuron initialization
    if seed:
        np.random.seed(seed)
    params.mufe_init = 3*np.random.uniform(0,1)
    params.mufi_init = 3*np.random.uniform(0,1)
    params.seem_init = 0.5*np.random.uniform(0,1)
    params.seim_init = 0.5*np.random.uniform(0,1)
    params.siem_init = 0.5*np.random.uniform(0,1)
    params.siim_init = 0.5*np.random.uniform(0,1)

    params.seev_init = 0.001*np.random.uniform(0,1)
    params.seiv_init = 0.001*np.random.uniform(0,1)
    params.siev_init = 0.01*np.random.uniform(0,1)
    params.siiv_init = 0.01*np.random.uniform(0,1)

    params.Q_e_init = 0.05*np.random.uniform(0,1)
    params.Q_i_init = 0.05*np.random.uniform(0,1)

    params.IA_init = 200.0*np.random.uniform(0,1)





    return params.__dict__
