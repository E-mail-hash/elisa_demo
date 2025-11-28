import numpy as np
import h5py

def loadDefaultParams(lookupTableFileName=None, seed=None):
    lookupTableFileName = 'models/aln_thalamus/aln-precalc/quantities_cascade.h5'

    class struct(object):
        pass
    params = struct()
    params.model = 'aln-thalamus'
    
    params.dt = 0.01
    params.duration = 2 * 1000
    params.seed = np.int64(0) 
    


    # local parameters for both TCR & TRN
    params.tau = 20.0
    params.Q_max = 400.0e-3  # 1/ms
    params.theta = -58.5  # mV
    params.sigma = 6.0
    params.C1 = 1.8137993642
    params.C_m = 1.0  # muF/cm^2
    params.gamma_e = 70.0e-3  # 1/ms
    params.gamma_r = 100.0e-3  # 1/ms
    params.g_L = 1.0  # AU
    params.g_GABA = 1.0  # ms
    params.g_AMPA = 1.0  # ms
    params.g_LK = 0.018  # mS/cm^2
    params.E_AMPA = 0.0  # mV
    params.E_GABA = -70.0  # mV
    params.E_L = -70.0  # mV
    params.E_K = -100.0  # mV
    params.E_Ca = 120.0  # mV

    # specific TCR
    params.g_T_t = 3.0  # mS/cm^2
    params.g_h = 0.062  # mS/cm^2
    params.E_h = -40.0  # mV
    params.alpha_Ca = -51.8e-6  # nmol
    params.tau_Ca = 10.0  # ms
    params.Ca_0 = 2.4e-4
    params.k1 = 2.5e7
    params.k2 = 4.0e-4
    params.k3 = 1.0e-1
    params.k4 = 1.0e-3
    params.n_P = 4.0
    params.g_inc = 2.0
    # connectivity
    params.N_tr = 5.0
    # noise
    params.d_phi = 0.0 

    # specific TRN
    params.g_T_r = 2.3  # mS/cm^2
    # connectivity
    params.N_rt = 3.0
    params.N_rr = 25.0

    # external input
    params.ext_current_t = 0.0
    params.ext_current_r = 0.0

    # neuron initialization
    if seed:
        np.random.seed(seed)
    params.V_t_init = np.random.uniform(-75, -50)
    params.V_r_init = np.random.uniform(-75, -50)
    params.Q_t_init = np.random.uniform(0.0, 200.0)
    params.Q_r_init = np.random.uniform(0.0, 200.0)
    params.Ca_init = 2.4e-4
    params.h_T_t_init = 0.0
    params.h_T_r_init = 0.0
    params.m_h1_init = 0.0
    params.m_h2_init = 0.0
    params.s_et_init = 0.0
    params.s_gt_init = 0.0
    params.s_er_init = 0.0
    params.s_gr_init = 0.0
    params.ds_et_init = 0.0
    params.ds_gt_init = 0.0
    params.ds_er_init = 0.0
    params.ds_gr_init = 0.0

    # aln 
    params.de = 1.0
    params.di = 1.0

    params.c_gl = 0.3
    params.ke_gl = 100

    params.Ke = 800.0
    params.Ki = 200.0
    
    params.tau_ou = 5.0
    params.sigma_ou = 0.05
    params.mue_ext_mean = 3.0
    params.mui_ext_mean = 1.0

    params.sigmae_ext = 1.5
    params.sigmai_ext = 1.5

    params.tau_se = 2.0
    params.tau_si = 5.0

    params.cee = 0.3
    params.cei = 0.3
    params.cie = 0.5
    params.cii = 0.5

    params.Jee_max = 4.0
    params.Jei_max = -8.0
    params.Jie_max = 8.0
    params.Jii_max = -4.0

    params.a = 12.0
    params.b = 60.0
    params.EA = -80.0
    params.tauA = 200.0

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
    
    params_dict = params.__dict__

    return params_dict

