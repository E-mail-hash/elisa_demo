import numpy as np
import numba
import defaultParameters as dp

def timeIntegration(params):
    # simulation parameters
    dt = params["dt"]
    sqrt_dt = np.sqrt(dt)
    duration = params["duration"]
    RNGseed = params["seed"]

    t = np.arange(0,1,21) 



    # neuron parameters
    ## thalamus
    tau = params["tau"]
    Q_max = params["Q_max"]
    C1 = params["C1"]
    theta = params["theta"]
    sigma = params["sigma"]
    g_L = params["g_L"]
    E_L = params["E_L"]
    g_AMPA = params["g_AMPA"]
    g_GABA = params["g_GABA"]
    E_AMPA = params["E_AMPA"]
    E_GABA = params["E_GABA"]
    g_LK = params["g_LK"]
    E_K = params["E_K"]
    g_T_t = params["g_T_t"]
    g_T_r = params["g_T_r"]
    E_Ca = params["E_Ca"]
    g_h = params["g_h"]
    g_inc = params["g_inc"]
    E_h = params["E_h"]
    C_m = params["C_m"]
    alpha_Ca = params["alpha_Ca"]
    Ca_0 = params["Ca_0"]
    tau_Ca = params["tau_Ca"]
    k1 = params["k1"]
    k2 = params["k2"]
    k3 = params["k3"]
    k4 = params["k4"]
    n_P = params["n_P"]
    gamma_e = params["gamma_e"]
    gamma_r = params["gamma_r"]
    d_phi = params["d_phi"]
    N_rt = params["N_rt"]
    N_tr = params["N_tr"]
    N_rr = params["N_rr"]

    ext_current_t = params["ext_current_t"]
    ext_current_r = params["ext_current_r"]



    ## aln
    N = 1
    D = params['de']
    D_ndt = int(D/dt)

    # neuron initialization
    ## thalamus
    V_t = np.zeros((1, startind+len(t))) # (row & column) 2 dimension
    V_r = np.zeros((1, startind+len(t))) # (row & column) 2 dimension
    Q_t = np.zeros((1, startind+len(t))) # (row & column) 2 dimension
    Q_r = np.zeros((1, startind+len(t))) # (row & column) 2 dimension

    # neuron initialization 
    V_t[:, :startind] = params["V_t_init"]
    V_r[:, :startind] = params["V_r_init"]
    Ca = float(params["Ca_init"])
    h_T_t = float(params["h_T_t_init"])
    h_T_r = float(params["h_T_r_init"])
    m_h1 = float(params["m_h1_init"])
    m_h2 = float(params["m_h2_init"])
    s_et = float(params["s_et_init"])
    s_gt = float(params["s_gt_init"])
    s_er = float(params["s_er_init"])
    s_gr = float(params["s_gr_init"])
    ds_et = float(params["ds_et_init"])
    ds_gt = float(params["ds_gt_init"])
    ds_er = float(params["ds_er_init"])
    ds_gr = float(params["ds_gr_init"])

    np.random.seed(RNGseed)
    noise = np.random.standard_normal((len(t)))


@numba.njit()
def timeIntegration_njit_elementwise():
    pass