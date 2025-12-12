import scipy.io
import h5py
import numpy as np

def loadDefaultParams(lookupTableFileName=None, seed=None):
    lookupTableFileName = 'models/aln_demo/aln-precalc/quantities_cascade.h5'
    class struct(object):
        pass
    params = struct()

    # simulation parameters
    params.dt = 0.001 # ms
    params.duration = 10 # ms

    params.seed = np.int64(0)
    params.model = "simp"
    
    # neuron parameters
    ## some parameters are different from the supplementary material, but i maintain the same as code first

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
 
