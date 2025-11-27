import h5py
import numpy as np

def loadDefaultParams(lookupTableFileName=None, seed=None):
    lookupTableFileName = 'models/aln/aln_precalc/quantities_cascade.h5'
    class struct(object):
        pass
    params = struct()

    # simulation parameters
    params.dt = 0.1
    params.duration = 2000

    params.seed = np.int64(0)
    
    # neuron parameters
    ## some parameters are different from the supplementary material, but i maintain the same as code first

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

    return params.__dict__
 
