import numpy as np
import numba
import defaultParameters as dp

def timeIntegration(params):
    # simulation parameters
    dt = params["dt"]
    sqrt_dt = np.sqrt(dt)
    duration = params["duration"]
    RNGseed = params["seed"]

    t = np.arange(1, round(duration, 6)/dt+1)

    # neuron parameters
    ## aln
    N = 1
    de = params['de']
    di = params['di']
    ndt_de = int(de/dt)
    ndt_di = int(di/dt)

    startind = int(np.max([ndt_de, ndt_di]))

    c_gl     = params['c_gl']       # global coupling strength between areas(unitless)
    Ke_gl    = params['Ke_gl']     # number of incoming E connections (to E population) from each area

    Jee_max = params['Jee_max']
    Jie_max = params['Jie_max']
    Jei_max = params['Jei_max']
    Jii_max = params['Jii_max']
    tau_exc = 0 
    tau_inh = 0 




    # neuron initialization
    '''
    these are all scalars in defaultParameters 
    '''
    mufe = float(params['mufe_init'])
    mufi = float(params['mufi_init'])
    seem = float(params['seem_init'])
    siem = float(params['siem_init'])
    seim = float(params['seim_init'])
    siim = float(params['siim_init'])
    
    ###? how we manage the startind here?
    Q_e = np.zeros((1, startind+len(t)))
    Q_e[:, :startind] = params['Q_e_init']
 
 

    # derivatives
    mue = Jee_max * seem + Jei_max * seim
    mui = Jie_max * siem + Jii_max * siim
    mufe_rhs = (mue - mufe)/tau_exc
    mufi_rhs = (mui - mufi)/tau_inh


@numba.njit()
def timeIntegration_njit_elementwise():
    pass