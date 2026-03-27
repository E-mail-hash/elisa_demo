from models.aln_thalamus import timeIntegration as ti_aln_thalamus
from models.aln_demo import timeIntegration as ti_aln_demo
from models.aln import timeIntegration as ti_aln
from models.thalamus import timeIntegration as ti_thalamus
from models.demo_model import timeIntegration as ti_demo
from models.spindle_demo import timeIntegration as ti_spindle1
from models.spindle_2 import timeIntegration as ti_spindle2
from models.spindle_3 import timeIntegration as ti_spindle3
def runModels(traj=0, manual_params=None):
    if traj != 0:
        params = traj.parameters.f_to_dict(short_names=True, fast_access=True)
        print("got traj")
    else:
        params = manual_params
    
    model = params["model"]
    

    if model == "spindle1":
        #t, Q_t, Q_r, V_t, V_r, a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk = runSpindle1(params)
        t, Q_t, Q_r, V_t, V_r, u_chunk, ur_chunk = runSpindle1(params)
    if model == "spindle2":
        #t, Q_t, Q_r, V_t, V_r, a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk = runSpindle1(params)
        t, Q_t, Q_r, V_t, V_r, Q_e, Q_i, V_e, V_i, c, V_e2, V_i2, c2 = runSpindle2(params)
    if model == "spindle3":
        #t, Q_t, Q_r, V_t, V_r, a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk = runSpindle1(params)
        t, Q_t, Q_r, V_t, V_r, Q_e, Q_i, V_e, V_i, c, V_e2, V_i2, c2 = runSpindle3(params)
    if model == "simp":
        Q_e, Q_i, t = runAlnDemo(params)
    if model == "demo":
        t, r, x, y = runDemo(params)
    if model == "thalamus":
        print("got thalamus model")
        t, V_t, V_r, Q_t, Q_r = runThalamus(params)
    if model == "aln":
        rates_exc, rates_inh, t, mufe, mufi = runAln(params)
    if model == "aln-thalamus":
        t, V_t, V_r, Q_t, Q_r, Q_e, Q_i = runAlnThalamus(params)   

    if traj != 0:
        if model == "spindle3":
            traj.f_add_result('spindle3_results_for_abcde.$', t=t, V_e=V_e, V_i=V_i, V_t=V_t, V_r=V_r)
        if model == "spindle1":
            traj.f_add_result('spindle1_results.$', t=t, Q_t=Q_t, Q_r=Q_r, V_t=V_t, V_r=V_r)
        if model == "simp":
            traj.f_add_result('aln_demo_results.$', t=t, Q_e=Q_e, Q_i=Q_i)
        if model == "thalamus":
            traj.f_add_result('thalamus_results.$', t=t, V_t=V_t, V_r=V_r, Q_t=Q_t, Q_r=Q_r)
        if model == "aln":
            traj.f_add_result('aln_results.$', t=t, rates_exc=rates_exc, rates_inh=rates_inh, mufe=mufe, mufi=mufi)
        if model == "aln-thalamus":
            traj.f_add_result('aln-thalamus_results.$', t=t, V_t=V_t, V_r=V_r, Q_t=Q_t, Q_r=Q_r, Q_e=Q_e, Q_i=Q_i)
    else:
        if model == "spindle1":
            #return t, Q_t, Q_r, V_t, V_r, a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk
            return t, Q_t, Q_r, V_t, V_r, u_chunk, ur_chunk
        if model == "spindle2":
            return t, Q_t, Q_r, V_t, V_r, Q_e, Q_i,V_e, V_i, c, V_e2, V_i2, c2
        if model == "spindle3":
            return t, Q_t, Q_r, V_t, V_r, Q_e, Q_i,V_e, V_i, c, V_e2, V_i2, c2
        if model == "simp":
            return t, Q_e, Q_i
        if model == "thalamus":
            return t, V_t, V_r, Q_t, Q_r 
        if model == "aln":
            return t, rates_exc, rates_inh, mufe, mufi
        if model == "aln-thalamus":
            return t, V_t, V_r, Q_t, Q_r, Q_e, Q_i
def runSpindle2(params):
    Q_t, Q_r, V_t, V_r, Q_e, Q_i, V_e, V_i, c, V_e2, V_i2, c2, t = ti_spindle2.timeIntegration(params)
    return t, Q_t, Q_r, V_t, V_r, Q_e, Q_i, V_e, V_i, c, V_e2, V_i2, c2
def runSpindle3(params):
    Q_t, Q_r, V_t, V_r, Q_e, Q_i, V_e, V_i, c, V_e2, V_i2, c2, t = ti_spindle3.timeIntegration(params)
    return t, Q_t, Q_r, V_t, V_r, Q_e, Q_i, V_e, V_i, c, V_e2, V_i2, c2


def runSpindle1(params):
    Q_t, Q_r, V_t, V_r, t, u_chunk, ur_chunk= ti_spindle1.timeintegration(params)
    #Q_t, Q_r, V_t, V_r, t, a_chunk, b_chunk, c_chunk, u_chunk , ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk= ti_spindle1.timeintegration(params)
    return t, Q_t, Q_r, V_t, V_r, u_chunk, ur_chunk
    #return t, Q_t, Q_r, V_t, V_r, a_chunk, b_chunk, c_chunk, u_chunk, ar_chunk, br_chunk, cr_chunk, dr_chunk, ur_chunk

def runAlnThalamus(params):
    t, V_t, V_r, Q_t, Q_r, Q_e, Q_i = ti_aln_thalamus.timeIntegration(params)
    return t, V_t, V_r, Q_t, Q_r, Q_e, Q_i

def runAlnDemo(params):
    Q_e, Q_i, t = ti_aln_demo.timeIntegration(params)
    return Q_e, Q_i, t

def runDemo(params):
    t, r, x, y = ti_demo.timeIntegration(params)
    return t, r, x, y

def runThalamus(params):
    t, V_t, V_r, Q_t, Q_r, *_  = ti_thalamus.timeIntegration(params)
    print("got timeIntegration")
    return t, V_t, V_r, Q_t, Q_r

def runAln(params):
    rates_exc, rates_inh, t, mufe, mufi, *_ = ti_aln.timeIntegration(params)
    return rates_exc, rates_inh, t, mufe, mufi



