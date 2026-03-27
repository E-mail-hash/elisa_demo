
def loadDefaultParams():

    class struct(object):
        pass
    params = struct()
    params.model = 'demo'
    
    params.dt = 0.1
    params.duration = 2

    params.a = 1
    params.b = 1
    params.c = 1
    params.d = 1

    params.x_init = 2
    params.y_init = 4

    params_dict = params.__dict__

    return params_dict
