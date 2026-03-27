from pypet import Environment, Parameter
import h5py

def add_parameters_single(traj, params):
    """Add parameters to traj
    just for demo
    """
    traj.f_add_parameter_group('simulation')
    traj.simulation.model = Parameter('model', params['model'])
    traj.simulation.dt = Parameter('dt', params['dt'])
    traj.simulation.duration = Parameter('duration', params['duration'])
    
    traj.f_add_parameter_group('demo')
    traj.demo.a = Parameter('a', params['a'])
    traj.demo.b = Parameter('b', params['b'])
    traj.demo.c = Parameter('c', params['c'])
    traj.demo.d = Parameter('d', params['d'])
    
    traj.f_add_parameter_group('init')
    traj.init.x_init = Parameter('x_init', params['x_init'])
    traj.init.y_init = Parameter('y_init', params['y_init'])

def add_parameters_all(traj, params):
    """Add parameters to traj
    used for all models not only demo
    """

    def addParametersRecursively(traj, params, current_level):
        if isinstance(current_level, str):
            current_level = [current_level]
        for key, value in params.items():
            if isinstance(value, dict):
                addParametersRecursively(traj, value, current_level+[key])
            else:
                param_address = ".".join(current_level+[key])
                value = "None" if value is None else value
                traj.f_add_parameter(param_address, value)
    
    addParametersRecursively(traj, params, [])
