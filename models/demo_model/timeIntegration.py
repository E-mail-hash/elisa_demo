import numpy as np

def timeIntegration(params):
    dt = params["dt"]
    duration = params["duration"]

    a = params["a"]
    b = params["b"]
    c = params["c"]
    d = params["d"]

    startind = 1
    t = np.arange(1, round(duration, 6)/dt+1) * dt # 1 dimension

    # model output 
    ## r, x, y will be longer than t(?)
    r = np.zeros((1, startind+len(t))) # (row & column) 2 dimension
    x = np.zeros((1, startind+len(t))) # row & column
    y = np.zeros((1, startind+len(t))) # row & column

    # init
    x[:, :startind] = params["x_init"]
    y[:, :startind] = params["y_init"]

    (
        t,
        r,
        x,
        y,
    ) = timeIntegration_njit_elementwise(
        startind,
        t, 
        dt,
        a,
        b,
        c,
        d,
        r,
        x,
        y,
    )

    return (t, r, x, y)

def timeIntegration_njit_elementwise(
        startind,
        t, 
        dt,
        a,
        b,
        c,
        d,
        r,
        x,
        y,
):
    def _firing_rate(voltage):
        return 1 / (1.0+voltage)

    for i in range(startind, startind+len(t)):
        # define derivatives
        d_x = a * x[0, i-1] + b
        d_y = c * x[0, i-1] + d * y[0, i-1]

        # Euler integration
        x[0, i] = x[0, i-1] + dt * d_x
        y[0, i] = y[0, i-1] + dt * d_y
        r[0, i] = _firing_rate(x[0, i]) * 1e3

    return t, r, x, y    



