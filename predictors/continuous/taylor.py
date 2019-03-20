
def predict_u(u, t=[]):
    """
    inputs:
        u- array or list 4x1 [u(t-3), u(t-2), u(t-1)]

    outputs:
        taylor_pred = u(t)
    """
    if t:
        taylor_pred = u[2] + (u[2]-u[1])/(t[2]-t[1]) + 0.5 * \
            ((u[2]-u[1])/(t[2]-t[1]) - (u[1] - u[0])/(t[1]-t[0]))/(t[2]-t[0])
    else:
        taylor_pred = u[2] + (u[2]-u[1]) + 0.5*((u[2]-u[1]) - (u[1] - u[0]))
    return taylor_pred

    