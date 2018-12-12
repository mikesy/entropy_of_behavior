def prepare_data(u,lag):
    u_t = []
    u_in = []
    u_out = u[lag:] #remove the first commands since no prediction can be made
    for samp_i in range(len(u)-lag):
        u_t = []
        for lag_i in range(lag):
            u_t.append([u[samp_i+lag_i]])
        #u_in.append([u[samp_i:samp_i+lag]])
        u_in.append(u_t)
    return u_in, u_out

def get_command_errors(ucmd, model, command_row, lag):

    u_err_all = []
    for segment_i in range(len(ucmd)):
        u_err = []
        u_seg = ucmd[segment_i][command_row]
        u_in, u_out = prepare_data(u_seg,lag)
        u_pred = model.predict(u_in)
        for u_i in range(len(u_out)):
            u_err.append(u_out[u_i]-u_pred[u_i])
        u_err_all.append(u_err)
    return u_err_all
