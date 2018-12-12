import numpy as np

def get_command_errors_1d(ucmd, acmd,command_row):
    """
    inputs:
        ucmd- user command
              dimensions  of [(number of segments) x (number command signals) x (time steps)]
              number of segments- if the command signals have been been split by tasks or other markers
              
        acmd- autonomy command
              *same as ucmd description
        command_row- the dimension of the command to compare
    outputs:
        u_err_all- the difference between ucmd and acmd for all timesteps of each segment
                   dimensions same as ucmd and acmd

    """

    # ensure inputs are of equal size
    if np.shape(ucmd) != np.shape(acmd):
        raise ValueError("ucmd and acmd shapes are not equal")
    # process
    u_err_all = []
    for segment_i in range(len(ucmd)):
        u_err = []
        u_seg = ucmd[segment_i][command_row]
        a_seg = acmd[segment_i][command_row]
        u_err = [u_i - a_i for u_i, a_i in zip(u_seg,a_seg)]
        u_err_all.append(u_err)
    return u_err_all


def get_command_errors_2d(ucmd, acmd, command_rows):
    """
    inputs:
        ucmd- user command
              dimensions  of [(number of segments) x (number command signals) x (time steps)]
              number of segments- if the command signals have been been split by tasks or other markers
              
        acmd- autonomy command
              *same as ucmd description
        command_row- the dimension of the command to compare
    outputs:
        u_err_all- the difference between ucmd and acmd for all timesteps of each segment
                   dimensions same as ucmd and acmd

    """

    # ensure inputs are of equal size

    # process

    # u_err_all = []
    # for segment_i in range(len(ucmd)):
    #     u_err = []
    #     u_seg = ucmd[segment_i][command_row]
    #     a_seg = acmd[segment_i][command_row]
    #     u_err = [u_i - a_i for u_i, a_i in zip(u_seg, a_seg)]
    #     u_err_all.append(u_err)
    # return u_err_all
