import entropy

def get_entropy(ucmd, dist, alpha):
    #pass alpha as zero if clean
    THETA_ROW = 2
    u_errors = get_command_errors_1d(ucmd, THETA_ROW)

    if dist == 'clean':
        alpha = entropy.calc_alpha(u_errors[0])   #only calculate for non-distracted

    # stats
    entropies = entropy.calc_entropy(u_errors,alpha)

    return entropies, alpha

def get_command_errors_1d(ucmd,  command_row):
    all_seg_errors = []
    for segment_i in range(0,len(ucmd)):        # cycle through segments split by L2
        u_seg = ucmd[segment_i][command_row]    # segments should be split by the L2 presses
        u_errors = []
        for u_i in range(0,len(u_seg)):
            u_snip = u_seg[u_i:4+u_i]
            if (4+u_i) <= len(u_seg):                                                                          # snippet for prediction
                taylor_pred = u_snip[2] + (u_snip[2]-u_snip[1]) + 0.5*((u_snip[2]-u_snip[1])- (u_snip[1] - u_snip[0])) # 2nd order taylor prediction
                pred_error = u_snip[3] - taylor_pred
                u_errors += [pred_error]
        all_seg_errors.append(u_errors)
    return all_seg_errors
