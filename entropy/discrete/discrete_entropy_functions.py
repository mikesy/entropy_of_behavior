from sklearn.metrics import confusion_matrix
import numpy as np
import math

def calc_entropy_using_confusion_matrix(x_true, x_pred, labels=[]):

    """
    Inputs:
        x_true [N x 1] array 
            N =  number of predictions 
                each element is an integer from 0 to infinity
                the number of unique integers is the number of classes
        x_pred [N x 1] array

    Output:
        entropy
    """
    if labels:
        cm = confusion_matrix(x_pred, x_true, labels=labels)
    else:
        cm = confusion_matrix(x_pred, x_true)

    return calc_entropy(cm)
    # num_classes = np.shape(cm)[0]
    # num_bins = num_classes**2
    # num_predictions = float(len(x_pred))
    # entropy = 0
    # for row in cm:
    #     for b in row:
    #         p_b = float(b)/num_predictions      # probability of bin b
    #         if p_b < 1e-10:
    #             p_b = 1e-10
    #         entropy_part = -p_b*np.log(p_b)/np.log(num_bins)
    #         if math.isnan(entropy_part):
    #             print('p_b = ',p_b)
    #             print(num_bins)
    #             print(x_true)
    #             print(x_pred)
    #         entropy += -p_b*np.log(p_b)/np.log(num_bins)
    # if math.isnan(entropy):
    #     print("got a nan")
    # return entropy

def calc_entropy(cm):
    num_classes = np.shape(cm)[0]
    num_bins = num_classes**2
    num_predictions = np.sum(cm.flatten())
    entropy = 0
    for row in cm:
        for b in row:
            p_b = float(b)/num_predictions      # probability of bin b
            if p_b < 1e-10:
                p_b = 1e-10
            # entropy_part = -p_b*np.log(p_b)/np.log(num_bins)
            # if math.isnan(entropy_part):
            #     print('p_b = ', p_b)
            #     print(num_bins)
            #     print(x_true)
            #     print(x_pred)
            entropy += -p_b*np.log(p_b)/np.log(num_bins)
    if math.isnan(entropy):
        print("got a nan")
    return entropy
