from sklearn.metrics import confusion_matrix
import numpy as np

def calc_entropy_using_confusion_matrix(x_true, x_pred):

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
    cm = confusion_matrix(x_pred, x_true)

    num_classes = np.shape(cm)[0]
    num_bins = num_classes**2
    num_predictions = float(len(x_pred))
    entropy = 0
    for row in cm:
        for b in row:
            p_b = float(b)/num_predictions      # probability of bin b
            if p_b < 1e-10:
                p_b = 1e-10
            entropy += -p_b*np.log(p_b)/np.log(num_bins)

    return entropy

