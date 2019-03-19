"""
This is the main access file for those needing access to the basics
Contains access to discrete and continuous classes for use in realtime.
"""
from entropy import discrete
import numpy as np

class EntropyOfBehavior:
    def __init__(self, predictor, prediction_window, entropy_estimation_window, N):
        self.p = predictor
        self.w_p = prediction_window
        self.w_e = entropy_estimation_window
        self.N = N

        # self.u_for_prediction = []
        self.u_true = np.array([[]])
        self.u_predicted = np.array([])
        self.entropy = 0

    def update_command_list(self, u):

        self.check_command(u)


        if u_true:
            if len(self.u_true) < self.w_p or len(self.u_true) < self.w_e:
                self.u_true = append(u)
            else:      
                self.u_true.pop(0)
                self.u_true.append(u)
        else:

        
    def check_command(self, u):
        if np.shape(u)[0] != 1:
            raise ValueError("dimension of u does not match N")
        else:
            if np.shape(u)[1] == self.N: 
                return True
        elif len(u) == self.N:
            return True
        else:
            raise ValueError("dimension of u does not match N")

    def predict_u(self):

        if len(self.u_true) >= self.w_p:
            u_p = self.p(self.u_true)
            if len(self.u_predicted) < self.w_e:
                self.u_predicted.append(u_p)
            else:
                self.u_predicted.pop(0)
                self.u_predicted.append(u_p)
        #     return True
        # else:
        #     return False

class DiscreteEntropyOfBehavior(EntropyOfBehavior):
    def __init__(self, predictor, prediction_window, entropy_estimation_window, N):
        EntropyOfBehavior.__init__(
            self, predictor, prediction_window, entropy_estimation_window, N)

        self.labels = [i for i in range(N)]

    def update(self, u_t):
        self.predict_u()
        self.update_command_list(u_t)
        self.estimate_entropy()

    def estimate_entropy(self):
        # don't necessarily need the full w_e to do this.. 
        discrete.calc_entropy_using_confusion_matrix(self.u_true, self.u_for_prediction, labels = self.labels)
        print("n")

class ContinuousEntropyOfBehavior(EntropyOfBehavior):
    def __init__(self, predictor, prediction_window, entropy_estimation_window, N, bin_size):
        EntropyOfBehavior.__init__(self, predictor, prediction_window, entropy_estimation_window, N)
        self.bin_size = bin_size
        
        if N > 1:
            raise ValueError("continuous multiple dimensions not implemented yet")

    def update(self, u_t):
        self.predict_u()
        self.update_command_list(u_t)
        self.estimate_entropy()

    def estimate_entropy(self):
        if self.u_true >= self.w_e:
            return 0
            # get errors..

        # else:

        # return 0
