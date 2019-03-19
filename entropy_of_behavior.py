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
        self.u_true = []
        self.u_predicted = []
        self.entropy = None

    def update_command_list(self, u):
        """
        inputs:
            u- 1d np array with N elements (use flatten() if more than 3d array)
        """
        self.check_command(u)
        u = np.array([u])   # necessary to append
        num_samples_collected = np.shape(self.u_true)[0]
        if num_samples_collected >=1:
            if num_samples_collected < self.w_p or num_samples_collected < self.w_e:
                self.u_true = np.append(self.u_true, u, axis=0)
            else:      
                self.u_true = self.u_true[1:]
                self.u_true = np.append(self.u_true, u, axis=0)
        else:
            self.u_true = u
        
    def check_command(self, u):
        u_shape  = np.shape(u)
        if type(u) is not np.ndarray:
            raise ValueError("wrong command type, use np array")
        elif len(u_shape) >=2:
            raise ValueError("wrong shape of np array, make sure 1d with N elements")
        elif u_shape[0] == self.N:
            return True
        else:
            raise ValueError("dimension of u does not match N")

    def predict_u(self):
        num_samples_collected = np.shape(self.u_true)[0]
        if num_samples_collected >= self.w_p:
            u_p = self.p(self.u_true)
            self.check_command(u_p)
            num_predictions = np.shape(self.u_predicted)[0]
            u_p = np.array([u_p])  # for append
            if num_predictions >= 1:
                
                if num_predictions < self.w_e:
                    self.u_predicted = np.append(self.u_predicted, u_p)
                else:
                    self.u_predicted = self.u_predicted[1:]
                    self.u_predicted = np.append(self.u_predicted, u_p)
            else:
                self.u_predicted = u_p


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
        discrete.calc_entropy_using_confusion_matrix(self.u_true, self.u_predicted, labels = self.labels)

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
            u_errors =  self.u_true - self.u_predicted

            
            # get errors..

        # else:

        # return 0
