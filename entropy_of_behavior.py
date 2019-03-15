"""
This is the main access file for those needing access to the basics
Contains access to discrete and continuous classes for use in realtime.
"""
from entropy import discrete

class EntropyOfBehavior:
    def __init__(self, predictor, prediction_window, entropy_estimation_window, N):
        self.p = predictor
        self.w_p = prediction_window
        self.w_e = entropy_estimation_window
        self.N = N

        self.u_for_prediction = []
        self.u_true = []
        self.u_predicted = []
        self.entropy = 0

    def update_command_list(self, u):
        self.check_command(u)

        if len(self.u_for_prediction) < self.w_p:
            self.u_for_prediction.append(u)
        else:      
            self.u_for_prediction.pop(0)
            self.u_for_prediction.append(u)
        
    def check_command(self, u):
        if len(u) == self.N:
            return True
        else:
            raise ValueError("dimension of u does not match N")

    def predict_u(self):
        u_p = self.p(self.u_for_prediction)
        if self.u_predicted < self.w_e:
            self.u_predicted.append(u_p)
        else:
            self.u_predicted.pop(0)
            self.u_for_prediction.append(u_p)



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
        discrete.calc_entropy_using_confusion_matrix(self.u_true, self.u_for_prediction, labels = self.labels)
        print("n")



class ContinousEntropyOfBehavior(EntropyOfBehavior):
    def __init__(self, predictor, prediction_window, entropy_estimation_window, N):
        EntropyOfBehavior.__init__(self, predictor, prediction_window, entropy_estimation_window, N)
        
        if N >= 1:
            raise ValueError("continuous multiple dimensions not implemented yet")

    def update(self, u_t):
        self.predict_u()
        self.update_command_list(u_t)
        self.estimate_entropy()

    def estimate_entropy(self):
        