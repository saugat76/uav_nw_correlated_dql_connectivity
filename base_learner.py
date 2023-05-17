from abc import abstractmethod


class BaseLearner:

    def __init__(self):
        self.no_iter = int(1e6)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999995

        self.alpha = 1.0
        self.alpha_min = 0.001
        self.alpha_decay = 0.999995

    @abstractmethod
    def learn(self):
        pass