from utils import *
from base_learner import *
from cvxpy import Variable, Problem, Minimize


class CEQLearning(BaseLearner):

    def __init__(self):
        super(CEQLearning, self).__init__()
        self.no_iter = 1e6
        # self.epsilon_decay = 0
        # self.epsilon = 0
        self.alpha = 1.0
        self.alpha_decay = (self.alpha_min / self.alpha) ** (1. / self.no_iter)

    def epsilon_greedy(self, pi, state, epsilon):
        state_a, state_b, state_ball = state
        prob = pi[state_a, state_b, state_ball]
        prob = np.abs(prob) / np.sum(np.abs(prob))
        temp = np.random.random()
        if temp < epsilon:
            actions = [np.random.choice([0, 1, 2, 3, 4], 1)[0], np.random.choice([0, 1, 2, 3, 4], 1)[0]]
        else:
            action_number = np.random.choice(range(prob.shape[0]), p=prob)
            actions = [action_number // 5, action_number % 5]
        return actions

    def get_additional_matrix(self, Q, agent):
        additional_matrix = []
        size = Q.shape[0]
        for i in range(size):
            for j in range(size):
                if i != j:
                    temp = [0 for _ in range(size * size)]
                    for k in range(size):
                        if agent == 0:
                            temp[i * size + k] = Q[i, k] - Q[j, k]
                        else:
                            temp[i + k * size] = Q[i, k] - Q[j, k]
                    additional_matrix.append(temp)
        return np.array(additional_matrix)

    def ce(self, Q_a, Q_b):
        size_a = Q_a.shape[0]
        size_b = Q_b.shape[0]
        joint_size = size_a * size_b
        w = Variable(joint_size)

        object_vec = -(Q_a + Q_b.transpose()).reshape((1, joint_size))
        object_func = object_vec * w
        sum_vec = np.ones((joint_size, 1))
        sum_func = w.T * sum_vec

        eye = np.identity(joint_size)
        additional_a = self.get_additional_matrix(Q_a, 0)
        additional_b = self.get_additional_matrix(Q_b, 1)
        total = -np.concatenate([additional_a.transpose(), additional_b.transpose(), eye], axis=1)
        total_func = w.T * total

        prob = Problem(Minimize(object_func),
                       [sum_func == 1, total_func <= 0])
        try:
            prob.solve()
            weights = w.value
            if np.isnan(weights).sum() > 0 or np.abs(weights.sum() - 1) > 0.00001:
                weights = None
                print("fail to find solution")
        except:
            weights = None
            print("cant find solution")
        return weights

    def update_weight(self, Q_a, Q_b, state):
        state_a, state_b, state_ball = state
        a = Q_a[state_a, state_b, state_ball]
        b = Q_b[state_a, state_b, state_ball]
        return self.ce(a, b)

    def compute_expected_value(self, Q, pi, state, agent):
        state_a, state_b, state_ball = state
        if agent == 0:
            value = (Q[state_a, state_b, state_ball].reshape((1, 25)) * pi[state_a, state_b, state_ball]).sum()
        else:
            value = (Q[state_a, state_b, state_ball].transpose().reshape((1, 25)) * pi[state_a, state_b, state_ball]).sum()
        return value

    def learn(self):
        errors = []
        Q_a = np.zeros((8, 8, 2, 5, 5))
        Q_b = np.zeros((8, 8, 2, 5, 5))
        pi = np.ones((8, 8, 2, 25)) * 1/25
        V_a = np.ones((8, 8, 2))
        V_b = np.ones((8, 8, 2))
        epsilon = epsilon_decay = 10**(np.log10(self.epsilon_min)/self.no_iter)
        alpha = alpha_decay = 10**(np.log10(self.alpha_min)/self.no_iter)
        gamma = self.gamma
        env = SoccerGame()
        i = 0
        while i < self.no_iter:
            env.reset()
            state = env.state_encode()
            prob = self.update_weight(Q_a, Q_b, state)
            if prob is not None:
                pi[state[0]][state[1]][state[2]] = prob
                V_a[state[0]][state[1]][state[2]] = self.compute_expected_value(Q_a, pi, state, 0)
                V_b[state[0]][state[1]][state[2]] = self.compute_expected_value(Q_b, pi, state, 1)
            while True:
                if i % 1000 == 1:
                    print(str(errors[-1]))
                before_value = Q_a[2][1][1][2][4]
                actions = self.epsilon_greedy(pi, state, epsilon)
                state_new, rewards, done = env.step(actions)
                prob = self.update_weight(Q_a, Q_b, state_new)
                if prob is not None:
                    pi[state_new[0]][state_new[1]][state_new[2]] = prob
                    V_a[state_new[0]][state_new[1]][state_new[2]] = self.compute_expected_value(Q_a, pi, state_new, 0)
                    V_b[state_new[0]][state_new[1]][state_new[2]] = self.compute_expected_value(Q_b, pi, state_new, 1)
                i += 1
                if done:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2], actions[0], actions[1]] + \
                        alpha * (rewards[0] + gamma * V_a[state_new[0], state_new[1], state_new[2]] -
                                 Q_a[state[0], state[1], state[2], actions[0], actions[1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2], actions[1], actions[0]] + \
                        alpha * (rewards[1] + gamma * V_b[state_new[0], state_new[1], state_new[2]] -
                                 Q_b[state[0], state[1], state[2], actions[1], actions[0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    break
                else:
                    Q_a[state[0], state[1], state[2], actions[0], actions[1]] = Q_a[state[0], state[1], state[2], actions[0], actions[1]] + \
                            alpha * (rewards[0] + gamma * V_a[state_new[0], state_new[1], state_new[2]] -
                            Q_a[state[0], state[1], state[2], actions[0], actions[1]])
                    Q_b[state[0], state[1], state[2], actions[1], actions[0]] = Q_b[state[0], state[1], state[2], actions[1], actions[0]] + \
                            alpha * (rewards[1] + gamma * V_b[state_new[0], state_new[1], state_new[2]] -
                            Q_b[state[0], state[1], state[2], actions[1], actions[0]])
                    after_value = Q_a[2][1][1][2][4]
                    errors.append(abs(before_value - after_value))
                    state = state_new
                # epsilon *= self.epsilon_decay
                # epsilon = max(self.epsilon_min, epsilon)
                # alpha *= self.alpha_decay
                # alpha = max(self.alpha_min, alpha)
                alpha = alpha_decay ** i
                epsilon = epsilon_decay ** i
        plot_error(errors, "ce_learning_final")
        return


if __name__ == "__main__":
    learner = CEQLearning()
    learner.learn()