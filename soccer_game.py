import numpy as np


class SoccerGame:

    def __init__(self):
        self.pos = [np.array([0, 2]), np.array([0, 1])]
        self.ball = 1
        self.actions = [np.array([-1, 0]), np.array([1, 0]), np.array([0, -1]), np.array([0, 1]), np.array([0, 0])]
        self.goal = [0, 3]

    def state_encode(self):
        return [self.pos[0][0] * 4 + self.pos[0][1], self.pos[1][0] * 4 + self.pos[1][1], self.ball]

    def reset(self):
        self.pos = [np.array([0, 2]), np.array([0, 1])]
        self.ball = 1
        return

    def move(self, agent, action):
        tentative_pos = self.pos.copy()
        other_agent = 1 - agent
        rewards = np.array([0, 0])
        done = False
        tentative_pos[agent] = tentative_pos[agent] + self.actions[action]
        if (tentative_pos[agent] == self.pos[other_agent]).all():
            # print("collision occurs")
            if self.ball == agent:
                self.ball = other_agent
        elif tentative_pos[agent][0] in range(0, 2) and tentative_pos[agent][1] in range(0, 4):
            self.pos[agent] = tentative_pos[agent]
            # print("position for {} is {}".format(agent, self.pos[agent]))
            if self.ball != agent:
                return self.state_encode(), rewards, done
            horizon_pos = self.pos[agent][1]
            if horizon_pos == self.goal[0]:
                rewards = np.array([100, -100])
                done = True
                return self.state_encode(), rewards, done
            elif horizon_pos == self.goal[1]:
                rewards = np.array([-100, 100])
                done = True
                return self.state_encode(), rewards, done
            else:
                return self.state_encode(), rewards, done
        return self.state_encode(), rewards, done

    def step(self, action):
        first_move = np.random.choice([0, 1], 1)[0]
        second_move = 1 - first_move
        # print("{} first move".format(first_move))
        first_action, second_action = action[first_move], action[second_move]

        encoded_state, rewards, done = self.move(first_move, first_action)
        if done:
            return encoded_state, rewards, done
        else:
            encoded_state, rewards, done = self.move(second_move, second_action)

        return encoded_state, rewards, done


if __name__ == "__main__":
    env = SoccerGame()
    action = [3,2]
    _, rewards, done = env.step(action)
    print(rewards, done)