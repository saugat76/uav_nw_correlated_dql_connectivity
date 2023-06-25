import cvxpy as cp
import numpy as np
import torch

# Define the variables
# Each variable corresponds to a joint action
p_A1_B1 = cp.Variable()
p_A1_B2 = cp.Variable()
p_A2_B1 = cp.Variable()
p_A2_B2 = cp.Variable()

# Q-values for each action for each player
Q1_A1 = 2
Q1_A2 = 3
Q2_B1 = 5
Q2_B2 = 1

# Define the constraints
constraints = [
    # Probabilities are non-negative
    p_A1_B1 >= 0,
    p_A1_B2 >= 0,
    p_A2_B1 >= 0,
    p_A2_B2 >= 0,

    # Probabilities sum to 1
    p_A1_B1 + p_A1_B2 + p_A2_B1 + p_A2_B2 == 1,

    # Player 1 should not prefer A2 over A1
    Q1_A1 * p_A1_B1 + Q1_A1 * p_A1_B2 >= Q1_A2 * p_A2_B1 + Q1_A2 * p_A2_B2,

    # Player 1 should not prefer A1 over A2
    Q1_A2 * p_A2_B1 + Q1_A2 * p_A2_B2 >= Q1_A1 * p_A1_B1 + Q1_A1 * p_A1_B2,

    # Player 2 should not prefer B2 over B1
    Q2_B1 * p_A1_B1 + Q2_B1 * p_A2_B1 >= Q2_B2 * p_A1_B2 + Q2_B2 * p_A2_B2,

    # Player 2 should not prefer B1 over B2
    Q2_B2 * p_A1_B2 + Q2_B2 * p_A2_B2 >= Q2_B1 * p_A1_B1 + Q2_B1 * p_A2_B1
]

# Define the objective function
objective = cp.Maximize(Q1_A1*p_A1_B1 + Q1_A1*p_A1_B2 + Q1_A2*p_A2_B1 + Q1_A2*p_A2_B2 +
                        Q2_B1*p_A1_B1 + Q2_B1*p_A2_B1 + Q2_B2*p_A1_B2 + Q2_B2*p_A2_B2)

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Print the optimal solution
print("p(A1, B1) = ", p_A1_B1.value)
print("p(A1, B2) = ", p_A1_B2.value)
print("p(A2, B1) = ", p_A2_B1.value)
print("p(A2, B2) = ", p_A2_B2.value)

NUM_UAV = 3



# Joint action size = number of agents ^ action size // for a state 
# Optimizing the joint action so setting as a variable for CE optimization 
action_size = 3
action_values_ind = torch.arange(3)
action_profile = torch.stack(torch.meshgrid(*([action_values_ind] * NUM_UAV)), dim=-1).reshape(-1, NUM_UAV)
prob_weight = cp.Variable((27), pos = True)

# Collect Q values for the corresponding states of each individual agents
q_complete = np.array([[2, 3, 5], [5, 1, 7], [7, 8, 9], [7, 1, 0]])

action_profile = action_profile.cpu().squeeze().numpy()

# print(np.where(action_profile[:, 1] == 1))

def indexing(agent_idx):
        indices = np.zeros((action_size, action_size ** (NUM_UAV - 1)), dtype=np.int16)
        for k in range(action_size):
            indices[k,:] = np.where(action_profile[:, agent_idx] == k)[0]
        return indices


# Computation of expected payoff matrix 
object_vec = np.zeros((27,3), dtype=np.float32)
for i in range(2):
    for k in range(action_size):
        indices = indexing(i)
        for k in range(action_size): 
            object_vec[indices[k, :], i] = q_complete[i, k] 

    # Maximize the sum of expected payoff (objective function)
object_func = cp.Maximize(sum(prob_weight @ object_vec))

# Constraint 1: Sum of the Probabilities should be equal to 1 // should follow for all agents
sum_func_constr = cp.sum(prob_weight) == 1 

# Constraint 2: Each probability value should be greater than 1 and smaller than 0 // should follow for all agents
prob_constr_1 = all(prob_weight) >= 0
prob_constr_2 = all(prob_weight) <= 1

# Constraint 3: To verify, agents have no incentive to unilaterally deviate form equilibirum
add_constraint = []
for i in range(NUM_UAV):
    indices_i = indexing(i)
    for k in range(action_size):
        for l in range(action_size):
            if k != l:
                utility_k = cp.sum(prob_weight[indices_i[k, :]]) * q_complete[i, k]
                utility_l = cp.sum(prob_weight[indices_i[l, :]]) * q_complete[i, l]
                add_constraint.append(utility_k >= utility_l)

# Define the problem with constraints
complete_constraint = [sum_func_constr, prob_constr_1, prob_constr_2] + add_constraint
opt_problem = cp.Problem(object_func, complete_constraint)

# Solve the optimization problem using linear programming
try:
    opt_problem.solve()
    # print(opt_problem.status)
    if opt_problem.status == "optimal":
        print("Found solution")
        weights = prob_weight.value
        print(weights)
        print('Max Weight:', np.max(weights))
        print("Best Joint Action:", np.argmax(weights))
    else:
        weights = None
        print("Failed to find an optimal solution")
except:
    weights = None
