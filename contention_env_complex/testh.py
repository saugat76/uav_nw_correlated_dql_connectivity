import cvxpy as cp

# Define the variables
# Each variable corresponds to a joint action
p_A1_B1 = cp.Variable()
p_A1_B2 = cp.Variable()
p_A2_B1 = cp.Variable()
p_A2_B2 = cp.Variable()

# Q-values for each action for each player
Q1_A1 = 2
Q1_A2 = 3
Q2_B1 = 1
Q2_B2 = 4

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
