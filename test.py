import numpy as np
import cvxpy as cp

num_agents = 5
num_actions = 5

# Create random payoff matrices with correct dimensions for each agent
payoff_matrices = [
    np.random.random((num_actions, num_actions)) for _ in range(num_agents)
]
print(payoff_matrices)
print(payoff_matrices[1])


# Define the optimization variables
correlated_probabilities = cp.Variable((num_actions, num_actions), nonneg=True)

# Define the optimization problem
objective = cp.Maximize(0)  # Placeholder objective since we only need constraints
constraints = []
for i in range(num_agents):
    # Calculate the expected payoff for each agent's action
    expected_payoff = cp.sum(cp.multiply(correlated_probabilities, payoff_matrices[i]))

    # Add the constraint that the expected payoff should be maximized for each agent
    constraints.append(cp.sum(cp.multiply(correlated_probabilities, payoff_matrices[i])) >= expected_payoff)
constraints.append(cp.sum(correlated_probabilities, 1) == 1)
constraints.append(all(correlated_probabilities) >= 0)
# Create the optimization problem
problem = cp.Problem(objective, constraints)

# Solve the optimization problem
problem.solve()

# Retrieve the correlated equilibrium probabilities
correlated_equilibrium = correlated_probabilities.value

# Print the correlated equilibrium
print("Correlated Equilibrium:")
for agent_index, mixed_strategy in enumerate(correlated_equilibrium):
    print(f"Agent {agent_index + 1}: {mixed_strategy}")
