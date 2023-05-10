 def correlated_equilibrium(self, epsilon_thres):
        # Note: Number of action given to all the agent in our case equals to NUM_UAV

        # Use linear programming to find a correlated equilibria 
        policy = pulp.LpProblem("CorrelatedEquilibrium", pulp.LpMaximize)
        # Defining the descision variable for each state s ; x
        # Each agent has joint action and state valyes 
        x = [[[[pulp.LpVariable(f"x_{s_1}_{s_2}_{a}_{i}", cat=pulp.LpBinary)
                    for i in range(NUM_UAV)] for a in range(NUM_UAV)]
                    for s_2 in range(self.Q.shape[0])] for s_1 in range(self.Q.shape[1])]
        # Expected joint//global Q-value (J) using linear combination of Q-values and the decision variable 
        # (probability of taking the action by agent )
        J = pulp.lpSum(self.Q[s_1][s_2][a][i] * x[s_1][a][i] for s_1 in range(self.Q.shape[0])
                                    for s_2 in range(self.Q.shape[1])
                                    for a in range(NUM_UAV)
                                    for i in range(NUM_UAV))

        # Adding the objective function to the LP problem 
        policy += J 

        # Addition of constraint for the optimization problem 
        # Constraint 1 - Sum of Probabilities Equal to 1 for all agents - should follow for all possible states
        # Add each constraint to the problem 
        for s_1 in range(self.Q.shape[0]):
            for s_2 in range(self.Q.shaper[1]):
                prob += pulp.lpSum(x[s_1][s_2][a][i] for a in range(NUM_UAV)
                                            for i in range(NUM_UAV)) == 1
               







## My next try 

    # def correlated_quilibrium(self, shared_q_values):

    #     # Function to generate additional Q matrix // other actions combination except for agent i
    #     def get_additional_matrix(self, shared_q_values):
    #         additional_mat = []
    #         for k in range(NUM_UAV):
    #             for v in range(NUM_UAV ** self.action_size):
    #                 if 

    #     # Since the size of action space for all the agents is same // only can use one variable to define the variable
    #     # Joint action size = number of agents x action size
    #     # Optimizing the joint action so setting as a variable for ce optimization 
    #     joint_action_size = NUM_UAV ** self.action_size
    #     prob_weight = Variable(joint_action_size)
    #     # Collect Q values for the corresponding states of each individual agents
    #     # Using negate value to use Minimize function for solving 
    #     for k in NUM_UAV:
    #         q_complete = q_complete.append(shared_q_values[k])
    #     q_complete = q_complete.reshape(1, joint_action_size)
    #     object_vec = -np.copy(q_complete)

    #     # Objective function
    #     object_func = object_vec * prob_weight

    #     # Constraint 1 // Sum of the Probabilities equate to 1
    #     sum_func = np.sum(prob_weight)
    #     # Constraint 2 // Total Func should be less than or equal to 0
    #     total_func = 

    #     # Constraint 3 // All the prob weight should be between 0 and 1
    #     # Included in the optimization problem itself

    #     # Define the problem with constraints
    #     opt_problem = Problem(Minimize(object_func),
    #                    [sum_func == 1, total_func <= 0, all(prob_weight) <= 1, all(prob_weight) >= 0])

    #     # Solve the optimization problem using LP
    #     try:
    #         opt_problem.solve()
    #         weights = prob_weight.value
    #         if np.isnan(weights).sum() > 0 or np.abs(weights.sum() - 1) > 0.00001:
    #             weights = None
    #             print("fail to find solution")
    #     except:
    #         weights = None
    #         print("cant find solution")
    #     return weights 
