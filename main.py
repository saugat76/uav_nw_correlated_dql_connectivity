###################################
## Incorporate Game Theory in RL ##
###################################

import random
import numpy as np
from collections import defaultdict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from scipy.io import savemat
from uav_env import UAVenv
from misc import final_render
import torch
from torch import nn 
from torch.utils.tensorboard import SummaryWriter
import wandb
import argparse
from distutils.util import strtobool
from collections import deque
import os
import math
import warnings
from cvxpy import Variable, Problem, Minimize
import time
import nashpy as nash

warnings.filterwarnings("ignore")

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

os.chdir = ("")

# Define arg parser with default values
def parse_args():
    parser = argparse.ArgumentParser()
    # Arguments for the experiments name / run / setup and Weights and Biases
    parser.add_argument("--exp-name", type=str, default="madql_uav", help="name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of experiment to ensure reproducibility")
    parser.add_argument("--torch-deterministic", type= lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggeled, 'torch-backends.cudnn.deterministic=False'")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-track", type=lambda x: bool(strtobool(x)), default=False, help="if toggled, this experiment will be tracked with Weights and Biases project")
    parser.add_argument("--wandb-name", type=str, default="UAV_Subband_Allocation_DQN_Pytorch", help="project name in Weight and Biases")
    parser.add_argument("--wandb-entity", type=str, default= None, help="entity(team) for Weights and Biases project")

    # Arguments specific to the Algotithm used 
    parser.add_argument("--env-id", type=str, default= "ma-custom-UAV-connectivity", help="id of developed custom environment")
    parser.add_argument("--num-env", type=int, default=1, help="number of parallel environment")
    parser.add_argument("--num-episode", type=int, default=351, help="number of episode, default value till the trainning is progressed")
    parser.add_argument("--num-steps", type=int, default= 100, help="number of steps/epoch use in every episode")
    parser.add_argument("--learning-rate", type=float, default= 3.5e-4, help="learning rate of the dql alggorithm used by every agent")
    parser.add_argument("--gamma", type=float, default= 0.95, help="discount factor used for the calculation of q-value, can prirotize future reward if kept high")
    parser.add_argument("--batch-size", type=int, default= 512, help="batch sample size used in a trainning batch")
    parser.add_argument("--epsilon", type=float, default= 0.1, help="epsilon to set the eploration vs exploitation")
    parser.add_argument("--update-rate", type=int, default= 10, help="steps at which the target network updates it's parameter from main network")
    parser.add_argument("--buffer-size", type=int, default=125000, help="size of replay buffer of each individual agent")
    parser.add_argument("--epsilon-min", type=float, default=0.1, help="maximum value of exploration-exploitation paramter, only used when epsilon deacay is set to True")
    parser.add_argument("--epsilon-decay", type=lambda x: bool(strtobool(x)), default=False, help="epsilon decay is used, explotation is prioritized at early episodes and on later epsidoe exploitation is prioritized, by default set to False")
    parser.add_argument("--epsilon-decay-steps", type=int, default=1, help="set the rate at which is the epsilon is deacyed, set value equates number of steps at which the epsilon reaches minimum")

    # Environment specific arguments 
    # Arguments for used inside the wireless UAV based enviornment  
    parser.add_argument("--num-user", type=int, default=100, help="number of user in defined environment")
    parser.add_argument("--num-uav", type=int, default=5, help="number of uav for the defined environment")
    parser.add_argument("--generate-user-distribution", type=lambda x: bool(strtobool(x)), default=False, help="if true generate a new user distribution, set true if changing number of users")
    parser.add_argument("--carrier-freq", type=int, default=2, help="set the frequency of the carrier signal in GHz")
    parser.add_argument("--coverage-xy", type=int, default=1000, help="set the length of target area (square)")
    parser.add_argument("--uav-height", type=int, default=350, help="define the altitude for all uav")
    parser.add_argument("--theta", type=int, default=60, help="angle of coverage for a uav in degree")
    parser.add_argument("--bw-uav", type=float, default=4e6, help="actual bandwidth of the uav")
    parser.add_argument("--bw-rb", type=float, default=180e3, help="bandwidth of a resource block")
    parser.add_argument("--grid-space", type=int, default=100, help="seperating space for grid")
    parser.add_argument("--uav-dis-th", type=int, default=1000, help="distance value that defines which uav agent share info")
    parser.add_argument("--dist-pri-param", type=float, default=1/5, help="distance penalty priority parameter used in level 3 info exchange")
    
    args = parser.parse_args()

    return args

# GPU configuration use for faster processing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DNN modeling
class NeuralNetwork(nn.Module):
    # NN is set to have same structure for all lvl of info exchange in this setup
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_stack = model = nn.Sequential(
            nn.Linear(self.state_size,400),
            nn.ReLU(),
            nn.Linear(400,400),
            nn.ReLU(),
            nn.Linear(400, self.action_size)
        ).to(device=device)

    def forward(self, x):
        x = x.to(device)
        Q_values = self.linear_stack(x)
        return Q_values

class DQL:
    # Initializing a Deep Neural Network
    def __init__(self):
        self.state_space = 10
        self.state_size = 2
        self.action_size = 5
        self.combined_action_size = NUM_UAV ** self.action_size
        self.combined_state_size = NUM_UAV * self.state_size
        self.replay_buffer = deque(maxlen = 125000)
        self.gamma = discount_factor
        self.epsilon = epsilon        
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = alpha
        self.main_network = NeuralNetwork(self.state_size, self.combined_action_size).to(device)
        self.target_network = NeuralNetwork(self.state_size, self.combined_action_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr = self.learning_rate)
        self.loss_func = nn.SmoothL1Loss()      # Huber Loss // Combines MSE and MAE
        self.steps_done = 0

    # Storing information of individual UAV information in their respective buffer
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # Deployment of epsilon greedy policy
    # One approach to apply correlated Q-learning is sharing of the Q-table/ Q-matrix across all the agents
    # Another approach proposed by Hu and Wellman [] is observe the action and rewards of other agents and 
    # Simulate the Q-values of all other agents, instead of transferring the Q-table which has overheads
    # Since there is gonna be transfer of information every time step so, it's not necessary to transfer complete Q-table
    # but just the all the Q-value of the corresponding states

    # Another approach can be to share the state and action information which is used to create a joint Q-Matrix/Q-Table
    # which is then used for determining the joint policy based on the linear programming optimization 

    # In out case, we are sharing the partial Q-values at every time step, this is used to find correlated equilibrium

    # It is also assumed that the UAV takes action following a specific order of social convention from higher priority to lower

    # In addition we have to make sure that the transfer of information is completed after every timestep instead of consecutively
    # It insures the each individual agent's calculated correlated equilibrium is the same 
    # Joint action are indexed in the order: [0,0,0,0,0], [0,0,0,0,1],... [1,0,0,0,0]...[4,4,4,4,4]
    # Where, the indexes of joint actions of the joint action represents the agents index

    def correlated_equilibrium(self, shared_q_values, agent_idx):

        # Additional constraint function // Generating constraint used for solving the optimization equation
        def generate_add_constraint(NUM_UAV, shared_q_values, prob_weight):
            add_constraint = []
            for v in range(NUM_UAV):
                # Computing probability of individual agents action space from joint action probability distribution
                # prob_individual.append(sum(prob_individual_complete
                #                            [v*(UAV_OB[v].action_size)**(NUM_UAV-1): (v+1)*(UAV_OB[v].action_size)**(NUM_UAV-1) - 1]))
                # Using this individual probability, defining a constraint
                # Extracting individual Q value
                # Q values arraged as positions defiend [0,0,0,0,0],[0,0,0,0,1]...//[1,0,0,0,0],[1,0,0,0,1]...//...[4,0,0,0,0]...[4,4,4,4,4]  
                temp_cumulative = 0
                Q_ind = shared_q_values[v, :].reshape(UAV_OB[v].action_size, (UAV_OB[v].action_size)**(NUM_UAV-1))
                for l in range(Q_ind.size(0)):
                    for n in range(Q_ind.size(1)):
                        for m in range(Q_ind.size(0)):
                                # Only computing only for the agents action other than currently selected
                                if l != m:
                                    temp_sum = Q_ind[l, n] - Q_ind[m, n]
                                    temp_prod = prob_weight[l* Q_ind.size(0) + n]*temp_sum
                                    temp_cumulative += temp_prod
                # Adding constraint corresponding to each individual agent
                add_constraint.append(temp_cumulative >= 0)
            return add_constraint


        # Joint action size = number of agents ^ action size // for a state 
        # Optimizing the joint action so setting as a variable for CE optimization 
        joint_action_size = NUM_UAV ** UAV_OB[agent_idx].action_size
        prob_weight = Variable(joint_action_size)
        
        # Collect Q values for the corresponding states of each individual agents
        # Using negate value to use Minimize function for solving 
        q_complete = -np.vstack([shared_q_values[k].ravel() for k in range(NUM_UAV)])

        # Objective function
        object_vec = q_complete.reshape((NUM_UAV, joint_action_size))
        object_func = Minimize(sum(object_vec * prob_weight))

        # Constraint 1: Sum of the Probabilities should be equal to 1 // should follow for all agents
        sum_func_constr = all(sum(prob_weight)) == 1
        
        # Constraint 2: Each probability value should be grater than 1 // should follow for all agents
        prob_constr = all(prob_weight) >= 0 and all(prob_weight) <= 1

        # Constraint 3: Total function should be less than or equal to 0
        add_constraint = generate_add_constraint(NUM_UAV, shared_q_values, prob_weight)
        total_func_constr = add_constraint

        # Define the problem with constraints
        complete_constraint = [sum_func_constr, prob_constr] + total_func_constr
        opt_problem = Problem(object_func, complete_constraint)

        # Solve the optimization problem using linear programming
        try:
            opt_problem.solve()
            weights = prob_weight.value
            if np.isnan(weights).sum() > 0 or np.abs(weights.sum() - 1) > 0.00001:
                weights = None
                print("Failed to find an optimal solution")
        except:
            weights = None
            print("Failed to find a solution")
        return weights
    
    # def correlated_equilibrium(self, shared_q_values, agent_idx):
    #     # Create the game using the payoff matrices
    #     q_complete = np.vstack([shared_q_values[k].ravel() for k in range(NUM_UAV)])
    #     game = nash.Game(q_complete)

    #     # Find the all equilibrium using the Lemke-Howson algorithm
    #     equilibria = game.lemke_howson_enumeration()

    #     # Iterate over the equilibria and find the correlated equilibrium
    #     correlated_eq = None
    #     for eqs in equilibria:
    #         eq_satisfies = True
    #         for eq in eqs:
    #             if None in eq or any(val == None for val in eq) or any(val < 0 for val in eq):
    #                 eq_satisfies = False
    #                 break
    #         if eq_satisfies:
    #             correlated_eq = eqs
    #             print("correlated_eq", eqs)
    #             break
         
    #     print(correlated_eq[0].shape)
    #     self.prob = correlated_eq
        

    def update_probs(self, shared_q_values, agent_idx):
        return self.correlated_equilibrium(shared_q_values, agent_idx)

    def epsilon_greedy(self, agent_idx):
        temp = random.random()
        # Epsilon decay policy is employed for faster convergence
        epsilon_thres = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1*self.steps_done/self.epsilon_decay)
        self.steps_done += 1 
        # Each agents possible state space is same so, prob varible // joining all index
        prob = self.pi
        # prob = np.abs(prob) / sum(prob)
        # Compare against a epsilon threshold to either explore or exploit
        if temp <= epsilon_thres:
            # If less than threshold action choosen randomly
            # Each iteration will generate action list from individual correlation device 
            # Action here is representing joint action so has a size of (1, 3125)
            actions = np.random.randint(0, UAV_OB[agent_idx].action_size ** NUM_UAV)
        else:
            # Else (high prob) choosing the action based correlated equilibrium 
            # Action choosen based on joint state and action and using linear programming to find a solution
            # Action index and action correspond to each other so
            choices = np.arange(0, UAV_OB[agent_idx].action_size ** NUM_UAV, dtype=int)
            actions = np.random.choice(choices, p=prob)
        return actions
       

    # Training of the DNN 
    def train(self,batch_size, dnn_epoch):
        for k in range(dnn_epoch):
            minibatch = random.sample(self.replay_buffer, batch_size)
            minibatch = np.vstack(minibatch)
            minibatch = minibatch.reshape(batch_size,5)
            state = torch.FloatTensor(np.vstack(minibatch[:,0]))
            action = torch.LongTensor(np.vstack(minibatch[:,1]))
            reward = torch.FloatTensor(np.vstack(minibatch[:,2]))
            next_state = torch.FloatTensor(np.vstack(minibatch[:,3]))
            done = torch.Tensor(np.vstack(minibatch[:,4]))
            state = state.to(device = device)
            action = action.to(device = device)
            reward = reward.to(device = device)
            next_state = next_state.to(device = device)
            done = done.to(device = device)
            diff = state - next_state
            done_local = (diff != 0).any(dim=1).float().to(device)

            Q_next = self.target_network(next_state).detach()
            target_Q = reward.squeeze() + self.gamma * Q_next.max(1)[0].view(batch_size, 1).squeeze() * done_local

            # Forward 
            # Loss calculation based on loss function
            target_Q = target_Q.float()
            Q_main = self.main_network(state).gather(1, action).squeeze()
            loss = self.loss_func(target_Q.cpu().detach(), Q_main.cpu())
            # Intialization of the gradient to zero and computation of the gradient
            self.optimizer.zero_grad()
            loss.backward()
            # For gradient clipping
            for param in self.main_network.parameters():
                param.grad.data.clamp_(-1,1)
            # Gradient descent
            self.optimizer.step()
            

if __name__ == "__main__":
    args = parse_args()
    u_env = UAVenv(args)
    GRID_SIZE = u_env.GRID_SIZE
    NUM_UAV = u_env.NUM_UAV
    NUM_USER = u_env.NUM_USER
    num_episode = args.num_episode
    num_epochs = args.num_steps
    discount_factor = args.gamma
    alpha = args.learning_rate
    batch_size = args.batch_size
    update_rate = args.update_rate
    epsilon = args.epsilon
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay_steps
    dnn_epoch = 1

    # Set the run id name to tack all the runs 
    run_id = f"{args.exp_name}__{u_env.NUM_UAV}__{args.seed}__{int(time.time())}"

    # Set the seed value from arg parser to ensure reproducibility 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms = args.torch_deterministic

    # If wandb tack is set to True // Track the training process, hyperparamters and results
    if args.wandb_track:
        wandb.init(
            # Set the wandb project where this run will be logged
            project=args.wandb_name,
            entity=args.wandb_entity,
            sync_tensorboard= True,
            # track hyperparameters and run metadata
            config=vars(args),
            name= run_id,
            save_code= True,
        )
    # Track everyruns inside run folder // Tensorboard files to keep track of the results
    writer = SummaryWriter(f"runs/{run_id}")
    # Store the hyper paramters used in run as a Scaler text inside the tensor board summary
    writer.add_text(
        "hyperparamaters", 
        "|params|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()]))
    )

    # Store environment specific parameters
    env_params = {'num_uav':NUM_UAV, 'num_user': NUM_USER, 'grid_size': GRID_SIZE, 'start_pos': str(u_env.state), 
                      'coverage_xy':u_env.COVERAGE_XY, 'uav_height': u_env.UAV_HEIGHT, 'bw_uav': u_env.BW_UAV, 
                      'bw_rb':u_env.BW_RB, 'actual_bw_uav':u_env.ACTUAL_BW_UAV, 'uav_dis_thres': u_env.UAV_DIST_THRS,
                      'dist_penalty_pri': u_env.dis_penalty_pri}
    writer.add_text(
        "environment paramters", 
        "|params|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in env_params.items()]))
    )

    # Initialize global step value
    global_step = 0

    # Keeping track of the episode reward
    episode_reward = np.zeros(num_episode)
    episode_user_connected = np.zeros(num_episode)

    # Keeping track of individual agents 
    episode_reward_agent = np.zeros((NUM_UAV, 1))

    # Plot the grid space
    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0:1, 0:1])

    # Create object of each UAV agent // Each agent is equpped with it's DQL system
    UAV_OB = []
    for k in range(NUM_UAV):
                UAV_OB.append(DQL())
    best_result = 0

    # Initializing probability distribution for all the agents 
    # Individually calculation of the correlated equilibrium // Individual prob dist for all agent
    for k in range(NUM_UAV):
        # Each agent is tracking possible probabilites by themself // so each agent has pi variable
        # Setting the probabilities to equal value // each combination of action has same probability
        dimension = UAV_OB[k].combined_action_size
        UAV_OB[k].pi = np.ones(dimension) * (1/(UAV_OB[k].action_size ** UAV_OB[k].action_size))

    # Start of the episode
    for i_episode in range(num_episode):
        print(i_episode)

        # Environment reset and get the states
        u_env.reset()

        # Get the initial states
        states = u_env.get_state()
        reward = np.zeros(NUM_UAV)
        
        for t in range(num_epochs):
            drone_act_list = []
            # Update the target network 
            for k in range(NUM_UAV):
                if t % update_rate == 0:
                    UAV_OB[k].target_network.load_state_dict(UAV_OB[k].main_network.state_dict())

            # Determining the actions for all drones // states are shared among UAVs
            # Sharing of Q-values is also completed in this loop
            
            # Intialization of shared_q_value variable 
            shared_q_values = torch.zeros(NUM_UAV, UAV_OB[1].combined_action_size)

            states_ten = torch.from_numpy(states)
            shared_state = states_ten.numpy().flatten()
            # To simplify is programming calculating once and sharing among UAVs
            # Sharing of the Q-values
            for k in range(NUM_UAV):
                state = states_ten[k, :]
                state = state.float()
                state = torch.unsqueeze(torch.FloatTensor(state), 0)
                q_values = UAV_OB[k].main_network(state)
                shared_q_values[k, :]= q_values.detach()

            for k in range(NUM_UAV):
                weights = UAV_OB[k].correlated_equilibrium(shared_q_values, k)
                if weights is not None:
                    UAV_OB[k].pi = weights
                action = UAV_OB[k].epsilon_greedy(k)
                print(f"Correlated action for agent {k} is {action}")
                drone_act_list.append(action)

            # Find the global reward for the combined set of actions for the UAV
            temp_data = u_env.step(drone_act_list)
            reward = temp_data[1]
            done = temp_data[2]
            next_state = u_env.get_state()

            # Store the transition information
            for k in range(NUM_UAV):
                ## Storing of the information on the individual UAV and it's reward value in itself.
                state = states_ten[k, :].numpy()
                next_sta = next_state[k, :]
                action = drone_act_list[k].cpu().numpy()
                reward_ind = reward[k]
                UAV_OB[k].store_transition(state, action, reward_ind, next_sta, done)

            # Calculation of the total episodic reward of all the UAVs 
            # Calculation of the total number of connected User for the combination of all UAVs
            ##########################
            ####   Custom logs    ####
            ##########################
            episode_reward[i_episode] += sum(reward)
            episode_user_connected[i_episode] += sum(temp_data[4])
            user_connected = temp_data[4]
            
            # Also calculting episodic reward for each agent // Add this in your main program 
            episode_reward_agent = np.add(episode_reward_agent, reward)

            states = next_state

            for k in range(NUM_UAV):
                if len(UAV_OB[k].replay_buffer) > batch_size:
                    UAV_OB[k].train(batch_size, dnn_epoch)

        # Update the probabilities for all the agents 
        for k in range(NUM_UAV):
            UAV_OB[k].update_probs(shared_q_values, k)

        #############################
        ####   Tensorboard logs  ####
        #############################
        # Track the same information regarding the performance in tensorboard log directory 
        writer.add_scalar("charts/episodic_reward", episode_reward[i_episode], i_episode)
        writer.add_scalar("charts/episodic_length", num_epochs, i_episode)
        writer.add_scalar("charts/connected_users", episode_user_connected[i_episode], i_episode)
        if args.wandb_track:
            wandb.log({"episodic_reward": episode_reward[i_episode], "episodic_length": num_epochs, "connected_users":episode_user_connected[i_episode], "global_steps": global_step})
            # wandb.log({"reward: "+ str(agent): reward[agent] for agent in range(NUM_UAV)})
            # wandb.log({"connected_users: "+ str(agent_l): user_connected[agent_l] for agent_l in range(NUM_UAV)})
        global_step += 1
        
        # Keep track of hyper parameter and other valuable information in tensorboard log directory 
        # Track the params of all agent
        # Since all agents are identical only tracking one agents params
        writer.add_scalar("params/learning_rate", UAV_OB[1].learning_rate, i_episode )
        writer.add_scalar("params/epsilon", UAV_OB[1].epsilon_thres, i_episode)

        
        if i_episode % 10 == 0:
            # Reset of the environment
            u_env.reset()
            # Get the states
            states = u_env.get_state()
            states_ten = torch.from_numpy(states)
            for t in range(100):
                drone_act_list = []
                for k in range(NUM_UAV):
                    state = states[k, :]
                    state = torch.unsqueeze(torch.FloatTensor(state),0)
                    correlated_actions = np.random.choice(0, 4, size=(NUM_UAV), p=UAV_OB[k].prob)
                    best_next_action = correlated_actions[k]
                    drone_act_list.append(best_next_action)
                temp_data = u_env.step(drone_act_list)
                states = u_env.get_state()
                states_fin = states
                if best_result < sum(temp_data[4]):
                    best_result = sum(temp_data[4])
                    best_state = states    

            # Custom logs and figures save / 
            custom_dir = f'custom_logs\{run_id}'
            if not os.path.exists(custom_dir):
                os.makedirs(custom_dir)
                
            u_env.render(ax1)
            ##########################
            ####   Custom logs    ####
            ##########################
            figure = plt.title("Simulation")
            # plt.savefig(custom_dir + f'\{i_episode}__{t}.png')

            #############################
            ####   Tensorboard logs  ####
            #############################
            # writer.add_figure("images/uav_users", figure, i_episode)
            writer.add_scalar("charts/connected_users_test", sum(temp_data[4]))

            print(drone_act_list)
            print("Number of user connected in ",i_episode," episode is: ", temp_data[4])
            print("Total user connected in ",i_episode," episode is: ", sum(temp_data[4]))


    def smooth(y, pts):
        box = np.ones(pts)/pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    ##########################
    ####   Custom logs    ####
    ##########################
    ## Save the data from the run as a file in custom logs
    mdict = {'num_episode':range(0, num_episode),'episodic_reward': episode_reward}
    savemat(custom_dir + f'\episodic_reward.mat', mdict)
    mdict_2 = {'num_episode':range(0, num_episode),'connected_user': episode_user_connected}
    savemat(custom_dir + f'\connected_users.mat', mdict_2)
    mdict_3 = {'num_episode':range(0, num_episode),'episodic_reward_agent': episode_reward_agent}
    savemat(custom_dir + f'\epsiodic_reward_agent.mat', mdict_3)
    
    # Plot the accumulated reward vs episodes // Save the figures in the respective directory 
    # Episodic Reward vs Episodes
    fig_1 = plt.figure()
    plt.plot(range(0, num_episode), episode_reward)
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title("Episode vs Episodic Reward")
    plt.savefig(custom_dir + f'\episode_vs_reward.png')
    plt.close()
    # Episode vs Connected Users
    fig_2 = plt.figure()
    plt.plot(range(0, num_episode), episode_user_connected)
    plt.xlabel("Episode")
    plt.ylabel("Connected User in Episode")
    plt.title("Episode vs Connected User in Episode")
    plt.savefig(custom_dir + f'\episode_vs_connected_users.png')
    plt.close()
    # Episodic Reward vs Episodes (Smoothed)
    fig_3 = plt.figure()
    smoothed = smooth(episode_reward, 10)
    plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
    plt.xlabel("Episode")
    plt.ylabel("Episodic Reward")
    plt.title("Smoothed Episode vs Episodic Reward")
    plt.savefig(custom_dir + f'\episode_vs_rewards(smoothed).png')
    plt.close()
    # Plot for best and final states 
    fig = plt.figure()
    final_render(states_fin, "final")
    plt.savefig(custom_dir + r'\final_users.png')
    plt.close()
    fig_4 = plt.figure()
    final_render(best_state, "best")
    plt.savefig(custom_dir + r'\best_users.png')
    plt.close()
    print(states_fin)
    print('Total Connected User in Final Stage', temp_data[4])
    print("Best State")
    print(best_state)
    print("Total Connected User (Best Outcome)", best_result)

    #############################
    ####   Tensorboard logs  ####
    #############################
    writer.add_figure("images/uav_users_best", fig_4)
    writer.add_text(
            "best outcome", str(best_state))
    writer.add_text(
            "best result", str(best_result))
    wandb.finish()
    writer.close()

