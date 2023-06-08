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
from cvxpy import Variable, Problem, Minimize, Maximize, multiply, matmul
import cvxpy
import time
import sys
from joblib import Parallel, delayed
from multiprocessing import Pool



os.chdir = ("")

# Define arg parser with default values
def parse_args():
    parser = argparse.ArgumentParser()
    # Arguments for the experiments name / run / setup and Weights and Biases
    parser.add_argument("--exp-name", type=str, default="correlated_madql_lstm_uav", help="name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of experiment to ensure reproducibility")
    parser.add_argument("--torch-deterministic", type= lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggeled, 'torch-backends.cudnn.deterministic=False'")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-track", type=lambda x: bool(strtobool(x)), default=False, help="if toggled, this experiment will be tracked with Weights and Biases project")
    parser.add_argument("--wandb-name", type=str, default="uav_nw_correlated_dql_connectivity", help="project name in Weight and Biases")
    parser.add_argument("--wandb-entity", type=str, default= None, help="entity(team) for Weights and Biases project")

    # Arguments specific to the Algotithm used 
    parser.add_argument("--env-id", type=str, default= "correlated-ma-custom-UAV-connectivity", help="id of developed custom environment")
    parser.add_argument("--num-env", type=int, default=1, help="number of parallel environment")
    parser.add_argument("--num-episode", type=int, default=351, help="number of episode, default value till the trainning is progressed")
    parser.add_argument("--max-steps", type=int, default= 100, help="max number of steps/epoch use in every episode")
    parser.add_argument("--learning-rate", type=float, default= 3.5e-4, help="learning rate of the dql alggorithm used by every agent")
    parser.add_argument("--gamma", type=float, default= 0.95, help="discount factor used for the calculation of q-value, can prirotize future reward if kept high")
    parser.add_argument("--batch-size", type=int, default= 512, help="batch sample size used in a trainning batch")
    parser.add_argument("--epsilon", type=float, default= 0.1, help="epsilon to set the eploration vs exploitation")
    parser.add_argument("--update-rate", type=int, default= 10, help="steps at which the target network updates it's parameter from main network")
    parser.add_argument("--buffer-size", type=int, default=125000, help="size of replay buffer of each individual agent")
    parser.add_argument("--epsilon-min", type=float, default=0.1, help="maximum value of exploration-exploitation paramter, only used when epsilon deacay is set to True")
    parser.add_argument("--epsilon-decay", type=lambda x: bool(strtobool(x)), default=False, help="epsilon decay is used, explotation is prioritized at early episodes and on later epsidoe exploitation is prioritized, by default set to False")
    parser.add_argument("--epsilon-decay-steps", type=int, default=1, help="set the rate at which is the epsilon is deacyed, set value equates number of steps at which the epsilon reaches minimum")
    parser.add_argument("--layers", type=int, default=2, help="set the number of layers for the target and main neural network")
    parser.add_argument("--nodes", type=int, default=400, help="set the number of nodes for the target and main neural network layers")
    parser.add_argument("--seed-sync", type=lambda x: bool(strtobool(x)), default=False, help="synchronize the seed value among agents, by default set to False")
    parser.add_argument("--lstm-layers", type=int, default=2, help="set the number of layers for the lstm layer of target and main neural network")
    parser.add_argument("--lstm-nodes", type=int, default=64, help="set the number of nodes for the lstm layer of target and main neural network layers")

    # Environment specific arguments
    # To be consitent with previous project addition of level 5 and 6
    parser.add_argument("--info-exchange-lvl", type=int, default=5, help="information exchange level between UAVs: 5 -> individual partial q-values, 6 -> q-values and state") 
     
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
    parser.add_argument("--reward-func", type=int, default=1, help="reward func used 1-> global reward across agents, 2-> independent reward")


    args = parser.parse_args()

    return args

# GPU configuration use for faster processing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# DNN modeling
class NeuralNetwork(nn.Module):
    # NN is set to have same structure for all lvl of info exchange in this setup
    def __init__(self, state_size, combined_action_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.state_size = state_size
        self.combined_action_size = combined_action_size
        self.action_size = action_size
        if args.layers == 2:
            self.lstm = nn.LSTM(NUM_UAV * self.action_size, args.lstm_nodes, args.lstm_layers)
            self.linear_stack = model = nn.Sequential(
                nn.Linear(self.state_size + args.lstm_nodes, args.nodes),
                nn.ReLU(),
                nn.Linear(args.nodes, args.nodes),
                nn.ReLU(),
                nn.Linear(args.nodes, self.combined_action_size)
            ).to(device=device)

        elif args.layers == 3:
            self.lstm = nn.LSTM(NUM_UAV * self.action_size, args.lstm_nodes, args.lstm_layers)
            self.linear_stack = model = nn.Sequential(
            nn.Linear(self.state_size + args.lstm_nodes, args.nodes),
            nn.ReLU(),
            nn.Linear(args.nodes, args.nodes),
            nn.ReLU(),
            nn.Linear(args.nodes, args.nodes),
            nn.ReLU(),
            nn.Linear(args.nodes, self.combined_action_size)
        ).to(device=device)


    def forward(self, x1, x2):
        x1 = x1.to(device)
        x2 = x2.to(device)
        lstm_out, _ = self.lstm(x2)
        linear_in = torch.cat((lstm_out.view(-1, args.lstm_nodes).squeeze(), x1), dim=-1)
        # try:
        #     linear_in = torch.cat((lstm_out.view(-1, args.lstm_nodes), x1.unsqueeze(0)), dim=-1)
        # except:
        #     linear_in = torch.cat((lstm_out.view(-1, args.lstm_nodes), x1), dim=-1)
        Q_values = self.linear_stack(linear_in)
        return Q_values

class DQL:
    # Initializing a Deep Neural Network
    def __init__(self):
        # lvl 1 info exchange only their respective state for lvl 4 all agents states 
        if args.info_exchange_lvl == 5:
            self.state_size = 2
        elif args.info_exchange_lvl == 6:
            self.state_size = args.num_uav * 2
        self.state_space = 10
        self.action_size = 5
        self.combined_action_size = self.action_size ** NUM_UAV
        self.combined_state_size = NUM_UAV * self.state_size
        self.replay_buffer = deque(maxlen = 125000)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        # Intialize the epsilon thres as epsilon as well which later changes based on epsilon decay algo
        self.epsilon_thres = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = alpha
        self.main_network = NeuralNetwork(self.state_size, self.combined_action_size, self.action_size).to(device)
        self.target_network = NeuralNetwork(self.state_size, self.combined_action_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr = self.learning_rate)
        self.loss_func = nn.SmoothL1Loss()      # Huber Loss // Combines MSE and MAE
        self.steps_done = 0

    # Storing information of individual UAV information in their respective buffer
    def store_transition(self, state, action, reward, next_state, done, previous_action, next_action):
        self.replay_buffer.append((state, action, reward, next_state, done, previous_action, next_action))

    #################################################
    ######      Correlated Equilibrium       ########
    #################################################

    # Deployment of epsilon greedy policy
    # One approach to apply correlated Q-learning is sharing of the Q-table/ Q-matrix across all the agents
    # Another approach proposed by Hu and Wellman [] is observe the action and rewards of other agents and 
    # Simulate the Q-values of all other agents, instead of transferring the Q-table which has overheads
    # Since there is gonna be transfer of information every time step so, it's not necessary to transfer complete Q-table
    # but just the all the Q-value of the corresponding states

    # Another approach can be to share the state and action information which is used to create a joint Q-Matrix/Q-Table
    # which is then used for determining the joint policy based on the linear programming optimization 

    # In out case, we are sharing the partial Q-values extracted from agents DQN at every time step, this is used to find correlated equilibrium

    # It is also assumed that the UAV takes action following a specific order of social convention from higher priority to lower

    # In addition we have to make sure that the transfer of information is completed after every timestep instead of consecutively
    # It insures the each individual agent's calculated correlated equilibrium is the same 
    # Joint action are indexed in the order: [0,0,0,0,0], [0,0,0,0,1],... [1,0,0,0,0]...[4,4,4,4,4]
    # Where, the indexes of joint actions of the joint action represents the agents index

    def correlated_equilibrium(self, shared_q_values, agent_idx):
        # Considering a deterministic system where the correleted action are fixed
        # Bruteforcing thorugh all the available option for each UAV agent and check for constraint satisfaction4
        shared_q_values = shared_q_values.to(device=device)
        max_ind = torch.argsort(torch.sum(shared_q_values, axis=0), descending = True).to(device=device)
        for k in max_ind:
            # Go over all the joint action indices which gives the max sum of Q's -> Descending order
            # Joint action value in form [0, 1, 1, 2, 3] from the joint action index
            current_complete_action = UAV_OB[agent_idx].action_profile[k, :]
            # Extracring the action value of the agent_idx from complete action selected
            current_ind_action = current_complete_action[agent_idx]
            # Extracting indices of the others except agent_idx
            excluded_idx = torch.arange(len(current_complete_action))[np.arange(len(current_complete_action)) != agent_idx]
            # Extracting the indcies where A-i matches which corresponds to the action profile index where all 
            # The value of current complete action matches excpet that of agent_idx
            Ai_actions = (UAV_OB[agent_idx].action_profile[:, excluded_idx] == current_complete_action[excluded_idx]).all(dim=1).nonzero().to(device=device)
            # Vectorizing the Q-value of all agents
            q_value_mat = shared_q_values[:, k] * torch.ones(NUM_UAV, Ai_actions.shape[0]).to(device=device)
            sum_contr =  q_value_mat.transpose(0, 1) - shared_q_values[:, Ai_actions.squeeze()]
            if torch.all(sum_contr >= 0):
                correlated_action_selected = k
                return correlated_action_selected
        return None


    def epsilon_greedy(self, agent_idx, state):
        temp = random.random()
        # Epsilon decay policy is employed for faster convergence
        self.epsilon_thres = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1*self.steps_done/self.epsilon_decay)
        self.steps_done += 1 
        # Each agents possible state space is same so, prob varible // joining all index
        choosen_action = self.correlated_choice
        # Compare against a epsilon threshold to either explore or exploit
        if temp <= self.epsilon_thres:
            # If less than threshold action choosen randomly
            # Each iteration will generate action list from individual correlation device 
            # Action here is representing joint action so has a value between (1, 3125)
            actions = np.random.randint(0, UAV_OB[agent_idx].action_size ** NUM_UAV, dtype=int)
        else:
            # Else (high prob) choosing the action based correlated equilibrium 
            # Action choosen based on correlated probabilities of joint action which is 
            # Calculated using linear programming to find a solution
            choices = np.arange(0, UAV_OB[agent_idx].action_size ** NUM_UAV, dtype=int)
            actions = choices[choosen_action]
        return actions
       

    # Training of the DNN 
    def train(self,batch_size, dnn_epoch, agent_idx):
        # In this function state is representing -> Location of UAV + Previous Computed Correlated Action
        for k in range(dnn_epoch):
            minibatch = random.sample(self.replay_buffer, batch_size)
            minibatch = np.vstack(minibatch)
            minibatch = minibatch.reshape(batch_size,7)
            state = torch.FloatTensor(np.vstack(minibatch[:,0]))
            action = torch.LongTensor(np.vstack(minibatch[:,1]))
            reward = torch.FloatTensor(np.vstack(minibatch[:,2]))
            next_state = torch.FloatTensor(np.vstack(minibatch[:,3]))
            done = torch.Tensor(np.vstack(minibatch[:,4]))
            previous_action = torch.Tensor(np.vstack(minibatch[:,5]))
            next_action = torch.Tensor(np.vstack(minibatch[:,6]))
            state = state.to(device = device)
            action = action.to(device = device)
            reward = reward.to(device = device)
            next_state = next_state.to(device = device)
            done = done.to(device = device)
            previous_action = previous_action.to(device=device)
            next_action = next_action.to(device=device)
            done_local = (done).any(dim=1).float().to(device)

            # Implementation of DQL algorithm 
            Q_next = self.target_network(next_state, next_action).detach()

            target_Q = reward.squeeze() + self.gamma * Q_next.max(1)[0].view(batch_size, 1).squeeze() * done_local

            # Forward 
            # Loss calculation based on loss function
            target_Q = target_Q.float()
            Q_main = self.main_network(state, previous_action).gather(1, action).squeeze()
            loss = self.loss_func(target_Q.cpu().detach(), Q_main.cpu())

            # Store the loss information for debugging purposes 
            self.loss = loss

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
    max_epochs = args.max_steps
    discount_factor = args.gamma
    alpha = args.learning_rate
    batch_size = args.batch_size
    update_rate = args.update_rate
    epsilon = args.epsilon
    epsilon_min = args.epsilon_min
    epsilon_decay = args.epsilon_decay_steps
    dnn_epoch = 1


    # Set the run id name to tack all the runs 
    run_id = f"{args.exp_name}__lvl{args.info_exchange_lvl}__{u_env.NUM_UAV}__{args.seed}__{int(time.time())}"

    # Set the seed value from arg parser to ensure reproducibility 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms = args.torch_deterministic

    # Synchronziation of seed value // Set a constant seed by randomization
    def seeding_sync(seed_state, np_seed_state, torch_seed_state):
        random.setstate(seed_state)
        np.random.set_state(np_seed_state)   
        torch.manual_seed(torch_seed_state)
        torch.use_deterministic_algorithms = torch_seed_state

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
    # Initialize action list for each UAV agent // output from the DQN and equilibria represents the indexes
    # Array to map those indexes to a joint action profile
    UAV_OB = []
    for k in range(NUM_UAV):
        UAV_OB.append(DQL())
        action_values_ind = torch.arange(UAV_OB[k].action_size)
        action_profile = torch.stack(torch.meshgrid(*([action_values_ind] * NUM_UAV)), dim=-1).reshape(-1, NUM_UAV)
        UAV_OB[k].action_profile = action_profile           
        # Each joint action can be extracted using the index value from the equilibrium and 
        # Independent action can be extracted using agents index

    best_result = 0

    # Initialize action list for each UAV agent // O/p from the DQN and equilibria represents the indexes
    # Array to map those indexes to a joint action profile

    # Start of the episode
    for i_episode in range(num_episode):
        print(i_episode)

        # Environment reset and get the states
        u_env.reset()

        # Get the initial states
        states = u_env.get_state()
        reward = np.zeros(NUM_UAV)

        next_action = torch.zeros(NUM_UAV, UAV_OB[0].action_size)
        
        for t in range(max_epochs):
            drone_act_list = []
            action_selected_list = []
            previous_action = torch.zeros(NUM_UAV, UAV_OB[0].action_size)

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
                if args.info_exchange_lvl == 5:
                    state = states_ten[k, :]
                elif args.info_exchange_lvl == 6:
                    state = states_ten.flatten()
                state = state.float()
                q_values = UAV_OB[k].main_network(state, previous_action.flatten().unsqueeze(0))
                shared_q_values[k, :]= q_values.detach()

            # Get the current seed state // Seed synchronization
            if args.seed_sync:
                seed_state = random.getstate()
                np_seed_state = np.random.get_state()
                torch_seed_state = torch.seed()

            for k in range(NUM_UAV):

                # Synchronization of seeding between agents // Synchronize randomness using same seed
                if args.seed_sync:
                    seeding_sync(seed_state, np_seed_state, torch_seed_state)
                # Note: seed synchronization might be creating the problem as it wont allow other agents to explore separately

                correlated_actions = UAV_OB[k].correlated_equilibrium(shared_q_values, k)
                if correlated_actions is not None:
                    UAV_OB[k].correlated_choice = correlated_actions
                else:
                    UAV_OB[k].correlated_choice = np.random.randint(0, UAV_OB[k].action_size ** NUM_UAV, dtype=int)  

                action = UAV_OB[k].epsilon_greedy(k, state)
                action_selected_list.append(action)
                # Action of the individual agent from the correlated action list
                # Correlated joint action // computed by individual agent
                action_selected = np.copy(UAV_OB[k].action_profile[action])
                
                #########################################################
                # Only one equilibria calculation // Can change if want a actually full distributed system
                # print(action_selected)
                # Trying a shortcut // Since the correlated action selection gives same results for all agents
                # Instead of computing in loop using the same value to see faster output
                drone_act_list = action_selected.tolist()
                for k in range(NUM_UAV-1):
                    action_selected_list.append(action)
                # If removed this need to adjust the store_transition function to action = correlated_action_list[k]
                break
                ########################################################
                
                # Individual action from the correleted joint action
                drone_act_list.append(action_selected[k])
            
            for k in range(NUM_UAV):
                next_action[k , drone_act_list[k]] = 1

            # Find the global reward for the combined set of actions for the UAV
            # Reward function design for both level 5 and 6 is the same so, we dont pass the argument
            # To the UAV environment for the computation in the step function
            temp_data = u_env.step(drone_act_list)
            reward = temp_data[1]
            done = temp_data[2]
            next_state = u_env.get_state()
            # Computation of other condition for done information
            # If done i.e. any UAV out-of-boundary or next-state == state
            done = (states_ten == next_state) or done

            # Store the transition information
            for k in range(NUM_UAV):
                ## Storing of the information on the individual UAV and it's reward value in itself.
                # If the lvl of info exchange is 5 -> partial q-values sharing among UAVs
                # Else if lvl info exchnage is 6 -> partial q-value and states
                if args.info_exchange_lvl == 5:
                    state = states_ten[k, :].numpy()
                    next_sta = next_state[k, :]
                elif args.info_exchange_lvl == 6:
                    state = states_ten.numpy().flatten()
                    next_sta = next_state.flatten()
                action = action_selected_list[k]
                done_individual = done[k]
                if args.reward_func == 1:
                    reward_ind = reward
                elif args.reward_func == 2:
                    reward_ind = reward[k]
                previous_action_passed = previous_action.flatten()
                next_action_passed = next_action.flatten()
                UAV_OB[k].store_transition(state, action, reward_ind, next_sta, done_individual, previous_action_passed, next_action_passed)

            # Calculation of the total episodic reward of all the UAVs 
            # Calculation of the total number of connected User for the combination of all UAVs
            ##########################
            ####   Custom logs    ####
            ##########################
            if args.reward_func == 1:
                episode_reward[i_episode] += reward
            elif args.reward_func == 2:
                episode_reward[i_episode] += np.sum(reward)
            episode_user_connected[i_episode] += np.sum(temp_data[4])
            user_connected = temp_data[4]
            
            # Also calculting episodic reward for each agent // Add this in your main program 
            episode_reward_agent = np.add(episode_reward_agent, reward)

            states = next_state
            previous_action = next_action

            for k in range(NUM_UAV):
                if len(UAV_OB[k].replay_buffer) > batch_size:
                    UAV_OB[k].train(batch_size, dnn_epoch, k)
                    if args.wandb_track:
                        wandb.log({f"loss__{k}" : UAV_OB[k].loss})
            
            # # If all UAVs are done the program_done is True
            done_program = all(done)
            if done_program:
                break


        #############################
        ####   Tensorboard logs  ####
        #############################
        # Track the same information regarding the performance in tensorboard log directory 
        writer.add_scalar("charts/episodic_reward", episode_reward[i_episode], i_episode)
        writer.add_scalar("charts/episodic_length", t, i_episode)
        writer.add_scalar("charts/connected_users", episode_user_connected[i_episode], i_episode)
        if args.wandb_track:
            wandb.log({"episodic_reward": episode_reward[i_episode], "episodic_length": t, "connected_users":episode_user_connected[i_episode], "global_steps": global_step})
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
            reward = np.zeros(NUM_UAV)

            for t in range(100):
                drone_act_list = []
                action_selected_list = []
                previous_action = torch.zeros(NUM_UAV, UAV_OB[0].action_size)
                states_ten = torch.from_numpy(states)
                for k in range(NUM_UAV):
                    if args.info_exchange_lvl == 5:
                        state = states_ten[k, :]
                    elif args.info_exchange_lvl == 6:
                        state = states_ten.flatten()
                    state = state.float()
                    choices = np.arange(0, UAV_OB[k].action_size, dtype=int)
                    q_values = UAV_OB[k].main_network(state, previous_action.flatten().unsqueeze(0))
                    shared_q_values[k, :]= q_values.detach()
                    correlated_actions = UAV_OB[k].correlated_equilibrium(shared_q_values, k)

                    if correlated_actions is not None:
                        UAV_OB[k].correlated_choice = correlated_actions
                    else:
                        UAV_OB[k].correlated_choice = np.random.randint(0, UAV_OB[k].action_size ** NUM_UAV, dtype=int)

                for k in range(NUM_UAV):
                    action_selected = np.copy(UAV_OB[k].action_profile[UAV_OB[k].correlated_choice])     
                    drone_act_list.append(action_selected[k])

                temp_data = u_env.step(drone_act_list)
                for k in range(NUM_UAV):
                    previous_action[k , drone_act_list[k]] = 1
                states = u_env.get_state()
                states_fin = states
                if best_result < sum(temp_data[4]):
                    best_result = sum(temp_data[4])
                    best_state = states     

            # Custom logs and figures save / 
            custom_dir = f'custom_logs\lvl_{args.info_exchange_lvl}\{run_id}'
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