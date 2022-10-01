import numpy as np
from scripts.hac.layer import Layer
import os
import logging

# Below class instantiates an agent
class Agent():
    def __init__(self, args, env, log_dir):

        self.args = args

        agent_params = env.agent_params

        # Set subgoal testing ratio each layer will use
        self.subgoal_test_perc = agent_params["subgoal_test_perc"]

        # Create agent with number of levels specified by user       
        self.layers = [Layer(i,args,env, agent_params) for i in range(args.n_layers)]        

        # Below attributes will be used help save network parameters
        self.log_dir = log_dir

        # Initialize actor/critic networks.  Load saved parameters if not retraining
        self.initialize_networks()   
        
        # goal_array will store goal for each layer of agent.
        self.goal_array = [None for i in range(args.n_layers)]

        # [subgoal_achieved, total_subgoal] for low-level policies
        self.subgoal_achieved_info = [[0, 0] for i in range(args.n_layers - 1)]

        self.current_state = None

        # Track number of low-level actions executed
        self.steps_taken = 0

        self.total_env_steps = 0

        self.layer_learning_started = [False for i in range(args.n_layers)]
        self.replay_buffer_sizes = [0 for i in range(args.n_layers)]

        # Below hyperparameter specifies number of Q-value updates made after each episode
        self.num_updates = 40

        self.other_params = agent_params


    # Determine whether or not each layer's goal was achieved.  Also, if applicable, return the highest level whose goal was achieved.
    def check_goals(self, env):

        # goal_status is vector showing status of whether a layer's goal has been achieved
        goal_status = [False for i in range(self.args.n_layers)]

        max_lay_achieved = None

        # Project current state onto the subgoal and end goal spaces
        proj_subgoal = env.project_state_to_subgoal(env.sim, self.current_state)
        proj_endgoal = env.project_state_to_endgoal(env.sim, self.current_state)

        for i in range(self.args.n_layers):

            goal_achieved = True

            # If at highest layer, compare to end goal threshold
            if i == self.args.n_layers - 1:
                # Check dimensions are appropriate         
                assert len(proj_endgoal) == len(self.goal_array[i]) == len(env.endgoal_thresholds), "Projected end goal, actual end goal, and end goal thresholds should have same dimensions"

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_endgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_endgoal[j]) > env.endgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If not highest layer, compare to subgoal thresholds
            else:

                # Check that dimensions are appropriate
                assert len(proj_subgoal) == len(self.goal_array[i]) == len(env.subgoal_thresholds), "Projected subgoal, actual subgoal, and subgoal thresholds should have same dimensions"           

                # Check whether layer i's goal was achieved by checking whether projected state is within the goal achievement threshold
                for j in range(len(proj_subgoal)):
                    if np.absolute(self.goal_array[i][j] - proj_subgoal[j]) > env.subgoal_thresholds[j]:
                        goal_achieved = False
                        break

            # If projected state within threshold of goal, mark as achieved
            if goal_achieved:
                goal_status[i] = True
                max_lay_achieved = i
            else:
                goal_status[i] = False
            

        return goal_status, max_lay_achieved


    def initialize_networks(self):

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # If not retraining, restore weights
        # if we are not retraining from scratch, just restore weights
        if self.args.retrain == False:
            print("Load models")
            self.load_model()

    # Save neural network parameters
    def save_model(self):
        for i in range(self.args.n_layers):
            self.layers[i].actor_critic.save(self.log_dir, str(i))

    def load_model(self):
        for i in range(self.args.n_layers):
            self.layers[i].actor_critic.load(self.log_dir, str(i))


    # Update actor and critic networks for each layer
    def learn(self, agent, total_env_steps):

        for i in range(len(self.layers)):   
            self.layers[i].learn(self.num_updates, agent, total_env_steps)

       
    # Train agent for an episode
    def train(self, env, episode_num, total_episodes):

        # Select final goal from final goal space
        self.goal_array[self.args.n_layers - 1] = env.get_next_goal(self.args.test)
        logging.info(f"Next End Goal: {self.goal_array[self.args.n_layers - 1]}")

        # Select initial state from in initial state space
        if self.args.env in ['hac-ant-four-rooms-v0', 'hac-ant-reacher-v0']:
            next_goal = self.goal_array[self.args.n_layers - 1]
        else:
            next_goal = None

        self.current_state = env.reset()
        # print("Initial State: ", self.current_state)

        # Reset step counter
        self.steps_taken = 0

        # Train for an episode
        goal_status, max_lay_achieved = self.layers[self.args.n_layers-1].train(self,env, episode_num = episode_num)

        # Update actor/critic networks if not testing
        if not self.args.test:
            self.learn(self, self.total_env_steps)

        # Return whether end goal was achieved
        return goal_status[self.args.n_layers-1]
