import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scripts.hac.utils import init_weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, 
        env, 
        batch_size, 
        layer_number, 
        args, 
        is_top_layer): 
        super(Actor, self).__init__()

        self.actor_name = 'actor_' + str(layer_number)

        self.is_top_layer = is_top_layer

        # Determine range of actor network outputs.  This will be used to configure outer layer of neural network
        if layer_number == 0:
            self.action_space_bounds = env.action_bounds
            self.action_offset = env.action_offset
        else:
            # Determine symmetric range of subgoal space and offset
            self.action_space_bounds = env.subgoal_bounds_symmetric
            self.action_offset = env.subgoal_bounds_offset        

        # Dimensions of action will depend on layer level
        if layer_number == 0:
            self.action_space_size = env.action_dim
        else:
            self.action_space_size = env.subgoal_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if not self.is_top_layer:
            self.goal_dim = env.subgoal_dim
        else:
            self.goal_dim = env.endgoal_dim

        self.state_dim = env.state_dim

        # actor
        self.actor = nn.Sequential(
                            nn.Linear(self.state_dim + self.goal_dim, 64),
                            nn.ReLU(),
                            nn.Linear(64, 64),
                            nn.ReLU(),
                            nn.Linear(64, self.action_space_size),
                            nn.Tanh()
                            )

        self.actor.apply(init_weights)

        self.action_space_bounds = torch.FloatTensor(self.action_space_bounds).to(device)
        self.action_offset = torch.FloatTensor(self.action_offset).to(device)
        
    def forward(self, state, goal):
        return self.actor(torch.cat([state, goal], 1)) * self.action_space_bounds + self.action_offset
        
class Critic(nn.Module):
    def __init__(self, 
        env, 
        layer_number, 
        args, 
        is_top_layer):

        super(Critic, self).__init__()

        self.critic_name = 'critic_' + str(layer_number)

        self.q_limit = -args.time_scale

        self.is_top_layer = is_top_layer

        # Dimensions of action placeholder will differ depending on layer level
        if layer_number == 0:
            self.action_dim = env.action_dim
        else:
            self.action_dim = env.subgoal_dim

        # Dimensions of goal placeholder will differ depending on layer level
        if not self.is_top_layer:
            self.goal_dim = env.subgoal_dim
        else:
            self.goal_dim = env.endgoal_dim

        self.state_dim = env.state_dim

        # Set parameters to give critic optimistic initialization near q_init
        self.q_init = -0.067
        self.q_offset = -np.log(self.q_limit/self.q_init - 1)

        # UVFA critic
        self.critic = nn.Sequential(
                        nn.Linear(self.state_dim + self.goal_dim + self.action_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        )            

        self.critic.apply(init_weights)

    def forward(self, state, action, goal):
        # rewards are in range [-H, 0]
        input_ = torch.cat([state, action, goal], 1)
        input_ = self.critic(input_)
        return torch.sigmoid(input_ + self.q_offset) * self.q_limit
    
class DDPG:
    def __init__(self, env, batch_size, layer_number, args, lr=0.001, gamma=0.95):

        self.is_top_layer = (layer_number == args.n_layers - 1)

        self.actor = Actor(env, batch_size, layer_number, args, self.is_top_layer).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic = Critic(env, layer_number, args, self.is_top_layer).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.q_limit = -args.time_scale
        
        self.mseLoss = torch.nn.MSELoss()

        self.gamma = gamma

        
    
    def select_action(self, state, goal):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        return self.actor(state, goal).detach().cpu().data.numpy().flatten()
    
    def update(self, buffer):
        
        # Sample a batch of transitions from replay buffer:
        state, action, reward, next_state, goal, done = buffer.get_batch()
        goal = torch.FloatTensor(goal).to(device)
        
        # convert np arrays into tensors
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).reshape((-1,1)).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape((-1,1)).to(device)
        
        # select next action
        next_action = self.actor(next_state, goal).detach()
        target_Q = self.critic(next_state, next_action, goal).detach()
        
        # Compute target Q-value:
        target_Q = reward + ((1-done) * self.gamma * target_Q)
        target_Q = torch.clamp(target_Q, self.q_limit, 0.0)

        # Optimize Critic:
        critic_loss = self.mseLoss(self.critic(state, action, goal), target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Compute actor loss:
        actor_loss = -self.critic(state, self.actor(state, goal), goal).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
                
    def save(self, directory, name):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, name))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, name))
        
    def load(self, directory, name):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, name), map_location='cpu'))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, name), map_location='cpu'))  
        
        
        
        
      
        
        
