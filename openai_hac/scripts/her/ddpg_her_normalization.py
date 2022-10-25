import torch

import os, sys

import random
import numpy as np
import copy
from datetime import datetime
from mpi4py import MPI
from her.replay_buffer import Replay_Buffer
from her.actor_critic import Actor, Critic
from her.her import HER
from her.normalizer import normalizer
from her.mpi import sync_networks, sync_grads
import rospy
from torch.utils.tensorboard import SummaryWriter


class OUNoise:
    """Ornstein-Uhlenbeck process. The Ornstein-Uhlenbeck process is a stationary Gauss-Markov process
    https://en.wikipedia.org/wiki/Ornstein-Uhlenbeck_process"""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for i in range(len(x))]
        )
        self.state = x + dx
        return self.state


class DDPG_HER_N:
    def __init__(self, params, env, env_params):

        # actor => policy network -> generate action
        # critic => DQN network

        # Create actor_critic pair
        self.env = env
        self.env_params = env_params
        self.params = params

        self.actor = Actor(self.env_params)
        self.critic = Critic(self.env_params)

        # sync networks across the CPUs
        sync_networks(self.actor)
        sync_networks(self.critic)
        # using Adam optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), self.params.lr_actor
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), self.params.lr_critic
        )

        # create the normalizer
        self.o_norm = normalizer(
            size=self.env_params["obs_dim"], default_clip_range=self.params.clip_range
        )
        self.g_norm = normalizer(
            size=self.env_params["goal_dim"], default_clip_range=self.params.clip_range
        )
        
        self.last_epoch = 0
        if MPI.COMM_WORLD.Get_rank() == 0:
            if os.path.exists(
                os.path.join(
                    self.params.save_dir, self.params.env_name, "actor_critic.pt"
                )
            ):
                checkpoint = torch.load(
                    os.path.join(
                        self.params.save_dir, self.params.env_name, "actor_critic.pt"
                    )
                )
                self.actor.load_state_dict(checkpoint["actor_state_dict"])
                self.critic.load_state_dict(checkpoint["critic_state_dict"])

                self.actor.eval()
                self.critic.eval()
                self.last_epoch = checkpoint["epoch"]
                #print(checkpoint['g_mean'])
                self.g_norm.mean = checkpoint["g_mean"]
                self.o_norm.mean = checkpoint["o_mean"]
                self.g_norm.std = checkpoint["g_std"]
                self.o_norm.std = checkpoint["o_std"]
                
        # Create target networks pair which lag (delay) the original networks
        self.actor_target = Actor(self.env_params)
        self.critic_target = Critic(self.env_params)

        # copy(loading) main (weight & bias) params into target network
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.params.cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.actor.to(device)
        self.critic.to(device)
        self.actor_target.to(device)
        self.critic_target.to(device)

        # TODO:
        self.her_module = HER(self.params.replay_k, self.re_compute_reward)

        self.ou_noise = OUNoise(self.env_params["action_dim"], self.params.seed)

        self.buffer = Replay_Buffer(
            self.env_params,
            self.params.buff_size,
            self.her_module.sample_her_transitions,
        )
        if MPI.COMM_WORLD.Get_rank() == 0:
            inpt, next_inpt, act = self._get_example()
            with SummaryWriter(
                "/home/qind/Desktop/catkin_ws/src/openai_hac/scripts/her/HER_approach_actor"
            ) as writer:
                writer.add_graph(self.actor, inpt)
            with SummaryWriter(
                "/home/qind/Desktop/catkin_ws/src/openai_hac/scripts/her/HER_approach_critic"
            ) as writer:
                writer.add_graph(self.critic, (next_inpt, act))
            with SummaryWriter(
                "/home/qind/Desktop/catkin_ws/src/openai_hac/scripts/her/HER_approach_actor_target"
            ) as writer:
                writer.add_graph(self.actor_target, inpt)
            with SummaryWriter(
                "/home/qind/Desktop/catkin_ws/src/openai_hac/scripts/her/HER_approach_critic_target"
            ) as writer:
                writer.add_graph(self.critic_target, (next_inpt, act))
            writer.close()

        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.params.save_dir):
                os.mkdir(self.params.save_dir)

            # path to save the model
            self.model_path = os.path.join(self.params.save_dir, self.params.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def re_compute_reward(self, a, g):
        r = np.array([])
        for i in range(len(a)):
            if self._is_out_range(np.array(a)[i]):
                r = np.append(r, -2)
            else:
                distance = np.linalg.norm(np.array(a)[i] - np.array(g)[i])
                if distance <= self.params.threshold:
                    r = np.append(r, 1)
                else:
                    r = np.append(r, -10 * distance)
        return r

    def _preproc_inputs(self, o, g):
        obs_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)

        inputs = np.concatenate([obs_norm, g_norm])

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.params.cuda:
            inputs = inputs.cuda()
        return inputs

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.params.clip_obs, self.params.clip_obs)
        g = np.clip(g, -self.params.clip_obs, self.params.clip_obs)
        return o, g

    def _update_normalizer(self, episode_transition):
        mb_obs, mb_ag, mb_g, mb_actions = episode_transition  # data of 1 episode
        num_eps = len(mb_obs)
        mb_obs_next = []
        mb_ag_next = []
        for i in range(num_eps):
            mb_obs_next.append(mb_obs[i][1:])
            mb_ag_next.append(mb_ag[i][1:])
        # get the number of normalization transitions
        num_transitions = num_eps

        # create the new buffer to store them
        buffer_temp = {
            "obs": mb_obs,
            "ag": mb_ag,
            "g": mb_g,
            "actions": mb_actions,
            "obs_next": mb_obs_next,
            "ag_next": mb_ag_next,
        }
        transitions = self.her_module.sample_her_transitions(
            buffer_temp, num_transitions
        )
        obs, g = transitions["obs"], transitions["g"]
        # pre process the obs and g
        transitions["obs"], transitions["g"] = self._preproc_og(obs, g)

        # update
        self.o_norm.update(transitions["obs"])
        self.g_norm.update(transitions["g"])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _generate_action_with_noise(self, obs, noise):
        # eval mode
        self.actor.eval()
        action = self.actor(obs)

        action = (
            action.detach().cpu().numpy().squeeze() + noise * self.ou_noise.sample()
        )
        self.actor.train()
        return np.clip(
            action, -self.env_params["action_max"], self.env_params["action_max"]
        )

    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.squeeze()

        # random actions...
        random_actions = np.random.uniform(
            low=-self.env_params["action_max"],
            high=self.env_params["action_max"],
            size=self.env_params["action_dim"],
        )
        # choose if use the random actions
        action += np.random.binomial(1, self.params.random_eps, 1)[0] * (
            random_actions - action
        )
        return action

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.params.polyak) * param.data
                + self.params.polyak * target_param.data
            )

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.params.batch_size)
        # pre-process the observation and goal
        o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
        transitions["obs"], transitions["g"] = self._preproc_og(o, g)
        transitions["obs_next"], transitions["g_next"] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions["obs"])
        g_norm = self.g_norm.normalize(transitions["g"])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions["obs_next"])
        g_next_norm = self.g_norm.normalize(transitions["g_next"])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32)  # reward tensor
        if self.params.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target(inputs_next_norm_tensor)
            q_next_value = self.critic_target(inputs_next_norm_tensor, actions_next)
            # q_next_value = q_next_value.detach()

            # Bellman equation (the estimation of optimal Q value)
            target_q_value = r_tensor + self.params.gamma * q_next_value
            # target_q_value = target_q_value.detach()

            # clip the q value
            clip_return = 1 / (1 - self.params.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic(inputs_norm_tensor, actions_tensor)

        # MS Bellman Error
        critic_loss = (target_q_value - real_q_value).pow(2).mean()  # MSE loss

        # the actor loss
        actions_real = self.actor(inputs_norm_tensor)
        actor_loss = -self.critic(
            inputs_norm_tensor, actions_real
        ).mean()  # gradient ascent

        actor_loss += (
            self.params.action_l2
            * (actions_real / self.env_params["action_max"]).pow(2).mean()
        )

        # update the actor network
        self.actor.train()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor)
        self.actor_optimizer.step()

        # update the critic_network
        self.critic.train()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic)
        self.critic_optimizer.step()

    def _is_out_range(self, cur_pos):
        if not (
            (self.params.position_x_min <= cur_pos[0] <= self.params.position_x_max)
            and (self.params.position_y_min <= cur_pos[1] <= self.params.position_y_max)
            and (self.params.position_z_min <= cur_pos[2] <= self.params.position_z_max)
        ):
            rospy.logwarn("Out of range")
            return True
        else:
            return False

    def _get_example(self):
        # sample the episodes
        transitions = {
            "obs": np.array([[5, 3, 6, 0.3, 2, 3, 5]]),
            "ag": np.array([[0.3, 0.2, 0.1]]),
            "g": np.array([[0.2, 0.2, 0.1]]),
            "actions": np.array([[0.1, 0.1, 0.1]]),
            "obs_next": np.array([[1, 2, 3, 0.1, 3, 5, 9]]),
            "ag_next": np.array([[0.2, 0.2, 0.3]]),
            "r": np.array([[0.2, 0.6]]),
        }
        # pre-process the observation and goal
        o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
        transitions["obs"], transitions["g"] = self._preproc_og(o, g)
        transitions["obs_next"], transitions["g_next"] = self._preproc_og(o_next, g)
        # start to do the update
        obs_norm = self.o_norm.normalize(transitions["obs"])
        g_norm = self.g_norm.normalize(transitions["g"])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions["obs_next"])
        g_next_norm = self.g_norm.normalize(transitions["g_next"])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions["actions"], dtype=torch.float32)
        r_tensor = torch.tensor(transitions["r"], dtype=torch.float32)  # reward tensor
        if self.params.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        return inputs_norm_tensor, inputs_next_norm_tensor, actions_tensor

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        cumulate_test_reward = []

        for t in range(self.params.test_episodes):
            rospy.logwarn("Test ep {}".format(t))
            per_success_rate = []
            observation = self.env.reset()
            obs = observation["observation"]
            g = observation["desired_goal"]
            cumulate_episode_reward = 0
            done = False
            ep = 0
            while not (done or (ep == self.params.max_ep_step)):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    self.actor.eval()
                    pi = self.actor(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                observation_new, reward, done, info = self.env.step(actions)

                obs = observation_new["observation"]
                g = observation_new["desired_goal"]
                cumulate_episode_reward += reward
                ep += 1
                per_success_rate.append(info["is_success"])
                total = sum(per_success_rate)
            total_success_rate.append(total)  # store result of episode
            cumulate_test_reward.append(cumulate_episode_reward / ep)
        cumulate_reward_avg = np.mean(np.array(cumulate_test_reward))
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)
        return cumulate_reward_avg, local_success_rate

    def train(self):

        with SummaryWriter(
            "/home/qind/Desktop/catkin_ws/src/openai_hac/scripts/her/HER_training_result"
        ) as writer:
            for epoch in range(self.last_epoch, self.params.num_epochs):
                rospy.logwarn("==================== Start Epoch {}".format(epoch + 1))
                for eps in range(self.params.num_episodes):
                    mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                    mb_r = []
                    rospy.logwarn("============== EP: {}".format(eps))
                    for roll in range(
                        self.params.num_rollouts_per_mpi
                    ):  # load data to each mpi

                        ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                        observation = self.env.reset()
                        obs = observation["observation"]
                        ag = observation["achieved_goal"]
                        g = observation["desired_goal"]
                        done, ep_r, ep_len = False, [], 0

                        for t in range(
                            self.params.max_ep_step
                        ):  # each step generates 1 action ->collect example data

                            # feed forward actor Network
                            with torch.no_grad():
                                inputs = self._preproc_inputs(obs, g)
                                action = self._generate_action_with_noise(
                                    inputs, self.params.noise_eps
                                )
                                action = self._select_actions(
                                    action
                                )  # random action or action with noise

                                # feed action into the environment and get feedback
                            rospy.logwarn(
                                "============== Step: {}, Rollout: {}".format(t, roll)
                            )
                            obs_nextt, reward, done, _ = self.env.step(action)
                            ep_r.append(reward)
                            ep_len += 1
                            done = False if ep_len == self.params.max_ep_step else done
                            obs_next = obs_nextt["observation"]
                            ag_next = obs_nextt["achieved_goal"]
                            # append rollout
                            ep_obs.append(obs.copy())  # track back of all steps
                            ep_ag.append(ag.copy())
                            ep_g.append(g.copy())
                            ep_actions.append(action.copy())

                            # re-assign the observation
                            obs = obs_next
                            ag = ag_next
                            check_range = self._is_out_range(obs[:3])
                            if (
                                (done and ep_len >= self.params.batch_size)
                                or (ep_len == self.params.max_ep_step)
                                or check_range
                            ):
                                rospy.logwarn(obs[:3])
                                break

                            # append (store) the last step
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())

                        mb_obs.append(ep_obs)
                        mb_ag.append(ep_ag)
                        mb_g.append(ep_g)
                        mb_actions.append(ep_actions)
                        mb_r.append(ep_r)
                    """mb_obs = np.array(mb_obs, dtype=object)
                    mb_ag = np.array(mb_ag, dtype=object)
                    mb_g = np.array(mb_g, dtype=object)
                    mb_actions = np.array(mb_actions, dtype=object)
                    mb_r = np.array(mb_r, dtype=object)"""
                    # store the episode in buffer
                    self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                    # update normalizer because we have new data (new episode)
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])

                    # train model after every episode
                    for _ in range(self.params.num_batches):
                        # train networks (Actor and Critic)
                        self._update_network()

                    # soft update target networks
                    self._soft_update_target_network(self.actor_target, self.actor)
                    self._soft_update_target_network(self.critic_target, self.critic)

                # evaluation every time done 1 epoch
                cumulate_reward, success_rate = self._eval_agent()
                
                
                rospy.logwarn("==================== Done Epoch {}".format(epoch + 1))
                writer.add_scalar(
                    "Distance(mm) mean error", cumulate_reward * 1000, epoch + 1
                )
                writer.add_scalar("Success rate", success_rate, epoch + 1)
                if MPI.COMM_WORLD.Get_rank() == 0:

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "actor_state_dict": self.actor.state_dict(),
                            "success_rate": success_rate,
                            "Avg_distance": cumulate_reward,
                            "critic_state_dict": self.critic.state_dict(),
                            "g_mean": self.g_norm.mean,
                            "g_std": self.g_norm.std,
                            "o_mean": self.o_norm.mean,
                            "o_std": self.o_norm.std,
                        },
                        self.model_path + "/actor_critic.pt",
                    )


class Test_DDPG_HER:
    def __init__(self, params, env, env_params) -> None:
    

        self.env = env
        self.env_params = env_params
        self.params = params

        self.actor = Actor(self.env_params)
        # using Adam optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), self.params.lr_actor
        )
        # create the normalizer
        self.o_norm = normalizer(
            size=self.env_params["obs_dim"], default_clip_range=self.params.clip_range
        )
        self.g_norm = normalizer(
            size=self.env_params["goal_dim"], default_clip_range=self.params.clip_range
        )

        if MPI.COMM_WORLD.Get_rank() == 0:
            if os.path.exists(
                os.path.join(
                    self.params.save_dir, self.params.env_name, "actor_critic.pt"
                )
            ):
                checkpoint = torch.load(
                    os.path.join(
                        self.params.save_dir, self.params.env_name, "actor_critic.pt"
                    )
                )
                self.actor.load_state_dict(checkpoint["actor_state_dict"])

                self.actor.eval()

                self.g_norm.mean = checkpoint['g_mean']
                self.o_norm.mean = checkpoint['o_mean']
                self.g_norm.std = checkpoint['g_std']
                self.o_norm.std = checkpoint['o_std']

        if self.params.cuda and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.actor.to(device)
  
        

    def _preproc_inputs(self, o, g):
        obs_norm = self.o_norm.normalize(o)
        g_norm = self.g_norm.normalize(g)

        inputs = np.concatenate([obs_norm, g_norm])

        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.params.cuda:
            inputs = inputs.cuda()
        return inputs

    def test(self):

        # test start
        total_success_rate = []
        cumulate_test_reward = []

        for t in range(self.params.test_episodes):
            rospy.logwarn("Test ep {}".format(t))
            per_success_rate = []
            observation = self.env.reset()
            obs = observation["observation"]
            g = observation["desired_goal"]
            cumulate_episode_reward = 0
            done = False
            ep = 0
            while not (done or (ep == self.params.max_ep_step)):
                with torch.no_grad():
                    input_tensor = self._preproc_inputs(obs, g)
                    self.actor.eval()
                    pi = self.actor(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()

                observation_new, reward, done, info = self.env.step(actions)

                obs = observation_new["observation"]
                g = observation_new["desired_goal"]
                cumulate_episode_reward += reward
                ep += 1
                per_success_rate.append(info["is_success"])
                total = sum(per_success_rate)   #0 or 1
            total_success_rate.append(total)  # store result of episode
            cumulate_test_reward.append(cumulate_episode_reward / ep)
                #need some meansurements for calculating more efficiently 

        cumulate_reward_avg = np.mean(np.array(cumulate_test_reward))
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate)

        print("Distance(mm) mean: {0} ".format(cumulate_reward_avg * 1000))
        print("Success rate avg {0}".format(local_success_rate))

        