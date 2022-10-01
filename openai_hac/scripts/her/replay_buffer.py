import threading
import numpy as np


class Replay_Buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params["max_timesteps"]
        self.size = buffer_size // self.T  # maximum number of episode can be stored
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {
            "obs": np.empty(
                [
                    self.size,
                ],
                dtype=object,
            ),
            "ag": np.empty(
                [
                    self.size,
                ],
                dtype=object,
            ),
            "g": np.empty(
                [
                    self.size,
                ],
                dtype=object,
            ),
            "actions": np.empty(
                [
                    self.size,
                ],
                dtype=object,
            ),
        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = len(mb_obs)  # number of episode
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            self.buffers["obs"][idxs] = mb_obs

            self.buffers["ag"][idxs] = mb_ag

            self.buffers["g"][idxs] = mb_g

            self.buffers["actions"][idxs] = mb_actions

            # self.n_transitions_stored += mb_obs.shape[1] * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][: self.current_size]
        temp_buffers["obs_next"] = []
        temp_buffers["ag_next"] = []
        for i in range(self.current_size):
            a = temp_buffers["obs"][i][1:]
            b = temp_buffers["ag"][i][1:]
            temp_buffers["obs_next"].append(a)
            temp_buffers["ag_next"].append(b)
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
