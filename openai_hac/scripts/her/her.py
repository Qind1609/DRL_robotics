import numpy as np


class HER:
    def __init__(self, replay_k, reward_func=None):

        self.replay_k = replay_k
        self.future_p = 1 - (1.0 / (1 + replay_k))

        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(
            episode_batch["actions"]
        )  # number of episodes in buffer
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        T = np.array([], dtype=np.int32)
        t_samples = np.array([], dtype=np.int32)
        j = 0
        transitions = {key: [] for key in episode_batch.keys()}
        for i in episode_idxs:

            T = np.append(T, len(episode_batch["actions"][i]))
            t_samples = np.append(t_samples, np.random.randint(T[j], size=1))

            for key in episode_batch.keys():
                transitions[key].append(episode_batch[key][i][t_samples[j]].copy())

            j += 1

        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        her_indexes = np.squeeze(her_indexes, axis=0)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        j = 0
        # replace desire goal with achieved goal in her_index
        for i in episode_idxs[her_indexes]:

            future_ag = episode_batch["ag"][i][future_t[j]]
            transitions["g"][her_indexes[j]] = future_ag
            j += 1

        # TODO:
        # re-compute reward for archived goal
        transitions["r"] = np.expand_dims(
            self.reward_func(transitions["ag_next"], transitions["g"]), 1
        )
        transitions = {
            k: np.array(transitions[k]).reshape(
                batch_size, *np.array(transitions[k]).shape[1:]
            )
            for k in transitions.keys()
        }
        return transitions
