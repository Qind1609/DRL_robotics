import numpy as np
from her import HER
from replay_buffer import Replay_Buffer
from normalizer import normalizer


def test_buffer():

    mb_obs = [
        [[1, 2, 3, 0.3], [3, 5, 6, 0.6], [5, 6, 9, 0.9], [2, 5, 6, 0.5]],
        [
            [2, 5, 4, 0.7],
            [5, 3, 6, 0.6],
            [1, 3, 5, 0.6],
            [3, 5, 8, 0.2],
            [1, 2, 3, 0.5],
        ],
        [[2, 5, 6, 0.3], [4, 5, 8, 0.5], [5, 3, 6, 0.8]],
    ]
    mb_ag = [
        [[0.3, 0.5, 0.6], [0.7, 0.5, 0.3], [0.5, 0.8, 0.9], [0.6, 0.3, 0.2]],
        [
            [0.3, 0.5, 0.6],
            [0.3, 0.2, 0.1],
            [0.2, 0.1, 0.3],
            [0.3, 0.2, 0.1],
            [0.3, 0.2, 0.1],
        ],
        [[0.1, 0.1, 0.1], [0.2, 0.3, 0.4], [0.1, 0.1, 0.1]],
    ]
    mb_g = [
        [[2, 3, 4], [2, 3, 4], [2, 3, 4]],
        [[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]],
        [[4, 5, 6], [4, 5, 6]],
    ]
    mb_actions = [
        [[0, 1, 0.1, 0.1], [-0.1, -0.1, 0.2], [0.3, 0.2, -0.2]],
        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
        [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
    ]
    """mb_obs = np.array(mb_obs, dtype=object)
    mb_ag = np.array(mb_ag, dtype=object)
    mb_g = np.array(mb_g, dtype=object)
    mb_actions = np.array(mb_actions, dtype=object)"""
    her = HER(2, compute_reward)

    env_params = {
        "obs_dim": 4,  # obs_dim (4,)
        "goal_dim": 3,  # goal_dim (3,)
        "action_dim": 3,  # action_dim (3,)
        "action_max": 0.01,  # high: [0.01, 0.01, 0.01] low:[-0.01, -0.01, -0.01]
        "max_timesteps": 1000,  # max_step for each ep
        "buff_size": 1000000,
    }
    buffer = Replay_Buffer(
        env_params,
        env_params["buff_size"],
        her.sample_her_transitions,
    )
    o_norm = normalizer(size=env_params["obs_dim"], default_clip_range=5)
    g_norm = normalizer(size=env_params["goal_dim"], default_clip_range=5)
    buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
    update_normalizer([mb_obs, mb_ag, mb_g, mb_actions], her, o_norm, g_norm)

    transitions = buffer.sample(2)
    print(transitions)
    o, o_next, g = transitions["obs"], transitions["obs_next"], transitions["g"]
    transitions["obs"], transitions["g"] = preproc_og(o, g)
    transitions["obs_next"], transitions["g_next"] = preproc_og(o_next, g)
    obs_norm1 = o_norm.normalize(transitions["obs"])
    g_norm1 = g_norm.normalize(transitions["g"])
    inputs_norm = np.concatenate([obs_norm1, g_norm1], axis=1)
    print(inputs_norm)
    obs_next_norm = o_norm.normalize(transitions["obs_next"])
    g_next_norm = g_norm.normalize(transitions["g_next"])
    inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
    print(inputs_next_norm)


def compute_reward(a, b):
    r = np.array([])
    for i in range(len(a)):
        r = np.append(r, np.linalg.norm(np.array(a)[i] - np.array(b)[i]))
    return r


def update_normalizer(episode_transition, her, o_norm, g_norm):
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
    transitions = her.sample_her_transitions(buffer_temp, num_transitions)
    obs, g = transitions["obs"], transitions["g"]
    # pre process the obs and g
    transitions["obs"], transitions["g"] = preproc_og(obs, g)

    # update
    o_norm.update(transitions["obs"])
    g_norm.update(transitions["g"])
    # recompute the stats
    o_norm.recompute_stats()
    g_norm.recompute_stats()


def preproc_og(o, g):
    o = np.clip(o, -200, 200)
    g = np.clip(g, -200, 200)
    return o, g


if __name__ == "__main__":
    test_buffer()
