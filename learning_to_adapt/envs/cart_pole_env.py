import numpy as np
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.logger import logger
import gym


class CartPoleEnv(Serializable):
    def __init__(self, task='cripple', reset_every_episode=False):
        Serializable.quick_init(self, locals())
        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self.first = True

        task = None if task == 'None' else task

        assert task in [None, 'cripple']

        # force changes
        # self.parameter_list = [10.0, 20.0, 50.0, 100.0]
        # length changes
        self.parameter_list = [0.5, 0.8, 1.0, 1.5, 2.0]
        self.chosen_parameter = 0

        self.task = task
        # self.current_obs = None
        self.env = gym.make('CartPole-v1')
        self.update_env()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def update_env(self):
        self.env.unwrapped.length = self.parameter_list[self.chosen_parameter]
        self.env.unwrapped.polemass_length = self.env.unwrapped.length * self.env.unwrapped.masspole

    # def get_current_obs(self):
    #     return self.current_obs

    def step(self, action):
        # when the action is provided as len 1 vector
        if isinstance(action, np.ndarray):
            action = action[0]

        # action must be integer as action_space is Discrete
        action = int(action)
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        # return false by default as the authors do the similar
        return next_obs, reward, False, info

    def reward(self, obs, action, next_obs) -> float:
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
        x, _, theta, _ = np.split(next_obs, next_obs.shape[1], axis=1)

        x_th_cond = np.logical_or(x < -self.env.x_threshold, x > self.env.x_threshold)
        theta_th_cond = np.logical_or(theta < -self.env.theta_threshold_radians, theta > self.env.theta_threshold_radians)
        combined_cond = np.logical_not(np.logical_or(x_th_cond, theta_th_cond))

        rewards = np.zeros_like(x)
        rewards[combined_cond] = 1.0

        return np.squeeze(rewards)

    def reset(self):
        if self.reset_every_episode and not self.first:
            self.reset_task()

        if self.first:
            self.first = False

        state, _ = self.env.reset()
        # return self.env.reset()

        return state

    def reset_task(self, value=None):
        if self.task == 'cripple':
            self.chosen_parameter = value if value is not None else np.random.randint(len(self.parameter_list))
            self.update_env()
        elif self.task is None:
            pass

        else:
            raise NotImplementedError

    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
        logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
        logger.logkv(prefix + 'StdForwardProgress', np.std(progs))


# if __name__ == '__main__':
#     from itertools import count
#
#     env = CartPoleEnv(task='cripple')
#
#     for iter in count():
#         print(f'running iter: {iter}')
#         env.reset()
#         env.reset_task()
#         for _ in range(1000):
#             obs = env.step(np.array([env.action_space.sample()]))
#
#             if obs[2]:
#                 print(obs)
#             # print(env.reward(None, None, obs))
#             env.render()
#
#             # if done:
#             #     break
