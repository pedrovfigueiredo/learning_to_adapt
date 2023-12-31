from learning_to_adapt.policies.base import Policy
from learning_to_adapt.utils.serializable import Serializable
import numpy as np
import gym


class MPCController(Policy, Serializable):
    def __init__(
            self,
            name,
            env,    # normalized env
            dynamics_model,
            reward_model=None,
            discount=1,
            use_cem=False,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            percent_elites=0.1,
            use_reward_model=False,
            alpha=0.1,
    ):
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.use_cem = use_cem
        self.num_cem_iters = num_cem_iters
        self.percent_elites = percent_elites
        self.env = env
        self.use_reward_model = use_reward_model
        self.alpha = alpha

        self.unwrapped_env = self.env

        # while hasattr(self.unwrapped_env, 'wrapped_env'):
        #     self.unwrapped_env = self.unwrapped_env.wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        Serializable.quick_init(self, locals())
        super(MPCController, self).__init__(env=env)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]

        if self.use_cem:
            action = self.get_cem_action(observation)
        else:
            action = self.get_rs_action(observation)

        return action, dict()

    def get_actions(self, observations):
        if self.use_cem:
            actions = self.get_cem_action(observations)
        else:
            actions = self.get_rs_action(observations)

        return actions, dict()

    def get_random_action(self, n):
        return np.array([self.env.action_space.sample() for _ in range(n)], dtype=np.float32)

    # def __reward(self, _, __, next_obs):
    #     # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    #     x, _, theta, _ = next_obs
    #
    #     if bool(
    #         x < -self.unwrapped_env.x_threshold
    #         or x > self.unwrapped_env.x_threshold
    #         or theta < -self.unwrapped_env.theta_threshold_radians
    #         or theta > self.unwrapped_env.theta_threshold_radians
    #     ):
    #         return 1.0
    #
    #     return 0


    def get_cem_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        act_dim = 1

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        mean = np.zeros((m, h * act_dim))
        std = np.ones((m, h * act_dim))
        clip_low = np.array([0] * h)
        clip_high = np.array([1] * h)

        for i in range(self.num_cem_iters):
            z = np.random.normal(size=(n, m,  h * act_dim))
            a = mean + z * std
            a_stacked = np.clip(a, clip_low, clip_high)
            a = a.reshape((n * m, h, act_dim))
            a = np.transpose(a, (1, 0, 2))
            returns = np.zeros((n * m,))

            for t in range(h):
                if t == 0:
                    cand_a = a[t].reshape((m, n, -1))
                    observation = np.repeat(observations, n, axis=0)
                next_observation = self.dynamics_model.predict(observation, a[t])
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation
            returns = returns.reshape(m, n)
            elites_idx = ((-returns).argsort(axis=-1) < num_elites).T
            elites = a_stacked[elites_idx]
            mean = mean * self.alpha + (1 - self.alpha) * np.mean(elites, axis=0)
            std = np.std(elites, axis=0)

        return cand_a[range(m), np.argmax(returns, axis=1)]

    def get_rs_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        returns = np.zeros((n * m,))

        a = self.get_random_action(h * n * m).reshape((h, n * m, -1))
        cand_a = a[0].reshape((m, n, -1))

        for t in range(h):
            if t == 0:
                observation = np.repeat(observations, n, axis=0)
            next_observation = self.dynamics_model.predict(observation, a[t])
            if self.use_reward_model:
                assert self.reward_model is not None
                rewards = self.reward_model.predict(observation, a[t], next_observation)
            else:
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        pass


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=None,
        discount=1,
        n_candidates=500,
        horizon=10,
        use_cem=False,
        num_cem_iters=5,
    )

    print(env.action_space)
    print(env.observation_space)
    print(policy.get_random_action(10))

    print('DONE')