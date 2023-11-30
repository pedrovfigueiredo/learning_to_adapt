from collections import defaultdict

import numpy as np

import gym
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor


class Options:
    def __init__(self, steps, gamma, alpha, epsilon):
        self.steps = steps
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon



class QLearning:
    def __init__(self, env, options: Options):
        assert str(env.action_space).startswith("Discrete") or str(
            env.action_space
        ).startswith("Tuple(Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.options = options
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))
        self.env = env

    def train_episode(self):
        """
        Run a single episode of the Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy
        while following an epsilon-greedy policy

        Use:
            self.env: OpenAI environment.
            self.options.steps: steps per episode
            self.epsilon_greedy_action(state): returns an epsilon greedy action
            np.argmax(self.Q[next_state]): action with highest q value
            self.options.gamma: Gamma discount factor.
            self.Q[state][action]: q value for ('state', 'action')
            self.options.alpha: TD learning rate.
            next_state, reward, done, _ = self.step(action): advance one step in the environment
        """

        # Reset the environment
        state, _ = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        for _ in range(self.options.steps):
            action = np.argmax(self.epsilon_greedy_action(state))
            nextState, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.Q[state][action] += self.options.alpha * (
                reward
                + self.options.gamma * np.max(self.Q[nextState])
                - self.Q[state][action]
            )

            state = nextState

            if done:
                break

    def __str__(self):
        return "Q-Learning"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.


        Returns:
            A function that takes an observation as input and returns a greedy
            action.
        """

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            qValues = self.Q[state]
            return np.argmax(qValues)

        return policy_fn

    def select_action(self, state) -> int:
        return self.create_greedy_policy()(state)

    def epsilon_greedy_action(self, state):
        """
        Return an epsilon-greedy action based on the current Q-values and
        epsilon.

        Use:
            self.env.action_space.n: size of the action space
            np.argmax(self.Q[state]): action with highest q value
        Returns:
            Probability of taking actions as a vector where each entry is the probability of taking that action
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        A = []

        aStar = self.create_greedy_policy()(state)
        e = self.options.epsilon
        nA = self.env.action_space.n
        for a in range(nA):
            if a == aStar:
                A.append(1 - e + e / nA)
            else:
                A.append(e / nA)

        return A

class ApproxQLearning(QLearning):
    # MountainCar-v0
    def __init__(self, env, options):
        self.estimator = Estimator(env)
        super().__init__(env, options)

    def train_episode(self):
        """
        Run a single episode of the approximated Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while following an epsilon-greedy policy

        Use:
            self.env: OpenAI environment.
            self.options.steps: steps per episode
            self.options.gamma: Gamma discount factor.
            self.estimator: The Q-function approximator
            self.estimator.predict(s,a): Returns the predicted q value for a given s,a pair
            self.estimator.update(s,a,y): Trains the estimator towards Q(s,a)=y
            next_state, reward, done, _ = self.step(action): To advance one step in the environment
        """

        # Reset the environment
        state, _ = self.env.reset()

        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        for t in range(self.options.steps):
            action = np.argmax(self.epsilon_greedy(state))
            nextState, reward, done, _ = self.step(action)
            target = reward + self.options.gamma * np.max(
                self.estimator.predict(nextState)
            )

            self.estimator.update(state, action, target)
            state = nextState

            if done:
                break

    def __str__(self):
        return "Approx Q-Learning"

    def epsilon_greedy(self, state):
        """
        Return an epsilon-greedy action based on the current Q-values and
        epsilon.

        Use:
            self.env.action_space.n: size of the action space
            np.argmax(self.Q[state]): action with highest q value
        Returns:
            Probability of taking actions as a vector where each entry is the probability of taking that action
        """
        ################################
        #   YOUR IMPLEMENTATION HERE   #
        ################################
        A = []
        aStar = self.create_greedy_policy()(state)
        e = self.options.epsilon
        nA = self.env.action_space.n
        for a in range(nA):
            if a == aStar:
                A.append(1 - e + e / nA)
            else:
                A.append(e / nA)

        return A

    def select_action(self, state) -> int:
        return self.create_greedy_policy()(state)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values from self.estimator.predict(s,a=None)

        Returns:
            A function that takes a state as input and returns a greedy
            action.
        """
        nA = self.env.action_space.n

        def policy_fn(state):
            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            action_values = self.estimator.predict(state)
            return np.argmax(action_values)

        return policy_fn

    # def plot(self, stats, smoothing_window=20, final=False):
    #     plotting.plot_episode_stats(stats, smoothing_window, final=final)
    #
    # def plot_q_function(self):
    #     plotting.plot_cost_to_go_mountain_car(self.env, self.estimator)


class Estimator:
    """
    Value Function approximator. Don't change!
    """

    def __init__(self, env):
        # Feature Preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for x in range(10000)]
        )
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion(
            [
                ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))
        # We create a separate model for each action in the environment's
        # action space. Alternatively we could somehow encode the action
        # into the features, but this way it's easier to code up.
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # We need to call partial_fit once to initialize the model
            # or we get a NotFittedError when trying to make a prediction
            # This is quite hacky.
            state, _ = env.reset()
            model.partial_fit([self.featurize_state(state)], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Makes value function predictions.

        Args:
            s: state to make a prediction for
            a: (Optional) action to make a prediction for

        Returns
            If an action a is given this returns a single number as the prediction.
            If no action is given this returns a vector or predictions for all actions
            in the environment where pred[i] is the prediction for action i.

        """
        features = self.featurize_state(s)
        if a is None:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])




def train_q_learning_agent(qLearner, num_episodes=500):
    for episode in range(num_episodes):
        print(f'running episode {episode}')

        for _ in range(num_episodes):
            qLearner.train_episode()


def test_q_learning_agent(qLearner, num_episodes: int = 5):
    env = qLearner.env
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = qLearner.select_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()

        print(f"Test Episode: {episode + 1}, Total Reward: {total_reward}")



if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # config
    steps = 1000
    gamma = 0.99
    alpha = 0.001
    epsilon = 0.1

    options = Options(steps, gamma, alpha, epsilon)


    qLearner = ApproxQLearning(env, options)

    train_q_learning_agent(qLearner, 500)
    test_q_learning_agent(qLearner, 5)

    env.close()