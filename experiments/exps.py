from stable_baselines3 import A2C, DQN, PPO
# import gym
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch

ENV = 'CartPole-v1'
ALGO = 'a2c'
SAVED_MODEL_PATH = f'../ckpts/{ALGO}_trained_model.zip'
saved_video_count = 0
TRAIN = True
ROLLOUTS = 20



def get_env():
    return gym.make(ENV, render_mode='rgb_array')

def run_episode(env, model):
    total_rewards = 0
    obs, _ = env.reset()
    done = False
    steps = 0
    max_steps = 1000

    while not done and steps < max_steps:
        action, _ = model.predict(obs)

        obs, reward, done, _, info = env.step(action)

        total_rewards += reward
        env.render()
        steps += 1

    return total_rewards

def save_video_with_cur_model(env, model, folder):
    assert model is not None
    global saved_video_count

    episode_trigger = lambda t: t % 2 == 0
    env = RecordVideo(env=env, video_folder=folder, step_trigger=episode_trigger, name_prefix=f'{ALGO}-{saved_video_count}')

    rewards_per_episode = run_episode(env, model)

    env.close_video_recorder()

    print(f'total rewards for one episode: {rewards_per_episode}')
    saved_video_count += 1

def get_algo_class():
    if ALGO == 'a2c':
        return A2C
    elif ALGO == 'dqn':
        return DQN
    elif ALGO == 'ppo':
        return PPO

    return None

def update_env_parameters(env):
    unwrapped = env.unwrapped
    unwrapped.total_mass = unwrapped.masspole + unwrapped.masscart
    unwrapped.polemass_length = unwrapped.masspole * unwrapped.length

if __name__ == '__main__':
    """
    default parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
    """

    env = get_env()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if TRAIN:
        # init
        model = get_algo_class()('MlpPolicy', env, verbose=1, device=device)
        # modify this with the algorithm

        model.learn(total_timesteps=1000000)

        # save trained model
        model.save(SAVED_MODEL_PATH)

    # load saved model
    model = get_algo_class().load(SAVED_MODEL_PATH, env=env, verbose=1, device=device)

    print('#############################\n BASE ENV\n#############################')
    save_video_with_cur_model(env, model, 'base')
    rewards = [run_episode(env, model) for _ in range(ROLLOUTS)]
    print(f'average rewards/episode: {sum(rewards)/len(rewards)}')

    # env.unwrapped.length = 1.0
    # update_env_parameters(env)
    #
    # print('#############################\n PERTURBED ENV BEFORE ADAPT\n#############################')
    # save_video_with_cur_model(env, model, 'pre-adapt')
    # rewards = [run_episode(env, model) for _ in range(ROLLOUTS)]
    # print(f'average rewards/episode: {sum(rewards) / len(rewards)}')
    #
    # # ADAPT
    # # model.learn(total_timesteps=500000)
    # for step in range(20):
    #     model.learn(total_timesteps=1)
    #     print(f'#############################\n PERTURBED ENV AFTER ADAPT - {step+1}\n#############################')
    #     rewards = [run_episode(env, model) for _ in range(ROLLOUTS)]
    #     print(f'average rewards/episode: {sum(rewards) / len(rewards)}')
    #     save_video_with_cur_model(env, model, 'post-adapt')