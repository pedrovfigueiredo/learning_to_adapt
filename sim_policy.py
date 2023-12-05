import joblib
import argparse
import os.path as osp
from learning_to_adapt.samplers.utils import rollout
import json
import numpy as np



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--param", type=str, help='Directory with the pkl and json file', default='data/grbal')
    parser.add_argument('--max_path_length', '-l', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', '-n', type=int, default=10,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    parser.add_argument('--ignore_done', action='store_true',
                        help='Whether stop animation when environment done or continue anyway')
    args = parser.parse_args()

    pkl_path = osp.join(args.param)

    print("Testing policy %s" % pkl_path)
    adapt_batch_size = 16
    data = joblib.load(pkl_path)
    policy = data['policy']
    # policy
    policy.dynamics_model.reset()
    env = data['env']
    env.task = 'cripple'
    path = rollout(env, policy, max_path_length=args.max_path_length,
                    animated=False, ignore_done=args.ignore_done,
                    adapt_batch_size=adapt_batch_size, num_rollouts=args.num_rollouts, adapt=True)
    rollout_r = [sum(p['rewards']) for p in path]
    print(rollout_r)
    print(f'avg of rewards over {args.num_rollouts} rollouts: {np.mean(rollout_r)}')
    # print(f'mean of losses for rollout({r}): {sum(path["pred_loss"]) / len(path["pred_loss"])}')
