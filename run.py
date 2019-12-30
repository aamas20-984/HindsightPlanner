import os
import sys
import time
import datetime
import numpy as np
import gym
from collections import defaultdict
import tensorflow as tf
import common.tf_util as U
from hindsight_planner import learn
import multiprocessing
from cmd_utils import make_vec_env, make_env
from common.vec_env import VecFrameStack, VecEnv
import logger


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

_game_envs = defaultdict(set)

def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = 'her'
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        U.get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        # if env_type == 'mujoco':
        #     env = VecNormalize(env)

    return env


def get_env_type(args):
    env_id = args.env

    # if args.env_type is not None:
    #     return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types {}'.format(env_id, _game_envs.keys())

    return env_type, env_id


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='FetchReach-v1')
    parser.add_argument('--num_env', help='Number of environment being run in parallel', type=int, default=1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_timesteps', '-nts', type=float, default=1e6)
    parser.add_argument('--hid_size', '-hsz', type=int)
    parser.add_argument('--pln_batch_size', '-bsz', type=int)
    parser.add_argument('--seq_len', '-sl', type=int)
    parser.add_argument('--replay_k', '-rk', type=int)
    parser.add_argument('--sg_smp', type=str, default='time', choices=['time','route'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--psi', type=int, default=5)
    parser.add_argument('--orient', type=int, default=0)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--render',default=False, action='store_true')
    parser.add_argument('--load_path',help='Path to load trained model from', default=None, type=str)
    parser.add_argument('--save_path',help='Path to save trained model to', default=None, type=str)
    parser.add_argument('--play', default=False, action='store_true')
    args = parser.parse_args()

    num_timesteps = int(args.num_timesteps)
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    os.makedirs(data_path, exist_ok=True)
    if args.sg_smp == 'time':
        logdir_prefix = 'hplanner_'
    else:
        logdir_prefix = 'hplanner2_'
    if num_timesteps == 0 and args.play:
        logdir = logdir_prefix + args.env + '_TEST_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        logdir = logdir_prefix + args.env + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(dir=logdir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        logger.configure(dir=logdir, format_strs=[])

    override_param = {}
    for attr in ['lr','hid_size', 'pln_batch_size', 'seq_len', 'replay_k']:
        if args.__getattribute__(attr) is not None:
            override_param[attr] = args.__getattribute__(attr)
    override_param['subgoal_strategy'] = args.sg_smp

    env = build_env(args)
    pi, planner = learn(
            env=env,
            total_timesteps=num_timesteps,
            seed=args.seed,
            policy_save_interval=args.psi,
            override_params=override_param,
            load_path=args.load_path,
            save_path=args.save_path
            )
    
    # model files are save in the training process, so here has no need to do

    env.close()

    if args.play:
        succ = []
        try:
            env = gym.make(args.env)
            obs = env.reset()

            for tt in range(500):
                succ_epi = []
                done = False
                obs = env.reset()
                subgoals = planner.plan(obs['achieved_goal'], obs['desired_goal']).reshape(-1, *obs['desired_goal'].shape)
                if args.env.startswith("Hand"):
                    subgoals[..., 3:] /= np.linalg.norm(subgoals[..., 3:], axis=-1, keepdims=True)
                env.set_subgoal(pos=subgoals[:-1])
                cur_subgo_idx = 0
                cur_goal = subgoals[cur_subgo_idx]
                episode_rew = 0
                info = {"running" : True}
                while not done:
                    info['is_subgoal'] = (cur_subgo_idx+1 < len(subgoals))
                    env_rew = env.compute_reward(obs['achieved_goal'], cur_goal, info)
                    succ_subgoal = (np.abs(env_rew) < 1e-5).astype(np.int32)
                    if succ_subgoal > 0 and cur_subgo_idx < subgoals.shape[0]-1:
                        env.set_subgoal(grey=cur_subgo_idx)
                    cur_subgo_idx = np.min([cur_subgo_idx+succ_subgoal, subgoals.shape[0]-1])
                    cur_goal = subgoals[cur_subgo_idx]
                    obs['desired_goal'] = cur_goal
                    action, _, _, _ = pi.step(obs)

                    obs, rew, done, res_info = env.step(action)
                    episode_rew += rew
                    succ_epi.append(res_info['is_success'])
                    env.render()
                    done = done.any() if isinstance(done, np.ndarray) else done
                    if done:
                        succ_res = np.array(succ_epi)[-3:]
                        succ_res = np.ceil(np.mean(succ_res) - 0.5)
                        print('Done ! episode result=', succ_res)
                        succ.append(succ_res)
                        succ_epi.clear()
            logger.log("Statistic of succ_rate: {}".format(np.mean(succ)))
        except KeyboardInterrupt:
            logger.log("Statistic of succ_rate: {}".format(np.mean(succ)))
            logger.info('Quit by hand !')
    


if __name__=='__main__' :
    main()
    

