import os
import numpy as np
import gym

from ddpg import DDPG
from planner import Planner
from sampler import make_sample_her_transitions
from sampler import make_sample_plans
from common.monitor import Monitor
import logger


DEFAULT_ENV_PARAMS = {
    'FetchReach-v1': {
        'n_cycles': 10,
    },
    'AntMazeU-v1':{
        'buffer_size': int(2.5E5),
        'relative_goals': True,
    },
    'AntMazeG-v1':{
        'buffer_size': int(2E5),
        'relative_goals': True,
    }
}

DEFAULT_AGENT_PARAMS = {
    # env
    'max_u': 1.,  # max absolute value of actions on different coordinates
    # ddpg
    'layers': 3,  # number of layers in the critic/actor networks
    'hidden': 256,  # number of neurons in each hidden layers
    'network_class': 'actor_critic:ActorCritic',
    'Q_lr': 0.001,  # critic learning rate
    'pi_lr': 0.001,  # actor learning rate
    'buffer_size': int(1E6),  # for experience replay
    'polyak': 0.95,  # polyak averaging coefficient
    'action_l2': 1.0,  # quadratic penalty on actions (before rescaling by max_u)
    'clip_obs': 200.,
    'scope': 'ddpg_pi',  # can be tweaked for testing
    'relative_goals': False,
    # training
    'n_cycles': 50,  # per epoch
    'rollout_batch_size': 2,  # per mpi thread
    'n_batches': 40,  # training batches per cycle
    'batch_size': 256,  # per mpi thread, measured in transitions and reduced to even multiple of chunk_length.
    'n_test_rollouts': 10,  # number of test rollouts per epoch, each consists of rollout_batch_size rollouts
    'test_with_polyak': False,  # run test episodes with the target network
    # exploration
    'random_eps': 0.3,  # percentage of time a random action is taken
    'noise_eps': 0.2,  # std of gaussian noise added to not-completely-random actions as a percentage of max_u
    'act_rdm_dec': 'None',        # supported modes: None, linear, sine
    # HER
    'replay_strategy': 'future',  # supported modes: future, none
    'replay_k': 4,  # number of additional goals used for replay, only used if off_policy_data=future
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values

    'bc_loss': 0, # whether or not to use the behavior cloning loss as an auxilliary loss
    'q_filter': 0, # whether or not a Q value filter should be used on the Actor outputs
    'num_demo': 100, # number of expert demo episodes
    'demo_batch_size': 128, #number of samples to be used from the demonstrations buffer, per mpi thread 128/1024 or 32/256
    'prm_loss_weight': 0.001, #Weight corresponding to the primary loss
    'aux_loss_weight':  0.0078, #Weight corresponding to the auxilliary loss also called the cloning loss
}

DEFAULT_PLANNER_PARAMS = {
    'scope': 'planner',
    'hid_size': 64,
    'optim_stepsize' : 0.001,       # learning rate
    'buffer_size': int(1E4),        # for plan replay
    'layerNorm' : False,            # whether or not to use layerNorm in RNN
    'seq_len' : 4,                  # 4 subgoals
    'pln_batch_size': 64,               # batch_size for training
    # normalization
    'norm_eps': 0.01,  # epsilon used for observation normalization
    'norm_clip': 5,  # normalized observations are cropped to this values
    # 'subgoal_strategy': 'time_sample',
}

CACHED_ENVS = {}

def cached_make_env(make_env):
    """
    Only creates a new environment from the provided function if one has not yet already been
    created. This is useful here because we need to infer certain properties of the env, e.g.
    its observation and action spaces, without any intend of actually using it.
    """
    if make_env not in CACHED_ENVS:
        env = make_env()
        CACHED_ENVS[make_env] = env
    return CACHED_ENVS[make_env]


def simple_goal_subtract(a, b):
    assert a.shape == b.shape
    return a - b

def config_params_get_policy(params, reuse=False, clip_return=True):
    env_name = params['env_name']

    def make_env(subrank=None):
        env = gym.make(env_name)
        if subrank is not None and logger.get_dir() is not None:
            try:
                from mpi4py import MPI
                mpi_rank = MPI.COMM_WORLD.Get_rank()
            except ImportError:
                MPI = None
                mpi_rank = 0
                logger.warn('Running with a single MPI process. This should work, but the results may differ from the ones publshed in Plappert et al.')

            max_episode_steps = env._max_episode_steps
            env =  Monitor(env,
                           os.path.join(logger.get_dir(), str(mpi_rank) + '.' + str(subrank)),
                           allow_early_resets=True)
            # hack to re-expose _max_episode_steps (ideally should replace reliance on it downstream)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    params['make_env'] = make_env
    env = cached_make_env(params['make_env'])
    env.reset()
    params['T'] = env.spec.max_episode_steps
    params['max_u'] = np.array(params['max_u']) if isinstance(params['max_u'], list) else params['max_u']
    params['gamma'] = 1. - 1. / params['T']
    if 'lr' in params:
        params['pi_lr'] = params['lr']
        params['Q_lr'] = params['lr']
        del params['lr']

    ddpg_params = dict()
    for name in ['buffer_size', 'hidden', 'layers',
                 'network_class',
                 'polyak',
                 'batch_size', 'Q_lr', 'pi_lr',
                 'norm_eps', 'norm_clip', 'max_u',
                 'action_l2', 'clip_obs', 'scope', 'relative_goals']:
        ddpg_params[name] = params[name]
        params['_' + name] = params[name]
        del params[name]

    params['ddpg_params'] = ddpg_params

    # configure_her
    def reward_fun(ag_2, g, info):  # vectorized
        return env.compute_reward(achieved_goal=ag_2, desired_goal=g, info=info)
    
    def goal_delta(g1, g2, weight=None):
        return env.goal_delta(pre_goal=g1, now_goal=g2, weight=weight)
    
    her_param = {
        'reward_fun': reward_fun,
        'replay_strategy' : params['replay_strategy'],
        'replay_k' : params['replay_k']
    }
    sample_her_transitions = make_sample_her_transitions(**her_param)
    params['reward_fun'] = reward_fun
    params['goal_delta'] = goal_delta

    # configure_dims
    env.reset()
    obs, _, _, info = env.step(env.action_space.sample())
    dims = {
        'o': obs['observation'].shape[0],
        'u': env.action_space.shape[0],
        'g': obs['desired_goal'].shape[0],
    }
    for key, value in info.items():
        value = np.array(value)
        if value.ndim == 0:
            value = value.reshape(1)
        dims['info_{}'.format(key)] = value.shape[0]
    params['dims'] = dims
    
    # configure_ddpg
    gamma = params['gamma']
    ddpg_params.update({
                        'input_dims' : dims,
                        'T': params['T'],
                        'clip_pos_returns': True,  # clip positive returns
                        'clip_return': (1. / (1. - gamma)) if clip_return else np.inf,  # max abs of return
                        'rollout_batch_size': params['rollout_batch_size'],
                        'subtract_goals': simple_goal_subtract,
                        'sample_transitions': sample_her_transitions,
                        'gamma': gamma
                    })
    ddpg_params['info'] = {
        'env_name': params['env_name']      }
    policy = DDPG(reuse=reuse, **ddpg_params, use_mpi=True)
    return policy


def config_params_get_planner(params, sess=None):
    env_name = params['env_name']
    if 'make_env' not in params:
        def make_env():
            env = gym.make(env_name)
            return env
        params['make_env'] = make_env

    env = cached_make_env(params['make_env'])
    env.reset()
    if 'dims' not in params:
        obs, _, _, info = env.step(env.action_space.sample())
        dims = {
            'o': obs['observation'].shape[0],
            'u': env.action_space.shape[0],
            'g': obs['desired_goal'].shape[0],
        }
        for key, value in info.items():
            value = np.array(value)
            if value.ndim == 0:
                value = value.reshape(1)
            dims['info_{}'.format(key)] = value.shape[0]
        params['dims'] = dims
    else:
        dims = params['dims']
    planner_params = dict()
    planner_params.update(DEFAULT_PLANNER_PARAMS)

    for attr in ['hid_size', 'pln_batch_size', 'seq_len']:
        if attr in params:
            planner_params[attr] = params[attr]
    params['seq_len'] = planner_params['seq_len']       # write into params for the rollout worker

    logger.save_params(params=planner_params, filename='planner_params.json')
    sample_func = make_sample_plans()
    planner_params['batch_size'] = planner_params['pln_batch_size']
    del planner_params['pln_batch_size']
    planner_params.update({
                        'inp_dim': dims['g'],
                        'out_dim': dims['g'],
                        'sample_func': sample_func
                        })
    
    planner = Planner(**planner_params, use_mpi=True)

    return planner 
