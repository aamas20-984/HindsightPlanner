
import os
import time
# import logz
import inspect
import tensorflow as tf
import gym
import numpy as np 
import numpy.matlib
import common.tf_util as U
from  common.utils import set_global_seeds
from common.mpi_adam import MpiAdam
from common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import matplotlib.pyplot as plt
import config
import logger
from rollout import RolloutWorker


def mpi_average(value):
    if not isinstance(value, list):
        value = [value]
    if not any(value):
        value = [0.]
    return mpi_moments(np.array(value))[0]

def train(policy, planner, rollout_worker, evaluator,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_path, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    if save_path:
        latest_mdl_path = save_path+'_latest'
        best_mdl_path = save_path
        periodic_policy_path = save_path+'_{}'
    
    best_success_rate = -1

    logger.info('Training......')
    # num_timesteps = n_epochs * n_cycles * rollout_length * number of rollout workers
    for epoch in range(n_epochs):
        logger.info('========== epoch {} ========='.format(epoch))
        logger.record_tabular('epoch', epoch)

        # train
        rollout_worker.clear_history()
        for _ in range(n_cycles):
            # logger.info('collect rollouts...')
            episode_for_act, episode_for_pln = rollout_worker.generate_rollouts(cur_progress=(epoch/n_epochs))
            # logger.info('store rollouts for policy')
            policy.store_episode(episode_for_act)
            # logger.info('store rollouts for planner, episodes_for_pln shape:', episode_for_pln.shape)
            planner.store_episode(episode_for_pln)
            # logger.info('training policy')
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()
            # logger.info('training planner')
            for _ in range(n_batches):
                planner.train(use_buffer=True)

        # test
        # logger.info("evaluate...")
        evaluator.clear_history()
        for ro in range(n_test_rollouts):
            evaluator.generate_rollouts()
        
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))
        for key, val in planner.logs():
            logger.record_tabular(key, mpi_average(val))
        if rank == 0:
            logger.dump_tabular()
        
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_path:
            best_success_rate = success_rate
            # logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            # evaluator.save_policy(latest_mdl_path)
            logger.info('Saving best policy+planner to {} ...'.format(best_mdl_path))
            evaluator.save_policy(best_mdl_path)
            evaluator.save_planner(best_mdl_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_path:
            # policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving lastest policy+planner to {} ...'.format(latest_mdl_path))
            evaluator.save_policy(latest_mdl_path)
            evaluator.save_planner(latest_mdl_path)
        elif rank==0 and policy_save_interval < 0 and epoch % (-policy_save_interval) == 0 and save_path:
            periodic_mdl_path = periodic_policy_path.format(epoch)
            logger.info('Saving periodic policy+planner to {} ...'.format(periodic_mdl_path))
            evaluator.save_policy(periodic_mdl_path)
            evaluator.save_planner(periodic_mdl_path)
        
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]
    
    return policy, planner


def learn(env, total_timesteps,
    seed=None,
    replay_strategy='future',
    policy_save_interval=5,
    clip_return=True,
    override_params=None,
    load_path=None,
    save_path=None,
    **kwargs
    ):

    # env = gym.make(env_name)
    
    override_params = override_params or {}
    if MPI is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        num_cpu = MPI.COMM_WORLD.Get_size()
    
    # Seed everything.
    rank_seed = seed + 1000000 * rank if seed is not None else None
    set_global_seeds(rank_seed)

    # prepare params
    logger.info("preparing parameters for NN models")
    params = config.DEFAULT_AGENT_PARAMS
    env_name = env.spec.id
    params['env_name'] = env_name
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])
    
    params.update(**override_params)
    params['rollout_per_worker'] = env.num_envs
    params['rollout_batch_size'] = params['rollout_per_worker']
    params['num_timesteps'] = total_timesteps
    logger.save_params(params=params, filename='ddpg_params.json')
    
    # initialize session
    # tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    # tf_config.gpu_options.allow_growth = True # may need if using GPU
    # sess = tf.Session(config=tf_config)
    # sess.__enter__()

    # get policy given params
    policy = config.config_params_get_policy(params=params, clip_return=clip_return)
    # get planner
    planner = config.config_params_get_planner(params=params)
    if load_path is not None:
        U.load_variables(load_path+'_pi')       # pi and planner are seperately stored.
    if load_path is not None:
        U.load_variables(load_path+'_pln')

    rollout_params = {
        'exploit': False,
        'act_rdm_dec': params['act_rdm_dec'],
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
        'reward_fun': params['reward_fun'],
        'goal_delta': params['goal_delta'],
        'subgoal_strategy': params['subgoal_strategy'],
        'subgoal_num': params['seq_len']+1,
        'subgoal_norm' : env_name.startswith('Hand')
    }
    eval_params = {
        'exploit': True,
        'act_rdm_dec': params['act_rdm_dec'],
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
        'reward_fun': params['reward_fun'],
        'subgoal_strategy': params['subgoal_strategy'],
        'goal_delta': params['goal_delta'],
        'subgoal_num': params['seq_len']+1,
        'subgoal_norm' : env_name.startswith('Hand')
    }
    for name in ['T', 'rollout_per_worker', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]
    
    eval_env = env

    rollout_worker = RolloutWorker(env, policy, params['dims'], logger, planner=planner, monitor=True, **rollout_params)
    evaluator = RolloutWorker(eval_env, policy, params['dims'], logger, planner=planner, **eval_params)

    n_cycles = params['n_cycles']
    n_epochs = total_timesteps // n_cycles // rollout_worker.T // rollout_worker.rollout_per_worker

    

    return train(
        save_path=save_path, policy=policy, planner=planner, rollout_worker=rollout_worker,
        evaluator=evaluator, n_epochs=n_epochs, n_test_rollouts=params['n_test_rollouts'],
        n_cycles=params['n_cycles'], n_batches=params['n_batches'],
        policy_save_interval=policy_save_interval
    )
