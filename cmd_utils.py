import gym
import os
import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from gym.wrappers import FlattenDictWrapper

from common.monitor import Monitor
from common.utils import set_global_seeds
from common.vec_env.subproc_vec_env import SubprocVecEnv
from common.vec_env.dummy_vec_env import DummyVecEnv
import logger

def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()
    def make_thunk(rank):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            logger_dir=logger_dir
        )

    set_global_seeds(seed)
    if num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(start_index)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None, flatten_dict_observations=True, wrapper_kwargs=None, logger_dir=None):
    wrapper_kwargs = wrapper_kwargs or {}
    # if env_type == 'atari':
    #     env = make_atari(env_id)
    # elif env_type == 'retro':
    #     import retro
    #     gamestate = gamestate or retro.State.DEFAULT
    #     env = retro_wrappers.make_retro(game=env_id, max_episode_steps=10000, use_restricted_actions=retro.Actions.DISCRETE, state=gamestate)
    # else:
    env = gym.make(env_id)

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        keys = env.observation_space.spaces.keys()
        env = gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))

    env.seed(seed + subrank if seed is not None else None)
    env = Monitor(env,
                  logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                  allow_early_resets=True)

    # if env_type == 'atari':
    #     env = wrap_deepmind(env, **wrapper_kwargs)
    # elif env_type == 'retro':
    #     env = retro_wrappers.wrap_deepmind_retro(env, **wrapper_kwargs)

    # if reward_scale != 1:
    #     env = retro_wrappers.RewardScaler(env, reward_scale)

    return env