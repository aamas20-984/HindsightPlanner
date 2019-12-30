from collections import deque

import numpy as np
from common.utils  import store_args, convert_episode_to_batch_major

class RolloutWorker:

    @store_args
    def __init__(self, venv, policy, dims, logger, T, reward_fun, goal_delta, planner=None, rollout_per_worker=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, act_rdm_dec='None', history_len=100, render=False, monitor=False, 
                 subgoal_num=5, subgoal_strategy='time', subgoal_norm=False, **kwargs):

        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_per_worker (int): the number of parallel rollouts that should be used per worker
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
            subgoal_strategy : strategy to sample subgoals, i.e. 'time' means choose the ac_go with equal timestep interval, 
                'route' means choose the ac_go with equal route interval,
                'random' means choose 'subgoal_num' ac_go randomly.
        """
        assert self.T > 0
        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]
        
        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.reset_all_rollouts()
        self.clear_history()
        self.qualify_traj_cnt = 0

    def reset_all_rollouts(self):
        self.obs_dict = self.venv.reset()
        self.initial_o = self.obs_dict['observation']
        self.initial_acgo = self.obs_dict['achieved_goal']
        self.dego = self.obs_dict['desired_goal']


    def generate_rollouts(self, cur_progress=0, last_cyc=False):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        # reset VecEnv (get initial_ob, desired goal, etc.)
        self.reset_all_rollouts()
        if self.act_rdm_dec == "linear":
            cur_random_eps = self.random_eps - (self.random_eps - 0.3) * cur_progress
        elif self.act_rdm_dec == "sine":
            cur_random_eps = self.random_eps - (self.random_eps - 0.3) * np.sin(np.min([cur_progress, 0.125]) * np.pi * 4)
        elif self.act_rdm_dec == "None":
            cur_random_eps = self.random_eps

        # variables for every timestep
        o = np.empty((self.rollout_per_worker, self.dims['o']), np.float32)  # observations
        acgo = np.empty((self.rollout_per_worker, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        acgo[:] = self.initial_acgo
        qualify = np.array([False] * self.rollout_per_worker)

        # episodic variable for each episode
        #  plan_goals:  goals that planner make for this episode
        if self.qualify_traj_cnt > 500 and self.planner is not None:
        # if self.planner is not None:
            plan_goals = self.planner.plan(self.initial_acgo, self.dego)           # shape = (rollout_per_worker, num of subgoals, dims['g'])
            # print("plan_goals shape: ", plan_goals.shape)
            if self.subgoal_norm:
                plan_goals[..., 3:] /= np.linalg.norm(plan_goals[..., 3:], axis=-1, keepdims=True)
            # self.venv.set_subgoal(plan_goals[:,:-1])
        else:
            plan_goals = self.dego.reshape(self.rollout_per_worker, 1, -1)
        wkr_idx = np.arange(self.rollout_per_worker)
        cur_subgo_idx = np.zeros(self.rollout_per_worker, np.int32)         # current subgoal index for each rollout
        self.dego = plan_goals[wkr_idx, cur_subgo_idx] #.reshape(self.rollout_per_worker, self.dims['g'])
        #  obs:      observations in this episode
        #  ac_goals: achieved goals in this episode
        #  acts:     actions in this episode
        #  goals:    goals that input to agent for quering action, namely subgoal in the plan_goals
        #  succ:     whether realize the original goal
        obs, ac_goals, acts, goals, succs = [], [], [], [], []
        routes = []
        dones = []
        info_values = [np.empty((self.T - 1, self.rollout_per_worker, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        tmp_info = {}
        # generate episodes
        for t in range(self.T):
            # judge agent has achieved the subgoal given by planner
            # print("acgo shape:", acgo.shape, ", dego shape:", self.dego.shape)
            # tmp_info['is_subgoal'] = [cur_subgo_idx[i]+1 < len(plan_goals[i]) for i in range(self.rollout_per_worker)]
            tmp_info['is_subgoal'] = (cur_subgo_idx[0]+1 < len(plan_goals[0]))
            env_rew = self.reward_fun(acgo, self.dego, tmp_info)
            succ_subgoal = (np.abs(env_rew) < 1e-5).astype(np.int32)
            cur_subgo_idx = np.min([cur_subgo_idx+succ_subgoal, np.array([plan_goals.shape[-2]-1]*self.rollout_per_worker)], axis=0)
            self.dego = plan_goals[wkr_idx,cur_subgo_idx] #.reshape(self.rollout_per_worker, self.dims['g'])
            # query actions from polciy
            pi_out = self.policy.get_actions(
                        o, acgo, self.dego,
                        compute_Q=self.compute_Q,
                        noise_eps=self.noise_eps if not self.exploit else 0.,
                        random_eps=cur_random_eps if not self.exploit else 0.,
                        use_target_net=self.use_target_net
                        )
            if self.compute_Q:
                u, Q = pi_out
                Qs.append(Q)
            else:
                u = pi_out

            if u.ndim==1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)
            
            o_new = np.empty((self.rollout_per_worker, self.dims['o']))
            ag_new = np.empty((self.rollout_per_worker, self.dims['g']))
            
            # step forward in VecEnv
            obs_dict_new, _, done, info = self.venv.step(u)
            if last_cyc:
                self.venv.render()

            o_new = obs_dict_new['observation']
            ag_new = obs_dict_new['achieved_goal']      # TODO: the original fetch_env just return the object pos as the achieved_goal
            succ = np.array([i.get('is_success', 0.0) for i in info])
            qualify = qualify | np.array([i.get('is_far_enough', False) for i in info])
            # for strategy 'route-sample'
            delta_route = self.goal_delta(acgo, ag_new)

            if any(done):
                # here we assume all environments are done is ~same number of steps, so we terminate rollouts whenever any of the envs returns done
                # trick with using vecenvs is not to add the obs from the environments that are "done", because those are already observations
                # after a reset
                # for HandManipulate Environment, use the `still_on_palm` information to filter out bad episodes
                qualify = qualify & np.array([i.get('still_on_palm', True) for i in info])
                break
            
            for i, _ in enumerate(info):
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[i][key]

            if np.isnan(o_new).any():
                self.logger.warn('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()
            
            dones.append(done)
            obs.append(o.copy())
            ac_goals.append(acgo.copy())
            succs.append(succ.copy())
            acts.append(u.copy())
            routes.append(delta_route.copy())
            goals.append(self.dego.copy())
            o[...] = o_new
            acgo[...] = ag_new
        obs.append(o.copy())
        ac_goals.append(acgo.copy())

        episode = dict(o=obs, u=acts, g=goals, ag=ac_goals)

        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        successful = np.array(succs)[-3:, :]
        successful = np.ceil(np.mean(successful, axis=0) - 0.5)
        assert successful.shape == (self.rollout_per_worker,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_per_worker

        # prepare the hindsight subgoals for planner training
        episode_for_pln = {}
        if self.subgoal_strategy == 'time':
            intval = len(ac_goals) // self.subgoal_num
            subgoal_idx = list(range(len(ac_goals)-1, -1, -intval))
            subgoal_idx = subgoal_idx[0:self.subgoal_num]
            subgoal_idx = np.array(subgoal_idx, dtype=np.int32)
            episode_for_pln['hsubgo'] = [ac_goals[i] for i in subgoal_idx]      # include the terminal goal; reverse order
        elif self.subgoal_strategy == 'route':
            cut_point = np.array([i for i in range(1, self.subgoal_num+1)]) / self.subgoal_num
            routes = np.swapaxes(routes, 0, 1)
            routes_sum = np.sum(routes, axis=-1, keepdims=True)
            routes = np.cumsum(routes, axis=-1) / routes_sum
            subgoal_idx = np.zeros(self.rollout_per_worker, dtype=np.int32)
            episode_for_pln['hsubgo'] = []
            for cut in cut_point:
                acgoal_in_cut = []
                for wk in range(self.rollout_per_worker):
                    while subgoal_idx[wk] < routes.shape[1] and routes[wk][subgoal_idx[wk]] < (cut-1e-3):
                        subgoal_idx[wk] += 1
                    assert subgoal_idx[wk] < routes.shape[1], 'routes[wk][subgoal_idx[wk]-1] is {}, cut is {}'.format(routes[wk][subgoal_idx[wk]-1], cut)
                    acgoal_in_cut.append(ac_goals[subgoal_idx[wk]][wk])
                episode_for_pln['hsubgo'].append(acgoal_in_cut)
            episode_for_pln['hsubgo'] = np.flipud(episode_for_pln['hsubgo'])
        else:
            raise NotImplementedError
        
        # print("rollout: hsubgo")
        # print(episode_for_pln['hsubgo'])
        # |key| * num_timesteps * rollout_per_worker -> |key| * rollout_per_worker * num_timesteps
        episode = convert_episode_to_batch_major(episode)
        episode_for_pln = convert_episode_to_batch_major(episode_for_pln)
        episode_for_pln['hsubgo'] = np.concatenate([self.initial_acgo.reshape(self.rollout_per_worker,1,-1)
                                          , episode_for_pln['hsubgo']], axis=-2)
        # remove the 'hsubgo' of those unqualified episode
        qual_idx = np.where(qualify)
        episode_for_pln['hsubgo'] = episode_for_pln['hsubgo'][qual_idx]
        self.qualify_traj_cnt += len(episode_for_pln['hsubgo'])
        return episode, episode_for_pln['hsubgo']
        

    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()
    
    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)
    
    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        self.policy.save(path+'_pi')
    
    def save_planner(self, path):
        """Pickles the current planner for later inspection.
        """
        assert (self.planner is not None)
        self.planner.save(path+'_pln')

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix != '':
            prefix = prefix.strip('/')
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs
