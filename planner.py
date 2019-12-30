import numpy as np
import numpy.matlib
import tensorflow as tf
import common.tf_util as U
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from common.utils import store_args
from common.mpi_adam import MpiAdam
from dataset import Dataset
from replay_buffer import PlanReplayBuffer
from normalizer import Normalizer
import os
import logger


class Planner(object):

    @store_args
    def __init__(self, inp_dim, hid_size, seq_len, out_dim, buffer_size, batch_size=64,
                 optim_stepsize=1e-3, sample_func=None, norm_eps=1e-2, norm_clip=5, scope='planner',
                 layerNorm=False, **kwargs):
        '''
        Implemention of LSTM Planner that produces given number of subgoals between src and dest.
        Args:
            inp_dim : dimension for the LSTM
            hid_size : cell_state_size
            seq_len : max_timesteps
            out_dim : dimension for LSTM output
        '''
        # self.main = lstm(hid_size, layerNorm)

        self.adamepsilon = 1e-6

        self.mode = tf.contrib.learn.ModeKeys.TRAIN  # TRAIN for training, INFER for prediction, EVAL for evaluation
        self.infer_outputs = None
        with tf.variable_scope(self.scope) :
            self._create_network()
        
        buffer_shape = [seq_len+2, out_dim]   # plus 2: the [0] is 'src', [1] is 'dest', [2:] are 'labels', 
        if self.sample_func is None:
            from sampler import make_sample_plans
            self.sample_func = make_sample_plans()
        self.buffer = PlanReplayBuffer(buffer_shape, buffer_size, self.sample_func)


    def _create_network(self):
        self.sess = U.get_session()

        self.inp_src = tf.placeholder(shape=[None, 1, self.inp_dim], dtype=tf.float32, name='input_src')
        self.inp_dest = tf.placeholder(shape=[None, 1, self.out_dim], dtype=tf.float32, name='input_dest')
        self.labels = tf.placeholder(shape=[None, self.seq_len, self.out_dim], dtype=tf.float32, name='label')
        self.src_seq_len = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
        self.tar_seq_len = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        # running averages
        # with tf.variable_scope('goal_stats_src'):
        #     self.goal_stats_src = Normalizer(self.inp_dim, self.norm_eps, self.norm_clip, sess=self.sess)
        with tf.variable_scope('goal_stats_dest'):
            self.goal_stats_dest = Normalizer(self.out_dim, self.norm_eps, self.norm_clip, sess=self.sess, PLN=True)
        
        # normalize inp_src, and goals labels
        inp_src = self.goal_stats_dest.normalize(self.inp_src)
        inp_dest = self.goal_stats_dest.normalize(self.inp_dest)
        goal_labels = self.goal_stats_dest.normalize(self.labels)
        with tf.variable_scope('goal_gen'):
            encoder_cell = tf.nn.rnn_cell.LSTMCell(self.hid_size)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, inp_src, sequence_length=self.src_seq_len, dtype=tf.float32)

            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hid_size)
            project_layer = tf.layers.Dense(self.out_dim)

            with tf.variable_scope("decode"):
                train_inp = tf.concat([inp_dest, goal_labels[:,:-1,:]], axis=-2)
                train_helper = tf.contrib.seq2seq.TrainingHelper(
                            train_inp,
                            sequence_length=self.tar_seq_len
                         )
                train_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, train_helper, encoder_state, 
                                    output_layer=project_layer)
                train_outputs, _, final_seq_len = tf.contrib.seq2seq.dynamic_decode(train_decoder, maximum_iterations=self.seq_len)
                self.train_outputs = train_outputs.rnn_output
            
            with tf.variable_scope("decode", reuse=True):
                infer_helper = ContinousInferHelper(inp_dest[:,0,:], self.tar_seq_len)
                infer_decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, infer_helper, encoder_state,
                                    output_layer=project_layer)
                infer_outputs, _, final_seq_len = tf.contrib.seq2seq.dynamic_decode(infer_decoder, maximum_iterations=self.seq_len)
                self.infer_outputs = self.goal_stats_dest.denormalize(infer_outputs.rnn_output)

            log_sigma = tf.get_variable(name="logstd", shape=[1, self.out_dim], initializer=U.normc_initializer(0.1))

            goals = train_outputs.rnn_output
            loss =   0.5 * tf.reduce_sum(tf.square((goal_labels - goals)/tf.exp(log_sigma)), axis=-1) \
                + 0.5 * np.log(2*np.pi) * tf.to_float(tf.shape(self.labels)[-1]) \
                + tf.reduce_sum(log_sigma, axis=-1)
            self.loss = tf.reduce_mean(loss)
            self.tr_outputs = self.goal_stats_dest.denormalize(self.train_outputs)       # just for inspect the correctness of training
        
        var_list = self._vars('')
        self.grads = U.flatgrad(self.loss, var_list)
        self.adam = MpiAdam(var_list, epsilon=self.adamepsilon)

        tf.variables_initializer(self._global_vars('')).run()
        self.adam.sync()
        

    def train(self, use_buffer=False, justEval=False, **kwargs):
        self.mode = tf.contrib.learn.ModeKeys.TRAIN
        if not use_buffer:
            src = np.reshape(kwargs['src'], [-1,1, self.inp_dim])
            dest = np.reshape(kwargs['dest'], [-1,1, self.out_dim])
            lbl = kwargs['lbl']
        else:
            episode_batch = self.buffer.sample(self.batch_size)
            src = np.reshape(episode_batch[:, 0,:], [-1, 1, self.inp_dim])
            lbl = episode_batch[:, 2:, :]
            dest = np.reshape(episode_batch[:, 1, :], [-1, 1, self.out_dim])
        src_seq_len = [1] * src.shape[0]
        tar_seq_len = [self.seq_len] * dest.shape[0]
        # compute grads
        loss, g,tr_sub_goals, te_sub_goals = self.sess.run([self.loss, self.grads, self.tr_outputs, self.infer_outputs], feed_dict={
                                self.inp_src  :src,
                                self.inp_dest: dest,
                                self.labels : lbl,
                                self.src_seq_len : src_seq_len,
                                self.tar_seq_len : tar_seq_len
                                })
        if not justEval:
            self.adam.update(g, stepsize=self.optim_stepsize)
        return loss, tr_sub_goals[-1], te_sub_goals[-1]
    
    def plan(self, src, dest):
        src = np.reshape(src, [-1, 1, self.inp_dim])
        dest = np.reshape(dest, [-1, 1, self.out_dim])
        src_seq_len = [1] * src.shape[0]
        tar_seq_len = [self.seq_len] * dest.shape[0]
        plan_goals = self.sess.run(self.infer_outputs, feed_dict={
                                    self.inp_src  :src,
                                    self.inp_dest: dest,
                                    self.src_seq_len : src_seq_len,
                                    self.tar_seq_len : tar_seq_len
                                    })
        
        assert plan_goals.shape[0] == src.shape[0] and plan_goals.shape[1] == self.seq_len
        plan_goals = np.flip(plan_goals, axis=-2)
        plan_goals = np.concatenate([plan_goals, dest], axis=-2)      # append the ultimate goal
        return plan_goals
    
    def store_episode(self, episode_batch, update_stats=True):
        """ episode_batch : [batch_size * (subgoal_num+1) * subgoal_dim]
        """
        isNull = episode_batch.shape[0] < 1
        if not isNull:
            self.buffer.store_episode(episode_batch)
        # logger.info("buffer store_episode done. updating statistics.")
        if update_stats:
            subgoals = episode_batch[:, 1:, :]
            self.goal_stats_dest.update(subgoals, isNull=isNull)
            # logger.info("ready to recomput_stats")
            # print(subgoals)
            self.goal_stats_dest.recompute_stats(inc=episode_batch.shape[0])

    
    def update_normalizer_stats(self, batch):
        # self.goal_stats_src.update(batch['src'])
        self.goal_stats_dest.update(batch['dest'])
        # self.goal_stats_src.recompute_stats()
        self.goal_stats_dest.recompute_stats()
    
    def _vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + '/' + scope)
        assert len(res) > 0
        return res

    def _global_vars(self, scope):
        res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope + '/' + scope)
        return res

    def save(self, save_path):
        assert self.infer_outputs is not None
        var_list = self._global_vars('')
        U.save_variables(save_path, variables=var_list, sess=self.sess)


    def load(self, load_path):
        if self.infer_outputs is None:
            self._create_network()
        var_list = self._global_vars('')
        U.load_variables(load_path, variables=var_list)
    
    def logs(self, prefix=''):
        logs = []
        logs += [('subgoals/buff_size', self.buffer.get_current_episode_size())]
        logs += [('goals/mean', np.mean(self.sess.run([self.goal_stats_dest.mean])))]
        logs += [('goals/std', np.mean(self.sess.run([self.goal_stats_dest.std])))]

        if prefix != '':
            prefix = prefix.strip('/')
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

##################################
### funtions for custom helper ###
##################################

class ContinousInferHelper(tf.contrib.seq2seq.Helper):

    def __init__(self, start_inputs, sequence_length, sample_ids_shape=None, sample_ids_dtype=None):
        self._start_inputs = start_inputs
        self._sequence_length = sequence_length
        self._batch_size = array_ops.size(sequence_length)
    
    def initialize(self):
        # all False at the initial step
        initial_finished = math_ops.equal(0, self._sequence_length)
        # TODO: designate the start_inputs
        # self.start_inputs = ???
        return (initial_finished, self._start_inputs)

    def sample(self, time, outputs, state):
        # del time, state  # unused by sample_fn
        # use argmax to fetch maximum from outputs
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), dtype=tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids):
        time += 1
        # this operation produces boolean tensor of [batch_size]
        elements_finished = (time >= self._sequence_length)
        # all_finished, type: boolean scalar, marking the end of batch
        all_finished = tf.reduce_all(elements_finished)
        # If finished, the next_inputs value doesn't matter
        next_inputs = tf.cond(all_finished, lambda: self._start_inputs, lambda: outputs)
        return elements_finished, next_inputs, state

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return dtypes.int32
