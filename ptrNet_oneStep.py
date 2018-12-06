from __future__ import absolute_import
from __future__ import  division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class PointerWrapper(tf.contrib.seq2seq.AttentionWrapper):
  """Customized AttentionWrapper for PointerNet."""

  def __init__(self,cell, attention_size, memory, memory_sequence_length=None,initial_cell_state=None,name=None):

    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(attention_size, memory, memory_sequence_length=memory_sequence_length, probability_fn=lambda x: x)

    # According to the paper, no need to concatenate the input and attention
    # Therefore, we make cell_input_fn to return input only
    cell_input_fn=lambda input, attention: input

    super(PointerWrapper, self).__init__(cell,
                                         attention_mechanism=attention_mechanism,
                                         attention_layer_size=None,
                                         alignment_history=False,
                                         cell_input_fn=cell_input_fn,
                                         output_attention=True,
                                         initial_cell_state=initial_cell_state,
                                         name=name)
  @property
  def output_size(self):
    return self.state_size.alignments

  def call(self, inputs, state):
    _, next_state = super(PointerWrapper, self).call(inputs, state)
    return next_state.alignments, next_state


class PointerNet:
    def __init__(self, iterator, params, mode):
        self.iterator = iterator
        self.params = params
        if mode == 'train':
            self.mode = tf.contrib.learn.ModeKeys.TRAIN
        elif mode == 'infer':
            self.mode = tf.contrib.learn.ModeKeys.INFER
        elif mode == 'eval':
            self.mode = tf.contrib.learn.ModeKeys.EVAL
        else:
            raise ValueError("Please choose mode: train, eval or infer")

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.global_step = tf.Variable(0, trainable=False)

        self.logits, self.loss= self.build_graph(mode)

        trainable_vars = tf.trainable_variables()

        opt = tf.train.AdamOptimizer(params.learning_rate)

        gradients = tf.gradients(self.loss, trainable_vars)
        clipped_grad , self.gradient_norm = tf.clip_by_global_norm(gradients, 5)
        self.update = opt.apply_gradients(zip(clipped_grad, trainable_vars), global_step=self.global_step)

        self.train_summary = tf.summary.merge([tf.summary.scalar('train_loss', self.loss)])

        self.saver=tf.train.Saver(tf.global_variables(), max_to_keep=params.max_to_keep)
        self.sess.run(tf.global_variables_initializer())


    def build_graph(self, mode):
        tf.logging.info("Building {} model...".format(mode))
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.src, self.src_length, self.trgt_in, self.trgt_out, self.trgt_length = self.iterator.get_next()
        enc_outputs, enc_state = self._build_encoder()
        logits = self._build_decoder(enc_outputs, enc_state)
        self.predicted = tf.contrib.framework.argsort(logits, axis=-1,direction='DESCENDING') +1

        loss = self._compute_loss(logits)
        return logits, loss

    def _build_encoder(self):
        dropout = self.params.dropout if self.mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

        with tf.variable_scope("encoder") as scope:
            cell = tf.contrib.rnn.LSTMBlockCell(self.params.num_units)

            if dropout > 0.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=(1.0-dropout))

            enc_outputs, enc_state =tf.nn.dynamic_rnn(cell, self.src, sequence_length=self.src_length, dtype = tf.float32)

        return enc_outputs, enc_state


    def _build_decoder(self, enc_outputs, enc_state):
        with tf.variable_scope("decoder") as decoder_scope:
            cell, decoder_initial_state = self._build_decoder_cell(enc_outputs, enc_state, self.src_length)

            initial_inputs = tf.fill([self.batch_size, self.params.input_dim], 0.0)
            logits, _ = cell.call(inputs = initial_inputs, state = decoder_initial_state)
        return logits

        #     if self.mode != tf.contrib.learn.ModeKeys.INFER:
        #         dec_inputs = self.trgt_in
        #
        #         if self.params.time_major:
        #             dec_inputs = tf.transpose(dec_inputs)
        #
        #         helper = tf.contrib.seq2seq.TrainingHelper(dec_inputs, self.trgt_length)
        #
        #         decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper,  decoder_initial_state)
        #
        #         outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope = decoder_scope)
        #         logits = outputs.rnn_output
        #         pointers = outputs.sample_id
        #
        #     else:
        #         def initialize_fn():
        #             finished = tf.tile([False], [self.batch_size])
        #             start_inputs = tf.fill([self.batch_size, self.params.input_dim], 0.0)
        #             return (finished, start_inputs)
        #
        #         def sample_fn(time, outputs, state):
        #             del time, state
        #             sample_ids =  tf.argmax(outputs, axis=-1, output_type=tf.int32)
        #             return sample_ids
        #
        #         def next_inputs_fn(time, outputs, state, sample_ids):
        #             del outputs
        #             # sample_ids = tf.argmax(outputs, axis=-1, output_type=tf.int32)
        #             # sample_ids.eval(session=self.sess)
        #             finished1 = tf.greater_equal(time, tf.cast(2, tf.int32))
        #             # finished1 = tf.greater_equal(time, tf.cast(self.src_length/100, tf.int32))
        #             # finished2 = tf.equal(sample_ids, tf.constant([0]))
        #             # finished = tf.logical_or(finished1, finished2)
        #             idx = tf.reshape(tf.stack([tf.range(self.batch_size, dtype=tf.int32), sample_ids], axis=1), (-1, 2))
        #             next_inputs = tf.gather_nd(self.src, idx)
        #             return (finished1, next_inputs, state)
        #
        #         helper = tf.contrib.seq2seq.CustomHelper(initialize_fn, sample_fn, next_inputs_fn)
        #         decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)
        #         outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=decoder_scope)
        #         logits = outputs.rnn_output
        #         pointers = outputs.sample_id
        #
        # return logits, pointers


    def _build_decoder_cell(self, enc_outputs, enc_state, src_length):
        if self.params.time_major:
            memory = tf.transpose(enc_outputs, [1,0,2])
        else:
            memory = enc_outputs


        cell = tf.contrib.rnn.LSTMBlockCell(self.params.num_units)

        cell = PointerWrapper(cell, self.params.num_units, memory, memory_sequence_length=src_length, name="attention")

        decoder_initial_state = cell.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state = enc_state)
        return cell, decoder_initial_state

    def _compute_loss(self,logits):
        trgt = self.trgt_out
        max_time = tf.shape(self.src)[1]
        multilabel = tf.reduce_sum(tf.one_hot(trgt, max_time+1, axis=-1), 1)
        multilabel = multilabel[:,1:]
        crossent = tf.nn.sigmoid_cross_entropy_with_logits(labels=multilabel, logits=logits)
        loss = tf.reduce_sum(crossent) / tf.to_float(self.batch_size)
        return loss


    def load_model(self):
        latest_ckpt = tf.train.latest_checkpoint(self.params.model_dir)
        self.saver.restore(self.sess, latest_ckpt)

    def save(self, global_step):
        self.saver.save(self.sess, self.params.model_dir, global_step=global_step)

    def train_step(self):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        return self.sess.run({'update': self.update, 'loss':self.loss, 'global_step':self.global_step, 'summary':self.train_summary}, {self.batch_size: self.params.batch_size})

    def eval(self):
        assert self.mode == tf.contrib.learn.ModeKeys.EVAL
        return self.sess.run({'true': self.trgt_out, 'predicted':self.predicted, 'loss':self.loss}, feed_dict={self.batch_size: self.params.batch_size})

    def infer(self):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return self.sess.run({'true': self.trgt_out, 'predicted':self.sample_ids}, feed_dict={self.batch_size: self.params.batch_size})


