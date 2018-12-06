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
    def __init__(self, params, mode):
        self.params = params
        self.mode = mode

        # self.sess = tf.Session()

        # self.logits, self.loss, self.sample_ids= self.build_graph(mode)

        # trainable_vars = tf.trainable_variables()

        # self.opt = tf.train.AdamOptimizer(params.learning_rate)

        # gradients = tf.gradients(self.loss, trainable_vars)
        # self.clipped_grad , self.gradient_norm = tf.clip_by_global_norm(gradients, 5)
        # self.update = self.opt.apply_gradients(zip(self.clipped_grad, trainable_vars), global_step=self.global_step)

        # self.train_summary = tf.summary.merge([tf.summary.scalar('train_loss', self.loss)])

        # self.saver=tf.train.Saver(tf.global_variables(), max_to_keep=params.max_to_keep)
        # self.sess.run(tf.global_variables_initializer())


    def build_graph(self, batch_data):
        tf.logging.info("Building {} model...".format(self.mode))
        if self.mode == 'train':
            self.mode = tf.contrib.learn.ModeKeys.TRAIN
        elif self.mode == 'infer':
            self.mode = tf.contrib.learn.ModeKeys.INFER
        elif self.mode == 'eval':
            self.mode = tf.contrib.learn.ModeKeys.EVAL
        else:
            raise ValueError("Please choose mode: train, eval or infer")

        self.src, self.src_length, self.trgt_in, self.trgt_out, self.trgt_length = batch_data
        self.batch_size = tf.shape(self.src_length)[0]
        # print("aaaaa:", self.batch_size.eval(session=tf.Session()))
        enc_outputs, enc_state = self._build_encoder()
        self.logits, self.sample_ids = self._build_decoder(enc_outputs, enc_state)


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

            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                dec_inputs = self.trgt_in

                if self.params.time_major:
                    dec_inputs = tf.transpose(dec_inputs)

                helper = tf.contrib.seq2seq.TrainingHelper(dec_inputs, self.trgt_length)

                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper,  decoder_initial_state)

                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope = decoder_scope)
                logits = outputs.rnn_output
                sample_ids = outputs.sample_id

            else:
                def initialize_fn():
                    finished = tf.tile([False], [self.batch_size])
                    start_inputs = tf.fill([self.batch_size, self.params.input_dim], 0.0)
                    return (finished, start_inputs)

                def sample_fn(time, outputs, state):
                    del time, state
                    sample_ids =  tf.argmax(outputs, axis=-1, output_type=tf.int32)
                    return sample_ids

                def next_inputs_fn(time, outputs, state, sample_ids):
                    del outputs
                    finished1 = tf.greater(time, tf.cast(self.src_length/100, tf.int32))
                    finished2 = tf.equal(sample_ids, 0) #pointing to first input point
                    finished = tf.logical_or(finished1, finished2)
                    idx = tf.reshape(tf.stack([tf.range(self.batch_size, dtype=tf.int32), sample_ids], axis=1), (-1, 2))
                    next_inputs = tf.gather_nd(self.src, idx)
                    return (finished, next_inputs, state)

                helper = tf.contrib.seq2seq.CustomHelper(initialize_fn, sample_fn, next_inputs_fn)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell, helper, decoder_initial_state)
                outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, scope=decoder_scope)
                logits = outputs.rnn_output
                sample_ids = outputs.sample_id

        return logits, sample_ids


    def _build_decoder_cell(self, enc_outputs, enc_state, src_length):
        if self.params.time_major:
            memory = tf.transpose(enc_outputs, [1,0,2])
        else:
            memory = enc_outputs


        cell = tf.contrib.rnn.LSTMBlockCell(self.params.num_units)

        cell = PointerWrapper(cell, self.params.num_units, memory, memory_sequence_length=src_length, name="attention")

        decoder_initial_state = cell.zero_state(self.batch_size, dtype=tf.float32).clone(cell_state = enc_state)

        return cell, decoder_initial_state

    def compute_loss(self):
        trgt_out = self.trgt_out
        max_time = tf.shape(trgt_out)[1]
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=trgt_out, logits=self.logits)
        trgt_weight = tf.sequence_mask(self.trgt_length, max_time, dtype=self.logits.dtype)
        self.loss = tf.reduce_sum(crossent * trgt_weight) / tf.to_float(self.batch_size)
        return self.loss


    # def load_model(self):
    #     latest_ckpt = tf.train.latest_checkpoint(self.params.model_dir)
    #     self.saver.restore(self.sess, latest_ckpt)
    #
    # def save(self, global_step):
    #     self.saver.save(self.sess, self.params.model_dir, global_step=global_step)
    #
    # def train_step(self):
    #     assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    #     return self.sess.run({'update': self.update, 'loss':self.loss, 'global_step':self.global_step, 'summary':self.train_summary}, {self.batch_size: self.params.batch_size})
    #
    #
    # def eval(self):
    #     assert self.mode == tf.contrib.learn.ModeKeys.EVAL
    #     return self.sess.run({'true': self.trgt_out, 'predicted':self.sample_ids, 'loss':self.loss}, feed_dict={self.batch_size: self.params.batch_size})
    #
    # def infer(self):
    #     assert self.mode == tf.contrib.learn.ModeKeys.INFER
    #     return self.sess.run({'true': self.trgt_out, 'predicted':self.sample_ids}, feed_dict={self.batch_size: self.params.batch_size})


