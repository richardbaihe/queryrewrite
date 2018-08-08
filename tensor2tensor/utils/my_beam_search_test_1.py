# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for tensor2tensor.beam_search."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
# from tensor2tensor.utils import beam_search
# from tensor2tensor.models import t2t
from tensor2tensor.models import transformer

import sys
sys.path.append('../')
from data_generators import problem_hparams
from utils import my_beam_search

SRC_VOCAB_SIZE = 30003
VOCAB_SIZE = 40960
BATCH_SIZE = 1
#INPUT_LENGTH = 7
last_position_only = False
decode_length = 40
beam_size = 4

import tensorflow as tf
import os

class BeamSearchTest():
  def __init__(self, hparams, p_hparams):
      self.model = self.getModel(hparams, p_hparams)
      self.logits, self.session = self.getSession()
      self.targets_vocab = hparams.problems[0].vocabulary["targets"]

  def getModel(self, hparams, p_hparams):
      return transformer.Transformer(
               hparams, tf.estimator.ModeKeys.PREDICT, p_hparams)

  def getSession(self):
      self.inputs_ph = tf.placeholder(tf.int32, [None, None, 1, 1])
      self.targets_ph = tf.placeholder(tf.int32, [None, None, None, 1])
      features = {
          "inputs": self.inputs_ph,
          "targets": self.targets_ph,
          "target_space_id": tf.constant(1, dtype=tf.int32)
      }
      model1 = self.model
      shadred_logits1, _, _ = model1.model_fn(features, False, last_position_only=last_position_only)
      # tf.get_variable_scope().reuse_variables()
      # logits1 = tf.concat(shadred_logits1, 0)
      saver1 = tf.train.Saver(name='model_saver')

      session1 = tf.Session()
      ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/ycliu/checkpoint'))
      if ckpt and ckpt.model_checkpoint_path:
          saver1.restore(session1, ckpt.model_checkpoint_path)
          print('restore from checkpoint')
      return shadred_logits1, session1

  def testShapes(self,inputs):
    BATCH_SIZE = np.shape(inputs)[0]
    features = {
        "inputs": inputs,
        "targets": tf.zeros([BATCH_SIZE,1,1,1],dtype=tf.int32),
        "problem_choice":np.array(0,dtype=np.int32),
        "input_sapce_id":np.array(4,dtype=np.int32),
        "target_space_id": tf.constant(9, dtype=tf.int32)
    }

    # batch_size = tf.shape(features["inputs"])[0]
    initial_ids = tf.zeros([BATCH_SIZE], dtype=tf.int32)

    inputs_old = features["inputs"]
    features["inputs"] = np.expand_dims(features["inputs"], 1)
    if len(features["inputs"].shape) < 5:
        features["inputs"] = np.expand_dims(features["inputs"], 4)
    # Expand the inputs in to the beam size.
    features["inputs"] = np.tile(features["inputs"], [1, beam_size, 1, 1, 1])
    s = np.shape(features["inputs"])
    features["inputs"] = np.reshape(features["inputs"],
                                    [s[0] * s[1], s[2], s[3], s[4]])
    # print('shape of features:', features["inputs"].shape)

    def symbols_to_logits_fn(ids):
      """Go from ids to logits."""
      ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])
      with tf.Session() as sess:
          ids = ids.eval()
      if ids.shape[1] > 1:
          for k in range(beam_size):
               print('seq_vocab %d' % k,self.targets_vocab.decode(np.squeeze(ids)[k]))
      # print('ids',type(ids),ids.shape)
      features["targets"] = ids
      feed = {self.inputs_ph: features['inputs'], self.targets_ph: ids}
      sharded_logits = self.session.run(self.logits, feed_dict=feed)
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
      logits = sharded_logits[0]  # Assuming we have one shard.
      # print('the first ten  of logits is : ', logits[:15])
      # print('the sum of logits is : ', np.sum(np.squeeze(logits)))
      # print('the sum of exp logits is : ', np.sum(np.exp(np.squeeze(logits))))
      if last_position_only:
          # print('logits', tf.squeeze(logits, axis=[1, 2, 3]).shape)
          return tf.squeeze(logits, axis=[1, 2, 3])
      current_output_position = np.shape(ids)[1]-1  # -1 due to the pad above.
      #print('current_output_position:',current_output_position)
      #print('logits:',logits.shape)
      logits = logits[:, current_output_position, :, :]
      return tf.squeeze(logits, axis=[1, 2])

    # decode_length = tf.shape(features["inputs"])[1] + tf.constant(decode_length)
    final_ids, final_probs = my_beam_search.beam_search(
        symbols_to_logits_fn, initial_ids, beam_size, decode_length, VOCAB_SIZE,
        0.6)

    final_output,final_prob = final_ids,final_probs
    print('the shape of final_output is : ',final_output.shape,final_prob.shape)
    for k in range(beam_size):
        print(self.targets_vocab.decode(final_output[0,k,:]),final_probs[:,k])
    result = {}
    for (each_inputs, each_outputs) in zip(features["inputs"], final_output[0]):
    	result = {"inputs": each_inputs, "outputs":each_outputs[1:]}
        yield result

if __name__ == "__main__":
  tf.test.main()
