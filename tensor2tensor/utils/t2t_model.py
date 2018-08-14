# Copyright 2017 Google Inc.
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

"""T2TModel Base Class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
import numpy as np
# Dependency imports

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import sys
sys.path.append("../")
from utils import beam_search
#from tensor2tensor.utils import expert_utils as eu

#from tensor2tensor.utils import registry
from models import common_layers
import tensorflow as tf
from utils import registry
from utils import expert_utils as eu
from utils import modality
def _with_timing(fn, msg):

  def fn_with_timing(*args, **kwargs):
    start_time = time.time()
    res = fn(*args, **kwargs)
    tf.logging.info("Doing %s took %.3f sec." % (msg, time.time() - start_time))
    return res

  return fn_with_timing


def _is_class_modality(mod):
  # TODO(lukaszkaiser): should be based on type, like CLASS_LABEL, not string.
  prefix = "class_label_modality_"
  try:
    if len(mod.name) < len(prefix):
      return False
    return mod.name[:len(prefix)] == prefix
  except:
    if len(mod) < len(prefix):
      return False
    return mod[:len(prefix)] == prefix


class T2TModel(object):
  """Abstract base class for models.

  Subclassess generally only need to override `build_model`.
  """

  def __init__(self,
               hparams,
               mode,
               problem_hparams,
               problem_idx=0,
               data_parallelism=None,
               ps_devices=None,
	       hparams_list=None):
    """Create a T2TModel.

    Args:
      hparams: a hyperparameters object.
      mode: The execution mode, as defined in tf.contrib.learn.ModeKeys.
      problem_hparams: a hyperparameters object.
      problem_idx: an integer.
      data_parallelism: a expert_utils.parallelism
        (specifies devices for data parallelism).
      ps_devices: a list of devices to be used for experts

    Returns:
      a T2TModel
    """
    if data_parallelism is None:
      data_parallelism = eu.Parallelism([""])
    if ps_devices is None:
      ps_devices = [""]
    for i in range(len(hparams_list)):
      if getattr(hparams_list[i],"mode", None) is not None:
        delattr(hparams_list[i],"mode")
      hparams_list[i].add_hparam("mode", mode)
      if mode != tf.contrib.learn.ModeKeys.TRAIN:
        for key in hparams_list[i].values():
          if key[-len("dropout"):] == "dropout":
            setattr(hparams_list[i], key, 0.0)
    self._hparams_list = hparams_list
    #hparams = copy.copy(hparams)
    #hparams.add_hparam("mode", mode)
    # when not in training mode, set all forms of dropout to zero.
    #if mode != tf.contrib.learn.ModeKeys.TRAIN:
    #  for key in hparams.values():
    #    if key[-len("dropout"):] == "dropout":
    #      setattr(hparams, key, 0.0)
    hparams = hparams_list[0]
    self._hparams = hparams
    self._data_parallelism = data_parallelism
    self._num_datashards = data_parallelism.n
    self._ps_devices = ps_devices
    self._problem_idx = problem_idx
    problem_hparams_list = [copy.copy(problem_hparams) for i in range(len(hparams_list))]
    # print('the initial problem_hparams is :', problem_hparams)
    # print('the initial hparams is :', hparams)
    for i in range(len(hparams_list)):
      #self._problem_hparams = problem_hparams
      #self._create_modalities(problem_hparams, hparams)
      self._create_modalities(problem_hparams_list[i], hparams_list[i])
    self._problem_hparams_list = problem_hparams_list
    # print('problem_hparams_list : ', self._problem_hparams_list)
    #for i in range(len(hparams_list)):
    #  print('problem_hparams %d' % i, self._problem_hparams_list[i])

  def _create_modalities(self, problem_hparams, hparams):
    """Construct modalities in problem_hparams."""
    #print('_create_modalities : ',hparams)
    input_modality_overrides = {}
    for override_str in hparams.input_modalities.split(";"):
      parts = override_str.split(":")
      feature_name = parts[0]
      modality_name = ":".join(parts[1:])
      input_modality_overrides[feature_name] = modality_name

    target_modality_name = None
    if hparams.target_modality:
      target_modality_name = hparams.target_modality

    input_modality = {}
    for f, modality_spec in six.iteritems(problem_hparams.input_modality):
      if isinstance(modality_spec, modality.Modality):
        return
      if f in input_modality_overrides:
        _warn_changed_modality_type(input_modality_overrides[f],
                                    modality_spec[0], f)
        modality_spec = (input_modality_overrides[f], modality_spec[1])
      input_modality[f] = registry.create_modality(modality_spec, hparams)
    problem_hparams.input_modality = input_modality

    target_modality_spec = problem_hparams.target_modality
    if isinstance(target_modality_spec, modality.Modality):
      return
    if target_modality_name:
      _warn_changed_modality_type(target_modality_name, target_modality_spec[0],
                                  "target")
      target_modality_spec = (target_modality_name, target_modality_spec[1])
    target_modality = registry.create_modality(target_modality_spec, hparams)
    problem_hparams.target_modality = target_modality
    # print('the new problem_hparams is :', problem_hparams)


  @property
  def has_input(self):
    return self._problem_hparams_list[0].input_modality

  def infer(self,
            features=None,
            decode_length=50,
            beam_size=1,
            top_beams=1,
            last_position_only=False,
            alpha=1.0):
    """A inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      last_position_only: a boolean, speed-up by computing last position only.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`.
    """
    if not self.has_input:
      # since there is no input, it is more interesting to see randomly
      # generated sequences, than to see the most likely sequence repeatedly.
      beam_size = 1
      self._hparams.sampling_method = "random"
    #if _is_class_modality(
    #    self._hparams.problems[self._problem_idx].target_modality):
    #  beam_size = 1  # No use to run beam-search for a single class.
    if beam_size == 1:
      tf.logging.info("Greedy Decoding")
      return self._greedy_infer(features, decode_length, last_position_only)
    else:
      tf.logging.info("Beam Decoding with beam size %d" % beam_size)
      return self._beam_decode(features, decode_length, beam_size, top_beams,
                               last_position_only, alpha)

  def _beam_decode(self, features, decode_length, beam_size, top_beams,
                   last_position_only, alpha, ensemble_num=1):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      last_position_only: a boolean, speed-up by computing last position only.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.

    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    """
    tf.logging.info('we use this beam_search')
    target_modality = self._hparams.problems[self._problem_idx].target_modality
    vocab_size = 84000#target_modality.top_dimensionality
    def symbols_to_logits_fn(ids):
      """Go from ids to logits."""
      ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])

      features["targets"] = ids
      self._coverage = None
      logits = tf.zeros([1,1,1,1,vocab_size],dtype=tf.float32)
      for i in range(ensemble_num):
          tf.logging.info('the %dth model_fn' % (i+1))
          #with tf.variable_scope("graph_%d" % (i+1)):            
          sharded_logits, _, _ = self.model_fn(
                  features, False, last_position_only=last_position_only, hparams=self._hparams_list[i], num=i)
      # now self._coverage is a coverage tensor for the first datashard.
      # it has shape [batch_size] and contains floats between 0 and
      # source_length.
          logits += sharded_logits[0]  # Assuming we have one shard.
      logits /= ensemble_num
      if last_position_only:
        return tf.squeeze(logits, axis=[1, 2, 3])
      current_output_position = tf.shape(ids)[1] - 1  # -1 due to the pad above.
      logits = logits[:, current_output_position, :, :]
      return tf.squeeze(logits, axis=[1, 2])

    batch_size = tf.shape(features["inputs"])[0]
    initial_ids = tf.zeros([batch_size], dtype=tf.int32)

    inputs_old = features["inputs"]
    features["inputs"] = tf.expand_dims(features["inputs"], 1)
    if len(features["inputs"].shape) < 5:
      features["inputs"] = tf.expand_dims(features["inputs"], 4)
    # Expand the inputs in to the beam size.
    features["inputs"] = tf.tile(features["inputs"], [1, beam_size, 1, 1, 1])
    s = tf.shape(features["inputs"])
    features["inputs"] = tf.reshape(features["inputs"],
                                    [s[0] * s[1], s[2], s[3], s[4]])
    
    #print('the inputs of feature in beam_search is :', tf.shape(features["inputs"])[3])
    #target_modality = self._hparams.problems[self._problem_idx].target_modality
    #vocab_size = target_modality.top_dimensionality
    # Setting decode length to input length + decode_length
    decode_length = tf.shape(features["inputs"])[1] + tf.constant(decode_length)
    ids, scores = beam_search.beam_search(symbols_to_logits_fn, initial_ids,
                                          beam_size, decode_length, vocab_size,
                                          alpha)

    # Set inputs back to the unexpanded inputs to not to confuse the Estimator!
    features["inputs"] = inputs_old

    # Return `top_beams` decodings (also remove initial id from the beam search)
    return_scores = True  # TODO(lukaszkaiser): make it work multi-problem.
    if top_beams == 1:
      if return_scores:
        return {"outputs": ids[:, 0, 1:], "scores": scores}
      return ids[:, 0, 1:]
    else:
      if return_scores:
        return {"outputs": ids[:, :top_beams, 1:], "scores": scores}
      return ids[:, :top_beams, 1:]

  def _greedy_infer(self, features, decode_length, last_position_only):
    """A slow greedy inference method.

    Quadratic time in decode_length.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      last_position_only: a boolean, speed-up by computing last position only.

    Returns:
       samples: an integer `Tensor`.
    """
    if not features:
      features = {}
    inputs_old = None
    if "inputs" in features and len(features["inputs"].shape) < 4:
      inputs_old = features["inputs"]
      features["inputs"] = tf.expand_dims(features["inputs"], 2)
    if not self.has_input:
      features["partial_targets"] = tf.to_int64(features["inputs"])

    def infer_step(recent_output, _):
      """Inference step."""
      recent_output.set_shape([None, None, None, 1])
      padded = tf.pad(recent_output, [[0, 0], [0, 1], [0, 0], [0, 0]])
      features["targets"] = padded
      # This is inefficient in that it generates samples at all timesteps,
      # not just the last one, except if last_position_only is set (dangerous).
      samples = self.sample(features, last_position_only=last_position_only)
      # Concatenate the already-generated recent_output with last timestep
      # of the newly-generated samples.
      if last_position_only:
        cur_sample = samples[:, -1, :, :]
      else:
        cur_sample = samples[:, tf.shape(recent_output)[1], :, :]
      cur_sample = tf.to_int64(tf.expand_dims(cur_sample, axis=1))
      samples = tf.concat([recent_output, cur_sample], axis=1)
      samples.set_shape([None, None, None, 1])
      return samples

    # Create an initial output tensor. This will be passed
    # to the infer_step, which adds one timestep at every iteration.
    if "partial_targets" in features:
      initial_output = tf.convert_to_tensor(features["partial_targets"])
    else:
      batch_size = tf.shape(features["inputs"])[0]
      initial_output = tf.zeros((batch_size, 0, 1, 1), dtype=tf.int64)
    # Hack: foldl complains when the output shape is less specified than the
    # input shape, so we confuse it about the input shape.
    initial_output = tf.slice(initial_output, [0, 0, 0, 0],
                              tf.shape(initial_output))
    if _is_class_modality(
        self._hparams.problems[self._problem_idx].target_modality):
      decode_length = 1
    else:
      decode_length = tf.shape(features["inputs"])[1] + decode_length
    result = tf.foldl(
        infer_step,
        tf.range(decode_length),
        initializer=initial_output,
        back_prop=False,
        parallel_iterations=1)
    if inputs_old is not None:  # Restore to not confuse Estimator.
      features["inputs"] = inputs_old
    return result

  def sample(self, features, last_position_only=False):
    """Run the model and extract samples.

    Args:
      features: an map of string to `Tensor`.
      last_position_only: a boolean, speed-up by computing last position only.

    Returns:
       samples: an integer `Tensor`.
    """
    sharded_logits, _, _ = self.model_fn(
        features, False, last_position_only=last_position_only)
    if self._hparams.sampling_method == "argmax":
      sharded_samples = self._data_parallelism(tf.argmax, sharded_logits, 4)
    else:
      assert self._hparams.sampling_method == "random"

      def _multinomial_squeeze(logits):
        reshaped_logits = tf.reshape(logits, [-1, tf.shape(logits)[-1]])
        choices = tf.multinomial(reshaped_logits, 1)
        choices = tf.reshape(choices,
                             tf.shape(logits)[:logits.get_shape().ndims - 1])
        return choices

      sharded_samples = self._data_parallelism(_multinomial_squeeze,
                                               sharded_logits)
    return tf.concat(sharded_samples, 0)

  def _shard_features(self, features):  # pylint: disable=missing-docstring
    sharded_features = dict()
    for k, v in six.iteritems(features):
      v = tf.convert_to_tensor(v)
      if not v.shape.as_list():
        v = tf.expand_dims(v, axis=-1)
        v = tf.tile(v, [self._num_datashards])
      sharded_features[k] = self._data_parallelism(tf.identity,
                                                   tf.split(
                                                       v, self._num_datashards,
                                                       0))
    return sharded_features
  def self_critic(self,
            features,
            decode_length=50,
            last_position_only=False,
            skip=False):
    def entity_keep(predictions,labels,origin_entities):
      # dict2num = [78003 + x for x in range(len(open('./oov.en', 'rU').readlines()))]
      padded_predictions, padded_labels = common_layers.pad_with_zeros(predictions, labels)
      entities = tf.tile(origin_entities,[1,1,tf.shape(padded_predictions)[1],1]) # batch num_enti len 3
      batch_size = tf.shape(entities)[0]
      length_size = tf.shape(entities)[-2]
      entities_size = tf.shape(entities)[-1]
      entities = tf.transpose(entities,perm=[0,1,3,2]) # batch num_enti 3 len
      outputs = tf.expand_dims(tf.to_int32(tf.argmax(padded_predictions, axis=-1)),-1)
      outputs = tf.transpose(outputs,perm=[0,2,3,1]) #batch 1 1 len
      outputs = tf.tile(outputs, [1, tf.shape(entities)[1], entities_size, 1]) # batch num_enti 3 len

      axis = list(range(1, len(outputs.get_shape())))
      outputs_mask = tf.to_int32(tf.equal(outputs, entities))
      outputs_nums = tf.sign(tf.reduce_sum(outputs_mask, axis=[-2])) # batch num_enti len
      entity_sums = tf.reduce_sum(tf.sign(origin_entities),axis=[-1]) # batch num_enti

      tri_diagonal = tf.eye(length_size,batch_shape=batch_size)
      tri_diagonal = tri_diagonal + tf.manip.roll(tri_diagonal,shift=1,axis=1) + \
                     tf.manip.roll(tri_diagonal, shift=2, axis=1)
      tri_diagonal = tf.slice(tri_diagonal,[0,0,0],[batch_size,length_size,length_size-(entities_size-1)])

      outputs_nums = tf.reduce_max(tf.matmul(outputs_nums, tri_diagonal),axis=-1)
      outputs_mask = tf.to_int32(tf.equal(outputs_nums, entity_sums)) # batch num_enti

      oov_ouputs_nums = tf.sign(tf.reduce_sum(outputs_mask, axis=-1))
      oov_entities_nums = tf.sign(tf.reduce_sum(origin_entities+1, axis=axis))
      total = tf.cast(oov_entities_nums, dtype=tf.float64)
      scores = tf.cast(tf.subtract(oov_entities_nums, oov_ouputs_nums),dtype=tf.float64)
      return tf.to_float(tf.div(scores, total))

      # label_word = tf.ones_like(padded_labels-5000) * word  # batch*len*1*1
      # label_mask = tf.to_int32(tf.equal(padded_labels-5000, label_word))
      # outputs = tf.to_int32(tf.argmax(padded_predictions, axis=-1))
      # axis = list(range(1, len(outputs.get_shape())))
      # outputs_word = tf.ones_like(outputs) * word  # batch*len*1*1
      # outputs_mask = tf.to_int32(tf.equal(outputs, outputs_word))
      # oov_ouputs_nums = tf.reduce_sum(outputs_mask, axis=axis)
      # oov_label_nums = tf.reduce_sum(label_mask, axis=axis)
      # temp = tf.subtract(oov_label_nums, oov_ouputs_nums)
      # total.append(tf.cast(oov_label_nums, dtype=tf.float64))
      # scores.append(tf.div(tf.cast(temp + tf.abs(temp), dtype=tf.float64), tf.convert_to_tensor(np.array(2.0))))
      # return tf.to_float(tf.div(tf.add_n(scores), tf.add_n(total)))
    new_features = features.copy()
    #new_features['inputs'] = features['sc_inputs']
    self._hparams.sampling_method ="random"
    sample_result = self._greedy_infer(features, decode_length, last_position_only)
    self._hparams.sampling_method = "argmax"
    greedy_result = self._greedy_infer(features, decode_length, last_position_only)
    reward = entity_keep(sample_result,features['inputs'],features['sc_inputs'])
    baseline = entity_keep(greedy_result,features['inputs'],features['sc_inputs'])
    new_features["targets"] = sample_result
    sharded_logits, training_loss, extra_loss = self.model_fn(new_features,skip=skip,reduce_sum=False)
    rl_loss = tf.multiply(tf.reduce_sum(training_loss,axis=list(range(1, len(training_loss.get_shape())))),tf.maximum(tf.zeros(tf.shape(reward),dtype=tf.float32),baseline-reward))
    rl_loss=tf.reduce_sum(rl_loss)
    return sharded_logits, rl_loss, extra_loss

  def model_fn(self, features, skip=False, last_position_only=False, hparams=None, num=0, reduce_sum=True):
    """Computes the entire model and produces sharded logits and training loss.

    Args:
      features: A dictionary of feature name to tensor.
      skip: a boolean, if we're just dummy-calling and actually skip this model
        (but we need to create variables to not confuse distributed training).
      last_position_only: a boolean, compute logits for only the last position.

    Returns:
      sharded_logits: a list of `Tensor`s, one per datashard.
      training_loss: a floating point `Scalar`.
    """
    start_time = time.time()
    dp = self._data_parallelism

    sharded_features = self._shard_features(features)
    # print('sharded_features in model_fn: ', sharded_features)
    # Construct the model bottom for inputs.
    transformed_features = {}
    all_previous_modalities = []

    for key, input_modality in six.iteritems(
        self._problem_hparams_list[num].input_modality):
      previous_modalities = [
          self._hparams_list[num].problems[i].input_modality[key].name
          for i in xrange(self._problem_idx)
      ]
      all_previous_modalities.extend(previous_modalities)
      # print('input_modality.name: ',input_modality.name)
      do_reuse = input_modality.name in all_previous_modalities
      with tf.variable_scope(input_modality.name, reuse=do_reuse):
        transformed_features[key] = input_modality.bottom_sharded(
            sharded_features[key], dp)
      all_previous_modalities.append(input_modality.name)

    # Target space id just gets copied to every shard.
    if "target_space_id" in features:
      transformed_features["target_space_id"] = [features["target_space_id"]
                                                ] * self._num_datashards

    # Targets are transformed by the autoregressive part of the modality
    previous_tgt_modalities = [
        self._hparams_list[num].problems[i].target_modality.name
        for i in xrange(self._problem_idx)
    ]
    all_previous_modalities.extend(previous_tgt_modalities)

    target_modality = self._problem_hparams_list[num].target_modality
    target_reuse = target_modality.name in previous_tgt_modalities
    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      transformed_features["targets"] = target_modality.targets_bottom_sharded(
          sharded_features["targets"], dp)
    # print('transformded_features',transformed_features)
    # Construct the model body.
    with tf.variable_scope("body", reuse=self._problem_idx > 0):
      if skip:
        body_outputs, extra_loss = transformed_features["targets"], 0.0
      else:
        body_outputs, extra_loss = self.model_fn_body_sharded(
            transformed_features, hparams=hparams)

    with tf.variable_scope(target_modality.name, reuse=target_reuse):
      if not last_position_only:
        sharded_logits, training_loss = (target_modality.top_sharded(
            body_outputs, sharded_features["targets"], self._data_parallelism))

        training_loss *= self._problem_hparams_list[num].loss_multiplier
      else:
        # Take body outputs for the last position only, and targets too.
        # TODO(lukaszkaiser): warning, this doesn't work for all modalities!
        last_position_body_outputs = [
            tf.expand_dims(body_shard[:, -1, :, :], axis=[1])
            for body_shard in body_outputs
        ]
        last_position_targets = [
            tf.expand_dims(target_shard[:, -1:, :, :], axis=[1])
            for target_shard in sharded_features["targets"]
        ]
        sharded_logits, training_loss = (target_modality.top_sharded(
            last_position_body_outputs, last_position_targets,
            self._data_parallelism))

        training_loss = None

    tf.logging.info("This model_fn took %.3f sec." % (time.time() - start_time))
    return sharded_logits, training_loss, extra_loss

  def model_fn_body_sharded(self, sharded_features, hparams=None):
    """Mixture-of-experts models will override this function.

    Compute model body on all datashards.

    Args:
      sharded_features: map from string to list of Tensors each with shape
         [batch, ?, ?, body_input_size]

    Returns:
      sharded_body_output:
          a list of Tensors, each with shape [batch, O, P, body_output_size]
      extra_loss: a Scalar.
   """ 
    
    print('this model fn use hparams: ', hparams)
    with tf.name_scope("model"):
      datashard_to_features = [{
          k: v[d]
          for k, v in six.iteritems(sharded_features)
      } for d in xrange(self._num_datashards)]
      output = self._data_parallelism(
          _with_timing(self.model_fn_body, "model_fn_body"),
          datashard_to_features,hparams=hparams)
      if isinstance(output, tuple):
        loss = tf.reduce_mean(output[1])
        output = output[0]
      else:
        loss = 0.0
      return output, loss

  def model_fn_body(self, features, hparams=None):
    """Most models will override this function.

    Compute label logits for one shard as a function of the transformed
    features.

    Args:
      features: A dictionary of key to Tensor.  Each Tensor has shape
         `[batch_size, ?, ?, hidden_size]`.

    Returns:
      a `Tensor` of logits with shape `[batch_size, O, P, body_output_size]`.
    """
    raise NotImplementedError("Abstract Method")

  @property
  def hparams(self):
    return self._hparams


def _warn_changed_modality_type(new_name, old_name, feature_name):
  new_type, new_name = registry.parse_modality_name(new_name)
  old_type, old_name = registry.parse_modality_name(old_name)
  if new_type != old_type:
    tf.logging.warning("%s has a designated modality type %s (%s) but has been "
                       "overriden with a modality of type %s (%s).",
                       feature_name, old_type, old_name, new_type, new_name)
