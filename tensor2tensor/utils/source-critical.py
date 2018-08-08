#!/usr/bin/env python
# encoding: utf-8
import tensorflow as tf
from tensorflow.python.ops import init_ops
import time
import os
import operator
from new_model import newmodel
import numpy as np

import sys
sys.path.append("../")
from models import models
from data_generators import problem_hparams
from utils import registry
import os

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("registry_help", False,
                  "If True, logs the contents of the registry and exits.")
flags.DEFINE_string("output_dir", "", "Base output directory for run.")
flags.DEFINE_string("model", "", "Which model to use.")
flags.DEFINE_string("hparams_set", "", "Which parameters to use.")
flags.DEFINE_string("hparams_range", "", "Parameters range.")
flags.DEFINE_string(
    "hparams", "",
    """A comma-separated list of `name=value` hyperparameter values. This flag
    is used to override hyperparameter settings either when manually selecting
    hyperparameters or when using Vizier. If a hyperparameter setting is
    specified by this flag then it must be a valid hyperparameter name for the
    model.""")
flags.DEFINE_string("problems", "", "Dash separated list of problems to "
                    "solve.")
flags.DEFINE_string("data_dir", "/tmp/data", "Directory with training data.")
flags.DEFINE_integer("train_steps", 250000,
                     "The number of steps to run training for.")
flags.DEFINE_integer("eval_steps", 10, "Number of steps in evaluation.")
flags.DEFINE_bool("eval_print", False, "Print eval logits and predictions.")
flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "How many recent checkpoints to keep.")
flags.DEFINE_bool("experimental_optimize_placement", False,
                  "Optimize ops placement with experimental session options.")
flags.DEFINE_string("decode_from_file", None, "Path to decode file")
flags.DEFINE_string("decode_to_file", None, "Path to inference output file")

flags.DEFINE_integer("decode_beam_size", 1, "The beam size for beam decoding")

flags.DEFINE_bool("self-critical", True, "rl.")

def train(hparams):
    def initializer():
        if hparams.initializer == "orthogonal":
            tf.logging.info("orthogonal")
            return tf.orthogonal_initializer(gain=hparams.initializer_gain)
        elif hparams.initializer == "uniform":
            max_val = 0.1 * hparams.initializer_gain
            return tf.random_uniform_initializer(-max_val, max_val)
        elif hparams.initializer == "normal_unit_scaling":
            return init_ops.variance_scaling_initializer(
                hparams.initializer_gain, mode="fan_avg", distribution="normal")
        elif hparams.initializer == "uniform_unit_scaling":
            return init_ops.variance_scaling_initializer(
                hparams.initializer_gain, mode="fan_avg", distribution="uniform")
        else:
            raise ValueError("Unrecognized initializer: %s" % hparams.initializer)

    hparams_list = []
    hparams_set_list = [hp.strip() for hp in FLAGS.hparams_set.split(",")]
    for hparams_set in hparams_set_list:
        hparams_ = create_hparams(hparams_set, data_dir=FLAGS.data_dir)
        hparams_list.append(hparams_)

    graph = tf.Graph()
    with graph.as_default() as g:
        inputs_ph = tf.placeholder(tf.int32, [None, None, 1, 1])
        targets_ph = tf.placeholder(tf.int32, [None, None, 1, 1])
        features = {"inputs": inputs_ph,
            "targets": targets_ph,
            "problem_choice": np.array(0).astype(np.int32),
            "input_space_id": np.array(4).astype(np.int32),
            "target_space_id": np.array(9).astype(np.int32)}
        with tf.variable_scope('',reuse=tf.AUTO_REUSE):
            model_rl = newmodel(hparams=hparams, mode=tf.contrib.learn.ModeKeys.TRAIN,
                               problem_hparams=hparams.problems[0],hparams_list=hparams_list)
            sample, r, b = model_rl.self_critic(features)
            reward = tf.placeholder(tf.float32, [None])
            baseline = tf.placeholder(tf.float32, [None])
            global_step, total_loss, train_op = model_rl.train(features,reward, baseline)
        saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.Session(config=sess_config, graph=graph) as sess:
        try:
            saver.restore(sess, tf.train.latest_checkpoint(hparams.output_dir))
            tf.logging.info('restore model from %s' % hparams.output_dir)
        except:
            sess.run(tf.global_variables_initializer())

        for epoch in range(1, hparams.train_steps + 1):
            train_input_fn = _train_batch_input_fn(hparams)
            tf.logging.info('epoch: {0}\n'.format(epoch))
            for batch in train_input_fn:
                start_time = time.time()

                inputs = batch['inputs']
                outputs = batch['outputs']
                feed_dict = {inputs_ph: inputs,targets_ph:outputs}
                sample_1, r_1, b_1 = sess.run([sample, r, b], feed_dict)
                feed_dict = {inputs_ph: inputs, targets_ph: sample_1, reward: r_1,
                                              baseline: b_1}
                step,total_loss,_ = sess.run([global_step,total_loss,train_op],
                                             feed_dict)

                start_time = time.time()
                # Train normal instances.
                tf.logging.info(
                    'model: {0}\tstep: {1}\tloss: {2:.4f}\ttime: {3:.4f}'.format
                        ('model_1', step, total_loss, time.time() - start_time))

                if step % 10 == 0:
                    saver.save(sess, FLAGS.output_dir, global_step=step)

        tf.logging.info("Finish training.")


def _train_batch_input_fn(hparams):
    problem_id = 0
    vocabulary_inputs = hparams.problems[problem_id].vocabulary["inputs"]
    vocabulary_targets = hparams.problems[problem_id].vocabulary["targets"]

    filename_source = FLAGS.data_dir + 'test.zh'
    filename_target = FLAGS.data_dir + 'test.en'

    inputs_list = [line.strip() for line in tf.gfile.Open(filename_source)]
    targets_list = [line.strip() for line in tf.gfile.Open(filename_target)]

    print(len(inputs_list), len(targets_list))
    num_train_batches = (len(inputs_list) - 1) // hparams.batch_size + 1
    print(" train batch %d" % num_train_batches)

    # all_train_input_fn = []
    for b in range(num_train_batches):
        batch_length_input = 0
        batch_length_target = 0
        batch_inputs = []
        batch_targets = []
        for inputs, targets in zip(
                inputs_list[b * hparams.batch_size:(b + 1) * hparams.batch_size],
                targets_list[b * hparams.batch_size:(b + 1) * hparams.batch_size]):
            input_ids = vocabulary_inputs.encode(inputs)
            target_ids = vocabulary_targets.encode(targets)
            input_ids.append(1)  # Assuming EOS=1.
            target_ids.append(1)  # Assuming EOS=1.
            batch_inputs.append(input_ids)
            batch_targets.append(target_ids)
            if len(input_ids) > batch_length_input:
                batch_length_input = len(input_ids)
            if len(target_ids) > batch_length_target:
                batch_length_target = len(target_ids)

        final_batch_inputs = []
        for input_ids in batch_inputs:
            assert len(input_ids) <= batch_length_input
            x = input_ids + [0] * (batch_length_input - len(input_ids))
            final_batch_inputs.append(x)

        final_batch_targets = []
        for target_ids in batch_targets:
            assert len(target_ids) <= batch_length_target
            y = target_ids + [0] * (batch_length_target - len(target_ids))
            final_batch_targets.append(y)

        if final_batch_inputs == [] and final_batch_targets == []:
            final_batch_inputs = [[]]
            final_batch_targets = [[]]

        tmp_batch = {
            "inputs": np.array(final_batch_inputs)[:, :, np.newaxis, np.newaxis].astype(np.int32),
            "outputs": np.array(final_batch_targets)[:, :, np.newaxis, np.newaxis].astype(np.int32),
        }
        yield tmp_batch

def create_hparams(params_id, data_dir):
    hparams = registry.hparams(params_id)()

    # Command line flags override any of the preceding hyperparameter values.
    if FLAGS.hparams:
        hparams = hparams.parse(FLAGS.hparams)
    hparams.add_hparam("data_dir", data_dir)

    # Add hparams for the problems
    hparams.problems = [
      problem_hparams.problem_hparams(problem, hparams)
      for problem in FLAGS.problems.split("-")
    ]
    return hparams

def run(data_dir, model, output_dir, train_steps, eval_steps, schedule):
    hparams = create_hparams(FLAGS.hparams_set, data_dir)

    hparams.add_hparam("output_dir", output_dir)
    hparams.add_hparam("train_steps", train_steps)
    hparams.add_hparam("eval_steps", eval_steps)
    hparams.add_hparam("decode_beam_size", FLAGS.decode_beam_size)

    train(hparams)
    # decode(hparams)

def log_registry():
  if FLAGS.registry_help:
    tf.logging.info(registry.help_string())
    sys.exit(0)

def validate_flags():
  if not FLAGS.model:
    raise ValueError("Must specify a model with --model.")
  if not FLAGS.problems:
    raise ValueError("Must specify a set of problems with --problems.")
  if not (FLAGS.hparams_set or FLAGS.hparams_range):
    raise ValueError("Must specify either --hparams_set or --hparams_range.")
  if not FLAGS.schedule:
    raise ValueError("Must specify --schedule.")
  if not FLAGS.output_dir:
    FLAGS.output_dir = "/tmp/tensor2tensor"
    tf.logging.warning("It is strongly recommended to specify --output_dir. "
                       "Using default output_dir=%s.", FLAGS.output_dir)

def main(_):
    log_registry()
    validate_flags()
    run(data_dir=FLAGS.data_dir,
        model=FLAGS.model,
        output_dir=FLAGS.output_dir,
        train_steps=FLAGS.train_steps,
        eval_steps=FLAGS.eval_steps,
        schedule=FLAGS.schedule)


if __name__ == '__main__':
    tf.app.run()

