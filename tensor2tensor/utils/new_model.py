import tensorflow as tf
import numpy as np
import math
import copy

import sys
sys.path.append("../")
from models import transformer


flags = tf.flags
FLAGS = flags.FLAGS
# Distributed training flags
flags.DEFINE_string("master", "", "Address of TensorFlow master.")
flags.DEFINE_string("schedule", "local_run",
                    "Method of tf.contrib.learn.Experiment to run.")
flags.DEFINE_integer("local_eval_frequency", 2000,
                     "Run evaluation every this steps during local training.")
flags.DEFINE_bool("locally_shard_to_cpu", False,
                  "Use CPU as a sharding device runnning locally. This allows "
                  "to test sharded model construction on a machine with 1 GPU.")
flags.DEFINE_bool("daisy_chain_variables", True,
                  "copy variables around in a daisy chain")
flags.DEFINE_bool("sync", False, "Sync compute on PS.")
flags.DEFINE_string("worker_job", "/job:worker", "name of worker job")
flags.DEFINE_integer("worker_gpu", 1, "How many GPUs to use.")
flags.DEFINE_integer("worker_replicas", 1, "How many workers to use.")
flags.DEFINE_integer("worker_id", 0, "Which worker task are we.")
flags.DEFINE_integer("ps_gpu", 0, "How many GPUs to use per ps.")
flags.DEFINE_string("gpu_order", "", "Optional order for daisy-chaining gpus."
                    " e.g. \"1 3 2 4\"")
flags.DEFINE_string("ps_job", "/job:ps", "name of ps job")
flags.DEFINE_integer("ps_replicas", 0, "How many ps replicas.")

# Decode flags
flags.DEFINE_bool("decode_use_last_position_only", False,
                  "In inference, use last position only for speedup.")
flags.DEFINE_integer("decode_shards", 1, "How many shards to decode.")
flags.DEFINE_integer("decode_problem_id", 0, "Which problem to decode.")
flags.DEFINE_integer("decode_extra_length", 30, "Added decode length.")
flags.DEFINE_integer("decode_batch_size", 32, "Batch size for decoding. "
                     "The decodes will be written to <filename>.decodes in"
                     "format result\tinput")
flags.DEFINE_float("decode_alpha", 1.0, "Alpha for length penalty")
flags.DEFINE_bool("decode_return_beams", False,
                  "Whether to return 1 (False) or all (True) beams. The \n "
                  "output file will have the format "
                  "<beam1>\t<beam2>..\t<input>")

class newmodel(transformer.Transformer):
    def __init__(self,
                 hparams,
                 mode,
                 problem_hparams,
                 problem_idx=0,
                 data_parallelism=None,
                 ps_devices=None,
                 hparams_list=None):
        super(newmodel, self).__init__(hparams, mode, problem_hparams,hparams_list=hparams_list)
        hparams = copy.copy(hparams)
        hparams.add_hparam("mode", mode)
    def train(self, features,reward,baseline):

        def learning_rate_decay(step):
            """Inverse-decay learning rate until warmup_steps, then decay."""
            warmup_steps = tf.to_float(
                hparams.learning_rate_warmup_steps * FLAGS.worker_replicas)
            step = tf.to_float(step)
            if hparams.learning_rate_decay_scheme == "noam":
                return 5000.0 * hparams.hidden_size ** -0.5 * tf.minimum(
                    (step + 1) * warmup_steps ** -1.5, (step + 1) ** -0.5)
            elif hparams.learning_rate_decay_scheme == "exp100k":
                return 0.94 ** (step // 100000)

            inv_base = tf.exp(tf.log(0.01) / warmup_steps)
            inv_decay = inv_base ** (warmup_steps - step)
            if hparams.learning_rate_decay_scheme == "sqrt":
                decay = _sqrt_decay(step - warmup_steps)
            elif hparams.learning_rate_decay_scheme == "exp10k":
                decay = _exp_decay_after(step - warmup_steps, 0.9995,
                                         FLAGS.train_steps - warmup_steps - 10000)
            elif hparams.learning_rate_decay_scheme == "exp50k":
                decay = _exp_decay_after(step - warmup_steps, 0.99995,
                                         FLAGS.train_steps - warmup_steps - 50000)
            elif hparams.learning_rate_decay_scheme == "exp500k":
                decay = _exp_decay_after(step - warmup_steps, 0.9999955,
                                         FLAGS.train_steps - warmup_steps - 500000)
            elif hparams.learning_rate_decay_scheme == "none":
                decay = tf.constant(1.0)
            else:
                raise ValueError("Unrecognized learning rate decay scheme: %s" %
                                 hparams.learning_rate_decay_scheme)
            return tf.cond(
                step < warmup_steps,
                lambda: inv_decay,
                lambda: decay,
                name="learning_rate_decay_warump_cond")

        hparams = self._hparams

        sharded_logits, training_loss, extra_loss = self.model_fn(features)
        rl_loss = tf.multiply(tf.reduce_sum(training_loss, axis=list(range(1, len(training_loss.get_shape())))),
                              baseline - reward)
        rl_loss = tf.reduce_sum(rl_loss)
        n = 0
        loss_moving_avgs = []
        with tf.variable_scope("losses_avg"):
            loss_moving_avgs.append(
                tf.get_variable(
                    "problem_%d/total_loss" % n, initializer=100.0, trainable=False))
            tf.get_variable(
                "problem_%d/training_loss" % n, initializer=100.0, trainable=False)
            tf.get_variable(
                "problem_%d/extra_loss" % n, initializer=100.0, trainable=False)

        with tf.variable_scope("losses_avg", reuse=True):
            loss_moving_avg = tf.get_variable("problem_%d/training_loss" % n)
            o1 = loss_moving_avg.assign(loss_moving_avg * 0.9 + rl_loss * 0.1)
            loss_moving_avg = tf.get_variable("problem_%d/extra_loss" % n)
            o2 = loss_moving_avg.assign(loss_moving_avg * 0.9 + extra_loss * 0.1)
            loss_moving_avg = tf.get_variable("problem_%d/total_loss" % n)
            total_loss = rl_loss + extra_loss
            o3 = loss_moving_avg.assign(loss_moving_avg * 0.9 + total_loss * 0.1)

        with tf.variable_scope("train_stats"):  # Count steps for this problem.
            problem_steps = tf.get_variable(
                "problem_%d_steps" % n, initializer=0, trainable=False)
            o4 = problem_steps.assign_add(1)
        with tf.control_dependencies([o1, o2, o3, o4]):  # Make sure the ops run.
            # Ensure the loss is a scalar here.
            total_loss = tf.reshape(total_loss, [], name="total_loss_control_id")

        result_list = [total_loss] + sharded_logits
        sharded_logits, total_loss = result_list[1:], result_list[0]

        self.global_step = tf.get_variable(name='global_step', dtype=tf.int64, shape=[],
                                           trainable=False, initializer=tf.zeros_initializer)

        with tf.name_scope("training_stats"):
            learning_rate = hparams.learning_rate * learning_rate_decay(self.global_step)
            learning_rate /= math.sqrt(float(FLAGS.worker_replicas))
            tf.summary.scalar("learning_rate", learning_rate)
            for n in range(len(hparams.problems)):
                with tf.variable_scope("losses_avg", reuse=True):
                    total_loss_var = tf.get_variable("problem_%d/total_loss" % n)
                    training_loss_var = tf.get_variable("problem_%d/training_loss" % n)
                    extra_loss_var = tf.get_variable("problem_%d/extra_loss" % n)
                tf.summary.scalar("loss_avg_%d/total_loss" % n, total_loss_var)
                tf.summary.scalar("loss_avg_%d/training_loss" % n, training_loss_var)
                tf.summary.scalar("loss_avg_%d/extra_loss" % n, extra_loss_var)
                with tf.variable_scope("train_stats", reuse=True):
                    nth_steps = tf.get_variable("problem_%d_steps" % n, dtype=tf.int32)
                tf.summary.scalar("problem_%d_frequency" % n,
                                  tf.to_float(nth_steps) / (tf.to_float(self.global_step) + 1.0))
        self.summary_op = tf.summary.merge_all()

        # Log trainable weights and add decay.
        total_size, weight_decay_loss = 0, 0.0
        all_weights = {v.name: v for v in tf.trainable_variables()}
        for v_name in sorted(list(all_weights)):
            v = all_weights[v_name]
            v_size = int(np.prod(np.array(v.shape.as_list())))
            # tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
            #                 v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size)
            total_size += v_size
            if hparams.weight_decay > 0.0 and len(v.shape.as_list()) > 1:
                # Add weight regularization if set and the weight is not a bias (dim>1).
                with tf.device(v._ref().device):  # pylint: disable=protected-access
                    v_loss = tf.nn.l2_loss(v) / v_size
                weight_decay_loss += v_loss
            is_body = len(v_name) > 5 and v_name[:5] == "body/"
            if hparams.weight_noise > 0.0 and is_body:
                # Add weight noise if set in hparams.
                with tf.device(v._ref().device):  # pylint: disable=protected-access
                    scale = learning_rate * 0.001
                    noise = tf.truncated_normal(v.shape) * hparams.weight_noise * scale
                    noise_op = v.assign_add(noise)
                with tf.control_dependencies([noise_op]):
                    total_loss = tf.identity(total_loss)
        tf.logging.info("Total trainable variables size: %d", total_size)
        if hparams.weight_decay > 0.0:
            total_loss += weight_decay_loss * hparams.weight_decay
        total_loss = tf.identity(total_loss, name="total_loss")
        self.loss = total_loss

        # Define the train_op for the TRAIN mode.
        opt = _ConditionalOptimizer(hparams.optimizer, learning_rate, hparams)
        tf.logging.info("Computing gradients for global model_fn.")
        self.train_op = tf.contrib.layers.optimize_loss(
            name="training",
            loss=total_loss,
            global_step=self.global_step,
            learning_rate=learning_rate,
            clip_gradients=hparams.clip_grad_norm or None,
            optimizer=opt,
            colocate_gradients_with_ops=True)

        tf.logging.info("Global model_fn finished.")
        return self.global_step, total_loss, self.train_op

    def eval(self):
        pass

    def decode(self, features):
        hparams = self._hparams
        for key in hparams.values():
            if key[-len("dropout"):] == "dropout":
                setattr(hparams, key, 0.0)
        self._hparams = hparams

        result_list = self.infer(
                        features,
                        beam_size=FLAGS.decode_beam_size,
                        top_beams=1,
                        last_position_only=FLAGS.decode_use_last_position_only,
                        alpha=FLAGS.decode_alpha,
                        decode_length=FLAGS.decode_extra_length)

        if not isinstance(result_list, dict):
            ret = {"outputs": result_list}
        else:
            ret = {
                      "outputs": result_list["outputs"],
                      "scores": result_list["scores"]
                  }
        if "inputs" in features:
            ret["inputs"] = features["inputs"]
        if "infer_targets" in features:
            ret["targets"] = features["infer_targets"]
        return ret

class _ConditionalOptimizer(tf.train.Optimizer):
  """Conditional optimizer."""

  def __init__(self, optimizer_name, lr, hparams):
    if optimizer_name == "Adam":
      # We change the default epsilon for Adam and re-scale lr.
      # Using LazyAdam as it's much faster for large vocabulary embeddings.
      self._opt = tf.contrib.opt.LazyAdamOptimizer(
          lr / 500.0,
          beta1=hparams.optimizer_adam_beta1,
          beta2=hparams.optimizer_adam_beta2,
          epsilon=hparams.optimizer_adam_epsilon)
    elif optimizer_name == "Momentum":
      self._opt = tf.train.MomentumOptimizer(
          lr, momentum=hparams.optimizer_momentum_momentum)
    else:
      self._opt = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](lr)

  def compute_gradients(self, loss, var_list, colocate_gradients_with_ops):
    return self._opt.compute_gradients(
        loss, var_list, colocate_gradients_with_ops=colocate_gradients_with_ops)

  def apply_gradients(self, gradients, global_step=None, name=None):
    return self._opt.apply_gradients(
        gradients, global_step=global_step, name=name)

def _sqrt_decay(step):
  """Decay like 1 / sqrt(step), multiplied by 500 to normalize."""
  return 500.0 / tf.sqrt(tf.maximum(step, 1.0))


def _exp_decay_after(step, rate, from_which_step):
  """Decay exponentially by rate (per step) starting at from_which_step."""
  return tf.cond(
      step < from_which_step,
      lambda: tf.constant(1.0),
      lambda: rate**(step - from_which_step),
      name="exponential_decay_step_cond")

