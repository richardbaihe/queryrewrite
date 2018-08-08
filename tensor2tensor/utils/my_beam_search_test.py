from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
from tensor2tensor.models import transformer

import sys
sys.path.append('../')
from data_generators import problem_hparams
from utils import my_beam_search

VOCAB_SIZE = 84000
BATCH_SIZE = 32
last_position_only = False
decode_length_1 = 15
beam_size = 4
import tensorflow as tf
import os

class BeamSearchTest():
    def __init__(self, hparams, p_hparams, ensemble_num):
        self.model = self.getModel(hparams, p_hparams)
        self.ensemble_num = ensemble_num
        self.session = self.getSession(hparams, p_hparams)
        self.targets_vocab = hparams.problems[0].vocabulary["targets"]


    def getModel(self, hparams, p_hparams):
        model =  transformer.Transformer(
               hparams, tf.estimator.ModeKeys.PREDICT, p_hparams)
        return model

    def getSession(self, hparams, p_hparams):
        #self.model = transformer.Transformer(
        #       hparams, tf.estimator.ModeKeys.PREDICT, p_hparams)
        #self.model_2 = transformer.Transformer(
        #       hparams, tf.estimator.ModeKeys.PREDICT, p_hparams)
        
        self.inputs_ph = tf.placeholder(tf.int32, [None, None, 1, 1])
        self.targets_ph = tf.placeholder(tf.int32, [None, None, None, 1])
        features = {
          "inputs": self.inputs_ph,
          "targets": self.targets_ph,
          "target_space_id": tf.constant(1, dtype=tf.int32)
        }
        #infer_graph = tf.Graph()
        #with infer_graph.as_default():
        for i in range(self.ensemble_num):    
	    with tf.variable_scope("graph_%d" % (i+1)):
                shadred_logits, _, _ = self.model.model_fn(features, False, last_position_only=last_position_only)
           # with tf.variable_scope("graph_2"):
           #     shadred_logits2, _, _ = self.model_2.model_fn(features, False, last_position_only=last_position_only)
            # for variable in tf.trainable_variables():
               # print(variable)
        saver = tf.train.Saver(name='model_saver')
        tf.get_variable_scope().reuse_variables()
 
        session = tf.Session()
        # session1.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/ycliu/model/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
            print('restore from checkpoint')
        return session

    def testShapes(self,inputs):
        BATCH_SIZE = np.shape(inputs)[0]
        features = {
            "inputs": inputs,
            "targets": tf.zeros([BATCH_SIZE,1,1,1], dtype=tf.int32),
            "problem_choice":tf.constant(0, dtype=tf.int32),
            "input_sapce_id":tf.constant(4, dtype=tf.int32),
            "target_space_id": tf.constant(9, dtype=tf.int32)
        }

        initial_ids = tf.zeros([BATCH_SIZE], dtype=tf.int32)
        inputs_old = features["inputs"]
        features["inputs"] = tf.expand_dims(features["inputs"], 1)
        if len(features["inputs"].shape) < 5:
            features["inputs"] = tf.expand_dims(features["inputs"], 4)
        # Expand the inputs in to the beam size.
        features["inputs"] = tf.tile(features["inputs"], [1, beam_size, 1, 1, 1])
        s = tf.shape(features["inputs"])
        features["inputs"] = tf.reshape(features["inputs"],
                                        [s[0] * s[1], s[2], s[3], s[4]])

        def symbols_to_logits_fn(ids):
            """Go from ids to logits."""
            ids = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            ids = tf.pad(ids[:, 1:], [[0, 0], [0, 1], [0, 0], [0, 0]])
            features["targets"] = ids
            logits = tf.zeros([1,1,1,1,VOCAB_SIZE])
            for i in range(self.ensemble_num):
            	with tf.variable_scope("graph_%d" % (i+1)):
                    sharded_logits, _, _ = self.model.model_fn(features, False, last_position_only=last_position_only)
		logits += sharded_logits[0]
            #with tf.variable_scope("graph_2"):
            #    sharded_logits2, _, _ = self.model_2.model_fn(features, False, last_position_only=last_position_only)
	    #	logits2 = sharded_logits2[0]
            logits /= self.ensemble_num
            #logits = sharded_logits[0]  # Assuming we have one shard.
            if last_position_only:
                return tf.squeeze(logits, axis=[1, 2, 3])
            current_output_position = tf.shape(ids)[1]-1  # -1 due to the pad above.
            logits = logits[:, current_output_position, :, :]
            return tf.squeeze(logits, axis=[1, 2])
        
        decode_length = tf.shape(features['inputs'])[1] + tf.constant(decode_length_1)
        final_ids, final_probs = my_beam_search.beam_search(
            symbols_to_logits_fn, initial_ids, beam_size, decode_length, VOCAB_SIZE, 1.0)

        final_output,final_probs = self.session.run([final_ids, final_probs])
        #print('the shape of final_output is : ',final_output.shape,final_probs.shape,inputs.shape)
        #for k in range(beam_size):
        #    print(self.targets_vocab.decode(final_output[0,k,:]),final_probs[:,k])
        result = {}
        for (each_inputs, each_outputs) in zip(inputs, final_output[:,0]):
            result = {"inputs": each_inputs, "outputs":each_outputs[1:]}
            yield result

if __name__ == "__main__":
    tf.test.main()


