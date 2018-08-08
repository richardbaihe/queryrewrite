"""Tests for Transformer."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from tensorflow.python.framework import ops

import numpy as np

import sys
#sys.path.append("../")
from tensor2tensor.data_generators import problem_hparams
from tensor2tensor.models import transformer
from tensor2tensor.utils import beam_search
import tensorflow as tf
import os

BATCH_SIZE = 1
INPUT_LENGTH = 5
TARGET_LENGTH = 15
VOCAB_SIZE = 29979
#TGT_VOCAB_SIZE = 29903
EOS_ID = 1

def mysort(seq, k):
    x = dict(enumerate(seq))
    seq_sort, index = quicksort(x, k)
    return seq_sort, index

def quicksort(seq, k):
    def partition(alist, start, end):
        if end <= start:
            return
        base = alist[start]
        index1, index2 = start, end
        while start < end:
            while start < end and alist[end][1] <= base[1]:
                end -= 1
            alist[start] = alist[end]
            while start < end and alist[start][1] >= base[1]:
                start += 1
            alist[end] = alist[start]
        alist[start] = base
        return start

    def find_least_k_nums(alist, k):
        length = len(alist)
        if length == k:
           return alist
        if not alist or k <= 0 or k > length:
            return
        start = 0
        end = length - 1
        index = partition(alist, start, end)
        while index != k:
            if index > k:
                index = partition(alist, start, index - 1)
            elif index < k:
                index = partition(alist, index + 1, end)
        return alist[:k]

    seq_top_k = find_least_k_nums(seq.items(), k)
    index, seq_top_k = zip(*seq_top_k)
    return np.array(seq_top_k), np.array(index)

def log_prob_from_logits(logits):
  return logits - np.log(np.sum(np.exp(logits)))


class TransformerTest():
  def __init__(self, hparams, p_hparams):
      self.model = self.getModel(hparams, p_hparams)
      self.logits, self.session = self.getSession()
      self.targets_vocab = hparams.problems[0].vocabulary["targets"]

  def getModel(self, hparams, p_hparams):
      return transformer.Transformer(
               hparams, tf.estimator.ModeKeys.PREDICT, p_hparams, data_parallelism=None)
  
  def getSession(self):
      self.inputs_ph = tf.placeholder(tf.int32, [BATCH_SIZE, None, 1, 1])
      self.targets_ph = tf.placeholder(tf.int32, [BATCH_SIZE, None, None, 1])
      features = {
          "inputs": self.inputs_ph,
          "targets": self.targets_ph,
          "target_space_id": tf.constant(1, dtype=tf.int32)
      }
      model1 = self.model
      sharded_logits1, _, _ = model1.model_fn(features,False,last_position_only=True)
      tf.get_variable_scope().reuse_variables()
      # logits1 = tf.nn.softmax(tf.concat(sharded_logits1, 0))
      # logits1 = tf.concat(shadred_logits1, 0)
      logits1 = sharded_logits1[0]
      saver1 = tf.train.Saver(name='model_saver')

      # with tf.Session(graph=graph1) as session1:
      session1 = tf.Session()
      # ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/ycliu/tensor2tensor-old/utils/checkpoint'))
      ckpt = tf.train.get_checkpoint_state(os.path.dirname('/home/ycliu/checkpoint'))
      if ckpt and ckpt.model_checkpoint_path:
          saver1.restore(session1, ckpt.model_checkpoint_path)
          print('restore from checkpoint')
      return  logits1, session1

  def testTransformer(self, inputs):
      logits1, session1 = self.logits, self.session
      targets = np.zeros([BATCH_SIZE, 1, 1, 1],dtype=np.int32)

      for input in inputs:          
          alive_seq = [targets]
      	  alive_log_probs = [0.0]
      	  beam_size = 2
     	  beam_length = 30
     	  finished_seq = []
          finished_scores = []
          length = 0
          input = input[np.newaxis, :, np.newaxis, np.newaxis]
          while alive_seq != []:
              length += 1
              if length > beam_length:
                  break
              tmp_seq = []
              tmp_scores = []
              for seq, scores in zip(alive_seq, alive_log_probs):
                  # print('seq',np.squeeze(seq))
                  if length>1:
                     print('seq_vocab',self.targets_vocab.decode(np.squeeze(seq)))
                  print('scores', scores)
                  if seq == []:
                     break
                  # seq = seq[np.newaxis, :, np.newaxis, np.newaxis]
                  # print('alive_seq', np.squeeze(seq)[1:])
                  # print('scores',scores)
                  feed = {self.inputs_ph: input, self.targets_ph: seq}
                  # with graph1.as_default():
                  res = session1.run(logits1, feed_dict=feed)
                  # print(res.shape)
                  # current_output_position = np.shape(seq)[1]-1
 		  # print('position : ',current_output_position)
                  # res = res[:,current_output_position,:,:,:]
                  # res = (res1+res1)/2.0
                  # topk_seq_scores, topk_seq_ids = mysort(res[0,-1,0,0], k=beam_size)
                  candidate_log_probs = log_prob_from_logits(np.squeeze(res))
                  # candidate_log_probs = np.squeeze(res)
                  # print('candidate_log_probs: ', candidate_log_probs.shape, np.max(candidate_log_probs))
                  log_probs = candidate_log_probs - scores
                  length_penalty = np.math.pow(((5. + (length+1)*1.0) / 6.), 0.6)
                  curr_scores = log_probs / length_penalty
                  topk_seq_ids = np.argpartition(-curr_scores, 2*beam_size)[:2*beam_size]
                  # print('topk_seq_ids', topk_seq_ids)
                  # topk_seq_scores = candidate_log_probs[topk_seq_ids]
                  topk_seq_scores = curr_scores[topk_seq_ids] * length_penalty
                  # length_penalty = np.math.pow(((5. + length*1.0) / 6.), 0.6)
                  # scores = scores / length_penalty
                  # topk_seq_scores = topk_seq_scores / length_penalty
                  # print('topk_seq_scores', topk_seq_scores)
                  # bug, should append all the tmp_seq
                  if length > 1:
                      tmp_seq += [np.append(np.squeeze(seq)[:-1], np.array([id, 0],dtype=np.int32))[np.newaxis,:,np.newaxis,np.newaxis] for id in topk_seq_ids]
                  else:
                      tmp_seq += [np.array([id,0],dtype=np.int32)[np.newaxis,:,np.newaxis,np.newaxis] for id in topk_seq_ids]    
              # tmp_scores += [scores-np.math.log(score) for score in topk_seq_scores]
                  tmp_scores += [-score for score in topk_seq_scores]
              # print('the tmp_seq is :', tmp_seq)
                  print('candidate_nextvocab :\t',self.targets_vocab.decode(np.squeeze(topk_seq_ids)))
                  print('candidate_scores :\t', topk_seq_scores)
              # print('the tmp_scores is : ', tmp_scores)
              # print('the shape of logits is :', len(res))

              tmp_seq = np.array(tmp_seq)
              tmp_scores = np.array(tmp_scores)
              # topk_scores, topk_ids = mysort(tmp_scores, k=beam_size)
              if len(tmp_scores) == beam_size:
                  topk_scores, topk_ids = mysort(tmp_scores, k=beam_size)
                  #topk_ids = range(beam_size)
                  #topk_scores = tmp_scores[topk_ids]
	      else:
	          topk_ids = np.argpartition(tmp_scores,beam_size)[:beam_size]
                  topk_scores = tmp_scores[topk_ids]
              #print('topk_ids',np.squeeze(tmp_seq[topk_ids]))
              #print('topk_socres', topk_scores)

              alive_seq = []
              alive_log_probs = []

              for seq, seq_scores in zip(tmp_seq[topk_ids], topk_scores):
                  if np.squeeze(seq)[-1] == EOS_ID:
                      finished_seq.append(seq)
                      finished_scores.append(seq_scores)
                      if len(finished_seq) > beam_size:
                          break
                  else:
                      alive_seq.append(seq)
                      alive_log_probs.append(seq_scores)
          # session2.close()
          finished_seq.append(alive_seq)
          print('finished_seq :',len(finished_seq[0]),finished_seq[0][0].shape)
          result = {"inputs": input, "outputs": finished_seq[0]}
          yield result

if __name__ == "__main__":
  test = TransformerTest()
  test.testTransformer()
