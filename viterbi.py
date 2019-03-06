from __future__ import division
from const import START_END_OBS, START_END_TAG
import math

class Viterbi:
  def __init__(self, obs_space, states, trans_prob, emit_prob):
    '''
      obs_space: all the possbile value of observations. In POS tagging, it is the vocabulary.
      states: the HMM states. In POS tagging, it is the Part of Speech tag.
      trans_prob: A matrix K by K, trans_prob[i][j] is prob of state i -> state j, 
                  where K is the size of the states.
      emit_prob: A matrix K by N, emit_prob[i][j] is the prob of observing obs_j given state i, 
                 where K is the size of the states, N is the size of the obs_space.
    '''
    self.obs_space, self.states, self.trans_prob, self.emit_prob = \
      obs_space, states, trans_prob, emit_prob

    self.obs_space_size = len(self.obs_space)
    self.states_size = len(self.states)

    #observation index lookup table
    self.indexOfObs = {}
    for idx, observation in enumerate(self.obs_space):
      self.indexOfObs[observation] = idx

  def decode(self, obs):
    '''
      obs: a sequence of observations. In POS tagging, it is a sequence of words/tokens.
    '''
    seq_len = len(obs)
    viterbi = [[0] * seq_len for i in range(self.states_size)]
    backptr = [[None] * seq_len for i in range(self.states_size)]

    #init start probability
    start_tag_idx = self.states.index(START_END_TAG)
    first_obs_idx = self.indexOfObs[obs[0]]
    for s in range(self.states_size):
      backptr[s][0] = 0
      if self.trans_prob[start_tag_idx][s] == 0:
        viterbi[s][0] = float("-inf")
      else:
        viterbi[s][0] = math.log(self.trans_prob[start_tag_idx][s]) + \
                        math.log(self.emit_prob[s][first_obs_idx])

    for t in range(1, seq_len):
      obs_t_idx = self.indexOfObs[obs[t]]
      for curr_s in range(self.states_size):
        max_path, max_prob = None, float("-inf")
        for prev_s in range(self.states_size):
          prob = viterbi[prev_s][t-1] + \
                 math.log(self.trans_prob[prev_s][curr_s]) + \
                 math.log(self.emit_prob[curr_s][obs_t_idx])
          if prob > max_prob:
            max_path, max_prob = prev_s, prob
        viterbi[curr_s][t] = max_prob
        backptr[curr_s][t] = max_path

    #trace backward to get the state sequence path
    state_seq = [None] * seq_len 
    state_idx_seq = [None] * seq_len

    #start tracing back from the one with the highest prob, 
    #in the case of POS tagging, the last one should be an end node.
    max_prob = viterbi[0][seq_len-1]
    for state_idx in range(1, self.states_size):
      if viterbi[state_idx][seq_len-1] > max_prob:
        max_prob = viterbi[state_idx][seq_len-1]
        state_idx_seq[seq_len-1] = state_idx

    state_seq[seq_len-1] = self.states[state_idx_seq[seq_len-1]] #get the actual tag as return result

    for t in range(seq_len-1, 0, -1):
      state_idx_seq[t-1] = backptr[state_idx_seq[t]][t]
      state_seq[t-1] = self.states[state_idx_seq[t-1]]

    return state_seq, max_prob
