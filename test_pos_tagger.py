import pytest
import logging

from viterbi import Viterbi
from const import START_END_OBS, START_END_TAG

logging.basicConfig(level=logging.DEBUG)

def test_viterbi_decode():
  '''
    Test case based on HW3 question.
  '''
  log = logging.getLogger('test viterbi')

  ZERO = 0.000000000000000001
  obs_space = ["moo", "hello", "quack", START_END_OBS]
  states = ["Cow", "Duck", START_END_TAG]
  trans_prob = [
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [1.0, ZERO, ZERO]
          ]
  emit_prob = [
            [0.9, 0.1, ZERO, ZERO],
            [ZERO, 0.4, 0.6, ZERO],
            [ZERO, ZERO, ZERO, 1.0]
          ]

  decoder = Viterbi(obs_space, states, trans_prob, emit_prob)

  obs = ["moo", "hello", "quack", START_END_OBS]
  seq, prob = decoder.decode(obs)

  log.debug("seq: " + str(seq))
  log.debug("log_prob: " + str(prob))
  assert prob - (-5.03903) < ZERO and \
         seq == ["Cow", "Duck", "Duck", START_END_TAG]

