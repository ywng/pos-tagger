import argparse
import pickle, datetime
from collections import defaultdict
from viterbi import Viterbi
from const import START_END_OBS, START_END_TAG
from const import UNKNOWN_WORDS_REPRESENTATION, PUNCTUATION, NOUN_SUFFIX, VERB_SUFFIX, ADJ_SUFFIX, ADV_SUFFIX
from score import score

MODEL_SAVE_PATH = "./model/model_%s.sav"

class HMMTagger:
  VOCAB_COUNT_THRESHOLD = 2
  ALPHA = 0.001 #smoothing parameter

  def train(self, train_data_path):
    self.vocab_list = self.__get_vocab(train_data_path)
    self.vocab_set = set(self.vocab_list)

    trans_count = defaultdict(lambda: defaultdict(int))
    emit_count = defaultdict(lambda: defaultdict(int))
    tag_count = defaultdict(int)

    with open(train_data_path, "r") as train_data:
      prev = START_END_OBS
      for line in train_data:
        if not line.split(): #empty line, means start/end of sentence
          word = START_END_OBS
          tag = START_END_TAG
        else:
          word, tag = line.strip().split("\t")
          if word not in self.vocab_set:
            word = self.__assign_unknown_cat(word)

        trans_count[prev][tag] += 1
        emit_count[tag][word] += 1
        tag_count[tag] += 1
        prev = tag
    self.tags = sorted(tag_count.keys())
    self.trans_prob = self.__generate_trans_prob_matrix(trans_count, tag_count)
    self.emit_prob = self.__generate_emit_prob_matrix(emit_count, tag_count)

  def eval(self, word_seq_path, tag_ans_path, output_path):
    self.test(word_seq_path, output_path)
    score(tag_ans_path, output_path)

  def test(self, word_seq_path, output_path):
    original_seq, processed_seq = self.__prepare_word_seq(word_seq_path)
    decoder = Viterbi(self.vocab_list, self.tags, self.trans_prob, self.emit_prob)
    tags_pred, prob = decoder.decode(processed_seq)

    with open(output_path, "w") as out:
      for word, tag in zip(original_seq, tags_pred):
        if not word:
          out.write("\n")
        else:
          out.write("{0}\t{1}\n".format(word, tag))

  def __prepare_word_seq(self, word_seq_path):
    original_seq, processed_seq = [], []
    with open(word_seq_path, "r") as seq_data:
      for line in seq_data:
        word = line.strip()
        original_seq.append(word)

        if not line.split(): #empty line, means start/end of sentence
          processed_seq.append(START_END_OBS)
        else:
          #unknown word
          if word not in self.vocab_set:
            word = self.__assign_unknown_cat(word)
          processed_seq.append(word)

    return original_seq, processed_seq

  def __generate_trans_prob_matrix(self, trans_count, tag_count):
    num_tags = len(self.tags)

    trans_prob = [[0] * num_tags for i in range(num_tags)]
    for i in range(num_tags):
        for j in range(num_tags):
            prev = self.tags[i]
            tag = self.tags[j]

            count = 0
            if ((prev in trans_count) and (tag in trans_count[prev])):
                count = trans_count[prev][tag]

            #smoothed transition probability
            trans_prob[i][j] = (count + HMMTagger.ALPHA) / (tag_count[prev] + HMMTagger.ALPHA * num_tags)

    return trans_prob

  def __generate_emit_prob_matrix(self, emit_count, tag_count):
    num_tags = len(self.tags)
    vocab_size = len(self.vocab_list)
    emit_prob = [[0] * vocab_size for i in range(num_tags)]

    for i in range(num_tags):
        for j in range(vocab_size):
            tag = self.tags[i]
            word = self.vocab_list[j]

            #smoothed emit probability
            count = 0
            if word in emit_count[tag]:
                count = emit_count[tag][word]

            emit_prob[i][j] = (count + HMMTagger.ALPHA) / (tag_count[tag] + HMMTagger.ALPHA * vocab_size)

    return emit_prob

  def __get_vocab(self, path):
    word_count = defaultdict(int)
    with open(path, "r") as train_data:
        for line in train_data:
          if line.split():
            emit, tag = line.split("\t")
            word_count[emit] += 1

    vocab = [k for k, v in word_count.items() if v >= HMMTagger.VOCAB_COUNT_THRESHOLD]
    # Add other unknown words into the vocab
    vocab.extend(UNKNOWN_WORDS_REPRESENTATION)
    vocab.append(START_END_OBS)
    vocab = sorted(vocab) #sort for easy human reading
    return vocab

  def __assign_unknown_cat(self, word):
    '''
      Classify the unknown words into different categories, for better performance.
    '''
    if any(c in PUNCTUATION for c in word):
      return "$$UNKNOWN_PUNCT$$"
    if any(c.isdigit() for c in word):
      return "$$UNKNOWN_DIGIT$$"
    if word[0].isupper() and "-" in word:
      return "$$UNKNOWN_PROPER_NOUN$$"

    if any(word.endswith(suffix) for suffix in ADV_SUFFIX):
      return "$$UNKNOWN_ADV$$"
    if any(word.endswith(suffix) for suffix in ADJ_SUFFIX):
      return "$$UNKNOWN_ADJ$$"
    if any(word.endswith(suffix) for suffix in VERB_SUFFIX):
      return "$$UNKNOWN_VERB$$"
    if any(word.endswith(suffix) for suffix in NOUN_SUFFIX):
      return "$$UNKNOWN_NOUN$$"

    if any(c.isupper() for c in word):
      return "$$UNKNOWN_PROPER_NOUN$$"

    return "$$UNKNOWN$$"


def save_model(model):
  pickle.dump(model, open(MODEL_SAVE_PATH%datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'wb'))

def load_model(path_to_model):
  model = pickle.load(open(path_to_model, 'rb'))
  return model

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', dest='train', action='store_true')
  parser.add_argument('--eval', dest='eval', action='store_true')
  parser.add_argument('--test', dest='test', action='store_true')

  parser.add_argument('--train_data', type=str, help='training data file location.')

  parser.add_argument('--model', type=str, help='the path to trained model.')
  parser.add_argument('--words', type=str, help='the word sequence to tag.')
  parser.add_argument('--tags', type=str, help='the ground true answers for the word sequence.')

  parser.add_argument('--output', type=str, help='the output path for the tagged word sequence.')


  args = parser.parse_args()

  tagger = HMMTagger()

  if args.train:
    #train the model
    if not args.train_data:
      fatal("Please provide the training data file path!")
    tagger.train(args.train_data)
    save_model(tagger)
  elif args.eval:
    #eval the model perf on dev corpus
    if not args.words or not args.tags:
      fatal("Please provide both the word sequence to eval and the tag answers!")
    if not args.model:
      fatal("Please provide the path to trained model")
    if not args.output:
      fatal("Please provide the output path for storing the tagged word sequence!")
    tagger = load_model(args.model)
    tagger.eval(args.words, args.tags, args.output)
  elif args.test:
    #test the model on unseen text
    if not args.words:
      fatal("Please provide the word sequence to test!")
    if not args.model:
      fatal("Please provide the path to trained model")
    if not args.output:
      fatal("Please provide the output path for storing the tagged word sequence!")
    tagger = load_model(args.model)
    tagger.test(args.words, args.output)

if __name__ == "__main__":
  main()