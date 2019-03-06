#HMM POS Tagger
This is to train a part-of-speech(POS) tagger based on HMM model, using WSJ from Penn Treebank Corpus.
The decoding of the POS state sequence is using Viterbi algorithm (dynamic programming).

##Handling of Unknown Words
It bases P(t|w) of unknown word on probability distribution of words which occur once in corpus.
To further classify different type of unknwon words, some heuristic rules are used:
* Unknown Proper Noun: usually the first letter is capital, and with/without hypen.
* Unknown Noun, Verb, Adjective, Adverb: they are open-class words, bases on common suffixes.

```
NOUN_SUFFIX = ["al", "ation", "action", "age", "ance", "acy", "cy", "dom", "ee", "ence", "er", "ery", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty", "ure"]
VERB_SUFFIX = ["ate", "en", "ify", "fy", "ise", "ize"]
ADJ_SUFFIX = ["al", "able", "ese", "esque", "ful", "i", "ian", "ible", "ic", "ical", "ish", "ive", "less", "ly", "zy", "sy", "lly", "dy", "ous"]
ADV_SUFFIX = ["ward", "wards", "wise", "ly", "ily", "ally"]
```
* Unknown Digit: if any character is a digit.
* Unknown Punctuation: if any character is a punctuation.
* Unknown: any unknown word that is not classified.

##Testing
You can use pytest to trigger my unit tests. In the source dir, run
```
pytest
```

##How to run
ssh into nyu mscis crunchy 1
```
ssh ywn202@access.cims.nyu.edu
ssh crunchy1.cims.nyu.edu
```
load the python module in crunchy 1.
```
module load python-3.6
```

###Training
It will produce a model object in ./model/
```
python main.py --train --train_data "./data/WSJ_02-21.pos"
```

###Eval
Running below script will tag the given word sequence. It will evalute and print the accuracy. 
For my implementation, with some heuristic handling of unknown words, **tagging accuracy is 95.391593 %**.
--model: path to the trained model file.
--words: the words file to be tagged.
--tags: the tags ground true.
--output: the path where the tag result will be output to.
```
python main.py --eval --model ./model/model_2019-03-06_01-19-20.sav --words ./data/WSJ_24.words --tags ./data/WSJ_24.pos --output ./output/wsj_24.pos
```

###Test
--model: path to the trained model file.
--words: the words file to be tagged.
--output: the path where the tag result will be output to.
```
python main.py --test --model ./model/model_2019-03-06_01-19-20.sav --words ./data/WSJ_23.words --output ./output/wsj_23.pos
```