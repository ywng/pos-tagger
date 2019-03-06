import string

START_END_OBS = "--start/end--"
START_END_TAG = "START_END"

UNKNOWN_WORDS_REPRESENTATION = [
	"$$UNKNOWN$$",
	"$$UNKNOWN_DIGIT$$",
	"$$UNKNOWN_PUNCT$$",
	"$$UNKNOWN_PROPER_NOUN$$",
	"$$UNKNOWN_NOUN$$",
	"$$UNKNOWN_VERB$$",
	"$$UNKNOWN_ADJ$$",
	"$$UNKNOWN_ADV$$",
]

PUNCTUATION = set(string.punctuation)

NOUN_SUFFIX = ["al", "ation", "action", "age", "ance", "acy", "cy", "dom", "ee", "ence", "er", "ery", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty", "ure"]
VERB_SUFFIX = ["ate", "en", "ify", "fy", "ise", "ize"]
ADJ_SUFFIX = ["al", "able", "ese", "esque", "ful", "i", "ian", "ible", "ic", "ical", "ish", "ive", "less", "ly", "zy", "sy", "lly", "dy", "ous"]
ADV_SUFFIX = ["ward", "wards", "wise", "ly", "ily", "ally"]