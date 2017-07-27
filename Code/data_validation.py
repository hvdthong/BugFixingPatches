import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn
import numpy as np


# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./data/noeq100train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/noeq100train.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("test_file", "./data/CNNall100testfinal.txt", "Data source for the test data.")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

x_text, x_code, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
test_text, test_code = data_helpers.load_data_text_and_code(FLAGS.test_file)

print len(x_text), len(x_code), len(test_text), len(test_code)
x_all_text, x_all_code = x_text + test_text, x_code + test_code
print len(x_all_text), len(x_all_code)

text_max_document_length = max([len(x.split(" ")) for x in x_all_text])
code_max_document_length = max([len(x.split(" ")) for x in x_all_code])
print 'Max length of text: %i -- Max length of code: %i' % (text_max_document_length, code_max_document_length)

text_vocab_processor = learn.preprocessing.VocabularyProcessor(text_max_document_length)
code_vocab_processor = learn.preprocessing.VocabularyProcessor(code_max_document_length)
x_text = np.array(list(text_vocab_processor.fit_transform(x_text)))
x_code = np.array(list(code_vocab_processor.fit_transform(x_code)))

print len(text_vocab_processor.vocabulary_._mapping), text_max_document_length
print len(code_vocab_processor.vocabulary_._mapping), code_max_document_length



