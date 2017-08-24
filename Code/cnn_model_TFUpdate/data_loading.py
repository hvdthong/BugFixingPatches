import numpy as np
from tensorflow.contrib import learn

import data_helpers


def loading_training_data(FLAGS):
    print("Loading data...")
    train_pos_text, train_pos_code = data_helpers.load_data_ver1_text_and_code(FLAGS.train_pos_data)
    print "Positive training data for text and code: %i, %i" % (len(train_pos_text), len(train_pos_code))
    train_neg_text, train_neg_code = data_helpers.load_data_ver1_text_and_code(FLAGS.train_neg_data)
    print "Negative training data for text and code: %i, %i" % (len(train_neg_text), len(train_neg_code))
    test_text, test_code = data_helpers.load_data_ver1_text_and_code(FLAGS.test_data)
    print "Testing data for text and code: %i, %i" % (len(test_text), len(test_code))
    label_pos_text, label_pos_code = data_helpers.load_data_ver1_text_and_code(FLAGS.label_pos_data)
    print "Positive labeled data for text and code: %i, %i" % (len(label_pos_text), len(label_pos_code))
    label_neg_text, label_neg_code = data_helpers.load_data_ver1_text_and_code(FLAGS.label_neg_data)
    print "Negative labeled data for text and code: %i, %i" % (len(label_neg_text), len(label_neg_code))

    all_text = train_pos_text + train_neg_text + test_text + label_pos_text + label_neg_text
    all_code = train_pos_code + train_neg_code + test_code + label_pos_code + label_neg_code
    print len(all_text), len(all_code)
    text_max_document_length = max([len(x.split(" ")) for x in all_text])
    code_max_document_length = max([len(x.split(" ")) for x in all_code])
    print 'Max length of text: %i -- Max length of code: %i' % (text_max_document_length, code_max_document_length)

    text_vocab_processor = learn.preprocessing.VocabularyProcessor(text_max_document_length)
    code_vocab_processor = learn.preprocessing.VocabularyProcessor(code_max_document_length)
    x_text = np.array(list(text_vocab_processor.fit_transform(all_text)))
    x_code = np.array(list(code_vocab_processor.fit_transform(all_code)))

    print 'Dictionary of text: %i -- Dictionary of code: %i' \
          % (len(text_vocab_processor.vocabulary_._mapping), len(code_vocab_processor.vocabulary_._mapping))

    length_train_pos, length_train_neg = len(train_pos_text), len(train_neg_text)

    x_text_pos, x_code_pos = x_text[:length_train_pos], x_code[:length_train_pos]
    x_text_neg, x_code_neg = x_text[length_train_pos:(length_train_pos + length_train_neg)], \
                             x_code[length_train_pos:(length_train_pos + length_train_neg)]
    print "Print the shape of training data"
    print x_text_pos.shape, x_code_pos.shape
    print x_text_neg.shape, x_code_neg.shape
    return x_text_pos, x_code_pos, x_text_neg, x_code_neg