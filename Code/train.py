import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn
import numpy as np
from cnn_patch import CNNPatch

# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./data/noeq100train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/noeq100train.neg", "Data source for the negative data.")
tf.flags.DEFINE_integer("seed", 10, "Random seed (default:123)")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim_text", 128, "Dimensionality of character embedding for text (default: 128)")
tf.flags.DEFINE_integer("embedding_dim_code", 64, "Dimensionality of character embedding for code (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2, 3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 32, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden layer units (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================
# Load data
print("Loading data...")
x_text, x_code, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
# print len(x_text), len(x_code), len(y)

text_max_document_length = max([len(x.split(" ")) for x in x_text])
code_max_document_length = max([len(x.split(" ")) for x in x_code])
print 'Max length of text: %i -- Max length of code: %i' % (text_max_document_length, code_max_document_length)

text_vocab_processor = learn.preprocessing.VocabularyProcessor(text_max_document_length)
code_vocab_processor = learn.preprocessing.VocabularyProcessor(code_max_document_length)
x_text = np.array(list(text_vocab_processor.fit_transform(x_text)))
x_code = np.array(list(code_vocab_processor.fit_transform(x_code)))

# Randomly shuffle data
np.random.seed(FLAGS.seed)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_text_shuffled, x_code_shuffled = x_text[shuffle_indices], x_code[shuffle_indices]
y_shuffled = y[shuffle_indices]

print 'Dictionary of text: %i -- Dictionary of code: %i' \
      % (len(text_vocab_processor.vocabulary_._mapping), len(code_vocab_processor.vocabulary_._mapping))
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn_ = CNNPatch(
            max_length_text=x_text_shuffled.shape[1],
            max_length_code=x_code_shuffled.shape[1],
            vocab_size_text=len(text_vocab_processor.vocabulary_),
            vocab_size_code=len(code_vocab_processor.vocabulary_),
            embedding_size_text=FLAGS.embedding_dim_text,
            embedding_size_code=FLAGS.embedding_dim_code,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            num_hidden=FLAGS.num_hidden,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        print 'hello'
