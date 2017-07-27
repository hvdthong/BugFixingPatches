import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn
import numpy as np

# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./data/train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/train.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1501154633/checkpoints/", "Checkpoint directory from training run")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

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
length_pos = data_helpers.data_size(FLAGS.positive_data_file)

text_max_document_length = max([len(x.split(" ")) for x in x_text])
code_max_document_length = max([len(x.split(" ")) for x in x_code])
print 'Max length of text: %i -- Max length of code: %i' % (text_max_document_length, code_max_document_length)

text_vocab_processor = learn.preprocessing.VocabularyProcessor(text_max_document_length)
code_vocab_processor = learn.preprocessing.VocabularyProcessor(code_max_document_length)
x_text = np.array(list(text_vocab_processor.fit_transform(x_text)))
x_code = np.array(list(code_vocab_processor.fit_transform(x_code)))

x_text_pos, x_code_pos = x_text[:length_pos], x_code[:length_pos]
x_text_neg, x_code_neg = x_text[length_pos:], x_code[length_pos:]
print x_text_pos.shape, x_code_pos.shape
print x_text_neg.shape, x_code_neg.shape

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_text_left = graph.get_operation_by_name("input_text_left").outputs[0]
        input_code_left = graph.get_operation_by_name("input_code_left").outputs[0]
        input_text_right = graph.get_operation_by_name("input_text_right").outputs[0]
        input_code_right = graph.get_operation_by_name("input_code_right").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        # lossValue = graph.get_operation_by_name("loss/loss").outputs[0]
        rankingScore_left = graph.get_operation_by_name("rank/ranking_left").outputs[0]
        rankingScore_right = graph.get_operation_by_name("rank/ranking_right").outputs[0]

        all_rankingScores = []
        for t, c in zip(x_text_neg, x_code_neg):
            feed_dict = {
                input_text_left: [t],
                input_code_left: [c],
                input_text_right: [t],
                input_code_right: [c],
                dropout_keep_prob: 1.0
            }
            # lossValue, rankingScore = sess.run([lossValue, rankingScore], feed_dict)
            # lossValue, l = sess.run([lossValue, rankingScore_left], feed_dict)
            # lossValue, l, r = sess.run([lossValue, rankingScore_left, rankingScore_right], feed_dict)
            # print lossValue, l
            l, r = sess.run([rankingScore_left, rankingScore_right], feed_dict)
            print l, r
            exit()
