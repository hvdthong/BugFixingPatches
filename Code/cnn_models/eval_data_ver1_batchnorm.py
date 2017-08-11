import tensorflow as tf
import data_helpers
import numpy as np
from tensorflow.contrib import learn

# Parameters
# ==================================================
# Data loading params
# tf.flags.DEFINE_string("train_pos_data", "./data_ver1/eq100_line_aug1.pos.contain"
#                        , "Data source for the positive training data")
# tf.flags.DEFINE_string("train_neg_data", "./data_ver1/eq100_line_aug1.neg.contain"
#                        , "Data source for the negative training data")
# tf.flags.DEFINE_string("test_data", "./data_ver1/extra100_line_aug1.neg.contain"
#                        , "Data source for the testing data, only contain negative bug fixing patches")
# tf.flags.DEFINE_string("label_pos_data", "./data_ver1/lbd100_line_aug1.pos.contain"
#                        , "Data source for the positive patches for testing data")
# tf.flags.DEFINE_string("label_neg_data", "./data_ver1/lbd100_line_aug1.neg.contain"
#                        , "Data source for the negative patches for testing data")

tf.flags.DEFINE_string("train_pos_data", "../data_ver1/eq100_line_aug1.pos.contain_maxtext150_maxcode300"
                       , "Data source for the positive training data")
tf.flags.DEFINE_string("train_neg_data", "../data_ver1/eq100_line_aug1.neg.contain_maxtext150_maxcode300"
                       , "Data source for the negative training data")
tf.flags.DEFINE_string("test_data", "../data_ver1/extra100_line_aug1.neg.contain_maxtext150_maxcode300"
                       , "Data source for the testing data, only contain negative bug fixing patches")
tf.flags.DEFINE_string("label_pos_data", "../data_ver1/lbd100_line_aug1.pos.contain_maxtext150_maxcode300"
                       , "Data source for the positive patches for testing data")
tf.flags.DEFINE_string("label_neg_data", "../data_ver1/lbd100_line_aug1.neg.contain_maxtext150_maxcode300"
                       , "Data source for the negative patches for testing data")

# ==================================================
# Eval Parameters
# checkpoint_number = "1502342722"
checkpoint_number = "1502415323"
tf.flags.DEFINE_string("checkpoint_dir", "./runs/" + checkpoint_number + "/checkpoints/",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_string("write_dir", "../results_test/", "Directory to write file results_test")
# ==================================================
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# ==================================================
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
# print len(train_pos_text), len(train_pos_code)
# print len(train_neg_text), len(train_neg_code)
# print len(test_text), len(test_code)
# print len(label_pos_text), len(label_pos_code)
# print len(label_neg_text), len(label_neg_code)

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


# ==================================================
# Load labeled data
x_test_text = x_text[(len(train_pos_text) + len(train_neg_text) + len(test_text)):]
x_test_code = x_code[(len(train_pos_code) + len(train_neg_code) + len(test_code)):]
print x_test_text.shape, x_test_code.shape

print "\nEvaluating...\n"


# Evaluation
# ==================================================
def checkpoint_results(checkpoint_file, path_write):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            print checkpoint_file
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text_left = graph.get_operation_by_name("input_text_left").outputs[0]
            input_code_left = graph.get_operation_by_name("input_code_left").outputs[0]
            input_text_right = graph.get_operation_by_name("input_text_right").outputs[0]
            input_code_right = graph.get_operation_by_name("input_code_right").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training_phase = graph.get_operation_by_name("is_training_phase").outputs[0]

            # Tensors we want to evaluate
            rankingScore_left = graph.get_operation_by_name("rank/ranking_left").outputs[0]
            rankingScore_right = graph.get_operation_by_name("rank/ranking_right").outputs[0]

            all_rankingScores = []
            for t, c in zip(x_test_text, x_test_code):
                feed_dict = {
                    input_text_left: [t],
                    input_code_left: [c],
                    input_text_right: [t],
                    input_code_right: [c],
                    dropout_keep_prob: 1.0,
                    is_training_phase: 0
                }
                l, r = sess.run([rankingScore_left, rankingScore_right], feed_dict)
                print len(all_rankingScores), l, r
                exit()
                all_rankingScores.append(float(l[0]))
            with open(path_write, "w") as file_:
                [file_.write(str(c) + '\n') for c in all_rankingScores]


checkpoints = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
for checkpoint_file in checkpoints.all_model_checkpoint_paths:
    num_checkpoint, model_name = checkpoint_file.strip().split('/')[-3], checkpoint_file.strip().split('/')[-1]
    path = FLAGS.write_dir + num_checkpoint + "_" + model_name
    print path
    checkpoint_results(checkpoint_file=checkpoint_file, path_write=path)