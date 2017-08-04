import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn
import numpy as np

# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("positive_data_file", "./data/train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/train.neg", "Data source for the negative data.")
tf.flags.DEFINE_string("test_data_file", "./data/id_testlabel.commits", "Data source for testing data.")

# Eval Parameters
# checkpoint_number = "1501168350"
checkpoint_number = "1501225832"
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/" + checkpoint_number + "/checkpoints/",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_string("write_dir", "./results_test/", "Directory to write file results_test")
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

# loading testing data
x_test_text, x_test_code = data_helpers.load_data_text_and_code(FLAGS.test_data_file)
x_test_text = np.array(list(text_vocab_processor.fit_transform(x_test_text)))
x_test_code = np.array(list(code_vocab_processor.fit_transform(x_test_code)))
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
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_text_left = graph.get_operation_by_name("input_text_left").outputs[0]
            input_code_left = graph.get_operation_by_name("input_code_left").outputs[0]
            input_text_right = graph.get_operation_by_name("input_text_right").outputs[0]
            input_code_right = graph.get_operation_by_name("input_code_right").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

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
                    dropout_keep_prob: 1.0
                }
                l, r = sess.run([rankingScore_left, rankingScore_right], feed_dict)
                print len(all_rankingScores), l, r
                all_rankingScores.append(float(l[0]))
            with open(path_write, "w") as file_:
                [file_.write(str(c) + '\n') for c in all_rankingScores]


checkpoints = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
for checkpoint_file in checkpoints.all_model_checkpoint_paths:
    num_checkpoint, model_name = checkpoint_file.strip().split('/')[-3], checkpoint_file.strip().split('/')[-1]
    path = FLAGS.write_dir + num_checkpoint + "_" + model_name
    print path
    checkpoint_results(checkpoint_file=checkpoint_file, path_write=path)
    # exit()

# print checkpoint_file
# print type(checkpoints)
# print checkpoints
# exit()
# for c in checkpoints:
#     print c
# print checkpoints
