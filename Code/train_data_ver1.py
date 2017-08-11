import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn
import numpy as np
from cnn_patch_update import CNNPatchUpdate
import os
import time
import datetime

# Parameters
# ==================================================
# Data loading params
tf.flags.DEFINE_string("train_pos_data", "./data_ver1/eq100_line_aug1.pos.contain"
                       , "Data source for the positive training data")
tf.flags.DEFINE_string("train_neg_data", "./data_ver1/eq100_line_aug1.neg.contain"
                       , "Data source for the negative training data")
tf.flags.DEFINE_string("test_data", "./data_ver1/extra100_line_aug1.neg.contain"
                       , "Data source for the testing data, only contain negative bug fixing patches")
tf.flags.DEFINE_string("label_pos_data", "./data_ver1/lbd100_line_aug1.pos.contain"
                       , "Data source for the positive patches for testing data")
tf.flags.DEFINE_string("label_neg_data", "./data_ver1/eq100_line_aug1.neg.contain"
                       , "Data source for the negative patches for testing data")
tf.flags.DEFINE_integer("seed", 10, "Random seed (default:123)")
# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim_text", 128, "Dimensionality of character embedding for text (default: 128)")
tf.flags.DEFINE_integer("embedding_dim_code", 64, "Dimensionality of character embedding for code (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1, 2, 3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 32, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden layer units (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_iters", 100000,
                        "Number of training iterations; the size of each iteration is the batch size (default: 1000)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 100, "Number of checkpoints to store (default: 5)")
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

exit()

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

print len(text_vocab_processor.vocabulary_._mapping)
print len(code_vocab_processor.vocabulary_._mapping)


x_text_pos, x_code_pos = x_text[:length_pos], x_code[:length_pos]
x_text_neg, x_code_neg = x_text[length_pos:], x_code[length_pos:]
print x_text_pos.shape, x_code_pos.shape
print x_text_neg.shape, x_code_neg.shape

# Randomly shuffle data
# np.random.seed(FLAGS.seed)
# shuffle_indices = np.random.permutation(np.arange(len(y)))
# x_text_shuffled, x_code_shuffled = x_text[shuffle_indices], x_code[shuffle_indices]
# y_shuffled = y[shuffle_indices]

print 'Dictionary of text: %i -- Dictionary of code: %i' \
      % (len(text_vocab_processor.vocabulary_._mapping), len(code_vocab_processor.vocabulary_._mapping))
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNNPatchUpdate(
            max_length_text=text_max_document_length,
            max_length_code=code_max_document_length,
            vocab_size_text=len(text_vocab_processor.vocabulary_),
            vocab_size_code=len(code_vocab_processor.vocabulary_),
            embedding_size_text=FLAGS.embedding_dim_text,
            embedding_size_code=FLAGS.embedding_dim_code,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # write vocabulary
        text_vocab_processor.save(os.path.join(out_dir, "vocab_text"))
        code_vocab_processor.save(os.path.join(out_dir, "vocab_code"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(left_text_batch, left_code_batch, right_text_batch, right_code_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_text_left: left_text_batch,
                cnn.input_code_left: left_code_batch,
                cnn.input_text_right: right_text_batch,
                cnn.input_code_right: right_code_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss],
                feed_dict)

            # _, step, loss = sess.run(
            #     [train_op, global_step, cnn.loss], feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            train_summary_writer.add_summary(summaries, step)

        for i in xrange(0, FLAGS.num_iters):
            # Generate batches
            batch_text_pos, batch_code_pos, batch_text_neg, batch_code_neg = data_helpers.batch_iter(
                text_pos=x_text_pos, code_pos=x_code_pos,
                text_neg=x_text_neg, code_neg=x_code_neg,
                batch_size=FLAGS.batch_size)

            train_step(left_text_batch=batch_text_pos, left_code_batch=batch_code_pos,
                       right_text_batch=batch_text_neg, right_code_batch=batch_code_pos)

            if (i + 1) % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=i)
                print("Saved model checkpoint to {}\n".format(path))