import tensorflow as tf
import data_helpers
from tensorflow.contrib import learn
import numpy as np
import os
import time
import datetime
from cnn_without_dropoutAndbatchnorm import CNNPatchWithoutDropOutAndBatchNorm
from cnn_withDropout_noBatchnorm import CNNPatchWithDropOutNoBatchNorm
from cnn_withDropoutAndBatchnorm import CNNPatchWithDropOutAndBatchNorm
from cnn_noDropout_withBatchnorm import CNNPatchWithoutDropOutWithBatchNorm
from cnn_withDropout_noBatchnorm_noFusionLayer import CNNPatchWithDropOutNoBatchNorm_noFusionLayer

# USE WITH BATCH NORMALIZATION SINCE WE NEED TO ADD "IS_TRAINING" IN BATCH NORMALIZATION
# Parameters
# ==================================================
# Data loading params
# tf.flags.DEFINE_string("train_pos_data", "../data_ver1/eq100_line_aug1.pos.contain"
#                        , "Data source for the positive training data")
# tf.flags.DEFINE_string("train_neg_data", "../data_ver1/eq100_line_aug1.neg.contain"
#                        , "Data source for the negative training data")
# tf.flags.DEFINE_string("test_data", "../data_ver1/extra100_line_aug1.neg.contain"
#                        , "Data source for the testing data, only contain negative bug fixing patches")
# tf.flags.DEFINE_string("label_pos_data", "../data_ver1/lbd100_line_aug1.pos.contain"
#                        , "Data source for the positive patches for testing data")
# tf.flags.DEFINE_string("label_neg_data", "../data_ver1/lbd100_line_aug1.neg.contain"
#                        , "Data source for the negative patches for testing data")
# tf.flags.DEFINE_integer("seed", 10, "Random seed (default:123)")
# tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation.")

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
tf.flags.DEFINE_integer("seed", 10, "Random seed (default:123)")
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim_text", 64, "Dimensionality of character embedding for text (default: 128)")
tf.flags.DEFINE_integer("embedding_dim_code", 64, "Dimensionality of character embedding for code (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1, 2, 3", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden layer units (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0, "L2 regularization lambda (default: 0.0)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_iters", 30000,
                        "Number of training iterations; the size of each iteration is the batch size (default: 1000)")
tf.flags.DEFINE_integer("evaluate_every", 5, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 100, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("num_devs", 2000, "Number of dev pairs for text and code")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
# Model
tf.flags.DEFINE_string("no_dropoutAndbatchnorm", "no_dropoutAndbatchnorm"
                       , "Without dropout and without batch normalization")
tf.flags.DEFINE_string("no_dropout_with_batchnorm", "no_dropout_with_batchnorm"
                       , "No dropout but with batch normalization")
tf.flags.DEFINE_string("with_dropout_no_batchnorm", "with_dropout_no_batchnorm"
                       , "With dropout but no batch normalization")
tf.flags.DEFINE_string("with_dropout_with_batchnorm", "with_dropout_with_batchnorm"
                       , "With dropout and with batch normalization")
tf.flags.DEFINE_string("with_dropout_no_batchnorm_noFusionLayer", "with_dropout_no_batchnorm_noFusionLayer"
                       , "With dropout and with batch normalization")
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
# train_pos_text, train_pos_code = data_helpers.load_data_ver1_text_and_code(FLAGS.train_pos_data)
# print "Positive training data for text and code: %i, %i" % (len(train_pos_text), len(train_pos_code))
# train_neg_text, train_neg_code = data_helpers.load_data_ver1_text_and_code(FLAGS.train_neg_data)
# print "Negative training data for text and code: %i, %i" % (len(train_neg_text), len(train_neg_code))
# test_text, test_code = data_helpers.load_data_ver1_text_and_code(FLAGS.test_data)
# print "Testing data for text and code: %i, %i" % (len(test_text), len(test_code))
label_pos_text, label_pos_code = data_helpers.load_data_ver1_text_and_code(FLAGS.label_pos_data)
print "Positive labeled data for text and code: %i, %i" % (len(label_pos_text), len(label_pos_code))
label_neg_text, label_neg_code = data_helpers.load_data_ver1_text_and_code(FLAGS.label_neg_data)
print "Negative labeled data for text and code: %i, %i" % (len(label_neg_text), len(label_neg_code))
# print len(train_pos_text), len(train_pos_code)
# print len(train_neg_text), len(train_neg_code)
# print len(test_text), len(test_code)
# print len(label_pos_text), len(label_pos_code)
# print len(label_neg_text), len(label_neg_code)

# all_text = train_pos_text + train_neg_text + test_text + label_pos_text + label_neg_text
# all_code = train_pos_code + train_neg_code + test_code + label_pos_code + label_neg_code
all_text = label_pos_text + label_neg_text
all_code = label_pos_code + label_neg_code
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

# length_train_pos, length_train_neg = len(train_pos_text), len(train_neg_text)
#
# x_text_pos, x_code_pos = x_text[:length_train_pos], x_code[:length_train_pos]
# x_text_neg, x_code_neg = x_text[length_train_pos:(length_train_pos + length_train_neg)], \
#                          x_code[length_train_pos:(length_train_pos + length_train_neg)]

x_text_pos, x_code_pos = x_text[:len(label_pos_text)], x_code[:(len(label_pos_text))]
x_text_neg, x_code_neg = x_text[len(label_pos_text):], x_code[len(label_pos_text):]

print "Print the shape of training data"
print x_text_pos.shape, x_code_pos.shape
print x_text_neg.shape, x_code_neg.shape

# Random positive and negative pool
np.random.seed(FLAGS.seed)
pos_shuffle_indices = np.random.permutation(np.arange(x_text_pos.shape[0]))
neg_shuffle_indices = np.random.permutation(np.arange(x_text_neg.shape[0]))
x_text_pos_shuffled, x_code_pos_shuffled = x_text_pos[pos_shuffle_indices], x_code_pos[pos_shuffle_indices]
x_text_neg_shuffled, x_code_neg_shuffled = x_text_neg[neg_shuffle_indices], x_code_neg[neg_shuffle_indices]

# Split train/dev
pos_dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(x_text_pos.shape[0]))
neg_dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(x_text_neg.shape[0]))

x_text_pos_train, x_text_pos_dev = \
    x_text_pos_shuffled[:pos_dev_sample_index], x_text_pos_shuffled[pos_dev_sample_index:]
x_code_pos_train, x_code_pos_dev = \
    x_code_pos_shuffled[:pos_dev_sample_index], x_code_pos_shuffled[pos_dev_sample_index:]
x_text_neg_train, x_text_neg_dev = \
    x_text_neg_shuffled[:neg_dev_sample_index], x_text_neg_shuffled[neg_dev_sample_index:]
x_code_neg_train, x_code_neg_dev = \
    x_code_neg_shuffled[:neg_dev_sample_index], x_code_neg_shuffled[neg_dev_sample_index:]

print "Print the shape of training data after we split it to train set and dev set"
print x_text_pos_train.shape, x_code_pos_train.shape
print x_text_neg_train.shape, x_code_neg_train.shape
print "Print the shape of dev data"
print x_text_pos_dev.shape, x_code_pos_dev.shape
print x_text_neg_dev.shape, x_code_neg_dev.shape

# Create a pair from dev data
pos_dev_index = [np.random.randint(0, x_text_pos_dev.shape[0]) for i in xrange(FLAGS.num_devs)]
neg_dev_index = [np.random.randint(0, x_text_neg_dev.shape[0]) for i in xrange(FLAGS.num_devs)]

x_text_pos_dev, x_code_pos_dev = x_text_pos_dev[pos_dev_index], x_code_pos_dev[pos_dev_index]
x_text_neg_dev, x_code_neg_dev = x_text_neg_dev[neg_dev_index], x_code_neg_dev[neg_dev_index]
print "Making a list of pairs for dev data"
print x_text_pos_dev.shape, x_code_pos_dev.shape
print x_text_neg_dev.shape, x_code_neg_dev.shape


# cnn_model_name = "with_dropout_with_batchnorm"
# cnn_model_name = "no_dropout_with_batchnorm"
cnn_model_name = "with_dropout_no_batchnorm_noFusionLayer"
print cnn_model_name
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if FLAGS.no_dropoutAndbatchnorm == cnn_model_name:
            cnn = CNNPatchWithoutDropOutAndBatchNorm(
                max_length_text=text_max_document_length,
                max_length_code=code_max_document_length,
                vocab_size_text=len(text_vocab_processor.vocabulary_),
                vocab_size_code=len(code_vocab_processor.vocabulary_),
                embedding_size_text=FLAGS.embedding_dim_text,
                embedding_size_code=FLAGS.embedding_dim_code,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.with_dropout_no_batchnorm == cnn_model_name:
            cnn = CNNPatchWithDropOutNoBatchNorm(
                max_length_text=text_max_document_length,
                max_length_code=code_max_document_length,
                vocab_size_text=len(text_vocab_processor.vocabulary_),
                vocab_size_code=len(code_vocab_processor.vocabulary_),
                embedding_size_text=FLAGS.embedding_dim_text,
                embedding_size_code=FLAGS.embedding_dim_code,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.with_dropout_with_batchnorm == cnn_model_name:
            cnn = CNNPatchWithDropOutAndBatchNorm(
                max_length_text=text_max_document_length,
                max_length_code=code_max_document_length,
                vocab_size_text=len(text_vocab_processor.vocabulary_),
                vocab_size_code=len(code_vocab_processor.vocabulary_),
                embedding_size_text=FLAGS.embedding_dim_text,
                embedding_size_code=FLAGS.embedding_dim_code,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.no_dropout_with_batchnorm == cnn_model_name:
            cnn = CNNPatchWithoutDropOutWithBatchNorm(
                max_length_text=text_max_document_length,
                max_length_code=code_max_document_length,
                vocab_size_text=len(text_vocab_processor.vocabulary_),
                vocab_size_code=len(code_vocab_processor.vocabulary_),
                embedding_size_text=FLAGS.embedding_dim_text,
                embedding_size_code=FLAGS.embedding_dim_code,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
        elif FLAGS.with_dropout_no_batchnorm_noFusionLayer == cnn_model_name:
            cnn = CNNPatchWithDropOutNoBatchNorm_noFusionLayer(
                max_length_text=text_max_document_length,
                max_length_code=code_max_document_length,
                vocab_size_text=len(text_vocab_processor.vocabulary_),
                vocab_size_code=len(code_vocab_processor.vocabulary_),
                embedding_size_text=FLAGS.embedding_dim_text,
                embedding_size_code=FLAGS.embedding_dim_code,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )
        else:
            print "You need to call a correct model name"
            exit()

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, grad_hist_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Train Summaries
        dev_summary_op = tf.merge_summary([loss_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # saver = tf.train.Saver(tf.all_variables(), max_to_keep=FLAGS.num_checkpoints)

        # write vocabulary
        text_vocab_processor.save(os.path.join(out_dir, "vocab_text"))
        code_vocab_processor.save(os.path.join(out_dir, "vocab_code"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())


        def train_step(left_text_batch, left_code_batch, right_text_batch, right_code_batch):
            """
            A single training step
            """
            # for t, c in zip(left_text_batch, left_code_batch):
            #     feed_dict = {
            #         cnn.input_text_left: [t],
            #         cnn.input_code_left: [c],
            #         cnn.input_text_right: [t],
            #         cnn.input_code_right: [c],
            #         cnn.dropout_keep_prob: 1.0,
            #         cnn.phase: 0
            #     }
            #     _, step, summaries, loss, s_l, s_r, r_l, r_r = sess.run(
            #         [train_op, global_step, train_summary_op, cnn.loss,
            #          cnn.score_left, cnn.score_right, cnn.ranking_left, cnn.ranking_right],
            #         feed_dict)
            #
            #     time_str = datetime.datetime.now().isoformat()
            #     print("{}: step {}, loss {:g}".format(time_str, step, loss))
            #     train_summary_writer.add_summary(summaries, step)
            #     print s_l, s_r, r_l, r_r
            #
            # exit()

            feed_dict = {
                cnn.input_text_left: left_text_batch,
                cnn.input_code_left: left_code_batch,
                cnn.input_text_right: right_text_batch,
                cnn.input_code_right: right_code_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                # cnn.phase: 1
            }
            _, step, summaries, loss = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(left_text_batch, left_code_batch, right_text_batch, right_code_batch):
            """
            Evaluate on a dev set
            """
            feed_dict = {
                cnn.input_text_left: left_text_batch,
                cnn.input_code_left: left_code_batch,
                cnn.input_text_right: right_text_batch,
                cnn.input_code_right: right_code_batch,
                cnn.dropout_keep_prob: 1.0,
                # cnn.phase: 0
            }
            step, summaries, loss = sess.run(
                [global_step, dev_summary_op, cnn.loss],
                feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

            # feed_dict = {
            #     cnn.input_text_left: left_text_batch,
            #     cnn.input_code_left: left_code_batch,
            #     cnn.input_text_right: right_text_batch,
            #     cnn.input_code_right: right_code_batch,
            #     cnn.dropout_keep_prob: 1.0,
            #     cnn.phase: 0
            # }
            # step, summaries, loss = sess.run(
            #     [global_step, dev_summary_op, cnn.loss],
            #     feed_dict)
            #
            # time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}".format(time_str, step, loss))

            # for x, y in zip(left_text_batch, left_code_batch):
            #     feed_dict = {
            #         cnn.input_text_left: [x],
            #         cnn.input_code_left: [y],
            #         cnn.input_text_right: [x],
            #         cnn.input_code_right: [y],
            #         cnn.dropout_keep_prob: 1.0,
            #         cnn.phase: 0
            #     }
            #     step, summaries, loss, rank_left, rank_right = sess.run(
            #         [global_step, dev_summary_op, cnn.loss, cnn.ranking_left, cnn.ranking_right],
            #         feed_dict)
            #
            #     time_str = datetime.datetime.now().isoformat()
            #     print("{}: step {}, loss {:g}".format(time_str, step, loss))
            #     print rank_left, rank_right
            #
            #     feed_dict = {
            #         cnn.input_text_left: [x],
            #         cnn.input_code_left: [y],
            #         cnn.input_text_right: [x],
            #         cnn.input_code_right: [y],
            #         cnn.dropout_keep_prob: 1.0,
            #         cnn.phase: 0
            #     }
            #     step, summaries, loss, rank_left, rank_right = sess.run(
            #         [global_step, dev_summary_op, cnn.loss, cnn.ranking_left, cnn.ranking_right],
            #         feed_dict)
            #
            #     time_str = datetime.datetime.now().isoformat()
            #     print("{}: step {}, loss {:g}".format(time_str, step, loss, cnn.ranking_left, cnn.ranking_right))
            #     print rank_left, rank_right
            #     exit()

            dev_summary_writer.add_summary(summaries, step)


        for i in xrange(0, FLAGS.num_iters):
            # Generate batches
            batch_text_pos, batch_code_pos, batch_text_neg, batch_code_neg = data_helpers.batch_iter(
                text_pos=x_text_pos_train, code_pos=x_code_pos_train,
                text_neg=x_text_neg_train, code_neg=x_code_neg_train,
                batch_size=FLAGS.batch_size)

            train_step(left_text_batch=batch_text_pos, left_code_batch=batch_code_pos,
                       right_text_batch=batch_text_neg, right_code_batch=batch_code_pos)

            if (i + 1) % FLAGS.evaluate_every == 0:
                print "\nEvaluation:"
                dev_step(left_text_batch=x_text_pos_dev, left_code_batch=x_code_pos_dev,
                         right_text_batch=x_text_neg_dev, right_code_batch=x_code_neg_dev)
                print ""

            if (i + 1) % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=i)
                print "Saved model checkpoint to {}\n".format(path)
