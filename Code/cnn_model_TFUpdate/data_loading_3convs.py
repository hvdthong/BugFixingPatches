import tensorflow as tf
from process_data_3convs import input_stable_path, len_path_data


def model_parameters():
    # Parameters
    # ==================================================
    # Data loading params
    tf.flags.DEFINE_integer("seed", 10, "Random seed (default:123)")
    tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation.")
    tf.flags.DEFINE_integer("folds", 5, "Number of folds in training data")
    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim_text", 32, "Dimensionality of character embedding for text (default: 128)")
    tf.flags.DEFINE_integer("embedding_dim_code", 16, "Dimensionality of character embedding for code (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", "1, 2, 3", "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", 32, "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_integer("num_hidden", 100, "Number of hidden layer units (default: 100)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 1e-5, "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate for optimization techniques")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 200)")
    tf.flags.DEFINE_integer("num_iters", 50000,
                            "Number of training iterations; the size of each iteration is the batch size "
                            "(default: 1000)")
    tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 100, "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_integer("num_devs", 2000, "Number of dev pairs for text and code")
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    return tf


def loading_training_data(msg, addedcode, removedcode, codefile):
    train_len = len_path_data(options="eq")
    print train_len
    msg_train, addedcode_train, removedcode_train, codefile_train = msg[:train_len], addedcode[:train_len], \
                                                                    removedcode[:train_len], codefile[:train_len]
    print msg_train.shape, addedcode_train.shape, removedcode_train.shape, codefile_train.shape
    exit()
