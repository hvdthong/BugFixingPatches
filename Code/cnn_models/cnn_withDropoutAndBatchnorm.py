import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm


class CNNPatchWithDropOutAndBatchNorm(object):
    """
    A CNN for bug fixing patches classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def pool_outputs(self, embedded_chars_expanded, W, b, max_length, filter_size):
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        # Apply batchnorm
        conv_bn = batch_norm(conv, decay=0.999, center=True, scale=True, is_training=self.phase)

        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv_bn, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, max_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        return pooled

    def h_pool(self, num_filters_total, pooled_outputs):
        h_pool_ = tf.reshape(tf.concat(len(pooled_outputs), pooled_outputs), [-1, num_filters_total])
        return h_pool_

    def h_fusion(self, W, h_pool_text, h_pool_code):
        transform = tf.matmul(h_pool_text, W)
        return tf.mul(transform, h_pool_code)

    def __init__(
            self, max_length_text, max_length_code, vocab_size_text,
            vocab_size_code, embedding_size_text, embedding_size_code, filter_sizes, num_filters,
            l2_reg_lambda=0.0):
        # Placeholders for input and dropout
        # Input left has higher rank to input right, we don't need y_label
        self.input_text_left = tf.placeholder(tf.int32, [None, max_length_text], name='input_text_left')
        self.input_code_left = tf.placeholder(tf.int32, [None, max_length_code], name='input_code_left')

        self.input_text_right = tf.placeholder(tf.int32, [None, max_length_text], name='input_text_right')
        self.input_code_right = tf.placeholder(tf.int32, [None, max_length_code], name='input_code_right')

        # self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.l2_loss = tf.constant(0.0)
        self.phase = tf.placeholder(tf.bool, name="is_training_phase")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_text = tf.Variable(
                tf.random_uniform([vocab_size_text, embedding_size_text], -1.0, 1.0),
                name="W_text")
            self.W_code = tf.Variable(
                tf.random_uniform([vocab_size_code, embedding_size_code], -1.0, 1.0),
                name="W_code")

        # Embedding layer for text and code
        embedded_chars_text_left = tf.nn.embedding_lookup(self.W_text, self.input_text_left)
        embedded_chars_expanded_text_left = tf.expand_dims(embedded_chars_text_left, -1)
        embedded_chars_code_left = tf.nn.embedding_lookup(self.W_code, self.input_code_left)
        embedded_chars_expanded_code_left = tf.expand_dims(embedded_chars_code_left, -1)

        embedded_chars_text_right = tf.nn.embedding_lookup(self.W_text, self.input_text_right)
        embedded_chars_expanded_text_right = tf.expand_dims(embedded_chars_text_right, -1)
        embedded_chars_code_right = tf.nn.embedding_lookup(self.W_code, self.input_code_right)
        embedded_chars_expanded_code_right = tf.expand_dims(embedded_chars_code_right, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_text_left, pooled_outputs_code_left = [], []
        pooled_outputs_text_right, pooled_outputs_code_right = [], []

        for i, filter_size in enumerate(filter_sizes):
            with tf.device('/cpu:' + str(filter_size)):
                # convolution + maxpool for text
                with tf.name_scope("conv-maxpool-text-%s" % filter_size):
                    filter_shape_text = [filter_size, embedding_size_text, 1, num_filters]
                    # Convolution Layer
                    W = tf.Variable(tf.truncated_normal(filter_shape_text, stddev=0.1), name="W_filter_text")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                    pooled_outputs_text_left.append(
                        self.pool_outputs(embedded_chars_expanded=embedded_chars_expanded_text_left, W=W, b=b,
                                          max_length=max_length_text, filter_size=filter_size))
                    pooled_outputs_text_right.append(
                        self.pool_outputs(embedded_chars_expanded=embedded_chars_expanded_text_right, W=W, b=b,
                                          max_length=max_length_text, filter_size=filter_size))

                with tf.name_scope("conv-maxpool-code-%s" % filter_size):
                    filter_shape_code = [filter_size, embedding_size_code, 1, num_filters]
                    # Convolution Layer
                    W = tf.Variable(tf.truncated_normal(filter_shape_code, stddev=0.1), name="W_filter_code")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                    pooled_outputs_code_left.append(
                        self.pool_outputs(embedded_chars_expanded=embedded_chars_expanded_code_left, W=W, b=b,
                                          max_length=max_length_code, filter_size=filter_size))
                    pooled_outputs_code_right.append(
                        self.pool_outputs(embedded_chars_expanded=embedded_chars_expanded_code_right, W=W, b=b,
                                          max_length=max_length_code, filter_size=filter_size))

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool_text_left = self.h_pool(num_filters_total=num_filters_total, pooled_outputs=pooled_outputs_text_left)
        h_pool_code_left = self.h_pool(num_filters_total=num_filters_total, pooled_outputs=pooled_outputs_code_left)

        h_pool_text_right = self.h_pool(num_filters_total=num_filters_total, pooled_outputs=pooled_outputs_text_right)
        h_pool_code_right = self.h_pool(num_filters_total=num_filters_total, pooled_outputs=pooled_outputs_code_right)

        with tf.name_scope("fusion"):
            W = tf.get_variable(
                name='W_fusion',
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            h_fusion_left = self.h_fusion(W=W, h_pool_text=h_pool_text_left, h_pool_code=h_pool_code_left)
            h_fusion_right = self.h_fusion(W=W, h_pool_text=h_pool_text_right, h_pool_code=h_pool_code_right)

        # Add dropout
        with tf.name_scope("dropout"):
            h_fusion_drop_left = tf.nn.dropout(h_fusion_left, self.dropout_keep_prob, name="h_fusion_drop_left")
            h_fusion_drop_right = tf.nn.dropout(h_fusion_right, self.dropout_keep_prob, name="h_fusion_drop_right")


        # Add output
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[h_fusion_left.get_shape()[1], 1],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            self.score_left = tf.nn.xw_plus_b(h_fusion_drop_left, W, b, name="score_left")
            self.score_right = tf.nn.xw_plus_b(h_fusion_drop_right, W, b, name="score_right")

        with tf.name_scope("loss"):
            losses = tf.sigmoid(tf.sub(self.score_left, self.score_right, name="subtract"))
            self.loss = tf.reduce_mean(losses, name="loss") + l2_reg_lambda * self.l2_loss

        with tf.name_scope("rank"):
            self.ranking_left = tf.sigmoid(self.score_left, name="ranking_left")
            self.ranking_right = tf.sigmoid(self.score_right, name="ranking_right")


