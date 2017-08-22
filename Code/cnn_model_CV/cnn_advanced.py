import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm


class CNNAdvanced(object):
    """
    A CNN for bug fixing patches classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Building CNN for each commit message and commit code.
    Adding extended features.
    """

    def pool_outputs(self, embedded_chars_expanded, W, b, max_length, filter_size, model):
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")

        if model == "cnn_batchnorm":
            # Apply batchnorm
            conv_bn = batch_norm(conv, center=False, scale=False, is_training=self.phase)
            # Apply nonlinearity -- using elu
            h = tf.nn.elu(tf.nn.bias_add(conv_bn, b), name="elu")
        else:
            # Apply nonlinearity -- using elu
            h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")

        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, max_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        return pooled

    def h_pool(self, num_filters_total, pooled_outputs):
        h_pool_ = tf.reshape(tf.concat(3, pooled_outputs), [-1, num_filters_total])
        return h_pool_

    # ==================================================
    # ==================================================
    def __init__(self, max_length_text, max_length_code, vocab_size_text, vocab_size_code,
                 embedding_size_text, embedding_size_code, filter_sizes, num_filters,
                 l2_reg_lambda, model):
        self.max_length_text = max_length_text
        self.max_length_code = max_length_code
        self.vocab_size_text = vocab_size_text
        self.vocab_size_code = vocab_size_code
        self.embedding_size_text = embedding_size_text
        self.embedding_size_code = embedding_size_code
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        self.model = model

    # ==================================================
    # ==================================================
    def _create_placeholders(self):
        # Placeholders for input and dropout
        # Input left has higher rank to input right, we don't need y_label
        self.input_text_left = tf.placeholder(tf.int32, [None, self.max_length_text], name='input_text_left')
        self.input_code_left = tf.placeholder(tf.int32, [None, self.max_length_code], name='input_code_left')

        self.input_text_right = tf.placeholder(tf.int32, [None, self.max_length_text], name='input_text_right')
        self.input_code_right = tf.placeholder(tf.int32, [None, self.max_length_code], name='input_code_right')
        self.l2_loss = tf.constant(0.0)  # we don't use regularization in our model
        self.num_filters_total = self.num_filters * len(self.filter_sizes)

        # Create a placeholder for dropout and batchnorm. They depends on the model that we are going to use
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.phase = tf.placeholder(tf.bool, name="is_training_phase")

    # ==================================================
    # ==================================================
    def _create_embedding_text_code_layer(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_text = tf.Variable(
                tf.random_uniform([self.vocab_size_text, self.embedding_size_text], -1.0, 1.0),
                name="W_text")
            self.W_code = tf.Variable(
                tf.random_uniform([self.vocab_size_code, self.embedding_size_code], -1.0, 1.0),
                name="W_code")

    def _create_embedding_text_layer(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding_text"):
            self.W_text = tf.Variable(
                tf.random_uniform([self.vocab_size_text, self.embedding_size_text], -1.0, 1.0),
                name="W_text")

    def _create_embedding_code_layer(self):
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding_code"):
            self.W_code = tf.Variable(
                tf.random_uniform([self.vocab_size_code, self.embedding_size_code], -1.0, 1.0),
                name="W_code")

    # ==================================================
    # ==================================================
    def _create_embedding_text_left_layer(self):
        # Embedding layer for text at the left layer
        embedded_chars_text_left = tf.nn.embedding_lookup(self.W_text, self.input_text_left)
        self.embedded_chars_expanded_text_left = tf.expand_dims(embedded_chars_text_left, -1)

    def _create_embedding_code_left_layer(self):
        embedded_chars_code_left = tf.nn.embedding_lookup(self.W_code, self.input_code_left)
        self.embedded_chars_expanded_code_left = tf.expand_dims(embedded_chars_code_left, -1)

    # ==================================================
    # ==================================================
    def _create_embedding_text_right_layer(self):
        embedded_chars_text_right = tf.nn.embedding_lookup(self.W_text, self.input_text_right)
        self.embedded_chars_expanded_text_right = tf.expand_dims(embedded_chars_text_right, -1)

    def _create_embedding_code_right_layer(self):
        embedded_chars_code_right = tf.nn.embedding_lookup(self.W_code, self.input_code_right)
        self.embedded_chars_expanded_code_right = tf.expand_dims(embedded_chars_code_right, -1)

    # ==================================================
    # ==================================================
    def _create_weight_conv_text_layer(self):
        self.w_filter_text, self.b_filter_text = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                with tf.name_scope("weight-conv-maxpool-text-%s" % filter_size):
                    filter_shape_text = [filter_size, self.embedding_size_text, 1, self.num_filters]
                    # Convolution Layer
                    w = tf.Variable(tf.truncated_normal(filter_shape_text, stddev=0.1), name="W_filter_text")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    self.w_filter_text.append(w)
                    self.b_filter_text.append(b)

    def _create_weight_conv_code_layer(self):
        self.w_filter_code, self.b_filter_code = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                with tf.name_scope("weight-conv-maxpool-code-%s" % filter_size):
                    filter_shape_code = [filter_size, self.embedding_size_code, 1, self.num_filters]
                    # Convolution Layer
                    w = tf.Variable(tf.truncated_normal(filter_shape_code, stddev=0.1), name="W_filter_code")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                    self.w_filter_code.append(w)
                    self.b_filter_code.append(b)

    def _create_conv_maxpool_text_left_layer(self):
        self.pooled_outputs_text_left = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                # convolution + maxpool for text
                self.pooled_outputs_text_left.append(
                    self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_text_left,
                                      W=self.w_filter_text[i], b=self.b_filter_text[i],
                                      max_length=self.max_length_text, filter_size=filter_size, model=self.model))

    def _create_conv_maxpool_code_left_layer(self):
        self.pooled_outputs_code_left = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                # convolution + maxpool for code
                self.pooled_outputs_code_left.append(
                    self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_code_left,
                                      W=self.w_filter_code[i], b=self.b_filter_code[i],
                                      max_length=self.max_length_code, filter_size=filter_size, model=self.model))

    def _create_conv_maxpool_text_right_layer(self):
        self.pooled_outputs_text_right = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                # convolution + maxpool for text
                self.pooled_outputs_text_right.append(
                    self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_text_right,
                                      W=self.w_filter_text[i], b=self.b_filter_text[i],
                                      max_length=self.max_length_text, filter_size=filter_size, model=self.model))

    def _create_conv_maxpool_code_right_layer(self):
        self.pooled_outputs_code_right = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device("/cpu:" + str(filter_size)):
                # convolutoin + maxpool for code
                self.pooled_outputs_code_right.append(
                    self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_code_right,
                                      W=self.w_filter_code[i], b=self.b_filter_text[i],
                                      max_length=self.max_length_code, filter_size=filter_size, model=self.model))

    def _create_conv_maxpool_left_right_layer(self):
        self.pooled_outputs_text_left, self.pooled_outputs_code_left = [], []
        self.pooled_outputs_text_right, self.pooled_outputs_code_right = [], []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.device('/cpu:' + str(filter_size)):
                # convolution + maxpool for text
                with tf.name_scope("conv-maxpool-text-%s" % filter_size):
                    filter_shape_text = [filter_size, self.embedding_size_text, 1, self.num_filters]
                    # Convolution Layer
                    W = tf.Variable(tf.truncated_normal(filter_shape_text, stddev=0.1), name="W_filter_text")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")

                    self.pooled_outputs_text_left.append(
                        self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_text_left, W=W, b=b,
                                          max_length=self.max_length_text, filter_size=filter_size, model=self.model))
                    self.pooled_outputs_text_right.append(
                        self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_text_right, W=W, b=b,
                                          max_length=self.max_length_text, filter_size=filter_size, model=self.model))
                # convolution + maxpool for code
                with tf.name_scope("conv-maxpool-code-%s" % filter_size):
                    filter_shape_code = [filter_size, self.embedding_size_code, 1, self.num_filters]
                    # Convolution Layer
                    W = tf.Variable(tf.truncated_normal(filter_shape_code, stddev=0.1), name="W_filter_code")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")

                    self.pooled_outputs_code_left.append(
                        self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_code_left, W=W, b=b,
                                          max_length=self.max_length_code, filter_size=filter_size, model=self.model))
                    self.pooled_outputs_code_right.append(
                        self.pool_outputs(embedded_chars_expanded=self.embedded_chars_expanded_code_right, W=W, b=b,
                                          max_length=self.max_length_code, filter_size=filter_size, model=self.model))

    # ==================================================
    # ==================================================
    def _create_pool_left_feature(self):
        with tf.name_scope("pool_feature_left"):
            h_pool_text_left = self.h_pool(num_filters_total=self.num_filters_total,
                                           pooled_outputs=self.pooled_outputs_text_left)
            h_pool_code_left = self.h_pool(num_filters_total=self.num_filters_total,
                                           pooled_outputs=self.pooled_outputs_code_left)
            self.h_fusion_left = tf.concat(1, [h_pool_text_left, h_pool_code_left], name="fusion_left")

    def _create_pool_right_feature(self):
        with tf.name_scope("pool_feature_right"):
            h_pool_text_right = self.h_pool(num_filters_total=self.num_filters_total,
                                            pooled_outputs=self.pooled_outputs_text_right)
            h_pool_code_right = self.h_pool(num_filters_total=self.num_filters_total,
                                            pooled_outputs=self.pooled_outputs_code_right)
            self.h_fusion_right = tf.concat(1, [h_pool_text_right, h_pool_code_right], name="fusion_right")

    def _create_pool_left_right_feature(self):
        with tf.name_scope("pool_feature"):
            h_pool_text_left = self.h_pool(num_filters_total=self.num_filters_total,
                                           pooled_outputs=self.pooled_outputs_text_left)
            h_pool_code_left = self.h_pool(num_filters_total=self.num_filters_total,
                                           pooled_outputs=self.pooled_outputs_code_left)

            h_pool_text_right = self.h_pool(num_filters_total=self.num_filters_total,
                                            pooled_outputs=self.pooled_outputs_text_right)
            h_pool_code_right = self.h_pool(num_filters_total=self.num_filters_total,
                                            pooled_outputs=self.pooled_outputs_code_right)

            self.h_fusion_left = tf.concat(1, [h_pool_text_left, h_pool_code_left], name="fusion_left")
            self.h_fusion_right = tf.concat(1, [h_pool_text_right, h_pool_code_right], name="fusion_right")

    # ==================================================
    # ==================================================
    def _create_dropout_left_right(self):
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_fusion_left = tf.nn.dropout(self.h_fusion_left, self.dropout_keep_prob, name="h_fusion_drop_left")
            self.h_fusion_right = tf.nn.dropout(self.h_fusion_right, self.dropout_keep_prob, name="h_fusion_drop_right")

    def _create_dropout_left(self):
        # Add dropout
        with tf.name_scope("dropout_left"):
            self.h_fusion_left = tf.nn.dropout(self.h_fusion_left, self.dropout_keep_prob, name="h_fusion_drop_left")

    def _create_dropout_right(self):
        # Add dropout
        with tf.name_scope("dropout_right"):
            self.h_fusion_right = tf.nn.dropout(self.h_fusion_right, self.dropout_keep_prob, name="h_fusion_drop_right")

    # ==================================================
    # ==================================================
    def _create_output_left_right(self):
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W_output",
                shape=[self.h_fusion_left.get_shape()[1], 1],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)

            self.score_left = tf.nn.xw_plus_b(self.h_fusion_left, W, b, name="score_left")
            self.score_right = tf.nn.xw_plus_b(self.h_fusion_right, W, b, name="score_right")

    def _create_weight_output_layer(self):
        with tf.name_scope("weight_output"):
            self.W_output = tf.get_variable(
                "W_output",
                shape=[self.h_fusion_left.get_shape()[1], 1],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b_output = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.l2_loss += tf.nn.l2_loss(self.W_output)
            self.l2_loss += tf.nn.l2_loss(self.b_output)

    def _create_output_left(self):
        with tf.name_scope("output_left"):
            self.score_left = tf.nn.xw_plus_b(self.h_fusion_left, self.W_output, self.b_output, name="score_left")

    def _create_output_right(self):
        with tf.name_scope("output_right"):
            self.score_right = tf.nn.xw_plus_b(self.h_fusion_right, self.W_output, self.b_output, name="score_right")

    # ==================================================
    # ==================================================
    def _create_loss_left_right(self):
        with tf.name_scope("loss"):
            losses = tf.sigmoid(tf.sub(self.score_left, self.score_right, name="subtract"))
            self.loss = tf.reduce_mean(losses, name="loss") + self.l2_reg_lambda * self.l2_loss

    # ==================================================
    # ==================================================
    def _create_rank_left_right(self):
        with tf.name_scope("rank"):
            self.ranking_left = tf.sigmoid(self.score_left, name="ranking_left")
            self.ranking_right = tf.sigmoid(self.score_right, name="ranking_right")

    def _create_rank_left(self):
        with tf.name_scope("rank_left"):
            self.ranking_left = tf.sigmoid(self.score_left, name="ranking_left")

    def _create_rank_right(self):
        with tf.name_scope("rank_right"):
            self.ranking_right = tf.sigmoid(self.score_right, name="ranking_right")

    def build_graph(self):
        if self.model == "cnn_dropout":
            # ==================================================
            # General
            # self._create_placeholders()
            # self._create_embedding_text_code_layer()
            # self._create_embedding_text_left_layer()
            # self._create_embedding_code_left_layer()
            # self._create_embedding_text_right_layer()
            # self._create_embedding_code_right_layer()
            # self._create_conv_maxpool_left_right_layer()
            # self._create_pool_left_right_feature()
            # self._create_dropout_left_right()
            # self._create_output_left_right()
            # self._create_loss_left_right()
            # self._create_rank_left_right()

            # ==================================================
            # More details
            self._create_placeholders()
            self._create_embedding_text_code_layer()
            self._create_embedding_text_left_layer()
            self._create_embedding_code_left_layer()
            self._create_embedding_text_right_layer()
            self._create_embedding_code_right_layer()
            self._create_weight_conv_text_layer()
            self._create_weight_conv_code_layer()
            self._create_conv_maxpool_text_left_layer()
            self._create_conv_maxpool_code_left_layer()
            self._create_conv_maxpool_text_right_layer()
            self._create_conv_maxpool_code_right_layer()
            self._create_pool_left_feature()
            self._create_pool_right_feature()
            self._create_dropout_left()
            self._create_dropout_right()
            self._create_weight_output_layer()
            self._create_output_left()
            self._create_output_right()
            self._create_loss_left_right()
            self._create_rank_left()
            self._create_rank_right()
        else:
            print "You need to give the correct input name"
            exit()
