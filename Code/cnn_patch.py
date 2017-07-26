import tensorflow as tf


class CNNPatch(object):
    """
    A CNN for bug fixing patches classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def network_model(self, max_length_text, max_length_code, input_text, input_code,
                      embedding_size_text, embedding_size_code, filter_sizes, num_filters):


        # Embedding layer for text and code
        embedded_chars_text = tf.nn.embedding_lookup(self.W_text, input_text)
        embedded_chars_expanded_text = tf.expand_dims(embedded_chars_text, -1)
        embedded_chars_code = tf.nn.embedding_lookup(self.W_code, input_code)
        embedded_chars_expanded_code = tf.expand_dims(embedded_chars_code, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs_text, pooled_outputs_code = [], []

        for i, filter_size in enumerate(filter_sizes):
            # convolution + maxpool for text
            with tf.name_scope("conv-maxpool-text-%s" % filter_size):
                filter_shape_text = [filter_size, embedding_size_text, 1, num_filters]
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape_text, stddev=0.1), name="W_filter_text")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded_text,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_length_text - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_text.append(pooled)

            # convolution + maxpool for code
            with tf.name_scope("conv-maxpool-code-%s" % filter_size):
                filter_shape_code = [filter_size, embedding_size_code, 1, num_filters]
                # Convolution Layer
                W = tf.Variable(tf.truncated_normal(filter_shape_code, stddev=0.1), name="W_filter_code")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expanded_code,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, max_length_code - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_code.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        h_pool_text = tf.reshape(tf.concat(pooled_outputs_text, len(pooled_outputs_text)), [-1, num_filters_total],
                                 name='h_pool_text')
        h_pool_code = tf.reshape(tf.concat(pooled_outputs_code, len(pooled_outputs_code)), [-1, num_filters_total],
                                 name='h_pool_code')

        # Adding fusion layer
        with tf.name_scope('fusion'):
            W = tf.get_variable(
                name='W_fusion',
                shape=[num_filters_total, num_filters_total],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            transform_left = tf.matmul(h_pool_text, W)
            h_fusion = tf.multiply(transform_left, h_pool_code)

        # print exit()
        # h_fusion = tf.matmul(tf.transpose(h_pool_text), h_pool_code, name='h_fusion')
        # flat_size = h_pool_text.get_shape()[1] * h_pool_code.get_shape()[1]
        # print h_fusion.get_shape()
        # print h_pool_text.get_shape(), h_pool_code.get_shape(), flat_size
        # # exit()
        # h_fusion_flat = tf.reshape(h_fusion, [-1])
        # # h_fusion_flat = tf.contrib.layers.flatten(h_fusion)
        # # h_fusion_flat = tf.squeeze(h_fusion)
        # print h_fusion_flat.get_shape()
        # exit()

        # Add dropout
        with tf.name_scope("dropout"):
            h_fusion_drop = tf.nn.dropout(h_fusion, self.dropout_keep_prob, name="h_fusion_drop")
        # print h_fusion_drop.get_shape()
        # print h_fusion_drop.get_shape()[1]
        # exit()

        with tf.name_scope('output'):
            W = tf.get_variable(
                "W_output",
                shape=[h_fusion_drop.get_shape()[1], 1],
                initializer=tf.contrib.layers.xavier_initializer())

            b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            # print W.get_shape(), h_fusion_drop.get_shape(), tf.transpose(h_fusion_drop).get_shape()
            # exit()

            ranking_score = tf.nn.xw_plus_b(h_fusion_drop, W, b, name="scores")
            # print ranking_score.get_shape()
            # exit()
        return ranking_score

    def __init__(
            self, max_length_text, max_length_code, vocab_size_text,
            vocab_size_code, embedding_size_text, embedding_size_code, filter_sizes, num_filters, num_hidden,
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

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W_text = tf.Variable(
                tf.random_uniform([vocab_size_text, embedding_size_text], -1.0, 1.0),
                name="W_text")
            self.W_code = tf.Variable(
                tf.random_uniform([vocab_size_code, embedding_size_code], -1.0, 1.0),
                name="W_code")

        with tf.variable_scope("siamese") as scope:
            self.score_left = self.network_model(max_length_text=max_length_text,
                                                 max_length_code=max_length_code,
                                                 input_text=self.input_text_left,
                                                 input_code=self.input_code_left,
                                                 embedding_size_text=embedding_size_text,
                                                 embedding_size_code=embedding_size_code,
                                                 filter_sizes=filter_sizes,
                                                 num_filters=num_filters)
            scope.reuse_variables()
            self.score_right = self.network_model(max_length_text=max_length_text,
                                                  max_length_code=max_length_code,
                                                  input_text=self.input_text_right,
                                                  input_code=self.input_code_right,
                                                  embedding_size_text=embedding_size_text,
                                                  embedding_size_code=embedding_size_code,
                                                  filter_sizes=filter_sizes,
                                                  num_filters=num_filters)

        with tf.name_scope("loss"):
            self.loss = tf.sigmoid(tf.subtract(self.score_left, self.score_right, name="loss"))

        print self.score_left.get_shape(), self.score_right.get_shape(), self.loss.get_shape()
        exit()

            # # Compute similarity
            # with tf.name_scope("similarity"):
            #     W = tf.get_variable(
            #         "W",
            #         shape=[num_filters_total, num_filters_total],
            #         initializer=tf.contrib.layers.xavier_initializer())
            #     self.transform_left = tf.matmul(self.h_pool_text, W)
            #     self.sims = tf.reduce_sum(tf.multiply(self.transform_left, self.h_pool_code), 1, keep_dims=True)
            #
            # # Keeping track of l2 regularization loss (optional)
            # l2_loss = tf.constant(0.0)
            #
            # # Make input for classification
            # self.new_input = tf.concat([self.h_pool_text, self.h_pool_code], 1, name='new_input')
            #
            # # Adding hidden layer
            # with tf.name_scope('hidden'):
            #     W = tf.get_variable(
            #         "W_hidden",
            #         shape=[2 * num_filters_total, num_hidden],
            #         initializer=tf.contrib.layers.xavier_initializer())
            #     b = tf.Variable(tf.constant(0.1, shape=[num_hidden]), name="b")
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.hidden_output = tf.nn.relu(tf.nn.xw_plus_b(self.new_input, W, b, name="hidden_output"))
            #
            # # Add dropout
            # with tf.name_scope("dropout"):
            #     self.h_drop = tf.nn.dropout(self.hidden_output, self.dropout_keep_prob, name="hidden_output_drop")
            #
            # # Final (unnormalized) scores and predictions
            # with tf.name_scope("output"):
            #     W = tf.get_variable(
            #         "W_output",
            #         shape=[num_hidden, 1],
            #         initializer=tf.contrib.layers.xavier_initializer())
            #     b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.rankingscores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")


            # self.normalizedscores = tf.nn.softmax(self.scores, name="normalizedscores")
            # self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # # CalculateMean cross-entropy loss
            # with tf.name_scope("loss"):
            #     losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            #     self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            #
            # # Accuracy
            # with tf.name_scope("accuracy"):
            #     correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")