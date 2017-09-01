from tensorflow.contrib import learn
import numpy as np
import tensorflow as tf


def pad_sentences(sentences, seq_len, padding_word="<PAD/>"):
    padded_sentences = []
    for sent in sentences:
        if len(sent.split()) < seq_len:
            num_padding = seq_len - len(sent.split())
            new_sent = sent + " " + " ".join([padding_word] * num_padding)
        else:
            new_sent = sent
        padded_sentences.append(new_sent)
    return padded_sentences


def pad_docs(docs, doc_len, seq_len, padding_word="<PAD/>"):
    new_docs = []
    for doc in docs:
        if len(doc) < doc_len:
            num_padding_doc = doc_len - len(doc)
            for i in xrange(num_padding_doc):
                doc.append(" ".join([padding_word] * seq_len))
        new_docs.append(doc)
    return new_docs


def flat_docs(docs):
    sents = [s for d in docs for s in d]
    return sents

num_filters = 10
embedding_size = 4
vocab_size = 12
max_length = 5
max_docs = 3
filter_size = 1

filter_sizes = [1, 2, 3]
# Start interactive session
sess = tf.InteractiveSession()
x = tf.placeholder(tf.int32, shape=[None, max_docs, max_length])
W_code = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W_code")
embedded_chars_code_left = tf.nn.embedding_lookup(W_code, x)
embedded_chars_expanded_code_left = tf.expand_dims(embedded_chars_code_left, -1)
# embedded_chars_expanded_code_left = tf.reshape(embedded_chars_code_left, [-1, 5, 4])
# embedded_chars_expanded_code_left = embedded_chars_code_left
# embedded_chars = embedded_chars_expanded_code_left[None, 0, :, :]
# embedded_chars = tf.expand_dims(embedded_chars, -1)
# shape_ex = tf.shape(embedded_chars_expanded_code_left)[0]

w_, b_ = [], []
pooled_outputs = []
for filter_size in filter_sizes:
    filter_shape = [1, filter_size, embedding_size, 1, num_filters]
    w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_filter_text")
    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

    conv = tf.nn.conv3d(
        embedded_chars_expanded_code_left,
        w,
        strides=[1, 1, 1, 1, 1],
        padding="VALID",
        name="conv")

    h = tf.nn.elu(tf.nn.bias_add(conv, b), name="elu")
    pooled = tf.nn.max_pool3d(
        h,
        ksize=[1, 1, max_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1, 1],
        padding='VALID',
        name="pool")

    pooled_outputs.append(pooled)
h_pool_ = tf.reshape(tf.concat(pooled_outputs, 4), [-1, 10 * len(filter_sizes)])


if __name__ == "__main__":
    sents = ["i love deep learning", "it is a great course"]
    sents_1 = ["i love book", "book is great", "reading"]
    docs = [sents, sents_1]
    max_sents = max([len(sent.split()) for sent in sents])
    max_docs = max([len(doc) for doc in docs])
    # pad_sents = pad_sentences(sents, max_sents, padding_word="<PAD/>")
    # pad_docs = pad_docs(docs, doc_len=max_docs, seq_len=max_sents, padding_word="<PAD/>")
    pad_sents = [pad_sentences(doc, max_sents, padding_word="<PAD/>") for doc in docs]
    # print pad_sents
    pad_sents = pad_docs(docs=pad_sents, doc_len=max_docs, seq_len=max_sents, padding_word="<PAD/>")
    # print max_sents, max_docs
    # print len(pad_sents)
    new_sents = flat_docs(pad_sents)
    # print new_sents
    # print len(new_sents)
    text_vocab_processor = learn.preprocessing.VocabularyProcessor(max_sents)
    x_text = np.array(list(text_vocab_processor.fit_transform(new_sents)))
    # print len(text_vocab_processor.vocabulary_)
    # exit()
# print x_text.shape
    # print x_text
    a = x_text[:(x_text.shape[0] / 2)].reshape(-1, 15)
    a = x_text[:(x_text.shape[0] / 2)]
    # print a.reshape(-1, 15)
    # print a.reshape(3, 5)
    # print a.reshape(-1, 5)
    sess.run(tf.global_variables_initializer())

    testing = embedded_chars_expanded_code_left.eval(feed_dict={x: [a]})
    print W_code.eval()
    print testing.shape
    print a
    print testing
    # print embedded_chars.eval(feed_dict={x: [a]})
    # print shape_ex.eval(feed_dict={x: [a]})
    print pooled.eval(feed_dict={x: [a]}).shape
    print h_pool_.eval(feed_dict={x: [a]}).shape
    # print value.eval(feed_dict={x: [a]})
    # print new_docs
    # print len(new_docs)
