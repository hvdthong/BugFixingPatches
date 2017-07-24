import tensorflow as tf
import data_helpers
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def similarity_score(pos, neg, num_random, repeat, seed):
    # np.random.seed(seed)
    pos_index, neg_index = np.random.choice(pos.shape[0], num_random), np.random.choice(neg.shape[0], num_random)
    # print pos_index, neg_index
    pos_values, neg_values = pos[pos_index], neg[neg_index]
    scores = []
    for v_p in pos_values:
        similar_v_p = [float(cosine_similarity(v_p, v_n)) for v_n in neg_values]
        scores += similar_v_p
        if num_random >= 100:
            print repeat, num_random, len(scores)
    scores = np.array(scores)
    # print scores.shape
    return np.mean(scores), np.std(scores)


tf.flags.DEFINE_string("positive_data_file", "./data/noeq100train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/noeq100train.neg", "Data source for the negative data.")
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
pos = data_helpers.load_data_all(FLAGS.positive_data_file)
neg = data_helpers.load_data_all(FLAGS.negative_data_file)
all = pos + neg
vectorizer = CountVectorizer(min_df=1)
X = vectorizer.fit_transform(all)
X_pos, X_neg = X[0:len(pos)], X[len(pos):]
# repeats, randoms = 10, [10, 30, 50, 70, 100, 500, 700, 1000]
repeats, randoms = 10, [1000]
for num in randoms:
    averages, sds = [], []
    for i in range(0, repeats):
        average, sd = similarity_score(pos=X_neg, neg=X_neg, num_random=num, repeat=i, seed=10)
        # average, sd = similarity_score(pos=X_pos, neg=X_pos, num_random=num, repeat=i, seed=10)
        # average, sd = similarity_score(pos=X_pos, neg=X_neg, num_random=num, repeat=i, seed=10)
        averages.append(average), sds.append(sd)
    print 'Random number for each positive and negative class is: %i' % num
    averages, sds = np.array(averages), np.array(sds)
    print np.mean(averages), np.mean(sds)
