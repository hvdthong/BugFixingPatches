import data_helpers
import tensorflow as tf
from gensim.models import Word2Vec


tf.flags.DEFINE_string("positive_data_file", "./data/noeq100train.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/noeq100train.neg", "Data source for the negative data.")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

x_text, x_code, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
x_text_new, x_code_new = [], []
for text, code in zip(x_text, x_code):
    x_text_new.append(text.split(' '))
    x_code_new.append(code.split(' '))

# model_text = Word2Vec(x_text_new, size=128, window=5, min_count=0, workers=4, iter=30)
# print model_text.most_similar('remove')

model_code = Word2Vec(x_code_new, size=64, window=5, min_count=0, workers=4)
print model_code.most_similar('tcomma')