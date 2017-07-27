import re
import numpy as np


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def seperate_text_code(data):
    x_text, x_code = [], []
    for v in data:
        v_split = v.split('&&')
        x_text.append(v_split[0])
        x_code.append(' '.join(v_split[1:]))
    return x_text, x_code


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text, x_code = seperate_text_code(x_text)

    x_text, x_code = [clean_str(sent) for sent in x_text], [clean_str(sent) for sent in x_code]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return x_text, x_code, y


def load_data_text_and_code(file):
    examples = list(open(file, "r").readlines())
    examples = [s.strip() for s in examples]
    examples = [clean_str(sent) for sent in examples]
    x_text, x_code = seperate_text_code(examples)
    x_text, x_code = [clean_str(sent) for sent in x_text], [clean_str(sent) for sent in x_code]
    return x_text, x_code


def load_data_all(data_file):
    # Load data from files
    examples = list(open(data_file, "r").readlines())
    examples = [s.strip() for s in examples]
    examples = [clean_str(sent) for sent in examples]
    return examples


def load_data_text_all(data_file):
    # Load data from files
    examples = list(open(data_file, "r").readlines())
    examples = [s.strip() for s in examples]
    x_text, x_code = seperate_text_code(examples)
    x_text = [clean_str(sent) for sent in x_text]
    return x_text


def data_size(data_file):
    # return data size
    examples = list(open(data_file, "r").readlines())
    examples = [s.strip() for s in examples]
    return len(examples)


def batch_iter(text_pos, code_pos, text_neg, code_neg, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    if shuffle:
        index_pos, index_neg = np.random.choice(len(text_pos), batch_size), np.random.choice(len(text_neg), batch_size)
        return text_pos[index_pos], code_pos[index_pos], text_neg[index_neg], code_neg[index_neg]
    else:
        print 'We need to randomly choose instances'
        exit()

        # data_pos = np.array(data_pos)
        # data_size = len(data)
        # num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        # for epoch in range(num_iters):
        #     # Shuffle the data at each epoch
        #     if shuffle:
        #         shuffle_indices = np.random.permutation(np.arange(data_size))
        #         shuffled_data = data[shuffle_indices]
        #     else:
        #         shuffled_data = data
        #     for batch_num in range(num_batches_per_epoch):
        #         start_index = batch_num * batch_size
        #         end_index = min((batch_num + 1) * batch_size, data_size)
        #         yield shuffled_data[start_index:end_index]
