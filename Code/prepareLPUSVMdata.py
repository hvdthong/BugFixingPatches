from boto.sdb.db.sequence import fib

from data_helpers import clean_str
from sklearn.feature_extraction.text import CountVectorizer


def lpu_svm_train_data(pos_file, neg_file, make_file):
    # Load data from files
    positive_examples = list(open(pos_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    pos_len = len(positive_examples)

    print len(positive_examples), len(negative_examples)
    # exit()
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    vec = CountVectorizer()
    data = vec.fit_transform(x_text)
    cols = data.shape[1]
    lines = []
    for i, d in enumerate(data):
        line = "+1 " if i < pos_len else "-1 "
        for j in range(cols):
            if d[0, j] > 0:
                line += str(j) + ":" + str(d[0, j]) + " "
        print i
        lines.append(line)
    with open(make_file, "w") as file_:
        [file_.write(c.strip() + '\n') for c in lines]


def lpu_svm_file_data(pos_id, neg_id, pos_file, neg_file, file_id, make_file):
    # Load all ids of pos and neg
    pos_id = list(open(pos_id, "r").readlines())
    pos_id = [s.strip().split(":")[1] for s in pos_id]
    neg_id = list(open(neg_id, "r").readlines())
    neg_id = [s.strip().split(":")[1] for s in neg_id]
    all_ids = pos_id + neg_id
    # Load ids of test lable
    file_id = list(open(file_id, "r").readlines())
    file_id = [s.strip().split("\t")[1] for s in file_id]

    print len(all_ids), len(file_id)
    # Index of all test label
    index = [all_ids.index(i) for i in file_id]

    # Load data from files
    positive_examples = list(open(pos_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    print len(positive_examples), len(negative_examples)
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    vec = CountVectorizer()
    data = vec.fit_transform(x_text)
    print data.shape

    x_file = data[index]
    print x_file.shape

    cols = data.shape[1]
    lines = []
    for i, d in enumerate(x_file):
        line = "-1 "
        for j in range(cols):
            if d[0, j] > 0:
                line += str(j + 1) + ":" + str(d[0, j]) + " "
        print i
        lines.append(line)

    with open(make_file, "w") as file_:
        [file_.write(c.strip() + '\n') for c in lines]


def modify_line(line):
    split_line = line.strip().split(" ")
    new_line = split_line[0] + " "
    for i in range(1, len(split_line)):
        tokens = split_line[i].split(":")
        ftr, value = str(int(tokens[0]) + 1), tokens[1]
        new_line += ftr + ":" + value + " "
    print new_line
    return new_line


def lpu_svm_fix_data(path_file):
    # Load all ids of pos and neg
    lines = list(open(path_file, "r").readlines())
    lines = [modify_line(l) for l in lines]

    with open(path_file, "w") as file_:
        [file_.write(c.strip() + '\n') for c in lines]



######################################################################################################
######################################################################################################
# pos_path, neg_path = "./data/train.pos", "./data/train.neg"
# new_file = "./data_lpusvm/train.txt"
# lpu_svm_train_data(pos_file=pos_path, neg_file=neg_path, make_file=new_file)

######################################################################################################
######################################################################################################
# pos_path, neg_path = "./data/train.pos", "./data/train.neg"
# pos_id, neg_id = "./data/all_ids.pos", "./data/all_ids.neg"
# file_id = "./data/id_testlabel.txt"
# new_file = "./data_lpusvm/test.txt"
# lpu_svm_file_data(pos_id=pos_id, neg_id=neg_id, pos_file=pos_path, neg_file=neg_path, file_id=file_id,
#                   make_file=new_file)

######################################################################################################
######################################################################################################
# path_file = "./data_lpusvm/train.txt"
# lpu_svm_fix_data(path_file=path_file)
path_file = "./data_lpusvm/test.txt"
lpu_svm_fix_data(path_file=path_file)