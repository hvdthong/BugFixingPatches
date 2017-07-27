import tensorflow as tf


def mapping_id_label(file_labels, file_test):
    # get the list of ids containing label (stable or non-stable)
    ids_label = list(open(file_labels, "r").readlines())
    ids_label = [e.split("\t")[1] for e in ids_label]

    # print len(ids_label)
    # exit()

    # get the list of ids and commit codes/messages of bug fixing patches
    tests = list(open(file_test, "r").readlines())
    ids = [e.split(":")[1] for e in tests]
    commits = [":".join(e.split(":")[1:]) for e in tests]

    for l in ids_label:
        print l

    for e in tests:
        if len(e.split(":")) != 3:
            label, id, features = e.split(':', 3)[0:3]
            print label, id, features
            exit()
            print e.split(":")[1]
            print ":".join(e.split(":")[2:])
            # print "something wrongs"
            # exit()
    exit()
    print ids[0], commits[0]
    print len(ids), len(commits)


def merging_ids(file_labels, train_labels, name_pos, name_neg):
    file = list(open(file_labels, "r").readlines())
    ids_file = [e.strip().split("\t")[1] for e in file]
    label_file = [e.strip().split("\t")[0] for e in file]

    train = list(open(train_labels, "r").readlines())
    ids_train = [e.strip().split(":")[1] for e in train]
    label_train = [e.strip().split(":")[0] for e in train]

    ids, labels = ids_file + ids_train, label_file + label_train

    pos_ = [l + ":" + i for i, l in zip(ids, labels) if int(l) == 1]
    neg_ = [l + ":" + i for i, l in zip(ids, labels) if int(l) == 0]

    with open(name_pos, "w") as pos_file, open(name_neg, 'w')as neg_file:
        [pos_file.write(p.strip() + '\n') for p in pos_]
        [neg_file.write(n.strip() + '\n') for n in neg_]
    print len(pos_), len(neg_)
    print file_labels, train_labels, name_pos, name_neg


def merging_data(train_data, test_data, namefile):
    train_ = list(open(train_data, "r").readlines())
    ids_train = [e.split(":")[1] for e in train_]
    commits_train = [":".join(e.split(":")[2:]) for e in train_]

    test_ = list(open(test_data, "r").readlines())
    ids_test = [e.split(":")[1] for e in test_]
    commits_test = [":".join(e.split(":")[2:]) for e in test_]

    ids_all, commits_all = ids_train + ids_test, commits_train + commits_test
    with open(namefile, "w") as file_:
        [file_.write(i.strip() + ":" + c.strip() + '\n') for i, c in zip(ids_all, commits_all)]
    print len(ids_all), len(commits_all)


def making_data(file_id, all_patches, file_name):
    file_ = list(open(file_id, "r").readlines())
    ids_file = [e.strip().split(":")[1] for e in file_]

    all_ = list(open(all_patches, "r").readlines())
    ids_all = [e.strip().split(":")[0] for e in all_]
    commits_all = [":".join(e.strip().split(":")[1:]) for e in all_]

    ids_overlap = [ids_all.index(i) for i in ids_file]
    commits_overlap = [commits_all[i] for i in ids_overlap]
    with open(file_name, "w") as file_:
        [file_.write(c.strip() + '\n') for c in commits_overlap]


# tf.flags.DEFINE_string("file_labels", "./data/id_testlabel.txt",
#                        "List of all bug fixing patches ids including ground truth label (labeled by human)")
# tf.flags.DEFINE_string("train_labels", "./data/consistId_noeq.txt",
#                        "List of all bug fixing patches ids using for training ranking model")
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
# merging_ids(file_labels=FLAGS.file_labels, train_labels=FLAGS.train_labels, name_pos="./data/all_ids.pos",
#             name_neg="./data/all_ids.neg")


###################################################################################################
###################################################################################################
# tf.flags.DEFINE_string("train_data", "./data/CNNall100noneq.txt",
#                        "List of all train bug fixing patches for training CNN ranking model")
# tf.flags.DEFINE_string("test_data", "./data/CNNall100test.txt",
#                        "List of all test bug fixing patches for training CNN ranking model")
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
# merging_data(train_data=FLAGS.train_data, test_data=FLAGS.test_data, namefile='./data/CNN_all.txt')

###################################################################################################
###################################################################################################
tf.flags.DEFINE_string("ids_neg", "./data/all_ids.neg",
                       "List of unknown bug fixing patches for training CNN ranking model")
tf.flags.DEFINE_string("ids_pos", "./data/all_ids.pos",
                       "List of stable bug fixing patches for training CNN ranking model")
tf.flags.DEFINE_string("all_patches", "./data/CNN_all.txt",
                       "List of stable bug fixing patches for training CNN ranking model")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
making_data(file_id=FLAGS.ids_neg, all_patches=FLAGS.all_patches, file_name="./data/train.neg")
making_data(file_id=FLAGS.ids_pos, all_patches=FLAGS.all_patches, file_name="./data/train.pos")
