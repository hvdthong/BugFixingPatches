import tensorflow as tf


def generate_test_data(testpatch, all_patches, file_name):
    file_ = list(open(testpatch, "r").readlines())
    ids_file = [e.strip().split("\t")[1] for e in file_]

    all_ = list(open(all_patches, "r").readlines())
    ids_all = [e.strip().split(":")[0] for e in all_]
    commits_all = [":".join(e.strip().split(":")[1:]) for e in all_]

    ids_overlap = [ids_all.index(i) for i in ids_file]
    commits_overlap = [commits_all[i] for i in ids_overlap]

    for c in commits_overlap:
        print c

    with open(file_name, "w") as file_:
        [file_.write(c.strip() + '\n') for c in commits_overlap]


###################################################################################################
###################################################################################################
tf.flags.DEFINE_string("file_labels", "./data/id_testlabel.txt",
                       "List of all bug fixing patches ids including ground truth label (labeled by human)")
tf.flags.DEFINE_string("all_patches", "./data/CNN_all.txt",
                       "List of stable bug fixing patches for training CNN ranking model")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

generate_test_data(testpatch=FLAGS.file_labels, all_patches=FLAGS.all_patches, file_name="./data/id_testlabel.commits")
