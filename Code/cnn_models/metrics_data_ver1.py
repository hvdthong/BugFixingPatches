from metrics import precision_recall_f1
from metrics import accuracy_at_k
import numpy as np
from metrics_helpers import average_precision
import tensorflow as tf


def mapping_predicted_true_label_data_ver1(pred_random_lbl, index_random_lbl, pos_file, neg_file):
    pos_file, neg_file = list(open(pos_file, "r").readlines()), list(open(neg_file, "r").readlines())
    pos_file, neg_file = [1 for v in pos_file], [0 for v in neg_file]
    all_ = pos_file + neg_file
    true_random_lbl = [all_[i] for i in index_random_lbl]
    num_pos = len([e for e in true_random_lbl if e == 1])
    predicted_ = [{"index": i, "score": float(p)} for i, p in enumerate(pred_random_lbl)]
    predicted_ = sorted(predicted_, key=lambda k: k["score"], reverse=True)
    predicted_label = [{"index": d.get("index"), "score": d.get("score"), "label": 1}
                       if i < num_pos else {"index": d.get("index"), "score": d.get("score"), "label": 0}
                       for i, d in enumerate(predicted_)]
    predicted_label = sorted(predicted_label, key=lambda k: k["index"])
    predicted_label = [d.get("label") for d in predicted_label]
    return predicted_label, true_random_lbl


def random_label(top100_lbl, pos_lbl, neg_lbl, result_lbl, randomFlag):
    if randomFlag == 1:
        # Test on random labeled data
        top100_lbl = list(open(top100_lbl, "r").readlines())
        top100_lbl = [t.strip().split("\t")[1] for t in top100_lbl]
        pos_lbl, neg_lbl = list(open(pos_lbl, "r").readlines()), list(open(neg_lbl, "r").readlines())
        pos_lbl, neg_lbl = [t.strip() for t in pos_lbl], [t.strip() for t in neg_lbl]
        result_lbl = list(open(result_lbl, "r").readlines())
        result_lbl = [t.strip() for t in result_lbl]
        all_lbl = pos_lbl + neg_lbl
        index_random_lbl = [all_lbl.index(l) for l in all_lbl if l not in top100_lbl]
        random_lbl = [result_lbl[i] for i in index_random_lbl]
        return random_lbl, index_random_lbl
    elif randomFlag == 2:
        # Test on all labeled data
        result_lbl = list(open(result_lbl, "r").readlines())
        result_lbl = [t.strip() for t in result_lbl]
        index_result_lbl = [i for i in range(0, len(result_lbl))]
        return result_lbl, index_result_lbl
    elif randomFlag == 3:
        # Test on top 100 labeled data
        top100_lbl = list(open(top100_lbl, "r").readlines())
        top100_lbl = [t.strip().split("\t")[1] for t in top100_lbl]
        pos_lbl, neg_lbl = list(open(pos_lbl, "r").readlines()), list(open(neg_lbl, "r").readlines())
        pos_lbl, neg_lbl = [t.strip() for t in pos_lbl], [t.strip() for t in neg_lbl]
        result_lbl = list(open(result_lbl, "r").readlines())
        result_lbl = [t.strip() for t in result_lbl]
        all_lbl = pos_lbl + neg_lbl
        index_top100_lbl = [all_lbl.index(l) for l in all_lbl if l in top100_lbl]
        top100_lbl = [result_lbl[i] for i in index_top100_lbl]
        return top100_lbl, index_top100_lbl


def printing_results_data_ver1(pred_random_lbl, index_random_lbl, pos_file, neg_file):
    pred, true = mapping_predicted_true_label_data_ver1(
        pred_random_lbl=pred_random_lbl, index_random_lbl=index_random_lbl, pos_file=pos_file, neg_file=neg_file)
    # precison, recall, f1 = precision_recall_f1(y_pred=pred, y_true=true)
    # print "Precision: %.3f Recall: %.3f F1: %.3f" % (precison, recall, f1)
    acc5 = accuracy_at_k(y_true=true, y_pred=pred, top_k=5)
    acc10 = accuracy_at_k(y_true=true, y_pred=pred, top_k=10)
    acc20 = accuracy_at_k(y_true=true, y_pred=pred, top_k=20)
    acc30 = accuracy_at_k(y_true=true, y_pred=pred, top_k=30)
    print "Acc5: %.3f Acc10: %.3f Acc20: %.3f Acc30: %.3f" % (acc5, acc10, acc20, acc30)
    y_pred = np.array(pred) * np.array(true)
    print "Average precision: %.3f" % (average_precision(y_pred))


print "Working with file metrics_data_ver1.py"
# randomFlag = 1
randomFlag = 2
########################################################################################
########################################################################################
# path_top100_lbl = "./data/id_testlabel.txt"
# path_pos_lbl, path_neg_lbl = "./data_ver1/lbd100_line_aug1.pos.id", "./data_ver1/lbd100_line_aug1.neg.id"
# results = "./svm_light/label_data_ver1_prediction"
# pred_random_lbl, index_random_lbl = random_label(top100_lbl=path_top100_lbl, pos_lbl=path_pos_lbl,
#                                                  neg_lbl=path_neg_lbl, result_lbl=results, randomFlag=randomFlag)
# printing_results_data_ver1(pred_random_lbl=pred_random_lbl, index_random_lbl=index_random_lbl,
#                            pos_file=path_pos_lbl, neg_file=path_neg_lbl)

########################################################################################
########################################################################################
# checkpoint_number = "1501225832"
# checkpoint_number = "1501829732"
# checkpoint_number = "1502330736"
# checkpoint_number = "1502332800"
checkpoint_number = "1502342722"
tf.flags.DEFINE_string("checkpoint_dir", "./runs/" + checkpoint_number + "/checkpoints/",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_string("write_dir", "../results_test/", "Directory to write file results_test")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
checkpoints = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

path_top100_lbl = "../data/id_testlabel.txt"
path_pos_lbl, path_neg_lbl = "../data_ver1/lbd100_line_aug1.pos.id_maxtext150_maxcode300", \
                             "../data_ver1/lbd100_line_aug1.neg.id_maxtext150_maxcode300"

for checkpoint_file in checkpoints.all_model_checkpoint_paths:
    num_checkpoint, model_name = checkpoint_file.strip().split('/')[-3], checkpoint_file.strip().split('/')[-1]
    path_results = FLAGS.write_dir + num_checkpoint + "_" + model_name
    print path_results
    pred_random_lbl, index_random_lbl = random_label(
        top100_lbl=path_top100_lbl, pos_lbl=path_pos_lbl,
        neg_lbl=path_neg_lbl, result_lbl=path_results, randomFlag=randomFlag)
    printing_results_data_ver1(pred_random_lbl=pred_random_lbl, index_random_lbl=index_random_lbl,
                               pos_file=path_pos_lbl, neg_file=path_neg_lbl)