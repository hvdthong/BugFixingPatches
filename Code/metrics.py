import numpy as np
from sklearn.metrics import confusion_matrix
from metrics_helpers import precision_at_k
from metrics_helpers import average_precision
from metrics_helpers import mean_average_precision
import tensorflow as tf


def hit_rate(y_true, y_pred, top_k=1):
    y_t, y_p = np.array(y_true), np.array(y_pred)
    if top_k > len(np.where(y_t == 1)[0]):
        print "Your topK should be smaller than the length of your"
        exit()
    else:
        y_pred_top = y_p[:top_k]
        return np.sum(y_pred_top)


def accuracy_at_k(y_true, y_pred, top_k=1):
    y_t, y_p = np.array(y_true), np.array(y_pred)
    if top_k > len(np.where(y_t == 1)[0]):
        print "Your topK should be smaller than the length of your"
        exit()
    else:
        y_pred_top = y_p[:top_k]
        return np.sum(y_pred_top) / float(top_k)


def mean_avg_prec(y_true, y_pred, top_k=1):
    y_q, p_q = np.array(y_true), np.array(y_pred)
    top_idx = (-np.array(y_pred)).argsort()
    top_idx = top_idx[:top_k] if top_k > 0 else top_idx

    # MAP += average_precision_score(y_q[top_idx], p_q[top_idx])
    rel_docs = sum(y_q)
    # print y_q[top_idx], p_q[top_idx], rel_docs
    idx_correct = [i for i, x in enumerate(y_q) if x == 1]
    if len(idx_correct) > 0:
        pred_corr = p_q[top_idx]
        print sum(pred_corr[idx_correct]) / rel_docs
        return sum(pred_corr[idx_correct]) / rel_docs


def accuracy(y_true, y_pred):
    y_t, y_p = np.array(y_true), np.array(y_pred)
    correct_predictions = float(sum(y_t == y_p))
    return correct_predictions / len(y_true)


def precision_recall_f1(y_pred, y_true):
    tp = confusion_matrix(y_true=y_true, y_pred=y_pred)[1][1]
    fp = confusion_matrix(y_true=y_true, y_pred=y_pred)[1][0]
    fn = confusion_matrix(y_true=y_true, y_pred=y_pred)[0][1]
    precision_ = tp / float(tp + fp)
    recall_ = tp / float(tp + fn)
    f1_ = 2 * precision_ * recall_ / float(precision_ + recall_)
    return precision_, recall_, f1_


def mapping_predicted_true_label(predicted_file, true_file):
    true_ = list(open(true_file, "r").readlines())
    true_label = [int(e.strip().split("\t")[0]) for e in true_]

    num_pos = len([e for e in true_ if int(e.strip().split("\t")[0]) == 1])
    predicted_ = list(open(predicted_file, "r").readlines())
    predicted_ = [{"index": i, "score": float(p)} for i, p in enumerate(predicted_)]
    predicted_ = sorted(predicted_, key=lambda k: k["score"], reverse=True)
    predicted_label = [
        {"index": d.get("index"), "score": d.get("score"), "label": 1} if i < num_pos else {"index": d.get("index"),
                                                                                            "score": d.get("score"),
                                                                                            "label": 0} for i, d in
        enumerate(predicted_)]
    predicted_label = sorted(predicted_label, key=lambda k: k["index"])
    predicted_label = [d.get("label") for d in predicted_label]
    return predicted_label, true_label


def printing_results(path_true_file, path_predicted):
    pred, true = mapping_predicted_true_label(predicted_file=path_predicted, true_file=path_true_file)
    precison, recall, f1 = precision_recall_f1(y_pred=pred, y_true=true)
    print "Precision: %.3f Recall: %.3f F1: %.3f" % (precison, recall, f1)
    acc5 = accuracy_at_k(y_true=true, y_pred=pred, top_k=5)
    acc10 = accuracy_at_k(y_true=true, y_pred=pred, top_k=10)
    # acc20 = accuracy_at_k(y_true=true, y_pred=pred, top_k=20)
    acc30 = accuracy_at_k(y_true=true, y_pred=pred, top_k=30)
    # acc40 = accuracy_at_k(y_true=true, y_pred=pred, top_k=40)
    acc50 = accuracy_at_k(y_true=true, y_pred=pred, top_k=50)
    print "Acc5: %.3f Acc10: %.3f Acc30: %.3f Acc50 %.3f" % (acc5, acc10, acc30, acc50)
    y_pred = np.array(pred) * np.array(true)
    print "Average precision: %.3f" % (average_precision(y_pred))


def accuracy_at_k_graph(path_true_file, path_predicted):
    pred, true = mapping_predicted_true_label(predicted_file=path_predicted, true_file=path_true_file)
    print pred
    print true


# true_label = "./data/id_testlabel.txt"
# predicted_scores = "./results_test/1501168350_model-999"
# printing_results(path_true_file=true_label, path_predicted=predicted_scores)

########################################################################################
########################################################################################
# true_label = "./data/id_testlabel.txt"
# checkpoint_number = "1501225832"
# tf.flags.DEFINE_string("checkpoint_dir", "./runs/" + checkpoint_number + "/checkpoints/",
#                        "Checkpoint directory from training run")
# tf.flags.DEFINE_string("write_dir", "./results_test/", "Directory to write file results_test")
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
# checkpoints = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
# for checkpoint_file in checkpoints.all_model_checkpoint_paths:
#     num_checkpoint, model_name = checkpoint_file.strip().split('/')[-3], checkpoint_file.strip().split('/')[-1]
#     path_results = FLAGS.write_dir + num_checkpoint + "_" + model_name
#     print path_results
#     # if path_results == "./results_test/1501225832_model-30999":
#     #     print 'hello'
#     printing_results(path_true_file=true_label, path_predicted=path_results)

########################################################################################
########################################################################################
true_label = "./data/id_testlabel.txt"
pred_label = "./data_lpusvm/results"
# printing_results(path_true_file=true_label, path_predicted=pred_label)
accuracy_at_k_graph(path_true_file=true_label, path_predicted=pred_label)