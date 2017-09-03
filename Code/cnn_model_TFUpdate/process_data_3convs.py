from padding import max_commit_file, padding_file
import numpy as np


def input_stable_path(options):
    paths = []
    if options == "eq":
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.stable.maxtext175.maxcode250")
    elif options == "extra":
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.stable.maxtext175.maxcode250")
    elif options == "lbd":
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.stable.maxtext175.maxcode250")
    else:
        print "Wrong options"
        exit()
    return paths


def input_data():
    print "hello"


def write_file_normal(new_file, info):
    with open(new_file, "w") as file_:
        [file_.write(str(c) + '\n') for c in info]


def write_file_special(new_file, info, options):
    if options == "msg":
        with open(new_file, "w") as file_:
            for c in info:
                print type(c)
                print c.shape
                # print list(c)
                c = c.tolist()
                c = map(str, c)
                print type(c)
                print c
                print len(c)
                print c[0]
                print " ".join(c)
                c = map(str, c.tolist())
                file_.write(" ".join(c) + '\n')
                exit()
            [file_.write(str(" ".join(list(c))) + '\n') for c in info]
    elif options == "code":
        print "hello"
    else:
        print "Your options are uncorrect"
        exit()


def finding_max_commit():
    maxlen_msg_eq, maxlen_code_eq, maxline_code_eq, maxlen_file_eq = max_commit_file(options="eq")
    maxlen_msg_extra, maxlen_code_extra, maxline_code_extra, maxlen_file_extra = max_commit_file(options="extra")
    maxlen_msg_lbd, maxlen_code_lbd, maxline_code_lbd, maxlen_file_lbd = max_commit_file(options="lbd")

    return max(maxlen_msg_eq, maxlen_msg_extra, maxlen_msg_lbd), \
           max(maxlen_code_eq, maxlen_code_extra, maxlen_code_lbd), \
           max(maxline_code_eq, maxline_code_extra, maxline_code_lbd), \
           max(maxlen_file_eq, maxlen_file_extra, maxlen_file_lbd)


def get_padding(pad):
    # removed id_ at the front of padding
    return [p.split("\t")[1:] for p in pad]


def padding_commit():
    maxlen_msg, maxlen_commitcode, maxline_commitcode, maxlen_file = finding_max_commit()
    print maxlen_msg, maxlen_commitcode, maxline_commitcode, maxlen_file

    pad_msg_eq, pad_addedcode_eq, pad_removedcode_eq = padding_file(options="eq", maxlen_msg=maxlen_msg,
                                                                    maxlen_code=maxlen_commitcode,
                                                                    maxline_code=maxline_commitcode)
    pad_msg_extra, pad_addedcode_extra, pad_removedcode_extra = padding_file(options="extra", maxlen_msg=maxlen_msg,
                                                                             maxlen_code=maxlen_commitcode,
                                                                             maxline_code=maxline_commitcode)
    pad_msg_lbd, pad_addedcode_lbd, pad_removedcode_lbd = padding_file(options="lbd", maxlen_msg=maxlen_msg,
                                                                       maxlen_code=maxlen_commitcode,
                                                                       maxline_code=maxline_commitcode)
    print len(pad_msg_eq), len(pad_msg_extra), len(pad_msg_lbd)
    pad_msg = pad_msg_eq + pad_msg_extra + pad_addedcode_lbd
    pad_addedcode = pad_addedcode_eq + pad_addedcode_extra + pad_addedcode_lbd
    pad_removedcode = pad_removedcode_eq + pad_removedcode_extra + pad_removedcode_lbd
    pad_msg, pad_addedcode, pad_removedcode = get_padding(pad_msg), \
                                              get_padding(pad_addedcode), get_padding(pad_removedcode)
    print len(pad_msg), len(pad_addedcode), len(pad_removedcode)
    return pad_msg, pad_addedcode, pad_removedcode


def build_vocab(sentences, options):
    if options == "msg":
        vocabulary = []
        for sent in sentences:
            vocabulary += sent[0].split()
        vocabulary = list(set(vocabulary))
    elif options == "code":
        vocabulary = []
        for doc in sentences:
            for d in doc:
                vocabulary += d.split()
        vocabulary = list(set(vocabulary))
    else:
        print "Please type correct options"
        exit()

    vocabulary.append("<UNK/>")
    return vocabulary


def build_input_data(sentences, options, vocab, maxinput):
    if options == "msg":
        new_sents = []
        for sent in sentences:
            new_sents.append(np.array([vocab.index(w) if w in vocab else vocab.index("<UNK/>")
                                       for w in sent[0].split()]))
            if len(new_sents) >= maxinput:
                return np.array(new_sents)
    elif options == "code":
        output = []
        for doc in sentences:
            new_doc = []
            for d in doc:
                new_doc.append(np.array([vocab.index(w) if w in vocab else vocab.index("<UNK/>") for w in d.split()]))
            output.append(np.array(new_doc))
            if len(output) >= maxinput:
                return np.array(output)


#################################################################################################################
#################################################################################################################
filename = "../preprocessing_twoconvlayers/eq100_extra100_lbd100_line_aug1.maxtext175.maxcode250"
pad_msg, pad_addedcode, pad_removedcode = padding_commit()
voca_msg = build_vocab(sentences=pad_msg, options="msg")
write_file_normal(new_file=filename + ".msg" + ".dict", info=voca_msg)
msg = build_input_data(sentences=pad_msg, options="msg", vocab=voca_msg, maxinput=100)
write_file_special(new_file=filename + ".msg" + ".input", info=msg, options="msg")

# vocab_code = build_vocab(sentences=pad_addedcode + pad_removedcode, options="code")
# added_code = build_input_data(sentences=pad_addedcode, options="code", vocab=vocab_code, maxinput=100
# removeded_code = build_input_data(sentences=pad_removedcode, options="code", vocab=vocab_code,
#                                   maxinput=100)
