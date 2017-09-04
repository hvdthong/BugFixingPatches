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


def len_path_data(options):
    if options == "eq" or options == "extra" or options == "lbd":
        path = input_stable_path(options=options)
    else:
        print "Wrong options"
        exit()
    length = len(list(open(path[0], "r").readlines()))
    return length


def input_path_data(options):
    paths = []
    if options == "msg":
        paths.append("../preprocessing_twoconvlayers/eq100_extra100_lbd100_line_aug1.maxtext175.maxcode250.msg.input")
    elif options == "addedcode":
        paths.append(
            "../preprocessing_twoconvlayers/eq100_extra100_lbd100_line_aug1.maxtext175.maxcode250.addedcode.input")
    elif options == "removedcode":
        paths.append(
            "../preprocessing_twoconvlayers/eq100_extra100_lbd100_line_aug1.maxtext175.maxcode250.removedcode.input")
    else:
        print "Your options are wrong, please retype again"
        exit()
    return paths[0]


def input_path_codefile(options):
    paths = []
    if options == "eq":
        paths.append(
            "../preprocessing_twoconvlayers/eq100_line_aug1.out.codefile.maxtext175.maxcode250.maxline30.mapping")
    elif options == "extra":
        paths.append(
            "../preprocessing_twoconvlayers/extra100_line_aug1.out.codefile.maxtext175.maxcode250.maxline30.mapping")
    elif options == "lbd":
        paths.append(
            "../preprocessing_twoconvlayers/eq100_line_aug1.out.codefile.maxtext175.maxcode250.maxline30.mapping")
    else:
        print "Your options are wrong, please retype again"
        exit()
    return paths[0]


def read_input_codefile(options):
    if options == "eq" or options == "extra" or options == "lbd":
        print "hello"
    else:
        print "Your options are wrong, please retype again"
        exit()


def read_input_codefile_all():
    print "hello"


def read_input_data(options):
    if options == "msg":
        path = input_path_data(options=options)
        msg = list(open(path, "r").readlines())
        msg = np.array([np.array(map(int, line.split())) for line in msg])
        return msg
    elif options == "addedcode" or options == "removedcode":
        path = input_path_data(options=options)
        code = list(open(path, "r").readlines())
        new_code = []
        for commit in code:
            commit_split = commit.split("\t")
            new_code.append(np.array([(np.array(map(int, line.split()))) for line in commit_split]))
        return np.array(new_code)
    else:
        print "Your options are wrong, please retype again"
        exit()


def write_file_normal(new_file, info):
    with open(new_file, "w") as file_:
        [file_.write(str(c) + '\n') for c in info]


def write_file_special(new_file, info, options):
    if options == "msg":
        with open(new_file, "w") as file_:
            for c in info:
                c = map(str, c.tolist())
                file_.write(" ".join(c) + '\n')
    elif options == "code":
        with open(new_file, "w") as file_:
            for d in info:
                d = [" ".join(map(str, l.tolist())) for l in d]
                file_.write("\t".join(d) + '\n')
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
    pad_msg = pad_msg_eq + pad_msg_extra + pad_msg_lbd
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
        index_empty = vocab.index("<PAD/>")
        for sent in sentences:
            line = []
            for w in sent[0].split():
                if w == "<PAD/>":
                    line.append(index_empty)
                else:
                    if w in vocab:
                        line.append(vocab.index(w))
                    else:
                        line.append(vocab.index("<UNK/>"))
            new_sents.append(np.array(line))
            if len(new_sents) >= maxinput:
                return np.array(new_sents)
            else:
                print len(new_sents)
    elif options == "code":
        output = []
        for doc in sentences:
            new_doc = []
            for d in doc:
                new_doc.append(np.array([vocab.index(w) if w in vocab else vocab.index("<UNK/>") for w in d.split()]))
            output.append(np.array(new_doc))
            if len(output) >= maxinput:
                return np.array(output)
            else:
                print len(output)


#################################################################################################################
#################################################################################################################
# filename = "../preprocessing_twoconvlayers/eq100_extra100_lbd100_line_aug1.maxtext175.maxcode250"
# pad_msg, pad_addedcode, pad_removedcode = padding_commit()
# options = "msg"
# vocab_msg = build_vocab(sentences=pad_msg, options=options)
# write_file_normal(new_file=filename + "." + options + ".dict", info=vocab_msg)
# msg = build_input_data(sentences=pad_msg, options=options, vocab=vocab_msg, maxinput=len(pad_msg))
# write_file_special(new_file=filename + "." + options + ".input", info=msg, options=options)

# options = "code"
# vocab_code = build_vocab(sentences=pad_addedcode + pad_removedcode, options=options)
# write_file_normal(new_file=filename + "." + options + ".dict", info=vocab_code)
# added_code = build_input_data(sentences=pad_addedcode, options=options, vocab=vocab_code, maxinput=len(pad_addedcode))
# write_file_special(new_file=filename + ".added" + options + ".input", info=added_code, options=options)
# removeded_code = build_input_data(sentences=pad_removedcode, options="code", vocab=vocab_code,
#                                   maxinput=len(pad_removedcode))
# write_file_special(new_file=filename + ".removed" + options + ".input", info=removeded_code, options=options)
