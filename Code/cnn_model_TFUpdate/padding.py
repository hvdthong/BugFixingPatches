from data_helpers import stemming_str, remove_stopwords
import sys

reload(sys)
sys.setdefaultencoding("utf-8")


def input_path_maxtext_maxcode_mapping(options):
    paths = []
    if options == "eq":
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.msg.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.addedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.removedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.codefile.maxtext175.maxcode250.mapping")
    elif options == "extra":
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.msg.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.addedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.removedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/extra100_line_aug1.out.codefile.maxtext175.maxcode250.mapping")
    elif options == "lbd":
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.msg.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.addedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.removedcode.maxtext175.maxcode250.mapping")
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.codefile.maxtext175.maxcode250.mapping")
    else:
        print "You select wrong options"
        exit()
    return paths


def input_path_maxtext_maxcode_maxline_mapping(options):
    paths = []
    if options == "eq":
        paths.append("../preprocessing_twoconvlayers/eq100_line_aug1.out.msg.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/eq100_line_aug1.out.addedcode.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/eq100_line_aug1.out.removedcode.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/eq100_line_aug1.out.codefile.maxtext175.maxcode250.maxline30.mapping")
    elif options == "extra":
        paths.append(
            "../preprocessing_twoconvlayers/extra100_line_aug1.out.msg.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/extra100_line_aug1.out.addedcode.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/extra100_line_aug1.out.removedcode.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/extra100_line_aug1.out.codefile.maxtext175.maxcode250.maxline30.mapping")
    elif options == "lbd":
        paths.append("../preprocessing_twoconvlayers/lbd100_line_aug1.out.msg.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/lbd100_line_aug1.out.addedcode.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/lbd100_line_aug1.out.removedcode.maxtext175.maxcode250.maxline30.mapping")
        paths.append(
            "../preprocessing_twoconvlayers/lbd100_line_aug1.out.codefile.maxtext175.maxcode250.maxline30.mapping")
    else:
        print "You select wrong options"
        exit()
    return paths


########################################################################################
########################################################################################
def max_commit_message_codefile(path_file):
    commits = list(open(path_file, "r").readlines())
    msg_length = [len(c.strip().split("\t")[1].split())
                  if len(c.strip().split("\t")) > 1 else 0 for c in commits]
    return max(msg_length)


def max_commit_code(path_file):
    commits = list(open(path_file, "r").readlines())
    msg_length = []
    lines = []
    for c in commits:
        code = c.strip().split("\t")[1:]
        if len(code) > 0:
            msg_length.append(max([len(line.split()) for line in code]))
            lines.append(len(code))
        else:
            msg_length.append(0)
            lines.append(0)
    return max(msg_length), max(lines)


def max_commit_all():
    eq_msg_path = input_path_maxtext_maxcode_maxline_mapping("eq")[0]
    extra_msg_path = input_path_maxtext_maxcode_maxline_mapping("extra")[0]
    lbd_msg_path = input_path_maxtext_maxcode_maxline_mapping("lbd")[0]
    print "Max length in commit messages:"
    print max(max_commit_message_codefile(eq_msg_path), max_commit_message_codefile(extra_msg_path),
              max_commit_message_codefile(lbd_msg_path))
    maxlen_msg = max(max_commit_message_codefile(eq_msg_path),
                     max_commit_message_codefile(extra_msg_path), max_commit_message_codefile(lbd_msg_path))

    eq_addedcode_path = input_path_maxtext_maxcode_maxline_mapping("eq")[1]
    extra_addedcode_path = input_path_maxtext_maxcode_maxline_mapping("extra")[1]
    lbd_addedcode_path = input_path_maxtext_maxcode_maxline_mapping("lbd")[1]
    print "Max length in added commit code:"
    maxlen_addedcode_eq, maxline_addedcode_eq = max_commit_code(eq_addedcode_path)
    maxlen_addedcode_extra, maxline_addedcode_extra = max_commit_code(extra_addedcode_path)
    maxlen_addedcode_lbd, maxline_addedcode_lbd = max_commit_code(lbd_addedcode_path)
    maxlen_addedcode = max(maxlen_addedcode_eq, maxlen_addedcode_extra, maxlen_addedcode_lbd)
    maxline_addedcode = max(maxline_addedcode_eq, maxline_addedcode_extra, maxline_addedcode_lbd)
    print maxlen_addedcode, maxline_addedcode

    eq_removedcode_path = input_path_maxtext_maxcode_maxline_mapping("eq")[2]
    extra_removedcode_path = input_path_maxtext_maxcode_maxline_mapping("extra")[2]
    lbd_removedcode_path = input_path_maxtext_maxcode_maxline_mapping("lbd")[2]
    print "Max length in removed commit code:"
    maxlen_removedcode_eq, maxline_removedcode_eq = max_commit_code(eq_removedcode_path)
    maxlen_removedcode_extra, maxline_removedcode_extra = max_commit_code(extra_removedcode_path)
    maxlen_removedcode_lbd, maxline_removedcode_lbd = max_commit_code(lbd_removedcode_path)
    maxlen_removedcode = max(maxlen_removedcode_eq, maxlen_removedcode_extra, maxlen_removedcode_lbd)
    maxline_removedcode = max(maxline_removedcode_eq, maxline_removedcode_extra, maxline_removedcode_lbd)
    print maxlen_removedcode, maxline_removedcode

    eq_codefile_path = input_path_maxtext_maxcode_maxline_mapping("eq")[3]
    extra_codefile_path = input_path_maxtext_maxcode_maxline_mapping("extra")[3]
    lbd_codefile_path = input_path_maxtext_maxcode_maxline_mapping("lbd")[3]
    print "Max length in code file name:"
    print max(max_commit_message_codefile(eq_codefile_path),
              max_commit_message_codefile(extra_codefile_path), max_commit_message_codefile(lbd_codefile_path))
    maxlen_codefile = max(max_commit_message_codefile(eq_codefile_path),
                          max_commit_message_codefile(extra_codefile_path),
                          max_commit_message_codefile(lbd_codefile_path))
    print "Max length in commit message is %i, commit code is %i and code file is %i " \
          % (maxlen_msg, max(maxlen_addedcode, maxlen_removedcode), maxlen_codefile)
    print "Max line in commit code is %i" % max(maxline_addedcode, maxline_removedcode)
    return maxlen_msg, max(maxlen_addedcode, maxlen_removedcode), \
           max(maxline_addedcode, maxline_removedcode), maxlen_codefile


def max_commit_file(options):
    paths = input_path_maxtext_maxcode_maxline_mapping(options)
    msg_path, addedcode_path, removedcode_path, codefile_path = paths[0], paths[1], paths[2], paths[3]
    maxlen_msg = max_commit_message_codefile(msg_path)
    print "Max length in commit messages with %s:%i" % (options, maxlen_msg)
    maxlen_addedcode = max_commit_code(addedcode_path)
    print "Max length and maxline in added commit code with %s:%i, %i" \
          % (options, maxlen_addedcode[0], maxlen_addedcode[1])
    maxlen_removedcode = max_commit_code(removedcode_path)
    print "Max length and maxline in removed commit code with %s:%i, %i" \
          % (options, maxlen_removedcode[0], maxlen_removedcode[1])
    maxlen_codefile = max_commit_message_codefile(codefile_path)
    print "Max length in commit code file name with %s:%i" % (options, maxlen_codefile)
    return maxlen_msg, max(maxlen_addedcode[0], maxlen_removedcode[0]), \
           max(maxlen_addedcode[1], maxlen_removedcode[1]), maxlen_codefile


########################################################################################
########################################################################################
def pad_sentences(sentences, seq_len, padding_word="<PAD/>"):
    padded_sentences = []
    for sent in sentences:
        if len(sent.split()) < seq_len:
            num_padding = seq_len - len(sent.split())
            new_sent = sent + " " + " ".join([padding_word] * num_padding)
        else:
            new_sent = sent
        padded_sentences.append(new_sent.strip())
    return padded_sentences


def pad_docs(docs, doc_len, seq_len, padding_word="<PAD/>"):
    new_docs = []
    for doc in docs:
        if len(doc) < doc_len:
            num_padding_doc = doc_len - len(doc)
            for i in xrange(num_padding_doc):
                doc.append(" ".join([padding_word] * seq_len))
        new_docs.append(doc)
    return new_docs


def padding_msg(path_file, max_len):
    commits = list(open(path_file, "r").readlines())
    ids = [c.strip().split("\t")[0] for c in commits]
    sents = [c.strip().split("\t")[1] if len(c.strip().split("\t")) > 1 else "" for c in commits]

    # sents = [stemming_str(sent) for sent in sents]
    # sents = [remove_stopwords(sent) for sent in sents]

    pad_sents = pad_sentences(sents, seq_len=max_len)
    pad_sents = [i + "\t" + p for i, p in zip(ids, pad_sents)]
    return pad_sents


def padding_code(path_file, maxlen_code, maxline_code):
    commits = list(open(path_file, "r").readlines())
    ids = [c.strip().split("\t")[0] for c in commits]
    codes = [c.strip().split("\t")[1:] for c in commits]
    pad_codes = [pad_sentences(c, seq_len=maxlen_code) for c in codes]
    pad_codes = pad_docs(docs=pad_codes, doc_len=maxline_code, seq_len=maxlen_code)
    pad_codes = [i + "\t" + "\t".join(p) for i, p in zip(ids, pad_codes)]
    return pad_codes


def padding_file(options, maxlen_msg, maxlen_code, maxline_code):
    paths = input_path_maxtext_maxcode_maxline_mapping(options)
    msg_path, addedcode_path, removedcode_path, codefile_path = paths[0], paths[1], paths[2], paths[3]
    pad_msg = padding_msg(path_file=msg_path, max_len=maxlen_msg)
    pad_addedcode = padding_code(path_file=addedcode_path, maxlen_code=maxlen_code, maxline_code=maxline_code)
    pad_removedcode = padding_code(path_file=removedcode_path, maxlen_code=maxlen_code, maxline_code=maxline_code)
    return pad_msg, pad_addedcode, pad_removedcode


def padding_all(maxlen_msg, maxlen_code, maxline_code):
    paths_eq = input_path_maxtext_maxcode_maxline_mapping("eq")
    paths_extra = input_path_maxtext_maxcode_maxline_mapping("extra")
    paths_lbd = input_path_maxtext_maxcode_maxline_mapping("lbd")

    eq_msg_path, extra_msg_path, lbd_msg_path = paths_eq[0], paths_extra[0], paths_lbd[0]
    eq_pad_msg = padding_msg(eq_msg_path, max_len=maxlen_msg)
    extra_pad_msg = padding_msg(extra_msg_path, max_len=maxlen_msg)
    lbd_pad_msg = padding_msg(lbd_msg_path, max_len=maxlen_msg)
    # print "Padding in commit messages for different files: eq:%i, extra:%i, lbd:%i" % \
    #       (len(eq_pad_msg), len(extra_pad_msg), len(lbd_pad_msg))
    all_pad_msg = eq_pad_msg + extra_pad_msg + lbd_pad_msg
    print "Padding in commit messages for different files: "
    print len(all_pad_msg)

    eq_added_path, extra_added_path, lbd_added_path = paths_eq[1], paths_extra[1], paths_lbd[1]
    eq_pad_add = padding_code(eq_added_path, maxlen_code=maxlen_code, maxline_code=maxline_code)
    exit()


########################################################################################
########################################################################################
# print max_commit_file(options="eq")
# print max_commit_file(options="extra")
# print max_commit_file(options="lbd")
# exit()
# padding_file(options="eq", maxlen_msg=max_msg, maxlen_code=max_code, maxline_code=max_linecode)
