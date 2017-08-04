def clean_commit_code(commit_code):
    codes = [c.strip() for c in commit_code if len(c.strip()) > 0]
    return codes


def extract_code_commit(commit_code):
    commit_code = clean_commit_code(commit_code)
    codes = [c.strip().split(" ")[1] if c.startswith("file:") else c.strip().split(" ")[3] for i, c in
             enumerate(commit_code)]
    return ",".join(codes)


def extract_info(messages):
    id_ = messages[0].strip().split(":")[1].strip()
    stable_ = messages[1].strip().split(":")[1].strip()
    author_ = messages[2].strip().split(":")[1].strip()
    committer_ = messages[3].strip().split(":")[1].strip()
    committer_date = messages[4].strip().split(":")[1].strip()
    author_date = messages[5].strip().split(":")[1].strip()
    message_commit = messages[8].strip()
    code_commit = extract_code_commit(messages[10:])
    line = id_ + ":" + stable_ + ":" + author_ + ":" + committer_ + ":" + committer_date + ":" + \
           author_date + ":" + message_commit + ":" + code_commit
    return line


def commit_info(path_file, make_file):
    commits = list(open(path_file, "r").readlines())
    commits_index = [i for i, c in enumerate(commits) if c.startswith("commit:")]
    info = [extract_info(commits[commits_index[i]:]) if i == len(commits_index) - 1 else extract_info(
        commits[commits_index[i]: (commits_index[i + 1])]) for i in range(0, len(commits_index))]
    print len(info), len(commits_index)

    info = [v for v in info if len(v.split(":")[7]) and len(v.split(":")[6]) > 0]
    with open(make_file, "w") as file_:
        [file_.write(str(c) + '\n') for c in info]


def doing_mapping(index, message, dict_index, dict_word):
    print index, message
    texts = message.strip().split(",")

    texts_index = [dict_index.index(t) for t in texts]
    texts_words = [dict_word[t] for t in texts_index]
    return " ".join(texts_words)


def mapping_task(commmits, dict_file):
    dict_ = list(open(dict_file, "r").readlines())
    dict_index = [d.strip().split(":")[0].strip() for d in dict_]
    dict_word = [d.strip().split(":")[1].strip() for d in dict_]

    print len(dict_index), len(dict_word)

    ids = [c.strip().split(":")[0] for c in commmits]
    contains = [doing_mapping(index=i, message=c.strip().split(":")[6], dict_index=dict_index, dict_word=dict_word)
                + "\t" + doing_mapping(index=i, message=c.strip().split(":")[7]
                                       , dict_index=dict_index, dict_word=dict_word)
                for i, c in enumerate(commmits)]
    return ids, contains


def mapping_dict(path_file, dict_file, make_pos_file, make_neg_file):
    commits = list(open(path_file, "r").readlines())
    pos_commits = [c for c in commits if c.strip().split(":")[1] == "true"]
    neg_commits = [c for c in commits if c.strip().split(":")[1] == "false"]
    print len(commits), len(pos_commits), len(neg_commits)

    if len(pos_commits) > 0:
        ids_pos, contains_pos = mapping_task(pos_commits, dict_file=dict_file)
        # write positive file
        with open(make_pos_file + ".id", "w") as file_:
            [file_.write(str(c) + '\n') for c in ids_pos]
        with open(make_pos_file + ".contain", "w") as file_:
            [file_.write(str(c) + '\n') for c in contains_pos]

    if len(neg_commits) > 0:
        ids_neg, contains_neg = mapping_task(neg_commits, dict_file=dict_file)
        # write negative file
        with open(make_neg_file + ".id", "w") as file_:
            [file_.write(str(c) + '\n') for c in ids_neg]
        with open(make_neg_file + ".contain", "w") as file_:
            [file_.write(str(c) + '\n') for c in contains_neg]


######################################################################################
######################################################################################
### Training File
path_file = "./data/eq100_line_aug1.out"
make_file = "./data/eq100_line_aug1.info"
commit_info(path_file=path_file, make_file=make_file)
######################################################################################
path_file_, dict_file_ = "./data/eq100_line_aug1.info", "./data/eq100.dict"
pos_file, neg_file = "./data/eq100_line_aug1.pos", "./data/eq100_line_aug1.neg"
mapping_dict(path_file=path_file_, dict_file=dict_file_, make_pos_file=pos_file, make_neg_file=neg_file)


######################################################################################
######################################################################################
### Testing File
path_file = "./data/extra100_line_aug1.out"
make_file = "./data/extra100_line_aug1.info"
commit_info(path_file=path_file, make_file=make_file)
######################################################################################
path_file_, dict_file_ = "./data/extra100_line_aug1.info", "./data/extra100.dict"
pos_file, neg_file = "./data/extra100_line_aug1.pos", "./data/extra100_line_aug1.neg"
mapping_dict(path_file=path_file_, dict_file=dict_file_, make_pos_file=pos_file, make_neg_file=neg_file)


######################################################################################
######################################################################################
### Label File
path_file = "./data/lbd100_line_aug1.out"
make_file = "./data/lbd100_line_aug1.info"
commit_info(path_file=path_file, make_file=make_file)
######################################################################################
path_file_, dict_file_ = "./data/lbd100_line_aug1.info", "./data/lbd100.dict"
pos_file, neg_file = "./data/lbd100_line_aug1.pos", "./data/lbd100_line_aug1.neg"
mapping_dict(path_file=path_file_, dict_file=dict_file_, make_pos_file=pos_file, make_neg_file=neg_file)
######################################################################################
######################################################################################