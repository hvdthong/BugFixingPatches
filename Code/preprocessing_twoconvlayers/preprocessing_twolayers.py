import matplotlib.pyplot as plt


def input_path(options):
    paths = []
    if options == "eq":
        paths.append("./eq100_line_aug1.out.msg")
        paths.append("./eq100_line_aug1.out.addedcode")
        paths.append("./eq100_line_aug1.out.removedcode")
        paths.append("./eq100_line_aug1.out.codefile")
    elif options == "extra":
        paths.append("./extra100_line_aug1.out.msg")
        paths.append("./extra100_line_aug1.out.addedcode")
        paths.append("./extra100_line_aug1.out.removedcode")
        paths.append("./extra100_line_aug1.out.codefile")
    elif options == "lbd":
        paths.append("./lbd100_line_aug1.out.msg")
        paths.append("./lbd100_line_aug1.out.addedcode")
        paths.append("./lbd100_line_aug1.out.removedcode")
        paths.append("./lbd100_line_aug1.out.codefile")
    else:
        print "Your options are not correct"
        exit()
    return paths


def input_path_maxtext_maxcode(options, maxtext, maxcode):
    paths = []
    if options == "eq":
        paths.append("./eq100_line_aug1.out.msg" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./eq100_line_aug1.out.addedcode" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./eq100_line_aug1.out.removedcode" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./eq100_line_aug1.out.codefile" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
    elif options == "extra":
        paths.append("./extra100_line_aug1.out.msg" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./extra100_line_aug1.out.addedcode" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./extra100_line_aug1.out.removedcode" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./extra100_line_aug1.out.codefile" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
    elif options == "lbd":
        paths.append("./lbd100_line_aug1.out.msg" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./lbd100_line_aug1.out.addedcode" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./lbd100_line_aug1.out.removedcode" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
        paths.append("./lbd100_line_aug1.out.codefile" + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode))
    else:
        print "Your options are not correct"
        exit()
    return paths


def input_path_dict(options):
    path = ""
    if options == "eq":
        path = "../raw_data/eq100.dict"
    elif options == "extra":
        path = "../raw_data/extra100.dict"
    elif options == "lbd":
         path = "../raw_data/lbd100.dict"
    else:
        print "Your options are not correct"
        exit()
    return path


def commits_index(path_file):
    commits = list(open(path_file, "r").readlines())
    i_commits = [i for i, c in enumerate(commits) if c.startswith("commit:")]
    return i_commits


def extract_num_file_in_commit_code(commits, index_commits):
    ids, fs = [], []
    for i in range(0, len(index_commits)):
        if i == len(index_commits) - 1:
            commit = commits[index_commits[i]:]
        else:
            commit = commits[index_commits[i]: (index_commits[i + 1])]
        id_ = commit[0].strip().split(":")[1].strip()
        code_commit = commit[10:]
        code_commit = [c.strip() for c in code_commit if len(c.strip()) > 0]
        files = [c.strip().split(" ")[1] for i, c in enumerate(code_commit) if c.startswith("file:")]
        ids.append(id_), fs.append(len(files))
    return ids, fs


def get_commit_commitindex(path_file):
    commits = list(open(path_file, "r").readlines())
    i_commits = commits_index(path_file)
    return commits, i_commits


def commit_code_less_two_files(path_file):
    # count the number of files in commit code
    commits, i_commits = get_commit_commitindex(path_file)
    print "Total commits: %i" % (len(i_commits))
    ids, fs = extract_num_file_in_commit_code(commits=commits, index_commits=i_commits)
    greater = [f for f in fs if f > 1]
    one = [f for f in fs if f == 1]
    zero = [f for f in fs if f == 0]
    print "Total commits which have more than two files in commit code: %i" % (len(greater))
    print "Total commits which have one file in commit code: %i" % (len(one))
    print "Total commits which have zero file in commit code: %i" % (len(zero))
    return ids, fs


def commit_one_file_commit_code(ids, fs, make_file):
    ids = [i for i, j in zip(ids, fs) if j == 1]
    with open(make_file, "w") as file_:
        [file_.write(str(c) + '\n') for c in ids]
    return ids


################################################################
def write_file(new_file, info):
    with open(new_file, "w") as file_:
        [file_.write(str(c) + '\n') for c in info]


def extract_code_filename(commit_code):
    for i, c in enumerate(commit_code):
        if c.startswith("file"):
            return c.strip().split(" ")[1]


def extract_code_addedcode(commit_code):
    added_code = []
    for i, c in enumerate(commit_code):
        if not c.startswith("file"):
            if "+" in c:
                type_code = c.strip().split(":")[2].strip()
                added_code.append("<" + type_code + ">," + c.strip().split(" ")[3])
    return added_code


def extract_code_removedcode(commit_code):
    removed_code = []
    for i, c in enumerate(commit_code):
        if not c.startswith("file"):
            if "-" in c:
                type_code = c.strip().split(":")[2].strip()
                removed_code.append("<" + type_code + ">," + c.strip().split(" ")[3])
    return removed_code


def extract_code_info(commit_code):
    name = extract_code_filename(commit_code)
    return name


def extract_info_commit(commit, ids):
    id_ = commit[0].strip().split(":")[1].strip()
    if id_ in ids:
        stable_ = commit[1].strip().split(":")[1].strip()
        commit_message = commit[8].strip()
        file_code = extract_code_filename(commit[10:])
        add_code = extract_code_addedcode(commit[10:])
        remove_code = extract_code_removedcode(commit[10:])
        return id_, stable_, commit_message, file_code, add_code, remove_code
    else:
        return [], [], [], [], [], []


def extract_info_update(path_file, ids):
    # extract information of commit from in ids (ids: commit id)
    commits, i_commits = get_commit_commitindex(path_file)
    print len(commits), len(i_commits), len(ids)

    id_stable, id_msg, id_codefile, id_addcode, id_removecode = [], [], [], [], []
    for i in range(0, len(i_commits)):
        if i == len(i_commits) - 1:
            id_, stable, message, file_code, add_code, remove_code = extract_info_commit(
                commit=commits[i_commits[i]:], ids=ids)
        else:
            id_, stable, message, file_code, add_code, remove_code = extract_info_commit(
                commit=commits[i_commits[i]:(i_commits[i + 1])], ids=ids)

        if len(id_) != 0:
            id_stable.append(id_ + ":" + stable)
            id_msg.append(id_ + ":" + message)
            id_codefile.append(id_ + ":" + file_code)
            id_addcode.append(id_ + ":" + "\t".join(add_code))
            id_removecode.append(id_ + ":" + "\t".join(remove_code))
    print len(id_stable), len(id_msg), len(id_codefile), len(id_addcode), len(id_removecode)
    write_file(new_file="./" + path_file.split("/")[-1] + ".stable", info=id_stable)
    write_file(new_file="./" + path_file.split("/")[-1] + ".msg", info=id_msg)
    write_file(new_file="./" + path_file.split("/")[-1] + ".codefile", info=id_codefile)
    write_file(new_file="./" + path_file.split("/")[-1] + ".addedcode", info=id_addcode)
    write_file(new_file="./" + path_file.split("/")[-1] + ".removedcode", info=id_removecode)


################################################################
def max_length_commit_msg(path_file):
    commits = list(open(path_file, "r").readlines())
    return max([len((c.strip().split(":")[-1]).split(",")) for c in commits])


def max_length_commit_code(path_file):
    commits = list(open(path_file, "r").readlines())
    lengths = []
    for c in commits:
        code = c.strip().split(":")[-1].split("\t")
        length = [len(t.split(",")) for t in code]
        lengths += length
    return max(lengths)


def hist_commit_msg(path_file):
    commits = list(open(path_file, "r").readlines())
    len_msgs = [len((c.strip().split(":")[-1]).split(",")) for c in commits]
    plt.hist(len_msgs, normed=True, bins=20)
    plt.title("Histogram of commit message")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()


def hist_commit_addedcode(path_file):
    commits = list(open(path_file, "r").readlines())
    lengths = []
    for c in commits:
        code = c.strip().split(":")[-1].split("\t")
        length = [len(t.split(",")) for t in code]
        lengths += [max(length)]
    plt.hist(lengths, bins=20)
    plt.title("Histogram of commit message")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.show()


################################################################
def filter_msg_maxlength(path_file, max_length):
    commits = list(open(path_file, "r").readlines())
    ids = []
    for c in commits:
        id_ = c.strip().split(":")[0]
        msg = len((c.strip().split(":")[-1]).split(","))
        if msg <= max_length:
            ids.append(id_)
    return ids


def filter_code_maxlength(path_file, max_length):
    commits = list(open(path_file, "r").readlines())
    ids = []
    for c in commits:
        id_ = c.strip().split(":")[0]
        code = c.strip().split(":")[-1].split("\t")
        length = max([len(t.split(",")) for t in code])
        if length <= max_length:
            ids.append(id_)
    return ids


################################################################
def select_ids(path_file, ids):
    commits = list(open(path_file, "r").readlines())
    commits_filter = [c.strip() for c in commits if c.strip().split(":")[0] in ids]
    return commits_filter


def create_maxtext_maxcode(options, maxtext, maxcode):
    print options, maxtext, maxcode
    print input_path(options)
    path_msg, path_addedcode, path_removedcode = input_path(options)[0], input_path(options)[1], input_path(options)[2]
    id_msg = filter_msg_maxlength(path_file=path_msg, max_length=maxtext)
    id_addedcode = filter_code_maxlength(path_file=path_addedcode, max_length=maxcode)
    id_removedcode = filter_code_maxlength(path_file=path_removedcode, max_length=maxcode)
    print path_msg, path_addedcode, path_removedcode
    print len(id_msg), len(id_addedcode), len(id_removedcode)
    ids = list(set(id_msg) & set(id_addedcode) & set(id_removedcode))
    print "Total commits have max length text %i and max length code %i: %i" \
          % (max_length_text, max_length_code, len(ids))
    write_file(new_file=input_path(options)[0] + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode)
               , info=select_ids(input_path(options)[0], ids=ids))
    write_file(new_file=input_path(options)[1] + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode)
               , info=select_ids(input_path(options)[1], ids=ids))
    write_file(new_file=input_path(options)[2] + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode)
               , info=select_ids(input_path(options)[2], ids=ids))
    write_file(new_file=input_path(options)[3] + ".maxtext" + str(maxtext) + ".maxcode" + str(maxcode)
               , info=select_ids(input_path(options)[3], ids=ids))


################################################################
def read_dict(path_file):
    dict_ = list(open(path_file, "r").readlines())
    dict_index = [d.strip().split(":")[0].strip() for d in dict_]
    dict_word = [d.strip().split(":")[1].strip() for d in dict_]
    return dict_index, dict_word


def doing_mapping_msg_file(index, message, dict_index, dict_word):
    # print index, message
    # print index
    texts = message.strip().split(",")
    # texts_index = [dict_index.index(t) for t in texts]
    # texts_words = [dict_word[t] for t in texts_index]
    # print index, texts
    # for t in texts:
    #     print int(t)
    texts_words = [dict_word[int(t) - 1] if t != "" else "" for t in texts]
    return " ".join(texts_words)


def mapping_msg_file(path_file, path_dict):
    dict_index, dict_word = read_dict(path_file=path_dict)
    commits = list(open(path_file, "r").readlines())
    ids = [c.strip().split(":")[0] for c in commits]
    files = [c.strip().split(":")[-1] for c in commits]
    words = [doing_mapping_msg_file(index=i, message=f, dict_index=dict_index, dict_word=dict_word)
             for i, f in enumerate(files)]
    mapping = [i + "\t" + w for i, w in zip(ids, words)]
    return mapping


def doing_mapping_added_removed_code(index, code, dict_index, dict_word):
    new_lines = ""
    for lines in code:
        split_lines = lines.split(",")
        if len(split_lines) > 1:
            text_words = [dict_word[int(split_lines[i]) - 1] for i in range(1, len(split_lines))]
            text_words = split_lines[0] + " " + " ".join(text_words)
            new_lines += "\t" + text_words
        else:
            return ""
    return new_lines.strip()


def mapping_added_removed_code(path_file, path_dict):
    dict_index, dict_word = read_dict(path_file=path_dict)
    commits = list(open(path_file, "r").readlines())
    ids = [c.strip().split(":")[0] for c in commits]
    codes = [c.strip().split(":")[-1].split("\t") for c in commits]
    words = [doing_mapping_added_removed_code(index=0, code=c,
                                              dict_index=dict_index, dict_word=dict_word) for c in codes]
    mapping = [i + "\t" + w for i, w in zip(ids, words)]
    return mapping


def create_mapping_dict(options, maxtext, maxcode):
    print options, maxtext, maxcode
    paths = input_path_maxtext_maxcode(options, maxtext=maxtext, maxcode=maxcode)
    path_msg, path_addedcode, path_removedcode, path_codefile = paths[0], paths[1], paths[2], paths[3]

    # print input_path_dict(options=options)
    mapping_msg = mapping_msg_file(path_file=path_msg, path_dict=input_path_dict(options=options))
    write_file(new_file=path_msg + ".mapping", info=mapping_msg)

    mapping_addedcode = mapping_added_removed_code(path_file=path_addedcode, path_dict=input_path_dict(options=options))
    write_file(new_file=path_addedcode + ".mapping", info=mapping_addedcode)

    mapping_removedcode = mapping_added_removed_code(path_file=path_removedcode,
                                                     path_dict=input_path_dict(options=options))
    write_file(new_file=path_removedcode + ".mapping", info=mapping_removedcode)

    mapping_codefile = mapping_msg_file(path_file=path_codefile, path_dict=input_path_dict(options=options))
    write_file(new_file=path_codefile + ".mapping", info=mapping_codefile)


################################################################
################################################################
################################################################
# path_file = "../raw_data/eq100_line_aug1.out"
# ids, fs = commit_code_less_two_files(path_file=path_file)
# make_file = "./eq100_line_aug1.out.id_onefile"
# ids = commit_one_file_commit_code(ids, fs, make_file)
# extract_info_update(path_file=path_file, ids=ids)

# path_file = "../raw_data/extra100_line_aug1.out"
# ids, fs = commit_code_less_two_files(path_file=path_file)
# make_file = "./extra100_line_aug1.out.id_onefile"
# ids = commit_one_file_commit_code(ids, fs, make_file)
# extract_info_update(path_file=path_file, ids=ids)
#
# path_file = "../raw_data/lbd100_line_aug1.out"
# ids, fs = commit_code_less_two_files(path_file=path_file)
# make_file = "./lbd100_line_aug1.out.id_onefile"
# ids = commit_one_file_commit_code(ids, fs, make_file)
# extract_info_update(path_file=path_file, ids=ids)
################################################################
# path_file = "./eq100_line_aug1.out.msg"
# print path_file, max_length_commit_msg(path_file=path_file)
# path_file = "./extra100_line_aug1.out.msg"
# print path_file, max_length_commit_msg(path_file=path_file)
# path_file = "./lbd100_line_aug1.out.msg"
# print path_file, max_length_commit_msg(path_file=path_file)
#
# path_file = "./eq100_line_aug1.out.codefile"
# print path_file, max_length_commit_msg(path_file=path_file)
# path_file = "./extra100_line_aug1.out.codefile"
# print path_file, max_length_commit_msg(path_file=path_file)
# path_file = "./lbd100_line_aug1.out.codefile"
# print path_file, max_length_commit_msg(path_file=path_file)
################################################################
# path_file = "./eq100_line_aug1.out.msg"
# hist_commit_msg(path_file=path_file)
# path_file = "./eq100_line_aug1.out.addedcode"
# hist_commit_addedcode(path_file=path_file)

################################################################
# path_added_file = "./eq100_line_aug1.out.addedcode"
# max_addedcode = max_length_commit_code(path_file=path_added_file)
# path_removed_file = "./eq100_line_aug1.out.removedcode"
# max_removedcode = max_length_commit_code(path_file=path_removed_file)
# print max(max_addedcode, max_removedcode)

################################################################
max_length_text, max_length_code = 175, 250
# create_maxtext_maxcode(options="eq", maxtext=max_length_text, maxcode=max_length_code)
# create_maxtext_maxcode(options="extra", maxtext=max_length_text, maxcode=max_length_code)
# create_maxtext_maxcode(options="lbd", maxtext=max_length_text, maxcode=max_length_code)

# create_mapping_dict(options="eq", maxtext=max_length_text, maxcode=max_length_code)
# create_mapping_dict(options="extra", maxtext=max_length_text, maxcode=max_length_code)
# create_mapping_dict(options="lbd", maxtext=max_length_text, maxcode=max_length_code)