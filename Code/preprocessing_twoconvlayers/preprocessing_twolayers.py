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
def length_commit_mmsg(commit):
    print "hello"

################################################################
# path_file = "../raw_data/eq100_line_aug1.out"
# ids, fs = commit_code_less_two_files(path_file=path_file)
# make_file = "./eq100_line_aug1.out.id_onefile"
# ids = commit_one_file_commit_code(ids, fs, make_file)
# extract_info_update(path_file=path_file, ids=ids)
################################################################
path_file = "./eq100_line_aug1.out.msg"
