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


def commit_code_less_two_files(path_file):
    # count the number of files in commit code
    commits = list(open(path_file, "r").readlines())
    i_commits = commits_index(path_file)
    print "Total commits: %i" % (len(i_commits))
    ids, fs = extract_num_file_in_commit_code(commits=commits, index_commits=i_commits)
    greater = [f for f in fs if f > 1]
    print "Total commits which have two files in commit code: %i" % (len(greater))
    return ids, fs


def commit_one_file_commit_code(ids, fs, make_file):
    ids = [i for i, j in zip(ids, fs) if j == 1]
    with open(make_file, "w") as file_:
        [file_.write(str(c) + '\n') for c in ids]
    return ids

################################################################
path_file = "../raw_data/eq100_line_aug1.out"
ids, fs = commit_code_less_two_files(path_file=path_file)
make_file = "./eq100_line_aug1.out.id_onefile"
commit_one_file_commit_code(ids, fs, make_file)


